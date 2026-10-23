# SPDX-License-Identifier: Apache-2.0
"""Loss replay keeps current assignments, scalar bits, gradients and ownership."""

from __future__ import annotations

import copy
import gc
import math
import pickle
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from markers import requires_cuda
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.plugins.precision import MixedPrecision
from torch.utils.data import DataLoader

from lucid_yolo.assign import stal, tal
from lucid_yolo.assign.tal import TaskAlignedAssigner
from lucid_yolo.cli.train import DetectionCLI, packaged_config
from lucid_yolo.data.targets import Targets
from lucid_yolo.losses import detection_loss
from lucid_yolo.losses.dual_loss import DualBranchLoss, DualLossOutput
from lucid_yolo.ptl import loss_capture as capture_module
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.loss_capture import _BF16_SUPPORTED, _LossCapture, _LossCaptureLifecycle
from lucid_yolo.ptl.module import DetectionLitModule


@pytest.fixture(autouse=True)
def _deterministic(deterministic_kernels: None) -> None:
    """Keep the byte oracle deterministic and restore the caller's numeric policy."""


def _inputs(
    count: int, batch: int = 2, device: str = "cuda", dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, ...]:
    """Make independent predictions and an asymmetric, possibly empty target batch.

    Examples:
        >>> _inputs(0, device="cpu")[5].shape
        torch.Size([2, 0, 4])
    """
    generator = torch.Generator(device=device).manual_seed(41 + count)
    points = torch.stack(
        torch.meshgrid(torch.arange(4, device=device), torch.arange(4, device=device), indexing="ij"), -1
    )
    points = points.reshape(-1, 2).float() * 8 + 4
    logits = [
        torch.randn((batch, 16, 3), generator=generator, device=device, dtype=dtype).requires_grad_() for _ in range(2)
    ]
    pred = [
        torch.cat(
            (
                torch.zeros((batch, 16, 2), device=device),
                torch.rand((batch, 16, 2), generator=generator, device=device) * 30 + 2,
            ),
            -1,
        ).requires_grad_()
        for _ in range(2)
    ]
    gt = torch.rand((batch, count, 4), generator=generator, device=device) * 12
    gt[..., 2:] += gt[..., :2] + 10
    labels = torch.arange(count, device=device).expand(batch, -1).remainder(3).contiguous()
    mask = torch.ones((batch, count), dtype=torch.bool, device=device)
    if count:
        mask[0, -1] = False
        labels[0, -1] = -1
    return (
        logits[0],
        pred[0],
        logits[1],
        pred[1],
        points,
        gt,
        labels,
        mask,
        torch.full((16,), 8.0, device=device),
        torch.tensor(0.37, dtype=torch.float64, device=device),
    )


def _native(source: DualBranchLoss, inputs: tuple[torch.Tensor, ...]) -> DualLossOutput:
    """Call the checked public criterion, keeping the branch weight a live tensor.

    Examples:
        >>> bool(torch.isfinite(_native(DualBranchLoss(), _inputs(0, device="cpu")).total))
        True
    """
    return source(*inputs[:8], strides=inputs[8], alpha=inputs[9])


def _bits(left: torch.Tensor, right: torch.Tensor) -> None:
    """Compare raw bits rather than rounded values or a numerical tolerance.

    Examples:
        >>> _bits(torch.tensor([1.0]), torch.tensor([1.0]))
    """
    assert left.dtype == right.dtype and left.shape == right.shape
    assert torch.equal(
        left.detach().contiguous().reshape(-1).view(torch.uint8),
        right.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def _same(left: DualLossOutput, right: DualLossOutput) -> None:
    """Compare every public scalar and every assignment field.

    Examples:
        >>> output = _native(DualBranchLoss(), _inputs(0, device="cpu"))
        >>> _same(output, output)
    """
    for a, b in (
        (left.total, right.total),
        *((getattr(left.o2m, key), getattr(right.o2m, key)) for key in ("total", "box", "cls", "l1")),
        *((getattr(left.o2o, key), getattr(right.o2o, key)) for key in ("total", "box", "cls", "l1")),
    ):
        _bits(a, b)
    for a, b in ((left.o2m_assign, right.o2m_assign), (left.o2o_assign, right.o2o_assign)):
        for key in ("fg_mask", "gt_index", "target_labels", "target_boxes", "align_weights"):
            _bits(getattr(a, key), getattr(b, key))


def test_cpu_stays_eager_and_copy_discards_recordings() -> None:
    """The off-device path never asks CUDA to record and keeps the public output."""
    holder, source, inputs = _LossCapture(), DualBranchLoss(), _inputs(3, device="cpu")
    with patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("CPU capture")):
        _same(_native(source, inputs), holder.run(source, *inputs))
    holder._disabled = True
    for restored in (copy.deepcopy(holder), pickle.loads(pickle.dumps(holder))):
        assert restored._objective is None and not restored._disabled


@pytest.mark.parametrize(
    "registration",
    [
        "register_module_forward_hook",
        "register_module_forward_pre_hook",
        "register_module_full_backward_hook",
        "register_module_full_backward_pre_hook",
    ],
)
def test_eligible_rejects_global_module_hooks_before_unpacking_inputs(registration: str) -> None:
    """A registered global hook exits before the function unpacks its input tuple.

    A single-element tuple would raise on the later unpacking line were the early
    exit skipped, so a clean ``False`` here proves the hook check fired first.
    """

    def observe(*args: object) -> None:
        pass

    with getattr(torch.nn.modules.module, registration)(observe):
        assert capture_module._eligible((torch.zeros(1),)) is False


@pytest.fixture
def _host_recorder(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Exercise host bookkeeping with CPU autograd, without claiming CUDA replay coverage.

    Only device eligibility, the stream handle, graph installation and its fence
    are substituted. Assignment, scoring, tensor ownership and backward are real.
    """
    monkeypatch.setattr(capture_module, "_eligible", lambda inputs: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: MagicMock(cuda_stream=1))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    recorder = MagicMock(side_effect=lambda objective, samples: objective)
    monkeypatch.setattr(torch.cuda, "make_graphed_callables", recorder)
    return recorder


def test_host_workspace_growth_shrinkage_and_output_ownership(_host_recorder: MagicMock) -> None:
    """Masked workspace reuse preserves real CPU loss bits, gradients and saved outputs."""
    holder, source = _LossCapture(), DualBranchLoss()
    saved = []
    for count in (3, 1, 7, 2):
        inputs = _inputs(count, device="cpu")
        reference = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in inputs)
        expected = _native(source, reference)
        actual = holder.run(source, *inputs)
        _same(expected, actual)
        expected.total.backward()
        actual.total.backward()
        assert not holder._pending_backward
        for left, right in zip(inputs[:4], reference[:4], strict=True):
            assert left.grad is not None and right.grad is not None
            _bits(left.grad, right.grad)
        saved.append((actual, copy.deepcopy(actual.o2m_assign)))
    assert _host_recorder.call_count == 2 and holder._capacity == 8
    for actual, assignment in saved:
        _bits(actual.o2m_assign.target_boxes, assignment.target_boxes)
    assert holder._workspace is not None
    assert not holder._workspace[2][:, 2:].any()
    holder.clear()
    assert holder._objective is None and holder._workspace is None


@pytest.mark.parametrize("collection_enabled", [False, True])
@pytest.mark.parametrize("failure", ["record", "fence"])
def test_host_capture_failure_restores_gc_and_releases_workspace(
    _host_recorder: MagicMock, collection_enabled: bool, failure: str
) -> None:
    """Exceptions at either recording boundary leave no published owner and restore GC."""
    holder, original = _LossCapture(), gc.isenabled()
    gc.enable() if collection_enabled else gc.disable()
    if failure == "record":
        _host_recorder.side_effect = RuntimeError("injected host recorder failure")
    try:
        with (
            patch("torch.cuda.synchronize", side_effect=RuntimeError("injected host fence failure")),
            pytest.raises(RuntimeError, match="injected host"),
        ):
            holder.run(DualBranchLoss(), *_inputs(3, device="cpu"))
        assert gc.isenabled() is collection_enabled
        assert holder._objective is None and holder._workspace is None
        assert holder._signature is None and holder._capacity == 0
    finally:
        gc.enable() if original else gc.disable()


@pytest.mark.parametrize("change", ["settings", "precision", "unreadable", "custom"])
def test_host_contract_changes_retire_once(_host_recorder: MagicMock, change: str) -> None:
    """Changed live policies retire a recording before it can reuse stale operations."""
    holder, source = _LossCapture(), DualBranchLoss()
    holder.run(source, *_inputs(3, device="cpu")).total.backward()
    if change == "settings":
        source._o2m_loss.box_gain += 1
    elif change == "custom":
        source._score = MagicMock(wraps=source._score)
    with patch.object(capture_module, "_precision_signature", wraps=capture_module._precision_signature) as precision:
        if change == "precision":
            precision.return_value = ("changed",)
        elif change == "unreadable":
            precision.side_effect = RuntimeError("policy query failed")
        with pytest.warns(UserWarning, match="stays eager") as warnings:
            _same(_native(source, _inputs(1, device="cpu")), holder.run(source, *_inputs(1, device="cpu")))
            holder.run(source, *_inputs(1, device="cpu"))
    assert len(warnings) == 1 and holder._disabled
    assert holder._objective is None and _host_recorder.call_count == 1


_FLOAT32_TINY = torch.finfo(torch.float32).tiny


@pytest.mark.parametrize("branch", ["o2m_assigner", "o2o_assigner"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("eps", 0.0),
        ("eps", math.nextafter(_FLOAT32_TINY, 0.0)),
        ("eps", float("nan")),
        ("eps", -1e-9),
        ("alpha", -1.0),
        ("alpha", float("nan")),
        ("beta", float("inf")),
        ("beta", float("nan")),
        ("topk", torch.tensor(1)),
        ("topk", True),
    ],
)
def test_source_signature_refuses_uncertified_settings(branch: str, field: str, value: object) -> None:
    """Invalid scalar policies cannot certify neutral padding, even without a CUDA runtime.

    The next float below the smallest normal float32 is the first epsilon that can
    flush to zero on the device, so it is refused while the smallest normal is not.
    """
    source = DualBranchLoss()
    setattr(getattr(source, branch), field, value)
    assert capture_module._source_signature(source) is None


@pytest.mark.parametrize("branch", ["o2m_assigner", "o2o_assigner"])
def test_source_signature_accepts_smallest_float32_epsilon(branch: str) -> None:
    """An epsilon equal to the smallest normal float32 still keeps zero-overlap IoUs finite."""
    source = DualBranchLoss()
    getattr(source, branch).eps = _FLOAT32_TINY
    assert capture_module._source_signature(source) is not None


@pytest.mark.parametrize("record_first", [False, True])
@pytest.mark.parametrize(
    "owner,name",
    [
        pytest.param(TaskAlignedAssigner, "_candidate_mask", id="base-candidate-mask"),
        pytest.param(tal, "_box_iou_cross", id="box-iou"),
        pytest.param(stal, "surrogate_boxes", id="surrogate-boxes"),
        pytest.param(detection_loss, "ciou_loss", id="ciou-loss"),
    ],
)
def test_host_replaced_delegated_stage_keeps_native_route(
    _host_recorder: MagicMock, record_first: bool, owner: object, name: str
) -> None:
    """Replacing a stage reached through super() or a module global retires replay.

    The surrogate filter calls its parent through ``super()`` and the recorded stages
    call module helpers by global name, so neither shows in the instance attributes.
    A replacement before or after the first recording must reach the native route.
    """
    holder, source = _LossCapture(), DualBranchLoss()
    if record_first:
        holder.run(source, *_inputs(3, device="cpu")).total.backward()
    original = getattr(owner, name)
    calls: list[int] = []

    def observed(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    with patch.object(owner, name, observed), pytest.warns(UserWarning, match="customized"):
        actual = holder.run(source, *_inputs(2, device="cpu"))
        replaced_calls = len(calls)
        expected = _native(source, _inputs(2, device="cpu"))
    assert replaced_calls > 0
    _same(expected, actual)
    assert holder._objective is None and holder._disabled


_GUARDED_INPUTS = {"match_logits": 0, "match_boxes": 1, "unique_logits": 2, "unique_boxes": 3, "target_boxes": 5}


@pytest.mark.parametrize("case", ["empty", "shape", "labels", *(f"nonfinite_{name}" for name in _GUARDED_INPUTS)])
def test_host_fallback_preserves_current_inputs(_host_recorder: MagicMock, case: str) -> None:
    """Existing recordings never hide empty, non-finite, reshaped or invalid current inputs.

    The recorder is stubbed to run the real objective, so a replay is visible as a
    call to the recorded forward. Capacity and recorder counts cannot tell a
    fallback from a replay at the same capacity, hence the forward-call count.
    """
    holder, source = _LossCapture(), DualBranchLoss()
    holder.run(source, *_inputs(3, device="cpu")).total.backward()
    inputs = _inputs(0 if case == "empty" else 1, batch=1 if case == "shape" else 2, device="cpu")
    original = capture_module._LossObjective.forward
    with patch.object(capture_module._LossObjective, "forward", autospec=True, side_effect=original) as replayed:
        if case == "labels":
            inputs[6][1, 0] = 3
            with pytest.raises(ValueError, match="gt_labels must be"):
                holder.run(source, *inputs)
        elif case.startswith("nonfinite_"):
            with torch.no_grad():
                inputs[_GUARDED_INPUTS[case.removeprefix("nonfinite_")]].view(-1)[0] = float("inf")
            expected, actual = _native(source, inputs), holder.run(source, *inputs)
            assert torch.equal(expected.total.isnan(), actual.total.isnan())
            assert torch.equal(expected.o2m_assign.fg_mask, actual.o2m_assign.fg_mask)
        else:
            _same(_native(source, inputs), holder.run(source, *inputs))
    assert replayed.call_count == 0
    assert _host_recorder.call_count == 1 and holder._capacity == 4


@pytest.mark.parametrize("name,index", [("anchor_points", 4), ("strides", 8), ("alpha", 9)])
def test_host_unpadded_nonfinite_inputs_replay_with_native_bits(
    _host_recorder: MagicMock, name: str, index: int
) -> None:
    """Anchor points, strides and alpha are never padded, so their non-finite values need no fallback.

    Padded rows only meet predictions and target boxes in arithmetic. Anchor points
    enter comparisons, strides scale each branch's own terms and alpha weights the two
    branches, so replay and the native route must agree even when the values are infinite.
    """
    holder, source = _LossCapture(), DualBranchLoss()
    holder.run(source, *_inputs(3, device="cpu")).total.backward()
    inputs = _inputs(2, device="cpu")
    torch.atleast_1d(inputs[index])[0] = float("inf")  # alpha is 0-dim; the others index as before
    original = capture_module._LossObjective.forward
    with patch.object(capture_module._LossObjective, "forward", autospec=True, side_effect=original) as replayed:
        expected, actual = _native(source, inputs), holder.run(source, *inputs)
    assert replayed.call_count == 1, name
    assert torch.equal(expected.total.isnan(), actual.total.isnan())
    _bits(expected.o2m_assign.fg_mask, actual.o2m_assign.fg_mask)
    _bits(expected.o2o_assign.fg_mask, actual.o2o_assign.fg_mask)


def test_host_skipped_backward_resets_at_batch_start(_host_recorder: MagicMock) -> None:
    """A skipped backward keeps same-batch calls eager until the automatic next-batch hook."""
    model = DetectionLitModule(0.5, 0.25, 64, 3, capture_loss=True)
    holder, source = model._loss_capture, DualBranchLoss()
    holder.run(source, *_inputs(3, device="cpu"))
    assert holder._pending_backward
    _same(_native(source, _inputs(1, device="cpu")), holder.run(source, *_inputs(1, device="cpu")))
    _LossCaptureLifecycle().on_train_batch_start(MagicMock(spec=Trainer), model, None, 1)
    assert not holder._pending_backward
    holder.run(source, *_inputs(1, device="cpu")).total.backward()
    assert _host_recorder.call_count == 1


def test_host_old_bf16_contract_stays_native(_host_recorder: MagicMock) -> None:
    """An unsupported BF16 runtime rejects replay before allocating a workspace."""
    holder, source = _LossCapture(), DualBranchLoss()
    with patch.object(capture_module, "_BF16_SUPPORTED", False):
        inputs = _inputs(1, device="cpu", dtype=torch.bfloat16)
        _same(_native(source, inputs), holder.run(source, *inputs))
    _host_recorder.assert_not_called()
    assert holder._workspace is None and not holder._disabled


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"task": "segment"}, id="segment"),
        pytest.param({"task": "obb"}, id="obb"),
        pytest.param({"compile_step": True}, id="compile_step"),
        pytest.param({"task": "keypoints", "num_keypoints": 3, "compile_step": True}, id="keypoints_compile_step"),
    ],
)
def test_unsupported_configurations_raise(kwargs: dict[str, object]) -> None:
    """The public option cannot silently promise replay for another task or compiler."""
    with pytest.raises(ValueError, match="capture_loss=True requires"):
        DetectionLitModule(0.5, 0.25, 64, 3, capture_loss=True, **kwargs)


@pytest.mark.parametrize("hook", ["on_fit_start", "on_fit_end", "on_exception"])
def test_fit_boundaries_reset_holder(hook: str) -> None:
    """Success, reuse and failure all clear the fit's transient owner."""
    model = DetectionLitModule(0.5, 0.25, 64, 3, capture_loss=True)
    model._loss_capture._disabled = True
    getattr(_LossCaptureLifecycle(), hook)(
        MagicMock(spec=Trainer), model, *([RuntimeError("stop")] if hook == "on_exception" else [])
    )
    assert not model._loss_capture._disabled


@requires_cuda
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="float32"),
        pytest.param(
            torch.bfloat16,
            id="bfloat16",
            marks=pytest.mark.skipif(
                not _BF16_SUPPORTED,
                reason="BF16 replay begins at the verified 2.9 runtime; separate test proves native fallback",
            ),
        ),
        pytest.param(torch.float16, id="float16"),
    ],
)
def test_variable_target_capacity_matches_native_bits_and_gradients(dtype: torch.dtype) -> None:
    """Growth, shrinkage and empty targets keep each branch's real assignment unchanged."""
    holder, source = _LossCapture(), DualBranchLoss()
    for count in (3, 1, 0, 7, 2, 0, 9, 1):
        inputs = _inputs(count, dtype=dtype)
        reference = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in inputs)
        with torch.autocast("cuda", enabled=dtype != torch.float32, dtype=dtype, cache_enabled=False):
            expected = _native(source, reference)
            actual = holder.run(source, *inputs)
        assert holder._objective is not None
        _same(expected, actual)
        expected.total.backward()
        actual.total.backward()
        assert not holder._pending_backward
        for a, b in zip(inputs[:4], reference[:4], strict=True):
            assert a.grad is not None and b.grad is not None
            _bits(a.grad, b.grad)
    assert holder._capacity == 16
    holder.clear()
    assert holder._objective is None and holder._workspace is None


@requires_cuda
def test_outputs_survive_replay_and_second_forward_stays_eager() -> None:
    """Saved results own their values and an outstanding backward cannot be overwritten."""
    holder, source, inputs = _LossCapture(), DualBranchLoss(), _inputs(3)
    first = holder.run(source, *inputs)
    saved = first.total.detach().clone(), first.o2m_assign.target_boxes.clone()
    second = holder.run(source, *_inputs(1))
    _same(second, _native(source, _inputs(1)))
    first.total.backward()
    assert not holder._pending_backward
    third = holder.run(source, *_inputs(2))
    third.total.backward()
    _bits(first.total, saved[0])
    _bits(first.o2m_assign.target_boxes, saved[1])


@requires_cuda
@pytest.mark.parametrize("bad,padded", [(3, False), (-1, False), (3, True)])
def test_current_invalid_labels_fail_before_replay(bad: int, padded: bool) -> None:
    """A valid first call never exempts the next batch, including its padding, from bounds checks."""
    holder, source = _LossCapture(), DualBranchLoss()
    first = holder.run(source, *_inputs(3))
    first.total.backward()
    inputs = _inputs(1)
    inputs[6][1, 0] = bad
    inputs[7][1, 0] = not padded
    with pytest.raises(ValueError, match="gt_labels must be"):
        holder.run(source, *inputs)


@requires_cuda
def test_replay_removes_python_assignment_and_scoring_calls() -> None:
    """A silent reversion to eager fails the work-count oracle after a real recording."""
    holder, source = _LossCapture(), DualBranchLoss()
    first = holder.run(source, *_inputs(3))
    first.total.backward()
    codes = {source.o2m_assigner._assign.__code__, source._score.__code__}
    calls: list[str] = []

    def observe(frame: object, event: str, argument: object) -> None:
        if event == "call" and frame.f_code in codes:
            calls.append(frame.f_code.co_name)

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        next_output = holder.run(source, *_inputs(2))
        next_output.total.backward()
    finally:
        sys.setprofile(previous)
    assert holder._objective is not None
    assert calls == []
    _native(source, _inputs(2)).total.backward()


@requires_cuda
def test_changed_gain_retires_recording_and_keeps_native_result() -> None:
    """Mutable loss settings cannot silently keep the old graph's constants."""
    holder, source = _LossCapture(), DualBranchLoss()
    holder.run(source, *_inputs(3)).total.backward()
    source._o2m_loss.box_gain = 2.5
    inputs = _inputs(2)
    with pytest.warns(UserWarning, match="settings.*changed"):
        actual = holder.run(source, *inputs)
    assert holder._objective is None and holder._disabled
    _same(_native(source, inputs), actual)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="detect"),
        pytest.param({"task": "keypoints", "num_keypoints": 3}, id="keypoints"),
    ],
)
def test_cpu_setup_warns_and_default_has_no_loss_callback(kwargs: dict[str, object]) -> None:
    """CPU compatibility leaves capture disabled and checkpoint tensor keys unchanged."""
    native = DetectionLitModule(0.5, 0.25, 64, 3, **kwargs)
    model = DetectionLitModule(0.5, 0.25, 64, 3, capture_loss=True, **kwargs)
    model._trainer = MagicMock(spec=Trainer, world_size=1)  # pose setup reads the attached trainer's world size
    assert native.configure_callbacks() == []
    assert native.state_dict().keys() == model.state_dict().keys()
    with pytest.warns(UserWarning, match="single-device CUDA"):
        model.setup("fit")
    assert not model._capture_loss_active


def test_cli_forwards_loss_option() -> None:
    """The shipped parser installs the lifecycle owner from its real model option."""
    cli = DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        run=False,
        args=[
            "--config",
            str(packaged_config("det_nano_smoke")),
            "--model.capture_loss",
            "true",
            "--trainer.accelerator",
            "cpu",
            "--trainer.logger",
            "false",
        ],
    )
    assert cli.model.hparams.capture_loss is True
    assert any(isinstance(callback, _LossCaptureLifecycle) for callback in cli.model.configure_callbacks())


@requires_cuda
def test_partial_batch_and_noncontiguous_inputs_keep_recording() -> None:
    """Unsupported signatures stay native without displacing the normal batch's graph."""
    holder, source = _LossCapture(), DualBranchLoss()
    holder.run(source, *_inputs(3)).total.backward()
    recording = holder._objective
    for original_inputs in (_inputs(1, batch=1), _inputs(2)):
        inputs = original_inputs
        if inputs[0].shape[0] == 2:
            values = list(inputs)
            values[0] = values[0].transpose(1, 2).contiguous().transpose(1, 2).detach().requires_grad_()
            inputs = tuple(values)
            assert not inputs[0].is_contiguous()
        actual = holder.run(source, *inputs)
        _same(_native(source, inputs), actual)
        actual.total.backward()
        assert holder._objective is recording
    holder.run(source, *_inputs(2)).total.backward()
    assert holder._objective is recording


@requires_cuda
def test_old_bf16_runtime_keeps_native_path() -> None:
    """An unverified BF16 runtime never silently records the operation."""
    holder, source, inputs = _LossCapture(), DualBranchLoss(), _inputs(3, dtype=torch.bfloat16)
    with patch("lucid_yolo.ptl.loss_capture._BF16_SUPPORTED", False), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = holder.run(source, *inputs)
        _same(_native(source, inputs), actual)
    assert holder._objective is None


@requires_cuda
def test_recording_failure_never_publishes_workspace() -> None:
    """A failed recorder clears its buffers and propagates the actionable failure."""
    holder, source = _LossCapture(), DualBranchLoss()
    with (
        patch("torch.cuda.make_graphed_callables", side_effect=RuntimeError("injected recording failure")),
        pytest.raises(RuntimeError, match="injected recording failure"),
    ):
        holder.run(source, *_inputs(3))
    assert holder._objective is None and holder._workspace is None and holder._capacity == 0


@requires_cuda
def test_settings_and_implementations_changed_before_capture_stay_observable() -> None:
    """Custom criteria cannot be hidden behind the built-in loss recording."""

    class CustomLoss(DualBranchLoss):
        def __call__(self, *args: object, **kwargs: object) -> DualLossOutput:
            return super().__call__(*args, **kwargs)

    holder, source, inputs = _LossCapture(), CustomLoss(), _inputs(3)
    with pytest.warns(UserWarning, match="customized"):
        actual = holder.run(source, *inputs)
    _same(_native(source, inputs), actual)
    assert holder._objective is None and holder._disabled


@requires_cuda
def test_public_fit_records_loss_and_releases_owner() -> None:
    """A real automatic Trainer activates replay and clears it after fit completion."""
    seen: list[int] = []

    class Probe(Callback):
        def on_train_batch_end(
            self, trainer: Trainer, pl_module: DetectionLitModule, outputs: object, batch: object, batch_idx: int
        ) -> None:
            assert pl_module._loss_capture._objective is not None
            seen.append(pl_module._loss_capture._capacity)

    batches = [
        (
            torch.rand(2, 3, 64, 64),
            [Targets(boxes=torch.tensor([[4.0, 4.0, 28.0, 28.0]]), labels=torch.tensor([1])) for _ in range(2)],
        )
        for _ in range(3)
    ]
    loader = DataLoader(batches, batch_size=None)
    model = DetectionLitModule(0.5, 0.25, 64, 3, capture_loss=True, warmup_epochs=0.0)
    trainer = Trainer(
        accelerator="cuda",
        devices=1,
        max_steps=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[Probe()],
    )
    trainer.fit(model, train_dataloaders=loader)
    assert len(seen) == 3 and all(capacity >= 1 for capacity in seen)
    assert model._loss_capture._objective is None and model._loss_capture._workspace is None


@requires_cuda
def test_custom_callable_without_assignment_attributes_keeps_native_route() -> None:
    """A replacement criterion need not expose the built-in criterion's private fields."""
    source = DualBranchLoss()
    holder, inputs = _LossCapture(), _inputs(2)
    criterion = source.__call__
    with pytest.warns(UserWarning, match="customized"):
        actual = holder.run(criterion, *inputs)
    _same(_native(source, inputs), actual)
    assert holder._objective is None


@requires_cuda
@pytest.mark.parametrize("record_first", [False, True])
def test_replaced_public_assigner_entry_stays_observable(record_first: bool) -> None:
    """Recording must not bypass a public assignment override before or after setup."""
    holder, source = _LossCapture(), DualBranchLoss()
    if record_first:
        holder.run(source, *_inputs(3)).total.backward()
    original = TaskAlignedAssigner.__call__
    calls: list[int] = []

    def observed(assigner: TaskAlignedAssigner, *args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(assigner, *args, **kwargs)

    with patch.object(TaskAlignedAssigner, "__call__", observed), pytest.warns(UserWarning, match="customized"):
        actual = holder.run(source, *_inputs(2))
    assert len(calls) == 2
    _same(_native(source, _inputs(2)), actual)
    assert holder._objective is None and holder._disabled


@requires_cuda
@pytest.mark.parametrize("collection_enabled", [False, True])
def test_recording_suspends_collection_and_restores_previous_mode(collection_enabled: bool) -> None:
    """Old graph cycles cannot be destroyed inside capture; the caller retains its GC policy."""
    was_enabled = gc.isenabled()
    recorder = torch.cuda.make_graphed_callables
    seen: list[bool] = []

    def record(*args: object, **kwargs: object) -> object:
        seen.append(gc.isenabled())
        return recorder(*args, **kwargs)

    gc.enable() if collection_enabled else gc.disable()
    try:
        holder = _LossCapture()
        with patch("torch.cuda.make_graphed_callables", record):
            holder.run(DualBranchLoss(), *_inputs(3)).total.backward()
        assert seen == [False]
        assert gc.isenabled() is collection_enabled
        holder.clear()
    finally:
        gc.enable() if was_enabled else gc.disable()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_fallback_preserves_keyword_only_custom_criterion(device: str) -> None:
    """A custom loss receives strides by keyword, as it does through the native module."""
    source, holder, inputs = DualBranchLoss(), _LossCapture(), _inputs(2, device=device)

    def criterion(*predictions_and_targets: torch.Tensor, strides: torch.Tensor, alpha: torch.Tensor) -> DualLossOutput:
        return source(*predictions_and_targets, strides=strides, alpha=alpha)

    if device == "cuda":
        with pytest.warns(UserWarning, match="customized"):
            actual = holder.run(criterion, *inputs)
    else:
        actual = holder.run(criterion, *inputs)
    _same(_native(source, inputs), actual)
    assert holder._objective is None


@requires_cuda
@pytest.mark.parametrize("record_first", [False, True])
def test_replaced_empty_assignment_stays_observable(record_first: bool) -> None:
    """Masked capacity cannot hide an override of the native empty-target result."""
    holder, source = _LossCapture(), DualBranchLoss()
    if record_first:
        holder.run(source, *_inputs(3)).total.backward()
    original = TaskAlignedAssigner._empty_result
    calls: list[int] = []

    def observed(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    with (
        patch.object(TaskAlignedAssigner, "_empty_result", staticmethod(observed)),
        pytest.warns(UserWarning, match="customized"),
    ):
        actual = holder.run(source, *_inputs(0))
    assert len(calls) == 2
    _same(_native(source, _inputs(0)), actual)
    assert holder._objective is None and holder._disabled


@requires_cuda
@pytest.mark.parametrize("record_first", [False, True])
@pytest.mark.parametrize(
    "count,index,value",
    [(0, 1, float("nan")), (0, 0, float("inf")), (3, 1, float("nan")), (3, 5, float("nan"))],
)
def test_nonfinite_inputs_preserve_native_outputs(count: int, index: int, value: float, record_first: bool) -> None:
    """Non-finite values cannot turn masked capacity into extra NaN contributions.

    The cases are the inputs that meet padded rows: predictions and target boxes, plus the
    empty-target route. Unpadded inputs replay by contract and are covered by
    ``test_host_unpadded_nonfinite_inputs_replay_with_native_bits``.
    """
    holder, source = _LossCapture(), DualBranchLoss()
    if record_first:
        holder.run(source, *_inputs(3)).total.backward()
    recording = holder._objective
    inputs = _inputs(count)
    with torch.no_grad():
        inputs[index].reshape(-1)[0] = value
    expected = _native(source, inputs)
    _same(expected, _native(source, inputs))
    with patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("non-finite capture")):
        actual = holder.run(source, *inputs)
    _same(expected, actual)
    assert holder._objective is recording and not holder._pending_backward
    if recording is not None:
        holder.run(source, *_inputs(2)).total.backward()
        assert holder._objective is recording


@requires_cuda
@pytest.mark.parametrize("record_first", [False, True])
def test_empty_targets_keep_native_finite_overflow_behavior(record_first: bool) -> None:
    """Finite decoded coordinates can still overflow area arithmetic on an empty batch."""
    holder, source = _LossCapture(), DualBranchLoss()
    if record_first:
        holder.run(source, *_inputs(3)).total.backward()
    recording, inputs = holder._objective, _inputs(0)
    with torch.no_grad():
        inputs[1][0, 0] = torch.tensor([0.0, -2e38, 0.0, 2e38], device="cuda")
    assert all(bool(value.isfinite().all()) for value in inputs)
    expected = _native(source, inputs)
    _same(expected, _native(source, inputs))
    assert bool(expected.total.isfinite())
    with patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("empty capture")):
        actual = holder.run(source, *inputs)
    _same(expected, actual)
    assert holder._objective is recording and not holder._pending_backward


@requires_cuda
@pytest.mark.parametrize("record_first", [False, True])
@pytest.mark.parametrize(
    "registration",
    [
        "register_module_forward_hook",
        "register_module_forward_pre_hook",
        "register_module_full_backward_hook",
        "register_module_full_backward_pre_hook",
    ],
)
def test_global_module_hooks_keep_native_criterion(registration: str, record_first: bool) -> None:
    """The auxiliary recorder must not introduce hook calls absent from the plain criterion."""
    holder, source = _LossCapture(), DualBranchLoss()
    if record_first:
        holder.run(source, *_inputs(3)).total.backward()
    recording, calls = holder._objective, []

    def observe(*args: object) -> None:
        calls.append(1)

    with getattr(torch.nn.modules.module, registration)(observe):
        inputs = _inputs(3)
        expected = _native(source, inputs)
        actual = holder.run(source, *inputs)
        _same(expected, actual)
        actual.total.backward()
    assert calls == [] and holder._objective is recording
    holder.run(source, *_inputs(2)).total.backward()
    assert holder._objective is not None


@requires_cuda
@pytest.mark.parametrize("mode", ["accumulation_clip", "overflow", "resume_assign", "network"])
def test_real_trainer_accumulation_amp_and_resume(mode: str, tmp_path: Path) -> None:
    """Real AMP, accumulation, clipping, network replay and resume keep native state bits."""

    def run(capture: bool) -> dict[str, object]:
        seed_everything(103, workers=True)
        batches = [
            (
                torch.rand(2, 3, 64, 64),
                [Targets(boxes=torch.tensor([[4.0, 4.0, 28.0, 28.0]]), labels=torch.tensor([1])) for _ in range(2)],
            )
            for _ in range(3)
        ]
        loader = DataLoader(batches, batch_size=None)
        seen: list[int] = []

        class Probe(Callback):
            injected = False

            def on_after_backward(self, trainer: Trainer, pl_module: DetectionLitModule) -> None:
                if mode == "overflow" and not self.injected:
                    parameter = next(value for value in pl_module.parameters() if value.grad is not None)
                    parameter.grad.fill_(float("inf"))
                    self.injected = True

            def on_train_batch_end(
                self, trainer: Trainer, pl_module: DetectionLitModule, outputs: object, batch: object, batch_idx: int
            ) -> None:
                if mode == "network":
                    assert pl_module._network_capture._network is not None
                if capture:
                    assert pl_module._loss_capture._objective is not None
                    seen.append(pl_module._loss_capture._capacity)

        def make_model() -> DetectionLitModule:
            return DetectionLitModule(
                0.5,
                0.25,
                64,
                3,
                capture_loss=capture,
                capture_network=mode == "network",
                capture_updates=True,
                capture_orthogonalization=True,
                warmup_epochs=0.0,
            )

        def make_trainer(epochs: int) -> tuple[Trainer, Probe]:
            probe = Probe()
            trainer = Trainer(
                accelerator="cuda",
                devices=1,
                plugins=[MixedPrecision("16-mixed", "cuda", scaler=torch.amp.GradScaler("cuda", init_scale=1.0))],
                max_epochs=epochs,
                accumulate_grad_batches=2,
                gradient_clip_val=1.0,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                limit_val_batches=0,
                num_sanity_val_steps=0,
                callbacks=[probe],
            )
            return trainer, probe

        model = make_model()
        trainer, probe = make_trainer(1)
        trainer.fit(model, train_dataloaders=loader)
        if mode == "overflow":
            assert probe.injected and trainer.precision_plugin.scaler.get_scale() == 0.5
        if mode == "resume_assign":
            checkpoint_path = tmp_path / f"{capture}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = make_model()
            model.load_state_dict(saved["state_dict"], assign=True)
            trainer, _ = make_trainer(2)
            trainer.fit(model, train_dataloaders=loader, ckpt_path=checkpoint_path)
        assert trainer.optimizers[0].state  # An all-skipped run cannot prove recovery.
        if capture:
            assert len(seen) == (6 if mode == "resume_assign" else 3)
            assert model._loss_capture._objective is None
        assert model._network_capture._network is None
        torch.cuda.synchronize()
        return {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(trainer.optimizers[0].state_dict()),
            "scaler": copy.deepcopy(trainer.precision_plugin.scaler.state_dict()),
        }

    reference = run(False)
    _same_training_state(reference, run(False))
    _same_training_state(reference, run(True))


def _same_training_state(left: object, right: object) -> None:
    """Compare nested training state with tensor-byte equality.

    Examples:
        ```pycon
        >>> _same_training_state({"value": torch.tensor([1.0]), "count": 2}, {"value": torch.tensor([1.0]), "count": 2})

        ```
    """
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        _bits(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same_training_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _same_training_state(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize("branch", ["o2m_assigner", "o2o_assigner"])
def test_constructor_valid_tiny_epsilon_keeps_native_loss(device: str, branch: str) -> None:
    """A positive Python epsilon that vanishes in float32 must not justify padded replay."""
    source, holder, inputs = DualBranchLoss(), _LossCapture(), _inputs(3, device=device)
    assigner = type(getattr(source, branch))(topk=getattr(source, branch).topk, eps=1e-50)
    setattr(source, branch, assigner)
    # Real, nondegenerate boxes have finite IoU without the epsilon; a padded
    # zero-area row would add 0/0 when the prediction also has zero area.
    with torch.no_grad():
        inputs[1].zero_()
        inputs[3].zero_()
    expected = _native(source, inputs)
    assert bool(torch.isfinite(expected.total))
    if device == "cuda":
        with pytest.warns(UserWarning, match="customized"):
            actual = holder.run(source, *inputs)
    else:
        actual = holder.run(source, *inputs)
    _same(expected, actual)
    assert holder._objective is None


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=requires_cuda)])
def replay_device(request: pytest.FixtureRequest) -> str:
    """Run a case through the stubbed CPU recorder, and again through real CUDA graph replay."""
    if request.param == "cpu":
        request.getfixturevalue("_host_recorder")
    return str(request.param)


def _replay_matches_native(
    holder: _LossCapture, source: DualBranchLoss, inputs: tuple[torch.Tensor, ...]
) -> DualLossOutput:
    """Replay one batch and compare every output bit and prediction gradient with the native route.

    Examples:
        >>> _replay_matches_native.__name__
        '_replay_matches_native'
    """
    reference = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in inputs)
    expected = _native(source, reference)
    actual = holder.run(source, *inputs)
    _same(expected, actual)
    expected.total.backward()
    actual.total.backward()
    assert not holder._pending_backward
    for left, right in zip(inputs[:4], reference[:4], strict=True):
        assert left.grad is not None and right.grad is not None
        _bits(left.grad, right.grad)
    return actual


def _crowded(count: int, device: str) -> tuple[torch.Tensor, ...]:
    """Make targets covering more anchors than the one-to-many top-k, with zero-overlap ties.

    Every target covers at least twelve of the sixteen anchors, against ``topk=10``.
    Five anchors per image overlap each target; the rest predict a far box, so their
    alignment is exactly zero and the tenth slot falls inside that tie. The right
    edge of every second target lies exactly on an anchor column (edge-inclusive).

    Examples:
        >>> _crowded(4, "cpu")[5][0, :, 2].tolist()
        [32.0, 28.0, 27.0, 32.0]
    """
    values = list(_inputs(count, device=device))
    gt = torch.zeros_like(values[5])
    gt[..., 2] = torch.tensor([32.0, 28.0, 27.0], device=device)[torch.arange(count, device=device) % 3]
    gt[..., 3] = 32.0
    values[5] = gt
    for index in (1, 3):
        boxes = torch.tensor([100.0, 100.0, 101.0, 101.0], device=device).expand(values[index].shape).clone()
        boxes[:, :5] = torch.tensor([0.0, 0.0, 30.0, 30.0], device=device)
        boxes[:, :5, 2:] += 2.0 * torch.arange(5, device=device).unsqueeze(-1)
        values[index] = boxes.requires_grad_()
    return tuple(values)


def _emptied(count: int, device: str, rows: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    """Remove every real target from the listed images, leaving the others populated.

    Examples:
        >>> _emptied(3, "cpu", (1,))[7].tolist()
        [[True, True, False], [False, False, False]]
    """
    values = list(_inputs(count, device=device))
    values[6], values[7] = values[6].clone(), values[7].clone()
    values[6][list(rows)], values[7][list(rows)] = -1, False
    return tuple(values)


def test_replay_follows_live_branch_weight(replay_device: str) -> None:
    """One recording serves any branch weight, because ``alpha`` is a live input.

    ``_inputs`` always carries 0.37, so a recording that baked the first weight in
    would pass every fixed-weight case. Same predictions and targets, three weights.
    """
    holder, source, totals = _LossCapture(), DualBranchLoss(), []
    for weight in (0.25, 0.5, 0.9):
        inputs = _inputs(3, device=replay_device)
        inputs = (*inputs[:9], torch.tensor(weight, dtype=torch.float64, device=replay_device))
        totals.append(float(_replay_matches_native(holder, source, inputs).total.detach()))
        recording = holder._objective
        assert recording is not None
    assert len(set(totals)) == 3
    assert holder._objective is recording and not holder._disabled


def test_replay_matches_native_top_k_truncation_and_ties(replay_device: str) -> None:
    """Truncated candidate sets and zero-overlap ties rank the same under padded capacity.

    More candidates than top-k, a tie across the top-k boundary and a fall from
    capacity 128 to three real rows are the cases where a padded row could displace a
    real anchor.
    """
    holder, source = _LossCapture(), DualBranchLoss()
    for count in (100, 3):
        actual = _replay_matches_native(holder, source, _crowded(count, replay_device))
        assert bool(torch.isfinite(actual.total))
        assert bool((actual.o2m_assign.fg_mask.sum(-1) < 16).all())
    assert holder._capacity == 128


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param((1,), id="second-image"),
        pytest.param((0,), id="first-image"),
        pytest.param((0, 1), id="every-image"),
    ],
)
def test_images_without_targets_match_native(replay_device: str, rows: tuple[int, ...]) -> None:
    """An image without real targets beside a populated one, or every mask false with N > 0.

    The native route keeps no foreground for those images; masked capacity must neither
    add one nor change the populated image's assignment, weights or gradients.
    """
    holder, source = _LossCapture(), DualBranchLoss()
    for count in (2, 5, 3, 9):
        actual = _replay_matches_native(holder, source, _emptied(count, replay_device, rows))
        for assignment in (actual.o2m_assign, actual.o2o_assign):
            assert not assignment.fg_mask[list(rows)].any()
            assert not assignment.align_weights[list(rows)].any()
    assert holder._objective is not None and holder._capacity == 16


@contextmanager
def _inference_with_grad() -> Iterator[None]:
    """Enter inference mode with grad mode switched back on, isolating the inference check.

    Examples:
        >>> with _inference_with_grad():
        ...     print(torch.is_inference_mode_enabled(), torch.is_grad_enabled())
        True True
    """
    with torch.inference_mode(), torch.enable_grad():
        yield


@pytest.mark.parametrize(
    "context",
    [
        pytest.param(torch.no_grad, id="no-grad"),
        pytest.param(torch.inference_mode, id="inference-mode"),
        pytest.param(_inference_with_grad, id="inference-mode-with-grad"),
        pytest.param(lambda: torch.autograd.set_detect_anomaly(True), id="anomaly-detection"),
        pytest.param(lambda: patch.object(torch.compiler, "is_compiling", return_value=True), id="compiling"),
    ],
)
def test_eligible_rejects_unsupported_context_before_unpacking_inputs(context: object) -> None:
    """Autograd and compiler contexts exit before any input is read.

    A single-element tuple would raise on the later unpacking line were the early exit
    skipped, so a clean ``False`` proves that the context clause itself fired.
    """
    with context():
        assert capture_module._eligible((torch.zeros(1),)) is False


def test_eligible_unpacks_inputs_without_unsupported_context() -> None:
    """Control for the context cases: outside them, the same tuple does reach unpacking."""
    with pytest.raises(ValueError, match="not enough values to unpack"):
        capture_module._eligible((torch.zeros(1),))


def _swap(index: int, transform: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[tuple[torch.Tensor, ...]], tuple]:
    """Return a change that rewrites one replay input and keeps the rest.

    Examples:
        >>> _swap(0, lambda value: value + 1)((torch.zeros(1), torch.zeros(1)))[0].tolist()
        [1.0]
    """

    def change(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        values = list(inputs)
        values[index] = transform(values[index])
        return tuple(values)

    return change


@requires_cuda
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_swap(0, torch.Tensor.detach), id="frozen-match-logits"),
        pytest.param(_swap(1, torch.Tensor.detach), id="frozen-match-boxes"),
        pytest.param(_swap(2, torch.Tensor.detach), id="frozen-unique-logits"),
        pytest.param(_swap(3, torch.Tensor.detach), id="frozen-unique-boxes"),
        pytest.param(_swap(4, torch.Tensor.requires_grad_), id="anchor-points-require-grad"),
        pytest.param(_swap(5, torch.Tensor.requires_grad_), id="target-boxes-require-grad"),
        pytest.param(_swap(5, lambda value: value.double()), id="target-boxes-float64"),
        pytest.param(_swap(5, lambda value: value.transpose(1, 2).contiguous().transpose(1, 2)), id="noncontiguous"),
        pytest.param(_swap(6, lambda value: value.int()), id="labels-int32"),
        pytest.param(_swap(6, lambda value: value[:, :-1].contiguous()), id="labels-shape"),
        pytest.param(_swap(7, lambda value: value.to(torch.uint8)), id="mask-uint8"),
        pytest.param(_swap(9, lambda value: value.float()), id="alpha-float32"),
        pytest.param(_swap(9, lambda value: value.reshape(1)), id="alpha-vector"),
        pytest.param(_swap(9, lambda value: value.cpu()), id="alpha-on-host"),
    ],
)
def test_eligible_rejects_unsupported_inputs(change: Callable[[tuple[torch.Tensor, ...]], tuple]) -> None:
    """Frozen predictions, trainable targets, another dtype, rank or device keep the native route."""
    inputs = _inputs(3)
    assert capture_module._eligible(inputs) is True
    assert capture_module._eligible(change(inputs)) is False


@requires_cuda
@pytest.mark.parametrize("name,value", [("current_device", 99), ("is_current_stream_capturing", True)])
def test_eligible_rejects_other_current_device_and_stream_capture(name: str, value: object) -> None:
    """A current device other than the predictions' device, or an open capture, keeps native."""
    inputs = _inputs(3)
    with patch.object(torch.cuda, name, return_value=value):
        assert capture_module._eligible(inputs) is False


@requires_cuda
def test_other_stream_keeps_native_route_and_recording() -> None:
    """A recording made on one stream is not replayed on another."""
    holder, source = _LossCapture(), DualBranchLoss()
    holder.run(source, *_inputs(3)).total.backward()
    recording, inputs = holder._objective, _inputs(2)
    other = torch.cuda.Stream()
    other.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(other), patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("stream")):
        actual = holder.run(source, *inputs)
        expected = _native(source, inputs)
    torch.cuda.synchronize()
    _same(expected, actual)
    assert holder._objective is recording and not holder._pending_backward
