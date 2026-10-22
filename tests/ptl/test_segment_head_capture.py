# SPDX-License-Identifier: Apache-2.0
"""Prediction replay preserves native gradients, including unused mask parameters."""

from __future__ import annotations

import copy
import dataclasses
import pickle
import re
import warnings
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from markers import requires_cuda
from pytorch_lightning import Callback, Trainer
from pytorch_lightning import __version__ as lightning_version
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.strategies import SingleDeviceStrategy
from torch.utils.data import DataLoader

from lucid_yolo.cli.train import DetectionCLI, packaged_config
from lucid_yolo.data.targets import Targets
from lucid_yolo.models.build import SegmentOutput
from lucid_yolo.models.heads.detect import DualDetectionHead, _build_box_stem, _DetectionBranch
from lucid_yolo.optim.update_replay import _RecordedSpans
from lucid_yolo.ptl import DetectionLitModule
from lucid_yolo.ptl.callbacks import EMACallback
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.network_capture import _network_signature
from lucid_yolo.ptl.segment_head_capture import (
    _BF16_REPLAY_SUPPORTED,
    _AttributePlan,
    _head_breach,
    _members,
    _parameter_breach,
    _PredictionGroup,
    _RecordedGroup,
    _SegmentHeadCapture,
    _SegmentHeadCaptureLifecycle,
    _stage_breach,
    _storage_breach,
    _tensor_signature,
)


@pytest.fixture(autouse=True)
def _deterministic(deterministic_kernels: None) -> None:
    """Use the public-data comparison's deterministic kernel policy."""


def _module(**kwargs: object) -> DetectionLitModule:
    """Build the small real segmentation architecture, preserving constructor validation.

    Examples:
        >>> _module(capture_segment_heads=True).hparams.capture_segment_heads
        True
    """
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=4, task="segment", **kwargs)


def _attach(module: DetectionLitModule) -> None:
    """Establish the runtime conditions a real CUDA trainer supplies.

    Examples:
        >>> callable(_attach)
        True
    """
    trainer = MagicMock(spec=Trainer)
    trainer.training = True
    trainer.accelerator = CUDAAccelerator()
    trainer.strategy = SingleDeviceStrategy(device=torch.device("cuda"))
    trainer.accumulate_grad_batches = 1
    module._trainer = trainer
    module.setup("fit")


def _bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare dtype, shape and bytes, including signed zero and NaN payloads.

    Examples:
        >>> _bits(torch.tensor([0.0]), torch.tensor([-0.0]))
        False
    """
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            right.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def _forward_segmentation_output(model: DetectionLitModule, images: torch.Tensor) -> SegmentOutput:
    """Run the real forward pass the replay certificate must reproduce exactly.

    Examples:
        >>> callable(_forward_segmentation_output)
        True
    """
    return model.forward_segmentation(images)


def _objective_from_output(output: SegmentOutput, masks_used: bool) -> torch.Tensor:
    """Build the squared-error objective from an already-computed forward output.

    A squared term is even: it cannot distinguish a prediction from its sign-flip,
    and the chain rule propagates that same blindness to every upstream gradient.
    Callers that need to detect a corrupted replay output compare the raw tensors
    in ``output`` directly (:func:`_assert_predictions_match`), never only this loss.

    Examples:
        >>> callable(_objective_from_output)
        True
    """
    detect = output.detect
    loss = sum(x.square().mean() for x in (detect.o2m_cls, detect.o2m_box, detect.o2o_cls, detect.o2o_box))
    assert output.semantic is not None
    loss = loss + output.semantic.square().mean()
    if masks_used:
        assert detect.o2m_coeff is not None and detect.o2o_coeff is not None
        loss = loss + detect.o2m_coeff.square().mean() + detect.o2o_coeff.square().mean()
        loss = loss + output.prototypes.square().mean()
    return loss


def _objective(model: DetectionLitModule, images: torch.Tensor, masks_used: bool) -> torch.Tensor:
    """Read detection/semantic predictions and optionally the instance-mask outputs.

    Examples:
        >>> callable(_objective)
        True
    """
    return _objective_from_output(_forward_segmentation_output(model, images), masks_used)


def _assert_predictions_match(native: SegmentOutput, replay: SegmentOutput, masks_used: bool) -> None:
    """Compare every raw prediction tensor directly, bypassing the squared objective's sign blindness.

    Examples:
        >>> callable(_assert_predictions_match)
        True
    """
    pairs = [
        (native.detect.o2m_cls, replay.detect.o2m_cls),
        (native.detect.o2m_box, replay.detect.o2m_box),
        (native.detect.o2o_cls, replay.detect.o2o_cls),
        (native.detect.o2o_box, replay.detect.o2o_box),
        (native.semantic, replay.semantic),
    ]
    if masks_used:
        pairs.extend(
            [
                (native.detect.o2m_coeff, replay.detect.o2m_coeff),
                (native.detect.o2o_coeff, replay.detect.o2o_coeff),
                (native.prototypes, replay.prototypes),
            ]
        )
    for left, right in pairs:
        assert left is not None and right is not None
        assert _bits(left, right)


def test_predictions_match_catches_sign_flip_squared_objective_cannot() -> None:
    """A squared loss is even, so a sign-flipped prediction leaves it and every upstream gradient unchanged.

    ``x.square()`` equals ``(-x).square()``, and the chain rule carries that same
    cancellation back through any parameter the flipped tensor depends on: a replay
    bug that returns ``-prediction`` instead of ``prediction`` would pass both the
    loss comparison and :func:`_assert_state_and_gradients` undetected. Only a direct
    comparison of the raw prediction tensors, as :func:`_assert_predictions_match`
    performs, can see it.
    """
    output = _forward_segmentation_output(_module(), torch.randn(1, 3, 64, 64))
    flipped = dataclasses.replace(output, detect=dataclasses.replace(output.detect, o2m_cls=-output.detect.o2m_cls))
    assert _bits(_objective_from_output(output, True), _objective_from_output(flipped, True))
    with pytest.raises(AssertionError):
        _assert_predictions_match(output, flipped, True)


def _assert_state_and_gradients(left: DetectionLitModule, right: DetectionLitModule) -> None:
    """Check every original parameter/buffer and gradient presence independently.

    Examples:
        >>> callable(_assert_state_and_gradients)
        True
    """
    assert left.state_dict().keys() == right.state_dict().keys()
    for name, tensor in left.state_dict().items():
        assert _bits(tensor, right.state_dict()[name]), name
    other = dict(right.named_parameters())
    for name, parameter in left.named_parameters():
        gradient, candidate = parameter.grad, other[name].grad
        assert (gradient is None) == (candidate is None), name
        if gradient is not None:
            assert candidate is not None and _bits(gradient, candidate), name


def test_default_and_cpu_fallback_preserve_checkpoint_keys() -> None:
    """An off or CPU request owns no recording and never calls the CUDA API."""
    native, replay = _module(), _module(capture_segment_heads=True)
    assert native.configure_callbacks() == []
    assert len(replay.configure_callbacks()) == 1
    assert native.state_dict().keys() == replay.state_dict().keys()
    with patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("CPU must stay native")):
        with pytest.warns(UserWarning, match="single-device CUDA"):
            replay.setup("fit")
        assert not replay._segment_head_capture.active
        with torch.no_grad():
            assert replay.eval().forward_segmentation(torch.zeros(2, 3, 64, 64)).semantic is None


@pytest.mark.parametrize(
    "kwargs",
    [pytest.param({"compile_step": True}, id="compile-step"), pytest.param({"task": "detect"}, id="detect-task")],
)
def test_incompatible_configuration_is_explicit(kwargs: dict[str, object]) -> None:
    """The request is never silently interpreted as another task's replay option.

    The refusal comes from the shared replay-option validator, before any stage is built
    or a hyperparameter recorded, as for the sibling capture flags. Its message names why
    the compiler is refused while the segment objective still runs eager.
    """
    options = {"depth": 0.34, "width": 0.25, "max_channels": 64, "num_classes": 4, "task": "segment"}
    options.update(kwargs)
    with (
        patch("lucid_yolo.ptl.module.build_detection_stages", side_effect=AssertionError("validated after build")),
        pytest.raises(
            ValueError, match=r"capture_segment_heads=True requires .* keeps its meaning once it is compiled"
        ),
    ):
        DetectionLitModule(**options, capture_segment_heads=True)


def test_copy_and_pickle_drop_all_transient_state() -> None:
    """Neither serialization nor EMA-style copying retains a recording or active trainer decision."""
    holder = _SegmentHeadCapture(True)
    holder.disabled = True
    holder.active = True
    holder.owner = torch.nn.Linear(2, 2)
    for restored in (copy.deepcopy(holder), pickle.loads(pickle.dumps(holder))):
        assert restored.requested and not restored.active and not restored.disabled
        assert restored.owner is None and restored.entries == {}


def test_later_retire_cause_stays_silent() -> None:
    """Only the first cause of a fit is reported; the holder is already native afterwards.

    A second retire in the same fit would repeat a warning about a state nothing can
    change any more, so it must not warn at all. Matches
    ``test_network_capture_guards.py::TestRetireWarning`` for the sibling detection-network
    recorder's identical disabled-guard contract.
    """
    holder = _SegmentHeadCapture(True)
    with pytest.warns(UserWarning, match=r"^capture_segment_heads=True stays eager: first cause$"):
        holder.retire("first cause")
    assert holder.disabled
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        holder.retire("second cause")


def test_empty_features_take_native_path_without_retiring() -> None:
    """An empty feature map sends that one call native; it does not retire replay for the fit.

    A zero-element batch has nothing to record, so ``runner`` declines the call and the native head
    runs. The holder must stay enabled, or every later full batch of the fit would silently lose replay.
    The CUDA stream-capture query is patched because a CPU-only build cannot answer it.
    """
    module = _module(capture_segment_heads=True)
    _attach(module)
    holder = module._segment_head_capture
    empty = (torch.empty(0, 8, 4, 4),) * 3
    with warnings.catch_warnings(), patch("torch.cuda.is_current_stream_capturing", return_value=False):
        warnings.simplefilter("error")
        assert holder.runner(module, empty) is None
    assert holder.active
    assert not holder.disabled


@pytest.mark.parametrize(
    ("alter", "reason"),
    [
        pytest.param(
            lambda module: setattr(module.head, "forward", module.head.forward),
            "the detection head overrides its native dispatch",
            id="head-dispatch",
        ),
        pytest.param(lambda module: None, "the features are not ordinary dense CUDA tensors", id="cpu-features"),
    ],
)
def test_runner_refusal_names_cause_and_retires(alter: Callable[[DetectionLitModule], object], reason: str) -> None:
    """Head and feature refusals reach the user as the warning's cause and keep the fit native.

    A user who turned the request on and sees it stay eager needs the reason in the warning, and
    every later call of the fit must skip the certificate. These two rows pin how ``runner`` wires a head
    and a feature refusal into ``retire``; the other causes are asserted where the contributor's guard tests
    reach them. Both are decided before any CUDA work, so a CPU module reaches them. The non-BF16 cause is not
    listed: ``_feature_breach`` returns on its first check for any non-CUDA tensor.
    """
    module = _module(capture_segment_heads=True)
    _attach(module)
    alter(module)
    holder = module._segment_head_capture
    features = (torch.zeros(1, 8, 4, 4),) * 3
    with (
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("refusal must not record")),
        pytest.warns(UserWarning, match=rf"^capture_segment_heads=True stays eager: {re.escape(reason)}$") as record,
    ):
        result = holder.runner(module, features)
    assert result is None
    assert holder.disabled and holder.entries == {} and holder.owner is None
    assert record[0].filename == __file__


def test_unreadable_precision_policy_retires_with_torch_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A precision query torch refuses retires with torch's own error text after the cause.

    Losing that text leaves a user unable to tell a mixed legacy/new precision API from any other reason the
    prediction groups went native. Matches the sibling recorder's
    ``test_network_capture_guards.py::TestPrecisionPolicy`` contract.
    """

    def refuse() -> tuple[object, ...]:
        raise RuntimeError("mix of the legacy and new APIs")

    monkeypatch.setattr("lucid_yolo.ptl.segment_head_capture._precision_signature", refuse)
    module = _module(capture_segment_heads=True)
    _attach(module)
    holder = module._segment_head_capture
    features = (torch.zeros(1, 8, 4, 4),) * 3
    with (
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        pytest.warns(UserWarning, match=r"stays eager: the CUDA precision policy cannot be read \(mix of the legacy"),
    ):
        result = holder.runner(module, features)
    assert result is None
    assert holder.disabled


@pytest.mark.parametrize("event", ["on_fit_start", "on_fit_end", "on_exception"])
def test_fit_boundaries_clear_holder(event: str) -> None:
    """Successful, failed and repeated fits all discard the previous eligibility decision."""
    module = _module(capture_segment_heads=True)
    module._segment_head_capture.disabled = True
    module._segment_head_capture.owner = module.head
    args = (MagicMock(spec=Trainer), module)
    getattr(_SegmentHeadCaptureLifecycle(), event)(*args, *([RuntimeError("abort")] if event == "on_exception" else []))
    assert module._segment_head_capture.owner is None
    assert not module._segment_head_capture.disabled


@pytest.mark.parametrize("method", ["forward", "forward_branch"])
def test_head_dispatch_override_uses_native_route(method: str) -> None:
    """A scheduling keyword cannot bypass a class or instance head override."""
    model = _module()
    assert _head_breach(model.head) is None
    original = getattr(model.head, method)
    with patch.object(model.head, method, original):
        assert _head_breach(model.head) is not None
    with patch.object(DualDetectionHead, method, lambda *args, **kwargs: None):
        assert _head_breach(model.head) is not None


@pytest.mark.parametrize("method", ["forward", "_base_predictions", "_coefficients", "_angles", "_keypoints"])
def test_branch_dispatch_override_uses_native_route(method: str) -> None:
    """Any branch method the recorded composition would bypass is certified too."""
    model = _module()
    original = getattr(model.head.o2m, method)
    with patch.object(model.head.o2m, method, original):
        assert _head_breach(model.head) is not None
    with patch.object(_DetectionBranch, method, lambda *args, **kwargs: None):
        assert _head_breach(model.head) is not None


@requires_cuda
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(None, id="no-autocast"),
        pytest.param(torch.float16, id="float16"),
        pytest.param(torch.bfloat16, id="bfloat16"),
    ],
)
@pytest.mark.parametrize("first_masks", [False, True])
def test_cuda_outputs_gradients_updates_and_unused_masks_are_exact(
    dtype: torch.dtype | None, first_masks: bool
) -> None:
    """Real graph replay keeps every gradient byte and unused-mask optimizer state unchanged."""
    torch.manual_seed(17)
    native = _module().cuda().to(memory_format=torch.channels_last).train()
    replay = _module(capture_segment_heads=True).cuda().to(memory_format=torch.channels_last).train()
    replay.load_state_dict(native.state_dict())
    _attach(native)
    _attach(replay)
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) for model in (native, replay)]
    for index, masks_used in enumerate((first_masks, not first_masks, first_masks)):
        images = torch.randn(2, 3, 64, 64, device="cuda").contiguous(memory_format=torch.channels_last)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        outputs = []
        losses = []
        for model in (native, replay):
            for entry in model._segment_head_capture.entries.values():
                entry.pending_backward = False
            with torch.autocast("cuda", dtype=dtype or torch.float16, enabled=dtype is not None, cache_enabled=False):
                output = _forward_segmentation_output(model, images)
                loss = _objective_from_output(output, masks_used)
            loss.backward()
            outputs.append(output)
            losses.append(loss)
        _assert_predictions_match(outputs[0], outputs[1], masks_used)
        assert _bits(*losses), index
        _assert_state_and_gradients(native, replay)
        for optimizer in optimizers:
            optimizer.step()
        _assert_state_and_gradients(native, replay)
        if not masks_used:
            assert all(p.grad is None for p in replay.protonet.parameters())
            assert all(p.grad is None for p in replay.head.o2m.coeff_stems.parameters())
    expected = 16 if dtype == torch.bfloat16 and _BF16_REPLAY_SUPPORTED else 0
    assert len(replay._segment_head_capture.entries) == expected
    copied = copy.deepcopy(replay)
    assert copied.state_dict().keys() == replay.state_dict().keys()
    assert copied._segment_head_capture.entries == {}
    replay._segment_head_capture.clear()


def test_recorded_group_run_replays_when_nothing_forces_native() -> None:
    """``run`` calls the graphed adapter, not ``.eager``, once the recorded signature matches.

    The CUDA tests above only assert ``len(entries) == 16`` and bit-identical output and
    gradients. Both still hold if this method's native/pending/signature guard were broken
    so every reuse silently fell back to ``.eager`` -- that branch is the exact native
    computation, so it cannot diverge from the native arm it is compared against. This
    isolates the guard with a mock adapter so a future regression that always, or never,
    takes the fallback fails here, on CPU, without a recorded CUDA graph.
    """
    group = _RecordedGroup.__new__(_RecordedGroup)
    group.pending_backward = False
    group.signature = "matching"
    group.spans = MagicMock(reached_by=MagicMock(return_value=False))
    group.adapter = MagicMock(return_value=(torch.zeros(1),))
    inputs = (object(),)
    with patch("lucid_yolo.ptl.segment_head_capture._input_signature", return_value="matching"):
        group.run(inputs)
    group.adapter.assert_called_once_with(*inputs)
    group.adapter.eager.assert_not_called()


def test_recorded_group_run_stays_native_when_backward_pending() -> None:
    """``run`` calls ``.eager``, not the graphed adapter, while a prior backward is pending."""
    group = _RecordedGroup.__new__(_RecordedGroup)
    group.pending_backward = True
    group.signature = "matching"
    group.spans = MagicMock(reached_by=MagicMock(return_value=False))
    group.adapter = MagicMock(eager=MagicMock(return_value=(torch.zeros(1),)))
    inputs = (object(),)
    with patch("lucid_yolo.ptl.segment_head_capture._input_signature", return_value="matching"):
        group.run(inputs)
    group.adapter.eager.assert_called_once_with(inputs)
    group.adapter.assert_not_called()


def test_recorded_group_output_without_backward_node_never_holds() -> None:
    """A replay of a fully frozen group has no backward to wait for, so nothing is held.

    The mocked adapter returns a tensor with no ``grad_fn``, so no backward hook can ever register and
    release the group; holding it would force every later forward of the fit native.
    """
    group = _RecordedGroup.__new__(_RecordedGroup)
    group.pending_backward = False
    group.signature = "matching"
    group.spans = MagicMock(reached_by=MagicMock(return_value=False))
    group.adapter = MagicMock(return_value=(torch.ones(3),))
    with patch("lucid_yolo.ptl.segment_head_capture._input_signature", return_value="matching"):
        group.run((object(),))
    assert not group.pending_backward


def _assert_close_or_both_none(left: torch.Tensor | None, right: torch.Tensor | None, name: str) -> None:
    """Require both tensors absent, or both present and equal within ``torch.testing`` tolerance.

    Examples:
        >>> _assert_close_or_both_none(None, None, "x")
        >>> _assert_close_or_both_none(torch.ones(1), torch.ones(1), "x")
    """
    assert (left is None) == (right is None), name
    if left is not None and right is not None:
        torch.testing.assert_close(left, right, msg=name)


@pytest.mark.parametrize("masks_used", [False, True])
def test_inactive_request_matches_native_train_forward_and_gradients(masks_used: bool) -> None:
    """A flag-on module that stays inactive on CPU computes the native forward and gradients.

    Both modules start from one state dict and see the same images in train mode. The request owns no recording,
    so every prediction tensor, every gradient and the presence of an unused mask parameter's gradient must match.
    ``_attach`` is not used: it would activate the holder.
    """
    torch.manual_seed(5)
    native, replay = _module().train(), _module(capture_segment_heads=True).train()
    replay.load_state_dict(native.state_dict())
    with pytest.warns(UserWarning, match="single-device CUDA"):
        replay.setup("fit")
    images = torch.randn(2, 3, 64, 64)
    outputs = [_forward_segmentation_output(model, images) for model in (native, replay)]
    for output in outputs:
        _objective_from_output(output, masks_used).backward()
    pairs = [
        (getattr(o.detect, f.name), getattr(r.detect, f.name), f.name)
        for o, r in [outputs]
        for f in dataclasses.fields(o.detect)
    ]
    pairs += [
        (outputs[0].prototypes, outputs[1].prototypes, "prototypes"),
        (outputs[0].semantic, outputs[1].semantic, "semantic"),
    ]
    pairs += [(p.grad, dict(replay.named_parameters())[n].grad, n) for n, p in native.named_parameters()]
    assert not replay._segment_head_capture.active
    for left, right, name in pairs:
        _assert_close_or_both_none(left, right, name)


def _segmentation_loader() -> torch.utils.data.DataLoader[tuple[torch.Tensor, list[Targets]]]:
    """Build a two-step loader of tiny synthetic segmentation batches with one rectangle per image.

    Examples:
        >>> len(_segmentation_loader())
        2
    """
    ring = torch.tensor([[4.0, 4.0], [28.0, 4.0], [28.0, 28.0], [4.0, 28.0]])
    batches = [
        (
            torch.randn(2, 3, 64, 64),
            [
                Targets(boxes=torch.tensor([[4.0, 4.0, 28.0, 28.0]]), labels=torch.tensor([1]), polygons=[ring])
                for _ in range(2)
            ],
        )
        for _ in range(2)
    ]
    return torch.utils.data.DataLoader(batches, batch_size=None)


def test_cpu_fit_wires_lifecycle_and_warns_once_without_replay() -> None:
    """A real CPU ``Trainer.fit`` attaches the lifecycle callback, warns once and never activates replay.

    Every other test drives ``MagicMock(spec=Trainer)`` and calls the hooks by hand. Here Lightning's own setup
    and callback dispatch run for two steps: the request is refused once at setup, each hook fires, and the
    holder stays inactive and empty. Replay itself needs CUDA and is covered by the ``gpu`` tests.
    """
    module = _module(capture_segment_heads=True)
    trainer = Trainer(
        accelerator="cpu",
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    batch_start, fit_end = _SegmentHeadCaptureLifecycle.on_train_batch_start, _SegmentHeadCaptureLifecycle.on_fit_end
    with (
        patch.object(
            _SegmentHeadCaptureLifecycle, "on_train_batch_start", autospec=True, side_effect=batch_start
        ) as starts,
        patch.object(_SegmentHeadCaptureLifecycle, "on_fit_end", autospec=True, side_effect=fit_end) as ends,
        warnings.catch_warnings(record=True) as record,
    ):
        warnings.simplefilter("always")
        trainer.fit(module, train_dataloaders=_segmentation_loader())
    refusals = [w for w in record if "capture_segment_heads=True is ignored" in str(w.message)]
    holder = module._segment_head_capture
    assert len(refusals) == 1
    assert any(isinstance(callback, _SegmentHeadCaptureLifecycle) for callback in trainer.callbacks)
    assert (starts.call_count, ends.call_count) == (2, 1)
    assert not holder.active and not holder.disabled and holder.entries == {}


def test_stage_helpers_and_shared_storage_are_not_certified() -> None:
    """The certificate covers callable helpers and byte overlap, including different Parameters."""
    model = _module()
    owner = torch.nn.ModuleList([model.head, model.proto_fusion, model.protonet, model.semantic])
    assert _stage_breach(owner) is None
    conv = model.head.o2m.box_stems[0][-1]
    with patch.object(conv, "_conv_forward", conv._conv_forward):
        assert _stage_breach(owner) is not None
    assert _storage_breach(owner, torch.device("cpu")) is None
    other = model.head.o2o.box_stems[0][-1]
    other.bias.data = conv.bias.data
    assert other.bias is not conv.bias
    assert _storage_breach(owner, torch.device("cpu")) == "prediction parameters or buffers share bytes"


def test_stage_forward_hook_names_local_cause() -> None:
    """A forward hook on a prediction stage is reported as a stage hook, not as a process-wide one.

    The contributor's hook test only checks that the word ``hook`` occurs, which a global-hook message
    satisfies as well, so it cannot tell the two causes apart.
    """
    model = _module()
    owner = torch.nn.ModuleList([model.head, model.protonet, model.semantic])
    model.head.o2m.box_stems[0][-1].register_forward_hook(lambda *_: None)
    assert _stage_breach(owner) == "a prediction stage carries a forward or backward hook"


def test_global_module_hook_is_reported_before_any_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process-wide module hook stays observable, so no stage is replayed while one is registered."""
    monkeypatch.setitem(torch.nn.modules.module._global_forward_hooks, 0, lambda *_: None)
    assert _stage_breach(torch.nn.ModuleList([torch.nn.Identity()])) == "a global module hook is registered"


def test_unlisted_stage_type_is_named_not_reported_as_forward_override() -> None:
    """A layer type outside the native stage list is reported by its class name.

    ``nn.Identity`` has no entry in the stage table. The refusal must say the type is unlisted, rather than
    claiming the layer overrides a forward it never had, so a maintainer knows to list or avoid the type.
    """
    owner = torch.nn.ModuleList([torch.nn.Identity()])
    assert _stage_breach(owner) == "a prediction stage of type Identity is not in the native stage list"


@requires_cuda
@pytest.mark.skipif(not _BF16_REPLAY_SUPPORTED, reason="BF16 replay runtime required")
def test_cuda_partial_batch_uses_native_without_another_recording() -> None:
    """A partial image signature neither overwrites the full recording nor grows its cache."""
    torch.manual_seed(23)
    native = _module().cuda().to(memory_format=torch.channels_last).train()
    replay = _module(capture_segment_heads=True).cuda().to(memory_format=torch.channels_last).train()
    replay.load_state_dict(native.state_dict())
    _attach(native)
    _attach(replay)
    for batch in (2, 1, 2):
        images = torch.randn(batch, 3, 64, 64, device="cuda").contiguous(memory_format=torch.channels_last)
        for model in (native, replay):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                loss = _objective(model, images, True)
            loss.backward()
        _assert_state_and_gradients(native, replay)
        assert len(replay._segment_head_capture.entries) == 16
    replay._segment_head_capture.clear()


@requires_cuda
@pytest.mark.skipif(not _BF16_REPLAY_SUPPORTED, reason="BF16 replay runtime required")
@pytest.mark.parametrize("change", ["hook", "fusion_hook", "storage", "negative_view", "helper", "frozen"])
def test_cuda_contract_changes_retire_and_keep_native_observability(change: str) -> None:
    """Hooks, storage changes, helper overrides and frozen parameters cannot reuse stale graphs."""
    model = _module(capture_segment_heads=True).cuda().to(memory_format=torch.channels_last).train()
    _attach(model)
    conv = model.head.o2m.box_stems[0][-1]
    complex_weight = None
    if change == "negative_view":
        complex_weight = torch.complex(torch.zeros_like(conv.weight), conv.weight.detach().clone())
        conv.weight.data = complex_weight.imag
    images = torch.randn(2, 3, 64, 64, device="cuda").contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        loss = _objective(model, images, True)
    loss.backward()
    assert len(model._segment_head_capture.entries) == 16
    model.zero_grad(set_to_none=True)
    calls = []
    handle = None
    conv = model.head.o2m.box_stems[0][-1]
    if change == "hook":
        handle = model.head.register_forward_hook(lambda *args: calls.append("head"))
    elif change == "fusion_hook":
        handle = model.proto_fusion.register_forward_hook(lambda *args: calls.append("fusion"))
    elif change == "storage":
        conv.weight.data = conv.weight.detach().clone()
    elif change == "negative_view":
        assert complex_weight is not None
        pointer, strides = conv.weight.data_ptr(), conv.weight.stride()
        conv.weight.data = complex_weight.conj().imag
        assert conv.weight.is_neg() and conv.weight.data_ptr() == pointer and conv.weight.stride() == strides
    elif change == "frozen":
        conv.weight.requires_grad_(False)
    else:
        original = conv._conv_forward

        def forward_helper(*args: object) -> torch.Tensor:
            """Keep an overridden helper observable on the native route.

            Examples:
                >>> callable(forward_helper)
                True
            """
            calls.append("helper")
            return original(*args)

        conv._conv_forward = forward_helper
    try:
        with (
            patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("must not re-record")),
            pytest.warns(UserWarning, match="capture_segment_heads=True stays eager"),
        ):
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                loss = _objective(model, images, True)
            loss.backward()
        assert model._segment_head_capture.disabled and model._segment_head_capture.entries == {}
        if change in ("hook", "fusion_hook", "helper"):
            assert calls
    finally:
        if handle is not None:
            handle.remove()
        model._segment_head_capture.clear()


@requires_cuda
def test_cuda_fit_replays_through_real_optimizer_callbacks_and_validation() -> None:
    """Replay the real segmentation objective across validation under automatic MuSGD and EMA.

    Observe actual CUDA graph launches after the first recording and again after
    validation, so cached entries alone cannot make a permanent native fallback
    pass. Other recorder options stay off to isolate the prediction graphs.

    Examples:
        >>> callable(test_cuda_fit_replays_through_real_optimizer_callbacks_and_validation)
        True
    """
    torch.manual_seed(0)
    seen: list[int] = []
    launches: list[int] = []
    validations: list[int] = []
    replay_count = [0]
    validation_count = [0]
    native_replay = torch.cuda.CUDAGraph.replay

    def _replay(graph: torch.cuda.CUDAGraph) -> None:
        """Count a real launch without changing its execution.

        Examples:
            >>> callable(_replay)
            True
        """
        replay_count[0] += 1
        native_replay(graph)

    class _EntryProbe(Callback):
        """Observe training after a completed validation epoch and its EMA weight swap."""

        def on_train_batch_end(
            self, trainer: Trainer, pl_module: DetectionLitModule, outputs: object, batch: object, batch_idx: int
        ) -> None:
            """Retain recording counts and completed graph launches for each real optimizer step."""
            seen.append(len(pl_module._segment_head_capture.entries))
            launches.append(replay_count[0])
            validations.append(validation_count[0])

        def on_validation_epoch_end(self, trainer: Trainer, pl_module: DetectionLitModule) -> None:
            """Count the validation epoch before training resumes."""
            validation_count[0] += 1

    def _batch() -> tuple[torch.Tensor, list[Targets], list[torch.Tensor]]:
        """Provide valid instance polygons and explicit prototype-grid masks.

        Examples:
            >>> callable(_batch)
            True
        """
        images = torch.rand(2, 3, 64, 64)
        ring = torch.tensor([[4.0, 4.0], [28.0, 4.0], [28.0, 28.0], [4.0, 28.0]])
        targets = [
            Targets(boxes=torch.tensor([[4.0, 4.0, 28.0, 28.0]]), labels=torch.tensor([1]), polygons=[ring.clone()])
            for _ in range(2)
        ]
        masks = [torch.zeros(1, 16, 16) for _ in range(2)]
        for mask in masks:
            mask[:, 1:7, 1:7] = 1.0
        return images, targets, masks

    train_loader = DataLoader([_batch(), _batch()], batch_size=None)
    val_loader = DataLoader([_batch()], batch_size=None)
    model = _module(
        capture_segment_heads=True, capture_orthogonalization=False, capture_updates=False, warmup_epochs=0.0
    )
    initial_weight = next(model.parameters()).detach().clone()
    trainer = Trainer(
        accelerator="cuda",
        devices=1,
        precision="bf16-mixed",
        max_epochs=2,
        deterministic=True,
        benchmark=False,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[EMACallback(), _EntryProbe()],
    )
    with (
        patch("torch.cuda.CUDAGraph.replay", _replay),
        patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record,
    ):
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    assert trainer.state.finished
    assert any(isinstance(callback, _SegmentHeadCaptureLifecycle) for callback in trainer.callbacks)
    assert trainer.optimizers[0].state
    assert not torch.equal(initial_weight, next(model.parameters()).detach().cpu())
    assert validations == [0, 0, 1, 1]
    assert validation_count[0] == 2
    expected = 16 if _BF16_REPLAY_SUPPORTED else 0
    assert seen == [expected] * 4
    assert record.call_count == expected
    if _BF16_REPLAY_SUPPORTED:
        assert len(launches) == 4 and launches[0] > 0
        assert all(before < after for before, after in pairwise(launches))
    else:
        assert launches == [0] * 4
    assert model._segment_head_capture.entries == {}


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_recipe_keeps_head_and_optimizer_options_independent(enabled: bool) -> None:
    """The public segmentation recipe forwards the new request without changing optimizer flags."""
    cli = DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        run=False,
        args=[
            "--config",
            str(packaged_config("seg_nano_smoke")),
            "--model.capture_segment_heads",
            str(enabled).lower(),
            "--model.capture_updates",
            "true",
            "--model.capture_orthogonalization",
            "true",
            "--trainer.accelerator",
            "cpu",
            "--trainer.logger",
            "false",
        ],
    )
    assert cli.model.hparams.capture_segment_heads is enabled
    assert cli.model.hparams.capture_updates and cli.model.hparams.capture_orthogonalization
    assert len(cli.model.configure_callbacks()) == int(enabled)


def test_old_checkpoint_loads_with_native_default(tmp_path: Path) -> None:
    """A checkpoint predating the option loads strictly with unchanged keys and no recording."""
    model = _module()
    hyperparameters = dict(model.hparams)
    del hyperparameters["capture_segment_heads"]
    path = tmp_path / "older.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": hyperparameters,
            "pytorch-lightning_version": lightning_version,
        },
        path,
    )
    restored = DetectionLitModule.load_from_checkpoint(path)
    assert not restored.hparams.capture_segment_heads
    assert restored._segment_head_capture.entries == {}
    _assert_state_and_gradients(model, restored)


def test_current_members_match_torch_enumeration_and_are_walked_once() -> None:
    """Shared checks consume one fresh topology walk without re-entering recursive tensor enumeration."""
    model = _module()
    owner = torch.nn.ModuleList([model.head, model.protonet, model.semantic])
    modules, parameters, buffers = _members(owner)
    assert tuple(map(id, modules)) == tuple(map(id, owner.modules()))
    assert tuple(map(id, parameters)) == tuple(map(id, owner.parameters()))
    assert tuple(map(id, buffers)) == tuple(map(id, owner.buffers()))
    assert _network_signature(owner)[1] == _tensor_signature(parameters + buffers)
    with (
        patch.object(owner, "modules", wraps=owner.modules) as walk,
        patch.object(owner, "parameters", side_effect=AssertionError("duplicate parameter walk")),
        patch.object(owner, "buffers", side_effect=AssertionError("duplicate buffer walk")),
    ):
        members = _members(owner)
        _tensor_signature(members[1] + members[2])
        _parameter_breach(members[1], members[2], torch.device("cpu"))
        _stage_breach(owner, members[0])
    assert walk.call_count == 1
    model.head.register_parameter("extra", torch.nn.Parameter(torch.zeros(1)))
    assert len(_members(owner)[1]) == len(parameters) + 1


@pytest.mark.parametrize("change", ["scalar", "list", "parameter_name", "buffer_name", "extra_attribute"])
def test_attribute_certificate_detects_mutations_outside_tensor_metadata(change: str) -> None:
    """Equal tensor storage never certifies scalar bindings, list contents or registration names."""
    model = _module()
    owner = torch.nn.ModuleList([model.head, model.protonet, model.semantic])
    conv = model.head.o2m.box_stems[0][-1]
    conv.padding_scalar = 0.0
    conv._reversed_padding_repeated_twice = list(conv._reversed_padding_repeated_twice)
    modules = tuple(owner.modules())
    certificate = _AttributePlan(modules)
    assert certificate.matches(modules)
    if change == "scalar":
        conv.padding_scalar = -0.0
    elif change == "list":
        conv._reversed_padding_repeated_twice[0] += 1
    elif change == "parameter_name":
        conv._parameters["renamed_bias"] = conv._parameters.pop("bias")
    elif change == "buffer_name":
        norm = next(layer for layer in model.head.modules() if isinstance(layer, torch.nn.BatchNorm2d))
        norm._buffers["renamed_count"] = norm._buffers.pop("num_batches_tracked")
    else:
        conv.new_attribute = True
    assert not certificate.matches(tuple(owner.modules()))


def test_attribute_certificate_keeps_live_buffer_values_and_mode_roundtrip() -> None:
    """The certificate rereads structure while in-place running-state updates stay live."""
    model = _module()
    owner = torch.nn.ModuleList([model.head, model.protonet, model.semantic])
    certificate = _AttributePlan(tuple(owner.modules()))
    buffer = next(model.head.buffers())
    before = buffer.clone()
    buffer.add_(1)
    assert not torch.equal(before, buffer)
    assert certificate.matches(tuple(owner.modules()))
    model.eval()
    assert not certificate.matches(tuple(owner.modules()))
    model.train()
    assert certificate.matches(tuple(owner.modules()))


@requires_cuda
@pytest.mark.skipif(not _BF16_REPLAY_SUPPORTED, reason="BF16 replay runtime required")
def test_cuda_second_forward_keeps_first_backward_activations() -> None:
    """A second forward stays native until the first graph's backward consumes its private activations."""
    torch.manual_seed(29)
    native = _module().cuda().to(memory_format=torch.channels_last).train()
    replay = _module(capture_segment_heads=True).cuda().to(memory_format=torch.channels_last).train()
    replay.load_state_dict(native.state_dict())
    _attach(native)
    _attach(replay)
    images = [torch.randn(2, 3, 64, 64, device="cuda").contiguous(memory_format=torch.channels_last) for _ in range(2)]
    losses = []
    for model in (native, replay):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            first = _objective(model, images[0], True)
            second = _objective(model, images[1], True)
            loss = first + second
        loss.backward()
        losses.append(loss)
    assert _bits(*losses)
    _assert_state_and_gradients(native, replay)
    assert len(replay._segment_head_capture.entries) == 16
    replay._segment_head_capture.clear()


def test_lazy_negative_parameter_view_is_not_certified_with_unchanged_storage() -> None:
    """A logical sign change keeps tensor identity and bytes while changing the native result."""
    layer = torch.nn.Conv2d(1, 1, 3, bias=False)
    complex_weight = torch.complex(torch.zeros_like(layer.weight), torch.ones_like(layer.weight))
    layer.weight.data = complex_weight.imag
    owner = torch.nn.ModuleList([layer])
    modules, parameters, buffers = _members(owner)
    attributes = _AttributePlan(modules)
    signature = _tensor_signature(parameters + buffers)
    assert _parameter_breach(parameters, buffers, torch.device("cpu")) is None
    image = torch.ones(1, 1, 5, 5)
    before = layer(image)
    layer.weight.data = complex_weight.conj().imag
    assert layer.weight.is_neg()
    assert _tensor_signature(parameters + buffers) == signature and attributes.matches(modules)
    assert torch.equal(layer(image), -before)
    assert _parameter_breach(parameters, buffers, torch.device("cpu")) is not None


@requires_cuda
def test_recorded_group_synchronizes_before_buffer_restore() -> None:
    """A warmup failure synchronizes the device before the BatchNorm snapshot is copied back.

    A prediction group warms up on its own side stream via
    ``torch.cuda.make_graphed_callables``, the same primitive
    :class:`~lucid_yolo.ptl.network_capture._NetworkCapture` uses. Without the
    drain, the restore on the current stream can race BatchNorm writes the
    failed warmup left queued; this is the same hazard
    ``test_network_capture_guards.py::TestFailedRecording`` covers for the
    sibling detection-network recording, exercised here for a prediction group.
    """
    stage = _build_box_stem(16).cuda().train()
    images = torch.randn(1, 16, 8, 8, device="cuda")
    adapter = _PredictionGroup((stage,), lambda inputs: (stage(inputs[0]),))
    spans = _RecordedSpans(list(stage.parameters()) + list(stage.buffers()), [images])
    order: list[str] = []

    def fail(network: torch.nn.Module, _samples: object) -> None:
        network(images)
        raise RuntimeError("injected warmup failure")

    def synchronize(device: object = None) -> None:
        order.append("synchronize")
        torch.cuda.current_stream().synchronize()

    buffer = next(stage.buffers())
    before = buffer.detach().clone()
    original_copy = buffer.copy_

    def restore(*args: object, **kwargs: object) -> torch.Tensor:
        order.append("restore")
        return original_copy(*args, **kwargs)

    with (
        patch("torch.cuda.make_graphed_callables", side_effect=fail),
        patch("torch.cuda.synchronize", side_effect=synchronize),
        patch.object(buffer, "copy_", side_effect=restore),
        pytest.raises(RuntimeError, match="injected warmup failure"),
    ):
        _RecordedGroup(adapter, (images,), spans)
    assert order[:2] == ["synchronize", "restore"]
    assert torch.equal(buffer, before)
