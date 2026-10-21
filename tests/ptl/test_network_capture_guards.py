# SPDX-License-Identifier: Apache-2.0
"""Network-capture guards name their cause and keep eager execution exact.

Every guard that sends a training forward back to eager execution is exercised here
on CPU wherever the guard is decided before any CUDA work: the retire warning, the
per-cause reasons and the recorded-contract drift checks. Behaviour that only exists
with a real CUDA recording is ``gpu``-marked and skipped without a device.
"""

from __future__ import annotations

import copy
import dataclasses
import re
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import Trainer

from lucid_yolo.ptl import DetectionLitModule
from lucid_yolo.ptl.network_capture import (
    _detection_stages,
    _DetectionNetwork,
    _FeatureNetwork,
    _ineligibility,
    _NetworkCapture,
    _NetworkCaptureLifecycle,
)


def _module() -> DetectionLitModule:
    """Build a small CPU detection network with the normal dual head.

    Examples:
        >>> _module().task
        'detect'
    """
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=4)


class TestRetireWarning:
    """The single retire warning keeps its stable prefix and names the broken contract."""

    def test_names_cause_after_prefix(self) -> None:
        """The warning reads as the shared prefix followed by the caller's reason.

        Existing callers match ``stays eager``; a user reading the warning needs the cause
        that follows it, so both halves are pinned exactly.
        """
        holder = _NetworkCapture()
        with pytest.warns(UserWarning, match=r"^capture_network=True stays eager: the input images require gradients$"):
            holder.retire("the input images require gradients")
        assert holder._disabled

    def test_later_cause_stays_silent(self) -> None:
        """Only the first cause of a fit is reported; the holder is already eager afterwards.

        A second retire in the same fit would repeat a warning about a state nothing can
        change any more, so it must not warn at all.
        """
        holder = _NetworkCapture()
        with pytest.warns(UserWarning, match="first cause"):
            holder.retire("first cause")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            holder.retire("second cause")

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(
                lambda holder, model, images: holder.run(images, (model.backbone, model.neck, model.head), MagicMock()),
                id="detection",
            ),
            pytest.param(
                lambda holder, model, images: holder.run_features(images, (model.backbone, model.neck)),
                id="pose-features",
            ),
        ],
    )
    def test_location_is_recorder_caller(
        self, call: Callable[[_NetworkCapture, DetectionLitModule, torch.Tensor], object]
    ) -> None:
        """A retirement decided inside the shared recording path names the code that called the recorder.

        Both compositions retire inside one shared helper, a frame deeper than a direct
        retire. Pointing into the recorder instead would break a user's warning filter
        keyed on the calling module; CPU stages retire there on their storage.
        """
        model = _module()
        with pytest.warns(UserWarning, match="not a strided CUDA tensor") as record:
            call(_NetworkCapture(), model, torch.zeros(1, 3, 64, 64))
        assert record[0].filename == __file__


class TestRunRetireReason:
    """A call that cannot replay names its own cause and still returns the eager result."""

    @pytest.mark.parametrize(
        ("requires_grad", "anomaly", "cause"),
        [
            pytest.param(True, False, "the input images require gradients", id="input-requires-grad"),
            pytest.param(False, True, "anomaly detection is enabled", id="anomaly-detection"),
            pytest.param(False, False, "not a strided CUDA tensor", id="cpu-stages"),
        ],
    )
    def test_warns_cause_and_runs_eager(self, requires_grad: bool, anomaly: bool, cause: str) -> None:
        """Each input-side or network-side cause appears after the prefix, and eager output returns.

        These guards are decided before any CUDA work, so a CPU network reaches them: the
        first two fire on the call itself, the last on the CPU-resident stages.
        """
        model = _module()
        holder = _NetworkCapture()
        images = torch.zeros(1, 3, 64, 64, requires_grad=requires_grad)
        eager = MagicMock(return_value="eager output")
        with (
            torch.autograd.set_detect_anomaly(anomaly),
            pytest.warns(UserWarning, match=f"^capture_network=True stays eager: .*{cause}"),
        ):
            result = holder.run(images, (model.backbone, model.neck, model.head), eager)
        assert result == "eager output"
        eager.assert_called_once_with(images)
        assert holder._disabled

    def test_bf16_cause_names_runtime_and_requirement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A BF16 batch below the verified runtime names the installed torch and the 2.9 floor.

        The gate compares release versions strictly, so 2.9 pre-release builds also land
        here; the warning has to say which version was seen for that to be diagnosable.
        """
        monkeypatch.setattr("lucid_yolo.ptl.network_capture._BF16_REPLAY_SUPPORTED", False)
        model = _module()
        holder = _NetworkCapture()
        images = torch.zeros(1, 3, 64, 64, dtype=torch.bfloat16)
        eager = MagicMock(return_value="eager output")
        expected = rf"stays eager: BF16 replay requires torch>=2\.9, .* torch {re.escape(torch.__version__)}$"
        with pytest.warns(UserWarning, match=expected):
            holder.run(images, (model.backbone, model.neck, model.head), eager)
        eager.assert_called_once_with(images)


class TestBatchNormEligibility:
    """A cumulative-average BatchNorm stays eager instead of aborting the recording."""

    @pytest.mark.parametrize(
        ("training", "cause"),
        [
            pytest.param(True, "BatchNorm with momentum=None", id="training"),
            pytest.param(False, "not a strided CUDA tensor", id="frozen"),
        ],
    )
    def test_momentum_none_retires_only_while_training(self, training: bool, cause: str) -> None:
        """Only a training BatchNorm that tracks running stats with ``momentum=None`` is refused.

        That combination reads ``num_batches_tracked`` on the host every forward, which
        CUDA stream capture rejects; a frozen BatchNorm reads nothing there, so it must
        reach the next check (here the CPU-storage one) rather than this reason.
        """
        model = _module()
        norm = next(module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d))
        norm.momentum = None
        norm.train(training)
        eager = MagicMock(return_value="eager output")
        with pytest.warns(UserWarning, match=f"stays eager: .*{cause}"):
            _NetworkCapture().run(torch.zeros(1, 3, 64, 64), (model.backbone, model.neck, model.head), eager)
        eager.assert_called_once()


class TestStageModuleBreach:
    """A recorded stage whose call runs code its class methods do not show is refused by cause."""

    @pytest.mark.parametrize(
        ("alter", "cause"),
        [
            pytest.param(
                lambda network: network.neck.register_forward_hook(lambda *_: None),
                "carries a forward or backward hook",
                id="hook",
            ),
            pytest.param(
                lambda network: vars(network.head).update(forward_branch=network.head.forward_branch),
                "overrides forward_branch on its instance",
                id="instance-forward-branch",
            ),
            pytest.param(lambda network: network.backbone.compile(), "is compiled", id="compiled"),
            pytest.param(
                lambda network: vars(network).update(forward=network.forward),
                "not a strided CUDA tensor",
                id="root-forward",
            ),
        ],
    )
    def test_names_cause(self, alter: Callable[[_DetectionNetwork], object], cause: str) -> None:
        """Each breach is named before the CPU-storage check, except an instance forward on the root.

        These guards are decided before any CUDA work, so CPU stages reach them. The graphed
        callable installs ``forward`` on the recorder's own root, so that override must fall
        through to the next check (here the CPU-resident storage) instead of retiring.
        """
        model = _module()
        network = _DetectionNetwork(model.backbone, model.neck, model.head)
        alter(network)
        assert cause in (_ineligibility(network) or "")

    def test_feature_reason_omits_head(self) -> None:
        """A pose feature recording holds no head, so its stage reason cannot send a user to one.

        The pose feature recording holds only backbone and neck; each head stem records
        separately and the head is judged by its certificate, so a reason naming the head
        would point at the wrong module.
        """
        model = _module()
        model.neck.register_forward_hook(lambda *_: None)
        reason = _ineligibility(_FeatureNetwork(model.backbone, model.neck))
        assert reason == "a recorded stage module carries a forward or backward hook"


class TestPrecisionPolicy:
    """The precision key reads the API torch accepts and reports a refused query verbatim."""

    def test_refused_query_text_reaches_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A policy query torch refuses retires with torch's own error text after the cause.

        Losing the text leaves a user unable to tell a mixed legacy/new precision API
        from any other reason the step went eager.
        """

        def refuse() -> tuple[object, ...]:
            raise RuntimeError("mix of the legacy and new APIs")

        monkeypatch.setattr("lucid_yolo.ptl.network_capture._precision_signature", refuse)
        model = _module()
        images = torch.zeros(1, 3, 64, 64)
        eager = MagicMock(return_value="eager output")
        with pytest.warns(UserWarning, match=r"cannot be read \(mix of the legacy and new APIs\)$"):
            result = _NetworkCapture().run(images, (model.backbone, model.neck, model.head), eager)
        assert result == "eager output"

    @pytest.mark.skipif(
        not hasattr(torch.backends.cuda.matmul, "fp32_precision"), reason="torch build lacks fp32_precision"
    )
    def test_new_api_policy_reads_and_distinguishes_values(self) -> None:
        """After the new ``fp32_precision`` API is used, the key still reads and tells TF32 from IEEE.

        Setting that API makes torch's legacy getters raise for the rest of the process,
        so the probe runs in a subprocess instead of poisoning this one.
        """
        probe = (
            "import torch\n"
            "from lucid_yolo.ptl.network_capture import _precision_signature\n"
            "torch.backends.cuda.matmul.fp32_precision = 'tf32'\n"
            "tf32 = _precision_signature()\n"
            "torch.backends.cuda.matmul.fp32_precision = 'ieee'\n"
            "print(tf32 != _precision_signature())\n"
        )
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "True"


_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class TestFailedRecording:
    """A recording that fails still drains queued device work before restoring buffers."""

    @pytest.mark.gpu
    @_cuda
    def test_synchronizes_before_buffer_restore(self) -> None:
        """A warmup failure synchronizes the device before the BatchNorm snapshot is copied back.

        The recorder warms up on its own side stream; without the drain, the restore on the
        current stream can race BatchNorm writes the failed warmup left queued.
        """
        model = _module().cuda()
        images = torch.randn(2, 3, 64, 64, device="cuda")
        order: list[str] = []

        def fail(network: torch.nn.Module, _inputs: object) -> None:
            network(images)
            raise RuntimeError("injected warmup failure")

        def synchronize(device: object = None) -> None:
            order.append("synchronize")
            torch.cuda.current_stream().synchronize()

        buffer = next(model.backbone.buffers())
        original_copy = buffer.copy_

        def copy(*args: object, **kwargs: object) -> torch.Tensor:
            order.append("restore")
            return original_copy(*args, **kwargs)

        with (
            patch("torch.cuda.make_graphed_callables", side_effect=fail),
            patch("torch.cuda.synchronize", side_effect=synchronize),
            patch.object(buffer, "copy_", side_effect=copy),
            pytest.raises(RuntimeError, match="injected warmup failure"),
        ):
            _NetworkCapture().run(images, (model.backbone, model.neck, model.head), model)
        assert order[:2] == ["synchronize", "restore"]


@pytest.fixture
def _deterministic_kernels() -> Iterator[None]:
    """Use deterministic kernels so eager and replayed gradients compare exactly."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


class TestReplayBackwardHold:
    """A replay holds later forwards eager until its own backward has read the static buffers."""

    def test_backward_releases_hold(self) -> None:
        """Waiting on an output with a backward node holds, and that node's backward releases it.

        The post-hook on the shared backward node is the observed boundary: once it fires,
        the static activations have been consumed and a new replay may overwrite them.
        """
        holder = _NetworkCapture()
        weight = torch.ones(3, requires_grad=True)
        output = weight * 2.0
        holder._await_backward((output,))
        held = holder._pending_backward
        output.sum().backward()
        assert (held, holder._pending_backward) == (True, False)

    def test_live_gradient_leaves_storage_before_backward(self) -> None:
        """A ``.grad`` alive when the replayed backward starts is copied out before that backward.

        The gradient tensor stands in for one adopted from the static gradient buffers:
        accumulating into it in place would be the silent doubling, so after the backward it
        must be untouched and the parameter must hold a separate accumulated copy.
        """
        network = _DetectionNetwork(torch.nn.Linear(3, 3), torch.nn.Identity(), torch.nn.Identity())
        weight = network.backbone.weight
        alias = torch.full_like(weight, 5.0)
        weight.grad = alias
        holder = _NetworkCapture()
        holder._network = network
        # `_run`'s capture block always sets this alongside `_network`; set it here too,
        # since `_before_replayed_backward` now walks this cached list instead of re-tracing
        # `network.parameters()` on every hold.
        holder._recorded_parameters = tuple(network.parameters())
        output = network.backbone(torch.ones(1, 3))
        holder._await_backward((output,))
        output.sum().backward()
        assert torch.equal(alias, torch.full_like(weight, 5.0))
        assert torch.equal(weight.grad, torch.full_like(weight, 6.0))

    def test_output_without_backward_node_never_holds(self) -> None:
        """A replay of a fully frozen network has no backward to wait for, so nothing is held."""
        holder = _NetworkCapture()
        holder._await_backward((torch.ones(3),))
        assert not holder._pending_backward

    def test_train_batch_start_drops_unconsumed_hold(self) -> None:
        """A new training batch releases a hold whose backward never ran.

        ``training_step`` may return ``None`` and skip the backward, which would otherwise
        keep every later forward of the fit eager.
        """
        model = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=4, capture_network=True)
        model._network_capture._pending_backward = True
        _NetworkCaptureLifecycle().on_train_batch_start(MagicMock(spec=Trainer), model, None, 0)
        assert not model._network_capture._pending_backward

    @pytest.mark.gpu
    @_cuda
    @pytest.mark.usefixtures("_deterministic_kernels")
    def test_two_forwards_before_backward_match_eager(self) -> None:
        """Two training forwards before one backward give the eager gradients exactly.

        Without the hold, the second call replays into the same static buffers and the
        single backward computes the first call's gradient from the second call's
        activations.
        """
        eager = _module().cuda()
        replay = copy.deepcopy(eager)
        holder = _NetworkCapture()
        first = torch.randn(2, 3, 64, 64, device="cuda")
        second = torch.randn(2, 3, 64, 64, device="cuda")
        stages = (replay.backbone, replay.neck, replay.head)
        eager_outputs = (eager(first), eager(second))
        replay_outputs = (holder.run(first, stages, replay), holder.run(second, stages, replay))
        sum(out.o2m_cls.square().mean() + out.o2o_box.square().mean() for out in eager_outputs).backward()
        sum(out.o2m_cls.square().mean() + out.o2o_box.square().mean() for out in replay_outputs).backward()
        for left, right in zip(eager.parameters(), replay.parameters(), strict=True):
            assert torch.equal(left.grad, right.grad)
        holder.clear()


class TestStageComposition:
    """The replayed tensor view and the module forward compose the stages identically."""

    def test_replay_view_matches_module_forward(self) -> None:
        """The tuple view rebuilt by field equals the module forward on every output field.

        This is the CPU half of the eager oracle: it catches a drifted composition or a
        reordered rebuild without needing a CUDA recording.
        """
        model = _module().eval()
        images = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            expected = model(images)
            actual = _DetectionNetwork.rebuild(_DetectionNetwork(model.backbone, model.neck, model.head)(images))
        for field in dataclasses.fields(expected):
            left, right = getattr(expected, field.name), getattr(actual, field.name)
            assert (left is None and right is None) or torch.equal(left, right), field.name

    def test_module_and_replay_view_share_composition(self) -> None:
        """Both forwards go through the one shared stage helper, so an edit cannot reach only one."""
        model = _module().eval()
        images = torch.rand(1, 3, 64, 64)
        with (
            patch("lucid_yolo.ptl.module._detection_stages", wraps=_detection_stages) as module_side,
            patch("lucid_yolo.ptl.network_capture._detection_stages", wraps=_detection_stages) as replay_side,
            torch.no_grad(),
        ):
            model(images)
            _DetectionNetwork(model.backbone, model.neck, model.head)(images)
        module_side.assert_called_once()
        replay_side.assert_called_once()


class TestRecordedContractDrift:
    """A change after recording names which part of the recorded contract moved."""

    @pytest.mark.parametrize(
        ("owners", "state", "precision", "cause"),
        [
            pytest.param((9, 2, 3), ("state",), ("precision",), "module was replaced", id="owners"),
            pytest.param((1, 2, 3), ("other state",), ("precision",), "storage, a module mode", id="state"),
            pytest.param((1, 2, 3), ("state",), ("other precision",), "precision policy changed", id="precision"),
        ],
    )
    def test_names_moved_part(
        self, owners: tuple[int, ...], state: tuple[object, ...], precision: tuple[object, ...], cause: str
    ) -> None:
        """Owners, storage and precision each produce their own reason against one recording."""
        holder = _NetworkCapture()
        holder._input, holder._state, holder._owners = ("shape", ("precision",)), ("state",), (1, 2, 3)
        assert cause in (holder._drift(state, owners, precision) or "")

    def test_unchanged_contract_has_no_reason(self) -> None:
        """The recorded owners, storage and precision produce no drift reason."""
        holder = _NetworkCapture()
        holder._input, holder._state, holder._owners = ("shape", ("precision",)), ("state",), (1, 2, 3)
        assert holder._drift(("state",), (1, 2, 3), ("precision",)) is None
