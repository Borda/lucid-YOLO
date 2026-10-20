# SPDX-License-Identifier: Apache-2.0
"""Exercise prediction-replay eligibility and lifetime decisions without a GPU.

CUDA graph execution is covered separately by the accelerator tests. These cases
use real CPU modules and tensors for the host-side contracts, with CUDA API
boundaries replaced only where needed to reach the decision under test.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from lucid_yolo.ptl import DetectionLitModule
from lucid_yolo.ptl import segment_head_capture as capture


def _module() -> DetectionLitModule:
    """Build the real segmentation stages with small channel counts.

    Examples:
        >>> _module().task
        'segment'
    """
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=4, task="segment")


@pytest.fixture
def _policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the CUDA-residency predicate when checking host-side feature policies.

    No tensor moves to CUDA and no graph is recorded. Layout, dtype, hooks and
    autograd metadata are still those of the real CPU tensors.

    Examples:
        >>> callable(_policy)
        True
    """
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(capture, "_BF16_REPLAY_SUPPORTED", True)


class TestReplayCallbackComposition:
    """The merged detection-loss callback and segmentation callback retain their own task scopes."""

    @pytest.mark.parametrize(
        ("task", "network", "loss", "heads", "expected"),
        [
            ("detect", False, False, False, []),
            ("detect", True, True, False, ["_NetworkCaptureLifecycle", "_LossCaptureLifecycle"]),
            ("segment", False, False, False, []),
            ("segment", False, False, True, ["_SegmentHeadCaptureLifecycle"]),
        ],
    )
    def test_keeps_each_requested_callback(
        self, task: str, network: bool, loss: bool, heads: bool, expected: list[str]
    ) -> None:
        """Resolving the shared callback method must not discard either independently enabled recorder."""
        model = DetectionLitModule(
            depth=0.34,
            width=0.25,
            max_channels=64,
            num_classes=4,
            task=task,
            capture_network=network,
            capture_loss=loss,
            capture_segment_heads=heads,
        )
        assert [type(callback).__name__ for callback in model.configure_callbacks()] == expected

    @pytest.mark.parametrize(("task", "cause"), [("detect", "capture_segment_heads"), ("segment", "capture_loss")])
    def test_incompatible_task_requests_are_still_explicit(self, task: str, cause: str) -> None:
        """Two independent options cannot bypass each other's task restrictions after integration."""
        with pytest.raises(ValueError, match=cause):
            DetectionLitModule(
                depth=0.34,
                width=0.25,
                max_channels=64,
                num_classes=4,
                task=task,
                capture_loss=True,
                capture_segment_heads=True,
            )


class TestStageContracts:
    """Observers and custom dispatch must stay visible through native execution."""

    @pytest.mark.parametrize("global_hook", [False, True])
    def test_hook_names_the_native_cause(self, global_hook: bool) -> None:
        """Both local and global observers prevent a recording from bypassing their calls."""
        owner = nn.ModuleList([nn.Conv2d(1, 1, 1)])
        if global_hook:
            handle = nn.modules.module.register_module_forward_hook(lambda *args: None)
        else:
            handle = owner[0].register_forward_pre_hook(lambda *args: None)
        try:
            assert "hook" in capture._stage_breach(owner)
        finally:
            handle.remove()
        assert capture._stage_breach(owner) is None

    @pytest.mark.parametrize("change", ["eval", "forward", "compiled", "helper", "cumulative"])
    def test_refuses_an_observable_stage_change(self, change: str) -> None:
        """Each rejected policy names its cause rather than silently recording a different call."""
        stage = nn.BatchNorm2d(2) if change == "cumulative" else nn.Conv2d(2, 2, 1)
        owner = nn.ModuleList([stage])
        if change == "eval":
            stage.eval()
            cause = "outside training mode"
        elif change == "forward":
            stage.forward = stage.forward
            cause = "forward implementation"
        elif change == "compiled":
            stage._compiled_call_impl = lambda *args: None
            cause = "forward implementation"
        elif change == "helper":
            stage._conv_forward = stage._conv_forward
            cause = "forward helper"
        else:
            stage.momentum = None
            cause = "cumulative-average"
        assert cause in capture._stage_breach(owner)


class TestStorageContracts:
    """Live parameter and buffer metadata is independent of the attribute certificate."""

    @pytest.mark.parametrize("change", ["empty", "frozen", "dtype", "hook", "post_hook", "buffer_grad"])
    def test_parameter_contract_names_the_breach(self, change: str) -> None:
        """A mutable tensor contract must be read again even when its Python binding is unchanged."""
        parameter = nn.Parameter(torch.ones(2))
        buffer = torch.ones(2)
        parameters = (parameter,)
        handle = None
        if change == "empty":
            parameters = ()
            cause = "plain strided storage"
        elif change == "frozen":
            parameter.requires_grad_(False)
            cause = "frozen"
        elif change == "dtype":
            parameter.data = parameter.double()
            cause = "not float32"
        elif change == "hook":
            handle = parameter.register_hook(lambda gradient: gradient)
            cause = "gradient hook"
        elif change == "post_hook":
            handle = parameter.register_post_accumulate_grad_hook(lambda value: None)
            cause = "gradient hook"
        else:
            buffer.requires_grad_(True)
            cause = "buffer requires gradients"
        try:
            assert cause in capture._parameter_breach(parameters, (buffer,), torch.device("cpu"))
        finally:
            if handle is not None:
                handle.remove()

    @pytest.mark.parametrize("change", ["empty", "device", "alias"])
    def test_storage_refuses_unrecordable_bytes(self, change: str) -> None:
        """Zero-sized, misplaced and shared parameter storage cannot establish independent recordings."""
        owner = nn.Module()
        owner.register_parameter("weight", nn.Parameter(torch.ones(2)))
        device = torch.device("cpu")
        if change == "empty":
            owner.register_buffer("empty", torch.empty(0))
        elif change == "device":
            device = torch.device("meta")
        else:
            owner.register_parameter("alias", owner.weight)
        reason = capture._storage_breach(owner, device)
        assert reason is not None
        assert ("share bytes" if change == "alias" else "empty or spans") in reason


class TestFeaturePolicy:
    """Feature-side policy decisions depend on real layout, dtype and autograd metadata."""

    def test_cpu_features_keep_the_native_route(self) -> None:
        """A real CPU tensor is rejected before CUDA-specific metadata or runtime checks."""
        reason = capture._feature_breach((torch.ones(2, dtype=torch.bfloat16),), (True, torch.bfloat16))
        assert "ordinary dense CUDA tensors" in reason

    @pytest.mark.usefixtures("_policy")
    @pytest.mark.parametrize("change", ["valid", "leaf", "retained", "hook"])
    def test_autograd_observers_keep_native(self, change: str) -> None:
        """Leaf ownership, retained gradients and gradient hooks must remain observable."""
        feature = torch.ones(2, 2, dtype=torch.bfloat16, requires_grad=True) * 2
        handle = None
        if change == "leaf":
            feature = feature.detach().requires_grad_()
        elif change == "retained":
            feature.retain_grad()
        elif change == "hook":
            handle = feature.register_hook(lambda gradient: gradient)
        try:
            reason = capture._feature_breach((feature,), (True, torch.bfloat16))
            if change == "valid":
                assert reason is None
            else:
                assert "feature gradients" in reason
        finally:
            if handle is not None:
                handle.remove()

    @pytest.mark.usefixtures("_policy")
    @pytest.mark.parametrize("change", ["anomaly", "inference", "compilation"])
    def test_runtime_modes_keep_native(self, change: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Graph-incompatible runtime modes retain the native execution path."""
        context = contextlib.nullcontext()
        if change == "anomaly":
            context = torch.autograd.set_detect_anomaly(True)
        elif change == "inference":
            context = torch.inference_mode()
        else:
            monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        with context:
            reason = capture._feature_breach((torch.ones(2, dtype=torch.bfloat16),), (True, torch.bfloat16))
        assert change in reason

    @pytest.mark.usefixtures("_policy")
    @pytest.mark.parametrize("change", ["precision", "dtype", "runtime"])
    def test_unverified_precision_keeps_native(self, change: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the verified precision/runtime combination may establish a recording."""
        feature = torch.ones(2, dtype=torch.bfloat16)
        precision = (True, torch.bfloat16)
        if change == "precision":
            precision = (False, torch.bfloat16)
        elif change == "dtype":
            feature = feature.float()
        else:
            monkeypatch.setattr(capture, "_BF16_REPLAY_SUPPORTED", False)
        reason = capture._feature_breach((feature,), precision)
        assert ("torch>=2.9" if change == "runtime" else "BF16 mixed precision") in reason


class TestRunnerCertificate:
    """The same owner may replay only while its live contract still matches."""

    @pytest.fixture
    def runtime(self, _policy: None, monkeypatch: pytest.MonkeyPatch) -> tuple:
        """Reach host-side certification with CPU storage and an isolated CUDA API boundary.

        Examples:
            >>> callable(TestRunnerCertificate.runtime)
            True
        """
        model = _module()
        model._trainer = SimpleNamespace(training=True, accumulate_grad_batches=1)
        holder = capture._SegmentHeadCapture(True)
        holder.active = True
        features = tuple(torch.ones(1, 2, size, size, dtype=torch.bfloat16) for size in (8, 4, 2))
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        monkeypatch.setattr(capture, "_precision_signature", lambda: (True, torch.bfloat16))
        return model, holder, features

    def test_reuses_the_same_valid_certificate(self, runtime: tuple) -> None:
        """Repeated certification retains its owner and attribute certificate without recording a graph."""
        model, holder, features = runtime
        assert holder.runner(model, features) is holder
        owner, attributes, tensors = holder.owner, holder.attributes, holder.tensors
        assert owner is not None and attributes is not None and tensors
        assert holder.runner(model, features) is holder
        assert holder.owner is owner and holder.attributes is attributes
        assert holder.tensors is tensors and not holder.entries

    @pytest.mark.parametrize(
        "change", ["inactive", "disabled", "trainer", "validating", "eval", "manual", "no_grad", "capturing"]
    )
    def test_nontraining_calls_keep_native_without_retiring(
        self, runtime: tuple, change: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Temporary nontraining conditions do not poison a later eligible training batch."""
        model, holder, features = runtime
        context = contextlib.nullcontext()
        if change == "inactive":
            holder.active = False
        elif change == "disabled":
            holder.disabled = True
        elif change == "trainer":
            model._trainer = None
        elif change == "validating":
            model._trainer.training = False
        elif change == "eval":
            model.eval()
        elif change == "manual":
            model.automatic_optimization = False
        elif change == "no_grad":
            context = torch.no_grad()
        else:
            monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with context:
            assert holder.runner(model, features) is None
        assert holder.disabled == (change == "disabled") and holder.owner is None

    @pytest.mark.parametrize(
        "change", ["accumulation", "precision_error", "levels", "replacement", "attributes", "storage", "precision"]
    )
    def test_contract_drift_retires_with_its_reason(
        self, runtime: tuple, change: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A permanent mismatch discards the certificate and reports the reason to the user."""
        model, holder, features = runtime
        if change in ("replacement", "attributes", "storage", "precision"):
            assert holder.runner(model, features) is holder
        if change == "accumulation":
            model._trainer.accumulate_grad_batches = 2
            cause = "accumulation changed"
        elif change == "precision_error":
            monkeypatch.setattr(
                capture, "_precision_signature", MagicMock(side_effect=RuntimeError("precision query refused"))
            )
            cause = "precision query refused"
        elif change == "levels":
            features = features[:2]
            cause = "stem count differs"
        elif change == "replacement":
            model.semantic = type(model.semantic)(4, 4)
            cause = "stage was replaced"
        elif change == "attributes":
            model.head.extra_scalar = 1
            cause = "attributes or registration names changed"
        elif change == "storage":
            parameter = next(model.head.parameters())
            parameter.data = parameter.detach().clone()
            cause = "storage, topology, modes or precision changed"
        else:
            monkeypatch.setattr(capture, "_precision_signature", lambda: (True, torch.bfloat16, "changed"))
            cause = "storage, topology, modes or precision changed"
        with pytest.warns(UserWarning, match=cause):
            assert holder.runner(model, features) is None
        assert holder.disabled and holder.owner is None and holder.attributes is None


class TestHeadTopology:
    """Only the segmentation dispatch and its existing level counts are certifiable."""

    @pytest.mark.parametrize("change", ["head", "branch", "coefficients", "angles", "keypoints", "levels"])
    def test_names_an_unsupported_topology(self, change: str) -> None:
        """A different prediction family never borrows the segmentation recording's call signature."""
        head = _module().head
        if change == "head":
            assert "head overrides" in capture._head_breach(nn.Identity())
            return
        if change == "branch":
            head.o2m = nn.Identity()
            cause = "branch overrides"
        elif change == "coefficients":
            head.o2m.coeff_stems = None
            cause = "segmentation topology"
        elif change == "angles":
            head.o2m.angle_stems = nn.ModuleList()
            cause = "segmentation topology"
        elif change == "keypoints":
            head.o2m.keypoint_stems = nn.ModuleList()
            cause = "segmentation topology"
        else:
            head.o2m.box_stems = nn.ModuleList(list(head.o2m.box_stems)[:-1])
            cause = "stem count differs"
        assert cause in capture._head_breach(head)

    def test_attribute_plan_checks_topology_and_sequence_length(self) -> None:
        """Changing module order or extending an attribute list invalidates the live certificate."""
        owner = nn.Sequential(nn.Conv2d(1, 1, 1), nn.SiLU())
        owner.labels = [object()]
        modules = tuple(owner.modules())
        plan = capture._AttributePlan(modules)
        assert not plan.matches(tuple(reversed(modules)))
        owner.labels.append(object())
        assert not plan.matches(modules)


class TestRecordingLifetime:
    """The host orchestration restores warmup state and preserves pending backward ownership."""

    @pytest.mark.parametrize("failed", [False, True])
    def test_warmup_restores_buffers_and_rng_after_synchronizing(self, failed: bool) -> None:
        """A CPU stand-in at the graph API boundary exercises the real snapshot and finally block.

        This checks orchestration, not CUDA graph correctness: the GPU regression
        separately checks the same ordering with a real CUDA prediction stage.
        """
        stage = nn.BatchNorm2d(2)
        inputs = (torch.ones(2, 2, 3, 3),)
        adapter = capture._PredictionGroup((stage,), lambda values: (stage(values[0]),))
        snapshots = {name: value.clone() for name, value in stage.named_buffers()}
        rng = torch.get_rng_state().clone()
        events = []
        stream = SimpleNamespace(cuda_stream=17)
        fork_rng = torch.random.fork_rng

        def warmup(network: nn.Module, samples: tuple) -> nn.Module:
            network(*samples)
            torch.rand(2)
            events.append("warmup")
            if failed:
                raise RuntimeError("injected recording failure")
            return network

        with (
            patch("torch.cuda.current_stream", return_value=stream),
            patch("torch.cuda.synchronize", side_effect=lambda device: events.append("synchronize")),
            patch("torch.random.fork_rng", side_effect=lambda devices: fork_rng(devices=[])),
            patch("torch.cuda.make_graphed_callables", side_effect=warmup) as record,
        ):
            if failed:
                with pytest.raises(RuntimeError, match="injected recording failure"):
                    capture._RecordedGroup(adapter, inputs, capture._RecordedSpans([], list(inputs)))
            else:
                entry = capture._RecordedGroup(adapter, inputs, capture._RecordedSpans([], list(inputs)))
                assert entry.stream is stream and not entry.pending_backward
                assert entry.parameters == tuple(stage.parameters())
            assert record.call_count == 1
        assert events == ["warmup", "synchronize"]
        assert torch.equal(torch.get_rng_state(), rng)
        for name, value in stage.named_buffers():
            assert torch.equal(value, snapshots[name]), name

    def test_backward_preserves_existing_gradient_storage_and_releases_hold(self) -> None:
        """Real CPU autograd invokes the registered hooks; only the recording API is replaced."""
        stage = nn.Linear(2, 1)
        parameter = stage.weight
        parameter.grad = torch.ones_like(parameter)
        existing = parameter.grad
        inputs = (torch.ones(1, 2, requires_grad=True) * 2,)
        adapter = capture._PredictionGroup((stage,), lambda values: (stage(values[0]),))
        with (
            patch("torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=17)),
            patch("torch.cuda.synchronize"),
            patch("torch.random.fork_rng", return_value=contextlib.nullcontext()),
            patch("torch.cuda.make_graphed_callables", side_effect=lambda network, samples: network),
        ):
            entry = capture._RecordedGroup(adapter, inputs, capture._RecordedSpans([], list(inputs)))
            output = entry.run(inputs)
            assert entry.pending_backward
            output[0].sum().backward()
        assert not entry.pending_backward
        assert parameter.grad is not existing
        assert torch.equal(existing, torch.ones_like(existing))
        assert torch.equal(parameter.grad, torch.full_like(parameter, 3))
        assert torch.equal(stage.bias.grad, torch.ones_like(stage.bias))

    @pytest.mark.parametrize(("stage", "requested"), [("fit", False), ("validate", True)])
    def test_inactive_nonfit_setup_does_not_warn(self, stage: str, requested: bool) -> None:
        """A disabled request or a separate validation stage does not report a refused training fit."""
        holder = capture._SegmentHeadCapture(requested)
        with patch.object(capture.warnings, "warn", side_effect=AssertionError("must not warn")):
            holder.setup(_module(), stage)
        assert not holder.active

    def test_clear_drains_each_recorded_stream_once(self) -> None:
        """Multiple groups sharing a stream do not cause duplicate fences or retain their owner."""
        holder = capture._SegmentHeadCapture(True)
        first, second = MagicMock(), MagicMock()
        holder.entries = {
            (1, "stem"): SimpleNamespace(stream=first),
            (2, "stem"): SimpleNamespace(stream=first),
            (3, "mask"): SimpleNamespace(stream=second),
        }
        holder.owner = nn.Identity()
        holder.clear()
        first.synchronize.assert_called_once_with()
        second.synchronize.assert_called_once_with()
        assert holder.entries == {} and holder.owner is None

    def test_new_batch_releases_a_skipped_backward_hold(self) -> None:
        """A skipped training backward cannot force every later batch onto the native path."""
        holder = capture._SegmentHeadCapture(True)
        entry = SimpleNamespace(pending_backward=True)
        holder.entries[(1, "stem")] = entry
        module = SimpleNamespace(_segment_head_capture=holder)
        capture._SegmentHeadCaptureLifecycle().on_train_batch_start(None, module, None, 0)
        assert not entry.pending_backward


class TestGroupDispatch:
    """Routing records once, preserves unsafe inputs and calls each original prediction routine."""

    @pytest.mark.parametrize("change", ["leaf", "retained", "hook", "strides"])
    def test_unsafe_input_stays_native_without_recording(self, change: str) -> None:
        """Input-side autograd observers and unsupported layouts remain observable on the native call."""
        feature = torch.ones(2, 3, requires_grad=True) * 2
        handle = None
        if change == "leaf":
            feature = feature.detach().requires_grad_()
        elif change == "retained":
            feature.retain_grad()
        elif change == "hook":
            handle = feature.register_hook(lambda gradient: gradient)
        else:
            feature = feature.t()
        eager = MagicMock(return_value=(feature,))
        try:
            with patch.object(capture, "_RecordedGroup", side_effect=AssertionError("must not record")):
                result = capture._SegmentHeadCapture(True)._run((1, "stem"), (), eager, (feature,))
            assert result[0] is feature
            eager.assert_called_once_with((feature,))
        finally:
            if handle is not None:
                handle.remove()

    def test_aliasing_input_stays_native_without_recording(self) -> None:
        """An input view into a prediction parameter's storage cannot become a static graph input."""
        stage = nn.Linear(2, 2, bias=False)
        feature = stage.weight.detach()
        holder = capture._SegmentHeadCapture(True)
        holder.tensors = (stage.weight,)
        eager = MagicMock(return_value=(feature,))
        with patch.object(capture, "_RecordedGroup", side_effect=AssertionError("must not record")):
            assert holder._run((1, "stem"), (stage,), eager, (feature,))[0] is feature
        eager.assert_called_once_with((feature,))
        assert holder.entries == {}

    def test_reuses_one_entry_for_the_same_group(self) -> None:
        """Only the recording boundary is replaced; real group identity controls reuse."""
        stage = nn.Linear(2, 2)
        feature = torch.ones(1, 2)
        holder = capture._SegmentHeadCapture(True)
        holder.tensors = tuple(stage.parameters())
        entry = MagicMock()
        entry.run.return_value = (feature,)
        eager = MagicMock()
        with patch.object(capture, "_RecordedGroup", return_value=entry) as record:
            for _ in range(2):
                assert holder._run((id(stage), "stem"), (stage,), eager, (feature,))[0] is feature
        record.assert_called_once()
        assert entry.run.call_count == 2
        eager.assert_not_called()

    @pytest.mark.parametrize("kind", ["stem", "mask", "coefficients", "base"])
    def test_routes_the_original_prediction_expression(self, kind: str) -> None:
        """The scheduling wrapper returns the actual native prediction, without changing its expression."""
        holder = capture._SegmentHeadCapture(True)
        model = _module()
        channels = tuple(
            next(layer for layer in stage.modules() if isinstance(layer, nn.Conv2d)).in_channels
            for stage in model.head.o2m.box_stems
        )
        features = tuple(torch.ones(1, channels[index], size, size) for index, size in enumerate((8, 4, 2)))
        with patch.object(holder, "_run", side_effect=lambda key, stages, eager, inputs: eager(inputs)):
            if kind in ("stem", "mask"):
                stage = nn.Identity()
                assert getattr(holder, kind)(stage, features[0]) is features[0]
            elif kind == "coefficients":
                branch = model.head.o2m
                assert torch.equal(holder.coefficients(branch, features), branch._coefficients(features))
            else:
                branch = model.head.o2m
                result = holder.base(branch, features)
                native = branch._base_predictions(features)
                assert all(torch.equal(left, right) for left, right in zip(result, native, strict=True))
