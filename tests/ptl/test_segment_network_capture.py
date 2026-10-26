# SPDX-License-Identifier: Apache-2.0
"""Shared segmentation replay keeps prediction gradients and the fit lifecycle native."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from markers import requires_cuda
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from torch.utils.data import DataLoader

from lucid_yolo.cli.train import DetectionCLI, packaged_config
from lucid_yolo.data.targets import Targets
from lucid_yolo.models.build import SegmentOutput
from lucid_yolo.models.heads.detect import DualHeadOutput
from lucid_yolo.ptl._graph_replay import copy_out_live_gradients
from lucid_yolo.ptl.callbacks import EMACallback
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.module import DetectionLitModule
from lucid_yolo.ptl.network_capture import _BF16_REPLAY_SUPPORTED


@pytest.fixture(autouse=True)
def _deterministic(deterministic_kernels_pinning_cudnn_benchmark: None) -> None:
    """Restore the caller's deterministic policy and cudnn benchmark flag after each test."""


def _module(**kwargs: object) -> DetectionLitModule:
    """Build the real segmentation stages at a small test width.

    Examples:
        >>> _module().task
        'segment'
    """
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=2, task="segment", **kwargs)


@contextmanager
def _no_retire_warning() -> Iterator[None]:
    """Fail on either replay holder reporting that it stays eager.

    Examples:
        >>> with _no_retire_warning():
        ...     pass
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*stays eager", category=UserWarning)
        yield


def _same(left: object, right: object) -> None:
    """Compare tensor bytes and the complete nested optimizer state.

    Examples:
        >>> _same(torch.tensor([1.0]), torch.tensor([1.0]))
    """
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, (tuple, list)) and len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _same(first, second)
    else:
        assert left == right


def _batch(empty: bool = False, size: int = 2) -> tuple[torch.Tensor, list[Targets], list[torch.Tensor]]:
    """Pair real image tensors with explicit instance masks on the prototype grid.

    Examples:
        >>> images, targets, masks = _batch(True, 1)
        >>> images.shape, targets[0].boxes.shape, masks[0].shape
        (torch.Size([1, 3, 64, 64]), torch.Size([0, 4]), torch.Size([0, 16, 16]))
    """
    images = torch.rand(size, 3, 64, 64)
    ring = torch.tensor([[8.0, 8.0], [48.0, 8.0], [48.0, 48.0], [8.0, 48.0]])
    targets = [
        Targets(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.int64))
        if empty
        else Targets(boxes=torch.tensor([[8.0, 8.0, 48.0, 48.0]]), labels=torch.tensor([1]), polygons=[ring])
        for _ in range(size)
    ]
    masks = [torch.zeros(0 if empty else 1, 16, 16) for _ in range(size)]
    for mask in masks:
        mask[:, 2:12, 2:12] = 1.0
    return images, targets, masks


class _WindowPosition(Callback):
    """Tell whether the current training batch continues an accumulation window rather than opening one.

    Lightning counts ``batch_idx`` from zero in every epoch and opens a window at each multiple of
    the accumulation, so a batch at any other index meets gradients its window accumulated.

    Examples:
        >>> position = _WindowPosition(3)
        >>> position.on_train_batch_start(None, None, None, 4)
        >>> position.inside
        True
    """

    def __init__(self, accumulation: int) -> None:
        self.accumulation, self.inside = accumulation, False

    def on_train_batch_start(self, trainer: object, pl_module: object, batch: object, batch_idx: int) -> None:
        """Record whether this batch follows another batch of the same window."""
        del trainer, pl_module, batch
        self.inside = batch_idx % self.accumulation != 0


def _live_gradient_spy(
    seen: set[tuple[int, ...]], position: _WindowPosition
) -> Callable[[Iterable[torch.Tensor]], None]:
    """Wrap the forward-time gradient copy so each group replaying onto accumulated gradients is recorded.

    The holder calls the copy only on its replay route, so a group in ``seen`` replayed inside an
    accumulation window and met the gradients that window's earlier batches left alive. A batch
    opening a window is not counted: Lightning clears gradients only after the training step's
    forward, so the previous step's gradients are alive there at every accumulation setting.

    Examples:
        >>> seen, position = set(), _WindowPosition(3)
        >>> position.inside = True
        >>> weight = torch.nn.Parameter(torch.zeros(1))
        >>> weight.grad = torch.ones(1)
        >>> _live_gradient_spy(seen, position)([weight])
        >>> len(seen)
        1
    """

    def copy_out(parameters: Iterable[torch.Tensor]) -> None:
        group = tuple(parameters)
        if position.inside and any(parameter.grad is not None for parameter in group):
            seen.add(tuple(id(parameter) for parameter in group))
        copy_out_live_gradients(group)

    return copy_out


def test_segment_request_preserves_defaults_and_checkpoint_keys() -> None:
    """A requested feature recorder adds no parameters and stays inactive off CUDA."""
    native, replay = _module(), _module(capture_network=True)
    assert not native.hparams.capture_network
    assert native.state_dict().keys() == replay.state_dict().keys()
    assert len(replay.configure_callbacks()) == 1
    with pytest.warns(UserWarning, match="single-device CUDA"):
        replay.setup("fit")
    with patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("CPU must stay native")):
        assert replay.forward_segmentation(torch.rand(2, 3, 64, 64)).prototypes.shape == (2, 32, 16, 16)
    assert replay._network_capture._network is None


@pytest.mark.parametrize(
    ("heads", "dtype"),
    [
        pytest.param(False, torch.float32, id="network-fp32"),
        pytest.param(False, torch.float16, id="network-fp16"),
        pytest.param(False, torch.bfloat16, id="network-bf16"),
        pytest.param(True, torch.bfloat16, id="heads-bf16"),
    ],
)
@requires_cuda
def test_segment_shared_replay_preserves_predictions_and_unused_gradients(heads: bool, dtype: torch.dtype) -> None:
    """Changing mask demand keeps every prediction, gradient and BatchNorm byte native."""
    torch.manual_seed(347)
    native = _module(capture_segment_heads=heads).cuda().to(memory_format=torch.channels_last)
    replay = copy.deepcopy(native)
    replay._capture_network_active = True
    replay._trainer = SimpleNamespace(training=True, accumulate_grad_batches=1)
    native._segment_head_capture.active = replay._segment_head_capture.active = heads
    native._trainer = SimpleNamespace(training=True, accumulate_grad_batches=1)
    for use_masks in (True, False, True):
        images = torch.rand(2, 3, 64, 64, device="cuda").to(memory_format=torch.channels_last)
        outputs = []
        for model in (native, replay):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                result = model.forward_segmentation(images)
                tensors = [result.detect.o2m_cls, result.detect.o2o_cls]
                if use_masks:
                    tensors += [result.prototypes, result.semantic, result.detect.o2m_coeff, result.detect.o2o_coeff]
                loss = sum(t.float().square().mean() for t in tensors if t is not None)
            loss.backward()
            outputs.append(result)
        for field in fields(DualHeadOutput):
            _same(getattr(outputs[0].detect, field.name), getattr(outputs[1].detect, field.name))
        _same(outputs[0].prototypes, outputs[1].prototypes)
        _same(outputs[0].semantic, outputs[1].semantic)
        _same(native.state_dict(), replay.state_dict())
        unused = 0
        for (name, left), (_, right) in zip(native.named_parameters(), replay.named_parameters(), strict=True):
            _same(left.grad, right.grad)
            if not use_masks and (
                "coeff_stems" in name or name.startswith(("protonet.", "semantic.", "proto_fusion."))
            ):
                assert left.grad is None and right.grad is None
                unused += 1
        assert use_masks or unused > 0, "no mask-only parameter matched the unused-gradient filter"
        assert (replay._network_capture._network is not None) == (dtype != torch.bfloat16 or _BF16_REPLAY_SUPPORTED)


#: Retirement warning of the head holder outside BF16 mixed precision.
_HEAD_PRECISION_RETIRE = "capture_segment_heads=True stays eager"


@pytest.mark.parametrize(
    ("heads", "precision", "accumulation", "retire"),
    [
        pytest.param(False, "bf16-mixed", 1, None, id="network-bf16"),
        pytest.param(False, "16-mixed", 1, None, id="network-fp16"),
        pytest.param(False, "32-true", 1, None, id="network-fp32"),
        pytest.param(True, "bf16-mixed", 1, None, id="heads-bf16"),
        pytest.param(False, "32-true", 2, None, id="network-fp32-accumulated"),
        pytest.param(True, "bf16-mixed", 2, None, id="heads-bf16-accumulated"),
        pytest.param(True, "16-mixed", 1, _HEAD_PRECISION_RETIRE, id="heads-fp16-stay-eager"),
    ],
)
@requires_cuda
def test_segment_fit_replays_across_empty_partial_validation_and_ema(
    heads: bool, precision: str, accumulation: int, retire: str | None
) -> None:
    """Real fits preserve MuSGD and EMA through validation, accumulation and unsupported head precision.

    Network-only cells cover every precision and accumulation 2, which carries live gradients
    into each second replay. Head replay adds the supported bf16 cell, the accumulated cell, where
    both holders replay and carry live gradients into each second replay, and the fp16
    retirement, whose warning must name the holder that stays eager.
    """
    torch.manual_seed(23)
    batches = [_batch(), _batch(True), _batch(size=1)]
    initial = _module().state_dict()
    snapshots = []
    recordings = []

    class Observe(Callback):
        """Observe the live feature recording before fit cleanup releases it."""

        def on_train_batch_end(
            self, trainer: Trainer, pl_module: DetectionLitModule, outputs: object, batch: object, batch_idx: int
        ) -> None:
            """Record whether the public route owns a real graphed network."""
            recordings.append(pl_module._network_capture._network is not None)

    for capture in (False, True):
        torch.manual_seed(23)
        model = _module(capture_network=capture, capture_segment_heads=heads, lr=1e-4, warmup_epochs=0.0)
        model.load_state_dict(initial)
        ema = EMACallback()
        trainer = Trainer(
            accelerator="cuda",
            devices=1,
            precision=precision,
            max_epochs=2,
            deterministic=True,
            benchmark=False,
            accumulate_grad_batches=accumulation,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            callbacks=[ema, Observe()],
            gradient_clip_val=1.0,
        )
        replays = precision != "bf16-mixed" or _BF16_REPLAY_SUPPORTED
        combined = capture and heads and precision == "bf16-mixed" and _BF16_REPLAY_SUPPORTED
        expectation = (
            pytest.warns(UserWarning, match=retire)
            if capture and retire
            else _no_retire_warning()
            if capture and replays
            else nullcontext()
        )
        with (
            expectation,
            patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record,
        ):
            trainer.fit(
                model,
                train_dataloaders=DataLoader(batches, batch_size=None),
                val_dataloaders=DataLoader([_batch()], batch_size=None),
            )
        assert trainer.state.finished
        feature_records = [call for call in record.call_args_list if type(call.args[0]).__name__ == "_FeatureNetwork"]
        assert len(feature_records) == int(capture and replays)
        head_records = [call for call in record.call_args_list if type(call.args[0]).__name__ == "_PredictionGroup"]
        assert len(head_records) >= int(combined)
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        snapshots.append(
            (
                copy.deepcopy(model.state_dict()),
                copy.deepcopy(trainer.optimizers[0].state_dict()),
                copy.deepcopy(ema.state_dict()),
                copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
            )
        )
        assert model._network_capture._network is None
        assert model._segment_head_capture.entries == {}
    _same(*snapshots)
    assert recordings[:6] == [False] * 6
    assert recordings[6:] == [replays] * 6


class _CudaMarked(torch.Tensor):
    """Exercise host routing with real CPU storage and a CUDA eligibility flag."""

    @property
    def is_cuda(self) -> bool:
        """Mark only the dispatch boundary; tensor arithmetic remains on CPU."""
        return True


#: Modes whose call stays native before the recorder is reached, so `run_features` is never consulted.
_NATIVE_BEFORE_RECORDER = (
    "inactive",
    "no-trainer",
    "validating",
    "manual-optimization",
    "cpu-images",
    "eval",
    "no-grad",
)


#: Reason completing the `capture_network=True stays eager: ` warning of each retiring mode; others run warning-free.
_RETIRE_REASON = {
    "override": "the segmentation forward is overridden or compiled$",
    "head-hook": "a segmentation head module is hooked, overridden or compiled$",
    "head-helper": "a prediction branch overrides its native dispatch$",
}


@pytest.mark.parametrize(
    "mode",
    [
        "record",
        "absent",
        "accumulate",
        "override",
        "compiled",
        "head-hook",
        "head-helper",
        "eval",
        "no-grad",
        "inactive",
        "no-trainer",
        "validating",
        "manual-optimization",
        "cpu-images",
    ],
)
def test_segment_host_routing_and_native_fallback(mode: str) -> None:
    """A real segmentation forward consumes replayed features or invokes both native stages."""
    model = _module(capture_network=True)
    model._capture_network_active = mode != "inactive"
    model.automatic_optimization = mode != "manual-optimization"
    model._trainer = (
        None
        if mode == "no-trainer"
        else SimpleNamespace(training=mode != "validating", accumulate_grad_batches=2 if mode == "accumulate" else 1)
    )
    images = torch.rand(2, 3, 64, 64)
    images = images if mode == "cpu-images" else images.as_subclass(_CudaMarked)
    features = model.neck(model.backbone(images))
    if mode == "override":
        model.forward_segmentation = model.forward_segmentation
    elif mode == "compiled":
        model._compiled_call_impl = lambda *_: None
    elif mode == "head-hook":
        model.head.register_forward_hook(lambda *_: None)
    elif mode == "head-helper":
        model.head.o2m._base_predictions = model.head.o2m._base_predictions
    elif mode == "eval":
        model.eval()
    expected_call = mode not in _NATIVE_BEFORE_RECORDER
    context = torch.no_grad() if mode == "no-grad" else nullcontext()
    reason = _RETIRE_REASON.get(mode)
    warning = (
        pytest.warns(UserWarning, match=f"capture_network=True stays eager: {reason}")
        if reason
        else warnings.catch_warnings(action="error")
    )
    with (
        context,
        warning,
        patch.object(
            model._network_capture, "run_features", return_value=features if mode == "record" else None
        ) as run,
        patch.object(model.backbone, "forward", wraps=model.backbone.forward) as backbone,
    ):
        actual = model.forward_segmentation(images)
    assert run.call_count == int(expected_call)
    assert backbone.call_count == int(mode != "record")
    assert model._network_capture.retired == (reason is not None)
    assert actual.prototypes.shape == (2, 32, 16, 16)
    if mode == "record":
        expected = model.head(features)
        for field in fields(DualHeadOutput):
            _same(getattr(actual.detect, field.name), getattr(expected, field.name))


class _OverridingModule(DetectionLitModule):
    """A subclass whose class-level `forward_segmentation` replaces the native one."""

    def forward_segmentation(self, images: torch.Tensor) -> SegmentOutput:
        """Delegate to the native forward, which the recorder can no longer certify by identity."""
        return super().forward_segmentation(images)


def _eligible_module(cls: type[DetectionLitModule] = DetectionLitModule) -> DetectionLitModule:
    """Build an activated segmentation fit whose images are eligible for replay.

    Examples:
        >>> _eligible_module()._capture_network_active
        True
    """
    model = cls(depth=0.34, width=0.25, max_channels=64, num_classes=2, task="segment", capture_network=True)
    model._capture_network_active = True
    model._trainer = SimpleNamespace(training=True, accumulate_grad_batches=1)
    return model


def test_segment_class_level_forward_override_retires_to_native() -> None:
    """A subclass overriding `forward_segmentation` on the class retires replay and stays native."""
    model = _eligible_module(_OverridingModule)
    images = torch.rand(2, 3, 64, 64).as_subclass(_CudaMarked)
    with (
        pytest.warns(UserWarning, match="stays eager: the segmentation forward is overridden or compiled$"),
        patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("overridden forward must stay native")),
    ):
        actual = model.forward_segmentation(images)
    assert model._network_capture.retired
    assert actual.prototypes.shape == (2, 32, 16, 16)


def test_segment_head_topology_swap_retires_with_topology_reason() -> None:
    """Swapping the class of one head submodule retires replay naming the changed topology."""
    model = _eligible_module()
    stem = model.head.o2m.box_stems[0]
    stem.__class__ = type("_SwappedStem", (type(stem),), {})
    images = torch.rand(2, 3, 64, 64).as_subclass(_CudaMarked)
    with (
        pytest.warns(UserWarning, match="stays eager: the segmentation head topology or class forward changed$"),
        patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("swapped head must stay native")),
    ):
        actual = model.forward_segmentation(images)
    assert model._network_capture.retired
    assert actual.prototypes.shape == (2, 32, 16, 16)


def test_retired_holder_skips_segment_replay_predicates() -> None:
    """A retired recording sends later steps native without re-walking the head certificate.

    Once retired, the holder answers every call eagerly and ignores later causes, so the
    dispatch and topology walks could change nothing; a step after retirement skips them
    and never reaches the recorder.
    """
    model = _module(capture_network=True)
    model._capture_network_active = True
    model._trainer = SimpleNamespace(training=True, accumulate_grad_batches=1)
    with pytest.warns(UserWarning, match="stays eager: a test cause"):
        model._network_capture.retire("a test cause")
    images = torch.rand(2, 3, 64, 64).as_subclass(_CudaMarked)
    with (
        warnings.catch_warnings(),
        patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("retired holder must stay native")),
        patch("lucid_yolo.ptl.module._segment_head_breach", autospec=True) as dispatch,
        patch("lucid_yolo.ptl.module._head_topology_breach", autospec=True) as topology,
        patch.object(model._network_capture, "run_features") as run,
    ):
        warnings.simplefilter("error")
        features = model._segmentation_features(images)
    dispatch.assert_not_called()
    topology.assert_not_called()
    run.assert_not_called()
    assert len(features) == 3
    assert model._network_capture.retired


def test_frozen_segment_features_retire_before_cuda_submission() -> None:
    """Frozen features stay native after unfreezing and report the shared-stage cause."""
    model = _module(capture_network=True)
    model._capture_network_active = True
    model._trainer = SimpleNamespace(training=True, accumulate_grad_batches=1)
    model.backbone.requires_grad_(False)
    model.neck.requires_grad_(False)
    images = torch.rand(2, 3, 64, 64).as_subclass(_CudaMarked)
    with (
        pytest.warns(UserWarning, match="shared network features are frozen"),
        patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("frozen stages must stay native")),
    ):
        model.forward_segmentation(images)
    assert model._network_capture._disabled
    model.backbone.requires_grad_(True)
    model.neck.requires_grad_(True)
    with (
        warnings.catch_warnings(),
        patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("retirement must stay sticky")),
    ):
        warnings.simplefilter("error")
        model.forward_segmentation(images)
    assert model._network_capture._disabled
    assert model._network_capture._network is None


@pytest.mark.parametrize("enabled", [False, True])
def test_segmentation_cli_keeps_feature_and_prediction_options_independent(enabled: bool) -> None:
    """The shipped segmentation recipe forwards both requests through the real CLI."""
    cli = DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        args=[
            "--config",
            str(packaged_config("seg_nano_smoke.yaml")),
            "--model.capture_network",
            str(enabled).lower(),
            "--model.capture_segment_heads",
            "true",
            "--trainer.accelerator",
            "cpu",
            "--trainer.logger",
            "false",
        ],
        run=False,
        save_config_callback=None,
    )
    assert cli.model.task == "segment"
    assert cli.model.hparams.capture_network is enabled
    assert cli.model.hparams.capture_segment_heads is True
    assert cli.datamodule._mask_targets


def test_segment_head_hook_retires_feature_replay_and_stays_observable() -> None:
    """A mutable head cannot change which shared outputs a recorded backward must consume."""
    model = _module(capture_network=True)
    model._capture_network_active = True
    model._trainer = SimpleNamespace(training=True, accumulate_grad_batches=1)
    images = torch.rand(2, 3, 64, 64).as_subclass(_CudaMarked)
    observed = []
    handle = model.head.register_forward_hook(lambda *_: observed.append(True))
    with (
        pytest.warns(UserWarning, match="stays eager: a segmentation head module is hooked, overridden or compiled$"),
        patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("custom head must stay native")),
    ):
        model.forward_segmentation(images)
    assert observed == [True] and model._network_capture._disabled
    handle.remove()
    model.forward_segmentation(images)
    assert model._network_capture._network is None


#: Trainer arguments shared by the CUDA lifecycle fits, which differ only in precision and callbacks.
_LIFECYCLE_FIT = {
    "accelerator": "cuda",
    "devices": 1,
    "max_epochs": 2,
    "deterministic": True,
    "benchmark": False,
    "logger": False,
    "enable_checkpointing": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}
_skip_no_bf16_replay = pytest.mark.skipif(not _BF16_REPLAY_SUPPORTED, reason="BF16 replay requires torch>=2.9")


class _FailingModule(DetectionLitModule):
    """A segmentation module whose second training step fails after the first recorded its graphs."""

    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        """Run the native step, then fail on the second batch of the fit."""
        if batch_idx == 1:
            raise RuntimeError("step failed after recording")
        return super().training_step(batch, batch_idx)  # type: ignore[arg-type]


class _FreezeBackbone(Callback):
    """Freeze the shared backbone after the first batch and release it after the second."""

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: DetectionLitModule, outputs: object, batch: object, batch_idx: int
    ) -> None:
        """Change the trainable set of a recorded network between steps."""
        pl_module.backbone.requires_grad_(batch_idx != 0)


def _fit_batches() -> DataLoader[object]:
    """Serve full, empty-mask and partial batches, as the fit parity test does.

    Examples:
        >>> len(_fit_batches())
        3
    """
    return DataLoader([_batch(), _batch(True), _batch(size=1)], batch_size=None)  # type: ignore[arg-type]


@requires_cuda
@_skip_no_bf16_replay
def test_segment_fit_exception_after_recording_releases_both_holders() -> None:
    """A failing step after recording leaves no graph, pending backward or head entry behind."""
    torch.manual_seed(23)
    model = _FailingModule(
        depth=0.34,
        width=0.25,
        max_channels=64,
        num_classes=2,
        task="segment",
        capture_network=True,
        capture_segment_heads=True,
    )
    trainer = Trainer(**_LIFECYCLE_FIT, precision="bf16-mixed")
    with (
        patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record,
        pytest.raises(RuntimeError, match="step failed after recording"),
    ):
        trainer.fit(model, train_dataloaders=_fit_batches())
    assert any(type(call.args[0]).__name__ == "_FeatureNetwork" for call in record.call_args_list)
    assert model._network_capture._network is None
    assert model._network_capture._pending_backward is False
    assert model._segment_head_capture.entries == {}


@requires_cuda
def test_segment_second_fit_records_once_per_fit() -> None:
    """Fitting the same module twice records its shared features once in each fit."""
    torch.manual_seed(23)
    model = _module(capture_network=True)
    counts = []
    for _ in range(2):
        trainer = Trainer(**_LIFECYCLE_FIT, precision="16-mixed")
        with patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record:
            trainer.fit(model, train_dataloaders=_fit_batches())
        counts.append(sum(type(call.args[0]).__name__ == "_FeatureNetwork" for call in record.call_args_list))
    assert counts == [1, 1]


@requires_cuda
def test_segment_freeze_after_recording_retires_once() -> None:
    """Freezing then releasing the backbone after recording reports a single retirement."""
    torch.manual_seed(23)
    model = _module(capture_network=True)
    trainer = Trainer(**_LIFECYCLE_FIT, precision="16-mixed", callbacks=[_FreezeBackbone()])
    with pytest.warns(UserWarning, match="capture_network=True stays eager") as record:
        trainer.fit(model, train_dataloaders=_fit_batches())
    assert sum("capture_network=True stays eager" in str(warning.message) for warning in record) == 1


@requires_cuda
def test_segment_accumulation_change_after_recording_keeps_replay_and_matches_native() -> None:
    """Raising accumulate_grad_batches after epoch one keeps the recording and the native result.

    The recording is made at accumulation 1; epoch two accumulates two batches, so live
    gradients reach a replay the recording never saw. One recording, no retirement and
    bitwise-equal weights and optimizer state show the change needs no fallback.
    """
    initial = _module().state_dict()
    snapshots = []
    for capture in (False, True):
        torch.manual_seed(23)
        model = _module(capture_network=capture)
        model.load_state_dict(initial)
        scheduler = GradientAccumulationScheduler({0: 1, 1: 2})
        trainer = Trainer(**_LIFECYCLE_FIT, precision="16-mixed", callbacks=[scheduler])
        with (
            _no_retire_warning(),
            patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record,
        ):
            trainer.fit(model, train_dataloaders=_fit_batches())
        feature_records = sum(type(call.args[0]).__name__ == "_FeatureNetwork" for call in record.call_args_list)
        assert feature_records == int(capture)
        assert model._network_capture._network is None
        snapshots.append((copy.deepcopy(model.state_dict()), copy.deepcopy(trainer.optimizers[0].state_dict())))
    _same(*snapshots)


@requires_cuda
@_skip_no_bf16_replay
def test_segment_head_replay_under_accumulation_matches_native() -> None:
    """Head replay at accumulation 3 ends with the weights, MuSGD, EMA and scaler bytes of a native fit.

    Mirrors ``test_segment_fit_replays_across_empty_partial_validation_and_ema`` with the head holder
    as the only difference between the arms. Each window opens with a full batch that replays, then
    a partial batch that misses the recorded signature and accumulates natively, then a second full
    batch whose replay meets the gradients both left. Every recorded prediction group must replay
    onto its window's accumulated gradients at least once, so a run whose in-window replay silently
    went native cannot pass; the gradients leave the static buffers at forward time. The
    empty-target batch, alone in the last window, leaves mask groups without a backward, so the
    skipped-backward hold and its release at the next batch run too. The replay arm warns nothing.
    """
    torch.manual_seed(23)
    batches = [_batch(), _batch(size=1), _batch(), _batch(True)]
    initial = _module().state_dict()
    snapshots = []
    for heads in (False, True):
        torch.manual_seed(23)
        model = _module(capture_segment_heads=heads, lr=1e-4, warmup_epochs=0.0)
        model.load_state_dict(initial)
        ema, position = EMACallback(), _WindowPosition(3)
        trainer = Trainer(
            **_LIFECYCLE_FIT,
            precision="bf16-mixed",
            accumulate_grad_batches=3,
            callbacks=[ema, position],
            gradient_clip_val=1.0,
        )
        live: set[tuple[int, ...]] = set()
        with (
            warnings.catch_warnings(),
            patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record,
            patch("lucid_yolo.ptl.segment_head_capture.copy_out_live_gradients", _live_gradient_spy(live, position)),
        ):
            warnings.filterwarnings("error", message="capture_segment_heads=True", category=UserWarning)
            trainer.fit(
                model,
                train_dataloaders=DataLoader(batches, batch_size=None),
                val_dataloaders=DataLoader([_batch()], batch_size=None),
            )
        assert trainer.state.finished
        head_records = [call for call in record.call_args_list if type(call.args[0]).__name__ == "_PredictionGroup"]
        assert bool(head_records) == heads
        assert len(live) == len(head_records)
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        snapshots.append(
            (
                copy.deepcopy(model.state_dict()),
                copy.deepcopy(trainer.optimizers[0].state_dict()),
                copy.deepcopy(ema.state_dict()),
                copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
            )
        )
        assert model._segment_head_capture.entries == {}
    _same(*snapshots)
