# SPDX-License-Identifier: Apache-2.0
"""Shared segmentation replay keeps prediction gradients and the fit lifecycle native."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from torch.utils.data import DataLoader

from lucid_yolo.cli.train import DetectionCLI, packaged_config
from lucid_yolo.data.targets import Targets
from lucid_yolo.models.heads.detect import DualHeadOutput
from lucid_yolo.ptl.callbacks import EMACallback
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.module import DetectionLitModule
from lucid_yolo.ptl.network_capture import _BF16_REPLAY_SUPPORTED


@pytest.fixture(autouse=True)
def _deterministic() -> Iterator[None]:
    """Restore the caller's deterministic policy after each exact tensor comparison."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warning)


def _module(**kwargs: object) -> DetectionLitModule:
    """Build the real segmentation stages at a small test width.

    Examples:
        >>> _module().task
        'segment'
    """
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=2, task="segment", **kwargs)


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


@pytest.mark.parametrize("heads", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph execution")
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
        for (name, left), (_, right) in zip(native.named_parameters(), replay.named_parameters(), strict=True):
            _same(left.grad, right.grad)
            if not use_masks and ("coeff_stems" in name or name.startswith("protonet.")):
                assert left.grad is None and right.grad is None
        assert (replay._network_capture._network is not None) == (dtype != torch.bfloat16 or _BF16_REPLAY_SUPPORTED)


@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires public CUDA training")
def test_segment_fit_replays_across_empty_partial_validation_and_ema(accumulation: int, precision: str) -> None:
    """Real fits preserve MuSGD and EMA through validation and unsupported accumulation."""
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
        model = _module(capture_network=capture, capture_segment_heads=True, lr=1e-4, warmup_epochs=0.0)
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
        with patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record:
            trainer.fit(
                model,
                train_dataloaders=DataLoader(batches, batch_size=None),
                val_dataloaders=DataLoader([_batch()], batch_size=None),
            )
        assert trainer.state.finished
        feature_records = [call for call in record.call_args_list if type(call.args[0]).__name__ == "_FeatureNetwork"]
        assert len(feature_records) == int(
            capture and accumulation == 1 and (precision != "bf16-mixed" or _BF16_REPLAY_SUPPORTED)
        )
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
    assert recordings[6:] == [accumulation == 1 and (precision != "bf16-mixed" or _BF16_REPLAY_SUPPORTED)] * 6


class _CudaMarked(torch.Tensor):
    """Exercise host routing with real CPU storage and a CUDA eligibility flag."""

    @property
    def is_cuda(self) -> bool:
        """Mark only the dispatch boundary; tensor arithmetic remains on CPU."""
        return True


@pytest.mark.parametrize(
    "mode", ["record", "absent", "accumulate", "override", "compiled", "head-hook", "head-helper", "eval", "no-grad"]
)
def test_segment_host_routing_and_native_fallback(mode: str) -> None:
    """A real segmentation forward consumes replayed features or invokes both native stages."""
    model = _module(capture_network=True)
    model._capture_network_active = True
    model._trainer = SimpleNamespace(training=True, accumulate_grad_batches=2 if mode == "accumulate" else 1)
    images = torch.rand(2, 3, 64, 64).as_subclass(_CudaMarked)
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
    expected_call = mode not in ("eval", "no-grad")
    context = torch.no_grad() if mode == "no-grad" else nullcontext()
    with (
        context,
        patch.object(
            model._network_capture, "run_features", return_value=features if mode == "record" else None
        ) as run,
        patch.object(model._network_capture, "retire") as retire,
        patch.object(model.backbone, "forward", wraps=model.backbone.forward) as backbone,
    ):
        actual = model.forward_segmentation(images)
    assert run.call_count == int(expected_call)
    assert backbone.call_count == int(mode != "record")
    assert retire.call_count == int(mode in ("accumulate", "override", "head-hook", "head-helper"))
    assert actual.prototypes.shape == (2, 32, 16, 16)
    if mode == "record":
        expected = model.head(features)
        for field in fields(DualHeadOutput):
            _same(getattr(actual.detect, field.name), getattr(expected, field.name))


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
        patch("lucid_yolo.ptl.module._segment_head_breach", autospec=True) as dispatch,
        patch("lucid_yolo.ptl.module._head_topology_breach", autospec=True) as topology,
        patch.object(model._network_capture, "run_features") as run,
    ):
        features = model._segmentation_features(images)
    dispatch.assert_not_called()
    topology.assert_not_called()
    run.assert_not_called()
    assert len(features) == 3


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
    model.forward_segmentation(images)
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
