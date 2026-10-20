# SPDX-License-Identifier: Apache-2.0
"""Pose replay retains every point/sigma tensor and its eager training gradients."""

from __future__ import annotations

import copy
from collections.abc import Iterator
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
from lucid_yolo.ptl import DetectionLitModule
from lucid_yolo.ptl import network_capture as capture_module
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.module import _pose_head_breach
from lucid_yolo.ptl.network_capture import _FeatureNetwork, _NetworkCapture


@pytest.fixture(autouse=True)
def _deterministic() -> Iterator[None]:
    """Match the deterministic kernel policy of the public-data fits."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _pose(**kwargs: object) -> DetectionLitModule:
    """Build a real small pose network with a dataset-specified point count.

    Examples:
        >>> _pose().head.num_keypoints
        5
    """
    return DetectionLitModule(
        depth=0.34,
        width=0.25,
        max_channels=64,
        num_classes=2,
        task="keypoints",
        num_keypoints=5,
        **kwargs,
    )


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare raw tensor bytes, including signed zero.

    Examples:
        >>> _bits_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
        False
    """
    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        return False
    return torch.equal(
        left.detach().contiguous().reshape(-1).view(torch.uint8),
        right.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def test_pose_features_preserve_fields() -> None:
    """The recorder's tensor tuple reconstructs native point and uncertainty fields."""
    torch.manual_seed(42)
    model = _pose().eval()
    images = torch.rand(2, 3, 64, 64)
    network = _FeatureNetwork(model.backbone, model.neck)
    with torch.no_grad():
        expected = model(images)
        tensors = network(images)
        actual = model.head(network.rebuild(tensors))
    assert len(tensors) == 3
    for field in fields(DualHeadOutput):
        left, right = getattr(expected, field.name), getattr(actual, field.name)
        if left is None:
            assert right is None
        else:
            assert right is not None and _bits_equal(left, right), field.name


def _replayed_pose(holder: _NetworkCapture, model: DetectionLitModule, image: torch.Tensor) -> DualHeadOutput:
    """Use the shared feature recording and native head of the production pose route.

    Examples:
        >>> holder = _NetworkCapture()
        >>> holder._disabled = True
        >>> output = _replayed_pose(holder, _pose().eval(), torch.zeros(2, 3, 64, 64))
        >>> output.o2o_keypoints.shape[-2:]
        torch.Size([5, 2])
    """
    features = holder.run_features(image, (model.backbone, model.neck))
    return model(image) if features is None else model.head(features)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph replay")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pose_replay_keeps_outputs_gradients_buffers(dtype: torch.dtype) -> None:
    """Changing images and repeated SGD updates follow the exact native trajectory."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    native = _pose().cuda().to(memory_format=torch.channels_last)
    replayed = copy.deepcopy(native)
    holder = _NetworkCapture()
    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-5) for model in (native, replayed)]
    for _ in range(4):
        image = torch.rand(2, 3, 64, 64, device="cuda").contiguous(memory_format=torch.channels_last)
        outputs = []
        for model, optimizer in zip((native, replayed), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
                output = model(image) if model is native else _replayed_pose(holder, model, image)
                tensors = [
                    getattr(output, field.name)
                    for field in fields(DualHeadOutput)
                    if getattr(output, field.name) is not None
                ]
                loss = sum(value.float().square().mean() for value in tensors)
            loss.backward()
            outputs.append(tensors)
            optimizer.step()
        assert (holder._network is not None) == (dtype != torch.bfloat16 or capture_module._BF16_REPLAY_SUPPORTED)
        for left, right in zip(*outputs, strict=True):
            assert _bits_equal(left, right)
        for (name, left), (_, right) in zip(native.named_parameters(), replayed.named_parameters(), strict=True):
            if left.grad is None:
                assert right.grad is None, name
            else:
                assert right.grad is not None and _bits_equal(left.grad, right.grad), name
        for name, value in native.state_dict().items():
            assert _bits_equal(value, replayed.state_dict()[name]), name


class _RecordingObserved(Callback):
    """Observe graph execution before the fit-boundary cleanup releases it."""

    def __init__(self) -> None:
        self.recorded = False

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: DetectionLitModule, outputs: object, batch: object, batch_idx: int
    ) -> None:
        """Require the public training step to create the actual recording."""
        del trainer, outputs, batch, batch_idx
        self.recorded |= pl_module._network_capture._network is not None


def _same_state(left: object, right: object) -> None:
    """Compare optimizer, scaler and model state by tensor bytes and exact metadata.

    Examples:
        >>> _same_state({"count": 2}, {"count": 2})
    """
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and _bits_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _same_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _same_state(a, b)
    else:
        assert left == right


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires public CUDA training")
@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("empty_batch", [False, True])
def test_pose_trainer_matches_native(precision: str, accumulation: int, empty_batch: bool) -> None:
    """Public pose training keeps flow/optimizer/scaler state through clipped accumulation."""
    torch.manual_seed(42)
    image = torch.rand(2, 3, 64, 64)
    target = Targets(
        boxes=torch.tensor([[8.0, 8.0, 48.0, 48.0]]),
        labels=torch.tensor([1]),
        keypoints=torch.tensor([[[16.0, 16.0]] * 5]),
        keypoint_vis=torch.ones(1, 5, dtype=torch.int64),
    )
    batches = [(image, [copy.deepcopy(target), copy.deepcopy(target)]) for _ in range(3)]
    if empty_batch:
        empty = Targets(
            boxes=torch.empty(0, 4),
            labels=torch.empty(0, dtype=torch.int64),
            keypoints=torch.empty(0, 5, 2),
            keypoint_vis=torch.empty(0, 5, dtype=torch.int64),
        )
        batches[1] = (image, [copy.deepcopy(empty), copy.deepcopy(empty)])
    initial = _pose().state_dict()
    states = []
    for capture in (False, True):
        torch.manual_seed(42)
        model = _pose(capture_network=capture, lr=1e-4)
        model.load_state_dict(initial)
        observed = _RecordingObserved()
        trainer = Trainer(
            accelerator="cuda",
            devices=1,
            max_epochs=1,
            precision=precision,
            accumulate_grad_batches=accumulation,
            gradient_clip_val=1.0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            callbacks=[observed],
        )
        trainer.fit(model, train_dataloaders=DataLoader(batches, batch_size=None))
        assert observed.recorded == (
            capture and accumulation == 1 and (precision != "bf16-mixed" or capture_module._BF16_REPLAY_SUPPORTED)
        )
        assert trainer.global_step == (3 + accumulation - 1) // accumulation
        assert model._network_capture._network is None
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        states.append(
            (
                copy.deepcopy(model.state_dict()),
                copy.deepcopy(trainer.optimizers[0].state_dict()),
                copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
            )
        )
    _same_state(*states)


@pytest.mark.parametrize("capture", [False, True])
def test_pose_cli_accepts_network_capture(capture: bool) -> None:
    """The shipped pose recipe exposes the existing recording flag without another knob."""
    cli = DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        run=False,
        args=[
            "--config",
            str(packaged_config("pose_nano_smoke")),
            "--model.capture_network",
            str(capture).lower(),
            "--trainer.accelerator",
            "cpu",
            "--trainer.logger",
            "false",
        ],
    )
    assert cli.model.hparams.capture_network is capture
    assert cli.model.head.num_keypoints == 17
    assert len(cli.model.configure_callbacks()) == int(capture)


def test_pose_head_certificate_survives_copy_and_rejects_overrides() -> None:
    """A native head stays eligible through deepcopy; changed dispatch cannot detach shared features."""
    model = copy.deepcopy(_pose())
    assert _pose_head_breach(model.head, model._pose_head_topology) is None
    with patch.object(model.head, "forward", wraps=model.head.forward):
        assert _pose_head_breach(model.head, model._pose_head_topology) is not None
    assert _pose_head_breach(model.head, model._pose_head_topology) is None
    with patch.object(type(model.head), "forward", lambda self, values: values):
        assert _pose_head_breach(model.head, model._pose_head_topology) is not None


def test_pose_head_certificate_rejects_forward_branch_overrides() -> None:
    """``forward`` only reaches submodules through ``forward_branch``; both dispatch points count."""
    model = copy.deepcopy(_pose())
    assert _pose_head_breach(model.head, model._pose_head_topology) is None
    with patch.object(model.head, "forward_branch", wraps=model.head.forward_branch):
        assert _pose_head_breach(model.head, model._pose_head_topology) is not None
    assert _pose_head_breach(model.head, model._pose_head_topology) is None
    with patch.object(type(model.head), "forward_branch", lambda self, features, branch: features):
        assert _pose_head_breach(model.head, model._pose_head_topology) is not None


def test_frozen_features_leave_full_native_call_to_owner() -> None:
    """A fully frozen feature extractor needs no recorded backward, even on CPU."""
    model = _pose()
    model.backbone.requires_grad_(False)
    model.neck.requires_grad_(False)
    holder = _NetworkCapture()
    with pytest.warns(UserWarning, match="shared pose features are frozen"):
        features = holder.run_features(torch.zeros(2, 3, 64, 64), (model.backbone, model.neck))
    assert features is None and holder._disabled and holder._network is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pose dispatch")
@pytest.mark.parametrize("location", ["head", "global"])
def test_pose_hook_retires_features_and_observes_native_root(location: str) -> None:
    """After recording, hooks keep their full native call boundary and retire the graph."""
    model = _pose(capture_network=True).cuda()
    model._trainer = SimpleNamespace(training=True, accumulate_grad_batches=1)
    model._capture_network_active = True
    image = torch.rand(2, 3, 64, 64, device="cuda")
    model._training_network(image).o2m_cls.sum().backward()
    model.zero_grad(set_to_none=True)
    assert model._network_capture._network is not None
    calls = []

    def observe(module: torch.nn.Module, args: object, output: object) -> None:
        """Observe only the intended public module in a global hook's larger call tree."""
        del args, output
        if module is (model if location == "global" else model.head):
            calls.append(module)

    handle = (
        model.head.register_forward_hook(observe)
        if location == "head"
        else torch.nn.modules.module.register_module_forward_hook(observe)
    )
    try:
        with pytest.warns(UserWarning, match="stays eager"):
            model._training_network(image)
        assert calls == [model if location == "global" else model.head]
        assert model._network_capture._disabled and model._network_capture._network is None
    finally:
        handle.remove()
