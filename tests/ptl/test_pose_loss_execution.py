# SPDX-License-Identifier: Apache-2.0
"""Pose loss execution reuses native assignment and independent branch reductions."""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.plugins.precision import MixedPrecision
from torch.utils.data import DataLoader

from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl.loss_capture import _BF16_SUPPORTED
from lucid_yolo.ptl.module import DetectionLitModule


def _pose(capture: bool, keypoint_loss: str = "rle") -> DetectionLitModule:
    """Construct the public pose option with a small matching annotation schema.

    Examples:
        >>> _pose(True).hparams.capture_loss
        True
    """
    return DetectionLitModule(
        0.5, 0.25, 64, 2, task="keypoints", num_keypoints=3, capture_loss=capture, keypoint_loss=keypoint_loss
    )


def _batch(count: int, visible: bool = True) -> tuple[torch.Tensor, list[Targets]]:
    """Build one empty image beside an asymmetric point-annotated image.

    Examples:
        >>> [len(target.boxes) for target in _batch(2)[1]]
        [0, 2]
    """
    boxes = torch.tensor([[4.0, 6.0, 36.0, 42.0], [24.0, 18.0, 59.0, 62.0]])[:count]
    points = torch.tensor([[[10.0, 12.0], [18.0, 20.0], [28.0, 30.0]], [[30.0, 24.0], [40.0, 32.0], [50.0, 52.0]]])
    targets = [
        Targets(
            boxes=boxes[:n].clone(),
            labels=torch.arange(n),
            keypoints=points[:n].clone(),
            keypoint_vis=torch.full((n, 3), 2 if visible else 0, dtype=torch.long),
        )
        for n in (0, count)
    ]
    return torch.rand(2, 3, 64, 64), targets


@pytest.mark.parametrize("keypoint_loss", ["rle", "laplace_nll"])
@pytest.mark.parametrize("count,visible", [(0, False), (1, False), (2, True)])
def test_pose_host_execution_keeps_loss_and_gradient_values(count: int, visible: bool, keypoint_loss: str) -> None:
    """CPU compatibility exercises routing while CUDA submission keeps native fallback."""
    seed_everything(19)
    native = _pose(False, keypoint_loss)
    executed = copy.deepcopy(native)
    executed._capture_loss_active = True
    executed._trainer = MagicMock(spec=Trainer, training=True)
    images, targets = _batch(count, visible)
    with (
        patch.object(native, "log"),
        patch.object(executed, "log"),
        patch.object(executed, "_paired_keypoint_terms", wraps=executed._paired_keypoint_terms) as paired,
        patch.object(executed._loss_capture, "run", wraps=executed._loss_capture.run) as replay,
        patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("CPU recording")),
    ):
        expected = native.training_step((images.clone(), targets), 0)
        actual = executed.training_step((images.clone(), targets), 0)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        expected.backward()
        actual.backward()
    assert replay.call_count == 1
    assert paired.call_count == int(keypoint_loss == "rle")
    assert executed._loss_capture._objective is None
    for (name, left), (other, right) in zip(native.named_parameters(), executed.named_parameters(), strict=True):
        assert name == other and (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert right.grad is not None
            assert (left.grad - right.grad).norm() <= 1e-5 * left.grad.norm() + 1e-6, name


@pytest.mark.parametrize("custom", ["_score_branch", "_rle_on", "_oks_on", "_gather_keypoint_positives"])
def test_pose_custom_scoring_keeps_native_dispatch(custom: str) -> None:
    """An overridden scoring boundary cannot be bypassed by the faster composition."""
    model = _pose(True)
    model._capture_loss_active = True
    model._trainer = MagicMock(spec=Trainer, training=True)
    original = getattr(model, custom)
    with (
        patch.object(model, "log"),
        patch.object(model, custom, wraps=original) as wrapped,
        patch.object(model, "_paired_keypoint_terms", side_effect=AssertionError("custom scoring paired")),
    ):
        model.training_step(_batch(1), 0)
    assert wrapped.call_count == 2


def test_pose_validation_keeps_original_criterion_signature() -> None:
    """Validation retains native loss dispatch and does not force the replay alpha argument."""
    model = _pose(True).eval()
    model._capture_loss_active = True
    original = model.loss
    seen = []

    def criterion(*args: object, **kwargs: object) -> object:
        seen.append(kwargs)
        return original(*args, **kwargs)

    model.loss = criterion
    with (
        patch.object(model, "log"),
        patch.object(model._loss_capture, "run", side_effect=AssertionError("validation recording")),
        patch.object(model, "_paired_keypoint_terms", side_effect=AssertionError("validation pairing")),
    ):
        model._shared_step(_batch(1), "val")
    assert len(seen) == 1 and "alpha" not in seen[0]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution needs a device")
@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
def test_pose_cuda_execution_keeps_objective_and_gradient_direction(precision: str) -> None:
    """A real automatic step records assignment and batches the same point objective."""
    if precision == "bf16-mixed" and not _BF16_SUPPORTED:
        pytest.skip("BF16 replay requires torch 2.9")
    seed_everything(23)
    batch = _batch(2)
    results = []

    class Observe(Callback):
        def __init__(self) -> None:
            self.gradients = {}
            self.loss = None
            self.recorded = False

        def on_after_backward(self, trainer: Trainer, pl_module: DetectionLitModule) -> None:
            self.gradients = {
                name: None if value.grad is None else value.grad.detach().cpu().clone()
                for name, value in pl_module.named_parameters()
            }

        def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: DetectionLitModule,
            outputs: dict[str, torch.Tensor],
            batch: object,
            batch_idx: int,
        ) -> None:
            self.loss = outputs["loss"].detach().cpu().clone()
            self.recorded = pl_module._loss_capture._objective is not None

    for capture in (False, True):
        seed_everything(17)
        model = _pose(capture)
        observed = Observe()
        plugin = (
            MixedPrecision("16-mixed", "cuda", scaler=torch.amp.GradScaler("cuda", init_scale=32.0))
            if precision == "16-mixed"
            else None
        )
        trainer = Trainer(
            accelerator="cuda",
            devices=1,
            precision=None if plugin is not None else precision,
            plugins=plugin,
            max_steps=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            callbacks=[observed],
            deterministic=True,
        )
        with patch.object(model, "_paired_keypoint_terms", wraps=model._paired_keypoint_terms) as paired:
            trainer.fit(model, train_dataloaders=DataLoader([batch], batch_size=None))
        assert paired.call_count == int(capture)
        assert observed.recorded is capture
        assert model._loss_capture._objective is None
        results.append(observed)
    epsilon = {
        "32-true": 1e-5,
        "16-mixed": 2 * torch.finfo(torch.float16).eps,
        "bf16-mixed": 2 * torch.finfo(torch.bfloat16).eps,
    }[precision]
    torch.testing.assert_close(results[0].loss, results[1].loss, rtol=epsilon, atol=1e-6)
    assert results[0].gradients.keys() == results[1].gradients.keys()
    for name, left in results[0].gradients.items():
        right = results[1].gradients[name]
        assert (left is None) == (right is None), name
        if left is not None:
            assert right is not None and torch.isfinite(left).all() and torch.isfinite(right).all()
            assert (left - right).norm() <= epsilon * left.norm() + 1e-5, name


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution needs a device")
def test_pose_cuda_execution_composes_with_network_replay() -> None:
    """Pose loss dispatch stays correct alongside the independent backbone/neck network replay.

    `capture_loss` reaches the network through `_training_network`, which runs
    `_network_capture.run_features` for pose when `capture_network` is also set — a
    second, independent recorder with its own graph pool. Nothing before this test ran
    the two together for `task="keypoints"`.
    """
    seed_everything(23)
    batches = [_batch(2) for _ in range(4)]
    results = []

    class Observe(Callback):
        def __init__(self) -> None:
            self.gradients: dict[str, torch.Tensor | None] = {}
            self.losses: list[torch.Tensor] = []
            self.active: list[tuple[bool, bool]] = []

        def on_after_backward(self, trainer: Trainer, pl_module: DetectionLitModule) -> None:
            self.gradients = {
                name: None if value.grad is None else value.grad.detach().cpu().clone()
                for name, value in pl_module.named_parameters()
            }

        def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: DetectionLitModule,
            outputs: dict[str, torch.Tensor],
            batch: object,
            batch_idx: int,
        ) -> None:
            self.losses.append(outputs["loss"].detach().cpu().clone())
            self.active.append(
                (pl_module._network_capture._network is not None, pl_module._loss_capture._objective is not None)
            )

    for capture_network, capture_loss in ((False, False), (True, True)):
        seed_everything(17)
        model = DetectionLitModule(
            0.5,
            0.25,
            64,
            2,
            task="keypoints",
            num_keypoints=3,
            capture_network=capture_network,
            capture_loss=capture_loss,
        )
        observed = Observe()
        trainer = Trainer(
            accelerator="cuda",
            devices=1,
            precision="32-true",
            max_steps=len(batches),
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            callbacks=[observed],
            deterministic=True,
        )
        trainer.fit(model, train_dataloaders=DataLoader(batches, batch_size=None))
        if capture_network:
            assert observed.active[-1] == (True, True)
        results.append(observed)

    for left, right in zip(results[0].losses, results[1].losses, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    assert results[0].gradients.keys() == results[1].gradients.keys()
    for name, left in results[0].gradients.items():
        right = results[1].gradients[name]
        assert (left is None) == (right is None), name
        if left is not None:
            assert right is not None and torch.isfinite(right).all()
            assert (left - right).norm() <= 1e-5 * left.norm() + 1e-6, name


def test_pose_custom_criterion_keeps_native_keywords() -> None:
    """A customized pose criterion need not accept the recorder's alpha keyword."""
    model = _pose(True)
    model._capture_loss_active = True
    model._trainer = MagicMock(spec=Trainer, training=True)
    original = model.loss
    calls = []

    def criterion(
        ml: torch.Tensor,
        mb: torch.Tensor,
        ol: torch.Tensor,
        ob: torch.Tensor,
        points: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        *,
        strides: torch.Tensor,
        gt_rboxes: torch.Tensor | None,
    ) -> object:
        calls.append(gt_rboxes)
        return original(ml, mb, ol, ob, points, boxes, labels, mask, strides=strides, gt_rboxes=gt_rboxes)

    model.loss = criterion
    with (
        patch.object(model, "log"),
        patch.object(model._loss_capture, "run", side_effect=AssertionError("custom criterion recorded")),
    ):
        result = model.training_step(_batch(1), 0)
    assert len(calls) == 1 and calls[0] is None and torch.isfinite(result)
