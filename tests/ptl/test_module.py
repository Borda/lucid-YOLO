# SPDX-License-Identifier: Apache-2.0
"""Tests for the detection LightningModule and its helpers (WP-034).

Covers the DoD ``test_training_step`` (a forward + dual loss over a synthetic
batch returns a finite scalar carrying gradients and logs every per-component
term), the :func:`~lit_yolo.ptl.module.pad_targets` ragged-to-dense contract
(including empty and all-empty batches), ``configure_optimizers`` building a
:class:`~lit_yolo.optim.musgd.MuSGD` over every parameter, the constant-LR /
automatic-optimization wiring proved by a ``fast_dev_run`` Lightning smoke run,
the ``alpha`` delegation seam (WP-035), and the task conditioning: an invalid
task is rejected, the ``_task_extra_loss`` stub is an inert zero for every task,
and a ``segment`` module trains identically to a ``detect`` module.

The module is built at n-scale multipliers with a low channel cap and a 160-px
input so the real backbone/neck/head stack runs in a couple of seconds on CPU.
Batches are synthetic tensors plus hand-built :class:`~lit_yolo.data.targets.Targets`
(no dataset fixture needed). Direct ``training_step`` calls monkeypatch
``module.log`` because Lightning's ``self.log`` requires trainer attachment; the
``fast_dev_run`` test exercises the real logging path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning import Trainer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lit_yolo.data.targets import Targets
from lit_yolo.optim.musgd import MuSGD
from lit_yolo.ptl import DetectionLitModule, collate_detection, pad_targets

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Class count of the tiny test head.
_NUM_CLASSES = 4
#: Square input side; divisible by every stride for an integer anchor grid.
_IMG_SIZE = 160
#: Images per synthetic batch.
_BATCH_SIZE = 2
#: The full set of metric keys a train step must log.
_TRAIN_LOG_KEYS = {
    "train/loss",
    "train/o2m_box",
    "train/o2m_cls",
    "train/o2m_l1",
    "train/o2o_box",
    "train/o2o_cls",
    "train/o2o_l1",
}


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed torch before each test so model init and synthetic batches are reproducible."""
    torch.manual_seed(0)
    yield


def _tiny_module(task: str = "detect") -> DetectionLitModule:
    """Build an n-scale module with a low channel cap for fast CPU tests."""
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=_NUM_CLASSES, task=task)


def _synthetic_targets(num_boxes: int) -> Targets:
    """Build ``num_boxes`` valid ``xyxy`` targets inside the test image."""
    top_left = torch.rand(num_boxes, 2) * 80.0
    size = torch.rand(num_boxes, 2) * 40.0 + 10.0
    boxes = torch.cat([top_left, top_left + size], dim=1)
    labels = torch.randint(0, _NUM_CLASSES, (num_boxes,))
    return Targets(boxes=boxes, labels=labels)


def _synthetic_batch() -> tuple[Tensor, list[Targets]]:
    """Build a two-image batch with ragged (2 and 1) instance counts."""
    images = torch.randn(_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)
    targets = [_synthetic_targets(2), _synthetic_targets(1)]
    return images, targets


class _SyntheticDetectionDataset(Dataset[tuple[Tensor, Targets]]):
    """Two-sample synthetic detection dataset for the Lightning smoke run."""

    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        """Return the sample count."""
        return self._size

    def __getitem__(self, index: int) -> tuple[Tensor, Targets]:
        """Return one ``(image, Targets)`` sample with two instances."""
        return torch.randn(3, _IMG_SIZE, _IMG_SIZE), _synthetic_targets(2)


def test_training_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """A training step returns a finite scalar requiring grad and logs every component."""
    module = _tiny_module()
    recorder = MagicMock()
    monkeypatch.setattr(module, "log", recorder)
    loss = module.training_step(_synthetic_batch(), 0)
    assert loss.ndim == 0
    assert bool(torch.isfinite(loss))
    assert loss.requires_grad
    logged_keys = {call.args[0] for call in recorder.call_args_list}
    assert logged_keys == _TRAIN_LOG_KEYS


def test_validation_step_logs_under_val_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A validation step returns a finite scalar and logs its total under ``val/``."""
    module = _tiny_module()
    recorder = MagicMock()
    monkeypatch.setattr(module, "log", recorder)
    loss = module.validation_step(_synthetic_batch(), 0)
    assert bool(torch.isfinite(loss))
    logged_keys = {call.args[0] for call in recorder.call_args_list}
    assert "val/loss" in logged_keys


def test_configure_optimizers_is_musgd_over_all_parameters() -> None:
    """configure_optimizers returns a MuSGD covering every module parameter."""
    module = _tiny_module()
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, MuSGD)
    optim_param_count = sum(len(group["params"]) for group in optimizer.param_groups)
    assert optim_param_count == len(list(module.parameters()))
    assert optim_param_count > 0


def test_alpha_property_delegates_to_loss() -> None:
    """The alpha property reads and writes the underlying DualBranchLoss weight."""
    module = _tiny_module()
    module.alpha = 0.8
    assert module.alpha == 0.8
    assert module.loss.alpha == 0.8


def test_pad_targets_pads_ragged_batch_with_mask() -> None:
    """A ragged batch pads to N_max with a mask marking the real rows."""
    first = Targets(boxes=torch.tensor([[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 5.0, 5.0]]), labels=torch.tensor([1, 2]))
    second = Targets(boxes=torch.tensor([[2.0, 2.0, 6.0, 6.0]]), labels=torch.tensor([3]))
    boxes, labels, mask = pad_targets([first, second])
    assert boxes.shape == (2, 2, 4)
    assert mask.tolist() == [[True, True], [True, False]]
    assert labels.tolist() == [[1, 2], [3, 0]]
    assert torch.equal(boxes[1, 1], torch.zeros(4))


def test_pad_targets_handles_empty_image() -> None:
    """An image with no instances contributes an all-False mask row."""
    annotated = Targets(boxes=torch.tensor([[0.0, 0.0, 4.0, 4.0]]), labels=torch.tensor([1]))
    boxes, _labels, mask = pad_targets([annotated, Targets.empty()])
    assert boxes.shape == (2, 1, 4)
    assert mask.tolist() == [[True], [False]]


def test_pad_targets_all_empty_batch_is_zero_width() -> None:
    """A batch whose every image is empty yields N_max = 0 tensors."""
    boxes, labels, mask = pad_targets([Targets.empty(), Targets.empty()])
    assert boxes.shape == (2, 0, 4)
    assert labels.shape == (2, 0)
    assert mask.shape == (2, 0)


def test_invalid_task_raises() -> None:
    """An unsupported task name is rejected at construction."""
    with pytest.raises(ValueError, match="task must be one of"):
        _tiny_module(task="pose")


@pytest.mark.parametrize(
    "task",
    [pytest.param("detect", id="detect"), pytest.param("segment", id="segment"), pytest.param("obb", id="obb")],
)
def test_task_extra_loss_is_inert_zero(task: str) -> None:
    """The task extra-loss stub returns a zero scalar for every accepted task."""
    module = _tiny_module(task=task)
    images, targets = _synthetic_batch()
    head_out = module(images)
    extra = module._task_extra_loss(head_out, targets)
    assert extra.ndim == 0
    assert float(extra) == 0.0


def test_segment_task_trains_identically_to_detect() -> None:
    """With identical weights a segment module yields the same loss as a detect module."""
    detect = _tiny_module(task="detect")
    segment = _tiny_module(task="segment")
    segment.load_state_dict(detect.state_dict())
    batch = _synthetic_batch()
    detect.log = MagicMock()  # type: ignore[method-assign]
    segment.log = MagicMock()  # type: ignore[method-assign]
    detect_loss = detect.training_step(batch, 0)
    segment_loss = segment.training_step(batch, 0)
    assert torch.equal(detect_loss, segment_loss)


def test_fast_dev_run_smoke() -> None:
    """A fast_dev_run Trainer drives one train and one val batch end to end."""
    module = _tiny_module()
    loader: DataLoader[tuple[Tensor, Targets]] = DataLoader(
        _SyntheticDetectionDataset(_BATCH_SIZE), batch_size=_BATCH_SIZE, collate_fn=collate_detection
    )
    trainer = Trainer(fast_dev_run=True, accelerator="cpu", logger=False, enable_progress_bar=False)
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.state.finished
    assert "train/loss" in trainer.callback_metrics
