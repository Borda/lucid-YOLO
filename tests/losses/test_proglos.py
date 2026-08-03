# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the progressive dual-loss schedule (WP-035).

Covers the DoD ``alpha(t)`` unit test reproducing R1 Eq. 3 exactly at
``t in {0, E/2, E-1}`` for ``E in {2, 10, 100}`` (plus the hand-computed midpoint
of an odd-``E`` run and the ``E == 1`` max-guard degenerate case), the
monotone-non-increasing shape of the ramp across every epoch, the
:class:`~lucid_yolo.losses.progressive.ProgressiveLossSchedule` delegating to the
pure function, and the module wiring: a three-epoch ``Trainer`` run steps
:attr:`~lucid_yolo.ptl.module.DetectionLitModule.alpha` through the exact schedule
values ``[0.8, 0.45, 0.1]`` epoch by epoch, captured by a callback that reads
``alpha`` on the first batch of each epoch (after the module's
``on_train_epoch_start`` hook has run).
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, cast

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lucid_yolo.data.targets import Targets
from lucid_yolo.losses import ProgressiveLossSchedule, progressive_alpha
from lucid_yolo.ptl import DetectionLitModule, collate_detection, unpack_batch

if TYPE_CHECKING:
    from pytorch_lightning import LightningModule

#: Class count of the tiny test head.
_NUM_CLASSES = 4
#: Square input side; divisible by every stride for an integer anchor grid.
_IMG_SIZE = 160


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed torch before each test so model init and synthetic batches are reproducible."""
    torch.manual_seed(0)


@pytest.mark.parametrize(
    ("epoch", "total_epochs", "expected"),
    [
        pytest.param(0, 2, 0.8, id="t0-E2"),
        pytest.param(0, 10, 0.8, id="t0-E10"),
        pytest.param(0, 100, 0.8, id="t0-E100"),
        pytest.param(1, 2, 0.1, id="tEnd-E2"),
        pytest.param(9, 10, 0.1, id="tEnd-E10"),
        pytest.param(99, 100, 0.1, id="tEnd-E100"),
        pytest.param(5, 11, 0.45, id="mid-E11"),
        pytest.param(0, 1, 0.8, id="degenerate-E1"),
    ],
)
def test_alpha_at_t0_mid_end(epoch: int, total_epochs: int, expected: float) -> None:
    """progressive_alpha reproduces R1 Eq. 3 exactly at t=0, the odd-E midpoint, and t=E-1."""
    assert progressive_alpha(epoch, total_epochs) == pytest.approx(expected)


def test_schedule_delegates_to_pure_function() -> None:
    """ProgressiveLossSchedule.alpha_at matches the pure function with the same endpoints."""
    schedule = ProgressiveLossSchedule(alpha_init=0.8, alpha_final=0.1)
    assert schedule.alpha_at(5, 11) == pytest.approx(progressive_alpha(5, 11, 0.8, 0.1))


@pytest.mark.parametrize(
    "total_epochs", [pytest.param(2, id="E2"), pytest.param(10, id="E10"), pytest.param(101, id="E101")]
)
def test_alpha_is_monotone_non_increasing(total_epochs: int) -> None:
    """The per-epoch weight never increases from one epoch to the next."""
    values = [progressive_alpha(epoch, total_epochs) for epoch in range(total_epochs)]
    assert all(later <= earlier for earlier, later in pairwise(values))


class _AlphaRecorder(Callback):
    """Records module.alpha on the first batch of every epoch, after the epoch-start hook."""

    def __init__(self) -> None:
        self.alphas: list[float] = []

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: object, batch_idx: int) -> None:
        """Capture alpha once per epoch — the first batch runs after on_train_epoch_start."""
        del trainer, batch
        if batch_idx == 0:
            self.alphas.append(cast("DetectionLitModule", pl_module).alpha)


class _SingleSampleDataset(Dataset[tuple[Tensor, Targets]]):
    """One-sample synthetic detection dataset for the schedule integration run."""

    def __len__(self) -> int:
        """Return the sample count."""
        return 1

    def __getitem__(self, index: int) -> tuple[Tensor, Targets]:
        """Return one ``(image, Targets)`` sample with a single instance."""
        del index
        boxes = torch.tensor([[8.0, 8.0, 48.0, 48.0]])
        labels = torch.tensor([1])
        return torch.randn(3, _IMG_SIZE, _IMG_SIZE), Targets(boxes=boxes, labels=labels)


def _collate_unpacked(batch: list[tuple[Tensor, Targets]]) -> tuple[Tensor, list[Targets]]:
    """Collate, then restore the ``list[Targets]`` the module consumes.

    No datamodule is attached here, so the transfer hook that restores the transport
    form never fires; this wrapper runs the same collate -> restore round-trip the
    datamodule performs in production (dequantizing the uint8 images and unpacking the
    targets) so the module receives its float images and ragged target list.
    """
    return unpack_batch(collate_detection(batch))


def test_module_alpha_steps_through_schedule() -> None:
    """A three-epoch run drives module.alpha through the exact ramp values epoch by epoch."""
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=_NUM_CLASSES)
    loader: DataLoader[tuple[Tensor, Targets]] = DataLoader(
        _SingleSampleDataset(), batch_size=1, collate_fn=_collate_unpacked
    )
    recorder = _AlphaRecorder()
    trainer = Trainer(
        max_epochs=3,
        limit_train_batches=1,
        limit_val_batches=0,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        callbacks=[recorder],
    )

    trainer.fit(module, train_dataloaders=loader)

    assert recorder.alphas == pytest.approx([0.8, 0.45, 0.1])
