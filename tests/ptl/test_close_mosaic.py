# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`~lucid_yolo.ptl.callbacks.CloseMosaicCallback` (WP-036).

Covers the DoD ``test_flip_epoch``: a 5-epoch dry run over the real
:class:`~lucid_yolo.ptl.datamodule.DetectionDataModule` (wired to the synthetic
detseg fixture, one batch per epoch) with ``close_mosaic=2`` disables mosaic from
epoch ``E - close_mosaic`` on and leaves it untouched before. A probe callback
records the pipeline's mosaic probability at the end of every epoch (after the
schedule has run and the epoch's batches have been drawn), so the per-epoch trace
is asserted directly. The parametrization also pins the two boundary contracts:
``close_mosaic=0`` never flips, and ``close_mosaic > max_epochs`` flips from epoch 0.

The Lightning module under test is a one-parameter stub — the callback drives the
*datamodule* seam, not the model, so the real detection stack is deliberately not
imported here. Two fast unit tests (no ``Trainer``) pin idempotency via the
duck-typed datamodule fallback and the negative-argument guard.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor, nn

from lucid_yolo.ptl.callbacks import CloseMosaicCallback
from lucid_yolo.ptl.datamodule import DetectionDataModule

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from lucid_yolo.data.targets import Targets

#: Small square input so the fixture pipeline runs in a fraction of a second.
_IMG_SIZE = 64
#: Epoch budget for the dry runs.
_MAX_EPOCHS = 5
#: Samples per batch.
_BATCH_SIZE = 2


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed torch before each test so the stub module and pipeline are reproducible."""
    torch.manual_seed(0)
    yield


class _OneParamModule(LightningModule):
    """Minimal Lightning module: consumes the image batch, returns a scalar loss.

    The mosaic schedule is a datamodule concern, so the module only has to run a
    real ``training_step`` with a gradient-carrying loss; it never inspects targets.
    """

    def __init__(self) -> None:
        super().__init__()
        self._weight = nn.Parameter(torch.zeros(1))

    def training_step(self, batch: tuple[Tensor, list[Targets]], batch_idx: int) -> Tensor:
        """Return a scalar loss coupling the image mean to the single parameter."""
        images, _targets = batch
        return (self._weight * images.mean()).sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return a zero-lr SGD so ``fit`` runs without perturbing the stub."""
        return torch.optim.SGD(self.parameters(), lr=0.0)


class _MosaicProbe(Callback):
    """Record the datamodule's mosaic probability at the end of every train epoch.

    Reading at ``on_train_epoch_end`` captures the value that epoch's batches were
    drawn with, independent of the order callbacks fire at ``on_train_epoch_start``.
    """

    def __init__(self, datamodule: DetectionDataModule) -> None:
        self._datamodule = datamodule
        self.per_epoch: list[float] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Append the pipeline's current mosaic probability."""
        self.per_epoch.append(self._datamodule.mosaic_prob)


class _StubDataModule:
    """Duck-typed datamodule exposing only the mosaic seam (not a DetectionDataModule)."""

    def __init__(self) -> None:
        self.mosaic_prob = 1.0
        self.set_calls = 0

    def set_mosaic_prob(self, prob: float) -> None:
        """Record every set call so idempotency can be asserted."""
        self.mosaic_prob = float(prob)
        self.set_calls += 1


def _datamodule(fixture_dir: Path) -> DetectionDataModule:
    """Build a datamodule pointing both splits at the fixture's single split."""
    split = fixture_dir / "train"
    annotation = split / "_annotations.coco.json"
    return DetectionDataModule(
        data_root=fixture_dir,
        batch_size=_BATCH_SIZE,
        num_workers=0,
        variant="m",
        img_size=_IMG_SIZE,
        train_images_dir=split,
        train_ann_file=annotation,
        val_images_dir=split,
        val_ann_file=annotation,
    )


def _run_schedule(datamodule: DetectionDataModule, close_mosaic: int) -> list[float]:
    """Run a ``_MAX_EPOCHS`` dry fit and return the per-epoch mosaic probabilities."""
    probe = _MosaicProbe(datamodule)
    trainer = Trainer(
        max_epochs=_MAX_EPOCHS,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        callbacks=[CloseMosaicCallback(close_mosaic=close_mosaic), probe],
    )
    trainer.fit(_OneParamModule(), datamodule=datamodule)
    return probe.per_epoch


@pytest.mark.parametrize(
    ("close_mosaic", "expected"),
    [
        pytest.param(2, [1.0, 1.0, 1.0, 0.0, 0.0], id="flips-at-E-minus-close"),
        pytest.param(0, [1.0, 1.0, 1.0, 1.0, 1.0], id="zero-never-flips"),
        pytest.param(10, [0.0, 0.0, 0.0, 0.0, 0.0], id="exceeds-max-flips-from-epoch-0"),
    ],
)
def test_flip_epoch(detseg_fixture_dir: Path, close_mosaic: int, expected: list[float]) -> None:
    """Mosaic probability follows the close-mosaic schedule across a 5-epoch dry run."""
    trace = _run_schedule(_datamodule(detseg_fixture_dir), close_mosaic)
    assert trace == expected


def test_repeated_flip_is_idempotent() -> None:
    """Once the window is entered the seam is set once; later epochs are no-ops."""
    callback = CloseMosaicCallback(close_mosaic=2)
    datamodule = _StubDataModule()
    module = MagicMock(spec=LightningModule)
    for epoch in (3, 4):  # both epochs are inside the close-mosaic window (E - 2 = 3)
        trainer = SimpleNamespace(current_epoch=epoch, max_epochs=_MAX_EPOCHS, datamodule=datamodule)
        callback.on_train_epoch_start(trainer, module)  # type: ignore[arg-type]
    assert datamodule.mosaic_prob == 0.0
    assert datamodule.set_calls == 1


def test_negative_close_mosaic_rejected() -> None:
    """A negative ``close_mosaic`` is rejected at construction."""
    with pytest.raises(ValueError, match="close_mosaic must be non-negative"):
        CloseMosaicCallback(close_mosaic=-1)
