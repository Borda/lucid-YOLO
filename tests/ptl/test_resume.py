# SPDX-License-Identifier: Apache-2.0
"""Deterministic checkpoint-and-resume tests for the detection loop (WP-039).

Covers the DoD ``test_trajectory_match`` — a seeded, uninterrupted four-epoch
:meth:`~pytorch_lightning.Trainer.fit` and a seeded run interrupted at a
two-epoch checkpoint then resumed to four epochs produce the **same** per-step
training-loss trajectory over the post-resume epochs — plus the supporting
continuity contracts: after resume the trainer's ``global_step`` and
``current_epoch`` continue from the checkpoint rather than restarting, and the
:class:`~lucid_yolo.optim.musgd.MuSGD` momentum buffers are restored (the
optimizer state is non-empty), so the resumed run continues the same momentum
rather than re-warming from zero.

Determinism is made trivial to reason about: the data stream is a fixed
in-memory :class:`~torch.utils.data.Dataset` of precomputed samples served with
``shuffle=False`` (so run A epoch 3 and run B's post-resume epoch 3 see the same
batches in the same order, without relying on sampler-state restoration, which
is out of this WP's scope), the trainer runs on CPU with ``deterministic=True``,
and the module carries no stochastic forward path, so the only state that must
cross the checkpoint boundary is the module weights and the optimizer momentum
buffers.

The interruption is modelled the way a real one happens: every run — reference,
interrupted, and resumed — is configured with the same ``max_epochs=4``, and the
interrupted run is merely *stopped early* after two epochs (via a
``should_stop`` callback) before its checkpoint is saved. Keeping ``max_epochs``
fixed matters because the alpha ramp (WP-035) is a pure function of
``current_epoch`` and ``max_epochs``: a differently-configured
``Trainer(max_epochs=2)`` would ramp alpha to its endpoint by epoch 1 and so
train the pre-checkpoint epochs on a *different* objective, producing a
checkpoint whose trajectory could never rejoin the reference. Alpha itself is
not checkpointed state — it is recomputed each epoch — so no restoration is
needed once ``max_epochs`` matches.

The module is built at n-scale multipliers with a low channel cap and a 64-px
input so the real backbone/neck/head stack trains in seconds on CPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
from pytorch_lightning import Callback, Trainer, seed_everything
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl import DetectionLitModule, collate_detection, unpack_batch

if TYPE_CHECKING:
    from pathlib import Path

#: Class count of the tiny test head.
_NUM_CLASSES = 4
#: Square input side; divisible by every stride (8, 16, 32) for an integer anchor grid.
_IMG_SIZE = 64
#: Images per batch.
_BATCH_SIZE = 2
#: Precomputed samples in the fixed dataset (``_DATASET_SIZE / _BATCH_SIZE`` batches per epoch).
_DATASET_SIZE = 8
#: Total epochs of the uninterrupted reference run.
_MAX_EPOCHS = 4
#: Epoch at which run B checkpoints and is later resumed.
_RESUME_EPOCH = 2
#: Optimizer steps per epoch (one per batch under automatic optimization).
_STEPS_PER_EPOCH = _DATASET_SIZE // _BATCH_SIZE
#: Instances per synthetic image.
_INSTANCES_PER_IMAGE = 2
#: Seed shared by every seeded run so weight init and batch order are reproducible.
_SEED = 1234


class _LossTrajectoryRecorder(Callback):
    """Record the scalar training loss of every optimizer step in fit order."""

    def __init__(self) -> None:
        super().__init__()
        self.losses: list[float] = []

    def on_train_batch_end(self, trainer: Trainer, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Append the step's scalar loss (automatic optimization wraps it as ``{"loss": ...}``)."""
        del trainer, pl_module, batch, batch_idx
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.losses.append(float(loss))


class _StopAfterEpoch(Callback):
    """Request trainer stop once ``epochs`` epochs have finished — a modelled interruption.

    Keeps ``max_epochs`` unchanged (so the alpha ramp is identical to the full run)
    while ending training early, exactly as an operator killing a run at epoch
    ``epochs`` would, leaving a checkpoint that a full-``max_epochs`` resume continues.
    """

    def __init__(self, epochs: int) -> None:
        super().__init__()
        self._epochs = epochs

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Any) -> None:
        """Set ``should_stop`` once the configured number of epochs has completed."""
        del pl_module
        if trainer.current_epoch + 1 >= self._epochs:
            trainer.should_stop = True


class _FixedDetectionDataset(Dataset[tuple[Tensor, Targets]]):
    """In-memory dataset of precomputed ``(image, Targets)`` samples (pure ``__getitem__``)."""

    def __init__(self, samples: list[tuple[Tensor, Targets]]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        """Return the sample count."""
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Targets]:
        """Return the precomputed sample at ``index`` (identical across epochs and runs)."""
        return self._samples[index]


def _build_samples() -> list[tuple[Tensor, Targets]]:
    """Precompute the fixed sample list from a local generator (independent of global RNG)."""
    generator = torch.Generator().manual_seed(_SEED)
    samples: list[tuple[Tensor, Targets]] = []
    for _ in range(_DATASET_SIZE):
        image = torch.randn(3, _IMG_SIZE, _IMG_SIZE, generator=generator)
        top_left = torch.rand(_INSTANCES_PER_IMAGE, 2, generator=generator) * 30.0
        size = torch.rand(_INSTANCES_PER_IMAGE, 2, generator=generator) * 20.0 + 8.0
        boxes = torch.cat([top_left, top_left + size], dim=1)
        labels = torch.randint(0, _NUM_CLASSES, (_INSTANCES_PER_IMAGE,), generator=generator)
        samples.append((image, Targets(boxes=boxes, labels=labels)))
    return samples


def _collate_unpacked(batch: list[tuple[Tensor, Targets]]) -> tuple[Tensor, list[Targets]]:
    """Collate, then restore the ``list[Targets]`` the module consumes.

    No datamodule is attached to these bare-loader ``Trainer.fit`` runs, so the
    transfer hook that restores the transport form never fires; this wrapper runs
    the same collate -> restore round-trip the datamodule performs in production
    (dequantizing the uint8 images and unpacking the targets) so the module
    receives its float images and ragged target list.
    """
    return unpack_batch(collate_detection(batch))


@pytest.fixture
def train_loader() -> DataLoader[tuple[Tensor, Targets]]:
    """Return an unshuffled loader over the fixed dataset — identical stream every run."""
    return DataLoader(
        _FixedDetectionDataset(_build_samples()),
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_unpacked,
    )


def _tiny_module() -> DetectionLitModule:
    """Build an n-scale module with a low channel cap for fast CPU training."""
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=_NUM_CLASSES)


def _trainer(*callbacks: Callback) -> Trainer:
    """Build a deterministic ``max_epochs=4`` CPU trainer with the given callbacks."""
    return Trainer(
        max_epochs=_MAX_EPOCHS,
        accelerator="cpu",
        deterministic=True,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=list(callbacks),
    )


def test_trajectory_match(train_loader: DataLoader[tuple[Tensor, Targets]], tmp_path: Path) -> None:
    """A resumed run reproduces the reference run's post-checkpoint loss trajectory exactly."""
    seed_everything(_SEED, workers=True)
    reference_recorder = _LossTrajectoryRecorder()
    _trainer(reference_recorder).fit(_tiny_module(), train_dataloaders=train_loader)

    seed_everything(_SEED, workers=True)
    interrupted = _trainer(_LossTrajectoryRecorder(), _StopAfterEpoch(_RESUME_EPOCH))
    interrupted.fit(_tiny_module(), train_dataloaders=train_loader)
    checkpoint = tmp_path / "epoch2.ckpt"
    interrupted.save_checkpoint(checkpoint)

    seed_everything(_SEED, workers=True)
    resumed_recorder = _LossTrajectoryRecorder()
    _trainer(resumed_recorder).fit(_tiny_module(), train_dataloaders=train_loader, ckpt_path=str(checkpoint))

    post_resume_reference = reference_recorder.losses[_RESUME_EPOCH * _STEPS_PER_EPOCH :]
    assert resumed_recorder.losses == pytest.approx(post_resume_reference, rel=1e-4)


def test_resume_continues_step_epoch_and_optimizer_state(
    train_loader: DataLoader[tuple[Tensor, Targets]], tmp_path: Path
) -> None:
    """After resume the global step and epoch continue and MuSGD momentum buffers are restored."""
    seed_everything(_SEED, workers=True)
    interrupted = _trainer(_LossTrajectoryRecorder(), _StopAfterEpoch(_RESUME_EPOCH))
    interrupted.fit(_tiny_module(), train_dataloaders=train_loader)
    checkpoint = tmp_path / "epoch2.ckpt"
    interrupted.save_checkpoint(checkpoint)

    seed_everything(_SEED, workers=True)
    resumed = _trainer(_LossTrajectoryRecorder())
    resumed.fit(_tiny_module(), train_dataloaders=train_loader, ckpt_path=str(checkpoint))

    assert resumed.global_step == _MAX_EPOCHS * _STEPS_PER_EPOCH
    assert resumed.current_epoch == _MAX_EPOCHS
    optimizer_state = resumed.optimizers[0].state_dict()["state"]
    assert optimizer_state
    assert all("momentum_buffer" in param_state for param_state in optimizer_state.values())
