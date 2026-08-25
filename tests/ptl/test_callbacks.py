# SPDX-License-Identifier: Apache-2.0
"""Tests for the checkpoint-path callback wiring in the detection CLI (WP-127).

Lightning's own default writes ``lightning_logs/version_N/checkpoints/epoch=X-step=Y.ckpt``
-- nothing in the path or filename names the task or scale variant, so identifying what a
saved checkpoint trained meant opening its ``hparams.yaml`` (surfaced re-scoring WP-064's
OBB-smoke checkpoint for WP-111's row 111). Covers the DoD: a real ``Trainer.fit`` run's
checkpoint filename carries its task and variant while ``lightning_logs/version_N``
numbering and the ``epoch=X-step=Y`` suffix stay exactly as before
(``test_fit_checkpoint_names_task_and_variant``), and
:class:`~lucid_yolo.cli.train.DetectionCLI` injects that callback by default unless a
config already places its own :class:`~pytorch_lightning.callbacks.ModelCheckpoint`
(``TestDetectionCLICheckpointInjection``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from lucid_yolo.cli.train import DetectionCLI, _checkpoint_filename, default_determinism, packaged_config
from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl import DetectionDataModule, DetectionLitModule, collate_detection, unpack_batch

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from torch import Tensor

#: Class count of the tiny test head.
_NUM_CLASSES = 4
#: Square input side; divisible by every stride (8, 16, 32) for an integer anchor grid.
_IMG_SIZE = 64
#: Images per batch.
_BATCH_SIZE = 2
#: Precomputed samples in the fixed dataset (two batches).
_DATASET_SIZE = 4
#: Instances per synthetic image.
_INSTANCES_PER_IMAGE = 2


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed torch before each test so the tiny module and dataset are reproducible."""
    torch.manual_seed(0)
    yield


class _FixedDetectionDataset(Dataset[tuple["Tensor", Targets]]):
    """In-memory dataset of precomputed ``(image, Targets)`` samples (pure ``__getitem__``)."""

    def __init__(self, samples: list[tuple[Tensor, Targets]]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        """Return the sample count."""
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Targets]:
        """Return the precomputed sample at ``index``."""
        return self._samples[index]


def _train_loader() -> DataLoader[tuple[Tensor, Targets]]:
    """Build a two-batch loader of synthetic detection samples.

    Examples:
        >>> loader = _train_loader()
        >>> len(loader.dataset)
        4
    """
    samples: list[tuple[Tensor, Targets]] = []
    for _ in range(_DATASET_SIZE):
        image = torch.randn(3, _IMG_SIZE, _IMG_SIZE)
        top_left = torch.rand(_INSTANCES_PER_IMAGE, 2) * 30.0
        size = torch.rand(_INSTANCES_PER_IMAGE, 2) * 20.0 + 8.0
        boxes = torch.cat([top_left, top_left + size], dim=1)
        labels = torch.randint(0, _NUM_CLASSES, (_INSTANCES_PER_IMAGE,))
        samples.append((image, Targets(boxes=boxes, labels=labels)))
    return DataLoader(
        _FixedDetectionDataset(samples),
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: unpack_batch(collate_detection(batch)),
    )


def _tiny_module() -> DetectionLitModule:
    """Build an n-scale module with a low channel cap for a fast CPU step.

    Examples:
        >>> _tiny_module().task
        'detect'
    """
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=_NUM_CLASSES)


class TestCheckpointFilename:
    """Unit coverage of the pure :func:`~lucid_yolo.cli.train._checkpoint_filename` helper."""

    @pytest.mark.parametrize(
        ("task", "variant", "expected"),
        [
            pytest.param("detect", "n", "detect_n_{epoch}-{step}", id="detect-n"),
            pytest.param("segment", "s", "segment_s_{epoch}-{step}", id="segment-s"),
            pytest.param("obb", "m", "obb_m_{epoch}-{step}", id="obb-m"),
        ],
    )
    def test_prefixes_task_and_variant_onto_the_default_suffix(self, task: str, variant: str, expected: str) -> None:
        """The task and variant prefix ``{epoch}-{step}``, Lightning's own default suffix.

        The suffix stays a literal template (not pre-filled) so
        :class:`~pytorch_lightning.callbacks.ModelCheckpoint` still fills it in from
        the run's own metrics -- only the run-identity prefix is new.
        """
        assert _checkpoint_filename(task, variant) == expected


def test_fit_checkpoint_names_task_and_variant(tmp_path: Path) -> None:
    """A real one-epoch fit's checkpoint filename carries its task and variant.

    Builds the exact callback :meth:`~lucid_yolo.cli.train.DetectionCLI.instantiate_trainer`
    would inject for a ``task="detect"``, ``variant="s"`` run, fits one epoch on a tiny
    synthetic loader, and checks the saved ``.ckpt`` under the resolved
    ``lightning_logs/version_0/checkpoints/`` directory: the ``lightning_logs/version_N``
    numbering and the ``epoch=X-step=Y`` suffix are unchanged, only the new
    ``detect_s_`` prefix is added.
    """
    module = _tiny_module()
    checkpoint = ModelCheckpoint(filename=_checkpoint_filename(module.task, "s"))
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        deterministic=True,
        default_root_dir=str(tmp_path),
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[checkpoint],
    )
    trainer.fit(module, train_dataloaders=_train_loader())

    checkpoints_dir = tmp_path / "lightning_logs" / "version_0" / "checkpoints"
    saved = list(checkpoints_dir.glob("detect_s_epoch=*-step=*.ckpt"))
    assert len(saved) == 1
    assert saved[0].name == f"detect_s_epoch=0-step={trainer.global_step}.ckpt"


class TestDetectionCLICheckpointInjection:
    """:meth:`DetectionCLI.instantiate_trainer` injects a task/variant checkpoint callback."""

    @staticmethod
    def _build_cli(*args: str) -> DetectionCLI:
        """Build the detection CLI in non-running mode, mirroring :func:`lucid_yolo.cli.train.main`."""
        return DetectionCLI(
            DetectionLitModule,
            DetectionDataModule,
            trainer_defaults={"deterministic": default_determinism()},
            seed_everything_default=0,
            run=False,
            args=list(args),
        )

    def test_injects_a_checkpoint_naming_the_configs_task_and_variant(self) -> None:
        """The det-smoke config (``task: detect``, default ``variant: n``) gets a matching filename."""
        cli = self._build_cli("--config", str(packaged_config("det_nano_smoke")))
        checkpoints = [callback for callback in cli.trainer.callbacks if isinstance(callback, ModelCheckpoint)]
        assert len(checkpoints) == 1
        assert checkpoints[0].filename == _checkpoint_filename("detect", "n")

    def test_a_user_supplied_checkpoint_callback_is_not_duplicated(self, tmp_path: Path) -> None:
        """A config that already places a ``ModelCheckpoint`` wins; none is injected alongside it."""
        override = tmp_path / "checkpoint_override.yaml"
        override.write_text(
            yaml.safe_dump(
                {
                    "trainer": {
                        "callbacks": [
                            {
                                "class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                                "init_args": {"filename": "custom"},
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        cli = self._build_cli("--config", str(packaged_config("det_nano_smoke")), "--config", str(override))
        checkpoints = [callback for callback in cli.trainer.callbacks if isinstance(callback, ModelCheckpoint)]
        assert len(checkpoints) == 1
        assert checkpoints[0].filename == "custom"
