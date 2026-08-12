# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the oriented acceptance instrument (WP-095).

Runs offline against a tiny tiled layout built by :mod:`lucid_yolo.data.tiles` from a
synthetic DOTA split, so the instrument is exercised over exactly the on-disk shape the
tier run feeds it — an untrained module scores near zero, which is fine: what is under
test is that ground truth, predictions and the accumulator meet correctly, not that a
random network detects anything.

The scoring contract itself is pinned separately and without a model: feeding the split's
own ground truth back in as perfect predictions must score 1.0, which is what says the
targets reach :func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map` in the frame and
the label space it expects. A wrong frame or a shifted label space fails that test while
"it ran and produced a float" passes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lucid_yolo.data import tiles as build
from lucid_yolo.eval import rotated_eval as evaluate
from lucid_yolo.eval.dota_eval import evaluate_rotated_map
from lucid_yolo.ptl.module import DetectionLitModule

#: Letterbox side and tile side used throughout: divisible by the level-32 stride.
IMG_SIZE = 64
#: Source image side, tiled into four windows at the patch/overlap below.
SIDE = 96
PATCH = 64
OVERLAP = 32


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed the global RNG before each test (the module is randomly initialized)."""
    torch.manual_seed(0)


@pytest.fixture
def tiled_root(tmp_path: Path) -> Path:
    """Build a one-image, one-class tiled layout with a single oriented object."""
    split_dir = tmp_path / "dota" / "val"
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "labelTxt").mkdir(parents=True)
    image = (torch.arange(3 * SIDE * SIDE, dtype=torch.uint8) % 251).reshape(3, SIDE, SIDE)
    build.write_png(image, str(split_dir / "images" / "P0001.png"))
    (split_dir / "labelTxt" / "P0001.txt").write_text("10 10 30 10 30 20 10 20 plane 0\n")

    build.convert_split(split_dir, tmp_path / "tiles", "val", patch=PATCH, overlap=OVERLAP)
    return tmp_path / "tiles"


def _datamodule(root: Path) -> object:
    """A val-only datamodule over the built split, set up and ready to iterate."""
    datamodule = evaluate.build_datamodule(root, "val", img_size=IMG_SIZE, batch_size=2, variant="n")
    datamodule.setup("validate")
    return datamodule


def test_the_eval_package_does_not_reach_up_into_the_training_layer() -> None:
    """Importing ``lucid_yolo.eval`` must not drag in ``lucid_yolo.ptl``.

    ``eval.checkpoint`` needs :class:`DetectionLitModule`, and that module imports
    ``eval.dota_eval`` — so re-exporting the loader from the eval package closes a cycle
    and every test module that imports both dies at collection with a partially
    initialized module. Re-exporting it is the natural tidying move, which is why the
    direction of the dependency is asserted rather than left to a comment.
    """
    probe = "import lucid_yolo.eval, sys; print(any(m.startswith('lucid_yolo.ptl') for m in sys.modules))"

    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "False"


def test_a_detection_checkpoint_is_refused_rather_than_scored(tiled_root: Path) -> None:
    """A module with no angle stem has no oriented number, so it raises instead of reporting one."""
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15).eval()

    with pytest.raises(ValueError, match="oriented checkpoint"):
        evaluate.score_split(module, _datamodule(tiled_root), torch.device("cpu"), img_size=IMG_SIZE)


def test_the_split_is_scored_tile_by_tile(tiled_root: Path) -> None:
    """Every tile of the split is scored, and the report counts the instances it saw."""
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()

    metrics, tiles, instances = evaluate.score_split(
        module, _datamodule(tiled_root), torch.device("cpu"), img_size=IMG_SIZE
    )

    assert tiles == 4
    assert instances >= 1
    assert 0.0 <= metrics["map_50"] <= 1.0


def test_the_limit_stops_early_without_changing_the_frame(tiled_root: Path) -> None:
    """``--limit`` scores a prefix of the split rather than a resampled subset."""
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()

    _, tiles, _ = evaluate.score_split(module, _datamodule(tiled_root), torch.device("cpu"), img_size=IMG_SIZE, limit=2)

    assert tiles == 2


def test_perfect_predictions_score_one(tiled_root: Path) -> None:
    """The split's own ground truth, fed back as detections, scores 1.0.

    This is what pins the frame and the label space: the targets the instrument collects
    are scored by the same accumulator the instrument uses, so a shifted label space or a
    ground truth left in tile coordinates while predictions are letterboxed would fail
    here while any "it produced a float" assertion passed.
    """
    datamodule = _datamodule(tiled_root)
    ground_truth: list[dict[str, torch.Tensor]] = []
    for batch in datamodule.val_dataloader():  # type: ignore[attr-defined]
        _, targets, _ = datamodule.on_after_batch_transfer(batch, 0)  # type: ignore[attr-defined]
        ground_truth.extend(
            {"rboxes": t.rboxes, "labels": t.labels.to(torch.long), "difficult": t.difficult} for t in targets
        )
    predictions = [
        {"rboxes": target["rboxes"], "labels": target["labels"], "scores": torch.ones(target["labels"].shape[0])}
        for target in ground_truth
    ]

    metrics = evaluate_rotated_map(predictions, ground_truth)

    assert metrics["map_50"] == pytest.approx(1.0)
