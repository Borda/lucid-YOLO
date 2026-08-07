# SPDX-License-Identifier: Apache-2.0
"""End-to-end gate on the ``scripts/eval_det.py`` entry point.

The library half of segmentation evaluation was gated by WP-053b and the script
half was not, which is exactly where the two came apart: the script loaded ground
truth without masks, so a segmentation checkpoint scored as a detector and said
nothing about it. These tests drive ``main`` the way a caller does -- a checkpoint
on disk, a COCO-layout data root, an output report -- and assert on what the
report contains.

The failure this file exists to catch is silent by construction. Every way the
mask path can drop out (ground truth loaded without masks, the module handed to
the evaluator unwrapped so its forward never returns a
:class:`~lucid_yolo.models.build.SegmentOutput`) produces a complete, plausible,
finite 12-statistic detection report and no error at all. Only the *presence* of
the ``segm_`` keys separates a working segmentation eval from a broken one, so
that is what is asserted -- never the values, which an untrained model makes
meaningless.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest
import torch

from lucid_yolo.eval.coco_eval import _METRIC_KEYS, _SEGM_PREFIX
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import eval_det  # noqa: E402  (path arranged just above)

_SPLIT = "train"
_ANNOTATION = "_annotations.coco.json"
_NUM_CLASSES = 4  # the detseg fixture carries category ids 1..4
_CANVAS = 160  # divisible by 32, and upscales the 128px fixtures so the inverse letterbox works
_LIMIT = 3  # a few images keep the mask decode and RLE encode quick
_SEGM_KEYS = {_SEGM_PREFIX + key for key in _METRIC_KEYS}


def _module(task: str) -> DetectionLitModule:
    """Build an untrained n-scale module for ``task``."""
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=_NUM_CLASSES, task=task)


def _write_checkpoint(module: DetectionLitModule, path: Path) -> Path:
    """Save ``module`` in the Lightning checkpoint layout ``load_from_checkpoint`` reads."""
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": "2.4.0",
            "epoch": 0,
            "global_step": 0,
            "loops": {},
        },
        path,
    )
    return path


def _data_root(fixture_dir: Path, tmp_path: Path) -> Path:
    """Lay the fixture split out under the ``val2017`` layout the script expects."""
    root = tmp_path / "coco"
    (root / "annotations").mkdir(parents=True)
    split_dir = fixture_dir / _SPLIT
    (root / "annotations" / "instances_val2017.json").write_text((split_dir / _ANNOTATION).read_text())
    (root / "val2017").symlink_to(split_dir, target_is_directory=True)
    return root


def _run(checkpoint: Path, root: Path, tmp_path: Path, *extra: str) -> dict[str, dict[str, float]]:
    """Run the script and return the ``report`` half of its JSON output."""
    output = tmp_path / f"report{len(extra)}.json"
    eval_det.main(
        [
            str(checkpoint),
            "--data-root",
            str(root),
            "--no-ema",
            "--limit",
            str(_LIMIT),
            "--img-size",
            str(_CANVAS),
            "--device",
            "cpu",
            "--output",
            str(output),
            *extra,
        ]
    )
    payload = json.loads(output.read_text())
    report: dict[str, dict[str, float]] = payload["report"]
    return report


def test_segmentation_checkpoint_reports_segm_statistics(detseg_fixture_dir: Path, tmp_path: Path) -> None:
    """A segmentation checkpoint reports the mask statistics on both decode paths.

    The gate on the whole script path. It fails if the ground truth is loaded
    without masks, and it fails if the module reaches the evaluator unwrapped --
    ``DetectionLitModule.forward`` returns the detection head's output alone, so
    the evaluator would never see prototypes and would report a detection run.
    """
    torch.manual_seed(0)
    checkpoint = _write_checkpoint(_module("segment"), tmp_path / "segment.ckpt")

    report = _run(checkpoint, _data_root(detseg_fixture_dir, tmp_path), tmp_path)

    assert set(report) == {"e2e", "nms"}
    for stats in report.values():
        assert set(stats) == set(_METRIC_KEYS) | _SEGM_KEYS
        assert all(value == pytest.approx(value) for value in stats.values())  # finite, not NaN


def test_detection_checkpoint_is_unchanged(detseg_fixture_dir: Path, tmp_path: Path) -> None:
    """A detection checkpoint reports the same twelve keys it always did, and no masks.

    Segmentation support must be reachable only through a checkpoint that carries
    a mask branch; a report padded with zeroed ``segm_`` entries would change what
    every existing detection run reports, and paying the ground-truth mask decode
    for a detector would be pure cost.
    """
    torch.manual_seed(0)
    checkpoint = _write_checkpoint(_module("detect"), tmp_path / "detect.ckpt")

    report = _run(checkpoint, _data_root(detseg_fixture_dir, tmp_path), tmp_path)

    for stats in report.values():
        assert set(stats) == set(_METRIC_KEYS)


def test_no_masks_forces_the_detection_reading(detseg_fixture_dir: Path, tmp_path: Path) -> None:
    """``--no-masks`` scores a segmentation checkpoint on boxes alone."""
    torch.manual_seed(0)
    checkpoint = _write_checkpoint(_module("segment"), tmp_path / "segment.ckpt")

    report = _run(checkpoint, _data_root(detseg_fixture_dir, tmp_path), tmp_path, "--no-masks")

    for stats in report.values():
        assert set(stats) == set(_METRIC_KEYS)
