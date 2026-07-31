# SPDX-License-Identifier: Apache-2.0
"""DoD test for WP-007: the synthetic micro-fixtures load and are deterministic.

Validates the A26 contract end to end: the detection/segmentation set carries
boxes *and* polygons, the oriented set carries four-corner rotated boxes, and
regeneration under a fixed seed is byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import synthetic

if TYPE_CHECKING:
    from pathlib import Path

_SPLIT = "train"
_ANNOTATION = "_annotations.coco.json"
_MIN_POLYGON_COORDS = 6  # >= 3 (x, y) points for a non-degenerate filled polygon
_OBB_CORNER_COORDS = 8  # exactly 4 (x, y) corner points for a rotated box


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_detseg_set_loads(detseg_fixture_dir: Path) -> None:
    """16 images exist, COCO parses, every annotation has a box and a polygon, >= 2 categories."""
    split_dir = detseg_fixture_dir / _SPLIT
    images = sorted(split_dir.glob("*.jpg"))
    assert len(images) == synthetic.DETSEG_NUM_IMAGES == 16

    doc = json.loads((split_dir / _ANNOTATION).read_text(encoding="utf-8"))
    assert len(doc["images"]) == 16
    assert doc["annotations"], "expected at least one annotation"

    for ann in doc["annotations"]:
        assert len(ann["bbox"]) == 4, "annotation is missing an axis-aligned bbox"
        polygons = ann.get("segmentation")
        assert polygons and len(polygons[0]) >= _MIN_POLYGON_COORDS, "annotation is missing a non-empty polygon"

    category_ids = {ann["category_id"] for ann in doc["annotations"]}
    assert len(category_ids) >= 2, f"need >= 2 distinct categories, got {sorted(category_ids)}"


def test_obb_set_loads(obb_fixture_dir: Path) -> None:
    """8 images exist and every annotation carries a four-point rotated box."""
    split_dir = obb_fixture_dir / _SPLIT
    images = sorted(split_dir.glob("*.jpg"))
    assert len(images) == synthetic.OBB_NUM_IMAGES == 8

    doc = json.loads((split_dir / _ANNOTATION).read_text(encoding="utf-8"))
    assert len(doc["images"]) == 8
    assert doc["annotations"], "expected at least one annotation"

    for ann in doc["annotations"]:
        corners = ann.get("segmentation")
        assert corners, "OBB annotation is missing its rotated-box corners"
        assert len(corners[0]) == _OBB_CORNER_COORDS, "rotated box must have exactly 4 corner points"


def test_detseg_generation_is_deterministic(tmp_path: Path) -> None:
    """Same seed -> byte-identical annotation JSON and identical image SHA-256 hashes (A26)."""
    first = synthetic.generate_detseg_fixtures(tmp_path / "run_a")
    second = synthetic.generate_detseg_fixtures(tmp_path / "run_b")

    ann_first = (first / _SPLIT / _ANNOTATION).read_bytes()
    ann_second = (second / _SPLIT / _ANNOTATION).read_bytes()
    assert ann_first == ann_second, "annotation JSON is not byte-identical across seeded runs"

    images_first = sorted((first / _SPLIT).glob("*.jpg"))
    images_second = sorted((second / _SPLIT).glob("*.jpg"))
    assert [p.name for p in images_first] == [p.name for p in images_second]
    assert images_first, "expected generated images to compare"
    for a, b in zip(images_first, images_second, strict=True):
        assert _sha256(a) == _sha256(b), f"image bytes differ across seeded runs: {a.name}"
