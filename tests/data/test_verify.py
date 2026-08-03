# SPDX-License-Identifier: Apache-2.0
"""Offline unit gate for the packaged dataset verifier (``lucid_yolo.data.verify``).

Every test builds a tiny fake COCO layout on ``tmp_path`` and exercises the
per-file existence checks without any network or real dataset: a clean root
passes, an interrupted-extraction root (annotations list more images than the
directory holds) fails with the right counts and a named missing sample, and a
missing annotation file is reported.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from lucid_yolo.data import verify


def _write_split(
    root: Path,
    split_dir: str,
    annotation_name: str,
    annotated: Sequence[str],
    *,
    on_disk: Sequence[str] | None = None,
) -> None:
    """Write a fake split: annotations listing ``annotated``, files ``on_disk``."""
    (root / split_dir).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    images = [{"id": i, "file_name": name, "height": 4, "width": 4} for i, name in enumerate(annotated)]
    (root / "annotations" / annotation_name).write_text(json.dumps({"images": images}), encoding="utf-8")
    for name in annotated if on_disk is None else on_disk:
        (root / split_dir / name).write_bytes(b"jpegbytes")


def _names(count: int) -> list[str]:
    """Return ``count`` COCO-style zero-padded jpg file names."""
    return [f"{i:012d}.jpg" for i in range(count)]


# --------------------------------------------------------------------------- #
# verify_split
# --------------------------------------------------------------------------- #


def test_verify_split_clean_passes(tmp_path: Path) -> None:
    """A split whose every annotated image is on disk verifies OK."""
    _write_split(tmp_path, "val2017", "instances_val2017.json", _names(3))

    result = verify.verify_split("val2017", tmp_path / "val2017", tmp_path / "annotations" / "instances_val2017.json")

    assert result.ok
    assert (result.expected, result.present, result.missing_count) == (3, 3, 0)
    assert result.missing == []
    assert result.annotation_present is True


def test_verify_split_missing_images_reports_counts_and_sample(tmp_path: Path) -> None:
    """A partial extraction fails with exact counts and the named missing files."""
    annotated = _names(5)
    _write_split(tmp_path, "val2017", "instances_val2017.json", annotated, on_disk=annotated[:2])

    result = verify.verify_split("val2017", tmp_path / "val2017", tmp_path / "annotations" / "instances_val2017.json")

    assert not result.ok
    assert (result.expected, result.present, result.missing_count) == (5, 2, 3)
    assert result.missing == annotated[2:]
    assert any("3 of 5 annotated images missing" in problem for problem in result.problems)


def test_verify_split_missing_annotation_file_fails(tmp_path: Path) -> None:
    """A split with no annotation JSON fails and flags the annotation absent."""
    (tmp_path / "val2017").mkdir()

    result = verify.verify_split("val2017", tmp_path / "val2017", tmp_path / "annotations" / "instances_val2017.json")

    assert not result.ok
    assert result.annotation_present is False
    assert any("annotation file missing" in problem for problem in result.problems)


def test_verify_split_unparsable_annotation_fails(tmp_path: Path) -> None:
    """A corrupt (non-JSON) annotation file fails and flags the annotation absent."""
    (tmp_path / "val2017").mkdir()
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "instances_val2017.json").write_text("{ not json", encoding="utf-8")

    result = verify.verify_split("val2017", tmp_path / "val2017", tmp_path / "annotations" / "instances_val2017.json")

    assert not result.ok
    assert result.annotation_present is False
    assert any("failed to parse" in problem for problem in result.problems)


def test_verify_split_missing_images_directory_fails(tmp_path: Path) -> None:
    """A split whose image directory is absent fails even with valid annotations."""
    (tmp_path / "annotations").mkdir()
    images = [{"id": 0, "file_name": "000000000000.jpg", "height": 4, "width": 4}]
    (tmp_path / "annotations" / "instances_val2017.json").write_text(json.dumps({"images": images}), encoding="utf-8")

    result = verify.verify_split("val2017", tmp_path / "val2017", tmp_path / "annotations" / "instances_val2017.json")

    assert not result.ok
    assert any("images directory missing" in problem for problem in result.problems)


def test_verify_split_caps_missing_sample(tmp_path: Path) -> None:
    """The missing sample is capped while ``missing_count`` stays the true total."""
    annotated = _names(30)
    _write_split(tmp_path, "val2017", "instances_val2017.json", annotated, on_disk=[])

    result = verify.verify_split(
        "val2017", tmp_path / "val2017", tmp_path / "annotations" / "instances_val2017.json", sample=4
    )

    assert result.missing_count == 30
    assert result.missing == annotated[:4]


# --------------------------------------------------------------------------- #
# verify_coco_root
# --------------------------------------------------------------------------- #


def test_verify_coco_root_clean_passes(tmp_path: Path) -> None:
    """A fully provisioned val root verifies OK end to end."""
    _write_split(tmp_path, "val2017", "instances_val2017.json", _names(4))

    result = verify.verify_coco_root(tmp_path, ["val"])

    assert result.ok
    assert [split.name for split in result.splits] == ["val2017"]


def test_verify_coco_root_incomplete_split_fails(tmp_path: Path) -> None:
    """A root with a partially extracted split reports not-ok with the shortfall."""
    annotated = _names(6)
    _write_split(tmp_path, "val2017", "instances_val2017.json", annotated, on_disk=annotated[:1])

    result = verify.verify_coco_root(tmp_path, ["val"])

    assert not result.ok
    (split,) = result.splits
    assert (split.expected, split.present, split.missing_count) == (6, 1, 5)


def test_verify_coco_root_collapses_duplicate_splits(tmp_path: Path) -> None:
    """Duplicate split names are collapsed to a single verification entry."""
    _write_split(tmp_path, "val2017", "instances_val2017.json", _names(2))

    result = verify.verify_coco_root(tmp_path, ["val", "val"])

    assert [split.name for split in result.splits] == ["val2017"]


def test_verify_coco_root_unknown_split_raises(tmp_path: Path) -> None:
    """An unknown split name raises a helpful ValueError."""
    with pytest.raises(ValueError, match="unknown split 'test'"):
        verify.verify_coco_root(tmp_path, ["test"])


# --------------------------------------------------------------------------- #
# format_report
# --------------------------------------------------------------------------- #


def test_format_report_clean_ends_in_pass(tmp_path: Path) -> None:
    """The clean-root report ends in the overall PASS verdict."""
    _write_split(tmp_path, "val2017", "instances_val2017.json", _names(2))

    report = verify.format_report(verify.verify_coco_root(tmp_path, ["val"]), tmp_path)

    assert report.splitlines()[-1] == "PASS: every annotated image is present"
    assert "PASS val2017" in report


def test_format_report_missing_lists_sample_and_fail(tmp_path: Path) -> None:
    """The failing report names the missing sample and ends in the FAIL verdict."""
    annotated = _names(15)
    _write_split(tmp_path, "val2017", "instances_val2017.json", annotated, on_disk=[])

    report = verify.format_report(verify.verify_coco_root(tmp_path, ["val"]), tmp_path)

    assert report.splitlines()[-1] == "FAIL: dataset is incomplete"
    assert "missing e.g.:" in report
    assert "more)" in report
