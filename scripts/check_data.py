# SPDX-License-Identifier: Apache-2.0
"""Validate a real COCO 2017 dataset root before a ``[DATA]`` run (WP-014).

Datasets are never committed and never auto-downloaded (AGENTS.md sec. 3); this
script is the ``make check-data`` gate that confirms a provisioned root actually
matches the COCO 2017 layout of blueprint sec. 14.3 before any ``[DATA]`` work
runs against it. For each split it checks that

    * the images directory exists and holds the expected number of image files
      (118,287 for ``train2017``, 5,000 for ``val2017``);
    * the ``annotations/instances_*.json`` file exists and parses;
    * the image count declared inside the annotation JSON matches the number of
      image files on disk.

The check core is importable (:func:`check_coco_root` returns a
:class:`DataCheck` whose ``ok`` flag drives the exit status); :func:`main` is a
thin CLI over it. The expected per-split counts are parameters (defaulting to the
real COCO 2017 totals) so the logic is unit-testable against a tiny fake layout.

Relationship to :mod:`lucid_yolo.data.verify`:
    This script is the developer-only ``make check-data`` gate and validates the
    layout by **counts** (fixed per-split totals plus annotation-vs-disk count
    parity). The packaged :mod:`lucid_yolo.data.verify` module performs the
    complementary **per-file existence** check exposed to pip-installed users via
    ``lucid-download --verify`` / ``--verify-only``. The two are kept separate on
    purpose: their logic differs materially (count parity vs. naming exactly which
    annotated images are absent), and this script is intentionally not shipped in
    the wheel.

Examples:
    Validate a provisioned root (exit 1 on any mismatch)::

        python scripts/check_data.py --data-root /data/coco
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

#: Expected image counts for the two COCO 2017 splits (blueprint sec. 14.3).
COCO_TRAIN_COUNT = 118287
COCO_VAL_COUNT = 5000
#: Image file suffixes counted on disk (COCO 2017 ships ``.jpg``).
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class SplitCheck:
    """Validation outcome for one dataset split.

    Attributes:
        name: Split name (e.g. ``"train2017"``).
        expected_images: Expected on-disk image count for this split.
        found_images: Image files actually found on disk (``-1`` if the directory
            is missing).
        annotation_images: Image entries declared in the annotation JSON (``-1``
            if the file is missing or failed to parse).
        problems: Human-readable problem descriptions; empty when the split is OK.
    """

    name: str
    expected_images: int
    found_images: int
    annotation_images: int
    problems: list[str]

    @property
    def ok(self) -> bool:
        """Return whether the split passed every check."""
        return not self.problems


@dataclass(frozen=True)
class DataCheck:
    """Aggregate validation outcome across all splits.

    Attributes:
        splits: The per-split :class:`SplitCheck` results.
    """

    splits: list[SplitCheck]

    @property
    def ok(self) -> bool:
        """Return whether every split passed."""
        return all(split.ok for split in self.splits)


def _count_images(images_dir: Path) -> int:
    """Count image files directly under ``images_dir`` (``-1`` if it is missing).

    Args:
        images_dir: Candidate split image directory.

    Returns:
        The number of files whose suffix is a known image suffix, or ``-1`` when
        the directory does not exist.
    """
    if not images_dir.is_dir():
        return -1
    return sum(1 for path in images_dir.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)


def _count_annotation_images(annotation_file: Path) -> tuple[int, str | None]:
    """Return the ``images`` count declared in ``annotation_file`` and any error.

    Args:
        annotation_file: Path to the split's ``instances_*.json``.

    Returns:
        A ``(count, error)`` pair: ``count`` is the number of image entries (or
        ``-1`` on failure) and ``error`` is a message when the file is missing or
        cannot be parsed, else ``None``.
    """
    if not annotation_file.is_file():
        return -1, f"annotation file missing: {annotation_file}"
    try:
        with annotation_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return -1, f"annotation file failed to parse: {annotation_file} ({error})"
    return len(payload.get("images", [])), None


def check_split(name: str, images_dir: Path, annotation_file: Path, expected_images: int) -> SplitCheck:
    """Validate one split's image directory and annotation file.

    Args:
        name: Split name for messages.
        images_dir: The split's image directory.
        annotation_file: The split's annotation JSON.
        expected_images: The image count this split must have.

    Returns:
        A :class:`SplitCheck` capturing the counts and any problems found.
    """
    problems: list[str] = []
    found = _count_images(images_dir)
    if found < 0:
        problems.append(f"images directory missing: {images_dir}")
    elif found != expected_images:
        problems.append(f"{name}: expected {expected_images} images, found {found}")
    ann_count, ann_error = _count_annotation_images(annotation_file)
    if ann_error is not None:
        problems.append(ann_error)
    elif found >= 0 and ann_count != found:
        problems.append(f"{name}: {ann_count} images in annotations, {found} on disk")
    return SplitCheck(
        name=name,
        expected_images=expected_images,
        found_images=found,
        annotation_images=ann_count,
        problems=problems,
    )


def check_coco_root(
    data_root: Path,
    expected_train: int = COCO_TRAIN_COUNT,
    expected_val: int = COCO_VAL_COUNT,
) -> DataCheck:
    """Validate a COCO 2017 root against the blueprint sec. 14.3 layout.

    Args:
        data_root: Directory holding ``train2017``, ``val2017`` and
            ``annotations``.
        expected_train: Expected ``train2017`` image count (defaults to the real
            COCO total, overridable so the logic is testable on a fake layout).
        expected_val: Expected ``val2017`` image count.

    Returns:
        A :class:`DataCheck` aggregating both splits' outcomes.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> check_coco_root(Path("/nonexistent")).ok
        False

        ```
    """
    annotations = data_root / "annotations"
    train = check_split("train2017", data_root / "train2017", annotations / "instances_train2017.json", expected_train)
    val = check_split("val2017", data_root / "val2017", annotations / "instances_val2017.json", expected_val)
    return DataCheck(splits=[train, val])


def format_report(result: DataCheck, data_root: Path) -> str:
    """Render a human-readable verdict for ``result``.

    Args:
        result: The aggregate check outcome.
        data_root: The root that was checked, named in the header.

    Returns:
        A multi-line report string ending in an overall PASS/FAIL verdict.
    """
    lines = [f"check-data: {data_root}"]
    for split in result.splits:
        if split.ok:
            lines.append(f"  PASS {split.name} — {split.found_images} images")
        else:
            lines.extend(f"  FAIL {problem}" for problem in split.problems)
    lines.append("PASS: dataset layout valid" if result.ok else "FAIL: dataset layout invalid")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Validate the COCO root named on the command line.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when the layout is valid, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description="Validate a COCO 2017 dataset root.")
    parser.add_argument("--data-root", type=Path, required=True, help="COCO 2017 root directory")
    args = parser.parse_args(argv)
    result = check_coco_root(args.data_root)
    print(format_report(result, args.data_root))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
