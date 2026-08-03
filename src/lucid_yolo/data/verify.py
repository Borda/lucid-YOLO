# SPDX-License-Identifier: Apache-2.0
"""Verify a provisioned COCO 2017 root actually holds every annotated image.

This is the packaged dataset preflight the shipped downloader exposes via
``lucid-download --verify`` / ``--verify-only``. For each requested split it
parses the annotation JSON and confirms that every ``file_name`` the annotations
enumerate is present on disk (a single :func:`os.scandir` per split directory and
a set difference, so it stays fast on the 118,287-image train split). It reports
per-split totals — expected, present, missing count, and a short sample of the
missing names — plus the annotation file's own existence, and drives a non-zero
exit when anything is missing.

Relationship to :mod:`scripts.check_data`:
    :mod:`scripts.check_data` is the repo's ``make check-data`` gate and validates
    the layout by **counts** — fixed per-split image totals (118,287 / 5,000) and
    annotation-vs-disk count *parity*. This module performs the complementary
    **per-file existence** check instead: it names exactly which annotated images
    are missing rather than only noticing that a count is off. The two checks are
    deliberately kept separate — their logic differs materially (counts vs.
    per-file presence), and unlike ``scripts/check_data.py`` (which is not shipped
    in the wheel) this module ships inside ``lucid_yolo`` so pip-installed users
    get a preflight without the developer tooling.

Examples:
    Verify an existing root before training::

        from pathlib import Path
        from lucid_yolo.data.verify import verify_coco_root

        result = verify_coco_root(Path("/data/coco"), ["val"])
        assert result.ok
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SplitVerification",
    "VerifyResult",
    "format_report",
    "verify_coco_root",
    "verify_split",
]

#: Default number of missing file names surfaced in a split's report sample.
_MISSING_SAMPLE = 10

#: Short split name -> (images directory, annotation file) under a data root.
_SPLIT_LAYOUT: dict[str, tuple[str, str]] = {
    "train": ("train2017", "instances_train2017.json"),
    "val": ("val2017", "instances_val2017.json"),
}


@dataclass(frozen=True)
class SplitVerification:
    """Per-file verification outcome for one dataset split.

    Attributes:
        name: Split directory name (e.g. ``"val2017"``).
        annotation_present: Whether the split's annotation JSON exists and parsed.
        expected: Number of images the annotations enumerate (``0`` when the
            annotation file is missing or failed to parse).
        present: How many of those enumerated images are actually on disk.
        missing: A capped, sorted sample of the missing file names.
        missing_count: Total number of enumerated images missing from disk.
        problems: Human-readable problem descriptions; empty when the split is OK.
    """

    name: str
    annotation_present: bool
    expected: int
    present: int
    missing: list[str]
    missing_count: int
    problems: list[str]

    @property
    def ok(self) -> bool:
        """Return whether the split passed every check."""
        return not self.problems


@dataclass(frozen=True)
class VerifyResult:
    """Aggregate per-file verification outcome across all requested splits.

    Attributes:
        splits: The per-split :class:`SplitVerification` results.
    """

    splits: list[SplitVerification]

    @property
    def ok(self) -> bool:
        """Return whether every verified split passed."""
        return all(split.ok for split in self.splits)


def _load_image_names(annotation_file: Path) -> tuple[list[str], str | None]:
    """Return the ``file_name`` list declared in ``annotation_file`` and any error.

    Args:
        annotation_file: Path to the split's ``instances_*.json``.

    Returns:
        A ``(names, error)`` pair: ``names`` is the declared image file names (an
        empty list on failure) and ``error`` is a message when the file is missing
        or cannot be parsed, else ``None``.
    """
    if not annotation_file.is_file():
        return [], f"annotation file missing: {annotation_file}"
    try:
        with annotation_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return [], f"annotation file failed to parse: {annotation_file} ({error})"
    names = [str(entry["file_name"]) for entry in payload.get("images", []) if "file_name" in entry]
    return names, None


def _present_names(images_dir: Path) -> set[str]:
    """Return the set of file names directly under ``images_dir``.

    Args:
        images_dir: The split's image directory.

    Returns:
        The names of regular files in the directory, or an empty set when the
        directory does not exist.
    """
    if not images_dir.is_dir():
        return set()
    with os.scandir(images_dir) as entries:
        return {entry.name for entry in entries if entry.is_file()}


def verify_split(
    name: str,
    images_dir: Path,
    annotation_file: Path,
    *,
    sample: int = _MISSING_SAMPLE,
) -> SplitVerification:
    """Verify every annotated image of one split is present on disk.

    Args:
        name: Split directory name used in messages (e.g. ``"val2017"``).
        images_dir: The split's image directory.
        annotation_file: The split's annotation JSON enumerating the images.
        sample: Maximum number of missing file names to keep in the report.

    Returns:
        A :class:`SplitVerification` capturing the counts, a missing sample, and
        any problems found.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> verify_split("val2017", Path("/nope"), Path("/nope.json")).ok
        False

        ```
    """
    problems: list[str] = []
    names, ann_error = _load_image_names(annotation_file)
    if ann_error is not None:
        problems.append(ann_error)
    if not images_dir.is_dir():
        problems.append(f"images directory missing: {images_dir}")
    on_disk = _present_names(images_dir)
    missing = sorted(n for n in names if n not in on_disk)
    if missing:
        problems.append(f"{name}: {len(missing)} of {len(names)} annotated images missing from {images_dir}")
    return SplitVerification(
        name=name,
        annotation_present=ann_error is None,
        expected=len(names),
        present=len(names) - len(missing),
        missing=missing[:sample],
        missing_count=len(missing),
        problems=problems,
    )


def verify_coco_root(
    data_root: Path,
    splits: Sequence[str] = ("val",),
    *,
    sample: int = _MISSING_SAMPLE,
) -> VerifyResult:
    """Verify each requested split of a COCO 2017 root is complete on disk.

    Args:
        data_root: Directory holding ``train2017`` / ``val2017`` and
            ``annotations``.
        splits: Short split names to verify (``"train"`` / ``"val"``); duplicates
            are collapsed and order is preserved.
        sample: Maximum number of missing file names kept per split report.

    Returns:
        A :class:`VerifyResult` aggregating the requested splits' outcomes.

    Raises:
        ValueError: If a split name is not a known COCO 2017 split.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> verify_coco_root(Path("/nonexistent"), ["val"]).ok
        False

        ```
    """
    annotations = data_root / "annotations"
    results: list[SplitVerification] = []
    for split in dict.fromkeys(splits):
        try:
            images_dir_name, annotation_name = _SPLIT_LAYOUT[split]
        except KeyError:
            known = ", ".join(sorted(_SPLIT_LAYOUT))
            raise ValueError(f"unknown split {split!r}; expected one of {known}") from None
        results.append(
            verify_split(
                images_dir_name,
                data_root / images_dir_name,
                annotations / annotation_name,
                sample=sample,
            )
        )
    return VerifyResult(splits=results)


def format_report(result: VerifyResult, data_root: Path) -> str:
    """Render a compact human-readable verdict for ``result``.

    Args:
        result: The aggregate verification outcome.
        data_root: The root that was verified, named in the header.

    Returns:
        A multi-line report string ending in an overall PASS/FAIL verdict.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> report = format_report(verify_coco_root(Path("/nope"), ["val"]), Path("/nope"))
        >>> report.splitlines()[-1]
        'FAIL: dataset is incomplete'

        ```
    """
    lines = [f"verify-data: {data_root}"]
    for split in result.splits:
        if split.ok:
            lines.append(f"  PASS {split.name} — {split.present}/{split.expected} annotated images present")
            continue
        lines.append(f"  FAIL {split.name} — {split.present}/{split.expected} present, {split.missing_count} missing")
        lines.extend(f"    - {problem}" for problem in split.problems)
        if split.missing:
            more = split.missing_count - len(split.missing)
            suffix = f" (+{more} more)" if more > 0 else ""
            lines.append(f"    missing e.g.: {', '.join(split.missing)}{suffix}")
    lines.append("PASS: every annotated image is present" if result.ok else "FAIL: dataset is incomplete")
    return "\n".join(lines)
