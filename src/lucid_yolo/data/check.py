# SPDX-License-Identifier: Apache-2.0
"""Validate a real COCO 2017 or DOTA-v1.0 dataset root before a ``[DATA]`` run (WP-014, WP-056).

Datasets are never committed and never auto-downloaded (AGENTS.md sec. 3); this is the
``lucid-data check`` gate that confirms a provisioned root actually matches the layout of
blueprint sec. 14.3 before any ``[DATA]`` work runs against it. For each COCO split it
checks that

    * the images directory exists and holds the expected number of image files
      (118,287 for ``train2017``, 5,000 for ``val2017``);
    * the ``annotations/instances_*.json`` file exists and parses;
    * the image count declared inside the annotation JSON matches the number of
      image files on disk.

For DOTA-v1.0 (``--dataset dota``) each split directory must hold ``images/`` and
``labelTxt/``, every image must have a label file of the same stem and vice
versa, every object line must parse (:mod:`lucid_yolo.data.dota`), and the totals
across the checked splits must match the published 2,806 images / 188,282
instances / 15 classes of AGENTS.md sec. 3.

    Those published totals describe the dataset **as published**. Whether a
    partially provisioned root — one split, or the annotated splits only — can
    meet them is settled at ``[DATA]`` time against real data, not here; the
    counts are parameters so a partial provisioning states its own expectation.
    Instances are counted from the label files directly, *every* object line
    including the ones flagged ``difficult``, because the published total does
    not break those out (A39 governs what a *loader* does with the flag, which is
    a different question from what is on disk).

The check core is importable (:func:`check_coco_root` and :func:`check_dota_root`
return a :class:`DataCheck` whose ``ok`` flag drives the exit status); :func:`run` is a
thin CLI over them. The expected counts are parameters (defaulting to the real dataset
totals) so the logic is unit-testable against a tiny fake layout.

Relationship to :mod:`lucid_yolo.data.verify`:
    This module validates the layout by **counts** — fixed per-split totals plus
    annotation-vs-disk count parity — and answers "is this the dataset as published".
    :mod:`lucid_yolo.data.verify` performs the complementary **per-file existence**
    check and answers "which annotated images are missing", which is what a download
    wants to know. The two stay separate because their logic differs materially, not
    because one of them was developer-only: both ship, as ``lucid-data check`` and
    ``lucid-data download --verify`` (WP-096).

    Neither runs inside ``fit``. A counts gate asserts the *published* totals, so
    coupling it to training would refuse a legitimate subset or smoke run at startup.

Examples:
    Validate a provisioned root (exit 1 on any mismatch)::

        lucid-data check --data-root /data/coco
        lucid-data check --data-root /data/dota --dataset dota
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from lucid_yolo.data.dota import DOTA_CLASSES, parse_dota_label_file

#: Expected image counts for the two COCO 2017 splits (blueprint sec. 14.3).
COCO_TRAIN_COUNT = 118287
COCO_VAL_COUNT = 5000
#: Published DOTA-v1.0 totals (AGENTS.md sec. 3; R18).
DOTA_IMAGE_COUNT = 2806
DOTA_INSTANCE_COUNT = 188282
DOTA_CLASS_COUNT = len(DOTA_CLASSES)
#: DOTA split directories checked by default; each holds ``images/`` and ``labelTxt/``.
DOTA_SPLITS = ("train", "val")
#: Image file suffixes counted on disk (COCO 2017 ships ``.jpg``).
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
#: Label file suffix of a DOTA ``labelTxt`` directory.
_LABEL_SUFFIXES = (".txt",)
#: Unmatched names listed in a pairing problem before the message is truncated.
_MAX_LISTED_NAMES = 3


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

    @property
    def summary(self) -> str:
        """Return the one-line count summary printed for a passing split."""
        return f"{self.found_images} images"


@dataclass(frozen=True)
class DotaSplitCheck:
    """Validation outcome for one DOTA-v1.0 split directory.

    Attributes:
        name: Split name (e.g. ``"train"``).
        found_images: Image files found under ``images/`` (``-1`` if it is
            missing).
        found_labels: Label files found under ``labelTxt/`` (``-1`` if it is
            missing).
        instances: Object lines parsed across the split's label files (``-1``
            when the label directory is missing); difficult instances included.
        classes: Class ids seen in this split.
        problems: Human-readable problem descriptions; empty when the split is OK.
    """

    name: str
    found_images: int
    found_labels: int
    instances: int
    classes: frozenset[int]
    problems: list[str]

    @property
    def ok(self) -> bool:
        """Return whether the split passed every check."""
        return not self.problems

    @property
    def summary(self) -> str:
        """Return the one-line count summary printed for a passing split."""
        return f"{self.found_images} images, {self.instances} instances, {len(self.classes)} classes"


@dataclass(frozen=True)
class DataCheck:
    """Aggregate validation outcome across all splits.

    Attributes:
        splits: The per-split results.
        problems: Root-level problems that belong to no single split (the
            cross-split totals); empty when there are none.
    """

    splits: list[SplitCheck] | list[DotaSplitCheck]
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether every split passed and no root-level problem was found."""
        return not self.problems and all(split.ok for split in self.splits)


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


def _stems(directory: Path, suffixes: tuple[str, ...]) -> set[str] | None:
    """Return the file stems directly under ``directory``, or ``None`` if it is missing.

    Args:
        directory: Candidate directory.
        suffixes: Lower-cased suffixes a file must carry to be counted.

    Returns:
        The set of stems of matching files, or ``None`` when the directory does
        not exist.

    Examples:
        >>> from pathlib import Path
        >>> _stems(Path("/nonexistent"), (".txt",)) is None
        True
    """
    if not directory.is_dir():
        return None
    return {path.stem for path in directory.iterdir() if path.suffix.lower() in suffixes}


def _pairing_problems(name: str, image_stems: set[str], label_stems: set[str]) -> list[str]:
    """Report images without a label file and label files without an image.

    Args:
        name: Split name for messages.
        image_stems: Stems found under ``images/``.
        label_stems: Stems found under ``labelTxt/``.

    Returns:
        One problem string per non-empty direction of the mismatch; empty when
        the two sets are equal.

    Examples:
        >>> _pairing_problems("train", {"P0001"}, set())
        ["train: 1 image(s) without a label file: ['P0001']"]
    """
    problems = []
    for missing, message in (
        (image_stems - label_stems, "image(s) without a label file"),
        (label_stems - image_stems, "label file(s) without an image"),
    ):
        if missing:
            listed = sorted(missing)[:_MAX_LISTED_NAMES]
            problems.append(f"{name}: {len(missing)} {message}: {listed}")
    return problems


def _scan_labels(labels_dir: Path) -> tuple[int, frozenset[int], list[str]]:
    """Parse every label file in ``labels_dir``, counting instances and classes.

    Args:
        labels_dir: The split's ``labelTxt`` directory.

    Returns:
        A ``(instances, classes, problems)`` triple. ``instances`` counts every
        object line, difficult ones included; ``problems`` holds one entry per
        label file that failed to parse, so a malformed file is reported rather
        than raised.

    Examples:
        >>> from pathlib import Path
        >>> _scan_labels(Path("/nonexistent"))
        (0, frozenset(), [])
    """
    instances = 0
    classes: set[int] = set()
    problems: list[str] = []
    for path in sorted(labels_dir.glob("*.txt")):
        try:
            objects = parse_dota_label_file(path)
        except (OSError, ValueError) as error:
            problems.append(f"label file failed to parse: {error}")
            continue
        instances += len(objects)
        classes.update(obj.label for obj in objects)
    return instances, frozenset(classes), problems


def check_dota_split(name: str, split_dir: Path) -> DotaSplitCheck:
    """Validate one DOTA-v1.0 split directory's layout, pairing and label files.

    Args:
        name: Split name for messages.
        split_dir: The split directory, expected to hold ``images/`` and
            ``labelTxt/``.

    Returns:
        A :class:`DotaSplitCheck` capturing the counts and any problems found.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> check_dota_split("train", Path("/nonexistent")).ok
        False

        ```
    """
    images_dir, labels_dir = split_dir / "images", split_dir / "labelTxt"
    image_stems = _stems(images_dir, _IMAGE_SUFFIXES)
    label_stems = _stems(labels_dir, _LABEL_SUFFIXES)
    problems: list[str] = []
    if image_stems is None:
        problems.append(f"images directory missing: {images_dir}")
    if label_stems is None:
        problems.append(f"labelTxt directory missing: {labels_dir}")
    if image_stems is not None and label_stems is not None:
        problems.extend(_pairing_problems(name, image_stems, label_stems))
    instances, classes, parse_problems = _scan_labels(labels_dir) if label_stems is not None else (-1, frozenset(), [])
    problems.extend(parse_problems)
    return DotaSplitCheck(
        name=name,
        found_images=-1 if image_stems is None else len(image_stems),
        found_labels=-1 if label_stems is None else len(label_stems),
        instances=instances,
        classes=classes,
        problems=problems,
    )


def check_dota_root(
    data_root: Path,
    splits: tuple[str, ...] = DOTA_SPLITS,
    expected_images: int = DOTA_IMAGE_COUNT,
    expected_instances: int = DOTA_INSTANCE_COUNT,
    expected_classes: int = DOTA_CLASS_COUNT,
) -> DataCheck:
    """Validate a DOTA-v1.0 root: per-split layout plus the published totals.

    Each split is checked by :func:`check_dota_split`; the totals across the
    checked splits are then compared with the published counts of AGENTS.md
    sec. 3 (2,806 images / 188,282 instances / 15 classes). Those describe the
    dataset **as published** — a partially provisioned root states its own
    expectation through the parameters rather than by weakening the default.

    Args:
        data_root: Directory holding the split directories.
        splits: Split directory names to check (defaults to the annotated
            ``train``/``val`` pair).
        expected_images: Expected image count summed over ``splits``.
        expected_instances: Expected object-line count summed over ``splits``,
            difficult instances included.
        expected_classes: Expected number of distinct classes seen.

    Returns:
        A :class:`DataCheck` aggregating the splits and the totals.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> check_dota_root(Path("/nonexistent")).ok
        False

        ```
    """
    checks = [check_dota_split(name, data_root / name) for name in splits]
    images = sum(max(check.found_images, 0) for check in checks)
    instances = sum(max(check.instances, 0) for check in checks)
    classes = frozenset[int]().union(*(check.classes for check in checks))
    problems = []
    if images != expected_images:
        problems.append(f"expected {expected_images} images across {list(splits)}, found {images}")
    if instances != expected_instances:
        problems.append(f"expected {expected_instances} instances across {list(splits)}, found {instances}")
    if len(classes) != expected_classes:
        problems.append(f"expected {expected_classes} classes across {list(splits)}, found {len(classes)}")
    return DataCheck(splits=checks, problems=problems)


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
            lines.append(f"  PASS {split.name} — {split.summary}")
        else:
            lines.extend(f"  FAIL {problem}" for problem in split.problems)
    lines.extend(f"  FAIL {problem}" for problem in result.problems)
    lines.append("PASS: dataset layout valid" if result.ok else "FAIL: dataset layout invalid")
    return "\n".join(lines)


#: Dataset layouts :func:`check_dataset` knows how to validate.
DATASETS = ("coco", "dota")


def check_dataset(data_root: Path, dataset: str = "coco") -> int:
    """Validate a provisioned dataset root against its published layout.

    Args:
        data_root: Dataset root directory.
        dataset: Which layout to validate, ``coco`` or ``dota``.

    Returns:
        ``0`` when the layout is valid, ``1`` otherwise.

    Raises:
        ValueError: If ``dataset`` is not a known layout.

    Examples:
        >>> code = check_dataset(Path("/nonexistent"))  # doctest: +ELLIPSIS
        check-data: /nonexistent
        ...
        FAIL: dataset layout invalid
        >>> code
        1
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; known layouts are {list(DATASETS)}")
    result = check_coco_root(data_root) if dataset == "coco" else check_dota_root(data_root)
    print(format_report(result, data_root))
    return 0 if result.ok else 1
