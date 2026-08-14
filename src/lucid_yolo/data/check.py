# SPDX-License-Identifier: Apache-2.0
"""Validate a provisioned COCO, DOTA-v1.0 or YOLO dataset root before a run (WP-014, WP-056, WP-099c).

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
versa, and every object line must parse (:mod:`lucid_yolo.data.dota`). The totals
across the checked splits are **reported**, and required only when the caller
states them (``--expected_images`` and its two siblings).

    The published 2,806 images / 188,282 instances / 15 classes of AGENTS.md
    sec. 3 describe the dataset **whole**, and are not the defaults: R18 sec. 4
    splits DOTA into half training, one sixth validation and one third testing,
    and releases ground truth for the first two only, so the annotated root this
    check is pointed at holds about two thirds of that and can never sum to it.
    Requiring them by default failed every correct download (WP-097). Instances
    are counted from the label files directly, *every* object line including the
    ones flagged ``difficult``, because the published total does not break those
    out (A39 governs what a *loader* does with the flag, which is a different
    question from what is on disk).

For a YOLO root (WP-099c) the root's own ``data.yaml`` is read first — it carries the
class list, so a labels tree without one is not a dataset this project can read — and
each split's images and labels directories are resolved by
:func:`~lucid_yolo.data.yolo.resolve_split_dirs`, the function
:meth:`~lucid_yolo.data.yolo.YoloDetectionDataset.from_root` resolves them with. Images
and label files must then pair by stem in **both** directions, and every row must parse
(:func:`~lucid_yolo.data.yolo.scan_yolo_label_file`), each rejection naming the file and
the 1-based line exactly as the reader's own does — a row's class index being inside the
declared ``names`` is part of that grammar, which is how the class count and the rows are
held to agree. Totals are reported and required only when stated, as for DOTA.

Which layout, and who decides:
    ``dataset`` left unstated means **infer**, through
    :func:`~lucid_yolo.data.layout.detect_layout` — the WP-099b probe (A63). This command
    is the pre-flight for ``lucid-yolo fit``, and a ``fit`` given no ``--data.layout``
    dispatches on that same probe: a check that defaulted to COCO would validate a question
    the run never asks, and the two would disagree about what the root is precisely when it
    matters. There is therefore one statement of what a COCO root and a YOLO root are, and
    this module holds neither of them.

    DOTA is stated, never inferred, and sits outside the probe by design rather than by
    omission: no training run reads a DOTA root at all. Its ``labelTxt`` tree is tiled into
    a COCO container first (``lucid-data build-tiles``, WP-094), so what the OBB tier trains
    on is a COCO layout. A root satisfying no convention says so *and* names
    ``--dataset dota``, because that operator is the only one inference cannot serve.

    The raise-versus-report line is the datamodule's (WP-099b): an argument the caller got
    wrong raises — an unknown layout name, a flag belonging to another layout — while every
    verdict about the disk, the probe's included, becomes a FAIL line in the report and exit
    1. This is the first command an operator runs against a fresh provisioning, and WP-097
    is the record of what a spurious failure costs there; a traceback would cost the same.

The check core is importable (:func:`check_coco_root`, :func:`check_dota_root` and
:func:`check_yolo_root` return a :class:`DataCheck` whose ``ok`` flag drives the exit
status); :func:`check_dataset` is the dispatch over them, and is what ``lucid-data check``
calls. Every expected count is a parameter, so the logic is unit-testable against a
tiny fake layout: COCO's per-split counts default to the published ones because a
COCO split either is that split or is not, while DOTA's and YOLO's totals default to
unstated — neither publishes a total this project can hold every root to.

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
    Validate a provisioned root (exit 1 on any mismatch); the layout is inferred unless
    the root is a DOTA one, which no probe covers::

        lucid-data check --data_root /data/coco
        lucid-data check --data_root /data/roboflow_export
        lucid-data check --data_root /data/dota --dataset dota

    Assert that a DOTA root holds both annotated splits whole::

        lucid-data check --data_root /data/dota --dataset dota \
            --expected_images 1869 --expected_instances 127843

    Check a root that ships only a training split — a third-party export whose validation
    split was never cut is a correct export, and failing it on a missing ``val`` would be
    the pre-flight refusing a dataset that trains::

        lucid-data check --data_root /data/roboflow_export --splits '[train]'
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from lucid_yolo.data.dota import DOTA_CLASSES, parse_dota_label_file
from lucid_yolo.data.layout import DATA_YAML_NAME, DatasetLayout, detect_layout
from lucid_yolo.data.yolo import IMAGE_SUFFIXES, YoloDataConfig, resolve_split_dirs, scan_yolo_label_file

if TYPE_CHECKING:
    from collections.abc import Callable

    #: One layout's label-directory scan: parse every label file under a directory and return
    #: ``(instances, classes, problems)``, one problem per file that failed to parse. The seam
    #: between :func:`_check_paired_split` and the two label grammars it is shared by.
    _LabelScan = Callable[[Path], tuple[int, frozenset[int], list[str]]]

#: Expected image counts for the two COCO 2017 splits (blueprint sec. 14.3).
COCO_TRAIN_COUNT = 118287
COCO_VAL_COUNT = 5000
#: Published DOTA-v1.0 totals (AGENTS.md sec. 3; R18). These cover all three splits,
#: testing included, and R18 releases no testing ground truth — so they are what a
#: caller asserting the *whole* dataset passes, never a default (see
#: :func:`check_dota_root`).
DOTA_IMAGE_COUNT = 2806
DOTA_INSTANCE_COUNT = 188282
#: Totals of the two annotated splits, **measured** on a complete DOTA-v1.0
#: train-plus-val provisioning from the official distribution (2026-08-13): 1,411 plus
#: 458 images, 98,990 plus 28,853 object lines with difficult instances included. Not a
#: published figure — R18 states only the whole-dataset totals above, and 2,806 minus
#: 1,869 is exactly the 937 images of the withheld testing third. Offered so a caller
#: has something to pass to ``--expected_images``, which is how a truncated download or
#: a half-unpacked archive gets caught; still not a default, because a root provisioned
#: with one split is a legitimate thing to check.
DOTA_ANNOTATED_IMAGE_COUNT = 1869
DOTA_ANNOTATED_INSTANCE_COUNT = 127843
DOTA_CLASS_COUNT = len(DOTA_CLASSES)
#: DOTA split directories checked by default; each holds ``images/`` and ``labelTxt/``.
DOTA_SPLITS = ("train", "val")
#: YOLO splits checked by default: the two a training run builds, which is what a pre-run
#: check is for. A ``data.yaml`` may also name a ``test:`` split, and requiring it would fail
#: an export whose test images were never downloaded for a split no run of this project opens.
YOLO_SPLITS = ("train", "val")
#: Image file suffixes counted on disk (COCO 2017 ships ``.jpg``).
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
#: Label file suffix of a DOTA ``labelTxt`` directory.
_LABEL_SUFFIXES = (".txt",)
#: Unmatched names listed in a pairing problem before the message is truncated.
_MAX_LISTED_NAMES = 3
#: Class names printed in a YOLO root's declared-classes note before it is truncated to a
#: count. Enough for a small export to be read back at a glance, short of COCO's 80.
_MAX_LISTED_CLASSES = 8


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
class PairedSplitCheck:
    """Validation outcome for one split of a layout whose labels are files beside its images.

    Both such layouts — DOTA-v1.0's ``labelTxt`` tree and the YOLO ``labels`` one — are
    checked the same way and report the same counts, so they share this container rather
    than each naming its own. It is the *shape* of the annotation that decides: one text
    file per image, paired by stem, versus COCO's single manifest (:class:`SplitCheck`).

    Attributes:
        name: Split name (e.g. ``"train"``).
        found_images: Image files found under the split's images directory (``-1``
            if it is missing).
        found_labels: Label files found under the split's labels directory (``-1``
            if it is missing).
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
        notes: Root-level observations that are **not** problems — the summed
            counts a caller may want to read off, printed but never failing the
            check. A count nobody stated an expectation for is information, and
            information reported as a problem trains an operator to skim past
            the report, which is how the unreachable totals clause survived.
    """

    splits: list[SplitCheck] | list[PairedSplitCheck]
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

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


def _check_paired_split(
    name: str,
    images_dir: Path,
    labels_dir: Path,
    image_suffixes: tuple[str, ...],
    scan: _LabelScan,
) -> PairedSplitCheck:
    """Validate one split of a layout that pairs each image with a label file of the same stem.

    The DOTA and YOLO layouts differ in where their two directories live and in the grammar
    of a label line; everything between — the two directories existing, the pairing holding
    in both directions, a malformed file being reported rather than raised — is one check,
    stated here once so the two cannot drift into reporting the same fault differently.

    Args:
        name: Split name for messages.
        images_dir: The split's images directory.
        labels_dir: The split's labels directory; its own directory name is what the
            missing-directory message says, so each layout's message names its own spelling.
        image_suffixes: Suffixes counted as images — the reading reader's own set.
        scan: The label-directory scan, returning ``(instances, classes, problems)``.

    Returns:
        A :class:`PairedSplitCheck` capturing the counts and any problems found.

    Examples:
        >>> _check_paired_split("train", Path("/no/images"), Path("/no/labels"), (".png",), _scan_labels).ok
        False
    """
    image_stems = _stems(images_dir, image_suffixes)
    label_stems = _stems(labels_dir, _LABEL_SUFFIXES)
    problems: list[str] = []
    if image_stems is None:
        problems.append(f"images directory missing: {images_dir}")
    if label_stems is None:
        problems.append(f"{labels_dir.name} directory missing: {labels_dir}")
    if image_stems is not None and label_stems is not None:
        problems.extend(_pairing_problems(name, image_stems, label_stems))
    instances, classes, parse_problems = scan(labels_dir) if label_stems is not None else (-1, frozenset(), [])
    problems.extend(parse_problems)
    return PairedSplitCheck(
        name=name,
        found_images=-1 if image_stems is None else len(image_stems),
        found_labels=-1 if label_stems is None else len(label_stems),
        instances=instances,
        classes=classes,
        problems=problems,
    )


def check_dota_split(name: str, split_dir: Path) -> PairedSplitCheck:
    """Validate one DOTA-v1.0 split directory's layout, pairing and label files.

    Args:
        name: Split name for messages.
        split_dir: The split directory, expected to hold ``images/`` and
            ``labelTxt/``.

    Returns:
        A :class:`PairedSplitCheck` capturing the counts and any problems found.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> check_dota_split("train", Path("/nonexistent")).ok
        False

        ```
    """
    return _check_paired_split(
        name,
        split_dir / "images",
        split_dir / "labelTxt",
        _IMAGE_SUFFIXES,
        _scan_labels,
    )


def check_dota_root(
    data_root: Path,
    splits: tuple[str, ...] = DOTA_SPLITS,
    expected_images: int | None = None,
    expected_instances: int | None = None,
    expected_classes: int | None = None,
) -> DataCheck:
    """Validate a DOTA-v1.0 root: per-split layout, and any totals the caller states.

    Each split is checked by :func:`check_dota_split`. The totals across the
    checked splits are then **reported**, and compared only against an
    expectation the caller actually supplied — an ``expected_*`` left at
    ``None`` produces a note, never a problem.

    The published counts (:data:`DOTA_IMAGE_COUNT`, :data:`DOTA_INSTANCE_COUNT`,
    :data:`DOTA_CLASS_COUNT`) are not defaults here, because they are not
    reachable by the roots this function is pointed at. They describe DOTA-v1.0
    whole, and R18 sec. 4 splits it into half training, one sixth validation and
    one third testing, releasing ground truth for the first two only: a
    correctly provisioned ``train``/``val`` root therefore holds about two
    thirds of 2,806 images and can never sum to it. Asserting them by default
    made the one command an operator runs first fail on a correct download,
    twice over, with the layout and pairing verdicts that *are* meaningful
    printed just above the noise (WP-097). Pass them explicitly to assert them.

    Args:
        data_root: Directory holding the split directories.
        splits: Split directory names to check (defaults to the annotated
            ``train``/``val`` pair; the testing split has no public labels).
        expected_images: Image count to require summed over ``splits``, or
            ``None`` to report the count without requiring one.
        expected_instances: Object-line count to require summed over ``splits``,
            difficult instances included, or ``None`` to only report it.
        expected_classes: Number of distinct classes to require, or ``None`` to
            only report it.

    Returns:
        A :class:`DataCheck` aggregating the splits, the stated totals, and a
        note carrying the observed ones.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> check_dota_root(Path("/nonexistent")).ok
        False

        ```
    """
    _require_splits(splits)
    checks = [check_dota_split(name, data_root / name) for name in splits]
    problems, notes = _totals(checks, splits, expected_images, expected_instances, expected_classes)
    return DataCheck(splits=checks, problems=problems, notes=notes)


def _require_splits(splits: tuple[str, ...]) -> None:
    """Raise :class:`ValueError` unless ``splits`` names at least one split.

    Every other bad argument in this module fails loudly; an empty ``splits`` fails
    *quietly*, which is worse here than anywhere else in it. No split checked means no
    problem found means ``PASS: dataset layout valid`` printed over a root nothing
    looked at — a clean green from the one command whose whole purpose is to be believed.
    Both root checkers guard rather than only :func:`check_dataset`, because they are
    documented as the importable core and a library caller reaches them without passing
    the entry point at all.

    Args:
        splits: The split names about to be checked.

    Raises:
        ValueError: If ``splits`` is empty.

    Examples:
        >>> _require_splits(("train",)) is None
        True
    """
    if not splits:
        raise ValueError(
            "splits names no split, which would check nothing and report a pass; "
            "omit it to check the layout's own splits"
        )


def _totals(
    checks: list[PairedSplitCheck],
    splits: tuple[str, ...],
    expected_images: int | None,
    expected_instances: int | None,
    expected_classes: int | None,
) -> tuple[list[str], list[str]]:
    """Sum a paired layout's per-split counts, and compare them with the totals stated.

    Shared by the DOTA and the YOLO root check, which reach the same question from
    different label grammars: neither layout publishes a per-root total this project can
    hold every provisioning to, so an ``expected_*`` left at ``None`` is reported and never
    required (WP-097).

    Args:
        checks: The per-split outcomes, in the order the splits were checked.
        splits: The split names, named in both messages.
        expected_images: Image count to require summed over ``splits``, or ``None``.
        expected_instances: Object-line count to require, or ``None``.
        expected_classes: Number of distinct classes to require, or ``None``.

    Returns:
        The problems raised by a stated total that was not met, and the one note carrying
        the observed totals.

    Examples:
        >>> _totals([], ("train",), 1, None, None)
        (["expected 1 images across ['train'], found 0"], ["totals across ['train']: 0 images, 0 instances, 0 classes"])
    """
    images = sum(max(check.found_images, 0) for check in checks)
    instances = sum(max(check.instances, 0) for check in checks)
    classes = frozenset[int]().union(*(check.classes for check in checks))
    stated = (
        (expected_images, images, "images"),
        (expected_instances, instances, "instances"),
        (expected_classes, len(classes), "classes"),
    )
    problems = [
        f"expected {expected} {noun} across {list(splits)}, found {found}"
        for expected, found, noun in stated
        if expected is not None and expected != found
    ]
    note = f"totals across {list(splits)}: {images} images, {instances} instances, {len(classes)} classes"
    return problems, [note]


def _scan_yolo_labels(labels_dir: Path, num_classes: int, oriented: bool) -> tuple[int, frozenset[int], list[str]]:
    """Parse every label file of a YOLO split, counting rows and the classes they name.

    Args:
        labels_dir: The split's ``labels`` directory.
        num_classes: Class count the root's ``data.yaml`` declares; a row naming anything
            outside it is a row written against a different class list, and fails here.
        oriented: Read the nine-field oriented rows rather than five-field boxes.

    Returns:
        An ``(instances, classes, problems)`` triple, with one problem per file that failed
        to parse — the reader's own message, so the file and the 1-based line are named.

    Examples:
        >>> _scan_yolo_labels(Path("/nonexistent"), 1, False)
        (0, frozenset(), [])
    """
    instances = 0
    classes: set[int] = set()
    problems: list[str] = []
    for path in sorted(labels_dir.glob("*.txt")):
        try:
            count, seen = scan_yolo_label_file(path, num_classes=num_classes, oriented=oriented)
        except (OSError, ValueError) as error:
            problems.append(f"label file failed to parse: {error}")
            continue
        instances += count
        classes.update(seen)
    return instances, frozenset(classes), problems


def check_yolo_split(name: str, data_root: Path, config: YoloDataConfig, oriented: bool = False) -> PairedSplitCheck:
    """Validate one split of a YOLO root: its two directories, their pairing and every row.

    The split's directories come from :func:`~lucid_yolo.data.yolo.resolve_split_dirs`, so a
    split the ``data.yaml`` points somewhere its own way is checked where the reader will
    look for it rather than where a convention says it should be (A58). A split entry that
    resolves nowhere is that split's problem, not an exception: the report is the product
    here, and one unbuilt split should not hide a second one's verdict.

    Args:
        name: Split name as keyed in the ``data.yaml``, e.g. ``"train"``.
        data_root: The dataset root holding the ``data.yaml``.
        config: The root's parsed ``data.yaml``, whose ``names`` bound every row's class.
        oriented: Read nine-field oriented rows instead of five-field boxes. Declared by
            the caller, never sniffed (WP-099): a nine-field file read as detection is a
            file that fails on every row, which is the report a mis-stated flag should give.

    Returns:
        A :class:`PairedSplitCheck` capturing the counts and any problems found.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> config = YoloDataConfig(root=Path("/nonexistent"), names=("car",), splits={})
        >>> check_yolo_split("train", Path("/nonexistent"), config).ok
        False

        ```
    """
    try:
        images_dir, labels_dir = resolve_split_dirs(data_root, name, config)
    except (FileNotFoundError, ValueError) as error:
        return PairedSplitCheck(
            name=name,
            found_images=-1,
            found_labels=-1,
            instances=-1,
            classes=frozenset(),
            problems=[str(error)],
        )
    return _check_paired_split(
        name,
        images_dir,
        labels_dir,
        IMAGE_SUFFIXES,
        lambda directory: _scan_yolo_labels(directory, len(config.names), oriented),
    )


def check_yolo_root(
    data_root: Path,
    splits: tuple[str, ...] = YOLO_SPLITS,
    oriented: bool = False,
    expected_images: int | None = None,
    expected_instances: int | None = None,
    expected_classes: int | None = None,
) -> DataCheck:
    """Validate a YOLO root: its ``data.yaml``, each split's pairing, and every label row.

    The ``data.yaml`` is read first and is the whole of the class space, so a root without
    one — or with one that contradicts itself, an ``nc`` disagreeing with its ``names``
    (A57) — has no split worth checking and is reported as that one problem. Otherwise each
    split in ``splits`` is checked by :func:`check_yolo_split`, and the totals are reported
    and required only where stated, as for DOTA.

    What this deliberately does **not** check: that an image decodes, that a box is
    plausible, or that a class the ``data.yaml`` declares is used by any row. The first two
    are the reader's job at the moment it reads (and would cost an epoch's decoding here);
    the third is not a fault at all — an export whose validation split happens to use six of
    its eight classes is a correct export, and failing it would be WP-097's mistake again.

    Args:
        data_root: Directory holding ``data.yaml`` and the split trees.
        splits: Split names to check; the two a training run builds by default, since this
            validates what ``fit`` will read. A ``test:`` entry is checked only when asked
            for by name, because no run of this project opens it.
        oriented: Read the nine-field oriented rows on every split (see
            :func:`check_yolo_split`).
        expected_images: Image count to require summed over ``splits``, or ``None`` to
            report the count without requiring one.
        expected_instances: Object-row count to require, or ``None`` to only report it.
        expected_classes: Number of distinct classes the rows must *use*, or ``None`` to
            only report it. The count the file declares is a note either way.

    Returns:
        A :class:`DataCheck` aggregating the splits, the stated totals, and notes carrying
        the observed totals and the declared class list.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> check_yolo_root(Path("/nonexistent")).ok
        False

        ```
    """
    _require_splits(splits)
    data_yaml = data_root / DATA_YAML_NAME
    try:
        config = YoloDataConfig.read(data_yaml)
    except (OSError, ValueError) as error:
        return DataCheck(splits=[], problems=[f"{DATA_YAML_NAME} unusable: {error}"])
    checks = [check_yolo_split(name, data_root, config, oriented=oriented) for name in splits]
    problems, notes = _totals(checks, splits, expected_images, expected_instances, expected_classes)
    return DataCheck(splits=checks, problems=problems, notes=[_declared_classes_note(config), *notes])


def _declared_classes_note(config: YoloDataConfig) -> str:
    """Render the class list a root's ``data.yaml`` declares, as a report note.

    Args:
        config: The root's parsed ``data.yaml``.

    Returns:
        The declared class count, with the names themselves when there are few enough to
        read at a glance — the check an operator actually performs by eye is "is this the
        label space I meant", and an 80-name COCO-sized list buries the rest of the report.

    Examples:
        >>> _declared_classes_note(YoloDataConfig(root=Path("."), names=("car", "truck"), splits={}))
        'data.yaml declares 2 classes: car, truck'
    """
    names = config.names
    listed = ", ".join(names) if len(names) <= _MAX_LISTED_CLASSES else f"{', '.join(names[:_MAX_LISTED_CLASSES])}, ..."
    return f"{DATA_YAML_NAME} declares {len(names)} classes: {listed}"


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
    lines.extend(f"  NOTE {note}" for note in result.notes)
    lines.extend(f"  FAIL {problem}" for problem in result.problems)
    lines.append("PASS: dataset layout valid" if result.ok else "FAIL: dataset layout invalid")
    return "\n".join(lines)


#: The DOTA-v1.0 layout, which is named and never inferred: no training run reads a DOTA
#: root — its ``labelTxt`` tree is tiled into a COCO container first (WP-094) — so it is
#: outside :func:`~lucid_yolo.data.layout.detect_layout`'s two tables by design.
DOTA_DATASET = "dota"
#: Dataset layouts :func:`check_dataset` knows how to validate: the two a run can be pointed
#: at, taken from the probe's own enum so this module cannot come to disagree with it about
#: what a root is, plus DOTA.
DATASETS = (DatasetLayout.COCO.value, DOTA_DATASET, DatasetLayout.YOLO.value)


def check_dataset(
    data_root: Path,
    dataset: str | None = None,
    oriented: bool = False,
    splits: tuple[str, ...] | None = None,
    expected_images: int | None = None,
    expected_instances: int | None = None,
    expected_classes: int | None = None,
) -> int:
    """Validate a provisioned dataset root against the layout it is written in.

    Args:
        data_root: Dataset root directory.
        dataset: Which layout to validate — ``coco``, ``dota`` or ``yolo``. Left unstated
            (the default) the root is **probed** by
            :func:`~lucid_yolo.data.layout.detect_layout`, which is the same dispatch a
            ``fit`` without ``--data.layout`` makes: this command is that run's pre-flight,
            so inferring differently would validate a question the run never asks. A DOTA
            root is outside the probe and must be named.
        oriented: ``yolo`` only — read the nine-field oriented label rows rather than
            five-field boxes. The variant is a property of the dataset and is declared,
            never sniffed (WP-099).
        splits: ``dota`` and ``yolo`` — the split names to check. Left unstated (the
            default) each layout keeps its own — :data:`DOTA_SPLITS`, :data:`YOLO_SPLITS` —
            which is a property of the layout and not of this call, so the default is never
            restated here. Naming a subset is how a root shipping only ``train`` passes: a
            third-party export whose validation split was never cut is a correct export, and
            failing it on a ``val`` nobody provisioned is WP-097's mistake in a new place.
            COCO refuses the flag rather than defaulting it, because its two splits *are*
            that layout — the names and the published per-split counts checked against them
            are one fact (:func:`check_coco_root`), so a subset of them is not a question
            the COCO branch can be asked.
        expected_images: ``dota`` and ``yolo`` — image count to require across the
            checked splits; omitted, the count is reported and not required.
        expected_instances: ``dota`` and ``yolo`` — object-line count to require,
            difficult instances included.
        expected_classes: ``dota`` and ``yolo`` — number of distinct classes to require.

    Returns:
        ``0`` when the layout is valid, ``1`` otherwise — including when the root satisfies
        no convention, which is a verdict about the disk and so is reported rather than
        raised.

    Raises:
        ValueError: If ``dataset`` is not a known layout, if ``splits`` names nothing at all,
            or if a flag is supplied for a layout it does not apply to. A flag silently
            ignored is worse than a rejected one: the report then reads as though it had been
            enforced — and an empty ``splits`` is the sharpest case of that, since checking
            no split at all yields a clean PASS about nothing.

    Examples:
        >>> code = check_dataset(Path("/nonexistent"))  # doctest: +ELLIPSIS
        check-data: /nonexistent
        ...
        FAIL: dataset layout invalid
        >>> code
        1

        Naming a subset narrows what is checked, and the totals note says which splits it
        summed:

        >>> code = check_dataset(Path("/nonexistent"), dataset="dota", splits=("train",))
        check-data: /nonexistent
          FAIL images directory missing: /nonexistent/train/images
          FAIL labelTxt directory missing: /nonexistent/train/labelTxt
          NOTE totals across ['train']: 0 images, 0 instances, 0 classes
        FAIL: dataset layout invalid
    """
    if dataset is not None and dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; known layouts are {list(DATASETS)}")
    if splits is not None:
        _require_splits(splits)
    if dataset is None:
        dataset, probe_problems = _infer_dataset(data_root)
        if dataset is None:
            print(format_report(DataCheck(splits=[], problems=probe_problems), data_root))
            return 1
    _check_flags_apply(dataset, oriented, splits, (expected_images, expected_instances, expected_classes))
    result = _check_root(
        data_root,
        dataset,
        oriented=oriented,
        splits=splits,
        expected_images=expected_images,
        expected_instances=expected_instances,
        expected_classes=expected_classes,
    )
    print(format_report(result, data_root))
    return 0 if result.ok else 1


def _infer_dataset(data_root: Path) -> tuple[str | None, list[str]]:
    """Ask the layout probe which convention ``data_root`` is written in.

    The probe raises on both of its undecidable states — a root satisfying neither
    convention, and one satisfying both (A63) — and both are verdicts about the disk, so
    they come back as report problems here. ``lucid-data check`` is the first command run
    against a fresh provisioning; a traceback where a report belongs is what WP-097 already
    paid for once.

    Args:
        data_root: Dataset root to probe.

    Returns:
        The layout's name and no problems, or ``None`` and the probe's own message, extended
        with the layout the probe cannot see.

    Examples:
        >>> layout, problems = _infer_dataset(Path("/nonexistent"))
        >>> layout is None, len(problems)
        (True, 1)
    """
    try:
        return detect_layout(data_root).value, []
    except (FileNotFoundError, ValueError) as error:
        return None, [
            f"{error}. Inference covers the two layouts a run reads; a DOTA-v1.0 root "
            f"(images beside labelTxt) is stated with --dataset {DOTA_DATASET}"
        ]


def _check_flags_apply(
    dataset: str,
    oriented: bool,
    splits: tuple[str, ...] | None,
    expectations: tuple[int | None, ...],
) -> None:
    """Reject a flag belonging to a layout other than the one being checked.

    Args:
        dataset: The layout that will be checked, stated or inferred.
        oriented: The oriented-rows flag.
        splits: The split names stated, or ``None`` for the layout's own.
        expectations: The three ``expected_*`` values.

    Raises:
        ValueError: If a flag applies to no layout in play. The counts are meaningless for
            COCO, whose per-split totals are published and are the check's own defaults; the
            oriented reading is a property of the YOLO row grammar alone; and COCO's splits
            are fixed by the same publication the counts come from, so
            :func:`check_coco_root` has no ``splits`` to narrow.

    Examples:
        >>> _check_flags_apply("yolo", True, ("train",), (None, None, None)) is None
        True
    """
    if dataset == DatasetLayout.COCO.value and any(expectation is not None for expectation in expectations):
        raise ValueError(
            f"expected_images/instances/classes apply to --dataset {DOTA_DATASET} or "
            f"--dataset {DatasetLayout.YOLO.value}, not {dataset!r}"
        )
    if oriented and dataset != DatasetLayout.YOLO.value:
        raise ValueError(
            f"oriented applies to --dataset {DatasetLayout.YOLO.value}, whose label rows carry "
            f"either variant, not {dataset!r}"
        )
    if splits is not None and dataset == DatasetLayout.COCO.value:
        raise ValueError(
            f"splits applies to --dataset {DOTA_DATASET} or --dataset {DatasetLayout.YOLO.value}, "
            f"whose split names are the root's own, not {dataset!r}"
        )


def _check_root(
    data_root: Path,
    dataset: str,
    *,
    oriented: bool,
    splits: tuple[str, ...] | None,
    expected_images: int | None,
    expected_instances: int | None,
    expected_classes: int | None,
) -> DataCheck:
    """Run the check belonging to one layout.

    Args:
        data_root: Dataset root directory.
        dataset: The layout to check, already validated and resolved.
        oriented: Whether YOLO rows are read as the oriented variant.
        splits: Split names to check, or ``None`` to keep the layout's own — which each
            branch names for itself here, since which splits a layout has by default is a
            fact about that layout rather than about this dispatch.
        expected_images: Image count to require, or ``None``.
        expected_instances: Object-line count to require, or ``None``.
        expected_classes: Distinct class count to require, or ``None``.

    Returns:
        That layout's :class:`DataCheck`.

    Examples:
        >>> _check_root(
        ...     Path("/nonexistent"),
        ...     "coco",
        ...     oriented=False,
        ...     splits=None,
        ...     expected_images=None,
        ...     expected_instances=None,
        ...     expected_classes=None,
        ... ).ok
        False
    """
    if dataset == DatasetLayout.COCO.value:
        return check_coco_root(data_root)
    if dataset == DOTA_DATASET:
        return check_dota_root(
            data_root,
            splits=DOTA_SPLITS if splits is None else splits,
            expected_images=expected_images,
            expected_instances=expected_instances,
            expected_classes=expected_classes,
        )
    return check_yolo_root(
        data_root,
        splits=YOLO_SPLITS if splits is None else splits,
        oriented=oriented,
        expected_images=expected_images,
        expected_instances=expected_instances,
        expected_classes=expected_classes,
    )
