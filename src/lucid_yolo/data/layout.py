# SPDX-License-Identifier: Apache-2.0
"""Resolve a split's image directory and annotation file from a dataset root (WP-098).

The reader takes any COCO-format annotation file, but the *layout* around it was pinned
to one dataset: the datamodule defaulted to ``train2017`` and ``instances_train2017.json``,
which is COCO 2017's own spelling and nothing else's. Every other COCO-format root — the
tiled oriented layout this project writes itself, the per-split exports the fixture
generator emits — reached the reader only by restating all four paths on the command
line. Four overrides that are always the same four values are not configuration; they
are a naming convention that had not been written down, and the same convention was
open-coded a third time inside the oriented evaluator.

:data:`CANDIDATES` is that convention, in order:

    COCO 2017 (R12)          <root>/train2017/   <root>/annotations/instances_train2017.json
    plain COCO / tiled       <root>/train/       <root>/annotations/instances_train.json
    images-subdirectory      <root>/images/train/ <root>/annotations/instances_train.json
    per-split export         <root>/train/       <root>/train/_annotations.coco.json

A candidate counts only when **both** its directory and its annotation file exist, so a
root holding one layout cannot be resolved into another's half; the first match wins, so
a root that somehow satisfies two is resolved the same way every time rather than by
directory-iteration order. When none matches, the COCO 2017 pair is returned unchanged.
That keeps this a naming rule rather than an existence check, and leaves the "no such
file" to the reader that opens it, with the path it actually wanted in the message. An
explicit override still wins over all of it.

Adding a layout is one row here. What this does **not** do is read a different *annotation*
format: a YOLO ``labels/*.txt`` tree is a reader (:mod:`lucid_yolo.data.yolo`, WP-099), not a
spelling. Its own naming convention lives beside this one as :data:`YOLO_CANDIDATES` /
:func:`resolve_yolo_split`, deliberately a **second table rather than four more rows**. A YOLO
split's annotations are a *directory*, so the "both halves exist" predicate that makes this
table safe would have to become an existence check that accepts either kind of thing — and a
root satisfying both conventions would then resolve to whichever row came first, handing a
labels directory to the COCO reader or an ``instances_*.json`` to the YOLO one. Which
convention applies is a property of the reader asking, so the reader asks its own table::

    per-split export       <root>/train/images/   <root>/train/labels/
    split-subdirectory     <root>/images/train/   <root>/labels/train/

One caller has no reader to ask with (WP-099b): :class:`~lucid_yolo.ptl.datamodule.DetectionDataModule`
is handed a ``data_root`` and has to *decide* which reader the root calls for, which is the
question the two tables above deliberately do not answer. :func:`detect_layout` answers it, and
it is a third thing rather than a merged table — it consults both tables and demands an
**unambiguous** answer:

* Neither convention satisfied raises, naming both and every path tried. The tables fall back
  instead of raising because a reader follows them and reports the file it could not open; a
  dispatcher has no such reader, so a fallback here would just pick one at random and let the
  wrong reader report a missing file it was never pointed at.
* **Both** satisfied raises too — the one place this module refuses precedence. Inside a table
  the rows are the same reader, so first-match is merely a tie-break between spellings; across
  the two tables the loser is a different *label space* and a different image set, and picking
  one silently trains on annotations nobody named. The caller states which it meant.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CANDIDATES",
    "DATA_YAML_NAME",
    "YOLO_CANDIDATES",
    "DatasetLayout",
    "detect_layout",
    "resolve_split",
    "resolve_yolo_split",
]

#: Layout conventions tried in order, as ``(images directory, annotation file)`` templates
#: taking ``{split}``. Paths are relative to the dataset root.
CANDIDATES: tuple[tuple[str, str], ...] = (
    ("{split}2017", "annotations/instances_{split}2017.json"),
    ("{split}", "annotations/instances_{split}.json"),
    ("images/{split}", "annotations/instances_{split}.json"),
    ("{split}", "{split}/_annotations.coco.json"),
)

#: YOLO-tree conventions tried in order, as ``(images directory, labels directory)`` templates
#: taking ``{split}``. The per-split spelling comes first because it is the one a published
#: export was read from (WP-099); both halves are directories, which is why this cannot be
#: rows of :data:`CANDIDATES`.
YOLO_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("{split}/images", "{split}/labels"),
    ("images/{split}", "labels/{split}"),
)

#: File a YOLO root publishes its class list and split entries under. It lives here rather than
#: in :mod:`lucid_yolo.data.yolo` (which re-exports it) because it is the same kind of fact as
#: the tables above — a name a dataset root is recognised by — and :func:`detect_layout` needs it
#: without importing the reader.
DATA_YAML_NAME = "data.yaml"

#: Splits :func:`detect_layout` probes when a caller names none: the two a training run builds.
_PROBE_SPLITS: tuple[str, ...] = ("train", "val")


class DatasetLayout(StrEnum):
    """Which reader a dataset root's own directory names call for (WP-099b).

    A :class:`~enum.StrEnum` so a configuration file, a CLI flag and a comparison against the
    plain spelling all keep working: ``DatasetLayout.YOLO == "yolo"`` is true.

    Attributes:
        COCO: An images directory beside a COCO ``instances_*.json``
            (:data:`CANDIDATES`, read by :class:`~lucid_yolo.data.coco.CocoDetectionDataset`).
        YOLO: A ``data.yaml`` beside per-split ``images``/``labels`` directories
            (:data:`YOLO_CANDIDATES`, read by :class:`~lucid_yolo.data.yolo.YoloDetectionDataset`).

    Examples:
        >>> DatasetLayout("yolo") is DatasetLayout.YOLO
        True
    """

    COCO = "coco"
    YOLO = "yolo"


def resolve_split(data_root: Path, split: str) -> tuple[Path, Path]:
    """Return the ``(images_dir, annotation_file)`` pair for ``split`` under ``data_root``.

    Args:
        data_root: Dataset root holding split directories and ``annotations/``.
        split: Split name without a layout suffix, ``"train"`` or ``"val"``.

    Returns:
        The first pair in :data:`CANDIDATES` whose directory and annotation file
        both exist; the COCO 2017 pair when none of them does.

    Examples:
        A root matching no layout resolves to the COCO 2017 names, unchanged::

            >>> images, annotations = resolve_split(Path("/nonexistent"), "val")
            >>> images.name, annotations.name
            ('val2017', 'instances_val2017.json')

        A tiled root built by ``lucid-data build-tiles`` resolves to its own::

            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     root = Path(tmp)
            ...     _ = (root / "val").mkdir()
            ...     _ = (root / "annotations").mkdir()
            ...     _ = (root / "annotations" / "instances_val.json").write_text("{}")
            ...     images, annotations = resolve_split(root, "val")
            ...     (images.name, annotations.name)
            ('val', 'instances_val.json')

    """
    candidates = [
        (data_root / images.format(split=split), data_root / annotations.format(split=split))
        for images, annotations in CANDIDATES
    ]
    for images_dir, annotation_file in candidates:
        if images_dir.is_dir() and annotation_file.is_file():
            return images_dir, annotation_file
    return candidates[0]


def resolve_yolo_split(data_root: Path, split: str) -> tuple[Path, Path]:
    """Return the ``(images_dir, labels_dir)`` pair for ``split`` under a YOLO ``data_root``.

    The sibling of :func:`resolve_split` for the tree :mod:`lucid_yolo.data.yolo` reads, and
    the same kind of rule: a naming convention with a deterministic fallback, never an
    existence check that raises. Both halves are directories here, so both are tested with
    ``is_dir()``.

    Args:
        data_root: Dataset root holding the split's images and labels trees.
        split: Split name, ``"train"`` or ``"val"`` (a YOLO ``data.yaml`` also spells its
            validation split ``"valid"`` in the directory it points at, which is why the
            dataset's own entry outranks this convention).

    Returns:
        The first pair in :data:`YOLO_CANDIDATES` whose two directories both exist; the
        per-split pair when none of them does.

    Examples:
        A root matching no convention resolves to the per-split names, unchanged::

            >>> images, labels = resolve_yolo_split(Path("/nonexistent"), "train")
            >>> (images.parent.name, images.name), labels.name
            (('train', 'images'), 'labels')

        A root written in the split-subdirectory spelling resolves to its own::

            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     root = Path(tmp)
            ...     (root / "images" / "val").mkdir(parents=True)
            ...     (root / "labels" / "val").mkdir(parents=True)
            ...     images, labels = resolve_yolo_split(root, "val")
            ...     (images.parent.name, labels.parent.name)
            ('images', 'labels')

    """
    candidates = [
        (data_root / images.format(split=split), data_root / labels.format(split=split))
        for images, labels in YOLO_CANDIDATES
    ]
    for images_dir, labels_dir in candidates:
        if images_dir.is_dir() and labels_dir.is_dir():
            return images_dir, labels_dir
    return candidates[0]


def detect_layout(data_root: Path, splits: Sequence[str] = _PROBE_SPLITS) -> DatasetLayout:
    """Return which of the two conventions ``data_root`` actually satisfies (WP-099b).

    The question :func:`resolve_split` and :func:`resolve_yolo_split` deliberately do not answer
    — see the module docstring. A root satisfies COCO when any probed split has both halves of a
    :data:`CANDIDATES` row on disk, and YOLO when it holds a :data:`DATA_YAML_NAME` *and* any
    probed split has both directories of a :data:`YOLO_CANDIDATES` row. Satisfying one is the
    answer; satisfying neither or both raises rather than guessing.

    A YOLO root whose ``data.yaml`` points its splits somewhere neither row names is legible to
    :meth:`~lucid_yolo.data.yolo.YoloDetectionDataset.from_root` and invisible here, which is one
    of the two reasons the caller can state the layout outright instead of asking.

    Args:
        data_root: Dataset root to probe.
        splits: Split names to probe, defaulting to the two a training run builds. A root is
            matched by *any* of them, so a dataset shipping only ``train`` still resolves.

    Returns:
        The layout the root satisfies.

    Raises:
        FileNotFoundError: If neither convention is satisfied. The message names both and every
            path tried, because a root that resolves to nothing is nearly always a root spelled
            in a third way, and the paths are what say which.
        ValueError: If both are — an ambiguity this module refuses to break by precedence, since
            the two readings are different label spaces rather than two spellings of one.

    Examples:
        A root written in a COCO spelling resolves to the COCO reader::

            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     root = Path(tmp)
            ...     (root / "train2017").mkdir()
            ...     (root / "annotations").mkdir()
            ...     _ = (root / "annotations" / "instances_train2017.json").write_text("{}")
            ...     detect_layout(root)
            <DatasetLayout.COCO: 'coco'>

        A ``data.yaml`` beside an images/labels pair resolves to the YOLO one::

            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     root = Path(tmp)
            ...     (root / "train" / "images").mkdir(parents=True)
            ...     (root / "train" / "labels").mkdir(parents=True)
            ...     _ = (root / DATA_YAML_NAME).write_text("names: [car]\\n")
            ...     detect_layout(root)
            <DatasetLayout.YOLO: 'yolo'>

    """
    coco_matched, coco_tried = _probe_coco(data_root, splits)
    yolo_matched, yolo_tried = _probe_yolo(data_root, splits)
    if coco_matched and not yolo_matched:
        return DatasetLayout.COCO
    if yolo_matched and not coco_matched:
        return DatasetLayout.YOLO
    named = ", ".join(splits)
    if coco_matched and yolo_matched:
        raise ValueError(
            f"{data_root}: satisfies both dataset conventions for splits {named}, which are two "
            f"different label spaces rather than two spellings of one; state which was meant "
            f"(layout={DatasetLayout.COCO.value!r} or layout={DatasetLayout.YOLO.value!r}). "
            f"COCO matched among {_listed(coco_tried)}; YOLO matched among {_listed(yolo_tried)}"
        )
    raise FileNotFoundError(
        f"{data_root}: satisfies no dataset convention for splits {named}. "
        f"COCO (an images directory beside its annotations JSON) tried {_listed(coco_tried)}; "
        f"YOLO (a {DATA_YAML_NAME} beside an images and a labels directory) tried {_listed(yolo_tried)}"
    )


def _probe_coco(data_root: Path, splits: Sequence[str]) -> tuple[bool, list[str]]:
    """Test ``data_root`` against :data:`CANDIDATES` and report what was tried.

    Args:
        data_root: Dataset root to probe.
        splits: Split names to probe.

    Returns:
        Whether any split matched a row outright, and every ``images + annotations`` pair tried.

    Examples:
        >>> matched, tried = _probe_coco(Path("/nonexistent"), ["val"])
        >>> matched, len(tried) == len(CANDIDATES)
        (False, True)
    """
    matched = False
    tried: list[str] = []
    for split in splits:
        for images, annotation in CANDIDATES:
            images_dir = data_root / images.format(split=split)
            annotation_file = data_root / annotation.format(split=split)
            tried.append(f"{images_dir} + {annotation_file}")
            matched = matched or (images_dir.is_dir() and annotation_file.is_file())
    return matched, tried


def _probe_yolo(data_root: Path, splits: Sequence[str]) -> tuple[bool, list[str]]:
    """Test ``data_root`` against :data:`YOLO_CANDIDATES` and report what was tried.

    The ``data.yaml`` is part of the predicate, not a detail: it carries the class names, so a
    labels tree without one is not a YOLO dataset this project can read, and dispatching to the
    reader on the directories alone would replace a layout error with a missing-file one.

    Args:
        data_root: Dataset root to probe.
        splits: Split names to probe.

    Returns:
        Whether the root holds a ``data.yaml`` *and* any split matched a row, and every path
        tried — the ``data.yaml`` first, then each ``images + labels`` pair.

    Examples:
        >>> matched, tried = _probe_yolo(Path("/nonexistent"), ["val"])
        >>> matched, tried[0]
        (False, '/nonexistent/data.yaml')
    """
    data_yaml = data_root / DATA_YAML_NAME
    matched = False
    tried: list[str] = [str(data_yaml)]
    for split in splits:
        for images, labels in YOLO_CANDIDATES:
            images_dir = data_root / images.format(split=split)
            labels_dir = data_root / labels.format(split=split)
            tried.append(f"{images_dir} + {labels_dir}")
            matched = matched or (images_dir.is_dir() and labels_dir.is_dir())
    return matched and data_yaml.is_file(), tried


def _listed(tried: Sequence[str]) -> str:
    """Join probed paths into one message fragment.

    Args:
        tried: The paths a probe tried, in probe order.

    Returns:
        The paths separated by ``; ``.

    Examples:
        >>> _listed(["a", "b"])
        'a; b'
    """
    return "; ".join(tried)
