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
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["CANDIDATES", "YOLO_CANDIDATES", "resolve_split", "resolve_yolo_split"]

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
