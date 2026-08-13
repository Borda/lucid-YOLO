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

Adding a layout is one row here. What this does **not** do is read a different
*annotation* format — a YOLO ``labels/*.txt`` tree is a reader, not a spelling, and is
its own work package.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["CANDIDATES", "resolve_split"]

#: Layout conventions tried in order, as ``(images directory, annotation file)`` templates
#: taking ``{split}``. Paths are relative to the dataset root.
CANDIDATES: tuple[tuple[str, str], ...] = (
    ("{split}2017", "annotations/instances_{split}2017.json"),
    ("{split}", "annotations/instances_{split}.json"),
    ("images/{split}", "annotations/instances_{split}.json"),
    ("{split}", "{split}/_annotations.coco.json"),
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
