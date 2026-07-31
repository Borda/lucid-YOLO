# SPDX-License-Identifier: Apache-2.0
"""Seeded synthetic micro-fixtures for the offline test gate (WP-007, A26).

Thin wrappers over ``fuse-augmentations`` (R21) that materialize two tiny
COCO datasets on disk: a detection/segmentation set whose annotations carry
*both* axis-aligned boxes and filled polygons, and an oriented-bounding-box set
whose annotations carry four rotated-box corner points. Generation is seeded, so
a fixed seed yields byte-identical output (A26 determinism guarantee); both
helpers are idempotent and skip regeneration when their annotation file already
exists.

These are unit-gate fixtures and offline development stand-ins only. Per the
dataset contract (AGENTS.md sec. 3, A26) they never substitute for real
COCO/DOTA during tier acceptance runs.

Examples:
    ```pycon
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     ds = generate_detseg_fixtures(Path(tmp))
    ...     (ds / "train" / "_annotations.coco.json").is_file()
    True

    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fuse_augmentations.data import generate_dataset  # type: ignore[import-untyped]
from fuse_augmentations.data.config import SplitRatios  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from pathlib import Path

#: Seeds are fixed so regeneration is byte-identical (A26 determinism validation).
DETSEG_SEED = 20260731
OBB_SEED = 20260732

#: Micro-dataset sizes prescribed by the WP-007 scope / A26.
DETSEG_NUM_IMAGES = 16
OBB_NUM_IMAGES = 8

#: Small canvas keeps the offline gate fast while still exercising every geometry field.
_IMG_SIZE = 128

#: Force a single ``train`` split so each set is one image folder plus one COCO JSON.
_SINGLE_SPLIT = SplitRatios(train=1.0, val=0.0, test=0.0)

#: Roboflow-style per-split COCO annotation filename emitted by the generator's CocoWriter.
_COCO_ANNOTATION = "_annotations.coco.json"

#: The single split all images land in given ``_SINGLE_SPLIT``.
_SPLIT = "train"


def _already_generated(dataset_dir: Path) -> bool:
    """Return whether ``dataset_dir`` already holds the expected COCO annotation file.

    Args:
        dataset_dir: Candidate dataset directory.

    Returns:
        ``True`` when the ``train`` split's ``_annotations.coco.json`` exists, so
        generation can be skipped.
    """
    return (dataset_dir / _SPLIT / _COCO_ANNOTATION).is_file()


def generate_detseg_fixtures(root: Path) -> Path:
    """Generate the 16-image detection/segmentation fixture set under ``root``.

    Uses ``task="segmentation"`` so every annotation carries both an axis-aligned
    ``bbox`` and a filled ``segmentation`` polygon. Writes into ``root/detseg`` and
    is idempotent: an existing set with its annotation file is left untouched.

    Args:
        root: Parent directory into which the ``detseg`` subdirectory is written
            (created if absent).

    Returns:
        The dataset directory (``root/detseg``), containing ``train/`` with 16
        images and one ``_annotations.coco.json``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     ds = generate_detseg_fixtures(Path(tmp))
        ...     ds.name
        'detseg'

        ```
    """
    dataset_dir = root / "detseg"
    if _already_generated(dataset_dir):
        return dataset_dir
    generate_dataset(
        dataset_dir,
        num_images=DETSEG_NUM_IMAGES,
        fmt="coco",
        task="segmentation",
        class_mode="shape",
        split_ratios=_SINGLE_SPLIT,
        seed=DETSEG_SEED,
        img_size=_IMG_SIZE,
    )
    return dataset_dir


def generate_obb_fixtures(root: Path) -> Path:
    """Generate the 8-image oriented-bounding-box fixture set under ``root``.

    Uses ``task="obb"`` so every annotation stores its rotated box as four corner
    points in the COCO ``segmentation`` field (COCO has no native oriented-box
    field). Writes into ``root/obb`` and is idempotent.

    Args:
        root: Parent directory into which the ``obb`` subdirectory is written
            (created if absent).

    Returns:
        The dataset directory (``root/obb``), containing ``train/`` with 8 images
        and one ``_annotations.coco.json``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     ds = generate_obb_fixtures(Path(tmp))
        ...     ds.name
        'obb'

        ```
    """
    dataset_dir = root / "obb"
    if _already_generated(dataset_dir):
        return dataset_dir
    generate_dataset(
        dataset_dir,
        num_images=OBB_NUM_IMAGES,
        fmt="coco",
        task="obb",
        class_mode="shape",
        split_ratios=_SINGLE_SPLIT,
        seed=OBB_SEED,
        img_size=_IMG_SIZE,
    )
    return dataset_dir
