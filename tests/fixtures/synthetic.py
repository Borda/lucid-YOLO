# SPDX-License-Identifier: Apache-2.0
"""Seeded synthetic micro-fixtures for the offline test gate (WP-007, A26).

Thin wrappers over ``fuse-augmentations`` (R21) that materialize tiny COCO
datasets on disk: a detection/segmentation set whose annotations carry *both*
axis-aligned boxes and filled polygons, an oriented-bounding-box set whose
annotations carry four rotated-box corner points, and a keypoints set whose
annotations carry animal-silhouette landmarks (WP-121b). Generation is seeded,
so a fixed seed yields byte-identical output (A26 determinism guarantee); every
helper is idempotent and skips regeneration when its annotation file already
exists.

The detection/segmentation and oriented-box sets pass an explicit geometric-only
``shapes=DEFAULT_SHAPES`` rather than relying on the vocabulary default, and the
emitted COCO ``categories`` list names those four shapes and nothing else --
``square``, ``rectangle``, ``triangle``, ``circle``. Being explicit is what keeps
that scoping visible in this file rather than left to a default the reader has to
go look up: R21's vocabulary has grown from two shape families to four across the
pins this project has used, and a fixture that named the whole union would have
moved its own category count every time upstream added a family (WP-146).
The keypoints set draws from a small, fixed animal subset instead (not every
animal fuse-augmentations ships) specifically to keep per-image class diversity
low for an overfit-style milestone run — more animals means a wider category
list an overfit run has to memorize before mAP means anything.

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
from fuse_augmentations.data.animals import animal_shapes  # type: ignore[import-untyped]
from fuse_augmentations.data.config import DEFAULT_SHAPES, SplitRatios  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from pathlib import Path

#: Seeds are fixed so regeneration is byte-identical (A26 determinism validation).
DETSEG_SEED = 20260731
OBB_SEED = 20260732
KEYPOINTS_SEED = 20260820

#: Micro-dataset sizes prescribed by the WP-007 scope / A26.
DETSEG_NUM_IMAGES = 16
OBB_NUM_IMAGES = 8
#: Smaller than the det/obb sets: animal silhouettes are larger relative to the
#: canvas than the geometric shapes, so fewer instances fit per image at the same
#: placement budget (WP-121b).
KEYPOINTS_NUM_IMAGES = 12

#: The first two animals in `AnimalShape` declaration order (duck, elephant) --
#: deliberately not the full 12-animal vocabulary, to keep the keypoints fixture's
#: category count low for an overfit-style milestone run (WP-121b).
KEYPOINTS_ANIMAL_COUNT = 2

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
        shapes=DEFAULT_SHAPES,
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
        shapes=DEFAULT_SHAPES,
        split_ratios=_SINGLE_SPLIT,
        seed=OBB_SEED,
        img_size=_IMG_SIZE,
    )
    return dataset_dir


def generate_keypoints_fixtures(root: Path) -> Path:
    """Generate the 12-image keypoints fixture set under ``root``.

    Uses ``task="keypoints"`` with only :data:`KEYPOINTS_ANIMAL_COUNT` animal
    silhouettes (not fuse-augmentations' full animal vocabulary), so every
    annotation carries a fixed-width landmark table over a small, overfit-friendly
    category count (WP-121b). Writes into ``root/keypoints`` and is idempotent.

    Args:
        root: Parent directory into which the ``keypoints`` subdirectory is
            written (created if absent).

    Returns:
        The dataset directory (``root/keypoints``), containing ``train/`` with 12
        images and one ``_annotations.coco.json``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     ds = generate_keypoints_fixtures(Path(tmp))
        ...     ds.name
        'keypoints'

        ```
    """
    dataset_dir = root / "keypoints"
    if _already_generated(dataset_dir):
        return dataset_dir
    generate_dataset(
        dataset_dir,
        num_images=KEYPOINTS_NUM_IMAGES,
        fmt="coco",
        task="keypoints",
        class_mode="shape",
        shapes=animal_shapes(KEYPOINTS_ANIMAL_COUNT),
        split_ratios=_SINGLE_SPLIT,
        seed=KEYPOINTS_SEED,
        img_size=_IMG_SIZE,
    )
    return dataset_dir
