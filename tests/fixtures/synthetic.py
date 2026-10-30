# SPDX-License-Identifier: Apache-2.0
"""Seeded synthetic micro-fixtures for the offline test gate (WP-007, A26).

Thin wrappers over ``fuse-augmentations`` (R21) that materialize tiny COCO datasets on disk: a detection/segmentation
set whose annotations carry *both* axis-aligned boxes and filled polygons, an oriented-bounding-box set whose
annotations carry four rotated-box corner points, and a keypoints set whose annotations carry animal-silhouette
landmarks (WP-121b). Generation is seeded, so a fixed seed yields byte-identical output (A26 determinism guarantee).

Caching:
    Every helper is idempotent, and the cache it consults is keyed on the whole generation request — the exact
    ``generate_dataset`` arguments plus the installed ``fuse-augmentations`` release — recorded in a
    ``.fixture-fingerprint.json`` beside the split. Existence of the annotation file is *not* the key: the generated
    tree is gitignored, so CI is always cold while a contributor's machine is always warm, and an existence-keyed cache
    meant any change to a seed, an image count, ``DEFAULT_SHAPES`` or the pinned upstream SHA left the warm machine
    running every fixture-consuming test against the previous scene while CI ran the new one (M-45). A fingerprint
    mismatch replaces the stale set with a regenerated one; an absent fingerprint counts as a mismatch, so caches
    written before this existed refresh once. ``make clean`` clears the whole cache directory.

Concurrency:
    ``make test`` runs under pytest-xdist, so several worker processes can reach a cold or stale cache at the same
    moment. A set whose fingerprint already matches is returned without taking any lock, so readers of a valid set never
    wait. Every other caller takes an exclusive ``fcntl.flock`` on a dot-prefixed lock file beside the target
    (``.detseg.lock`` beside ``detseg``) and checks the fingerprint again once it holds the lock; inspecting the set,
    generating its replacement and publishing it all happen under that one lock. A worker that judged the set stale just
    before another published the current one therefore finds it current on the second check and returns it untouched.
    Only a lock holder ever changes the target, and it replaces only a set that is not current, so a path any caller has
    returned stays present and complete for the rest of the session.

    The replacement is generated into a private dot-prefixed staging directory beside the target, fingerprint written
    last, and published with one ``rename``, so a lock-free reader finds either no set, a complete stale one or a
    complete current one, never a half-written tree. A crash or failed rename mid-replacement can leave the target
    absent, but only where a set no caller had returned used to be: the next caller, which waits on the same lock, finds
    it missing and generates it again. The kernel drops the lock when its holder exits, however it exits. The lock file
    is never deleted, since deleting it while another process waits on it would let two holders in at once;
    ``make clean`` removes it with the rest of the cache.

    The invariant is for callers sharing one generation request. Two sessions running different code against one cache
    directory each hold a different fingerprint, and each replaces the other's set. ``fcntl`` is POSIX-only: CI runs
    Linux and macOS, and Windows is not supported.

The detection/segmentation and oriented-box sets pass an explicit geometric-only ``shapes=DEFAULT_SHAPES`` rather than
relying on the vocabulary default, and the emitted COCO ``categories`` list names those four shapes and nothing else --
``square``, ``rectangle``, ``triangle``, ``circle``. Being explicit is what keeps that scoping visible in this file
rather than left to a default the reader has to go look up: R21's vocabulary has grown from two shape families to four
across the pins this project has used, and a fixture that named the whole union would have moved its own category count
every time upstream added a family (WP-146). The keypoints set draws from a small, fixed animal subset instead (not
every animal fuse-augmentations ships) specifically to keep per-image class diversity low for an overfit-style milestone
run — more animals means a wider category list an overfit run has to memorize before mAP means anything.

These are unit-gate fixtures and offline development stand-ins only. Per the dataset contract (AGENTS.md sec. 3, A26)
they never substitute for real COCO/DOTA during tier acceptance runs.

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

import contextlib
import fcntl
import json
import shutil
import tempfile
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fuse_augmentations.data import generate_dataset  # type: ignore[import-untyped]
from fuse_augmentations.data.animals import AnimalShape  # type: ignore[import-untyped]
from fuse_augmentations.data.config import DEFAULT_SHAPES, SplitRatios  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Seeds are fixed so regeneration is byte-identical (A26 determinism validation).
DETSEG_SEED = 20260731
OBB_SEED = 20260732
KEYPOINTS_SEED = 20260820

#: Micro-dataset sizes prescribed by the WP-007 scope / A26.
DETSEG_NUM_IMAGES = 16
OBB_NUM_IMAGES = 8
#: Smaller than the det/obb sets: animal silhouettes are larger relative to the canvas than the geometric shapes, so
#: fewer instances fit per image at the same placement budget (WP-121b).
KEYPOINTS_NUM_IMAGES = 12

#: The first two animals in `AnimalShape` declaration order (duck, elephant) -- deliberately not the full 12-animal
#: vocabulary, to keep the keypoints fixture's category count low for an overfit-style milestone run (WP-121b).
KEYPOINTS_ANIMAL_COUNT = 2

#: Small canvas keeps the offline gate fast while still exercising every geometry field.
_IMG_SIZE = 128

#: Force a single ``train`` split so each set is one image folder plus one COCO JSON.
_SINGLE_SPLIT = SplitRatios(train=1.0, val=0.0, test=0.0)

#: Roboflow-style per-split COCO annotation filename emitted by the generator's CocoWriter.
_COCO_ANNOTATION = "_annotations.coco.json"

#: The single split all images land in given ``_SINGLE_SPLIT``.
_SPLIT = "train"

#: Cache key written beside a generated set, naming everything that determines its content. Dot-prefixed so it cannot be
#: mistaken for dataset content by the ``*.jpg`` globs and split readers that walk these trees.
_FINGERPRINT_NAME = ".fixture-fingerprint.json"

#: Distribution whose generator produces every fixture here; its identity is part of the cache key, since a bump changes
#: the scene without changing any argument below.
_GENERATOR_DIST = "fuse-augmentations"

#: Suffix of the advisory lock file kept beside each published set. The file is dot-prefixed like the staging
#: directories, so directory walks over the cache skip it.
_LOCK_SUFFIX = ".lock"


def _generator_identity() -> dict[str, str]:
    """Return the installed generator release, as specifically as the install allows.

    The declared version is not enough on its own: this project pins ``fuse-augmentations`` to a git commit, and a dev
    version string (``0.12.0.dev0``) does not move when the pinned SHA does. Pip records the commit in the
    distribution's ``direct_url.json``, so that file is folded in when present, which is what actually makes a SHA bump
    invalidate the cache.

    Returns:
        A ``version`` entry always, plus a ``direct_url`` entry naming the VCS revision when the distribution was
        installed from one (absent for a plain PyPI install, where the version alone is exact).
    """
    identity = {"version": metadata.version(_GENERATOR_DIST)}
    direct_url = metadata.distribution(_GENERATOR_DIST).read_text("direct_url.json")
    if direct_url is not None:
        identity["direct_url"] = direct_url
    return identity


def _fingerprint(spec: dict[str, Any]) -> str:
    """Render one generation request as its canonical cache key.

    Args:
        spec: The exact keyword arguments handed to ``generate_dataset``.

    Returns:
        Sorted-key JSON of ``spec`` plus :func:`_generator_identity`. Values with no JSON form (``SplitRatios``) fall
        back to ``repr``, which moves with every field they carry; the shape enums are string-valued and serialize as
        their own names.
    """
    return json.dumps({"generator": _generator_identity(), "spec": spec}, sort_keys=True, default=str)


def _is_current(dataset_dir: Path, fingerprint: str) -> bool:
    """Report whether ``dataset_dir`` holds a complete set generated for ``fingerprint``.

    The fingerprint file is the completion marker: it is written last, inside the staging directory, before the tree is
    published, so its presence means every other file is already there. A directory that disappears mid-check (another
    worker moving a stale set aside) reads as not current.

    Args:
        dataset_dir: Directory a set is published at.
        fingerprint: The cache key the caller needs, from :func:`_fingerprint`.

    Returns:
        ``True`` when the recorded fingerprint equals ``fingerprint`` and the split's annotation document exists.
    """
    try:
        recorded = (dataset_dir / _FINGERPRINT_NAME).read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return recorded == fingerprint and (dataset_dir / _SPLIT / _COCO_ANNOTATION).is_file()


def _lock_path(dataset_dir: Path) -> Path:
    """Return the advisory lock file guarding ``dataset_dir``: ``.<name>.lock`` beside it.

    Args:
        dataset_dir: Directory a set is published at.

    Returns:
        The lock file path in ``dataset_dir``'s parent; it may not exist yet.
    """
    return dataset_dir.parent / f".{dataset_dir.name}{_LOCK_SUFFIX}"


@contextlib.contextmanager
def _exclusive(dataset_dir: Path) -> Iterator[None]:
    """Hold the cross-process lock that every change to ``dataset_dir`` is made under.

    The lock is an ``fcntl.flock`` on a dot-prefixed file beside ``dataset_dir``. It belongs to the open file
    description, so two processes exclude each other, and so do two threads of one process that each open the file.
    Closing the file releases it, and the kernel releases it when the holder dies, so a crashed worker cannot leave the
    cache locked. The file is created on first use and never deleted: unlinking it while another caller waits on the old
    inode would let a third caller lock a new file and hold the lock alongside it.

    Args:
        dataset_dir: Directory the set is published at; its parent must already exist.

    Yields:
        Nothing; the lock is held for the body of the ``with`` block.
    """
    with _lock_path(dataset_dir).open("a") as handle:  # append mode creates the file without truncating it
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _replace(dataset_dir: Path, spec: dict[str, Any], fingerprint: str) -> None:
    """Generate a fresh set and swap it in for whatever ``dataset_dir`` holds now.

    The caller must hold :func:`_exclusive` for ``dataset_dir`` and must have found the set there not current, which is
    what makes moving it aside safe: no caller has returned it. The set is generated into a private dot-prefixed staging
    directory beside ``dataset_dir``, its fingerprint written last, and published with one ``rename``, so a lock-free
    reader never sees a half-written tree. Whatever ends up in the staging directory, the moved-aside set included, is
    removed on the way out, whether the swap succeeded or not.

    Args:
        dataset_dir: Directory the set is published at.
        spec: The exact keyword arguments to hand ``generate_dataset``.
        fingerprint: The cache key ``spec`` renders to, from :func:`_fingerprint`.

    Raises:
        OSError: When moving the old set aside or renaming the new one in fails; the target is then absent and the next
            caller generates it again.
    """
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_dir.name}-staging-", dir=dataset_dir.parent))
    try:
        candidate = staging / dataset_dir.name
        generate_dataset(candidate, **spec)
        (candidate / _FINGERPRINT_NAME).write_text(fingerprint, encoding="utf-8")
        with contextlib.suppress(FileNotFoundError):  # a cold cache has nothing to move aside
            dataset_dir.rename(staging / f"{dataset_dir.name}-stale")
        candidate.rename(dataset_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _generate_cached(dataset_dir: Path, spec: dict[str, Any]) -> Path:
    """Return ``dataset_dir``, generating it unless a matching fingerprint is already there.

    A set whose fingerprint matches is returned without taking the lock. Otherwise the caller takes :func:`_exclusive`
    and checks again, because another worker may have published the current set while this one waited; only a set still
    not current under the lock is replaced, by :func:`_replace`. A mismatched or missing fingerprint replaces the whole
    directory rather than regenerating into it: a smaller image count than last time would otherwise leave the surplus
    images behind, and a set that is partly old and partly new is worse than either. Generating under the lock means one
    pytest-xdist worker builds each set while the rest wait for it, instead of every worker building a copy and all but
    one throwing theirs away.

    Args:
        dataset_dir: Directory the set is published at (replaced when stale).
        spec: The exact keyword arguments to hand ``generate_dataset``.

    Returns:
        ``dataset_dir``, holding a set that matches ``spec`` and the installed generator.
    """
    fingerprint = _fingerprint(spec)
    if _is_current(dataset_dir, fingerprint):
        return dataset_dir
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive(dataset_dir):
        if not _is_current(dataset_dir, fingerprint):
            _replace(dataset_dir, spec, fingerprint)
    return dataset_dir


def generate_detseg_fixtures(root: Path) -> Path:
    """Generate the 16-image detection/segmentation fixture set under ``root``.

    Uses ``task="segmentation"`` so every annotation carries both an axis-aligned ``bbox`` and a filled ``segmentation``
    polygon. Writes into ``root/detseg`` and is idempotent: an existing set with its annotation file is left untouched.

    Args:
        root: Parent directory into which the ``detseg`` subdirectory is written (created if absent).

    Returns:
        The dataset directory (``root/detseg``), containing ``train/`` with 16 images and one
        ``_annotations.coco.json``.

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
    return _generate_cached(
        root / "detseg",
        {
            "num_images": DETSEG_NUM_IMAGES,
            "fmt": "coco",
            "task": "segmentation",
            "class_mode": "shape",
            "shapes": DEFAULT_SHAPES,
            "split_ratios": _SINGLE_SPLIT,
            "seed": DETSEG_SEED,
            "img_size": _IMG_SIZE,
        },
    )


def generate_obb_fixtures(root: Path) -> Path:
    """Generate the 8-image oriented-bounding-box fixture set under ``root``.

    Uses ``task="obb"`` so every annotation stores its rotated box as four corner points in the COCO ``segmentation``
    field (COCO has no native oriented-box field). Writes into ``root/obb`` and is idempotent.

    Args:
        root: Parent directory into which the ``obb`` subdirectory is written (created if absent).

    Returns:
        The dataset directory (``root/obb``), containing ``train/`` with 8 images and one ``_annotations.coco.json``.

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
    return _generate_cached(
        root / "obb",
        {
            "num_images": OBB_NUM_IMAGES,
            "fmt": "coco",
            "task": "obb",
            "class_mode": "shape",
            "shapes": DEFAULT_SHAPES,
            "split_ratios": _SINGLE_SPLIT,
            "seed": OBB_SEED,
            "img_size": _IMG_SIZE,
        },
    )


def generate_keypoints_fixtures(root: Path) -> Path:
    """Generate the 12-image keypoints fixture set under ``root``.

    Uses ``task="keypoints"`` with only :data:`KEYPOINTS_ANIMAL_COUNT` animal silhouettes (not fuse-augmentations' full
    animal vocabulary), so every annotation carries a fixed-width landmark table over a small, overfit-friendly category
    count (WP-121b). Writes into ``root/keypoints`` and is idempotent.

    Args:
        root: Parent directory into which the ``keypoints`` subdirectory is written (created if absent).

    Returns:
        The dataset directory (``root/keypoints``), containing ``train/`` with 12 images and one
        ``_annotations.coco.json``.

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
    return _generate_cached(
        root / "keypoints",
        {
            "num_images": KEYPOINTS_NUM_IMAGES,
            "fmt": "coco",
            "task": "keypoints",
            "class_mode": "shape",
            "shapes": tuple(AnimalShape)[:KEYPOINTS_ANIMAL_COUNT],
            "split_ratios": _SINGLE_SPLIT,
            "seed": KEYPOINTS_SEED,
            "img_size": _IMG_SIZE,
        },
    )
