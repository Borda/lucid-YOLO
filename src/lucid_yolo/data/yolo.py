# SPDX-License-Identifier: Apache-2.0
"""YOLO-format dataset reading: a ``data.yaml`` plus a tree of label text files (WP-099).

The format is a directory convention rather than a schema. A dataset ships a ``data.yaml``
naming its class list and its per-split image directories, and one ``.txt`` file per image
whose stem matches the image's::

    data.yaml
    train/images/30074_jpg.rf.10fa6c.jpg      train/labels/30074_jpg.rf.10fa6c.txt
    valid/images/...                          valid/labels/...

Each label line is one object, whitespace-separated, class index first and every coordinate
**normalized** to the image side it measures — ``x`` divided by the width, ``y`` by the
height::

    cls cx cy w h                              (detection, five fields)
    cls x1 y1 x2 y2 x3 y3 x4 y4                (oriented, nine fields)

The class index is 0-based into the ``names`` list, so the list order *is* the label space
and no remapping happens on read — unlike :mod:`lucid_yolo.data.coco`, whose category ids
are sparse and get compacted.

Which reading applies is the caller's ``oriented`` flag, never a sniffed field count: a file
whose rows carry the wrong number of fields for the declared variant is a file this reader
cannot account for, and guessing the variant from it is how a mis-exported tree gets parsed
into plausible nonsense. Every rejection names the file and the 1-based line, as
:mod:`lucid_yolo.data.dota` does.

Output contract:
    Targets are the same :class:`~lucid_yolo.data.targets.Targets` the COCO path produces, so
    assignment, loss and metric code is untouched. Detection reading fills ``boxes`` (pixel
    ``xyxy``, clamped to the canvas) and ``labels``. Oriented reading denormalizes each row's
    quadrilateral and routes it through :func:`~lucid_yolo.data.rotated_geom.polygons_to_rboxes`,
    exactly as :mod:`lucid_yolo.data.dota` and the oriented COCO path do, with ``boxes`` the
    envelope of that same ring — so ``boxes[i]`` and ``rboxes[i]`` are one object by
    construction. ``polygons`` stays empty on both readings, and ``difficult`` is all-``False``:
    the format carries no per-instance flag, which is what A51's default already means.

When a malformed row is found, and by whom (L-31):
    This reader validates a row **lazily**, at the ``__getitem__`` that needs it, where
    :mod:`lucid_yolo.data.coco` validates its whole annotation file at construction. The
    asymmetry is the formats': one COCO split is one JSON, so parsing it early costs one
    read, while a YOLO split is one text file per image, and opening every one of them at
    construction would front-load a full pass over the tree onto every run — including the
    runs that never reach most of it.

    The consequence is real and is not papered over: a tree with one bad row hours into an
    epoch raises there, not at ``fit`` start. What makes that acceptable is that the
    whole-tree scan exists as its own step — ``lucid-data check`` parses **every** row of
    every split through :func:`scan_yolo_label_file`, this module's own grammar, and reports
    the file and 1-based line of each rejection. It is the pre-flight for ``lucid-yolo fit``
    (see :mod:`lucid_yolo.data.check`), and running it is how a YOLO root is held to the
    guarantee the COCO reader gives for free. Skipping it is choosing the late failure.

Provenance:
    R18 for the quadrilateral convention the oriented rows carry and for the long-edge fit they
    are converted by. The format itself was read from a **published dataset export** — the
    Roboflow Universe ``gongx/cars-jnnoy`` v1 YOLO export (CC BY 4.0), vendored under
    ``supervision/releases/v0.29.0/cars-dataset/`` on this machine — never from any
    implementation's reader (AGENTS.md prime directive). Measured from its 70-image train
    split: fields lie in ``(0, 1]``, derived corners land exactly on ``0`` and ``1`` (the
    export clips boxes to the image), class indices are ``0, 1, 2`` against ``nc: 3``, its
    ``data.yaml`` writes ``train: ../train/images`` beside a real ``train/images`` tree, and
    every image has a label file with no empty ones among them.

Assumptions:
    * **Split entries are relative to the ``data.yaml``, with the published ``../`` prefix
      absorbed.** The export writes ``../train/images`` for a directory that actually sits at
      ``<root>/train/images`` beside the yaml, so the literal reading escapes the root. Both
      are tried, literal first, and neither resolving is an error naming every path attempted
      — not a third guess.
    * **A row's fields must lie in ``[0, 1]``, exactly, with no tolerance.** A coordinate
      outside it is not a slightly-off box, it is a coordinate that was never normalized — a
      file written in pixels reads as boxes hundreds of image-widths away, and clamping turns
      the whole file into a wall of degenerate edge boxes it would then train on. The bound is
      compared exactly because the values arrive as decimal text and the published export
      already writes its clipped corners as exact ``0`` and ``1``.
    * **A derived pixel box is clamped to the canvas**, as :mod:`lucid_yolo.data.coco` clamps
      COCO's ``xywh``: the file's own numbers must be normalized (rejected above), but the
      geometry they describe may still round a half-pixel past an edge.
    * **A missing label file is an error; an empty one is a background image.** Writing a
      zero-byte file is a positive statement that the image holds no objects; not writing one
      is silence, and reading silence as "no objects" turns a half-finished export into a
      dataset that trains on backgrounds without saying so. ``allow_missing_labels=True`` opts
      into the other reading for a dataset that genuinely omits them.
    * **Images are enumerated by sorted file name** over :data:`IMAGE_SUFFIXES`. The format has
      no manifest — there is no ``images`` array as in COCO — so the directory listing is the
      index, sorted for a stable, reproducible order.
    * **The oriented row is exactly nine fields, class first.** The roadmap row states eight
      normalized polygon coordinates, and class-first follows the detection row; a ten-field
      row (R18's trailing ``difficult`` flag, which the normalized variant has no published
      spelling for here) is rejected rather than assumed away.
    * **``names`` may be a list or an index-keyed mapping.** Only the list form is evidenced;
      a mapping is accepted when its keys are exactly ``0..K-1``, which validates itself, and
      rejected otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from os.path import normpath
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import yaml
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from lucid_yolo.data.layout import DATA_YAML_NAME, resolve_yolo_split
from lucid_yolo.data.rotated_geom import polygons_to_rboxes
from lucid_yolo.data.targets import Targets
from lucid_yolo.data.transforms import GeometricTransform, boxes_from_polygons

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "DATA_YAML_NAME",
    "IMAGE_SUFFIXES",
    "YoloDataConfig",
    "YoloDetectionDataset",
    "load_yolo_targets",
    "resolve_split_dirs",
    "scan_yolo_label_file",
]

#: Image extensions the reader enumerates, lower-cased. ``.jpg`` is what the published export
#: writes; ``.png`` is what this project's synthetic fixtures do (A26).
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png")
#: Split keys a ``data.yaml`` may name, in the spelling it uses for each.
_SPLIT_KEYS: tuple[str, ...] = ("train", "val", "test")
#: Fields on a detection row: the class index and ``cx cy w h``.
_DETECTION_FIELDS = 5
#: Fields on an oriented row: the class index and four normalized ``(x, y)`` corners.
_ORIENTED_FIELDS = 9
#: Corner count of the quadrilateral form of a rotated box.
_QUAD_CORNERS = 4
#: Column count of a point ``(x, y)``.
_POINT_DIM = 2
#: Directory component naming images, and the one naming their labels beside it.
_IMAGES_COMPONENT = "images"
_LABELS_COMPONENT = "labels"
#: 8-bit image scale factor mapping ``uint8`` pixels into ``[0, 1]`` float.
_UINT8_MAX = 255.0


@dataclass(frozen=True)
class _Row:
    """One parsed label line, still normalized.

    Attributes:
        label: Class index, 0-based into the dataset's ``names``.
        coords: The row's normalized coordinates, already range-checked.
    """

    label: int
    coords: tuple[float, ...]


@dataclass(frozen=True)
class YoloDataConfig:
    """A dataset's own ``data.yaml``: its class names and its split image directories.

    Split entries are kept as written and resolved on demand by :meth:`images_dir`, so a
    ``test:`` line pointing at a directory that was never downloaded costs nothing until
    someone asks for that split.

    Attributes:
        root: Directory holding the ``data.yaml``; split entries resolve against it.
        names: Class names in index order — the index into this tuple *is* the class id.
        splits: Raw split entries as written in the file, keyed by split name.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> text = "names:\\n- car\\n- truck\\nnc: 2\\ntrain: ../train/images\\n"
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = (root / DATA_YAML_NAME).write_text(text, encoding="utf-8")
        ...     (root / "train" / "images").mkdir(parents=True)
        ...     config = YoloDataConfig.read(root / DATA_YAML_NAME)
        ...     (config.names, config.images_dir("train").name)
        (('car', 'truck'), 'images')

        ```
    """

    root: Path
    names: tuple[str, ...]
    splits: Mapping[str, str]

    @classmethod
    def read(cls, data_yaml: Path) -> YoloDataConfig:
        """Parse a ``data.yaml`` into its class names and split entries.

        Keys the format does not define (the export's ``roboflow:`` metadata block, for
        instance) are ignored. ``nc``, when present, must agree with the length of ``names``:
        a file disagreeing with itself is one whose class space cannot be trusted.

        Args:
            data_yaml: Path to the dataset's ``data.yaml``.

        Returns:
            The parsed configuration, rooted at the file's own directory.

        Raises:
            ValueError: If the document is not a mapping, ``names`` is missing or malformed,
                or ``nc`` contradicts it. The file is named in every message.
            OSError: If the file cannot be read.

        Examples:
            ```pycon
            >>> import tempfile
            >>> from pathlib import Path
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     path = Path(tmp) / DATA_YAML_NAME
            ...     _ = path.write_text("names: {0: plane, 1: ship}\\nval: valid/images\\n")
            ...     YoloDataConfig.read(path).names
            ('plane', 'ship')

            ```
        """
        source = str(data_yaml)
        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: expected a YAML mapping; got {type(payload).__name__}")
        names = _parse_names(payload.get("names"), source)
        _check_class_count(payload.get("nc"), names, source)
        splits = {key: str(payload[key]) for key in _SPLIT_KEYS if payload.get(key) is not None}
        return cls(root=data_yaml.parent, names=names, splits=splits)

    def images_dir(self, split: str) -> Path:
        """Resolve one split's image directory from the entry the file wrote.

        The entry is tried literally against the file's own directory first, then with its
        leading ``..`` components dropped — the published export writes ``../train/images``
        for a tree that sits at ``<root>/train/images``, and the literal reading of that
        escapes the root.

        Args:
            split: Split name as keyed in the file, e.g. ``"train"`` or ``"val"``.

        Returns:
            The first candidate that is a directory.

        Raises:
            KeyError: If the file names no such split.
            FileNotFoundError: If no candidate exists; the message lists every path tried.

        Examples:
            ```pycon
            >>> import tempfile
            >>> from pathlib import Path
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     root = Path(tmp)
            ...     (root / "valid" / "images").mkdir(parents=True)
            ...     config = YoloDataConfig(root=root, names=("car",), splits={"val": "../valid/images"})
            ...     config.images_dir("val") == root / "valid" / "images"
            True

            ```
        """
        if split not in self.splits:
            known = ", ".join(sorted(self.splits)) or "none"
            raise KeyError(f"{self.root / DATA_YAML_NAME}: no {split!r} split; the file names {known}")
        candidates = _split_candidates(self.root, self.splits[split])
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"{self.root / DATA_YAML_NAME}: {split!r} entry {self.splits[split]!r} "
            f"resolves to no directory; tried {tried}"
        )


class YoloDetectionDataset(Dataset[tuple[Tensor, Targets]]):
    """YOLO-format dataset yielding the same ``(image, Targets)`` pairs the COCO reader does.

    Images are the sorted :data:`IMAGE_SUFFIXES` files of ``images_dir``; each one's label file
    is the same stem under ``labels_dir``. Labels are parsed per sample rather than at
    construction — one image's text is a handful of rows, so the copy-on-write pressure that
    made the COCO reader precompute (WP-073) does not arise, and reading them here keeps the
    image's own decoded size as the denormalizing scale. A malformed row therefore raises at
    the ``__getitem__`` that reads it, not at construction; ``lucid-data check`` is the
    pre-flight that parses every row of the tree up front, and the module docstring states
    why the two readers differ here.

    Args:
        images_dir: Directory holding the split's images.
        labels_dir: Directory holding one ``.txt`` per image, matched by stem.
        names: Class names in index order; the length bounds every row's class index.
        transforms: Optional per-image geometric transform applied to every sample.
        oriented: Read nine-field rows as quadrilaterals and fit rotated boxes, instead of
            five-field ``cls cx cy w h`` rows. The variant is declared, never sniffed.
        allow_missing_labels: Read an image with no label file as a background image instead
            of raising. ``False`` (the default) treats the absence as the broken export it
            usually is; an empty label file means "no objects" either way.

    Attributes:
        names: The class names, in index order.
        keypoint_flip_pairs: Always ``None``. Declared so both readers answer the
            question A64 makes the *dataset's* to answer, rather than leaving the caller
            to ask only the reader it expects to have points. The answer here is not a
            placeholder: the YOLO label row carries no keypoint fields to name sides
            with, which is the same reason ``keypoint_targets=True`` is refused on a
            YOLO root.

    Examples:
        ```pycon
        >>> YoloDetectionDataset  # doctest: +SKIP
        >>> # ds = YoloDetectionDataset.from_root(root, "train")
        >>> # image, targets = ds[0]  # image: (3, H, W) float32 in [0, 1]

        ```
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        names: Sequence[str],
        transforms: GeometricTransform | None = None,
        *,
        oriented: bool = False,
        allow_missing_labels: bool = False,
    ) -> None:
        self._images_dir = images_dir
        self._labels_dir = labels_dir
        self.names = tuple(names)
        self._transforms = transforms
        self._oriented = bool(oriented)
        self._allow_missing_labels = bool(allow_missing_labels)
        self.keypoint_flip_pairs: list[tuple[int, int]] | None = None
        self._images = sorted(
            path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    @classmethod
    def from_root(
        cls,
        data_root: Path,
        split: str,
        transforms: GeometricTransform | None = None,
        *,
        oriented: bool = False,
        allow_missing_labels: bool = False,
    ) -> YoloDetectionDataset:
        """Build a split's dataset from a dataset root alone.

        The root's ``data.yaml`` is authoritative: it supplies the class names, and its own
        split entry supplies the image directory when it resolves. Where the two directories
        come from is :func:`resolve_split_dirs`, shared with the pre-run check so a validated
        root is the tree this reader then opens.

        Args:
            data_root: Directory holding ``data.yaml`` and the split trees.
            split: Split name, e.g. ``"train"`` or ``"val"``.
            transforms: Optional per-image geometric transform.
            oriented: Read the oriented nine-field rows (see the class docstring).
            allow_missing_labels: Read an image with no label file as a background image.

        Returns:
            The dataset for that split.

        Raises:
            ValueError: If the ``data.yaml`` is malformed, or the images directory's final
                component is not ``images`` so no labels directory can be derived from it.
            OSError: If ``data.yaml`` or the images directory is missing.

        Examples:
            ```pycon
            >>> YoloDetectionDataset.from_root  # doctest: +SKIP
            >>> # ds = YoloDetectionDataset.from_root(root, "val", oriented=True)

            ```
        """
        config = YoloDataConfig.read(data_root / DATA_YAML_NAME)
        images_dir, labels_dir = resolve_split_dirs(data_root, split, config)
        return cls(
            images_dir,
            labels_dir,
            config.names,
            transforms,
            oriented=oriented,
            allow_missing_labels=allow_missing_labels,
        )

    def __len__(self) -> int:
        """Return the number of images in the split."""
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Tensor, Targets]:
        """Return the ``(image, Targets)`` pair for image ``index``.

        Args:
            index: Zero-based index into the sorted image listing.

        Returns:
            A CHW float32 image in ``[0, 1]`` and its targets in pixel coordinates, after the
            optional ``transforms``.

        Raises:
            FileNotFoundError: If the image has no label file and ``allow_missing_labels`` is
                ``False``; the message names both the image and the path it looked for.
            ValueError: If any label row is malformed; the message names the file and line.
        """
        image_path = self._images[index]
        image = self._load_image(image_path)
        _, height, width = image.shape
        targets = self._load_targets(image_path, height=int(height), width=int(width))
        if self._transforms is not None:
            image, targets = self._transforms(image, targets)
        return image, targets

    def _load_image(self, path: Path) -> Tensor:
        """Decode one image into a CHW float32 RGB tensor in ``[0, 1]``.

        Args:
            path: Path to the image file.

        Returns:
            A ``(3, H, W)`` float32 tensor scaled into ``[0, 1]``.

        Raises:
            OSError: If the file is missing or cannot be decoded; the path is named.
        """
        try:
            raw = read_image(str(path), ImageReadMode.RGB)
        except (RuntimeError, OSError) as error:
            raise OSError(f"failed to read image {path}: {error}") from error
        return cast("Tensor", raw.to(torch.float32) / _UINT8_MAX)

    def _load_targets(self, image_path: Path, *, height: int, width: int) -> Targets:
        """Parse one image's label file, applying the missing-file policy.

        Args:
            image_path: The image whose label file is wanted.
            height: Decoded image height, the ``y`` denormalizing scale.
            width: Decoded image width, the ``x`` denormalizing scale.

        Returns:
            The image's targets; :meth:`~lucid_yolo.data.targets.Targets.empty` for an empty
            label file, and for a missing one under ``allow_missing_labels``.

        Raises:
            FileNotFoundError: If the label file is missing and absences are not allowed.
        """
        label_file = self._labels_dir / f"{image_path.stem}.txt"
        if not label_file.is_file():
            if self._allow_missing_labels:
                return Targets.empty()
            raise FileNotFoundError(
                f"{image_path}: no label file at {label_file}. A YOLO tree has no image manifest, so an "
                f"absent file cannot be told from a broken export; write an empty file for a background "
                f"image, or pass allow_missing_labels=True"
            )
        return load_yolo_targets(
            label_file,
            height=height,
            width=width,
            num_classes=len(self.names),
            oriented=self._oriented,
        )


def load_yolo_targets(label_file: Path, *, height: int, width: int, num_classes: int, oriented: bool) -> Targets:
    """Parse one label file into pixel-space :class:`~lucid_yolo.data.targets.Targets`.

    Blank lines are skipped; every other line must be a well-formed row of the declared
    variant. ``oriented`` and ``num_classes`` have no defaults on purpose: the variant is a
    property of the dataset the caller opened, and the class count is what bounds a row's
    class index into a real name.

    Args:
        label_file: Path to the image's ``.txt`` label file.
        height: Image height in pixels, the ``y`` denormalizing scale.
        width: Image width in pixels, the ``x`` denormalizing scale.
        num_classes: Number of classes; a row's class index must be below it.
        oriented: Read nine-field quadrilateral rows instead of five-field boxes.

    Returns:
        The image's targets: ``boxes``/``labels`` on the detection reading, plus ``rboxes``
        paired 1:1 with them on the oriented one. Empty targets for a file with no rows.

    Raises:
        ValueError: If any row has the wrong field count for the variant, a non-integer or
            out-of-range class index, a non-numeric coordinate, or a coordinate outside
            ``[0, 1]``. The message names the file and the 1-based line number.
        OSError: If the file cannot be read.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "frame.txt"
        ...     _ = path.write_text("1 0.5 0.5 0.5 0.25\\n", encoding="utf-8")
        ...     targets = load_yolo_targets(path, height=40, width=100, num_classes=3, oriented=False)
        >>> targets.boxes.tolist(), targets.labels.tolist()
        ([[25.0, 15.0, 75.0, 25.0]], [1])

        ```
    """
    lines = label_file.read_text(encoding="utf-8").splitlines()
    rows = _parse_rows(lines, str(label_file), num_classes=num_classes, oriented=oriented)
    if not rows:
        return Targets.empty()
    labels = torch.tensor([row.label for row in rows], dtype=torch.int64)
    if oriented:
        return _oriented_targets(rows, labels, height=height, width=width)
    return Targets(boxes=_detection_boxes(rows, height=height, width=width), labels=labels)


def resolve_split_dirs(data_root: Path, split: str, config: YoloDataConfig) -> tuple[Path, Path]:
    """Return the ``(images_dir, labels_dir)`` a split is actually read from.

    The dataset's own ``data.yaml`` entry outranks the naming convention: the published
    export writes ``val: ../valid/images`` for a directory no
    :data:`~lucid_yolo.data.layout.YOLO_CANDIDATES` row names (A58). A split the file does
    not mention falls back to that table
    (:func:`~lucid_yolo.data.layout.resolve_yolo_split`), which is a naming rule with a
    deterministic fallback and cannot fail.

    It is a function rather than three lines inside
    :meth:`YoloDetectionDataset.from_root` because ``lucid-data check`` validates a root
    *before* a run reads it (WP-099c), and a pre-run check resolving splits its own way
    would validate a different tree than the one that gets trained on.

    Args:
        data_root: Directory holding ``data.yaml`` and the split trees.
        split: Split name as keyed in the file, e.g. ``"train"`` or ``"val"``.
        config: The root's parsed ``data.yaml``.

    Returns:
        The split's images directory and the labels directory beside it.

    Raises:
        FileNotFoundError: If the file names the split but its entry resolves to no
            directory; the message lists every path tried.
        ValueError: If the resolved images directory has no ``images`` component, leaving
            no labels directory to derive from it.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     (root / "valid" / "images").mkdir(parents=True)
        ...     config = YoloDataConfig(root=root, names=("car",), splits={"val": "../valid/images"})
        ...     images, labels = resolve_split_dirs(root, "val", config)
        ...     (images.parent.name, images.name, labels.name)
        ('valid', 'images', 'labels')

        ```
    """
    if split not in config.splits:
        return resolve_yolo_split(data_root, split)
    images_dir = config.images_dir(split)
    return images_dir, _labels_dir_for(images_dir)


def scan_yolo_label_file(label_file: Path, *, num_classes: int, oriented: bool) -> tuple[int, frozenset[int]]:
    """Parse one label file for its object count and class ids, without its image.

    The same grammar :func:`load_yolo_targets` reads, stopping before the denormalization:
    a validator (``lucid-data check``, WP-099c) wants to know that every row parses and what
    classes the file names, and decoding the image to obtain a pixel scale it then discards
    would make checking a split as expensive as an epoch of it. Every rejection is therefore
    the reader's own, named by file and 1-based line.

    Args:
        label_file: Path to an image's ``.txt`` label file.
        num_classes: Number of classes the dataset declares; a row's class index must be
            below it, which is what makes a file written against another class list fail
            here rather than train.
        oriented: Read the nine-field oriented rows instead of five-field boxes. Declared,
            never sniffed, exactly as on the reader (see the module docstring).

    Returns:
        The number of object rows (blank lines excluded) and the distinct class ids they name.

    Raises:
        ValueError: If any row is malformed, naming the file and the 1-based line.
        OSError: If the file cannot be read.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "frame.txt"
        ...     _ = path.write_text("1 0.5 0.5 0.5 0.25\\n\\n0 0.2 0.2 0.1 0.1\\n", encoding="utf-8")
        ...     scan_yolo_label_file(path, num_classes=3, oriented=False)
        (2, frozenset({0, 1}))

        ```
    """
    lines = label_file.read_text(encoding="utf-8").splitlines()
    rows = _parse_rows(lines, str(label_file), num_classes=num_classes, oriented=oriented)
    return len(rows), frozenset(row.label for row in rows)


def _parse_names(raw: object, source: str) -> tuple[str, ...]:
    """Parse the ``names`` value into class names in index order.

    A list is taken in order. A mapping is accepted only when its keys are exactly the
    integers ``0..K-1``, which makes the ordering the file's own statement rather than this
    reader's guess at it.

    Args:
        raw: The ``names`` value as loaded.
        source: File path used in error messages.

    Returns:
        The class names, index-ordered.

    Raises:
        ValueError: If ``names`` is missing, empty, or a mapping that is not index-keyed.

    Examples:
        >>> _parse_names(["car", "truck"], "mem")
        ('car', 'truck')
        >>> _parse_names({1: "ship", 0: "plane"}, "mem")
        ('plane', 'ship')
    """
    if isinstance(raw, list):
        names = [str(name) for name in raw]
    elif isinstance(raw, dict):
        keys = sorted(raw)
        if keys != list(range(len(keys))) or not all(isinstance(key, int) for key in keys):
            raise ValueError(f"{source}: names mapping must be keyed by 0..K-1; got keys {sorted(map(str, raw))}")
        names = [str(raw[key]) for key in keys]
    else:
        raise ValueError(f"{source}: names must be a list or an index-keyed mapping; got {type(raw).__name__}")
    if not names:
        raise ValueError(f"{source}: names is empty; a dataset with no classes cannot be read")
    return tuple(names)


def _check_class_count(raw: object, names: tuple[str, ...], source: str) -> None:
    """Reject an ``nc`` that contradicts the ``names`` it is published beside.

    Args:
        raw: The ``nc`` value as loaded, or ``None`` when the key is absent.
        names: The parsed class names.
        source: File path used in error messages.

    Raises:
        ValueError: If ``nc`` is present and differs from ``len(names)``.

    Examples:
        >>> _check_class_count(2, ("car", "truck"), "mem") is None
        True
    """
    if raw is None:
        return
    if int(cast("int", raw)) != len(names):
        raise ValueError(f"{source}: nc={raw} contradicts {len(names)} names")


def _split_candidates(root: Path, entry: str) -> list[Path]:
    """Return the paths a split entry may mean, literal reading first.

    Args:
        root: Directory holding the ``data.yaml``.
        entry: The split entry exactly as written.

    Returns:
        An absolute candidate for an absolute entry; otherwise the literal reading against
        ``root``, followed by the same entry with its leading ``..`` components dropped when
        that says something different.

    Examples:
        >>> [str(p) for p in _split_candidates(Path("/data/set"), "../train/images")]
        ['/data/train/images', '/data/set/train/images']
        >>> [str(p) for p in _split_candidates(Path("/data/set"), "train/images")]
        ['/data/set/train/images']
    """
    written = Path(entry)
    if written.is_absolute():
        return [written]
    literal = Path(normpath(str(root / written)))
    stripped = root.joinpath(*written.parts[_leading_parents(written) :])
    return [literal] if stripped == literal else [literal, stripped]


def _leading_parents(written: Path) -> int:
    """Count the leading ``..`` components of a relative path.

    Args:
        written: The entry as written.

    Returns:
        How many ``..`` components the path opens with.

    Examples:
        >>> _leading_parents(Path("../../train/images"))
        2
    """
    count = 0
    for part in written.parts:
        if part != "..":
            break
        count += 1
    return count


def _labels_dir_for(images_dir: Path) -> Path:
    """Return the labels directory beside an images directory.

    The last component spelled ``images`` becomes ``labels``, which is what both published
    spellings agree on: ``<split>/images`` pairs with ``<split>/labels`` and ``images/<split>``
    with ``labels/<split>``. Rewriting the *last* such component rather than the first keeps a
    root that happens to live under a directory called ``images`` from being rewritten at its
    root instead of at its split.

    Args:
        images_dir: The split's images directory.

    Returns:
        The same path with that component replaced.

    Raises:
        ValueError: If no component is ``images``, leaving nothing to derive from.

    Examples:
        >>> _labels_dir_for(Path("/data/set/train/images"))
        PosixPath('/data/set/train/labels')
        >>> _labels_dir_for(Path("/data/set/images/train"))
        PosixPath('/data/set/labels/train')
    """
    parts = list(images_dir.parts)
    if _IMAGES_COMPONENT not in parts:
        raise ValueError(
            f"cannot derive a labels directory from {images_dir}: no {_IMAGES_COMPONENT!r} component to replace"
        )
    index = len(parts) - 1 - parts[::-1].index(_IMAGES_COMPONENT)
    parts[index] = _LABELS_COMPONENT
    return Path(*parts)


def _parse_rows(lines: Sequence[str], source: str, *, num_classes: int, oriented: bool) -> list[_Row]:
    """Parse every non-blank line of a label file into a normalized row.

    Args:
        lines: The file's lines, without their terminators.
        source: File path used in error messages.
        num_classes: Bound on a row's class index.
        oriented: Whether rows are the nine-field oriented variant.

    Returns:
        One :class:`_Row` per object line, in file order.

    Examples:
        >>> _parse_rows(["", "0 0.5 0.5 0.2 0.2"], "mem", num_classes=1, oriented=False)[0].label
        0
    """
    rows: list[_Row] = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(_parse_row(stripped, source, number, num_classes=num_classes, oriented=oriented))
    return rows


def _parse_row(line: str, source: str, number: int, *, num_classes: int, oriented: bool) -> _Row:
    """Parse one object line of the declared variant.

    Args:
        line: The stripped, non-empty line.
        source: File path used in error messages.
        number: 1-based line number used in error messages.
        num_classes: Bound on the class index.
        oriented: Whether the nine-field oriented variant is expected.

    Returns:
        The parsed row, coordinates still normalized.

    Raises:
        ValueError: If the field count is wrong for the variant, or any field fails to parse.

    Examples:
        >>> _parse_row("2 0.1 0.2 0.3 0.4", "mem", 1, num_classes=3, oriented=False).coords
        (0.1, 0.2, 0.3, 0.4)
    """
    expected = _ORIENTED_FIELDS if oriented else _DETECTION_FIELDS
    fields = line.split()
    if len(fields) != expected:
        shape = "class and eight polygon coordinates" if oriented else "class and cx cy w h"
        raise ValueError(f"{source}:{number}: expected {expected} fields ({shape}); got {len(fields)} in {line!r}")
    return _Row(
        label=_parse_class(fields[0], source, number, num_classes),
        coords=_parse_normalized(fields[1:], source, number),
    )


def _parse_class(raw: str, source: str, number: int, num_classes: int) -> int:
    """Parse a class field into an index into the dataset's ``names``.

    A token that is not an integer and an integer naming no class are different mistakes: the
    first is a malformed file, the second a file written against a different class list.

    Args:
        raw: The class field exactly as written.
        source: File path used in error messages.
        number: 1-based line number used in error messages.
        num_classes: Number of classes the dataset declares.

    Returns:
        The class index.

    Raises:
        ValueError: If the field is not an integer, or is outside ``0..num_classes-1``.

    Examples:
        >>> _parse_class("2", "mem", 1, 3)
        2
    """
    try:
        label = int(raw)
    except ValueError:
        raise ValueError(f"{source}:{number}: class index must be an integer; got {raw!r}") from None
    if not 0 <= label < num_classes:
        raise ValueError(f"{source}:{number}: class index {label} is outside 0..{num_classes - 1}")
    return label


def _parse_normalized(fields: Sequence[str], source: str, number: int) -> tuple[float, ...]:
    """Parse a row's coordinate fields, rejecting anything outside ``[0, 1]``.

    Args:
        fields: The coordinate fields following the class index.
        source: File path used in error messages.
        number: 1-based line number used in error messages.

    Returns:
        The coordinates as floats, in file order.

    Raises:
        ValueError: If a field is not a number, or lies outside ``[0, 1]`` — which means it
            was never normalized, not that it is slightly off.

    Examples:
        >>> _parse_normalized(["0.0", "1.0"], "mem", 1)
        (0.0, 1.0)
    """
    try:
        coords = tuple(float(value) for value in fields)
    except ValueError:
        raise ValueError(f"{source}:{number}: coordinates must be numeric; got {list(fields)}") from None
    for coord in coords:
        if not 0.0 <= coord <= 1.0:
            raise ValueError(
                f"{source}:{number}: coordinate {coord} is outside [0, 1]; "
                f"this format's coordinates are normalized by the image side"
            )
    return coords


def _detection_boxes(rows: Sequence[_Row], *, height: int, width: int) -> Tensor:
    """Denormalize ``cls cx cy w h`` rows into pixel ``xyxy`` boxes clamped to the canvas.

    Args:
        rows: Parsed detection rows.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        An ``(N, 4)`` float32 tensor of ``xyxy`` boxes.

    Examples:
        >>> _detection_boxes([_Row(0, (0.5, 0.5, 1.0, 1.0))], height=20, width=10).tolist()
        [[0.0, 0.0, 10.0, 20.0]]
    """
    boxes = []
    for row in rows:
        cx, cy, w, h = row.coords
        boxes.append(
            [
                min(max((cx - w / 2) * width, 0.0), float(width)),
                min(max((cy - h / 2) * height, 0.0), float(height)),
                min(max((cx + w / 2) * width, 0.0), float(width)),
                min(max((cy + h / 2) * height, 0.0), float(height)),
            ]
        )
    return torch.tensor(boxes, dtype=torch.float32)


def _oriented_targets(rows: Sequence[_Row], labels: Tensor, *, height: int, width: int) -> Targets:
    """Denormalize quadrilateral rows and fit them to rotated boxes and their envelopes.

    Both modalities come from the *same* ring, which is what makes ``boxes[i]`` and
    ``rboxes[i]`` one object rather than two lists that happen to be ordered alike — the
    instance-axis invariant :mod:`lucid_yolo.data.dota` states and every rotated transform
    relies on. ``polygons`` is left empty, as on the other two oriented readings: the quad is
    already carried by ``rboxes`` up to the rectangle fit.

    Args:
        rows: Parsed oriented rows, each carrying eight normalized coordinates.
        labels: ``(N,)`` int64 class ids aligned with ``rows``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Targets whose ``boxes``, ``labels`` and ``rboxes`` share one instance axis.

    Examples:
        >>> import torch
        >>> row = _Row(0, (0.1, 0.1, 0.5, 0.1, 0.5, 0.3, 0.1, 0.3))
        >>> targets = _oriented_targets([row], torch.tensor([0]), height=10, width=10)
        >>> targets.boxes.tolist(), [round(v, 4) for v in targets.rboxes[0].tolist()]
        ([[1.0, 1.0, 5.0, 3.0]], [3.0, 2.0, 4.0, 2.0, 0.0])
    """
    scale = torch.tensor([width, height], dtype=torch.float32)
    rings = [torch.tensor(row.coords, dtype=torch.float32).reshape(_QUAD_CORNERS, _POINT_DIM) * scale for row in rows]
    return Targets(
        boxes=boxes_from_polygons(rings),
        labels=labels,
        rboxes=polygons_to_rboxes(torch.stack(rings, dim=0)),
    )
