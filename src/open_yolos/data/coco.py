# SPDX-License-Identifier: Apache-2.0
"""COCO-format detection/segmentation dataset and the scale-aware policy (WP-014).

:class:`CocoDetectionDataset` reads a COCO *instances* JSON once, indexes its
annotations by image id, and yields ``(image, Targets)`` pairs consumed by the
Phase 1 augmentation pipeline. It is deliberately a *thin* reader: image decoding
goes through :func:`torchvision.io.read_image` (torchvision is a core dependency,
blueprint sec. 5.9), boxes are converted from COCO ``xywh`` to the project's
``xyxy`` convention, category ids are remapped to a contiguous ``int64`` label
space, and each annotation's first segmentation ring is parsed into a ``(P, 2)``
float32 polygon so the box/polygon/rbox modalities of
:class:`~open_yolos.data.targets.Targets` stay in lock-step.

Crowd / RLE policy:
    COCO carries two kinds of ``segmentation``: a list of flat polygon rings
    (``[[x, y, x, y, ...], ...]``) for ordinary instances, and a run-length dict
    for crowd regions. This reader keeps ordinary polygon instances only —
    ``iscrowd=1`` annotations and any annotation whose ``segmentation`` is not a
    usable polygon ring (an RLE dict, an empty list, or a ring with fewer than
    three points) are **skipped**. That keeps every retained box paired with
    exactly one ring, so :class:`~open_yolos.data.targets.Targets` sees a consistent
    "one ring per box" set (or an empty set) rather than a mix.

:func:`build_scale_policy` returns the size-aware augmentation strengths of
[R1] Table S3 (blueprint sec. 5.9): the ``n`` recipe is mildest and larger
variants grow stronger. The exact per-variant tuples are this project's reading
of that table; the acceptance gate is Tier-A *behaviour* (the augmentation
pipeline runs and stays geometrically consistent), not the literal constants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from open_yolos.data.targets import Targets

if TYPE_CHECKING:
    from pathlib import Path

    from open_yolos.data.transforms import GeometricTransform

__all__ = ["CocoDetectionDataset", "build_scale_policy"]

#: Points ``(x, y)`` per polygon vertex; a flat ring must be a multiple of this.
_POINT_STRIDE = 2
#: Minimum vertices for a polygon to bound any area.
_MIN_RING_POINTS = 3
#: 8-bit image scale factor mapping ``uint8`` pixels into ``[0, 1]`` float.
_UINT8_MAX = 255.0

#: Size-aware augmentation strengths per variant ([R1] Table S3; blueprint 5.9).
#: ``scale`` is the :class:`~open_yolos.data.affine.RandomAffine` half-range, ``mixup``
#: and ``copy_paste`` are the per-sample assembly probabilities.
_SCALE_POLICY: dict[str, dict[str, float]] = {
    "n": {"scale": 0.5, "mixup": 0.0, "copy_paste": 0.1},
    "s": {"scale": 0.9, "mixup": 0.05, "copy_paste": 0.15},
    "m": {"scale": 0.9, "mixup": 0.1, "copy_paste": 0.4},
    "l": {"scale": 0.9, "mixup": 0.1, "copy_paste": 0.5},
    "x": {"scale": 0.9, "mixup": 0.2, "copy_paste": 0.6},
}


def build_scale_policy(variant: str) -> dict[str, float]:
    """Return the size-aware augmentation strengths for a model ``variant``.

    The values transcribe [R1] Table S3 (blueprint sec. 5.9): the ``n`` variant is
    mildest (``scale`` 0.5, no mixup, light copy-paste) and larger variants grow
    stronger, up to ``x`` (``scale`` 0.9, ``mixup`` 0.2, ``copy_paste`` 0.6). The
    exact tuples are this project's reading of that table; the gate is Tier-A
    behaviour, not the literal constants.

    Args:
        variant: One of ``"n"``, ``"s"``, ``"m"``, ``"l"``, ``"x"``.

    Returns:
        A fresh ``dict`` with keys ``"scale"``, ``"mixup"`` and ``"copy_paste"``.

    Raises:
        ValueError: If ``variant`` is not a known size letter.

    Examples:
        ```pycon
        >>> build_scale_policy("n")
        {'scale': 0.5, 'mixup': 0.0, 'copy_paste': 0.1}
        >>> build_scale_policy("x")["copy_paste"]
        0.6

        ```
    """
    try:
        policy = _SCALE_POLICY[variant]
    except KeyError:
        known = ", ".join(sorted(_SCALE_POLICY))
        raise ValueError(f"unknown variant {variant!r}; expected one of {known}") from None
    return dict(policy)


@dataclass(frozen=True)
class _ImageRecord:
    """One image's identity resolved from the COCO ``images`` list.

    Attributes:
        image_id: COCO image id, the key annotations reference.
        file_name: Image file name relative to the dataset's images directory.
        height: Image height in pixels (clip bound for boxes/polygons).
        width: Image width in pixels.
    """

    image_id: int
    file_name: str
    height: int
    width: int


class CocoDetectionDataset(Dataset[tuple[Tensor, Targets]]):
    """COCO-format detection/segmentation dataset yielding ``(image, Targets)``.

    The instances JSON is parsed once at construction: category ids are remapped
    to contiguous ``int64`` labels (the mapping is exposed as
    :attr:`category_id_to_label` / :attr:`label_to_category_id`) and annotations
    are grouped by image id. Each ``__getitem__`` decodes its image to a CHW
    float32 tensor in ``[0, 1]``, converts every kept annotation's ``xywh`` box to
    ``xyxy`` (clamped to the image bounds), and parses its first segmentation ring
    into a ``(P, 2)`` polygon. Crowd (``iscrowd=1``) and non-polygon (RLE / empty /
    degenerate) annotations are skipped, so a retained image carries one ring per
    box or an empty target set.

    An optional single-image ``transforms`` (any
    :class:`~open_yolos.data.transforms.GeometricTransform`, e.g.
    :class:`~open_yolos.data.letterbox.Letterbox` for validation) is applied to each
    sample before it is returned. Multi-image assemblies (mosaic, mixup,
    copy-paste) are composed at the datamodule level, not here.

    Args:
        images_dir: Directory holding the image files named by the annotations.
        annotation_file: Path to the COCO ``instances_*.json`` file.
        transforms: Optional per-image geometric transform applied to every
            sample. Defaults to ``None`` (raw decoded sample).

    Attributes:
        category_id_to_label: Mapping from COCO category id to contiguous label.
        label_to_category_id: The inverse mapping, label to COCO category id.

    Examples:
        ```pycon
        >>> CocoDetectionDataset  # doctest: +SKIP
        >>> # ds = CocoDetectionDataset(images_dir, annotation_file)
        >>> # image, targets = ds[0]  # image: (3, H, W) float32 in [0, 1]

        ```
    """

    def __init__(
        self,
        images_dir: Path,
        annotation_file: Path,
        transforms: GeometricTransform | None = None,
    ) -> None:
        self._images_dir = images_dir
        self._transforms = transforms
        with annotation_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        self.category_id_to_label, self.label_to_category_id = _build_category_maps(payload["categories"])
        self._images = _build_image_records(payload["images"])
        self._anns_by_image = _group_annotations(payload["annotations"])

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Tensor, Targets]:
        """Return the ``(image, Targets)`` pair for image ``index``.

        Args:
            index: Zero-based image index (COCO image ids are re-indexed to a dense
                ``0..len-1`` order at construction).

        Returns:
            A CHW float32 image in ``[0, 1]`` and its :class:`~open_yolos.data.targets.Targets`
            (``xyxy`` boxes, contiguous ``int64`` labels, one polygon ring per box),
            after the optional ``transforms``.
        """
        record = self._images[index]
        image = self._load_image(record.file_name)
        targets = self._build_targets(record)
        if self._transforms is not None:
            image, targets = self._transforms(image, targets)
        return image, targets

    def _load_image(self, file_name: str) -> Tensor:
        """Decode ``file_name`` into a CHW float32 RGB tensor in ``[0, 1]``.

        Args:
            file_name: Image file name relative to the images directory.

        Returns:
            A ``(3, H, W)`` float32 tensor scaled into ``[0, 1]``.

        Raises:
            OSError: If the file is missing or cannot be decoded; the offending
                path is named in the message.
        """
        path = self._images_dir / file_name
        try:
            raw = read_image(str(path), ImageReadMode.RGB)
        except (RuntimeError, OSError) as error:
            raise OSError(f"failed to read image {path}: {error}") from error
        return cast("Tensor", raw.to(torch.float32) / _UINT8_MAX)

    def _build_targets(self, record: _ImageRecord) -> Targets:
        """Assemble the :class:`~open_yolos.data.targets.Targets` for one image."""
        boxes: list[list[float]] = []
        labels: list[int] = []
        polygons: list[Tensor] = []
        for ann in self._anns_by_image.get(record.image_id, ()):
            parsed = self._parse_annotation(ann, record)
            if parsed is None:
                continue
            box, label, ring = parsed
            boxes.append(box)
            labels.append(label)
            polygons.append(ring)
        if not boxes:
            return Targets.empty()
        return Targets(
            boxes=torch.tensor(boxes, dtype=torch.float32),
            labels=torch.tensor(labels, dtype=torch.int64),
            polygons=polygons,
        )

    def _parse_annotation(self, ann: dict[str, object], record: _ImageRecord) -> tuple[list[float], int, Tensor] | None:
        """Parse one annotation into ``(xyxy_box, label, ring)`` or ``None`` to skip.

        Crowd annotations and annotations without a usable polygon ring are
        skipped (returning ``None``) per the module's crowd/RLE policy.
        """
        if int(cast("int", ann.get("iscrowd", 0))) == 1:
            return None
        ring = _parse_ring(ann.get("segmentation"))
        if ring is None:
            return None
        box = _xywh_to_xyxy(ann["bbox"], record.height, record.width)  # type: ignore[arg-type]
        label = self.category_id_to_label[int(ann["category_id"])]  # type: ignore[call-overload]
        return box, label, ring


def _build_category_maps(categories: list[dict[str, object]]) -> tuple[dict[int, int], dict[int, int]]:
    """Build contiguous label maps from a COCO ``categories`` list.

    Category ids are sorted ascending and assigned dense labels ``0..K-1`` so the
    label space is contiguous regardless of the source id numbering.

    Args:
        categories: The COCO ``categories`` entries (each with an ``"id"``).

    Returns:
        A ``(category_id_to_label, label_to_category_id)`` pair of dicts.
    """
    sorted_ids = sorted(int(cat["id"]) for cat in categories)  # type: ignore[call-overload]
    category_id_to_label = {cat_id: label for label, cat_id in enumerate(sorted_ids)}
    label_to_category_id = {label: cat_id for cat_id, label in category_id_to_label.items()}
    return category_id_to_label, label_to_category_id


def _build_image_records(images: list[dict[str, object]]) -> list[_ImageRecord]:
    """Build the ordered image-record list from the COCO ``images`` entries.

    Records are sorted by image id so indexing is stable and reproducible.

    Args:
        images: The COCO ``images`` entries.

    Returns:
        Image records ordered by ascending image id.
    """
    records = [
        _ImageRecord(
            image_id=int(img["id"]),  # type: ignore[call-overload]
            file_name=str(img["file_name"]),
            height=int(img["height"]),  # type: ignore[call-overload]
            width=int(img["width"]),  # type: ignore[call-overload]
        )
        for img in images
    ]
    records.sort(key=lambda record: record.image_id)
    return records


def _group_annotations(annotations: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    """Group COCO annotations by their ``image_id``.

    Args:
        annotations: The COCO ``annotations`` entries.

    Returns:
        A dict mapping image id to the list of its annotations (source order).
    """
    grouped: dict[int, list[dict[str, object]]] = {}
    for ann in annotations:
        grouped.setdefault(int(ann["image_id"]), []).append(ann)  # type: ignore[call-overload]
    return grouped


def _parse_ring(segmentation: object) -> Tensor | None:
    """Parse a COCO ``segmentation`` field into a ``(P, 2)`` float32 ring or ``None``.

    Only the first polygon ring of an ordinary (list-typed) segmentation is
    returned. RLE dicts, empty lists and rings with fewer than three points yield
    ``None`` so the caller skips the annotation.

    Args:
        segmentation: The annotation's ``segmentation`` value.

    Returns:
        A ``(P, 2)`` float32 tensor, or ``None`` when no usable polygon exists.
    """
    if not isinstance(segmentation, list) or not segmentation:
        return None
    flat = segmentation[0]
    if not isinstance(flat, list) or len(flat) < _MIN_RING_POINTS * _POINT_STRIDE:
        return None
    ring = torch.tensor(flat, dtype=torch.float32)
    if ring.numel() % _POINT_STRIDE != 0:
        return None
    return ring.reshape(-1, _POINT_STRIDE)


def _xywh_to_xyxy(bbox: list[float], height: int, width: int) -> list[float]:
    """Convert a COCO ``xywh`` box to ``xyxy``, clamped to the image bounds.

    Args:
        bbox: The COCO ``[x, y, w, h]`` box in pixels.
        height: Image height, the ``y`` clamp bound.
        width: Image width, the ``x`` clamp bound.

    Returns:
        The ``[x1, y1, x2, y2]`` box clamped to ``[0, width] x [0, height]``.
    """
    x, y, w, h = (float(v) for v in bbox)
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + w, 0.0), float(width))
    y2 = min(max(y + h, 0.0), float(height))
    return [x1, y1, x2, y2]
