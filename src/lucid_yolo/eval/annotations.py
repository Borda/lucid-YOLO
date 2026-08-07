# SPDX-License-Identifier: Apache-2.0
"""COCO annotation loading and letterboxed batching for the dual-path evaluator.

The reader side of :class:`~lucid_yolo.eval.coco_eval.DualPathEvaluator`: it turns a
COCO ``instances`` file into the evaluator's two inputs — the per-image ground-truth
mapping in **original** image coordinates (A10) and the ``(images, image_ids,
orig_sizes)`` batch stream — without going through the training datamodule, whose
transforms would move the boxes into the letterboxed frame.

``iscrowd`` flags and annotation areas are preserved so the COCO crowd-ignore rule
and the small/medium/large breakdown stay faithful; degenerate boxes (non-list
``bbox``, wrong arity, non-positive extent) are dropped rather than propagated as
zero-area targets.

Instance **masks** (WP-053b) are opt-in: pass an ``image_size`` to
:func:`annotations_to_target` (or ``with_masks=True`` to
:func:`load_eval_annotations`) and every target additionally carries a ``(M, H,
W)`` bool ``masks`` tensor at the original image size, aligned row-for-row with
``boxes``. The default stays detection-only, so the existing callers neither
change shape nor pay the segmentation-decode cost.

This lives in the library rather than in a script because more than one entry point
consumes it — the val2017 checkpoint evaluation and the synthetic-shapes regression
producer — and a second copy of the annotation-to-target conversion would be free to
drift from the first.

Provenance: R12 (COCO annotation format), R1 sec. 4.4. Assumptions: A9, A10.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from faster_coco_eval import mask as coco_mask
from torchvision.io import ImageReadMode, read_image

from lucid_yolo.data.targets import Targets

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from torch import Tensor

    from lucid_yolo.data.letterbox import Letterbox

#: Divisor mapping ``uint8`` pixel values onto the unit float range the model expects.
_UINT8_MAX = 255.0

#: Element count of a well-formed COCO ``bbox`` (``[x, y, width, height]``).
_XYWH_LEN = 4


@dataclass(frozen=True)
class EvalImage:
    """One evaluation image: its identity and its original coordinate frame.

    Attributes:
        image_id: COCO image id, the key both predictions and targets are matched on.
        file_name: Image file name relative to the split's image directory.
        height: Original pixel height, before letterboxing.
        width: Original pixel width, before letterboxing.

    Examples:
        >>> record = EvalImage(image_id=7, file_name="000007.jpg", height=480, width=640)
        >>> record.image_id, record.width
        (7, 640)
    """

    image_id: int
    file_name: str
    height: int
    width: int


def empty_target(image_size: tuple[int, int] | None = None) -> dict[str, Tensor]:
    """Return the ground-truth mapping of an image carrying no usable annotation.

    Args:
        image_size: Original image ``(height, width)``. When given, the target also
            carries an empty ``(0, height, width)`` ``masks`` tensor, so an
            annotation-free image still satisfies the segm metric's requirement that
            **every** target dict hold a ``masks`` key. ``None`` (the default) keeps
            the detection-only shape.

    Returns:
        A target dict whose ``boxes``, ``labels``, ``iscrowd`` and ``area`` entries are
        all empty, with the dtypes torchmetrics expects, plus ``masks`` when
        ``image_size`` is given.

    Examples:
        >>> target = empty_target()
        >>> tuple(target["boxes"].shape), target["labels"].dtype
        ((0, 4), torch.int64)
        >>> sorted(empty_target(image_size=(6, 8)))  # masks join the detection keys
        ['area', 'boxes', 'iscrowd', 'labels', 'masks']
        >>> tuple(empty_target(image_size=(6, 8))["masks"].shape)
        (0, 6, 8)
    """
    target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.long),
        "iscrowd": torch.zeros((0,), dtype=torch.long),
        "area": torch.zeros((0,), dtype=torch.float32),
    }
    if image_size is not None:
        target["masks"] = torch.zeros((0, *image_size), dtype=torch.bool)
    return target


def annotations_to_target(
    annotations: Sequence[dict[str, object]],
    image_size: tuple[int, int] | None = None,
) -> dict[str, Tensor]:
    """Convert one image's COCO annotations into a torchmetrics target mapping.

    Boxes are converted from COCO ``xywh`` to ``xyxy`` and left in original image
    coordinates. Annotations whose ``bbox`` is not a list, does not hold exactly four
    values, or has a non-positive width or height are skipped — a degenerate box would
    otherwise enter the match as an unmatchable zero-area target and depress recall.

    Passing ``image_size`` additionally decodes each surviving annotation's
    ``segmentation`` into a bool mask at the original image size (see
    :func:`annotation_mask`), stacked row-for-row with ``boxes``. The **same**
    ``bbox`` filter governs both, so a dropped degenerate annotation drops its mask
    too and the two tensors cannot describe different instances. ``iscrowd``
    annotations are treated exactly as the box path treats them — kept, with the flag
    recorded, never skipped — so a crowd region carries its mask as well.

    Args:
        annotations: The raw annotation dicts belonging to a single image.
        image_size: Original image ``(height, width)``. When given, masks are decoded
            and the target carries a ``(M, height, width)`` bool ``masks`` entry.
            ``None`` (the default) is detection-only and pays no decode cost.

    Returns:
        A target dict with ``boxes`` (``xyxy``), ``labels`` (category ids), ``iscrowd``
        and ``area`` — plus ``masks`` when ``image_size`` is given;
        :func:`empty_target` when nothing survives filtering.

    Examples:
        >>> target = annotations_to_target([{"bbox": [1.0, 2.0, 3.0, 4.0], "category_id": 5}])
        >>> target["boxes"].tolist(), target["labels"].tolist()
        ([[1.0, 2.0, 4.0, 6.0]], [5])
        >>> tuple(annotations_to_target([{"bbox": [0.0, 0.0, 0.0, 4.0], "category_id": 1}])["boxes"].shape)
        (0, 4)
        >>> polygon = {"bbox": [1.0, 2.0, 3.0, 4.0], "category_id": 5, "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]]}
        >>> masked = annotations_to_target([polygon], image_size=(8, 8))
        >>> tuple(masked["masks"].shape), int(masked["masks"].sum())
        ((1, 8, 8), 12)
    """
    boxes: list[list[float]] = []
    labels: list[int] = []
    iscrowd: list[int] = []
    area: list[float] = []
    masks: list[Tensor] = []
    for annotation in annotations:
        raw_bbox = annotation["bbox"]
        if not isinstance(raw_bbox, list):
            continue
        bbox = [float(value) for value in raw_bbox]
        if len(bbox) != _XYWH_LEN or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        x, y, width, height = bbox
        boxes.append([x, y, x + width, y + height])
        labels.append(int(cast("int", annotation["category_id"])))
        iscrowd.append(int(cast("int", annotation.get("iscrowd", 0))))
        area.append(float(cast("float", annotation.get("area", width * height))))
        if image_size is not None:
            masks.append(annotation_mask(annotation.get("segmentation"), image_size))
    if not boxes:
        return empty_target(image_size)
    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "iscrowd": torch.tensor(iscrowd, dtype=torch.long),
        "area": torch.tensor(area, dtype=torch.float32),
    }
    if image_size is not None:
        target["masks"] = torch.stack(masks)
    return target


def annotation_mask(segmentation: object, image_size: tuple[int, int]) -> Tensor:
    """Decode one COCO ``segmentation`` field into a bool mask at the original size.

    COCO stores instance segmentation in three interchangeable encodings — a list of
    flat polygon vertex lists, an uncompressed RLE (``counts`` a list of run lengths),
    and a compressed RLE (``counts`` a byte string, the form ``iscrowd`` regions use).
    All three are dispatched through :func:`faster_coco_eval.mask.segmToRle`, which
    also merges a multi-part polygon into the single RLE the instance deserves; a
    per-encoding branch written here would be a second decoder free to disagree with
    the one the metric itself uses. Note its ``(width, height)`` argument order, the
    reverse of the ``frPyObjects`` convention.

    An annotation with no ``segmentation`` field (or an empty one) yields an all-zero
    mask rather than being skipped: dropping the row would leave ``masks`` shorter
    than ``boxes``, and every later row would then describe a different instance in
    the two tensors.

    Args:
        segmentation: The raw ``segmentation`` value of one COCO annotation, or
            ``None``.
        image_size: Original image ``(height, width)`` the mask is decoded onto.

    Returns:
        A ``(height, width)`` bool tensor, ``True`` on the instance.

    Examples:
        >>> mask = annotation_mask([[1, 1, 4, 1, 4, 3, 1, 3]], image_size=(6, 6))
        >>> mask.shape, mask.dtype
        (torch.Size([6, 6]), torch.bool)
        >>> int(mask.sum())
        6
        >>> int(annotation_mask(None, image_size=(6, 6)).sum())  # no segmentation field
        0
    """
    height, width = image_size
    if not segmentation:
        return torch.zeros(image_size, dtype=torch.bool)
    rle = coco_mask.segmToRle(segmentation, width, height)
    decoded: np.ndarray[Any, np.dtype[np.uint8]] = np.ascontiguousarray(coco_mask.decode(rle))
    return torch.from_numpy(decoded).to(torch.bool)


def load_eval_annotations(
    ann_file: Path,
    with_masks: bool = False,
) -> tuple[list[EvalImage], dict[int, dict[str, Tensor]], dict[int, int]]:
    """Parse a COCO instances file into eval images, target dicts, and the label map.

    Every image in the file gets a target entry, including images with no annotation,
    so the evaluator sees the full split rather than only the annotated part of it.
    The label map reproduces :class:`~lucid_yolo.data.coco.CocoDetectionDataset`'s
    sorted-category-id order, so predicted contiguous labels can be mapped back into
    the target dicts' category-id space.

    Args:
        ann_file: Path to a COCO ``instances`` JSON file (val2017 or any compatible
            subset or synthetic split).
        with_masks: When ``True``, every target additionally carries the ``masks``
            entry :func:`annotations_to_target` decodes at that image's own
            ``(height, width)`` — the ground truth the segm half of the metric scores
            against. Defaults to ``False``: detection-only callers keep the target
            shape they have and skip the segmentation decode entirely.

    Returns:
        A triple of the image records sorted by ascending image id, the target dict per
        image id in original coordinates, and the contiguous-label to category-id map.

    Examples:
        >>> images, targets, label_map = load_eval_annotations(ann_file)  # doctest: +SKIP
        >>> len(images) == len(targets)  # doctest: +SKIP
        True
    """
    payload = json.loads(ann_file.read_text())
    sorted_ids = sorted(int(category["id"]) for category in payload["categories"])
    label_to_category = dict(enumerate(sorted_ids))
    images = sorted(
        (
            EvalImage(
                image_id=int(image["id"]),
                file_name=str(image["file_name"]),
                height=int(image["height"]),
                width=int(image["width"]),
            )
            for image in payload["images"]
        ),
        key=lambda image: image.image_id,
    )
    sizes: dict[int, tuple[int, int] | None] = {
        image.image_id: (image.height, image.width) if with_masks else None for image in images
    }
    targets: dict[int, dict[str, Tensor]] = {image.image_id: empty_target(sizes[image.image_id]) for image in images}
    grouped: dict[int, list[dict[str, object]]] = {}
    for annotation in payload["annotations"]:
        grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
    for image_id, image_annotations in grouped.items():
        targets[image_id] = annotations_to_target(image_annotations, sizes[image_id])
    return images, targets, label_to_category


def letterboxed_batches(
    images: Sequence[EvalImage],
    images_dir: Path,
    letterbox: Letterbox,
    batch_size: int,
) -> Iterator[tuple[Tensor, list[int], list[tuple[int, int]]]]:
    """Yield ``(images, image_ids, orig_sizes)`` batches for the dual-path evaluator.

    Each image is read as RGB, scaled to the unit float range, and letterboxed with the
    validation geometry. The original ``(height, width)`` travels with the batch so the
    evaluator can invert the letterbox and score in original coordinates (A10).

    Args:
        images: The eval image records to iterate, in order.
        images_dir: Directory holding the image files named by ``file_name``.
        letterbox: The validation letterbox applied to every image.
        batch_size: Number of images per yielded batch.

    Yields:
        Batches matching the :class:`~lucid_yolo.eval.coco_eval.DualPathEvaluator`
        dataloader contract.

    Examples:
        >>> for batch, ids, sizes in letterboxed_batches(images, path, letterbox, 8):  # doctest: +SKIP
        ...     batch.shape[0] == len(ids) == len(sizes)  # doctest: +SKIP
        True
    """
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        tensors = []
        for image in chunk:
            raw = read_image(str(images_dir / image.file_name), ImageReadMode.RGB)
            letterboxed, _ = letterbox(raw.to(torch.float32) / _UINT8_MAX, Targets.empty())
            tensors.append(letterboxed)
        yield (
            torch.stack(tensors),
            [image.image_id for image in chunk],
            [(image.height, image.width) for image in chunk],
        )
