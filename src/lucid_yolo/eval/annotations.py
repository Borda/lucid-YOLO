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

Numeric metadata outside the COCO domain is **refused**, not carried
(:func:`_require_valid_metadata`): a non-finite box coordinate, a non-finite or
negative ``area``, or an ``iscrowd`` that is neither ``0`` nor ``1`` raises naming the
image and annotation id. That is a different class from the degenerate boxes above —
a dropped degenerate box is an annotation this reader can read and decline, whereas
these values have no reading at all, and both scorers answer ``-1`` for every summary
when one reaches them, which is an undefined metric report with nothing in it pointing
at the annotation that caused it. Unusual-but-legal input keeps working: coordinates
off the image edge, ``area = 0``, and ``iscrowd = 1`` are all untouched.

Instance **masks** (WP-053b) are opt-in: pass an ``image_size`` to
:func:`annotations_to_target` (or ``with_masks=True`` to
:func:`load_eval_annotations`) and every target additionally carries a ``(M, H,
W)`` bool ``masks`` tensor at the original image size, aligned row-for-row with
``boxes``. The default stays detection-only, so the existing callers neither
change shape nor pay the segmentation-decode cost.

Instance **keypoints** (WP-134) are opt-in the same way: ``with_keypoints=True``
adds ``keypoints`` ``(M, K, 2)``, ``visibility`` ``(M, K)`` and ``num_keypoints``
``(M,)`` to every target, parsed from the annotation's own flat COCO
``keypoints`` field by :func:`~lucid_yolo.data.coco.parse_coco_keypoints` — the
parser the training reader uses, not a second one. ``num_keypoints`` is read from
the field COCO supplies rather than recounted from ``visibility``, because it is
that supplied value the OKS protocol ignores an instance on.

With masks on, the returned ground truth is **lazy** (:class:`LazyTargets`): a
mapping that decodes an image's masks when that image is looked up and keeps
nothing afterwards. Eager decoding is not an option at COCO scale -- val2017's
~36.8k instance annotations at ~0.3 MB per original-resolution bool mask is
about 11 GB resident before the first image is even read, where the evaluator
needs only one batch's worth alive at a time.

This lives in the library rather than in a script because more than one entry point
consumes it — the val2017 checkpoint evaluation and the synthetic-shapes regression
producer — and a second copy of the annotation-to-target conversion would be free to
drift from the first.

Provenance: R12 (COCO annotation format), R1 sec. 4.4. Assumptions: A9, A10.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping  # runtime import: LazyTargets subclasses it
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from faster_coco_eval import mask as coco_mask
from torch import Tensor  # runtime import: LazyTargets' base class subscripts it
from torchvision.io import ImageReadMode, read_image

from lucid_yolo.data.coco import parse_coco_keypoints
from lucid_yolo.data.targets import Targets

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from lucid_yolo.data.letterbox import Letterbox

#: Divisor mapping ``uint8`` pixel values onto the unit float range the model expects.
_UINT8_MAX = 255.0

#: Element count of a well-formed COCO ``bbox`` (``[x, y, width, height]``).
_XYWH_LEN = 4

#: Coordinate count of a keypoint ``(x, y)``.
_POINT_DIM = 2


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


def empty_target(image_size: tuple[int, int] | None = None, with_keypoints: bool = False) -> dict[str, Tensor]:
    """Return the ground-truth mapping of an image carrying no usable annotation.

    Args:
        image_size: Original image ``(height, width)``. When given, the target also
            carries an empty ``(0, height, width)`` ``masks`` tensor, so an
            annotation-free image still satisfies the segm metric's requirement that
            **every** target dict hold a ``masks`` key. ``None`` (the default) keeps
            the detection-only shape.
        with_keypoints: When ``True``, the target additionally carries empty
            ``keypoints``, ``visibility`` and ``num_keypoints`` entries. The point
            count is ``0`` rather than the dataset's ``K``: no instance is present to
            have points, and nothing iterates the point axis of an empty instance axis.

    Returns:
        A target dict whose ``boxes``, ``labels``, ``iscrowd`` and ``area`` entries are
        all empty, with the dtypes torchmetrics expects, plus ``masks`` when
        ``image_size`` is given and the three keypoint entries when ``with_keypoints``
        is set.

    Examples:
        >>> target = empty_target()
        >>> tuple(target["boxes"].shape), target["labels"].dtype
        ((0, 4), torch.int64)
        >>> sorted(empty_target(image_size=(6, 8)))  # masks join the detection keys
        ['area', 'boxes', 'iscrowd', 'labels', 'masks']
        >>> tuple(empty_target(image_size=(6, 8))["masks"].shape)
        (0, 6, 8)
        >>> tuple(empty_target(with_keypoints=True)["keypoints"].shape)
        (0, 0, 2)
    """
    target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.long),
        "iscrowd": torch.zeros((0,), dtype=torch.long),
        "area": torch.zeros((0,), dtype=torch.float32),
    }
    if image_size is not None:
        target["masks"] = torch.zeros((0, *image_size), dtype=torch.bool)
    if with_keypoints:
        target["keypoints"] = torch.zeros((0, 0, _POINT_DIM), dtype=torch.float32)
        target["visibility"] = torch.zeros((0, 0), dtype=torch.long)
        target["num_keypoints"] = torch.zeros((0,), dtype=torch.long)
    return target


def _require_valid_metadata(annotation: dict[str, object], bbox: Sequence[float]) -> None:
    """Refuse one annotation whose numeric metadata falls outside the COCO domain.

    Three fields, each with a domain the format states and the metric relies on: the
    box coordinates must be finite, ``area`` must be finite and non-negative, and
    ``iscrowd`` must be ``0`` or ``1``. A value outside them is not scored to a wrong
    number — both backends answer ``-1`` for *every* summary — so it is refused here,
    at the one place that reads the annotation and still knows which one it is.

    Legal-but-unusual values are deliberately untouched: a coordinate off the image
    edge is a normal COCO occurrence, ``area = 0`` is what a protocol may assign a
    hairline instance, and ``iscrowd = 1`` is the crowd flag itself.

    Args:
        annotation: The raw annotation dict, read only for its ids and its ``area``
            and ``iscrowd`` fields.
        bbox: That annotation's ``bbox`` already parsed to floats, of any length —
            the arity filter is the caller's, and a wrongly sized box still has
            coordinates worth checking.

    Raises:
        ValueError: If a coordinate or the area is non-finite, the area is negative,
            or ``iscrowd`` is outside ``{0, 1}``. The message names the image id, the
            annotation id, the field and the value.

    Examples:
        >>> _require_valid_metadata({"id": 3, "image_id": 7, "iscrowd": 1}, [1.0, 2.0, 3.0, 4.0])
        >>> _require_valid_metadata({"id": 3, "image_id": 7}, [float("nan"), 2.0, 3.0, 4.0])
        Traceback (most recent call last):
        ValueError: annotation 3 of image 7: bbox must be finite; got [nan, 2.0, 3.0, 4.0]
    """
    where = f"annotation {annotation.get('id', '?')} of image {annotation.get('image_id', '?')}"
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"{where}: bbox must be finite; got {list(bbox)}")
    raw_area = annotation.get("area")
    if raw_area is not None:
        area = float(cast("float", raw_area))
        if not math.isfinite(area) or area < 0.0:
            raise ValueError(f"{where}: area must be finite and non-negative; got {raw_area!r}")
    crowd = annotation.get("iscrowd", 0)
    if crowd not in (0, 1):
        raise ValueError(f"{where}: iscrowd must be 0 or 1; got {crowd!r}")


def _require_valid_annotations(annotations: Iterable[dict[str, object]]) -> None:
    """Run :func:`_require_valid_metadata` over a whole file's annotations.

    The eager path validates while it builds its targets, so it refuses a malformed
    file at load. :class:`LazyTargets` builds nothing until the evaluator looks an
    image up, which would move the same refusal to the middle of a scoring run — this
    is the one cheap pass that keeps both paths failing at the same moment.

    Args:
        annotations: Every annotation record of the file, in any order.

    Raises:
        ValueError: From :func:`_require_valid_metadata`, on the first bad record.

    Examples:
        >>> _require_valid_annotations([{"id": 1, "image_id": 2, "bbox": [0.0, 0.0, 1.0, 1.0]}])
    """
    for annotation in annotations:
        raw_bbox = annotation.get("bbox")
        if not isinstance(raw_bbox, list):
            continue
        _require_valid_metadata(annotation, [float(value) for value in raw_bbox])


def annotations_to_target(
    annotations: Sequence[dict[str, object]],
    image_size: tuple[int, int] | None = None,
    with_keypoints: bool = False,
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
        with_keypoints: When ``True``, each surviving annotation's flat COCO
            ``keypoints`` field is parsed into ``keypoints`` ``(M, K, 2)`` and
            ``visibility`` ``(M, K)``, and its supplied ``num_keypoints`` count is
            carried as ``num_keypoints`` ``(M,)``. Governed by the **same** ``bbox``
            filter as the boxes, for the reason the masks are.

    Returns:
        A target dict with ``boxes`` (``xyxy``), ``labels`` (category ids), ``iscrowd``
        and ``area`` — plus ``masks`` when ``image_size`` is given and the three
        keypoint entries when ``with_keypoints`` is set;
        :func:`empty_target` when nothing survives filtering.

    Raises:
        ValueError: If an annotation's box coordinates or ``area`` are non-finite, its
            ``area`` is negative, or its ``iscrowd`` is outside ``{0, 1}``
            (:func:`_require_valid_metadata`). Distinct from the degenerate-box filter
            above, which drops rather than refuses.

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
        >>> posed = {"bbox": [1.0, 2.0, 3.0, 4.0], "category_id": 1, "keypoints": [1, 2, 2, 3, 4, 0]}
        >>> target = annotations_to_target([posed], with_keypoints=True)
        >>> target["visibility"].tolist(), target["num_keypoints"].tolist()
        ([[2, 0]], [1])
    """
    boxes: list[list[float]] = []
    labels: list[int] = []
    iscrowd: list[int] = []
    area: list[float] = []
    masks: list[Tensor] = []
    points: list[Tensor] = []
    visibility: list[Tensor] = []
    num_keypoints: list[int] = []
    for annotation in annotations:
        raw_bbox = annotation["bbox"]
        if not isinstance(raw_bbox, list):
            continue
        bbox = [float(value) for value in raw_bbox]
        # Before the extent filter, not after: a NaN width fails every comparison, so
        # `bbox[2] <= 0` is False for it and the malformed box would pass as a good one.
        _require_valid_metadata(annotation, bbox)
        if len(bbox) != _XYWH_LEN or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        x, y, width, height = bbox
        boxes.append([x, y, x + width, y + height])
        labels.append(int(cast("int", annotation["category_id"])))
        iscrowd.append(int(cast("int", annotation.get("iscrowd", 0))))
        area.append(float(cast("float", annotation.get("area", width * height))))
        if image_size is not None:
            masks.append(annotation_mask(annotation.get("segmentation"), image_size))
        if with_keypoints:
            # `.get` rather than `[...]`: an annotation file with no keypoints at all --
            # `instances_val2017.json` handed to the pose protocol -- then fails with the
            # parser's own message about the field, not a bare KeyError.
            coords, visible = parse_coco_keypoints(annotation.get("keypoints"), str(annotation.get("image_id", "?")))
            points.append(coords)
            visibility.append(visible)
            num_keypoints.append(int(cast("int", annotation.get("num_keypoints", int(visible.gt(0).sum())))))
    if not boxes:
        return empty_target(image_size, with_keypoints=with_keypoints)
    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "iscrowd": torch.tensor(iscrowd, dtype=torch.long),
        "area": torch.tensor(area, dtype=torch.float32),
    }
    if image_size is not None:
        target["masks"] = torch.stack(masks)
    if with_keypoints:
        target["keypoints"] = torch.stack(points)
        target["visibility"] = torch.stack(visibility)
        target["num_keypoints"] = torch.tensor(num_keypoints, dtype=torch.long)
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


class LazyTargets(Mapping[int, dict[str, Tensor]]):
    """Ground truth that decodes an image's masks on lookup and keeps nothing after.

    A drop-in for the ``dict`` the detection path returns — same keys, same target
    dicts — differing only in *when* the work happens. That distinction is what
    makes segmentation evaluation runnable at COCO scale: the eager mapping holds
    every instance mask of every image at original resolution simultaneously
    (~11 GB on val2017), while the evaluator consumes them one batch at a time and
    the metric RLE-encodes each one on ``update``, so nothing needs to stay dense.

    Lookups are not memoised. A repeated lookup re-decodes rather than growing the
    footprint this class exists to bound; the evaluator visits each image once.

    The keypoint option composes with this rather than being lost to it: masks are why
    the mapping is lazy, and ``with_keypoints`` decides what each looked-up target
    additionally carries, so a caller asking for both gets both. The two options are
    documented as independent, and a target that quietly dropped its points would reach
    an OKS consumer looking complete.

    Attributes:
        grouped: Raw COCO annotations per image id; a missing id means an image
            with no annotations, which still gets an :func:`empty_target`.
        sizes: Original ``(height, width)`` per image id, the grid masks decode onto.
        with_keypoints: Whether each looked-up target also carries ``keypoints``,
            ``visibility`` and ``num_keypoints``.

    Examples:
        >>> polygon = {"bbox": [1.0, 2.0, 3.0, 4.0], "category_id": 5, "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]]}
        >>> targets = LazyTargets({7: [polygon]}, {7: (8, 8), 9: (8, 8)})
        >>> sorted(targets), tuple(targets[7]["masks"].shape)
        ([7, 9], (1, 8, 8))
        >>> tuple(targets[9]["masks"].shape)  # an image with no annotations
        (0, 8, 8)
        >>> posed = dict(polygon, keypoints=[1, 2, 2, 3, 4, 0], num_keypoints=1)
        >>> both = LazyTargets({7: [posed]}, {7: (8, 8), 9: (8, 8)}, with_keypoints=True)
        >>> sorted(both[7])  # masks and points, on one target
        ['area', 'boxes', 'iscrowd', 'keypoints', 'labels', 'masks', 'num_keypoints', 'visibility']
        >>> sorted(both[9])  # and on an image with no annotations
        ['area', 'boxes', 'iscrowd', 'keypoints', 'labels', 'masks', 'num_keypoints', 'visibility']
    """

    def __init__(
        self,
        grouped: Mapping[int, Sequence[dict[str, object]]],
        sizes: Mapping[int, tuple[int, int]],
        with_keypoints: bool = False,
    ) -> None:
        self._grouped = grouped
        self._sizes = sizes
        self._with_keypoints = bool(with_keypoints)

    def __getitem__(self, image_id: int) -> dict[str, Tensor]:
        """Return the target of ``image_id``, decoding its masks now."""
        size = self._sizes[image_id]
        annotations = self._grouped.get(image_id)
        if not annotations:
            return empty_target(size, with_keypoints=self._with_keypoints)
        return annotations_to_target(annotations, size, with_keypoints=self._with_keypoints)

    def __iter__(self) -> Iterator[int]:
        """Iterate the image ids, in the order the annotation file's images were sorted."""
        return iter(self._sizes)

    def __len__(self) -> int:
        """Return the number of images, annotated or not."""
        return len(self._sizes)


def load_eval_annotations(
    ann_file: Path,
    with_masks: bool = False,
    with_keypoints: bool = False,
) -> tuple[list[EvalImage], Mapping[int, dict[str, Tensor]], dict[int, int]]:
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
        with_keypoints: When ``True``, every target additionally carries the
            ``keypoints``, ``visibility`` and ``num_keypoints`` entries
            :func:`annotations_to_target` parses — the ground truth the OKS protocol
            scores against, read off a ``person_keypoints`` file. Independent of
            ``with_masks``: on its own it needs no laziness, since a whole split's
            points are a few megabytes where its masks are gigabytes, and combined with
            ``with_masks`` it rides the :class:`LazyTargets` lookup that the masks
            require. Either way the entries are on every target.

    Returns:
        A triple of the image records sorted by ascending image id, the per-image-id
        ground truth in original coordinates, and the contiguous-label to category-id
        map. The ground truth is a plain ``dict`` for the detection default and a
        :class:`LazyTargets` when ``with_masks`` is set; both satisfy the evaluator's
        ``Mapping`` contract.

    Raises:
        ValueError: If any annotation's numeric metadata is outside the COCO domain
            (:func:`_require_valid_metadata`). Refused here rather than at first
            lookup, so the lazy and eager paths fail at the same moment — the load —
            instead of part-way through a scoring run.

    Examples:
        >>> images, targets, label_map = load_eval_annotations(ann_file)  # needs a COCO file  # doctest: +SKIP
        >>> len(images) == len(targets)  # needs a COCO file  # doctest: +SKIP
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
    grouped: dict[int, list[dict[str, object]]] = {}
    for annotation in payload["annotations"]:
        grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
    if with_masks:
        sizes = {image.image_id: (image.height, image.width) for image in images}
        _require_valid_annotations(payload["annotations"])
        return images, LazyTargets(grouped, sizes, with_keypoints=with_keypoints), label_to_category
    targets: dict[int, dict[str, Tensor]] = {
        image.image_id: empty_target(with_keypoints=with_keypoints) for image in images
    }
    for image_id, image_annotations in grouped.items():
        targets[image_id] = annotations_to_target(image_annotations, with_keypoints=with_keypoints)
    return images, targets, label_to_category


def read_letterboxed_image(path: Path, letterbox: Letterbox) -> tuple[Tensor, tuple[int, int]]:
    """Read one image file and letterbox it exactly as the evaluation path does.

    The whole of what happens to an image between the disk and the model: read as RGB,
    scaled to the unit float range the model was trained on, letterboxed with the
    validation geometry. :func:`letterboxed_batches` is this function in a loop, and
    single-image inference (:func:`lucid_yolo.predict.predict_image`) is one call of it
    — which is why it is a function rather than three lines inside the batcher. A second
    copy would be free to read ``ImageReadMode.UNCHANGED``, or to scale by ``256``, and
    the model would answer plausibly either way.

    Args:
        path: Image file to read.
        letterbox: The validation letterbox applied to it.

    Returns:
        The letterboxed ``(3, out_h, out_w)`` float image, and the **original**
        ``(height, width)`` the file decoded to — the frame an inverse letterbox maps
        predictions back onto (A10).

    Examples:
        >>> image, orig_size = read_letterboxed_image(path, letterbox)  # needs an image file  # doctest: +SKIP
        >>> image.shape[0]  # needs an image file  # doctest: +SKIP
        3
    """
    raw = read_image(str(path), ImageReadMode.RGB)
    orig_size = (int(raw.shape[-2]), int(raw.shape[-1]))
    letterboxed, _ = letterbox(raw.to(torch.float32) / _UINT8_MAX, Targets.empty())
    return letterboxed, orig_size


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
        >>> for batch, ids, sizes in letterboxed_batches(images, path, letterbox, 8):  # needs images  # doctest: +SKIP
        ...     batch.shape[0] == len(ids) == len(sizes)  # needs images  # doctest: +SKIP
        True
    """
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        tensors = [read_letterboxed_image(images_dir / image.file_name, letterbox)[0] for image in chunk]
        yield (
            torch.stack(tensors),
            [image.image_id for image in chunk],
            [(image.height, image.width) for image in chunk],
        )
