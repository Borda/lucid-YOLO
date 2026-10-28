# SPDX-License-Identifier: Apache-2.0
"""COCO bbox, mask, and OKS keypoint evaluation (WP-043, WP-053b, WP-124).

The detection acceptance instrument of blueprint sec. 5.11: ``bbox`` mAP50-95 on
val2017, reported for **both** decode paths (E2E and non-E2E) from one checkpoint
in one pass, exactly as [R1, Table 7] does. The expected E2E deficit of 0.6-0.8
AP ([R1] sec. 4.4) is itself a claim measured against this instrument later; this
module only builds the measurement.

The metric engine is
:class:`torchmetrics.detection.MeanAveragePrecision` pinned to its
``faster_coco_eval`` backend (WP-069) — a COCOeval-faithful reimplementation, so
the numbers match the classic COCO protocol with no compiled-extension dependency.
The installed torchmetrics 1.9 wrapper exposes only ``bbox`` and ``segm`` IoU
types, so :func:`evaluate_keypoints` drives the same backend's
:class:`faster_coco_eval.COCOeval_faster` class directly for COCO OKS evaluation.

Three pieces compose the protocol:

- :func:`detections_to_predictions` turns a fixed-size ``(B, 300, 6)`` A9
  detection batch (the shared output of
  :class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` and
  :class:`~lucid_yolo.decode.nms_path.NMSDecoder`) into torchmetrics prediction
  dicts: the ``xyxy`` corners stay ``xyxy`` (the metric's ``box_format``), the
  contiguous class label is mapped **back** to its original COCO category id, and
  score-zero padding rows are dropped.
- :func:`evaluate_bbox` drives ``MeanAveragePrecision`` over aligned prediction
  and target dicts and returns the 12 summary statistics as a named dict.
- :class:`DualPathEvaluator` runs the model forward **once per batch**, decodes
  both paths from the same dense outputs, un-letterboxes each back to original
  image coordinates (A10), and returns ``{"e2e": ..., "nms": ...}`` — one
  command, one checkpoint, both paths (the sec. 5.11 contract).

Segmentation (WP-053b) is layered on top without disturbing any of that. A
prediction dict may additionally carry ``masks``, filtered by the *same* score
mask as the boxes (:func:`detections_to_predictions`); :func:`evaluate_segm`
scores masks alone and :func:`evaluate_bbox_and_segm` scores both in **one**
metric pass via torchmetrics' tuple ``iou_type=("bbox", "segm")``, so the two
numbers come from one traversal of one matching. The combined report keeps the
bbox statistics under their bare names and prefixes the mask ones with
``segm_``: a detection-only model's report is byte-for-byte the report it was
before, and the presence of a ``segm_`` key is exactly the statement "this
checkpoint has a mask branch".

Keypoints (WP-124) add two one-shot functions. :func:`keypoints_to_predictions`
filters a fixed-size ``(B, N, K, 2)`` batch and maps contiguous labels back to COCO
category ids; :func:`evaluate_keypoints` turns those tensors and WP-121's ground
truth tensors into an in-memory COCO document, then reports the backend's ten
standard OKS statistics. The citable COCO 17-point sigma table is fixed by
:data:`COCO_KEYPOINT_OKS_SIGMAS` rather than inherited from a dependency default.

WP-134 puts both behind :class:`DualPathEvaluator` — a keypoint checkpoint is now
scored by the same one-pass, two-path machinery the other three tasks use — and
makes the OKS ground truth canonical. ``area``, ``iscrowd``, ``num_keypoints`` and
``bbox`` are read from the annotation when the target dict carries them instead of
being rebuilt from visible-point extent, because those supplied values are what
COCO's protocol normalizes, buckets and ignores on. The reconstruction survives as
the fallback for ground truth that has no such metadata.

The ground truth is supplied as torchmetrics target dicts
(``{image_id: {"boxes": ..., "labels": ...}}`` in original image coordinates,
category-id labels) rather than a ground-truth index object: torchmetrics scores
predictions directly against target tensors. Segmentation adds a ``masks`` entry
to those dicts (:func:`~lucid_yolo.eval.annotations.load_eval_annotations` with
``with_masks=True``), in the same original coordinates.

The 101-point recall grid is configured rather than defaulted (WP-092, A46). torchmetrics
builds it with a float32 ``torch.linspace``, which overshoots ``k/100`` at 36 of the 101
indices, so a class whose attained recall lands exactly on one of those boundaries forfeits
that point and ``1/101`` of its average precision — always downward, never up. That is not
an exotic case: a class with 5, 10, 20, 25, 50 or 100 ground truths lands on a grid point
at *every* recall it can attain, and 20-36% of those are forfeited. :data:`COCO_RECALL_GRID`
supplies the correctly rounded hundredths instead, through the metric's own documented
``rec_thresholds`` argument, which makes this instrument agree exactly with the oriented one
(:func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map`) at the boundaries where they used
to differ. Only the ``map`` family moves; average recall never entered the grid.

The split into private modules (WP-205) changes no import path.
:mod:`~lucid_yolo.eval._coco_streaming` holds the metric configuration, the batch-by-batch
scorers and the hotcoco backend (WP-138), :mod:`~lucid_yolo.eval._coco_keypoints` the OKS
keypoint scoring, and every entry point and constant named above still imports from here. A test
patching a name the moved code looks up patches it in the module that now holds that code.

Provenance: R1 sec. 3.2.1, R1 Table 7, R1 sec. 4.4, R22, R23. Assumptions: A9, A10, A37, A46.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from lucid_yolo.assign.grid import HEAD_STRIDES, anchor_grid
from lucid_yolo.decode.common import (
    BOX_CORNERS,
    LABEL_COLUMN,
    SCORE_COLUMN,
    gather_by_anchor,
    points_to_letterboxed_original,
    to_letterboxed_original,
)
from lucid_yolo.decode.per_task import gather_keypoint_sets
from lucid_yolo.eval._coco_keypoints import _KEYPOINT_AREA_RANGES as _KEYPOINT_AREA_RANGES  # noqa: PLC0414 - re-export
from lucid_yolo.eval._coco_keypoints import _KEYPOINT_MAX_DETS as _KEYPOINT_MAX_DETS  # noqa: PLC0414 - re-export
from lucid_yolo.eval._coco_keypoints import (
    COCO_KEYPOINT_OKS_SIGMAS,
    SYMBOL_KEYPOINT_OKS_SIGMA,
    _KeypointScorer,
    evaluate_keypoints,
    keypoints_to_predictions,
)
from lucid_yolo.eval._coco_keypoints import _pin_keypoint_params as _pin_keypoint_params  # noqa: PLC0414 - re-export
from lucid_yolo.eval._coco_streaming import (
    _METRIC_KEYS,
    _SEGM_PREFIX,
    COCO_RECALL_GRID,
    _HotcocoStreamingScorer,
    _named_stats,
    _new_metric,
    _StreamingScorer,
)
from lucid_yolo.eval._coco_streaming import hotcoco_available as hotcoco_available  # noqa: PLC0414 - re-export
from lucid_yolo.eval.segment_decode import decode_instance_masks, masks_to_original
from lucid_yolo.models.build import SegmentOutput

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Tensor, nn

    from lucid_yolo.data.letterbox import Letterbox
    from lucid_yolo.decode.common import IndexingDecoder
    from lucid_yolo.models.heads.detect import DualHeadOutput

__all__ = [
    "COCO_KEYPOINT_OKS_SIGMAS",
    "COCO_RECALL_GRID",
    "SYMBOL_KEYPOINT_OKS_SIGMA",
    "DualPathEvaluator",
    "detections_to_predictions",
    "evaluate_bbox",
    "evaluate_bbox_and_segm",
    "evaluate_keypoints",
    "evaluate_segm",
    "gather_keypoints",
    "keypoints_to_predictions",
]


def detections_to_predictions(
    detections: Tensor,
    label_to_category: Mapping[int, int],
    score_floor: float = 0.0,
    masks: Sequence[Tensor] | None = None,
    keypoints: Tensor | None = None,
) -> list[dict[str, Tensor]]:
    """Convert a fixed-size A9 detection batch into torchmetrics prediction dicts.

    Each row of the ``(B, 300, 6)`` batch is the A9 tuple
    ``[x1, y1, x2, y2, score, class]``. A row survives when its ``score`` is
    strictly above ``score_floor`` — which, at the default ``0.0``, drops the
    score-zero padding rows that fill the fixed shape. The ``xyxy`` corners are
    kept as-is (torchmetrics is driven with ``box_format="xyxy"``) and each
    contiguous class label is mapped **back** to its original COCO category id via
    ``label_to_category``.

    When ``masks`` is given, each image's mask stack is filtered by the **same**
    boolean score mask that filters its boxes — one mask tensor indexes every
    per-detection field, so no off-by-one drift between the box list and the mask
    list is representable. That matters more than it looks: masks and boxes that
    disagree by one row still produce a plausible-looking report, with every bbox
    number right and every segm number scored against the neighbouring object.

    Args:
        detections: Detections of shape ``(B, 300, 6)`` (or any ``(B, N, 6)``),
            the shared output of either decode path.
        label_to_category: Mapping from contiguous class label to original COCO
            category id (e.g.
            :attr:`~lucid_yolo.data.coco.CocoDetectionDataset.label_to_category_id`).
        score_floor: Rows with ``score`` at or below this are dropped. Defaults to
            ``0.0`` (drop only the score-zero padding rows).
        masks: Optional per-image binary instance masks, one ``(N, H, W)`` tensor
            per image in the batch, row-aligned with that image's detections.
            Images may differ in ``(H, W)`` — after the inverse letterbox each is
            on its own original grid (A10) — which is why this is a sequence and
            not one stacked tensor. ``None`` (the default) yields detection-only
            prediction dicts with no ``masks`` key at all.
        keypoints: Optional per-detection point coordinates of shape
            ``(B, N, K, 2)``, row-aligned with ``detections`` and already in
            original image coordinates. One tensor rather than a sequence because
            ``K`` is the same for every image, unlike a mask's ``(H, W)``. Filtered
            by the same score mask, for the same reason the masks are: a keypoint
            set paired with the neighbouring detection's score reports a
            plausible pose at a correct box.

    Returns:
        A length-``B`` list of prediction dicts, each with ``boxes`` (``(M, 4)``
        ``xyxy``), ``scores`` (``(M,)``) and ``labels`` (``(M,)`` long, category
        ids) — the per-image shape :class:`MeanAveragePrecision` consumes — plus
        ``masks`` (``(M, H, W)`` bool) when ``masks`` is given and ``keypoints``
        (``(M, K, 2)``) when ``keypoints`` is given.

    Examples:
        ```pycon
        >>> import torch
        >>> dets = torch.tensor(
        ...     [[[0.0, 0.0, 4.0, 4.0, 0.9, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
        ... )  # one real detection, one padding row
        >>> preds = detections_to_predictions(dets, label_to_category={1: 42})
        >>> len(preds), tuple(preds[0]["boxes"].shape)  # padding row dropped
        (1, (1, 4))
        >>> preds[0]["labels"].tolist()
        [42]
        >>> stack = torch.zeros(2, 4, 4, dtype=torch.bool)  # one mask row per detection row
        >>> stack[0, :, :] = True
        >>> masked = detections_to_predictions(dets, {1: 42}, masks=[stack])
        >>> tuple(masked[0]["masks"].shape)  # the padding row's mask went with it
        (1, 4, 4)
        >>> points = torch.zeros(1, 2, 3, 2)  # one point set per detection row
        >>> posed = detections_to_predictions(dets, {1: 42}, keypoints=points)
        >>> tuple(posed[0]["keypoints"].shape)  # the padding row's points went with it
        (1, 3, 2)

        ```
    """
    dense = detections.detach().to(device="cpu", dtype=torch.float32)
    image_masks: Sequence[Tensor | None] = [None] * dense.shape[0] if masks is None else list(masks)
    image_points: Sequence[Tensor | None] = (
        [None] * dense.shape[0] if keypoints is None else list(keypoints.detach().to(device="cpu"))
    )
    return [
        _image_to_prediction(image, label_to_category, score_floor, one_masks, one_points)
        for image, one_masks, one_points in zip(dense, image_masks, image_points, strict=True)
    ]


def _image_to_prediction(
    detections: Tensor,
    label_to_category: Mapping[int, int],
    score_floor: float,
    masks: Tensor | None = None,
    keypoints: Tensor | None = None,
) -> dict[str, Tensor]:
    """Convert one image's ``(N, 6)`` detections into a prediction dict (padding dropped)."""
    keep = detections[:, SCORE_COLUMN] > score_floor
    kept = detections[keep]
    labels = torch.tensor(
        [label_to_category[int(label)] for label in kept[:, LABEL_COLUMN]],
        dtype=torch.long,
    )
    prediction = {
        "boxes": kept[:, :BOX_CORNERS],
        "scores": kept[:, SCORE_COLUMN],
        "labels": labels,
    }
    if masks is not None:
        prediction["masks"] = masks.detach().cpu()[keep].to(torch.bool)
    if keypoints is not None:
        prediction["keypoints"] = keypoints.detach().cpu()[keep].to(torch.float32)
    return prediction


def evaluate_bbox(
    preds: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
) -> dict[str, float]:
    """Score predictions against targets with ``MeanAveragePrecision`` bbox mAP.

    Runs :class:`torchmetrics.detection.MeanAveragePrecision` (``faster_coco_eval``
    backend, ``box_format="xyxy"``) over the position-aligned ``preds`` and
    ``targets`` and returns its 12 summary statistics as a named dict (see
    :data:`_METRIC_KEYS`). The backend can chatter on ``stdout``; that is
    redirected away. An empty ``preds`` list yields an all-zero stat dict rather
    than driving the metric through a degenerate no-update path. torchmetrics
    reports ``-1.0`` for a size bucket (small/medium/large) with no ground-truth
    boxes; that sentinel is passed through unchanged.

    Args:
        preds: Per-image prediction dicts (``boxes``/``scores``/``labels``), as
            produced by :func:`detections_to_predictions`.
        targets: Per-image ground-truth dicts (``boxes``/``labels``), aligned by
            position with ``preds`` and in the same coordinate and label space.

    Returns:
        A dict from statistic name to value: ``map`` (mAP50-95), ``map_50``,
        ``map_75``, the small/medium/large AP breakdown, and the six
        average-recall (``mar_*``) entries.

    Examples:
        ```pycon
        >>> import torch
        >>> box = torch.tensor([[10.0, 12.0, 30.0, 42.0]])
        >>> preds = [{"boxes": box, "scores": torch.tensor([1.0]), "labels": torch.tensor([5])}]
        >>> targets = [{"boxes": box, "labels": torch.tensor([5])}]
        >>> stats = evaluate_bbox(preds, targets)
        >>> round(stats["map"], 3), round(stats["map_50"], 3)
        (1.0, 1.0)

        ```
    """
    if not preds:
        return dict.fromkeys(_METRIC_KEYS, 0.0)
    return _named_stats(_compute_metric(preds, targets, "bbox"), combined=False)


def evaluate_segm(
    preds: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
) -> dict[str, float]:
    """Score predicted instance masks against target masks with segm mAP.

    The mask-side twin of :func:`evaluate_bbox`, and identical to it in every
    respect but the ``iou_type``: the same backend, the same 12 statistics under
    the same names, the same all-zero dict for an empty ``preds`` list, the same
    ``-1.0`` sentinel for an empty size bucket. The overlap that drives the
    matching is mask intersection-over-union rather than box
    intersection-over-union, so the ``boxes`` entries of both dicts are ignored
    here — they must still be present, since torchmetrics reads them for the
    small/medium/large area breakdown.

    Use :func:`evaluate_bbox_and_segm` when both metrics are wanted: it computes
    them in one pass instead of two.

    Args:
        preds: Per-image prediction dicts carrying ``masks`` (``(M, H, W)`` bool)
            beside the detection entries, as produced by
            :func:`detections_to_predictions` with its ``masks`` argument.
        targets: Per-image ground-truth dicts carrying ``masks`` at the same
            resolution, aligned by position with ``preds``.

    Returns:
        The same named 12-statistic dict :func:`evaluate_bbox` returns, computed
        over mask overlap.

    Examples:
        ```pycon
        >>> import torch
        >>> box = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
        >>> mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        >>> mask[0, 1:3, 1:3] = True
        >>> preds = [{"boxes": box, "scores": torch.tensor([1.0]), "labels": torch.tensor([5]), "masks": mask}]
        >>> targets = [{"boxes": box, "labels": torch.tensor([5]), "masks": mask}]
        >>> round(evaluate_segm(preds, targets)["map"], 3)
        1.0

        ```
    """
    if not preds:
        return dict.fromkeys(_METRIC_KEYS, 0.0)
    computed = _compute_metric(preds, targets, "segm")
    return {key: float(computed[key]) for key in _METRIC_KEYS}


def evaluate_bbox_and_segm(
    preds: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
) -> dict[str, float]:
    """Score boxes and masks together in one ``MeanAveragePrecision`` pass.

    Driving the metric with the tuple ``iou_type=("bbox", "segm")`` evaluates both
    overlaps over one traversal of one set of prediction and target dicts, so the
    two columns of the report are guaranteed to describe the same detections — two
    separate evaluations could silently be fed differently filtered inputs.

    The returned names are chosen so a segmentation report is a **superset** of a
    detection report: the box statistics keep the bare names
    :func:`evaluate_bbox` gives them (``map``, ``map_50``, ...) and the mask ones
    are prefixed ``segm_`` (``segm_map``, ``segm_map_50``, ...). A caller reading
    ``report["map"]`` therefore reads the bbox mAP whether or not the checkpoint
    has a mask branch, and the presence of any ``segm_`` key is the signal that it
    does.

    Args:
        preds: Per-image prediction dicts carrying ``masks`` beside the detection
            entries.
        targets: Per-image ground-truth dicts carrying ``masks``, aligned by
            position with ``preds``.

    Returns:
        A 24-entry dict: the 12 bbox statistics under their bare names plus the 12
        mask statistics under ``segm_``-prefixed names. An empty ``preds`` list
        yields the same key set with every value ``0.0``.

    Examples:
        ```pycon
        >>> import torch
        >>> box = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
        >>> mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        >>> mask[0, 1:3, 1:3] = True
        >>> preds = [{"boxes": box, "scores": torch.tensor([1.0]), "labels": torch.tensor([5]), "masks": mask}]
        >>> targets = [{"boxes": box, "labels": torch.tensor([5]), "masks": mask}]
        >>> stats = evaluate_bbox_and_segm(preds, targets)
        >>> round(stats["map"], 3), round(stats["segm_map"], 3)
        (1.0, 1.0)

        ```
    """
    if not preds:
        return dict.fromkeys([*_METRIC_KEYS, *(_SEGM_PREFIX + key for key in _METRIC_KEYS)], 0.0)
    return _named_stats(_compute_metric(preds, targets, ("bbox", "segm")), combined=True)


def _compute_metric(
    preds: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    iou_type: str | tuple[str, ...],
) -> Mapping[str, Tensor]:
    """Update a fresh metric with everything at once and return its raw compute dict."""
    metric = _new_metric(iou_type)
    with contextlib.redirect_stdout(io.StringIO()):
        metric.update(preds, targets)
        computed: Mapping[str, Tensor] = metric.compute()
    return computed


@dataclass(frozen=True)
class _BatchGeometry:
    """The per-batch quantities both decode paths share.

    Grouped into one object so the per-path decode takes a handful of arguments
    rather than a parameter list long enough to permute silently, and so the two
    paths provably run on the *same* anchor grid, the same canvas and the same
    original sizes.

    Attributes:
        anchor_points: Anchor-centre ``(x, y)`` coordinates ``(A, 2)``.
        strides: Per-anchor level stride ``(A,)``.
        canvas: The letterboxed ``(height, width)`` the model saw.
        orig_sizes: Per-image original ``(height, width)`` to map back onto (A10).
        prototypes: Raw prototype maps ``(B, K, Hp, Wp)`` when the model has a
            mask branch, ``None`` for a detection-only checkpoint.
    """

    anchor_points: Tensor
    strides: Tensor
    canvas: tuple[int, int]
    orig_sizes: Sequence[tuple[int, int]]
    prototypes: Tensor | None


def _segment_parts(output: object) -> tuple[DualHeadOutput, Tensor | None]:
    """Split a model forward result into its dual-head output and its prototypes.

    A segmentation model returns a
    :class:`~lucid_yolo.models.build.SegmentOutput` wrapping the dual-head output;
    a detection-only model returns the dual-head output itself. The mask branch is
    detected by that type, not by testing whether coefficients happen to be
    present: prototypes and coefficients are two halves of Eq. 7, and a checkpoint
    carrying one without the other cannot produce a mask at all.
    """
    if isinstance(output, SegmentOutput):
        return output.detect, output.prototypes
    return cast("DualHeadOutput", output), None


def gather_keypoints(keypoints: Tensor, anchor_index: Tensor) -> Tensor:
    """Select each detection's decoded point set by its own source anchor index.

    The point-set name of :func:`~lucid_yolo.decode.common.gather_by_anchor`, public since
    WP-137 because :class:`~lucid_yolo.ptl.module.DetectionLitModule`'s own epoch-level
    ``val/oks_mAP`` needs the identical gather on the identical contract. It delegates to
    the shared gather rather than carrying a second copy of its padding rule — the drift
    this project's duplication guard exists to catch (WP-109's 3-strides lesson). It is a
    function of its own rather than an alias so the parameter names it has been called by
    since WP-137 keep binding: ``gather_keypoints(keypoints=..., anchor_index=...)``.

    Args:
        keypoints: Decoded dense point coordinates of shape ``(B, A, K, 2)``.
        anchor_index: Source anchor index per detection, shape ``(B, N)``.

    Returns:
        Point coordinates of shape ``(B, N, K, 2)``, padding rows all zero.

    Examples:
        ```pycon
        >>> keypoints = torch.arange(12.0).reshape(1, 3, 2, 2)
        >>> gathered = gather_keypoints(keypoints=keypoints, anchor_index=torch.tensor([[2, -1]]))
        >>> gathered.shape
        torch.Size([1, 2, 2, 2])
        >>> gathered[0, 0]
        tensor([[ 8.,  9.],
                [10., 11.]])
        >>> gathered[0, 1]
        tensor([[0., 0.],
                [0., 0.]])

        ```
    """
    return gather_by_anchor(keypoints, anchor_index)


class DualPathEvaluator:
    """Evaluate both decode paths from one checkpoint in a single loader pass.

    Implements the blueprint sec. 5.11 contract: for each batch the model runs
    **once**, both the suppression-free E2E path
    (:class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` over the one-to-one branch)
    and the non-E2E path (:class:`~lucid_yolo.decode.nms_path.NMSDecoder` over the
    dense one-to-many branch) are decoded from the same
    :class:`~lucid_yolo.models.heads.detect.DualHeadOutput`, each un-letterboxed to
    original image coordinates (A10), and accumulated into separate prediction
    lists. :meth:`evaluate` returns ``{"e2e": ..., "nms": ...}`` scored by
    :func:`evaluate_bbox`.

    Given a **segmentation** model — one returning a
    :class:`~lucid_yolo.models.build.SegmentOutput` — each path additionally
    decodes its own instance masks and the report gains the ``segm_``-prefixed
    statistics of :func:`evaluate_bbox_and_segm`. Three properties make that
    addition safe rather than merely present:

    - each path's mask coefficients are gathered by the anchor indices **its own**
      decoder reports (:meth:`~lucid_yolo.decode.topk_e2e.TopKDecoder.decode_with_indices`),
      never by a second ranking computed here — the E2E path reads ``o2o_coeff``
      and the dense path ``o2m_coeff``, from their own branches;
    - masks are assembled while the boxes are still in the letterboxed frame,
      because :func:`~lucid_yolo.eval.segment_decode.decode_instance_masks` crops
      to the predicted box on that canvas, and only then is each image's stack
      landed in original coordinates by
      :func:`~lucid_yolo.eval.segment_decode.masks_to_original`. Mapping the boxes
      first and cropping afterwards leaves every bbox number right and every segm
      number quietly wrong;
    - a detection-only model takes exactly the path it took before — the plain
      decoder call, :func:`evaluate_bbox`, no ``masks`` key anywhere.

    Given a **keypoint** model — one whose head output carries point offsets — the
    same three properties hold with ``o2o_keypoints``/``o2m_keypoints`` in place of
    the coefficients, and the report gains the ten ``oks_``-prefixed statistics of
    :func:`evaluate_keypoints` beside its box ones. Two differences from the mask
    case, both forced by what the pieces are:
    :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints` reads the whole
    dense anchor axis and offers no per-detection indexing, so this path decodes
    the batch and *then* gathers rather than the other way round; and OKS is one
    matching over one document with no incremental update, so the points are
    accumulated for the split and scored once (:class:`_KeypointScorer`) where the
    boxes stream. The ground truth must then carry ``keypoints``, ``visibility``
    and — for the canonical protocol rather than a reconstruction of it —
    ``num_keypoints``, ``area`` and ``iscrowd``, which
    :func:`~lucid_yolo.eval.annotations.load_eval_annotations` supplies under
    ``with_keypoints=True``.

    The ``dataloader`` passed to :meth:`evaluate` yields
    ``(images, image_ids, orig_sizes)`` batches, where ``images`` is a letterboxed
    ``(B, 3, H, W)`` tensor (``H``/``W`` divisible by 32), ``image_ids`` are the
    ``B`` COCO image ids, and ``orig_sizes`` are the ``B`` original ``(height,
    width)`` pairs the boxes are mapped back onto. The ground truth is a mapping
    from image id to a torchmetrics target dict in the same original coordinates —
    which must carry ``masks`` when the model has a mask branch (see
    :func:`~lucid_yolo.eval.annotations.load_eval_annotations` with
    ``with_masks=True``).

    Args:
        model: Any module whose forward maps ``(B, 3, H, W)`` images to a
            :class:`~lucid_yolo.models.heads.detect.DualHeadOutput` (e.g.
            :class:`~lucid_yolo.ptl.module.DetectionLitModule` or a bare
            backbone/neck/head composition) or to a
            :class:`~lucid_yolo.models.build.SegmentOutput`.
        e2e_decoder: The one-to-one E2E decoder, called
            ``(o2o_cls, o2o_box, anchor_points, strides) -> (B, 300, 6)``; for a
            segmentation model it must also offer ``decode_with_indices`` with the
            same signature, returning the detections and their anchor indices.
        nms_decoder: The dense-branch NMS decoder, called with the one-to-many
            outputs and the same signature.
        label_to_category: Mapping from contiguous class label to original COCO
            category id, applied to both paths' detections so predicted labels
            share the target dicts' category-id space.
        letterbox: The validation :class:`~lucid_yolo.data.letterbox.Letterbox`
            whose geometry (its ``allow_upscale`` setting) inverts the resize.
        strides: The head's per-level input strides. Defaults to
            :data:`~lucid_yolo.assign.grid.HEAD_STRIDES`.
        keypoint_sigmas: Per-point OKS sigmas, used only when the model has a
            keypoint branch, and then required to have one entry per predicted
            point. Defaults to :data:`COCO_KEYPOINT_OKS_SIGMAS`, which is COCO's
            17-point person table and therefore wrong for any other ``K``.

    Examples:
        ```pycon
        >>> evaluator = DualPathEvaluator(  # doctest: +SKIP
        ...     model, TopKDecoder(), NMSDecoder(), dataset.label_to_category_id, letterbox
        ... )
        >>> report = evaluator.evaluate(val_loader, targets, torch.device("cpu"))  # doctest: +SKIP
        >>> sorted(report)  # doctest: +SKIP
        ['e2e', 'nms']

        ```
    """

    def __init__(
        self,
        model: nn.Module,
        e2e_decoder: nn.Module,
        nms_decoder: nn.Module,
        label_to_category: Mapping[int, int],
        letterbox: Letterbox,
        strides: tuple[int, int, int] = HEAD_STRIDES,
        keypoint_sigmas: Sequence[float] = COCO_KEYPOINT_OKS_SIGMAS,
        backend: str = "faster_coco_eval",
    ) -> None:
        if backend not in ("faster_coco_eval", "hotcoco"):
            raise ValueError(f"backend must be one of ('faster_coco_eval', 'hotcoco'), got {backend!r}")
        self._model = model
        self._e2e_decoder = e2e_decoder
        self._nms_decoder = nms_decoder
        self._label_to_category = dict(label_to_category)
        self._letterbox = letterbox
        self._strides = strides
        self._keypoint_sigmas = tuple(keypoint_sigmas)
        #: Box/mask scoring engine. The **library** default is conservative (WP-138):
        #: constructing this class without naming a backend scores with
        #: ``faster_coco_eval``, so every caller and test that predates that WP means
        #: exactly what it meant before. The **CLI** default is not the same thing and
        #: should not be read as one — ``lucid-eval`` defaults ``--eval_backend`` to
        #: ``auto``, which resolves to ``hotcoco`` whenever hotcoco is usable on the
        #: host and records a reason when it falls back
        #: (:func:`~lucid_yolo.cli.eval._resolve_eval_backend`). So the shipped tool's
        #: reports are hotcoco reports on any host that can run hotcoco, and the
        #: engine that produced one is read from the report's own ``eval_backend``
        #: field rather than assumed.
        #:
        #: What licenses that, and what it does not cover.
        #: ``tests/eval/test_coco_eval.py::TestHotcocoParity`` holds the two engines to
        #: the same numbers — ``1e-6`` absolute, on CPU — for the 12 :data:`_METRIC_KEYS`
        #: box statistics, the 24 box+mask statistics, targets carrying their own
        #: ``iscrowd`` and ``area`` through the box pass *and* the mask pass (WP-166),
        #: the never-updated all-zero case, and one end-to-end
        #: :meth:`evaluate` run over a real fixture. Not covered by any comparison:
        #: the per-class entries this module drops from its report
        #: (``map_per_class``, ``mar_100_per_class``, ``classes``), and any device other
        #: than CPU — the parity evidence is CPU-only, so an accelerator run is
        #: unverified rather than known-equal.
        #: OKS keypoint scoring is untouched either way: it stays hand-driven
        #: ``faster_coco_eval`` (A73), the same choice R36 made.
        self._backend = backend

    def evaluate(
        self,
        dataloader: Iterable[tuple[Tensor, Sequence[int], Sequence[tuple[int, int]]]],
        targets: Mapping[int, dict[str, Tensor]],
        device: torch.device,
    ) -> dict[str, dict[str, float]]:
        """Run both paths over ``dataloader`` and return their bbox mAP reports.

        Args:
            dataloader: Iterable of ``(images, image_ids, orig_sizes)`` batches
                (see the class docstring for the batch contract).
            targets: Ground truth keyed by COCO image id, each a torchmetrics
                target dict (``boxes`` ``xyxy`` in original coordinates, ``labels``
                as category ids). Every ``image_id`` yielded by ``dataloader`` must
                be present. A mapping that materialises a target on lookup
                (:class:`~lucid_yolo.eval.annotations.LazyTargets`) is what keeps a
                segmentation run's ground-truth masks off the heap.
            device: Device the model and image batches run on.

        Returns:
            ``{"e2e": stats, "nms": stats}`` where each ``stats`` is the named
            12-metric dict of :func:`evaluate_bbox`, or the 24-entry bbox+segm
            dict of :func:`evaluate_bbox_and_segm` when the model has a mask
            branch.

        Note:
            Each batch is folded into its path's metric as it is produced rather
            than accumulated and scored at the end. For boxes the difference is
            invisible; for masks it is the difference between a bounded footprint
            and hundreds of gigabytes (see :class:`_StreamingScorer`).
        """
        self._model.to(device).eval()
        new_scorer = _HotcocoStreamingScorer if self._backend == "hotcoco" else _StreamingScorer
        e2e_scorer, nms_scorer = new_scorer(), new_scorer()
        e2e_oks, nms_oks = _KeypointScorer(), _KeypointScorer()
        with torch.no_grad():
            for images, image_ids, orig_sizes in dataloader:
                e2e_batch, nms_batch = self._predict_batch(images.to(device), device, orig_sizes)
                gt_batch = [targets[int(image_id)] for image_id in image_ids]
                e2e_scorer.update(e2e_batch, gt_batch)
                nms_scorer.update(nms_batch, gt_batch)
                if e2e_batch and "keypoints" in e2e_batch[0]:
                    e2e_oks.update(e2e_batch, gt_batch, image_ids)
                    nms_oks.update(nms_batch, gt_batch, image_ids)
        return {
            "e2e": e2e_scorer.compute() | e2e_oks.compute(self._keypoint_sigmas),
            "nms": nms_scorer.compute() | nms_oks.compute(self._keypoint_sigmas),
        }

    def _predict_batch(
        self,
        images: Tensor,
        device: torch.device,
        orig_sizes: Sequence[tuple[int, int]],
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        """Forward once, then build both paths' prediction dicts from the shared outputs."""
        head_out, prototypes = _segment_parts(self._model(images))
        anchor_points, strides = anchor_grid((int(images.shape[-2]), int(images.shape[-1])), device, self._strides)
        geometry = _BatchGeometry(
            anchor_points=anchor_points,
            strides=strides,
            canvas=(int(images.shape[-2]), int(images.shape[-1])),
            orig_sizes=orig_sizes,
            prototypes=prototypes,
        )
        e2e = self._decode_path(
            self._e2e_decoder,
            head_out.o2o_cls,
            head_out.o2o_box,
            head_out.o2o_coeff,
            head_out.o2o_keypoints,
            geometry,
        )
        nms = self._decode_path(
            self._nms_decoder,
            head_out.o2m_cls,
            head_out.o2m_box,
            head_out.o2m_coeff,
            head_out.o2m_keypoints,
            geometry,
        )
        return e2e, nms

    def _decode_path(
        self,
        decoder: nn.Module,
        cls_logits: Tensor,
        raw_ltrb: Tensor,
        coefficients: Tensor | None,
        keypoints: Tensor | None,
        geometry: _BatchGeometry,
    ) -> list[dict[str, Tensor]]:
        """Decode one path into prediction dicts, with masks or keypoints when the model has them.

        The per-anchor fields are read from **this** path's own branch — the E2E
        path gets ``o2o_coeff``/``o2o_keypoints`` and the dense path
        ``o2m_coeff``/``o2m_keypoints`` — and gathered by the anchor indices this
        path's own decoder reports. Crossing the two produces a plausible report
        in which every box is right and every pose belongs to the other branch's
        ranking.
        """
        segmenting = geometry.prototypes is not None and coefficients is not None
        if not segmenting and keypoints is None:
            detections = decoder(cls_logits, raw_ltrb, geometry.anchor_points, geometry.strides)
            mapped = self._to_original(detections, geometry.canvas, geometry.orig_sizes)
            return detections_to_predictions(mapped, self._label_to_category)
        indexing = cast("IndexingDecoder", decoder)
        detections, anchor_index = indexing.decode_with_indices(
            cls_logits, raw_ltrb, geometry.anchor_points, geometry.strides
        )
        masks = (
            self._path_masks(detections, anchor_index, coefficients, geometry)
            if geometry.prototypes is not None and coefficients is not None
            else None
        )
        points = None if keypoints is None else self._path_keypoints(keypoints, anchor_index, geometry)
        mapped = self._to_original(detections, geometry.canvas, geometry.orig_sizes)
        return detections_to_predictions(mapped, self._label_to_category, masks=masks, keypoints=points)

    def _path_keypoints(self, keypoints: Tensor, anchor_index: Tensor, geometry: _BatchGeometry) -> Tensor:
        """Decode this path's dense point offsets, gather each detection's set, and land it in original coordinates.

        Decode first, gather second.
        :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints` composes the raw
        offsets with the whole anchor grid and has no per-detection indexing, so
        the order is forced: the dense ``(B, A, K, 2)`` tensor is decoded for the
        batch, and only then is each kept row taken —
        :func:`~lucid_yolo.decode.per_task.gather_keypoint_sets`, the composition
        prediction and export decode through as well.
        """
        gathered = gather_keypoint_sets(keypoints, anchor_index, geometry.anchor_points, geometry.strides)
        return self._points_to_original(gathered, geometry.canvas, geometry.orig_sizes)

    def _path_masks(
        self,
        detections: Tensor,
        anchor_index: Tensor,
        coefficients: Tensor,
        geometry: _BatchGeometry,
    ) -> list[Tensor]:
        """Assemble one path's instance masks and land each image's stack in original coordinates.

        The boxes handed to
        :func:`~lucid_yolo.eval.segment_decode.decode_instance_masks` are the
        letterboxed-frame ones the head predicted, since that is the frame its
        A11 crop and ``image_size`` describe; the inverse letterbox is applied
        afterwards, per image, exactly as the box path applies its own inverse.

        One image is decoded at a time rather than the whole batch at once: the
        intermediate is ``(B, N, H, W)`` at canvas resolution, which for a full
        300-detection batch at 640 px is measured in gigabytes, and each image's
        result lands on its own original grid anyway.

        Only the image's **real** detections are decoded. ``detections`` is the fixed
        ``max_det`` block :func:`~lucid_yolo.decode.common.pad_detections` produces, so
        an image with 25 objects still arrives as 300 rows of which 275 carry a
        score-zero, all-zero box; decoding those costs a full canvas-sized mask each and
        yields nothing, since an empty box crops every pixel away. Both decoders emit
        score-descending, and the padding is appended, so the real rows are a leading
        run and a prefix slice is exact rather than a filter. The stack is padded back to
        ``max_det`` all-``False`` rows before it is returned, because
        :meth:`_image_to_prediction` indexes it with a boolean mask over every row —
        all-``False`` is what the full decode computed for those rows anyway, verified
        rather than assumed. Measured end to end, un-letterboxing included, at 640 px with
        25 real detections in a 300-row block: 183.6 ms per image before, 8.7 ms after.
        Cost here is the per-mask upsample to canvas, which does not care whether a row's
        box is real, so the saving is the padding fraction and nothing else — an image
        whose detections fill the block sees no change, and never a regression.
        """
        assert geometry.prototypes is not None  # narrowed by the caller's guard
        gathered = gather_by_anchor(coefficients, anchor_index)
        padded_rows = detections.shape[1]
        masks: list[Tensor] = []
        for image, orig_size in enumerate(geometry.orig_sizes):
            window = slice(image, image + 1)
            real = int(torch.count_nonzero(detections[image, :, SCORE_COLUMN] > 0.0))
            original = (int(orig_size[0]), int(orig_size[1]))
            if real == 0:
                masks.append(torch.zeros((padded_rows, *original), dtype=torch.bool))
                continue
            canvas_masks = decode_instance_masks(
                geometry.prototypes[window],
                gathered[window, :real],
                detections[window, :real, :BOX_CORNERS],
                image_size=geometry.canvas,
            )
            decoded = masks_to_original(canvas_masks[0].cpu(), self._letterbox, original)
            image_masks = decoded.new_zeros((padded_rows, *decoded.shape[1:]))
            image_masks[:real] = decoded
            masks.append(image_masks)
        return masks

    def _points_to_original(
        self,
        points: Tensor,
        letterboxed_size: tuple[int, int],
        orig_sizes: Sequence[tuple[int, int]],
    ) -> Tensor:
        """Un-letterbox each image's ``(N, K, 2)`` point sets back to its original coordinates (A10).

        One :func:`~lucid_yolo.decode.common.points_to_letterboxed_original` call per
        image, since each image has its own original size — the same point inverse
        :func:`~lucid_yolo.predict.predict_keypoints` lands its single image through, so
        the two cannot disagree by a pad.
        """
        canvas = (int(letterboxed_size[0]), int(letterboxed_size[1]))
        allow_upscale = self._letterbox.allow_upscale
        mapped = [
            points_to_letterboxed_original(
                image_points.cpu(),
                (int(size[0]), int(size[1])),
                canvas,
                allow_upscale=allow_upscale,
            )
            for image_points, size in zip(points, orig_sizes, strict=True)
        ]
        return torch.stack(mapped)

    def _to_original(
        self,
        detections: Tensor,
        letterboxed_size: tuple[int, int],
        orig_sizes: Sequence[tuple[int, int]],
    ) -> Tensor:
        """Un-letterbox each image's detections back to its original coordinates."""
        canvas = (int(letterboxed_size[0]), int(letterboxed_size[1]))
        allow_upscale = self._letterbox.allow_upscale
        mapped = [
            to_letterboxed_original(
                image_detections.unsqueeze(0).cpu(),
                orig_size=(int(size[0]), int(size[1])),
                letterboxed_size=canvas,
                allow_upscale=allow_upscale,
            )
            for image_detections, size in zip(detections, orig_sizes, strict=True)
        ]
        return torch.cat(mapped, dim=0)
