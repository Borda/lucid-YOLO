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

Keypoints (WP-124) add two one-shot functions without introducing a streaming
model evaluator before the pose decode pipeline exists. :func:`keypoints_to_predictions`
filters a fixed-size ``(B, N, K, 2)`` batch and maps contiguous labels back to COCO
category ids; :func:`evaluate_keypoints` turns those tensors and WP-121's ground
truth tensors into an in-memory COCO document, then reports the backend's ten
standard OKS statistics. The citable COCO 17-point sigma table is fixed by
:data:`COCO_KEYPOINT_OKS_SIGMAS` rather than inherited from a dependency default.

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
at *every* recall it can attain, and 20-36% of those are forfeited. :data:`_RECALL_GRID`
supplies the correctly rounded hundredths instead, through the metric's own documented
``rec_thresholds`` argument, which makes this instrument agree exactly with the oriented one
(:func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map`) at the boundaries where they used
to differ. Only the ``map`` family moves; average recall never entered the grid.

Provenance: R1 sec. 3.2.1, R1 Table 7, R1 sec. 4.4, R22, R23. Assumptions: A9, A10, A37, A46.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch
from faster_coco_eval import COCO, COCOeval_faster
from torchmetrics.detection import MeanAveragePrecision

from lucid_yolo.assign.grid import HEAD_STRIDES, anchor_grid
from lucid_yolo.decode.common import BOX_CORNERS, LABEL_COLUMN, SCORE_COLUMN, to_letterboxed_original
from lucid_yolo.eval.segment_decode import decode_instance_masks, masks_to_original
from lucid_yolo.models.build import SegmentOutput

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Tensor, nn

    from lucid_yolo.data.letterbox import Letterbox
    from lucid_yolo.models.heads.detect import DualHeadOutput

__all__ = [
    "COCO_KEYPOINT_OKS_SIGMAS",
    "DualPathEvaluator",
    "detections_to_predictions",
    "evaluate_bbox",
    "evaluate_bbox_and_segm",
    "evaluate_keypoints",
    "evaluate_segm",
    "keypoints_to_predictions",
]

#: The 12 scalar :class:`MeanAveragePrecision` metrics, AP first then AR, in the
#: order torchmetrics reports them (``map`` = mAP50-95). The per-class and
#: ``classes`` entries of the raw compute dict are deliberately excluded.
_METRIC_KEYS: tuple[str, ...] = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
    "mar_small",
    "mar_medium",
    "mar_large",
)

#: Name prefix of the mask statistics in a combined bbox+segm report. The box
#: statistics deliberately keep their bare names, so a detection-only report and
#: the bbox half of a segmentation report are read the same way.
_SEGM_PREFIX = "segm_"

#: Number of interpolated recall points in the COCO protocol: 0.00 to 1.00 by 0.01 (R12).
_RECALL_POINTS = 101

#: The COCO recall grid as **correctly rounded** hundredths (WP-092, A46).
#:
#: Passed to the metric rather than left to its default, because torchmetrics builds this
#: grid with a **float32** ``torch.linspace`` and widens the result to Python floats: at 36
#: of the 101 indices the stored threshold is then strictly greater than the ``k/100`` it
#: stands for. The 65th is ``0.6500000357627869``, against the ``0.65`` that a recall of
#: ``13/20`` attains exactly. A class landing on such a boundary fails the comparison,
#: forfeits that point, and loses ``1/101`` of its average precision — a bias that is
#: small, one-directional and always downward.
#:
#: Evaluating ``k/100`` in float64 removes that bias outright rather than tightening it,
#: and the reason is arithmetic rather than empirical. IEEE division is correctly rounded,
#: so ``hits/positives`` and ``k/100`` land on the *same* double whenever they are equal as
#: rationals: the equality case — the only one floating point was getting wrong — is then
#: decided exactly, with no epsilon to choose. The unequal cases stay exact too at every
#: scale this instrument sees, since two distinct rationals with denominators at most ``Q``
#: differ by at least ``1/(100 Q)`` while doubles near 1 resolve ``2**-52``; the comparison
#: is settled by the values rather than by rounding for ``Q`` up to roughly ``4.5e13``,
#: against COCO classes whose ground-truth counts are measured in tens of thousands.
#:
#: This does not make the axis-aligned instrument bit-identical to
#: :func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map`, which decides the same question
#: in int64 and is exact for *any* ``Q``; it makes the two agree on every input either can
#: be given in practice. ``tests/eval/test_coco_eval.py::TestRecallGridBoundary`` pins the
#: agreement at the boundaries, and the 36-index defect itself, so a torchmetrics that
#: fixes its own grid is noticed rather than silently worked around forever.
_RECALL_GRID: tuple[float, ...] = tuple(index / (_RECALL_POINTS - 1) for index in range(_RECALL_POINTS))

#: The COCO 17-point OKS per-keypoint sigmas (R12), in COCO's own point order
#: (nose, l/r eye, l/r ear, l/r shoulder, l/r elbow, l/r wrist, l/r hip, l/r
#: knee, l/r ankle). Named explicitly here rather than left to
#: faster_coco_eval's internal default (Params.setKpParams) so the value this
#: project reports against is citable and version-independent, even though it
#: currently equals that library default exactly -- verify this against
#: `faster_coco_eval.core.cocoeval.Params.setKpParams`'s source before trusting
#: this comment; if the library ever changes its own default, this project's
#: reported numbers must not silently move with it.
COCO_KEYPOINT_OKS_SIGMAS: tuple[float, ...] = (
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
)

#: The ten scalar statistics in COCO's keypoint protocol. Unlike bbox/segm,
#: keypoints have medium and large area buckets only, with no small bucket.
_KEYPOINT_METRIC_KEYS: tuple[str, ...] = (
    "AP_all",
    "AP_50",
    "AP_75",
    "AP_medium",
    "AP_large",
    "AR_all",
    "AR_50",
    "AR_75",
    "AR_medium",
    "AR_large",
)


def detections_to_predictions(
    detections: Tensor,
    label_to_category: Mapping[int, int],
    score_floor: float = 0.0,
    masks: Sequence[Tensor] | None = None,
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

    Returns:
        A length-``B`` list of prediction dicts, each with ``boxes`` (``(M, 4)``
        ``xyxy``), ``scores`` (``(M,)``) and ``labels`` (``(M,)`` long, category
        ids) — the per-image shape :class:`MeanAveragePrecision` consumes — plus
        ``masks`` (``(M, H, W)`` bool) when ``masks`` is given.

    Examples:
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
    """
    dense = detections.detach().to(device="cpu", dtype=torch.float32)
    if masks is None:
        return [_image_to_prediction(image, label_to_category, score_floor) for image in dense]
    return [
        _image_to_prediction(image, label_to_category, score_floor, image_masks)
        for image, image_masks in zip(dense, masks, strict=True)
    ]


def keypoints_to_predictions(
    keypoints: Tensor,
    scores: Tensor,
    labels: Tensor,
    label_to_category: Mapping[int, int],
    score_floor: float = 0.0,
) -> list[dict[str, Tensor]]:
    """Convert fixed-size keypoint batches into COCO prediction dicts.

    Each row of ``keypoints`` is one instance's ``K`` absolute-pixel ``(x, y)``
    coordinates, such as the output of
    :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints`. A row survives
    when its per-instance detection score is strictly above ``score_floor`` — at
    the default ``0.0``, this drops the score-zero padding rows that fill a fixed
    decoder shape. Each contiguous class label is mapped **back** to its original
    COCO category id via ``label_to_category``.

    Args:
        keypoints: Absolute pixel coordinates of shape ``(B, N, K, 2)``.
        scores: Per-instance detection confidence of shape ``(B, N)``.
        labels: Contiguous class labels of shape ``(B, N)``.
        label_to_category: Mapping from contiguous class label to original COCO
            category id.
        score_floor: Rows with ``score`` at or below this are dropped. Defaults to
            ``0.0`` (drop only score-zero padding rows).

    Returns:
        A length-``B`` list of prediction dicts, each with ``keypoints``
        (``(M, K, 2)`` float), ``scores`` (``(M,)``), and ``labels`` (``(M,)``
        long, original COCO category ids).

    Examples:
        >>> import torch
        >>> points = torch.tensor(
        ...     [[[[10.0, 12.0], [20.0, 22.0]], [[0.0, 0.0], [0.0, 0.0]]]]
        ... )  # one real instance, one padding row
        >>> scores = torch.tensor([[0.9, 0.0]])
        >>> labels = torch.tensor([[1, 0]])
        >>> preds = keypoints_to_predictions(points, scores, labels, {1: 42, 0: 1})
        >>> tuple(preds[0]["keypoints"].shape), preds[0]["labels"].tolist()
        ((1, 2, 2), [42])
    """
    dense_keypoints = keypoints.detach().to(device="cpu", dtype=torch.float32)
    dense_scores = scores.detach().to(device="cpu", dtype=torch.float32)
    dense_labels = labels.detach().to(device="cpu")
    predictions: list[dict[str, Tensor]] = []
    for image_keypoints, image_scores, image_labels in zip(dense_keypoints, dense_scores, dense_labels, strict=True):
        keep = image_scores > score_floor
        mapped_labels = torch.tensor(
            [label_to_category[int(label)] for label in image_labels[keep]],
            dtype=torch.long,
        )
        predictions.append(
            {
                "keypoints": image_keypoints[keep],
                "scores": image_scores[keep],
                "labels": mapped_labels,
            }
        )
    return predictions


def _image_to_prediction(
    detections: Tensor,
    label_to_category: Mapping[int, int],
    score_floor: float,
    masks: Tensor | None = None,
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
        >>> import torch
        >>> box = torch.tensor([[10.0, 12.0, 30.0, 42.0]])
        >>> preds = [{"boxes": box, "scores": torch.tensor([1.0]), "labels": torch.tensor([5])}]
        >>> targets = [{"boxes": box, "labels": torch.tensor([5])}]
        >>> stats = evaluate_bbox(preds, targets)
        >>> round(stats["map"], 3), round(stats["map_50"], 3)
        (1.0, 1.0)
    """
    if not preds:
        return dict.fromkeys(_METRIC_KEYS, 0.0)
    return _named_stats(_compute_metric(preds, targets, "bbox"), combined=False)


def evaluate_keypoints(
    preds: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    sigmas: Sequence[float] = COCO_KEYPOINT_OKS_SIGMAS,
) -> dict[str, float]:
    """Score keypoint predictions with COCO's object keypoint similarity protocol.

    Builds the minimal in-memory COCO ground-truth and result documents needed by
    :class:`faster_coco_eval.COCOeval_faster`, then runs its native keypoint path.
    Ground-truth ``visibility`` follows WP-121's tensor convention directly:
    ``v == 0`` points are unlabeled and excluded from OKS, while ``v > 0`` points
    define both the match distances and each instance's tight bounding-box area.
    Predicted visibility is the COCO placeholder ``2`` because OKS reads only the
    predicted coordinates and ground-truth visibility. An empty prediction input
    yields all ten statistics at ``0.0`` rather than entering a degenerate backend
    path.

    Args:
        preds: Per-image prediction dicts with ``keypoints`` (``(M, K, 2)``),
            ``scores`` (``(M,)``), and ``labels`` (``(M,)`` original COCO category
            ids), as produced by :func:`keypoints_to_predictions`.
        targets: Position-aligned ground-truth dicts with ``keypoints``
            (``(M, K, 2)``), ``visibility`` (``(M, K)`` int64), and ``labels``
            (``(M,)`` original COCO category ids).
        sigmas: Per-keypoint OKS sigmas in the tensors' point order. Defaults to
            :data:`COCO_KEYPOINT_OKS_SIGMAS`, COCO's 17-point person convention.

    Returns:
        A ten-entry dict containing ``AP_all``, ``AP_50``, ``AP_75``, the medium
        and large AP buckets, and the corresponding five ``AR_*`` statistics.
        COCO keypoint evaluation defines no small-area bucket.

    Examples:
        >>> import torch
        >>> points = torch.tensor([[[10.0, 10.0], [50.0, 50.0]]])
        >>> preds = [{"keypoints": points, "scores": torch.tensor([1.0]), "labels": torch.tensor([1])}]
        >>> targets = [{"keypoints": points, "visibility": torch.tensor([[2, 2]]), "labels": torch.tensor([1])}]
        >>> stats = evaluate_keypoints(preds, targets, sigmas=[1.0, 1.0])
        >>> round(stats["AP_all"], 3), round(stats["AP_50"], 3)
        (1.0, 1.0)
    """
    if not preds:
        return dict.fromkeys(_KEYPOINT_METRIC_KEYS, 0.0)

    images = [{"id": image_id} for image_id in range(len(preds))]
    annotations: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    category_ids: set[int] = set()
    annotation_id = 1

    for image_id, (prediction, target) in enumerate(zip(preds, targets, strict=True)):
        target_keypoints = target["keypoints"].detach().to(device="cpu", dtype=torch.float32)
        target_visibility = target["visibility"].detach().to(device="cpu", dtype=torch.long)
        target_labels = target["labels"].detach().to(device="cpu", dtype=torch.long)
        for instance_keypoints, instance_visibility, instance_label in zip(
            target_keypoints, target_visibility, target_labels, strict=True
        ):
            visible = instance_visibility > 0
            if bool(visible.any()):
                visible_points = instance_keypoints[visible]
                minimum = visible_points.amin(dim=0)
                extent = visible_points.amax(dim=0) - minimum
                width, height = float(extent[0]), float(extent[1])
                x_min, y_min = float(minimum[0]), float(minimum[1])
                area = max(width * height, 1.0)
                bbox = [x_min, y_min, width, height]
            else:
                area = 1.0
                bbox = [0.0, 0.0, 1.0, 1.0]

            category_id = int(instance_label)
            category_ids.add(category_id)
            flat_keypoints = torch.cat(
                (instance_keypoints, instance_visibility.to(torch.float32).unsqueeze(-1)),
                dim=-1,
            ).reshape(-1)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "keypoints": flat_keypoints.tolist(),
                    "num_keypoints": int(visible.sum()),
                    "iscrowd": 0,
                    "area": area,
                    "bbox": bbox,
                }
            )
            annotation_id += 1

        prediction_keypoints = prediction["keypoints"].detach().to(device="cpu", dtype=torch.float32)
        prediction_scores = prediction["scores"].detach().to(device="cpu", dtype=torch.float32)
        prediction_labels = prediction["labels"].detach().to(device="cpu", dtype=torch.long)
        for instance_keypoints, instance_score, instance_label in zip(
            prediction_keypoints, prediction_scores, prediction_labels, strict=True
        ):
            category_id = int(instance_label)
            category_ids.add(category_id)
            visibility = torch.full(
                (instance_keypoints.shape[0], 1),
                2.0,
                dtype=instance_keypoints.dtype,
            )
            flat_keypoints = torch.cat((instance_keypoints, visibility), dim=-1).reshape(-1)
            results.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "keypoints": flat_keypoints.tolist(),
                    "score": float(instance_score),
                }
            )

    if not results:
        return dict.fromkeys(_KEYPOINT_METRIC_KEYS, 0.0)

    annotation_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_id, "name": str(category_id)} for category_id in sorted(category_ids)],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_ground_truth = COCO(annotation_dict)
        coco_predictions = coco_ground_truth.loadRes(results)
        evaluator = COCOeval_faster(
            coco_ground_truth,
            coco_predictions,
            iouType="keypoints",
            kpt_oks_sigmas=list(sigmas),
        )
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return cast("dict[str, float]", evaluator.stats_as_dict)


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
        >>> import torch
        >>> box = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
        >>> mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        >>> mask[0, 1:3, 1:3] = True
        >>> preds = [{"boxes": box, "scores": torch.tensor([1.0]), "labels": torch.tensor([5]), "masks": mask}]
        >>> targets = [{"boxes": box, "labels": torch.tensor([5]), "masks": mask}]
        >>> round(evaluate_segm(preds, targets)["map"], 3)
        1.0
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
        >>> import torch
        >>> box = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
        >>> mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        >>> mask[0, 1:3, 1:3] = True
        >>> preds = [{"boxes": box, "scores": torch.tensor([1.0]), "labels": torch.tensor([5]), "masks": mask}]
        >>> targets = [{"boxes": box, "labels": torch.tensor([5]), "masks": mask}]
        >>> stats = evaluate_bbox_and_segm(preds, targets)
        >>> round(stats["map"], 3), round(stats["segm_map"], 3)
        (1.0, 1.0)
    """
    if not preds:
        return dict.fromkeys([*_METRIC_KEYS, *(_SEGM_PREFIX + key for key in _METRIC_KEYS)], 0.0)
    return _named_stats(_compute_metric(preds, targets, ("bbox", "segm")), combined=True)


def _new_metric(iou_type: str | tuple[str, ...]) -> MeanAveragePrecision:
    """Construct the metric for one ``iou_type``, on the exact recall grid.

    The single place the metric is configured, so the backend, the box format, the
    recall grid and the detection-cap warning setting cannot differ between the
    one-shot entry points and the streaming one :class:`DualPathEvaluator` drives.
    Passing :data:`_RECALL_GRID` here is therefore what gives ``bbox``, ``segm``,
    the combined pass and the streaming path one definition of average precision
    rather than four.

    Why this route, and not the other two the work package weighed:
        ``rec_thresholds`` is a **documented constructor parameter** of
        :class:`~torchmetrics.detection.MeanAveragePrecision` (1.9), forwarded
        verbatim to the backend as ``params.recThrs`` in float64 — not a private
        attribute reached into after construction. The alternative of rebuilding
        average precision from the backend's raw matching would replace one
        supported keyword with a reimplementation of all twelve statistics against
        genuinely undocumented internals, to arrive at the same numbers; and
        carrying the bias in the report instead would be choosing to publish a
        known-low figure while the correction sat behind a public argument.

    What happens when this stops being true:
        Removed parameter — ``MeanAveragePrecision`` raises ``TypeError`` at
        construction, loudly, on its own. Accepted-but-ignored parameter — the
        quiet regression, which would silently restore the downward bias — is
        caught by the check below. Stored-but-unused is caught in the gate by
        ``tests/eval/test_coco_eval.py::TestRecallGridBoundary``, which asserts the
        sampled values rather than the configuration. None of the three can reach a
        report as a slightly-too-low mAP.
    """
    metric = MeanAveragePrecision(
        backend="faster_coco_eval",
        box_format="xyxy",
        iou_type=iou_type,  # type: ignore[arg-type]
        rec_thresholds=list(_RECALL_GRID),
    )
    if tuple(metric.rec_thresholds) != _RECALL_GRID:
        raise RuntimeError(
            "MeanAveragePrecision did not honour the rec_thresholds it was given, so its "
            "recall grid is not the exact one this module requires (A46). Refusing to "
            f"report a silently biased mAP. Expected {_RECALL_POINTS} exact hundredths, "
            f"got {list(metric.rec_thresholds)[:3]}... — pin torchmetrics to a version "
            "whose rec_thresholds argument is honoured, or rework _new_metric."
        )
    # The fixed 300-row decoder output routinely exceeds COCO's top-100 detection
    # cap; keeping only the 100 highest-scoring per image is the standard protocol
    # (COCOeval's maxDets), not a misconfiguration, so silence the per-call warning.
    metric.warn_on_many_detections = False
    return metric


def _named_stats(computed: Mapping[str, Tensor], combined: bool) -> dict[str, float]:
    """Rename one ``compute()`` dict onto this module's reported statistic names.

    torchmetrics names its outputs ``map``/``mar_*`` for a single ``iou_type`` but
    ``bbox_map``/``segm_map`` for a tuple of them. Both readings live here so the
    one-shot and streaming paths cannot report the same run under different names.
    """
    if not combined:
        return {key: float(computed[key]) for key in _METRIC_KEYS}
    stats = {key: float(computed[f"bbox_{key}"]) for key in _METRIC_KEYS}
    stats.update({_SEGM_PREFIX + key: float(computed[_SEGM_PREFIX + key]) for key in _METRIC_KEYS})
    return stats


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


class _StreamingScorer:
    """One decode path's metric, updated batch by batch rather than at the end.

    Accumulating every image's predictions and targets and scoring once is fine
    for boxes and impossible for masks. A segmentation eval of val2017 would hold
    both paths' predicted masks *and* the ground-truth masks dense and
    simultaneously — hundreds of gigabytes at original resolution. Updating per
    batch bounds that to one batch, because ``MeanAveragePrecision.update``
    RLE-encodes each mask into its state immediately (torchmetrics 1.9,
    ``_get_safe_item_values``); the dense tensors are free the moment the batch is
    scored.

    The ``iou_type`` is decided by the first batch, from whether its predictions
    carry masks, and the metric is constructed only then: a run that yields no
    batches at all reports the same all-zero dict the one-shot entry points do,
    rather than raising from a metric that was never updated.
    """

    def __init__(self) -> None:
        self._metric: MeanAveragePrecision | None = None
        self._combined = False

    def update(self, preds: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]) -> None:
        """Fold one batch into the metric state, constructing it on the first call."""
        if not preds:
            return
        if self._metric is None:
            self._combined = "masks" in preds[0]
            self._metric = _new_metric(("bbox", "segm") if self._combined else "bbox")
        with contextlib.redirect_stdout(io.StringIO()):
            self._metric.update(preds, targets)

    def compute(self) -> dict[str, float]:
        """Return the named statistics, or an all-zero dict if nothing was scored."""
        if self._metric is None:
            return dict.fromkeys(_METRIC_KEYS, 0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            computed: Mapping[str, Tensor] = self._metric.compute()
        return _named_stats(computed, self._combined)


class _IndexingDecoder(Protocol):
    """A decoder that reports the source anchor of every detection it emits.

    The interface :class:`DualPathEvaluator` needs from both decoders to evaluate
    a segmentation model — stated structurally rather than by naming the two
    concrete classes, so a caller may substitute its own decoder as the
    detection-only path already allows.
    """

    def decode_with_indices(
        self,
        cls_logits: Tensor,
        raw_ltrb: Tensor,
        anchor_points: Tensor,
        strides: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the A9 detections and the anchor index of each of their rows."""
        ...


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


def _gather_coefficients(coefficients: Tensor, anchor_index: Tensor) -> Tensor:
    """Select each detection's mask coefficients by its own source anchor index.

    ``anchor_index`` comes from the decoder that produced the detections, so row
    ``n`` of the result is the coefficient row of the anchor row ``n``'s box came
    from. Padding rows carry
    :data:`~lucid_yolo.decode.common.PAD_ANCHOR_INDEX`; they are clamped to a
    valid gather position and then zeroed, which yields an all-zero coefficient
    vector, a mask logit of exactly 0, a probability of 0.5, and therefore an
    empty mask under the strictly-greater A37 threshold. Those rows also carry
    score 0 and are dropped by :func:`detections_to_predictions` regardless.
    """
    real = anchor_index >= 0
    index = anchor_index.clamp(min=0).unsqueeze(-1).expand(-1, -1, coefficients.shape[-1])
    gathered = coefficients.gather(1, index)
    return gathered * real.unsqueeze(-1).to(gathered.dtype)


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

    Examples:
        >>> evaluator = DualPathEvaluator(  # doctest: +SKIP
        ...     model, TopKDecoder(), NMSDecoder(), dataset.label_to_category_id, letterbox
        ... )
        >>> report = evaluator.evaluate(val_loader, targets, torch.device("cpu"))  # doctest: +SKIP
        >>> sorted(report)  # doctest: +SKIP
        ['e2e', 'nms']
    """

    def __init__(
        self,
        model: nn.Module,
        e2e_decoder: nn.Module,
        nms_decoder: nn.Module,
        label_to_category: Mapping[int, int],
        letterbox: Letterbox,
        strides: tuple[int, int, int] = HEAD_STRIDES,
    ) -> None:
        self._model = model
        self._e2e_decoder = e2e_decoder
        self._nms_decoder = nms_decoder
        self._label_to_category = dict(label_to_category)
        self._letterbox = letterbox
        self._strides = strides

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
        e2e_scorer, nms_scorer = _StreamingScorer(), _StreamingScorer()
        with torch.no_grad():
            for images, image_ids, orig_sizes in dataloader:
                e2e_batch, nms_batch = self._predict_batch(images.to(device), device, orig_sizes)
                gt_batch = [targets[int(image_id)] for image_id in image_ids]
                e2e_scorer.update(e2e_batch, gt_batch)
                nms_scorer.update(nms_batch, gt_batch)
        return {"e2e": e2e_scorer.compute(), "nms": nms_scorer.compute()}

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
        e2e = self._decode_path(self._e2e_decoder, head_out.o2o_cls, head_out.o2o_box, head_out.o2o_coeff, geometry)
        nms = self._decode_path(self._nms_decoder, head_out.o2m_cls, head_out.o2m_box, head_out.o2m_coeff, geometry)
        return e2e, nms

    def _decode_path(
        self,
        decoder: nn.Module,
        cls_logits: Tensor,
        raw_ltrb: Tensor,
        coefficients: Tensor | None,
        geometry: _BatchGeometry,
    ) -> list[dict[str, Tensor]]:
        """Decode one path into prediction dicts, with masks when the model has them."""
        if geometry.prototypes is None or coefficients is None:
            detections = decoder(cls_logits, raw_ltrb, geometry.anchor_points, geometry.strides)
            mapped = self._to_original(detections, geometry.canvas, geometry.orig_sizes)
            return detections_to_predictions(mapped, self._label_to_category)
        indexing = cast("_IndexingDecoder", decoder)
        detections, anchor_index = indexing.decode_with_indices(
            cls_logits, raw_ltrb, geometry.anchor_points, geometry.strides
        )
        masks = self._path_masks(detections, anchor_index, coefficients, geometry)
        mapped = self._to_original(detections, geometry.canvas, geometry.orig_sizes)
        return detections_to_predictions(mapped, self._label_to_category, masks=masks)

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
        """
        assert geometry.prototypes is not None  # narrowed by the caller's guard
        gathered = _gather_coefficients(coefficients, anchor_index)
        masks: list[Tensor] = []
        for image, orig_size in enumerate(geometry.orig_sizes):
            window = slice(image, image + 1)
            canvas_masks = decode_instance_masks(
                geometry.prototypes[window],
                gathered[window],
                detections[window, :, :BOX_CORNERS],
                image_size=geometry.canvas,
            )
            original = (int(orig_size[0]), int(orig_size[1]))
            masks.append(masks_to_original(canvas_masks[0].cpu(), self._letterbox, original))
        return masks

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
