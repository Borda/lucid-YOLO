# SPDX-License-Identifier: Apache-2.0
"""Rotated mAP50-95 for the DOTA-v1.0 validation protocol (WP-063, A24).

The oriented acceptance instrument of the blueprint: rotated mAP50-95 on the DOTA-v1.0
**validation** split, scored with **exact polygon-intersection IoU** rather than a
Gaussian surrogate. :mod:`lucid_yolo.losses.probiou` approximates a rotated box by its
uniform-density Gaussian because a loss needs a smooth gradient; a metric needs the area
itself, so this module computes it — Sutherland-Hodgman convex clipping of one
quadrilateral against the other, then the shoelace formula.

Three pieces compose the protocol, mirroring the axis-aligned
:mod:`lucid_yolo.eval.coco_eval`:

- :func:`rotated_iou` — the exact pairwise overlap kernel.
- :func:`evaluate_rotated_map` — the COCO-style accumulator running on that kernel.
- :func:`rotated_detections_to_predictions` and :func:`tiled_targets_to_ground_truth` —
  the adapters from the A45 oriented detection tuple and from WP-057's tiled targets.

Scope — what this module is **not** (WP-107):
    It scores predictions against ground truth **as supplied**, one evaluation unit at a
    time. It does **not** merge detections from the overlapping 1024 px tiles of
    :func:`~lucid_yolo.data.tiling.tile_windows` back onto whole DOTA images. That step is
    genuinely underspecified on an NMS-free path — two overlapping tiles both detect the
    same object at full confidence and there is no suppression stage to remove the
    duplicate — so it needs a policy decided rather than inherited. WP-063 deferred it to
    WP-088, whose scope never took it up, and 0.3.0 shipped without it; it now lives in
    :mod:`lucid_yolo.eval.tile_merge` (WP-107), which states the rule and argues it. What
    this module scores is whatever evaluation unit it is handed: hand it tiles and the
    honest reading is **per-tile** mAP, hand it merged source images and it is whole-image
    mAP, and the two are not interchangeable — a tile-level score never pays the
    duplicate-detection cost that whole-image evaluation charges.

Protocol constants, fixed here as decisions:
    R1 states the evaluation target ("rotated mAP50-95 on DOTA-v1.0 val", Tables 10-11)
    but settles none of the five constants below. Each is pinned in code and named where
    it lives, so a reader learns the protocol from the module rather than from a report.

    1. **Ten IoU thresholds**, 0.50 to 0.95 in steps of 0.05 (:data:`IOU_THRESHOLDS`).
    2. **101-point interpolated recall** (:data:`RECALL_POINTS`), the COCO convention.
       Chosen so this instrument and the axis-aligned one — which is COCOeval-faithful
       through its ``faster_coco_eval`` backend — define mAP the same way, and a reader
       comparing the two reports is comparing numbers rather than definitions. Whether
       they actually did depended on the dtype each grid was built in, because sampling
       "has recall reached ``k/100``" is an equality test that rounding can settle::

           torch.linspace float32  (torchmetrics default, no longer used)  36 missed
           numpy.linspace float64  (raw pycocotools / faster_coco_eval)    10 missed
           exact integers          (this module)                            0 missed

       A "missed" boundary is one where the stored grid value exceeds the correctly
       rounded ``k/100``, so a class whose recall reaches exactly ``k/100`` fails the
       comparison and forfeits that point — ``1/101`` of that class's AP, always
       downward. Until WP-092 that made ``evaluate_bbox`` the lower of the two at 36
       boundaries; it now supplies the metric an exact grid through its documented
       ``rec_thresholds`` argument, so the axis-aligned instrument forfeits none either
       and the two agree at every boundary they can reach. They arrive there by
       different arithmetic — int64 rationals here, correctly rounded float64 division
       there — so the agreement is a measured property rather than a shared code path,
       which is why both sides pin it. :func:`_average_precision` documents the
       arithmetic; ``tests/eval/test_dota_eval.py`` pins the exact values and the
       agreement, and ``tests/eval/test_coco_eval.py`` pins the dependency defect so
       that a torchmetrics which fixes its own grid is noticed rather than worked
       around forever.
    3. **maxDets = 300** (:data:`MAX_DETECTIONS`), not COCO's 100. The oriented output
       caps at 300 detections per image
       (:func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk`), so a 100 cap would
       silently discard two thirds of what the model emits and report the remainder as
       the model's recall.
    4. **Classes with zero ground truths are excluded** from the mean, the COCO
       convention, rather than scored as zero — and "zero" counts *non-difficult* ground
       truths, since a class whose instances are all difficult has an empty recall
       denominator and averaging a 0/0 into the mean would be arithmetic, not a
       measurement.
    5. **Difficult ground truths follow R18's devkit convention** (VOC-style): a detection
       matched to a difficult ground truth is neither a true positive nor a false positive
       — it is discarded — and difficult instances do not count towards the recall
       denominator. See the difficult-flag note below; this is load-bearing, not
       decorative.

The difficult flag, and what the caller must do about it:
    :func:`~lucid_yolo.data.tiling.crop_targets` flags every instance clipped below
    :data:`~lucid_yolo.data.tiling.DIFFICULT_AREA_FRACTION` of its original area as
    difficult, so real DOTA val tiles carry difficult instances whether or not the source
    label file did. The evaluation loader must therefore read labels with
    ``keep_difficult=True`` (:func:`~lucid_yolo.data.dota.load_dota_targets`) and pass the
    flags through: dropping difficult instances at load does not make them neutral, it
    converts every detection that finds one into a **false positive** against a ground
    truth that was silently deleted.

    The matching rule implemented here prefers a non-difficult match: a detection takes
    the best *unmatched non-difficult* ground truth at or above the threshold; failing
    that, an overlap with any difficult ground truth discards it; failing that it is a
    false positive. The literal VOC devkit instead takes the argmax over *all* ground
    truths and only then inspects the flag, so the two differ in one corner — a difficult
    instance overlapping a detection more than a matchable non-difficult one shadows it,
    costing a true positive. Difficult instances are never consumed, so several detections
    may be discarded against the same one, which is the devkit's behaviour.

Precision, and why there is no float64 here:
    :mod:`lucid_yolo.data.tiling` clips in float64 because DOTA coordinates reach 10^4 px
    and a float32 shoelace difference loses the precision its 0.7 threshold is compared
    at. That escape is not available on this path: evaluation runs on MPS, which has no
    float64 — the same constraint :mod:`lucid_yolo.losses.probiou` restructured its algebra
    for. The working dtype here follows the input instead, and the conditioning is fixed
    structurally: intersection over union is **translation invariant**, so every pair is
    re-centred on its own midpoint and only then expanded to corners. A pair of 50 px
    boxes at ``x = 12000`` is clipped at coordinates near zero rather than near 12000,
    which is where a float32 shoelace has its precision, and the remaining operations are
    all like-signed sums over small numbers.

    The order of those two steps is the whole of it, and it is not a detail. Worst
    absolute IoU error over 300 overlapping 20-80 px pairs per row, measured against a
    float64 shapely evaluation, as the pair's common offset from the origin grows::

        common offset   no shift at all   shift after   shift before (shipped)
        0               1.5e-07           2.2e-07       1.4e-07
        1e3             5.1e-05           1.8e-06       1.7e-07
        1e4             6.3e-03           2.7e-05       1.6e-07
        1e5             1.7e+00           1.4e-04       1.4e-07
        1e6             1.5e+00           1.4e-03       1.4e-07

    Shifting *after* the corners are expanded still leaves them rounded at absolute scale
    — a cliff that merely starts later. Shifting *before* makes the kernel scale-free, at
    the cost of canonicalizing ``M * N`` boxes rather than ``M + N``. That price is worth
    paying now rather than later: tiles are 1024 px local, so nothing in this work package
    exercises the cliff, but WP-064 evaluates on whole DOTA images whose coordinates reach
    10^4, and a metric that quietly loses three digits at that scale would be found by
    nobody.

Provenance: R1 Tables 10-11, R12 (the 101-point and zero-GT conventions), R13, R18 sec. 4
and its devkit. Assumptions: A21, A23, A24, A39, A45.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lucid_yolo.data.rotated_geom import rboxes_to_polygons

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lucid_yolo.data.tiling import TiledTargets

__all__ = [
    "IOU_THRESHOLDS",
    "MAX_DETECTIONS",
    "RECALL_POINTS",
    "evaluate_rotated_map",
    "rotated_detections_to_predictions",
    "rotated_iou",
    "tiled_targets_to_ground_truth",
]

#: The ten IoU thresholds of the protocol, 0.50 to 0.95 in steps of 0.05 (decision 1).
#: Built by arithmetic rather than written out so the step is the definition.
IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * step, 2) for step in range(10))

#: Recall sample count of the interpolated average precision (decision 2): the COCO
#: 101-point grid ``0.00, 0.01, ..., 1.00``, matching :mod:`lucid_yolo.eval.coco_eval`.
RECALL_POINTS = 101

#: Detections scored per image (decision 3). 300, not COCO's 100, because
#: :func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk` emits 300 and a lower cap would
#: measure a truncation rather than the model.
MAX_DETECTIONS = 300

#: The four summary statistics. ``map``/``map_50``/``map_75`` carry the names
#: :mod:`lucid_yolo.eval.coco_eval` gives the same quantities, so the oriented and
#: axis-aligned reports share one vocabulary where they overlap. ``mar_300`` is the
#: honest analogue of its ``mar_100`` — the cap differs, so the name does too. The
#: ``*_small``/``*_medium``/``*_large`` breakdown is deliberately absent: those buckets
#: are COCO area constants (R12), and this protocol does not define area ranges for
#: oriented boxes, so reporting them would be inventing a threshold rather than citing one.
_METRIC_KEYS: tuple[str, ...] = ("map", "map_50", "map_75", "mar_300")

#: Index of the 0.50 and 0.75 thresholds within :data:`IOU_THRESHOLDS`.
_IOU_50_INDEX = 0
_IOU_75_INDEX = 5

#: Column count of a long-edge rotated box ``(cx, cy, w, h, theta)``.
_RBOX_COLUMNS = 5
#: Column index of the score within the A45 oriented detection tuple.
_SCORE_COLUMN = 5
#: Column index of the integral class label within the A45 oriented detection tuple.
_LABEL_COLUMN = 6
#: Width of the A45 oriented detection tuple ``[cx, cy, w, h, theta, score, class]``.
_DETECTION_WIDTH = 7

#: Corner count of the quadrilateral form of a rotated box.
_QUAD_CORNERS = 4
#: Vertex bound on the intersection of two convex quadrilaterals. Clipping a 4-gon by
#: ``j`` half-planes yields at most ``4 + j`` vertices, so 8 bounds every intermediate
#: stage of the four-edge pass as well as its result.
_MAX_INTERSECTION_CORNERS = 8
#: Minimum vertex count for a polygon to enclose any area.
_MIN_AREA_CORNERS = 3


def rotated_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Compute exact pairwise intersection-over-union between rotated boxes.

    Each box is expanded to its four corners by
    :func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons` — which canonicalizes first,
    so ``theta`` and ``theta + pi`` give identical overlaps — and every pair's
    intersection is obtained by clipping one quadrilateral against the other's four edges
    (Sutherland-Hodgman, valid because both are convex) and taking the shoelace area of
    what survives. No sampling, no Gaussian surrogate, no float64: see the module
    docstring on the per-pair midpoint shift that makes float32 sufficient.

    Conventions and degenerate cases:

    - Clipping is **edge-inclusive** (a vertex exactly on a clip edge counts as inside),
      matching :func:`~lucid_yolo.data.rotated_geom.points_in_rboxes` and WP-055's
      containment choice. Boxes sharing only an edge or a corner therefore intersect in a
      zero-area polygon and score exactly ``0.0`` either way — inclusivity changes which
      vertices are kept, never the area.
    - A box with zero or negative extent encloses no area. Its polygon has zero or
      reversed winding, both of which clamp to zero area, so every IoU involving it is
      ``0.0``. The union is guarded against division by zero, so a degenerate pair yields
      ``0.0`` rather than ``NaN`` — the same refusal :mod:`lucid_yolo.losses.probiou`
      makes.

    Args:
        boxes_a: ``(M, 5)`` rotated boxes ``(cx, cy, w, h, theta)``, canonical or not.
        boxes_b: ``(N, 5)`` rotated boxes in the same form.

    Returns:
        ``(M, N)`` IoU in ``[0, 1]``, in the dtype
        :func:`torch.result_type` gives the two inputs.

    Raises:
        ValueError: If either argument is not a 2-D ``(K, 5)`` tensor.

    Examples:
        >>> import torch
        >>> box = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]])
        >>> float(rotated_iou(box, box))  # a box against itself
        1.0
        >>> shifted = torch.tensor([[2.0, 0.0, 4.0, 2.0, 0.0]])  # half its width along +x
        >>> round(float(rotated_iou(box, shifted)), 4)
        0.3333
        >>> touching = torch.tensor([[4.0, 0.0, 4.0, 2.0, 0.0]])  # shares one edge only
        >>> float(rotated_iou(box, touching))
        0.0
        >>> square = torch.tensor([[0.0, 0.0, 3.0, 3.0, 0.2]])
        >>> turned = torch.tensor([[0.0, 0.0, 3.0, 3.0, 0.2 + torch.pi / 2]])
        >>> round(float(rotated_iou(square, turned)), 5)  # the same square, folded
        1.0
    """
    _check_rboxes(boxes_a, "boxes_a")
    _check_rboxes(boxes_b, "boxes_b")
    dtype = torch.result_type(boxes_a, boxes_b)
    left, right = boxes_a.to(dtype), boxes_b.to(dtype)
    if left.shape[0] == 0 or right.shape[0] == 0:
        return torch.zeros((left.shape[0], right.shape[0]), dtype=dtype, device=left.device)

    # Translation invariance is what buys float32 the headroom float64 would otherwise be
    # needed for: each pair is re-centred on its own midpoint *before* its corners are
    # expanded, so no coordinate in the clipping arithmetic ever carries the absolute
    # offset. Re-centring after the expansion would leave the corners themselves rounded
    # at absolute scale, which is a cliff rather than a constant (see the module docstring).
    shape = (left.shape[0], right.shape[0], _RBOX_COLUMNS)
    midpoint = (left[:, None, :2] + right[None, :, :2]) / 2
    subject = _local_polygons(left[:, None].expand(shape), midpoint)
    clip = _local_polygons(right[None, :].expand(shape), midpoint)

    area_a = _shoelace(subject).clamp_min(0)
    area_b = _shoelace(clip).clamp_min(0)
    intersection = _intersection_area(subject, clip)
    union = area_a + area_b - intersection
    tiny = torch.finfo(union.dtype).tiny
    return torch.where(union > tiny, intersection / union.clamp_min(tiny), torch.zeros_like(union)).clamp(0.0, 1.0)


def rotated_detections_to_predictions(detections: Tensor, score_floor: float = 0.0) -> list[dict[str, Tensor]]:
    """Convert a fixed-size oriented detection batch into per-image prediction dicts.

    Each row of the ``(B, N, 7)`` batch is the A45 tuple
    ``[cx, cy, w, h, theta, score, class]`` that
    :func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk` emits. A row survives when its
    ``score`` is strictly above ``score_floor``, which at the default ``0.0`` drops the
    score-zero padding rows that fill the fixed shape — the same rule
    :func:`~lucid_yolo.eval.coco_eval.detections_to_predictions` applies to the
    axis-aligned tuple.

    Unlike the COCO path there is no label remapping. DOTA class ids are already the
    contiguous indices into :data:`~lucid_yolo.data.dota.DOTA_CLASSES` that the head
    predicts, so a mapping table would be the identity and one more thing to keep aligned.

    Args:
        detections: Oriented detections of shape ``(B, N, 7)``, typically ``(B, 300, 7)``.
        score_floor: Rows with ``score`` at or below this are dropped. Defaults to ``0.0``
            (drop only the score-zero padding rows).

    Returns:
        A length-``B`` list of prediction dicts with ``rboxes`` (``(M, 5)``), ``scores``
        (``(M,)``) and ``labels`` (``(M,)`` long) — the per-image shape
        :func:`evaluate_rotated_map` consumes.

    Raises:
        ValueError: If ``detections`` is not a 3-D ``(B, N, 7)`` tensor.

    Examples:
        >>> import torch
        >>> dets = torch.tensor(
        ...     [
        ...         [
        ...             [5.0, 5.0, 4.0, 2.0, 0.3, 0.9, 1.0],
        ...             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ...         ]
        ...     ]
        ... )  # one real detection, one padding row
        >>> preds = rotated_detections_to_predictions(dets)
        >>> len(preds), tuple(preds[0]["rboxes"].shape)  # padding row dropped
        (1, (1, 5))
        >>> preds[0]["labels"].tolist(), preds[0]["scores"].tolist()
        ([1], [0.8999999761581421])
    """
    if detections.ndim != 3 or detections.shape[2] != _DETECTION_WIDTH:
        raise ValueError(f"detections must be (B, N, {_DETECTION_WIDTH}); got shape {tuple(detections.shape)}")
    dense = detections.detach().to(device="cpu", dtype=torch.float32)
    return [_image_to_prediction(image, score_floor) for image in dense]


def tiled_targets_to_ground_truth(tiled: TiledTargets) -> dict[str, Tensor]:
    """Convert one tile's :class:`~lucid_yolo.data.tiling.TiledTargets` into a target dict.

    The rotated boxes, class labels and difficult flags travel together on the single
    instance axis WP-056 establishes and WP-057 preserves, so the flag reaching the
    accumulator is provably the flag of the box beside it. That flag is the R18 one OR-ed
    with "this part fell below 70% of its original area", which is why the evaluation
    loader must keep difficult instances rather than filter them (module docstring).

    Args:
        tiled: One window's targets and their R18 bookkeeping, as returned by
            :func:`~lucid_yolo.data.tiling.crop_targets` or
            :func:`~lucid_yolo.data.tiling.tile_image_targets`.

    Returns:
        A target dict with ``rboxes`` (``(N, 5)``), ``labels`` (``(N,)`` long) and
        ``difficult`` (``(N,)`` bool) — the per-image shape
        :func:`evaluate_rotated_map` consumes.

    Examples:
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> from lucid_yolo.data.tiling import TiledTargets
        >>> targets = Targets(
        ...     boxes=torch.tensor([[0.0, 0.0, 4.0, 2.0]]),
        ...     labels=torch.tensor([3]),
        ...     rboxes=torch.tensor([[2.0, 1.0, 4.0, 2.0, 0.0]]),
        ... )
        >>> tiled = TiledTargets(targets, torch.tensor([True]), torch.tensor([0.5]))
        >>> ground_truth = tiled_targets_to_ground_truth(tiled)
        >>> ground_truth["labels"].tolist(), ground_truth["difficult"].tolist()
        ([3], [True])
    """
    return {
        "rboxes": tiled.targets.rboxes.detach().cpu(),
        "labels": tiled.targets.labels.detach().cpu().to(torch.long),
        "difficult": tiled.difficult.detach().cpu().to(torch.bool),
    }


def evaluate_rotated_map(
    preds: Sequence[Mapping[str, Tensor]],
    targets: Sequence[Mapping[str, Tensor]],
    *,
    max_detections: int | None = MAX_DETECTIONS,
) -> dict[str, float]:
    """Score oriented predictions against oriented ground truth with rotated mAP50-95.

    Implements the five protocol decisions of the module docstring: ten IoU thresholds,
    101-point interpolated recall, a 300-detection cap per image, exclusion of classes
    without non-difficult ground truth, and R18's difficult-instance rule. Matching is
    greedy and one-to-one — within each class and threshold, detections are taken in
    descending score order and each claims the best still-unmatched ground truth at or
    above the threshold.

    Matching is performed per image and the resulting decisions are then merged in global
    score order for the precision-recall curve. That is not an approximation: a ground
    truth can only be claimed by a detection in its own image, so the per-image order and
    the global order consume ground truths identically.

    Args:
        preds: Per-image prediction dicts with ``rboxes`` (``(M, 5)``), ``scores``
            (``(M,)``) and ``labels`` (``(M,)``), as produced by
            :func:`rotated_detections_to_predictions`. Images carrying more than
            ``max_detections`` rows are capped here, by score, across all classes.
        targets: Per-image ground-truth dicts with ``rboxes`` (``(N, 5)``), ``labels``
            (``(N,)``) and ``difficult`` (``(N,)`` bool), aligned by position with
            ``preds`` and in the same coordinate and label space. ``difficult`` may be
            omitted, which is read as no instance being difficult.
        max_detections: Detections scored per evaluation unit (decision 3). Defaults to
            :data:`MAX_DETECTIONS`, the per-tile cap A47 fixes. ``None`` scores every
            detection supplied, which is what :mod:`lucid_yolo.eval.tile_merge`'s
            whole-image unit needs: a merged source image is many forward passes, each
            already capped at emission, so re-capping the merged image at 300 would
            measure a truncation rather than the model. ``mar_300`` keeps A47's name
            wherever the cap goes, so a report quoting a non-default cap must name it.

    Returns:
        ``{"map": ..., "map_50": ..., "map_75": ..., "mar_300": ...}``. ``map`` is the
        mean over classes and thresholds; ``mar_300`` is the mean recall attained at the
        300-detection cap. An evaluation with no scorable class yields zeros throughout.

    Raises:
        ValueError: If ``preds`` and ``targets`` differ in length.

    Examples:
        >>> import torch
        >>> box = torch.tensor([[10.0, 10.0, 4.0, 2.0, 0.3]])
        >>> preds = [{"rboxes": box, "scores": torch.tensor([0.9]), "labels": torch.tensor([2])}]
        >>> targets = [{"rboxes": box, "labels": torch.tensor([2]), "difficult": torch.tensor([False])}]
        >>> stats = evaluate_rotated_map(preds, targets)
        >>> round(stats["map"], 3), round(stats["map_50"], 3), round(stats["mar_300"], 3)
        (1.0, 1.0, 1.0)
        >>> miss = [{"rboxes": box, "labels": torch.tensor([2]), "difficult": torch.tensor([True])}]
        >>> evaluate_rotated_map(preds, miss)["map"]  # the only class is all-difficult
        0.0
    """
    if len(preds) != len(targets):
        raise ValueError(f"preds and targets must align by position; got {len(preds)} and {len(targets)}")
    labels = _scorable_labels(targets)
    if not labels:
        return dict.fromkeys(_METRIC_KEYS, 0.0)
    capped = [_cap_detections(prediction, max_detections) for prediction in preds]
    curves = [
        _class_curves([_split_class(p, t, label) for p, t in zip(capped, targets, strict=True)]) for label in labels
    ]
    precision = torch.stack([curve[0] for curve in curves])
    recall = torch.stack([curve[1] for curve in curves])
    return {
        "map": float(precision.mean()),
        "map_50": float(precision[:, _IOU_50_INDEX].mean()),
        "map_75": float(precision[:, _IOU_75_INDEX].mean()),
        "mar_300": float(recall.mean()),
    }


@dataclass(frozen=True)
class _ImageSplit:
    """One image's detections and ground truths, restricted to a single class.

    Attributes:
        scores: ``(D,)`` detection scores in descending order.
        iou: ``(D, G)`` overlap of each detection with each ground truth, rows in the
            same descending-score order as ``scores``.
        difficult: ``(G,)`` bool R18 flags of the ground truths.
    """

    scores: Tensor
    iou: Tensor
    difficult: Tensor


def _local_polygons(pairs: Tensor, midpoint: Tensor) -> Tensor:
    """Expand per-pair rotated boxes to corners in the frame centred on ``midpoint``.

    The subtraction happens on the **centre**, before
    :func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons` adds the half-extents, so
    the corner coordinates are born small instead of being made small afterwards. That is
    the whole precision story of this module: a corner expanded at absolute DOTA scale is
    already rounded to that scale's float32 spacing, and no later shift recovers it.

    Args:
        pairs: ``(M, N, 5)`` rotated boxes, broadcast to the pair grid.
        midpoint: ``(M, N, 2)`` centre each pair is re-expressed about.

    Returns:
        ``(M, N, 4, 2)`` corner coordinates in the per-pair local frame.

    Examples:
        >>> import torch
        >>> box = torch.tensor([[[10.0, 10.0, 4.0, 2.0, 0.0]]])
        >>> _local_polygons(box, torch.tensor([[[10.0, 10.0]]]))[0, 0].tolist()
        [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]]
    """
    shifted = torch.cat([pairs[..., :2] - midpoint, pairs[..., 2:]], dim=-1)
    corners = rboxes_to_polygons(shifted.reshape(-1, _RBOX_COLUMNS))
    return corners.reshape(*pairs.shape[:2], _QUAD_CORNERS, corners.shape[-1])


def _intersection_area(subject: Tensor, clip: Tensor) -> Tensor:
    """Clip ``subject`` against every edge of ``clip`` and return the surviving area.

    One Sutherland-Hodgman pass per clip edge, each followed by a compaction back to
    :data:`_MAX_INTERSECTION_CORNERS` slots — which loses nothing, since an intermediate
    result cannot exceed that bound (see the constant's note).

    Args:
        subject: ``(M, N, 4, 2)`` corners of the clipped quadrilateral, per pair.
        clip: ``(M, N, 4, 2)`` corners of the clipping quadrilateral, per pair.

    Returns:
        ``(M, N)`` intersection area, never negative.

    Examples:
        >>> import torch
        >>> unit = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        >>> pair = unit[None, None]
        >>> float(_intersection_area(pair, pair))
        4.0
    """
    polygon = subject
    valid = torch.ones(subject.shape[:-1], dtype=torch.bool, device=subject.device)
    for corner in range(_QUAD_CORNERS):
        start = clip[..., corner, :]
        end = clip[..., (corner + 1) % _QUAD_CORNERS, :]
        polygon, valid = _clip_by_edge(polygon, valid, start, end)
        polygon, valid = _compact(polygon, valid)
    enclosed = valid.sum(dim=-1) >= _MIN_AREA_CORNERS
    return torch.where(enclosed, _shoelace(polygon), torch.zeros_like(enclosed, dtype=polygon.dtype)).clamp_min(0)


def _clip_by_edge(polygon: Tensor, valid: Tensor, start: Tensor, end: Tensor) -> tuple[Tensor, Tensor]:
    """Run one Sutherland-Hodgman step against the directed edge ``start -> end``.

    Emits two slots per input vertex — the vertex itself when it is inside, and the
    crossing point when the edge to its successor changes side — so the output shape is a
    pure function of the input shape and no data-dependent resize is needed. The interior
    test is ``cross >= 0``, which is **edge-inclusive** and matches the positive winding
    :func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons` guarantees.

    Args:
        polygon: ``(M, N, K, 2)`` vertices, valid ones compacted to the front.
        valid: ``(M, N, K)`` prefix mask of live vertices.
        start: ``(M, N, 2)`` first endpoint of the clip edge.
        end: ``(M, N, 2)`` second endpoint of the clip edge.

    Returns:
        ``(M, N, 2K, 2)`` vertices and their ``(M, N, 2K)`` validity mask.

    Examples:
        >>> import torch
        >>> square = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])[None, None]
        >>> live = torch.ones(square.shape[:-1], dtype=torch.bool)
        >>> _, mask = _clip_by_edge(square, live, torch.zeros(1, 1, 2), torch.tensor([[[1.0, 0.0]]]))
        >>> int(mask.sum())  # the whole square lies on the inside of the x axis
        4
    """
    edge = (end - start)[..., None, :]
    offset = polygon - start[..., None, :]
    distance = edge[..., 0] * offset[..., 1] - edge[..., 1] * offset[..., 0]
    successor = _successor(polygon, valid)
    next_distance = _successor(distance[..., None], valid)[..., 0]

    inside = distance >= 0
    crossing = inside != (next_distance >= 0)
    denominator = distance - next_distance
    step = distance / torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    crossed = polygon + step[..., None] * (successor - polygon)

    vertices = torch.stack([polygon, crossed], dim=-2).flatten(-3, -2)
    kept = torch.stack([valid & inside, valid & crossing], dim=-1).flatten(-2, -1)
    return vertices, kept


def _successor(values: Tensor, valid: Tensor) -> Tensor:
    """Return each slot's cyclic successor among the live vertices.

    ``valid`` is a prefix mask, so the successor of slot ``i`` is slot ``i + 1`` when that
    slot is live and slot ``0`` otherwise — the ring closes at the first vertex rather
    than wandering into the padding.

    Args:
        values: ``(M, N, K, C)`` per-vertex values.
        valid: ``(M, N, K)`` prefix mask of live vertices.

    Returns:
        ``(M, N, K, C)`` values of each slot's successor.

    Examples:
        >>> import torch
        >>> values = torch.tensor([[[[1.0], [2.0], [3.0]]]])
        >>> mask = torch.tensor([[[True, True, False]]])
        >>> _successor(values, mask).flatten().tolist()  # slot 1 wraps to slot 0
        [2.0, 1.0, 1.0]
    """
    rolled = values.roll(-1, dims=-2)
    rolled_valid = valid.roll(-1, dims=-1)
    return torch.where(rolled_valid[..., None], rolled, values[..., :1, :])


def _compact(polygon: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Move live vertices to the front, truncate to the vertex bound, and pad with vertex 0.

    Restores the prefix-mask invariant :func:`_successor` relies on, and replaces every
    dead slot with the first live vertex so :func:`_shoelace` may run mask-free: the
    padding edges are zero-length and contribute nothing to the area.

    Args:
        polygon: ``(M, N, K, 2)`` vertices in emission order.
        valid: ``(M, N, K)`` mask of live vertices, in any arrangement.

    Returns:
        ``(M, N, 8, 2)`` compacted vertices and their ``(M, N, 8)`` prefix mask.

    Examples:
        >>> import torch
        >>> pts = torch.tensor([[[[9.0, 9.0], [1.0, 1.0], [2.0, 2.0]]]])
        >>> mask = torch.tensor([[[False, True, True]]])
        >>> kept, live = _compact(pts, mask)
        >>> kept[0, 0, :3].tolist(), live[0, 0, :3].tolist()
        ([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]], [True, True, False])
    """
    order = torch.argsort(valid.logical_not().to(torch.uint8), dim=-1, stable=True)[..., :_MAX_INTERSECTION_CORNERS]
    gathered = polygon.gather(-2, order[..., None].expand(*order.shape, polygon.shape[-1]))
    kept = valid.gather(-1, order)
    return torch.where(kept[..., None], gathered, gathered[..., :1, :]), kept


def _shoelace(polygon: Tensor) -> Tensor:
    """Return the signed area of each polygon by the shoelace formula.

    Positive for the winding :func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons`
    emits. Repeated vertices contribute zero, which is what lets padded rings be measured
    without a mask.

    Args:
        polygon: ``(..., K, 2)`` vertices in ring order.

    Returns:
        ``(...)`` signed area.

    Examples:
        >>> import torch
        >>> square = torch.tensor([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]])
        >>> float(_shoelace(square))
        6.0
    """
    successor = polygon.roll(-1, dims=-2)
    cross = polygon[..., 0] * successor[..., 1] - successor[..., 0] * polygon[..., 1]
    return 0.5 * cross.sum(dim=-1)


def _check_rboxes(rboxes: Tensor, name: str) -> None:
    """Raise :class:`ValueError` unless ``rboxes`` is a 2-D ``(K, 5)`` tensor.

    Examples:
        >>> import torch
        >>> _check_rboxes(torch.zeros((0, 5)), "boxes_a")
    """
    if rboxes.ndim != 2 or rboxes.shape[1] != _RBOX_COLUMNS:
        raise ValueError(f"{name} must be (K, {_RBOX_COLUMNS}); got shape {tuple(rboxes.shape)}")


def _image_to_prediction(detections: Tensor, score_floor: float) -> dict[str, Tensor]:
    """Convert one image's ``(N, 7)`` oriented detections into a prediction dict.

    Examples:
        >>> import torch
        >>> rows = torch.tensor([[1.0, 2.0, 4.0, 2.0, 0.1, 0.5, 3.0], [0.0] * 7])
        >>> _image_to_prediction(rows, 0.0)["labels"].tolist()
        [3]
    """
    kept = detections[detections[:, _SCORE_COLUMN] > score_floor]
    return {
        "rboxes": kept[:, :_RBOX_COLUMNS],
        "scores": kept[:, _SCORE_COLUMN],
        "labels": kept[:, _LABEL_COLUMN].to(torch.long),
    }


def _cap_detections(prediction: Mapping[str, Tensor], cap: int | None) -> Mapping[str, Tensor]:
    """Keep at most ``cap`` rows of one image, the highest-scoring ones.

    The cap is applied across all classes at once, as COCO's ``maxDets`` is: it models
    what the detector may emit for the image, not what it may emit per class. ``None``
    disables it for an evaluation unit whose detections were already capped at emission.

    Examples:
        >>> import torch
        >>> many = {
        ...     "rboxes": torch.zeros(400, 5),
        ...     "scores": torch.arange(400, dtype=torch.float32),
        ...     "labels": torch.zeros(400, dtype=torch.long),
        ... }
        >>> tuple(_cap_detections(many, MAX_DETECTIONS)["scores"].shape)
        (300,)
        >>> tuple(_cap_detections(many, None)["scores"].shape)
        (400,)
    """
    scores = prediction["scores"]
    if cap is None or scores.shape[0] <= cap:
        return prediction
    keep = torch.topk(scores, cap).indices
    return {key: value[keep] for key, value in prediction.items()}


def _difficult_flags(target: Mapping[str, Tensor]) -> Tensor:
    """Return a target's difficult flags, defaulting to all-false when it carries none.

    Examples:
        >>> import torch
        >>> _difficult_flags({"labels": torch.tensor([1, 2])}).tolist()
        [False, False]
    """
    flags = target.get("difficult")
    if flags is None:
        return torch.zeros(target["labels"].shape[0], dtype=torch.bool)
    return flags.to(torch.bool)


def _scorable_labels(targets: Sequence[Mapping[str, Tensor]]) -> list[int]:
    """Return the class labels carrying at least one non-difficult ground truth.

    Decision 4: a class without them has an empty recall denominator, so it is excluded
    from the mean rather than contributing a zero. That covers both the class absent from
    the ground truth entirely and the class whose every instance is flagged difficult.

    Examples:
        >>> import torch
        >>> targets = [{"labels": torch.tensor([4, 7]), "difficult": torch.tensor([False, True])}]
        >>> _scorable_labels(targets)  # class 7 is all-difficult
        [4]
    """
    labels: set[int] = set()
    for target in targets:
        scorable = target["labels"][~_difficult_flags(target)]
        labels.update(int(label) for label in scorable)
    return sorted(labels)


def _split_class(
    prediction: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    label: int,
) -> _ImageSplit:
    """Restrict one image's detections and ground truths to ``label`` and pair them up.

    The detections are sorted by descending score here, once, rather than inside the
    per-threshold matching: the order is a property of the image, not of the threshold.

    Examples:
        >>> import torch
        >>> box = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]])
        >>> prediction = {"rboxes": box, "scores": torch.tensor([0.7]), "labels": torch.tensor([1])}
        >>> target = {"rboxes": box, "labels": torch.tensor([1]), "difficult": torch.tensor([False])}
        >>> split = _split_class(prediction, target, 1)
        >>> tuple(split.iou.shape), float(split.iou[0, 0])
        ((1, 1), 1.0)
    """
    keep = prediction["labels"] == label
    scores = prediction["scores"][keep]
    order = torch.argsort(scores, descending=True, stable=True)
    boxes = prediction["rboxes"][keep][order]

    difficult = _difficult_flags(target)
    gt_keep = target["labels"] == label
    return _ImageSplit(
        scores=scores[order],
        iou=rotated_iou(boxes, target["rboxes"][gt_keep]),
        difficult=difficult[gt_keep],
    )


def _class_curves(splits: Sequence[_ImageSplit]) -> tuple[Tensor, Tensor]:
    """Return one class's average precision and recall at each of the ten thresholds.

    Called only for a label :func:`_scorable_labels` returned, so ``splits`` is non-empty
    and carries at least one non-difficult ground truth between them.

    Examples:
        >>> import torch
        >>> split = _ImageSplit(torch.tensor([0.9]), torch.ones(1, 1), torch.tensor([False]))
        >>> precision, recall = _class_curves([split])
        >>> float(precision[0]), float(recall[0])
        (1.0, 1.0)
    """
    positives = sum(int((~split.difficult).sum()) for split in splits)
    scores = torch.cat([split.scores for split in splits])
    order = torch.argsort(scores, descending=True, stable=True)
    matched = [_match_image(split) for split in splits]
    true_positive = torch.cat([value[0] for value in matched], dim=1)[:, order]
    false_positive = torch.cat([value[1] for value in matched], dim=1)[:, order]
    results = [
        _threshold_curve(hits, misses, positives) for hits, misses in zip(true_positive, false_positive, strict=True)
    ]
    return torch.tensor([value[0] for value in results]), torch.tensor([value[1] for value in results])


def _threshold_curve(true_positive: Tensor, false_positive: Tensor, positives: int) -> tuple[float, float]:
    """Reduce one threshold's score-ordered match sequence to ``(average precision, recall)``.

    Args:
        true_positive: ``(N,)`` per-detection true-positive flags in descending score order.
        false_positive: ``(N,)`` per-detection false-positive flags in the same order.
        positives: The class's non-difficult ground-truth count.

    Examples:
        >>> import torch
        >>> miss = torch.tensor([False])
        >>> _threshold_curve(miss, ~miss, 1)  # the only detection missed
        (0.0, 0.0)
    """
    scored = true_positive | false_positive
    return _average_precision(true_positive[scored], false_positive[scored], positives)


def _match_image(split: _ImageSplit) -> tuple[Tensor, Tensor]:
    """Greedily assign one image's detections to its ground truths, at every threshold at once.

    Decision 5 lives here. A detection claims the best still-unmatched **non-difficult**
    ground truth at or above the threshold; failing that, an overlap with any difficult
    ground truth discards it (neither true nor false positive, and the difficult instance
    stays available for further detections); failing that it is a false positive.

    The ten thresholds are matched **together** rather than in ten passes (WP-104). They
    share the overlap matrix and the score order and differ only in what counts as a hit,
    so a pass per threshold re-walks the same detections ten times to consume a different
    subset of the same ground truths; carrying the availability state as ``(T, G)`` walks
    them once. The greedy order is unchanged — it is still descending score within the
    image, resolved independently per threshold — which is why this is a restructuring
    and not a redefinition.

    Detections that reach no ground truth at the **lowest** threshold skip the walk
    entirely. Such a detection cannot become a true positive at any threshold, cannot be
    discarded by a difficult instance at any threshold, and consumes nothing, so it is a
    false positive everywhere and the zero-initialised rows already say so. On an
    early-epoch model, where almost every one of the 300 emitted detections lands nowhere
    near a target, that is nearly the whole loop.

    Args:
        split: One image's detections, overlaps and difficult flags for a single class.

    Returns:
        ``(T, D)`` true-positive and false-positive flags, one row per
        :data:`IOU_THRESHOLDS` entry, in the split's own descending-score order.

    Examples:
        >>> import torch
        >>> iou = torch.tensor([[0.9, 0.0], [0.8, 0.0]])  # two detections, one good target
        >>> split = _ImageSplit(torch.tensor([0.9, 0.5]), iou, torch.tensor([False, False]))
        >>> true_positive, false_positive = _match_image(split)
        >>> true_positive[0].tolist(), false_positive[0].tolist()  # at 0.50 the second duplicates
        ([True, False], [False, True])
        >>> true_positive[-1].tolist()  # at 0.95 neither overlap is enough
        [False, False]
    """
    thresholds = torch.tensor(IOU_THRESHOLDS, dtype=split.iou.dtype)
    detections, ground_truths = split.iou.shape
    true_positive = torch.zeros(len(IOU_THRESHOLDS), detections, dtype=torch.bool)
    discarded = torch.zeros_like(true_positive)
    if ground_truths == 0:
        return true_positive, ~true_positive
    available = (~split.difficult).expand(len(IOU_THRESHOLDS), ground_truths).clone()
    reachable = split.iou.max(dim=1).values >= float(thresholds.min())
    for detection in reachable.nonzero().flatten().tolist():
        overlaps = split.iou[detection]
        candidates = torch.where(available, overlaps, torch.full_like(overlaps, -1.0))
        best_overlap, best = candidates.max(dim=1)
        hit = best_overlap >= thresholds
        true_positive[hit, detection] = True
        available[hit, best[hit]] = False
        difficult_overlaps = overlaps[split.difficult]
        if difficult_overlaps.numel():
            reached = (difficult_overlaps.unsqueeze(0) >= thresholds.unsqueeze(1)).any(dim=1)
            discarded[:, detection] = reached & ~hit
    return true_positive, ~(true_positive | discarded)


def _average_precision(true_positive: Tensor, false_positive: Tensor, positives: int) -> tuple[float, float]:
    """Reduce a score-ordered match sequence to interpolated AP and attained recall.

    Decision 2: the precision-recall curve is made monotone by a running maximum from the
    right and then sampled at :data:`RECALL_POINTS` evenly spaced recall values, each
    taking the precision of the first point whose recall reaches it and zero when none
    does. That is COCO's definition, so this AP and
    :func:`~lucid_yolo.eval.coco_eval.evaluate_bbox`'s are the same statistic.

    Why the recall comparison is integer arithmetic:
        Both sides of "has recall reached this grid point" are **rationals** — attained
        recall is ``hits / positives``, the grid point is ``i / 100`` — and the sample is
        decided by whether they are equal, which floating point may not be able to
        represent. In float32 the case ``hits = 13``, ``positives = 20``, ``i = 65`` is
        decided **wrongly**: ``13 / 20`` rounds to ``0.6499999761581421`` while
        ``linspace(0, 1, 101)[65]`` rounds to ``0.6500000357627869``, so the point that
        should sample the curve's last precision falls off the end and samples zero
        instead. Attained recall landing exactly on a grid point is **common**, not
        exotic — any ``positives`` dividing 100 hits it routinely — and each occurrence
        silently removes ``1/101`` of that class's average precision.

        Comparing ``hits * 100 >= i * positives`` in int64 decides the same question
        exactly, for every input, in every dtype, on every device. It is not a tolerance:
        there is no epsilon to choose and no precision that could be insufficient. The
        float64 the reference implementation uses happens to decide this case correctly,
        which is why the bug surfaced as a fixture-dependent near-miss against it rather
        than as an obvious break.

    The empty guard covers two cases of unequal standing. A class with no detections at
    all is ordinary and reachable — it scores zero. ``positives == 0`` should be
    unreachable, since :func:`_scorable_labels` admits only classes with a non-difficult
    instance; it is nonetheless checked because the consequence of being wrong is a
    division by zero, and torch answers that with a silent ``inf`` rather than an
    exception. A guard against a silently corrupted metric is worth its line.

    Examples:
        >>> import torch
        >>> hit = torch.tensor([True])
        >>> _average_precision(hit, ~hit, 1)
        (1.0, 1.0)
        >>> # One of two ground truths found. The average is 51/101 = 0.5050, not 1/2:
        >>> # the 101-point grid includes both endpoints, so 51 of its points lie at or
        >>> # below recall 0.5 and each of those samples a precision of 1.
        >>> average, attained = _average_precision(hit, ~hit, 2)
        >>> round(average, 4), attained
        (0.505, 0.5)
    """
    if positives == 0 or true_positive.numel() == 0:
        return 0.0, 0.0
    hits = true_positive.to(device="cpu", dtype=torch.int64).cumsum(0)
    ranked = hits + false_positive.to(device="cpu", dtype=torch.int64).cumsum(0)
    precision = hits.to(torch.float64) / ranked.clamp_min(1)
    envelope = torch.flip(torch.cummax(torch.flip(precision, (0,)), dim=0).values, (0,))
    # Exact rational test: recall >= i/(R-1) is hits*(R-1) >= i*positives over integers.
    grid = torch.arange(RECALL_POINTS, dtype=torch.int64)
    index = torch.searchsorted(hits * (RECALL_POINTS - 1), grid * positives)
    reachable = index < hits.numel()
    sampled = torch.where(reachable, envelope[index.clamp(max=hits.numel() - 1)], torch.zeros(1, dtype=torch.float64))
    return float(sampled.mean()), float(hits[-1]) / positives
