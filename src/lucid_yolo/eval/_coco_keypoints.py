# SPDX-License-Identifier: Apache-2.0
"""COCO OKS keypoint scoring behind :mod:`lucid_yolo.eval.coco_eval` (WP-124, WP-134, WP-205).

Holds the keypoint half of the instrument: the OKS sigmas (COCO's 17-point table and A67's
uniform value) and the keypoint protocol's detection cap, area partition and statistic names;
:func:`keypoints_to_predictions` and :func:`evaluate_keypoints`, which drive
:class:`faster_coco_eval.COCOeval_faster` over an in-memory COCO document built from the
prediction and ground-truth tensors; and :class:`_KeypointScorer`, the whole-split accumulator
:class:`~lucid_yolo.eval.coco_eval.DualPathEvaluator` scores a keypoint checkpoint through.

Moved out of :mod:`lucid_yolo.eval.coco_eval` unchanged, and re-exported from it under every
name a caller or test reaches it by. A patch of a name these functions look up targets this
module, not that one.
"""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING, cast

import torch
from faster_coco_eval import COCO, COCOeval_faster

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

#: Name prefix of the ten OKS statistics in a keypoint report. Same argument as
#: :data:`_SEGM_PREFIX`: the box statistics keep their bare names, so a keypoint
#: report is a superset of a detection one, and the presence of an ``oks_`` key is
#: exactly the statement "this checkpoint has a keypoint branch".
_OKS_PREFIX = "oks_"

#: The COCO 17-point OKS per-keypoint sigmas (R12), in COCO's own point order
#: (nose, l/r eye, l/r ear, l/r shoulder, l/r elbow, l/r wrist, l/r hip, l/r
#: knee, l/r ankle). Named explicitly here rather than left to
#: faster_coco_eval's internal default (Params.setKpParams) so the value this
#: project reports against is citable and version-independent. It agrees with
#: that library default to within float rounding rather than exactly: the
#: library writes the table as ``np.array([0.26, 0.25, ...]) / 10.0`` while this
#: one writes each quotient literally, and the largest elementwise difference is
#: 1.3877787807814457e-17 (measured against the installed faster_coco_eval,
#: `faster_coco_eval.core.cocoeval.Params.setKpParams`). Re-measure that figure
#: rather than trusting this comment; if the library ever moves its own default
#: by more than rounding, this project's reported numbers must not follow.
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

#: A uniform per-point OKS sigma for a keypoint schema that is not COCO's
#: 17-point human table (A67), at R12's own median sigma value. Not a
#: simplification: a non-uniform vector encodes which landmarks *human
#: annotators* localize less reliably, and for an analytically-placed or
#: otherwise non-human schema that structure has never been measured, so any
#: per-point vector this project invented would fabricate it. Uniform is a
#: monotone rescaling of OKS, so it sets only where a floor sits and never
#: which run wins (WP-137).
SYMBOL_KEYPOINT_OKS_SIGMA = 0.072

#: COCO's keypoint detection cap and area partition (R12), pinned for exactly the
#: reason :data:`COCO_KEYPOINT_OKS_SIGMAS` is: ``Params.setKpParams`` sets all
#: three, and a report that cites its sigmas but inherits its detection cap and
#: its medium/large split from whatever the library happens to default to is
#: citable in one field out of three. The keypoint protocol caps at 20 detections
#: per image and defines no small bucket, so the ranges below are COCO's
#: ``all``/``medium``/``large`` in that order (``0``, ``32**2``, ``96**2``, and
#: ``1e5**2`` as the open upper bound). Equal to the installed faster_coco_eval's
#: own defaults today, which
#: ``tests/eval/test_keypoint_eval.py::TestKeypointEvaluationParams`` pins so a
#: library that moves them is noticed rather than silently followed.
_KEYPOINT_MAX_DETS: tuple[int, ...] = (20,)
_KEYPOINT_AREA_RANGES: tuple[tuple[float, float], ...] = (
    (0.0, 1e10),
    (1024.0, 9216.0),
    (9216.0, 1e10),
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
        ```pycon
        >>> import torch
        >>> points = torch.tensor(
        ...     [[[[10.0, 12.0], [20.0, 22.0]], [[0.0, 0.0], [0.0, 0.0]]]]
        ... )  # one real instance, one padding row
        >>> scores = torch.tensor([[0.9, 0.0]])
        >>> labels = torch.tensor([[1, 0]])
        >>> preds = keypoints_to_predictions(points, scores, labels, {1: 42, 0: 1})
        >>> tuple(preds[0]["keypoints"].shape), preds[0]["labels"].tolist()
        ((1, 2, 2), [42])

        ```
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


def _visible_extent(points: Tensor, visible: Tensor) -> tuple[list[float], float]:
    """Reconstruct an instance's ``xywh`` box and area from its visible points alone.

    The fallback for ground truth carrying no annotation metadata — a synthetic
    fixture, or the WP-124 one-shot callers. It is **not** what COCO scores: OKS
    normalizes by the annotation's own ``area``, which counts the whole segmented
    person and not the hull of the labeled joints, and the medium/large bucket
    boundary is drawn on that same figure. :func:`_instance_metadata` prefers the
    supplied fields wherever they exist and reaches here only when they do not.
    """
    if not bool(visible.any()):
        return [0.0, 0.0, 1.0, 1.0], 1.0
    visible_points = points[visible]
    minimum = visible_points.amin(dim=0)
    extent = visible_points.amax(dim=0) - minimum
    width, height = float(extent[0]), float(extent[1])
    return [float(minimum[0]), float(minimum[1]), width, height], max(width * height, 1.0)


def _instance_metadata(
    target: Mapping[str, Tensor],
    index: int,
    points: Tensor,
    visible: Tensor,
) -> dict[str, object]:
    """Return one ground-truth instance's COCO ``area``/``iscrowd``/``num_keypoints``/``bbox``.

    Each field is taken from the annotation when the target carries it and
    reconstructed only when it does not, because the backend's keypoint path reads
    all four as supplied values: ``area`` normalizes OKS and selects the
    medium/large bucket, ``iscrowd`` and a zero ``num_keypoints`` both mark an
    instance ignored. Recomputing them from visible-point extent — which is what
    this function's fallback does, and what WP-124 did unconditionally — scores a
    different protocol under the same name.
    """
    boxes = target.get("boxes")
    areas = target.get("area")
    crowds = target.get("iscrowd")
    counts = target.get("num_keypoints")
    reconstructed_bbox, reconstructed_area = _visible_extent(points, visible)
    if boxes is not None:
        x_min, y_min, x_max, y_max = (float(value) for value in boxes[index])
        reconstructed_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    return {
        "bbox": reconstructed_bbox,
        "area": float(areas[index]) if areas is not None else reconstructed_area,
        "iscrowd": int(crowds[index]) if crowds is not None else 0,
        "num_keypoints": int(counts[index]) if counts is not None else int(visible.sum()),
    }


def _keypoint_annotations(
    targets: list[dict[str, Tensor]],
    image_ids: Sequence[int],
) -> tuple[list[dict[str, object]], set[int]]:
    """Build the COCO ground-truth annotation records and the category ids they use."""
    annotations: list[dict[str, object]] = []
    category_ids: set[int] = set()
    annotation_id = 1
    for image_id, target in zip(image_ids, targets, strict=True):
        points = target["keypoints"].detach().to(device="cpu", dtype=torch.float32)
        visibility = target["visibility"].detach().to(device="cpu", dtype=torch.long)
        labels = target["labels"].detach().to(device="cpu", dtype=torch.long)
        for index, (instance_points, instance_visibility, instance_label) in enumerate(
            zip(points, visibility, labels, strict=True)
        ):
            visible = instance_visibility > 0
            category_id = int(instance_label)
            category_ids.add(category_id)
            flat = torch.cat(
                (instance_points, instance_visibility.to(torch.float32).unsqueeze(-1)),
                dim=-1,
            ).reshape(-1)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": int(image_id),
                    "category_id": category_id,
                    "keypoints": flat.tolist(),
                    **_instance_metadata(target, index, instance_points, visible),
                }
            )
            annotation_id += 1
    return annotations, category_ids


def _keypoint_results(
    preds: list[dict[str, Tensor]],
    image_ids: Sequence[int],
) -> tuple[list[dict[str, object]], set[int]]:
    """Build the COCO detection records for predicted keypoints, visibility fixed at 2."""
    results: list[dict[str, object]] = []
    category_ids: set[int] = set()
    for image_id, prediction in zip(image_ids, preds, strict=True):
        points = prediction["keypoints"].detach().to(device="cpu", dtype=torch.float32)
        scores = prediction["scores"].detach().to(device="cpu", dtype=torch.float32)
        labels = prediction["labels"].detach().to(device="cpu", dtype=torch.long)
        for instance_points, instance_score, instance_label in zip(points, scores, labels, strict=True):
            category_id = int(instance_label)
            category_ids.add(category_id)
            visibility = torch.full((instance_points.shape[0], 1), 2.0, dtype=instance_points.dtype)
            flat = torch.cat((instance_points, visibility), dim=-1).reshape(-1)
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": category_id,
                    "keypoints": flat.tolist(),
                    "score": float(instance_score),
                }
            )
    return results, category_ids


def _pin_keypoint_params(evaluator: COCOeval_faster) -> None:
    """Set one keypoint evaluator's detection cap and area partition from this module's constants.

    :data:`COCO_KEYPOINT_OKS_SIGMAS` is passed at construction so the OKS this
    project reports is a citable number rather than an inherited library default.
    ``maxDets`` and ``areaRng`` decide the AR lines and the medium/large split of
    the same report, so they are pinned here for the same reason and from the same
    kind of named constant (WP-166) — the values themselves are COCO's own, and
    equal what ``Params.setKpParams`` sets today.

    The read-back is not ceremony: hotcoco's ``params`` is copy-on-read, which
    makes an assignment to it a silent no-op (see :func:`_hotcoco_stats`), and an
    unnoticed no-op here would quietly restore exactly the library-default
    dependence this closes.
    """
    evaluator.params.maxDets = list(_KEYPOINT_MAX_DETS)
    evaluator.params.areaRng = [list(bounds) for bounds in _KEYPOINT_AREA_RANGES]
    pinned_ranges = tuple(tuple(float(bound) for bound in bounds) for bounds in evaluator.params.areaRng)
    if tuple(evaluator.params.maxDets) != _KEYPOINT_MAX_DETS or pinned_ranges != _KEYPOINT_AREA_RANGES:
        raise RuntimeError(
            "COCOeval_faster did not keep the keypoint maxDets/areaRng it was given, so the "
            "detection cap and area partition behind this report are the library's rather than "
            f"COCO's. Expected {list(_KEYPOINT_MAX_DETS)} and "
            f"{[list(bounds) for bounds in _KEYPOINT_AREA_RANGES]}, got "
            f"{list(evaluator.params.maxDets)} and {[list(bounds) for bounds in pinned_ranges]}."
        )


def evaluate_keypoints(
    preds: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    sigmas: Sequence[float] = COCO_KEYPOINT_OKS_SIGMAS,
    image_ids: Sequence[int] | None = None,
) -> dict[str, float]:
    """Score keypoint predictions with COCO's object keypoint similarity protocol.

    Builds the minimal in-memory COCO ground-truth and result documents needed by
    :class:`faster_coco_eval.COCOeval_faster`, then runs its native keypoint path.
    Ground-truth ``visibility`` follows WP-121's tensor convention directly:
    ``v == 0`` points are unlabeled and excluded from OKS, while ``v > 0`` points
    define the match distances. Predicted visibility is the COCO placeholder ``2``
    because OKS reads only the predicted coordinates and ground-truth visibility.
    An empty prediction input yields all ten statistics at ``0.0`` rather than
    entering a degenerate backend path.

    The remaining four ground-truth fields — ``area``, ``iscrowd``,
    ``num_keypoints`` and ``bbox`` — are taken from the annotation whenever the
    target dict carries them (WP-134), and reconstructed from visible-point extent
    only when it does not. That distinction is the protocol, not bookkeeping:
    ``area`` is what OKS divides by and what draws the medium/large boundary, and
    it counts the whole segmented person rather than the hull of the labeled
    joints; ``iscrowd`` and ``num_keypoints == 0`` are the two flags the backend
    ignores an instance on. Ground truth loaded by
    :func:`~lucid_yolo.eval.annotations.load_eval_annotations` with
    ``with_keypoints=True`` carries all four; the WP-124 one-shot shape carries
    none and keeps the reconstruction it always had.

    Args:
        preds: Per-image prediction dicts with ``keypoints`` (``(M, K, 2)``),
            ``scores`` (``(M,)``), and ``labels`` (``(M,)`` original COCO category
            ids), as produced by :func:`keypoints_to_predictions`.
        targets: Position-aligned ground-truth dicts with ``keypoints``
            (``(M, K, 2)``), ``visibility`` (``(M, K)`` int64), and ``labels``
            (``(M,)`` original COCO category ids), optionally also ``boxes``
            (``xyxy``), ``area``, ``iscrowd`` and ``num_keypoints``.
        sigmas: Per-keypoint OKS sigmas in the tensors' point order. Defaults to
            :data:`COCO_KEYPOINT_OKS_SIGMAS`, COCO's 17-point person convention.
        image_ids: The COCO image id of each position, so the evaluated image set
            is the real one. Defaults to ``range(len(preds))``. An image scored
            with no prediction still belongs here — dropping it would drop its
            ground truth from the document and raise every statistic.

    Returns:
        A ten-entry dict containing ``AP_all``, ``AP_50``, ``AP_75``, the medium
        and large AP buckets, and the corresponding five ``AR_*`` statistics.
        COCO keypoint evaluation defines no small-area bucket.

    Examples:
        ```pycon
        >>> import torch
        >>> points = torch.tensor([[[10.0, 10.0], [50.0, 50.0]]])
        >>> preds = [{"keypoints": points, "scores": torch.tensor([1.0]), "labels": torch.tensor([1])}]
        >>> targets = [{"keypoints": points, "visibility": torch.tensor([[2, 2]]), "labels": torch.tensor([1])}]
        >>> stats = evaluate_keypoints(preds, targets, sigmas=[1.0, 1.0])
        >>> round(stats["AP_all"], 3), round(stats["AP_50"], 3)
        (1.0, 1.0)

        ```
    """
    if not preds:
        return dict.fromkeys(_KEYPOINT_METRIC_KEYS, 0.0)
    ids = list(range(len(preds))) if image_ids is None else [int(image_id) for image_id in image_ids]
    annotations, target_categories = _keypoint_annotations(targets, ids)
    results, _ = _keypoint_results(preds, ids)
    if not results:
        return dict.fromkeys(_KEYPOINT_METRIC_KEYS, 0.0)

    # COCO fixes the evaluated category set from the ground truth, so a category
    # only ever predicted is not a class the split asks about: including it would
    # add an all-false-positive class whose AP joins the mean. Neutral on this
    # backend today -- an empty ground-truth class scores the -1 sentinel and is
    # dropped from the mean, and loadRes accepts a result whose category is absent
    # from the table (both measured) -- so this is the protocol stated rather than
    # a number changed (WP-166).
    annotation_dict = {
        "images": [{"id": image_id} for image_id in ids],
        "annotations": annotations,
        "categories": [{"id": category_id, "name": str(category_id)} for category_id in sorted(target_categories)],
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
        _pin_keypoint_params(evaluator)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return cast("dict[str, float]", evaluator.stats_as_dict)


class _KeypointScorer:
    """One decode path's OKS statistics, accumulated whole and scored once.

    The opposite choice from :class:`_StreamingScorer`, and for the opposite
    reason. Masks force streaming because a split's worth of them is measured in
    hundreds of gigabytes; a split's worth of points is measured in megabytes
    (val2017's person annotations are ~11k instances of 17 ``(x, y)`` pairs), and
    COCO's keypoint protocol is a single matching over the whole document anyway —
    :class:`faster_coco_eval.COCOeval_faster` has no incremental update.

    Every scored image is recorded, including the ones that produced no
    prediction: their ground truth belongs in the document, and omitting it would
    silently delete the recall those images cost.
    """

    def __init__(self) -> None:
        self._preds: list[dict[str, Tensor]] = []
        self._targets: list[dict[str, Tensor]] = []
        self._image_ids: list[int] = []

    def update(
        self,
        preds: list[dict[str, Tensor]],
        targets: list[dict[str, Tensor]],
        image_ids: Sequence[int],
    ) -> None:
        """Record one batch's predictions, ground truth and COCO image ids."""
        self._preds.extend(preds)
        self._targets.extend(targets)
        self._image_ids.extend(int(image_id) for image_id in image_ids)

    def compute(self, sigmas: Sequence[float]) -> dict[str, float]:
        """Return the ten ``oks_``-prefixed statistics, or nothing if no image was scored."""
        if not self._preds:
            return {}
        stats = evaluate_keypoints(self._preds, self._targets, sigmas, image_ids=self._image_ids)
        return {_OKS_PREFIX + key: value for key, value in stats.items()}
