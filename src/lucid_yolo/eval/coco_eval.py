# SPDX-License-Identifier: Apache-2.0
"""torchmetrics bbox mAP evaluation for both inference paths (WP-043, WP-069).

The detection acceptance instrument of blueprint sec. 5.11: ``bbox`` mAP50-95 on
val2017, reported for **both** decode paths (E2E and non-E2E) from one checkpoint
in one pass, exactly as [R1, Table 7] does. The expected E2E deficit of 0.6-0.8
AP ([R1] sec. 4.4) is itself a claim measured against this instrument later; this
module only builds the measurement.

The metric engine is
:class:`torchmetrics.detection.MeanAveragePrecision` pinned to its
``faster_coco_eval`` backend (WP-069) — a COCOeval-faithful reimplementation, so
the numbers match the classic COCO protocol with no compiled-extension dependency.

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

The ground truth is supplied as torchmetrics target dicts
(``{image_id: {"boxes": ..., "labels": ...}}`` in original image coordinates,
category-id labels) rather than a ground-truth index object: torchmetrics scores
predictions directly against target tensors. Segmentation adds a ``masks`` entry
to those dicts (:func:`~lucid_yolo.eval.annotations.load_eval_annotations` with
``with_masks=True``), in the same original coordinates.

Provenance: R1 sec. 3.2.1, R1 Table 7, R1 sec. 4.4, R22, R23. Assumptions: A9, A10, A37.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torchmetrics.detection import MeanAveragePrecision

from lucid_yolo.assign.grid import make_anchor_points
from lucid_yolo.decode.common import BOX_CORNERS, SCORE_COLUMN, to_letterboxed_original
from lucid_yolo.eval.segment_decode import decode_instance_masks, masks_to_original
from lucid_yolo.models.build import SegmentOutput

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Tensor, nn

    from lucid_yolo.data.letterbox import Letterbox
    from lucid_yolo.models.heads.detect import DualHeadOutput

__all__ = [
    "DualPathEvaluator",
    "detections_to_predictions",
    "evaluate_bbox",
    "evaluate_bbox_and_segm",
    "evaluate_segm",
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

#: Feature-level input-pixel strides of the P3/P4/P5 detection head (8, 16, 32).
_STRIDES: tuple[int, int, int] = (8, 16, 32)

#: Column index of the integral class label within the A9 detection tuple.
_LABEL_COLUMN = 5


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
        [label_to_category[int(label)] for label in kept[:, _LABEL_COLUMN]],
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
    computed = _compute_metric(preds, targets, "bbox")
    return {key: float(computed[key]) for key in _METRIC_KEYS}


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
    computed = _compute_metric(preds, targets, ("bbox", "segm"))
    stats = {key: float(computed[f"bbox_{key}"]) for key in _METRIC_KEYS}
    stats.update({_SEGM_PREFIX + key: float(computed[_SEGM_PREFIX + key]) for key in _METRIC_KEYS})
    return stats


def _compute_metric(
    preds: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    iou_type: str | tuple[str, ...],
) -> Mapping[str, Tensor]:
    """Run ``MeanAveragePrecision`` for one ``iou_type`` and return its raw compute dict.

    The single place the metric is constructed and driven, so the backend, the box
    format and the detection-cap warning setting cannot differ between the bbox,
    segm and combined entry points. Note that torchmetrics names its outputs
    ``map``/``mar_*`` for a single ``iou_type`` but ``bbox_map``/``segm_map`` for a
    tuple of them; the callers own that renaming.
    """
    metric = MeanAveragePrecision(backend="faster_coco_eval", box_format="xyxy", iou_type=iou_type)  # type: ignore[arg-type]
    # The fixed 300-row decoder output routinely exceeds COCO's top-100 detection
    # cap; keeping only the 100 highest-scoring per image is the standard protocol
    # (COCOeval's maxDets), not a misconfiguration, so silence the per-call warning.
    metric.warn_on_many_detections = False
    with contextlib.redirect_stdout(io.StringIO()):
        metric.update(preds, targets)
        computed: Mapping[str, Tensor] = metric.compute()
    return computed


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


def _score_path(preds: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]) -> dict[str, float]:
    """Score one path's predictions, with segm mAP only when they carry masks."""
    if preds and "masks" in preds[0]:
        return evaluate_bbox_and_segm(preds, targets)
    return evaluate_bbox(preds, targets)


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
        strides: The head's per-level input strides. Defaults to ``(8, 16, 32)``.

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
        strides: tuple[int, int, int] = _STRIDES,
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
                be present.
            device: Device the model and image batches run on.

        Returns:
            ``{"e2e": stats, "nms": stats}`` where each ``stats`` is the named
            12-metric dict of :func:`evaluate_bbox`, or the 24-entry bbox+segm
            dict of :func:`evaluate_bbox_and_segm` when the model has a mask
            branch.
        """
        self._model.to(device).eval()
        e2e_preds: list[dict[str, Tensor]] = []
        nms_preds: list[dict[str, Tensor]] = []
        gt_targets: list[dict[str, Tensor]] = []
        with torch.no_grad():
            for images, image_ids, orig_sizes in dataloader:
                e2e_batch, nms_batch = self._predict_batch(images.to(device), device, orig_sizes)
                e2e_preds.extend(e2e_batch)
                nms_preds.extend(nms_batch)
                gt_targets.extend(targets[int(image_id)] for image_id in image_ids)
        return {"e2e": _score_path(e2e_preds, gt_targets), "nms": _score_path(nms_preds, gt_targets)}

    def _predict_batch(
        self,
        images: Tensor,
        device: torch.device,
        orig_sizes: Sequence[tuple[int, int]],
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        """Forward once, then build both paths' prediction dicts from the shared outputs."""
        head_out, prototypes = _segment_parts(self._model(images))
        anchor_points, strides = self._anchor_grid(images.shape[-2:], device)
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

    def _anchor_grid(self, size: torch.Size, device: torch.device) -> tuple[Tensor, Tensor]:
        """Build the ``(anchor_points, stride_per_anchor)`` grid for a canvas size."""
        height, width = int(size[0]), int(size[1])
        feature_sizes = [(height // stride, width // stride) for stride in self._strides]
        points, strides = make_anchor_points(feature_sizes, list(self._strides))
        return points.to(device), strides.to(device)

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
