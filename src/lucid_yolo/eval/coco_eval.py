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

The ground truth is supplied as torchmetrics target dicts
(``{image_id: {"boxes": ..., "labels": ...}}`` in original image coordinates,
category-id labels) rather than a ground-truth index object: torchmetrics scores
predictions directly against target tensors.

Provenance: R1 sec. 3.2.1, R1 Table 7, R1 sec. 4.4, R22, R23. Assumptions: A9, A10.
"""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING

import torch
from torchmetrics.detection import MeanAveragePrecision

from lucid_yolo.assign.grid import make_anchor_points
from lucid_yolo.decode.common import BOX_CORNERS, SCORE_COLUMN, to_letterboxed_original

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Tensor, nn

    from lucid_yolo.data.letterbox import Letterbox

__all__ = ["DualPathEvaluator", "detections_to_predictions", "evaluate_bbox"]

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

#: Feature-level input-pixel strides of the P3/P4/P5 detection head (8, 16, 32).
_STRIDES: tuple[int, int, int] = (8, 16, 32)

#: Column index of the integral class label within the A9 detection tuple.
_LABEL_COLUMN = 5


def detections_to_predictions(
    detections: Tensor,
    label_to_category: Mapping[int, int],
    score_floor: float = 0.0,
) -> list[dict[str, Tensor]]:
    """Convert a fixed-size A9 detection batch into torchmetrics prediction dicts.

    Each row of the ``(B, 300, 6)`` batch is the A9 tuple
    ``[x1, y1, x2, y2, score, class]``. A row survives when its ``score`` is
    strictly above ``score_floor`` — which, at the default ``0.0``, drops the
    score-zero padding rows that fill the fixed shape. The ``xyxy`` corners are
    kept as-is (torchmetrics is driven with ``box_format="xyxy"``) and each
    contiguous class label is mapped **back** to its original COCO category id via
    ``label_to_category``.

    Args:
        detections: Detections of shape ``(B, 300, 6)`` (or any ``(B, N, 6)``),
            the shared output of either decode path.
        label_to_category: Mapping from contiguous class label to original COCO
            category id (e.g.
            :attr:`~lucid_yolo.data.coco.CocoDetectionDataset.label_to_category_id`).
        score_floor: Rows with ``score`` at or below this are dropped. Defaults to
            ``0.0`` (drop only the score-zero padding rows).

    Returns:
        A length-``B`` list of prediction dicts, each with ``boxes`` (``(M, 4)``
        ``xyxy``), ``scores`` (``(M,)``) and ``labels`` (``(M,)`` long, category
        ids) — the per-image shape :class:`MeanAveragePrecision` consumes.

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
    """
    dense = detections.detach().to(device="cpu", dtype=torch.float32)
    return [_image_to_prediction(image_detections, label_to_category, score_floor) for image_detections in dense]


def _image_to_prediction(
    detections: Tensor,
    label_to_category: Mapping[int, int],
    score_floor: float,
) -> dict[str, Tensor]:
    """Convert one image's ``(N, 6)`` detections into a prediction dict (padding dropped)."""
    kept = detections[detections[:, SCORE_COLUMN] > score_floor]
    labels = torch.tensor(
        [label_to_category[int(label)] for label in kept[:, _LABEL_COLUMN]],
        dtype=torch.long,
    )
    return {
        "boxes": kept[:, :BOX_CORNERS],
        "scores": kept[:, SCORE_COLUMN],
        "labels": labels,
    }


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
    metric = MeanAveragePrecision(backend="faster_coco_eval", box_format="xyxy")
    # The fixed 300-row decoder output routinely exceeds COCO's top-100 detection
    # cap; keeping only the 100 highest-scoring per image is the standard protocol
    # (COCOeval's maxDets), not a misconfiguration, so silence the per-call warning.
    metric.warn_on_many_detections = False
    with contextlib.redirect_stdout(io.StringIO()):
        metric.update(preds, targets)
        computed = metric.compute()
    return {key: float(computed[key]) for key in _METRIC_KEYS}


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

    The ``dataloader`` passed to :meth:`evaluate` yields
    ``(images, image_ids, orig_sizes)`` batches, where ``images`` is a letterboxed
    ``(B, 3, H, W)`` tensor (``H``/``W`` divisible by 32), ``image_ids`` are the
    ``B`` COCO image ids, and ``orig_sizes`` are the ``B`` original ``(height,
    width)`` pairs the boxes are mapped back onto. The ground truth is a mapping
    from image id to a torchmetrics target dict in the same original coordinates.

    Args:
        model: Any module whose forward maps ``(B, 3, H, W)`` images to a
            :class:`~lucid_yolo.models.heads.detect.DualHeadOutput` (e.g.
            :class:`~lucid_yolo.ptl.module.DetectionLitModule` or a bare
            backbone/neck/head composition).
        e2e_decoder: The one-to-one E2E decoder, called
            ``(o2o_cls, o2o_box, anchor_points, strides) -> (B, 300, 6)``.
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
            12-metric dict of :func:`evaluate_bbox`.
        """
        self._model.to(device).eval()
        e2e_preds: list[dict[str, Tensor]] = []
        nms_preds: list[dict[str, Tensor]] = []
        gt_targets: list[dict[str, Tensor]] = []
        with torch.no_grad():
            for images, image_ids, orig_sizes in dataloader:
                e2e_batch, nms_batch = self._decode_batch(images.to(device), device)
                e2e_batch = self._to_original(e2e_batch, images.shape[-2:], orig_sizes)
                nms_batch = self._to_original(nms_batch, images.shape[-2:], orig_sizes)
                e2e_preds.extend(detections_to_predictions(e2e_batch, self._label_to_category))
                nms_preds.extend(detections_to_predictions(nms_batch, self._label_to_category))
                gt_targets.extend(targets[int(image_id)] for image_id in image_ids)
        return {"e2e": evaluate_bbox(e2e_preds, gt_targets), "nms": evaluate_bbox(nms_preds, gt_targets)}

    def _decode_batch(self, images: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
        """Forward once, then decode both paths from the shared dense outputs."""
        head_out = self._model(images)
        anchor_points, strides = self._anchor_grid(images.shape[-2:], device)
        e2e_batch = self._e2e_decoder(head_out.o2o_cls, head_out.o2o_box, anchor_points, strides)
        nms_batch = self._nms_decoder(head_out.o2m_cls, head_out.o2m_box, anchor_points, strides)
        return e2e_batch, nms_batch

    def _anchor_grid(self, size: torch.Size, device: torch.device) -> tuple[Tensor, Tensor]:
        """Build the ``(anchor_points, stride_per_anchor)`` grid for a canvas size."""
        height, width = int(size[0]), int(size[1])
        feature_sizes = [(height // stride, width // stride) for stride in self._strides]
        points, strides = make_anchor_points(feature_sizes, list(self._strides))
        return points.to(device), strides.to(device)

    def _to_original(
        self,
        detections: Tensor,
        letterboxed_size: torch.Size,
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
