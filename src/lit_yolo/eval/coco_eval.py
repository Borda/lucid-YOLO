# SPDX-License-Identifier: Apache-2.0
"""pycocotools bbox mAP evaluation for both inference paths (WP-043).

The detection acceptance instrument of blueprint sec. 5.11: pycocotools
``bbox`` mAP50-95 on val2017, reported for **both** decode paths (E2E and
non-E2E) from one checkpoint in one pass, exactly as [R1, Table 7] does. The
expected E2E deficit of 0.6-0.8 AP ([R1] sec. 4.4) is itself a claim measured
against this instrument later; this module only builds the measurement.

Three pieces compose the protocol:

- :func:`detections_to_coco` turns a fixed-size ``(B, 300, 6)`` A9 detection
  batch (the shared output of :class:`~lit_yolo.decode.topk_e2e.TopKDecoder` and
  :class:`~lit_yolo.decode.nms_path.NMSDecoder`) into pycocotools result dicts:
  ``xyxy`` corners become ``xywh``, the contiguous class label is mapped **back**
  to its original COCO category id, and score-zero padding rows are dropped.
- :func:`evaluate_bbox` wraps :class:`pycocotools.cocoeval.COCOeval` and returns
  the 12 summary statistics as a named dict.
- :class:`DualPathEvaluator` runs the model forward **once per batch**, decodes
  both paths from the same dense outputs, un-letterboxes each back to original
  image coordinates (A10), and returns ``{"e2e": ..., "nms": ...}`` — one
  command, one checkpoint, both paths (the sec. 5.11 contract).

pycocotools writes progress banners to ``stdout``; the wrappers here redirect
that chatter so a caller (or a test) sees only the returned numbers.

Provenance: R1 sec. 3.2.1, R1 Table 7, R1 sec. 4.4. Assumptions: A9, A10.
"""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from lit_yolo.assign.grid import make_anchor_points
from lit_yolo.decode.common import BOX_CORNERS, SCORE_COLUMN, to_letterboxed_original

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Tensor, nn

    from lit_yolo.data.letterbox import Letterbox

__all__ = ["DualPathEvaluator", "detections_to_coco", "evaluate_bbox"]

#: The 12 :attr:`COCOeval.stats` in fixed pycocotools order (AP first, AR after).
_STAT_NAMES: tuple[str, ...] = (
    "map50_95",
    "map50",
    "map75",
    "map_small",
    "map_medium",
    "map_large",
    "recall_1",
    "recall_10",
    "recall_100",
    "recall_small",
    "recall_medium",
    "recall_large",
)

#: Feature-level input-pixel strides of the P3/P4/P5 detection head (8, 16, 32).
_STRIDES: tuple[int, int, int] = (8, 16, 32)

#: Column index of the integral class label within the A9 detection tuple.
_LABEL_COLUMN = 5


def detections_to_coco(
    detections: Tensor,
    image_ids: Sequence[int],
    label_to_category: Mapping[int, int],
    score_floor: float = 0.0,
) -> list[dict[str, object]]:
    """Convert a fixed-size A9 detection batch into pycocotools result dicts.

    Each row of the ``(B, 300, 6)`` batch is the A9 tuple
    ``[x1, y1, x2, y2, score, class]``. A row is emitted as one result dict when
    its ``score`` is strictly above ``score_floor`` — which, at the default
    ``0.0``, drops the score-zero padding rows that fill the fixed shape. The
    ``xyxy`` corners become a COCO ``xywh`` box and the contiguous class label is
    mapped **back** to its original COCO category id via ``label_to_category``.

    Args:
        detections: Detections of shape ``(B, 300, 6)`` (or any ``(B, N, 6)``),
            the shared output of either decode path.
        image_ids: Length-``B`` COCO image ids, aligned with the batch axis.
        label_to_category: Mapping from contiguous class label to original COCO
            category id (e.g.
            :attr:`~lit_yolo.data.coco.CocoDetectionDataset.label_to_category_id`).
        score_floor: Rows with ``score`` at or below this are dropped. Defaults to
            ``0.0`` (drop only the score-zero padding rows).

    Returns:
        A flat list of result dicts, each with ``image_id``, ``category_id``,
        ``bbox`` (``[x, y, w, h]``) and ``score`` keys.

    Raises:
        ValueError: If ``detections`` batch size does not match ``image_ids``.

    Examples:
        >>> import torch
        >>> dets = torch.tensor(
        ...     [[[0.0, 0.0, 4.0, 4.0, 0.9, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
        ... )  # one real detection, one padding row
        >>> records = detections_to_coco(dets, image_ids=[7], label_to_category={1: 42})
        >>> len(records)  # padding row dropped
        1
        >>> records[0]["bbox"], records[0]["category_id"], records[0]["image_id"]
        ([0.0, 0.0, 4.0, 4.0], 42, 7)
    """
    if detections.shape[0] != len(image_ids):
        raise ValueError(f"batch size {detections.shape[0]} does not match {len(image_ids)} image ids")
    dense = detections.detach().to(device="cpu", dtype=torch.float64)
    results: list[dict[str, object]] = []
    for image_detections, image_id in zip(dense, image_ids, strict=True):
        results.extend(_image_to_coco(image_detections, int(image_id), label_to_category, score_floor))
    return results


def _image_to_coco(
    detections: Tensor,
    image_id: int,
    label_to_category: Mapping[int, int],
    score_floor: float,
) -> list[dict[str, object]]:
    """Convert one image's ``(N, 6)`` detections into result dicts (padding dropped)."""
    records: list[dict[str, object]] = []
    for row in detections:
        score = float(row[SCORE_COLUMN])
        if score <= score_floor:
            continue
        x1, y1, x2, y2 = (float(value) for value in row[:BOX_CORNERS])
        records.append(
            {
                "image_id": image_id,
                "category_id": label_to_category[int(row[_LABEL_COLUMN])],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            }
        )
    return records


def evaluate_bbox(coco_gt: COCO, results: list[dict[str, object]]) -> dict[str, float]:
    """Score detection results against ground truth with COCOeval ``bbox`` mAP.

    Runs the standard :class:`pycocotools.cocoeval.COCOeval` ``bbox`` pipeline
    (evaluate, accumulate, summarize) over ``results`` and returns its 12 summary
    statistics as a named dict (see :data:`_STAT_NAMES`). pycocotools' progress
    banners are redirected away from ``stdout``. An empty ``results`` list yields
    an all-zero stat dict rather than driving pycocotools through a degenerate
    empty-result path.

    Args:
        coco_gt: Ground-truth :class:`pycocotools.coco.COCO` object.
        results: Result dicts as produced by :func:`detections_to_coco`.

    Returns:
        A dict from statistic name to value: ``map50_95``, ``map50``, ``map75``,
        the small/medium/large AP breakdown, and the six average-recall entries.

    Examples:
        >>> from pycocotools.coco import COCO  # doctest: +SKIP
        >>> stats = evaluate_bbox(coco_gt, results)  # doctest: +SKIP
        >>> round(stats["map50_95"], 3)  # doctest: +SKIP
        1.0
    """
    if not results:
        return dict.fromkeys(_STAT_NAMES, 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    stats = coco_eval.stats
    return {name: float(stats[index]) for index, name in enumerate(_STAT_NAMES)}


class DualPathEvaluator:
    """Evaluate both decode paths from one checkpoint in a single loader pass.

    Implements the blueprint sec. 5.11 contract: for each batch the model runs
    **once**, both the suppression-free E2E path
    (:class:`~lit_yolo.decode.topk_e2e.TopKDecoder` over the one-to-one branch)
    and the non-E2E path (:class:`~lit_yolo.decode.nms_path.NMSDecoder` over the
    dense one-to-many branch) are decoded from the same
    :class:`~lit_yolo.models.heads.detect.DualHeadOutput`, each un-letterboxed to
    original image coordinates (A10), and accumulated into separate result lists.
    :meth:`evaluate` returns ``{"e2e": ..., "nms": ...}`` scored by
    :func:`evaluate_bbox`.

    The ``dataloader`` passed to :meth:`evaluate` yields
    ``(images, image_ids, orig_sizes)`` batches, where ``images`` is a letterboxed
    ``(B, 3, H, W)`` tensor (``H``/``W`` divisible by 32), ``image_ids`` are the
    ``B`` COCO image ids, and ``orig_sizes`` are the ``B`` original ``(height,
    width)`` pairs the boxes are mapped back onto.

    Args:
        model: Any module whose forward maps ``(B, 3, H, W)`` images to a
            :class:`~lit_yolo.models.heads.detect.DualHeadOutput` (e.g.
            :class:`~lit_yolo.ptl.module.DetectionLitModule` or a bare
            backbone/neck/head composition).
        e2e_decoder: The one-to-one E2E decoder, called
            ``(o2o_cls, o2o_box, anchor_points, strides) -> (B, 300, 6)``.
        nms_decoder: The dense-branch NMS decoder, called with the one-to-many
            outputs and the same signature.
        label_to_category: Mapping from contiguous class label to original COCO
            category id, applied to both paths' detections.
        letterbox: The validation :class:`~lit_yolo.data.letterbox.Letterbox`
            whose geometry (its ``allow_upscale`` setting) inverts the resize.
        strides: The head's per-level input strides. Defaults to ``(8, 16, 32)``.

    Examples:
        >>> evaluator = DualPathEvaluator(  # doctest: +SKIP
        ...     model, TopKDecoder(), NMSDecoder(), dataset.label_to_category_id, letterbox
        ... )
        >>> report = evaluator.evaluate(val_loader, coco_gt, torch.device("cpu"))  # doctest: +SKIP
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
        coco_gt: COCO,
        device: torch.device,
    ) -> dict[str, dict[str, float]]:
        """Run both paths over ``dataloader`` and return their bbox mAP reports.

        Args:
            dataloader: Iterable of ``(images, image_ids, orig_sizes)`` batches
                (see the class docstring for the batch contract).
            coco_gt: Ground-truth :class:`pycocotools.coco.COCO` object.
            device: Device the model and image batches run on.

        Returns:
            ``{"e2e": stats, "nms": stats}`` where each ``stats`` is the named
            12-metric dict of :func:`evaluate_bbox`.
        """
        self._model.to(device).eval()
        e2e_results: list[dict[str, object]] = []
        nms_results: list[dict[str, object]] = []
        with torch.no_grad():
            for images, image_ids, orig_sizes in dataloader:
                e2e_batch, nms_batch = self._decode_batch(images.to(device), device)
                e2e_batch = self._to_original(e2e_batch, images.shape[-2:], orig_sizes)
                nms_batch = self._to_original(nms_batch, images.shape[-2:], orig_sizes)
                e2e_results.extend(detections_to_coco(e2e_batch, image_ids, self._label_to_category))
                nms_results.extend(detections_to_coco(nms_batch, image_ids, self._label_to_category))
        return {"e2e": evaluate_bbox(coco_gt, e2e_results), "nms": evaluate_bbox(coco_gt, nms_results)}

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
