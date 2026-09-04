# SPDX-License-Identifier: Apache-2.0
"""Confidence-threshold + class-wise NMS decoding for the dense branch (WP-042).

The non-E2E inference decode: it turns the **one-to-many** branch's dense outputs
into detections the classic way — a confidence threshold followed by class-wise
non-maximum suppression — to produce the "mAP (non-E2E)" comparison column
against the suppression-free one-to-one path (blueprint sec. 5.5). Where
:class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` keeps two heavily overlapping
same-class high-score boxes, this path suppresses the lower-scoring one; that
contrast is the whole point of reporting both columns.

The pipeline over the dense outputs is:

1. sigmoid the class logits and take, per anchor, its single best class and that
   class's score (**single-label** convention — one detection per surviving
   anchor; a multi-label variant that emits one detection per class above the
   threshold is out of scope here);
2. drop anchors whose best-class score is below ``conf_threshold`` (default
   ``0.001``, the usual permissive evaluation-time threshold that lets the mAP
   integration see the low-confidence tail);
3. class-wise NMS via :func:`torchvision.ops.batched_nms` at ``iou_threshold``
   (default ``0.7``) — boxes of *different* classes never suppress each other, so
   overlapping detections of distinct classes all survive;
4. keep at most ``max_det`` detections per image (default ``300``; R1
   sec. 3.2.1);
5. pad to a **fixed** ``(B, max_det, 6)`` shape with score-zero rows.

The fixed-size A9 output tuple ``[x1, y1, x2, y2, score, class]`` is identical to
the one-to-one path's, so both decoders feed one evaluator interchangeably. Rows
beyond the survivors carry ``score == 0`` and are ignored by a score-filtering
evaluation loop. :func:`torchvision.ops.batched_nms` returns the kept indices in
descending score order, so the score column comes out non-increasing.

Provenance: R1 sec. 3.2.1, R3 sec. 4. Assumptions: A9.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.ops import batched_nms

from lucid_yolo.decode.common import pad_anchor_indices, pad_detections
from lucid_yolo.models.heads.detect import decode_ltrb

__all__ = ["NMSDecoder"]

#: Default per-image detection cap (R1 sec. 3.2.1, R3 sec. 4, A9).
_DEFAULT_MAX_DET = 300

#: Default evaluation-time confidence threshold — permissive so the mAP
#: integration sees the low-confidence tail.
_DEFAULT_CONF_THRESHOLD = 0.001

#: Default class-wise NMS IoU threshold.
_DEFAULT_IOU_THRESHOLD = 0.7


class NMSDecoder(nn.Module):
    """Confidence-threshold + class-wise NMS decoder for the dense branch (A9).

    Turns the one-to-many branch's raw dense outputs into a fixed-size batch of
    score-ranked detections via a confidence threshold and class-wise
    non-maximum suppression (the non-E2E path, blueprint sec. 5.5). Emits the
    same ``(B, max_det, 6)`` A9 tuple as
    :class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` so both paths feed one
    evaluator. The module owns no parameters — it is a pure functional transform
    wrapped as an :class:`~torch.nn.Module`.

    Args:
        conf_threshold: Anchors whose best-class score is strictly below this
            value are dropped before NMS. Defaults to ``0.001`` (the permissive
            evaluation convention).
        iou_threshold: Class-wise NMS IoU threshold; a lower-scoring box is
            suppressed when its IoU with a kept higher-scoring box of the *same*
            class exceeds this. Defaults to ``0.7``.
        max_det: Maximum detections kept per image; also the fixed output length.
            Defaults to ``300`` (R1 sec. 3.2.1).

    Examples:
        >>> import torch
        >>> from lucid_yolo.assign.grid import make_anchor_points
        >>> points, strides = make_anchor_points([(2, 2)], [8])
        >>> cls_logits = torch.full((1, 4, 3), -5.0)
        >>> cls_logits[0, 1, 2] = 5.0  # one confident anchor, class 2
        >>> raw_ltrb = torch.ones(1, 4, 4)
        >>> decoder = NMSDecoder(max_det=100)
        >>> detections = decoder(cls_logits, raw_ltrb, points, strides)
        >>> detections.shape  # fixed (B, max_det, 6)
        torch.Size([1, 100, 6])
        >>> int(detections[0, 0, 5])  # class of the top detection
        2
    """

    def __init__(
        self,
        conf_threshold: float = _DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = _DEFAULT_IOU_THRESHOLD,
        max_det: int = _DEFAULT_MAX_DET,
    ) -> None:
        super().__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

    def forward(self, cls_logits: Tensor, raw_ltrb: Tensor, anchor_points: Tensor, strides: Tensor) -> Tensor:
        """Decode raw dense outputs into fixed-size NMS-filtered detections.

        Args:
            cls_logits: Raw dense class logits of shape ``(B, A, C)``.
            raw_ltrb: Raw dense ltrb distances of shape ``(B, A, 4)`` (order l, t,
                r, b), aligned with ``cls_logits`` on the anchor axis.
            anchor_points: Anchor-centre ``(x, y)`` coordinates of shape
                ``(A, 2)`` in input pixels, as returned by
                :func:`~lucid_yolo.assign.grid.make_anchor_points`.
            strides: Per-anchor level stride of shape ``(A,)``.

        Returns:
            Detections of shape ``(B, max_det, 6)`` whose last axis is the A9
            tuple ``[x1, y1, x2, y2, score, class]``, sorted by descending score
            with score-zero padding rows filling any shortfall below ``max_det``.

        Examples:
            >>> import torch
            >>> from lucid_yolo.assign.grid import make_anchor_points
            >>> points, strides = make_anchor_points([(1, 2)], [8])
            >>> cls_logits = torch.zeros(1, 2, 1)
            >>> raw_ltrb = torch.zeros(1, 2, 4)
            >>> NMSDecoder(max_det=5)(cls_logits, raw_ltrb, points, strides).shape
            torch.Size([1, 5, 6])
        """
        detections, _ = self.decode_with_indices(cls_logits, raw_ltrb, anchor_points, strides)
        return detections

    def decode_with_indices(
        self,
        cls_logits: Tensor,
        raw_ltrb: Tensor,
        anchor_points: Tensor,
        strides: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode as :meth:`forward` does, and report the source anchor of each row.

        The whole decode lives here and :meth:`forward` is its box-only view, so
        the two cannot report different survivor sets. The extra return exists for
        the same reason as its one-to-one twin
        (:meth:`~lucid_yolo.decode.topk_e2e.TopKDecoder.decode_with_indices`): the
        segmentation decode (WP-053b) gathers each detection's mask coefficients
        from the dense ``(B, A, K)`` map by these indices. On this path the kept
        set is chosen by the threshold and the suppression, not by a ranking any
        caller could reproduce, so the indices have to be carried out through the
        **same** ``keep``/``order`` selection that built the boxes — recomputing a
        second selection outside is exactly how a mask ends up describing another
        anchor's object.

        Args:
            cls_logits: Raw dense class logits of shape ``(B, A, C)``.
            raw_ltrb: Raw dense ltrb distances of shape ``(B, A, 4)``, aligned
                with ``cls_logits`` on the anchor axis.
            anchor_points: Anchor-centre ``(x, y)`` coordinates of shape
                ``(A, 2)`` in input pixels.
            strides: Per-anchor level stride of shape ``(A,)``.

        Returns:
            A pair ``(detections, anchor_indices)``: the ``(B, max_det, 6)`` A9
            batch :meth:`forward` returns, and the ``(B, max_det)`` long tensor
            naming the anchor each row came from, with
            :data:`~lucid_yolo.decode.common.PAD_ANCHOR_INDEX` on the padding rows.

        Examples:
            >>> import torch
            >>> from lucid_yolo.assign.grid import make_anchor_points
            >>> points, strides = make_anchor_points([(1, 2)], [8])
            >>> cls_logits = torch.tensor([[[-5.0], [5.0]]])  # only anchor 1 clears the threshold
            >>> raw_ltrb = torch.full((1, 2, 4), 0.25)  # a real 4 px box on each anchor
            >>> decoder = NMSDecoder(conf_threshold=0.5, max_det=3)
            >>> detections, anchors = decoder.decode_with_indices(cls_logits, raw_ltrb, points, strides)
            >>> detections.shape
            torch.Size([1, 3, 6])
            >>> anchors  # anchor 1 survives; the remaining rows are padding
            tensor([[ 1, -1, -1]])

            A box has to enclose an area to be a detection: zero ``ltrb`` distances put
            ``x2`` exactly on ``x1``, and such a row is dropped with the sub-threshold
            ones rather than spending one of the ``max_det`` slots.

            >>> flat = torch.zeros(1, 2, 4)
            >>> _, none_kept = decoder.decode_with_indices(cls_logits, flat, points, strides)
            >>> none_kept
            tensor([[-1, -1, -1]])
        """
        boxes = decode_ltrb(raw_ltrb, anchor_points, strides)  # (B, A, 4)
        confidence = cls_logits.sigmoid()
        scores, classes = confidence.max(dim=-1)  # both (B, A), single-label per anchor
        decoded = [self._decode_image(boxes[i], scores[i], classes[i]) for i in range(boxes.shape[0])]
        return torch.stack([image for image, _ in decoded]), torch.stack([index for _, index in decoded])

    def _decode_image(self, boxes: Tensor, scores: Tensor, classes: Tensor) -> tuple[Tensor, Tensor]:
        """Threshold, class-wise NMS, cap, and pad one image's anchors.

        Args:
            boxes: Decoded ``xyxy`` boxes of shape ``(A, 4)``.
            scores: Per-anchor best-class score of shape ``(A,)``, in ``[0, 1]``.
            classes: Per-anchor best-class index of shape ``(A,)`` (integral).

        Returns:
            A pair of the ``(max_det, 6)`` A9 detections ``[x1, y1, x2, y2,
            score, class]`` sorted by descending score and padded with score-zero
            rows, and the ``(max_det,)`` source anchor index of each row (padding
            rows carry :data:`~lucid_yolo.decode.common.PAD_ANCHOR_INDEX`). The
            index is threaded through the same two selections the boxes pass —
            the threshold's ``keep`` mask and the suppression's ``order`` — rather
            than recovered afterwards.
        """
        # Degenerate rows are dropped at the threshold, not at the decode: `decode_ltrb`
        # is a pure `(l, t, r, b) -> xyxy` transcription whose values are pinned by the
        # frozen goldens, so clamping there would move numbers this boundary can leave
        # untouched. `batched_nms` has no opinion on a box whose `x2 <= x1` — such a box
        # has non-positive area, so its IoU against everything is zero and it suppresses
        # nothing, which lets an untrained-region anchor cluster consume `max_det` slots
        # and push real detections out of the budget. Strict `>` therefore drops the
        # zero-area case as well as the inverted one; neither describes a region.
        # The mask stays folded into `keep` so the anchor indices below are threaded
        # through exactly one selection, which is this method's stated invariant.
        # Finiteness is asserted rather than inferred from the two comparisons. `NaN`
        # corners would indeed fail them, but an infinite pair does not: `l = r = inf`
        # decodes to `[-inf, -inf, inf, inf]`, which is correctly *ordered* and would
        # have passed as a box covering the plane.
        keep = (
            (scores >= self.conf_threshold)
            & boxes.isfinite().all(dim=-1)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        anchors = keep.nonzero(as_tuple=False).flatten()  # source anchor row of each survivor
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        order = batched_nms(boxes, scores, classes, self.iou_threshold)[: self.max_det]
        detection = torch.cat(
            (boxes[order], scores[order].unsqueeze(-1), classes[order].unsqueeze(-1).to(boxes.dtype)),
            dim=-1,
        )
        return pad_detections(detection, self.max_det), pad_anchor_indices(anchors[order], self.max_det)
