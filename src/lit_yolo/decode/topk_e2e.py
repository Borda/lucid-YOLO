# SPDX-License-Identifier: Apache-2.0
"""Score-based top-k end-to-end decoding for the one-to-one branch (WP-041).

The inference-facing decode of the suppression-free path. This module deliberately
carries no suppression-operator token in its text, so the WP-041 meta-test can pin
the property structurally by reading the source. Over the one-to-one branch's dense
outputs it ranks anchors purely by classification confidence and keeps the top
``k``; there is **no IoU computation and no non-maximum suppression** — two heavily
overlapping high-score boxes both survive, which is the defining property of the
end-to-end path (R3 sec. 4). The detection cap is 300 (R1 sec. 3.2.1, A9).

The pipeline is: sigmoid the class logits, decode the raw ltrb distances into
``xyxy`` boxes with :func:`~lit_yolo.models.heads.detect.decode_ltrb`, reduce to
the score-ranked detections with
:func:`~lit_yolo.models.heads.detect.o2o_topk`, pad to a **fixed** ``(B, 300,
6)`` shape, then optionally zero the score of entries below a confidence
threshold. The fixed-size output is the export-friendly contract: the shape does
not depend on the anchor count or on how many detections clear the threshold, so
a traced/exported graph has a static output shape. Rows beyond the available
detections (or below the threshold) carry ``score == 0`` and are ignored by an
evaluation loop that filters on score.

The A9 detection tuple is ``[x1, y1, x2, y2, score, class]`` with ``score`` in
``[0, 1]`` and ``class`` the integral class index stored as a float.

Provenance: R3 sec. 4, R1 sec. 3.2.1. Assumptions: A9.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lit_yolo.data.letterbox import Letterbox
from lit_yolo.models.heads.detect import decode_ltrb, o2o_topk

__all__ = ["TopKDecoder", "to_letterboxed_original"]

#: Width of the A9 detection tuple ``[x1, y1, x2, y2, score, class]``.
_DET_WIDTH = 6

#: Number of box-coordinate columns (the ``xyxy`` corners) of a detection tuple.
_BOX_CORNERS = 4

#: Column index of the confidence score within the A9 detection tuple.
_SCORE_COLUMN = 4

#: Default per-image detection cap (R1 sec. 3.2.1, R3 sec. 4, A9).
_DEFAULT_TOPK = 300


class TopKDecoder(nn.Module):
    """Score-based top-k end-to-end decoder for the one-to-one branch (A9).

    Turns the one-to-one branch's raw dense outputs into a fixed-size batch of
    score-ranked detections without any IoU computation or non-maximum
    suppression (R3 sec. 4). The module owns no parameters — it is a pure
    functional transform wrapped as an :class:`~torch.nn.Module` so it composes
    into an inference graph and traces/exports cleanly with a static output
    shape.

    Args:
        k: Maximum detections kept per image; also the fixed output length.
            Defaults to 300 (R1 sec. 3.2.1, R3 sec. 4).
        conf_threshold: Detections whose confidence is strictly below this value
            have their score zeroed (the row is kept so the output shape is
            fixed). Defaults to ``0.0`` — pure top-k, nothing zeroed.

    Examples:
        >>> import torch
        >>> from lit_yolo.assign.grid import make_anchor_points
        >>> points, strides = make_anchor_points([(2, 2)], [8])
        >>> cls_logits = torch.full((1, 4, 3), -5.0)
        >>> cls_logits[0, 1, 2] = 5.0  # one confident anchor, class 2
        >>> raw_ltrb = torch.ones(1, 4, 4)
        >>> decoder = TopKDecoder(k=2)
        >>> detections = decoder(cls_logits, raw_ltrb, points, strides)
        >>> detections.shape  # fixed (B, k, 6) even though only 4 anchors exist
        torch.Size([1, 2, 6])
        >>> int(detections[0, 0, 5])  # class of the top detection
        2
    """

    def __init__(self, k: int = _DEFAULT_TOPK, conf_threshold: float = 0.0) -> None:
        super().__init__()
        self.k = k
        self.conf_threshold = conf_threshold

    def forward(self, cls_logits: Tensor, raw_ltrb: Tensor, anchor_points: Tensor, strides: Tensor) -> Tensor:
        """Decode raw one-to-one outputs into fixed-size top-k detections.

        Args:
            cls_logits: Raw one-to-one class logits of shape ``(B, A, C)``.
            raw_ltrb: Raw one-to-one ltrb distances of shape ``(B, A, 4)``
                (order l, t, r, b), aligned with ``cls_logits`` on the anchor
                axis.
            anchor_points: Anchor-centre ``(x, y)`` coordinates of shape
                ``(A, 2)`` in input pixels, as returned by
                :func:`~lit_yolo.assign.grid.make_anchor_points`.
            strides: Per-anchor level stride of shape ``(A,)``.

        Returns:
            Detections of shape ``(B, k, 6)`` whose last axis is the A9 tuple
            ``[x1, y1, x2, y2, score, class]``, sorted by descending score with
            score-zero padding rows filling any shortfall below ``k``.

        Examples:
            >>> import torch
            >>> from lit_yolo.assign.grid import make_anchor_points
            >>> points, strides = make_anchor_points([(1, 2)], [8])
            >>> cls_logits = torch.zeros(1, 2, 1)
            >>> raw_ltrb = torch.zeros(1, 2, 4)
            >>> TopKDecoder(k=5)(cls_logits, raw_ltrb, points, strides).shape
            torch.Size([1, 5, 6])
        """
        boxes = decode_ltrb(raw_ltrb, anchor_points, strides)
        detections = o2o_topk(cls_logits, boxes, k=self.k)
        detections = self._pad_to_k(detections)
        if self.conf_threshold > 0.0:
            detections = self._zero_below_threshold(detections)
        return detections

    def _pad_to_k(self, detections: Tensor) -> Tensor:
        """Pad ``detections`` to a fixed length of ``k`` rows with zero rows.

        :func:`o2o_topk` returns ``min(k, A)`` rows; when the anchor count ``A``
        is below ``k`` the shortfall is filled with all-zero rows (score 0)
        appended after the ranked detections, so the output length is always
        ``k`` and the descending-score ordering is preserved.

        Args:
            detections: Ranked detections of shape ``(B, min(k, A), 6)``.

        Returns:
            Detections of shape ``(B, k, 6)``.
        """
        batch, kept, _ = detections.shape
        if kept >= self.k:
            return detections
        padding = detections.new_zeros(batch, self.k - kept, _DET_WIDTH)
        return torch.cat((detections, padding), dim=1)

    def _zero_below_threshold(self, detections: Tensor) -> Tensor:
        """Zero the score of detections below :attr:`conf_threshold`.

        The row is retained (its box and class untouched) so the output shape
        stays fixed; only the score column is masked. Because the scores arrive
        sorted descending, the below-threshold entries form a contiguous suffix,
        so zeroing them keeps the column non-increasing.

        Args:
            detections: Detections of shape ``(B, k, 6)``, score-sorted
                descending.

        Returns:
            Detections of shape ``(B, k, 6)`` with sub-threshold scores set to 0.
        """
        scores = detections[..., _SCORE_COLUMN : _SCORE_COLUMN + 1]
        kept = (scores >= self.conf_threshold).to(scores.dtype)
        masked_scores = scores * kept
        return torch.cat(
            (detections[..., :_SCORE_COLUMN], masked_scores, detections[..., _SCORE_COLUMN + 1 :]),
            dim=-1,
        )


def to_letterboxed_original(
    detections: Tensor,
    orig_size: tuple[int, int],
    letterboxed_size: tuple[int, int],
    allow_upscale: bool = True,
) -> Tensor:
    """Un-letterbox detection boxes back to original-image coordinates (A10).

    Eval-time hook: maps the ``xyxy`` box corners of each detection from the
    letterboxed canvas the model saw back to the original image via the exact
    analytic inverse of :class:`~lit_yolo.data.letterbox.Letterbox`. The score
    and class columns pass through untouched. Padding / sub-threshold rows
    (score 0) are mapped like any other row; they are identified downstream by
    their zero score, not by their coordinates.

    Args:
        detections: Detections of shape ``(B, N, 6)`` with the A9 tuple
            ``[x1, y1, x2, y2, score, class]`` in letterboxed-canvas pixels.
        orig_size: Original image ``(height, width)``.
        letterboxed_size: Letterboxed canvas ``(height, width)`` the boxes live
            in.
        allow_upscale: The letterbox ``allow_upscale`` setting used at resize
            time; must match so the inverse recovers the exact geometry.
            Defaults to ``True`` (the transform default).

    Returns:
        Detections of shape ``(B, N, 6)`` with box corners in original-image
        coordinates and score/class unchanged.

    Examples:
        >>> import torch
        >>> # A 2x4 image letterboxed into a 4x4 canvas gains 1px top/bottom pads.
        >>> boxes = torch.tensor([[[0.0, 1.0, 4.0, 3.0, 0.9, 0.0]]])
        >>> mapped = to_letterboxed_original(boxes, orig_size=(2, 4), letterboxed_size=(4, 4))
        >>> mapped[0, 0, :4]
        tensor([0., 0., 4., 2.])
        >>> mapped[0, 0, 4:]  # score and class survive the round trip
        tensor([0.9000, 0.0000])
    """
    batch, num_det, _ = detections.shape
    letterbox = Letterbox(letterboxed_size, allow_upscale=allow_upscale)
    corner_points = detections[..., :_BOX_CORNERS].reshape(-1, 2)
    mapped_points = letterbox.inverse_map(corner_points, orig_size, letterboxed_size)
    mapped_boxes = mapped_points.reshape(batch, num_det, _BOX_CORNERS)
    return torch.cat((mapped_boxes, detections[..., _BOX_CORNERS:]), dim=-1)
