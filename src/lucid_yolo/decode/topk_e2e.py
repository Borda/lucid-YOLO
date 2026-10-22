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
``xyxy`` boxes with :func:`~lucid_yolo.models.heads.detect.decode_ltrb`, reduce to
the score-ranked detections with
:func:`~lucid_yolo.models.heads.detect.o2o_topk_with_indices`, pad to a **fixed** ``(B, 300,
6)`` shape, then optionally zero the score of entries below a confidence
threshold. The fixed-size output is the export-friendly contract: the shape does
not depend on the anchor count or on how many detections clear the threshold, so
a traced/exported graph has a static output shape. Rows beyond the available
detections (or below the threshold) carry ``score == 0`` and are ignored by an
evaluation loop that filters on score.

The A9 detection tuple is ``[x1, y1, x2, y2, score, class]`` with ``score`` in
``[0, 1]`` and ``class`` the integral class index stored as a float.

:class:`RotatedTopKDecoder` is the oriented twin over the same branch: canonical
rotated boxes from :func:`~lucid_yolo.models.heads.obb.decode_rboxes`, ranked by
:func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk_with_indices`, padded to the same
fixed ``k`` rows with the same anchor-index contract, emitting the A45 tuple
``[cx, cy, w, h, theta, score, class]``. It lives in this module so the structural pin on
this file's text covers both suppression-free decoders.

Provenance: R3 sec. 4, R1 sec. 3.2.1, R1 Eq. 13. Assumptions: A9, A23, A45.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lucid_yolo.decode.common import MAX_DETECTIONS, SCORE_COLUMN, pad_anchor_indices, pad_detections
from lucid_yolo.models.heads.detect import decode_ltrb, o2o_topk_with_indices
from lucid_yolo.models.heads.obb import decode_rboxes, o2o_rotated_topk_with_indices

__all__ = ["RotatedTopKDecoder", "TopKDecoder"]


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
        ```pycon
        >>> import torch
        >>> from lucid_yolo.assign.grid import make_anchor_points
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

        ```
    """

    def __init__(self, k: int = MAX_DETECTIONS, conf_threshold: float = 0.0) -> None:
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
                :func:`~lucid_yolo.assign.grid.make_anchor_points`.
            strides: Per-anchor level stride of shape ``(A,)``.

        Returns:
            Detections of shape ``(B, k, 6)`` whose last axis is the A9 tuple
            ``[x1, y1, x2, y2, score, class]``, sorted by descending score with
            score-zero padding rows filling any shortfall below ``k``.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.assign.grid import make_anchor_points
            >>> points, strides = make_anchor_points([(1, 2)], [8])
            >>> cls_logits = torch.zeros(1, 2, 1)
            >>> raw_ltrb = torch.zeros(1, 2, 4)
            >>> TopKDecoder(k=5)(cls_logits, raw_ltrb, points, strides).shape
            torch.Size([1, 5, 6])

            ```
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
        the two cannot report different selections. The extra return is what any
        per-anchor quantity outside the A9 tuple must be gathered by — the
        segmentation decode's mask coefficients (WP-053b): a coefficient row taken
        from a differently ranked selection would yield a plausible mask of the
        wrong object beside a correct box.

        Args:
            cls_logits: Raw one-to-one class logits of shape ``(B, A, C)``.
            raw_ltrb: Raw one-to-one ltrb distances of shape ``(B, A, 4)``,
                aligned with ``cls_logits`` on the anchor axis.
            anchor_points: Anchor-centre ``(x, y)`` coordinates of shape
                ``(A, 2)`` in input pixels.
            strides: Per-anchor level stride of shape ``(A,)``.

        Returns:
            A pair ``(detections, anchor_indices)``: the ``(B, k, 6)`` A9 batch
            :meth:`forward` returns, and the ``(B, k)`` long tensor naming the
            anchor each row came from, with
            :data:`~lucid_yolo.decode.common.PAD_ANCHOR_INDEX` on the padding rows.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.assign.grid import make_anchor_points
            >>> points, strides = make_anchor_points([(1, 2)], [8])
            >>> cls_logits = torch.tensor([[[-5.0], [5.0]]])  # anchor 1 is the confident one
            >>> raw_ltrb = torch.zeros(1, 2, 4)
            >>> detections, anchors = TopKDecoder(k=3).decode_with_indices(
            ...     cls_logits, raw_ltrb, points, strides
            ... )
            >>> detections.shape
            torch.Size([1, 3, 6])
            >>> anchors  # anchor 1 ranks first; the third row is padding
            tensor([[ 1,  0, -1]])

            ```
        """
        boxes = decode_ltrb(raw_ltrb, anchor_points, strides)
        detections, anchor_indices = o2o_topk_with_indices(cls_logits, boxes, k=self.k)
        detections = self._pad_to_k(detections)
        if self.conf_threshold > 0.0:
            detections = self._zero_below_threshold(detections)
        return detections, pad_anchor_indices(anchor_indices, self.k)

    def _pad_to_k(self, detections: Tensor) -> Tensor:
        """Pad ``detections`` to a fixed length of ``k`` rows with zero rows.

        :func:`o2o_topk_with_indices` returns ``min(k, A)`` rows; when the anchor count ``A``
        is below ``k`` the shortfall is filled with all-zero rows (score 0)
        appended after the ranked detections, so the output length is always
        ``k`` and the descending-score ordering is preserved. Delegates to the
        shared :func:`~lucid_yolo.decode.common.pad_detections` so both decode
        paths pad identically.

        Args:
            detections: Ranked detections of shape ``(B, min(k, A), 6)``.

        Returns:
            Detections of shape ``(B, k, 6)``.
        """
        return pad_detections(detections, self.k)

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
        scores = detections[..., SCORE_COLUMN : SCORE_COLUMN + 1]
        kept = (scores >= self.conf_threshold).to(scores.dtype)
        masked_scores = scores * kept
        return torch.cat(
            (detections[..., :SCORE_COLUMN], masked_scores, detections[..., SCORE_COLUMN + 1 :]),
            dim=-1,
        )


class RotatedTopKDecoder(nn.Module):
    """Score-based top-k end-to-end decoder for the one-to-one branch's oriented outputs (A45).

    The oriented twin of :class:`TopKDecoder`, and deliberately the same contract: the
    one-to-one branch's raw ltrb distances and raw angles (R1 Eq. 13) become canonical
    rotated boxes through :func:`~lucid_yolo.models.heads.obb.decode_rboxes` (A23), the
    anchors are ranked purely by classification confidence through
    :func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk_with_indices` — no overlap is
    computed and no box is dropped for overlapping another — and the result is padded
    to exactly ``k`` rows, the anchor indices beside it padded with
    :data:`~lucid_yolo.decode.common.PAD_ANCHOR_INDEX`.

    The padding is what this class adds over calling the ranking directly, which returns
    ``min(k, A)`` rows: on a canvas with fewer than ``k`` anchors (a side of 96 px or
    less at the three head strides) the row count would otherwise depend on the canvas,
    unlike every other decoder's. On a canvas with at least ``k`` anchors the ``k`` rows
    are the ranking's own, bit for bit. Rows beyond the available anchors carry
    ``score == 0``, so a caller filtering on score drops them.

    Args:
        k: Maximum detections kept per image; also the fixed output length. Defaults to
            :data:`~lucid_yolo.decode.common.MAX_DETECTIONS` (A9).

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.assign.grid import make_anchor_points
        >>> points, strides = make_anchor_points([(1, 2)], [8])
        >>> cls_logits = torch.tensor([[[-5.0], [5.0]]])  # anchor 1 is the confident one
        >>> raw_ltrb = torch.ones(1, 2, 4)
        >>> raw_angles = torch.zeros(1, 2, 1)
        >>> detections = RotatedTopKDecoder(k=3)(cls_logits, raw_ltrb, raw_angles, points, strides)
        >>> detections.shape  # fixed (B, k, 7) even though only 2 anchors exist
        torch.Size([1, 3, 7])
        >>> detections[0, 2]  # the padding row: score zero, no box
        tensor([0., 0., 0., 0., 0., 0., 0.])

        ```
    """

    def __init__(self, k: int = MAX_DETECTIONS) -> None:
        super().__init__()
        self.k = k

    def forward(
        self,
        cls_logits: Tensor,
        raw_ltrb: Tensor,
        raw_angles: Tensor,
        anchor_points: Tensor,
        strides: Tensor,
    ) -> Tensor:
        """Decode raw one-to-one outputs into fixed-size top-k oriented detections.

        Args:
            cls_logits: Raw one-to-one class logits of shape ``(B, A, C)``.
            raw_ltrb: Raw one-to-one ltrb distances of shape ``(B, A, 4)``.
            raw_angles: Raw one-to-one angles in radians, shape ``(B, A, 1)`` or
                ``(B, A)``.
            anchor_points: Anchor-centre ``(x, y)`` coordinates of shape ``(A, 2)``.
            strides: Per-anchor level stride of shape ``(A,)``.

        Returns:
            Detections of shape ``(B, k, 7)`` whose last axis is the A45 tuple
            ``[cx, cy, w, h, theta, score, class]``, sorted by descending score with
            score-zero padding rows filling any shortfall below ``k``.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.assign.grid import make_anchor_points
            >>> points, strides = make_anchor_points([(1, 2)], [8])
            >>> cls_logits, raw_ltrb, raw_angles = torch.zeros(1, 2, 1), torch.zeros(1, 2, 4), torch.zeros(1, 2, 1)
            >>> RotatedTopKDecoder(k=5)(cls_logits, raw_ltrb, raw_angles, points, strides).shape
            torch.Size([1, 5, 7])

            ```
        """
        detections, _ = self.decode_with_indices(cls_logits, raw_ltrb, raw_angles, anchor_points, strides)
        return detections

    def decode_with_indices(
        self,
        cls_logits: Tensor,
        raw_ltrb: Tensor,
        raw_angles: Tensor,
        anchor_points: Tensor,
        strides: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode as :meth:`forward` does, and report the source anchor of each row.

        The whole decode lives here and :meth:`forward` is its detections-only view, as
        in :class:`TopKDecoder`, so the two cannot report different selections. The
        indices are what any per-anchor value outside the A45 tuple is gathered by.

        Args:
            cls_logits: Raw one-to-one class logits of shape ``(B, A, C)``.
            raw_ltrb: Raw one-to-one ltrb distances of shape ``(B, A, 4)``.
            raw_angles: Raw one-to-one angles in radians, shape ``(B, A, 1)`` or
                ``(B, A)``.
            anchor_points: Anchor-centre ``(x, y)`` coordinates of shape ``(A, 2)``.
            strides: Per-anchor level stride of shape ``(A,)``.

        Returns:
            A pair ``(detections, anchor_indices)``: the ``(B, k, 7)`` A45 batch
            :meth:`forward` returns, and the ``(B, k)`` long tensor naming the anchor each
            row came from, with :data:`~lucid_yolo.decode.common.PAD_ANCHOR_INDEX` on the
            padding rows.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.assign.grid import make_anchor_points
            >>> points, strides = make_anchor_points([(1, 2)], [8])
            >>> cls_logits = torch.tensor([[[-5.0], [5.0]]])
            >>> raw_ltrb, raw_angles = torch.ones(1, 2, 4), torch.zeros(1, 2, 1)
            >>> decoder = RotatedTopKDecoder(k=3)
            >>> _, anchors = decoder.decode_with_indices(cls_logits, raw_ltrb, raw_angles, points, strides)
            >>> anchors  # anchor 1 ranks first; the third row is padding
            tensor([[ 1,  0, -1]])

            ```
        """
        rboxes = decode_rboxes(raw_ltrb, raw_angles, anchor_points, strides)
        detections, anchor_indices = o2o_rotated_topk_with_indices(cls_logits, rboxes, k=self.k)
        return pad_detections(detections, self.k), pad_anchor_indices(anchor_indices, self.k)
