# SPDX-License-Identifier: Apache-2.0
"""Confidence-threshold + class-wise **rotated** NMS decoding for the dense branch (WP-091b).

The oriented counterpart of :class:`~lucid_yolo.decode.nms_path.NMSDecoder`, and it
consumes the same branch that one does: the **one-to-many** dense outputs
(``o2m_cls``, ``o2m_box``, ``o2m_angle``), never the one-to-one branch's. That is the
whole framing of this module. R1's claim for the one-to-one branch is that it needs no
NMS at all, and :func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk` is what serves it;
nothing here is a fix, a fallback or a repair for that path. This decoder exists so the
oriented tier can report the same "mAP (non-E2E)" comparison column the axis-aligned tier
reports (blueprint sec. 5.5) — a suppression baseline the suppression-free number is
measured *against*. Pointing it at the one-to-one branch would answer, and would answer a
question nobody asked.

**Suppression is by rotated overlap, and that is the only reason this module exists.**
Running the axis-aligned :class:`~lucid_yolo.decode.nms_path.NMSDecoder` over an oriented
box's centre-and-extent columns would suppress by *upright* overlap — the overlap of the
boxes' axis-aligned envelopes — and keep or drop the wrong boxes with nothing in the
output to show it had. The two measures are genuinely different quantities: two 20x2 bars
crossing at right angles through one centre have **identical** envelopes, so their
envelope IoU is exactly ``1.0`` while their rotated IoU is ``0.053``. Under any threshold
below 1 an envelope-based decoder deletes one of two objects that barely touch. The
overlap here is :func:`~lucid_yolo.data.rotated_geom.rotated_iou`, the exact
polygon-intersection measure the oriented evaluator also scores with (A24), so the decoder
and the instrument that grades it agree on what "overlap" means by construction rather
than by coincidence — they call one function in the geometry module, not two
implementations that happen to match (WP-091c).

The pipeline over the dense oriented outputs mirrors the axis-aligned path step for step:

1. decode the ltrb distances and raw angles into **canonical** rotated boxes with
   :func:`~lucid_yolo.models.heads.obb.decode_rboxes`, which ends in
   :func:`~lucid_yolo.data.rotated_geom.canonicalize` (A23) — so suppression compares
   long-edge representatives and two boxes described a half turn apart cannot read as
   different objects;
2. sigmoid the class logits and take, per anchor, its single best class and that class's
   score (the **single-label** convention :class:`~lucid_yolo.decode.nms_path.NMSDecoder`
   follows);
3. drop anchors whose best-class score is below ``conf_threshold``;
4. class-wise greedy suppression at ``iou_threshold`` (:data:`ROTATED_NMS_IOU_THRESHOLD`,
   **A61**) — boxes of *different* classes never suppress each other;
5. keep at most ``max_det`` detections per image (300; R1 sec. 3.2.1, A9, A47);
6. pad to a **fixed** ``(B, max_det, 7)`` shape with score-zero rows.

The output is the A45 tuple ``[cx, cy, w, h, theta, score, class]``, identical to what
:func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk` emits, so both oriented paths feed
:func:`~lucid_yolo.eval.dota_eval.rotated_detections_to_predictions` interchangeably —
which is what makes the comparison column a comparison rather than two incomparable
numbers. Rows beyond the survivors carry ``score == 0``.

**No ``decode_with_indices`` here, deliberately.** Both axis-aligned decoders report the
source anchor of each row because the segmentation decode has to gather mask coefficients
by it (WP-053b), and a second selection computed outside would be free to disagree. The
oriented tuple carries everything the oriented path predicts — there is no per-anchor
quantity left outside it — so an index return would be a shape with no consumer, which is
how a contract starts drifting from what it is checked against. Add it when something
needs to gather by it.

**Cost.** The suppression is a Python-level greedy loop rather than a fused kernel: there
is no batched rotated-NMS operator in torchvision, and writing one is not this package.
Each iteration is one ``(1, M)`` rotated-IoU call against the remaining candidates, and
the loop stops as soon as ``max_det`` survivors are kept — exact rather than approximate,
because greedy survivors arrive in descending score order and the output is truncated to
``max_det`` anyway, so the iterations skipped could only have produced rows the cap
discards. The bound is therefore ``max_det`` iterations, and it is ``conf_threshold`` that
governs how wide each one is.

Provenance: R1 sec. 3.2.1, R3 sec. 4, R13, R18 sec. 4.
Assumptions: A9, A23, A24, A45, A47, A61 (the suppression threshold).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lucid_yolo.data.rotated_geom import rotated_iou
from lucid_yolo.decode.common import pad_detections
from lucid_yolo.models.heads.obb import decode_rboxes

__all__ = ["ROTATED_NMS_IOU_THRESHOLD", "RotatedNMSDecoder"]

#: Rotated-IoU threshold above which a lower-scoring same-class box is suppressed
#: (**A61**). Carried over unchanged from the axis-aligned
#: :class:`~lucid_yolo.decode.nms_path.NMSDecoder`, so the two comparison columns differ by
#: the overlap *measure* alone and a difference between them can be attributed to the
#: geometry rather than to a second knob turned at the same time. No source on the
#: allowlist states a value for rotated suppression on this architecture; the register row
#: records both that gap and the measured exposure the carry-over accepts.
ROTATED_NMS_IOU_THRESHOLD = 0.7

#: Default per-image detection cap (R1 sec. 3.2.1, A9, A47).
_DEFAULT_MAX_DET = 300

#: Default evaluation-time confidence threshold — permissive, so the mAP integration sees
#: the low-confidence tail, exactly as the axis-aligned path's default does.
_DEFAULT_CONF_THRESHOLD = 0.001


class RotatedNMSDecoder(nn.Module):
    """Confidence-threshold + class-wise rotated-NMS decoder for the dense branch (A45, A61).

    Turns the **one-to-many** branch's dense oriented outputs into a fixed-size batch of
    score-ranked oriented detections, suppressing by exact rotated overlap rather than by
    the overlap of the boxes' upright envelopes. Emits the same ``(B, max_det, 7)`` A45
    tuple as :func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk`, so the suppression
    column and the suppression-free column are the same kind of number. The module owns no
    parameters — it is a pure functional transform wrapped as an
    :class:`~torch.nn.Module`.

    This is the *o2m* path's decoder and a comparison baseline. It is not a repair to the
    one-to-one path, whose freedom from NMS is the architecture's claim rather than a
    limitation to work around.

    Args:
        conf_threshold: Anchors whose best-class score is strictly below this value are
            dropped before suppression. Defaults to ``0.001`` (the permissive evaluation
            convention).
        iou_threshold: Rotated-IoU threshold; a lower-scoring box is suppressed when its
            rotated IoU with a kept higher-scoring box of the *same* class exceeds this.
            Defaults to :data:`ROTATED_NMS_IOU_THRESHOLD` (A61).
        max_det: Maximum detections kept per image; also the fixed output length. Defaults
            to ``300`` (R1 sec. 3.2.1, A9, A47).

    Examples:
        >>> import torch
        >>> from lucid_yolo.assign.grid import make_anchor_points
        >>> points, strides = make_anchor_points([(2, 2)], [8])
        >>> cls_logits = torch.full((1, 4, 3), -5.0)
        >>> cls_logits[0, 1, 2] = 5.0  # one confident anchor, class 2
        >>> raw_ltrb = torch.ones(1, 4, 4)
        >>> angles = torch.zeros(1, 4, 1)
        >>> decoder = RotatedNMSDecoder(max_det=100)
        >>> detections = decoder(cls_logits, raw_ltrb, angles, points, strides)
        >>> detections.shape  # fixed (B, max_det, 7)
        torch.Size([1, 100, 7])
        >>> int(detections[0, 0, 6])  # class of the top detection
        2
    """

    def __init__(
        self,
        conf_threshold: float = _DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = ROTATED_NMS_IOU_THRESHOLD,
        max_det: int = _DEFAULT_MAX_DET,
    ) -> None:
        super().__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

    def forward(
        self,
        cls_logits: Tensor,
        raw_ltrb: Tensor,
        angles: Tensor,
        anchor_points: Tensor,
        strides: Tensor,
    ) -> Tensor:
        """Decode raw dense oriented outputs into fixed-size rotated-NMS-filtered detections.

        Args:
            cls_logits: Raw dense class logits of shape ``(B, A, C)``.
            raw_ltrb: Raw dense ltrb distances of shape ``(B, A, 4)`` (order l, t, r, b),
                aligned with ``cls_logits`` on the anchor axis.
            angles: Raw dense orientation angles in radians, shape ``(B, A, 1)`` or
                ``(B, A)``, as R1 Eq. 13 emits them — any real number, canonicalized by
                the decode rather than by the head (A23).
            anchor_points: Anchor-centre ``(x, y)`` coordinates of shape ``(A, 2)`` in
                input pixels, as returned by
                :func:`~lucid_yolo.assign.grid.make_anchor_points`.
            strides: Per-anchor level stride of shape ``(A,)``.

        Returns:
            Detections of shape ``(B, max_det, 7)`` whose last axis is the A45 tuple
            ``[cx, cy, w, h, theta, score, class]``, canonical (``w >= h``, ``theta`` in
            ``[-pi/4, 3*pi/4)``), sorted by descending score, with score-zero padding rows
            filling any shortfall below ``max_det``.

        Examples:
            >>> import torch
            >>> from lucid_yolo.assign.grid import make_anchor_points
            >>> points, strides = make_anchor_points([(1, 2)], [8])
            >>> cls_logits = torch.zeros(1, 2, 1)
            >>> raw_ltrb = torch.zeros(1, 2, 4)
            >>> angles = torch.zeros(1, 2, 1)
            >>> RotatedNMSDecoder(max_det=5)(cls_logits, raw_ltrb, angles, points, strides).shape
            torch.Size([1, 5, 7])
            >>> # Nothing above the threshold still returns the fixed shape, all padding.
            >>> empty = RotatedNMSDecoder(conf_threshold=0.99, max_det=2)
            >>> empty(cls_logits, raw_ltrb, angles, points, strides)
            tensor([[[0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.]]])
        """
        rboxes = decode_rboxes(raw_ltrb, angles, anchor_points, strides)  # (B, A, 5), canonical
        confidence = cls_logits.sigmoid()
        scores, classes = confidence.max(dim=-1)  # both (B, A), single-label per anchor
        return torch.stack([self._decode_image(rboxes[i], scores[i], classes[i]) for i in range(rboxes.shape[0])])

    def _decode_image(self, rboxes: Tensor, scores: Tensor, classes: Tensor) -> Tensor:
        """Threshold, class-wise rotated suppression, cap, and pad one image's anchors.

        Args:
            rboxes: Canonical rotated boxes of shape ``(A, 5)``, each ``(cx, cy, w, h,
                theta)``.
            scores: Per-anchor best-class score of shape ``(A,)``, in ``[0, 1]``.
            classes: Per-anchor best-class index of shape ``(A,)`` (integral).

        Returns:
            The ``(max_det, 7)`` A45 detections ``[cx, cy, w, h, theta, score, class]``
            sorted by descending score and padded with score-zero rows.
        """
        # A non-finite row is dropped here rather than left to the suppression, because
        # suppression cannot remove it: `rotated_iou` scores every pair involving a
        # non-finite box `0.0`, that zero clears no threshold, and the row therefore
        # survives every round of `_suppress` and is emitted as a detection carrying a
        # real score. The threshold is the only boundary that sees it as a row rather
        # than as an overlap.
        keep = (scores >= self.conf_threshold) & rboxes.isfinite().all(dim=-1)
        rboxes, scores, classes = rboxes[keep], scores[keep], classes[keep]
        order = self._suppress(rboxes, scores, classes)
        detection = torch.cat(
            (rboxes[order], scores[order].unsqueeze(-1), classes[order].unsqueeze(-1).to(rboxes.dtype)),
            dim=-1,
        )
        return pad_detections(detection, self.max_det)

    def _suppress(self, rboxes: Tensor, scores: Tensor, classes: Tensor) -> Tensor:
        """Return the surviving rows, score-descending, after class-wise rotated suppression.

        The classic greedy rule with the overlap measure swapped: take the highest-scoring
        remaining candidate, keep it, and discard every lower-scoring candidate of the
        **same class** whose rotated IoU with it exceeds :attr:`iou_threshold`. A
        different-class candidate is never discarded, so overlapping detections of
        distinct classes all survive — the property
        :func:`torchvision.ops.batched_nms` gives the axis-aligned path by offsetting
        coordinates per class, obtained here by testing the label directly because there
        is no offset that separates *rotated* boxes without also turning them.

        The sort is stable, so equal scores keep their anchor order and the decode is
        reproducible. The loop stops at :attr:`max_det` survivors: they arrive in
        descending score order and the caller truncates there regardless, so nothing that
        would have appeared in the output is skipped.

        Args:
            rboxes: Canonical rotated boxes of shape ``(N, 5)`` that cleared the
                confidence threshold.
            scores: Their scores, shape ``(N,)``.
            classes: Their integral class indices, shape ``(N,)``.

        Returns:
            A long tensor of at most ``max_det`` indices into ``rboxes``, in descending
            score order.
        """
        order = torch.argsort(scores, descending=True, stable=True)
        kept: list[Tensor] = []
        while order.numel() and len(kept) < self.max_det:
            best, rest = order[0], order[1:]
            kept.append(best)
            if not rest.numel():
                break
            overlap = rotated_iou(rboxes[best].unsqueeze(0), rboxes[rest])[0]
            order = rest[~((classes[rest] == classes[best]) & (overlap > self.iou_threshold))]
        return torch.stack(kept) if kept else order.new_empty((0,))
