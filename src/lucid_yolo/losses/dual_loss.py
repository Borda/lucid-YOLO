# SPDX-License-Identifier: Apache-2.0
"""Dual-branch detection loss composition for the anchor-free head (WP-028).

Transcribed by hand from the reproduction's technical specification sections
3.2.1 and 3.3.2 (R1) and the consistent dual-assignment lineage (R6). The
detection head carries two branches that share neck features:

- the **one-to-many** (o2m) branch trains with dense supervision — a
  Small-Target-Aware assignment with ``topk = 10`` (many positive anchors per
  ground truth), which drives fast, high-recall learning; and
- the **one-to-one** (o2o) branch trains with a *unique* assignment — a
  :class:`~lucid_yolo.assign.one_to_one.UniqueAssigner` (``topk = 7`` candidate
  set reduced by a ``topk2 = 1`` filter to exactly one positive per ground
  truth), which learns NMS-free, one-prediction-per-object outputs.

:class:`DualBranchLoss` scores each branch with its own
:class:`~lucid_yolo.losses.detection_loss.DetectionBranchLoss` (both sharing the
same gains) against its own assignment, then combines the two totals::

    L = alpha * L_o2m + (1 - alpha) * L_o2o

``alpha`` is a plain, settable attribute — static here (default ``0.5``). The
epoch-scheduled ramp ``(0.8, 0.2) -> (0.1, 0.9)`` (R1 Eq. 2-3) is out of scope
for this work package: it lands as the WP-035 progressive-loss hook, which sets
this attribute per epoch. Exposing ``alpha`` as a bare attribute is exactly the
seam that hook writes to.

The two assigners are exposed as :attr:`DualBranchLoss.o2m_assigner` and
:attr:`DualBranchLoss.o2o_assigner` so callers can inspect or reconfigure the
assignment independently of the loss. Both consume per-anchor class
*probabilities*, so :meth:`DualBranchLoss.__call__` maps each branch's raw
logits through a sigmoid for the assigner while feeding the raw logits to the
loss; the assigner runs under ``no_grad``, so this mapping never carries a
gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from lucid_yolo.assign.one_to_one import UniqueAssigner
from lucid_yolo.assign.stal import SmallTargetAssigner
from lucid_yolo.losses.detection_loss import DetectionBranchLoss, DetectionLossOutput

__all__ = ["DualBranchLoss", "DualLossOutput"]


@dataclass(frozen=True)
class DualLossOutput:
    """Combined dual-branch loss and its two per-branch breakdowns.

    ``total`` is the ``alpha``-weighted combination of the two branch totals;
    the ``o2m`` and ``o2o`` fields carry each branch's full
    :class:`~lucid_yolo.losses.detection_loss.DetectionLossOutput` (its pre-gain
    ``box``/``cls``/``l1`` terms and gain-weighted ``total``), and ``alpha`` is
    the branch weight used for this call.

    Attributes:
        total: ``alpha * o2m.total + (1 - alpha) * o2o.total``; the scalar to
            backpropagate.
        o2m: One-to-many branch loss output (dense ``topk = 10`` assignment).
        o2o: One-to-one branch loss output (unique ``topk = 7 -> 1`` assignment).
        alpha: Branch weight applied to the o2m total (o2o gets ``1 - alpha``).
    """

    total: Tensor
    o2m: DetectionLossOutput
    o2o: DetectionLossOutput
    alpha: float


class DualBranchLoss:
    """Compose the one-to-many and one-to-one detection-branch losses.

    Holds two :class:`~lucid_yolo.losses.detection_loss.DetectionBranchLoss`
    instances (identical gains), a dense o2m assigner
    (:class:`~lucid_yolo.assign.stal.SmallTargetAssigner`, ``topk = 10``), and a
    unique o2o assigner (:class:`~lucid_yolo.assign.one_to_one.UniqueAssigner`,
    ``topk = 7`` reduced to one). The object is stateless apart from its gains,
    assigners, and the settable :attr:`alpha`, so a single instance is reused
    across the training run. It is coordinate-scale agnostic in the same way as
    the per-branch loss: predicted and ground-truth boxes only need to share a
    frame (see :class:`~lucid_yolo.losses.detection_loss.DetectionBranchLoss`).

    Args:
        box_gain: Weight on each branch's Complete-IoU term (A13 default ``7.5``).
        cls_gain: Weight on each branch's classification term (A13 default ``0.5``).
        l1_gain: Weight on each branch's L1 box term (A13 default ``6.0``).
        o2m_topk: Candidate count for the dense o2m assignment (default ``10``).
        o2o_topk: Candidate-set size for the o2o assignment before the unique
            ``topk2 = 1`` reduction (default ``7``).
        alpha: Initial branch weight for ``L = alpha * L_o2m + (1 - alpha) *
            L_o2o`` (static default ``0.5``; the WP-035 schedule sets it per
            epoch).
        s_min: Small-target surrogate threshold shared by both assigners (the
            smallest stride, ``8.0`` at 640 input).
        s_ref: Small-target surrogate replacement size shared by both assigners
            (the next stride, ``16.0`` at 640 input).

    Examples:
        >>> import torch
        >>> from lucid_yolo.assign import make_anchor_points
        >>> loss = DualBranchLoss()
        >>> points, _ = make_anchor_points([(2, 2)], [8])  # 4 anchors
        >>> logits = torch.zeros(1, 4, 1)
        >>> boxes = torch.zeros(1, 4, 4)
        >>> empty_boxes = torch.zeros(1, 0, 4)
        >>> empty_labels = torch.zeros(1, 0, dtype=torch.long)
        >>> empty_mask = torch.zeros(1, 0, dtype=torch.bool)
        >>> out = loss(logits, boxes, logits, boxes, points, empty_boxes, empty_labels, empty_mask)
        >>> bool(torch.isfinite(out.total))  # no ground truth: finite, cls-only
        True
    """

    def __init__(
        self,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        l1_gain: float = 6.0,
        o2m_topk: int = 10,
        o2o_topk: int = 7,
        alpha: float = 0.5,
        s_min: float = 8.0,
        s_ref: float = 16.0,
    ) -> None:
        self._o2m_loss = DetectionBranchLoss(box_gain, cls_gain, l1_gain)
        self._o2o_loss = DetectionBranchLoss(box_gain, cls_gain, l1_gain)
        self.o2m_assigner = SmallTargetAssigner(topk=o2m_topk, s_min=s_min, s_ref=s_ref)
        self.o2o_assigner = UniqueAssigner(topk=o2o_topk, s_min=s_min, s_ref=s_ref)
        self.alpha = alpha

    def __call__(
        self,
        o2m_logits: Tensor,
        o2m_boxes: Tensor,
        o2o_logits: Tensor,
        o2o_boxes: Tensor,
        anchor_points: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        gt_mask: Tensor,
        strides: Tensor | None = None,
    ) -> DualLossOutput:
        """Score both branches against their own assignments and combine them.

        Args:
            o2m_logits: ``(B, A, C)`` raw class logits of the one-to-many branch.
            o2m_boxes: ``(B, A, 4)`` predicted ``xyxy`` boxes of the o2m branch.
            o2o_logits: ``(B, A, C)`` raw class logits of the one-to-one branch.
            o2o_boxes: ``(B, A, 4)`` predicted ``xyxy`` boxes of the o2o branch.
            anchor_points: ``(A, 2)`` anchor-centre ``(x, y)`` pixels shared by
                both branches (see :func:`lucid_yolo.assign.make_anchor_points`).
            gt_boxes: ``(B, N, 4)`` padded ground-truth boxes in ``xyxy`` pixels.
            gt_labels: ``(B, N)`` long class ids; padded slots are ignored.
            gt_mask: ``(B, N)`` bool marking real ground truths.
            strides: Optional ``(A,)`` per-anchor level strides, forwarded to
                both branch losses so their L1 terms are measured in stride
                units (A13 revision, WP-078). The training path always passes
                strides.

        Returns:
            A :class:`DualLossOutput` with the combined ``total``, each branch's
            :class:`~lucid_yolo.losses.detection_loss.DetectionLossOutput`, and the
            ``alpha`` used.

        Examples:
            >>> import torch
            >>> from lucid_yolo.assign import make_anchor_points
            >>> loss = DualBranchLoss()
            >>> loss.alpha = 0.25  # e.g. a scheduled value from WP-035
            >>> points, _ = make_anchor_points([(2, 2)], [8])
            >>> logits = torch.zeros(1, 4, 1, requires_grad=True)
            >>> boxes = torch.zeros(1, 4, 4, requires_grad=True)
            >>> gt = torch.tensor([[[0.0, 0.0, 8.0, 8.0]]])
            >>> out = loss(logits, boxes, logits, boxes, points, gt, torch.tensor([[0]]), torch.tensor([[True]]))
            >>> out.total.backward()
            >>> bool(torch.isfinite(logits.grad).all())
            True
        """
        o2m_assign = self.o2m_assigner(o2m_logits.sigmoid(), o2m_boxes, anchor_points, gt_boxes, gt_labels, gt_mask)
        o2o_assign = self.o2o_assigner(o2o_logits.sigmoid(), o2o_boxes, anchor_points, gt_boxes, gt_labels, gt_mask)
        o2m_out = self._o2m_loss(o2m_logits, o2m_boxes, o2m_assign, strides)
        o2o_out = self._o2o_loss(o2o_logits, o2o_boxes, o2o_assign, strides)
        total = self.alpha * o2m_out.total + (1.0 - self.alpha) * o2o_out.total
        return DualLossOutput(total=total, o2m=o2m_out, o2o=o2o_out, alpha=self.alpha)
