# SPDX-License-Identifier: Apache-2.0
"""Per-branch detection loss for the anchor-free head (WP-027).

Transcribed by hand from the reproduction's technical specification section
3.3.2 (R1) and the Task-aligned One-stage Object Detection paper (R4,
arXiv:2108.07755), which fixes the alignment-weighted normalization convention.
This loss consumes one detection branch's dense predictions together with the
:class:`~open_yolos.assign.tal.AssignResult` produced for that branch and returns
the three weighted terms plus their gain-weighted sum.

For one branch the detection loss is::

    L = box_gain * L_box + l1_gain * L_l1 + cls_gain * L_cls

with the three terms (A13 gains ``box=7.5``, ``cls=0.5``, ``l1=6.0`` for the
from-scratch set):

- ``L_cls`` — binary-cross-entropy-with-logits between the predicted class
  logits and *soft* targets. The target for a positive anchor is a one-hot at
  its assigned class scaled by that anchor's normalized alignment weight
  (:attr:`AssignResult.align_weights`), and zero at every background anchor.
  The per-element losses are summed over classes and anchors and normalized by
  the total alignment-weight sum (clamped to a minimum of one). This is the
  TAL-weighted classification objective (R4): well-aligned anchors carry a
  larger positive target, so classification and localization quality are
  coupled. Background anchors still contribute an all-zero-target BCE term, so
  the classification head keeps training even on images with no positives.
- ``L_box`` — Complete-IoU loss (A1; :func:`open_yolos.losses.ciou.ciou_loss`) on
  the positive anchors only, each weighted by its alignment weight and
  normalized by the same weight sum.
- ``L_l1`` — element-wise L1 between predicted and target boxes summed over the
  four ``xyxy`` coordinates, on the positive anchors only, with the same
  weighting and normalization. This is the term the legacy DFL gain field
  scales in the DFL-free head (R1 S3 note): the gain parameter is named
  ``l1_gain`` and corresponds to that field (A13). The term is coordinate-scale
  agnostic — feeding stride-normalized or pixel boxes is the caller's choice
  and only rescales this component.

When a batch contains no positive anchors the box and L1 terms are exactly
zero (and stay connected to ``pred_boxes`` with zero gradient), while the
classification term trains against the all-background target; no term produces
``NaN`` or a non-finite gradient.

Component convention: :class:`DetectionLossOutput` reports the three terms
**pre-gain** (the raw ``L_box``/``L_cls``/``L_l1`` values), while ``total``
applies the gains. Scaling a gain therefore changes ``total`` linearly but
leaves the reported component unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F

from open_yolos.assign.tal import AssignResult
from open_yolos.losses.ciou import ciou_loss

__all__ = ["DetectionBranchLoss", "DetectionLossOutput"]


@dataclass(frozen=True)
class DetectionLossOutput:
    """Detection-branch loss terms for one branch.

    The three component tensors are the **pre-gain** values (the raw ``L_box``,
    ``L_cls`` and ``L_l1`` objectives); ``total`` is the gain-weighted sum
    ``box_gain * box + cls_gain * cls + l1_gain * l1``. Every field is a scalar
    (zero-dimensional) tensor carrying gradients back to the predictions.

    Attributes:
        total: Gain-weighted sum of the three terms; the value to backpropagate.
        box: Pre-gain Complete-IoU term ``L_box``.
        cls: Pre-gain TAL-weighted classification term ``L_cls``.
        l1: Pre-gain L1 box term ``L_l1``.
    """

    total: Tensor
    box: Tensor
    cls: Tensor
    l1: Tensor


class DetectionBranchLoss:
    """Alignment-weighted detection loss for a single head branch.

    The object is stateless apart from its three gain hyper-parameters and holds
    no learnable state, so one instance is reused across the training run. It is
    coordinate-scale agnostic: ``pred_boxes`` and ``assign.target_boxes`` only
    need to share a coordinate frame (stride-normalized or pixel), which the
    caller controls.

    Args:
        box_gain: Weight on the Complete-IoU term (A13 default ``7.5``).
        cls_gain: Weight on the classification term (A13 default ``0.5``).
        l1_gain: Weight on the L1 box term; corresponds to the legacy DFL gain
            field in the DFL-free head (A13 default ``6.0``).

    Examples:
        >>> import torch
        >>> from open_yolos.assign.tal import AssignResult
        >>> loss = DetectionBranchLoss()
        >>> logits = torch.zeros(1, 2, 1)  # B=1, A=2, C=1
        >>> boxes = torch.zeros(1, 2, 4)
        >>> assign = AssignResult(
        ...     fg_mask=torch.zeros(1, 2, dtype=torch.bool),
        ...     gt_index=torch.full((1, 2), -1),
        ...     target_labels=torch.full((1, 2), -1),
        ...     target_boxes=torch.zeros(1, 2, 4),
        ...     align_weights=torch.zeros(1, 2),
        ... )
        >>> out = loss(logits, boxes, assign)  # no positives: box and l1 vanish
        >>> float(out.box), float(out.l1)
        (0.0, 0.0)
    """

    def __init__(self, box_gain: float = 7.5, cls_gain: float = 0.5, l1_gain: float = 6.0) -> None:
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.l1_gain = l1_gain

    def __call__(self, pred_logits: Tensor, pred_boxes: Tensor, assign: AssignResult) -> DetectionLossOutput:
        """Compute the detection-branch loss for one branch of one batch.

        Args:
            pred_logits: ``(B, A, C)`` raw (pre-sigmoid) class logits.
            pred_boxes: ``(B, A, 4)`` predicted boxes in ``xyxy``, in the same
                coordinate frame as ``assign.target_boxes``.
            assign: The :class:`~open_yolos.assign.tal.AssignResult` for this
                branch, supplying the foreground mask, target labels and boxes,
                and normalized alignment weights.

        Returns:
            A :class:`DetectionLossOutput` with the pre-gain terms and their
            gain-weighted ``total``.

        Examples:
            >>> import torch
            >>> from open_yolos.assign.tal import AssignResult
            >>> loss = DetectionBranchLoss(box_gain=7.5, cls_gain=0.5, l1_gain=6.0)
            >>> logits = torch.zeros(1, 1, 1, requires_grad=True)
            >>> boxes = torch.zeros(1, 1, 4, requires_grad=True)
            >>> assign = AssignResult(
            ...     fg_mask=torch.ones(1, 1, dtype=torch.bool),
            ...     gt_index=torch.zeros(1, 1, dtype=torch.long),
            ...     target_labels=torch.zeros(1, 1, dtype=torch.long),
            ...     target_boxes=torch.tensor([[[0.0, 0.0, 2.0, 2.0]]]),
            ...     align_weights=torch.ones(1, 1),
            ... )
            >>> out = loss(logits, boxes, assign)
            >>> out.total.backward()
            >>> bool(torch.isfinite(boxes.grad).all())
            True
        """
        weight_sum = assign.align_weights.sum().clamp(min=1.0)
        l_cls = self._classification_loss(pred_logits, assign, weight_sum)
        l_box, l_l1 = self._box_losses(pred_boxes, assign, weight_sum)
        total = self.box_gain * l_box + self.cls_gain * l_cls + self.l1_gain * l_l1
        return DetectionLossOutput(total=total, box=l_box, cls=l_cls, l1=l_l1)

    @staticmethod
    def _classification_loss(pred_logits: Tensor, assign: AssignResult, weight_sum: Tensor) -> Tensor:
        """TAL-weighted BCE-with-logits, normalized by the alignment-weight sum.

        The soft target is a per-anchor one-hot at the assigned class scaled by
        the anchor's alignment weight, and all-zero at background anchors (whose
        weight is zero and whose ``fg_mask`` is ``False``).
        """
        num_classes = pred_logits.shape[-1]
        labels = assign.target_labels.clamp(min=0)  # (B, A); background -> class 0, masked out below
        one_hot = F.one_hot(labels, num_classes).to(pred_logits.dtype)  # (B, A, C)
        soft_target = one_hot * assign.align_weights.unsqueeze(-1) * assign.fg_mask.unsqueeze(-1).to(pred_logits.dtype)
        bce = F.binary_cross_entropy_with_logits(pred_logits, soft_target, reduction="none")  # (B, A, C)
        return bce.sum() / weight_sum

    @staticmethod
    def _box_losses(pred_boxes: Tensor, assign: AssignResult, weight_sum: Tensor) -> tuple[Tensor, Tensor]:
        """Alignment-weighted CIoU and L1 terms over positive anchors.

        Boolean-masking with an all-``False`` ``fg_mask`` yields empty positive
        tensors whose weighted sums are zero yet stay connected to ``pred_boxes``
        with zero gradient — so a batch with no positives is finite, not a
        special case.
        """
        fg_mask = assign.fg_mask
        pred_pos = pred_boxes[fg_mask]  # (P, 4)
        target_pos = assign.target_boxes[fg_mask]  # (P, 4)
        weights = assign.align_weights[fg_mask]  # (P,)
        box_terms = ciou_loss(pred_pos, target_pos)  # (P,)
        l1_terms = (pred_pos - target_pos).abs().sum(dim=-1)  # (P,)
        l_box = (box_terms * weights).sum() / weight_sum
        l_l1 = (l1_terms * weights).sum() / weight_sum
        return l_box, l_l1
