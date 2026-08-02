# SPDX-License-Identifier: Apache-2.0
"""Small-Target-Aware Label Assignment (STAL) for the detection head (WP-026).

Transcribed by hand from the reproduction's technical specification section
3.3.3 (Eq. 4-6) and its operational consequence in section 4. STAL is a
minimal, surgical modification of the Task-Aligned Assigner (:mod:`open_yolos.assign.tal`):
it changes **only** the candidate-filtering step so that very small ground
truths are guaranteed eligible anchors. Alignment scoring, top-k selection,
conflict resolution, target normalization, and the returned target boxes are
all inherited unchanged and continue to operate on the **original** ground-truth
box — never the surrogate.

The mechanism is a per-ground-truth surrogate box. For a ground truth with
centre ``(x, y)`` and size ``(w, h)`` the surrogate ``g_tilde`` keeps the same
centre and inflates each dimension **independently**::

    d_tilde = s_ref  if d < s_min  else d      for d in (w, h)

with ``s_min = 8`` (the smallest stride) and ``s_ref = 16`` (the next stride) at
a 640-pixel input. The centre-inside candidate test runs against ``g_tilde``; a
ground truth smaller than ``s_min`` in a dimension is thereby widened to the
next stride's footprint, so a sub-``8x8`` box that vanilla TAL would leave with
zero centre-inside anchors at stride 8 is guaranteed at least one candidate.

Everything after candidate selection is the base assigner's: the alignment
metric ``t = s**alpha * u**beta`` uses the IoU ``u`` of predictions against the
original box, the returned ``target_boxes`` are the original ground truth, and
the normalized ``align_weights`` derive from that original-box IoU.
"""

from __future__ import annotations

import torch
from torch import Tensor

from open_yolos.assign.tal import TaskAlignedAssigner

__all__ = ["SmallTargetAssigner", "surrogate_boxes"]


def surrogate_boxes(gt_boxes: Tensor, s_min: float, s_ref: float) -> Tensor:
    """Build centre-preserving surrogate boxes with per-dimension size clamping.

    Each box is widened independently per dimension: a dimension smaller than
    ``s_min`` is replaced by ``s_ref``; a dimension at least ``s_min`` is left
    untouched. The box centre never moves. Used only for the STAL candidate
    filter; scoring, targets, and regression keep the original boxes.

    Args:
        gt_boxes: ``(..., 4)`` boxes in ``xyxy`` pixels. Any leading batch/ground
            -truth axes are preserved.
        s_min: Size threshold below which a dimension is inflated (the smallest
            stride, ``8.0`` at 640 input).
        s_ref: Replacement size for an inflated dimension (the next stride,
            ``16.0`` at 640 input).

    Returns:
        A ``(..., 4)`` tensor of ``xyxy`` surrogate boxes sharing ``gt_boxes``'s
        dtype, device, and leading shape.

    Examples:
        >>> import torch
        >>> boxes = torch.tensor([[[2.0, 5.0, 8.0, 25.0]]])  # 6 wide, 20 tall
        >>> surrogate_boxes(boxes, 8.0, 16.0)  # width -> 16, height kept, centre fixed
        tensor([[[-3.,  5., 13., 25.]]])
    """
    x1, y1, x2, y2 = gt_boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    width = x2 - x1
    height = y2 - y1
    width_tilde = torch.where(width < s_min, torch.full_like(width, s_ref), width)
    height_tilde = torch.where(height < s_min, torch.full_like(height, s_ref), height)
    half_w = width_tilde * 0.5
    half_h = height_tilde * 0.5
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


class SmallTargetAssigner(TaskAlignedAssigner):
    """Task-Aligned assigner with small-target-aware candidate filtering (STAL).

    Identical to :class:`~open_yolos.assign.tal.TaskAlignedAssigner` except that the
    centre-inside candidate test runs against a per-ground-truth surrogate box
    (see :func:`surrogate_boxes`) whose dimensions below ``s_min`` are inflated to
    ``s_ref``. All other stages — alignment metric, top-k, conflict resolution,
    target boxes, and normalized weights — are inherited unchanged and operate on
    the original ground-truth boxes.

    Args:
        topk: Number of highest-alignment anchors kept per ground truth.
        alpha: Exponent on the classification score in ``t = s**alpha * u**beta``
            (A2 default ``1.0``).
        beta: Exponent on the IoU in ``t = s**alpha * u**beta`` (A2 default
            ``6.0``).
        eps: Small constant guarding the IoU union and the normalization
            denominator.
        s_min: Dimension threshold below which the surrogate inflates a box side
            (the smallest stride, ``8.0`` at 640 input).
        s_ref: Replacement side length for an inflated dimension (the next
            stride, ``16.0`` at 640 input).

    Examples:
        >>> import torch
        >>> from open_yolos.assign import make_anchor_points
        >>> assigner = SmallTargetAssigner(topk=4)
        >>> points, _ = make_anchor_points([(4, 4)], [8])  # centres at 4, 12, 20, 28
        >>> gt = torch.tensor([[[5.0, 5.0, 11.0, 11.0]]])  # 6x6, no centre inside
        >>> scores = torch.full((1, 16, 1), 0.9)
        >>> boxes = gt.expand(1, 16, 4).contiguous()  # every pred == the GT box
        >>> labels = torch.tensor([[0]])
        >>> mask = torch.tensor([[True]])
        >>> out = assigner(scores, boxes, points, gt, labels, mask)
        >>> int(out.fg_mask.sum())  # vanilla TAL would yield 0 here
        4
    """

    def __init__(
        self,
        topk: int,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
        s_min: float = 8.0,
        s_ref: float = 16.0,
    ) -> None:
        super().__init__(topk, alpha, beta, eps)
        self.s_min = s_min
        self.s_ref = s_ref

    def _candidate_mask(self, anchor_points: Tensor, gt_boxes: Tensor, gt_mask: Tensor) -> Tensor:
        """Eligibility mask against the surrogate box; ``(B, N, A)``.

        The only overridden stage: the centre-inside test uses surrogate boxes
        (small dimensions inflated to ``s_ref``) so tiny ground truths gain
        candidates. Every downstream stage still sees the original ``gt_boxes``.
        """
        surrogate = surrogate_boxes(gt_boxes, self.s_min, self.s_ref)
        return super()._candidate_mask(anchor_points, surrogate, gt_mask)
