# SPDX-License-Identifier: Apache-2.0
"""Task-Aligned Assigner (TAL) for the anchor-free detection head (WP-025).

Transcribed by hand from the Task-aligned One-stage Object Detection paper
(R4, arXiv:2108.07755) and the reproduction's technical specification section
3.3.3. TAL replaces IoU-threshold label assignment with a single *alignment*
metric that couples the classification and localization tasks, so the anchors
selected as positives are exactly those a jointly-good prediction would fire on.

For a ground-truth box with class ``c`` and an anchor whose predicted class
probability for ``c`` is ``s`` and whose predicted box has IoU ``u`` with the
ground truth, the alignment metric is::

    t = s ** alpha * u ** beta

with ``alpha = 1`` and ``beta = 6`` (A2). Assignment proceeds per image:

1. **Candidate filter.** Only anchors whose centre lies inside the ground-truth
   box are eligible; all others get ``t = 0`` and can never be selected.
2. **Top-k select.** Each ground truth keeps its ``k`` highest-``t`` eligible
   anchors as positives (``k`` degrades to the eligible count when it is larger).
3. **Conflict resolution.** An anchor selected by several ground truths is kept
   only for the one giving it the highest ``t``.
4. **Target normalization.** Positive anchors carry a soft weight rather than a
   hard 1. Following R4's target-normalization rule, each ground truth's
   positives are rescaled by ``u_max / t_max`` — the largest IoU over its
   positives divided by the largest ``t`` over its positives — so the
   best-aligned anchor of each ground truth reaches a weight of ``u_max`` and
   the others fall off in proportion to their ``t``.

Background convention for the returned tensors: ``gt_index`` and
``target_labels`` are ``-1``, ``target_boxes`` are zero, and ``align_weights``
are zero at every non-positive anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = ["AssignResult", "TaskAlignedAssigner"]

#: Column count of an ``xyxy`` axis-aligned box.
_BOX_DIM = 4


@dataclass(frozen=True)
class AssignResult:
    """Per-anchor assignment produced by :class:`TaskAlignedAssigner`.

    All tensors are batched with a leading ``B`` (images) and ``A`` (anchors)
    axis. A non-positive (background) anchor is marked by ``fg_mask`` being
    ``False`` and carries the documented background sentinels in the other
    fields (``-1`` index/label, zero box, zero weight).

    Attributes:
        fg_mask: ``(B, A)`` bool; ``True`` at positive (foreground) anchors.
        gt_index: ``(B, A)`` long; index of the assigned ground truth into the
            padded ``N`` axis, or ``-1`` at background anchors.
        target_labels: ``(B, A)`` long; class id of the assigned ground truth,
            or ``-1`` at background anchors.
        target_boxes: ``(B, A, 4)`` float; ``xyxy`` box of the assigned ground
            truth, or zeros at background anchors.
        align_weights: ``(B, A)`` float; the per-ground-truth-normalized
            alignment metric at positive anchors, zero elsewhere.
    """

    fg_mask: Tensor
    gt_index: Tensor
    target_labels: Tensor
    target_boxes: Tensor
    align_weights: Tensor


def _box_iou_cross(gt_boxes: Tensor, pred_boxes: Tensor, eps: float) -> Tensor:
    """Cross IoU between every ground-truth and every anchor box, per image.

    Args:
        gt_boxes: ``(B, N, 4)`` ground-truth boxes in ``xyxy``.
        pred_boxes: ``(B, A, 4)`` predicted boxes in ``xyxy``.
        eps: Small constant guarding the union division.

    Returns:
        ``(B, N, A)`` IoU tensor.

    Examples:
        >>> import torch
        >>> gt = torch.tensor([[[0.0, 0.0, 2.0, 2.0]]])
        >>> pred = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]])
        >>> _box_iou_cross(gt, pred, 1e-9).round(decimals=4)
        tensor([[[1.0000, 0.1429]]])
    """
    gt = gt_boxes.unsqueeze(2)  # (B, N, 1, 4)
    pred = pred_boxes.unsqueeze(1)  # (B, 1, A, 4)
    top_left = torch.maximum(gt[..., :2], pred[..., :2])
    bottom_right = torch.minimum(gt[..., 2:], pred[..., 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    gt_area = (gt[..., 2] - gt[..., 0]).clamp(min=0) * (gt[..., 3] - gt[..., 1]).clamp(min=0)
    pred_area = (pred[..., 2] - pred[..., 0]).clamp(min=0) * (pred[..., 3] - pred[..., 1]).clamp(min=0)
    union = gt_area + pred_area - intersection
    return intersection / (union + eps)


class TaskAlignedAssigner:
    """Task-Aligned label assigner for the anchor-free detection head.

    The assigner is stateless apart from its hyper-parameters and holds no
    learnable parameters; a single instance is reused across the training run.
    All work runs under ``torch.no_grad`` — assignment produces targets, never
    gradients.

    Args:
        topk: Number of highest-alignment anchors kept per ground truth. When a
            ground truth has fewer eligible (centre-inside) anchors than
            ``topk``, only the eligible ones are kept.
        alpha: Exponent on the classification score in ``t = s**alpha * u**beta``
            (A2 default ``1.0``).
        beta: Exponent on the IoU in ``t = s**alpha * u**beta`` (A2 default
            ``6.0``).
        eps: Small constant guarding the IoU union and the normalization
            denominator.

    Examples:
        >>> import torch
        >>> assigner = TaskAlignedAssigner(topk=1)
        >>> scores = torch.tensor([[[0.9], [0.1]]])  # (B=1, A=2, C=1)
        >>> boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 7.0, 7.0]]])
        >>> points = torch.tensor([[1.0, 1.0], [6.0, 6.0]])
        >>> gt_boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0]]])
        >>> gt_labels = torch.tensor([[0]])
        >>> gt_mask = torch.tensor([[True]])
        >>> out = assigner(scores, boxes, points, gt_boxes, gt_labels, gt_mask)
        >>> out.fg_mask
        tensor([[ True, False]])
    """

    def __init__(self, topk: int, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9) -> None:
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def __call__(
        self,
        pred_scores: Tensor,
        pred_boxes: Tensor,
        anchor_points: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        gt_mask: Tensor,
    ) -> AssignResult:
        """Assign ground truths to anchors for a batch of images.

        Args:
            pred_scores: ``(B, A, C)`` predicted class probabilities in ``[0, 1]``.
            pred_boxes: ``(B, A, 4)`` predicted boxes in ``xyxy`` pixels.
            anchor_points: ``(A, 2)`` anchor-centre ``(x, y)`` pixels, shared
                across the batch (see :func:`lit_yolo.assign.make_anchor_points`).
            gt_boxes: ``(B, N, 4)`` padded ground-truth boxes in ``xyxy`` pixels.
            gt_labels: ``(B, N)`` long class ids of the ground truths; entries at
                padded slots are ignored.
            gt_mask: ``(B, N)`` bool; ``True`` marks a real ground truth, ``False``
                a padding slot that is never assigned.

        Returns:
            An :class:`AssignResult` with per-anchor foreground mask, assigned
            ground-truth index, target labels and boxes, and normalized
            alignment weights.

        Examples:
            >>> import torch
            >>> assigner = TaskAlignedAssigner(topk=1)
            >>> scores = torch.zeros(1, 4, 1)  # no ground truth -> all background
            >>> boxes = torch.zeros(1, 4, 4)
            >>> points = torch.zeros(4, 2)
            >>> empty_boxes = torch.zeros(1, 0, 4)
            >>> empty_labels = torch.zeros(1, 0, dtype=torch.long)
            >>> empty_mask = torch.zeros(1, 0, dtype=torch.bool)
            >>> out = assigner(scores, boxes, points, empty_boxes, empty_labels, empty_mask)
            >>> bool(out.fg_mask.any())
            False
        """
        with torch.no_grad():
            batch, num_anchors = pred_scores.shape[0], pred_scores.shape[1]
            if gt_boxes.shape[1] == 0:
                return self._empty_result(batch, num_anchors, pred_boxes.dtype, pred_boxes.device)

            candidate_mask = self._candidate_mask(anchor_points, gt_boxes, gt_mask)
            align_metric, iou = self._alignment_metric(pred_scores, pred_boxes, gt_boxes, gt_labels, candidate_mask)
            mask_pos = self._select_topk(align_metric, candidate_mask)
            mask_pos = self._resolve_conflicts(mask_pos, align_metric)
            return self._build_result(mask_pos, align_metric, iou, gt_boxes, gt_labels)

    def _candidate_mask(self, anchor_points: Tensor, gt_boxes: Tensor, gt_mask: Tensor) -> Tensor:
        """Eligibility mask ``(B, N, A)``: anchor centre inside a real GT box.

        Subclasses (see :class:`lit_yolo.assign.stal.SmallTargetAssigner`) override
        this single step to filter against a surrogate box while leaving every
        downstream stage on the original ground truth; the base implementation is
        stateless and ignores ``self``.
        """
        px = anchor_points[:, 0]  # (A,)
        py = anchor_points[:, 1]  # (A,)
        x1 = gt_boxes[..., 0].unsqueeze(-1)  # (B, N, 1)
        y1 = gt_boxes[..., 1].unsqueeze(-1)
        x2 = gt_boxes[..., 2].unsqueeze(-1)
        y2 = gt_boxes[..., 3].unsqueeze(-1)
        inside = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)  # (B, N, A)
        return inside & gt_mask.unsqueeze(-1)

    def _alignment_metric(
        self,
        pred_scores: Tensor,
        pred_boxes: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        candidate_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Alignment metric ``t`` and cross IoU, both ``(B, N, A)``.

        ``t`` is zeroed outside the candidate set so it can feed top-k directly;
        the raw (unmasked) IoU is returned for the target normalization.
        """
        num_anchors = pred_scores.shape[1]
        iou = _box_iou_cross(gt_boxes, pred_boxes, self.eps)  # (B, N, A)
        scores = pred_scores.permute(0, 2, 1)  # (B, C, A)
        class_index = gt_labels.clamp(min=0).unsqueeze(-1).expand(-1, -1, num_anchors)  # (B, N, A)
        gt_scores = torch.gather(scores, 1, class_index)  # (B, N, A)
        align_metric = gt_scores.pow(self.alpha) * iou.pow(self.beta)
        align_metric = align_metric * candidate_mask
        return align_metric, iou

    def _select_topk(self, align_metric: Tensor, candidate_mask: Tensor) -> Tensor:
        """Keep each GT's top-``k`` anchors by alignment; ``(B, N, A)`` bool.

        Non-candidate anchors have ``t = 0``, so when ``topk`` exceeds the
        eligible count the surplus slots land on zeros and are removed by the
        intersection with ``candidate_mask``.
        """
        k = min(self.topk, align_metric.shape[-1])
        _, topk_index = align_metric.topk(k, dim=-1, largest=True)  # (B, N, k)
        mask_topk = torch.zeros_like(align_metric, dtype=torch.bool)
        mask_topk.scatter_(-1, topk_index, True)
        return mask_topk & candidate_mask

    @staticmethod
    def _resolve_conflicts(mask_pos: Tensor, align_metric: Tensor) -> Tensor:
        """Give each anchor to the highest-``t`` GT that selected it; ``(B, N, A)``."""
        selected_per_anchor = mask_pos.sum(dim=1)  # (B, A)
        if int(selected_per_anchor.max().item()) <= 1:
            return mask_pos
        num_gt = mask_pos.shape[1]
        contested = (selected_per_anchor > 1).unsqueeze(1)  # (B, 1, A)
        masked_align = align_metric.masked_fill(~mask_pos, -1.0)
        best_gt = masked_align.argmax(dim=1)  # (B, A) — a GT that selected the anchor
        is_best = F.one_hot(best_gt, num_gt).permute(0, 2, 1).bool()  # (B, N, A)
        return torch.where(contested, is_best & mask_pos, mask_pos)

    def _build_result(
        self,
        mask_pos: Tensor,
        align_metric: Tensor,
        iou: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
    ) -> AssignResult:
        """Gather per-anchor targets and normalized weights from the final mask."""
        mask_pos_f = mask_pos.to(align_metric.dtype)
        fg_mask = mask_pos.any(dim=1)  # (B, A)
        gt_index_raw = mask_pos_f.argmax(dim=1).long()  # (B, A); 0 at background
        background = torch.full_like(gt_index_raw, -1)
        gt_index = torch.where(fg_mask, gt_index_raw, background)

        gather_index = gt_index_raw.clamp(min=0)  # (B, A) valid for gather
        target_labels = torch.gather(gt_labels, 1, gather_index)
        target_labels = torch.where(fg_mask, target_labels, background)
        box_index = gather_index.unsqueeze(-1).expand(-1, -1, _BOX_DIM)
        target_boxes = torch.gather(gt_boxes, 1, box_index) * fg_mask.unsqueeze(-1).to(gt_boxes.dtype)

        align_pos = align_metric * mask_pos_f
        iou_pos = iou * mask_pos_f
        t_max = align_pos.amax(dim=-1, keepdim=True)  # (B, N, 1)
        u_max = iou_pos.amax(dim=-1, keepdim=True)  # (B, N, 1)
        norm_align = align_pos * (u_max / (t_max + self.eps))  # (B, N, A)
        align_weights = norm_align.amax(dim=1)  # (B, A); one owning GT per anchor

        return AssignResult(
            fg_mask=fg_mask,
            gt_index=gt_index,
            target_labels=target_labels,
            target_boxes=target_boxes,
            align_weights=align_weights,
        )

    @staticmethod
    def _empty_result(batch: int, num_anchors: int, dtype: torch.dtype, device: torch.device) -> AssignResult:
        """All-background result for an image batch with zero ground truths."""
        return AssignResult(
            fg_mask=torch.zeros((batch, num_anchors), dtype=torch.bool, device=device),
            gt_index=torch.full((batch, num_anchors), -1, dtype=torch.long, device=device),
            target_labels=torch.full((batch, num_anchors), -1, dtype=torch.long, device=device),
            target_boxes=torch.zeros((batch, num_anchors, _BOX_DIM), dtype=dtype, device=device),
            align_weights=torch.zeros((batch, num_anchors), dtype=dtype, device=device),
        )
