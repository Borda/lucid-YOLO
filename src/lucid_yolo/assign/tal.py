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

Oriented ground truths (WP-061, A25) enter through the optional ``gt_rboxes``
argument of :meth:`TaskAlignedAssigner.__call__`, which swaps the centre-inside
test for the point-in-rotated-rect test of
:func:`~lucid_yolo.data.rotated_geom.points_in_rboxes`. It changes **step 1 only**:
the alignment metric, the IoU, the returned targets and the weights all keep
running on the axis-aligned ``gt_boxes``. Omitting the argument leaves every
tensor operation on the axis-aligned path exactly as it was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from lucid_yolo.data.rotated_geom import points_in_rboxes

__all__ = ["AssignResult", "TaskAlignedAssigner"]

#: Column count of an ``xyxy`` axis-aligned box.
_BOX_DIM = 4
#: Column count of a long-edge rotated box ``(cx, cy, w, h, theta)``.
_RBOX_DIM = 5


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


def _rotated_inside(anchor_points: Tensor, gt_rboxes: Tensor) -> Tensor:
    """Point-in-rotated-rect containment for every GT and anchor; ``(B, N, A)``.

    Calls :func:`~lucid_yolo.data.rotated_geom.points_in_rboxes` once per **image**
    rather than once on the batch flattened into its box axis. The primitive
    materializes a ``(P, M, 2)`` offset tensor, so folding ``B`` into ``M`` would grow
    that intermediate by a factor of ``B`` for anchor/box pairs no image ever reads.
    The loop is over images — a handful — never over boxes.

    Args:
        anchor_points: ``(A, 2)`` anchor-centre ``(x, y)`` pixels.
        gt_rboxes: ``(B, N, 5)`` rotated ground truths ``(cx, cy, w, h, theta)``.

    Returns:
        ``(B, N, A)`` bool tensor, ``True`` where the anchor centre lies inside or
        exactly on the boundary of the rotated ground truth.

    Examples:
        >>> import torch
        >>> points = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
        >>> rboxes = torch.tensor([[[0.0, 0.0, 6.0, 2.0, 0.0]]])  # 6x2 box at the origin
        >>> _rotated_inside(points, rboxes).tolist()  # (1, 1, 2): the second point is out
        [[[True, False]]]
    """
    per_image = [points_in_rboxes(anchor_points, rboxes) for rboxes in gt_rboxes]  # each (A, N)
    return torch.stack(per_image).transpose(1, 2)  # (B, A, N) -> (B, N, A)


def _check_rbox_batch(gt_rboxes: Tensor, gt_boxes: Tensor) -> None:
    """Raise :class:`ValueError` unless ``gt_rboxes`` is ``(B, N, 5)`` alongside ``gt_boxes``.

    The rotated ground truths pair one-to-one with the axis-aligned ones: entry
    ``(b, n)`` of each describes the same object, the rotated one deciding candidacy
    and the axis-aligned one everything downstream.

    Examples:
        >>> import torch
        >>> _check_rbox_batch(torch.zeros((2, 3, 5)), torch.zeros((2, 3, 4)))
    """
    expected = (*gt_boxes.shape[:2], _RBOX_DIM)
    if tuple(gt_rboxes.shape) != expected:
        raise ValueError(f"gt_rboxes must be {expected} to match gt_boxes; got {tuple(gt_rboxes.shape)}")


def _check_hyperparameters(alpha: float, beta: float, eps: float) -> None:
    """Raise :class:`ValueError` unless the exponents and the union guard are usable.

    These three are the assigner's numerical safeguards, and each of them has
    values that switch it off rather than tune it — which is why they are checked
    here rather than left to fail somewhere downstream as a ``NaN`` weight.

    ``eps`` guards the IoU union division (:func:`_box_iou_cross`). At ``eps = 0``
    a pair of zero-area boxes divides ``0 / 0`` and the IoU comes out ``NaN``
    instead of the ``0.0`` overlap the guard exists to produce; a negative ``eps``
    is worse, since it can flip the sign of a small union and hand top-k a
    *negative* alignment for an anchor that genuinely overlaps.

    A non-finite exponent makes ``t = s**alpha * u**beta`` non-finite at every
    anchor, so the ranking carries no information at all. A negative one inverts
    it: the worst-aligned anchor scores highest and top-k reads that as the best
    candidate, which is a silently wrong assignment rather than a crash. Zero is
    allowed for either exponent — it drops that factor from the metric, an
    unusual but coherent request (``beta = 0`` is pure classification alignment).

    Args:
        alpha: Exponent on the classification score; must be finite and ``>= 0``.
        beta: Exponent on the IoU; must be finite and ``>= 0``.
        eps: Constant added to the IoU union; must be finite and ``> 0``.

    Raises:
        ValueError: If any of the three lies outside its documented range. The
            message names the offending parameter and the value it was given.

    Examples:
        >>> _check_hyperparameters(1.0, 6.0, 1e-9)
        >>> _check_hyperparameters(1.0, 6.0, 0.0)
        Traceback (most recent call last):
        ValueError: eps must be finite and > 0, got 0.0
        >>> _check_hyperparameters(1.0, -6.0, 1e-9)
        Traceback (most recent call last):
        ValueError: beta must be finite and >= 0, got -6.0
    """
    for name, exponent in (("alpha", alpha), ("beta", beta)):
        if not math.isfinite(exponent) or exponent < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {exponent}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and > 0, got {eps}")


def _check_labels(gt_labels: Tensor, gt_mask: Tensor, num_classes: int) -> None:
    """Raise :class:`ValueError` unless every real ground-truth label names a predicted class.

    :meth:`TaskAlignedAssigner._alignment_metric` gathers each ground truth's
    predicted score by using its label as a column index, behind a
    ``clamp(min=0)`` that exists for the padding slots. That clamp silently
    absorbs a **negative** label: the ground truth is then scored against class 0,
    assigned on that score, and trained towards its own (different) label, with
    nothing anywhere reporting it. A label at or above ``num_classes`` is not
    absorbed but fails inside ``torch.gather`` as an index error that names no
    tensor a caller would recognise — and on an accelerator, not even reliably at
    the call that caused it. Checking the range up front turns both into one
    message naming the bound.

    The two bounds have different scopes, because the gather does. Its index
    tensor is built from **every** slot, padded ones included — nothing there
    consults ``gt_mask`` — so the upper bound is a precondition on the whole
    tensor: a padding slot holding ``num_classes`` fails the gather just as a real
    one does, whatever the mask says. The lower bound applies to real ground
    truths only, since that is precisely the case the ``clamp(min=0)`` does not
    genuinely neutralize: at a padding slot the clamped row is discarded by
    ``candidate_mask`` and never reaches a target, while at a real one it is kept
    and scored against class 0. Both tests are vacuously true for an empty batch.

    Args:
        gt_labels: ``(B, N)`` long class ids of the padded ground truths.
        gt_mask: ``(B, N)`` bool; ``True`` marks a real ground truth.
        num_classes: Column count ``C`` of the predicted score tensor.

    Raises:
        ValueError: If any slot carries a label at or above ``C``, or if a real
            ground truth carries a negative one.

    Examples:
        >>> import torch
        >>> labels = torch.tensor([[0, -1]])  # -1 is a padding slot's leftover
        >>> _check_labels(labels, torch.tensor([[True, False]]), num_classes=3)
        >>> _check_labels(labels, torch.tensor([[True, True]]), num_classes=3)
        Traceback (most recent call last):
        ValueError: gt_labels must be in [0, 3) at real ground truths; got [-1]
        >>> _check_labels(torch.tensor([[0, 7]]), torch.tensor([[True, False]]), num_classes=3)
        Traceback (most recent call last):
        ValueError: gt_labels must be < 3 at every slot, padding included; got [7]
    """
    too_large = gt_labels >= num_classes
    if bool(too_large.any()):
        raise ValueError(
            f"gt_labels must be < {num_classes} at every slot, padding included; got {gt_labels[too_large].tolist()}"
        )
    negative = (gt_labels < 0) & gt_mask
    if bool(negative.any()):
        raise ValueError(
            f"gt_labels must be in [0, {num_classes}) at real ground truths; got {gt_labels[negative].tolist()}"
        )


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
            (A2 default ``1.0``). Must be finite and ``>= 0``.
        beta: Exponent on the IoU in ``t = s**alpha * u**beta`` (A2 default
            ``6.0``). Must be finite and ``>= 0``.
        eps: Small constant guarding the IoU union division; must be finite and
            ``> 0``. It does **not** guard the target-normalization denominator:
            that quantity is ``s * u_max**6``, which an absolute floor swamps at
            small overlap (see :meth:`_build_result`), so the denominator is
            floored at the dtype's smallest normal instead.

    Raises:
        ValueError: If ``topk`` is below 1, or if any of ``alpha``, ``beta`` and
            ``eps`` takes a value that disables the safeguard it exists to be
            (see :func:`_check_hyperparameters`).

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
        _check_hyperparameters(alpha, beta, eps)
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
        gt_rboxes: Tensor | None = None,
    ) -> AssignResult:
        """Assign ground truths to anchors for a batch of images.

        Args:
            pred_scores: ``(B, A, C)`` predicted class probabilities in ``[0, 1]``.
            pred_boxes: ``(B, A, 4)`` predicted boxes in ``xyxy`` pixels.
            anchor_points: ``(A, 2)`` anchor-centre ``(x, y)`` pixels, shared
                across the batch (see :func:`lucid_yolo.assign.make_anchor_points`).
            gt_boxes: ``(B, N, 4)`` padded ground-truth boxes in ``xyxy`` pixels.
            gt_labels: ``(B, N)`` long class ids of the ground truths. Every entry
                must be below ``C`` — the score gather indexes padded slots too —
                and every *real* one must also be non-negative; a padded slot may
                hold any negative value and is ignored. Checked at entry by
                :func:`_check_labels` unless Python runs with ``-O``.
            gt_mask: ``(B, N)`` bool; ``True`` marks a real ground truth, ``False``
                a padding slot that is never assigned.
            gt_rboxes: Optional ``(B, N, 5)`` rotated ground truths
                ``(cx, cy, w, h, theta)`` in the long-edge convention, entry ``(b, n)``
                describing the same object as ``gt_boxes[b, n]``. When given, candidacy
                switches to point-in-rotated-rect containment (WP-061, A25) and every
                other stage — IoU, alignment metric, targets, weights — keeps running on
                ``gt_boxes``. When omitted (the default) the assignment is the
                axis-aligned one, unchanged.

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

            Rotated candidacy drops an anchor the axis-aligned box would have kept:

            >>> two = TaskAlignedAssigner(topk=2)
            >>> points = torch.tensor([[0.0, 0.0], [3.0, 3.0]])
            >>> gt = torch.tensor([[[-4.0, -4.0, 4.0, 4.0]]])  # 8x8, both anchors inside
            >>> rotated = torch.tensor([[[0.0, 0.0, 8.0, 2.0, 0.0]]])  # same centre, flat
            >>> scores, preds = torch.full((1, 2, 1), 0.9), gt.expand(1, 2, 4).contiguous()
            >>> labels, mask = torch.tensor([[0]]), torch.tensor([[True]])
            >>> two(scores, preds, points, gt, labels, mask).fg_mask
            tensor([[True, True]])
            >>> two(scores, preds, points, gt, labels, mask, rotated).fg_mask
            tensor([[ True, False]])
        """
        with torch.no_grad():
            batch, num_anchors = pred_scores.shape[0], pred_scores.shape[1]
            # Guarded by `__debug__` because reading the verdict costs a device sync,
            # and this is the only one the assignment still pays (see
            # `_resolve_conflicts`). Under `python -O` the assigner runs sync-free and
            # a bad label reverts to the silent mis-scoring `_check_labels` describes.
            if __debug__:
                _check_labels(gt_labels, gt_mask, pred_scores.shape[-1])
            if gt_boxes.shape[1] == 0:
                return self._empty_result(batch, num_anchors, pred_boxes.dtype, pred_boxes.device)

            candidate_mask = self._candidate_mask(anchor_points, gt_boxes, gt_mask, gt_rboxes)
            align_metric, iou = self._alignment_metric(pred_scores, pred_boxes, gt_boxes, gt_labels, candidate_mask)
            mask_pos = self._select_topk(align_metric, candidate_mask)
            mask_pos = self._resolve_conflicts(mask_pos, align_metric)
            mask_pos = self._finalize_mask(mask_pos, align_metric)
            return self._build_result(mask_pos, align_metric, iou, gt_boxes, gt_labels)

    def _candidate_mask(
        self,
        anchor_points: Tensor,
        gt_boxes: Tensor,
        gt_mask: Tensor,
        gt_rboxes: Tensor | None = None,
    ) -> Tensor:
        """Eligibility mask ``(B, N, A)``: anchor centre inside a real GT box.

        Subclasses (see :class:`lucid_yolo.assign.stal.SmallTargetAssigner`) override
        this single step to filter against a surrogate box while leaving every
        downstream stage on the original ground truth; the base implementation is
        stateless and ignores ``self``.

        With ``gt_rboxes`` supplied the containment test becomes the
        point-in-rotated-rect test of
        :func:`~lucid_yolo.data.rotated_geom.points_in_rboxes` and ``gt_boxes`` is not
        read here at all — it stays the box every later stage scores against. Both
        tests are edge-inclusive, so an anchor exactly on a boundary counts as a
        candidate either way (A25 inherits that choice from the primitive).
        """
        if gt_rboxes is not None:
            _check_rbox_batch(gt_rboxes, gt_boxes)
            return _rotated_inside(anchor_points, gt_rboxes) & gt_mask.unsqueeze(-1)
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

        Non-candidates are pushed *below* every candidate before the ranking rather
        than merely zeroed alongside them. Zeroing alone is not enough: ``t`` is zero
        at a candidate too whenever the prediction misses its ground truth entirely —
        the state a freshly initialized head is in, since :func:`decode_ltrb` applies
        no non-negativity and inverted distances decode to zero-IoU boxes. With the
        whole row tied at zero, ``topk`` fills its ``k`` slots from the global tie
        order, which need not contain a single one of *this* ground truth's
        candidates, and the intersection below then returns zero positives for a
        ground truth that had several. Filling non-candidates with ``-1`` (the
        sentinel :meth:`_resolve_conflicts` already uses, and safe because ``t`` is
        non-negative by construction) makes the ranking pick candidates first, so a
        ground truth with ``c`` candidates always keeps ``min(k, c)`` of them. The
        intersection with ``candidate_mask`` still matters when ``k`` exceeds ``c``:
        those surplus slots land on the ``-1`` fill and are dropped here.
        """
        k = min(self.topk, align_metric.shape[-1])
        ranked = align_metric.masked_fill(~candidate_mask, -1.0)  # (B, N, A)
        _, topk_index = ranked.topk(k, dim=-1, largest=True)  # (B, N, k)
        mask_topk = torch.zeros_like(align_metric, dtype=torch.bool)
        mask_topk.scatter_(-1, topk_index, True)
        return mask_topk & candidate_mask

    @staticmethod
    def _resolve_conflicts(mask_pos: Tensor, align_metric: Tensor) -> Tensor:
        """Give each anchor to the highest-``t`` GT that selected it; ``(B, N, A)``.

        Resolution runs unconditionally, with no early exit for the common case in
        which no anchor was claimed twice. That skip is decidable —
        ``selected_per_anchor.max() <= 1`` settles it — but only on the *host*, so
        taking it costs a device sync on every assignment call to save work the
        ``torch.where`` below already discards anchor by anchor. Measured at
        640-input shapes (``B=16``, ``A=8400``, ``N=20``, four repeats of a
        median-of-50 over the public ``__call__``): dropping the sync is 2-5%
        faster on MPS in both the contested and the uncontested regime, and within
        run-to-run noise on CPU, where the delta flips sign across repeats. The two
        forms are bit-identical on every returned field — with nothing contested,
        ``contested`` is all-``False`` and ``torch.where`` returns ``mask_pos``
        untouched — so the skip was only ever an optimization, and a negative one.
        """
        selected_per_anchor = mask_pos.sum(dim=1)  # (B, A)
        num_gt = mask_pos.shape[1]
        contested = (selected_per_anchor > 1).unsqueeze(1)  # (B, 1, A)
        masked_align = align_metric.masked_fill(~mask_pos, -1.0)
        best_gt = masked_align.argmax(dim=1)  # (B, A) — a GT that selected the anchor
        is_best = F.one_hot(best_gt, num_gt).permute(0, 2, 1).bool()  # (B, N, A)
        return torch.where(contested, is_best & mask_pos, mask_pos)

    def _finalize_mask(self, mask_pos: Tensor, align_metric: Tensor) -> Tensor:
        """Post-conflict hook on the positive mask; base assigner is a no-op.

        Runs after conflict resolution and immediately before the targets are
        gathered, so it sees a mask in which every anchor already belongs to at
        most one ground truth. The base implementation returns ``mask_pos``
        unchanged and ignores ``self``; subclasses override this single seam to
        reduce the mask further (see :class:`lucid_yolo.assign.one_to_one.UniqueAssigner`,
        which keeps a single positive per ground truth for the one-to-one branch).

        Args:
            mask_pos: ``(B, N, A)`` bool positive mask after conflict resolution.
            align_metric: ``(B, N, A)`` candidate-masked alignment metric ``t``,
                non-negative at candidates and zero elsewhere.

        Returns:
            The ``(B, N, A)`` positive mask to gather targets from; the base
            assigner returns its input untouched.
        """
        del align_metric
        return mask_pos

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
        # The denominator is floored at the dtype's smallest normal, not at ``self.eps``.
        # ``t_max = s * u_max**6`` collapses as the sixth power of the overlap, so any
        # *absolute* floor is eventually larger than the quantity it guards: at IoU 0.01,
        # ``t_max`` is ~5e-13 and a 1e-9 floor shrinks every one of that ground truth's
        # weights by ~2000x, training a positive as background while ``fg_mask`` still
        # calls it foreground. Flooring at ``tiny`` only ever guards a true 0/0, and there
        # ``align_pos`` is zero too (``t_max`` is its own row maximum), so the product is
        # zero rather than NaN. Sub-normal ``t_max`` — IoU below ~1e-6 — still under-scales,
        # but such a positive carries a vanishing ``u_max`` weight either way.
        floor = torch.finfo(align_pos.dtype).tiny
        norm_align = align_pos * (u_max / t_max.clamp(min=floor))  # (B, N, A)
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
