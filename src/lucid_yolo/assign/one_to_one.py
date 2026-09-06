# SPDX-License-Identifier: Apache-2.0
"""One-to-one label assignment for the detection head's o2o branch (WP-028).

Transcribed by hand from the reproduction's technical specification sections
3.2.1 and 3.3.2 (R1) and the consistent dual-assignment lineage (R6). The
one-to-one branch trains with a *unique* assignment: exactly one anchor per
ground truth, so the branch learns NMS-free, one-prediction-per-object outputs.

:class:`UniqueAssigner` is a minimal, surgical extension of the
Small-Target-Aware assigner (:class:`~lucid_yolo.assign.stal.SmallTargetAssigner`).
It keeps STAL's small-target candidate filter and every scoring stage, then adds
a single secondary filter after the base pipeline has produced its positive
mask: among each ground truth's positives it keeps only the one anchor with the
highest alignment metric ``t = s**alpha * u**beta``. This is the ``topk2 = 1``
reduction applied on top of the ``topk = 7`` candidate set (R1 3.2.1): the wider
candidate set is selected and de-conflicted exactly as in STAL, then collapsed
to a unique assignment.

Ordering — reduce **after** conflict resolution. The base pipeline first gives
each contested anchor to the ground truth it aligns best with, and only then is
each ground truth collapsed to its single best *remaining* anchor. A ground
truth whose top anchor is claimed by a better-aligned neighbour therefore falls
back to its next candidate rather than losing its positive outright, so two
ground truths whose best anchors collide both keep a positive whenever candidates
remain.

Invariant: after assignment every ground truth holds **at most one** positive
anchor and every anchor holds at most one ground truth. A real ground truth with
at least one candidate keeps exactly one positive, with a single documented
exception — when several ground truths contend for the *same* lone candidate,
the highest-alignment ground truth wins it (ties broken by the lowest ground
-truth index, inherited from :meth:`TaskAlignedAssigner._resolve_conflicts`) and a
loser left with no other candidate gets zero positives. An anchor cannot serve
two ground truths in a one-to-one assignment, so this collision is irreducible.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lucid_yolo.assign.stal import SmallTargetAssigner

__all__ = ["UniqueAssigner"]


class UniqueAssigner(SmallTargetAssigner):
    """STAL assigner reduced to exactly one positive anchor per ground truth.

    Identical to :class:`~lucid_yolo.assign.stal.SmallTargetAssigner` — the same
    small-target surrogate candidate filter, alignment metric, top-``k`` select,
    and conflict resolution — with one extra stage: a ``topk2 = 1`` reduction
    (:meth:`_finalize_mask`) that keeps only each ground truth's highest-alignment
    anchor after conflict resolution. The default ``topk = 7`` matches the
    one-to-one recipe's candidate set (R1 3.2.1); the reduction makes the branch's
    assignment unique regardless of ``topk``.

    Args:
        topk: Size of the candidate set kept per ground truth before the unique
            reduction (the o2o recipe default ``7``).
        alpha: Exponent on the classification score in ``t = s**alpha * u**beta``
            (A2 default ``1.0``); finite and ``>= 0``.
        beta: Exponent on the IoU in ``t = s**alpha * u**beta`` (A2 default
            ``6.0``); finite and ``>= 0``.
        eps: Small constant guarding the IoU union; finite and ``> 0``. Validated
            by :class:`~lucid_yolo.assign.tal.TaskAlignedAssigner`, which also
            documents why the normalization denominator uses its own floor rather
            than this value.
        s_min: Dimension threshold below which the surrogate inflates a box side
            (the smallest stride, ``8.0`` at 640 input); finite and ``> 0``.
        s_ref: Replacement side length for an inflated dimension (the next
            stride, ``16.0`` at 640 input); finite, ``> 0``, and ``>= s_min``.
            Validated with ``s_min`` by
            :class:`~lucid_yolo.assign.stal.SmallTargetAssigner`, which this
            constructor forwards both to unchanged.

    Examples:
        >>> import torch
        >>> from lucid_yolo.assign import make_anchor_points
        >>> assigner = UniqueAssigner(topk=7)
        >>> points, _ = make_anchor_points([(4, 4)], [8])  # centres at 4, 12, 20, 28
        >>> gt = torch.tensor([[[0.0, 0.0, 32.0, 32.0]]])  # covers all 16 anchors
        >>> scores = torch.full((1, 16, 1), 0.9)
        >>> boxes = gt.expand(1, 16, 4).contiguous()  # every pred == the GT box
        >>> labels = torch.tensor([[0]])
        >>> mask = torch.tensor([[True]])
        >>> out = assigner(scores, boxes, points, gt, labels, mask)
        >>> int(out.fg_mask.sum())  # STAL(topk=7) would yield 7 here
        1
    """

    def __init__(
        self,
        topk: int = 7,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
        s_min: float = 8.0,
        s_ref: float = 16.0,
    ) -> None:
        super().__init__(topk, alpha, beta, eps, s_min, s_ref)

    def _finalize_mask(self, mask_pos: Tensor, align_metric: Tensor) -> Tensor:
        """Keep each ground truth's single highest-alignment anchor; ``(B, N, A)``.

        The ``topk2 = 1`` reduction. ``align_metric`` is non-negative at candidate
        anchors, so filling non-positives with ``-1`` lets an :func:`torch.argmax`
        along the anchor axis pick each ground truth's best *positive* anchor. The
        final ``& mask_pos`` drops the spurious pick a ground truth with no
        positives (all ``-1``) would otherwise receive, so such a ground truth
        stays empty rather than gaining a phantom anchor.

        Args:
            mask_pos: ``(B, N, A)`` bool positive mask after conflict resolution;
                each anchor already belongs to at most one ground truth.
            align_metric: ``(B, N, A)`` candidate-masked alignment metric ``t``.

        Returns:
            A ``(B, N, A)`` bool mask with at most one ``True`` per ground truth
            (the highest-``t`` anchor of its positives).

        Examples:
            >>> import torch
            >>> from lucid_yolo.assign.one_to_one import UniqueAssigner
            >>> mask = torch.tensor([[[True, True, False]]])  # one GT, two positives
            >>> align = torch.tensor([[[0.2, 0.9, 0.0]]])  # anchor 1 aligns best
            >>> UniqueAssigner()._finalize_mask(mask, align)
            tensor([[[False,  True, False]]])
        """
        masked_align = align_metric.masked_fill(~mask_pos, -1.0)  # (B, N, A)
        best_anchor = masked_align.argmax(dim=-1, keepdim=True)  # (B, N, 1)
        unique = torch.zeros_like(mask_pos)
        unique.scatter_(-1, best_anchor, True)
        return unique & mask_pos
