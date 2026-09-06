# SPDX-License-Identifier: Apache-2.0
"""Small-Target-Aware Label Assignment (STAL) for the detection head (WP-026).

Transcribed by hand from the reproduction's technical specification section
3.3.3 (Eq. 4-6) and its operational consequence in section 4. STAL is a
minimal, surgical modification of the Task-Aligned Assigner (:mod:`lucid_yolo.assign.tal`):
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

For oriented ground truths (WP-061) the same surrogate is built by
:func:`surrogate_rboxes` on the rotated ``(w, h)`` and the containment test is the
rotated one. R1 sec. 3.3.3 states Eq. 4-6 generically over a box's dimensions and
never restates it for oriented boxes; reading those dimensions as the rotated
``(w, h)`` — the box's own edge lengths, not the sides of its axis-aligned envelope
— is this project's decision, registered as A25 and not the paper's words.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lucid_yolo.assign.tal import TaskAlignedAssigner
from lucid_yolo.data.rotated_geom import canonicalize

__all__ = ["SmallTargetAssigner", "surrogate_boxes", "surrogate_rboxes"]

#: Column count of a long-edge rotated box ``(cx, cy, w, h, theta)``.
_RBOX_DIM = 5


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
            stride, ``8.0`` at 640 input); must be finite and ``> 0``.
        s_ref: Replacement size for an inflated dimension (the next stride,
            ``16.0`` at 640 input); must be finite, ``> 0``, and ``>= s_min``. The pair
            is checked once when :class:`SmallTargetAssigner` is constructed
            (:func:`_check_surrogate_sizes`) and not again here, so a direct caller of
            this function gets the silent no-op, empty, or plane-spanning surrogate that
            check describes rather than an error.

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


def surrogate_rboxes(gt_rboxes: Tensor, s_min: float, s_ref: float) -> Tensor:
    """Build centre- and angle-preserving rotated surrogates with per-dimension clamping.

    The rotated counterpart of :func:`surrogate_boxes`: each rotated box keeps its
    centre and its ``theta``, and an edge shorter than ``s_min`` is replaced by
    ``s_ref``. Used only for the STAL candidate filter on oriented ground truths;
    scoring, targets, and regression keep the original boxes.

    A25 — the clamped dimensions are the box's **own** ``(w, h)``, the rotated edge
    lengths, not the sides of its axis-aligned envelope. R1 sec. 3.3.3 gives Eq. 4-6
    generically over a box's dimensions and never restates it for oriented boxes, so
    this reading is the project's, not the paper's.

    Inflating ``h`` past ``w`` would leave the long-edge convention (``w >= h``), so
    the result is passed through :func:`~lucid_yolo.data.rotated_geom.canonicalize`,
    which swaps the pair and turns ``theta`` by ``pi/2`` — the same rectangle, named
    the way the rest of the codebase expects. Containment is invariant under that
    move, so the candidate set does not depend on it.

    Args:
        gt_rboxes: ``(..., 5)`` rotated boxes ``(cx, cy, w, h, theta)``. Any leading
            batch/ground-truth axes are preserved.
        s_min: Edge length below which a dimension is inflated (the smallest stride,
            ``8.0`` at 640 input); must be finite and ``> 0``.
        s_ref: Replacement edge length for an inflated dimension (the next stride,
            ``16.0`` at 640 input); must be finite, ``> 0``, and ``>= s_min``, checked
            at assigner construction and not here on the same terms as
            :func:`surrogate_boxes`.

    Returns:
        A ``(..., 5)`` tensor of canonical rotated surrogate boxes sharing
        ``gt_rboxes``'s dtype, device, and leading shape.

    Examples:
        >>> import torch
        >>> boxes = torch.tensor([[[10.0, 10.0, 10.0, 4.0, 0.0]]])  # 10 long, 4 thin
        >>> # h -> 16 overtakes w, so the canonical answer swaps the pair: theta += pi/2
        >>> [round(v, 4) for v in surrogate_rboxes(boxes, 8.0, 16.0)[0, 0].tolist()]
        [10.0, 10.0, 16.0, 10.0, 1.5708]
    """
    cx, cy, width, height, theta = gt_rboxes.unbind(-1)
    width_tilde = torch.where(width < s_min, torch.full_like(width, s_ref), width)
    height_tilde = torch.where(height < s_min, torch.full_like(height, s_ref), height)
    inflated = torch.stack((cx, cy, width_tilde, height_tilde, theta), dim=-1)
    return canonicalize(inflated.reshape(-1, _RBOX_DIM)).reshape(inflated.shape)


def _check_surrogate_sizes(s_min: float, s_ref: float) -> None:
    """Raise :class:`ValueError` unless the surrogate sizes can widen a small ground truth.

    The pair *is* STAL: :func:`surrogate_boxes` maps a dimension ``d`` to
    ``s_ref if d < s_min else d`` and the candidate filter runs on the result. Each of
    the two has values that switch that mapping off — or invert it — rather than tune
    it, and ``torch.where`` accepts every one of them without complaint, so a bad value
    comes out as a silently wrong candidate set rather than as a failure.

    ``s_min`` is the threshold, compared against a box dimension that is never negative.
    At ``s_min <= 0`` — and at ``nan``, where every comparison is ``False`` — no
    dimension is ever below it, nothing is ever inflated, and the class degrades to the
    :class:`~lucid_yolo.assign.tal.TaskAlignedAssigner` it subclasses: the sub-``8x8``
    ground truth of the module docstring is back to zero candidates, which is the whole
    of what STAL exists to prevent. At ``+inf`` the opposite happens — *every* dimension
    counts as below the threshold, so every ground truth is filtered on an
    ``s_ref``-sided footprint instead of its own, and a large object loses candidates it
    genuinely covers.

    ``s_ref`` is the replacement, and it becomes the surrogate's whole width or height.
    At ``s_ref = 0`` the surrogate collapses to a point at the ground truth's centre; at
    ``s_ref < 0`` it is inverted, its ``x1`` above its ``x2``; at ``nan`` every corner is
    ``nan`` and every containment comparison is ``False``. All three leave an inflated
    ground truth with *zero* candidates — the same failure as a disabled ``s_min``, now
    aimed at exactly the smallest targets. At ``+inf`` the surrogate spans the plane and
    every anchor in the image becomes a candidate for that one ground truth.

    ``s_ref >= s_min`` is a coherence bound rather than the recipe. A dimension in
    ``(s_ref, s_min)`` is *shrunk* by a replacement below the threshold: the surrogate
    then sits strictly inside the original box, so the subclass hands the base assigner
    fewer candidates than the untouched ground truth would have had — a 6-wide box that
    vanilla TAL gives one candidate keeps none under ``s_min = 8``, ``s_ref = 4``.
    Equality is accepted and is the boundary: every dimension it touches is below
    ``s_min`` by definition, so it is still strictly widened. The module docstring's
    ``s_min = 8`` and ``s_ref = 16`` read as a strict inequality, but those are the two
    smallest strides at a 640-pixel input — a recipe, not a constraint — so only the
    non-strict bound is enforced here.

    Neither size is bounded in magnitude beyond that. A ceiling would be the image or
    feature-map extent and a floor the stride the boxes are being rescued onto, and the
    assigner is handed none of the three — its constructor takes no image size and its
    ``__call__`` no stride list, only anchor points that have already been built from
    one. A threshold too small to ever fire in practice is therefore indistinguishable
    here from a deliberately conservative one, and both are left to the caller's
    judgement; only the values that cannot fire *at all* are refused.

    Args:
        s_min: Dimension threshold below which a side is inflated; must be finite and
            ``> 0``.
        s_ref: Replacement side length for an inflated dimension; must be finite,
            ``> 0``, and ``>= s_min``.

    Raises:
        ValueError: If either size is non-finite or non-positive, or if ``s_ref`` lies
            below ``s_min``. The message names the offending parameter and its value.

    Examples:
        >>> _check_surrogate_sizes(8.0, 16.0)
        >>> _check_surrogate_sizes(8.0, 8.0)  # the accepted boundary: still widens
        >>> _check_surrogate_sizes(0.0, 16.0)
        Traceback (most recent call last):
        ValueError: s_min must be finite and > 0, got 0.0
        >>> _check_surrogate_sizes(8.0, 4.0)
        Traceback (most recent call last):
        ValueError: s_ref must be >= s_min, got 4.0 < 8.0
    """
    for name, size in (("s_min", s_min), ("s_ref", s_ref)):
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {size}")
    # Both sizes are finite by here, so this comparison cannot quietly pass on a ``nan``
    # the way ``nan >= nan`` would have had the two checks run in the other order.
    if s_ref < s_min:
        raise ValueError(f"s_ref must be >= s_min, got {s_ref} < {s_min}")


class SmallTargetAssigner(TaskAlignedAssigner):
    """Task-Aligned assigner with small-target-aware candidate filtering (STAL).

    Identical to :class:`~lucid_yolo.assign.tal.TaskAlignedAssigner` except that the
    centre-inside candidate test runs against a per-ground-truth surrogate box
    (see :func:`surrogate_boxes`) whose dimensions below ``s_min`` are inflated to
    ``s_ref``. All other stages — alignment metric, top-k, conflict resolution,
    target boxes, and normalized weights — are inherited unchanged and operate on
    the original ground-truth boxes.

    Passing ``gt_rboxes`` to :meth:`~lucid_yolo.assign.tal.TaskAlignedAssigner.__call__`
    switches the same filter to rotated ground truths: the surrogate is built by
    :func:`surrogate_rboxes` on the rotated ``(w, h)`` and the containment test becomes
    point-in-rotated-rect (A25).

    Args:
        topk: Number of highest-alignment anchors kept per ground truth.
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
            stride, ``16.0`` at 640 input); finite, ``> 0``, and ``>= s_min``. The
            strict ``s_ref > s_min`` of the stride recipe is documented but not
            enforced; the non-strict bound is (see :func:`_check_surrogate_sizes`).

    Raises:
        ValueError: If ``topk``, ``alpha``, ``beta`` or ``eps`` falls outside the base
            assigner's range, or if ``s_min`` or ``s_ref`` takes a value that disables
            or inverts the surrogate inflation (see :func:`_check_surrogate_sizes`).

    Examples:
        >>> import torch
        >>> from lucid_yolo.assign import make_anchor_points
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
        _check_surrogate_sizes(s_min, s_ref)
        self.s_min = s_min
        self.s_ref = s_ref

    def _candidate_mask(
        self,
        anchor_points: Tensor,
        gt_boxes: Tensor,
        gt_mask: Tensor,
        gt_rboxes: Tensor | None = None,
    ) -> Tensor:
        """Eligibility mask against the surrogate box; ``(B, N, A)``.

        The only overridden stage: the containment test uses surrogate boxes (small
        dimensions inflated to ``s_ref``) so tiny ground truths gain candidates. Every
        downstream stage still sees the original ``gt_boxes``.

        With ``gt_rboxes`` supplied the surrogate is the rotated one
        (:func:`surrogate_rboxes`, clamped on the rotated ``(w, h)`` per A25) and the
        base class runs its rotated containment test; ``gt_boxes`` is handed over
        untouched because that path does not read it.
        """
        if gt_rboxes is not None:
            surrogate_r = surrogate_rboxes(gt_rboxes, self.s_min, self.s_ref)
            return super()._candidate_mask(anchor_points, gt_boxes, gt_mask, surrogate_r)
        surrogate = surrogate_boxes(gt_boxes, self.s_min, self.s_ref)
        return super()._candidate_mask(anchor_points, surrogate, gt_mask)
