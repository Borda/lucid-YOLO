# SPDX-License-Identifier: Apache-2.0
"""Square-object angle loss for oriented boxes (WP-060, A22, A42).

Transcribed by hand from R1 sec. 3.4.3, Eq. 14-15. It is the auxiliary term that
disambiguates *square* boxes, which the rotated IoU loss cannot: R1 states that "the
double-angle penalty is used as auxiliary supervision for square and near-square boxes,
for which rotations separated by 90deg become geometrically ambiguous. Elongated boxes
receive smaller ``omega_i`` and remain primarily constrained by the rotated IoU loss" —
that IoU loss being :mod:`lucid_yolo.losses.probiou` (WP-059).

The premise is the one WP-055 builds the long-edge convention on: "since an oriented
rectangle is unchanged under a 180deg rotation, ``(x, y, w, h, theta)`` and
``(x, y, w, h, theta + pi)`` represent the same geometry. The angular residual should
therefore be measured modulo ``pi`` rather than on the real line." Hence Eq. 14::

    d_theta_i       = theta_hat_i - theta_star_i
    d_theta_tilde_i = d_theta_i - round(d_theta_i / pi) * pi

and Eq. 15, over the foreground assignments ``F`` with TAL's assignment weights ``q_i``::

    L_angle = (1/S) * sum_{i in F} q_i * omega_i * sin^2(2 * d_theta_tilde_i)
    S       = max(sum_{i in F} q_i, 1)
    omega_i = exp( - ln^2(w_star_i / h_star_i) / lambda^2 )

``lambda = 3`` is the paper's own value rather than an assumption of this project: R1
Table 11 measures 50.2 mAP there against 49.0 with no angle loss at all, and — the reason
the value is not a free knob — 47.1 at ``lambda = 5``, which is *worse* than omitting the
term. Larger ``lambda`` widens ``omega`` until elongated boxes
receive the double-angle penalty too, and for those a 90deg error is a genuine error the
IoU term already punishes correctly.

The three-factor structure is what makes the term auxiliary rather than a second angle
objective: ``q_i`` restricts it to well-aligned foreground anchors, ``omega_i`` restricts
it to near-square targets (1.000 at 1:1, 0.948 at 2:1, 0.750 at 5:1, 0.555 at 10:1, 0.369
at 20:1 — measured, ``lambda = 3``, natural log), and ``sin^2(2 d_theta_tilde)`` is zero
at both 0 and 90deg, so it penalizes being *diagonally* wrong without asking a square to
choose between its two indistinguishable orientations.

Weight in the total objective
    A22: the scalar weight of ``L_angle`` in the total objective is ``1.0``, and R1 does
    not state it — a registered gap, not a derivation. :func:`square_angle_loss` therefore
    returns the **pre-gain** term, following
    :class:`~lucid_yolo.losses.detection_loss.DetectionLossOutput`'s convention, and the
    caller that assembles the oriented objective (WP-088) owns the gain.

The wrapped range, and the tie
    Eq. 14 as written leans on ``round``, whose tie rule is unstated. It matters:
    :func:`torch.round` breaks halves to even, so ``round(0.5) = 0`` but
    ``round(1.5) = 2``, and ``d_theta = pi/2`` would wrap to ``+pi/2`` while
    ``d_theta = 3*pi/2`` — the same residual — wrapped to ``-pi/2``. The interval would be
    the closed ``[-pi/2, pi/2]``, with its two ends both attainable and equivalent.

    :func:`wrap_angle_delta` breaks the tie **upwards** instead (``n = floor(d/pi + 1/2)``,
    equivalently ``round`` with halves away from zero on the positive side), which lands
    every residual in the half-open ``[-pi/2, pi/2)`` — length ``pi``, no gap, no overlap,
    and both ties at ``d_theta = +-pi/2`` mapping to the single representative ``-pi/2``.
    It differs from :func:`torch.round` only on the exact half-integers, where both answers
    describe the same rotation; and since ``sin^2(2x)`` has period ``pi/2``, the choice is
    invisible in the loss. It is made anyway, because a range the caller may rely on is
    worth more than a tie nobody hits.

    The range is a guarantee, not a near-certainty, so the implementation follows
    :func:`lucid_yolo.data.rotated_geom._wrap_theta`: an in-range input short-circuits (so
    the wrap is exactly idempotent), and the remainder path is clamped to the neighbour
    below ``pi/2`` in the working dtype, because ``torch.remainder`` may return its own
    divisor for an input an ulp below a multiple of it.

The quarter-turn fold
    ``sin^2(2x)`` is evaluated after folding the residual once more, from
    ``[-pi/2, pi/2)`` into ``[-pi/4, pi/4)`` by ``+-pi/2``. Mathematically this is free —
    the function's period is ``pi/2`` — but numerically it is what makes the square
    tie-break of :func:`~lucid_yolo.data.rotated_geom.canonicalize` cost nothing.

    That function folds an exact square towards zero, so a square target arrives as one of
    two representatives ``pi/2`` apart. Their residuals differ by ``pi/2``, and ``sin``
    evaluated at two arguments that differ by a floating-point ``pi/2`` does not return
    bit-equal results — measured 0.22984886 against 0.22984877, agreeing to seven digits
    and no further, for one pair of a square's two representatives. The
    fold reconciles them *before* the sine: for ``|u|`` in ``[pi/4, pi/2]`` the shift by
    ``pi/2`` is a Sterbenz subtraction (the operands are within a factor of two), hence
    exact, so both representatives collapse onto the same float and the loss is bit-equal
    rather than equal to tolerance. The ties fold likewise: ``d_theta = +-pi/2`` wraps to
    ``-pi/2`` and folds to exactly ``0``, so a 90deg error on a square is scored as a true
    zero.

Degenerate targets (A42)
    ``omega`` divides by the target height, so sides are floored at ``min_side`` — 1e-4,
    the same value :mod:`lucid_yolo.losses.probiou` floors at, though not for the same
    reason: A41 justifies that floor by ``B_D``'s logarithmic dependence on it, and nothing
    of that argument carries here. A target collapsed in *both* dimensions has ratio 1 and
    receives ``omega = 1``, i.e. **full** square-object weight, which is the honest
    consequence of the formula rather than a choice R1 supports — it says nothing about
    degenerate targets. A42 records it. The floor is what keeps the backward pass finite:
    an unfloored zero height sends ``ln(w/h)`` to ``inf`` and its gradient to ``NaN``.
"""

import math

import torch
from torch import Tensor

__all__ = ["aspect_ratio_weight", "square_angle_loss", "wrap_angle_delta"]

#: R1's aspect-ratio bandwidth for ``omega``; Table 11's best, stated by the paper.
_LAMBDA: float = 3.0
#: Default floor on target sides, in the caller's coordinate frame (A42).
_MIN_SIDE: float = 1e-4

_PI = math.pi
_HALF_PI = math.pi / 2
_QUARTER_PI = math.pi / 4


def wrap_angle_delta(delta: Tensor) -> Tensor:
    """Reduce an angular residual modulo ``pi`` into ``[-pi/2, pi/2)`` (R1 Eq. 14).

    The residual of two orientations, measured on the circle a rectangle actually lives on
    rather than on the real line: ``delta`` and ``delta + pi`` describe the same relative
    rotation and return the same value. Ties at ``+-pi/2`` go to ``-pi/2`` — see the module
    docstring for why the tie is settled here rather than left to :func:`torch.round`.

    Exactly idempotent: an already-wrapped value is returned bit for bit.

    Args:
        delta: Angular residuals in radians, any shape, any real magnitude.

    Returns:
        The residuals reduced into ``[-pi/2, pi/2)`` as represented in the input dtype,
        same shape and dtype.

    Examples:
        >>> import torch
        >>> wrap_angle_delta(torch.tensor([0.3, 3.4416, -3.4416, 1.5708])).round(decimals=4)
        tensor([ 0.3000,  0.3000, -0.3000, -1.5708])
    """
    in_range = (delta >= -_HALF_PI) & (delta < _HALF_PI)
    wrapped = torch.remainder(delta + _HALF_PI, _PI) - _HALF_PI
    wrapped = torch.where(wrapped < -_HALF_PI, wrapped + _PI, wrapped)
    wrapped = torch.where(wrapped >= _HALF_PI, wrapped - _PI, wrapped)
    low = torch.tensor(-_HALF_PI, dtype=wrapped.dtype, device=wrapped.device)
    wrapped = wrapped.clamp(min=low, max=_greatest_below_half_pi(wrapped))
    return torch.where(in_range, delta, wrapped)


def aspect_ratio_weight(width: Tensor, height: Tensor, lam: float = _LAMBDA, min_side: float = _MIN_SIDE) -> Tensor:
    """Per-target ``omega``, R1 Eq. 15's aspect-ratio-aware factor.

    A log-Gaussian in the target's aspect ratio: ``1`` for an exact square and decaying as
    the box elongates, so the double-angle penalty is spent on the boxes whose orientation
    is genuinely ambiguous. Symmetric under swapping ``width`` and ``height``, since the
    logarithm of the reciprocal ratio only changes sign.

    Args:
        width: Target box widths ``w_star``, any shape broadcastable against ``height``.
        height: Target box heights ``h_star``.
        lam: R1's ``lambda``, the bandwidth in log-ratio units (Table 11's ``3.0``).
            Larger values extend square-object supervision to more elongated boxes.
        min_side: Floor applied to both sides before the ratio is taken (A42), keeping a
            collapsed target finite instead of dividing by zero.

    Returns:
        ``omega`` per target, in ``(0, 1]``, shaped like the broadcast of the inputs.

    Raises:
        ValueError: If ``lam`` is not strictly positive.

    Examples:
        >>> import torch
        >>> ratios = torch.tensor([1.0, 2.0, 5.0, 10.0, 20.0])
        >>> aspect_ratio_weight(ratios, torch.ones(5)).round(decimals=4)
        tensor([1.0000, 0.9480, 0.7499, 0.5548, 0.3689])
    """
    if lam <= 0:
        raise ValueError(f"lam must be > 0; got {lam}")
    log_ratio = torch.log(width.clamp(min=min_side) / height.clamp(min=min_side))
    return torch.exp(-(log_ratio**2) / (lam * lam))


def square_angle_loss(
    pred_theta: Tensor,
    target_theta: Tensor,
    target_width: Tensor,
    target_height: Tensor,
    weights: Tensor,
    lam: float = _LAMBDA,
    min_side: float = _MIN_SIDE,
) -> Tensor:
    """Assignment-weighted square-object angle loss (R1 Eq. 15).

    The **pre-gain** term: A22 puts its weight in the total objective at ``1.0`` but R1
    does not state one, so the gain stays with the caller that assembles the oriented
    objective (WP-088), exactly as ``box``/``cls``/``l1`` are reported pre-gain by
    :class:`~lucid_yolo.losses.detection_loss.DetectionLossOutput`.

    Inputs are per-anchor and aligned. Passing only the foreground rows is the intended
    use, but a whole branch may be passed with background anchors carrying ``q_i = 0``:
    they contribute nothing to either the sum or the normalizer, which is the same
    arrangement :class:`~lucid_yolo.losses.detection_loss.DetectionBranchLoss` relies on.
    An empty foreground gives an exact zero that is still connected to ``pred_theta``, so
    a batch without positives needs no special case.

    Args:
        pred_theta: ``(...,)`` predicted orientations in radians. Any range: the residual
            is reduced modulo ``pi`` (Eq. 14), so an unwrapped head output is fine.
        target_theta: ``(...,)`` target orientations ``theta_star``, same shape.
        target_width: ``(...,)`` target widths ``w_star``, for ``omega``.
        target_height: ``(...,)`` target heights ``h_star``, for ``omega``.
        weights: ``(...,)`` TAL assignment weights ``q_i``, non-negative. Their sum,
            floored at ``1``, is R1's normalizer ``S``.
        lam: R1's ``lambda`` for ``omega`` (Table 11's ``3.0``).
        min_side: Floor applied to the target sides (A42).

    Returns:
        Scalar (zero-dimensional) loss, non-negative and bounded above by the largest
        ``omega`` present, carrying gradient back to ``pred_theta``.

    Raises:
        ValueError: If ``lam`` is not strictly positive.

    Examples:
        >>> import torch
        >>> pred = torch.tensor([0.10, 0.90, 0.30])
        >>> target = torch.tensor([0.05, 0.10, 0.30])  # third is already exact
        >>> width, height = torch.tensor([9.0, 8.0, 8.0]), torch.tensor([9.0, 8.0, 2.0])
        >>> square_angle_loss(pred, target, width, height, torch.tensor([0.6, 0.9, 0.5])).round(decimals=4)
        tensor(0.4526)
        >>> empty = torch.zeros(0)
        >>> square_angle_loss(empty, empty, empty, empty, empty)
        tensor(0.)
    """
    residual = _fold_quarter_turn(wrap_angle_delta(pred_theta - target_theta))
    penalty = torch.sin(2 * residual) ** 2
    omega = aspect_ratio_weight(target_width, target_height, lam=lam, min_side=min_side)
    return (weights * omega * penalty).sum() / weights.sum().clamp(min=1.0)


def _fold_quarter_turn(residual: Tensor) -> Tensor:
    """Fold a wrapped residual from ``[-pi/2, pi/2)`` into ``[-pi/4, pi/4)``.

    Free under ``sin^2(2x)``, whose period is ``pi/2``, and exact in floating point: the
    shift by ``pi/2`` is a Sterbenz subtraction for every value it touches. That is what
    makes the two representatives of a square target score bit-identically rather than to
    seven digits (module docstring).

    Examples:
        >>> import torch
        >>> _fold_quarter_turn(torch.tensor([0.2, 1.0, -1.0])).round(decimals=4)
        tensor([ 0.2000, -0.5708,  0.5708])
    """
    folded = torch.where(residual >= _QUARTER_PI, residual - _HALF_PI, residual)
    return torch.where(folded < -_QUARTER_PI, folded + _HALF_PI, folded)


def _greatest_below_half_pi(like: Tensor) -> Tensor:
    """Return the largest value below ``pi/2`` representable in ``like``'s dtype.

    Clamping to ``pi/2`` itself would leave the residual on the excluded end of the
    half-open range, and the neighbour below it differs between float32 and float64, so it
    cannot be written as one literal.

    Examples:
        >>> import torch
        >>> float(_greatest_below_half_pi(torch.zeros(1))) < math.pi / 2
        True
    """
    high = torch.tensor(_HALF_PI, dtype=like.dtype, device=like.device)
    return torch.nextafter(high, torch.full_like(high, -math.inf))
