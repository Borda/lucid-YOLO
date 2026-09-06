# SPDX-License-Identifier: Apache-2.0
"""Probabilistic-IoU (ProbIoU) rotated-box regression loss (WP-059, A19, A41).

Transcribed by hand from R17 (arXiv:2106.06072) for the long-edge rotated boxes
``(cx, cy, w, h, theta)`` WP-055 establishes. This is the rotated sibling of
:mod:`lucid_yolo.losses.ciou`: same paired-rows contract, same epsilon discipline, and
the same refusal to let a degenerate box produce a ``NaN``.

Each box becomes a Gaussian whose covariance is R17's uniform-density second moment,
``Sigma = R diag(w^2/12, h^2/12) R^T``, i.e. with ``a' = w^2/12`` and ``b' = h^2/12``::

    a = a' cos^2(theta) + b' sin^2(theta)
    b = a' sin^2(theta) + b' cos^2(theta)
    c = 0.5 (a' - b') sin(2 theta)

The two Gaussians are compared by their Bhattacharyya distance ``B_D = B1 + B2``::

    B1 = (1/4) [ (a1+a2)(y1-y2)^2 + (b1+b2)(x1-x2)^2 + 2(c1+c2)(x2-x1)(y1-y2) ]
               / [ (a1+a2)(b1+b2) - (c1+c2)^2 ]

    B2 = (1/2) ln( [ (a1+a2)(b1+b2) - (c1+c2)^2 ]
                   / [ 4 sqrt( (a1 b1 - c1^2)(a2 b2 - c2^2) ) ] )

whence the Bhattacharyya coefficient ``B_C = exp(-B_D)``, the Hellinger distance
``H_D = sqrt(1 - B_C)`` in ``[0, 1]``, and ``ProbIoU = 1 - H_D``. R17 proposes two
losses and suggests starting training with the second and switching to the first:
``L1 = H_D = 1 - ProbIoU`` on ``[0, 1]``, and ``L2 = B_D`` on ``[0, inf)``, which has no
vanishing gradient. Both are exposed here (:func:`probiou_hellinger_loss`,
:func:`probiou_bhattacharyya_loss`); which one the oriented training path selects is
WP-088's choice, not this module's.

Numerics
    The formulas above are *evaluated* differently than they are written, because as
    written they cancel. ``(a1+a2)(b1+b2) - (c1+c2)^2`` is the determinant of the summed
    covariance, and for a thin box the two products agree to several digits before the
    subtraction; ``B1``'s numerator is a positive-definite quadratic form assembled from
    a signed cross term the same way. The identities that remove both subtractions:

    - The 2x2 adjugate is **linear**, so ``adj(Sigma1 + Sigma2) = adj(Sigma1) +
      adj(Sigma2)``, and ``B1``'s numerator — which is exactly ``d^T adj(Sigma1+Sigma2)
      d`` for the centre offset ``d`` — becomes ``sum_i [ b'_i (d . u_i)^2 + a'_i
      (d . v_i)^2 ]`` over each box's own axes ``u_i = (cos, sin)``, ``v_i = (-sin, cos)``.
    - ``det(Sigma1 + Sigma2) = det Sigma1 + det Sigma2 + tr(Sigma1 adj Sigma2)``, which in
      the box parameters is ``a'1 b'1 + a'2 b'2 + (a'1 b'2 + a'2 b'1) cos^2(dtheta) +
      (a'1 a'2 + b'1 b'2) sin^2(dtheta)`` for ``dtheta = theta1 - theta2``.
    - Rotation preserves the determinant, so ``det Sigma_i = w_i^2 h_i^2 / 144`` exactly,
      and ``B2 = (1/2) ln(1 + e)`` with the non-negative excess
      ``e = [ (w1 h1 - w2 h2)^2 + (w1 h2 - w2 h1)^2 cos^2(dtheta)
      + (w1 w2 - h1 h2)^2 sin^2(dtheta) ] / (4 w1 h1 w2 h2)``.

    Every accumulation is now a sum of like-signed terms. What that buys, measured in
    float32 against a float64 evaluation of the literal form above
    (``tests/losses/test_probiou.py`` pins the agreement):

    - ProbIoU is exactly scale invariant — the same pair scaled from 1e3 to 1e6 px gives
      bit-comparable results — so sheer magnitude is not the problem WP-057's shoelace
      had. **Aspect ratio** is. Over a randomized 1024 px sweep the form here holds
      1.7e-6 to 2.9e-6 relative at every ratio from 1:1 to 1000:1, while the literal one
      goes 1.5e-5 at 1:1, 2.2e-5 at 10:1, 1.6e-3 at 100:1, and at 1000:1 returns ``NaN``
      for part of the sweep, its summed determinant having cancelled to a negative value
      before the logarithm. DOTA's bridges and harbours are the boxes that reach those
      ratios.
    - The converged regime is where it decides the loss. For 20:1 boxes differing by 1e-4
      of their own size, the literal float32 ``B_D`` is off by 377% and its ``L1`` by
      2.2e-3 in absolute terms; the form used here holds ``L1`` to 3.2e-6.
    - ``B_D >= 0`` holds by construction (a ratio of positive sums plus a ``log1p`` of a
      non-negative quantity), so ``1 - B_C`` never goes negative and the square root is
      always real.
    - Identity is exact rather than approximate: every squared difference above vanishes
      bit for bit when a box is compared with itself, so ``B_D`` is exactly ``0`` and
      ``ProbIoU`` exactly ``1``.

    The obvious alternative — float64 internals cast back out, this repository's habit
    for geometry (``data/affine.py``, ``eval/segment_decode.py``) — is not available: a
    regression loss runs on the training device, and D12c puts training on Apple MPS,
    which has no float64. Working precision therefore follows the input dtype **at
    float32 and above**, and the algebra is what makes float32 sufficient. Head-room is
    ample either way: the widest intermediate is a product of four sides, so float32
    saturates only beyond ~1e9 px.

    Below float32 the algebra is not enough, so :func:`_working_dtype` raises the working
    precision to float32 and the result is cast back to the caller's dtype. The head-room
    argument is what fails first: float16 saturates at 65504, and the ``excess`` above is
    a sum of squared side-products over a product of four sides, so a merely *finite*
    degenerate pair — a 1e-3 px box against a 20x3 one — overflows its numerator to
    ``Inf`` and reports an infinite ``B_D`` for two boxes that are simply far apart. The
    Hellinger branch fails second and more quietly: the ``sqrt`` derivative at the
    radicand floor is ~5e9, which is itself ``Inf`` in float16, so a *coincident* pair —
    the case :data:`_RADICAND_FLOOR` exists to keep differentiable — backpropagated
    ``NaN``. bfloat16 has float32's exponent range and so overflows neither, but its
    8-bit mantissa loses the cancellation-free form's whole point. Since the promotion is
    a no-op cast for float32 and float64 inputs, it changes no value any current caller
    computes; a true ``B_D`` above 65504 still returns ``Inf`` in float16, which is an
    honest overflow of the dtype the caller asked for rather than an artefact of the
    intermediate arithmetic.

    ``1 - B_C`` is computed as ``-expm1(-B_D)``: at ``B_D = 1e-8`` the naive difference
    returns exactly ``0`` in float32, which would report a 1e-4 Hellinger distance as
    perfect overlap.

Degenerate boxes (A41)
    Both denominators vanish for a zero-area box, so sides are floored at ``min_side``,
    1e-4 in whatever coordinate frame the caller works in. ``B_D`` depends on that floor
    only logarithmically — one decade of floor moves it by ``ln 10`` — so the value is
    not load-bearing; it is chosen an order of magnitude at a time. Below ~1e-10 the
    product of four floored sides underflows float32 and a degenerate-against-degenerate
    pair returns ``NaN``, and above ~1e-3 the floor would start to distort genuinely
    small boxes in a normalized frame. A clamped side has zero gradient, which is the
    intended behaviour: the loss does not chase a box that has collapsed.
"""

import math

import torch
from torch import Tensor

#: Column count of a long-edge rotated box ``(cx, cy, w, h, theta)``.
_RBOX_DIM = 5
#: Default floor on box sides, in the caller's coordinate frame (A41).
_MIN_SIDE: float = 1e-4
#: Floor inside the Hellinger square root. ``sqrt`` has an infinite derivative at zero,
#: so a coincident pair would backpropagate ``0 * inf = NaN``; clamping the radicand
#: keeps the local derivative finite while the clamp itself zeroes the gradient there.
#: A coincident pair takes the exact-zero branch and never sees it; the smallest non-zero
#: ``B_D`` a float32 pair can produce is of order 1e-13, so the floor binds only in
#: float64 and only for pairs already indistinguishable at that dtype's resolution.
_RADICAND_FLOOR: float = 1e-20


def _check_min_side(min_side: float) -> None:
    """Reject a ``min_side`` that would disable the A41 floor (M-23).

    The floor is the only thing standing between a zero-area box and the vanishing
    denominators the module docstring describes. ``min_side = 0`` therefore removes
    the safeguard entirely and a degenerate pair returns ``NaN``; a negative floor
    never binds at all, since :meth:`~torch.Tensor.clamp` leaves every non-negative
    side untouched; and a non-finite one turns every box in the batch degenerate.
    All three are accepted silently by ``clamp`` itself, so the check is here.

    The *upper* end is left to the caller's judgement rather than enforced: the A41
    argument gives ~1e-10 and ~1e-3 as the outer bounds of a sensible frame, but
    both depend on the coordinate frame the caller works in, which this function
    cannot see.

    Args:
        min_side: The candidate side floor.

    Raises:
        ValueError: If ``min_side`` is not finite and strictly positive.

    Examples:
        >>> _check_min_side(1e-4)  # accepted: nothing returned
        >>> try:
        ...     _check_min_side(0.0)
        ... except ValueError as error:
        ...     print(error)
        min_side must be finite and > 0; got 0.0
    """
    if not math.isfinite(min_side) or min_side <= 0.0:
        raise ValueError(f"min_side must be finite and > 0; got {min_side}")


def probiou_bhattacharyya_loss(pred: Tensor, target: Tensor, min_side: float = _MIN_SIDE) -> Tensor:
    """Bhattacharyya distance ``B_D`` between aligned pairs of rotated boxes (R17's L2).

    R17's unbounded loss: zero for coincident boxes and growing without bound as they
    separate, which is the property the paper credits with not vanishing far from the
    target. Evaluated through the cancellation-free identities in the module docstring,
    so the result is non-negative by construction and exactly ``0`` for a box against
    itself.

    Args:
        pred: Predicted boxes of shape ``(..., 5)``, ``(cx, cy, w, h, theta)`` with
            ``theta`` in radians. Any leading shape broadcasts against ``target``.
        target: Target boxes of shape ``(..., 5)``, same convention.
        min_side: Floor applied to ``w`` and ``h`` before they become variances (A41);
            keeps a zero-area box finite instead of dividing by zero. Must be finite
            and strictly positive; ``0`` or a negative floor never binds and restores
            the division by zero it exists to prevent (M-23).

    Returns:
        ``B_D`` per pair, shaped like the broadcast of the two inputs without their last
        dimension. Non-negative, unbounded above.

    Raises:
        ValueError: If either input's last dimension is not ``5``, or if ``min_side``
            is not finite and strictly positive.

    Examples:
        >>> import torch
        >>> pred = torch.tensor([[3.0, 4.0, 9.0, 2.0, 0.4]])
        >>> target = torch.tensor([[5.0, 1.0, 7.0, 3.0, -0.2]])
        >>> probiou_bhattacharyya_loss(pred, target).round(decimals=4)
        tensor([1.8495])
    """
    _check_rboxes(pred, "pred")
    _check_rboxes(target, "target")
    _check_min_side(min_side)

    out_dtype = torch.result_type(pred, target)
    work_dtype = _working_dtype(out_dtype)
    pred, target = pred.to(work_dtype), target.to(work_dtype)

    p_cx, p_cy, p_w, p_h, p_theta = _unpack(pred, min_side)
    t_cx, t_cy, t_w, t_h, t_theta = _unpack(target, min_side)

    # Variances along each box's own long (w) and short (h) axes: R17's uniform-density
    # second moment for a rectangle.
    p_var_w, p_var_h = p_w * p_w / 12, p_h * p_h / 12
    t_var_w, t_var_h = t_w * t_w / 12, t_h * t_h / 12

    delta_theta = p_theta - t_theta
    cos_sq = torch.cos(delta_theta) ** 2
    sin_sq = torch.sin(delta_theta) ** 2

    centre_term = _centre_term(
        p_cx - t_cx, p_cy - t_cy, (p_theta, p_var_w, p_var_h), (t_theta, t_var_w, t_var_h), cos_sq, sin_sq
    )
    # det(Sigma1 + Sigma2) relative to 4 sqrt(det Sigma1 det Sigma2), minus one: the ratio
    # inside R17's B2 logarithm, written so that it is a sum of squares over a product of
    # sides and therefore exactly zero when the two covariances agree.
    excess = (
        (p_w * p_h - t_w * t_h) ** 2 + (p_w * t_h - t_w * p_h) ** 2 * cos_sq + (p_w * t_w - p_h * t_h) ** 2 * sin_sq
    ) / (4 * p_w * p_h * t_w * t_h)
    return (centre_term + 0.5 * torch.log1p(excess)).to(out_dtype)


def probabilistic_iou(pred: Tensor, target: Tensor, min_side: float = _MIN_SIDE) -> Tensor:
    """ProbIoU similarity ``1 - H_D`` between aligned pairs of rotated boxes.

    R17's Gaussian-overlap analogue of IoU: ``1`` for coincident boxes, decaying towards
    ``0`` as they separate in position, size or orientation. It is a Hellinger-distance
    similarity between two Gaussians, **not** the overlap area of two rectangles — R17
    claims no equality with the polygon IoU, which is WP-063's subject (A24).

    Args:
        pred: Predicted boxes of shape ``(..., 5)``, ``(cx, cy, w, h, theta)``.
        target: Target boxes of shape ``(..., 5)``, same convention.
        min_side: Floor applied to ``w`` and ``h`` (A41). Must be finite and strictly
            positive (M-23).

    Returns:
        ProbIoU per pair, in ``[0, 1]``; exactly ``1`` for a box against itself.

    Raises:
        ValueError: If either input's last dimension is not ``5``, or if ``min_side``
            is not finite and strictly positive.

    Examples:
        >>> import torch
        >>> box = torch.tensor([[3.0, 4.0, 9.0, 2.0, 0.4]])
        >>> probabilistic_iou(box, box)
        tensor([1.])
        >>> other = torch.tensor([[5.0, 1.0, 7.0, 3.0, -0.2]])
        >>> probabilistic_iou(box, other).round(decimals=4)
        tensor([0.0820])
    """
    return 1 - probiou_hellinger_loss(pred, target, min_side=min_side)


def probiou_hellinger_loss(pred: Tensor, target: Tensor, min_side: float = _MIN_SIDE) -> Tensor:
    """Hellinger distance ``H_D = 1 - ProbIoU`` per aligned pair (R17's L1).

    R17's bounded loss, and the one it suggests switching to once training has started
    with :func:`probiou_bhattacharyya_loss`. Its derivative with respect to ``B_D`` is
    unbounded as the boxes coincide — inherent to the square root, not to this
    implementation — so a coincident pair is given exactly ``0`` with a zero gradient
    rather than the ``NaN`` an unguarded ``sqrt`` backward would produce.

    Args:
        pred: Predicted boxes of shape ``(..., 5)``, ``(cx, cy, w, h, theta)``.
        target: Target boxes of shape ``(..., 5)``, same convention.
        min_side: Floor applied to ``w`` and ``h`` (A41). Must be finite and strictly
            positive (M-23).

    Returns:
        Loss per pair, in ``[0, 1]``; exactly ``0`` for a box against itself and
        approaching ``1`` for boxes with no meaningful overlap.

    Raises:
        ValueError: If either input's last dimension is not ``5``, or if ``min_side``
            is not finite and strictly positive.

    Examples:
        >>> import torch
        >>> pred = torch.tensor([[3.0, 4.0, 9.0, 2.0, 0.4]])
        >>> target = torch.tensor([[5.0, 1.0, 7.0, 3.0, -0.2]])
        >>> probiou_hellinger_loss(pred, target).round(decimals=4)
        tensor([0.9180])
    """
    # The promotion has to wrap the square root as well as the covariance arithmetic, not
    # just be inherited from the call below: the derivative of `sqrt` at _RADICAND_FLOOR is
    # ~5e9, which is `Inf` in float16, so a coincident pair backpropagates `0 * Inf = NaN`
    # however exactly `B_D` was computed.
    out_dtype = torch.result_type(pred, target)
    work_dtype = _working_dtype(out_dtype)
    distance = probiou_bhattacharyya_loss(pred.to(work_dtype), target.to(work_dtype), min_side=min_side)
    # 1 - exp(-B_D) directly: the naive difference underflows to exactly 0 in float32
    # around B_D = 1e-8, reporting a small Hellinger distance as a perfect match.
    radicand = -torch.expm1(-distance)
    hellinger = torch.where(radicand > 0, torch.sqrt(radicand.clamp(min=_RADICAND_FLOOR)), torch.zeros_like(radicand))
    return hellinger.to(out_dtype)


def _working_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the precision the covariance arithmetic runs in for a given input dtype.

    Identity at float32 and above — those inputs keep the exact behaviour the module
    docstring's error tables were measured with — and float32 for the half dtypes, whose
    exponent range (float16) or mantissa (bfloat16) the cancellation-free algebra cannot
    compensate for on its own.

    Args:
        dtype: The dtype :func:`torch.result_type` gives the two box arguments.

    Returns:
        The dtype the intermediate covariance and division work is carried out in.

    Examples:
        >>> import torch
        >>> _working_dtype(torch.float16), _working_dtype(torch.bfloat16)
        (torch.float32, torch.float32)
        >>> _working_dtype(torch.float32), _working_dtype(torch.float64)
        (torch.float32, torch.float64)
    """
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _centre_term(
    offset_x: Tensor,
    offset_y: Tensor,
    pred_gaussian: tuple[Tensor, Tensor, Tensor],
    target_gaussian: tuple[Tensor, Tensor, Tensor],
    cos_sq: Tensor,
    sin_sq: Tensor,
) -> Tensor:
    """Evaluate R17's ``B1`` from the centre offset resolved in each box's own frame.

    The numerator is ``d^T adj(Sigma1 + Sigma2) d``; the adjugate is linear in 2x2, so it
    splits into one non-negative contribution per box (module docstring). The denominator
    is ``4 det(Sigma1 + Sigma2)``, expanded the same cancellation-free way.

    Args:
        offset_x: ``x1 - x2`` centre offset.
        offset_y: ``y1 - y2`` centre offset.
        pred_gaussian: ``(theta, var_w, var_h)`` of the first box.
        target_gaussian: ``(theta, var_w, var_h)`` of the second box.
        cos_sq: ``cos^2(theta1 - theta2)``.
        sin_sq: ``sin^2(theta1 - theta2)``.

    Returns:
        ``B1`` per pair, non-negative.

    Examples:
        >>> import torch
        >>> one = (torch.tensor([0.0]), torch.tensor([3.0]), torch.tensor([1.0]))
        >>> zero, unit = torch.zeros(1), torch.ones(1)
        >>> _centre_term(unit, zero, one, one, unit, zero).round(decimals=4)
        tensor([0.0417])
    """
    p_theta, p_var_w, p_var_h = pred_gaussian
    t_theta, t_var_w, t_var_h = target_gaussian
    p_along, p_across = _resolve(offset_x, offset_y, p_theta)
    t_along, t_across = _resolve(offset_x, offset_y, t_theta)
    numerator = p_var_h * p_along**2 + p_var_w * p_across**2 + t_var_h * t_along**2 + t_var_w * t_across**2
    summed_det = (
        p_var_w * p_var_h
        + t_var_w * t_var_h
        + (p_var_w * t_var_h + t_var_w * p_var_h) * cos_sq
        + (p_var_w * t_var_w + p_var_h * t_var_h) * sin_sq
    )
    return numerator / (4 * summed_det)


def _resolve(offset_x: Tensor, offset_y: Tensor, theta: Tensor) -> tuple[Tensor, Tensor]:
    """Project a centre offset onto a box's own axes, ``(along w, along h)``.

    Examples:
        >>> import math, torch
        >>> along, across = _resolve(torch.ones(1), 2 * torch.ones(1), torch.tensor([math.pi / 2]))
        >>> [round(float(v), 4) for v in (along, across)]
        [2.0, -1.0]
    """
    cos, sin = torch.cos(theta), torch.sin(theta)
    return offset_x * cos + offset_y * sin, offset_y * cos - offset_x * sin


def _unpack(rboxes: Tensor, min_side: float) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Split rotated boxes into their five columns, flooring the two side lengths (A41).

    Examples:
        >>> import torch
        >>> flat = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.5]])
        >>> [round(float(v), 4) for v in _unpack(flat, 1e-4)]
        [1.0, 2.0, 3.0, 0.0001, 0.5]
    """
    centre_x, centre_y, width, height, theta = rboxes.unbind(dim=-1)
    return centre_x, centre_y, width.clamp(min=min_side), height.clamp(min=min_side), theta


def _check_rboxes(rboxes: Tensor, name: str) -> None:
    """Raise :class:`ValueError` unless ``rboxes`` has a trailing rotated-box dimension.

    Deliberately looser than :func:`lucid_yolo.data.rotated_geom._check_2d`, which
    requires exactly 2-D: everything here is elementwise in the leading shape, so a
    ``(4, 1, 5)`` prediction against a ``(1, 3, 5)`` target is a well-defined pairwise
    matrix and refusing it would be the guard inventing a restriction the arithmetic
    does not have. The two are not a duplicate pair, whatever the shared name suggests
    (WP-091e). Only the trailing five is structural — it is what :func:`_unpack`
    unbinds — so that is all this asserts, and the message says ``(..., 5)`` rather
    than naming rows the function never counts.

    Examples:
        >>> import torch
        >>> _check_rboxes(torch.zeros((0, 5)), "pred")
        >>> _check_rboxes(torch.zeros((2, 3, 5)), "pred")
    """
    if rboxes.ndim == 0 or rboxes.shape[-1] != _RBOX_DIM:
        raise ValueError(f"{name} must be (..., {_RBOX_DIM}); got shape {tuple(rboxes.shape)}")
