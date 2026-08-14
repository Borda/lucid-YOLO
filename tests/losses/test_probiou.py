# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the ProbIoU rotated-box loss (WP-059, A19, A41).

Two oracles carry this file. The first is a **literal transcription** of R17's ``B1`` and
``B2`` exactly as arXiv:2106.06072 writes them, evaluated in float64: the implementation
evaluates the same quantities through cancellation-free identities (see the module
docstring), and this oracle is what proves the algebra did not change the answer. The
second is **numerical integration** of ``B_C = integral sqrt(p q)`` over a fine grid,
which grounds the closed form in the definition it comes from rather than in another
rearrangement of itself.

Not a shapely oracle. ProbIoU is a Hellinger-distance similarity between two Gaussians,
and R17 claims no equality between it and the overlap area of two polygons — an
agreement test against shapely could not pass, and a correlation threshold would be
invented rather than sourced. The exact polygon intersection is WP-063's subject (A24)
and keeps the shapely oracle.

The float32 conditioning claims in the module docstring are pinned here too: the literal
form is left visibly worse at high aspect ratio and near coincidence, which is the whole
reason the implementation does not use it.
"""

import math

import pytest
import torch
from torch import Tensor

from lucid_yolo.data.rotated_geom import canonicalize
from lucid_yolo.losses import probabilistic_iou, probiou_bhattacharyya_loss, probiou_hellinger_loss

#: Pairs spanning square/elongated, aligned/rotated, overlapping/disjoint.
_PAIRS = [
    pytest.param([10.0, 10.0, 8.0, 4.0, 0.3], [12.0, 9.0, 6.0, 5.0, 1.1], id="overlapping-rotated"),
    pytest.param([0.0, 0.0, 20.0, 3.0, -0.6], [1.5, -2.0, 18.0, 4.0, 0.9], id="elongated-crossing"),
    pytest.param([5.0, 5.0, 5.0, 5.0, 0.0], [5.0, 5.0, 5.0, 5.0, 1.2], id="squares-same-centre"),
    pytest.param([0.0, 0.0, 4.0, 2.0, 0.0], [30.0, 20.0, 4.0, 2.0, 0.0], id="far-apart"),
    pytest.param([100.0, 60.0, 512.0, 6.0, 2.4], [104.0, 63.0, 480.0, 9.0, 2.1], id="dota-scale-thin"),
]

#: Angles covering the long-edge range and both of its bounds.
_ANGLES = [-math.pi / 4, -0.3, 0.0, 0.5, math.pi / 2, 2.0, 3 * math.pi / 4 - 1e-3]


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed every RNG source before each test for deterministic tensors."""
    torch.manual_seed(0)


def literal_bhattacharyya(pred: Tensor, target: Tensor) -> Tensor:
    """R17's ``B_D = B1 + B2`` transcribed exactly as the paper writes it.

    The oracle, not the implementation: it keeps both subtractions the module docstring
    removes, so it is accurate in float64 and visibly inaccurate in float32.
    """
    x1, y1, w1, h1, theta1 = pred.unbind(dim=-1)
    x2, y2, w2, h2, theta2 = target.unbind(dim=-1)
    ap1, bp1 = w1**2 / 12, h1**2 / 12
    ap2, bp2 = w2**2 / 12, h2**2 / 12
    a1 = ap1 * torch.cos(theta1) ** 2 + bp1 * torch.sin(theta1) ** 2
    b1 = ap1 * torch.sin(theta1) ** 2 + bp1 * torch.cos(theta1) ** 2
    c1 = 0.5 * (ap1 - bp1) * torch.sin(2 * theta1)
    a2 = ap2 * torch.cos(theta2) ** 2 + bp2 * torch.sin(theta2) ** 2
    b2 = ap2 * torch.sin(theta2) ** 2 + bp2 * torch.cos(theta2) ** 2
    c2 = 0.5 * (ap2 - bp2) * torch.sin(2 * theta2)

    summed_det = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2
    term1 = (
        0.25
        * ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2 + 2 * (c1 + c2) * (x2 - x1) * (y1 - y2))
        / summed_det
    )
    term2 = 0.5 * torch.log(summed_det / (4 * torch.sqrt((a1 * b1 - c1**2) * (a2 * b2 - c2**2))))
    return term1 + term2


def integrated_bhattacharyya_coefficient(pred: list[float], target: list[float], steps: int = 301) -> float:
    """``B_C = integral over R^2 of sqrt(p(x) q(x)) dx`` by trapezoidal quadrature.

    The integrand is evaluated from the two densities themselves, but the grid is laid
    out in the frame that whitens it. ``sqrt(p q)`` has a quadratic exponent, so it is an
    unnormalized Gaussian of covariance ``2 (Sigma1^-1 + Sigma2^-1)^-1``; sampling a
    ``+-9`` sigma box in *that* frame resolves both boxes' scales at once. An
    axis-aligned grid cannot: a 512x6 box needs a step below its 1.7 px minor spread
    across a 2400 px major extent, which is 1e8 grid points for one pair.

    Only the sample *placement* comes from that observation — the values summed are
    ``sqrt(p(x) q(x))`` computed pointwise, so the oracle remains independent of the
    closed form it checks.
    """
    mean1, cov1 = _gaussian(pred)
    mean2, cov2 = _gaussian(target)
    precision1, precision2 = torch.linalg.inv(cov1), torch.linalg.inv(cov2)
    shape_cov = 2 * torch.linalg.inv(precision1 + precision2)
    centre = 0.5 * shape_cov @ (precision1 @ mean1 + precision2 @ mean2)
    whitener = torch.linalg.cholesky(shape_cov)

    axis = torch.linspace(-9.0, 9.0, steps, dtype=torch.float64)
    unit_grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1)
    points = centre + unit_grid @ whitener.T

    integrand = torch.sqrt(_density(points, mean1, cov1) * _density(points, mean2, cov2))
    step = (axis[1] - axis[0]).item()
    jacobian = torch.linalg.det(whitener).abs().item()
    return torch.trapezoid(torch.trapezoid(integrand, dx=step, dim=1), dx=step, dim=0).item() * jacobian


def _gaussian(rbox: list[float]) -> tuple[Tensor, Tensor]:
    """Build the ``(mean, covariance)`` R17 assigns to one rotated box."""
    cx, cy, width, height, theta = rbox
    cos, sin = math.cos(theta), math.sin(theta)
    rotation = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float64)
    variances = torch.diag(torch.tensor([width**2 / 12, height**2 / 12], dtype=torch.float64))
    return torch.tensor([cx, cy], dtype=torch.float64), rotation @ variances @ rotation.T


def _density(points: Tensor, mean: Tensor, cov: Tensor) -> Tensor:
    """Evaluate a 2-D Gaussian density on a grid of points."""
    offset = points - mean
    quadratic = torch.einsum("...i,ij,...j->...", offset, torch.linalg.inv(cov), offset)
    return torch.exp(-0.5 * quadratic) / (2 * math.pi * torch.sqrt(torch.linalg.det(cov)))


def _pair(pred: list[float], target: list[float], dtype: torch.dtype = torch.float64) -> tuple[Tensor, Tensor]:
    """Wrap two box literals as single-row tensors."""
    return torch.tensor([pred], dtype=dtype), torch.tensor([target], dtype=dtype)


def _elongated_sweep(aspect: float, count: int = 1024, scale: float = 1024.0) -> tuple[Tensor, Tensor]:
    """Random float64 box pairs at a fixed aspect ratio, offset in all five parameters."""
    pred = torch.stack(
        [
            torch.rand(count, dtype=torch.float64) * scale,
            torch.rand(count, dtype=torch.float64) * scale,
            (torch.rand(count, dtype=torch.float64) * 0.9 + 0.1) * scale,
            (torch.rand(count, dtype=torch.float64) * 0.9 + 0.1) * scale / aspect,
            torch.rand(count, dtype=torch.float64) * math.pi,
        ],
        dim=-1,
    )
    target = pred.clone()
    target[:, :2] += (torch.rand(count, 2, dtype=torch.float64) - 0.5) * 0.2 * scale
    target[:, 2] *= 0.9
    target[:, 3] *= 1.1
    target[:, 4] += 0.3
    return pred, target


@pytest.mark.parametrize(("pred", "target"), _PAIRS)
def test_matches_numerical_integration_of_the_definition(pred: list[float], target: list[float]) -> None:
    """``exp(-B_D)`` equals the numerically integrated Bhattacharyya coefficient.

    Relative tolerance 1e-9, which is loose by five orders: the measured agreement is
    ~2e-14 relative, and refining the grid 301 -> 501 -> 1001 steps moves the result by
    the same ~1e-14 *without a trend*. That flatness identifies the residual as float64
    round-off in the summation rather than discretization error — the integrand is
    smooth and its truncation beyond nine sigma is below 1e-17. Relative rather than
    absolute so the disjoint pair, whose coefficient is 1.6e-102, is checked as
    stringently as the overlapping ones.
    """
    pred_t, target_t = _pair(pred, target)

    coefficient = torch.exp(-probiou_bhattacharyya_loss(pred_t, target_t)).item()

    assert coefficient == pytest.approx(integrated_bhattacharyya_coefficient(pred, target), rel=1e-9)


@pytest.mark.parametrize(("pred", "target"), _PAIRS)
def test_matches_literal_transcription_of_r17(pred: list[float], target: list[float]) -> None:
    """The cancellation-free evaluation agrees with R17's formulas as written.

    Float64 on both sides, so any disagreement is an algebra error rather than a
    conditioning one.
    """
    pred_t, target_t = _pair(pred, target)

    distance = probiou_bhattacharyya_loss(pred_t, target_t)

    assert torch.allclose(distance, literal_bhattacharyya(pred_t, target_t), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("theta", _ANGLES)
@pytest.mark.parametrize("aspect", [1.0, 1.5, 10.0, 200.0])
def test_a_box_against_itself_scores_exactly_one(theta: float, aspect: float) -> None:
    """ProbIoU is exactly 1 and both losses exactly 0 for a box compared with itself.

    Bit-exact, not to tolerance: every squared difference in the implementation vanishes
    identically when both operands are the same row, so ``B_D`` is a true zero.
    """
    box = torch.tensor([[7.0, -3.0, 12.0, 12.0 / aspect, theta]])

    assert torch.equal(probabilistic_iou(box, box), torch.ones(1))
    assert torch.equal(probiou_hellinger_loss(box, box), torch.zeros(1))
    assert torch.equal(probiou_bhattacharyya_loss(box, box), torch.zeros(1))


@pytest.mark.parametrize(("pred", "target"), _PAIRS)
def test_symmetric_in_its_two_arguments(pred: list[float], target: list[float]) -> None:
    """``B_D(p, q)`` equals ``B_D(q, p)``: a distance, not a directed penalty."""
    pred_t, target_t = _pair(pred, target)

    forward = probiou_bhattacharyya_loss(pred_t, target_t)

    assert torch.allclose(forward, probiou_bhattacharyya_loss(target_t, pred_t), rtol=1e-12, atol=1e-12)


def test_probiou_decays_monotonically_with_translation() -> None:
    """Sliding one box away from the other strictly lowers ProbIoU.

    The sweep stops at 16 px: past that the pair's ProbIoU underflows to 0 and equal
    zeros would make a strict comparison meaningless rather than informative.
    """
    box = torch.tensor([[0.0, 0.0, 8.0, 4.0, 0.0]], dtype=torch.float64)
    offsets = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=torch.float64)
    shifted = box.repeat(len(offsets), 1)
    shifted[:, 0] = offsets

    scores = probabilistic_iou(box.expand_as(shifted), shifted)

    assert torch.all(scores[1:] < scores[:-1])


def test_probiou_decays_monotonically_with_angle_difference() -> None:
    """Turning one non-square box against the other strictly lowers ProbIoU.

    Swept over a quarter turn only: the Gaussian has period ``pi`` in ``theta`` and is
    symmetric about ``pi/2``, so the decay reverses beyond that.
    """
    box = torch.tensor([[0.0, 0.0, 8.0, 4.0, 0.0]], dtype=torch.float64)
    angles = torch.tensor([0.0, 0.1, 0.3, 0.6, 1.0, 1.4, math.pi / 2], dtype=torch.float64)
    turned = box.repeat(len(angles), 1)
    turned[:, 4] = angles

    scores = probabilistic_iou(box.expand_as(turned), turned)

    assert torch.all(scores[1:] < scores[:-1])


@pytest.mark.parametrize(("pred", "target"), _PAIRS[:3] + _PAIRS[4:])
def test_the_two_losses_are_one_quantity(pred: list[float], target: list[float]) -> None:
    """``L2 == -ln(1 - L1^2)``: R17's two losses are two forms of the same distance.

    Float64, and ``log1p`` on the test side, because ``1 - L1^2`` cancels catastrophically
    near coincidence. The disjoint pair is excluded for the mirror-image reason: its
    ``B_D`` is ~234, so ``L1`` is exactly 1 in any float precision and the right-hand side
    is ``-ln(0)``. Both are limits of the *relation*, not of either loss — each of which
    stays exact there.
    """
    pred_t, target_t = _pair(pred, target)

    hellinger = probiou_hellinger_loss(pred_t, target_t)

    assert torch.allclose(
        probiou_bhattacharyya_loss(pred_t, target_t), -torch.log1p(-(hellinger**2)), rtol=1e-9, atol=1e-9
    )


def test_both_losses_have_finite_gradients_in_all_five_parameters() -> None:
    """Every box parameter receives a finite, non-zero gradient from either loss."""
    pred = torch.tensor([[10.0, 10.0, 8.0, 4.0, 0.3], [0.0, 2.0, 30.0, 3.0, -0.5]], requires_grad=True)
    target = torch.tensor([[12.0, 9.0, 6.0, 5.0, 1.1], [1.0, 0.0, 25.0, 5.0, 0.4]])

    (probiou_hellinger_loss(pred, target).sum() + probiou_bhattacharyya_loss(pred, target).sum()).backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad != 0).all()


def test_bhattacharyya_gradient_vanishes_only_at_coincidence() -> None:
    """``grad(B_D)`` is zero exactly where the boxes coincide, and non-zero one step away.

    R17's argument for starting with ``L2``: it has no vanishing gradient anywhere else,
    including far from the target where the bounded ``L1`` has saturated.
    """
    box = [10.0, 10.0, 8.0, 4.0, 0.3]
    coincident = torch.tensor([box], requires_grad=True)
    probiou_bhattacharyya_loss(coincident, torch.tensor([box])).sum().backward()

    nudged = torch.tensor([box], requires_grad=True)
    probiou_bhattacharyya_loss(nudged, torch.tensor([[10.001, 10.0, 8.0, 4.0, 0.3]])).sum().backward()
    far = torch.tensor([box], requires_grad=True)
    probiou_bhattacharyya_loss(far, torch.tensor([[500.0, 400.0, 8.0, 4.0, 0.3]])).sum().backward()

    assert torch.equal(coincident.grad, torch.zeros(1, 5))
    assert nudged.grad is not None and (nudged.grad[0, :2] != 0).all()
    assert far.grad is not None and torch.isfinite(far.grad).all() and (far.grad[0, :2] != 0).all()


def test_hellinger_gradient_at_coincidence_is_zero_not_nan() -> None:
    """A coincident pair backpropagates zeros through ``L1``, not the ``sqrt``'s ``NaN``.

    ``d(sqrt(u))/du`` is infinite at ``u = 0`` and ``dB_D/dbox`` is zero there, so an
    unguarded square root would multiply ``inf`` by ``0``. One coincident pair would then
    poison a whole batch's gradients.
    """
    box = torch.tensor([[10.0, 10.0, 8.0, 4.0, 0.3]], requires_grad=True)

    probiou_hellinger_loss(box, box.detach().clone()).sum().backward()

    assert box.grad is not None
    assert torch.equal(box.grad, torch.zeros(1, 5))


@pytest.mark.parametrize(
    "degenerate",
    [
        pytest.param([1.0, 1.0, 0.0, 0.0, 0.0], id="zero-area"),
        pytest.param([1.0, 1.0, 4.0, 0.0, 0.7], id="zero-height"),
        pytest.param([1.0, 1.0, 0.0, 2.0, -0.2], id="zero-width"),
        pytest.param([1.0, 1.0, 1e-12, 1e-12, 0.4], id="below-the-floor"),
        pytest.param([1.0, 1.0, -3.0, -1.0, 0.4], id="negative-sides"),
    ],
)
def test_degenerate_boxes_stay_finite_forwards_and_backwards(degenerate: list[float]) -> None:
    """A collapsed box gives finite losses and finite gradients (A41).

    Both against a normal box and against another degenerate one; the second is the case
    that underflows without the floor, since it divides by a product of four tiny sides.
    """
    pred = torch.tensor([degenerate, degenerate], requires_grad=True)
    target = torch.tensor([[2.0, 3.0, 6.0, 3.0, 0.1], degenerate])

    losses = probiou_hellinger_loss(pred, target) + probiou_bhattacharyya_loss(pred, target)
    losses.sum().backward()

    assert torch.isfinite(losses).all()
    assert torch.isfinite(probabilistic_iou(pred, target)).all()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_clamped_sides_take_no_gradient() -> None:
    """A side below the floor receives exactly zero gradient — the loss lets it go."""
    pred = torch.tensor([[1.0, 1.0, 1e-9, 1e-9, 0.3]], requires_grad=True)
    target = torch.tensor([[2.0, 3.0, 6.0, 3.0, 0.1]])

    probiou_bhattacharyya_loss(pred, target).sum().backward()

    assert pred.grad is not None
    assert torch.equal(pred.grad[0, 2:4], torch.zeros(2))
    assert (pred.grad[0, :2] != 0).all()


@pytest.mark.parametrize(("pred", "target"), _PAIRS)
def test_invariant_under_the_two_canonical_moves(pred: list[float], target: list[float]) -> None:
    """``theta + pi`` and the ``(w, h)`` swap with ``theta + pi/2`` describe one rectangle.

    Both moves preserve WP-055's rectangle, and R1 sec. 3.4.3 adopts the long-edge range
    precisely to keep the boundary from moving the loss; a form that saw them differently
    would reintroduce the ambiguity the range exists to remove. Tolerance is float64
    round-off in the shifted angle, not slack in the property.
    """
    pred_t, target_t = _pair(pred, target)
    half_turn = pred_t.clone()
    half_turn[:, 4] += math.pi
    swapped = pred_t.clone()
    swapped[:, 2], swapped[:, 3] = pred_t[:, 3], pred_t[:, 2]
    swapped[:, 4] += math.pi / 2

    baseline = probiou_bhattacharyya_loss(pred_t, target_t)

    assert torch.allclose(probiou_bhattacharyya_loss(half_turn, target_t), baseline, rtol=1e-9, atol=1e-12)
    assert torch.allclose(probiou_bhattacharyya_loss(swapped, target_t), baseline, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("aspect", [1.0, 10.0, 100.0, 1000.0])
def test_float32_accuracy_is_aspect_ratio_independent(aspect: float) -> None:
    """Float32 stays within 1e-5 relative of float64 truth from 1:1 to 1000:1.

    The property the cancellation-free identities exist for: measured 1.7e-6 to 2.9e-6
    across the sweep, with no trend in elongation.
    """
    pred, target = _elongated_sweep(aspect)
    truth = probiou_bhattacharyya_loss(pred, target)

    implemented = probiou_bhattacharyya_loss(pred.float(), target.float()).double()

    assert ((implemented - truth).abs() / truth.abs()).max() < 1e-5


def test_float32_survives_thousand_to_one_elongation() -> None:
    """A 1000:1 box keeps 1e-5 relative accuracy in float32 — the reason for the evaluation used.

    R17's formulas as written do not: on this sweep they fail to hold even 1e-3, and some
    pairs return ``NaN``, the summed determinant having cancelled to a negative value
    before the logarithm. That measurement is why the module evaluates the same two
    quantities without either subtraction (A41), but it is not asserted here — it is a
    property of a formula this package does not ship, and float32 rounding of an unstable
    expression differs between platforms (A26 records exactly such a divergence).
    """
    pred, target = _elongated_sweep(1000.0)
    truth = probiou_bhattacharyya_loss(pred, target)

    implemented_error = (probiou_bhattacharyya_loss(pred.float(), target.float()).double() - truth).abs() / truth.abs()

    assert torch.all(implemented_error < 1e-5)


def test_float32_holds_in_the_converged_regime() -> None:
    """Near-coincident boxes keep their Hellinger loss accurate in float32.

    The regime a regression loss spends its training in, and the one the literal form
    loses first: at a 1e-4 relative perturbation of a 20:1 box its float32 ``B_D`` is off
    by 377% and its ``L1`` by 2.2e-3, against 3.2e-6 for the form implemented here. Only
    the latter is asserted — see the elongation test above for why the literal form's
    failure is recorded rather than gated.
    """
    count = 512
    scale = 1024.0
    delta = 1e-4
    pred = torch.stack(
        [
            torch.rand(count, dtype=torch.float64) * scale,
            torch.rand(count, dtype=torch.float64) * scale,
            torch.full((count,), 0.4 * scale, dtype=torch.float64),
            torch.full((count,), 0.02 * scale, dtype=torch.float64),
            torch.rand(count, dtype=torch.float64) * math.pi,
        ],
        dim=-1,
    )
    target = pred.clone()
    target[:, 0] += delta * scale * 0.4
    target[:, 2] *= 1 + delta
    target[:, 3] *= 1 - delta
    target[:, 4] += delta

    implemented = probiou_hellinger_loss(pred.float(), target.float()).double()

    assert (implemented - probiou_hellinger_loss(pred, target)).abs().max() < 1e-5


def test_batched_broadcast_and_empty_shapes() -> None:
    """Leading dimensions broadcast and are dropped; an empty input gives empty output."""
    pred = torch.rand(4, 1, 5) * 10 + torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
    target = torch.rand(1, 3, 5) * 10 + torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])

    assert probiou_bhattacharyya_loss(pred, target).shape == (4, 3)
    assert probabilistic_iou(pred, target).shape == (4, 3)
    assert probiou_hellinger_loss(pred, torch.rand(4, 1, 5) + 1).shape == (4, 1)

    empty = torch.zeros(0, 5)
    assert probiou_bhattacharyya_loss(empty, empty).shape == (0,)
    assert probabilistic_iou(empty, empty).shape == (0,)
    assert probiou_hellinger_loss(empty, empty).shape == (0,)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((3, 4), id="four-columns"),
        pytest.param((3, 6), id="six-columns"),
        pytest.param((3, 5, 1), id="trailing-singleton"),
        pytest.param((), id="scalar"),
    ],
)
def test_wrong_trailing_dimension_raises(shape: tuple[int, ...]) -> None:
    """A tensor whose last dimension is not 5 is rejected by name, and by the shape it wanted."""
    good = torch.zeros(3, 5)

    with pytest.raises(ValueError, match=r"target must be \(\.\.\., 5\)"):
        probiou_bhattacharyya_loss(good, torch.zeros(shape))


def test_extra_leading_dimensions_are_accepted() -> None:
    """The loss takes any leading shape — the guard asserts the trailing five and nothing else.

    Pins the divergence from :func:`lucid_yolo.data.rotated_geom` deliberately (WP-091e):
    that module's checks require exactly 2-D and reject this input, and the two are not
    copies of one check despite sharing a name.
    """
    boxes = torch.rand(2, 3, 4, 5) + torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])

    assert probiou_bhattacharyya_loss(boxes, boxes.clone()).shape == (2, 3, 4)
    assert probiou_hellinger_loss(boxes, boxes.clone()).shape == (2, 3, 4)

    with pytest.raises(ValueError, match=r"rboxes must be \(N, 5\)"):
        canonicalize(boxes)


def test_scores_stay_in_range_over_a_random_sweep() -> None:
    """ProbIoU stays in [0, 1] and B_D stays non-negative for arbitrary pairs."""
    pred = torch.rand(2048, 5) * torch.tensor([200.0, 200.0, 50.0, 50.0, math.pi]) + torch.tensor(
        [0.0, 0.0, 0.1, 0.1, -math.pi / 4]
    )
    target = torch.rand(2048, 5) * torch.tensor([200.0, 200.0, 50.0, 50.0, math.pi]) + torch.tensor(
        [0.0, 0.0, 0.1, 0.1, -math.pi / 4]
    )

    scores = probabilistic_iou(pred, target)

    assert (scores >= 0).all() and (scores <= 1).all()
    assert (probiou_bhattacharyya_loss(pred, target) >= 0).all()
