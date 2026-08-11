# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the square-object angle loss (WP-060, A22, A42).

R1 Eq. 14-15 is short enough that a transcription oracle would only restate the
implementation, so this file gates the three things the equations leave to whoever writes
them down.

The **wrap range** is the first. Eq. 14 says ``round`` and stops; :func:`torch.round`
breaks halves to even, which would make the residual of ``pi/2`` and of ``3*pi/2`` — the
same rotation — land on opposite ends of a *closed* interval. The tests here pin the
half-open ``[-pi/2, pi/2)`` the module settles on, at the boundary and at both ulps
around it, the way ``tests/data/test_rotated_geom.py`` pins the long-edge range.

The **square tie-break** is the second, and the reason the module folds by a quarter turn
before taking the sine. :func:`~lucid_yolo.data.rotated_geom.canonicalize` folds an exact
square towards zero, so a square target reaches the loss as one of two representatives
``pi/2`` apart. ``sin^2(2x)`` cannot tell them apart mathematically, but two *floats*
``pi/2`` apart do not give bit-equal sines — so the invariance is asserted with
:func:`torch.equal` on exactly representable inputs rather than assumed.

The **profile of omega** is the third: it is the only place a hyper-parameter enters, and
R1 Table 11 shows ``lambda`` is not a free knob (50.2 mAP at 3, 47.1 at 5, against 49.0
with no angle loss at all). The measured values are pinned against the paper's quoted
table so a change to the formula, the log base or the default cannot pass silently.
"""

import math

import pytest
import torch
from torch import Tensor

from lucid_yolo.data.rotated_geom import canonicalize
from lucid_yolo.losses import aspect_ratio_weight, square_angle_loss, wrap_angle_delta

_PI = math.pi
_HALF_PI = math.pi / 2
_QUARTER_PI = math.pi / 4

#: ``omega`` at ``lambda = 3``, natural log — the table R1's aspect-ratio factor implies.
_OMEGA_TABLE = [
    pytest.param(1.0, 1.0000, id="square"),
    pytest.param(2.0, 0.9480, id="2-to-1"),
    pytest.param(5.0, 0.7499, id="5-to-1"),
    pytest.param(10.0, 0.5548, id="10-to-1"),
    pytest.param(20.0, 0.3689, id="20-to-1"),
]

#: ``sin^2(2 d)`` is zero at a quarter turn and maximal halfway between: the double-angle
#: penalty ignores the 90deg ambiguity of a square and punishes the diagonal error.
_EXTREMA = [
    pytest.param(0.0, 0.0, id="exact"),
    pytest.param(_HALF_PI, 0.0, id="quarter-turn"),
    pytest.param(-_HALF_PI, 0.0, id="quarter-turn-back"),
    pytest.param(_PI, 0.0, id="half-turn"),
    pytest.param(_QUARTER_PI, 1.0, id="diagonal"),
    pytest.param(-_QUARTER_PI, 1.0, id="diagonal-back"),
    pytest.param(3 * _QUARTER_PI, 1.0, id="diagonal-next-quadrant"),
]


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed every RNG source before each test for deterministic tensors."""
    torch.manual_seed(0)


def _step_ulps(value: float, steps: int) -> float:
    """Return ``value`` moved ``steps`` float32 ulps, negative steps moving downwards."""
    current = torch.tensor(value, dtype=torch.float32)
    towards = torch.tensor(-torch.inf if steps < 0 else torch.inf, dtype=torch.float32)
    for _ in range(abs(steps)):
        current = torch.nextafter(current, towards)
    return float(current)


def _wrap_inputs() -> list[float]:
    """Residuals spanning several turns, plus both ulps around every wrap boundary."""
    plain = [0.0, 0.3, -0.3, 1.2, -1.2, 7.5, -7.5, 100.0, -100.0]
    boundaries = [_HALF_PI, -_HALF_PI, 3 * _HALF_PI, -3 * _HALF_PI, 5 * _HALF_PI]
    return plain + [_step_ulps(bound, step) for bound in boundaries for step in (-1, 0, 1)]


def _loss_curve(pred: Tensor, target: float = 0.0, side: float = 8.0) -> Tensor:
    """Loss at each swept prediction, every point evaluated as its own single-anchor batch.

    A single anchor of weight one normalizes by ``S = 1`` and, for the square target used
    here, carries ``omega = 1`` — so each entry is the bare penalty the loss puts on that
    angular error, read through the public function rather than reimplemented.
    """
    one, sides = torch.ones(1), torch.full((1,), side)
    return torch.stack(
        [square_angle_loss(point.reshape(1), torch.tensor([target]), sides, sides, one) for point in pred]
    )


def _single(pred: float, target: float, width: float = 8.0, height: float = 8.0, weight: float = 1.0) -> Tensor:
    """Loss for one anchor, given as plain floats."""
    return square_angle_loss(
        torch.tensor([pred]),
        torch.tensor([target]),
        torch.tensor([width]),
        torch.tensor([height]),
        torch.tensor([weight]),
    )


@pytest.mark.parametrize("delta", _wrap_inputs())
def test_wrap_range(delta: float) -> None:
    """Every residual lands in the half-open ``[-pi/2, pi/2)``, boundary ulps included.

    Half-open with no gap and no overlap is the property Eq. 14's unstated ``round`` tie
    rule decides: an interval closed at both ends would admit ``+pi/2`` and ``-pi/2`` as
    two answers to the same rotation. Idempotence is asserted alongside because the
    remainder path is not exact near its own boundary — an already-wrapped value must
    short-circuit rather than be recomputed.
    """
    wrapped = wrap_angle_delta(torch.tensor([delta], dtype=torch.float32))

    assert bool(wrapped.ge(-_HALF_PI).all())
    assert bool(wrapped.lt(_HALF_PI).all())
    assert torch.equal(wrap_angle_delta(wrapped), wrapped)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_wrap_ties_take_the_low_end(dtype: torch.dtype) -> None:
    """Both ``+-pi/2`` ties wrap to ``-pi/2``, in either working precision.

    This is the "no overlap" half of the range guarantee, and where the module parts
    company with :func:`torch.round`: half-to-even would send ``+pi/2`` to ``+pi/2`` and
    the equivalent ``+3*pi/2`` to ``-pi/2``, two representatives of one rotation.

    Bit-exact only at ``+-pi/2``, and to rounding three half-turns out: ``-3*pi/2`` is not
    representable in float32, and the stored value sits 1.2e-8 off the true tie, which the
    remainder magnifies to 2 ulps (measured ``-1.5707961`` against ``-1.5707964``). The
    tie rule is a statement about the tie, not about floats that are merely near one.
    """
    ties = torch.tensor([_HALF_PI, -_HALF_PI], dtype=dtype)

    wrapped = wrap_angle_delta(ties)

    assert torch.equal(wrapped, torch.full((2,), -_HALF_PI, dtype=dtype))
    assert torch.allclose(
        wrap_angle_delta(torch.tensor([3 * _HALF_PI, -3 * _HALF_PI], dtype=dtype)),
        torch.full((2,), -_HALF_PI, dtype=dtype),
        atol=1e-6,
    )


def test_wrap_is_invariant_to_whole_half_turns() -> None:
    """Adding ``pi`` to a residual does not change where it wraps to (Eq. 14's premise)."""
    base = torch.tensor([0.0, 0.3, -0.3, 1.4, -1.4, 1.5])

    wrapped = wrap_angle_delta(base)

    assert torch.allclose(wrap_angle_delta(base + _PI), wrapped, atol=1e-6)
    assert torch.allclose(wrap_angle_delta(base - _PI), wrapped, atol=1e-6)


@pytest.mark.parametrize(("ratio", "expected"), _OMEGA_TABLE)
def test_omega_profile(ratio: float, expected: float) -> None:
    """``omega`` follows R1's log-Gaussian in the aspect ratio at ``lambda = 3``.

    The paper quotes this table to three significant figures (0.750 at 5:1, 0.369 at
    20:1); the four-decimal values asserted here are the measured ones. The tolerance is
    tight enough that a switch to base-10 logs (0.9472 at 5:1) or to ``lambda = 5``
    (0.9016 there) fails it.
    """
    omega = aspect_ratio_weight(torch.tensor([3.0 * ratio]), torch.tensor([3.0]))

    assert omega.item() == pytest.approx(expected, abs=5e-5)


def test_omega_is_symmetric_under_swapping_the_sides() -> None:
    """A box and its transpose share one ``omega``: only the log-ratio's sign changes.

    Load-bearing because :func:`~lucid_yolo.data.rotated_geom.canonicalize` may have
    swapped ``w`` and ``h`` (with ``theta += pi/2``) before the target reached the loss.
    """
    width, height = torch.tensor([9.0, 2.0, 40.0]), torch.tensor([3.0, 7.0, 1.0])

    omega = aspect_ratio_weight(width, height)

    assert torch.allclose(aspect_ratio_weight(height, width), omega, rtol=1e-6)


def test_omega_decays_strictly_as_the_target_elongates() -> None:
    """Elongated targets are down-weighted, squares are not — R1's stated intent.

    "Elongated boxes receive smaller ``omega_i`` and remain primarily constrained by the
    rotated IoU loss": the square is the unique maximum and the decay is monotone.
    """
    ratios = torch.tensor([1.0, 1.2, 2.0, 3.0, 5.0, 10.0, 50.0, 500.0])

    omega = aspect_ratio_weight(ratios, torch.ones(8))

    assert omega[0] == 1.0
    assert torch.all(omega[1:] < omega[:-1])


def test_omega_widens_with_lambda() -> None:
    """A larger ``lambda`` spends the square-object penalty on more elongated boxes.

    The mechanism behind R1 Table 11's regression at ``lambda = 5`` (47.1 mAP, below the
    49.0 of no angle loss at all): a 10:1 box goes from 0.55 of the penalty to 0.81.
    """
    sides = torch.tensor([10.0])

    wide = aspect_ratio_weight(sides, torch.ones(1), lam=5.0)

    assert wide.item() == pytest.approx(0.8089, abs=5e-5)
    assert wide.item() > aspect_ratio_weight(sides, torch.ones(1)).item()


def test_omega_rejects_a_non_positive_lambda() -> None:
    """``lambda <= 0`` divides by zero and would return ``NaN`` for a square, not raise."""
    with pytest.raises(ValueError, match="lam must be"):
        aspect_ratio_weight(torch.ones(1), torch.ones(1), lam=0.0)


@pytest.mark.parametrize(("error", "expected"), _EXTREMA)
def test_sin2_extrema(error: float, expected: float) -> None:
    """The double-angle penalty is zero at every quarter turn and maximal between them.

    A single square anchor of weight one, so ``S`` and ``omega`` are both 1 and the loss
    *is* ``sin^2(2 d_theta_tilde)``. The zeros at ``+-pi/2`` are the point of the term:
    for a square those rotations are the same box, so a 90deg "error" is not one.
    """
    loss = _single(pred=error, target=0.0)

    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_boundary_continuity() -> None:
    """The loss crosses the wrap boundary without a step, over a sweep that spans ``[0, 1]``.

    A prediction swept from ``pi/4`` to ``3*pi/4`` against a target at zero takes the
    residual through the ``pi/2`` wrap boundary at the midpoint, and the penalty falls
    from 1 to 0 and back — so the sweep exercises the full range of the term rather than
    only its neighbourhood of zero. Continuity is asserted as a Lipschitz bound: the
    penalty's derivative ``2 sin(4x)`` is bounded by 2, so no adjacent pair may differ by
    more than twice the sweep step. A wrap that jumped would break that by ~1.0, not by a
    rounding margin.
    """
    swept = torch.linspace(_QUARTER_PI, 3 * _QUARTER_PI, 2001)
    step = float(swept[1] - swept[0])

    curve = _loss_curve(swept)

    assert float(curve.diff().abs().max()) < 2 * step + 1e-6
    assert float(curve.min()) < 1e-6
    assert float(curve.max()) > 0.999


def test_boundary_ulps_agree_with_the_boundary_itself() -> None:
    """One ulp either side of the wrap boundary scores what the boundary scores.

    The residual is exactly ``pi/2`` there, which wraps to ``-pi/2`` and folds to an exact
    zero — so the loss is bit-exactly zero at the boundary, and within an ulp's worth of
    it on both sides rather than merely finite.
    """
    boundary = _single(pred=_HALF_PI, target=0.0)

    assert torch.equal(boundary, torch.zeros(()))
    assert _single(pred=_step_ulps(_HALF_PI, -1), target=0.0).item() < 1e-12
    assert _single(pred=_step_ulps(_HALF_PI, 1), target=0.0).item() < 1e-12


def test_both_representatives_of_a_square_target_score_identically() -> None:
    """A square's two canonical representatives give a bit-identical loss.

    :func:`~lucid_yolo.data.rotated_geom.canonicalize` folds an exact square towards zero,
    so which of the two angles ``pi/2`` apart the loss receives is an accident of the
    target's provenance. The inputs are chosen dyadic so ``target + pi/2`` is exact in
    float32 and the comparison tests the loss rather than the test's own arithmetic; the
    tolerance-free version of this invariance is what the module's quarter-turn fold
    exists for.
    """
    low = torch.tensor([0.25])
    high = low + _HALF_PI
    square = torch.full((1,), 8.0)
    assert torch.equal(canonicalize(torch.tensor([[0.0, 0.0, 8.0, 8.0, float(high)]]))[:, 4], low)

    loss = square_angle_loss(torch.tensor([0.5]), low, square, square, torch.ones(1))

    assert torch.equal(square_angle_loss(torch.tensor([0.5]), high, square, square, torch.ones(1)), loss)


def test_square_representative_invariance_holds_across_a_random_sweep() -> None:
    """The same invariance at arbitrary angles, where the shift by ``pi/2`` itself rounds.

    Bit-equality is a property of exactly representable inputs; away from those, the two
    representatives differ by a rounded ``pi/2`` before the loss sees them, so the
    agreement is to float32 round-off. Asserted separately so a regression in the fold
    cannot hide behind a loose tolerance in the exact case.
    """
    pred = torch.rand(256) * _PI - _HALF_PI
    target = torch.rand(256) * _PI - _HALF_PI
    square, weights = torch.full((256,), 5.0), torch.rand(256)

    loss = square_angle_loss(pred, target, square, square, weights)

    assert square_angle_loss(pred, target + _HALF_PI, square, square, weights).item() == pytest.approx(
        loss.item(), rel=1e-6
    )


def test_loss_is_invariant_to_half_turns_of_either_angle() -> None:
    """``theta`` and ``theta + pi`` are one rectangle, so they are one loss (Eq. 14).

    The premise the whole term is built on, checked on the loss rather than on the wrap:
    an implementation that measured the residual on the real line would fail here.
    """
    pred, target = torch.rand(64) * 4 - 2, torch.rand(64) * 4 - 2
    width, height, weights = torch.rand(64) * 20 + 1, torch.rand(64) * 5 + 1, torch.rand(64)

    loss = square_angle_loss(pred, target, width, height, weights)

    assert square_angle_loss(pred + _PI, target, width, height, weights).item() == pytest.approx(loss.item(), rel=1e-5)
    assert square_angle_loss(pred, target - _PI, width, height, weights).item() == pytest.approx(loss.item(), rel=1e-5)


def test_normalizer_is_the_weight_sum_floored_at_one() -> None:
    """``S = max(sum q_i, 1)``: the floor binds below one, the sum divides above it."""
    quarter = _single(pred=_QUARTER_PI, target=0.0, weight=0.25)
    doubled = square_angle_loss(
        torch.full((2,), _QUARTER_PI), torch.zeros(2), torch.full((2,), 8.0), torch.full((2,), 8.0), torch.ones(2)
    )

    assert quarter.item() == pytest.approx(0.25, abs=1e-6)
    assert doubled.item() == pytest.approx(1.0, abs=1e-6)


def test_background_anchors_of_zero_weight_change_nothing() -> None:
    """A whole branch may be passed: ``q_i = 0`` rows leave both the sum and ``S`` alone.

    The arrangement :class:`~lucid_yolo.losses.detection_loss.DetectionBranchLoss` already
    relies on, so the oriented head (WP-088) need not mask before calling.
    """
    pred, target = torch.tensor([0.4, 1.9]), torch.tensor([0.1, 0.2])
    sides, weights = torch.tensor([6.0, 6.0]), torch.tensor([0.7, 0.3])
    padded = (torch.cat([pred, torch.tensor([2.5])]), torch.cat([target, torch.tensor([-1.0])]))
    padded_sides = torch.cat([sides, torch.tensor([100.0])])

    loss = square_angle_loss(pred, target, sides, sides, weights)

    assert torch.allclose(
        square_angle_loss(*padded, padded_sides, padded_sides, torch.cat([weights, torch.zeros(1)])), loss, rtol=1e-7
    )


def test_no_foreground_gives_an_exact_zero_that_still_carries_gradient() -> None:
    """An empty assignment is a zero loss with a zero gradient, not a special case."""
    pred = torch.zeros(0, requires_grad=True)
    empty = torch.zeros(0)

    loss = square_angle_loss(pred, empty, empty, empty, empty)
    loss.backward()

    assert torch.equal(loss, torch.zeros(()))
    assert pred.grad is not None and pred.grad.shape == (0,)


def test_gradient_reaches_the_prediction_and_stays_finite() -> None:
    """Every predicted angle receives a finite, non-zero gradient; the targets take none."""
    pred = torch.tensor([0.3, -0.9, 2.2], requires_grad=True)
    target, width, height = (
        torch.tensor([0.1, 0.4, -0.5]),
        torch.tensor([8.0, 3.0, 20.0]),
        torch.tensor([8.0, 3.0, 4.0]),
    )

    square_angle_loss(pred, target, width, height, torch.tensor([0.5, 0.8, 0.2])).backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad != 0).all()


@pytest.mark.parametrize(
    ("width", "height"),
    [
        pytest.param(0.0, 0.0, id="zero-area"),
        pytest.param(4.0, 0.0, id="zero-height"),
        pytest.param(0.0, 2.0, id="zero-width"),
        pytest.param(1e-12, 1e-12, id="below-the-floor"),
        pytest.param(-3.0, -1.0, id="negative-sides"),
    ],
)
def test_degenerate_targets_stay_finite_forwards_and_backwards(width: float, height: float) -> None:
    """A collapsed target gives a finite loss and a finite gradient (A42).

    Without the side floor ``ln(w/h)`` reaches ``+-inf`` and its square poisons the whole
    batch's gradient through the sum.
    """
    pred = torch.tensor([0.7, 0.2], requires_grad=True)
    sides = (torch.tensor([width, 5.0]), torch.tensor([height, 2.0]))

    loss = square_angle_loss(pred, torch.tensor([0.1, 0.1]), *sides, torch.tensor([0.5, 0.5]))
    loss.backward()

    assert torch.isfinite(loss).all()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_loss_is_bounded_by_one_over_a_random_sweep() -> None:
    """The term stays in ``[0, 1]``: ``omega <= 1``, ``sin^2 <= 1``, and ``S`` is the weight sum."""
    pred, target = torch.rand(4096) * 20 - 10, torch.rand(4096) * 20 - 10
    width, height = torch.rand(4096) * 100 + 0.5, torch.rand(4096) * 100 + 0.5

    loss = square_angle_loss(pred, target, width, height, torch.rand(4096) * 2)

    assert 0.0 <= loss.item() <= 1.0
