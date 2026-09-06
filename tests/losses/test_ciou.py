# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Complete-IoU loss (WP-024).

Covers a hand-derived closed-form value, the identity and far-apart boundary
cases, the aspect-ratio ordering property, gradient finiteness for normal and
degenerate boxes, and batched/empty shape handling. The math is transcribed
from R10 (arXiv:1911.08287); expected values are derived by hand in the test
comments rather than lifted from any reference implementation.

Three later groups pin what the values alone cannot: the ``atan2`` origin guard
(A10), which is version-dependent behaviour no gradient assertion can observe on
one installed Torch; the safeguard-parameter validation (M-23); and the stated
float32 coordinate range (L-12), which is documented rather than enforced.
"""

import math

import pytest
import torch

from lucid_yolo.losses import box_iou_aligned, ciou_loss, complete_iou
from lucid_yolo.losses.ciou import _aspect_angle


def test_against_closed_form() -> None:
    """CIoU matches a fully hand-derived value to 1e-6.

    Boxes (xyxy):
        pred   = [0, 0, 4, 2]  -> w=4, h=2, center (2, 1), area 8
        target = [1, 0, 3, 4]  -> w=2, h=4, center (2, 2), area 8

    Intersection: x in [1, 3] -> 2, y in [0, 2] -> 2, so inter = 4.
    Union = 8 + 8 - 4 = 12, so IoU = 4 / 12 = 1/3.
    Center distance^2 = (2-2)^2 + (1-2)^2 = 1.
    Enclosing box = [0, 0, 4, 4], diagonal^2 = 4^2 + 4^2 = 32.
    v = (4/pi^2) * (atan(2/4) - atan(4/2))^2.
    alpha = v / ((1 - IoU) + v).
    CIoU = IoU - 1/32 - alpha * v.
    """
    pred = torch.tensor([[0.0, 0.0, 4.0, 2.0]])
    target = torch.tensor([[1.0, 0.0, 3.0, 4.0]])

    iou = 1.0 / 3.0
    distance_penalty = 1.0 / 32.0
    v = (4.0 / math.pi**2) * (math.atan(2.0 / 4.0) - math.atan(4.0 / 2.0)) ** 2
    alpha = v / ((1.0 - iou) + v)
    expected = iou - distance_penalty - alpha * v

    got = complete_iou(pred, target)
    assert got.shape == (1,)
    assert math.isclose(got.item(), expected, abs_tol=1e-6)


def test_iou_aligned_matches_hand_value() -> None:
    """box_iou_aligned returns the paired IoU, not a cross matrix."""
    pred = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
    target = torch.tensor([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 2.0]])
    iou = box_iou_aligned(pred, target)
    assert iou.shape == (2,)
    # First pair: inter 1, union 7 -> 1/7. Second pair: identical -> 1.
    assert torch.allclose(iou, torch.tensor([1.0 / 7.0, 1.0]), atol=1e-6)


def test_identical_boxes_score_one_zero_loss() -> None:
    """Identical boxes give CIoU == 1 and loss == 0."""
    boxes = torch.tensor([[3.0, 5.0, 11.0, 9.0], [-2.0, -2.0, 4.0, 1.0]])
    assert torch.allclose(complete_iou(boxes, boxes), torch.ones(2), atol=1e-6)
    assert torch.allclose(ciou_loss(boxes, boxes), torch.zeros(2), atol=1e-6)


def test_disjoint_far_boxes_go_negative() -> None:
    """Far-apart disjoint boxes push CIoU below 0 and loss above 1."""
    pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    target = torch.tensor([[100.0, 100.0, 101.0, 101.0]])
    ciou = complete_iou(pred, target)
    assert ciou.item() < 0.0
    assert ciou_loss(pred, target).item() > 1.0


def test_matching_aspect_ratio_scores_higher() -> None:
    """Same IoU and centers: the matching-aspect pair scores strictly higher.

    Both candidate preds share the target's center and the same IoU with it
    (equal area, fully contained, symmetric about the center); only the aspect
    ratio differs, so CIoU must rank the aspect-matched candidate above the
    aspect-mismatched one via the ``alpha * v`` term.
    """
    target = torch.tensor([[-2.0, -1.0, 2.0, 1.0]])  # w=4, h=2, center (0, 0)
    matched = torch.tensor([[-1.0, -0.5, 1.0, 0.5]])  # w=2, h=1 -> same 2:1 ratio
    mismatched = torch.tensor([[-0.5, -1.0, 0.5, 1.0]])  # w=1, h=2 -> 1:2 ratio

    # Same area and same center => identical IoU for both candidates.
    assert math.isclose(
        box_iou_aligned(matched, target).item(),
        box_iou_aligned(mismatched, target).item(),
        abs_tol=1e-6,
    )
    assert complete_iou(matched, target).item() > complete_iou(mismatched, target).item()


def test_gradients_finite_normal_and_degenerate() -> None:
    """backward() yields finite grads for normal and zero-width pred boxes."""
    pred = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 1.0, 3.0]],  # second row: zero width
        requires_grad=True,
    )
    target = torch.tensor([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 4.0]])
    loss = ciou_loss(pred, target).sum()
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_batched_and_empty_shapes() -> None:
    """(N,4) inputs give (N,) outputs; N=0 works without crashing."""
    pred = torch.rand(5, 4)
    pred = torch.stack([pred[:, 0], pred[:, 1], pred[:, 0] + 1.0, pred[:, 1] + 1.0], dim=1)
    target = pred.clone() + 0.25
    assert complete_iou(pred, target).shape == (5,)
    assert ciou_loss(pred, target).shape == (5,)

    empty = torch.zeros(0, 4)
    assert complete_iou(empty, empty).shape == (0,)
    assert box_iou_aligned(empty, empty).shape == (0,)
    assert ciou_loss(empty, empty).shape == (0,)


class TestDegenerateGradientMagnitude:
    """Degenerate boxes take a *bounded* gradient, not merely a finite one (WP-170).

    ``test_gradients_finite_normal_and_degenerate`` above asserts only
    ``isfinite``, which the aspect term satisfied while backpropagating 3.0e5
    from a collapsed box: ``atan(w / (h + eps))`` is finite at ``h = 0`` but its
    derivative there is ``1 / eps``. Finiteness is therefore the wrong predicate
    for this input class, and these tests pin the magnitude instead.
    """

    #: Gradient magnitude a degenerate box must stay under. Two orders above the
    #: ~1e-1 a well-formed pair produces here and two below the 3.0e5 the guarded
    #: ratio produced, so it separates the two regimes without pinning an exact value.
    GRADIENT_BOUND = 1.0e3

    def test_collapsed_box_gradient_is_bounded(self) -> None:
        """A zero-area prediction backpropagates a small gradient, not a 1/eps spike.

        The measured regression: against this target, ``[5, 5, 5, 5]`` gave
        ``dL/dx1 = 301186.5`` through ``atan(w / (h + eps))``, and ``0.0``
        through ``atan2(w, h)``.
        """
        target = torch.tensor([[0.0, 0.0, 10.0, 20.0]])
        pred = torch.tensor([[5.0, 5.0, 5.0, 5.0]], requires_grad=True)

        ciou_loss(pred, target).sum().backward()

        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert float(pred.grad.abs().max()) < self.GRADIENT_BOUND

    def test_a_thin_box_does_not_spike_as_height_vanishes(self) -> None:
        """The bound holds as ``h`` is driven to zero, not only exactly at it.

        ``1 / (h + eps)`` grows without bound as ``h`` shrinks, so a test sitting
        only on ``h == 0`` could be satisfied by a special case there. Each row
        below has a real width and a height stepping down to zero.
        """
        heights = [1.0, 1e-2, 1e-4, 1e-6, 0.0]
        pred = torch.tensor([[0.0, 0.0, 4.0, h] for h in heights], requires_grad=True)
        target = torch.tensor([[0.0, 0.0, 10.0, 20.0]] * len(heights))

        ciou_loss(pred, target).sum().backward()

        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert float(pred.grad.abs().max()) < self.GRADIENT_BOUND

    def test_inverted_box_is_gradient_dead_by_design(self) -> None:
        """An inverted box takes exactly zero gradient on the inverted axis.

        Documented behaviour rather than an accident (``losses/ciou.py`` module
        docstring): the clamped extents make CIoU blind to a box that names no
        region, and un-inverting it is the decode boundary's job, not this loss's.
        Pinned so that a future change to the clamp cannot alter it silently.
        """
        target = torch.tensor([[0.0, 0.0, 10.0, 20.0]])
        pred = torch.tensor([[10.0, 10.0, 0.0, 0.0]], requires_grad=True)

        ciou_loss(pred, target).sum().backward()

        assert pred.grad is not None
        assert float(pred.grad[0, 0]) == 0.0  # dL/dx1
        assert float(pred.grad[0, 2]) == 0.0  # dL/dx2

    def test_a_well_formed_pair_keeps_an_informative_gradient(self) -> None:
        """The bound above is not satisfied by zeroing everything.

        Guards the obvious wrong fix -- clamping the aspect term away entirely --
        by asserting a normal overlapping pair still moves.
        """
        target = torch.tensor([[0.0, 0.0, 10.0, 20.0]])
        pred = torch.tensor([[1.0, 1.0, 9.0, 19.0]], requires_grad=True)

        ciou_loss(pred, target).sum().backward()

        assert pred.grad is not None
        assert float(pred.grad.abs().max()) > 0.0
        assert float(pred.grad.abs().max()) < self.GRADIENT_BOUND


class TestAtan2OriginGuard:
    """The zero pair never reaches ``atan2``, on any Torch in the declared range (A10).

    WP-170's finite-gradient guarantee was measured on Torch 2.13, whose ``atan2``
    backward special-cases the origin and returns zero partials. Torch 2.4 — the
    floor ``pyproject.toml`` declares — divides by ``w^2 + h^2`` with no such guard
    and returns ``NaN`` for the same pair. Because only one end of that range is
    ever installed, an assertion on the gradient cannot see the difference: these
    tests assert the *call* instead, which is version-independent, and pin the
    guard as a no-op everywhere else.
    """

    @staticmethod
    def _recording_atan2(seen: list[tuple[float, float]]):
        """Wrap ``torch.atan2``, recording every argument pair it is handed."""
        real = torch.atan2

        def spy(width: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
            broadcast = torch.broadcast_tensors(width.detach(), height.detach())
            pairs = torch.stack(broadcast, dim=-1).reshape(-1, 2)
            seen.extend((float(w), float(h)) for w, h in pairs)
            return real(width, height)

        return spy

    def test_a_collapsed_prediction_never_hands_atan2_the_origin(self, monkeypatch) -> None:
        """The pair the module's own doctest produces is substituted before the call."""
        seen: list[tuple[float, float]] = []
        monkeypatch.setattr(torch, "atan2", self._recording_atan2(seen))

        pred = torch.tensor([[5.0, 5.0, 5.0, 5.0]], requires_grad=True)
        ciou_loss(pred, torch.tensor([[0.0, 0.0, 10.0, 20.0]])).sum().backward()

        assert seen, "atan2 was never called; the test no longer exercises the aspect term"
        assert (0.0, 0.0) not in seen

    def test_a_zero_size_target_never_hands_atan2_the_origin(self, monkeypatch) -> None:
        """The target side is guarded too: a zero-area ground truth reaches the same term."""
        seen: list[tuple[float, float]] = []
        monkeypatch.setattr(torch, "atan2", self._recording_atan2(seen))

        pred = torch.tensor([[0.0, 0.0, 4.0, 2.0]], requires_grad=True)
        ciou_loss(pred, torch.tensor([[1.0, 1.0, 1.0, 1.0]])).sum().backward()

        assert seen
        assert (0.0, 0.0) not in seen

    def test_an_inverted_prediction_never_hands_atan2_the_origin(self, monkeypatch) -> None:
        """Inverted extents clamp to zero, so they arrive at the aspect term as the origin."""
        seen: list[tuple[float, float]] = []
        monkeypatch.setattr(torch, "atan2", self._recording_atan2(seen))

        pred = torch.tensor([[10.0, 10.0, 0.0, 0.0]], requires_grad=True)
        ciou_loss(pred, torch.tensor([[0.0, 0.0, 10.0, 20.0]])).sum().backward()

        assert seen
        assert (0.0, 0.0) not in seen

    def test_the_guard_is_bit_identical_to_a_bare_atan2_away_from_the_origin(self) -> None:
        """Value *and* gradient match the unguarded call exactly, not to a tolerance.

        This is what makes the guard safe to add to a landed loss: every pair that
        is not exactly ``(0, 0)`` takes the same arithmetic it always did.
        """
        torch.manual_seed(0)
        sides = torch.rand(256, 2) * 100.0 + 1e-3

        guarded_input = sides.clone().requires_grad_(True)
        guarded = _aspect_angle(guarded_input[:, 0], guarded_input[:, 1])
        guarded.sum().backward()

        bare_input = sides.clone().requires_grad_(True)
        bare = torch.atan2(bare_input[:, 0], bare_input[:, 1])
        bare.sum().backward()

        assert torch.equal(guarded, bare)
        assert guarded_input.grad is not None and bare_input.grad is not None
        assert torch.equal(guarded_input.grad, bare_input.grad)

    def test_one_zero_side_is_not_the_origin_and_stays_unguarded(self) -> None:
        """``atan2(w, 0) = pi/2`` is exact and keeps its own gradient; only the pair is special."""
        sides = torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True)

        angles = _aspect_angle(sides[:, 0], sides[:, 1])
        angles.sum().backward()

        assert torch.equal(angles, torch.atan2(sides[:, 0].detach(), sides[:, 1].detach()))
        assert sides.grad is not None
        assert float(sides.grad.abs().sum()) > 0.0

    def test_the_origin_itself_stays_finite_in_value_and_gradient(self) -> None:
        """The behaviour the guard preserves: zero contribution, zero gradient."""
        sides = torch.zeros(1, 2, requires_grad=True)

        angle = _aspect_angle(sides[:, 0], sides[:, 1])
        angle.sum().backward()

        assert float(angle.detach()) == 0.0
        assert sides.grad is not None
        assert torch.equal(sides.grad, torch.zeros(1, 2))


class TestSafeguardParameterValidation:
    """``eps`` is rejected at the values that disable the guard it exists to be (M-23)."""

    @pytest.mark.parametrize("bad", [0.0, -1e-7, float("nan"), float("inf"), float("-inf")])
    def test_a_disabling_eps_is_refused_by_name(self, bad: float) -> None:
        """Zero, negative and non-finite epsilons all raise, and the message names the parameter."""
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        for loss in (box_iou_aligned, complete_iou, ciou_loss):
            with pytest.raises(ValueError, match="eps must be finite and > 0"):
                loss(boxes, boxes, eps=bad)

    def test_a_usable_eps_is_accepted(self) -> None:
        """The check refuses only the disabling values; a smaller working eps still passes."""
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        assert float(ciou_loss(boxes, boxes, eps=1e-12)) == 0.0


class TestCoordinateRange:
    """The documented float32 coordinate ceiling, pinned from both sides (L-12).

    The module states a range rather than enforcing one -- validating a tensor
    would cost a device sync per batch in the training hot path -- so these tests
    are what keeps the stated range honest as the arithmetic changes.
    """

    @staticmethod
    def _loss_and_grad(coordinate: float) -> tuple[torch.Tensor, torch.Tensor]:
        pred = torch.tensor([[0.0, 0.0, coordinate, coordinate]], requires_grad=True)
        target = torch.tensor([[0.0, 0.0, coordinate / 2, coordinate / 2]])
        loss = ciou_loss(pred, target)
        loss.sum().backward()
        assert pred.grad is not None
        return loss, pred.grad

    @pytest.mark.parametrize("coordinate", [1.0e5, 1.0e18, 1.0e19])
    def test_inside_the_documented_range_the_loss_and_gradients_are_finite(self, coordinate: float) -> None:
        """Pixel-frame coordinates and the two decades below the edge all behave."""
        loss, grad = self._loss_and_grad(coordinate)
        assert bool(torch.isfinite(loss).all())
        assert bool(torch.isfinite(grad).all())

    def test_at_the_documented_edge_the_squared_terms_overflow(self) -> None:
        """1e20 is the measured failure the docstring names; it is pinned, not fixed.

        Squaring 1e20 gives 1e40 against float32's 3.4e38 ceiling, so the areas and
        the enclosing diagonal saturate to ``Inf`` and the loss reduces to ``NaN``.
        If a future change makes this finite, the documented range is wrong and
        should be widened -- which is the point of asserting it.
        """
        loss, grad = self._loss_and_grad(1.0e20)
        assert not bool(torch.isfinite(loss).all())
        assert not bool(torch.isfinite(grad).all())
