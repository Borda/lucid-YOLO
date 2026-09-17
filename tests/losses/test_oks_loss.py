# SPDX-License-Identifier: Apache-2.0
"""Unit gates for the WP-178 OKS loss (R12's definition, A66, A75).

R1 names the term and states no formula, so nothing here can check a published
number; what a unit test *can* pin is that the module computes R12's OKS exactly
as written — a hand-computed two-point instance to ``1e-6`` — and every edge the
training step will hit: a perfect prediction scores zero, a ``v = 0`` point is
outside the graph rather than merely multiplied by zero (A66), an instance with no
labelled point is left out of the mean rather than scored as ``0 / 0``, the empty
batch returns an attached finite zero, the area floor holds, the sigma table
matches the input's ``K``, and the module carries no parameters and no persistent
state a checkpoint could pick up.
"""

from __future__ import annotations

import math

import pytest
import torch

from lucid_yolo.losses import OKSLoss

#: A two-point sigma table with unequal tolerances, so a hand computation that
#: silently swapped the two points would not come out the same.
_SIGMAS = (0.1, 0.2)
#: Tolerance of the hand-computed OKS reference, computed in float64.
_REFERENCE_ATOL = 1e-6


def _two_point_instance() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One instance, two labelled points, offsets ``(3, 4)`` and ``(0, 1)``, area 100.

    Examples:
        >>> mu_hat, mu_gt, area, visibility = _two_point_instance()
        >>> mu_hat.shape, area.tolist(), visibility.tolist()
        (torch.Size([1, 2, 2]), [100.0], [[2, 1]])
    """
    mu_hat = torch.zeros(1, 2, 2, dtype=torch.float64)
    mu_gt = torch.tensor([[[3.0, 4.0], [0.0, 1.0]]], dtype=torch.float64)
    return mu_hat, mu_gt, torch.tensor([100.0], dtype=torch.float64), torch.tensor([[2, 1]])


class TestDefinition:
    """The value is R12's ``1 - OKS``, with ``s^2`` the area and ``k_i = 2 sigma_i``."""

    def test_matches_a_hand_computed_oks(self) -> None:
        """One instance, two points: ``1 - mean_i exp(-d_i^2 / (2 * area * (2 sigma_i)^2))``."""
        mu_hat, mu_gt, area, visibility = _two_point_instance()
        first = math.exp(-25.0 / (2.0 * 100.0 * (2.0 * 0.1) ** 2))  # d^2 = 3^2 + 4^2
        second = math.exp(-1.0 / (2.0 * 100.0 * (2.0 * 0.2) ** 2))  # d^2 = 0^2 + 1^2
        expected = 1.0 - (first + second) / 2.0

        loss = OKSLoss(sigmas=_SIGMAS)(mu_hat, mu_gt, area, visibility)

        assert float(loss) == pytest.approx(expected, abs=_REFERENCE_ATOL)

    def test_a_perfect_prediction_scores_zero(self) -> None:
        """``OKS = 1`` at zero distance, so the loss is exactly zero there."""
        _, mu_gt, area, visibility = _two_point_instance()

        loss = OKSLoss(sigmas=_SIGMAS)(mu_gt.clone(), mu_gt, area, visibility)

        assert float(loss) == 0.0

    def test_instances_are_meaned_after_their_own_per_point_mean(self) -> None:
        """Two instances weigh equally however many labelled points each carries.

        R12 normalizes each instance by *its* labelled count before anything is
        averaged across instances; a single flat mean over points would let a
        fully-labelled instance outvote a sparsely-labelled one.
        """
        loss_fn = OKSLoss(sigmas=_SIGMAS)
        mu_hat, mu_gt, area, _ = _two_point_instance()
        both = torch.tensor([[2, 2]])
        one = torch.tensor([[2, 0]])
        alone_both = float(loss_fn(mu_hat, mu_gt, area, both))
        alone_one = float(loss_fn(mu_hat, mu_gt, area, one))

        paired = loss_fn(
            torch.cat([mu_hat, mu_hat]), torch.cat([mu_gt, mu_gt]), torch.cat([area, area]), torch.cat([both, one])
        )

        assert float(paired) == pytest.approx((alone_both + alone_one) / 2.0, abs=_REFERENCE_ATOL)

    def test_the_area_is_floored_at_one_square_pixel(self) -> None:
        """A zero or negative area scores as one square pixel rather than dividing by zero."""
        loss_fn = OKSLoss(sigmas=_SIGMAS)
        mu_hat, mu_gt, _, visibility = _two_point_instance()
        floored = loss_fn(mu_hat, mu_gt, torch.tensor([1.0], dtype=torch.float64), visibility)

        degenerate = loss_fn(mu_hat, mu_gt, torch.tensor([0.0], dtype=torch.float64), visibility)

        assert math.isfinite(float(degenerate))
        assert float(degenerate) == float(floored)


class TestVisibilityMasking:
    """``v = 0`` points are excluded before any arithmetic (A66)."""

    def test_an_unlabelled_point_is_outside_the_graph(self) -> None:
        """Corrupting a ``v = 0`` coordinate with ``inf`` changes neither the loss nor a gradient.

        The decisive form of the check: a mask applied *after* the distance would keep
        the value right and put ``0 * inf = NaN`` into ``mu_hat``'s gradient.
        """
        loss_fn = OKSLoss(sigmas=_SIGMAS)
        mu_hat, mu_gt, area, _ = _two_point_instance()
        visibility = torch.tensor([[2, 0]])
        clean = float(loss_fn(mu_hat, mu_gt, area, visibility))
        corrupted_gt = mu_gt.clone()
        corrupted_gt[0, 1] = float("inf")
        mu_hat = mu_hat.clone().requires_grad_(True)

        loss = loss_fn(mu_hat, corrupted_gt, area, visibility)

        loss.backward()
        assert float(loss.detach()) == clean
        assert mu_hat.grad is not None
        assert bool(torch.isfinite(mu_hat.grad).all())
        assert bool((mu_hat.grad[0, 1] == 0).all())  # nothing flows to the unlabelled point

    def test_an_instance_without_labelled_points_is_left_out_of_the_mean(self) -> None:
        """An all-``v = 0`` instance has no OKS (R12's denominator is zero) and is skipped."""
        loss_fn = OKSLoss(sigmas=_SIGMAS)
        mu_hat, mu_gt, area, visibility = _two_point_instance()
        alone = float(loss_fn(mu_hat, mu_gt, area, visibility))

        with_unlabelled = loss_fn(
            torch.cat([mu_hat, mu_hat]),
            torch.cat([mu_gt, mu_gt]),
            torch.cat([area, area]),
            torch.cat([visibility, torch.zeros_like(visibility)]),
        )

        assert float(with_unlabelled) == pytest.approx(alone, abs=_REFERENCE_ATOL)

    def test_an_all_unlabelled_batch_is_a_finite_attached_zero(self) -> None:
        """No labelled point anywhere: zero, finite, and still on ``mu_hat``'s graph."""
        mu_hat = torch.zeros(2, 2, 2, requires_grad=True)

        loss = OKSLoss(sigmas=_SIGMAS)(mu_hat, torch.ones(2, 2, 2), torch.ones(2), torch.zeros(2, 2, dtype=torch.int64))

        assert float(loss.detach()) == 0.0
        assert loss.requires_grad

    def test_an_empty_batch_is_a_finite_attached_zero(self) -> None:
        """``N = 0``: zero, finite, and still on ``mu_hat``'s graph."""
        mu_hat = torch.zeros(0, 2, 2, requires_grad=True)

        loss = OKSLoss(sigmas=_SIGMAS)(
            mu_hat, torch.zeros(0, 2, 2), torch.zeros(0), torch.zeros(0, 2, dtype=torch.int64)
        )

        assert float(loss.detach()) == 0.0
        assert loss.requires_grad


class TestModuleShape:
    """No parameters, no persistent state, gradient where it belongs."""

    def test_holds_no_parameters_and_an_empty_state_dict(self) -> None:
        """``k`` is a non-persistent buffer: nothing to train, nothing for a checkpoint to gain.

        The pinned key set of a keypoints module is the detection keys plus the point
        stems plus ``rle_loss.``; a persistent buffer here would add a key and stop
        every earlier pose checkpoint loading strictly.
        """
        loss_fn = OKSLoss(sigmas=_SIGMAS)

        assert list(loss_fn.parameters()) == []
        assert dict(loss_fn.state_dict()) == {}
        assert loss_fn.k.tolist() == [pytest.approx(2.0 * sigma) for sigma in _SIGMAS]

    def test_the_buffer_follows_the_module_dtype(self) -> None:
        """Non-persistent still means device- and dtype-placed with the module."""
        loss_fn = OKSLoss(sigmas=_SIGMAS).to(torch.float64)

        assert loss_fn.k.dtype == torch.float64

    def test_gradient_reaches_the_prediction(self) -> None:
        """A labelled, imperfect point leaves a non-zero gradient on ``mu_hat``."""
        mu_hat, mu_gt, area, visibility = _two_point_instance()
        mu_hat = mu_hat.requires_grad_(True)

        OKSLoss(sigmas=_SIGMAS)(mu_hat, mu_gt, area, visibility).backward()

        assert mu_hat.grad is not None
        assert bool((mu_hat.grad != 0).any())

    def test_rejects_a_point_count_that_is_not_the_sigma_tables(self) -> None:
        """``K`` must match the table — a mismatch is a schema error, not a broadcast."""
        with pytest.raises(ValueError, match="sigma table has 2 keypoints but the inputs carry 3"):
            OKSLoss(sigmas=_SIGMAS)(
                torch.zeros(1, 3, 2), torch.zeros(1, 3, 2), torch.ones(1), torch.ones(1, 3, dtype=torch.int64)
            )

    @pytest.mark.parametrize("sigmas", [(), (0.1, 0.0), (0.1, -0.2)])
    def test_rejects_an_empty_or_non_positive_sigma_table(self, sigmas: tuple[float, ...]) -> None:
        """A zero sigma makes a point's tolerance zero and the loss constant off the optimum."""
        with pytest.raises(ValueError, match="non-empty sequence of positive values"):
            OKSLoss(sigmas=sigmas)
