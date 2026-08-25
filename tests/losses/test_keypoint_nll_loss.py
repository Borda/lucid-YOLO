# SPDX-License-Identifier: Apache-2.0
"""Unit gates for the WP-135 Laplace-NLL ablation loss (R14 Table 7, A65, A66).

This loss exists to be *compared against* :class:`~lucid_yolo.losses.rle_loss.RLELoss`,
so what these tests pin is exactly what a comparison depends on:

* it carries **no trainable parameters** -- the single structural difference from
  RLE, and the one an optimizer, a checkpoint and a same-seed weight draw all
  observe;
* its value is the closed-form Laplace negative log-likelihood, checked against a
  hand-computation that never calls the implementation's own internals, so a
  dropped or doubled term cannot pass by agreeing with itself;
* it is RLE's value **minus the flow term** and nothing else, checked against
  ``RLELoss`` directly on shared inputs -- the property that makes a paired run
  attributable to the flow rather than to some other difference;
* A66's visibility rule and the zero-visible-points contract behave exactly as
  RLE's do, since a control that masked differently would compare two things at
  once;
* gradient reaches both ``mu_hat`` and ``sigma_raw``, the "learnable variance"
  half of Table 7's row name.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import Laplace

from lucid_yolo.losses import LaplaceNLLLoss, RLELoss
from lucid_yolo.losses.keypoint_nll_loss import _MIN_SIGMA

#: Tolerance on float32 log-density arithmetic, matching ``test_rle_loss.py``'s.
_LAPLACE_ATOL = 1e-5


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed torch so the ``RLELoss`` comparison arm's flow initialization is deterministic.

    This loss draws no RNG of its own -- that is one of the properties under test
    -- but the tests that pin it *against* ``RLELoss`` construct a flow, whose
    ``Linear`` layers do. Seeded here rather than in those two bodies so the file
    follows ``test_rle_loss.py``'s own fixture precedent.
    """
    torch.manual_seed(0)


def _reference_loss(mu_hat: torch.Tensor, sigma_raw: torch.Tensor, mu_gt: torch.Tensor) -> torch.Tensor:
    """Recompute the per-point loss from R14 Eq. 8 without the flow term, independently.

    Written out of ``torch.distributions`` and ``torch.log`` directly rather than
    by calling anything in the module under test, so agreement is evidence about
    the formula rather than about the implementation agreeing with itself. Assumes
    every point visible; masking is a separate concern with its own tests.

    Examples:
        >>> import torch
        >>> zeros = torch.zeros(1, 1, 2)
        >>> _reference_loss(zeros, zeros, zeros).shape
        torch.Size([1, 1])
    """
    sigma_hat = torch.sigmoid(sigma_raw).clamp(min=_MIN_SIGMA)
    residual = (mu_gt - mu_hat) / sigma_hat
    return -Laplace(0.0, 1.0).log_prob(residual).sum(dim=-1) + torch.log(sigma_hat).sum(dim=-1)


class TestParameterFreedom:
    """The ablation holds no weights of its own -- Table 7's whole structural claim."""

    def test_exposes_no_trainable_parameters(self) -> None:
        """``LaplaceNLLLoss()`` reports an empty parameter list, where ``RLELoss`` reports a flow.

        The difference the ablation is *for*. A version that accidentally kept a
        learnable term -- a scale, a bias, a residual flow layer -- would still
        produce plausible loss values and would still train, while no longer being
        the control R14 Table 7 published, and a paired run against it would
        attribute the flow's effect to something that was present in both arms.
        """
        ablation = LaplaceNLLLoss()

        assert list(ablation.parameters()) == []
        assert list(RLELoss().parameters()) != []

    def test_contributes_no_state_dict_entries(self) -> None:
        """Its ``state_dict`` is empty, so a checkpoint of an ablation run carries no loss keys.

        Follows from having no parameters but is asserted separately because it is
        the property a *checkpoint* observes: an ablation checkpoint and a
        detection checkpoint of the same model differ only by the point stems, and
        anything held here would silently widen that difference.
        """
        assert dict(LaplaceNLLLoss().state_dict()) == {}


class TestClosedFormValue:
    """The number matches R14 Eq.

    8 with the flow term removed, and only that.
    """

    def test_matches_a_hand_computed_laplace_negative_log_likelihood(self) -> None:
        """The loss equals ``-log Laplace(0, 1)(x_bar) + log sigma_hat``, summed over both axes.

        A fixed, hand-chosen batch scored against an independent recomputation
        from ``torch.distributions``: this is the check that a sign, a missing
        ``log sigma_hat`` Jacobian, or a mean over the wrong axis cannot survive,
        none of which any other test in this file would notice.
        """
        loss_fn = LaplaceNLLLoss()
        mu_hat = torch.tensor([[[0.3, -0.2], [1.0, 0.5]]])
        sigma_raw = torch.tensor([[[0.1, -0.4], [0.0, 2.0]]])
        mu_gt = torch.tensor([[[0.5, -1.5], [0.9, 0.1]]])
        visibility = torch.tensor([[2, 2]])

        loss = loss_fn(mu_hat, sigma_raw, mu_gt, visibility)

        expected = _reference_loss(mu_hat, sigma_raw, mu_gt).mean()
        assert torch.allclose(loss, expected, atol=_LAPLACE_ATOL)

    def test_is_the_rle_loss_with_exactly_the_flow_term_removed(self) -> None:
        """On shared inputs, ``RLELoss`` minus this loss is precisely the flow's log-density.

        The claim the whole ablation rests on, stated as an equation rather than
        as prose in a docstring: the two losses differ by ``-log G_phi(x_bar)`` and
        by nothing else, so a difference measured between two training runs is
        attributable to the flow. A drifted residual, a different ``sigma_hat``
        activation or a different reduction in either module breaks this and
        nothing else in either suite would report it.
        """
        rle = RLELoss()
        ablation = LaplaceNLLLoss()
        mu_hat = torch.tensor([[[0.3, -0.2], [0.1, 0.4]]])
        sigma_raw = torch.tensor([[[0.1, -0.4], [0.2, 0.0]]])
        mu_gt = torch.tensor([[[0.5, -1.5], [-0.3, 0.2]]])
        visibility = torch.tensor([[2, 2]])

        difference = rle(mu_hat, sigma_raw, mu_gt, visibility) - ablation(mu_hat, sigma_raw, mu_gt, visibility)

        residual = (mu_gt - mu_hat) / torch.sigmoid(sigma_raw)
        flow_term = -rle.flow.log_density(residual.reshape(-1, 2)).mean()
        assert torch.allclose(difference, flow_term, atol=_LAPLACE_ATOL)

    def test_a_far_out_residual_stays_finite_rather_than_overflowing(self) -> None:
        """A residual two orders outside the ``O(1)`` range gives a large finite loss and gradient.

        The module docstring claims this ablation degrades more gently than RLE
        under A70's off-canvas points -- the unit Laplace's log-density is linear
        in ``|x_bar|`` where the flow's latent is exponential in a compounding
        log-scale (A72). Asserting finiteness here keeps that claim a measured
        property rather than an argument, and pins the behaviour a pixel-frame
        caller would actually meet.
        """
        mu_hat = torch.zeros(3, 5, 2, requires_grad=True)
        sigma_raw = torch.zeros(3, 5, 2, requires_grad=True)
        mu_gt = torch.full((3, 5, 2), 300.0)
        visibility = torch.ones(3, 5, dtype=torch.int64)

        loss = LaplaceNLLLoss()(mu_hat, sigma_raw, mu_gt, visibility)
        loss.backward()

        assert torch.isfinite(loss)
        assert torch.isfinite(mu_hat.grad).all()
        assert torch.isfinite(sigma_raw.grad).all()


class TestVisibilityMasking:
    """A66 behaves exactly as it does for RLE -- the control must mask identically."""

    def test_unlabeled_points_are_excluded(self) -> None:
        """An instance whose only point is ``v == 0`` contributes nothing to the loss.

        Compares a batch carrying one visible and one unlabeled point against the
        same batch with the unlabeled point's target corrupted -- an unchanged loss
        proves that point was never read. Mirrors ``test_rle_loss.py``'s own
        assertion of the same contract, because a control that trained on a
        different point set than RLE would compare two differences at once.
        """
        loss_fn = LaplaceNLLLoss()
        mu_hat = torch.zeros(2, 1, 2)
        sigma_raw = torch.zeros(2, 1, 2)
        mu_gt = torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]])
        visibility = torch.tensor([[2], [0]])

        baseline = loss_fn(mu_hat, sigma_raw, mu_gt, visibility)

        mu_gt_corrupted = mu_gt.clone()
        mu_gt_corrupted[1, 0] = torch.tensor([999.0, -999.0])
        corrupted = loss_fn(mu_hat, sigma_raw, mu_gt_corrupted, visibility)

        assert torch.equal(baseline, corrupted)

    def test_occluded_but_labeled_points_are_included(self) -> None:
        """A ``v == 1`` (occluded) point changes the loss exactly like ``v == 2`` does.

        A66 reads COCO's ``v in {1, 2}`` as equally valid supervision; this asserts
        the mask does not silently narrow to ``v == 2`` only, the same way
        ``test_rle_loss.py`` asserts it for the loss this one controls for.
        """
        loss_fn = LaplaceNLLLoss()
        mu_hat = torch.zeros(1, 1, 2)
        mu_gt = torch.tensor([[[1.0, 1.0]]])
        sigma_raw = torch.zeros(1, 1, 2)

        loss_occluded = loss_fn(mu_hat, sigma_raw, mu_gt, torch.tensor([[1]]))
        loss_visible = loss_fn(mu_hat, sigma_raw, mu_gt, torch.tensor([[2]]))

        assert torch.equal(loss_occluded, loss_visible)

    def test_no_visible_points_gives_a_finite_zero_still_attached_to_the_graph(self) -> None:
        """All-``v=0`` input returns a finite zero, not ``NaN``, still connected to gradients.

        Mirrors :func:`~lucid_yolo.losses.mask_loss.instance_mask_loss`'s
        no-positives contract: a ``0 / 0`` mean over an empty mask would be
        ``NaN``, and any loss returning ``NaN`` here would poison the whole
        multi-task sum it gets added into. An ablation arm that went ``NaN`` on an
        instance-free batch would end the very run it exists to produce.
        """
        loss_fn = LaplaceNLLLoss()
        mu_hat = torch.zeros(3, 2, 2, requires_grad=True)
        sigma_raw = torch.zeros(3, 2, 2, requires_grad=True)
        mu_gt = torch.zeros(3, 2, 2)
        visibility = torch.zeros(3, 2, dtype=torch.int64)

        loss = loss_fn(mu_hat, sigma_raw, mu_gt, visibility)
        loss.backward()

        assert math.isclose(float(loss.detach()), 0.0, abs_tol=1e-8)
        assert mu_hat.grad is not None
        assert torch.equal(mu_hat.grad, torch.zeros_like(mu_hat))


def test_gradient_reaches_both_the_location_and_the_learned_scale() -> None:
    """Both ``mu_hat`` and ``sigma_raw`` receive non-zero gradient from one backward pass.

    "Learnable variance" is half of what R14 Table 7 names this row, and the
    ``log sigma_hat`` Jacobian term is the only thing giving the likelihood a
    reason to prefer a small scale. A loss that reached ``mu_hat`` alone would be
    an L1 regression wearing an NLL's name, and would still train, still fall, and
    still look entirely ordinary in a loss curve.
    """
    loss_fn = LaplaceNLLLoss()
    mu_hat = torch.tensor([[[0.3, -0.2]]], requires_grad=True)
    sigma_raw = torch.tensor([[[0.1, -0.4]]], requires_grad=True)
    mu_gt = torch.tensor([[[1.0, 1.0]]])
    visibility = torch.tensor([[2]])

    loss_fn(mu_hat, sigma_raw, mu_gt, visibility).backward()

    assert mu_hat.grad is not None
    assert sigma_raw.grad is not None
    assert bool((mu_hat.grad != 0).any())
    assert bool((sigma_raw.grad != 0).any())


def test_toy_convergence_mu_moves_toward_ground_truth() -> None:
    """Optimizing ``mu_hat`` and ``sigma_raw`` against a fixed target reduces the loss over steps.

    Mirrors the RLE suite's own toy-convergence case, and this repo's MuSGD
    precedent (``tests/optim/test_toy_convergence.py``): a minimal, fully-seeded
    scenario driving the loss end to end through an optimizer, rather than a
    single forward-pass assertion. Run for the ablation because "it optimizes at
    all" is the one thing a comparison arm cannot be assumed to do -- an arm that
    silently failed to converge would read as evidence for the flow.
    """
    loss_fn = LaplaceNLLLoss()
    mu_hat = torch.zeros(1, 1, 2, requires_grad=True)
    sigma_raw = torch.zeros(1, 1, 2, requires_grad=True)
    mu_gt = torch.tensor([[[2.0, -1.5]]])
    visibility = torch.tensor([[2]])
    optimizer = torch.optim.Adam([mu_hat, sigma_raw], lr=0.1)

    first_loss = float(loss_fn(mu_hat, sigma_raw, mu_gt, visibility).detach())
    for _ in range(50):
        optimizer.zero_grad()
        loss = loss_fn(mu_hat, sigma_raw, mu_gt, visibility)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        last_loss = float(loss_fn(mu_hat, sigma_raw, mu_gt, visibility))

    assert last_loss < first_loss
    assert torch.allclose(mu_hat.detach(), mu_gt, atol=0.5)
