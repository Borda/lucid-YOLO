# SPDX-License-Identifier: Apache-2.0
"""Unit gates for the WP-123 RLE loss (R14, A65, A66).

Since the flow's own correctness cannot be checked against a training-run
metric until the [GPU][PRINCIPAL]-gated pose-smoke tier (WP-125), every property a
unit test *can* pin here is load-bearing: invertibility (the flow's synthesis
and inverse must be exact inverses of each other, or the density evaluation is
wrong), the log-determinant bookkeeping (checked against autograd's own
Jacobian, not just self-consistency), that the flow's induced density actually
integrates to one over its domain (the single most discriminating check of a
change-of-variables implementation), the fixed Laplace base term against
``torch.distributions`` directly, R14's own "gradient shortcut" claim (the
paper's stated reason ``Q``/flow are summed rather than composed), sigma's
sigmoid range (A65), visibility masking (A66), and a toy convergence run
mirroring this repo's existing MuSGD toy-convergence precedent
(``tests/optim/test_toy_convergence.py``).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import Laplace

from lucid_yolo.losses import RLELoss
from lucid_yolo.losses.rle_loss import _MIN_SIGMA, _AffineCoupling, _RealNVPFlow

_LAPLACE_ATOL = 1e-5


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed torch so flow initialization and sampled inputs are deterministic."""
    torch.manual_seed(0)


class TestAffineCouplingInvertibility:
    """A single coupling layer's synthesis and inverse are exact inverses."""

    def test_invert_undoes_synthesize(self) -> None:
        """``invert(synthesize(z)) == z`` for random 2-D inputs, both transform axes."""
        z = torch.randn(16, 2)
        for transform_index in (0, 1):
            layer = _AffineCoupling(transform_index=transform_index)
            x, _ = layer.synthesize(z)
            z_back, _ = layer.invert(x)
            assert torch.allclose(z_back, z, atol=1e-5)

    def test_conditioning_axis_is_unchanged_by_synthesize(self) -> None:
        """The coordinate a layer conditions on rides through untouched.

        RealNVP's coupling identity only transforms one axis per layer; the
        other must be a literal pass-through, not merely numerically close.
        """
        z = torch.randn(8, 2)
        layer = _AffineCoupling(transform_index=1)

        x, _ = layer.synthesize(z)

        assert torch.equal(x[..., 0], z[..., 0])

    def test_rejects_a_transform_index_outside_the_2d_residual(self) -> None:
        """A ``transform_index`` other than 0 or 1 raises rather than silently no-op.

        The 2-D residual has exactly two coordinates; any other index would
        name a coordinate that does not exist rather than transform nothing.
        """
        with pytest.raises(ValueError, match="transform_index must be 0 or 1"):
            _AffineCoupling(transform_index=2)


class TestRealNVPFlowInvertibility:
    """The full 6-layer stack inverts exactly, matching the single-layer property."""

    def test_invert_undoes_synthesize(self) -> None:
        """``invert(synthesize(z)) == z`` for a batch of random latents through all 6 layers."""
        flow = _RealNVPFlow()
        z = torch.randn(32, 2)

        x, _ = flow.synthesize(z)
        z_back, _ = flow.invert(x)

        assert torch.allclose(z_back, z, atol=1e-4)

    def test_forward_and_inverse_log_determinants_are_negatives(self) -> None:
        """The stack's forward log-det and its inverse's log-det sum to zero.

        A layer's inverse divides by exactly what synthesis multiplied by, so
        undoing a transform must exactly cancel its own log-determinant --
        this is a stronger check than invertibility alone, since it also
        pins the log-det *bookkeeping*, not just the recovered value.
        """
        flow = _RealNVPFlow()
        z = torch.randn(10, 2)

        x, forward_log_det = flow.synthesize(z)
        _, inverse_log_det = flow.invert(x)

        assert torch.allclose(forward_log_det, -inverse_log_det, atol=1e-4)


def test_log_determinant_matches_autograd_jacobian() -> None:
    """The analytic per-layer log-determinant matches autograd's own Jacobian determinant.

    Self-consistency between :meth:`synthesize` and :meth:`invert` alone could
    not catch a sign error or a missing term that happened to cancel between
    the two; comparing against an independently computed Jacobian determinant
    (``torch.autograd.functional.jacobian``) cannot make that mistake, since it
    never calls :meth:`invert` at all.
    """
    layer = _AffineCoupling(transform_index=1)
    z = torch.randn(2)

    def synthesize_one(point: torch.Tensor) -> torch.Tensor:
        out, _ = layer.synthesize(point.unsqueeze(0))
        return out.squeeze(0)

    with torch.no_grad():
        jacobian = torch.autograd.functional.jacobian(synthesize_one, z)
        _, analytic_log_det = layer.synthesize(z.unsqueeze(0))

    autograd_log_det = jacobian.det().abs().log()
    assert math.isclose(float(autograd_log_det), float(analytic_log_det.squeeze(0)), abs_tol=1e-4)


def test_flow_density_integrates_to_one() -> None:
    """The flow's induced density, numerically integrated over a 2-D grid, is close to 1.

    The single most discriminating test of the change-of-variables
    bookkeeping: a base-density term without its accompanying log-determinant
    (or vice versa) would not raise an error anywhere else in this suite, but
    it would not integrate to a proper probability density either.
    """
    flow = _RealNVPFlow()
    grid_1d = torch.linspace(-8.0, 8.0, 161)
    cell = float(grid_1d[1] - grid_1d[0])
    grid_x, grid_y = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
    points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)

    with torch.no_grad():
        density = flow.log_density(points).exp()

    integral = float(density.sum()) * cell * cell

    assert math.isclose(integral, 1.0, abs_tol=0.05)


def test_laplace_base_term_matches_torch_distributions_directly() -> None:
    """The Laplace base term, summed over both axes, matches ``torch.distributions`` exactly.

    Pins the "sum over the two coordinate axes" convention the module
    docstring states (``Q`` evaluated per axis, then summed) against a
    hand-picked residual whose per-axis log-probabilities are independently
    computable.
    """
    residual = torch.tensor([[0.5, -1.5]])

    summed = Laplace(0.0, 1.0).log_prob(residual).sum(dim=-1)

    expected = Laplace(0.0, 1.0).log_prob(torch.tensor(0.5)) + Laplace(0.0, 1.0).log_prob(torch.tensor(-1.5))
    assert torch.allclose(summed, expected.unsqueeze(0), atol=_LAPLACE_ATOL)


class TestSigmaRange:
    """sigma_hat stays in the sigmoid's open (0, 1) range (A65, R14 sec.

    3.3).
    """

    def test_large_positive_raw_input_approaches_but_never_reaches_one(self) -> None:
        """A large positive raw sigma logit saturates near, but strictly below, 1.

        15.0 rather than an even larger value: float32's precision near 1.0 is
        coarser than near 0, and a sufficiently extreme logit rounds
        ``sigmoid`` to exactly ``1.0`` — a float32 saturation artifact, not a
        property of :func:`torch.sigmoid`'s true open range, so the input is
        chosen to stay inside where the distinction is still representable.
        """
        loss_fn = RLELoss()
        mu_hat = torch.zeros(1, 1, 2)
        mu_gt = torch.zeros(1, 1, 2)
        sigma_raw = torch.full((1, 1, 2), 15.0, requires_grad=True)
        visibility = torch.tensor([[2]])

        loss_fn(mu_hat, sigma_raw, mu_gt, visibility)
        sigma_hat = torch.sigmoid(sigma_raw)

        assert bool((sigma_hat < 1.0).all())
        assert bool((sigma_hat > 0.99).all())

    def test_large_negative_raw_input_stays_above_the_floor(self) -> None:
        """A large negative raw sigma logit saturates near, but strictly above, the numeric floor."""
        sigma_raw = torch.full((1, 1, 2), -100.0)

        sigma_hat = torch.sigmoid(sigma_raw).clamp(min=_MIN_SIGMA)

        assert bool((sigma_hat >= _MIN_SIGMA).all())
        assert bool((sigma_hat > 0.0).all())


class TestVisibilityMasking:
    """Only ``v >= 1`` points contribute to the loss (A66)."""

    def test_unlabeled_points_are_excluded(self) -> None:
        """An instance whose only point is ``v == 0`` contributes nothing to the loss.

        Compares a batch carrying one visible and one unlabeled point against
        the same batch with the unlabeled point's target corrupted -- an
        unchanged loss proves that point was never read.
        """
        loss_fn = RLELoss()
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

        A66 reads COCO's ``v in {1, 2}`` as equally valid supervision; this
        asserts the mask does not silently narrow to ``v == 2`` only.
        """
        loss_fn = RLELoss()
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
        multi-task sum it gets added into.
        """
        loss_fn = RLELoss()
        mu_hat = torch.zeros(3, 2, 2, requires_grad=True)
        sigma_raw = torch.zeros(3, 2, 2, requires_grad=True)
        mu_gt = torch.zeros(3, 2, 2)
        visibility = torch.zeros(3, 2, dtype=torch.int64)

        loss = loss_fn(mu_hat, sigma_raw, mu_gt, visibility)
        loss.backward()

        assert math.isclose(float(loss.detach()), 0.0, abs_tol=1e-8)
        assert mu_hat.grad is not None
        assert torch.equal(mu_hat.grad, torch.zeros_like(mu_hat))


def test_gradient_shortcut_reaches_mu_and_sigma_with_the_flow_frozen() -> None:
    """R14's own claim: gradients through the Laplace base term do not depend on the flow.

    R14 sec. 3.2 motivates summing a fixed base density with the learned flow
    correction (rather than only using the flow) as a "gradient shortcut" --
    the regression head keeps receiving a clean signal even when the flow's
    own gradient is unavailable. Freezing every flow parameter and confirming
    ``mu_hat``/``sigma_raw`` still receive nonzero gradient is a direct test
    of that claim, not an incidental property.
    """
    loss_fn = RLELoss()
    for parameter in loss_fn.flow.parameters():
        parameter.requires_grad_(False)

    mu_hat = torch.tensor([[[0.3, -0.2]]], requires_grad=True)
    sigma_raw = torch.tensor([[[0.1, -0.4]]], requires_grad=True)
    mu_gt = torch.tensor([[[1.0, 1.0]]])
    visibility = torch.tensor([[2]])

    loss = loss_fn(mu_hat, sigma_raw, mu_gt, visibility)
    loss.backward()

    assert mu_hat.grad is not None
    assert sigma_raw.grad is not None
    assert bool((mu_hat.grad != 0).any())
    assert bool((sigma_raw.grad != 0).any())


def test_toy_convergence_mu_moves_toward_ground_truth() -> None:
    """Optimizing only ``mu_hat`` against a fixed target reduces the loss over steps.

    Mirrors this repo's MuSGD toy-convergence precedent
    (``tests/optim/test_toy_convergence.py``): a minimal, fully-seeded scenario
    exercising the loss end to end through an optimizer step, rather than a
    single forward-pass assertion.
    """
    loss_fn = RLELoss()
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


class TestCouplingScaleBound:
    """A72: the coupling log-scale is a tanh times a learned scale, not the raw output.

    RealNVP states this parameterization as a stability measure, and R14 cites RealNVP for
    the coupling construction rather than restating it -- so the detail is easy to miss and
    was, until this loss was first composed into a training run. The stack multiplies
    ``exp(log_scale)`` once per layer, so an unbounded log-scale compounds: a measured
    residual of 27 reached the latent as ``4e11`` and the loss as ``7e22``. These pin the
    bound and, more importantly, the behaviour it exists for -- finiteness an order of
    magnitude outside the residual range the equations are usually read at.
    """

    def test_the_log_scale_stays_inside_its_learned_bound(self) -> None:
        """However extreme the conditioning value, ``|log_scale| <= scale``."""
        layer = _AffineCoupling(transform_index=0)
        bound = float(layer.conditioner.scale.detach().abs())

        log_scale, _ = layer.conditioner(torch.tensor([[-1e6], [-10.0], [0.0], [10.0], [1e6]]))

        assert torch.all(log_scale.abs() <= bound + _LAPLACE_ATOL)

    def test_a_far_out_residual_still_yields_a_finite_loss_and_gradient(self) -> None:
        """The regression this bound exists to prevent, stated as the property that failed.

        A residual two orders of magnitude outside the ``O(1)`` range R14's equations are
        read at is exactly what A70's off-canvas points produce, and it must produce a large
        finite number rather than an overflow -- large is a likelihood doing its job,
        non-finite is a run that ends.
        """
        mu_hat = torch.zeros(3, 5, 2, requires_grad=True)
        sigma_raw = torch.zeros(3, 5, 2, requires_grad=True)
        mu_gt = torch.full((3, 5, 2), 300.0)
        visibility = torch.ones(3, 5, dtype=torch.int64)

        loss = RLELoss()(mu_hat, sigma_raw, mu_gt, visibility)
        loss.backward()

        assert torch.isfinite(loss)
        assert torch.isfinite(mu_hat.grad).all()
        assert torch.isfinite(sigma_raw.grad).all()

    def test_the_bound_is_learnable_rather_than_a_constant(self) -> None:
        """RealNVP's scale is *learned*; a hard constant would be a different layer.

        Pinning it as a parameter keeps the deviation honest: A72 claims to restore the
        paper's parameterization, and a frozen constant would not be it.
        """
        layer = _AffineCoupling(transform_index=1)

        names = {name for name, _ in layer.conditioner.named_parameters()}

        assert "scale" in names
        assert layer.conditioner.scale.requires_grad


class TestUnlabeledPointsReachNoGradient:
    """A ``v == 0`` point cannot influence the backward pass either (WP-170, A66).

    ``TestVisibilityMasking`` above asserts the loss *value* is independent of an
    unlabeled point's target, and it was: the mask was applied to the reduction,
    which makes the number right. Everything upstream of the reduction still ran.
    Every ``v == 0`` residual was divided by ``sigma_hat``, pushed through the
    RealNVP stack and evaluated under the standard-normal base, so one non-finite
    unlabeled annotation produced ``NaN`` gradients on the shared flow weights and
    on every *visible* point's ``mu_hat`` — while the reported loss stayed healthy
    and the value assertion above stayed green.

    Value-independence and gradient-independence are different properties and only
    the second one matters for training; these tests pin the second.
    """

    #: Placeholders a real unlabeled COCO point plausibly carries.
    CORRUPTIONS = (float("inf"), float("-inf"), float("nan"), 1e30, -999.0)

    def _grads(self, corrupt: float | None) -> tuple[float, list[torch.Tensor]]:
        """Return the loss value and every gradient produced by one backward pass.

        Builds the loss from a fixed seed each call so the flow's weights are
        identical between the clean and the corrupted run, making the two directly
        comparable under ``torch.equal``.

        Args:
            corrupt: Value written into the unlabeled point's target, or ``None``
                to leave the clean target in place.

        Returns:
            ``(loss_value, gradients)`` — the scalar loss, and the gradients of
            ``mu_hat``, ``sigma_raw`` and every flow parameter in a fixed order.

        Examples:
            >>> case = TestUnlabeledPointsReachNoGradient()
            >>> value, grads = case._grads(None)
            >>> isinstance(value, float) and len(grads) > 2
            True
        """
        torch.manual_seed(0)
        loss_fn = RLELoss()
        mu_hat = torch.zeros(1, 2, 2, requires_grad=True)
        sigma_raw = torch.zeros(1, 2, 2, requires_grad=True)
        mu_gt = torch.tensor([[[0.1, 0.1], [0.3, -0.2]]])
        if corrupt is not None:
            mu_gt = mu_gt.clone()
            mu_gt[0, 1] = torch.tensor([corrupt, corrupt])
        loss = loss_fn(mu_hat, sigma_raw, mu_gt, torch.tensor([[2, 0]]))
        loss.backward()
        assert mu_hat.grad is not None and sigma_raw.grad is not None
        return float(loss.detach()), [mu_hat.grad, sigma_raw.grad, *(p.grad for p in loss_fn.parameters())]

    @pytest.mark.parametrize("corrupt", CORRUPTIONS)
    def test_gradients_are_identical_to_the_clean_run(self, corrupt: float) -> None:
        """Corrupting the unlabeled target changes no gradient anywhere, bit for bit.

        The strong form of A66: not "the masked point contributes little" but "the
        masked point is not in the graph", which ``torch.equal`` against the clean
        run is what actually establishes.
        """
        clean_value, clean_grads = self._grads(None)
        value, grads = self._grads(corrupt)

        assert value == clean_value
        for clean, actual in zip(clean_grads, grads, strict=True):
            assert torch.equal(clean, actual)

    @pytest.mark.parametrize("corrupt", CORRUPTIONS)
    def test_every_gradient_stays_finite(self, corrupt: float) -> None:
        """No ``NaN`` reaches ``mu_hat``, ``sigma_raw`` or the flow's own weights.

        The flow's weights are the consequence that outlives the step: they are
        shared across every point in the batch, so one unlabeled annotation
        poisoned the density for all of them.
        """
        _, grads = self._grads(corrupt)

        assert all(bool(torch.isfinite(grad).all()) for grad in grads)

    @pytest.mark.parametrize("corrupt", CORRUPTIONS)
    def test_a_masked_point_does_not_raise_from_argument_validation(self, corrupt: float) -> None:
        """A non-finite unlabeled target is not rejected by the base distribution.

        ``torch.distributions`` validates its arguments by default, so before the
        selection moved upstream the standard-normal base raised ``ValueError`` on
        a residual belonging to a point the loss had already decided to ignore --
        a masked-out annotation crashing the loss outright.
        """
        mu_gt = torch.tensor([[[0.1, 0.1], [corrupt, corrupt]]])

        loss = RLELoss()(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), mu_gt, torch.tensor([[2, 0]]))

        assert bool(torch.isfinite(loss))

    def test_the_flow_only_sees_the_labelled_points(self) -> None:
        """Selection happens before the flow, so its input is ``(V, 2)`` not ``(N*K, 2)``.

        The structural claim behind the two assertions above, and the reason the
        fix also costs less: three of four points here are unlabeled and never
        reach the RealNVP stack at all.
        """
        seen: list[tuple[int, ...]] = []
        loss_fn = RLELoss()
        original = loss_fn.flow.log_density

        def _record(x: torch.Tensor) -> torch.Tensor:
            """Record the shape the flow is called with, then defer to the real method."""
            seen.append(tuple(x.shape))
            return original(x)

        loss_fn.flow.log_density = _record  # type: ignore[method-assign]
        loss_fn(
            torch.zeros(2, 2, 2),
            torch.zeros(2, 2, 2),
            torch.zeros(2, 2, 2),
            torch.tensor([[2, 0], [0, 0]]),
        )

        assert seen == [(1, 2)]
