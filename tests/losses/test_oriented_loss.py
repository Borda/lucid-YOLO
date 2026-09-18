# SPDX-License-Identifier: Apache-2.0
"""Unit gates for the WP-088 oriented branch terms (A41, A49, A50).

The module had no test file of its own: its behaviour was covered only through the
oriented training path, which is where L-09 was found. The gap that mattered is the
compensating pair A50 argues for and nothing asserted — ProbIoU floors a collapsed
side at ``min_side`` and gives it **zero** gradient there (A41), so the rotated IoU
term alone cannot pull a collapsed extent back out, and ``rl1`` is kept for exactly
that reason. These tests tie the two together: whenever ``rbox``'s gradient on a
side is floored away, ``rl1``'s is not.

Also covered: the form lookup, the normalizer's provenance, and the no-ground-truth
path, all of which the module docstring states and none of which had an assertion
outside a doctest.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from lucid_yolo.assign.tal import AssignResult
from lucid_yolo.losses.angle_loss import square_angle_loss
from lucid_yolo.losses.oriented_loss import (
    DEFAULT_ROTATED_IOU_FORM,
    ROTATED_IOU_FORMS,
    OrientedLossOutput,
    oriented_branch_terms,
)
from lucid_yolo.losses.probiou import _MIN_SIDE

#: Per-anchor stride of the single test level, in the frame the L1 term is measured in.
_STRIDE = 8.0


def _single_positive_assignment(weight: float = 1.0) -> AssignResult:
    """One anchor, one instance, foreground — the smallest assignment that scores anything.

    Args:
        weight: The positive anchor's alignment weight ``q_i``.

    Returns:
        A two-anchor :class:`~lucid_yolo.assign.tal.AssignResult` whose first anchor
        is a positive on instance ``0`` and whose second is background.

    Examples:
        >>> assignment = _single_positive_assignment()
        >>> bool(assignment.fg_mask[0, 0]), bool(assignment.fg_mask[0, 1])
        (True, False)
    """
    return AssignResult(
        fg_mask=torch.tensor([[True, False]]),
        gt_index=torch.tensor([[0, -1]]),
        target_labels=torch.tensor([[0, -1]]),
        target_boxes=torch.zeros(1, 2, 4),
        align_weights=torch.tensor([[weight, 0.0]]),
    )


def _terms_for(pred_rboxes: Tensor, gt_rboxes: Tensor, weight: float = 1.0) -> OrientedLossOutput:
    """Score one branch against a single-positive assignment on one stride level.

    Args:
        pred_rboxes: ``(1, 2, 5)`` predicted canonical rotated boxes.
        gt_rboxes: ``(1, N, 5)`` padded ground-truth rotated boxes.
        weight: The positive anchor's alignment weight.

    Returns:
        The branch's three pre-gain oriented terms.

    Examples:
        >>> import torch
        >>> box = torch.tensor([[4.0, 4.0, 8.0, 2.0, 0.25]])
        >>> terms = _terms_for(box.expand(1, 2, 5), box.unsqueeze(0))
        >>> float(terms.rbox), float(terms.rl1), float(terms.angle)
        (0.0, 0.0, 0.0)
    """
    return oriented_branch_terms(
        pred_rboxes,
        pred_rboxes[..., 4],
        gt_rboxes,
        _single_positive_assignment(weight),
        torch.tensor([_STRIDE, _STRIDE]),
    )


class TestFlooredExtentKeepsAGradient:
    """A collapsed side is gradient-dead in ``rbox`` and alive in ``rl1`` (L-09, A50).

    A41 floors ``w`` and ``h`` at ``min_side`` before they become variances, and a
    clamped side has zero gradient by construction: the loss does not chase a box
    that has collapsed. A50's argument for keeping the L1 term is that this is
    precisely the state an untrained head starts in — raw ltrb distances routinely
    decode to ``r < -l`` — so something must still pull the extent back out. Nothing
    asserted that until now.
    """

    @staticmethod
    def _side_gradients(collapsed_side: int) -> tuple[float, float]:
        """Return ``(d rbox / d side, d rl1 / d side)`` for a prediction collapsed on one side.

        Args:
            collapsed_side: Column index of the collapsed extent, ``2`` for ``w``
                and ``3`` for ``h``.

        Returns:
            The two gradient magnitudes, taken on separate graphs so neither term's
            backward pass can contaminate the other's.
        """
        target = torch.tensor([[[4.0, 4.0, 8.0, 6.0, 0.2]]])
        magnitudes = []
        for term in ("rbox", "rl1"):
            box = torch.tensor([4.0, 4.0, 8.0, 6.0, 0.2])
            box[collapsed_side] = 0.0
            pred = box.expand(1, 2, 5).clone().requires_grad_(True)
            terms = _terms_for(pred, target)
            getattr(terms, term).backward()
            assert pred.grad is not None
            magnitudes.append(float(pred.grad[0, 0, collapsed_side].abs()))
        return magnitudes[0], magnitudes[1]

    @pytest.mark.parametrize(("name", "column"), [("width", 2), ("height", 3)])
    def test_rl1_is_non_zero_exactly_where_rbox_is_floored(self, name: str, column: int) -> None:
        """The compensating pair, asserted as a pair: one term stops, the other does not."""
        rbox_gradient, rl1_gradient = self._side_gradients(column)

        assert rbox_gradient == 0.0, f"ProbIoU is expected to be gradient-dead on a collapsed {name}"
        assert rl1_gradient > 0.0, f"nothing would pull a collapsed {name} back out"

    def test_the_floor_is_what_kills_the_rbox_gradient(self) -> None:
        """Names the mechanism: a side above the floor still moves the rotated IoU term."""
        target = torch.tensor([[[4.0, 4.0, 8.0, 6.0, 0.2]]])
        box = torch.tensor([4.0, 4.0, 8.0, 6.0, 0.2])
        box[2] = _MIN_SIDE * 100.0  # small, but clear of the clamp
        pred = box.expand(1, 2, 5).clone().requires_grad_(True)

        _terms_for(pred, target).rbox.backward()

        assert pred.grad is not None
        assert float(pred.grad[0, 0, 2].abs()) > 0.0

    def test_a_doubly_collapsed_prediction_still_moves(self) -> None:
        """Both extents floored at once — the untrained-head state A50 names — stays trainable."""
        target = torch.tensor([[[4.0, 4.0, 8.0, 6.0, 0.2]]])
        pred = torch.tensor([4.0, 4.0, 0.0, 0.0, 0.2]).expand(1, 2, 5).clone().requires_grad_(True)

        terms = _terms_for(pred, target)
        (terms.rbox + terms.rl1).backward()

        assert pred.grad is not None
        assert bool(torch.isfinite(pred.grad).all())
        assert float(pred.grad[0, 0, 2].abs()) > 0.0
        assert float(pred.grad[0, 0, 3].abs()) > 0.0


class TestNormalizer:
    """Both weighted terms divide by the alignment-weight sum floored at one."""

    def test_a_weight_sum_above_one_divides_by_the_sum(self) -> None:
        """Two unit weights halve a term that a single unit weight reports in full."""
        target = torch.tensor([[[4.0, 4.0, 8.0, 6.0, 0.2]]])
        pred = torch.tensor([5.0, 4.0, 8.0, 6.0, 0.2]).expand(1, 2, 5).clone()

        both_positive = oriented_branch_terms(
            pred,
            pred[..., 4],
            target,
            AssignResult(
                fg_mask=torch.tensor([[True, True]]),
                gt_index=torch.tensor([[0, 0]]),
                target_labels=torch.tensor([[0, 0]]),
                target_boxes=torch.zeros(1, 2, 4),
                align_weights=torch.tensor([[1.0, 1.0]]),
            ),
            torch.tensor([_STRIDE, _STRIDE]),
        )
        one_positive = _terms_for(pred, target)

        assert float(both_positive.rl1) == pytest.approx(float(one_positive.rl1), rel=1e-6)

    def test_below_unit_weight_the_floor_makes_the_term_a_weighted_sum(self) -> None:
        """The documented consequence of ``clamp(min=1.0)``: a sparse batch weighs less."""
        target = torch.tensor([[[4.0, 4.0, 8.0, 6.0, 0.2]]])
        pred = torch.tensor([5.0, 4.0, 8.0, 6.0, 0.2]).expand(1, 2, 5).clone()

        full = _terms_for(pred, target, weight=1.0)
        quarter = _terms_for(pred, target, weight=0.25)

        assert float(quarter.rl1) == pytest.approx(0.25 * float(full.rl1), rel=1e-6)


class TestFormSelection:
    """The rotated-IoU form is a named lookup with a stated default (A49)."""

    def test_the_default_is_the_bounded_hellinger_form(self) -> None:
        """A49 selects ``H_D`` so A13's ``box_gain`` transfers without inventing a magnitude."""
        assert DEFAULT_ROTATED_IOU_FORM == "hellinger"
        assert set(ROTATED_IOU_FORMS) == {"hellinger", "bhattacharyya"}

    def test_the_unbounded_form_scores_a_separated_pair_higher(self) -> None:
        """``B_D`` is unbounded above where ``H_D`` saturates towards one."""
        target = torch.tensor([[[4.0, 4.0, 8.0, 6.0, 0.2]]])
        pred = torch.tensor([400.0, 400.0, 8.0, 6.0, 0.2]).expand(1, 2, 5).clone()
        assignment = _single_positive_assignment()
        strides = torch.tensor([_STRIDE, _STRIDE])

        bounded = oriented_branch_terms(pred, pred[..., 4], target, assignment, strides, form="hellinger")
        unbounded = oriented_branch_terms(pred, pred[..., 4], target, assignment, strides, form="bhattacharyya")

        assert float(unbounded.rbox) > float(bounded.rbox)
        assert float(bounded.rbox) <= 1.0

    def test_an_unknown_form_names_the_alternatives(self) -> None:
        """The error is actionable rather than a bare ``KeyError``."""
        target = torch.tensor([[[4.0, 4.0, 8.0, 6.0, 0.2]]])
        pred = target.expand(1, 2, 5).clone()

        with pytest.raises(ValueError, match="unknown rotated-IoU form"):
            oriented_branch_terms(
                pred, pred[..., 4], target, _single_positive_assignment(), torch.tensor([_STRIDE, _STRIDE]), form="l2"
            )


class TestNoGroundTruth:
    """An empty ground truth gives three exact zeros that still carry gradient."""

    def test_every_term_is_zero_finite_and_connected(self) -> None:
        """``N == 0`` pads to a tensor no ``gather`` can index; the step needs no special case."""
        pred = torch.tensor([4.0, 4.0, 8.0, 6.0, 0.2]).expand(1, 2, 5).clone().requires_grad_(True)

        terms = _terms_for(pred, torch.zeros(1, 0, 5))
        (terms.rbox + terms.rl1 + terms.angle).backward()

        assert float(terms.rbox.detach()) == 0.0
        assert float(terms.rl1.detach()) == 0.0
        assert float(terms.angle.detach()) == 0.0
        assert pred.grad is not None
        assert torch.equal(pred.grad, torch.zeros_like(pred))


def _masked_oriented_terms_oracle(
    pred_rboxes: Tensor, pred_theta: Tensor, gt_rboxes: Tensor, assign: AssignResult, strides: Tensor
) -> OrientedLossOutput:
    """The pre-WP-183 ``oriented_branch_terms`` body, verbatim: boolean gathers over the positives.

    The Hellinger form only — the reference the dense form is held against, with the
    ``N == 0`` early return left out because that path did not change.

    Examples:
        >>> box = torch.tensor([[4.0, 4.0, 8.0, 2.0, 0.25]])
        >>> terms = _masked_oriented_terms_oracle(
        ...     box.expand(1, 2, 5), box[:, 4].expand(1, 2), box.unsqueeze(0),
        ...     _single_positive_assignment(), torch.tensor([_STRIDE, _STRIDE]),
        ... )
        >>> float(terms.rbox), float(terms.rl1), float(terms.angle)
        (0.0, 0.0, 0.0)
    """
    rotated_iou_loss = ROTATED_IOU_FORMS["hellinger"]
    fg_mask = assign.fg_mask
    weight_sum = assign.align_weights.sum().clamp(min=1.0)
    instance_index = assign.gt_index.clamp(min=0).unsqueeze(-1).expand(-1, -1, 5)
    target_pos = gt_rboxes.gather(1, instance_index)[fg_mask]  # (P, 5)
    pred_pos = pred_rboxes[fg_mask]  # (P, 5)
    weights = assign.align_weights[fg_mask]  # (P,)

    rbox_terms = rotated_iou_loss(pred_pos, target_pos)  # (P,)
    stride_pos = strides.expand(fg_mask.shape)[fg_mask].unsqueeze(-1)  # (P, 1)
    l1_terms = ((pred_pos[:, :4] - target_pos[:, :4]) / stride_pos).abs().sum(dim=-1)  # (P,)
    return OrientedLossOutput(
        rbox=(rbox_terms * weights).sum() / weight_sum,
        rl1=(l1_terms * weights).sum() / weight_sum,
        angle=square_angle_loss(pred_theta[fg_mask], target_pos[:, 4], target_pos[:, 2], target_pos[:, 3], weights),
    )


def _random_rboxes(*leading: int) -> Tensor:
    """Random long-edge rotated boxes ``(cx, cy, w, h, theta)`` with positive sides.

    Examples:
        >>> boxes = _random_rboxes(2, 3)
        >>> tuple(boxes.shape), bool((boxes[..., 2:4] > 0).all())
        ((2, 3, 5), True)
    """
    centre = torch.rand(*leading, 2) * 100.0
    sides = torch.rand(*leading, 2) * 40.0 + 1.0
    theta = (torch.rand(*leading, 1) - 0.5) * torch.pi
    return torch.cat([centre, sides, theta], dim=-1)


def _random_assign(fg_mask: Tensor, num_instances: int) -> AssignResult:
    """A contract-conforming assignment whose positives index random instances in ``[0, N)``.

    ``gt_index`` is drawn rather than fixed at ``0`` so the ``gather`` over the padded
    instance axis is exercised, not only its first column.

    Examples:
        >>> assign = _random_assign(torch.tensor([[True, False]]), num_instances=3)
        >>> int(assign.gt_index[0, 1]), float(assign.align_weights[0, 1])
        (-1, 0.0)
    """
    batch, anchors = fg_mask.shape
    background = torch.full((batch, anchors), -1)
    gt_index = torch.where(fg_mask, torch.randint(0, num_instances, (batch, anchors)), background)
    return AssignResult(
        fg_mask=fg_mask,
        gt_index=gt_index,
        target_labels=torch.where(fg_mask, torch.zeros_like(gt_index), background),
        target_boxes=torch.zeros(batch, anchors, 4),
        align_weights=torch.rand(batch, anchors) * fg_mask,
    )


class TestDenseOrientedTermsMatchMaskedReference:
    """The dense ``(B, A)`` oriented terms reproduce the masked-gather implementation (WP-183).

    Each scenario scores one random assignment with the shipped dense form and with
    the pre-change oracle above, on separate leaf tensors for ``pred_rboxes`` and
    ``pred_theta`` so the two backward passes cannot accumulate. All three terms and
    both gradients must agree to ``1e-6``, and the gradients must be finite.
    """

    @staticmethod
    def _score_both(
        pred_rboxes: Tensor, pred_theta: Tensor, gt_rboxes: Tensor, assign: AssignResult
    ) -> tuple[tuple[OrientedLossOutput, Tensor, Tensor], tuple[OrientedLossOutput, Tensor, Tensor]]:
        """Return ``(terms, d/d pred_rboxes, d/d pred_theta)`` from the dense form and from the oracle."""
        strides = torch.tensor([8.0, 16.0, 32.0, 8.0, 16.0, 32.0])[: assign.fg_mask.shape[1]]
        dense_rboxes, dense_theta = pred_rboxes.clone().requires_grad_(True), pred_theta.clone().requires_grad_(True)
        dense = oriented_branch_terms(dense_rboxes, dense_theta, gt_rboxes, assign, strides)
        (dense.rbox + dense.rl1 + dense.angle).backward()
        oracle_rboxes = pred_rboxes.clone().requires_grad_(True)
        oracle_theta = pred_theta.clone().requires_grad_(True)
        oracle = _masked_oriented_terms_oracle(oracle_rboxes, oracle_theta, gt_rboxes, assign, strides)
        (oracle.rbox + oracle.rl1 + oracle.angle).backward()
        assert dense_rboxes.grad is not None and dense_theta.grad is not None
        assert oracle_rboxes.grad is not None and oracle_theta.grad is not None
        return (dense, dense_rboxes.grad, dense_theta.grad), (oracle, oracle_rboxes.grad, oracle_theta.grad)

    @staticmethod
    def _assert_agree(
        dense: tuple[OrientedLossOutput, Tensor, Tensor], oracle: tuple[OrientedLossOutput, Tensor, Tensor]
    ) -> None:
        """Every term and both gradients agree to ``1e-6``, and the dense gradients are finite."""
        dense_terms, dense_rbox_grad, dense_theta_grad = dense
        oracle_terms, oracle_rbox_grad, oracle_theta_grad = oracle
        assert torch.allclose(dense_terms.rbox, oracle_terms.rbox, atol=1e-6)
        assert torch.allclose(dense_terms.rl1, oracle_terms.rl1, atol=1e-6)
        assert torch.allclose(dense_terms.angle, oracle_terms.angle, atol=1e-6)
        assert bool(torch.isfinite(dense_rbox_grad).all()) and bool(torch.isfinite(dense_theta_grad).all())
        assert torch.allclose(dense_rbox_grad, oracle_rbox_grad, atol=1e-6)
        assert torch.allclose(dense_theta_grad, oracle_theta_grad, atol=1e-6)

    def test_mixed_batch_with_one_empty_image(self) -> None:
        """A ``B=4`` batch where one image has no positive scores identically in value and gradient.

        Positives index three padded instances at random, so the dense gather is held
        against the masked one on every column of the instance axis. The empty image
        carries all-zero padded ground truth, as a collated batch does, so every one
        of its background anchors scores densely against a ``(0, 0, 0, 0, 0)`` box —
        the row the masked form drops and the dense form must carry as exact zeros.
        """
        fg_mask = torch.rand(4, 6) < 0.5
        fg_mask[1] = False
        fg_mask[0, 0] = True
        assign = _random_assign(fg_mask, num_instances=3)
        pred_rboxes = _random_rboxes(4, 6)
        gt_rboxes = _random_rboxes(4, 3)
        gt_rboxes[1] = 0.0

        dense, oracle = self._score_both(pred_rboxes, pred_rboxes[..., 4] * 3.0, gt_rboxes, assign)

        self._assert_agree(dense, oracle)

    def test_all_background_batch_is_zero_and_connected(self) -> None:
        """No positive anywhere gives three exact zeros and finite, all-zero gradients.

        The ground truth is padded but every anchor is background, so the masked form
        gathers nothing and the dense form selects nothing — both must be the same
        zero, still connected to both predictions.
        """
        assign = _random_assign(torch.zeros(4, 6, dtype=torch.bool), num_instances=2)
        pred_rboxes = _random_rboxes(4, 6)

        (dense, rbox_grad, theta_grad), (oracle, _, _) = self._score_both(
            pred_rboxes, pred_rboxes[..., 4], _random_rboxes(4, 2), assign
        )

        assert float(dense.rbox.detach()) == 0.0 and float(dense.rl1.detach()) == 0.0
        assert float(dense.angle.detach()) == 0.0 and float(oracle.angle.detach()) == 0.0
        assert torch.equal(rbox_grad, torch.zeros_like(pred_rboxes))
        assert torch.equal(theta_grad, torch.zeros_like(pred_rboxes[..., 4]))

    def test_collapsed_background_predictions_leak_nothing(self) -> None:
        """Collapsed and zero predictions on background anchors reach neither the value nor the gradient.

        A background anchor scores against instance ``0`` in the dense form and against
        nothing in the masked one; planting a fully collapsed box and an all-zero box
        there proves the ``torch.where`` selection keeps whatever ProbIoU makes of them
        out of the sum and the gradient.
        """
        fg_mask = torch.tensor([[True, False, False], [False, True, False]])
        assign = _random_assign(fg_mask, num_instances=2)
        pred_rboxes = _random_rboxes(2, 3)
        pred_rboxes[0, 1] = torch.tensor([4.0, 4.0, 0.0, 0.0, 0.3])  # both sides collapsed
        pred_rboxes[0, 2] = torch.zeros(5)
        pred_rboxes[1, 2] = torch.zeros(5)

        (dense, rbox_grad, theta_grad), oracle = self._score_both(
            pred_rboxes, pred_rboxes[..., 4], _random_rboxes(2, 2), assign
        )

        self._assert_agree((dense, rbox_grad, theta_grad), oracle)
        assert torch.equal(rbox_grad[~fg_mask], torch.zeros(4, 5))
        assert torch.equal(theta_grad[~fg_mask], torch.zeros(4))
