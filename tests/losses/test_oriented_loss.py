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
