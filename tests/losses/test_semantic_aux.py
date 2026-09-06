# SPDX-License-Identifier: Apache-2.0
"""Unit gates for the WP-051 auxiliary semantic BCE+Dice loss (A17, A36).

Covers the two terms at both confidence extremes, the unit-coefficient reading of
"equal weight", the empty-prediction-versus-empty-target smoothing case, and the
end-to-end contract with :class:`~lucid_yolo.models.heads.semantic.SemanticAux`'s
training-mode output.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from lucid_yolo.losses import SemanticAuxOutput, semantic_aux_loss
from lucid_yolo.models import SemanticAux

#: Logit magnitude at which sigmoid saturates well past float32 BCE resolution.
_CONFIDENT = 20.0


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed every RNG source before each test for deterministic tensors."""
    torch.manual_seed(0)


def test_confident_correct_predictions_drive_both_terms_to_zero() -> None:
    """Saturated logits matching their targets leave both terms near zero.

    Catches an inverted sign in either term (a Dice returning the coefficient
    rather than ``1 - coefficient``, or a BCE fed the complement target), which
    would peak exactly where the loss should bottom out.
    """
    targets = torch.zeros(2, 3, 4, 5)
    targets[:, 0, 1:3, 1:4] = 1.0
    logits = torch.where(targets > 0.5, _CONFIDENT, -_CONFIDENT)

    out = semantic_aux_loss(logits, targets)

    assert out.bce.item() < 1e-6
    assert abs(out.dice.item()) < 1e-4


def test_confident_wrong_predictions_make_both_terms_large() -> None:
    """Saturated logits opposing their targets push BCE high and Dice to ~1.

    Complements the correct-prediction floor: a term that is near zero at the
    optimum but also flat elsewhere would train nothing.
    """
    targets = torch.zeros(2, 3, 4, 5)
    targets[:, 0, 1:3, 1:4] = 1.0
    logits = torch.where(targets > 0.5, -_CONFIDENT, _CONFIDENT)

    out = semantic_aux_loss(logits, targets)

    assert out.bce.item() > 1.0
    assert out.dice.item() > 0.9


def test_total_is_the_unweighted_sum_of_the_two_terms() -> None:
    """``total`` equals ``bce + dice`` bit for bit, pinning the A36 unit coefficients.

    Catches a gain silently baked into the term (e.g. a 0.5 or 4.0 on Dice),
    which would make the reported components incomparable with the value that is
    actually backpropagated.
    """
    logits = torch.randn(2, 3, 5, 7)
    targets = (torch.rand(2, 3, 5, 7) > 0.5).float()

    out = semantic_aux_loss(logits, targets)

    assert isinstance(out, SemanticAuxOutput)
    assert torch.equal(out.total, out.bce + out.dice)


def test_empty_target_and_empty_prediction_smooth_to_zero_dice() -> None:
    """An all-negative prediction against an all-zero target gives Dice ~0, not NaN.

    This is the A36 smoothing case: with intersection and cardinality both ~0,
    an unsmoothed ratio is 0/0. It is the common case for the many classes absent
    from any given image, so a NaN here would poison nearly every batch.
    """
    logits = torch.full((2, 3, 4, 4), -_CONFIDENT)
    targets = torch.zeros(2, 3, 4, 4)

    out = semantic_aux_loss(logits, targets)

    assert torch.isfinite(out.dice).all()
    assert abs(out.dice.item()) < 1e-4


def test_consumes_semantic_branch_training_output_directly() -> None:
    """SemanticAux's train-mode logits feed the loss and gradients reach its classifier.

    Catches a shape or activation contract drift between the branch and the loss
    — the branch emits raw ``(B, C, H, W)`` logits and this loss applies its own
    sigmoid, so an adapter appearing between them would mean one of the two is
    wrong.
    """
    branch = SemanticAux(6, num_classes=3).train()
    feature = torch.randn(2, 6, 5, 7)
    targets = (torch.rand(2, 3, 5, 7) > 0.5).float()

    logits = branch(feature)
    assert logits is not None
    out = semantic_aux_loss(logits, targets)
    out.total.backward()

    assert out.total.shape == ()
    assert torch.isfinite(out.total).all()
    assert branch.classifier.weight.grad is not None
    assert torch.isfinite(branch.classifier.weight.grad).all()
    assert (branch.classifier.weight.grad != 0.0).any()


def test_dice_is_averaged_per_class_not_pooled_over_the_batch() -> None:
    """A tiny wrong class keeps its full weight beside a large correct one.

    Pins the A36 reduction: Dice is one ratio per ``(B, C)`` pair, averaged
    afterwards. Pooling every element into a single ratio instead would let the
    large, perfectly predicted class absorb the small, entirely wrong one, and
    the term would report roughly 0.33 where the per-pair mean reports 0.47 —
    both finite and plausible, so only the value distinguishes them.
    """
    logits = torch.full((1, 2, 4, 4), _CONFIDENT)  # both classes predicted present
    targets = torch.zeros(1, 2, 4, 4)
    targets[:, 0] = 1.0  # class 0 correct everywhere, class 1 wrong everywhere

    dice = semantic_aux_loss(logits, targets).dice

    correct_pair = 1.0 - (2.0 * 16.0 + 1.0) / (16.0 + 16.0 + 1.0)
    wrong_pair = 1.0 - 1.0 / (16.0 + 1.0)
    expected = torch.tensor((correct_pair + wrong_pair) / 2.0)
    assert torch.allclose(dice, expected, atol=1e-5), "Dice must average per (B, C) pair, not pool the batch"


class TestEmptyBatch:
    """An empty batch returns a defined zero, not the ``nan`` an empty mean gives (M-25).

    Every other loss in the package reduces so that an empty input stays finite —
    ``mask_loss``, ``rle_loss`` and ``keypoint_nll_loss`` all divide by
    ``max(n, 1)``. This one reduced with ``mean`` on both terms and returned ``nan``
    in all three fields. Reachability is low; the inconsistency was the finding.
    """

    #: The shapes an empty batch can take: no images, no classes, an empty grid.
    EMPTY_SHAPES: ClassVar[list[tuple[int, ...]]] = [(0, 3, 4, 4), (2, 0, 4, 4), (2, 3, 0, 4), (2, 3, 4, 0)]

    @pytest.mark.parametrize("shape", EMPTY_SHAPES)
    def test_every_term_is_an_exact_finite_zero(self, shape: tuple[int, ...]) -> None:
        """All three fields are exactly zero and finite, whichever dimension is empty."""
        out = semantic_aux_loss(torch.zeros(*shape), torch.zeros(*shape))

        for name, term in (("total", out.total), ("bce", out.bce), ("dice", out.dice)):
            assert bool(torch.isfinite(term)), f"{name} is not finite"
            assert float(term) == 0.0, f"{name} is not exactly zero"

    def test_the_zero_still_carries_gradient_to_the_logits(self) -> None:
        """The zero is a function of ``logits``, so the composition site needs no branch."""
        logits = torch.zeros(0, 3, 4, 4, requires_grad=True)

        semantic_aux_loss(logits, torch.zeros(0, 3, 4, 4)).total.backward()

        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_the_terms_stay_scalar(self) -> None:
        """Shape contract is unchanged: zero-dimensional tensors, as for a full batch."""
        out = semantic_aux_loss(torch.zeros(0, 3, 4, 4), torch.zeros(0, 3, 4, 4))

        assert out.total.ndim == 0
        assert out.bce.ndim == 0
        assert out.dice.ndim == 0

    def test_a_non_empty_batch_is_untouched_by_the_guard(self) -> None:
        """The guard is an early return, so every non-empty reduction is bit-unchanged.

        Pins the reason the fix is a branch rather than a swap of the two divisors:
        ``reduction="mean"`` and ``.mean()`` still reduce every real batch, so no
        value produced on any device can move.
        """
        logits = torch.randn(2, 3, 4, 5)
        targets = (torch.rand(2, 3, 4, 5) > 0.5).float()

        out = semantic_aux_loss(logits, targets)

        probabilities = logits.sigmoid()
        intersection = (probabilities * targets).sum(dim=(-2, -1))
        cardinality = probabilities.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        expected_dice = (1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()
        expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

        assert torch.equal(out.bce, expected_bce)
        assert torch.equal(out.dice, expected_dice)
