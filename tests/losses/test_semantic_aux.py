# SPDX-License-Identifier: Apache-2.0
"""Unit gates for the WP-051 auxiliary semantic BCE+Dice loss (A17, A36).

Covers the two terms at both confidence extremes, the unit-coefficient reading of
"equal weight", the empty-prediction-versus-empty-target smoothing case, and the
end-to-end contract with :class:`~lucid_yolo.models.heads.semantic.SemanticAux`'s
training-mode output.
"""

from __future__ import annotations

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
