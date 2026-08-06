# SPDX-License-Identifier: Apache-2.0
"""Auxiliary semantic-segmentation loss for the training-only branch (WP-051).

Consumes the raw per-class logits :meth:`SemanticAux.forward
<lucid_yolo.models.heads.semantic.SemanticAux.forward>` returns in training mode
(A17) and supervises them with two terms of equal weight:

- ``bce`` — mean binary-cross-entropy-with-logits over every ``(B, C, H, W)``
  element. This is the per-pixel term; it is well behaved on the overwhelmingly
  background maps the branch produces, which is exactly why the classifier
  carries the A30 prior-probability bias init.
- ``dice`` — soft Dice on the sigmoid probabilities, ``1 - (2|p n t| + s) /
  (|p| + |t| + s)``. Dice is the region term: it scores overlap rather than
  per-pixel agreement, so a class occupying a handful of pixels still produces a
  gradient the dense BCE would drown out.

Two choices the papers leave open are recorded as A36. First, Dice is computed
per ``(B, C)`` pair — contracting the spatial dimensions only — and then averaged
over the batch and the classes. Pooling the whole batch into one ratio instead
would let a single large class dominate; the per-pair form gives every class in
every image the same say. Second, the smoothing constant ``s = 1.0`` appears in
both numerator and denominator, so an empty prediction against an empty target
scores ``1 - 1/1 = 0`` — exactly zero loss — rather than ``0 / 0``.

``total`` is ``bce + dice``: "equal weight" is read as unit coefficients on both
terms. Any overall gain on the auxiliary objective belongs at the composition
site that adds this loss to the detection terms, not baked in here, so that the
reported components stay comparable across gain settings — the same pre-gain
component convention
:class:`~lucid_yolo.losses.detection_loss.DetectionLossOutput` uses.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F

__all__ = ["SemanticAuxOutput", "semantic_aux_loss"]

#: Dice smoothing constant, in both numerator and denominator (A36).
_DICE_SMOOTH: float = 1.0


@dataclass(frozen=True)
class SemanticAuxOutput:
    """Auxiliary semantic loss terms for one batch.

    ``bce`` and ``dice`` are the two pre-combination terms and ``total`` is their
    plain sum, the unit-coefficient reading of "equal weight" (A36). Every field
    is a scalar (zero-dimensional) tensor carrying gradients back to the logits.

    Attributes:
        total: ``bce + dice``; the value to backpropagate.
        bce: Mean binary-cross-entropy-with-logits term.
        dice: Soft-Dice term on sigmoid probabilities.

    Examples:
        >>> import torch
        >>> out = SemanticAuxOutput(total=torch.tensor(1.5), bce=torch.tensor(1.0), dice=torch.tensor(0.5))
        >>> float(out.total)
        1.5
    """

    total: Tensor
    bce: Tensor
    dice: Tensor


def semantic_aux_loss(logits: Tensor, targets: Tensor) -> SemanticAuxOutput:
    """Equal-weight BCE + soft-Dice supervision for the auxiliary semantic branch.

    Args:
        logits: Raw per-class logits ``(B, C, H, W)``, i.e. exactly what
            :meth:`SemanticAux.forward
            <lucid_yolo.models.heads.semantic.SemanticAux.forward>` returns in
            training mode. The sigmoid is applied here, so the branch must not
            activate them itself.
        targets: Binary per-class targets ``(B, C, H, W)`` as floats in
            ``{0.0, 1.0}``, on the same grid as ``logits``.

    Returns:
        A :class:`SemanticAuxOutput` holding the two terms and their sum.

    Examples:
        >>> import torch
        >>> logits = torch.full((1, 1, 2, 2), 20.0)  # confidently positive
        >>> targets = torch.ones(1, 1, 2, 2)
        >>> out = semantic_aux_loss(logits, targets)
        >>> bool(out.bce.abs() < 1e-6) and bool(out.dice.abs() < 1e-6)
        True
        >>> bool(out.total.equal(out.bce + out.dice))
        True
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

    probabilities = logits.sigmoid()
    # Contract the spatial dims only: one Dice ratio per (B, C) pair, then mean.
    intersection = (probabilities * targets).sum(dim=(-2, -1))  # (B, C)
    cardinality = probabilities.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))  # (B, C)
    dice = (1.0 - (2.0 * intersection + _DICE_SMOOTH) / (cardinality + _DICE_SMOOTH)).mean()

    return SemanticAuxOutput(total=bce + dice, bce=bce, dice=dice)
