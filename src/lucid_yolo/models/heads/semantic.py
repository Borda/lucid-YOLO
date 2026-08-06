# SPDX-License-Identifier: Apache-2.0
"""Training-only auxiliary semantic branch over ``F_proto`` (WP-050).

The branch attaches a single 1x1 classifier to the Eq. 8 fused feature and emits
one raw logit map per class at that feature's own resolution. It exists purely to
shape the shared prototype features during training: :meth:`SemanticAux.forward`
returns ``None`` outside training mode, so an evaluated or exported model
provably carries no semantic computation and the fused-parameter gate has exactly
one convolution to account for. The BCE+Dice supervision lands in WP-051 and
consumes these logits directly.

Provenance: R1 sec. 3.4.1 ("training-only branch"). Assumptions: A17, A30.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from lucid_yolo.models.heads.detect import init_cls_prior_bias

__all__ = ["SemanticAux"]


class SemanticAux(nn.Module):
    """Auxiliary per-class semantic logits from the fused prototype feature.

    A17's "lightweight conv" is read literally: exactly one 1x1
    :class:`~torch.nn.Conv2d` from the fused-feature width to ``num_classes``,
    with no hidden width, normalization, or activation. Keeping it to a single
    convolution is what lets the WP-052 fused-parameter gate state precisely
    which parameters disappear when the branch is dropped.

    The output convolution carries the A30 prior-probability bias init
    (:func:`~lucid_yolo.models.heads.detect.init_cls_prior_bias`) because this is
    exactly the setting A30 was written for: a dense sigmoid classifier over a
    map that is overwhelmingly background, whose per-pixel BCE would otherwise
    open orders of magnitude too large. That is the opposite of the coefficient
    stems (WP-047), which deliberately omit the init because tanh regressands
    have no prior-probability interpretation.

    Logits are returned raw. The WP-051 loss uses a with-logits BCE plus a Dice
    term and applies its own sigmoid, so an activation here would double-squash
    the objective.

    Args:
        in_channels: Channel count of the fused feature, i.e.
            :attr:`~lucid_yolo.models.heads.proto.ProtoFusion.out_channels`.
        num_classes: Number of object classes, one logit map each.

    Attributes:
        num_classes: Number of emitted class logit maps.
        classifier: The single 1x1 output convolution.

    Examples:
        >>> import torch
        >>> semantic = SemanticAux(64, num_classes=80)
        >>> semantic.train()(torch.zeros(1, 64, 80, 80)).shape
        torch.Size([1, 80, 80, 80])
        >>> semantic.eval()(torch.zeros(1, 64, 80, 80)) is None
        True
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        """Initialize the single 1x1 classifier with the A30 prior-probability bias.

        Args:
            in_channels: Channel count of the fused prototype feature.
            num_classes: Number of object classes.
        """
        super().__init__()
        self.num_classes: int = num_classes
        self.classifier = nn.Conv2d(in_channels, num_classes, 1)
        init_cls_prior_bias(self.classifier)

    def forward(self, feature: Tensor) -> Tensor | None:
        """Return raw per-class semantic logits, or ``None`` outside training.

        The eval-mode ``None`` is the branch's removal proof: it is tied to
        :attr:`~torch.nn.Module.training` alone, never to gradient mode, so
        inference under :func:`torch.no_grad` and export both skip the
        convolution entirely.

        Args:
            feature: Fused prototype feature with shape ``(B, C, H, W)``.

        Returns:
            Raw logits of shape ``(B, num_classes, H, W)`` at the input's own
            spatial size in training mode, otherwise ``None``.
        """
        if not self.training:
            return None
        return cast(Tensor, self.classifier(feature))
