# SPDX-License-Identifier: Apache-2.0
"""Unit gates for the WP-050 training-only auxiliary semantic branch.

Pins the three properties the branch is defined by: it is provably absent at
eval, it is exactly one 1x1 convolution (A17), and it emits raw prior-initialized
logits at the fused feature's own resolution (A30). The BCE+Dice supervision and
any model wiring remain outside this module's scope.
"""

from __future__ import annotations

import torch

from lucid_yolo.models import ProtoFusion, SemanticAux
from lucid_yolo.models.heads.detect import CLS_PRIOR_PROB

#: Small fused-feature width; unequal to the class count so shapes cannot alias.
_IN_CHANNELS = 6

#: Class count kept distinct from every spatial size used below.
_NUM_CLASSES = 3


def test_eval_mode_inactive() -> None:
    """Eval mode returns None and train mode returns a tensor, independent of grad mode.

    Catches an inactivity check accidentally keyed on ``torch.no_grad`` rather
    than ``Module.training``: that would leave the branch alive under a
    gradient-enabled eval pass, so the "training-only" removal claim would be
    false exactly where it matters for export.
    """
    semantic = SemanticAux(_IN_CHANNELS, _NUM_CLASSES)
    feature = torch.randn(2, _IN_CHANNELS, 5, 7)

    semantic.eval()
    eval_with_grad = semantic(feature)
    with torch.no_grad():
        eval_without_grad = semantic(feature)
    semantic.train()
    train_output = semantic(feature)

    assert eval_with_grad is None, "eval mode must skip the branch even with grad enabled"
    assert eval_without_grad is None, "eval mode must skip the branch under no_grad"
    assert isinstance(train_output, torch.Tensor)


def test_output_keeps_input_resolution() -> None:
    """Logits keep the odd, non-square input size, proving no resampling happens.

    A fixed scale factor or a hard-coded proto grid would silently rescale the
    map and misalign it with the WP-051 semantic target.
    """
    semantic = SemanticAux(_IN_CHANNELS, _NUM_CLASSES)
    feature = torch.randn(2, _IN_CHANNELS, 13, 21)

    output = semantic(feature)

    assert output is not None
    assert output.shape == (2, _NUM_CLASSES, 13, 21)


def test_prior_probability_bias_init() -> None:
    """A zero input yields the bias alone, whose sigmoid must equal the A30 prior.

    Catches a dropped or diverging prior init: default zero bias would open the
    dense background BCE at ~0.69 nats per element instead of ~0.01.
    """
    semantic = SemanticAux(_IN_CHANNELS, _NUM_CLASSES)

    output = semantic(torch.zeros(1, _IN_CHANNELS, 4, 6))

    assert output is not None
    assert torch.allclose(output.sigmoid(), torch.full_like(output, CLS_PRIOR_PROB), atol=1e-6)


def test_output_is_raw_logits() -> None:
    """A zeroed weight and a bias of 5.0 must surface as 5.0, not a squashed value.

    Catches an activation slipped in ahead of the WP-051 with-logits loss, which
    would double-squash the objective.
    """
    semantic = SemanticAux(_IN_CHANNELS, _NUM_CLASSES)
    semantic.classifier.weight.data.zero_()
    assert semantic.classifier.bias is not None
    semantic.classifier.bias.data.fill_(5.0)

    output = semantic(torch.randn(1, _IN_CHANNELS, 4, 6))

    assert output is not None
    assert torch.equal(output, torch.full_like(output, 5.0))


def test_parameter_count_is_a_single_pointwise_convolution() -> None:
    """Parameters equal one 1x1 conv's weights plus biases, pinning the A17 reading.

    A heavier stack (ConvBNAct units, a hidden width, normalization) would change
    what the WP-052 fused-parameter gate has to remove, so the minimal reading is
    pinned numerically rather than left to review.
    """
    semantic = SemanticAux(_IN_CHANNELS, _NUM_CLASSES)

    total = sum(parameter.numel() for parameter in semantic.parameters())

    assert total == _IN_CHANNELS * _NUM_CLASSES + _NUM_CLASSES


def test_consumes_fusion_output_directly() -> None:
    """ProtoFusion's fused feature feeds the branch with no adapter in between.

    Catches a channel-contract drift between ``ProtoFusion.out_channels`` and the
    branch's expected input width.
    """
    channels = (4, 8, 16)
    fusion = ProtoFusion(channels).eval()
    semantic = SemanticAux(fusion.out_channels, _NUM_CLASSES)
    features = tuple(torch.randn(1, channels[index], 80 // (2**index), 80 // (2**index)) for index in range(3))

    output = semantic(fusion(features))

    assert output is not None
    assert output.shape == (1, _NUM_CLASSES, 80, 80)
