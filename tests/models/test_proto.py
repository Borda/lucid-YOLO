# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-048 multi-scale prototype-feature fusion.

Covers the literal Eq. 8 sum, including the unprojected ``X_1`` identity path,
target-size nearest-neighbour upsampling, and independent coarse-level
projections. Prototype generation, mask assembly, and losses are intentionally
outside this module's scope.
"""

from __future__ import annotations

import pytest
import torch

from lucid_yolo.models import ProtoFusion

#: Small, unequal widths prove each coarse level projects into the P3 width.
_CHANNELS = (2, 3, 5)


def _features(
    spatial_sizes: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    batch: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic stride-ordered feature maps with the requested sizes."""
    return tuple(
        torch.randn(batch, channels, height, width)
        for channels, (height, width) in zip(_CHANNELS, spatial_sizes, strict=True)
    )


def test_fusion_eq8() -> None:
    """Eq. 8 adds each projected coarse-level bias to the unprojected P3 tensor."""
    torch.manual_seed(0)
    fusion = ProtoFusion(_CHANNELS).eval()
    x1, x2, x3 = _features(((12, 20), (6, 10), (3, 5)))
    biases = (torch.tensor((0.25, -0.5)), torch.tensor((1.0, 0.75)))

    for projection, bias in zip(fusion.projections, biases, strict=True):
        projection.weight.data.zero_()
        assert projection.bias is not None
        projection.bias.data.copy_(bias)

    with torch.no_grad():
        fused = fusion((x1, x2, x3))

    expected = x1 + biases[0].view(1, -1, 1, 1) + biases[1].view(1, -1, 1, 1)
    assert torch.allclose(fused, expected), "fusion must implement X1 + U(phi4(X2)) + U(phi5(X3))"


def test_fusion_identity_when_projections_vanish() -> None:
    """Zero coarse projections leave the finest feature map exactly unchanged."""
    torch.manual_seed(0)
    fusion = ProtoFusion(_CHANNELS).eval()
    features = _features(((12, 20), (6, 10), (3, 5)))
    for projection in fusion.projections:
        projection.weight.data.zero_()
        assert projection.bias is not None
        projection.bias.data.zero_()

    with torch.no_grad():
        fused = fusion(features)

    assert torch.equal(fused, features[0]), "X1 must be added directly, without a level-one projection"


def test_fusion_output_matches_x1_shape_and_channel_contract() -> None:
    """The fused map preserves P3's batch, channel, and spatial dimensions."""
    fusion = ProtoFusion(_CHANNELS).eval()
    features = _features(((12, 20), (6, 10), (3, 5)), batch=1)

    with torch.no_grad():
        fused = fusion(features)

    assert fusion.out_channels == features[0].shape[1]
    assert fused.shape == features[0].shape


@pytest.mark.parametrize(
    "spatial_sizes",
    [
        ((12, 20), (6, 10), (3, 5)),  # 96x160 input: non-square feature maps.
        ((13, 21), (7, 11), (4, 6)),  # Odd P3 size: fixed scale factors cannot align.
    ],
)
def test_fusion_uses_x1_target_size(
    spatial_sizes: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> None:
    """Nearest interpolation targets X1's exact size for non-square and odd maps."""
    fusion = ProtoFusion(_CHANNELS).eval()
    features = _features(spatial_sizes)

    with torch.no_grad():
        fused = fusion(features)

    assert fused.shape[-2:] == features[0].shape[-2:]


def test_coarse_level_projections_are_disjoint() -> None:
    """P4 and P5 own independent learnable 1x1 projection parameters."""
    fusion = ProtoFusion(_CHANNELS)
    p4_parameter_ids = {id(parameter) for parameter in fusion.projections[0].parameters()}
    p5_parameter_ids = {id(parameter) for parameter in fusion.projections[1].parameters()}

    assert p4_parameter_ids and p5_parameter_ids
    assert p4_parameter_ids.isdisjoint(p5_parameter_ids)
