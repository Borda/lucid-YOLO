# SPDX-License-Identifier: Apache-2.0
"""Unit gates for WP-048-051 prototype fusion, generation, and assembly.

Covers the literal Eq. 8 sum, including the unprojected ``X_1`` identity path,
target-size nearest-neighbour upsampling, and independent coarse-level
projections. Also covers the Eq. 9 raw prototype stack and the Eq. 7 linear mask
assembly; the mask losses themselves remain outside this module's scope.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from lucid_yolo.models import ProtoFusion, ProtoNet, assemble_masks
from lucid_yolo.models.blocks import ConvBNAct
from lucid_yolo.models.heads.detect import DEFAULT_NUM_COEFFS

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


@pytest.mark.parametrize(
    ("p3_size", "prototype_size"),
    [
        ((80, 80), (160, 160)),  # 640-pixel input: P3 at stride 8 (A15).
        ((13, 21), (26, 42)),  # Odd, non-square P3 still doubles structurally.
    ],
)
def test_proto_resolution(
    p3_size: tuple[int, int],
    prototype_size: tuple[int, int],
) -> None:
    """Proto maps double each P3 dimension without a hard-coded input resolution."""
    prototypes = ProtoNet(_CHANNELS[0]).eval()
    feature = torch.randn(1, _CHANNELS[0], *p3_size)

    with torch.no_grad():
        output = prototypes(feature)

    assert output.shape == (1, DEFAULT_NUM_COEFFS, *prototype_size)
    assert output.shape[-2] == 2 * feature.shape[-2]
    assert output.shape[-1] == 2 * feature.shape[-1]


def test_proto_count_matches_coefficient_width() -> None:
    """Prototype count defaults to K and respects an explicit coefficient width."""
    default = ProtoNet(_CHANNELS[0])
    custom = ProtoNet(_CHANNELS[0], num_prototypes=8).eval()

    with torch.no_grad():
        output = custom(torch.randn(1, _CHANNELS[0], 5, 7))

    assert default.num_prototypes == DEFAULT_NUM_COEFFS
    assert output.shape[1] == 8


def test_proto_output_is_unactivated() -> None:
    """Raw prototype logits retain values outside the range of tanh or sigmoid."""
    prototypes = ProtoNet(_CHANNELS[0], num_prototypes=1).eval()
    output_conv = prototypes.layers[-1]
    assert isinstance(output_conv, nn.Conv2d)
    output_conv.weight.data.zero_()
    assert output_conv.bias is not None
    output_conv.bias.data.fill_(5.0)

    with torch.no_grad():
        output = prototypes(torch.randn(1, _CHANNELS[0], 4, 6))

    assert torch.equal(output, torch.full_like(output, 5.0))


def test_proto_upsample_precedes_final_spatial_conv() -> None:
    """The fourth 3x3 unit runs after nearest upsampling at the proto grid."""
    prototypes = ProtoNet(_CHANNELS[0])

    assert len(prototypes.layers) == 6
    assert isinstance(prototypes.layers[3], nn.Upsample)
    assert isinstance(prototypes.layers[4], ConvBNAct)
    assert isinstance(prototypes.layers[5], nn.Conv2d)


def test_proto_grid_is_refined_after_upsampling() -> None:
    """A spatial unit runs at proto resolution, so output is not a bare 2x upsample.

    Complements the structural order check with an observable consequence: if the
    upsample were the last spatial operation, every 2x2 output block would be
    constant, because nearest interpolation replicates each source pixel. A 3x3
    convolution at the proto grid breaks that replication.
    """
    torch.manual_seed(0)
    prototypes = ProtoNet(_CHANNELS[0], num_prototypes=4).eval()

    with torch.no_grad():
        output = prototypes(torch.randn(1, _CHANNELS[0], 8, 10))

    assert not torch.equal(output[..., 0::2, :], output[..., 1::2, :]), "rows within a 2x2 block are replicated"
    assert not torch.equal(output[..., 0::2], output[..., 1::2]), "columns within a 2x2 block are replicated"


def test_fusion_and_protonet_run_from_neck_feature_triple() -> None:
    """The Eq. 8 output feeds Eq. 9 directly at the expected proto resolution."""
    channels = (4, 8, 16)
    fusion = ProtoFusion(channels).eval()
    prototypes = ProtoNet(fusion.out_channels).eval()
    features = tuple(torch.randn(1, channels[index], 80 // (2**index), 80 // (2**index)) for index in range(3))

    with torch.no_grad():
        output = prototypes(fusion(features))

    assert output.shape == (1, DEFAULT_NUM_COEFFS, 160, 160)


def test_assemble_masks_shape_contract() -> None:
    """(B, N, K) coefficients against (B, K, H, W) prototypes give (B, N, H, W).

    Catches a transposed einsum subscript: several wrong contractions still
    produce a well-formed tensor, and only an N != K != H != W shape separates
    them from the intended one.
    """
    prototypes = torch.randn(2, 5, 7, 11)
    coefficients = torch.randn(2, 3, 5)

    masks = assemble_masks(prototypes, coefficients)

    assert masks.shape == (2, 3, 7, 11)


def test_assemble_masks_matches_explicit_weighted_sum() -> None:
    """Each output map equals the hand-built sum of c_ik * P_k over all K prototypes.

    Pins Eq. 7 as a plain linear combination: a stray activation, normalization,
    or crop would break the exact per-instance sum this rebuilds by hand.
    """
    prototypes = torch.randn(1, 4, 3, 5)
    coefficients = torch.randn(1, 2, 4)

    masks = assemble_masks(prototypes, coefficients)

    for instance in range(coefficients.shape[1]):
        expected = sum(coefficients[0, instance, k] * prototypes[0, k] for k in range(prototypes.shape[1]))
        assert torch.allclose(masks[0, instance], expected, atol=1e-6)


def test_assemble_masks_one_hot_selects_a_single_prototype() -> None:
    """A one-hot coefficient vector reproduces exactly the prototype it selects.

    Catches a contraction that averages over K or that indexes the wrong axis:
    both keep the output shape but neither returns the selected map untouched.
    """
    prototypes = torch.randn(1, 4, 3, 5)
    coefficients = torch.zeros(1, 1, 4)
    coefficients[0, 0, 2] = 1.0

    masks = assemble_masks(prototypes, coefficients)

    assert torch.allclose(masks[0, 0], prototypes[0, 2])
