# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-021 detection neck (test_neck.py).

Covers :class:`lit_yolo.models.DetectionNeck`: that the three outputs preserve the
stride-8/16/32 resolution of the backbone taps with width-scaled channel counts for
the ``n``/``s`` multipliers, that the attention tail carries exactly one PSABlock
inner unit regardless of the depth multiplier, that a backbone+neck pass on a small
input runs clean in eval mode, and that the fuse convolutions account for the
concatenated channel counts.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from lit_yolo.models import DetectionBackbone, DetectionNeck, PSABlock

# (depth, width, max_channels, backbone taps (P3, P4, P5), neck outputs (N3, N4, N5))
_N_SCALE = (0.50, 0.25, 1024, (128, 128, 256), (64, 128, 256))
_S_SCALE = (0.50, 0.50, 1024, (256, 256, 512), (128, 256, 512))


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed torch RNG so weight init and random inputs are deterministic."""
    torch.manual_seed(0)


@pytest.mark.parametrize(
    ("depth", "width", "max_channels", "in_channels", "expected_channels"),
    [
        pytest.param(*_N_SCALE, id="n-scale"),
        pytest.param(*_S_SCALE, id="s-scale"),
    ],
)
def test_output_shapes(
    depth: float,
    width: float,
    max_channels: int,
    in_channels: tuple[int, int, int],
    expected_channels: tuple[int, int, int],
) -> None:
    """The N3/N4/N5 outputs keep strides 8/16/32 with width-scaled channel counts."""
    neck = DetectionNeck(in_channels=in_channels, depth=depth, width=width, max_channels=max_channels).eval()
    in_p3, in_p4, in_p5 = in_channels
    features = (
        torch.randn(1, in_p3, 80, 80),
        torch.randn(1, in_p4, 40, 40),
        torch.randn(1, in_p5, 20, 20),
    )

    with torch.no_grad():
        n3, n4, n5 = neck(features)

    assert neck.channels == expected_channels, "channels property must match output widths"
    ch3, ch4, ch5 = expected_channels
    assert n3.shape == (1, ch3, 80, 80), "N3 output must stay stride 8 with width-scaled channels"
    assert n4.shape == (1, ch4, 40, 40), "N4 output must stay stride 16 with width-scaled channels"
    assert n5.shape == (1, ch5, 20, 20), "N5 output must stay stride 32 with width-scaled channels"


@pytest.mark.parametrize(
    "depth",
    [
        pytest.param(0.50, id="d0.50"),
        pytest.param(1.00, id="d1.00"),
    ],
)
def test_attention_tail_fixed_single_unit(depth: float) -> None:
    """The final stage carries exactly one PSABlock inner unit, independent of depth."""
    neck = DetectionNeck(in_channels=(256, 256, 512), depth=depth, width=0.5, max_channels=1024)

    tail = neck.bottom_up_p5
    assert len(tail.blocks) == 1, "attention-tail C3k2 is fixed at n=1, never depth-scaled"
    assert any(isinstance(m, PSABlock) for m in tail.modules()), "final stage must contain a PSABlock"


def test_depth_scales_non_attention_stages() -> None:
    """The three x2 stages track the depth multiplier while the tail stays at n=1."""
    neck = DetectionNeck(in_channels=(256, 256, 512), depth=1.00, width=0.5, max_channels=1024)

    for stage in (neck.top_down_p4, neck.top_down_p3, neck.bottom_up_p4):
        assert len(stage.blocks) == 2, "x2 neck stages repeat max(1, round(2 * depth)) inner units"
    assert len(neck.bottom_up_p5.blocks) == 1, "attention tail stays fixed at one inner unit"


def test_backbone_neck_forward_small_input_clean() -> None:
    """A stride-32-divisible small input flows through backbone then neck cleanly."""
    backbone = DetectionBackbone(depth=0.50, width=0.25, max_channels=1024).eval()
    neck = DetectionNeck(in_channels=backbone.channels, depth=0.50, width=0.25, max_channels=1024).eval()

    with torch.no_grad():
        n3, n4, n5 = neck(backbone(torch.randn(1, 3, 128, 128)))

    assert n3.shape == (1, 64, 16, 16), "N3 output must be input / 8"
    assert n4.shape == (1, 128, 8, 8), "N4 output must be input / 16"
    assert n5.shape == (1, 256, 4, 4), "N5 output must be input / 32"


def test_concat_channel_accounting() -> None:
    """Each fuse conv's input width equals the concatenated feed for the s-scale neck."""
    # s-scale: in=(256, 256, 512); scaled widths T4=256, N3=128, N4=256, N5=512.
    neck = DetectionNeck(in_channels=(256, 256, 512), depth=0.50, width=0.5, max_channels=1024)

    def fuse_in(stage: nn.Module) -> int:
        return stage.cv1.conv.in_channels  # type: ignore[no-any-return, union-attr]

    assert fuse_in(neck.top_down_p4) == 512 + 256, "top-down P4 fuses upsampled P5 with P4"
    assert fuse_in(neck.top_down_p3) == 256 + 256, "top-down P3 fuses upsampled T4 with P3"
    assert fuse_in(neck.bottom_up_p4) == 128 + 256, "bottom-up P4 fuses downsampled N3 with T4"
    assert fuse_in(neck.bottom_up_p5) == 256 + 512, "bottom-up P5 fuses downsampled N4 with P5"
