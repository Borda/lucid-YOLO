# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-020 detection backbone (test_backbone.py).

Covers :class:`lit_yolo.models.DetectionBackbone`: that the three taps land at
strides 8/16/32 with width-scaled channel counts for the ``n``/``s`` multipliers,
that the ``m`` multiplier's ``max_channels`` cap clamps the P5 tap, that inner-unit
repeat counts track the depth multiplier, and that a forward pass on a small input
runs clean in eval mode.
"""

from __future__ import annotations

import pytest
import torch

from lit_yolo.models import DetectionBackbone

# (depth, width, max_channels, expected tap channels (P3, P4, P5))
_N_SCALE = (0.50, 0.25, 1024, (128, 128, 256))
_S_SCALE = (0.50, 0.50, 1024, (256, 256, 512))
_M_SCALE = (0.50, 1.00, 512, (512, 512, 512))


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed torch RNG so weight init and random inputs are deterministic."""
    torch.manual_seed(0)


@pytest.mark.parametrize(
    ("depth", "width", "max_channels", "expected_channels"),
    [
        pytest.param(*_N_SCALE, id="n-scale"),
        pytest.param(*_S_SCALE, id="s-scale"),
    ],
)
def test_tap_shapes(depth: float, width: float, max_channels: int, expected_channels: tuple[int, int, int]) -> None:
    """P3/P4/P5 taps land at strides 8/16/32 with the width-scaled channel counts."""
    backbone = DetectionBackbone(depth=depth, width=width, max_channels=max_channels).eval()
    x = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        p3, p4, p5 = backbone(x)

    assert backbone.channels == expected_channels, "channels property must match tap widths"
    ch3, ch4, ch5 = expected_channels
    assert p3.shape == (1, ch3, 80, 80), "P3 tap must be stride 8 with width-scaled channels"
    assert p4.shape == (1, ch4, 40, 40), "P4 tap must be stride 16 with width-scaled channels"
    assert p5.shape == (1, ch5, 20, 20), "P5 tap must be stride 32 with width-scaled channels"


def test_max_channels_clamp() -> None:
    """The m-scale (mc=512) clamps the P5 tap to 512, not 1024 * width."""
    depth, width, max_channels, expected_channels = _M_SCALE
    backbone = DetectionBackbone(depth=depth, width=width, max_channels=max_channels).eval()

    with torch.no_grad():
        _, _, p5 = backbone(torch.randn(1, 3, 640, 640))

    assert backbone.channels == expected_channels, "P5 must clamp to mc, not 1024 * width"
    assert p5.shape == (1, 512, 20, 20), "clamped P5 tap keeps stride 32 at 512 channels"


@pytest.mark.parametrize(
    ("depth", "expected_repeats"),
    [
        pytest.param(0.50, 1, id="d0.50-one-unit"),
        pytest.param(1.00, 2, id="d1.00-two-units"),
    ],
)
def test_repeats_scale_with_depth(depth: float, expected_repeats: int) -> None:
    """Every CSP stage repeats max(1, round(2 * depth)) inner units."""
    backbone = DetectionBackbone(depth=depth, width=0.25, max_channels=1024)

    for stage in (backbone.stage_p2, backbone.stage_p3, backbone.stage_p4, backbone.stage_p5):
        assert len(stage.blocks) == expected_repeats, "C3k2 inner-unit count must track depth"
    assert len(backbone.attn_p5.blocks) == expected_repeats, "C2PSA PSABlock count must track depth"


def test_forward_small_input_clean() -> None:
    """A stride-32-divisible small input flows through cleanly in eval mode."""
    backbone = DetectionBackbone(depth=0.50, width=0.25, max_channels=1024).eval()

    with torch.no_grad():
        p3, p4, p5 = backbone(torch.randn(1, 3, 128, 128))

    assert p3.shape == (1, 128, 16, 16), "P3 tap must be input / 8"
    assert p4.shape == (1, 128, 8, 8), "P4 tap must be input / 16"
    assert p5.shape == (1, 256, 4, 4), "P5 tap must be input / 32"
