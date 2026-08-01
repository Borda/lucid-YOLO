# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-016 convolutional primitives (test_blocks.py).

Covers the three blocks in ``lit_yolo.models.blocks``: stride-1/stride-2 spatial
behaviour for odd kernels, the depthwise group count, the bottleneck channel
expansion ratio, the residual add, and a clean eval-mode forward pass.
"""

from __future__ import annotations

import pytest
import torch

from lit_yolo.models import Bottleneck, C3k, C3k2, ConvBNAct, DepthwiseConv


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed torch RNG so weight init and random inputs are deterministic."""
    torch.manual_seed(0)


def test_primitives() -> None:
    """The three primitives satisfy their shape, wiring, and residual contracts."""
    # --- ConvBNAct: odd kernels preserve size at stride 1, halve at stride 2 ---
    x = torch.randn(2, 8, 16, 16)
    for kernel_size in (3, 5):
        same = ConvBNAct(8, 12, kernel_size, stride=1).eval()
        down = ConvBNAct(8, 12, kernel_size, stride=2).eval()
        with torch.no_grad():
            same_out = same(x)
            down_out = down(x)
        assert same_out.shape == (2, 12, 16, 16), f"stride-1 k={kernel_size} spatial mismatch"
        assert down_out.shape == (2, 12, 8, 8), f"stride-2 k={kernel_size} spatial mismatch"

    # --- DepthwiseConv: one group per input channel ---
    dw = DepthwiseConv(8, 8, kernel_size=5).eval()
    assert dw.conv.conv.groups == 8, "depthwise conv must use groups == in_channels"
    with torch.no_grad():
        dw_out = dw(x)
    assert dw_out.shape == (2, 8, 16, 16)

    # --- Bottleneck: expansion ratio honoured in the hidden conv ---
    block = Bottleneck(8, 8, shortcut=True, expansion=0.25).eval()
    assert block.cv1.conv.out_channels == int(8 * 0.25), "hidden channels must be int(out_ch * e)"

    # --- Bottleneck: residual actually adds when shortcut is active ---
    z = torch.randn(2, 8, 8, 8)
    assert block.add_shortcut is True
    with torch.no_grad():
        feed_forward = block.cv2(block.cv1(z))  # the no-shortcut path
        actual = block(z)
    assert torch.allclose(actual, feed_forward + z), "active shortcut must add the identity"
    assert not torch.allclose(actual, feed_forward), "output must differ from the no-shortcut path"

    # --- Bottleneck: no residual when channels differ, even with shortcut=True ---
    no_add = Bottleneck(8, 16, shortcut=True).eval()
    assert no_add.add_shortcut is False, "shortcut requires matching in/out channels"
    w = torch.randn(2, 8, 8, 8)
    with torch.no_grad():
        assert torch.allclose(no_add(w), no_add.cv2(no_add.cv1(w)))


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "n", "expansion"),
    [
        pytest.param(64, 64, 1, 0.5, id="square-n1"),
        pytest.param(32, 128, 2, 0.5, id="widen-n2"),
        pytest.param(64, 64, 3, 0.25, id="narrow-e025"),
    ],
)
def test_c3k_shapes(in_channels: int, out_channels: int, n: int, expansion: float) -> None:
    """C3k preserves spatial size, honours the channel contract, and stacks n bottlenecks."""
    block = C3k(in_channels, out_channels, n=n, expansion=expansion).eval()
    hidden = int(out_channels * expansion)
    # --- split streams and pre-fuse concat width both follow the expansion ratio ---
    assert block.cv1.conv.out_channels == hidden, "cv1 must squeeze to int(out_ch * e)"
    assert block.cv3.conv.in_channels == 2 * hidden, "cv3 fuses two hidden-channel streams"
    assert len(block.blocks) == n, "refined stream must hold n inner bottlenecks"
    # --- inner bottlenecks keep full hidden width (expansion 1.0 inside C3k) ---
    assert block.blocks[0].cv1.conv.out_channels == hidden, "inner bottleneck uses expansion 1.0"
    x = torch.randn(2, in_channels, 8, 8)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, out_channels, 8, 8), "C3k must preserve spatial resolution"


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "n", "e", "c3k"),
    [
        pytest.param(64, 64, 1, 0.5, False, id="bottleneck-n1"),
        pytest.param(32, 256, 2, 0.25, False, id="bottleneck-e025-n2"),
        pytest.param(64, 128, 2, 0.5, True, id="c3k-inner-n2"),
        pytest.param(48, 96, 3, 0.5, True, id="c3k-inner-n3"),
    ],
)
def test_c3k2_shapes(in_channels: int, out_channels: int, n: int, e: float, c3k: bool) -> None:
    """C3k2 preserves spatial size and accumulates (2 + n) * hidden channels before fusion."""
    block = C3k2(in_channels, out_channels, n=n, e=e, c3k=c3k).eval()
    hidden = int(out_channels * e)
    # --- cv1 lifts to two halves; dense chaining accumulates (2 + n) * hidden ---
    assert block.cv1.conv.out_channels == 2 * hidden, "cv1 must produce two hidden-channel halves"
    assert block.cv2.conv.in_channels == (2 + n) * hidden, "pre-fuse concat width is (2 + n) * hidden"
    assert len(block.blocks) == n, "must hold n densely chained inner units"
    # --- inner-unit type switches on the c3k flag (module introspection) ---
    expected_inner = C3k if c3k else Bottleneck
    assert all(isinstance(unit, expected_inner) for unit in block.blocks), "inner unit type must match c3k flag"
    x = torch.randn(2, in_channels, 8, 8)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, out_channels, 8, 8), "C3k2 must preserve spatial resolution"


def test_c3k2_inner_factory_override() -> None:
    """A custom inner_block_factory replaces the default selection for every inner unit."""
    built: list[tuple[int, bool]] = []

    def factory(hidden_channels: int, shortcut: bool) -> torch.nn.Module:
        built.append((hidden_channels, shortcut))
        return DepthwiseConv(hidden_channels, hidden_channels, kernel_size=3)

    block = C3k2(32, 64, n=2, e=0.5, inner_block_factory=factory, shortcut=False).eval()
    assert built == [(32, False), (32, False)], "factory receives (hidden, shortcut) once per inner unit"
    assert all(isinstance(unit, DepthwiseConv) for unit in block.blocks), "override supplies the inner units"
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        assert block(x).shape == (2, 64, 8, 8)


def test_c3k2_param_count_increases_with_depth() -> None:
    """Parameter count strictly increases as more inner units are added."""
    counts = [sum(p.numel() for p in C3k2(64, 64, n=n).parameters()) for n in (1, 2, 3)]
    assert counts[0] < counts[1] < counts[2], "each additional inner unit must add parameters"
