# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-016 convolutional primitives (test_blocks.py).

Covers the three blocks in ``lit_yolo.models.blocks``: stride-1/stride-2 spatial
behaviour for odd kernels, the depthwise group count, the bottleneck channel
expansion ratio, the residual add, and a clean eval-mode forward pass.
"""

from __future__ import annotations

import pytest
import torch

from lit_yolo.models import Bottleneck, ConvBNAct, DepthwiseConv


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
