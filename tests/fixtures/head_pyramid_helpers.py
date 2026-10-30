# SPDX-License-Identifier: Apache-2.0
"""Feature-pyramid and anchor-grid helpers shared by the detection-head test modules.

The dual detection head consumes three neck outputs at strides 8/16/32. Several ``tests/models`` modules build such
feature maps and the matching anchor grid for a square input; they live here once instead of being copied per module.
Tests import this module by bare name because ``tests/conftest.py`` puts ``tests/fixtures`` on ``sys.path``.
"""

from __future__ import annotations

import torch

from lucid_yolo.assign.grid import make_anchor_points

#: The head's level strides, in the (8, 16, 32) order the neck emits.
STRIDES = [8, 16, 32]


def feature_sizes(input_size: int) -> list[tuple[int, int]]:
    """Return the per-level ``(H, W)`` cell counts for a square input.

    Examples:
        >>> feature_sizes(640)
        [(80, 80), (40, 40), (20, 20)]
    """
    return [(input_size // stride, input_size // stride) for stride in STRIDES]


def neck_features(input_size: int, channels: tuple[int, int, int], batch: int = 2) -> tuple[torch.Tensor, ...]:
    """Build random neck-style feature maps at strides 8/16/32 for a square input.

    Examples:
        >>> features = neck_features(128, (16, 32, 64))
        >>> [f.shape for f in features]
        [torch.Size([2, 16, 16, 16]), torch.Size([2, 32, 8, 8]), torch.Size([2, 64, 4, 4])]
    """
    return tuple(
        torch.randn(batch, ch, height, width)
        for ch, (height, width) in zip(channels, feature_sizes(input_size), strict=True)
    )


def square_anchor_grid(input_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the shared anchor points and per-anchor strides for a square input.

    Examples:
        >>> points, strides = square_anchor_grid(128)
        >>> points.shape, strides.shape
        (torch.Size([336, 2]), torch.Size([336]))
    """
    return make_anchor_points(feature_sizes(input_size), STRIDES)
