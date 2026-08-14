# SPDX-License-Identifier: Apache-2.0
"""Anchor-point grid for the anchor-free detection head (WP-025).

The detector is anchor-free: every feature-map cell contributes exactly one
anchor *point* (not a set of anchor boxes), located at the cell centre. For a
level with stride ``s`` the cell in row ``i`` (top to bottom) and column ``j``
(left to right) maps to the pixel-space point ``((j + 0.5) * s, (i + 0.5) * s)``
on the network input (A11: centres at ``(i + 0.5) * stride``; sources R4, R5).

Points are laid out level by level and, within a level, in row-major order — the
same order a ``(H, W)`` feature map flattens to ``(H * W,)`` — so the returned
tensors align position-for-position with a head that flattens and concatenates
its per-level predictions in the same ``(strides[0], strides[1], ...)`` order.
At a 640-pixel input with strides ``[8, 16, 32]`` this yields
``80*80 + 40*40 + 20*20 = 8400`` points.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["HEAD_STRIDES", "anchor_grid", "make_anchor_points"]

#: Column count of an ``(x, y)`` anchor point.
_POINT_DIM = 2

#: Input-pixel strides of the P3/P4/P5 detection levels (blueprint sec. 5.4). Public
#: vocabulary about the architecture rather than a local convenience: every consumer that
#: builds a grid for the head — the training step, both evaluators, single-image
#: inference — is describing the *same* three levels, and three private copies of the
#: triple are three chances for one to be edited alone.
HEAD_STRIDES: tuple[int, int, int] = (8, 16, 32)


def make_anchor_points(
    feature_sizes: list[tuple[int, int]],
    strides: list[int],
) -> tuple[Tensor, Tensor]:
    """Build the concatenated anchor-point grid across feature levels.

    Args:
        feature_sizes: One ``(height, width)`` cell count per level, ordered to
            match ``strides``. ``height`` is the number of rows, ``width`` the
            number of columns of the feature map.
        strides: The input-pixel stride of each level, same length and order as
            ``feature_sizes``.

    Returns:
        A pair ``(anchor_points, stride_per_anchor)`` where ``anchor_points`` has
        shape ``(A, 2)`` holding ``(x, y)`` cell-centre coordinates in input
        pixels (``float32``), and ``stride_per_anchor`` has shape ``(A,)`` giving
        the owning level's stride for each point (``float32``). ``A`` is the sum
        of ``height * width`` over all levels; points are ordered by level and,
        within a level, row-major.

    Raises:
        ValueError: If ``feature_sizes`` and ``strides`` differ in length.

    Examples:
        >>> import torch
        >>> points, strides = make_anchor_points([(2, 2)], [4])
        >>> points
        tensor([[2., 2.],
                [6., 2.],
                [2., 6.],
                [6., 6.]])
        >>> strides
        tensor([4., 4., 4., 4.])
    """
    if len(feature_sizes) != len(strides):
        raise ValueError(
            f"feature_sizes and strides must have equal length, got {len(feature_sizes)} and {len(strides)}"
        )

    points_per_level: list[Tensor] = []
    strides_per_level: list[Tensor] = []
    for (height, width), stride in zip(feature_sizes, strides, strict=True):
        shift_x = (torch.arange(width, dtype=torch.float32) + 0.5) * stride
        shift_y = (torch.arange(height, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        level_points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        points_per_level.append(level_points)
        strides_per_level.append(torch.full((height * width,), float(stride), dtype=torch.float32))

    if not points_per_level:
        return (
            torch.zeros((0, _POINT_DIM), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
        )
    return torch.cat(points_per_level, dim=0), torch.cat(strides_per_level, dim=0)


def anchor_grid(
    canvas: tuple[int, int],
    device: torch.device,
    strides: tuple[int, ...] = HEAD_STRIDES,
) -> tuple[Tensor, Tensor]:
    """Build the head's anchor grid for one input canvas, on ``device``.

    :func:`make_anchor_points` takes feature-map sizes; every caller that has an *image*
    rather than a feature map divides the canvas by the strides to get them, then moves
    both tensors onto the device the model runs on. That three-line step was written
    once per consumer — the dual-path evaluator, the oriented evaluator, single-image
    inference — which is three places for the stride triple to be edited alone, and the
    grid is what pairs a prediction with the pixel it came from. It lives here now,
    beside the function it wraps.

    The canvas is a ``(height, width)`` pair rather than a square side because the
    rectangular case is the general one: a square canvas is ``(s, s)``, while a square
    signature cannot express a letterbox that is not square.

    Args:
        canvas: Input canvas ``(height, width)`` in pixels, each divisible by every
            stride.
        device: Device the returned tensors are moved to — the one the model runs on.
        strides: Per-level input-pixel strides. Defaults to :data:`HEAD_STRIDES`; a
            caller that carries its own (the evaluator takes them as a constructor
            argument) passes them through.

    Returns:
        The ``(A, 2)`` anchor centres in canvas pixels and their ``(A,)`` strides, both
        on ``device``.

    Examples:
        >>> import torch
        >>> points, strides = anchor_grid((64, 64), torch.device("cpu"))
        >>> points.shape  # 8x8 + 4x4 + 2x2 cells
        torch.Size([84, 2])
        >>> sorted(set(strides.tolist()))
        [8.0, 16.0, 32.0]
    """
    height, width = canvas
    feature_sizes = [(height // stride, width // stride) for stride in strides]
    points, stride_per_anchor = make_anchor_points(feature_sizes, list(strides))
    return points.to(device), stride_per_anchor.to(device)
