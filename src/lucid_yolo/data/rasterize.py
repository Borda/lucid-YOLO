# SPDX-License-Identifier: Apache-2.0
"""Polygon-to-pixel-mask rasterisation shared by augmentation and supervision (WP-087).

One rasteriser, two consumers. :class:`~lucid_yolo.data.mixup.CopyPaste` uses it to
decide which pixels a polygon **moves** between images; segmentation target
construction uses it to decide which pixels that same polygon **supervises**. A
second implementation would let the two disagree — the model would be trained
against masks that do not match the augmentation which produced its pixels — so
the function lives here, in neither caller, and both import it.

The fill rule (even-odd ray casting) and the boundary convention (half-open pixel
blocks: left/top edges inside, right/bottom outside) are load-bearing and are
**not** to be "improved": the half-open block is the convention
:mod:`lucid_yolo.losses.mask_loss` crops against (A11), so changing it here would
silently shift every mask target by a pixel.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["rasterize_polygon", "rasterize_polygons"]


def rasterize_polygon(ring: Tensor, height: int, width: int) -> Tensor:
    """Rasterise a polygon ring to a boolean pixel mask by the even-odd rule.

    Each pixel is tested at its integer-coordinate centre ``(x, y)`` with the
    classic ray-casting (PNPOLY) even-odd crossing test, vectorised over the whole
    ``height x width`` grid: for every polygon edge, pixels whose horizontal ray to
    ``-inf`` crosses that edge flip their inside/outside parity. The test is
    ``O(P * H * W)`` in the ring's point count ``P`` — acceptable at training-time
    resolutions (e.g. 640 px) and kept deliberately simple over a scanline sweep.

    Boundary pixels (a centre lying exactly on an edge) follow PNPOLY's half-open
    convention: the left/top edges count as inside, the right/bottom as outside, so
    an axis-aligned rectangle rasterises to a clean half-open pixel block.

    Args:
        ring: ``(P, 2)`` float polygon points ``(x, y)``; ``P >= 3`` for any area.
        height: Mask height in pixels.
        width: Mask width in pixels.

    Returns:
        ``(height, width)`` boolean mask, ``True`` where a pixel centre is inside
        the polygon.

    Examples:
        ```pycon
        >>> import torch
        >>> square = torch.tensor([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
        >>> rasterize_polygon(square, 6, 6).sum().item()
        9

        ```
    """
    ys = torch.arange(height, dtype=torch.float32).view(height, 1)
    xs = torch.arange(width, dtype=torch.float32).view(1, width)
    inside = torch.zeros((height, width), dtype=torch.bool)
    point_count = ring.shape[0]
    for i in range(point_count):
        yi = ring[i, 1]
        yj = ring[i - 1, 1]
        # A horizontal ray at row `ys` crosses edge (i-1 -> i) only where the edge
        # straddles that row; `straddles` is False for horizontal edges, so the
        # divide-by-zero below lands only on masked-out entries.
        straddles = (yi > ys) != (yj > ys)
        xi = ring[i, 0]
        xj = ring[i - 1, 0]
        x_cross = (xj - xi) * (ys - yi) / (yj - yi) + xi
        inside = inside ^ (straddles & (xs < x_cross))
    return inside


def rasterize_polygons(polygons: list[Tensor], height: int, width: int) -> Tensor:
    """Rasterise one mask per polygon ring and stack them instance-first.

    The per-instance counterpart of :func:`rasterize_polygon`, for target
    construction where each ground-truth instance needs its own mask rather than
    a merged one. Rings are rasterised independently and are free to overlap; no
    occlusion or priority ordering is applied here.

    An empty ``polygons`` list returns an empty ``(0, height, width)`` tensor
    rather than raising — an image with no instances is ordinary training data,
    and :func:`torch.stack` on an empty list would raise instead.

    Args:
        polygons: One ``(P_i, 2)`` float ring per instance; may be empty.
        height: Mask height in pixels, shared by every instance.
        width: Mask width in pixels, shared by every instance.

    Returns:
        ``(len(polygons), height, width)`` boolean mask stack, in the order the
        rings were given.

    Examples:
        ```pycon
        >>> import torch
        >>> square = torch.tensor([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
        >>> rasterize_polygons([square, square], 6, 6).shape
        torch.Size([2, 6, 6])
        >>> rasterize_polygons([], 6, 6).shape
        torch.Size([0, 6, 6])

        ```
    """
    if not polygons:
        return torch.zeros((0, height, width), dtype=torch.bool)
    return torch.stack([rasterize_polygon(ring, height, width) for ring in polygons])
