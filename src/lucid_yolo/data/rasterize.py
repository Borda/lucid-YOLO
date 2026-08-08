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

#: Element budget for the batched crossing test's ``(instances, points, H, W)``
#: intermediate. At 64 M booleans that is ~64 MB, which a 160-px prototype grid with
#: COCO-scale instance counts sits an order of magnitude under; it exists so that one
#: unusually detailed ring cannot size the whole stack.
_CHUNK_ELEMENTS = 64_000_000


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
    ys = torch.arange(height, dtype=torch.float32).view(1, height, 1)
    xs = torch.arange(width, dtype=torch.float32).view(1, 1, width)
    # Every edge at once, not one Python iteration each. The looped form cost one
    # full height x width tensor op per vertex, on CPU, inside the training step:
    # at a 160-px prototype grid a 64-point COCO ring took 1.65 ms, so batch 32
    # spent most of a second per step rasterising while the GPU sat idle. Edges
    # here are the (i-1 -> i) pairs the loop walked, produced by rolling the ring.
    previous = ring.roll(1, dims=0)
    yi, xi = ring[:, 1].view(-1, 1, 1), ring[:, 0].view(-1, 1, 1)
    yj, xj = previous[:, 1].view(-1, 1, 1), previous[:, 0].view(-1, 1, 1)
    # A horizontal ray at row `ys` crosses an edge only where the edge straddles
    # that row; `straddles` is False for horizontal edges, so the divide-by-zero
    # below lands only on masked-out entries.
    straddles = (yi > ys) != (yj > ys)  # (P, height, 1)
    x_cross = (xj - xi) * (ys - yi) / (yj - yi) + xi  # (P, height, 1)
    crossings = straddles & (xs < x_cross)  # (P, height, width)
    # Even-odd parity over the edges: the XOR fold the loop performed, as a sum.
    return crossings.sum(dim=0) % 2 == 1


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

    # Every instance in one crossing test rather than one call each. A mosaic batch
    # carries far more instances than an unaugmented one -- 1138 across 32 images in
    # a measured COCO batch, 35 an image -- and per-ring calls made that 1138 passes
    # over the grid, on CPU, inside the training step.
    #
    # Rings are padded to a common point count by repeating the last vertex, which
    # adds only zero-length edges: their endpoints share a y, so `straddles` is False
    # and they contribute no crossing. The roll below still pairs vertex 0 with the
    # true last vertex, so the closing edge is counted exactly once.
    counts = [int(ring.shape[0]) for ring in polygons]
    longest = max(counts)
    padded = torch.stack(
        [
            ring if count == longest else torch.cat([ring, ring[-1:].expand(longest - count, 2)])
            for ring, count in zip(polygons, counts, strict=True)
        ]
    )  # (N, P, 2)

    # The intermediate is (instances, points, height, width) booleans, so a single
    # unusually detailed ring would otherwise size the whole stack. Chunking bounds
    # it without changing the result: instances are independent.
    per_instance = max(longest * height * width, 1)
    chunk_size = max(1, _CHUNK_ELEMENTS // per_instance)
    chunks = [
        _crossing_parity(padded[start : start + chunk_size], height, width)
        for start in range(0, len(polygons), chunk_size)
    ]
    return torch.cat(chunks)


def _crossing_parity(rings: Tensor, height: int, width: int) -> Tensor:
    """Even-odd crossing test for a padded ``(N, P, 2)`` ring stack, all edges at once.

    Args:
        rings: ``(N, P, 2)`` float rings, padded by repeated vertices.
        height: Mask height in pixels.
        width: Mask width in pixels.

    Returns:
        ``(N, height, width)`` boolean masks.

    Examples:
        >>> import torch
        >>> square = torch.tensor([[[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]]])
        >>> int(_crossing_parity(square, 6, 6).sum())
        9
    """
    ys = torch.arange(height, dtype=torch.float32).view(1, 1, height, 1)
    xs = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    previous = rings.roll(1, dims=1)
    yi, xi = rings[..., 1].unsqueeze(-1).unsqueeze(-1), rings[..., 0].unsqueeze(-1).unsqueeze(-1)
    yj, xj = previous[..., 1].unsqueeze(-1).unsqueeze(-1), previous[..., 0].unsqueeze(-1).unsqueeze(-1)
    straddles = (yi > ys) != (yj > ys)  # (N, P, height, 1)
    x_cross = (xj - xi) * (ys - yi) / (yj - yi) + xi  # (N, P, height, 1)
    return (straddles & (xs < x_cross)).sum(dim=1) % 2 == 1
