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

#: Element budget for the batched crossing test's ``(edges, H, W)`` intermediate. At
#: 64 M entries that is ~64 MB of int32, which a 160-px prototype grid with COCO-scale
#: instance counts sits an order of magnitude under; it exists so that an image with
#: unusually many edges cannot size the whole stack at once.
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
    # Every edge at once, not one Python iteration each. The looped form cost one
    # full height x width tensor op per vertex, on CPU, inside the training step:
    # at a 160-px prototype grid a 64-point COCO ring took 1.65 ms, so batch 32
    # spent most of a second per step rasterising while the GPU sat idle. Edges
    # here are the (i-1 -> i) pairs the loop walked, produced by rolling the ring,
    # and the parity is the XOR fold the loop performed, as a sum.
    return _edge_crossings(ring, ring.roll(1, dims=0), height, width).sum(dim=0) % 2 == 1


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

    # Every instance in one crossing test rather than one call each: a mosaic batch
    # carries 35 instances an image in a measured COCO batch, and per-ring calls made
    # one pass over the grid each, on CPU, inside the training step.
    #
    # The edges are flattened across instances rather than the rings padded to a
    # common point count. Padding costs `instances * longest_ring` edge tests, so a
    # single 500-point ring made a 7-instance image cost what 3500 points would --
    # measured 0.93 ms to 20.24 ms at a 160-px grid, and COCO rings are that ragged.
    # Flattening costs the true point total instead, and parity is then accumulated
    # per instance rather than read off a padded axis.
    counts = torch.tensor([int(ring.shape[0]) for ring in polygons])
    points = torch.cat(polygons)  # (E, 2)
    starts = torch.repeat_interleave(counts.cumsum(0) - counts, counts)  # (E,)
    local = torch.arange(points.shape[0]) - starts
    # The closing edge is the (last -> first) pair, so vertex 0's predecessor is the
    # ring's last vertex: floor-semantics `%` maps local 0 to count - 1.
    previous = points[starts + (local - 1) % torch.repeat_interleave(counts, counts)]
    instance = torch.repeat_interleave(torch.arange(len(polygons)), counts)  # (E,)

    # Crossings are counted into an int accumulator, so chunking the edge axis cannot
    # change the result -- addition is associative over the integers, and the parity
    # is only read at the end.
    crossings = torch.zeros((len(polygons), height, width), dtype=torch.int32)
    chunk_size = max(1, _CHUNK_ELEMENTS // max(height * width, 1))
    for start in range(0, points.shape[0], chunk_size):
        stop = start + chunk_size
        edge_crossings = _edge_crossings(points[start:stop], previous[start:stop], height, width)
        crossings.index_add_(0, instance[start:stop], edge_crossings.to(torch.int32))
    return crossings % 2 == 1


def _edge_crossings(ends: Tensor, starts: Tensor, height: int, width: int) -> Tensor:
    """Test which pixels' horizontal rays cross each of ``E`` independent edges.

    The per-edge core of the even-odd rule, shared by the instance-batched path: the
    caller decides which edges belong to which ring and folds the parity itself.

    Args:
        ends: ``(E, 2)`` float edge end points ``(x, y)``.
        starts: ``(E, 2)`` float edge start points, aligned with ``ends``.
        height: Mask height in pixels.
        width: Mask width in pixels.

    Returns:
        ``(E, height, width)`` boolean crossings.

    Examples:
        >>> import torch
        >>> ends = torch.tensor([[4.0, 1.0]])
        >>> starts = torch.tensor([[4.0, 4.0]])
        >>> int(_edge_crossings(ends, starts, 6, 6).sum())
        12
    """
    ys = torch.arange(height, dtype=torch.float32).view(1, height, 1)
    xs = torch.arange(width, dtype=torch.float32).view(1, 1, width)
    yi, xi = ends[:, 1].view(-1, 1, 1), ends[:, 0].view(-1, 1, 1)
    yj, xj = starts[:, 1].view(-1, 1, 1), starts[:, 0].view(-1, 1, 1)
    # A horizontal ray at row `ys` crosses an edge only where the edge straddles that
    # row; `straddles` is False for horizontal edges, so the divide-by-zero below
    # lands only on masked-out entries.
    straddles = (yi > ys) != (yj > ys)  # (E, height, 1)
    x_cross = (xj - xi) * (ys - yi) / (yj - yi) + xi  # (E, height, 1)
    return straddles & (xs < x_cross)  # (E, height, width)
