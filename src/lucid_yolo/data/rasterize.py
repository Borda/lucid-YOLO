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

import math

import torch
from torch import Tensor

__all__ = ["rasterize_polygon", "rasterize_polygons"]

#: Element budget for the crossing test's ``(edges, h, w)`` intermediate. At 64 M
#: entries that is ~64 MB of bool, which a windowed ring sits orders of magnitude
#: under; it exists so that one unusually detailed ring spanning the whole grid
#: cannot size the stack at once.
_CHUNK_ELEMENTS = 64_000_000

#: Points below which a ring encloses no area, so no pixel centre can be inside it.
_MIN_RING_POINTS = 3


def rasterize_polygon(ring: Tensor, height: int, width: int) -> Tensor:
    """Rasterise a polygon ring to a boolean pixel mask by the even-odd rule.

    Each pixel is tested at its integer-coordinate centre ``(x, y)`` with the
    classic ray-casting (PNPOLY) even-odd crossing test, vectorised over every edge
    at once: pixels whose horizontal ray to ``-inf`` crosses an edge flip their
    inside/outside parity. Only the ring's own bounding window is tested, since no
    pixel outside it can be inside the ring (:func:`_ring_window`), which keeps the
    cost ``O(P * h * w)`` in the ring's own extent rather than the whole grid;
    the rule itself is deliberately kept simple over a scanline sweep.

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
    mask = torch.zeros((height, width), dtype=torch.bool)
    _fill_ring(mask, ring, height, width)
    return mask


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
    masks = torch.zeros((len(polygons), height, width), dtype=torch.bool)
    for index, ring in enumerate(polygons):
        _fill_ring(masks[index], ring, height, width)
    return masks


def _ring_window(ring: Tensor, height: int, width: int) -> tuple[int, int, int, int]:
    """Return the ``(y0, y1, x0, x1)`` pixel window that can hold the ring's interior.

    Outside a ring's own bounding box no pixel centre can be inside it, and the
    crossing test already says so: a row above or below the box straddles no edge,
    a column right of the box has every crossing behind it, and a column left of it
    sees all of them, which is an even count for any closed ring. So the window is
    exactly where the answer can be ``True`` — not an approximation of it.

    The bounds are widened to whole pixels (``floor`` of the low corner, ``floor``
    of the high corner plus one), which may admit one row or column that scores
    ``False`` anyway. Erring outward keeps the window a pure restriction of the
    full-grid computation rather than a second, subtly different rule.

    Args:
        ring: ``(P, 2)`` float polygon points ``(x, y)``.
        height: Mask height in pixels.
        width: Mask width in pixels.

    Returns:
        The half-open window ``(y0, y1, x0, x1)``, clamped to the grid; empty
        (``y1 <= y0`` or ``x1 <= x0``) when the ring lies entirely outside it.

    Examples:
        >>> import torch
        >>> _ring_window(torch.tensor([[1.2, 2.5], [4.0, 2.5], [4.0, 5.5]]), 8, 8)
        (2, 6, 1, 5)
    """
    low_x, low_y = (float(value) for value in ring.amin(dim=0))
    high_x, high_y = (float(value) for value in ring.amax(dim=0))
    return (
        max(0, math.floor(low_y)),
        min(height, math.floor(high_y) + 1),
        max(0, math.floor(low_x)),
        min(width, math.floor(high_x) + 1),
    )


def _fill_ring(mask: Tensor, ring: Tensor, height: int, width: int) -> None:
    """Set ``mask`` ``True`` inside ``ring``, testing only the pixels that can be.

    Restricting the crossing test to the ring's own window is where the cost goes:
    the test is ``O(P * h * w)``, and a COCO instance covers a small part of the
    prototype grid, so a full-grid pass spends nearly all of it proving pixels
    nowhere near the polygon are outside. Measured on a batch of 64 mosaic images
    (2170 instances, 160-px grid) the whole batch's rasterisation was 4454 ms on one
    thread, which was starving the training loop's dataloader workers.

    The window slices the **coordinate ranges**; the ring is never translated. Every
    surviving pixel is therefore tested with the identical ``ys``/``xs`` values and
    the identical edge arithmetic the full-grid form used, so the result is equal bit
    for bit rather than merely equivalent — translating the ring instead would
    re-round every crossing.

    Args:
        mask: ``(height, width)`` boolean mask to write into, in place.
        ring: ``(P, 2)`` float polygon points ``(x, y)``.
        height: Mask height in pixels.
        width: Mask width in pixels.

    Examples:
        >>> import torch
        >>> mask = torch.zeros((6, 6), dtype=torch.bool)
        >>> square = torch.tensor([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
        >>> _fill_ring(mask, square, 6, 6)
        >>> int(mask.sum())
        9
    """
    if ring.shape[0] < _MIN_RING_POINTS:
        return  # no area to enclose, and `amin` on an empty ring would raise
    y0, y1, x0, x1 = _ring_window(ring, height, width)
    if y1 <= y0 or x1 <= x0:
        return

    ys = torch.arange(y0, y1, dtype=torch.float32).view(1, y1 - y0, 1)
    xs = torch.arange(x0, x1, dtype=torch.float32).view(1, 1, x1 - x0)
    # Edges are the (i-1 -> i) pairs, all at once rather than one Python iteration
    # each: the per-vertex loop cost a full tensor op per point.
    previous = ring.roll(1, dims=0)
    # Crossings are counted into an int accumulator, so chunking the edge axis cannot
    # change the result -- addition is associative over the integers and the parity is
    # only read at the end. It bounds the (edges, h, w) intermediate of a ring that is
    # both unusually detailed and grid-spanning.
    crossings = torch.zeros((y1 - y0, x1 - x0), dtype=torch.int32)
    chunk_size = max(1, _CHUNK_ELEMENTS // max((y1 - y0) * (x1 - x0), 1))
    for start in range(0, ring.shape[0], chunk_size):
        ends, begins = ring[start : start + chunk_size], previous[start : start + chunk_size]
        yi, xi = ends[:, 1].view(-1, 1, 1), ends[:, 0].view(-1, 1, 1)
        yj, xj = begins[:, 1].view(-1, 1, 1), begins[:, 0].view(-1, 1, 1)
        # A horizontal ray at row `ys` crosses an edge only where the edge straddles
        # that row; `straddles` is False for horizontal edges, so the divide-by-zero
        # below lands only on masked-out entries.
        straddles = (yi > ys) != (yj > ys)  # (E, h, 1)
        x_cross = (xj - xi) * (ys - yi) / (yj - yi) + xi  # (E, h, 1)
        crossings += (straddles & (xs < x_cross)).sum(dim=0, dtype=torch.int32)
    mask[y0:y1, x0:x1] = crossings % 2 == 1
