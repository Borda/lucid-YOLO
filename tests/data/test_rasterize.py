# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the shared polygon rasteriser (WP-087).

The rasteriser was private to :mod:`lucid_yolo.data.mixup` until WP-087 promoted
it to :mod:`lucid_yolo.data.rasterize`, where copy-paste augmentation and
segmentation target construction both consume it. These tests pin the two things
that promotion could silently change — the even-odd fill rule and the half-open
boundary convention — plus the batch helper's shape contract and the identity of
the function copy-paste actually calls.
"""

from __future__ import annotations

import math

import pytest
import torch

from lucid_yolo.data import mixup, rasterize
from lucid_yolo.data.rasterize import rasterize_polygon, rasterize_polygons
from lucid_yolo.data.targets import Targets

#: Grid side used by the single-polygon cases.
_SIDE = 10

#: Corners of the triangle case, offset off the pixel lattice on every edge.
_TRIANGLE = torch.tensor([[0.3, 0.4], [10.3, 0.4], [0.3, 10.4]])

#: Grid side large enough to hold ``_TRIANGLE`` with a margin.
_TRIANGLE_SIDE = 12


def _square_ring(x_min: float, y_min: float, x_max: float, y_max: float) -> torch.Tensor:
    """Build the four-point ring of an axis-aligned rectangle, clockwise from top-left."""
    return torch.tensor([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])


def test_axis_aligned_square_fills_exact_half_open_block() -> None:
    """A rectangle rasterises to the exact half-open pixel block, not one shifted or grown by a row.

    The half-open convention (left/top edges inside, right/bottom outside) is what
    :mod:`lucid_yolo.losses.mask_loss` crops against (A11). A rasteriser that
    counted the closing edges too, or rounded rather than truncated, would still
    return a plausible connected blob — this compares against the exact index
    block, so an off-by-one on either side fails.
    """
    mask = rasterize_polygon(_square_ring(2.0, 2.0, 6.0, 6.0), _SIDE, _SIDE)

    expected = torch.zeros((_SIDE, _SIDE), dtype=torch.bool)
    expected[2:6, 2:6] = True
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


def test_triangle_matches_the_analytic_half_plane_lattice() -> None:
    """A diagonal edge includes exactly the pixel centres the three half-planes admit.

    A right triangle is the cheapest shape whose edges are not axis-aligned, so it
    is where a broken crossing test (a wrong parity flip, an inclusive rather than
    exclusive ``xs < x_cross``) shows up. The corners sit off the pixel lattice, so
    no centre lies on an edge and the expected mask is the unambiguous conjunction
    of the three half-planes, computed here independently of the implementation.
    """
    mask = rasterize_polygon(_TRIANGLE, _TRIANGLE_SIDE, _TRIANGLE_SIDE)

    ys = torch.arange(_TRIANGLE_SIDE, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(_TRIANGLE_SIDE, dtype=torch.float32).view(1, -1)
    expected = (xs > 0.3) & (ys > 0.4) & (xs + ys < 10.7)
    assert torch.equal(mask, expected)
    # 45 lattice centres against a continuous area of 50: pixel-centre sampling of
    # a diagonal edge is short by O(perimeter), which is the convention, not a bug.
    assert int(mask.sum()) == 45


def _looped_rasterize_polygon(ring: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """The per-vertex reference the vectorised crossing test replaced.

    Kept as an oracle rather than deleted: it is the form every segmentation
    target and every copy-paste mask was produced by until the vectorisation, so
    it is what "the output did not change" has to be measured against.
    """
    ys = torch.arange(height, dtype=torch.float32).view(height, 1)
    xs = torch.arange(width, dtype=torch.float32).view(1, width)
    inside = torch.zeros((height, width), dtype=torch.bool)
    for i in range(ring.shape[0]):
        yi, yj = ring[i, 1], ring[i - 1, 1]
        straddles = (yi > ys) != (yj > ys)
        xi, xj = ring[i, 0], ring[i - 1, 0]
        x_cross = (xj - xi) * (ys - yi) / (yj - yi) + xi
        inside = inside ^ (straddles & (xs < x_cross))
    return inside


def _regular_ring(vertices: int, radius: float = 5.1) -> torch.Tensor:
    """Build a ``(vertices, 2)`` ring on an irrational-radius circle, off the pixel lattice."""
    angles = torch.arange(vertices, dtype=torch.float32) * (2 * math.pi / vertices)
    return torch.stack([7.3 + radius * torch.cos(angles), 6.9 + (radius - 0.4) * torch.sin(angles)], dim=-1)


@pytest.mark.parametrize("vertices", [3, 4, 17, 64])
def test_vectorised_crossing_test_matches_the_per_vertex_loop(vertices: int) -> None:
    """All edges at once produces the mask the per-vertex loop produced, exactly.

    The loop cost one full ``height x width`` tensor op per vertex, on CPU, inside
    the training step: a 64-point ring on a 160-px prototype grid took 1.65 ms, so
    a batch of 32 spent most of a second per step rasterising while the GPU idled.
    Doing every edge in one shot is a speed change only if the parity is identical
    — the fold moved from an XOR chain to a sum modulo two, and horizontal edges
    still divide by zero before being masked out, so the ``NaN`` handling has to
    agree too.

    Vertex counts span a triangle up to a ring of COCO-like complexity, and the
    radius is irrational so no vertex or crossing lands on a pixel centre, where
    the two forms could agree by rounding rather than by construction.
    """
    angles = torch.arange(vertices, dtype=torch.float32) * (2 * math.pi / vertices)
    ring = torch.stack([7.3 + 5.1 * torch.cos(angles), 6.9 + 4.7 * torch.sin(angles)], dim=-1)

    vectorised = rasterize_polygon(ring, 16, 16)

    assert torch.equal(vectorised, _looped_rasterize_polygon(ring, 16, 16))
    assert bool(vectorised.any())  # not vacuously equal on two empty masks


def test_batch_helper_stacks_one_mask_per_ring() -> None:
    """The batch helper returns one mask per instance in the given order, not a merged mask.

    Segmentation supervision needs per-instance masks; a helper that unioned the
    rings would still return a ``bool`` tensor of the right height and width and
    would be caught only by the leading dimension and the per-ring content.
    """
    left = _square_ring(1.0, 1.0, 3.0, 3.0)
    right = _square_ring(5.0, 5.0, 8.0, 8.0)

    masks = rasterize_polygons([left, right], _SIDE, _SIDE)

    assert masks.shape == (2, _SIDE, _SIDE)
    assert masks.dtype == torch.bool
    assert torch.equal(masks[0], rasterize_polygon(left, _SIDE, _SIDE))
    assert torch.equal(masks[1], rasterize_polygon(right, _SIDE, _SIDE))


@pytest.mark.parametrize(
    "vertex_counts",
    [
        pytest.param((3, 3, 3), id="uniform"),
        pytest.param((5, 41, 7), id="one-long-ring"),
        pytest.param((64, 3, 17, 4), id="mixed"),
    ],
)
def test_batched_rings_match_one_call_per_ring(vertex_counts: tuple[int, ...]) -> None:
    """Rasterising a whole image at once gives what per-ring calls give, whatever the point counts.

    Each ring is rasterised inside its own window, so the instances of one image no
    longer share anything — not a padded point axis, not a grid. That is what makes
    a ragged image cheap, and it is also what could go wrong silently: a stack
    assembled from per-ring windows can be plausible in shape and content while a
    ring landed in the wrong row of it.

    The counts deliberately include a ring an order of magnitude longer than its
    neighbours, which is exactly the case padding handled correctly but slowly.
    """
    rings = [_regular_ring(count, radius=3.0 + index) for index, count in enumerate(vertex_counts)]

    masks = rasterize_polygons(rings, 16, 16)

    expected = torch.stack([rasterize_polygon(ring, 16, 16) for ring in rings])
    assert torch.equal(masks, expected)
    assert bool(masks.all(dim=0).any())  # not vacuously equal on empty masks


def test_chunking_the_edge_axis_does_not_change_the_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk budget small enough to split every ring yields the unchunked masks exactly.

    A ring's edge axis is chunked so that one that is both unusually detailed and
    grid-spanning cannot size the whole ``(edges, h, w)`` intermediate at once.
    Crossings are counted into an integer accumulator for precisely this reason —
    the parity is read only at the end — so a chunk boundary falling mid-ring must
    not change a single pixel.
    """
    rings = [_regular_ring(count, radius=3.0 + index) for index, count in enumerate((5, 41, 7))]
    unchunked = rasterize_polygons(rings, 16, 16)

    monkeypatch.setattr(rasterize, "_CHUNK_ELEMENTS", 2)  # two edges a chunk at any window
    chunked = rasterize_polygons(rings, 16, 16)

    assert torch.equal(chunked, unchunked)
    assert bool(unchunked.any())


def test_batch_helper_returns_empty_stack_for_no_rings() -> None:
    """An image with no instances yields a ``(0, H, W)`` stack instead of raising.

    ``torch.stack([])`` raises, so the empty case needs its own branch; without it
    every batch containing one unlabelled image would crash target construction.
    """
    masks = rasterize_polygons([], _SIDE, _SIDE)

    assert masks.shape == (0, _SIDE, _SIDE)
    assert masks.dtype == torch.bool


def test_copy_paste_uses_the_promoted_rasteriser() -> None:
    """Copy-paste rasterises through the promoted function itself, not a private copy.

    The whole point of the promotion is that the pixels an instance *moves* and the
    pixels it *supervises* come from one implementation. A leftover private copy in
    :mod:`lucid_yolo.data.mixup` would pass every other test here while letting the
    two definitions drift apart.
    """
    ring = _square_ring(1.0, 1.0, 5.0, 5.0)
    source_targets = Targets(boxes=torch.tensor([[1.0, 1.0, 5.0, 5.0]]), labels=torch.tensor([5]), polygons=[ring])
    copy_paste = mixup.CopyPaste(p=1.0)

    out_image, _ = copy_paste(
        [(torch.zeros(3, _SIDE, _SIDE), Targets.empty()), (torch.ones(3, _SIDE, _SIDE), source_targets)]
    )

    assert mixup._rasterize_polygon is rasterize_polygon
    assert torch.equal(out_image[0] == 1.0, rasterize_polygon(ring, _SIDE, _SIDE))


@pytest.mark.parametrize(
    ("offset", "span"),
    [
        pytest.param((8.0, 8.0), 4.0, id="well-inside"),
        pytest.param((0.0, 0.0), 6.0, id="touches-the-top-left-corner"),
        pytest.param((12.0, 12.0), 8.0, id="spills-off-the-bottom-right"),
        pytest.param((-5.0, 3.0), 7.0, id="spills-off-the-left"),
        pytest.param((-20.0, -20.0), 5.0, id="entirely-outside"),
        pytest.param((0.0, 0.0), 40.0, id="covers-the-whole-grid"),
    ],
)
def test_windowed_rasterisation_matches_the_full_grid_rule(offset: tuple[float, float], span: float) -> None:
    """Testing only a ring's own window gives what testing every pixel gave, exactly.

    Rings are rasterised inside their bounding window because a COCO instance
    covers a small part of the prototype grid, and a full-grid pass spends nearly
    all of its time proving distant pixels are outside — measured 4454 ms for one
    batch of 64 mosaic images on a single thread, which starved the loader workers.

    The window is a restriction of the full-grid computation, not a second rule:
    the coordinate ranges are sliced and the ring is never translated, so every
    surviving pixel sees the identical arithmetic. The oracle here is the original
    per-vertex loop over the whole grid, so a window that clipped a row, or a
    translation that re-rounded a crossing, fails rather than merely shifting a
    boundary pixel nobody checks. The cases that matter are the ones at the edges
    of the clamp: a ring on the border, one hanging off each side, one entirely
    outside, and one with no restriction left to make.
    """
    angles = torch.arange(9, dtype=torch.float32) * (2 * math.pi / 9)
    centre = torch.tensor(offset) + span
    ring = centre + torch.stack([span * torch.cos(angles), span * 0.8 * torch.sin(angles)], dim=-1)

    windowed = rasterize_polygon(ring, _SIDE * 2, _SIDE * 2)

    assert torch.equal(windowed, _looped_rasterize_polygon(ring, _SIDE * 2, _SIDE * 2))


@pytest.mark.parametrize("points", [0, 1, 2])
def test_a_ring_without_area_rasterises_to_nothing(points: int) -> None:
    """A ring of fewer than three points encloses no area and yields an empty mask.

    Fewer than three points cannot bound a region, and the crossing test agrees —
    every edge is traversed twice in opposite directions, so the parity cancels.
    The guard exists because the window is derived from the ring's extent, and
    ``amin`` over a zero-point ring raises rather than returning an empty box.
    """
    ring = torch.rand(points, 2) * _SIDE

    mask = rasterize_polygon(ring, _SIDE, _SIDE)

    assert mask.shape == (_SIDE, _SIDE)
    assert not bool(mask.any())
