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

import torch

from lucid_yolo.data import mixup
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
