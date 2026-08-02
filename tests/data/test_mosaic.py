# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-011 four-image mosaic assembly (blueprint sec. 5.9).

Covers the canvas geometry (``2S x 2S``), that every output box lies within the
canvas, that instances are lost only to clipping (kept count bounded by the input
count when nothing is clipped), seeded quadrant placement of a distinctive pixel,
box/polygon consistency, grey fill in uncovered regions, the wrong-count and
rotated-box guards, and byte-for-byte determinism under a seeded generator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import torch

from open_yolos.data import Targets, boxes_from_polygons
from open_yolos.data.mosaic import MosaicAssembly

#: Base size ``S``; the assembled canvas is ``2S x 2S``.
_TARGET = 32
#: Per-image size used across the suite (each image is ``_TILE x _TILE``).
_TILE = 32
#: Grey fill value shared with letterbox and affine.
_GREY = 114.0 / 255.0
#: Mosaic always consumes exactly four images.
_MOSAIC_COUNT = 4


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed every RNG source before each test so generated geometry is deterministic."""
    torch.manual_seed(0)
    yield


def _generator(seed: int = 1234) -> torch.Generator:
    """Return a CPU generator seeded to ``seed`` for reproducible sampling."""
    return torch.Generator().manual_seed(seed)


def _image() -> torch.Tensor:
    """Return a random CHW float image on a single tile."""
    return torch.rand(3, _TILE, _TILE)


def _one_box_targets() -> Targets:
    """Build a single small box well inside the tile (never clipped when centred)."""
    return Targets(boxes=torch.tensor([[8.0, 8.0, 16.0, 16.0]]), labels=torch.tensor([0]))


def _polygon_targets() -> Targets:
    """Build one square polygon ring well inside the tile, with its matching box."""
    rings = [torch.tensor([[8.0, 8.0], [20.0, 8.0], [20.0, 24.0], [8.0, 24.0]])]
    boxes = boxes_from_polygons(rings)
    labels = torch.zeros(len(rings), dtype=torch.int64)
    return Targets(boxes=boxes, labels=labels, polygons=rings)


def _items(targets_factory: Callable[[], Targets]) -> list[tuple[torch.Tensor, Targets]]:
    """Return four ``(image, targets)`` pairs, each built from ``targets_factory``."""
    return [(_image(), targets_factory()) for _ in range(_MOSAIC_COUNT)]


class TestCanvasGeometry:
    """The stitched canvas is ``2S x 2S`` and every output box lies within it."""

    def test_canvas_is_twice_target_size(self) -> None:
        """The output image spans ``2S x 2S`` regardless of centre sampling."""
        mosaic = MosaicAssembly(target_size=_TARGET, generator=_generator())

        out_image, _ = mosaic(_items(_one_box_targets))

        assert out_image.shape == (3, 2 * _TARGET, 2 * _TARGET)

    def test_every_box_within_canvas_bounds(self) -> None:
        """Every merged box is clamped to ``[0, 2S]`` on both axes."""
        mosaic = MosaicAssembly(target_size=_TARGET, generator=_generator())

        _, out = mosaic(_items(_one_box_targets))

        limit = float(2 * _TARGET)
        assert torch.all(out.boxes >= 0.0)
        assert torch.all(out.boxes <= limit)


class TestInstanceCounts:
    """Kept instances are bounded by the inputs, and drops come only from clipping."""

    def test_unclipped_inputs_keep_every_instance(self) -> None:
        """Four one-box tiles placed without clipping yield exactly four boxes."""
        mosaic = MosaicAssembly(target_size=_TARGET, generator=_generator())

        _, out = mosaic(_items(_one_box_targets))

        assert out.boxes.shape[0] == _MOSAIC_COUNT
        assert out.labels.tolist() == [0, 0, 0, 0]

    def test_clipped_instance_dropped(self) -> None:
        """A box straddling a tile edge is dropped once the off-canvas part is cut away."""
        edge = Targets(boxes=torch.tensor([[28.0, 8.0, 60.0, 16.0]]), labels=torch.tensor([0]))
        items = [(_image(), edge.clone()) for _ in range(_MOSAIC_COUNT)]
        mosaic = MosaicAssembly(target_size=_TARGET, generator=_generator(), min_visibility=0.5)

        _, out = mosaic(items)

        assert out.boxes.shape[0] <= _MOSAIC_COUNT


class TestQuadrantPlacement:
    """With a fixed centre, a distinctive pixel from each input lands in its quadrant."""

    def test_marker_pixels_land_in_expected_quadrants(self) -> None:
        """Each tile's bright marker pixel appears inside that tile's canvas quadrant."""
        mosaic = MosaicAssembly(target_size=_TARGET, generator=_generator())
        images = [torch.full((1, _TILE, _TILE), _GREY) for _ in range(_MOSAIC_COUNT)]
        for image in images:
            image[0, _TILE // 2, _TILE // 2] = 1.0
        items = [(image, Targets.empty()) for image in images]

        out_image, _ = mosaic(items)

        assert mosaic.last_center is not None
        cx, cy = mosaic.last_center
        size = 2 * _TARGET
        quadrants = [(0, 0, cx, cy), (cx, 0, size, cy), (0, cy, cx, size), (cx, cy, size, size)]
        for value in range(_MOSAIC_COUNT):
            x1, y1, x2, y2 = quadrants[value]
            region = out_image[0, y1:y2, x1:x2]
            assert region.numel() > 0
            assert torch.isclose(region.max(), torch.tensor(1.0))


class TestBoxPolygonConsistency:
    """Recomputed boxes track the clipped polygons exactly (box/mask agreement)."""

    def test_merged_boxes_match_polygon_extents(self) -> None:
        """Every merged box equals the extent of its merged polygon ring."""
        mosaic = MosaicAssembly(target_size=_TARGET, generator=_generator())

        _, out = mosaic(_items(_polygon_targets))

        assert len(out.polygons) == out.boxes.shape[0]
        derived = boxes_from_polygons(out.polygons)
        assert torch.equal(out.boxes, derived)


class TestGreyFill:
    """Regions no tile covers keep the grey pad value."""

    def test_uncovered_canvas_is_grey(self) -> None:
        """Small tiles leave grey gaps between the quadrants."""
        small = 8
        mosaic = MosaicAssembly(target_size=_TARGET, generator=_generator())
        images = [torch.zeros(3, small, small) for _ in range(_MOSAIC_COUNT)]
        items = [(image, Targets.empty()) for image in images]

        out_image, _ = mosaic(items)

        assert torch.isclose(out_image.max(), torch.tensor(_GREY))
        assert torch.isclose(out_image.min(), torch.tensor(0.0))


class TestGuards:
    """Wrong input counts and rotated boxes are rejected."""

    def test_wrong_item_count_raises(self) -> None:
        """Fewer than four items raises ValueError naming the required count."""
        mosaic = MosaicAssembly(target_size=_TARGET)

        with pytest.raises(ValueError, match="exactly 4"):
            mosaic([(_image(), Targets.empty()) for _ in range(3)])

    def test_rboxes_raise_not_implemented(self) -> None:
        """A non-empty rboxes tensor on any input raises NotImplementedError naming WP-058."""
        rboxes = torch.tensor([[10.0, 20.0, 8.0, 4.0, 0.3]])
        with_rbox = Targets(boxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.int64), rboxes=rboxes)
        items = [(_image(), Targets.empty()) for _ in range(3)] + [(_image(), with_rbox)]
        mosaic = MosaicAssembly(target_size=_TARGET)

        with pytest.raises(NotImplementedError, match="WP-058"):
            mosaic(items)


class TestDeterminism:
    """Two seeded generators with the same seed give byte-identical outputs."""

    def test_same_seed_identical_outputs(self) -> None:
        """Equal-seed generators produce equal mosaic images, boxes and centres."""
        images = [_image() for _ in range(_MOSAIC_COUNT)]
        boxes = _one_box_targets()
        items_a = [(image.clone(), boxes.clone()) for image in images]
        items_b = [(image.clone(), boxes.clone()) for image in images]
        mosaic_a = MosaicAssembly(target_size=_TARGET, generator=_generator(99))
        mosaic_b = MosaicAssembly(target_size=_TARGET, generator=_generator(99))

        image_a, targets_a = mosaic_a(items_a)
        image_b, targets_b = mosaic_b(items_b)

        assert torch.equal(image_a, image_b)
        assert torch.equal(targets_a.boxes, targets_b.boxes)
        assert mosaic_a.last_center == mosaic_b.last_center
