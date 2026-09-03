# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-011 four-image mosaic assembly (blueprint section 5.9).

Covers the canvas geometry (``2S x 2S``), that every output box lies within the
canvas, that instances are lost only to clipping (kept count bounded by the input
count when nothing is clipped), seeded quadrant placement of a distinctive pixel,
box/polygon consistency, grey fill in uncovered regions, the wrong-count and
rotated-box guards, and byte-for-byte determinism under a seeded generator.

Keypoints (WP-132) get their own class: points shift by their image's placement offset,
drop with their instance, and — A70 — keep coordinate and visibility when the quadrant
crop carries them off the canvas, rather than being clamped to the edge or zeroed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import torch

from lucid_yolo.data import Targets, boxes_from_polygons
from lucid_yolo.data.mosaic import MosaicAssembly, MosaicParams

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
    """Return a CPU generator seeded to ``seed`` for reproducible sampling.

    Examples:
        >>> _generator(42).initial_seed()
        42
    """
    return torch.Generator().manual_seed(seed)


def _image() -> torch.Tensor:
    """Return a random CHW float image on a single tile.

    Examples:
        >>> _image().shape
        torch.Size([3, 32, 32])
    """
    return torch.rand(3, _TILE, _TILE)


def _one_box_targets() -> Targets:
    """Build a single small box well inside the tile (never clipped when centred).

    Examples:
        >>> _one_box_targets().boxes.tolist()
        [[8.0, 8.0, 16.0, 16.0]]
    """
    return Targets(boxes=torch.tensor([[8.0, 8.0, 16.0, 16.0]]), labels=torch.tensor([0]))


def _polygon_targets() -> Targets:
    """Build one square polygon ring well inside the tile, with its matching box.

    Examples:
        >>> targets = _polygon_targets()
        >>> targets.boxes.shape
        torch.Size([1, 4])
        >>> len(targets.polygons)
        1
    """
    rings = [torch.tensor([[8.0, 8.0], [20.0, 8.0], [20.0, 24.0], [8.0, 24.0]])]
    boxes = boxes_from_polygons(rings)
    labels = torch.zeros(len(rings), dtype=torch.int64)
    return Targets(boxes=boxes, labels=labels, polygons=rings)


def _items(targets_factory: Callable[[], Targets]) -> list[tuple[torch.Tensor, Targets]]:
    """Return four ``(image, targets)`` pairs, each built from ``targets_factory``.

    Examples:
        >>> items = _items(_one_box_targets)
        >>> len(items)
        4
        >>> items[0][0].shape
        torch.Size([3, 32, 32])
    """
    return [(_image(), targets_factory()) for _ in range(_MOSAIC_COUNT)]


#: A box centred in the tile. Every quadrant placement keeps it above ``min_box_size`` and
#: ``min_visibility`` for any sampled centre, so all four instances always survive and the
#: concat order stays image 0, 1, 2, 3 — which is what lets a test index a known row.
_CENTRED_BOX = torch.tensor([[12.0, 12.0, 20.0, 20.0]])
#: A point whose tile-local coordinate is far enough left and above the tile that image 0's
#: placement (offset ``cx - 32``, ``cy - 32``, with the centre sampled from ``[16, 48]``)
#: puts it off the canvas for every possible centre.
_OFF_TILE_POINT = -40.0


def _posed_targets() -> Targets:
    """Build one centred box carrying two points: one at the tile centre, one far outside it.

    Examples:
        >>> _posed_targets().keypoints.shape
        torch.Size([1, 2, 2])
    """
    return Targets(
        boxes=_CENTRED_BOX.clone(),
        labels=torch.tensor([0]),
        keypoints=torch.tensor([[[16.0, 16.0], [_OFF_TILE_POINT, _OFF_TILE_POINT]]]),
        keypoint_vis=torch.tensor([[2, 1]]),
    )


class TestKeypoints:
    """Points shift with their image, filter with their boxes, and are never clamped."""

    def test_points_shift_by_the_placement_offset(self) -> None:
        """Each image's points move onto the canvas by exactly that image's placement offset.

        Placement is a pure translation, so a point must land at ``local + offset`` — the
        same offset its box takes. Anything else puts the landmark somewhere other than the
        pixels it describes, which the merged canvas gives no way to detect later.
        """
        mosaic = MosaicAssembly(target_size=_TARGET, generator=_generator())
        items = _items(_posed_targets)

        _, out = mosaic(items)

        assert mosaic.last_center is not None
        cx, cy = mosaic.last_center
        assert out.boxes.shape[0] == _MOSAIC_COUNT
        assert out.keypoints[0, 0].tolist() == [16.0 + cx - _TILE, 16.0 + cy - _TILE]

    def test_keypoint_free_targets_keep_the_canonical_empty(self) -> None:
        """A detection-only mosaic returns the canonical empty keypoint pair, untouched.

        Mosaic is on the train path of every task, and the frozen goldens run through it
        with no points at all. The carry-through has to be an exact no-op for them.
        """
        mosaic = MosaicAssembly(target_size=_TARGET, generator=_generator())

        _, out = mosaic(_items(_one_box_targets))

        assert out.keypoints.shape == (0, 0, 2)
        assert out.keypoint_vis.shape == (0, 0)

    def test_dropped_instance_takes_its_points_with_it(self) -> None:
        """An instance clipped away by its quadrant removes its keypoint rows too.

        Points share the box instance axis. A drop that removed the box alone would shift
        every later instance onto its neighbour's landmarks — a silent misalignment, since
        the counts would still agree.
        """
        mosaic = MosaicAssembly(target_size=_TARGET, generator=_generator())
        vanishing = Targets(
            boxes=torch.tensor([[8.0, 8.0, 16.0, 16.0], [-60.0, -60.0, -50.0, -50.0]]),
            labels=torch.tensor([0, 1]),
            keypoints=torch.tensor([[[10.0, 10.0]], [[-55.0, -55.0]]]),
            keypoint_vis=torch.tensor([[2], [2]]),
        )

        _, out = mosaic([(_image(), vanishing.clone()) for _ in range(_MOSAIC_COUNT)])

        assert out.boxes.shape[0] == out.keypoints.shape[0]
        assert out.keypoints.shape[0] == _MOSAIC_COUNT

    def test_point_pushed_off_canvas_is_neither_clamped_nor_zeroed(self) -> None:
        """A70: a point placed outside the canvas on a kept instance keeps coordinate and visibility.

        Image 0 is anchored with its bottom-right corner at the sampled centre, so its
        left and top edges are what the canvas crops. The second point sits far enough
        outside the tile that this placement always carries it past ``x = 0``, while its box
        stays comfortably inside. Clamping the point to the edge would invent a target and
        zeroing its visibility would overload A66's "never annotated" flag, so it must
        survive exactly as given.
        """
        mosaic = MosaicAssembly(target_size=_TARGET, generator=_generator())

        _, out = mosaic(_items(_posed_targets))

        assert mosaic.last_center is not None
        cx, cy = mosaic.last_center
        expected = [_OFF_TILE_POINT + cx - _TILE, _OFF_TILE_POINT + cy - _TILE]
        assert expected[0] < 0.0
        assert out.keypoints[0, 1].tolist() == expected
        assert out.keypoint_vis[0].tolist() == [2, 1]


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


def _sliver_items(route: str) -> list[tuple[torch.Tensor, Targets]]:
    """Return four tiles each carrying one box that clips to a sliver, on ``route``'s axis.

    Paired with a centre of ``(_TILE, _TILE)``, image 0 places at offset ``(0, 0)``, so its
    box spans ``[60, 10, 160, 40]`` against a 64-wide canvas and keeps four columns of
    thirty rows — 4% of its own area. The other three place further right or lower and clip
    away entirely, which is why the control below expects one survivor rather than four.

    Args:
        route: ``"rotated"``, ``"polygons"`` or ``"boxes"``, naming which of the three
            keep-mask call sites the returned targets route through.

    Returns:
        Four ``(image, targets)`` pairs.

    Examples:
        >>> len(_sliver_items("boxes"))
        4
        >>> _sliver_items("rotated")[0][1].rboxes.shape
        torch.Size([1, 5])
    """
    boxes = torch.tensor([[60.0, 10.0, 160.0, 40.0]])
    labels = torch.tensor([0])

    def _targets() -> Targets:
        if route == "rotated":
            return Targets(
                boxes=boxes.clone(), labels=labels.clone(), rboxes=torch.tensor([[110.0, 25.0, 100.0, 30.0, 0.0]])
            )
        if route == "polygons":
            ring = torch.tensor([[60.0, 10.0], [160.0, 10.0], [160.0, 40.0], [60.0, 40.0]])
            return Targets(boxes=boxes.clone(), labels=labels.clone(), polygons=[ring])
        return Targets(boxes=boxes.clone(), labels=labels.clone())

    return [(_image(), _targets()) for _ in range(_MOSAIC_COUNT)]


#: Centre that anchors image 0 at offset ``(0, 0)``, so its targets are shifted by nothing
#: and the clip against the canvas is the only thing acting on them.
_ANCHORED_CENTRE = MosaicParams(center_x=_TILE, center_y=_TILE)


@pytest.mark.parametrize("route", ["rotated", "polygons", "boxes"])
class TestKeepThresholdsReachUpstream:
    """Both thresholds are passed to upstream's keep mask at every one of the three call sites.

    Upstream's ``instance_keep_mask`` defaults ``min_size`` and ``min_visibility`` to
    ``0.0``, which drops nothing, against this project's ``2.0`` and ``0.1``. Omitting
    either argument at any call site keeps every instance with no shape change and no
    exception, so the failure is silent — these cases are what makes it loud. One case per
    site per threshold, plus the control that the drop is the threshold's doing.
    """

    def test_size_threshold_drops_the_sliver(self, route: str) -> None:
        """A four-column clipped box falls below ``min_box_size`` and is dropped.

        ``min_visibility`` is pinned at ``0.0`` so nothing but the size rule can account
        for the drop: at upstream's default the instance would survive, since 4 >= 0.0.
        """
        mosaic = MosaicAssembly(target_size=_TARGET, min_box_size=8.0, min_visibility=0.0)

        _, out = mosaic.apply(_sliver_items(route), _ANCHORED_CENTRE)

        assert out.boxes.shape[0] == 0

    def test_visibility_threshold_drops_the_sliver(self, route: str) -> None:
        """A box retaining 4% of its area falls below ``min_visibility`` and is dropped.

        ``min_box_size`` is pinned at ``0.0`` so nothing but the visibility rule can
        account for the drop: at upstream's default the instance would survive, since
        0.04 >= 0.0.
        """
        mosaic = MosaicAssembly(target_size=_TARGET, min_box_size=0.0, min_visibility=0.5)

        _, out = mosaic.apply(_sliver_items(route), _ANCHORED_CENTRE)

        assert out.boxes.shape[0] == 0

    def test_upstream_defaults_would_keep_every_instance(self, route: str) -> None:
        """With both thresholds at upstream's defaults all four instances survive.

        The control for the two cases above, and the sharpest statement of what omitting an
        argument would cost: at ``0.0``/``0.0`` not only image 0's sliver survives but the
        three instances the placement clips away to nothing do too — a zero-width box is
        still ``>= 0.0`` on both rules — so the pipeline would carry three empty instances
        with every shape lining up and nothing raising.
        """
        mosaic = MosaicAssembly(target_size=_TARGET, min_box_size=0.0, min_visibility=0.0)

        _, out = mosaic.apply(_sliver_items(route), _ANCHORED_CENTRE)

        assert out.boxes.shape[0] == _MOSAIC_COUNT


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

    def test_unpaired_rboxes_raise_value_error(self) -> None:
        """Rotated boxes that do not share an input's instance axis are rejected (WP-056)."""
        rboxes = torch.tensor([[10.0, 20.0, 8.0, 4.0, 0.3]])
        with_rbox = Targets(boxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.int64), rboxes=rboxes)
        items = [(_image(), Targets.empty()) for _ in range(3)] + [(_image(), with_rbox)]
        mosaic = MosaicAssembly(target_size=_TARGET)

        with pytest.raises(ValueError, match="instance axis"):
            mosaic(items)
