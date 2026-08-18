# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-057 overlapping crop tiling (A21, A39).

Two halves. The placement half is pure integer geometry — window counts, flush edge
windows, and the DoD's ``test_coverage_no_gaps``, which paints every window onto a mask
and demands no unpainted pixel for a sweep of sizes, patches and overlaps including ones
that divide evenly, leave a remainder, or are shorter than the patch on one or both axes.

The re-mapping half exercises R18's 0.7 rule: an object wholly inside a window keeps its
annotation translated and stays easy; an object straddling a boundary appears in *both*
windows with the flags the rule prescribes, and the two parts' visible fractions sum to
one — the area-additivity the whole rule rests on. Clipped areas are checked against a
**shapely** oracle over a randomized sweep of rotated boxes against windows; shapely is a
dev-only dependency (pyproject ``[dependency-groups] dev``) and appears here as an oracle
only — never in ``src/``.

DOTA-v1.0 itself is not on this machine, so nothing here claims real-data verification;
the one image-level test uses the A26 synthetic OBB fixture set.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry

from lucid_yolo.data import (
    CROP_OVERLAP,
    DIFFICULT_AREA_FRACTION,
    PATCH_SIZE,
    Targets,
    TiledTargets,
    crop_image,
    crop_targets,
    points_in_rboxes,
    polygons_to_rboxes,
    rboxes_to_polygons,
    tile_image_targets,
    tile_windows,
)

#: Placement sweep: even division, remainder, short on one axis, short on both, and the
#: A21 defaults against a DOTA-sized image.
_PLACEMENT_CASES = [
    pytest.param((10, 10), 6, 2, id="even-division"),
    pytest.param((11, 13), 6, 2, id="remainder-both-axes"),
    pytest.param((4, 13), 6, 2, id="short-height"),
    pytest.param((11, 5), 6, 2, id="short-width"),
    pytest.param((4, 5), 6, 2, id="short-both"),
    pytest.param((6, 6), 6, 0, id="exact-patch-no-overlap"),
    pytest.param((7, 6), 6, 5, id="unit-stride"),
    pytest.param((4000, 3000), PATCH_SIZE, CROP_OVERLAP, id="dota-sized-a21-defaults"),
]


def _rbox(cx: float, cy: float, w: float, h: float, theta: float) -> torch.Tensor:
    """Build a one-row ``(1, 5)`` rotated-box tensor.

    Examples:
        >>> _rbox(1.0, 2.0, 3.0, 4.0, 0.5).tolist()
        [[1.0, 2.0, 3.0, 4.0, 0.5]]
    """
    return torch.tensor([[cx, cy, w, h, theta]], dtype=torch.float32)


def _targets(rboxes: torch.Tensor, labels: list[int]) -> Targets:
    """Build targets whose axis-aligned boxes are the envelopes of ``rboxes`` (WP-056).

    Examples:
        >>> targets = _targets(_rbox(10.0, 10.0, 4.0, 2.0, 0.0), [1])
        >>> targets.boxes.tolist(), targets.labels.tolist()
        ([[8.0, 9.0, 12.0, 11.0]], [1])
    """
    polygons = rboxes_to_polygons(rboxes)
    boxes = torch.cat([polygons.amin(dim=1), polygons.amax(dim=1)], dim=1)
    return Targets(boxes=boxes, labels=torch.tensor(labels, dtype=torch.int64), rboxes=rboxes)


def _random_rboxes(
    generator: torch.Generator, count: int, centres: tuple[float, float], extents: tuple[float, float]
) -> torch.Tensor:
    """Draw ``count`` canonical-extent rotated boxes with centres and extents in the given ranges.

    Examples:
        >>> gen = torch.Generator().manual_seed(0)
        >>> _random_rboxes(gen, count=2, centres=(0.0, 10.0), extents=(1.0, 5.0)).shape
        torch.Size([2, 5])
    """
    centre = torch.rand((count, 2), generator=generator) * (centres[1] - centres[0]) + centres[0]
    extent = torch.rand((count, 2), generator=generator) * (extents[1] - extents[0]) + extents[0]
    angle = torch.rand((count, 1), generator=generator) * (2 * math.pi) - math.pi
    return torch.cat([centre, extent.sort(dim=1, descending=True).values, angle], dim=1).float()


def _shapely_clip(rbox: torch.Tensor, window: torch.Tensor) -> BaseGeometry:
    """Oracle clip: shapely's intersection of one rotated box's polygon with the window.

    Examples:
        >>> geom = _shapely_clip(_rbox(5.0, 5.0, 4.0, 4.0, 0.0), torch.tensor([0, 0, 10, 10]))
        >>> round(geom.area, 4)
        16.0
    """
    corners = rboxes_to_polygons(rbox.reshape(1, 5).to(torch.float64))[0]
    x0, y0, x1, y1 = (float(v) for v in window.tolist())
    return Polygon(corners.tolist()).intersection(shapely_box(x0, y0, x1, y1))


def _clip_vertices(rbox: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Oracle clipped-region vertices as a ``(Q, 2)`` float32 tensor, ring closure dropped.

    Examples:
        >>> _clip_vertices(_rbox(5.0, 5.0, 4.0, 4.0, 0.0), torch.tensor([0, 0, 10, 10])).shape
        torch.Size([4, 2])
    """
    return torch.tensor(_shapely_clip(rbox, window).exterior.coords[:-1], dtype=torch.float32)


def _inflate(rbox: torch.Tensor, slack: float = 2e-3) -> torch.Tensor:
    """Widen a ``(5,)`` box by ``slack`` on each side, absorbing float32 slack in a containment test.

    Examples:
        >>> _inflate(torch.tensor([1.0, 1.0, 4.0, 2.0, 0.0])).shape
        torch.Size([1, 5])
    """
    grown = rbox.clone().reshape(1, 5)
    grown[:, 2:4] += 2 * slack
    return grown


@pytest.mark.parametrize(("size", "patch", "overlap"), _PLACEMENT_CASES)
def test_coverage_no_gaps(size: tuple[int, int], patch: int, overlap: int) -> None:
    """Every pixel of the source falls in at least one window, and none runs past an edge."""
    height, width = size
    windows = tile_windows(size, patch=patch, overlap=overlap)
    covered = torch.zeros((height, width), dtype=torch.bool)
    for x0, y0, x1, y1 in windows.tolist():
        assert 0 <= x0 < x1 <= width, f"window {(x0, y0, x1, y1)} escapes the image width"
        assert 0 <= y0 < y1 <= height, f"window {(x0, y0, x1, y1)} escapes the image height"
        assert (x1 - x0, y1 - y0) == (min(patch, width), min(patch, height))
        covered[y0:y1, x0:x1] = True
    assert bool(covered.all()), f"{int((~covered).sum())} pixels of {size} uncovered"


def test_window_placement_divides_evenly() -> None:
    """Windows land on the stride grid, with no extra flush window, when the size divides."""
    assert tile_windows((10, 10), patch=6, overlap=2).tolist() == [
        [0, 0, 6, 6],
        [4, 0, 10, 6],
        [0, 4, 6, 10],
        [4, 4, 10, 10],
    ]


def test_window_placement_remainder_pulls_last_window_flush() -> None:
    """A remainder adds one window flush against the far edge, overlapping more than nominal."""
    starts = [x0 for x0, _, _, _ in tile_windows((6, 11), patch=6, overlap=2).tolist()]
    assert starts == [0, 4, 5]  # nominal stride 4, then a 1 px step to sit flush at 11 - 6


def test_window_smaller_than_patch_is_the_image() -> None:
    """An axis shorter than the patch yields a single window of the axis's own length."""
    assert tile_windows((3, 5), patch=8, overlap=2).tolist() == [[0, 0, 5, 3]]
    assert tile_windows((3, 20), patch=8, overlap=2).tolist() == [[0, 0, 8, 3], [6, 0, 14, 3], [12, 0, 20, 3]]


@pytest.mark.parametrize(
    ("size", "patch", "overlap"),
    [((0, 5), 4, 1), ((5, 5), 0, 1), ((5, 5), 4, 4), ((5, 5), 4, -1)],
)
def test_window_placement_rejects_impossible_parameters(size: tuple[int, int], patch: int, overlap: int) -> None:
    """A non-positive size or patch, or an overlap outside ``[0, patch)``, is rejected."""
    with pytest.raises(ValueError, match=r"must be|must satisfy"):
        tile_windows(size, patch=patch, overlap=overlap)


def test_defaults_are_the_paper_patch_and_the_a21_overlap() -> None:
    """The published 1024 px patch and A21's 200 px overlap are the defaults."""
    assert (PATCH_SIZE, CROP_OVERLAP, DIFFICULT_AREA_FRACTION) == (1024, 200, 0.7)
    assert tile_windows((2048, 2048)).shape[0] == 9  # starts 0, 824, 1024 on each axis


def test_instance_wholly_inside_keeps_its_annotation_translated() -> None:
    """An object fully inside a window is translated, fully visible and not difficult."""
    rboxes = _rbox(60.0, 40.0, 20.0, 10.0, 0.4)  # envelope x in [48.8, 71.2], y in [31.5, 48.5]
    targets = _targets(rboxes, [3])
    window = torch.tensor([40, 20, 100, 80])
    tiled = crop_targets(targets, window, difficult=torch.tensor([False]))
    expected = rboxes.clone()
    expected[:, :2] -= torch.tensor([40.0, 20.0])
    assert torch.allclose(tiled.targets.rboxes, expected, atol=1e-5)
    assert torch.allclose(tiled.targets.boxes, targets.boxes - torch.tensor([40.0, 20.0, 40.0, 20.0]), atol=1e-5)
    assert tiled.targets.labels.tolist() == [3]
    assert tiled.difficult.tolist() == [False]
    assert tiled.visible_fraction.item() == pytest.approx(1.0, abs=1e-6)


def test_straddling_instance_appears_in_both_windows_with_areas_summing_to_one() -> None:
    """R18's split: both parts survive, flagged by the 0.7 rule, and their areas sum to the whole."""
    # A 40x10 axis-aligned box centred at x = 100, cut at x = 108: 80% left, 20% right.
    targets = _targets(_rbox(100.0, 50.0, 40.0, 10.0, 0.0), [7])
    left = crop_targets(targets, torch.tensor([0, 0, 108, 100]), difficult=torch.tensor([False]))
    right = crop_targets(targets, torch.tensor([108, 0, 200, 100]), difficult=torch.tensor([False]))
    assert left.targets.labels.tolist() == right.targets.labels.tolist() == [7]
    assert left.visible_fraction.item() == pytest.approx(0.7, abs=1e-6)
    assert right.visible_fraction.item() == pytest.approx(0.3, abs=1e-6)
    assert left.visible_fraction.item() + right.visible_fraction.item() == pytest.approx(1.0, abs=1e-6)
    # U == 0.7 is not below the threshold, so the left part keeps the original annotation
    # (its box still hangs past the window, exactly as WP-056 leaves boxes unclipped).
    assert left.difficult.tolist() == [False]
    assert torch.allclose(left.targets.rboxes, targets.rboxes, atol=1e-5)
    assert float(left.targets.boxes[0, 2]) == pytest.approx(120.0, abs=1e-5)
    # The right part is below it: flagged, and re-fitted to the 12x10 sliver it kept.
    assert right.difficult.tolist() == [True]
    assert torch.allclose(right.targets.rboxes[0, 2:4], torch.tensor([12.0, 10.0]), atol=1e-4)
    assert torch.allclose(right.targets.boxes, torch.tensor([[0.0, 45.0, 12.0, 55.0]]), atol=1e-4)


@pytest.mark.parametrize(("cut", "expect_difficult"), [(160.0, False), (140.0, True)])
def test_difficult_flag_follows_the_seventy_percent_threshold(cut: float, expect_difficult: bool) -> None:
    """A part is flagged iff its visible fraction is strictly below 0.7 (R18)."""
    targets = _targets(_rbox(150.0, 50.0, 40.0, 10.0, 0.0), [0])  # spans x in [130, 170]
    tiled = crop_targets(targets, torch.tensor([0, 0, int(cut), 100]), difficult=torch.tensor([False]))
    assert tiled.difficult.tolist() == [expect_difficult]
    assert (tiled.visible_fraction.item() < DIFFICULT_AREA_FRACTION) is expect_difficult


def test_incoming_difficult_flag_survives_any_visible_fraction() -> None:
    """An object already flagged difficult stays difficult even when wholly visible."""
    targets = _targets(_rbox(50.0, 50.0, 20.0, 10.0, -0.3), [2])
    tiled = crop_targets(targets, torch.tensor([0, 0, 100, 100]), difficult=torch.tensor([True]))
    assert tiled.visible_fraction.item() == pytest.approx(1.0, abs=1e-6)
    assert tiled.difficult.tolist() == [True]


def test_instance_outside_the_window_is_dropped() -> None:
    """``U == 0`` drops the instance; the surviving axis stays 1:1 across every modality."""
    rboxes = torch.cat([_rbox(20.0, 20.0, 10.0, 6.0, 0.0), _rbox(300.0, 300.0, 10.0, 6.0, 0.0)])
    difficult = torch.zeros(2, dtype=torch.bool)
    tiled = crop_targets(_targets(rboxes, [1, 4]), torch.tensor([0, 0, 100, 100]), difficult=difficult)
    assert tiled.targets.labels.tolist() == [1]
    assert tiled.targets.boxes.shape[0] == tiled.targets.rboxes.shape[0] == 1
    assert tiled.difficult.shape == tiled.visible_fraction.shape == (1,)


def test_empty_targets_and_empty_window_give_shaped_empties() -> None:
    """No instances at all, and a window that catches none, both yield valid empty targets."""
    empty = crop_targets(Targets.empty(), torch.tensor([0, 0, 10, 10]), difficult=torch.zeros(0, dtype=torch.bool))
    missed = crop_targets(
        _targets(_rbox(500.0, 500.0, 10.0, 10.0, 0.0), [0]),
        torch.tensor([0, 0, 10, 10]),
        difficult=torch.tensor([False]),
    )
    for tiled in (empty, missed):
        assert tiled.targets.boxes.shape == (0, 4)
        assert tiled.targets.rboxes.shape == (0, 5)
        assert tiled.targets.labels.shape == (0,)
        assert tiled.difficult.shape == (0,) and tiled.difficult.dtype == torch.bool
        assert tiled.visible_fraction.shape == (0,)
        assert tiled.targets.polygons == []


def test_emitted_rboxes_are_canonical_long_edge_boxes() -> None:
    """Re-fitted parts come back canonical: ``w >= h`` and ``theta`` in ``[-pi/4, 3*pi/4)``."""
    generator = torch.Generator().manual_seed(57)
    angles = torch.linspace(-math.pi, math.pi, 24)
    rboxes = torch.stack([torch.tensor([50.0, 50.0, 40.0, 12.0, float(theta)]) for theta in angles])
    rboxes[:, :2] += torch.rand((rboxes.shape[0], 2), generator=generator) * 20.0 - 10.0
    tiled = crop_targets(
        _targets(rboxes, [0] * rboxes.shape[0]),
        torch.tensor([0, 0, 55, 55]),
        difficult=torch.zeros(rboxes.shape[0], dtype=torch.bool),
    )
    kept = tiled.targets.rboxes
    assert kept.shape[0] > 0
    assert bool((kept[:, 2] >= kept[:, 3]).all())
    assert bool(((kept[:, 4] >= -math.pi / 4) & (kept[:, 4] < 3 * math.pi / 4)).all())


def test_clipped_area_matches_shapely_oracle() -> None:
    """Visible fractions agree with shapely over a randomized rotated-box/window sweep."""
    generator = torch.Generator().manual_seed(1024)
    rboxes = _random_rboxes(generator, count=120, centres=(-100.0, 300.0), extents=(4.0, 84.0))
    window = torch.tensor([0, 0, 200, 150])
    tiled = crop_targets(_targets(rboxes, [0] * rboxes.shape[0]), window, difficult=torch.zeros(120, dtype=torch.bool))
    oracle = torch.tensor(
        [_shapely_clip(rbox, window).area / float(rbox[2] * rbox[3]) for rbox in rboxes],
        dtype=torch.float64,
    )
    assert tiled.targets.labels.shape[0] == int((oracle > 0).sum())
    assert torch.allclose(tiled.visible_fraction.double(), oracle[oracle > 0], atol=1e-6)


def test_emitted_box_contains_its_clipped_region_and_never_grows() -> None:
    """Each emitted box encloses the shapely-clipped region, with extents never above the whole's.

    The fitted rectangle of a cut corner may reach outside the window — it is a rotated
    rectangle around a triangular sliver — so containment, not window-boundedness, is the
    property worth pinning.
    """
    generator = torch.Generator().manual_seed(7)
    rboxes = _random_rboxes(generator, count=40, centres=(20.0, 80.0), extents=(10.0, 40.0))
    window = torch.tensor([0, 0, 50, 50])
    tiled = crop_targets(_targets(rboxes, [0] * 40), window, difficult=torch.zeros(40, dtype=torch.bool))
    survivors = [index for index, rbox in enumerate(rboxes) if _shapely_clip(rbox, window).area > 0.0]
    origin = window[:2].to(torch.float32)
    assert bool((tiled.visible_fraction < DIFFICULT_AREA_FRACTION).any())  # the case under test occurs
    enclosed = [
        bool(
            points_in_rboxes(
                _clip_vertices(rboxes[source], window) - origin, _inflate(tiled.targets.rboxes[part])
            ).all()
        )
        for part, source in enumerate(survivors)
    ]
    assert all(enclosed)
    assert bool((tiled.targets.rboxes[:, 2] <= rboxes[survivors][:, 2] + 1e-3).all())
    assert bool((tiled.targets.rboxes[:, 3] <= rboxes[survivors][:, 3] + 1e-3).all())


def test_axis_aligned_part_reduces_to_the_box_intersection() -> None:
    """For an axis-aligned object the re-fitted part is exactly the box/window intersection."""
    targets = _targets(_rbox(20.0, 20.0, 20.0, 8.0, 0.0), [5])  # x in [10, 30], y in [16, 24]
    tiled = crop_targets(targets, torch.tensor([0, 0, 23, 60]), difficult=torch.tensor([False]))
    assert tiled.visible_fraction.item() == pytest.approx(0.65, abs=1e-6)  # 13 of 20 px wide
    assert torch.allclose(tiled.targets.boxes, torch.tensor([[10.0, 16.0, 23.0, 24.0]]), atol=1e-4)
    assert torch.allclose(tiled.targets.rboxes, torch.tensor([[16.5, 20.0, 13.0, 8.0, 0.0]]), atol=1e-4)


def test_crop_image_matches_its_window() -> None:
    """The image crop is the window's slice of the source, in window order."""
    image = torch.arange(3 * 12 * 16, dtype=torch.float32).reshape(3, 12, 16)
    window = tile_windows((12, 16), patch=10, overlap=4)[1]
    assert torch.equal(crop_image(image, window), image[:, 0:10, 6:16])


def test_tile_image_targets_walks_every_window() -> None:
    """The composition yields one triple per window, each tile matching its own window."""
    image = torch.zeros(3, 30, 40)
    targets = _targets(_rbox(20.0, 15.0, 12.0, 6.0, 0.2), [1])
    tiles = list(tile_image_targets(image, targets, difficult=torch.tensor([False]), patch=16, overlap=4))
    windows = tile_windows((30, 40), patch=16, overlap=4)
    assert len(tiles) == windows.shape[0]
    for (window, tile, tiled), expected in zip(tiles, windows, strict=True):
        assert torch.equal(window, expected)
        assert tile.shape == (3, int(window[3] - window[1]), int(window[2] - window[0]))
        assert isinstance(tiled, TiledTargets)
    assert sum(tiled.targets.labels.shape[0] for _, _, tiled in tiles) > 0


class TestCropTargets:
    """Tests for ``crop_targets``' own input validation."""

    @pytest.mark.parametrize(
        ("targets_kwargs", "difficult", "match"),
        [
            ({"polygons": [torch.zeros((3, 2))]}, torch.tensor([False]), "no polygons"),
            ({}, torch.tensor([0]), "must be bool"),
            ({}, torch.tensor([False, True]), "1-D of length 1"),
        ],
    )
    def test_rejects_broken_inputs(
        self, targets_kwargs: dict[str, list[torch.Tensor]], difficult: torch.Tensor, match: str
    ) -> None:
        """Polygons, a non-bool flag tensor and a mismatched flag length are all rejected."""
        base = _targets(_rbox(5.0, 5.0, 4.0, 2.0, 0.0), [0])
        targets = Targets(boxes=base.boxes, labels=base.labels, rboxes=base.rboxes, **targets_kwargs)
        with pytest.raises((TypeError, ValueError), match=match):
            crop_targets(targets, torch.tensor([0, 0, 10, 10]), difficult=difficult)

    def test_rejects_a_broken_input_instance_axis(self) -> None:
        """A ``rboxes`` axis that does not match the instance axis is rejected, not zipped short."""
        targets = Targets(
            boxes=torch.zeros((1, 4)), labels=torch.zeros(1, dtype=torch.int64), rboxes=torch.zeros((2, 5))
        )
        with pytest.raises(ValueError, match="must share the instance axis"):
            crop_targets(targets, torch.tensor([0, 0, 10, 10]), difficult=torch.tensor([False]))

    def test_rejects_a_degenerate_window(self) -> None:
        """A window with no extent, or the wrong shape, is a caller error rather than an empty crop."""
        targets = _targets(_rbox(5.0, 5.0, 4.0, 2.0, 0.0), [0])
        with pytest.raises(ValueError, match="positive extent"):
            crop_targets(targets, torch.tensor([10, 0, 10, 5]), difficult=torch.tensor([False]))
        with pytest.raises(ValueError, match=r"window must be \(4,\)"):
            crop_targets(targets, torch.tensor([0, 0, 10]), difficult=torch.tensor([False]))


def test_zero_area_annotation_is_dropped() -> None:
    """A degenerate zero-extent annotation has no ``U`` to score and is dropped rather than emitted."""
    targets = _targets(_rbox(5.0, 5.0, 4.0, 0.0, 0.0), [0])
    tiled = crop_targets(targets, torch.tensor([0, 0, 10, 10]), difficult=torch.tensor([False]))
    assert tiled.targets.labels.tolist() == []
    assert tiled.visible_fraction.shape == (0,)


@pytest.mark.parametrize(
    ("difficult", "fraction", "error", "match"),
    [
        pytest.param(torch.tensor([0]), torch.ones(1), TypeError, "difficult must be bool", id="flag-dtype"),
        pytest.param(
            torch.tensor([True]),
            torch.ones(1, dtype=torch.float64),
            TypeError,
            "visible_fraction must be float32",
            id="fraction-dtype",
        ),
        pytest.param(
            torch.tensor([True, False]), torch.ones(1), ValueError, "difficult must be 1-D of length 1", id="flag-len"
        ),
        pytest.param(
            torch.tensor([True]),
            torch.ones(2),
            ValueError,
            "visible_fraction must be 1-D of length 1",
            id="fraction-len",
        ),
    ],
)
def test_tiled_targets_rejects_companions_off_the_instance_axis(
    difficult: torch.Tensor, fraction: torch.Tensor, error: type[Exception], match: str
) -> None:
    """The container validates its own per-instance companions against the instance axis."""
    targets = _targets(_rbox(5.0, 5.0, 4.0, 2.0, 0.0), [0])
    with pytest.raises(error, match=match):
        TiledTargets(targets=targets, difficult=difficult, visible_fraction=fraction)


def test_tiled_targets_rejects_a_broken_instance_axis() -> None:
    """``boxes`` and ``rboxes`` must stay 1:1 in an emitted tile (WP-056's invariant)."""
    targets = Targets(boxes=torch.zeros((1, 4)), labels=torch.zeros(1, dtype=torch.int64), rboxes=torch.zeros((2, 5)))
    with pytest.raises(ValueError, match="boxes/rboxes length mismatch"):
        TiledTargets(targets=targets, difficult=torch.zeros(1, dtype=torch.bool), visible_fraction=torch.zeros(1))


def test_image_shape_is_checked_before_any_cropping() -> None:
    """Both image entry points reject anything that is not a CHW tensor."""
    targets = _targets(_rbox(5.0, 5.0, 4.0, 2.0, 0.0), [0])
    with pytest.raises(ValueError, match=r"image must be \(C, H, W\)"):
        crop_image(torch.zeros(4, 4), torch.tensor([0, 0, 2, 2]))
    with pytest.raises(ValueError, match=r"image must be \(C, H, W\)"):
        list(tile_image_targets(torch.zeros(4, 4), targets, difficult=torch.tensor([False])))


def test_tiles_of_a_synthetic_obb_image_account_for_every_instance(obb_fixture_dir: Path) -> None:
    """On the A26 synthetic OBB fixtures, the tiles of an image account for all of every instance.

    A dataset-shaped pass over real rotated annotations — DOTA itself is not on this
    machine, so nothing here is a real-data check. Because the windows cover the image
    with no gap, the areas an instance contributes across all tiles must sum to at least
    the area it has inside the image (more, where windows overlap).
    """
    quads = _fixture_obb_quads(obb_fixture_dir)
    rboxes = polygons_to_rboxes(quads)
    image_size = _fixture_image_size(obb_fixture_dir)
    image_rect = torch.tensor([0, 0, image_size[1], image_size[0]])
    totals = [
        _tiled_area_total(torch.zeros(3, *image_size), rboxes[index : index + 1]) for index in range(rboxes.shape[0])
    ]
    inside = [_shapely_clip(rboxes[index], image_rect).area for index in range(rboxes.shape[0])]
    assert rboxes.shape[0] > 0
    assert all(total >= area - 1e-3 for total, area in zip(totals, inside, strict=True))
    assert all(total > 0.0 for total in totals)


def _tiled_area_total(image: torch.Tensor, rboxes: torch.Tensor) -> float:
    """Sum one instance's clipped area over every tile of ``image`` (patch 64, overlap 16).

    Examples:
        >>> image = torch.zeros(3, 20, 20)
        >>> round(_tiled_area_total(image, _rbox(10.0, 10.0, 4.0, 4.0, 0.0)), 4)
        16.0
    """
    targets = _targets(rboxes, [0])
    difficult = torch.zeros(1, dtype=torch.bool)
    original = float(rboxes[0, 2] * rboxes[0, 3])
    tiles = tile_image_targets(image, targets, difficult=difficult, patch=64, overlap=16)
    return sum(float(tiled.visible_fraction.sum()) * original for _, _, tiled in tiles)


def _fixture_obb_quads(dataset_dir: Path) -> torch.Tensor:
    """Read the rotated quadrilaterals of the fixture image that carries the most instances.

    Examples:
        >>> callable(_fixture_obb_quads)  # needs a live obb_fixture_dir fixture (generated dataset dir)
        True
    """
    annotations = json.loads((dataset_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    by_image: dict[int, list[list[float]]] = {}
    for annotation in annotations["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation["segmentation"][0])
    busiest = max(by_image.values(), key=len)
    return torch.tensor(busiest, dtype=torch.float32).reshape(-1, 4, 2)


def _fixture_image_size(dataset_dir: Path) -> tuple[int, int]:
    """Return the fixture set's ``(height, width)``; the generator writes one square size.

    Examples:
        >>> callable(_fixture_image_size)  # needs a live obb_fixture_dir fixture (generated dataset dir)
        True
    """
    annotations = json.loads((dataset_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    record = annotations["images"][0]
    return int(record["height"]), int(record["width"])
