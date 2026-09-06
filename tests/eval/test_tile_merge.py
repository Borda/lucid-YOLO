# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the whole-image tile merge (WP-107).

Two clauses carry this file, and they pull in opposite directions on purpose.

The first is that the merge **removes** the seam duplicate an NMS-free path cannot
suppress: an object placed in the overlap band of two adjacent tiles and detected by both
must reach the score once. The second is that the merge **changes nothing else**: over a
single-tile image there is no seam, no duplicate and nothing to own, so the whole-image
figure must reproduce the per-tile figure exactly. A merge that quietly improved the
number would pass the first clause and fail the second, which is why the second is
written as an exact equality rather than a tolerance.

Everything runs on hand-built fixtures (A26): a COCO container written in the test, and
detections written as A45 tuples. No model, no datamodule, no download, no weights (D14)
— the merge is a function of geometry and the tiling record, so it is tested as one.

The cores are derived from :func:`~lucid_yolo.data.tiling.tile_windows`' real placement
rather than from windows invented here, so the flush-against-the-edge trailing window and
the short-axis single window are exercised as the build actually emits them.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from lucid_yolo.data.rotated_geom import rboxes_to_polygons
from lucid_yolo.data.tiling import tile_windows
from lucid_yolo.eval.dota_eval import evaluate_rotated_map, rotated_detections_to_predictions
from lucid_yolo.eval.tile_merge import (
    TileWindow,
    core_bounds,
    in_core,
    load_tile_index,
    merge_whole_images,
    tile_detections_to_source,
)

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

#: Side of the fixture tiles, and of the letterbox canvas they are scored on. Equal by
#: construction so the letterbox is the identity and the single-tile equality is exact
#: rather than merely close: an inverse that divides by a ratio of one changes no bits.
_TILE = 64

#: The two-tile fixture geometry. A 6 px patch at 2 px overlap over a 10 px axis places
#: windows at 0 and 4, so the overlap band is ``[4, 6)`` and the core boundary its
#: midpoint, ``x = 5``. Small enough to reason about by hand, and produced by the same
#: `tile_windows` the build calls.
_PATCH = 6
_OVERLAP = 2


def _ring(rbox: list[float]) -> list[float]:
    """Flatten a rotated box into the COCO ``segmentation`` ring the build writes.

    Args:
        rbox: A ``[cx, cy, w, h, theta]`` long-edge box.

    Returns:
        The eight-value quadrilateral ring.

    Examples:
        >>> len(_ring([0.0, 0.0, 2.0, 2.0, 0.0]))
        8
    """
    corners = rboxes_to_polygons(torch.tensor([rbox], dtype=torch.float32))[0]
    return [float(value) for value in corners.reshape(-1)]


def _annotation(image_id: int, rbox: list[float], *, label: int = 0, difficult: bool = False) -> dict[str, object]:
    """Build one tile annotation record in tile-local pixels.

    Args:
        image_id: The tile's image id.
        rbox: The instance's tile-local rotated box.
        label: Dense class label; the fixture's categories run from id 1.
        difficult: R18's flag (A53).

    Returns:
        The annotation record.

    Examples:
        >>> ann = _annotation(1, [0.0, 0.0, 2.0, 2.0, 0.0], label=2)
        >>> ann["image_id"], ann["category_id"], ann["difficult"], ann["iscrowd"]
        (1, 3, 0, 0)
    """
    return {
        "id": 0,
        "image_id": image_id,
        "category_id": label + 1,
        "segmentation": [_ring(rbox)],
        "difficult": int(difficult),
        "iscrowd": 0,
    }


def _layout(
    tmp_path: Path,
    windows: list[TileWindow],
    annotations: list[dict[str, object]],
    *,
    provenance: bool = True,
) -> Path:
    """Write a tiled COCO container of the shape ``lucid-data build-tiles`` emits.

    Args:
        tmp_path: Directory the file is written into.
        windows: One entry per tile, in image-id order; each record is written under the
            window's own :attr:`~lucid_yolo.eval.tile_merge.TileWindow.tile_id`, so a
            fixture's ids and the ids the index reads back are the same statement.
        annotations: Tile annotations, referencing image ids from one.
        provenance: When ``False``, the A53 window keys are omitted, producing the
            ordinary COCO container the merge must decline rather than misread.

    Returns:
        Path of the written file.

    Examples:
        >>> import json, tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     out = _layout(Path(tmp), [TileWindow("a.png", (0, 0), (4, 4), 1)], [])
        ...     data = json.loads(out.read_text())
        >>> data["images"][0]["file_name"], data["images"][0]["window"]
        ('tile_0.png', [0, 0, 4, 4])
    """
    images: list[dict[str, object]] = []
    for index, window in enumerate(windows):
        record: dict[str, object] = {
            "id": window.tile_id,
            "file_name": f"tile_{index}.png",
            "height": window.size[0],
            "width": window.size[1],
        }
        if provenance:
            record["source_image"] = window.source_image
            record["window"] = [
                window.origin[0],
                window.origin[1],
                window.origin[0] + window.size[1],
                window.origin[1] + window.size[0],
            ]
        images.append(record)
    path = tmp_path / "instances_val.json"
    path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "plane"}, {"id": 2, "name": "ship"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def _detections(rows: list[list[float]], pad: int = 1) -> Tensor:
    """Build one tile's ``(N, 7)`` A45 detection block, padded as the head emits it.

    Args:
        rows: One ``[cx, cy, w, h, theta, score, class]`` tuple per detection.
        pad: Score-zero padding rows appended after them.

    Returns:
        The ``(len(rows) + pad, 7)`` block.

    Examples:
        >>> _detections([[1.0, 2.0, 3.0, 4.0, 0.0, 0.9, 0.0]], pad=2).shape
        torch.Size([3, 7])
    """
    return torch.tensor([*rows, *([[0.0] * 7] * pad)], dtype=torch.float32)


def _two_tile_windows() -> list[TileWindow]:
    """Return the two windows a 10x6 source image tiles into at the fixture geometry.

    Returns:
        Both tiles of ``source.png``, in window order.

    Examples:
        >>> [(w.source_image, w.origin, w.size) for w in _two_tile_windows()]
        [('source.png', (0, 0), (6, 6)), ('source.png', (4, 0), (6, 6))]
    """
    return [
        TileWindow("source.png", (int(window[0]), int(window[1])), (_PATCH, _PATCH), tile_id)
        for tile_id, window in enumerate(tile_windows((_PATCH, 10), patch=_PATCH, overlap=_OVERLAP), start=1)
    ]


class TestCorePartition:
    """The cores are a partition of the plane, derived from the recorded windows alone."""

    def test_the_boundary_sits_at_the_midpoint_of_the_overlap_band(self) -> None:
        """Windows at 0 and 4 with width 6 overlap on ``[4, 6)``, so the boundary is 5."""
        cores = core_bounds(_two_tile_windows())
        assert cores[0, 2].item() == pytest.approx(5.0)
        assert cores[1, 0].item() == pytest.approx(5.0)

    def test_one_tile_owns_the_whole_plane(self) -> None:
        """A source image shorter than the patch tiles into one window, which owns all of it.

        This is what makes the single-tile equality structural rather than incidental:
        with one core there is nothing an ownership filter can remove.
        """
        cores = core_bounds([TileWindow("small.png", (0, 0), (4, 4), 1)])
        assert cores.tolist() == [[-float("inf"), -float("inf"), float("inf"), float("inf")]]

    def test_every_point_is_owned_exactly_once(self) -> None:
        """Over a grid spanning the source image and beyond it, ownership sums to one.

        The window set includes the trailing flush-against-the-edge window an 11 px axis
        forces, whose overlap with its predecessor is wider than the nominal one.
        """
        windows = [
            TileWindow("wide.png", (int(window[0]), int(window[1])), (_PATCH, _PATCH), tile_id)
            for tile_id, window in enumerate(tile_windows((11, 11), patch=_PATCH, overlap=_OVERLAP), start=1)
        ]
        cores = core_bounds(windows)
        axis = torch.arange(-3.0, 15.0, 0.5)
        points = torch.cartesian_prod(axis, axis)
        owners = torch.stack([in_core(points, core) for core in cores])
        assert owners.sum(dim=0).unique().tolist() == [1]

    def test_a_centre_on_a_boundary_belongs_to_the_far_side(self) -> None:
        """Low bounds are inclusive and high bounds exclusive, so boundaries never double."""
        cores = core_bounds(_two_tile_windows())
        on_boundary = torch.tensor([[5.0, 3.0]])
        assert in_core(on_boundary, cores[0]).tolist() == [False]
        assert in_core(on_boundary, cores[1]).tolist() == [True]

    def test_a_grid_row_sharing_one_origin_still_partitions(self) -> None:
        """Two windows at the same x origin and the same width are the ordinary grid case.

        A uniform tiler emits one x origin per column and reuses it down every row, so
        the shared origin is the rule rather than the exception; the refusal below must
        not reach it. Four windows, two columns by two rows, still own the plane once.
        """
        windows = [
            TileWindow("grid.png", (x, y), (6, 6), 1 + 2 * row + column)
            for row, y in enumerate((0, 4))
            for column, x in enumerate((0, 4))
        ]
        cores = core_bounds(windows)
        axis = torch.arange(-3.0, 13.0, 0.5)
        points = torch.cartesian_prod(axis, axis)

        owners = torch.stack([in_core(points, core) for core in cores])

        assert owners.sum(dim=0).unique().tolist() == [1]

    def test_two_windows_sharing_an_origin_with_different_spans_are_refused(self) -> None:
        """An origin needing two boundaries is refused, not silently partitioned by one of them.

        The cores are keyed on the window start, so a start carrying two different ends
        loses one of them and the axis is then partitioned by whichever span was read
        last — every detection between the two ends judged against a boundary belonging
        to the other tiling. No tiler this project ships produces it; a hand-written or
        externally produced layout can.
        """
        windows = [TileWindow("mixed.png", (0, 0), (6, 6), 1), TileWindow("mixed.png", (0, 0), (6, 9), 2)]

        with pytest.raises(ValueError, match="share the x origin 0 with different spans"):
            core_bounds(windows)


class TestTileIndex:
    """Reading the A53 window provenance back out of a tiled container."""

    def test_windows_and_ground_truth_are_read_in_image_id_order(self, tmp_path: Path) -> None:
        """Entry ``i`` describes the tile the unshuffled loader yields ``i``-th."""
        windows = _two_tile_windows()
        index = load_tile_index(_layout(tmp_path, windows, [_annotation(2, [1.0, 3.0, 2.0, 1.0, 0.0], label=1)]))
        assert index is not None
        assert [window.origin for window in index.windows] == [(0, 0), (4, 0)]
        assert index.ground_truth[0]["labels"].tolist() == []
        assert index.ground_truth[1]["labels"].tolist() == [1]

    def test_an_ordinary_coco_container_has_no_merge(self, tmp_path: Path) -> None:
        """No window keys means no tiles, so the caller reports the per-tile figure alone."""
        assert load_tile_index(_layout(tmp_path, _two_tile_windows(), [], provenance=False)) is None

    def test_a_half_provenanced_container_is_refused(self, tmp_path: Path) -> None:
        """Merging part of a split and silently dropping the rest is worse than failing."""
        path = _layout(tmp_path, _two_tile_windows(), [])
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["images"][1]["window"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="partially tiled"):
            load_tile_index(path)


class TestSeamDuplicate:
    """The first DoD clause: one object across a seam yields one detection, not two."""

    #: Source ``x`` of the straddling object: inside the overlap band ``[4, 6)``, so both
    #: windows contain it and both annotate it, and left of the core boundary at ``x = 5``,
    #: so the first tile owns it.
    _SEAM_X = 4.5

    @staticmethod
    def _straddling(tmp_path: Path) -> tuple[Path, list[dict[str, Tensor]]]:
        """Build the straddle, plus a second object the seam does not touch.

        The second object matters to the scoring clause below and to nothing else. COCO
        average precision is blind to false positives ranked *after* full recall is
        attained, so a fixture with the straddler alone would score 1.0 whether the
        duplicate was removed or not; a target that is found later leaves recall to attain
        after the duplicate, and the duplicate then costs precision where it is measured.
        The duplicate is given the **higher** score of the two, so the merge is not
        flattered by the ranking either.

        Returns:
            The container path and the two tiles' predictions, in source pixels.
        """
        windows = _two_tile_windows()
        seam = TestSeamDuplicate._SEAM_X
        annotations = [
            _annotation(1, [seam, 3.0, 2.0, 1.0, 0.0]),
            _annotation(1, [1.5, 3.0, 2.0, 1.0, 0.0]),
            _annotation(2, [seam - 4.0, 3.0, 2.0, 1.0, 0.0]),
        ]
        blocks = [
            _detections([[seam, 3.0, 2.0, 1.0, 0.0, 0.90, 0.0], [1.5, 3.0, 2.0, 1.0, 0.0, 0.50, 0.0]]),
            _detections([[seam - 4.0, 3.0, 2.0, 1.0, 0.0, 0.95, 0.0]]),
        ]
        return _layout(tmp_path, windows, annotations), [
            tile_detections_to_source(block, window, (_PATCH, _PATCH))
            for block, window in zip(blocks, windows, strict=True)
        ]

    def test_both_tiles_detect_it_before_the_merge(self, tmp_path: Path) -> None:
        """The duplicate is real: both tiles place a detection at the same source centre."""
        _, mapped = self._straddling(tmp_path)
        centres = torch.cat([entry["rboxes"][:, 0] for entry in mapped])
        assert int(_at(centres, self._SEAM_X).sum()) == 2

    def test_the_merge_leaves_exactly_one(self, tmp_path: Path) -> None:
        """The whole DoD clause: one object across a seam, one detection after the merge."""
        path, mapped = self._straddling(tmp_path)
        index = load_tile_index(path)
        assert index is not None
        predictions, ground_truth, names = merge_whole_images(index, mapped)
        assert names == ["source.png"]
        at_seam = _at(predictions[0]["rboxes"][:, 0], self._SEAM_X)
        assert int(at_seam.sum()) == 1
        assert predictions[0]["scores"][at_seam].tolist() == pytest.approx([0.9])
        assert int(_at(ground_truth[0]["rboxes"][:, 0], self._SEAM_X).sum()) == 1

    def test_the_duplicate_it_removes_is_worth_a_sixth_of_the_score(self, tmp_path: Path) -> None:
        """The clause restated as the number it protects, against one fixed ground truth.

        Both sides are scored against the **merged** ground truth, with the cap off as the
        whole-image path uses it. Merged predictions score 1.0; the same detections
        without the ownership filter score 0.835, because the surviving duplicate is a
        false positive ranked above a target still to be found. That gap is the
        duplicate-detection cost a per-tile score never pays, measured.
        """
        path, mapped = self._straddling(tmp_path)
        index = load_tile_index(path)
        assert index is not None
        predictions, ground_truth, _ = merge_whole_images(index, mapped)
        unmerged = [{key: torch.cat([entry[key] for entry in mapped], dim=0) for key in ("rboxes", "scores", "labels")}]
        assert evaluate_rotated_map(predictions, ground_truth, max_detections=None)["map_50"] == pytest.approx(1.0)
        assert evaluate_rotated_map(unmerged, ground_truth, max_detections=None)["map_50"] == pytest.approx(
            0.8349835, abs=1e-6
        )


class TestSingleTileEquality:
    """The second DoD clause: no seam, no duplicate, therefore no change to the number."""

    @staticmethod
    def _fixture(tmp_path: Path) -> tuple[Path, Tensor]:
        """Build a one-tile source image and a detection set that partly hits it.

        Deliberately imperfect — one detection lands on a target, one is slightly off,
        one lands on nothing, and a whole class is never detected — so every compared key
        is an interior value rather than a saturated 1.0 that many wrong implementations
        would also produce.

        Returns:
            The container path and the tile's detection block.
        """
        window = TileWindow("solo.png", (0, 0), (_TILE, _TILE), 1)
        annotations = [
            _annotation(1, [20.0, 20.0, 8.0, 4.0, 0.0]),
            _annotation(1, [40.0, 40.0, 6.0, 3.0, 0.3]),
            _annotation(1, [50.0, 12.0, 6.0, 3.0, 0.0], label=1),
        ]
        detections = _detections(
            [
                [20.0, 20.0, 8.0, 4.0, 0.0, 0.95, 0.0],
                [40.5, 40.0, 6.0, 3.0, 0.3, 0.80, 0.0],
                [10.0, 55.0, 5.0, 2.0, 0.0, 0.60, 0.0],
            ]
        )
        return _layout(tmp_path, [window], annotations), detections

    def test_the_whole_image_figure_reproduces_the_per_tile_one(self, tmp_path: Path) -> None:
        """Exact equality on every key, with no tolerance anywhere.

        A tolerance here would hide the defect it guards. The per-tile side is what the
        existing path reports: the detections as the head emits them, against the tile's
        own ground truth. The whole-image side is the new
        path over the same detections. With one tile the letterbox is the identity, the
        window origin is zero and the single core is infinite, so every step of the merge
        is a no-op and any difference is the merge changing the score for a reason
        unrelated to merging.
        """
        path, detections = self._fixture(tmp_path)
        index = load_tile_index(path)
        assert index is not None
        per_tile = evaluate_rotated_map(
            rotated_detections_to_predictions(detections.unsqueeze(0)), [index.ground_truth[0]]
        )
        predictions, ground_truth, _ = merge_whole_images(
            index, [tile_detections_to_source(detections, index.windows[0], (_TILE, _TILE))]
        )
        whole_image = evaluate_rotated_map(predictions, ground_truth, max_detections=None)
        assert whole_image == per_tile

    def test_the_compared_figure_is_not_saturated(self, tmp_path: Path) -> None:
        """Guard on the guard: an equality between two 1.0s would prove nothing."""
        path, detections = self._fixture(tmp_path)
        index = load_tile_index(path)
        assert index is not None
        per_tile = evaluate_rotated_map(
            rotated_detections_to_predictions(detections.unsqueeze(0)), [index.ground_truth[0]]
        )
        assert all(0.0 < value < 1.0 for value in per_tile.values()), per_tile


class TestOwnershipCost:
    """The rule's price, pinned so nobody removes it by turning the merge into NMS."""

    def test_a_detection_only_the_non_owning_tile_made_is_dropped(self, tmp_path: Path) -> None:
        """Whole-image recall is the owner's recall, not the union of both tiles'.

        This is the cost of refusing a confidence-ranked union: the neighbouring tile saw
        the object and the merge discards it anyway, because ownership is decided by
        where the detection is and never by whether another tile also fired. A change
        that makes this test pass with one surviving detection has reintroduced
        suppression at the seam — see the module docstring before making it.
        """
        windows = _two_tile_windows()
        path = _layout(tmp_path, windows, [_annotation(1, [4.5, 3.0, 2.0, 1.0, 0.0])])
        index = load_tile_index(path)
        assert index is not None
        predictions, _, _ = merge_whole_images(
            index,
            [
                tile_detections_to_source(_detections([]), windows[0], (_PATCH, _PATCH)),
                tile_detections_to_source(
                    _detections([[0.5, 3.0, 2.0, 1.0, 0.0, 0.8, 0.0]]), windows[1], (_PATCH, _PATCH)
                ),
            ],
        )
        assert predictions[0]["rboxes"].shape[0] == 0


class TestGroundTruthReconstruction:
    """The ground truth is rebuilt by the same rule, and the R18 0.7 rule makes it safe."""

    def test_a_target_seen_whole_by_two_tiles_is_counted_once(self, tmp_path: Path) -> None:
        """R18 keeps the original annotation above the threshold, so both copies share a centre.

        The cores partition the plane, so exactly one of two identically-placed copies
        survives — double-counting a non-difficult ground truth is structurally
        impossible rather than merely unlikely.
        """
        windows = _two_tile_windows()
        path = _layout(
            tmp_path,
            windows,
            [_annotation(1, [4.5, 3.0, 2.0, 1.0, 0.0]), _annotation(2, [0.5, 3.0, 2.0, 1.0, 0.0])],
        )
        index = load_tile_index(path)
        assert index is not None
        _, ground_truth, _ = merge_whole_images(index, [{"rboxes": torch.zeros(0, 5)} | _empty() for _ in windows])
        assert ground_truth[0]["labels"].tolist() == [0]

    @staticmethod
    def _with_a_straggler(tmp_path: Path) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        """Build a merge whose ground truth keeps a clipped, difficult copy nobody removed.

        Two real targets sit in the first tile, and the second tile carries a clipped part
        flagged *difficult* whose centre its own core owns. Such a straggler needs an
        object wider than the overlap — only then can clipping displace a centre across a
        core boundary — so it is written directly here rather than derived from a tiling
        run; the point under test is what the accumulator does with it.

        The detections find both real targets and the straggler, and the straggler's is
        ranked **between** them, so a detection the true annotation set would have scored
        as a false positive lands where precision is still being measured.

        Returns:
            The merged predictions and ground truth.
        """
        windows = _two_tile_windows()
        path = _layout(
            tmp_path,
            windows,
            [
                _annotation(1, [3.0, 3.0, 2.0, 1.0, 0.0]),
                _annotation(1, [1.0, 3.0, 2.0, 1.0, 0.0]),
                _annotation(2, [1.5, 3.0, 2.0, 1.0, 0.0], difficult=True),
            ],
        )
        index = load_tile_index(path)
        assert index is not None
        predictions, ground_truth, _ = merge_whole_images(
            index,
            [
                tile_detections_to_source(
                    _detections([[3.0, 3.0, 2.0, 1.0, 0.0, 0.9, 0.0], [1.0, 3.0, 2.0, 1.0, 0.0, 0.3, 0.0]]),
                    windows[0],
                    (_PATCH, _PATCH),
                ),
                tile_detections_to_source(
                    _detections([[1.5, 3.0, 2.0, 1.0, 0.0, 0.6, 0.0]]), windows[1], (_PATCH, _PATCH)
                ),
            ],
        )
        return predictions, ground_truth

    def test_a_surviving_clipped_copy_costs_no_recall(self, tmp_path: Path) -> None:
        """A48 keeps a difficult straggler out of the denominator and lets it shadow nothing.

        This is the half of the residual that is genuinely safe: the straggler is carried
        as difficult, the two real targets are the whole recall denominator, and both are
        found — so the merge loses nothing to it.
        """
        predictions, ground_truth = self._with_a_straggler(tmp_path)

        assert ground_truth[0]["difficult"].tolist() == [False, False, True]
        assert evaluate_rotated_map(predictions, ground_truth, max_detections=None)["map_50"] == pytest.approx(1.0)

    def test_a_surviving_clipped_copy_absorbs_a_would_be_false_positive(self, tmp_path: Path) -> None:
        """The merge's one flattering residual, measured rather than asserted away.

        The same predictions are scored against the ground truth the merge produced and
        against the whole-image annotation set a straggler-free reconstruction would have
        given — the two real targets alone. Against the true set the straggler's detection
        is a false positive ranked above a target still to be found and costs 0.165 of
        ``map_50``; the merge discards it instead, exactly as R18's devkit discards a
        detection on a difficult instance. That gap is the direction a whole-image figure
        from this module errs in, and its size on this fixture.
        """
        predictions, ground_truth = self._with_a_straggler(tmp_path)
        real = [{key: value[~ground_truth[0]["difficult"]] for key, value in ground_truth[0].items()}]

        assert evaluate_rotated_map(predictions, ground_truth, max_detections=None)["map_50"] == pytest.approx(1.0)
        assert evaluate_rotated_map(predictions, real, max_detections=None)["map_50"] == pytest.approx(
            0.8349835, abs=1e-6
        )


class TestCoordinates:
    """What crosses from a tile into its source image, and what does not."""

    def test_score_zero_padding_rows_never_reach_the_merge(self) -> None:
        """Padding rows inverse-map to a real coordinate; dropping them by score is the guard."""
        window = TileWindow("a.png", (100, 200), (_TILE, _TILE), 1)
        mapped = tile_detections_to_source(_detections([[8.0, 8.0, 4.0, 2.0, 0.0, 0.7, 0.0]], pad=5), window, (64, 64))
        assert mapped["scores"].tolist() == pytest.approx([0.7])

    def test_a_non_identity_letterbox_is_undone_before_the_translation(self) -> None:
        """A 32x64 tile letterboxed into 64x64 gains 16 px pads; the inverse removes them.

        The order is the point: the letterbox inverse is the existing exact one and the
        window origin is added afterwards, so nothing here recomputes a pad or a ratio.
        """
        window = TileWindow("a.png", (10, 20), (32, 64), 1)
        mapped = tile_detections_to_source(_detections([[32.0, 32.0, 8.0, 4.0, 0.25, 0.9, 0.0]]), window, (64, 64))
        assert mapped["rboxes"][0, :2].tolist() == pytest.approx([42.0, 36.0])
        assert mapped["rboxes"][0, 2:].tolist() == pytest.approx([8.0, 4.0, 0.25])

    def test_a_source_image_whose_detections_are_all_disowned_still_appears(self, tmp_path: Path) -> None:
        """Predictions and ground truth align by position, so an empty image is an entry."""
        windows = [*_two_tile_windows(), TileWindow("other.png", (0, 0), (_PATCH, _PATCH), 3)]
        index = load_tile_index(_layout(tmp_path, windows, []))
        assert index is not None
        predictions, ground_truth, names = merge_whole_images(index, [_empty() | {"rboxes": torch.zeros(0, 5)}] * 3)
        assert names == ["source.png", "other.png"]
        assert len(predictions) == len(ground_truth) == 2


class TestDetectionCap:
    """A47's 300 is per forward pass; a merged source image is many of them."""

    def test_the_whole_image_path_scores_every_merged_detection(self) -> None:
        """With the cap off a 400-detection image keeps all 400, so recall is the model's."""
        boxes = torch.stack(
            [torch.tensor([float(index) * 10.0, 5.0, 4.0, 2.0, 0.0]) for index in range(400)],
        )
        predictions = [
            {"rboxes": boxes, "scores": torch.linspace(0.1, 0.9, 400), "labels": torch.zeros(400, dtype=torch.long)}
        ]
        targets = [{"rboxes": boxes[-1:], "labels": torch.zeros(1, dtype=torch.long)}]
        assert evaluate_rotated_map(predictions, targets, max_detections=None)["map_50"] == pytest.approx(1.0)

    def test_the_default_cap_is_unchanged(self) -> None:
        """The per-tile path must be bit-identical: the same call without the argument caps."""
        boxes = torch.stack(
            [torch.tensor([float(index) * 10.0, 5.0, 4.0, 2.0, 0.0]) for index in range(400)],
        )
        predictions = [
            {"rboxes": boxes, "scores": torch.linspace(0.1, 0.9, 400), "labels": torch.zeros(400, dtype=torch.long)}
        ]
        targets = [{"rboxes": boxes[:1], "labels": torch.zeros(1, dtype=torch.long)}]
        assert evaluate_rotated_map(predictions, targets)["map_50"] == pytest.approx(0.0)


class TestPartialSplit:
    """``--limit`` stops part-way through a split; a half-merged image is not scored."""

    def test_an_incomplete_trailing_source_image_is_dropped(self, tmp_path: Path) -> None:
        """One tile of a two-tile image would report a recall the pipeline never attempted."""
        windows = [*_two_tile_windows(), TileWindow("other.png", (0, 0), (_PATCH, _PATCH), 3)]
        index = load_tile_index(_layout(tmp_path, windows, []))
        assert index is not None
        _, _, names = merge_whole_images(index, [_empty() | {"rboxes": torch.zeros(0, 5)}])
        assert names == []

    def test_a_prefix_ending_on_a_boundary_keeps_its_images(self, tmp_path: Path) -> None:
        """Both tiles of the first image are present, so that image is complete and scored."""
        windows = [*_two_tile_windows(), TileWindow("other.png", (0, 0), (_PATCH, _PATCH), 3)]
        index = load_tile_index(_layout(tmp_path, windows, []))
        assert index is not None
        _, _, names = merge_whole_images(index, [_empty() | {"rboxes": torch.zeros(0, 5)}] * 2)
        assert names == ["source.png"]


def _at(centres: Tensor, coordinate: float) -> Tensor:
    """Return the mask of centres sitting at ``coordinate`` on one axis.

    ``tensor == pytest.approx(x)`` collapses to a single bool, which is not what a count
    of detections at one place needs.

    Args:
        centres: ``(N,)`` coordinates along one axis.
        coordinate: The value to match.

    Returns:
        ``(N,)`` bool mask.

    Examples:
        >>> _at(torch.tensor([1.0, 2.0, 2.0]), 2.0).tolist()
        [False, True, True]
    """
    return torch.isclose(centres, torch.full_like(centres, coordinate))


def _empty() -> dict[str, Tensor]:
    """Return an empty prediction dict, for fixtures whose detections are beside the point.

    Returns:
        A prediction dict with no rows.

    Examples:
        >>> {key: tuple(value.shape) for key, value in _empty().items()}
        {'rboxes': (0, 5), 'scores': (0,), 'labels': (0,)}
    """
    return {
        "rboxes": torch.zeros(0, 5),
        "scores": torch.zeros(0),
        "labels": torch.zeros(0, dtype=torch.long),
    }
