# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-008 target container and transform helpers (blueprint section 5.9).

Covers construction-time validation, deep-copy independence, aligned filtering across every modality, concatenation
semantics, the empty identity, and the two pure geometry helpers against hand-computed cases.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from lucid_yolo.data import (
    Compose,
    GeometricTransform,
    Targets,
    apply_affine_to_points,
    boxes_from_polygons,
    polygons_to_rboxes,
)

#: A rectangle whose long edge runs at 45 degrees: its envelope and its rotated fit differ.
_TILTED_QUAD = torch.tensor([[0.0, 10.0], [20.0, 30.0], [30.0, 20.0], [10.0, 0.0]])
#: An axis-aligned rectangle, ``4 x 2`` at the origin.
_FLAT_QUAD = torch.tensor([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])


def _boxes(n: int) -> torch.Tensor:
    """Return ``n`` distinct float32 xyxy boxes.

    Examples:
        >>> _boxes(2).tolist()
        [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]
    """
    base = torch.arange(n, dtype=torch.float32).reshape(n, 1)
    return torch.cat([base, base, base + 1.0, base + 1.0], dim=1)


def _labels(n: int) -> torch.Tensor:
    """Return ``n`` int64 labels.

    Examples:
        >>> _labels(3).tolist()
        [0, 1, 2]
    """
    return torch.arange(n, dtype=torch.int64)


def _rboxes(m: int) -> torch.Tensor:
    """Return ``m`` float32 long-edge rotated boxes.

    Examples:
        >>> _rboxes(2).tolist()
        [[0.0, 0.0, 2.0, 1.0, 0.0], [1.0, 1.0, 3.0, 2.0, 0.0]]
    """
    base = torch.arange(m, dtype=torch.float32).reshape(m, 1)
    return torch.cat([base, base, base + 2.0, base + 1.0, base * 0.0], dim=1)


def _zero_posed(n_points: int, n: int = 1) -> Targets:
    """Return ``n`` instances, each carrying ``n_points`` all-zero keypoints with zero visibility.

    Examples:
        >>> _zero_posed(3, n=2).keypoints.shape
        torch.Size([2, 3, 2])
        >>> _zero_posed(3, n=2).keypoint_vis.shape
        torch.Size([2, 3])
    """
    return Targets(
        boxes=_boxes(n),
        labels=_labels(n),
        keypoints=torch.zeros((n, n_points, 2)),
        keypoint_vis=torch.zeros((n, n_points), dtype=torch.int64),
    )


class TestTargetsValidation:
    """Construction-time shape and dtype invariants."""

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            pytest.param(
                {"boxes": _boxes(2), "labels": _labels(3)}, ValueError, "length mismatch", id="labels-length-mismatch"
            ),
            pytest.param(
                {"boxes": _boxes(2), "labels": _labels(2), "polygons": [torch.zeros((3, 2))]},
                ValueError,
                "polygons count",
                id="polygon-count-not-zero-or-n",
            ),
            pytest.param(
                {"boxes": torch.zeros((1, 4), dtype=torch.float64), "labels": _labels(1)},
                TypeError,
                "boxes must be float32",
                id="boxes-wrong-dtype",
            ),
            pytest.param(
                {"boxes": _boxes(1), "labels": torch.zeros(1, dtype=torch.int32)},
                TypeError,
                "labels must be int64",
                id="labels-wrong-dtype",
            ),
            pytest.param(
                {"boxes": _boxes(1), "labels": _labels(1), "rboxes": torch.zeros((1, 4))},
                ValueError,
                r"rboxes must be \(M, 5\)",
                id="rboxes-wrong-shape",
            ),
        ],
    )
    def test_malformed_construction_raises(self, kwargs: dict[str, Any], error: type[Exception], match: str) -> None:
        """Mismatched lengths, a polygon count other than 0 or N, wrong dtypes and a non-5-column rbox are rejected.

        Each case breaks exactly one documented invariant of the container, and the message must name it: a labels
        count that differs from the box count, a polygon count that is neither 0 nor N, non-float32 boxes, non-int64
        labels and rotated boxes without the 5 long-edge columns.
        """
        with pytest.raises(error, match=match):
            Targets(**kwargs)

    def test_valid_construction_succeeds(self) -> None:
        """A well-formed set with every modality constructs without error."""
        t = Targets(
            boxes=_boxes(2), labels=_labels(2), polygons=[torch.zeros((3, 2)), torch.ones((4, 2))], rboxes=_rboxes(1)
        )
        assert len(t.labels) == 2
        assert len(t.polygons) == 2
        assert t.rboxes.shape == (1, 5)


class TestKeypoints:
    """Keypoint tensors share the box instance axis and a common keypoint count."""

    def test_valid_construction_succeeds(self) -> None:
        """Float32 ``(N, K, 2)`` points and int64 ``(N, K)`` visibility values are accepted."""
        keypoints = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        keypoint_vis = torch.tensor([[2, 1], [0, 2]])
        targets = Targets(boxes=_boxes(2), labels=_labels(2), keypoints=keypoints, keypoint_vis=keypoint_vis)
        assert torch.equal(targets.keypoints, keypoints)
        assert torch.equal(targets.keypoint_vis, keypoint_vis)

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            pytest.param(
                {"keypoints": torch.zeros((1, 2, 3)), "keypoint_vis": torch.zeros((1, 2), dtype=torch.int64)},
                ValueError,
                r"keypoints must be \(N, K, 2\)",
                id="no-xy-columns",
            ),
            pytest.param(
                {
                    "keypoints": torch.zeros((1, 2, 2), dtype=torch.float64),
                    "keypoint_vis": torch.zeros((1, 2), dtype=torch.int64),
                },
                TypeError,
                "keypoints must be float32",
                id="keypoints-wrong-dtype",
            ),
            pytest.param(
                {"keypoints": torch.zeros((1, 2, 2)), "keypoint_vis": torch.zeros((1, 2), dtype=torch.int32)},
                TypeError,
                "keypoint_vis must be int64",
                id="visibility-wrong-dtype",
            ),
            pytest.param(
                {"keypoints": torch.zeros((1, 2, 2)), "keypoint_vis": torch.zeros((1, 1), dtype=torch.int64)},
                ValueError,
                r"keypoint_vis must be \(N, K\)",
                id="visibility-shape-mismatch",
            ),
            pytest.param(
                {"keypoints": torch.zeros((1, 2, 2))},
                ValueError,
                r"keypoint_vis must be \(N, K\)",
                id="keypoints-with-default-empty-visibility",
            ),
            pytest.param(
                {"keypoint_vis": torch.zeros((1, 2), dtype=torch.int64)},
                ValueError,
                "keypoint_vis count must be 0",
                id="visibility-without-keypoints",
            ),
        ],
    )
    def test_malformed_keypoint_channels_raise(
        self, kwargs: dict[str, Any], error: type[Exception], match: str
    ) -> None:
        """Keypoints without xy columns, wrong dtypes, and keypoint/visibility pairs that disagree are rejected.

        Visibility must hold one int64 scalar per instance/keypoint coordinate pair; present keypoints cannot be paired
        with the default empty visibility tensor, and absent keypoints cannot be paired with visibility rows.
        """
        with pytest.raises(error, match=match):
            Targets(boxes=_boxes(1), labels=_labels(1), **kwargs)

    def test_clone_independent(self) -> None:
        """Mutating cloned keypoint tensors leaves their source tensors unchanged."""
        original = Targets(
            boxes=_boxes(1),
            labels=_labels(1),
            keypoints=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            keypoint_vis=torch.tensor([[2, 1]]),
        )
        clone = original.clone()
        clone.keypoints.add_(10.0)
        clone.keypoint_vis.fill_(0)
        assert original.keypoints.tolist() == [[[1.0, 2.0], [3.0, 4.0]]]
        assert original.keypoint_vis.tolist() == [[2, 1]]

    def test_filter_keeps_selected_keypoint_rows(self) -> None:
        """The instance mask selects matching keypoint and visibility rows."""
        targets = Targets(
            boxes=_boxes(3),
            labels=_labels(3),
            keypoints=torch.arange(12, dtype=torch.float32).reshape(3, 2, 2),
            keypoint_vis=torch.tensor([[0, 1], [1, 2], [2, 0]]),
        )
        kept = targets.filter(torch.tensor([True, False, True]))
        assert torch.equal(kept.keypoints, targets.keypoints[[0, 2]])
        assert torch.equal(kept.keypoint_vis, targets.keypoint_vis[[0, 2]])

    def test_concat_matching_keypoints_succeeds(self) -> None:
        """Matching keypoint counts concatenate on the shared instance axis."""
        first = Targets(
            boxes=_boxes(1),
            labels=_labels(1),
            keypoints=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            keypoint_vis=torch.tensor([[2, 1]]),
        )
        second = Targets(
            boxes=_boxes(2),
            labels=_labels(2),
            keypoints=torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
            keypoint_vis=torch.tensor([[0, 1], [1, 2]]),
        )
        merged = Targets.concat([first, second])
        assert merged.keypoints.shape == (3, 2, 2)
        assert merged.keypoint_vis.tolist() == [[2, 1], [0, 1], [1, 2]]

    def test_concat_mixed_keypoint_presence_raises(self) -> None:
        """Concatenating annotated and keypoint-absent targets is rejected."""
        without_keypoints = Targets(boxes=_boxes(1), labels=_labels(1))
        with pytest.raises(ValueError, match="mixed keypoint presence"):
            Targets.concat([_zero_posed(2), without_keypoints])

    def test_concat_mismatched_keypoint_count_raises(self) -> None:
        """Targets with different K values cannot describe one concatenated keypoint batch."""
        with pytest.raises(ValueError, match="keypoint counts"):
            Targets.concat([_zero_posed(2), _zero_posed(3)])

    def test_empty_has_keypoint_shapes(self) -> None:
        """The empty identity carries canonical zero-instance keypoint tensors."""
        targets = Targets.empty()
        assert targets.keypoints.shape == (0, 0, 2)
        assert targets.keypoint_vis.shape == (0, 0)


class TestSharedKeypointCount:
    """The one ``K`` read both batch-level joins share: ``Targets.concat`` and the packed transport."""

    def test_skips_items_without_point_sets(self) -> None:
        """An empty or box-only item's point axis of ``0`` never outvotes the batch's ``K``.

        Only the middle item carries points (``K = 4``); the leading empty and trailing box-only items must be ignored
        rather than forcing the batch to ``K = 0``.
        """
        box_only = Targets(boxes=_boxes(1), labels=_labels(1))
        assert Targets._shared_keypoint_count([Targets.empty(), _zero_posed(4, n=2), box_only]) == 4

    @pytest.mark.parametrize(
        "items",
        [
            pytest.param([], id="no-items"),
            pytest.param([Targets.empty(), Targets(boxes=_boxes(1), labels=_labels(1))], id="no-point-sets"),
        ],
    )
    def test_zero_without_point_sets(self, items: list[Targets]) -> None:
        """A batch that annotates no points at all has ``K = 0``, the item-free batch included."""
        assert Targets._shared_keypoint_count(items) == 0

    def test_refuses_differing_counts_naming_both(self) -> None:
        """Two point-carrying items with different ``K`` are refused, the first item's count first."""
        with pytest.raises(ValueError, match="differing keypoint counts: 2 and 3"):
            Targets._shared_keypoint_count([_zero_posed(2), Targets.empty(), _zero_posed(3)])


class TestClone:
    """Deep-copy independence."""

    def test_clone_independent(self) -> None:
        """Mutating a clone's tensors leaves the original unchanged.

        Boxes, labels, a polygon ring and the rotated boxes are each modified in place on the clone, so a shared
        buffer in any modality shows up as a changed original.
        """
        original = Targets(
            boxes=_boxes(2), labels=_labels(2), polygons=[torch.zeros((3, 2)), torch.zeros((3, 2))], rboxes=_rboxes(1)
        )
        clone = original.clone()
        clone.boxes.add_(5.0)
        clone.labels.add_(9)
        clone.polygons[0].add_(1.0)
        clone.rboxes.add_(2.0)
        assert torch.equal(original.boxes, _boxes(2))
        assert torch.equal(original.labels, _labels(2))
        assert float(original.polygons[0].sum()) == 0.0
        assert torch.equal(original.rboxes, _rboxes(1))


class TestFilter:
    """Aligned selection across modalities."""

    def test_filter_keeps_boxes_labels_polygons_aligned(self) -> None:
        """One instance mask selects the same rows of boxes, labels and polygons."""
        polys = [torch.full((3, 2), float(i)) for i in range(3)]
        t = Targets(boxes=_boxes(3), labels=_labels(3), polygons=polys)
        kept = t.filter(torch.tensor([True, False, True]))
        assert kept.labels.tolist() == [0, 2]
        assert torch.equal(kept.boxes, _boxes(3)[[0, 2]])
        assert [float(p[0, 0]) for p in kept.polygons] == [0.0, 2.0]

    def test_filter_wrong_mask_length_raises(self) -> None:
        """An instance mask of the wrong length is rejected."""
        t = Targets(boxes=_boxes(2), labels=_labels(2))
        with pytest.raises(ValueError, match="keep mask"):
            t.filter(torch.tensor([True]))

    def test_filter_requires_rkeep_when_rboxes_present(self) -> None:
        """Filtering targets that carry rotated boxes demands a separate rkeep mask."""
        t = Targets(boxes=_boxes(2), labels=_labels(2), rboxes=_rboxes(2))
        with pytest.raises(ValueError, match="rkeep mask is required"):
            t.filter(torch.tensor([True, False]))

    def test_filter_rejects_rkeep_when_rboxes_absent(self) -> None:
        """Supplying rkeep when there are no rotated boxes is a usage error."""
        t = Targets(boxes=_boxes(2), labels=_labels(2))
        with pytest.raises(ValueError, match="rkeep must be None"):
            t.filter(torch.tensor([True, False]), rkeep=torch.tensor([True]))

    def test_filter_applies_independent_rbox_mask(self) -> None:
        """rkeep selects the rotated-box axis independently of the instance axis."""
        t = Targets(boxes=_boxes(2), labels=_labels(2), rboxes=_rboxes(3))
        kept = t.filter(torch.tensor([True, False]), rkeep=torch.tensor([False, True, True]))
        assert kept.boxes.shape[0] == 1
        assert torch.equal(kept.rboxes, _rboxes(3)[[1, 2]])


class TestConcat:
    """Merging several images' targets."""

    def test_concat_merges_counts_and_preserves_dtype(self) -> None:
        """Concatenation sums instance/rbox counts and keeps canonical dtypes."""
        a = Targets(boxes=_boxes(1), labels=_labels(1), rboxes=_rboxes(1))
        b = Targets(boxes=_boxes(2), labels=_labels(2), rboxes=_rboxes(2))
        merged = Targets.concat([a, b])
        assert merged.boxes.shape[0] == 3
        assert merged.rboxes.shape[0] == 3
        assert merged.boxes.dtype == torch.float32
        assert merged.labels.dtype == torch.int64

    def test_concat_merges_polygons(self) -> None:
        """Polygon rings from every input are concatenated in order."""
        a = Targets(boxes=_boxes(1), labels=_labels(1), polygons=[torch.zeros((3, 2))])
        b = Targets(boxes=_boxes(2), labels=_labels(2), polygons=[torch.ones((3, 2)), torch.ones((4, 2))])
        merged = Targets.concat([a, b])
        assert len(merged.polygons) == 3

    def test_concat_mixed_polygon_presence_raises(self) -> None:
        """Concatenating a set with polygons and one without is rejected."""
        with_polys = Targets(boxes=_boxes(1), labels=_labels(1), polygons=[torch.zeros((3, 2))])
        without = Targets(boxes=_boxes(2), labels=_labels(2))
        with pytest.raises(ValueError, match="mixed polygon presence"):
            Targets.concat([with_polys, without])

    def test_concat_empty_list_returns_empty(self) -> None:
        """Concatenating nothing yields the empty identity."""
        merged = Targets.concat([])
        assert merged.boxes.shape == (0, 4)
        assert merged.labels.shape == (0,)


class TestEmpty:
    """The empty identity round-trips through filter and concat."""

    def test_empty_has_canonical_shapes(self) -> None:
        """empty() carries zero-length tensors of the right rank and dtype."""
        t = Targets.empty()
        assert t.boxes.shape == (0, 4)
        assert t.labels.shape == (0,)
        assert t.rboxes.shape == (0, 5)
        assert t.polygons == []

    def test_empty_round_trips_through_filter(self) -> None:
        """Filtering empty targets with an empty mask returns empty targets."""
        kept = Targets.empty().filter(torch.zeros(0, dtype=torch.bool))
        assert kept.boxes.shape == (0, 4)

    def test_empty_concat_identity(self) -> None:
        """Concatenating empty targets with a populated set changes nothing."""
        populated = Targets(boxes=_boxes(2), labels=_labels(2))
        merged = Targets.concat([Targets.empty(), populated, Targets.empty()])
        assert merged.boxes.shape[0] == 2
        assert torch.equal(merged.boxes, _boxes(2))


class TestFromQuads:
    """One quadrilateral per instance becomes its envelope box and its rotated fit."""

    def test_tilted_quad_gives_envelope_box_and_fitted_rbox(self) -> None:
        """A tilted rectangle yields its axis-aligned envelope and its 45-degree long-edge box.

        Both modalities come from the one ring, which is the instance-axis invariant every oriented reader relies on;
        a tilted quad is the case where the two visibly differ.
        """
        targets = Targets.from_quads([_TILTED_QUAD], torch.tensor([3]))

        assert targets.boxes.tolist() == [[0.0, 0.0, 30.0, 30.0]]
        assert targets.rboxes[0].tolist() == pytest.approx([15.0, 15.0, 28.284271, 14.142136, math.pi / 4], abs=1e-5)
        assert targets.labels.tolist() == [3]
        assert targets.difficult.tolist() == [False]

    def test_matches_separate_envelope_and_fit_bit_for_bit(self) -> None:
        """The result equals the two calls the three oriented readers used to make by hand.

        The constructor replaced three copies of ``boxes_from_polygons`` plus ``polygons_to_rboxes`` in the COCO, YOLO
        and DOTA readers; any difference here would move every oriented target those readers produce.
        """
        quads = [_TILTED_QUAD, _FLAT_QUAD]

        targets = Targets.from_quads(quads, _labels(2))

        assert torch.equal(targets.boxes, boxes_from_polygons(quads))
        assert torch.equal(targets.rboxes, polygons_to_rboxes(torch.stack(quads, dim=0)))

    def test_optional_channels_carried_on_shared_axis(self) -> None:
        """Difficult flags and keypoints given alongside the quads arrive unchanged.

        The COCO reader passes R18's ``difficult`` flag and, in keypoint mode, the point channels; dropping either
        here would silently lose the evaluation's ignore rule or the pose supervision.
        """
        difficult = torch.tensor([True, False])
        keypoints = torch.ones((2, 3, 2), dtype=torch.float32)
        keypoint_vis = torch.full((2, 3), 2, dtype=torch.int64)

        targets = Targets.from_quads(
            [_TILTED_QUAD, _FLAT_QUAD], _labels(2), difficult=difficult, keypoints=keypoints, keypoint_vis=keypoint_vis
        )

        assert torch.equal(targets.difficult, difficult)
        assert torch.equal(targets.keypoints, keypoints)
        assert torch.equal(targets.keypoint_vis, keypoint_vis)

    def test_no_quads_yields_empty_targets(self) -> None:
        """An image with no instance is a valid empty set, not a failed stack of nothing."""
        targets = Targets.from_quads([], _labels(0))

        assert targets.boxes.shape == (0, 4)
        assert targets.rboxes.shape == (0, 5)
        assert targets.labels.shape == (0,)


class TestApplyAffineToPoints:
    """Homogeneous point warping."""

    def test_rotation_plus_translation_hand_computed(self) -> None:
        """A +90 deg rotation then (2, 3) translation matches the hand-computed result.

        The unit points (1, 0) and (0, 1) rotate onto (0, 1) and (-1, 0) and then shift to (2, 4) and (1, 3).
        """
        theta = math.pi / 2.0
        matrix = torch.tensor(
            [
                [math.cos(theta), -math.sin(theta), 2.0],
                [math.sin(theta), math.cos(theta), 3.0],
                [0.0, 0.0, 1.0],
            ]
        )
        points = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        expected = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
        torch.testing.assert_close(apply_affine_to_points(points, matrix), expected)

    def test_wrong_point_shape_raises(self) -> None:
        """Points that are not (K, 2) are rejected."""
        with pytest.raises(ValueError, match=r"points must be \(K, 2\)"):
            apply_affine_to_points(torch.zeros((3, 3)), torch.eye(3))


class TestBoxesFromPolygons:
    """Polygon-extent boxes."""

    def test_hand_computed_extent(self) -> None:
        """The enclosing box equals the ring's min/max corners."""
        ring = torch.tensor([[1.0, 2.0], [5.0, 2.0], [5.0, 8.0], [1.0, 8.0]])
        torch.testing.assert_close(boxes_from_polygons([ring]), torch.tensor([[1.0, 2.0, 5.0, 8.0]]))

    def test_empty_list_returns_empty_boxes(self) -> None:
        """No rings yields an empty (0, 4) float32 tensor."""
        out = boxes_from_polygons([])
        assert out.shape == (0, 4)
        assert out.dtype == torch.float32

    def test_empty_ring_raises(self) -> None:
        """A ring with no points has no extent and is rejected."""
        with pytest.raises(ValueError, match="no points"):
            boxes_from_polygons([torch.zeros((0, 2))])


def test_compose_conforms_to_protocol_and_chains() -> None:
    """Compose satisfies GeometricTransform and threads the pair through in order.

    Two shifting transforms are chained, so both the image (+1 each) and the box coordinates (+1 each) must reflect
    two applications.
    """

    def shift(image: torch.Tensor, targets: Targets) -> tuple[torch.Tensor, Targets]:
        moved = targets.clone()
        moved.boxes.add_(1.0)
        return image + 1.0, moved

    pipeline = Compose([shift, shift])
    assert isinstance(pipeline, GeometricTransform)
    image = torch.zeros(3, 4, 4)
    out_image, out_targets = pipeline(image, Targets(boxes=_boxes(1), labels=_labels(1)))
    assert float(out_image[0, 0, 0]) == 2.0
    assert torch.equal(out_targets.boxes, _boxes(1) + 2.0)
