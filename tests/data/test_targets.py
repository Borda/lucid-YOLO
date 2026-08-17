# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-008 target container and transform helpers (blueprint section 5.9).

Covers construction-time validation, deep-copy independence, aligned filtering
across every modality, concatenation semantics, the empty identity, and the two
pure geometry helpers against hand-computed cases.
"""

from __future__ import annotations

import math

import pytest
import torch

from lucid_yolo.data import (
    Compose,
    GeometricTransform,
    Targets,
    apply_affine_to_points,
    boxes_from_polygons,
)


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


class TestTargetsValidation:
    """Construction-time shape and dtype invariants."""

    def test_boxes_labels_length_mismatch_raises(self) -> None:
        """A labels count that differs from the box count is rejected."""
        with pytest.raises(ValueError, match="length mismatch"):
            Targets(boxes=_boxes(2), labels=_labels(3))

    def test_polygon_count_not_zero_or_n_raises(self) -> None:
        """A polygon count that is neither 0 nor N is rejected."""
        with pytest.raises(ValueError, match="polygons count"):
            Targets(boxes=_boxes(2), labels=_labels(2), polygons=[torch.zeros((3, 2))])

    def test_boxes_wrong_dtype_raises(self) -> None:
        """Non-float32 boxes are rejected with a clear dtype error."""
        with pytest.raises(TypeError, match="boxes must be float32"):
            Targets(boxes=torch.zeros((1, 4), dtype=torch.float64), labels=_labels(1))

    def test_labels_wrong_dtype_raises(self) -> None:
        """Non-int64 labels are rejected with a clear dtype error."""
        with pytest.raises(TypeError, match="labels must be int64"):
            Targets(boxes=_boxes(1), labels=torch.zeros(1, dtype=torch.int32))

    def test_rboxes_wrong_shape_raises(self) -> None:
        """Rotated boxes without the 5 long-edge columns are rejected."""
        with pytest.raises(ValueError, match=r"rboxes must be \(M, 5\)"):
            Targets(boxes=_boxes(1), labels=_labels(1), rboxes=torch.zeros((1, 4)))

    def test_valid_construction_succeeds(self) -> None:
        """A well-formed set with every modality constructs without error."""
        t = Targets(
            boxes=_boxes(2), labels=_labels(2), polygons=[torch.zeros((3, 2)), torch.ones((4, 2))], rboxes=_rboxes(1)
        )
        assert len(t.labels) == 2
        assert len(t.polygons) == 2
        assert t.rboxes.shape == (1, 5)


class TestClone:
    """Deep-copy independence."""

    def test_clone_is_independent(self) -> None:
        """Mutating a clone's tensors leaves the original unchanged."""
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

    def test_empty_is_concat_identity(self) -> None:
        """Concatenating empty targets with a populated set changes nothing."""
        populated = Targets(boxes=_boxes(2), labels=_labels(2))
        merged = Targets.concat([Targets.empty(), populated, Targets.empty()])
        assert merged.boxes.shape[0] == 2
        assert torch.equal(merged.boxes, _boxes(2))


class TestApplyAffineToPoints:
    """Homogeneous point warping."""

    def test_rotation_plus_translation_hand_computed(self) -> None:
        """A +90 deg rotation then (2, 3) translation matches the hand-computed result."""
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
    """Compose satisfies GeometricTransform and threads the pair through in order."""

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
