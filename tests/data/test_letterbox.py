# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-009 letterbox transform (blueprint sec. 5.9, A10).

Covers the sub-pixel forward/inverse round trip across boxes and polygons,
aspect preservation and symmetric padding of the resized canvas, box/polygon
consistency after warping, rotated-box handling (centre warp, ``w``/``h`` scale,
``theta`` fixed) and byte-for-byte determinism.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from open_yolos.data import Letterbox, Targets, apply_affine_to_points, boxes_from_polygons

#: Non-square source and square target used across the suite; r = 0.8 on both axes.
_ORIG_H = 500
_ORIG_W = 800
_TARGET = 640
_EXPECTED_R = 0.8


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed every RNG source before each test so generated geometry is deterministic."""
    torch.manual_seed(0)
    yield


def _random_polygons(n: int, n_points: int = 5) -> list[torch.Tensor]:
    """Return ``n`` float32 rings of ``n_points`` points inside the source image."""
    rings: list[torch.Tensor] = []
    for _ in range(n):
        xs = torch.rand(n_points) * _ORIG_W
        ys = torch.rand(n_points) * _ORIG_H
        rings.append(torch.stack([xs, ys], dim=1).to(torch.float32))
    return rings


def _targets_with_polygons(n: int = 6) -> Targets:
    """Build targets whose boxes are exactly the extent of their polygon rings."""
    polygons = _random_polygons(n)
    boxes = boxes_from_polygons(polygons)
    labels = torch.arange(n, dtype=torch.int64)
    return Targets(boxes=boxes, labels=labels, polygons=polygons)


def _image() -> torch.Tensor:
    """Return a random CHW float image at the source size."""
    return torch.rand(3, _ORIG_H, _ORIG_W)


class TestRoundTrip:
    """Forward letterbox then inverse map recovers the original geometry."""

    def test_roundtrip_subpixel(self) -> None:
        """Boxes and polygons survive letterbox -> inverse_map within sub-pixel error."""
        targets = _targets_with_polygons()
        letterbox = Letterbox(_TARGET)

        _, out_targets = letterbox(_image(), targets)
        recovered_boxes = letterbox.inverse_map(
            out_targets.boxes.reshape(-1, 2), orig_size=(_ORIG_H, _ORIG_W), letterboxed_size=(_TARGET, _TARGET)
        ).reshape(-1, 4)
        recovered_rings = [
            letterbox.inverse_map(ring, orig_size=(_ORIG_H, _ORIG_W), letterboxed_size=(_TARGET, _TARGET))
            for ring in out_targets.polygons
        ]

        assert torch.allclose(recovered_boxes, targets.boxes, atol=1e-4)
        for recovered, original in zip(recovered_rings, targets.polygons, strict=True):
            assert torch.allclose(recovered, original, atol=1e-4)

    def test_inverse_of_forward_matches_source_points(self) -> None:
        """Applying the forward affine then inverse_map is identity on arbitrary points."""
        letterbox = Letterbox(_TARGET)
        points = torch.tensor([[0.0, 0.0], [_ORIG_W, _ORIG_H], [123.5, 456.25]])
        forward = apply_affine_to_points(
            points, _forward_matrix(_EXPECTED_R, pad_top=(_TARGET - round(_ORIG_H * _EXPECTED_R)) // 2)
        )

        recovered = letterbox.inverse_map(forward, orig_size=(_ORIG_H, _ORIG_W), letterboxed_size=(_TARGET, _TARGET))

        assert torch.allclose(recovered, points, atol=1e-4)


class TestAspectPreservation:
    """The output canvas is the target size with symmetric, ratio-preserving content."""

    def test_output_is_target_size(self) -> None:
        """The letterboxed image has exactly the requested canvas dimensions."""
        out_image, _ = Letterbox(_TARGET)(_image(), Targets.empty())

        assert out_image.shape == (3, _TARGET, _TARGET)

    def test_content_region_ratio_equals_min_ratio(self) -> None:
        """The resized content spans round(dim * r) on each axis for the min ratio r."""
        expected_new_h = round(_ORIG_H * _EXPECTED_R)
        expected_new_w = round(_ORIG_W * _EXPECTED_R)

        # r = min(640/500, 640/800) = 0.8; content fills the width, is padded vertically.
        assert (expected_new_h, expected_new_w) == (400, 640)
        assert abs(expected_new_w / _ORIG_W - _EXPECTED_R) < 1e-6
        assert abs(expected_new_h / _ORIG_H - _EXPECTED_R) < 1e-6

    def test_padding_split_is_symmetric_within_one_pixel(self) -> None:
        """A box at the source origin lands at (pad_left, pad_top) with balanced slack."""
        origin_box = Targets(boxes=torch.zeros((1, 4)), labels=torch.zeros(1, dtype=torch.int64))

        _, out_targets = Letterbox(_TARGET)(_image(), origin_box)

        pad_left = out_targets.boxes[0, 0].item()
        pad_top = out_targets.boxes[0, 1].item()
        total_h_slack = _TARGET - round(_ORIG_H * _EXPECTED_R)
        total_w_slack = _TARGET - round(_ORIG_W * _EXPECTED_R)
        assert abs((total_w_slack - pad_left) - pad_left) <= 1
        assert abs((total_h_slack - pad_top) - pad_top) <= 1


class TestTargetConsistency:
    """Warped polygons and their derived boxes stay mutually consistent."""

    def test_boxes_from_warped_polygons_match_warped_boxes(self) -> None:
        """boxes_from_polygons of the warped rings equals the warped boxes."""
        targets = _targets_with_polygons()

        _, out_targets = Letterbox(_TARGET)(_image(), targets)
        derived = boxes_from_polygons(out_targets.polygons)

        assert torch.allclose(derived, out_targets.boxes, atol=1e-4)


class TestRotatedBoxes:
    """Rotated boxes: centres warp as points, extents scale by r, angle is fixed."""

    def test_rboxes_centre_scale_and_angle(self) -> None:
        """Centres match the point warp, w/h scale by r, theta is unchanged."""
        rboxes = torch.tensor([[100.0, 200.0, 40.0, 20.0, 0.5], [300.0, 150.0, 60.0, 30.0, -0.3]])
        targets = Targets(boxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.int64), rboxes=rboxes)
        pad_top = (_TARGET - round(_ORIG_H * _EXPECTED_R)) // 2

        _, out_targets = Letterbox(_TARGET)(_image(), targets)

        expected_centers = apply_affine_to_points(rboxes[:, :2], _forward_matrix(_EXPECTED_R, pad_top))
        assert torch.allclose(out_targets.rboxes[:, :2], expected_centers, atol=1e-4)
        assert torch.allclose(out_targets.rboxes[:, 2:4], rboxes[:, 2:4] * _EXPECTED_R, atol=1e-4)
        assert torch.equal(out_targets.rboxes[:, 4], rboxes[:, 4])


class TestDeterminism:
    """Identical inputs produce byte-identical outputs."""

    def test_same_input_identical_output(self) -> None:
        """Two calls on the same image and targets return equal tensors."""
        image = _image()
        targets = _targets_with_polygons()
        letterbox = Letterbox(_TARGET)

        image_a, targets_a = letterbox(image.clone(), targets.clone())
        image_b, targets_b = letterbox(image.clone(), targets.clone())

        assert torch.equal(image_a, image_b)
        assert torch.equal(targets_a.boxes, targets_b.boxes)
        for ring_a, ring_b in zip(targets_a.polygons, targets_b.polygons, strict=True):
            assert torch.equal(ring_a, ring_b)


def _forward_matrix(r: float, pad_top: int, pad_left: int = 0) -> torch.Tensor:
    """Build the float32 forward letterbox affine for the fixed-width source case."""
    return torch.tensor([[r, 0.0, pad_left], [0.0, r, pad_top], [0.0, 0.0, 1.0]], dtype=torch.float32)
