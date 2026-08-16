# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-010 random-affine transform (blueprint section 5.9).

Covers the box/mask consistency contract (the DoD), the identity transform,
recovery of a seeded translation on both the image and the box centres, canvas
clipping (partial clip and fully-outside drop with label alignment), visibility
filtering, the rotated-box guard, and byte-for-byte determinism under a seeded
generator.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from lucid_yolo.data import RandomAffine, Targets, boxes_from_polygons

#: Square canvas used across the suite.
_CANVAS = 64


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed every RNG source before each test so generated geometry is deterministic."""
    torch.manual_seed(0)
    yield


def _generator(seed: int = 1234) -> torch.Generator:
    """Return a CPU generator seeded to ``seed`` for reproducible sampling."""
    return torch.Generator().manual_seed(seed)


def _image() -> torch.Tensor:
    """Return a random CHW float image on the canvas."""
    return torch.rand(3, _CANVAS, _CANVAS)


def _polygon_targets() -> Targets:
    """Build two square polygon rings well inside the canvas, with matching boxes."""
    rings = [
        torch.tensor([[10.0, 12.0], [30.0, 12.0], [30.0, 34.0], [10.0, 34.0]]),
        torch.tensor([[40.0, 20.0], [55.0, 20.0], [55.0, 50.0], [40.0, 50.0]]),
    ]
    boxes = boxes_from_polygons(rings)
    labels = torch.arange(len(rings), dtype=torch.int64)
    return Targets(boxes=boxes, labels=labels, polygons=rings)


class TestBoxMaskConsistency:
    """Recomputed boxes track the clipped polygons exactly (the WP-010 DoD)."""

    def test_box_mask_consistency(self) -> None:
        """After a rotation+scale warp, boxes equal boxes_from_polygons(output rings)."""
        affine = RandomAffine(degrees=25.0, translate=0.0, scale=0.2, shear=0.0, generator=_generator())

        _, out = affine(_image(), _polygon_targets())

        derived = boxes_from_polygons(out.polygons)
        assert torch.equal(out.boxes, derived)
        for ring, box in zip(out.polygons, out.boxes, strict=True):
            assert torch.all(ring[:, 0] >= box[0] - 1e-4) and torch.all(ring[:, 0] <= box[2] + 1e-4)
            assert torch.all(ring[:, 1] >= box[1] - 1e-4) and torch.all(ring[:, 1] <= box[3] + 1e-4)


class TestIdentity:
    """A fully-zeroed transform leaves image and geometry unchanged."""

    def test_identity_leaves_boxes_exact(self) -> None:
        """Zero rotation/translation/scale/shear returns the input boxes exactly."""
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0)
        targets = _polygon_targets()

        _, out = affine(_image(), targets)

        assert torch.equal(out.boxes, targets.boxes)
        for got, want in zip(out.polygons, targets.polygons, strict=True):
            assert torch.equal(got, want)

    def test_identity_leaves_image_within_tolerance(self) -> None:
        """The identity grid_sample reproduces the image within interpolation tolerance."""
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0)
        image = _image()

        out_image, _ = affine(image, Targets.empty())

        assert torch.allclose(out_image, image, atol=1e-5)


class TestKnownTranslation:
    """A seeded translation-only warp shifts image content and box centres alike."""

    def test_box_centres_shift_by_sampled_offset(self) -> None:
        """Box centres move by exactly the sampled (tx, ty) recovered from last_params."""
        affine = RandomAffine(degrees=0.0, translate=0.15, scale=0.0, shear=0.0, generator=_generator())
        targets = _polygon_targets()
        in_centres = torch.stack(
            [(targets.boxes[:, 0] + targets.boxes[:, 2]) / 2, (targets.boxes[:, 1] + targets.boxes[:, 3]) / 2], dim=1
        )

        _, out = affine(_image(), targets)

        assert affine.last_params is not None
        offset = torch.tensor([affine.last_params.translate_x, affine.last_params.translate_y])
        out_centres = torch.stack(
            [(out.boxes[:, 0] + out.boxes[:, 2]) / 2, (out.boxes[:, 1] + out.boxes[:, 3]) / 2], dim=1
        )
        assert torch.allclose(out_centres - in_centres, offset.expand_as(out_centres), atol=1e-4)

    def test_bright_pixel_shifts_same_direction(self) -> None:
        """A single bright pixel lands near centre + sampled offset (content follows boxes)."""
        affine = RandomAffine(degrees=0.0, translate=0.15, scale=0.0, shear=0.0, generator=_generator())
        image = torch.full((1, _CANVAS, _CANVAS), 114.0 / 255.0)
        row0, col0 = _CANVAS // 2, _CANVAS // 2
        image[0, row0, col0] = 1.0

        out_image, _ = affine(image, Targets.empty())

        assert affine.last_params is not None
        flat_idx = int(out_image[0].argmax())
        peak_row, peak_col = divmod(flat_idx, _CANVAS)
        assert abs(peak_col - (col0 + affine.last_params.translate_x)) <= 1.0
        assert abs(peak_row - (row0 + affine.last_params.translate_y)) <= 1.0


class TestClipping:
    """Boxes crossing the canvas edge are clamped; fully-outside instances are dropped."""

    def test_partial_box_clipped_to_canvas(self) -> None:
        """A box translated off the right edge is clamped to the canvas width."""
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0)
        targets = Targets(boxes=torch.tensor([[50.0, 10.0, 80.0, 40.0]]), labels=torch.tensor([0]))

        _, out = affine(_image(), targets)

        assert out.boxes.shape[0] == 1
        assert out.boxes[0, 2].item() == pytest.approx(_CANVAS)
        assert out.boxes[0, 0].item() == pytest.approx(50.0)

    def test_fully_outside_dropped_and_labels_aligned(self) -> None:
        """An off-canvas instance is removed while surviving labels stay aligned."""
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0)
        targets = Targets(
            boxes=torch.tensor([[5.0, 5.0, 25.0, 25.0], [200.0, 200.0, 240.0, 240.0]]),
            labels=torch.tensor([7, 9]),
        )

        _, out = affine(_image(), targets)

        assert out.boxes.shape[0] == 1
        assert out.labels.tolist() == [7]


class TestVisibilityFilter:
    """Instances whose kept-area fraction falls below the threshold are removed."""

    def test_mostly_clipped_instance_removed(self) -> None:
        """A box with only a thin sliver inside the canvas is dropped by min_visibility."""
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, min_visibility=0.5)
        targets = Targets(boxes=torch.tensor([[60.0, 10.0, 160.0, 40.0]]), labels=torch.tensor([0]))

        _, out = affine(_image(), targets)

        assert out.boxes.shape[0] == 0


class TestRotatedBoxesGuard:
    """The rotated path filters, so it demands WP-056's instance-axis invariant."""

    def test_unpaired_rboxes_raise_value_error(self) -> None:
        """Rotated boxes that do not share the instance axis with boxes are rejected."""
        rboxes = torch.tensor([[10.0, 20.0, 8.0, 4.0, 0.3]])
        targets = Targets(boxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.int64), rboxes=rboxes)
        affine = RandomAffine(degrees=10.0)

        with pytest.raises(ValueError, match="instance axis"):
            affine(_image(), targets)


class TestWarpTo:
    """warp_to composes a post-affine into the image warp, leaving targets at canvas scale."""

    def test_identity_post_matrix_matches_call_targets(self) -> None:
        """An identity post-affine to the same canvas reproduces __call__'s targets."""
        image = _image()
        targets = _polygon_targets()
        identity = torch.eye(3, dtype=torch.float64)
        call_affine = RandomAffine(degrees=12.0, translate=0.1, scale=0.2, shear=3.0, generator=_generator(5))
        warp_affine = RandomAffine(degrees=12.0, translate=0.1, scale=0.2, shear=3.0, generator=_generator(5))

        _, call_targets = call_affine(image.clone(), targets.clone())
        _, warp_targets = warp_affine.warp_to(image.clone(), targets.clone(), identity, _CANVAS, _CANVAS)

        assert torch.equal(warp_targets.boxes, call_targets.boxes)
        for got, want in zip(warp_targets.polygons, call_targets.polygons, strict=True):
            assert torch.equal(got, want)

    def test_post_matrix_downscales_image_to_output_size(self) -> None:
        """A half-scale post-affine emits a half-size image in one resample."""
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0)
        half = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
        out = _CANVAS // 2

        out_image, _ = affine.warp_to(_image(), Targets.empty(), half, out, out)

        assert out_image.shape == (3, out, out)


class TestDeterminism:
    """Two seeded generators with the same seed give byte-identical outputs."""

    def test_same_seed_identical_outputs(self) -> None:
        """Equal-seed generators produce equal warped images and boxes."""
        image = _image()
        targets = _polygon_targets()
        affine_a = RandomAffine(degrees=15.0, translate=0.1, scale=0.2, shear=5.0, generator=_generator(99))
        affine_b = RandomAffine(degrees=15.0, translate=0.1, scale=0.2, shear=5.0, generator=_generator(99))

        image_a, targets_a = affine_a(image.clone(), targets.clone())
        image_b, targets_b = affine_b(image.clone(), targets.clone())

        assert torch.equal(image_a, image_b)
        assert torch.equal(targets_a.boxes, targets_b.boxes)
        for ring_a, ring_b in zip(targets_a.polygons, targets_b.polygons, strict=True):
            assert torch.equal(ring_a, ring_b)
