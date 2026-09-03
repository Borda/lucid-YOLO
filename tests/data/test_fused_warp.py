# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-070 fused affine+letterbox single warp, delegated in WP-156.

The fusion is now ``RandomAffine(letterbox=...)``: the affine and the
aspect-preserving fit are two transforms in one upstream segment, composed into
one matrix and one resample. The local class that used to hold that composition
is gone, and this file follows the behaviour rather than the class.

Covers the output contract (shape/dtype/value range), hand-computed box mapping
for identity-affine cases on both the mosaic-square and single-image branches,
byte-for-byte equivalence of the targets against the two-transform
``RandomAffine`` then ``Letterbox`` path (the fusion changes only the image
resampling, never the geometry), the rotated-box guard, and determinism under a
seeded generator.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from lucid_yolo.data import (
    Letterbox,
    RandomAffine,
    Targets,
    boxes_from_polygons,
)
from lucid_yolo.data.transforms import GeometricTransform


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed every RNG source before each test so generated geometry is deterministic."""
    torch.manual_seed(0)
    yield


def _generator(seed: int = 1234) -> torch.Generator:
    """Return a CPU generator seeded to ``seed`` for reproducible sampling.

    Examples:
        >>> _generator(7).initial_seed()
        7
    """
    return torch.Generator().manual_seed(seed)


def _polygon_targets(canvas: int) -> Targets:
    """Build three polygon rings inside a ``canvas``-sized square, with matching boxes.

    Examples:
        >>> targets = _polygon_targets(64)
        >>> targets.boxes.shape, targets.labels.tolist(), len(targets.polygons)
        (torch.Size([3, 4]), [0, 1, 2], 3)
    """
    rings = [
        torch.tensor([[10.0, 12.0], [30.0, 12.0], [30.0, 34.0], [10.0, 34.0]]),
        torch.tensor([[40.0, 20.0], [55.0, 20.0], [55.0, 50.0], [40.0, 50.0]]),
        torch.tensor([[5.0, 5.0], [canvas - 3.0, 8.0], [canvas - 5.0, canvas - 4.0], [7.0, canvas - 6.0]]),
    ]
    boxes = boxes_from_polygons(rings)
    labels = torch.arange(len(rings), dtype=torch.int64)
    return Targets(boxes=boxes, labels=labels, polygons=rings)


class TestProtocol:
    """The fused transform conforms to the geometric-transform protocol."""

    def test_is_geometric_transform(self) -> None:
        """A letterboxing RandomAffine satisfies the runtime GeometricTransform protocol."""
        assert isinstance(RandomAffine(letterbox=32), GeometricTransform)


class TestOutputContract:
    """The emitted image has the target shape, float32 dtype and in-range values."""

    @pytest.mark.parametrize(
        ("in_h", "in_w"),
        [
            pytest.param(256, 256, id="mosaic-square"),
            pytest.param(200, 360, id="single-wide"),
            pytest.param(360, 200, id="single-tall"),
            pytest.param(90, 120, id="single-upscale"),
        ],
    )
    def test_image_shape_dtype_range(self, in_h: int, in_w: int) -> None:
        """Any source size letterboxes to a square float32 image within the fill range."""
        fused = RandomAffine(degrees=10.0, translate=0.1, scale=0.2, generator=_generator(), letterbox=128)

        out_image, _ = fused(torch.rand(3, in_h, in_w), Targets.empty())

        assert out_image.shape == (3, 128, 128)
        assert out_image.dtype == torch.float32
        assert float(out_image.min()) >= 0.0 and float(out_image.max()) <= 1.0


class TestKnownBoxMapping:
    """Identity-affine cases map boxes to hand-computed letterboxed coordinates."""

    def test_single_image_scale_one_symmetric_pad(self) -> None:
        """A 4x8 source into an 8x8 canvas keeps x and shifts y by the top pad of 2."""
        fused = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, letterbox=8)
        # r = min(8/4, 8/8) = 1.0; new = (4, 8); pad_top = (8 - 4) // 2 = 2, pad_left = 0.
        targets = Targets(boxes=torch.tensor([[1.0, 1.0, 5.0, 3.0]]), labels=torch.tensor([0]))

        _, out = fused(torch.rand(3, 4, 8), targets)

        assert torch.equal(out.boxes, torch.tensor([[1.0, 3.0, 5.0, 5.0]]))

    def test_mosaic_square_pure_half_scale(self) -> None:
        """An 8x8 source into a 4x4 canvas is a pure 0.5 scale with no padding."""
        fused = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, letterbox=4)
        # r = min(4/8, 4/8) = 0.5; new = (4, 4); pad = 0 on both axes.
        targets = Targets(boxes=torch.tensor([[2.0, 2.0, 6.0, 6.0]]), labels=torch.tensor([0]))

        _, out = fused(torch.rand(3, 8, 8), targets)

        assert torch.equal(out.boxes, torch.tensor([[1.0, 1.0, 3.0, 3.0]]))


class TestTwoStepEquivalence:
    """Fused targets equal the two-transform affine-then-letterbox targets exactly."""

    @pytest.mark.parametrize(
        ("in_h", "in_w"),
        [
            pytest.param(256, 256, id="mosaic-square"),
            pytest.param(200, 360, id="single-wide"),
            pytest.param(360, 200, id="single-tall"),
            pytest.param(90, 120, id="single-upscale"),
        ],
    )
    def test_targets_match_two_step_path(self, in_h: int, in_w: int) -> None:
        """Boxes, polygons and labels are byte-identical to RandomAffine then Letterbox."""
        image = torch.rand(3, in_h, in_w)
        targets = _polygon_targets(min(in_h, in_w))
        kwargs = {"degrees": 12.0, "translate": 0.1, "scale": 0.3, "shear": 4.0}

        two_affine = RandomAffine(**kwargs, generator=_generator(7))
        warped_image, warped_targets = two_affine(image.clone(), targets.clone())
        _, two_targets = Letterbox(128)(warped_image, warped_targets)
        fused = RandomAffine(**kwargs, generator=_generator(7), letterbox=128)
        _, fused_targets = fused(image.clone(), targets.clone())

        assert torch.equal(fused_targets.boxes, two_targets.boxes)
        assert torch.equal(fused_targets.labels, two_targets.labels)
        for got, want in zip(fused_targets.polygons, two_targets.polygons, strict=True):
            assert torch.equal(got, want)


class TestRotatedBoxesGuard:
    """The fused warp inherits the affine's instance-axis requirement (WP-056)."""

    def test_unpaired_rboxes_raise_value_error(self) -> None:
        """Rotated boxes that do not share the instance axis with boxes are rejected."""
        rboxes = torch.tensor([[10.0, 20.0, 8.0, 4.0, 0.3]])
        targets = Targets(boxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.int64), rboxes=rboxes)
        fused = RandomAffine(degrees=10.0, letterbox=64)

        with pytest.raises(ValueError, match="instance axis"):
            fused(torch.rand(3, 96, 96), targets)
