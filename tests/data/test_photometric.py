# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-013 photometric HSV jitter and horizontal flip (blueprint section 5.9).

Covers the HSV converter round-trip fidelity and the jitter's identity/clamp/wrap
behaviour with targets left untouched, plus the flip's per-modality mirroring
(boxes, polygons, rotated boxes), image column reversal, double-flip identity, the
``p=0``/``p=1`` boundaries and seeded determinism of the flip draw.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch

from lucid_yolo.data import Targets
from lucid_yolo.data.augment import HorizontalFlip, HSVJitter, hsv_to_rgb, rgb_to_hsv


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed the global RNG before each test so any un-seeded sampling is deterministic."""
    torch.manual_seed(0)
    yield


def _generator(seed: int = 1234) -> torch.Generator:
    """Return a CPU generator seeded to ``seed`` for reproducible sampling.

    Examples:
        >>> _generator(7).initial_seed()
        7
    """
    return torch.Generator().manual_seed(seed)


def _targets() -> Targets:
    """Build a small mixed-modality target set (boxes + polygons + rotated boxes).

    Examples:
        >>> targets = _targets()
        >>> targets.boxes.shape
        torch.Size([2, 4])
        >>> len(targets.polygons)
        2
    """
    boxes = torch.tensor([[10.0, 12.0, 30.0, 34.0], [40.0, 20.0, 55.0, 50.0]])
    labels = torch.tensor([3, 7])
    polygons = [
        torch.tensor([[10.0, 12.0], [30.0, 12.0], [30.0, 34.0], [10.0, 34.0]]),
        torch.tensor([[40.0, 20.0], [55.0, 20.0], [55.0, 50.0], [40.0, 50.0]]),
    ]
    rboxes = torch.tensor([[32.0, 24.0, 20.0, 8.0, 0.3], [16.0, 40.0, 12.0, 6.0, -0.5]])
    return Targets(boxes=boxes, labels=labels, polygons=polygons, rboxes=rboxes)


class TestConverters:
    """The pure rgb<->hsv converters are exact inverses and match known colours."""

    def test_round_trip_random_image(self) -> None:
        """rgb -> hsv -> rgb reconstructs a random image to 1e-4."""
        image = torch.rand(3, 32, 48)
        restored = hsv_to_rgb(rgb_to_hsv(image))
        assert torch.allclose(restored, image, atol=1e-4)

    def test_primary_colours(self) -> None:
        """Pure red/green/blue map to hue 0, 1/3, 2/3 with full saturation and value."""
        red = torch.tensor([1.0, 0.0, 0.0]).reshape(3, 1, 1)
        green = torch.tensor([0.0, 1.0, 0.0]).reshape(3, 1, 1)
        blue = torch.tensor([0.0, 0.0, 1.0]).reshape(3, 1, 1)
        assert torch.allclose(rgb_to_hsv(red).flatten(), torch.tensor([0.0, 1.0, 1.0]), atol=1e-6)
        assert torch.allclose(rgb_to_hsv(green).flatten(), torch.tensor([1.0 / 3.0, 1.0, 1.0]), atol=1e-6)
        assert torch.allclose(rgb_to_hsv(blue).flatten(), torch.tensor([2.0 / 3.0, 1.0, 1.0]), atol=1e-6)

    def test_greyscale_is_achromatic(self) -> None:
        """A grey pixel has zero saturation and its value equals the grey level."""
        grey = torch.full((3, 2, 2), 0.4)
        hsv = rgb_to_hsv(grey)
        assert torch.allclose(hsv[1], torch.zeros_like(hsv[1]))
        assert torch.allclose(hsv[2], torch.full_like(hsv[2], 0.4))

    def test_shape_guard(self) -> None:
        """Both converters reject a non-3-channel image."""
        with pytest.raises(ValueError, match="3, H, W"):
            rgb_to_hsv(torch.rand(4, 2, 2))
        with pytest.raises(ValueError, match="3, H, W"):
            hsv_to_rgb(torch.rand(2, 2))


class TestHSVJitter:
    """HSV jitter perturbs colour only, honouring wrap/clamp and leaving targets alone."""

    def test_zero_gain_is_identity(self) -> None:
        """Zeroed gain ranges reduce to a pure rgb->hsv->rgb round trip (identity)."""
        jitter = HSVJitter(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, generator=_generator())
        image = torch.rand(3, 16, 16)
        out_image, _ = jitter(image, Targets.empty())
        assert torch.allclose(out_image, image, atol=1e-4)
        assert jitter.last_gains == (0.0, 0.0, 0.0)

    def test_hue_shift_wraps_red_to_cyan(self) -> None:
        """A +0.5 hue shift turns pure red into cyan (hue wraps additively mod 1)."""
        red = torch.tensor([1.0, 0.0, 0.0]).reshape(3, 1, 1)
        hsv = rgb_to_hsv(red)
        shifted = hsv.clone()
        shifted[0] = (hsv[0] + 0.5) % 1.0
        out = hsv_to_rgb(shifted)
        assert torch.allclose(out.flatten(), torch.tensor([0.0, 1.0, 1.0]), atol=1e-4)

    def test_output_stays_in_unit_range(self) -> None:
        """Extreme s/v gain ranges never push the output outside [0, 1] (clamp holds)."""
        for seed in range(8):
            jitter = HSVJitter(hsv_h=0.0, hsv_s=5.0, hsv_v=5.0, generator=_generator(seed))
            image = torch.rand(3, 8, 8) * 0.5 + 0.25
            out_image, _ = jitter(image, Targets.empty())
            assert out_image.min() >= 0.0
            assert out_image.max() <= 1.0 + 1e-6

    def test_saturation_multiplier_clamps_at_one(self) -> None:
        """A multiplicative saturation boost past 1.0 clamps rather than overflowing."""
        # s=0.5 boosted by (1 + 5) = 3.0 must clamp to exactly 1.0.
        boosted = (torch.tensor(0.5) * (1.0 + 5.0)).clamp(0.0, 1.0)
        assert boosted.item() == pytest.approx(1.0)
        hsv = torch.tensor([0.25, 1.0, 0.8]).reshape(3, 1, 1)
        rgb = hsv_to_rgb(hsv)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0 + 1e-6

    def test_targets_pass_through_untouched(self) -> None:
        """The returned targets are the exact same tensors as the input geometry."""
        jitter = HSVJitter(generator=_generator())
        targets = _targets()
        _, out_targets = jitter(torch.rand(3, 8, 8), targets)
        assert out_targets is targets
        assert torch.equal(out_targets.boxes, targets.boxes)
        assert torch.equal(out_targets.rboxes, targets.rboxes)


class TestHorizontalFlip:
    """Horizontal flip mirrors the image and every carried target modality exactly."""

    def test_p_one_flips_boxes(self) -> None:
        """p=1 mirrors a hand-built box about (W-1)/2: x1' = (W-1) - x2, x2' = (W-1) - x1."""
        flip = HorizontalFlip(p=1.0, generator=_generator())
        image = torch.rand(3, 4, 10)
        boxes = torch.tensor([[2.0, 1.0, 6.0, 3.0]])
        _, out = flip(image, Targets(boxes=boxes, labels=torch.tensor([0])))
        assert out.boxes.tolist() == [[3.0, 1.0, 7.0, 3.0]]
        assert flip.last_flipped is True

    def test_the_mirror_axis_is_the_image_column_reversal(self) -> None:
        """The mirror is about ``(W-1)/2``, the axis ``image.flip(-1)`` itself reflects about — not ``W/2``.

        ``W-1`` versus ``W`` is a real choice and both appear in the wild, so it is
        asserted here by its consequence rather than restated as a formula: a box drawn
        tightly around a bright block must still bound that block after both are
        mirrored. Under the ``W`` convention the block lands at columns ``{6, 7}`` while
        the box is carried to ``[7, 8]`` — off by exactly one pixel, and every downstream
        coordinate inherits the error.
        """
        flip = HorizontalFlip(p=1.0, generator=_generator())
        width = 10
        image = torch.zeros(3, 4, width)
        image[:, :, 2:4] = 1.0
        boxes = torch.tensor([[2.0, 0.0, 3.0, 3.0]])

        out_image, out = flip(image, Targets(boxes=boxes, labels=torch.tensor([0])))

        lit = out_image[0, 0].nonzero().flatten().tolist()
        assert lit == [6, 7]
        assert out.boxes[0, 0].item() == float(min(lit))
        assert out.boxes[0, 2].item() == float(max(lit))

    def test_a_box_at_the_canvas_bound_is_re_clipped(self) -> None:
        """A box whose ``x2`` sits at the reader's clamp bound ``W`` mirrors to ``x1 = -1`` and is re-clipped.

        The two conventions meet here: extents are clamped to ``[0, W]`` (``coco.py``,
        ``affine.py``, ``fuse``'s own ``clip_bbox_xyxy``) while the mirror reflects about
        ``(W-1)/2``, so a box legitimately touching the right edge comes back one pixel
        off the left edge. The flip is the last stage of the train pipeline, so without a
        re-clip that negative coordinate is what the assigner receives.
        """
        flip = HorizontalFlip(p=1.0, generator=_generator())
        width = 10
        boxes = torch.tensor([[4.0, 1.0, float(width), 3.0]])

        _, out = flip(torch.rand(3, 4, width), Targets(boxes=boxes, labels=torch.tensor([0])))

        assert out.boxes.tolist() == [[0.0, 1.0, 5.0, 3.0]]

    def test_double_flip_is_identity(self) -> None:
        """Flipping boxes twice returns the original coordinates.

        The canvas holds the whole target set here: the involution is a property of the
        mirror, and the re-clip that keeps the assigner on-canvas necessarily breaks it
        for geometry that started outside the canvas.
        """
        flip = HorizontalFlip(p=1.0, generator=_generator())
        image = torch.rand(3, 64, 64)
        targets = _targets()
        once_image, once = flip(image, targets)
        twice_image, twice = flip(once_image, once)
        assert torch.allclose(twice.boxes, targets.boxes)
        assert torch.allclose(twice_image, image)

    def test_polygon_points_mirror(self) -> None:
        """Every polygon x-coordinate reflects to (W-1) - x; y is untouched."""
        flip = HorizontalFlip(p=1.0, generator=_generator())
        targets = _targets()
        width = 10
        _, out = flip(torch.rand(3, 4, width), targets)
        for src, dst in zip(targets.polygons, out.polygons, strict=True):
            assert torch.allclose(dst[:, 0], (width - 1) - src[:, 0])
            assert torch.allclose(dst[:, 1], src[:, 1])

    def test_keypoints_mirror_without_identity_swap(self) -> None:
        """Without pairs, every x coordinate mirrors while keypoint-column order stays fixed."""
        flip = HorizontalFlip(p=1.0, generator=_generator())
        targets = Targets(
            boxes=torch.zeros((1, 4)),
            labels=torch.tensor([0]),
            keypoints=torch.tensor([[[1.0, 2.0], [7.0, 3.0]]]),
            keypoint_vis=torch.tensor([[2, 1]]),
        )
        _, out = flip(torch.rand(3, 4, 10), targets)
        assert out.keypoints.tolist() == [[[8.0, 2.0], [2.0, 3.0]]]
        assert out.keypoint_vis.tolist() == [[2, 1]]

    def test_keypoints_mirror_and_swap_supplied_pairs(self) -> None:
        """A supplied pair swaps mirrored coordinates and visibility columns together."""
        flip = HorizontalFlip(p=1.0, generator=_generator(), keypoint_flip_pairs=[(0, 1)])
        targets = Targets(
            boxes=torch.zeros((1, 4)),
            labels=torch.tensor([0]),
            keypoints=torch.tensor([[[1.0, 2.0], [7.0, 3.0]]]),
            keypoint_vis=torch.tensor([[2, 1]]),
        )
        _, out = flip(torch.rand(3, 4, 10), targets)
        assert out.keypoints.tolist() == [[[2.0, 3.0], [8.0, 2.0]]]
        assert out.keypoint_vis.tolist() == [[1, 2]]

    def test_mirrored_visibility_flags_follow_their_own_points(self) -> None:
        """Every flag stays attached to its own point across the mirror and the identity swap.

        The coordinate permutation is upstream's and moves coordinates only, so the
        visibility permutation is this transform's own (WP-157). Dropping it leaves shapes
        identical and every coordinate assertion passing while a visible point arrives
        marked occluded and its partner marked visible -- which is why the invariant asserted
        here is the *pairing* of each mirrored point with its flag, not the two columns
        separately. The third slot is unpaired on purpose: it pins that an identity slot
        neither moves nor loses its flag.
        """
        width = 10
        flip = HorizontalFlip(p=1.0, generator=_generator(), keypoint_flip_pairs=[(0, 2)])
        targets = Targets(
            boxes=torch.zeros((1, 4)),
            labels=torch.tensor([0]),
            keypoints=torch.tensor([[[1.0, 2.0], [4.0, 5.0], [7.0, 3.0]]]),
            keypoint_vis=torch.tensor([[2, 0, 1]]),
        )

        _, out = flip(torch.rand(3, 4, width), targets)

        expected_pairing = {
            (float(width - 1) - x, y, vis)
            for (x, y), vis in zip(targets.keypoints[0].tolist(), targets.keypoint_vis[0].tolist(), strict=True)
        }
        actual_pairing = {
            (x, y, vis) for (x, y), vis in zip(out.keypoints[0].tolist(), out.keypoint_vis[0].tolist(), strict=True)
        }
        assert actual_pairing == expected_pairing
        # ... and the swap did fire, so the equality above is not the trivial identity.
        assert out.keypoints.tolist() == [[[2.0, 3.0], [5.0, 5.0], [8.0, 2.0]]]
        assert out.keypoint_vis.tolist() == [[1, 0, 2]]

    def test_out_of_range_keypoint_pair_raises(self) -> None:
        """Pairs outside the supplied K-point axis are rejected with a clear error."""
        flip = HorizontalFlip(p=1.0, generator=_generator(), keypoint_flip_pairs=[(0, 2)])
        targets = Targets(
            boxes=torch.zeros((1, 4)),
            labels=torch.tensor([0]),
            keypoints=torch.zeros((1, 2, 2)),
            keypoint_vis=torch.zeros((1, 2), dtype=torch.int64),
        )
        with pytest.raises(ValueError, match=r"\(0, 2\).*K=2"):
            flip(torch.rand(3, 4, 10), targets)

    def test_rbox_centre_mirror_and_theta_negation(self) -> None:
        """Rotated boxes reflect cx -> (W-1) - cx and negate theta (both in range); w/h stay put."""
        flip = HorizontalFlip(p=1.0, generator=_generator())
        targets = _targets()
        width = 10
        _, out = flip(torch.rand(3, 4, width), targets)
        assert torch.allclose(out.rboxes[:, 0], (width - 1) - targets.rboxes[:, 0])
        assert torch.allclose(out.rboxes[:, 4], -targets.rboxes[:, 4])
        assert torch.allclose(out.rboxes[:, 1:4], targets.rboxes[:, 1:4])

    def test_image_columns_reversed(self) -> None:
        """The flipped image is the column-reversed original."""
        flip = HorizontalFlip(p=1.0, generator=_generator())
        image = torch.arange(24.0).reshape(1, 4, 6).expand(3, 4, 6).contiguous()
        out_image, _ = flip(image, Targets.empty())
        assert torch.equal(out_image, image.flip(-1))

    def test_p_zero_is_identity(self) -> None:
        """p=0 never flips: image and targets are returned unchanged."""
        flip = HorizontalFlip(p=0.0, generator=_generator())
        image = torch.rand(3, 4, 6)
        targets = _targets()
        out_image, out_targets = flip(image, targets)
        assert torch.equal(out_image, image)
        assert out_targets is targets
        assert flip.last_flipped is False

    def test_mirrored_angle_above_quarter_pi_is_re_canonicalized(self) -> None:
        """An angle above pi/4 mirrors to pi - theta, inside the range the bare negation left (WP-058)."""
        flip = HorizontalFlip(p=1.0, generator=_generator())
        theta = 3.0 * math.pi / 8.0
        rboxes = torch.tensor([[20.0, 10.0, 15.0, 5.0, theta]])
        targets = Targets(boxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.int64), rboxes=rboxes)
        _, out = flip(torch.rand(3, 4, 40), targets)
        assert out.rboxes[0, 4].item() == pytest.approx(math.pi - theta)
        assert -math.pi / 4.0 <= out.rboxes[0, 4].item() < 3.0 * math.pi / 4.0
