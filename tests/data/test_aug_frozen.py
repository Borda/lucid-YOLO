# SPDX-License-Identifier: Apache-2.0
"""Tier B frozen augmentation expectations: characterized, not derived (WP-149).

Every case here states its parameters explicitly, calls a transform's ``apply``
and asserts a literal result. **These values pin behaviour; they do not prove
it.** Each case names why no closed form was available -- a composite of
rotation, shear, scale and translation followed by a clip; a re-fit of a
rectangle under a transform that is not a similarity; a rasterised triangle; a
padding split that resolves a half-pixel by convention. Where a closed form does
exist, the case belongs in ``tests/data/test_aug_contract.py`` instead, and every
one that could be moved there has been.

A failure here therefore reads as *behaviour moved*, never as *the new code is
wrong*. Diagnosing it means deciding which of the two implementations is right,
and the answer is not in this file.

**Parameters, never seeds.** Nothing here touches a generator. An upstream engine
will not draw the same numbers from the same seed -- different call order,
different distributions, different consumption of the stream -- so a seed-keyed
expectation would break at a swap for reasons unrelated to correctness, and the
only repair available would be a re-freeze at exactly the moment the guard was
supposed to hold.

**Targets are exact; pixels are not.** Geometry is asserted to float tolerance
because it is computed through a matrix. Image expectations belong in
``goldens/aug_invariants.json`` as means and counts within a band, because the
upstream engine samples at a different half-pixel convention, may execute through
a different backend, and collapses several warps into one resample. A byte-exact
pixel expectation would fail at swap time for a *correct* implementation.

A frozen value may move only in its own work package, with what moved, from which
value to which, and why the new one is correct recorded in
``docs/ENGINEERING_LOG.md`` -- and never in the same commit as a call-site swap,
because that combination is indistinguishable from adjusting the test until the
new code passes.
"""

from __future__ import annotations

import torch

from lucid_yolo.data.affine import AffineParams, RandomAffine
from lucid_yolo.data.augment import FlipParams, HorizontalFlip, HSVJitter, HSVParams
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.mixup import CopyPaste, CopyPasteParams
from lucid_yolo.data.mosaic import MosaicAssembly, MosaicParams
from lucid_yolo.data.targets import Targets

#: Geometry tolerance: these coordinates come out of a float32 matrix product.
GEOM = 1e-4

#: The one composite affine every multi-parameter case states. Rotation, both
#: shears, a scale and a translation are all non-zero, so no single parameter can
#: be dropped without a case here moving.
COMPOSITE = AffineParams(angle=0.35, shear_x=0.12, shear_y=-0.08, scale=1.15, translate_x=2.5, translate_y=-1.5)


def _two_instance_scene() -> Targets:
    """Build the two-instance box-and-polygon scene the affine cases warp.

    Examples:
        ```pycon
        >>> _two_instance_scene().boxes.shape
        torch.Size([2, 4])

        ```
    """
    return Targets(
        boxes=torch.tensor([[6.0, 6.0, 18.0, 18.0], [20.0, 4.0, 28.0, 14.0]]),
        labels=torch.tensor([0, 1]),
        polygons=[
            torch.tensor([[6.0, 6.0], [18.0, 6.0], [18.0, 18.0], [6.0, 18.0]]),
            torch.tensor([[20.0, 4.0], [28.0, 4.0], [28.0, 14.0], [20.0, 14.0]]),
        ],
    )


class TestAffineComposite:
    """Tier B -- a rotation, two shears, a scale and a translation, then a clip."""

    def test_boxes_land_on_frozen_coordinates(self) -> None:
        """The two warped boxes are pinned to the coordinates today's code produces.

        No closed form: the envelope of a rectangle under a composite that includes
        shear is not the rectangle's own corners re-fitted, the canvas clip then
        truncates both boxes at different edges, and the result depends on the order
        the transform composes rotation, shear and translation about the centre.
        Deriving it by hand would reproduce the implementation rather than check it.
        """
        _, warped = RandomAffine().apply(torch.zeros(3, 32, 32), _two_instance_scene(), COMPOSITE)

        expected = torch.tensor([[6.7768, 0.3623, 23.2884, 17.5889], [23.3996, 2.4148, 32.0000, 16.1548]])
        assert torch.allclose(warped.boxes, expected, atol=GEOM)

    def test_a_polygon_ring_lands_on_frozen_vertices(self) -> None:
        """The first instance's ring is pinned vertex by vertex under the same composite.

        No closed form for the same reason as the box, with one addition: the ring
        carries four points through the matrix and is then clipped against the
        canvas, which can add vertices. Freezing the ring rather than only its
        derived box is what would catch a ring path that drifted from the box path
        while both stayed self-consistent.
        """
        single = Targets(
            boxes=torch.tensor([[6.0, 6.0, 18.0, 18.0]]),
            labels=torch.tensor([0]),
            polygons=[torch.tensor([[6.0, 6.0], [18.0, 6.0], [18.0, 18.0], [6.0, 18.0]])],
        )

        _, warped = RandomAffine().apply(torch.zeros(3, 32, 32), single, COMPOSITE)

        expected = torch.tensor(
            [[9.945715, 0.362253], [23.288427, 4.054957], [20.119549, 17.588881], [6.776836, 13.896176]]
        )
        assert torch.allclose(warped.polygons[0], expected, atol=GEOM)


class TestAffineOntoLetterboxCanvas:
    """Tier B -- the same composite, then an aspect-preserving downscale, in one resample."""

    def test_boxes_land_on_frozen_output_canvas_coordinates(self) -> None:
        """The fused path's output-canvas boxes are pinned.

        No closed form: this composes the affine case's composite with a letterbox
        whose ratio and padding are resolved per sample from the source size, so the
        result is a product of two matrices neither of which has a hand-checkable
        envelope after the intermediate clip.
        """
        _, warped = RandomAffine(letterbox=24).apply(torch.zeros(3, 32, 32), _two_instance_scene(), COMPOSITE)

        expected = torch.tensor([[5.0826, 0.2717, 17.4663, 13.1917], [17.5497, 1.8111, 24.0000, 12.1161]])
        assert torch.allclose(warped.boxes, expected, atol=GEOM)


class TestLetterboxOddPadding:
    """Tier B -- how a half-pixel of leftover padding is split between the two sides."""

    def test_an_odd_leftover_puts_the_extra_pixel_below(self) -> None:
        """A 10x20 source into a 15x15 canvas scales by 0.75 and offsets by 3, not 3.75.

        Not derived, because it is not derivable: the ratio 0.75 leaves 7.5 rows of
        padding, and how that half pixel is resolved -- floor, ceil, or split
        unevenly between the two sides -- is a convention rather than a consequence.
        This value is what every prediction's inverse map depends on, so a change to
        it moves every reported box by a pixel.
        """
        matrix, out_h, out_w = Letterbox(15).forward_affine(10, 20)

        assert (out_h, out_w) == (15, 15)
        assert torch.allclose(matrix[:2, :2], torch.tensor([[0.75, 0.0], [0.0, 0.75]], dtype=matrix.dtype), atol=GEOM)
        assert torch.allclose(matrix[:2, 2], torch.tensor([0.0, 3.0], dtype=matrix.dtype), atol=GEOM)


class TestMosaicOffCentre:
    """Tier B -- an off-centre stitch, where placement meets the canvas clip and the filter."""

    def test_an_off_centre_stitch_keeps_eight_frozen_boxes(self) -> None:
        """A centre at ``(26, 38)`` on four 32-pixel images yields eight boxes at frozen coordinates.

        The four offsets are derivable and the tier-A case covers them. What is not
        derivable is the interaction pinned here: two of the four quadrants push
        instances past a canvas edge, so the clip truncates them and the
        ``min_box_size`` / ``min_visibility`` filter then decides which truncated
        instances survive. Eight of eight surviving is a fact about those thresholds
        against this geometry, not about the placement arithmetic.
        """
        items = [(torch.zeros(3, 32, 32), _two_instance_scene()) for _ in range(4)]

        _, merged = MosaicAssembly(target_size=32).apply(items, MosaicParams(center_x=26, center_y=38))

        expected = torch.tensor(
            [
                [0.0, 12.0, 12.0, 24.0],
                [14.0, 10.0, 22.0, 20.0],
                [32.0, 12.0, 44.0, 24.0],
                [46.0, 10.0, 54.0, 20.0],
                [0.0, 44.0, 12.0, 56.0],
                [14.0, 42.0, 22.0, 52.0],
                [32.0, 44.0, 44.0, 56.0],
                [46.0, 42.0, 54.0, 52.0],
            ]
        )
        assert merged.boxes.shape[0] == 8
        assert torch.allclose(merged.boxes, expected, atol=GEOM)


class TestPhotometricFrozen:
    """Tier B -- a hue rotation on a saturated colour, where the conversion pair has no shortcut."""

    def test_a_hue_rotation_on_pure_red_lands_on_frozen_rgb(self) -> None:
        """Red at 0.8 under gains ``(+0.06 hue, -0.25 saturation, +0.2 value)`` freezes to a literal RGB.

        The value and saturation halves are multiplicative and derivable on a pure
        primary. The hue half is not: rotating hue by a fraction of the circle moves
        the colour into a different RGB sector, and the output is a piecewise
        function of where in that sector it lands. This is the case that would catch
        an asymmetry between the RGB-to-HSV and HSV-to-RGB conversions, which cancel
        exactly at zero gain and do not cancel here.
        """
        red = torch.zeros(3, 2, 2)
        red[0] = 0.8

        jittered, _ = HSVJitter().apply(red, Targets.empty(), HSVParams(hue=0.1, saturation=-0.25, value=0.2))

        assert torch.allclose(jittered[:, 0, 0], torch.tensor([0.96, 0.672, 0.24]), atol=1e-5)


class TestRotatedUnderShear:
    """Tier B -- an rbox re-fitted under a transform that is not a similarity."""

    def test_a_sheared_rotated_box_lands_on_frozen_parameters(self) -> None:
        """The composite's shear makes the re-fit lossy, and the five numbers are pinned.

        No closed form, and this is the sharpest example of why the tier exists: a
        similarity maps a rectangle to a rectangle, so a pure rotation re-fits
        exactly and belongs in tier A. A shear maps it to a parallelogram, and the
        re-fit is then a choice of enclosing rectangle rather than a recovery of the
        original. What that choice is, is the implementation's, and this pins it in
        canonical long-edge form.
        """
        targets = Targets(
            boxes=torch.tensor([[8.0, 9.0, 20.0, 15.0]]),
            labels=torch.tensor([0]),
            rboxes=torch.tensor([[14.0, 12.0, 12.0, 6.0, 0.35]]),
        )

        _, warped = RandomAffine().apply(torch.zeros(3, 32, 32), targets, COMPOSITE)

        expected = torch.tensor([[17.256418, 9.591017, 14.028654, 6.856925, 0.616364]])
        assert torch.allclose(warped.rboxes, expected, atol=GEOM)


class TestKeypointsThroughMirrorAndWarp:
    """Tier B -- flip pairs composed with the affine composite, on the keypoint axis."""

    def test_mirrored_then_warped_keypoints_land_on_frozen_coordinates(self) -> None:
        """A mirror with a flip-pair swap followed by the composite freezes three points.

        The mirror alone is derivable and tier A covers it. What is pinned here is
        the composition: the swap reorders the keypoint axis, and the composite then
        moves each point by a matrix that includes shear, so the frozen coordinates
        are keyed to *both* the permutation and the warp. A regression in either --
        a swap that stopped firing, or a keypoint path that drifted from the image
        path -- moves these numbers.
        """
        targets = Targets(
            boxes=torch.tensor([[6.0, 6.0, 18.0, 18.0]]),
            labels=torch.tensor([0]),
            keypoints=torch.tensor([[[8.0, 8.0], [16.0, 16.0], [12.0, 7.0]]]),
            keypoint_vis=torch.tensor([[2, 2, 1]]),
        )
        flip = HorizontalFlip(keypoint_flip_pairs=[(0, 1)])

        _, mirrored = flip.apply(torch.zeros(3, 32, 32), targets, FlipParams(flipped=True))
        _, warped = RandomAffine().apply(torch.zeros(3, 32, 32), mirrored, COMPOSITE)

        expected = torch.tensor([[[17.312017, 14.410050], [28.319744, 7.849238], [24.136246, 5.490510]]])
        assert torch.allclose(warped.keypoints, expected, atol=GEOM)
        assert warped.keypoint_vis.tolist() == [[2, 2, 1]]


class TestCopyPasteRasterisedTriangle:
    """Tier B -- how many pixels a non-axis-aligned ring covers."""

    def test_a_triangle_ring_writes_a_frozen_pixel_count(self) -> None:
        """The triangle ``(2,2)-(10,3)-(6,11)`` rasterises to exactly 32 pixels on a 16-pixel canvas.

        No closed form: an axis-aligned ring's pixel count follows from A11's
        half-open rule and tier A states it, but a triangle's does not -- every
        sloped edge resolves against the pixel grid one row at a time, and the count
        is the sum of those resolutions. Its analytic area is 34, so the two-pixel
        gap is exactly the boundary rule this case exists to pin.
        """
        triangle = torch.tensor([[2.0, 2.0], [10.0, 3.0], [6.0, 11.0]])
        source = Targets(boxes=torch.tensor([[2.0, 2.0, 10.0, 11.0]]), labels=torch.tensor([4]), polygons=[triangle])
        items = [(torch.zeros(3, 16, 16), Targets.empty()), (torch.ones(3, 16, 16), source)]

        pasted, merged = CopyPaste(p=1.0).apply(items, CopyPasteParams(selected=(0,)))

        assert int((pasted[0] > 0.5).sum()) == 32
        assert merged.labels.tolist() == [4]
