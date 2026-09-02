# SPDX-License-Identifier: Apache-2.0
"""Tier A and tier C augmentation expectations: derived results and invariants (WP-148).

Two of the guard's four tiers live here, and neither carries a characterization
value, so the whole file reviews as mathematics rather than as a record of what
the code happened to print.

**Tier A -- derived.** Hand-computed closed-form results, independent of any
implementation: an identity returns its input, an integer-pixel translation is an
exact shifted copy, a quarter turn maps a known box to a known box, a letterbox
forward-then-inverse is the identity, two mirrors compose to the identity
including the keypoint permutation, a stated mosaic centre places four known
boxes at four derived offsets. This is the only tier that proves the code is
*right* rather than *unchanged*, which is why every case that can be moved into
it is.

**Tier C -- invariants.** Parameter-free properties constraining the modalities
against each other: every modality on the instance axis survives or dies
together, a box derived from a transported ring equals the separately transported
box, rotated boxes leaving a transform are canonical long-edge, out-of-canvas
pixels equal the fill exactly. Cheapest tier to write and the likeliest to catch
a real integration defect, because a shape-compatible wrong answer preserves
shapes and not relationships.

Both tiers are written against the WP-147 sample/apply seam: every case states
its parameters and calls ``apply``, never ``__call__``, so nothing here depends
on a generator and nothing here re-freezes when a sampler changes. Coverage
reaches every transform, including the ones that stay in this repository --
a transform that is refactored around still needs its guard already written.

Pixel expectations are exact only where analytically forced: an identity, an
integer-pixel translation, and the fill region. Everywhere else this file asserts
geometry, which is exact, and leaves pixels to tier B's probes and bands.
"""

from __future__ import annotations

import math

import torch

from lucid_yolo.data.affine import AffineParams, RandomAffine
from lucid_yolo.data.augment import FlipParams, HorizontalFlip, HSVJitter, HSVParams
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.mixup import CopyPaste, CopyPasteParams, Mixup, MixupParams
from lucid_yolo.data.mosaic import MosaicAssembly, MosaicParams
from lucid_yolo.data.rasterize import rasterize_polygon
from lucid_yolo.data.rotated_geom import canonicalize
from lucid_yolo.data.targets import Targets
from lucid_yolo.data.transforms import apply_affine_to_points, boxes_from_polygons

#: The grey fill every geometric transform pads with (the YOLO-lineage 114/255).
FILL = 114.0 / 255.0
#: Float32 resampling tolerance: one bilinear pass through ``grid_sample`` at
#: exactly-hit pixel centres, which is a copy up to the last bit of a float32.
EXACT = 1e-6


def _identity_params() -> AffineParams:
    """Build the affine that maps every pixel to itself.

    Examples:
        ```pycon
        >>> _identity_params().scale
        1.0

        ```
    """
    return AffineParams(angle=0.0, shear_x=0.0, shear_y=0.0, scale=1.0, translate_x=0.0, translate_y=0.0)


def _translation(dx: float, dy: float) -> AffineParams:
    """Build a pure pixel translation.

    Args:
        dx: Horizontal shift in pixels.
        dy: Vertical shift in pixels.

    Examples:
        ```pycon
        >>> _translation(2.0, 0.0).translate_x
        2.0

        ```
    """
    return AffineParams(angle=0.0, shear_x=0.0, shear_y=0.0, scale=1.0, translate_x=dx, translate_y=dy)


def _keeps_everything() -> RandomAffine:
    """Build an affine whose instance filter drops nothing, isolating the geometry.

    The filter has its own cases; a geometry case that also filtered would confound
    a wrong coordinate with a dropped instance.

    Examples:
        ```pycon
        >>> _keeps_everything().min_box_size
        0.0

        ```
    """
    return RandomAffine(min_box_size=0.0, min_visibility=0.0)


def _boxed(boxes: torch.Tensor) -> Targets:
    """Wrap ``boxes`` in a target set with sequential labels and nothing else.

    Args:
        boxes: ``(N, 4)`` xyxy boxes.

    Examples:
        ```pycon
        >>> _boxed(torch.zeros((2, 4))).labels.tolist()
        [0, 1]

        ```
    """
    return Targets(boxes=boxes, labels=torch.arange(boxes.shape[0]))


class TestAffineDerived:
    """Tier A -- closed-form affine results, independent of any implementation."""

    def test_identity_returns_the_image_unchanged(self) -> None:
        """An identity affine resamples every pixel centre onto itself.

        This is the case that would catch a half-pixel error in the pixel-to-
        normalised change of basis: any off-by-half in either direction shows up
        here as a blurred copy, while every geometric assertion still passes.
        """
        image = torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(0))

        warped, _ = _keeps_everything().apply(image, Targets.empty(), _identity_params())

        assert torch.allclose(warped, image, atol=EXACT)

    def test_integer_translation_is_an_exact_shifted_copy(self) -> None:
        """A two-pixel translation copies columns exactly and fills the vacated ones.

        At ``align_corners=False`` an integer translation lands on pixel centres, so
        no interpolation error can enter; the vacated columns must be the grey fill
        rather than an edge-replicated or zero-padded value.
        """
        image = torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(1))

        warped, _ = _keeps_everything().apply(image, Targets.empty(), _translation(2.0, 0.0))

        assert torch.allclose(warped[:, :, 2:], image[:, :, :-2], atol=EXACT)
        assert torch.allclose(warped[:, :, :2], torch.full_like(warped[:, :, :2], FILL), atol=EXACT)

    def test_quarter_turn_maps_a_known_box_to_a_known_box(self) -> None:
        """A 90-degree rotation about the canvas centre maps ``[1,2,3,6]`` to ``[2,1,6,3]``.

        Derived, not observed: on an 8-pixel canvas the centre is ``(4, 4)`` and the
        rotation sends ``(x, y)`` to ``(4 - (y - 4), 4 + (x - 4))``, so the corners
        ``(1, 2)`` and ``(3, 6)`` land on ``(6, 1)`` and ``(2, 3)``, whose envelope
        is the expected box. A similarity maps a rectangle to a rectangle, so the
        envelope is exact rather than an over-approximation.
        """
        quarter = AffineParams(angle=math.pi / 2, shear_x=0.0, shear_y=0.0, scale=1.0, translate_x=0.0, translate_y=0.0)

        _, warped = _keeps_everything().apply(
            torch.zeros(3, 8, 8), _boxed(torch.tensor([[1.0, 2.0, 3.0, 6.0]])), quarter
        )

        assert torch.allclose(warped.boxes, torch.tensor([[2.0, 1.0, 6.0, 3.0]]), atol=EXACT)

    def test_a_known_scale_scales_box_coordinates_about_the_centre(self) -> None:
        """Doubling about the centre sends ``[2,2,4,4]`` to ``[0,0,4,4]`` on an 8-pixel canvas.

        Derived: a scale of two about ``(4, 4)`` maps ``x`` to ``4 + 2(x - 4)``, so
        ``2`` goes to ``0`` and ``4`` stays at ``4``. Choosing a box that stays inside
        the canvas keeps the clip out of the assertion.
        """
        doubled = AffineParams(angle=0.0, shear_x=0.0, shear_y=0.0, scale=2.0, translate_x=0.0, translate_y=0.0)

        _, warped = _keeps_everything().apply(
            torch.zeros(3, 8, 8), _boxed(torch.tensor([[2.0, 2.0, 4.0, 4.0]])), doubled
        )

        assert torch.allclose(warped.boxes, torch.tensor([[0.0, 0.0, 4.0, 4.0]]), atol=EXACT)


class TestLetterboxDerived:
    """Tier A -- the letterbox's ratio, padding and exact analytic inverse."""

    def test_forward_then_inverse_is_the_identity(self) -> None:
        """Mapping points forward through the letterbox and back returns them exactly.

        The inverse is what every prediction path uses to return detections to
        original-image coordinates, so a drift here is a systematic coordinate error
        in every reported box rather than a training-time nuisance.
        """
        letterbox = Letterbox(16)
        points = torch.tensor([[0.0, 0.0], [10.0, 5.0], [19.0, 9.0]])
        matrix, out_h, out_w = letterbox.forward_affine(10, 20)

        forward = apply_affine_to_points(points, matrix.to(points.dtype))
        recovered = letterbox.inverse_map(forward, (10, 20), (out_h, out_w))

        assert torch.allclose(recovered, points, atol=EXACT)

    def test_ratio_and_padding_are_the_aspect_preserving_ones(self) -> None:
        """A 10x20 source into a 16x16 canvas scales by 0.8 and pads 4 rows above.

        Derived: the ratio is ``min(16/10, 16/20) = 0.8``, the scaled content is
        ``8x16``, and the 8 leftover rows split symmetrically into 4 above and 4
        below. The horizontal axis is exactly filled, so its offset is zero.
        """
        matrix, out_h, out_w = Letterbox(16).forward_affine(10, 20)

        assert (out_h, out_w) == (16, 16)
        assert torch.allclose(matrix[:2, :2], torch.tensor([[0.8, 0.0], [0.0, 0.8]], dtype=matrix.dtype), atol=EXACT)
        assert torch.allclose(matrix[:2, 2], torch.tensor([0.0, 4.0], dtype=matrix.dtype), atol=EXACT)


class TestFlipDerived:
    """Tier A -- mirroring is an involution, on pixels and on keypoint identities."""

    def test_two_mirrors_restore_the_image_and_every_modality(self) -> None:
        """Flipping twice returns the image, boxes, keypoints and visibilities exactly.

        The keypoint half is the part worth pinning: a mirror that reflected
        coordinates but applied the left/right swap only once would compose to a
        silent identity on pixels and a permutation on identities, supervising every
        sided point toward its opposite number.
        """
        flip = HorizontalFlip(keypoint_flip_pairs=[(0, 1)])
        image = torch.rand(3, 4, 6, generator=torch.Generator().manual_seed(2))
        targets = Targets(
            boxes=torch.tensor([[1.0, 1.0, 3.0, 3.0]]),
            labels=torch.tensor([0]),
            keypoints=torch.tensor([[[1.0, 1.0], [3.0, 3.0]]]),
            keypoint_vis=torch.tensor([[2, 1]]),
        )

        once_image, once = flip.apply(image, targets, FlipParams(flipped=True))
        twice_image, twice = flip.apply(once_image, once, FlipParams(flipped=True))

        assert torch.equal(twice_image, image)
        assert torch.allclose(twice.boxes, targets.boxes, atol=EXACT)
        assert torch.allclose(twice.keypoints, targets.keypoints, atol=EXACT)
        assert torch.equal(twice.keypoint_vis, targets.keypoint_vis)

    def test_one_mirror_reflects_a_box_about_the_vertical_axis(self) -> None:
        """On a 6-wide canvas the box ``[0,0,1,2]`` mirrors to ``[5,0,6,2]``.

        Derived: ``x1' = W - x2`` and ``x2' = W - x1``, which is the only mapping
        that both reflects and keeps ``x1 < x2``.
        """
        _, mirrored = HorizontalFlip().apply(
            torch.zeros(3, 2, 6), _boxed(torch.tensor([[0.0, 0.0, 1.0, 2.0]])), FlipParams(flipped=True)
        )

        assert torch.allclose(mirrored.boxes, torch.tensor([[5.0, 0.0, 6.0, 2.0]]), atol=EXACT)


class TestRotatedDerived:
    """Tier A -- a similarity maps a rectangle to a rectangle, so an rbox re-fits exactly."""

    def test_pure_rotation_refits_a_rotated_box_exactly(self) -> None:
        """Rotating by a quarter turn moves the centre and adds to the angle, nothing else.

        Derived: a rotation is a similarity, so the warped corners are still a
        rectangle of the same side lengths; only the centre moves and the angle
        advances. The result is re-canonicalized, so the expectation is stated in
        canonical long-edge form rather than in whatever form the warp produced.
        """
        quarter = AffineParams(angle=math.pi / 2, shear_x=0.0, shear_y=0.0, scale=1.0, translate_x=0.0, translate_y=0.0)
        rbox = torch.tensor([[6.0, 4.0, 4.0, 2.0, 0.0]])
        targets = Targets(boxes=torch.tensor([[4.0, 3.0, 8.0, 5.0]]), labels=torch.tensor([0]), rboxes=rbox)

        _, warped = _keeps_everything().apply(torch.zeros(3, 8, 8), targets, quarter)

        expected_centre = torch.tensor([4.0, 6.0])
        assert torch.allclose(warped.rboxes[0, :2], expected_centre, atol=1e-4)
        assert torch.allclose(warped.rboxes[0, 2:4], torch.tensor([4.0, 2.0]), atol=1e-4)


class TestPhotometricDerived:
    """Tier A -- HSV gains with a closed form on a known colour."""

    def test_zero_gains_are_the_identity(self) -> None:
        """Zero gains round-trip RGB through HSV and back to the same pixels.

        The conversion pair is exercised by every jittered sample, so an asymmetry
        between them would bias every training image by a fixed amount that no
        geometric assertion could see.
        """
        image = torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(3))

        jittered, _ = HSVJitter().apply(image, Targets.empty(), HSVParams(hue=0.0, saturation=0.0, value=0.0))

        assert torch.allclose(jittered, image, atol=1e-5)

    def test_a_value_gain_scales_a_grey_image_exactly(self) -> None:
        """On a grey image, value is the pixel itself, so a gain of ``+0.5`` scales it by 1.5.

        Derived: grey has zero saturation and undefined hue, so the value channel is
        the pixel value and the multiplicative gain applies to it directly.
        """
        grey = torch.full((3, 4, 4), 0.4)

        jittered, _ = HSVJitter().apply(grey, Targets.empty(), HSVParams(hue=0.0, saturation=0.0, value=0.5))

        assert torch.allclose(jittered, torch.full_like(grey, 0.6), atol=1e-5)


class TestAssemblyDerived:
    """Tier A -- mosaic placement, mixup blending and copy-paste, all from stated parameters."""

    def test_a_stated_mosaic_centre_places_four_known_boxes(self) -> None:
        """With the centre at ``(8, 8)`` and four 8-pixel images, each box shifts by its quadrant.

        Derived: image 0 is anchored so its bottom-right corner sits at the centre,
        giving offset ``(0, 0)``; images 1, 2 and 3 take offsets ``(8, 0)``, ``(0, 8)``
        and ``(8, 8)``. The same box in all four inputs therefore lands at four
        translations of itself, which is a stronger statement than four boxes being
        present somewhere on the canvas.
        """
        box = _boxed(torch.tensor([[1.0, 1.0, 5.0, 5.0]]))
        items = [(torch.full((3, 8, 8), float(index)), box.clone()) for index in range(4)]

        canvas, merged = MosaicAssembly(target_size=8, min_box_size=0.0, min_visibility=0.0).apply(
            items, MosaicParams(center_x=8, center_y=8)
        )

        expected = torch.tensor(
            [[1.0, 1.0, 5.0, 5.0], [9.0, 1.0, 13.0, 5.0], [1.0, 9.0, 5.0, 13.0], [9.0, 9.0, 13.0, 13.0]]
        )
        assert torch.allclose(merged.boxes, expected, atol=EXACT)
        assert [float(canvas[0, y, x]) for y, x in ((0, 0), (0, 15), (15, 0), (15, 15))] == [0.0, 1.0, 2.0, 3.0]

    def test_mixup_at_a_stated_factor_is_the_convex_blend(self) -> None:
        """``lam = 0.25`` on constant images returns exactly ``0.25a + 0.75b``.

        Constant inputs make the expectation a single number, so a transposed blend
        (``lam`` applied to the wrong image) fails rather than producing a plausible
        intermediate value.
        """
        items = [(torch.ones(3, 4, 4), Targets.empty()), (torch.zeros(3, 4, 4), Targets.empty())]

        blended, _ = Mixup(p=1.0).apply(items, MixupParams(lam=0.25))

        assert torch.allclose(blended, torch.full((3, 4, 4), 0.25), atol=EXACT)

    def test_mixup_at_unit_factor_returns_the_first_image(self) -> None:
        """``lam = 1.0`` is the identity on the first image, by the blend's own algebra."""
        first = torch.rand(3, 4, 4, generator=torch.Generator().manual_seed(4))
        items = [(first, Targets.empty()), (torch.zeros(3, 4, 4), Targets.empty())]

        blended, _ = Mixup(p=1.0).apply(items, MixupParams(lam=1.0))

        assert torch.allclose(blended, first, atol=EXACT)

    def test_copy_paste_writes_exactly_the_rasterised_mask(self) -> None:
        """The pasted pixels are exactly those the ring rasterises to, and no others.

        The paste is a mask write, so its correctness is entirely the mask's: an
        off-by-one in the rasteriser shows up here as a border row of destination
        pixels that should have been overwritten, or source pixels outside the ring.
        """
        ring = torch.tensor([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
        source = Targets(boxes=torch.tensor([[1.0, 1.0, 4.0, 4.0]]), labels=torch.tensor([7]), polygons=[ring])
        items = [(torch.zeros(3, 6, 6), Targets.empty()), (torch.ones(3, 6, 6), source)]

        pasted, merged = CopyPaste(p=1.0).apply(items, CopyPasteParams(selected=(0,)))

        assert torch.equal(pasted[0] > 0.5, rasterize_polygon(ring, 6, 6))
        assert merged.labels.tolist() == [7]


class TestRasterisationDerived:
    """Tier A -- the half-open rasterisation rule (A11), stated as a pixel count."""

    def test_an_axis_aligned_ring_covers_its_half_open_interior(self) -> None:
        """The ring spanning ``[1,4] x [1,4]`` rasterises to the 3x3 block at rows and columns 1..3.

        Derived from A11's half-open convention: the upper edge is excluded, so a
        span of three units covers three pixels rather than four. This is the rule
        every mask supervision target is built on, and it stays in this repository
        after the augmentation engine moves.
        """
        ring = torch.tensor([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])

        mask = rasterize_polygon(ring, 6, 6)

        expected = torch.zeros((6, 6), dtype=torch.bool)
        expected[1:4, 1:4] = True
        assert torch.equal(mask, expected)


class TestModalityInvariants:
    """Tier C -- parameter-free properties tying the modalities to each other."""

    def test_every_modality_survives_or_dies_together(self) -> None:
        """The instance axis stays aligned across boxes, labels, polygons and keypoints.

        A translation that pushes one of two instances off the canvas must drop that
        instance from *every* modality. Dropping it from boxes alone would leave a
        keypoint set supervising the surviving box with the departed instance's
        points -- shape-compatible, and wrong in a way no shape check can see.
        """
        far = _translation(-10.0, 0.0)
        targets = Targets(
            boxes=torch.tensor([[1.0, 1.0, 5.0, 5.0], [12.0, 1.0, 15.0, 5.0]]),
            labels=torch.tensor([3, 4]),
            polygons=[
                torch.tensor([[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]]),
                torch.tensor([[12.0, 1.0], [15.0, 1.0], [15.0, 5.0], [12.0, 5.0]]),
            ],
            keypoints=torch.tensor([[[2.0, 2.0], [4.0, 4.0]], [[13.0, 2.0], [14.0, 4.0]]]),
            keypoint_vis=torch.tensor([[2, 2], [2, 2]]),
        )

        _, warped = RandomAffine().apply(torch.zeros(3, 16, 16), targets, far)

        kept = warped.boxes.shape[0]
        assert kept == warped.labels.shape[0] == len(warped.polygons) == warped.keypoints.shape[0]
        assert kept == warped.keypoint_vis.shape[0]
        assert warped.labels.tolist() == [4]

    def test_a_derived_box_equals_the_transported_box(self) -> None:
        """For an instance that never touches an edge, the ring and the box agree.

        This is the alignment invariant the whole ring-transport route rests on: the
        box carried through the transform must equal the box re-derived from the
        ring carried through the same transform. It holds only while both ride one
        matrix, which is exactly the property a split geometric segment would break
        -- silently, since both outputs stay well-formed.
        """
        shifted = _translation(2.0, 1.0)
        ring = torch.tensor([[4.0, 4.0], [9.0, 4.0], [9.0, 10.0], [4.0, 10.0]])
        targets = Targets(boxes=torch.tensor([[4.0, 4.0, 9.0, 10.0]]), labels=torch.tensor([0]), polygons=[ring])

        _, warped = RandomAffine().apply(torch.zeros(3, 24, 24), targets, shifted)

        assert torch.allclose(boxes_from_polygons(warped.polygons), warped.boxes, atol=1e-4)

    def test_rotated_boxes_leave_a_transform_canonical(self) -> None:
        """Whatever a warp does to an rbox, its long-edge form is restored on the way out.

        Canonical form is a task convention (A22) rather than a resampling property,
        so it has to be re-established after every warp; a transform that returned a
        short-edge-first box would be read by the assigner as a differently oriented
        object of a different aspect.
        """
        skewed = AffineParams(angle=0.7, shear_x=0.0, shear_y=0.0, scale=1.2, translate_x=1.0, translate_y=-2.0)
        rbox = torch.tensor([[12.0, 12.0, 3.0, 7.0, 0.4]])
        targets = Targets(boxes=torch.tensor([[8.0, 8.0, 16.0, 16.0]]), labels=torch.tensor([0]), rboxes=rbox)

        _, warped = _keeps_everything().apply(torch.zeros(3, 24, 24), targets, skewed)

        assert torch.allclose(warped.rboxes, canonicalize(warped.rboxes), atol=1e-5)

    def test_out_of_canvas_pixels_are_exactly_the_fill(self) -> None:
        """A translation past the canvas leaves nothing but the grey fill.

        The fill value is a convention shared with the letterbox and the mosaic pad;
        a transform that filled with zeros instead would darken every border in
        training while every geometric assertion stayed green.
        """
        image = torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(5))

        warped, _ = _keeps_everything().apply(image, Targets.empty(), _translation(64.0, 64.0))

        assert torch.allclose(warped, torch.full_like(warped, FILL), atol=EXACT)

    def test_mixup_concatenates_both_label_sets_unweighted(self) -> None:
        """The blend factor weighs pixels and never labels.

        Mixup's supervision contract is that both label sets are present in full:
        the loss sees every instance of both images, and the blend appears only in
        the pixels. Weighting or dropping labels by ``lam`` would be a different
        method wearing the same name.
        """
        first = Targets(boxes=torch.tensor([[0.0, 0.0, 2.0, 2.0]]), labels=torch.tensor([1]))
        second = Targets(boxes=torch.tensor([[2.0, 2.0, 4.0, 4.0]]), labels=torch.tensor([9]))
        items = [(torch.ones(3, 4, 4), first), (torch.zeros(3, 4, 4), second)]

        _, merged = Mixup(p=1.0).apply(items, MixupParams(lam=0.9))

        assert merged.labels.tolist() == [1, 9]
        assert merged.boxes.shape[0] == 2

    def test_a_keypoint_rides_its_own_pixel_through_a_warp(self) -> None:
        """A point marking a bright pixel still marks it after an integer translation.

        This ties the image transport and the keypoint transport to each other
        rather than checking each alone: a keypoint path that used a different sign
        or a different centre convention from the image path would move the point
        off its feature while both outputs stayed inside the canvas.
        """
        image = torch.full((3, 16, 16), 0.0)
        image[:, 5, 6] = 1.0
        targets = Targets(
            boxes=torch.tensor([[4.0, 3.0, 9.0, 8.0]]),
            labels=torch.tensor([0]),
            keypoints=torch.tensor([[[6.0, 5.0]]]),
            keypoint_vis=torch.tensor([[2]]),
        )

        warped_image, warped = _keeps_everything().apply(image, targets, _translation(3.0, 2.0))

        x, y = warped.keypoints[0, 0].round().to(torch.int64).tolist()
        assert (x, y) == (9, 7)
        assert float(warped_image[0, y, x]) > 0.5
