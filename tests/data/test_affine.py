# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-010 random-affine transform (blueprint section 5.9).

Covers the box/mask consistency contract (the DoD), the identity transform,
recovery of a seeded translation on both the image and the box centres, canvas
clipping (partial clip and fully-outside drop with label alignment), visibility
filtering, the rotated-box guard, and byte-for-byte determinism under a seeded
generator.

Keypoints (WP-132) get their own class, including A70's case: this transform
manufactures it — translate and scale can push a point off-canvas while its box still
survives the visibility filter — and A70 says such a point is carried through with its
true coordinate and its visibility, neither clamped to the edge nor zeroed.
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
    """Return a CPU generator seeded to ``seed`` for reproducible sampling.

    Examples:
        >>> torch.rand(1, generator=_generator()).item()
        0.028979241847991943
    """
    return torch.Generator().manual_seed(seed)


def _image() -> torch.Tensor:
    """Return a random CHW float image on the canvas.

    Examples:
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> _image().shape
        torch.Size([3, 64, 64])
    """
    return torch.rand(3, _CANVAS, _CANVAS)


def _polygon_targets() -> Targets:
    """Build two square polygon rings well inside the canvas, with matching boxes.

    Examples:
        >>> targets = _polygon_targets()
        >>> targets.boxes.tolist()
        [[10.0, 12.0, 30.0, 34.0], [40.0, 20.0, 55.0, 50.0]]
        >>> targets.labels.tolist()
        [0, 1]
    """
    rings = [
        torch.tensor([[10.0, 12.0], [30.0, 12.0], [30.0, 34.0], [10.0, 34.0]]),
        torch.tensor([[40.0, 20.0], [55.0, 20.0], [55.0, 50.0], [40.0, 50.0]]),
    ]
    boxes = boxes_from_polygons(rings)
    labels = torch.arange(len(rings), dtype=torch.int64)
    return Targets(boxes=boxes, labels=labels, polygons=rings)


def _posed_targets(boxes: torch.Tensor, keypoints: torch.Tensor, visibility: torch.Tensor) -> Targets:
    """Build keypoint-carrying targets whose points ride the box instance axis.

    Examples:
        >>> t = _posed_targets(
        ...     torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        ...     torch.tensor([[[1.0, 2.0]]]),
        ...     torch.tensor([[2]]),
        ... )
        >>> t.keypoints.shape, t.keypoint_vis.shape
        (torch.Size([1, 1, 2]), torch.Size([1, 1]))
    """
    return Targets(
        boxes=boxes,
        labels=torch.arange(boxes.shape[0], dtype=torch.int64),
        keypoints=keypoints,
        keypoint_vis=visibility,
    )


class TestKeypoints:
    """Keypoints ride the affine's matrix, filter with their boxes, and are never clipped."""

    def test_points_warp_by_the_sampled_translation(self) -> None:
        """A translation-only warp moves each keypoint by exactly the sampled offset.

        This is the contract that makes a keypoint target usable at all: the points must
        land where the image content lands. A transform that carried them unwarped — or
        dropped them, as this one did before WP-132 — supervises the model against the
        pre-augmentation scene.
        """
        affine = RandomAffine(degrees=0.0, translate=0.15, scale=0.0, shear=0.0, generator=_generator())
        points = torch.tensor([[[12.0, 14.0], [28.0, 32.0]]])
        targets = _posed_targets(torch.tensor([[10.0, 12.0, 30.0, 34.0]]), points, torch.tensor([[2, 1]]))

        _, out = affine(_image(), targets)

        assert affine.last_params is not None
        offset = torch.tensor([affine.last_params.translate_x, affine.last_params.translate_y])
        assert torch.allclose(out.keypoints - points, offset.expand_as(points), atol=1e-4)

    def test_visibility_survives_the_warp_unchanged(self) -> None:
        """Visibility codes are carried across the warp verbatim, not recomputed.

        The affine moves coordinates; it has no basis for revising whether a point was
        annotated. Silently rewriting these values would corrupt the flag the OKS scorer
        and the loss mask both read.
        """
        affine = RandomAffine(degrees=0.0, translate=0.15, scale=0.0, shear=0.0, generator=_generator())
        visibility = torch.tensor([[2, 1]])
        targets = _posed_targets(
            torch.tensor([[10.0, 12.0, 30.0, 34.0]]), torch.tensor([[[12.0, 14.0], [28.0, 32.0]]]), visibility
        )

        _, out = affine(_image(), targets)

        assert torch.equal(out.keypoint_vis, visibility)

    def test_keypoint_free_targets_keep_the_canonical_empty(self) -> None:
        """A detection-only warp returns the canonical empty keypoint pair, untouched.

        Every golden in the repo sits on this path. If the keypoint carry-through were not
        an exact no-op for point-free targets, detect/segment/obb results would move.
        """
        affine = RandomAffine(degrees=0.0, translate=0.1, scale=0.0, shear=0.0, generator=_generator())

        _, out = affine(_image(), _polygon_targets())

        assert out.keypoints.shape == (0, 0, 2)
        assert out.keypoint_vis.shape == (0, 0)

    def test_dropped_instance_takes_its_points_with_it(self) -> None:
        """When the canvas filter drops an instance, the surviving keypoint rows stay aligned.

        The points share the box instance axis, so a filter that removed a box without
        removing its point set would silently re-pair every later instance with the wrong
        landmarks — a misalignment no shape check downstream would catch.
        """
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0)
        targets = _posed_targets(
            torch.tensor([[5.0, 5.0, 25.0, 25.0], [200.0, 200.0, 240.0, 240.0]]),
            torch.tensor([[[6.0, 7.0]], [[210.0, 220.0]]]),
            torch.tensor([[2], [2]]),
        )

        _, out = affine(_image(), targets)

        assert out.boxes.shape[0] == 1
        assert out.keypoints.tolist() == [[[6.0, 7.0]]]

    def test_point_pushed_off_canvas_is_neither_clamped_nor_zeroed(self) -> None:
        """A70: a point warped outside the canvas on a kept instance keeps coordinate and visibility.

        The affine translates a box that stays comfortably inside the canvas while one of
        its landmarks crosses the right edge. Clamping that landmark to the boundary would
        supervise the model toward a location the object is not at, and demoting it to
        ``v=0`` would collide with A66's "never annotated" meaning of that flag. Both are
        rejected, so the transform must leave the true coordinate standing.
        """
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0)
        outside = torch.tensor([[[20.0, 20.0], [_CANVAS + 30.0, _CANVAS + 12.0]]])
        targets = _posed_targets(torch.tensor([[10.0, 10.0, 40.0, 40.0]]), outside, torch.tensor([[2, 2]]))

        _, out = affine(_image(), targets)

        assert out.boxes.shape[0] == 1
        assert torch.allclose(out.keypoints, outside, atol=1e-4)
        assert out.keypoint_vis.tolist() == [[2, 2]]

    def test_polygon_path_warps_points_from_the_unclamped_matrix(self) -> None:
        """On the polygon path an off-canvas point is still unclamped, though rings are clamped.

        The two modalities take deliberately opposite treatment on the same call: a ring is
        clamped because it is rasterised onto the canvas, while A70 keeps a point where it
        truly is. Warping points from the clipped rings would quietly re-impose the clamp
        A70 rejects.
        """
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0)
        ring = torch.tensor([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]])
        outside = torch.tensor([[[_CANVAS + 25.0, 20.0]]])
        targets = Targets(
            boxes=boxes_from_polygons([ring]),
            labels=torch.tensor([0]),
            polygons=[ring],
            keypoints=outside,
            keypoint_vis=torch.tensor([[2]]),
        )

        _, out = affine(_image(), targets)

        assert torch.allclose(out.keypoints, outside, atol=1e-4)


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


def _sliver_targets(route: str) -> Targets:
    """Build one box clipped to a sliver, on the modality axis naming ``route``.

    The box spans ``[60, 10, 160, 40]`` on a 64-wide canvas, so clipping leaves four
    columns of thirty rows — 4% of its own area — which every threshold pairing below
    is chosen against.

    Args:
        route: ``"rotated"``, ``"polygons"`` or ``"boxes"``, naming which of the three
            keep-mask call sites the returned targets route through.

    Returns:
        Targets carrying exactly one instance.

    Examples:
        >>> _sliver_targets("boxes").boxes.tolist()
        [[60.0, 10.0, 160.0, 40.0]]
        >>> _sliver_targets("rotated").rboxes.shape
        torch.Size([1, 5])
    """
    boxes = torch.tensor([[60.0, 10.0, 160.0, 40.0]])
    labels = torch.tensor([0])
    if route == "rotated":
        return Targets(boxes=boxes, labels=labels, rboxes=torch.tensor([[110.0, 25.0, 100.0, 30.0, 0.0]]))
    if route == "polygons":
        ring = torch.tensor([[60.0, 10.0], [160.0, 10.0], [160.0, 40.0], [60.0, 40.0]])
        return Targets(boxes=boxes, labels=labels, polygons=[ring])
    return Targets(boxes=boxes, labels=labels)


@pytest.mark.parametrize("route", ["rotated", "polygons", "boxes"])
class TestKeepThresholdsReachUpstream:
    """Both thresholds are passed to upstream's keep mask at every one of the three call sites.

    Upstream's ``instance_keep_mask`` defaults ``min_size`` and ``min_visibility`` to
    ``0.0``, which drops nothing, against this project's ``2.0`` and ``0.1``. Omitting
    either argument at any call site keeps every instance with no shape change and no
    exception, so the failure is silent — these cases are what makes it loud. One case per
    site per threshold, plus the control that the drop is the threshold's doing.
    """

    def test_size_threshold_drops_the_sliver(self, route: str) -> None:
        """A four-column clipped box falls below ``min_box_size`` and is dropped.

        ``min_visibility`` is pinned at ``0.0`` so nothing but the size rule can account
        for the drop: if ``min_size`` were left to upstream's default the instance would
        survive, since 4 >= 0.0.
        """
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, min_box_size=8.0, min_visibility=0.0)

        _, out = affine(_image(), _sliver_targets(route))

        assert out.boxes.shape[0] == 0

    def test_visibility_threshold_drops_the_sliver(self, route: str) -> None:
        """A box retaining 4% of its area falls below ``min_visibility`` and is dropped.

        ``min_box_size`` is pinned at ``0.0`` so nothing but the visibility rule can
        account for the drop: at upstream's default of ``0.0`` the instance would survive,
        since 0.04 >= 0.0.
        """
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, min_box_size=0.0, min_visibility=0.5)

        _, out = affine(_image(), _sliver_targets(route))

        assert out.boxes.shape[0] == 0

    def test_upstream_defaults_would_keep_the_sliver(self, route: str) -> None:
        """With both thresholds at upstream's defaults the same instance survives.

        The control for the two cases above: it pins that they fail for the reason claimed
        — the thresholds this project passes — rather than because the sliver is dropped by
        the clip, the warp or anything else on the path.
        """
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, min_box_size=0.0, min_visibility=0.0)

        _, out = affine(_image(), _sliver_targets(route))

        assert out.boxes.shape[0] == 1


class TestRotatedBoxesGuard:
    """The rotated path filters, so it demands WP-056's instance-axis invariant."""

    def test_unpaired_rboxes_raise_value_error(self) -> None:
        """Rotated boxes that do not share the instance axis with boxes are rejected."""
        rboxes = torch.tensor([[10.0, 20.0, 8.0, 4.0, 0.3]])
        targets = Targets(boxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.int64), rboxes=rboxes)
        affine = RandomAffine(degrees=10.0)

        with pytest.raises(ValueError, match="instance axis"):
            affine(_image(), targets)


class TestLetterboxCanvas:
    """``letterbox=`` composes the aspect-preserving fit into the same single warp (WP-156)."""

    def test_a_same_size_letterbox_matches_the_plain_targets(self) -> None:
        """A letterbox onto the source canvas reproduces the plain path's targets exactly.

        The fit is the identity when target and source agree, so the letterboxed
        transform must land where the plain one does. This is the case that catches a
        composition applied on the wrong side: an inverted or transposed letterbox
        matrix is still the identity here only if the fit really is one.
        """
        image = _image()
        targets = _polygon_targets()
        plain = RandomAffine(degrees=12.0, translate=0.1, scale=0.2, shear=3.0, generator=_generator(5))
        boxed = RandomAffine(
            degrees=12.0, translate=0.1, scale=0.2, shear=3.0, generator=_generator(5), letterbox=_CANVAS
        )

        _, plain_targets = plain(image.clone(), targets.clone())
        _, boxed_targets = boxed(image.clone(), targets.clone())

        assert torch.equal(boxed_targets.boxes, plain_targets.boxes)
        for got, want in zip(boxed_targets.polygons, plain_targets.polygons, strict=True):
            assert torch.equal(got, want)

    def test_a_smaller_letterbox_emits_its_own_canvas(self) -> None:
        """A half-size letterbox emits a half-size image from one resample.

        The output size comes from the fit rather than from the input, which is the
        whole point of fusing the two: the source canvas reaches the training canvas
        without an intermediate full-size image ever existing.
        """
        out = _CANVAS // 2
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, letterbox=out)

        out_image, _ = affine(_image(), Targets.empty())

        assert out_image.shape == (3, out, out)


class TestGateStreamSeeding:
    """The discarded activation gates draw from a stream the caller's seed determines (L-32).

    Upstream draws a per-transform activation gate whether or not it can fail, so this class
    holds a second generator purely to keep those draws out of the caller's sequence. Left
    unseeded, that stream was reproducible only because ``torch.Generator()`` happens to
    start from a fixed constant *and* every gate here sits at ``prob = 1.0`` so its value is
    thrown away — two premises the code relied on and stated only half of. Seeding it from
    the caller's generator makes the reproducibility a property of this class instead.
    """

    def test_a_seeded_caller_seeds_the_gate_stream(self) -> None:
        """A caller-supplied generator determines the gate stream rather than leaving it default.

        The default ``torch.Generator()`` state is the same constant in every process, so an
        unseeded gate stream looks reproducible under test while being reproducible for a
        reason nothing here owns.
        """
        affine = RandomAffine(generator=_generator(1234))

        assert affine._gate_stream.initial_seed() != torch.Generator().initial_seed()

    def test_different_caller_seeds_give_different_gate_streams(self) -> None:
        """Two affines seeded differently draw their gates from different streams.

        Every affine sharing one stream is the shape of the bug this guards: two pipelines
        built under different seeds would agree on a sequence neither of them chose.
        """
        first = RandomAffine(generator=_generator(1))
        second = RandomAffine(generator=_generator(2))

        assert first._gate_stream.initial_seed() != second._gate_stream.initial_seed()

    def test_the_same_caller_seed_gives_the_same_gate_stream(self) -> None:
        """Rebuilding an affine at one seed rebuilds the same gate stream.

        The point of seeding it at all is reproducibility, so the property the module
        docstring claims — one seed determines the whole pipeline — must hold here too.
        """
        first = RandomAffine(generator=_generator(7))
        second = RandomAffine(generator=_generator(7))

        assert first._gate_stream.initial_seed() == second._gate_stream.initial_seed()

    def test_the_gate_stream_is_not_the_callers_own(self) -> None:
        """The gate draws never come out of the caller's generator.

        This is the property WP-079 was opened by losing: a gate drawn from the caller's
        stream shifts every subsequent transform's draw, so an upstream change in how many
        gates exist would silently re-roll the whole augmentation sequence.
        """
        caller = _generator(1234)
        affine = RandomAffine(generator=caller)

        assert affine._gate_stream is not caller
        assert affine._gate_stream.initial_seed() != caller.initial_seed()

    def test_an_unseeded_caller_leaves_the_gate_stream_anonymous(self) -> None:
        """With no caller generator there is no seed to derive from, and none is invented.

        A default-constructed affine has nothing to be reproducible *with*; giving its gate
        stream a fabricated seed would suggest otherwise.
        """
        affine = RandomAffine()

        assert affine._gate_stream.initial_seed() == torch.Generator().initial_seed()
