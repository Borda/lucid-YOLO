# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-058 rotated-aware augmentations (A40).

The DoD test is the module-level ``test_roundtrip``: a similarity warp composed with its
own inverse returns every rotated box in an angle sweep to tolerance. Around it sit the
properties WP-058 exists to establish.

Canonical output everywhere:
    Every transform is swept over the whole ``[-pi/4, 3*pi/4)`` range **and** past both
    ends, and its output must satisfy ``w >= h`` and the angle range. That is what closes
    the WP-013 flip defect, where negating an angle above ``pi/4`` left the range.

Geometry, not parameters:
    A rotated box is compared by its **corners**, never by ``(w, h, theta)``: the square
    tie-break and the ``w``/``h`` swap inside ``canonicalize`` mean two different parameter
    triples can describe one rectangle, and WP-055's row warns downstream not to assume
    which representative comes back. ``_box_distance`` is therefore an order-agnostic
    symmetric nearest-corner distance between two boxes' corner rings.

The shear residual, derived rather than tuned:
    Under a similarity the warped box is still a rectangle and the re-fit is exact. Under
    shear the warped quad is a parallelogram with half-vectors ``p`` (along the fitted long
    axis) and ``q``; the fit returns ``|q|`` along ``p``'s perpendicular instead of ``q``
    itself, so **every** corner is displaced by exactly ``2 |q| sin(s / 2)`` where ``s`` is
    the angle by which ``q`` misses that perpendicular. The sheared test asserts that
    identity — an equality against a quantity computed from the transform's own matrix —
    rather than a tolerance chosen to pass.

Dropping, not flagging:
    Augmentation drops what leaves the canvas (A40), by the axis-aligned path's own
    ``min_box_size`` / ``min_visibility`` rule, and drops it from every modality at once,
    which the instance-axis assertions pin.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch
from torch import Tensor

from lucid_yolo.data import (
    CopyPaste,
    HorizontalFlip,
    Mixup,
    MosaicAssembly,
    RandomAffine,
    Targets,
    canonicalize,
    clip_rboxes_to_canvas,
    rboxes_to_polygons,
    warp_rboxes,
)
from lucid_yolo.data.transforms import apply_affine_to_points

#: Inclusive lower bound of the canonical angle range.
_THETA_LOW = -math.pi / 4.0
#: Exclusive upper bound of the canonical angle range.
_THETA_HIGH = 3.0 * math.pi / 4.0
#: Side of the square canvas the single-image transforms run on.
_CANVAS = 128
#: Mosaic base size ``S``; the assembled canvas is ``2S``.
_MOSAIC_SIDE = 32

#: Angles spanning the canonical range, its two ends, and inputs outside it.
_SWEEP_ANGLES = (
    _THETA_LOW,
    -0.4,
    0.0,
    math.pi / 8.0,
    math.pi / 4.0,
    1.0,
    math.pi / 2.0,
    2.0,
    _THETA_HIGH - 1e-3,
    -1.5,
    -3.0,
    3.5,
)

_THETA_CASES = [
    pytest.param(_THETA_LOW, id="range-low"),
    pytest.param(0.0, id="zero"),
    pytest.param(math.pi / 4.0, id="quarter-pi"),
    pytest.param(1.0, id="above-quarter-pi"),
    pytest.param(math.pi / 2.0, id="half-pi"),
    pytest.param(_THETA_HIGH - 1e-3, id="range-high"),
    pytest.param(-1.5, id="out-of-range-low"),
    pytest.param(3.5, id="out-of-range-high"),
]


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


def _rboxes(*thetas: float, cx: float = 64.0, cy: float = 60.0, w: float = 20.0, h: float = 8.0) -> Tensor:
    """Build one ``(M, 5)`` rotated box per angle, all at the same centre and extents.

    Examples:
        >>> _rboxes(0.0, 0.5).shape
        torch.Size([2, 5])
        >>> _rboxes(0.0)[0].tolist()
        [64.0, 60.0, 20.0, 8.0, 0.0]
    """
    return torch.tensor([[cx, cy, w, h, theta] for theta in thetas])


def _paired(rboxes: Tensor) -> Targets:
    """Wrap rotated boxes as WP-056 targets: one label each, boxes the rotated envelopes.

    Examples:
        >>> targets = _paired(_rboxes(0.0, 0.5))
        >>> targets.boxes.shape
        torch.Size([2, 4])
        >>> targets.labels.tolist()
        [0, 1]
    """
    corners = rboxes_to_polygons(rboxes)
    boxes = torch.cat([corners.amin(dim=1), corners.amax(dim=1)], dim=1)
    return Targets(boxes=boxes, labels=torch.arange(rboxes.shape[0], dtype=torch.int64), rboxes=rboxes)


def _image(side: int = _CANVAS) -> Tensor:
    """Return a deterministic CHW image of the given square side.

    Examples:
        >>> _image(4).shape
        torch.Size([3, 4, 4])
    """
    return torch.rand(3, side, side)


def _similarity(angle: float, scale: float, tx: float, ty: float) -> Tensor:
    """Return the ``3x3`` float64 rotation-scale-translation matrix (a similarity).

    Examples:
        >>> _similarity(0.0, 1.0, 2.0, 3.0).tolist()
        [[1.0, -0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]
    """
    cos, sin = scale * math.cos(angle), scale * math.sin(angle)
    return torch.tensor([[cos, -sin, tx], [sin, cos, ty], [0.0, 0.0, 1.0]], dtype=torch.float64)


def _ring_distance(first: Tensor, second: Tensor) -> Tensor:
    """Symmetric nearest-corner distance between two ``(M, 4, 2)`` corner batches.

    Examples:
        >>> corners = torch.zeros(1, 4, 2)
        >>> float(_ring_distance(corners, corners))
        0.0
    """
    pairwise = torch.cdist(first, second)
    forward = pairwise.min(dim=2).values.amax(dim=1)
    backward = pairwise.min(dim=1).values.amax(dim=1)
    return torch.maximum(forward, backward)


def _box_distance(first: Tensor, second: Tensor) -> Tensor:
    """Per-instance corner distance between two ``(M, 5)`` rotated-box batches.

    Examples:
        >>> rboxes = _rboxes(0.3)
        >>> float(_box_distance(rboxes, rboxes))
        0.0
    """
    return _ring_distance(rboxes_to_polygons(first), rboxes_to_polygons(second))


def _envelopes(rboxes: Tensor) -> Tensor:
    """Return the ``(M, 4)`` axis-aligned envelopes of a rotated-box batch.

    Examples:
        >>> _envelopes(_rboxes(0.0)).tolist()
        [[54.0, 56.0, 74.0, 64.0]]
    """
    corners = rboxes_to_polygons(rboxes)
    return torch.cat([corners.amin(dim=1), corners.amax(dim=1)], dim=1)


def _assert_canonical(rboxes: Tensor) -> None:
    """Assert every row satisfies the long-edge invariants ``w >= h`` and the angle range.

    Examples:
        >>> _assert_canonical(canonicalize(_rboxes(0.0)))
        >>> try:
        ...     _assert_canonical(_rboxes(-1.5))  # below _THETA_LOW, not canonical
        ... except AssertionError:
        ...     print("not canonical")
        not canonical
    """
    assert bool((rboxes[:, 2] >= rboxes[:, 3]).all())
    assert bool((rboxes[:, 4] >= _THETA_LOW).all())
    assert bool((rboxes[:, 4] < _THETA_HIGH).all())


def _assert_one_instance_axis(targets: Targets) -> None:
    """Assert boxes, labels and rotated boxes still share one instance axis (WP-056).

    Examples:
        >>> _assert_one_instance_axis(_paired(_rboxes(0.0, 0.5)))
    """
    assert targets.labels.shape[0] == targets.boxes.shape[0]
    assert targets.rboxes.shape[0] == targets.boxes.shape[0]


def test_roundtrip() -> None:
    """A similarity warp composed with its inverse returns every swept angle's box (DoD)."""
    rboxes = _rboxes(*_SWEEP_ANGLES)
    matrix = _similarity(angle=0.6, scale=1.7, tx=12.0, ty=-5.0)

    restored = warp_rboxes(warp_rboxes(rboxes, matrix), torch.inverse(matrix))

    _assert_canonical(restored)
    assert float(_box_distance(restored, canonicalize(rboxes)).max()) < 1e-3


class TestCanonicalOutput:
    """Every transform emits canonical long-edge boxes, for any incoming angle."""

    @pytest.mark.parametrize("theta", _THETA_CASES)
    def test_flip_output_is_canonical(self, theta: float) -> None:
        """The horizontal flip re-canonicalizes the mirrored angle."""
        flip = HorizontalFlip(p=1.0, generator=_generator())

        _, out = flip(_image(), _paired(_rboxes(theta)))

        _assert_canonical(out.rboxes)

    @pytest.mark.parametrize("theta", _THETA_CASES)
    def test_affine_output_is_canonical(self, theta: float) -> None:
        """A rotating, scaling, shearing affine re-fits to a canonical box."""
        affine = RandomAffine(degrees=25.0, translate=0.05, scale=0.2, shear=10.0, generator=_generator())

        _, out = affine(_image(), _paired(_rboxes(theta)))

        _assert_canonical(out.rboxes)

    @pytest.mark.parametrize("theta", _THETA_CASES)
    def test_mosaic_output_is_canonical(self, theta: float) -> None:
        """Mosaic placement leaves every surviving rotated box canonical."""
        mosaic = MosaicAssembly(target_size=_MOSAIC_SIDE, generator=_generator())
        carried = _paired(_rboxes(theta, cx=30.0, cy=30.0, w=4.0, h=2.0))
        items = [(_image(_MOSAIC_SIDE), carried)] + [(_image(_MOSAIC_SIDE), Targets.empty()) for _ in range(3)]

        _, out = mosaic(items)

        _assert_canonical(out.rboxes)

    @pytest.mark.parametrize("theta", _THETA_CASES)
    def test_mixup_output_is_canonical(self, theta: float) -> None:
        """Mixup carries rotated boxes through the blend still canonical."""
        mixup = Mixup(p=1.0, generator=_generator())
        first = _paired(canonicalize(_rboxes(theta)))
        second = _paired(canonicalize(_rboxes(-theta)))

        _, out = mixup([(_image(), first), (_image(), second)])

        _assert_canonical(out.rboxes)


class TestHorizontalFlipRotated:
    """The mirror is exact, verified against independently mirrored corners."""

    @pytest.mark.parametrize("theta", _THETA_CASES)
    def test_mirrored_box_matches_mirrored_corners(self, theta: float) -> None:
        """The flipped box describes the rectangle obtained by mirroring the corners alone."""
        flip = HorizontalFlip(p=1.0, generator=_generator())
        targets = _paired(_rboxes(theta))
        expected = rboxes_to_polygons(targets.rboxes).clone()
        expected[:, :, 0] = _CANVAS - expected[:, :, 0]

        _, out = flip(_image(), targets)

        assert float(_ring_distance(rboxes_to_polygons(out.rboxes), expected).max()) < 1e-3

    def test_double_flip_restores_the_original_geometry(self) -> None:
        """Flipping twice returns every swept angle's rectangle to its starting place."""
        flip = HorizontalFlip(p=1.0, generator=_generator())
        targets = _paired(_rboxes(*_SWEEP_ANGLES))

        image, once = flip(_image(), targets)
        _, twice = flip(image, once)

        assert float(_box_distance(twice.rboxes, canonicalize(targets.rboxes)).max()) < 1e-4

    def test_nothing_is_dropped_by_the_mirror(self) -> None:
        """An isometry keeps every instance, so all three modalities keep their length."""
        flip = HorizontalFlip(p=1.0, generator=_generator())
        targets = _paired(_rboxes(*_SWEEP_ANGLES))

        _, out = flip(_image(), targets)

        _assert_one_instance_axis(out)
        assert out.rboxes.shape[0] == len(_SWEEP_ANGLES)


class TestAffineRotated:
    """The affine warps corners and re-fits: exact under a similarity, a fit under shear."""

    def test_identity_affine_returns_the_input_boxes(self) -> None:
        """A zero-parameter affine is the identity on every swept rotated box."""
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, generator=_generator())
        targets = _paired(_rboxes(*_SWEEP_ANGLES))

        _, out = affine(_image(), targets)

        assert float(_box_distance(out.rboxes, canonicalize(targets.rboxes)).max()) < 1e-4

    def test_similarity_warp_is_exact(self) -> None:
        """With no shear the warped corners are reproduced exactly, not approximated."""
        affine = RandomAffine(degrees=20.0, translate=0.05, scale=0.3, shear=0.0, generator=_generator())
        targets = _paired(_rboxes(*_SWEEP_ANGLES))
        corners = rboxes_to_polygons(targets.rboxes)

        _, out = affine(_image(), targets)

        warped = apply_affine_to_points(corners.reshape(-1, 2).double(), affine.last_matrix).reshape(-1, 4, 2)
        assert float(_ring_distance(rboxes_to_polygons(out.rboxes), warped.float()).max()) < 1e-3

    def test_sheared_warp_matches_the_derived_fit_residual(self) -> None:
        """Under shear every corner misses by exactly ``2 |q| sin(s / 2)`` (module docstring)."""
        affine = RandomAffine(degrees=10.0, translate=0.0, scale=0.1, shear=15.0, generator=_generator(5))
        targets = _paired(_rboxes(*_SWEEP_ANGLES))
        corners = rboxes_to_polygons(targets.rboxes)

        _, out = affine(_image(), targets)

        warped = apply_affine_to_points(corners.reshape(-1, 2).double(), affine.last_matrix).reshape(-1, 4, 2)
        residual = _ring_distance(rboxes_to_polygons(out.rboxes), warped.float()).double()
        expected = _parallelogram_fit_residual(warped)
        # The sweep spans orientations the shear barely tilts and ones it tilts hard; the
        # identity must hold for all of them, and the hard cases prove the case is real.
        assert float(expected.max()) > 0.5
        assert torch.allclose(residual, expected, atol=1e-3)

    def test_envelope_tracks_the_rotated_geometry(self) -> None:
        """An unclipped instance's box is exactly the envelope of its rotated box (A40)."""
        affine = RandomAffine(degrees=20.0, translate=0.05, scale=0.2, shear=5.0, generator=_generator())
        targets = _paired(_rboxes(*_SWEEP_ANGLES))

        _, out = affine(_image(), targets)

        assert torch.allclose(out.boxes, _envelopes(out.rboxes), atol=1e-4)

    def test_offcanvas_instance_is_dropped_from_every_modality(self) -> None:
        """An instance outside the canvas leaves, taking its box and label with it."""
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, generator=_generator())
        inside = _rboxes(0.4)
        outside = _rboxes(0.4, cx=-200.0, cy=-200.0)
        targets = _paired(torch.cat([inside, outside], dim=0))

        _, out = affine(_image(), targets)

        _assert_one_instance_axis(out)
        assert out.labels.tolist() == [0]

    def test_clipped_instance_keeps_its_envelope_inside_the_canvas(self) -> None:
        """A straddling instance survives with an envelope cut to the canvas edge."""
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, generator=_generator())
        targets = _paired(_rboxes(0.0, cx=float(_CANVAS) - 4.0, cy=60.0))

        _, out = affine(_image(), targets)

        assert out.boxes[0, 2].item() == pytest.approx(float(_CANVAS))
        assert out.boxes[0, 0].item() == pytest.approx(float(_CANVAS) - 14.0)

    def test_clipped_rotated_box_keeps_its_orientation_and_covers_the_survivor(self) -> None:
        """The re-fit invents no angle: it keeps theta and contains the region that survived."""
        affine = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, generator=_generator())
        targets = _paired(_rboxes(0.6, cx=float(_CANVAS) - 4.0, cy=60.0))

        _, out = affine(_image(), targets)

        fitted = _envelopes(out.rboxes)
        assert out.rboxes[0, 4].item() == pytest.approx(0.6, abs=1e-5)
        assert bool((fitted[:, :2] <= out.boxes[:, :2] + 1e-4).all())
        assert bool((fitted[:, 2:] >= out.boxes[:, 2:] - 1e-4).all())

    def test_zero_angle_box_agrees_with_the_axis_aligned_path(self) -> None:
        """A theta=0 instance gets the same box whether or not the rotated path runs."""
        rotated = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, generator=_generator())
        plain = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, generator=_generator())
        targets = _paired(_rboxes(0.0))

        _, with_rboxes = rotated(_image(), targets)
        _, without = plain(_image(), Targets(boxes=targets.boxes.clone(), labels=targets.labels.clone()))

        assert torch.allclose(with_rboxes.boxes, without.boxes, atol=1e-5)

    def test_polygons_alongside_rboxes_are_rejected(self) -> None:
        """The oriented path carries no polygons, so a mixed input raises (WP-056)."""
        affine = RandomAffine(degrees=10.0, generator=_generator())
        targets = _paired(_rboxes(0.3))
        mixed = Targets(
            boxes=targets.boxes,
            labels=targets.labels,
            polygons=[rboxes_to_polygons(targets.rboxes)[0]],
            rboxes=targets.rboxes,
        )

        with pytest.raises(ValueError, match="no polygons"):
            affine(_image(), mixed)


class TestMosaicRotated:
    """Placement is a pure translation; the canvas edge is what drops instances."""

    def test_placed_box_follows_the_quadrant_offset(self) -> None:
        """The surviving box sits at its image-local position plus the sampled offset."""
        mosaic = MosaicAssembly(target_size=_MOSAIC_SIDE, generator=_generator())
        carried = _paired(_rboxes(0.4, cx=30.0, cy=30.0, w=4.0, h=2.0))
        items = [(_image(_MOSAIC_SIDE), carried)] + [(_image(_MOSAIC_SIDE), Targets.empty()) for _ in range(3)]

        _, out = mosaic(items)

        cx, cy = mosaic.last_center
        expected = _rboxes(0.4, cx=30.0 + cx - _MOSAIC_SIDE, cy=30.0 + cy - _MOSAIC_SIDE, w=4.0, h=2.0)
        assert float(_box_distance(out.rboxes, expected).max()) < 1e-4

    def test_offcanvas_instance_drops_every_modality(self) -> None:
        """An instance placed outside the canvas leaves all three modalities together."""
        mosaic = MosaicAssembly(target_size=_MOSAIC_SIDE, generator=_generator())
        kept = _rboxes(0.4, cx=30.0, cy=30.0, w=4.0, h=2.0)
        lost = _rboxes(0.4, cx=-100.0, cy=-100.0, w=4.0, h=2.0)
        carried = _paired(torch.cat([kept, lost], dim=0))
        items = [(_image(_MOSAIC_SIDE), carried)] + [(_image(_MOSAIC_SIDE), Targets.empty()) for _ in range(3)]

        _, out = mosaic(items)

        _assert_one_instance_axis(out)
        assert out.labels.tolist() == [0]

    def test_envelope_tracks_the_rotated_geometry(self) -> None:
        """The merged box is the envelope of the merged rotated box (A40)."""
        mosaic = MosaicAssembly(target_size=_MOSAIC_SIDE, generator=_generator())
        carried = _paired(_rboxes(0.4, cx=30.0, cy=30.0, w=4.0, h=2.0))
        items = [(_image(_MOSAIC_SIDE), carried)] + [(_image(_MOSAIC_SIDE), Targets.empty()) for _ in range(3)]

        _, out = mosaic(items)

        assert torch.allclose(out.boxes, _envelopes(out.rboxes), atol=1e-4)


class TestMixupRotated:
    """Mixup moves no geometry; copy-paste has no rotated instance to move."""

    def test_rboxes_are_concatenated_untouched(self) -> None:
        """Both inputs' rotated boxes survive the blend bit for bit, in input order."""
        mixup = Mixup(p=1.0, generator=_generator())
        first = _paired(canonicalize(_rboxes(0.3)))
        second = _paired(canonicalize(_rboxes(1.4, cx=20.0, cy=20.0)))

        _, out = mixup([(_image(), first), (_image(), second)])

        _assert_one_instance_axis(out)
        assert torch.equal(out.rboxes, torch.cat([first.rboxes, second.rboxes], dim=0))

    def test_copy_paste_still_rejects_rotated_boxes(self) -> None:
        """Copy-paste transfers polygon masks, which the oriented path does not carry."""
        copy_paste = CopyPaste(p=1.0, generator=_generator())
        carried = _paired(_rboxes(0.3))

        with pytest.raises(NotImplementedError, match="polygon"):
            copy_paste([(_image(), Targets.empty()), (_image(), carried)])


class TestClipRboxesToCanvas:
    """The clip helper: untouched inside, re-fitted across an edge, zeroed outside."""

    def test_inside_box_is_returned_bit_for_bit(self) -> None:
        """A box wholly inside the canvas takes no clip and no re-fit."""
        rboxes = canonicalize(_rboxes(*_SWEEP_ANGLES))

        clipped, envelopes = clip_rboxes_to_canvas(rboxes, height=float(_CANVAS), width=float(_CANVAS))

        assert torch.equal(clipped, rboxes)
        assert torch.allclose(envelopes, _envelopes(rboxes))

    def test_outside_box_collapses_to_zeros(self) -> None:
        """A box with nothing inside the canvas yields a zero box the caller then drops."""
        rboxes = _rboxes(0.4, cx=-500.0, cy=-500.0)

        clipped, envelopes = clip_rboxes_to_canvas(rboxes, height=float(_CANVAS), width=float(_CANVAS))

        assert torch.equal(clipped, torch.zeros_like(clipped))
        assert torch.equal(envelopes, torch.zeros_like(envelopes))

    def test_partial_box_is_refitted_at_its_own_orientation(self) -> None:
        """An axis-aligned box across the right edge keeps theta and loses only the overhang."""
        rboxes = _rboxes(0.0, cx=float(_CANVAS) - 4.0, cy=60.0)

        clipped, envelopes = clip_rboxes_to_canvas(rboxes, height=float(_CANVAS), width=float(_CANVAS))

        assert clipped[0].tolist() == pytest.approx([float(_CANVAS) - 7.0, 60.0, 14.0, 8.0, 0.0])
        assert envelopes[0].tolist() == pytest.approx([float(_CANVAS) - 14.0, 56.0, float(_CANVAS), 64.0])


def _parallelogram_fit_residual(warped: Tensor) -> Tensor:
    """Return the per-corner displacement the rectangle fit costs on a warped parallelogram.

    The warped quad has half-vectors ``p`` (first edge) and ``q`` (last edge) about its
    centre; the fit keeps ``p`` and replaces ``q`` by ``|q|`` along ``p``'s perpendicular,
    displacing every corner by ``2 |q| sin(s / 2)`` for the misalignment ``s``.

    Examples:
        >>> corners = rboxes_to_polygons(_rboxes(0.0)).double()  # an unsheared rectangle
        >>> _parallelogram_fit_residual(corners).tolist()
        [0.0]
    """
    half_first = (warped[:, 1] - warped[:, 0]) / 2.0
    half_second = (warped[:, 3] - warped[:, 0]) / 2.0
    perpendicular = torch.stack([-half_first[:, 1], half_first[:, 0]], dim=1)
    cosine = (half_second * perpendicular).sum(dim=1).abs() / (half_second.norm(dim=1) * perpendicular.norm(dim=1))
    misalignment = torch.arccos(cosine.clamp(max=1.0))
    return 2.0 * half_second.norm(dim=1) * torch.sin(misalignment / 2.0)
