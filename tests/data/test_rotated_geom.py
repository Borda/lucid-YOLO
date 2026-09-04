# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-055 long-edge rotated-geometry primitives (A23, A25).

Covers the long-edge canonical form (``w >= h``, ``theta`` in ``[-pi/4, 3*pi/4)``)
including its exact idempotence and the square tie-break, the ``theta``/``theta + pi``
equivalence a rectangle's 180-degree symmetry implies, the rotated-box/polygon round
trip, and point-in-rotated-rect containment — the primitive WP-061 selects rotated
assignment candidates with.

No RNG is used: every input is written out, so there is no seeding fixture.
"""

from __future__ import annotations

import math

import pytest
import torch

from lucid_yolo.data import canonicalize, points_in_rboxes, polygons_to_rboxes, rboxes_to_polygons
from lucid_yolo.data.rotated_geom import rotated_iou

#: Canonical angle bounds as float32 tensors. The invariant is a float32 one and
#: ``float32(-pi/4)`` is strictly *below* the float64 ``-math.pi / 4``, so comparing a
#: canonicalized boundary angle against the float64 literal would reject a correct
#: result. Comparing tensor-to-scalar casts the scalar to float32, which is the
#: comparison the implementation makes.
_THETA_LOW = torch.tensor(-math.pi / 4, dtype=torch.float32)
_THETA_HIGH = torch.tensor(3 * math.pi / 4, dtype=torch.float32)
#: Largest float32 strictly below the exclusive upper bound of the canonical range.
_JUST_BELOW_HIGH = float(torch.nextafter(_THETA_HIGH, torch.tensor(0.0)))

#: Angle/extent sweep shared by the canonical-form, idempotence and geometry-preservation
#: gates: in-range and out-of-range angles both directions, ``w < h`` inputs, the exact
#: range boundaries and the ambiguous square.
_CANON_CASES = [
    pytest.param(6.0, 3.0, 0.3, id="in-range"),
    pytest.param(6.0, 3.0, 0.0, id="theta-zero"),
    pytest.param(6.0, 3.0, math.pi / 2, id="theta-half-pi"),
    pytest.param(6.0, 3.0, -math.pi / 4, id="lower-bound-exact"),
    pytest.param(6.0, 3.0, 3 * math.pi / 4, id="upper-bound-exact"),
    pytest.param(6.0, 3.0, _JUST_BELOW_HIGH, id="just-below-upper-bound"),
    pytest.param(3.0, 6.0, 0.4, id="needs-swap"),
    pytest.param(3.0, 6.0, -2.9, id="needs-swap-and-wrap"),
    pytest.param(6.0, 3.0, 10 * math.pi + 0.3, id="many-turns-positive"),
    pytest.param(6.0, 3.0, -10 * math.pi - 0.3, id="many-turns-negative"),
    pytest.param(5.0, 5.0, 1.0, id="square-upper-half"),
    pytest.param(5.0, 5.0, -math.pi / 4, id="square-lower-bound"),
]

#: Angles for the ``theta``/``theta + pi`` equivalence gate. Kept at modest magnitude on
#: purpose: the sweep's many-turn angles carry a float32 ulp of ~2e-6 at 30 rad, which
#: swamps the agreement tolerance without saying anything about the symmetry itself.
_THETA_PI_CASES = [
    pytest.param(6.0, 3.0, 0.0, id="axis-aligned"),
    pytest.param(6.0, 3.0, 0.3, id="in-range"),
    pytest.param(6.0, 3.0, -0.7, id="near-lower-bound"),
    pytest.param(6.0, 3.0, 2.2, id="upper-half-range"),
    pytest.param(3.0, 6.0, 0.4, id="needs-swap"),
    pytest.param(5.0, 5.0, 0.2, id="square"),
]

#: Non-square boxes for the polygon round trip; the square is covered separately
#: because a reconstructed square's ``w``/``h`` can differ by one ulp, which legitimately
#: rotates its canonical angle by ``pi/2``.
_ROUNDTRIP_CASES = [
    pytest.param(6.0, 3.0, 0.3, id="in-range"),
    pytest.param(6.0, 3.0, 0.0, id="axis-aligned"),
    pytest.param(3.0, 6.0, 0.4, id="needs-swap"),
    pytest.param(6.0, 3.0, 2.2, id="upper-half-range"),
    pytest.param(9.0, 1.0, -0.6, id="elongated"),
]


def _rbox(w: float, h: float, theta: float) -> torch.Tensor:
    """Return a single ``(1, 5)`` float32 rotated box centred at ``(7, -2)``.

    Examples:
        >>> _rbox(6.0, 3.0, 0.3)
        tensor([[ 7.0000, -2.0000,  6.0000,  3.0000,  0.3000]])
    """
    return torch.tensor([[7.0, -2.0, w, h, theta]], dtype=torch.float32)


def _step_ulps(value: torch.Tensor, steps: int) -> float:
    """Return ``value`` moved ``steps`` float32 ulps, negative steps moving downwards.

    Examples:
        >>> _step_ulps(torch.tensor(1.0), 1) > 1.0
        True
        >>> _step_ulps(torch.tensor(1.0), 0) == 1.0
        True
    """
    current = value.to(torch.float32)
    towards = torch.tensor(-torch.inf if steps < 0 else torch.inf, dtype=torch.float32)
    for _ in range(abs(steps)):
        current = torch.nextafter(current, towards)
    return float(current)


def _point_grid() -> torch.Tensor:
    """Return a deterministic 25x25 point lattice straddling the test boxes' centre.

    The fractional offsets keep every point clear of any test box's edges, so a
    containment comparison cannot flip on a point sitting exactly on a boundary.

    Examples:
        >>> grid = _point_grid()
        >>> grid.shape
        torch.Size([625, 2])
        >>> [round(v, 3) for v in grid[0].tolist()]
        [1.123, -7.923]
    """
    axis = torch.arange(-6.0, 6.5, 0.5, dtype=torch.float32)
    xs, ys = torch.meshgrid(axis + 7.123, axis - 1.923, indexing="ij")
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)


@pytest.mark.parametrize(("w", "h", "theta"), _THETA_PI_CASES)
def test_theta_pi_equivalence(w: float, h: float, theta: float) -> None:
    """``theta`` and ``theta + pi`` describe one rectangle: same canonical box, same corners.

    Bit-equality is not available — building ``theta + pi`` in float32 already rounds —
    so the canonical parameters agree to ``1e-6`` and the corners to ``1e-5``. Corner
    *order* is exact by construction, because :func:`rboxes_to_polygons` winds the
    canonical box rather than the box it was handed.
    """
    stacked = torch.cat([_rbox(w, h, theta), _rbox(w, h, theta + math.pi)], dim=0)

    canonical, polygons = canonicalize(stacked), rboxes_to_polygons(stacked)

    assert torch.allclose(canonical[0], canonical[1], atol=1e-6)
    assert torch.allclose(polygons[0], polygons[1], atol=1e-5)


@pytest.mark.parametrize(("w", "h", "theta"), _CANON_CASES)
def test_canonicalization(w: float, h: float, theta: float) -> None:
    """Canonical output always satisfies ``w >= h`` and ``theta`` in ``[-pi/4, 3*pi/4)``."""
    box = _rbox(w, h, theta)

    out = canonicalize(box)

    assert out.dtype == torch.float32
    assert bool(out[:, 2].ge(out[:, 3]).all())
    assert bool(out[:, 4].ge(_THETA_LOW).all())
    assert bool(out[:, 4].lt(_THETA_HIGH).all())


class TestCanonicalize:
    """Canonicalization is exactly idempotent, geometry-preserving and unique for squares."""

    @pytest.mark.parametrize(("w", "h", "theta"), _CANON_CASES)
    def test_idempotent(self, w: float, h: float, theta: float) -> None:
        """``canonicalize(canonicalize(x))`` equals ``canonicalize(x)`` bit for bit."""
        once = canonicalize(_rbox(w, h, theta))

        twice = canonicalize(once)

        assert torch.equal(twice, once)

    @pytest.mark.parametrize(
        "step",
        [pytest.param(step, id=f"step{step:+d}") for step in (-2, -1, 0, 1, 2)],
    )
    @pytest.mark.parametrize(
        "bound",
        [pytest.param(_THETA_LOW, id="low"), pytest.param(_THETA_HIGH, id="high")],
    )
    def test_angle_an_ulp_either_side_of_a_bound_lands_in_range(self, bound: float, step: int) -> None:
        """Angles within a couple of ulps of either range bound still canonicalize into range.

        Shifting by whole multiples of pi cannot rescue an angle this close to a bound at
        float32: the remainder rounds to pi and the shift returns the input untouched. A
        similarity round trip of a box at exactly -pi/4 reaches that case in practice.
        """
        theta = _step_ulps(bound, step)

        out = canonicalize(_rbox(4.0, 2.0, theta))

        assert bool(out[:, 4].ge(_THETA_LOW).all())
        assert bool(out[:, 4].lt(_THETA_HIGH).all())
        assert torch.equal(canonicalize(out), out)

    @pytest.mark.parametrize(("w", "h", "theta"), _CANON_CASES)
    def test_containment_unchanged(self, w: float, h: float, theta: float) -> None:
        """Canonicalizing leaves the set of points the box contains unchanged."""
        box = _rbox(w, h, theta)
        grid = _point_grid()

        before, after = points_in_rboxes(grid, box), points_in_rboxes(grid, canonicalize(box))

        assert torch.equal(before, after)

    @pytest.mark.parametrize(
        ("theta", "expected"),
        [
            pytest.param(0.3, 0.3, id="already-in-lower-half"),
            pytest.param(-math.pi / 4, -math.pi / 4, id="lower-bound-kept"),
            pytest.param(1.0, 1.0 - math.pi / 2, id="upper-half-folded"),
            pytest.param(math.pi / 4, math.pi / 4 - math.pi / 2, id="quarter-pi-folded"),
            pytest.param(2.2, 2.2 - math.pi / 2, id="near-upper-bound-folded"),
        ],
    )
    def test_square_angle_folded_towards_zero(self, theta: float, expected: float) -> None:
        """An exact square resolves its ``pi/2`` ambiguity into ``[-pi/4, pi/4)``."""
        out = canonicalize(_rbox(5.0, 5.0, theta))

        assert out[0, 4].item() == pytest.approx(expected, abs=1e-6)

    def test_empty_input(self) -> None:
        """Zero boxes canonicalize to a correctly shaped empty tensor."""
        out = canonicalize(torch.zeros((0, 5), dtype=torch.float32))

        assert out.shape == (0, 5)
        assert out.dtype == torch.float32


class TestPolygonConversion:
    """Corner emission and the quad-to-box inverse agree on one rectangle."""

    def test_axis_aligned_corner_order(self) -> None:
        """An unrotated box winds from its top-left corner clockwise as displayed."""
        box = torch.tensor([[5.0, 3.0, 4.0, 2.0, 0.0]], dtype=torch.float32)

        polygons = rboxes_to_polygons(box)

        assert polygons.shape == (1, 4, 2)
        assert polygons[0].tolist() == [[3.0, 2.0], [7.0, 2.0], [7.0, 4.0], [3.0, 4.0]]

    @pytest.mark.parametrize(("w", "h", "theta"), _ROUNDTRIP_CASES)
    def test_round_trip_reproduces_canonical_box(self, w: float, h: float, theta: float) -> None:
        """``polygons_to_rboxes(rboxes_to_polygons(b))`` reproduces ``canonicalize(b)``."""
        box = _rbox(w, h, theta)

        recovered = polygons_to_rboxes(rboxes_to_polygons(box))

        assert torch.allclose(recovered, canonicalize(box), atol=1e-5)

    def test_square_round_trip_preserves_geometry(self) -> None:
        """A square survives the round trip as the same rectangle despite ulp-level w/h drift.

        The tie-break fires on exact equality only, so a reconstructed square whose sides
        differ by one ulp may come back rotated by ``pi/2`` — the same rectangle under a
        different canonical representative, which containment sees through.
        """
        box = _rbox(5.0, 5.0, 0.3)
        grid = _point_grid()

        recovered = polygons_to_rboxes(rboxes_to_polygons(box))

        assert torch.equal(points_in_rboxes(grid, recovered), points_in_rboxes(grid, box))

    @pytest.mark.parametrize("start", [pytest.param(k, id=f"start{k}") for k in range(4)])
    @pytest.mark.parametrize("reverse", [pytest.param(False, id="forward"), pytest.param(True, id="reversed")])
    def test_ring_order_does_not_change_the_box(self, start: int, reverse: bool) -> None:
        """Every cyclic rotation and reversal of one quad's ring yields the same canonical box.

        DOTA writes its eight coordinates in whatever order the annotator drew them, so all
        eight orderings of a given ring must load as one box.
        """
        box = _rbox(6.0, 3.0, 0.3)
        ring = rboxes_to_polygons(box).roll(start, dims=1)

        reordered = polygons_to_rboxes(ring.flip(dims=(1,)) if reverse else ring)

        assert torch.allclose(reordered, canonicalize(box), atol=1e-5)

    def test_approximate_rectangle_averages_opposite_sides(self) -> None:
        """A quad perturbed off rectangularity returns the mean of each opposite side pair."""
        quad = torch.tensor([[[0.0, -1.0], [10.0, -2.0], [10.0, 2.0], [0.0, 1.0]]], dtype=torch.float32)

        out = polygons_to_rboxes(quad)

        # Long sides both measure hypot(10, 1); short sides measure 4 and 2 -> mean 3.
        # The centre is the vertex centroid, and the long-edge direction is the mean of
        # the two long edges, which cancels the +-1 vertical perturbation back to theta=0.
        assert out[0, :2].tolist() == [5.0, 0.0]
        assert out[0, 2].item() == pytest.approx(math.hypot(10.0, 1.0), abs=1e-5)
        assert out[0, 3].item() == pytest.approx(3.0, abs=1e-5)
        assert out[0, 4].item() == pytest.approx(0.0, abs=1e-6)

    def test_empty_inputs(self) -> None:
        """Zero boxes and zero quads convert to correctly shaped empty tensors."""
        polygons = rboxes_to_polygons(torch.zeros((0, 5), dtype=torch.float32))
        boxes = polygons_to_rboxes(torch.zeros((0, 4, 2), dtype=torch.float32))

        assert polygons.shape == (0, 4, 2)
        assert boxes.shape == (0, 5)


class TestPointsInRboxes:
    """Point-in-rotated-rect containment, the WP-061 assignment primitive (A25)."""

    def test_centre_is_inside(self) -> None:
        """Every box contains its own centre."""
        boxes = torch.tensor([[7.0, -2.0, 6.0, 3.0, 0.3], [0.0, 0.0, 4.0, 4.0, -1.0]], dtype=torch.float32)

        inside = points_in_rboxes(boxes[:, :2], boxes)

        assert bool(inside.diagonal().all())

    def test_point_just_beyond_a_corner_is_outside(self) -> None:
        """A point nudged diagonally past a corner falls outside."""
        box = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]], dtype=torch.float32)
        beyond = torch.tensor([[2.001, 1.001]], dtype=torch.float32)

        inside = points_in_rboxes(beyond, box)

        assert not bool(inside[0, 0])

    def test_boundary_is_inclusive(self) -> None:
        """A point exactly on a corner counts as inside."""
        box = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]], dtype=torch.float32)
        corner = torch.tensor([[2.0, 1.0]], dtype=torch.float32)

        inside = points_in_rboxes(corner, box)

        assert bool(inside[0, 0])

    def test_rotated_box_excludes_a_point_its_envelope_includes(self) -> None:
        """A point inside the axis-aligned envelope but outside the rotated rect is rejected."""
        box = torch.tensor([[0.0, 0.0, 10.0, 2.0, math.pi / 4]], dtype=torch.float32)
        point = torch.tensor([[3.0, -3.0]], dtype=torch.float32)
        corners = rboxes_to_polygons(box)[0]

        inside = points_in_rboxes(point, box)

        assert bool((point[0] >= corners.amin(dim=0)).all())
        assert bool((point[0] <= corners.amax(dim=0)).all())
        assert not bool(inside[0, 0])

    def test_output_is_points_by_boxes(self) -> None:
        """The result is a ``(P, M)`` bool tensor over every point/box pair."""
        points = torch.zeros((5, 2), dtype=torch.float32)
        boxes = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]], dtype=torch.float32).repeat(3, 1)

        inside = points_in_rboxes(points, boxes)

        assert inside.shape == (5, 3)
        assert inside.dtype == torch.bool

    @pytest.mark.parametrize(
        ("n_points", "n_boxes"),
        [
            pytest.param(0, 3, id="no-points"),
            pytest.param(4, 0, id="no-boxes"),
            pytest.param(0, 0, id="neither"),
        ],
    )
    def test_empty_inputs_give_correctly_shaped_output(self, n_points: int, n_boxes: int) -> None:
        """Empty points or boxes give an empty bool tensor of the right shape."""
        points = torch.zeros((n_points, 2), dtype=torch.float32)
        boxes = torch.zeros((n_boxes, 5), dtype=torch.float32)

        inside = points_in_rboxes(points, boxes)

        assert inside.shape == (n_points, n_boxes)
        assert inside.dtype == torch.bool


class TestShapeValidation:
    """Wrong-shaped inputs are rejected rather than silently broadcast."""

    def test_rboxes_wrong_width(self) -> None:
        """A four-column tensor is not a rotated-box tensor."""
        with pytest.raises(ValueError, match=r"rboxes must be \(N, 5\)"):
            canonicalize(torch.zeros((2, 4), dtype=torch.float32))

    def test_points_wrong_width(self) -> None:
        """Points must be ``(P, 2)``."""
        with pytest.raises(ValueError, match=r"points must be \(N, 2\)"):
            points_in_rboxes(torch.zeros((2, 3), dtype=torch.float32), torch.zeros((1, 5), dtype=torch.float32))

    def test_polygons_wrong_shape(self) -> None:
        """Quads must be ``(M, 4, 2)``."""
        with pytest.raises(ValueError, match=r"polygons must be \(M, 4, 2\)"):
            polygons_to_rboxes(torch.zeros((2, 5, 2), dtype=torch.float32))


class TestRotatedIouRejectsImpossibleGeometry:
    """A row that is not a rectangle scores ``0.0``, never a plausible overlap (WP-170).

    ``rotated_iou``'s docstring has always promised this for zero and negative
    extents, but the promise was left to the winding to keep and the winding does
    not keep it: :func:`canonicalize` swaps a pair of signed extents back into
    ``w >= h`` order, and two negative extents are a half-turn that rebuilds the
    very rectangle the caller wrote as impossible. Non-finite rows reached the
    same zero for an unrelated reason -- every comparison against ``NaN`` is false
    -- which is a different statement about a different failure and was worth
    making on purpose.
    """

    #: A well-formed 4x2 box at the origin, the honest counterpart of the rows below.
    VALID = (0.0, 0.0, 4.0, 2.0, 0.0)

    def test_two_negative_extents_no_longer_score_a_perfect_match(self) -> None:
        """``[-4, -2]`` against itself scored ``1.0``; it is an invalid row, so ``0.0``.

        The half-turn case: negating both extents names the same rectangle, so
        canonicalization produced a correctly wound polygon and the shoelace had
        nothing left to object to.
        """
        negative = torch.tensor([[0.0, 0.0, -4.0, -2.0, 0.0]])

        assert float(rotated_iou(negative, negative)) == 0.0

    def test_a_negative_row_does_not_match_a_real_box(self) -> None:
        """``[-4, -2]`` scored ``1.0`` against the honest ``[4, 2]`` describing that region.

        Worse than the self-pair: an impossible row was not merely self-consistent,
        it was accepted as an exact match for a real detection.
        """
        negative = torch.tensor([[0.0, 0.0, -4.0, -2.0, 0.0]])
        valid = torch.tensor([self.VALID])

        assert float(rotated_iou(negative, valid)) == 0.0
        assert float(rotated_iou(valid, negative)) == 0.0

    @pytest.mark.parametrize("extents", [(-4.0, 2.0), (4.0, -2.0), (0.0, 2.0), (4.0, 0.0), (0.0, 0.0)])
    def test_mixed_sign_and_zero_extents_score_zero(self, extents: tuple[float, float]) -> None:
        """Every non-positive extent combination encloses no area, against anything."""
        width, height = extents
        invalid = torch.tensor([[0.0, 0.0, width, height, 0.0]])
        valid = torch.tensor([self.VALID])

        assert float(rotated_iou(invalid, invalid)) == 0.0
        assert float(rotated_iou(invalid, valid)) == 0.0

    @pytest.mark.parametrize("column", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_in_any_column_scores_zero(self, column: int, bad: float) -> None:
        """A non-finite ``cx``, ``cy``, ``w``, ``h`` or ``theta`` yields ``0.0``, not ``NaN``.

        Stated by the validity mask now rather than reached by accident through
        false ``NaN`` comparisons, so ``inf`` -- which compares perfectly well --
        is covered by the same rule as ``nan``.
        """
        row = torch.tensor([self.VALID])
        row[0, column] = bad
        valid = torch.tensor([self.VALID])

        overlap = rotated_iou(row, valid)

        assert float(overlap) == 0.0
        assert bool(torch.isfinite(overlap).all())

    def test_valid_pairs_are_untouched_by_the_validity_mask(self) -> None:
        """The refusal costs nothing on well-formed geometry: identity and a known overlap."""
        valid = torch.tensor([self.VALID])
        shifted = torch.tensor([[2.0, 0.0, 4.0, 2.0, 0.0]])  # half its width along +x

        assert float(rotated_iou(valid, valid)) == 1.0
        assert float(rotated_iou(valid, shifted)) == pytest.approx(1.0 / 3.0, abs=1e-4)

    def test_an_invalid_row_does_not_poison_its_neighbours(self) -> None:
        """Only the pairs involving the bad row collapse; the rest of the matrix stands.

        The mask is per-box and folded into the pair grid, so a single corrupt
        detection in a batch cannot zero the overlaps of the good ones beside it.
        """
        rows = torch.tensor([self.VALID, (0.0, 0.0, -4.0, -2.0, 0.0), (2.0, 0.0, 4.0, 2.0, 0.0)])

        overlap = rotated_iou(rows, rows)

        assert float(overlap[0, 0]) == 1.0
        assert float(overlap[2, 2]) == 1.0
        assert float(overlap[0, 2]) == pytest.approx(1.0 / 3.0, abs=1e-4)
        assert float(overlap[1].abs().max()) == 0.0  # the invalid row, against everything
        assert float(overlap[:, 1].abs().max()) == 0.0
