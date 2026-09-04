# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the anchor-point grid (WP-025).

Covers the A11 cell-centre placement ``(i + 0.5) * stride``, the row-major
within-level and level-major across-level ordering, the per-anchor stride
mapping, the 8400-point count at a 640 input, and argument validation. Expected
coordinates are enumerated by hand, not lifted from any reference.
"""

import pytest
import torch

from lucid_yolo.assign import anchor_grid, make_anchor_points


def test_single_level_centers_are_row_major() -> None:
    """A 2x2 grid at stride 4 yields the four cell centres in row-major order."""
    points, strides = make_anchor_points([(2, 2)], [4])
    expected = torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]])
    assert torch.equal(points, expected)
    assert torch.equal(strides, torch.tensor([4.0, 4.0, 4.0, 4.0]))


def test_non_square_level_uses_height_rows_width_cols() -> None:
    """A (height=1, width=3) grid spans three columns in one row."""
    points, _ = make_anchor_points([(1, 3)], [2])
    # Centres: ((j+0.5)*2, 0.5*2) for j in 0..2 -> x in {1,3,5}, y == 1.
    expected = torch.tensor([[1.0, 1.0], [3.0, 1.0], [5.0, 1.0]])
    assert torch.equal(points, expected)


def test_multi_level_concatenates_in_level_order() -> None:
    """Levels concatenate in order; stride_per_anchor labels each point's level."""
    points, strides = make_anchor_points([(2, 2), (1, 1)], [4, 8])
    # Level 0: 4 points (stride 4); level 1: 1 point at (4, 4) (stride 8).
    assert points.shape == (5, 2)
    assert torch.equal(points[4], torch.tensor([4.0, 4.0]))
    assert torch.equal(strides, torch.tensor([4.0, 4.0, 4.0, 4.0, 8.0]))


def test_total_count_is_8400_at_640_input() -> None:
    """The canonical [8,16,32] pyramid at 640 px produces 8400 anchor points."""
    points, strides = make_anchor_points([(80, 80), (40, 40), (20, 20)], [8, 16, 32])
    assert points.shape == (8400, 2)
    assert strides.shape == (8400,)
    # First point of the finest level sits at the stride-8 cell centre.
    assert torch.equal(points[0], torch.tensor([4.0, 4.0]))


def test_length_mismatch_raises() -> None:
    """Unequal feature_sizes / strides lengths are rejected."""
    with pytest.raises(ValueError, match="equal length"):
        make_anchor_points([(2, 2)], [4, 8])


@pytest.mark.parametrize(
    "canvas",
    [
        pytest.param((100, 100), id="floors-to-96"),
        pytest.param((641, 641), id="one-past-640"),
        pytest.param((640, 641), id="width-only"),
        pytest.param((0, 640), id="empty-height"),
        pytest.param((-640, 640), id="negative-height"),
    ],
)
def test_a_canvas_the_strides_do_not_divide_is_refused(canvas: tuple[int, int]) -> None:
    """A canvas that does not tile every level raises instead of flooring silently (WP-171).

    The docstring has declared the precondition since WP-025 and nothing checked it, so
    the floor division answered anyway: ``(100, 100)`` returned the grid of a 96 px
    canvas, and ``(641, 641)`` returned the *same 8400 anchors as 640* — a grid pairing
    every prediction with the wrong pixel, with no error anywhere to read as wrong. The
    two negative sides are separate cases because divisibility alone does not refuse
    them: ``0 % 32`` and ``-640 % 32`` are both zero, so positivity is its own condition.
    """
    with pytest.raises(ValueError, match="canvas"):
        anchor_grid(canvas, torch.device("cpu"))


def test_the_640_canvas_the_evaluators_use_still_builds_its_8400_anchors() -> None:
    """The guard leaves the deployed canvas untouched, at the size every protocol runs.

    A precondition added to a function four call sites already depend on is worth
    asserting from the accepting side too: the refusal above is only correct if 640 —
    what both COCO evaluators, single-image inference and the export graph pass — is
    still built, and built identically.
    """
    points, strides = anchor_grid((640, 640), torch.device("cpu"))

    assert points.shape == (8400, 2)
    assert sorted(set(strides.tolist())) == [8.0, 16.0, 32.0]
