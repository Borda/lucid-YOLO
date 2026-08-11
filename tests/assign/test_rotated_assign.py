# SPDX-License-Identifier: Apache-2.0
"""Unit tests for rotated containment in TAL and STAL (WP-061, A25).

The rotated path is one optional argument, ``gt_rboxes``, on the assigner call: it
replaces the centre-inside candidate test with the point-in-rotated-rect test of
:func:`lucid_yolo.data.rotated_geom.points_in_rboxes`, and STAL clamps the rotated
``(w, h)`` instead of the axis-aligned ones. Nothing else moves — scoring, targets,
and weights keep running on the axis-aligned ``gt_boxes``.

The claims covered. A tiny rotated ground truth gets zero candidates under
vanilla TAL and gains them under the STAL surrogate (the rotated analogue of
``test_stal.py::test_tiny_box_gains_candidates``), and its targets stay the
axis-aligned box. Containment genuinely respects rotation: a 45-degree square admits
its diamond of anchors and rejects the corner anchors that lie inside its axis-aligned
envelope. Candidate sets stay separated per image and per ground truth, which is what
pins the ``(B, N, A)`` layout of the rotated mask. The rotated surrogate clamps each
edge independently and re-canonicalizes when the short edge overtakes the long one.
And the axis-aligned assignment is bit-identical to the values frozen at 485ac14, the
commit before this work package — every expected value below was produced by running
that code, not by running the code under test.
"""

import math

import pytest
import torch

from lucid_yolo.assign import (
    AssignResult,
    SmallTargetAssigner,
    TaskAlignedAssigner,
    UniqueAssigner,
    make_anchor_points,
    surrogate_rboxes,
)

#: 6x6 ground truth turned 45 degrees about (8, 8); no stride-8 anchor centre inside it.
_TINY_RBOX = torch.tensor([[[8.0, 8.0, 6.0, 6.0, math.pi / 4]]])
#: The axis-aligned box the tiny scene scores against; candidacy never reads it.
_TINY_GT = torch.tensor([[[5.0, 5.0, 11.0, 11.0]]])
#: Anchors (4, 4), (12, 4), (4, 12), (12, 12) — inside the 16x16 rotated surrogate.
_TINY_STAL_CANDIDATES = (0, 1, 4, 5)

#: 16x16 ground truth turned 45 degrees about (12, 12) — a diamond on the anchor grid.
_DIAMOND_RBOX = torch.tensor([[[12.0, 12.0, 16.0, 16.0, math.pi / 4]]])
#: The same square unturned: the axis-aligned box that scores the diamond scene.
_DIAMOND_GT = torch.tensor([[[4.0, 4.0, 20.0, 20.0]]])
#: Anchors (12, 4), (4, 12), (12, 12), (20, 12), (12, 20) — the diamond's own points.
_DIAMOND_CANDIDATES = (1, 4, 5, 6, 9)
#: Anchors (4, 4), (20, 4), (4, 20), (20, 20) — inside the envelope, outside the diamond.
_ENVELOPE_ONLY = (0, 2, 8, 10)
#: Half-extent of the diamond's axis-aligned envelope: ``(16 / 2) * sqrt(2)``.
_DIAMOND_ENVELOPE_HALF = 8.0 * math.sqrt(2.0)


def _rotated_scene(
    gt_rboxes: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Assemble a single-image scene on a 4x4 stride-8 grid around one rotated GT.

    Anchor centres are x, y in {4, 12, 20, 28}. Class-0 score is 0.9 at every anchor
    and every predicted box equals ``gt_boxes``, so the alignment metric is flat and
    the positive set is decided purely by candidacy.
    """
    points, _ = make_anchor_points([(4, 4)], [8])  # (16, 2)
    scores = torch.full((1, 16, 1), 0.9)
    pred_boxes = gt_boxes.expand(1, 16, 4).contiguous()
    gt_labels = torch.tensor([[0]])
    gt_mask = torch.tensor([[True]])
    return scores, pred_boxes, points, gt_boxes, gt_labels, gt_mask, gt_rboxes


def _positive_indices(result: AssignResult) -> tuple[int, ...]:
    """Anchor indices marked foreground in a single-image assignment result."""
    return tuple(int(index) for index in result.fg_mask[0].nonzero().flatten())


def test_tiny_rotated_gt() -> None:
    """A 6x6 rotated GT yields zero positives under vanilla TAL and four under STAL."""
    scene = _rotated_scene(_TINY_RBOX, _TINY_GT)

    vanilla = TaskAlignedAssigner(topk=4)(*scene)
    stal = SmallTargetAssigner(topk=4)(*scene)

    assert int(vanilla.fg_mask.sum()) == 0
    assert int(stal.fg_mask.sum()) >= 1
    assert _positive_indices(stal) == _TINY_STAL_CANDIDATES


def test_rotated_targets_stay_axis_aligned() -> None:
    """STAL positives on a rotated GT carry the axis-aligned box, not the surrogate."""
    scene = _rotated_scene(_TINY_RBOX, _TINY_GT)

    out = SmallTargetAssigner(topk=4)(*scene)

    assert torch.equal(out.target_boxes[0, _TINY_STAL_CANDIDATES[0]], _TINY_GT[0, 0])
    assert torch.equal(out.target_labels[0, _TINY_STAL_CANDIDATES[0]], torch.tensor(0))
    assert torch.equal(out.align_weights[0, list(_TINY_STAL_CANDIDATES)], torch.ones(4))


def test_rotation_excludes_envelope_corners() -> None:
    """A 45-degree square admits its diamond of anchors and rejects its envelope corners."""
    rotated_scene = _rotated_scene(_DIAMOND_RBOX, _DIAMOND_GT)
    axis_aligned_scene = rotated_scene[:-1]  # same scene, rotated candidacy switched off

    rotated = TaskAlignedAssigner(topk=16)(*rotated_scene)
    axis_aligned = TaskAlignedAssigner(topk=16)(*axis_aligned_scene)

    assert _positive_indices(rotated) == _DIAMOND_CANDIDATES
    assert _positive_indices(axis_aligned) == tuple(sorted(_DIAMOND_CANDIDATES + _ENVELOPE_ONLY))
    # Anchor 0 sits at (4, 4): 8 px from the centre on each axis, so inside the envelope
    # that spans 12 +- 11.3137 — and 11.3137 px along the box's own long axis, outside it.
    assert abs(4.0 - 12.0) <= _DIAMOND_ENVELOPE_HALF
    assert not bool(rotated.fg_mask[0, 0])
    assert bool(axis_aligned.fg_mask[0, 0])


def test_rotated_candidates_are_per_image_and_per_gt() -> None:
    """Two images with two differently turned GTs each keep their own candidate sets."""
    points, _ = make_anchor_points([(4, 4)], [8])  # (16, 2)
    rboxes = torch.tensor(
        [
            [[12.0, 12.0, 16.0, 16.0, math.pi / 4], [20.0, 28.0, 16.0, 4.0, 0.0]],  # diamond, flat bar
            [[4.0, 12.0, 16.0, 4.0, math.pi / 2], [12.0, 12.0, 16.0, 16.0, math.pi / 4]],  # bar, padded
        ]
    )
    gt_boxes = _DIAMOND_GT.expand(2, 2, 4).contiguous()
    scores = torch.full((2, 16, 1), 0.9)
    pred_boxes = _DIAMOND_GT.expand(2, 16, 4).contiguous()
    gt_labels = torch.zeros(2, 2, dtype=torch.long)
    gt_mask = torch.tensor([[True, True], [True, False]])  # image 1's diamond is padding

    out = TaskAlignedAssigner(topk=16)(scores, pred_boxes, points, gt_boxes, gt_labels, gt_mask, rboxes)

    expected = torch.full((2, 16), -1, dtype=torch.long)
    expected[0, list(_DIAMOND_CANDIDATES)] = 0  # (12, 4), (4, 12), (12, 12), (20, 12), (12, 20)
    expected[0, [13, 14, 15]] = 1  # the flat bar along y = 28
    expected[1, [0, 4, 8]] = 0  # the bar turned a quarter along x = 4
    assert torch.equal(out.gt_index, expected)  # image 1's padded diamond claims nothing


@pytest.mark.parametrize(
    ("rbox", "expected"),
    [
        pytest.param(
            [0.0, 0.0, 20.0, 6.0, 0.0],  # short edge below s_min, long edge above
            [0.0, 0.0, 20.0, 16.0, 0.0],  # h -> 16, still shorter than w: no swap
            id="w20-h6-height-clamped",
        ),
        pytest.param(
            [0.0, 0.0, 10.0, 4.0, 0.0],  # h -> 16 overtakes w = 10
            [0.0, 0.0, 16.0, 10.0, math.pi / 2],  # canonical again: pair swapped, theta + pi/2
            id="w10-h4-swap-after-clamp",
        ),
        pytest.param(
            [0.0, 0.0, 6.0, 6.0, 0.0],  # both edges below s_min
            [0.0, 0.0, 16.0, 16.0, 0.0],  # both -> 16, angle untouched
            id="w6-h6-both-clamped",
        ),
        pytest.param(
            [0.0, 0.0, 20.0, 20.0, 0.5],  # both edges at or above s_min
            [0.0, 0.0, 20.0, 20.0, 0.5],  # untouched
            id="w20-h20-untouched",
        ),
    ],
)
def test_rotated_surrogate_clamps_per_edge(rbox: list[float], expected: list[float]) -> None:
    """surrogate_rboxes clamps each rotated edge independently, keeping centre and rectangle."""
    boxes = torch.tensor([[rbox]])  # (1, 1, 5)

    surrogate = surrogate_rboxes(boxes, s_min=8.0, s_ref=16.0)[0, 0]

    assert torch.allclose(surrogate, torch.tensor(expected), atol=1e-6)
    assert torch.equal(surrogate[:2], boxes[0, 0, :2])  # centre never moves
    assert float(surrogate[2]) >= float(surrogate[3])  # long-edge convention holds


def _frozen_scene() -> tuple[torch.Tensor, ...]:
    """Rebuild the deterministic two-image scene the frozen expectations were taken on.

    Two ground truths per image (the second image's second slot is padding), scores and
    predicted boxes derived from the anchor index by arithmetic so the scene carries no
    RNG and no stored fixture.
    """
    points, _ = make_anchor_points([(4, 4)], [8])  # (16, 2)
    anchor = torch.arange(16, dtype=torch.float32)
    scores = torch.stack(
        [torch.stack([((anchor * 3 + b * 5 + c * 7) % 10) / 10 for c in range(2)], dim=-1) for b in range(2)]
    )  # (2, 16, 2)
    half = (4.0 + (anchor % 3) * 2.0).unsqueeze(-1)  # (16, 1)
    pred = torch.cat([points - half, points + half], dim=-1)  # (16, 4)
    pred_boxes = torch.stack([pred, pred + 1.0])  # (2, 16, 4)
    gt_boxes = torch.tensor(
        [
            [[2.0, 2.0, 18.0, 18.0], [10.0, 10.0, 30.0, 30.0]],
            [[0.0, 0.0, 32.0, 32.0], [20.0, 20.0, 28.0, 28.0]],
        ]
    )
    gt_labels = torch.tensor([[0, 1], [1, 0]])
    gt_mask = torch.tensor([[True, True], [True, False]])
    return scores, pred_boxes, points, gt_boxes, gt_labels, gt_mask


def _expected_result(rows: list[tuple[int, int, int, int, list[float], float]]) -> AssignResult:
    """Expand frozen foreground rows into a full :class:`AssignResult` over 2 images, 16 anchors.

    Each row is ``(image, anchor, gt_index, label, box, align_weight)``; every anchor not
    listed carries the documented background sentinels (``-1`` index and label, zero box,
    zero weight), so a stray positive breaks the comparison just as a wrong value does.
    """
    fg_mask = torch.zeros(2, 16, dtype=torch.bool)
    gt_index = torch.full((2, 16), -1, dtype=torch.long)
    labels = torch.full((2, 16), -1, dtype=torch.long)
    boxes = torch.zeros(2, 16, 4)
    weights = torch.zeros(2, 16)
    for image, anchor, gt, label, box, weight in rows:
        fg_mask[image, anchor] = True
        gt_index[image, anchor] = gt
        labels[image, anchor] = label
        boxes[image, anchor] = torch.tensor(box)
        weights[image, anchor] = weight
    return AssignResult(fg_mask, gt_index, labels, boxes, weights)


#: Assignment frozen at 485ac14 (pre-WP-061) on ``_frozen_scene``, as
#: ``(image, anchor, gt_index, label, target_box, align_weight)`` foreground rows.
#: Regenerate only from a commit whose axis-aligned path is known good, never from the
#: code under test. TAL and STAL share these rows: no ground truth in the scene is tiny,
#: so the STAL surrogate is the identity here and must not move a single value.
_FROZEN_O2M = [
    (0, 1, 0, 0, [2.0, 2.0, 18.0, 18.0], 0.006481748539954424),
    (0, 4, 0, 0, [2.0, 2.0, 18.0, 18.0], 0.004321165382862091),
    (0, 5, 0, 0, [2.0, 2.0, 18.0, 18.0], 0.6202530860900879),
    (0, 10, 1, 1, [10.0, 10.0, 30.0, 30.0], 0.35999977588653564),
    (0, 14, 1, 1, [10.0, 10.0, 30.0, 30.0], 0.2395859658718109),
    (1, 2, 0, 1, [0.0, 0.0, 32.0, 32.0], 0.06244543939828873),
    (1, 5, 0, 1, [0.0, 0.0, 32.0, 32.0], 0.249998539686203),
    (1, 8, 0, 1, [0.0, 0.0, 32.0, 32.0], 0.046834077686071396),
]
#: The same scene under the one-to-one reduction: one positive per ground truth.
_FROZEN_UNIQUE = [
    (0, 5, 0, 0, [2.0, 2.0, 18.0, 18.0], 0.6202530860900879),
    (0, 10, 1, 1, [10.0, 10.0, 30.0, 30.0], 0.35999977588653564),
    (1, 5, 0, 1, [0.0, 0.0, 32.0, 32.0], 0.249998539686203),
]


@pytest.mark.parametrize(
    ("assigner", "frozen"),
    [
        pytest.param(TaskAlignedAssigner(topk=3), _FROZEN_O2M, id="tal"),
        pytest.param(SmallTargetAssigner(topk=3), _FROZEN_O2M, id="stal"),
        pytest.param(UniqueAssigner(topk=3), _FROZEN_UNIQUE, id="unique"),
    ],
)
def test_axis_aligned_assignment_is_bit_identical(
    assigner: TaskAlignedAssigner,
    frozen: list[tuple[int, int, int, int, list[float], float]],
) -> None:
    """Omitting gt_rboxes reproduces the pre-WP-061 assignment exactly, in every field."""
    out = assigner(*_frozen_scene())

    expected = _expected_result(frozen)
    assert torch.equal(out.fg_mask, expected.fg_mask)
    assert torch.equal(out.gt_index, expected.gt_index)
    assert torch.equal(out.target_labels, expected.target_labels)
    assert torch.equal(out.target_boxes, expected.target_boxes)
    assert torch.equal(out.align_weights, expected.align_weights)


def test_rboxes_must_match_gt_boxes_shape() -> None:
    """A rotated batch that does not pair with gt_boxes is rejected, not broadcast."""
    scene = _rotated_scene(_TINY_RBOX, _TINY_GT)
    mismatched = _TINY_RBOX.expand(1, 3, 5).contiguous()  # three rotated GTs, one axis-aligned

    with pytest.raises(ValueError, match=r"gt_rboxes must be \(1, 1, 5\)"):
        TaskAlignedAssigner(topk=4)(*scene[:-1], mismatched)
