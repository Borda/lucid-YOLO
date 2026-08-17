# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the one-to-one UniqueAssigner (WP-028).

UniqueAssigner extends STAL with a ``topk2 = 1`` reduction: after the base
pipeline (small-target candidate filter, alignment scoring, top-``k`` select,
conflict resolution) produces its positive mask, each ground truth keeps only
its single highest-alignment anchor. These tests cover the invariant directly on
a 4x4 stride-8 grid whose 16 anchor centres sit at x, y in {4, 12, 20, 28}: a GT
with many candidates collapses to exactly one positive; every real GT on a
multi-GT scene keeps exactly one; and the two documented collision outcomes —
the losing GT falls back to another candidate when one remains, and gets zero
when its only candidate is claimed by a better-aligned GT. Expected anchors are
computed by hand from the grid geometry and the alignment metric
``t = s * u**6``, never lifted from a reference implementation.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from lucid_yolo.assign import SmallTargetAssigner, UniqueAssigner, make_anchor_points

#: The anchor whose centre (12, 12) is the lone stride-8 candidate of an 8x8 GT.
_CENTRE_ANCHOR = 5


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed every RNG source before each test for deterministic tensors."""
    torch.manual_seed(0)


def _grid_points() -> Tensor:
    """Return the 16 stride-8 anchor centres of a 4x4 grid; x fastest, row-major.

    Examples:
        >>> points = _grid_points()
        >>> points.shape
        torch.Size([16, 2])
        >>> points[:2].tolist()
        [[4.0, 4.0], [12.0, 4.0]]
    """
    points, _ = make_anchor_points([(4, 4)], [8])
    return points


def test_reduces_many_candidates_to_one() -> None:
    """A GT with many STAL candidates keeps exactly one positive; STAL keeps several."""
    gt = torch.tensor([[[0.0, 0.0, 32.0, 32.0]]])  # centre (16, 16); all 16 anchors inside
    pred_boxes = torch.zeros(1, 16, 4)
    pred_boxes[0, _CENTRE_ANCHOR] = gt[0, 0]  # only this anchor aligns (IoU 1.0)
    scores = torch.full((1, 16, 1), 0.9)
    labels = torch.tensor([[0]])
    mask = torch.tensor([[True]])
    points = _grid_points()

    stal = SmallTargetAssigner(topk=7)(scores, pred_boxes, points, gt, labels, mask)
    unique = UniqueAssigner(topk=7)(scores, pred_boxes, points, gt, labels, mask)

    assert int(stal.fg_mask.sum()) > 1
    assert int(unique.fg_mask.sum()) == 1
    assert int(unique.gt_index[0, _CENTRE_ANCHOR]) == 0  # the IoU-1.0 anchor wins


def test_exactly_one_positive_per_gt() -> None:
    """Two well-separated GTs each keep exactly one positive on disjoint anchors."""
    gt = torch.tensor([[[0.0, 0.0, 16.0, 16.0], [16.0, 16.0, 32.0, 32.0]]])  # (1, 2, 4)
    pred_boxes = torch.zeros(1, 16, 4)
    pred_boxes[0, 0] = gt[0, 0]  # anchor (4, 4): best for GT 0
    pred_boxes[0, 15] = gt[0, 1]  # anchor (28, 28): best for GT 1
    scores = torch.full((1, 16, 1), 0.9)
    labels = torch.tensor([[0, 0]])
    mask = torch.tensor([[True, True]])

    out = UniqueAssigner(topk=7)(scores, pred_boxes, _grid_points(), gt, labels, mask)

    assert int(out.fg_mask.sum()) == 2
    assert int((out.gt_index[0] == 0).sum()) == 1
    assert int((out.gt_index[0] == 1).sum()) == 1
    assert int(out.gt_index[0, 0]) == 0
    assert int(out.gt_index[0, 15]) == 1


def test_collision_loser_falls_back_when_candidate_remains() -> None:
    """When two GTs' best anchor collides, the loser keeps another candidate.

    GT 0 (8x8) has the single candidate anchor (12, 12); GT 1 (32x32) covers
    every anchor including (12, 12). The shared anchor predicts GT 0's box, so it
    aligns far better with GT 0 (IoU 1.0) than GT 1 (IoU 0.0625); conflict
    resolution gives it to GT 0, and GT 1 falls back to one of its remaining
    candidates. Both GTs end with exactly one positive on distinct anchors.
    """
    gt = torch.tensor([[[8.0, 8.0, 16.0, 16.0], [0.0, 0.0, 32.0, 32.0]]])
    pred_boxes = torch.zeros(1, 16, 4)
    pred_boxes[0, _CENTRE_ANCHOR] = gt[0, 0]  # aligns with GT 0, not GT 1
    scores = torch.full((1, 16, 1), 0.9)
    labels = torch.tensor([[0, 0]])
    mask = torch.tensor([[True, True]])

    out = UniqueAssigner(topk=7)(scores, pred_boxes, _grid_points(), gt, labels, mask)

    assert int((out.gt_index[0] == 0).sum()) == 1
    assert int((out.gt_index[0] == 1).sum()) == 1
    assert int(out.gt_index[0, _CENTRE_ANCHOR]) == 0  # GT 0 wins the contested anchor
    gt1_anchor = int((out.gt_index[0] == 1).nonzero()[0])
    assert gt1_anchor != _CENTRE_ANCHOR  # GT 1 fell back to a different anchor


def test_collision_loser_gets_zero_without_remaining_candidate() -> None:
    """Two identical single-candidate GTs: the lower-index GT wins, the other gets zero.

    Both GTs are the same 8x8 box whose only stride-8 candidate is anchor
    (12, 12), so their alignment ties. Conflict resolution breaks the tie by the
    lowest GT index (GT 0), and GT 1 — left with no other candidate — keeps no
    positive. This is the irreducible one-to-one collision: an anchor cannot
    serve two GTs.
    """
    gt = torch.tensor([[[8.0, 8.0, 16.0, 16.0], [8.0, 8.0, 16.0, 16.0]]])
    pred_boxes = torch.zeros(1, 16, 4)
    pred_boxes[0, _CENTRE_ANCHOR] = gt[0, 0]
    scores = torch.full((1, 16, 1), 0.9)
    labels = torch.tensor([[0, 0]])
    mask = torch.tensor([[True, True]])

    out = UniqueAssigner(topk=7)(scores, pred_boxes, _grid_points(), gt, labels, mask)

    assert int(out.fg_mask.sum()) == 1
    assert int((out.gt_index[0] == 0).sum()) == 1
    assert int((out.gt_index[0] == 1).sum()) == 0
    assert int(out.gt_index[0, _CENTRE_ANCHOR]) == 0


def test_zero_gt_batch_is_all_background() -> None:
    """A batch with no ground truths yields no positives."""
    empty_boxes = torch.zeros(1, 0, 4)
    empty_labels = torch.zeros(1, 0, dtype=torch.long)
    empty_mask = torch.zeros(1, 0, dtype=torch.bool)
    scores = torch.full((1, 16, 1), 0.9)
    pred_boxes = torch.zeros(1, 16, 4)

    out = UniqueAssigner(topk=7)(scores, pred_boxes, _grid_points(), empty_boxes, empty_labels, empty_mask)

    assert bool(out.fg_mask.any()) is False
