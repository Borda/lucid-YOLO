# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Small-Target-Aware Label Assignment (STAL, WP-026).

STAL modifies only the candidate-filtering step of the Task-Aligned Assigner:
a per-ground-truth surrogate box (centre preserved, each dimension below
``s_min`` inflated to ``s_ref``) drives the centre-inside test, while scoring,
targets, and regression keep the original box. The three DoD tests here cover
that contract directly: a sub-8x8 ground truth that vanilla TAL leaves with
zero candidates gains candidates under STAL (the Phase-3 gate scenario); the
surrogate helper clamps each dimension independently without moving centres;
and STAL's returned targets and weights derive from the original box, never the
surrogate. All expected values are computed by hand from the technical
specification section 3.3.3 (Eq. 4-6) and section 4, never lifted from a
reference implementation.
"""

import pytest
import torch

from lucid_yolo.assign import (
    SmallTargetAssigner,
    TaskAlignedAssigner,
    make_anchor_points,
    surrogate_boxes,
)

#: Original 6x6 tiny ground truth: centre (8, 8), no stride-8 anchor centre inside.
_TINY_GT = torch.tensor([[[5.0, 5.0, 11.0, 11.0]]])
#: STAL surrogate of ``_TINY_GT``: both dims inflated to 16, centred at (8, 8).
_TINY_SURROGATE = torch.tensor([0.0, 0.0, 16.0, 16.0])
#: The four stride-8 anchors whose centres fall inside the 16x16 surrogate.
_STAL_CANDIDATES = (0, 1, 4, 5)


def _tiny_gt_scene(
    pred_boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble a single-image scene around the 6x6 tiny GT on a 4x4 stride-8 grid.

    Anchor centres are x, y in {4, 12, 20, 28}; none lies inside the 6x6 GT box
    ``[5, 5, 11, 11]``. Class-0 scores are 0.9 at every candidate anchor; the
    caller supplies the ``(1, 16, 4)`` predicted boxes so each test controls the
    IoU against the original ground truth.
    """
    points, _ = make_anchor_points([(4, 4)], [8])  # (16, 2)
    scores = torch.zeros(1, 16, 1)
    for anchor in _STAL_CANDIDATES:
        scores[0, anchor, 0] = 0.9
    gt_labels = torch.tensor([[0]])
    gt_mask = torch.tensor([[True]])
    return points, scores, pred_boxes, _TINY_GT, gt_labels, gt_mask


def test_tiny_box_gains_candidates() -> None:
    """A sub-8x8 GT yields 0 positives under vanilla TAL and >= 1 under STAL."""
    pred_boxes = _TINY_GT.expand(1, 16, 4).contiguous()  # every pred == the GT box, IoU 1.0
    points, scores, boxes, gt_boxes, gt_labels, gt_mask = _tiny_gt_scene(pred_boxes)

    vanilla = TaskAlignedAssigner(topk=4)(scores, boxes, points, gt_boxes, gt_labels, gt_mask)
    stal = SmallTargetAssigner(topk=4)(scores, boxes, points, gt_boxes, gt_labels, gt_mask)

    assert int(vanilla.fg_mask.sum()) == 0
    assert int(stal.fg_mask.sum()) >= 1


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        pytest.param(
            [7.0, 0.0, 13.0, 20.0],  # 6 wide, 20 tall, centre (10, 10)
            [2.0, 0.0, 18.0, 20.0],  # width -> 16, height kept
            id="w6-h20-width-clamped",
        ),
        pytest.param(
            [0.0, 7.0, 20.0, 13.0],  # 20 wide, 6 tall, centre (10, 10)
            [0.0, 2.0, 20.0, 18.0],  # width kept, height -> 16
            id="w20-h6-height-clamped",
        ),
        pytest.param(
            [7.0, 7.0, 13.0, 13.0],  # 6x6, centre (10, 10)
            [2.0, 2.0, 18.0, 18.0],  # both dims -> 16
            id="w6-h6-both-clamped",
        ),
        pytest.param(
            [0.0, 0.0, 20.0, 20.0],  # 20x20, centre (10, 10)
            [0.0, 0.0, 20.0, 20.0],  # untouched
            id="w20-h20-untouched",
        ),
    ],
)
def test_per_dim_clamp(box: list[float], expected: list[float]) -> None:
    """surrogate_boxes clamps each dimension independently and never moves centres."""
    gt = torch.tensor([[box]])  # (1, 1, 4)
    original_centre = torch.tensor([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])

    surrogate = surrogate_boxes(gt, s_min=8.0, s_ref=16.0)[0, 0]

    assert torch.equal(surrogate, torch.tensor(expected))
    surrogate_centre = torch.tensor([(surrogate[0] + surrogate[2]) * 0.5, (surrogate[1] + surrogate[3]) * 0.5])
    assert torch.equal(surrogate_centre, original_centre)


def test_targets_unchanged() -> None:
    """STAL positives carry the ORIGINAL GT box; weights derive from original-box IoU."""
    pred_boxes = torch.zeros(1, 16, 4)
    pred_boxes[0, 0] = _TINY_GT[0, 0]  # IoU 1.0 vs original
    for anchor in (1, 4, 5):
        pred_boxes[0, anchor] = torch.tensor([5.0, 5.0, 8.0, 8.0])  # IoU 0.25 vs original
    points, scores, boxes, gt_boxes, gt_labels, gt_mask = _tiny_gt_scene(pred_boxes)

    out = SmallTargetAssigner(topk=4)(scores, boxes, points, gt_boxes, gt_labels, gt_mask)

    positive = torch.tensor(_STAL_CANDIDATES)
    fg_expected = torch.zeros(16, dtype=torch.bool)
    fg_expected[positive] = True
    assert torch.equal(out.fg_mask[0], fg_expected)
    for anchor in _STAL_CANDIDATES:  # every positive's target is the original 6x6 box, not the surrogate
        assert torch.equal(out.target_boxes[0, anchor], _TINY_GT[0, 0])
        assert not torch.equal(out.target_boxes[0, anchor], _TINY_SURROGATE)

    # t = score * IoU_original**6; normalized by u_max / t_max over the GT's positives.
    weights_expected = torch.zeros(16)
    weights_expected[0] = 1.0  # best-aligned anchor reaches u_max = 1.0
    weights_expected[torch.tensor([1, 4, 5])] = 0.25**6  # = 0.9 * 0.25**6 * (1.0 / 0.9)
    assert torch.allclose(out.align_weights[0], weights_expected, atol=1e-9)
