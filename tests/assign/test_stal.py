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

    Examples:
        >>> points, scores, boxes, gt_boxes, gt_labels, gt_mask = _tiny_gt_scene(torch.zeros(1, 16, 4))
        >>> points.shape, scores.shape, gt_boxes.shape
        (torch.Size([16, 2]), torch.Size([1, 16, 1]), torch.Size([1, 1, 4]))
        >>> int((scores[0, :, 0] == 0.9).sum())  # the four STAL candidate anchors
        4
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


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        pytest.param("s_min", 0.0, id="s-min-zero"),
        pytest.param("s_min", -8.0, id="s-min-negative"),
        pytest.param("s_min", float("nan"), id="s-min-nan"),
        pytest.param("s_min", float("inf"), id="s-min-inf"),
        pytest.param("s_ref", 0.0, id="s-ref-zero"),
        pytest.param("s_ref", -16.0, id="s-ref-negative"),
        pytest.param("s_ref", float("nan"), id="s-ref-nan"),
        pytest.param("s_ref", float("inf"), id="s-ref-inf"),
    ],
)
def test_surrogate_sizes_that_disable_the_inflation_are_rejected(parameter: str, value: float) -> None:
    """``s_min`` and ``s_ref`` are validated at construction, naming the parameter.

    The surrogate is the whole of STAL, and both of its sizes accept values that switch
    it off rather than tune it. A threshold at or below zero — or a ``nan`` one, which no
    dimension ever compares below — inflates nothing, so the class silently becomes the
    plain assigner it subclasses; an infinite threshold inflates *everything*, filtering
    a large ground truth on a footprint that is not its own. A zero, negative, or ``nan``
    replacement leaves the inflated ground truth with no candidates at all, and an
    infinite one makes every anchor in the image a candidate for it. Only ``topk``,
    ``alpha``, ``beta`` and ``eps`` were checked before, so each of these constructed an
    assigner that then mis-assigned quietly for the rest of the run.
    """
    with pytest.raises(ValueError, match=rf"^{parameter} must be finite"):
        SmallTargetAssigner(topk=4, **{parameter: value})


def test_a_replacement_below_the_threshold_is_rejected() -> None:
    """``s_ref < s_min`` raises, because a replacement under the threshold shrinks the box.

    A dimension lying between the two sizes is replaced by a *smaller* one, so the
    surrogate sits strictly inside the original ground truth and the subclass hands the
    base assigner fewer candidates than the untouched box would have had on its own —
    the exact opposite of what STAL exists for, on exactly the targets it exists for,
    and with nothing at the call site to show for it.
    """
    with pytest.raises(ValueError, match=r"^s_ref must be >= s_min, got 4\.0 < 8\.0$"):
        SmallTargetAssigner(topk=4, s_min=8.0, s_ref=4.0)


def test_a_replacement_equal_to_the_threshold_is_accepted() -> None:
    """``s_ref == s_min`` constructs and still inflates; the boundary is usable, not refused.

    The check refuses the values that disable or invert the inflation, not every unusual
    one. At ``s_ref == s_min`` every dimension the surrogate touches is below ``s_min`` by
    definition and so is still strictly widened: the 6x6 ground truth of this module
    becomes the 8x8 box ``[4, 4, 12, 12]``, whose corners are exactly the four stride-8
    anchor centres vanilla TAL leaves it without. Refusing the boundary would reject a
    conservative but coherent recipe — raise every side to at least the smallest stride.
    """
    pred_boxes = _TINY_GT.expand(1, 16, 4).contiguous()  # every pred == the GT box, IoU 1.0
    points, scores, boxes, gt_boxes, gt_labels, gt_mask = _tiny_gt_scene(pred_boxes)

    out = SmallTargetAssigner(topk=4, s_min=8.0, s_ref=8.0)(scores, boxes, points, gt_boxes, gt_labels, gt_mask)

    assert set(out.fg_mask[0].nonzero(as_tuple=True)[0].tolist()) == set(_STAL_CANDIDATES)
