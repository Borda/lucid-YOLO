# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Task-Aligned Assigner (WP-025).

The centrepiece ``test_alignment_and_topk`` builds a 4x4 stride-4 grid (16
anchors) with two disjoint ground-truth boxes and hand-crafted predicted scores
and boxes, so every candidate anchor, its alignment metric ``t = s * u**6``, the
top-k winners, and the per-ground-truth-normalized weights can be enumerated by
hand. Supporting tests isolate the centre-inside filter, zero-ground-truth
images, padded ground truths, graceful top-k degradation, and conflict
resolution. All expected values are derived from the R4 formulation, never
lifted from a reference implementation.
"""

import math

import pytest
import torch

from lucid_yolo.assign import AssignResult, TaskAlignedAssigner, make_anchor_points


def _build_dual_gt_scene() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two disjoint GTs on a 4x4 stride-4 grid with crafted preds.

    Anchor centres are x, y in {2, 6, 10, 14}; row-major index ``i*4 + j``.
    GT0 = [0, 0, 8, 8] (class 0): candidates {0, 1, 4, 5} (centres <= 8).
    GT1 = [8, 8, 16, 16] (class 1): candidates {10, 11, 14, 15} (centres >= 8).

    Examples:
        >>> points, scores, boxes, gt_boxes, gt_labels, gt_mask = _build_dual_gt_scene()
        >>> points.shape, scores.shape, boxes.shape
        (torch.Size([16, 2]), torch.Size([1, 16, 2]), torch.Size([1, 16, 4]))
        >>> gt_boxes.shape, gt_labels.tolist(), gt_mask.tolist()
        (torch.Size([1, 2, 4]), [[0, 1]], [[True, True]])
    """
    points, _ = make_anchor_points([(4, 4)], [4])  # (16, 2)
    scores = torch.zeros(1, 16, 2)
    boxes = torch.zeros(1, 16, 4)

    # GT0 candidates: (score class 0, predicted box) engineered for known IoU.
    scores[0, 0, 0], boxes[0, 0] = 0.9, torch.tensor([0.0, 0.0, 8.0, 8.0])  # IoU 1.0
    scores[0, 1, 0], boxes[0, 1] = 0.8, torch.tensor([0.0, 0.0, 4.0, 4.0])  # IoU 0.25
    scores[0, 4, 0], boxes[0, 4] = 0.6, torch.tensor([0.0, 0.0, 8.0, 4.0])  # IoU 0.5
    scores[0, 5, 0], boxes[0, 5] = 0.5, torch.tensor([4.0, 4.0, 8.0, 8.0])  # IoU 0.25

    # GT1 candidates.
    scores[0, 10, 1], boxes[0, 10] = 0.7, torch.tensor([8.0, 8.0, 16.0, 16.0])  # IoU 1.0
    scores[0, 11, 1], boxes[0, 11] = 0.6, torch.tensor([8.0, 8.0, 12.0, 12.0])  # IoU 0.25
    scores[0, 14, 1], boxes[0, 14] = 0.55, torch.tensor([8.0, 8.0, 16.0, 12.0])  # IoU 0.5
    scores[0, 15, 1], boxes[0, 15] = 0.4, torch.tensor([12.0, 12.0, 16.0, 16.0])  # IoU 0.25

    gt_boxes = torch.tensor([[[0.0, 0.0, 8.0, 8.0], [8.0, 8.0, 16.0, 16.0]]])
    gt_labels = torch.tensor([[0, 1]])
    gt_mask = torch.tensor([[True, True]])
    return points, scores, boxes, gt_boxes, gt_labels, gt_mask


def test_alignment_and_topk() -> None:
    """Top-3 selection, targets, and normalized weights match the hand case.

    Per GT the metric is ``t = score * IoU**6``. GT0 candidate t values are
    idx0=0.9, idx4=0.6/64, idx1=0.8/4096, idx5=0.5/4096; top-3 keep {0, 4, 1}
    and drop idx5. GT1 keeps {10, 14, 11} and drops idx15. Normalization scales
    each GT's positives by ``u_max / t_max`` (u_max == 1.0 here), so the
    best-aligned anchor of each GT reaches weight 1.0.
    """
    points, scores, boxes, gt_boxes, gt_labels, gt_mask = _build_dual_gt_scene()
    assigner = TaskAlignedAssigner(topk=3)

    out = assigner(scores, boxes, points, gt_boxes, gt_labels, gt_mask)

    positive = torch.tensor([0, 1, 4, 10, 11, 14])
    fg_expected = torch.zeros(16, dtype=torch.bool)
    fg_expected[positive] = True
    assert torch.equal(out.fg_mask[0], fg_expected)

    gt_expected = torch.full((16,), -1, dtype=torch.long)
    gt_expected[torch.tensor([0, 1, 4])] = 0
    gt_expected[torch.tensor([10, 11, 14])] = 1
    assert torch.equal(out.gt_index[0], gt_expected)
    assert torch.equal(out.target_labels[0], gt_expected)  # labels equal gt ids here
    assert torch.equal(out.target_boxes[0, 0], torch.tensor([0.0, 0.0, 8.0, 8.0]))
    assert torch.equal(out.target_boxes[0, 10], torch.tensor([8.0, 8.0, 16.0, 16.0]))

    t_gt0 = torch.tensor([0.9 * 1.0**6, 0.8 * 0.25**6, 0.6 * 0.5**6])  # idx 0, 1, 4
    t_gt1 = torch.tensor([0.7 * 1.0**6, 0.6 * 0.25**6, 0.55 * 0.5**6])  # idx 10, 11, 14
    weights_expected = torch.zeros(16)
    weights_expected[torch.tensor([0, 1, 4])] = t_gt0 * (1.0 / t_gt0.max())
    weights_expected[torch.tensor([10, 11, 14])] = t_gt1 * (1.0 / t_gt1.max())
    assert torch.allclose(out.align_weights[0], weights_expected, atol=1e-6)


def test_returns_assign_result_with_documented_shapes() -> None:
    """The result is an AssignResult with the documented per-field shapes."""
    points, scores, boxes, gt_boxes, gt_labels, gt_mask = _build_dual_gt_scene()

    out = TaskAlignedAssigner(topk=3)(scores, boxes, points, gt_boxes, gt_labels, gt_mask)

    assert isinstance(out, AssignResult)
    assert out.fg_mask.shape == (1, 16)
    assert out.gt_index.shape == (1, 16)
    assert out.target_labels.shape == (1, 16)
    assert out.target_boxes.shape == (1, 16, 4)
    assert out.align_weights.shape == (1, 16)


def test_center_inside_filter_excludes_outside_anchor() -> None:
    """An anchor whose centre lies outside the GT box is never a candidate."""
    points = torch.tensor([[2.0, 2.0], [6.0, 2.0]])  # first inside, second outside
    scores = torch.full((1, 2, 1), 0.9)
    boxes = torch.tensor([[[0.0, 0.0, 3.0, 3.0], [0.0, 0.0, 3.0, 3.0]]])  # both would score high
    gt_boxes = torch.tensor([[[0.0, 0.0, 3.0, 3.0]]])
    gt_labels = torch.tensor([[0]])
    gt_mask = torch.tensor([[True]])

    out = TaskAlignedAssigner(topk=5)(scores, boxes, points, gt_boxes, gt_labels, gt_mask)

    assert torch.equal(out.fg_mask[0], torch.tensor([True, False]))
    assert torch.equal(out.gt_index[0], torch.tensor([0, -1]))


def test_zero_gt_image_is_all_background() -> None:
    """An image with no ground truths yields empty fg_mask and finite weights."""
    points = torch.zeros(4, 2)
    scores = torch.zeros(1, 4, 1)
    boxes = torch.zeros(1, 4, 4)
    empty_boxes = torch.zeros(1, 0, 4)
    empty_labels = torch.zeros(1, 0, dtype=torch.long)
    empty_mask = torch.zeros(1, 0, dtype=torch.bool)

    out = TaskAlignedAssigner(topk=3)(scores, boxes, points, empty_boxes, empty_labels, empty_mask)

    assert not out.fg_mask.any()
    assert (out.gt_index == -1).all()
    assert torch.isfinite(out.align_weights).all()


def test_all_masked_gts_give_no_nan_weights() -> None:
    """A present-but-masked GT (zero positives) normalizes without NaN."""
    points, scores, boxes, gt_boxes, gt_labels, _ = _build_dual_gt_scene()
    all_masked = torch.tensor([[False, False]])

    out = TaskAlignedAssigner(topk=3)(scores, boxes, points, gt_boxes, gt_labels, all_masked)

    assert not out.fg_mask.any()
    assert torch.isfinite(out.align_weights).all()
    assert (out.align_weights == 0).all()


def test_padded_gt_is_never_assigned() -> None:
    """A padded GT (gt_mask False) attracts no anchors even with a real box."""
    points, _ = make_anchor_points([(4, 4)], [4])
    scores = torch.zeros(1, 16, 2)
    boxes = torch.zeros(1, 16, 4)
    # Both GTs share a box that contains anchors {0,1,4,5}; only GT0 is real.
    scores[0, 0, 0], boxes[0, 0] = 0.9, torch.tensor([0.0, 0.0, 8.0, 8.0])
    scores[0, 1, 1], boxes[0, 1] = 0.9, torch.tensor([0.0, 0.0, 8.0, 8.0])  # would feed GT1
    gt_boxes = torch.tensor([[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 8.0]]])
    gt_labels = torch.tensor([[0, 1]])
    gt_mask = torch.tensor([[True, False]])

    out = TaskAlignedAssigner(topk=3)(scores, boxes, points, gt_boxes, gt_labels, gt_mask)

    assert (out.gt_index != 1).all()
    assert out.gt_index[0, 0] == 0


def test_topk_larger_than_candidate_count_degrades() -> None:
    """A top-k exceeding the eligible count keeps exactly the eligible anchors."""
    points, _ = make_anchor_points([(4, 4)], [4])  # centres x,y in {2,6,10,14}
    scores = torch.zeros(1, 16, 1)
    boxes = torch.zeros(1, 16, 4)
    # GT covers y<=3 so only row-0 anchors (2,2) and (6,2): indices 0 and 1.
    scores[0, 0, 0], boxes[0, 0] = 0.9, torch.tensor([0.0, 0.0, 7.0, 3.0])
    scores[0, 1, 0], boxes[0, 1] = 0.8, torch.tensor([0.0, 0.0, 7.0, 3.0])
    gt_boxes = torch.tensor([[[0.0, 0.0, 7.0, 3.0]]])
    gt_labels = torch.tensor([[0]])
    gt_mask = torch.tensor([[True]])

    out = TaskAlignedAssigner(topk=100)(scores, boxes, points, gt_boxes, gt_labels, gt_mask)

    expected = torch.zeros(16, dtype=torch.bool)
    expected[torch.tensor([0, 1])] = True
    assert torch.equal(out.fg_mask[0], expected)


def test_conflict_resolution_assigns_highest_t() -> None:
    """An anchor claimed by two GTs goes to the one giving it the higher t."""
    points = torch.tensor([[5.0, 5.0]])  # single anchor inside both GTs
    scores = torch.tensor([[[0.9, 0.3]]])  # class 0 outranks class 1
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])  # IoU 1.0 with both GTs
    gt_boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]])
    gt_labels = torch.tensor([[0, 1]])
    gt_mask = torch.tensor([[True, True]])

    out = TaskAlignedAssigner(topk=1)(scores, boxes, points, gt_boxes, gt_labels, gt_mask)

    assert out.fg_mask[0, 0]
    assert out.gt_index[0, 0] == 0
    assert out.target_labels[0, 0] == 0


@pytest.mark.parametrize(
    "topk",
    [
        pytest.param(1, id="k-far-below-c"),
        pytest.param(3, id="k-one-below-c"),
        pytest.param(4, id="k-equals-c"),
        pytest.param(6, id="k-above-c-below-anchor-count"),
    ],
)
def test_all_zero_metric_still_keeps_min_k_c_of_the_gts_own_candidates(topk: int) -> None:
    """A ground truth with an all-zero alignment metric keeps ``min(k, c)`` of its candidates.

    This is the state the head is in at initialization: ``decode_ltrb`` applies no
    non-negativity, so raw distances decode to inverted boxes whose IoU with every
    ground truth is exactly zero and whose metric ``t = s * u**6`` is therefore
    uniformly zero. The scene puts the only real ground truth in the bottom-right
    quadrant of a 4x4 stride-4 grid, so its four candidates are anchors
    ``{10, 11, 14, 15}`` — deliberately not the low indices a tie-broken ``topk``
    reaches for first. Ranking the ties globally instead of within the ground
    truth's own candidate set makes the intersection with that set return zero
    positives for a ground truth that had four.
    """
    points, _ = make_anchor_points([(4, 4)], [4])  # centres x, y in {2, 6, 10, 14}
    scores = torch.full((1, 16, 1), 0.9)
    boxes = torch.zeros(1, 16, 4)  # degenerate predictions: IoU 0 everywhere, so t == 0
    gt_boxes = torch.tensor([[[8.0, 8.0, 16.0, 16.0]]])
    gt_labels = torch.tensor([[0]])
    gt_mask = torch.tensor([[True]])

    out = TaskAlignedAssigner(topk=topk)(scores, boxes, points, gt_boxes, gt_labels, gt_mask)

    positives = out.fg_mask[0].nonzero(as_tuple=True)[0]
    assert positives.numel() == min(topk, 4)
    assert set(positives.tolist()) <= {10, 11, 14, 15}
    assert (out.gt_index[0][positives] == 0).all()


@pytest.mark.parametrize(
    "u_max",
    [
        pytest.param(1.0, id="u-max-1.0"),
        pytest.param(0.5, id="u-max-0.5"),
        pytest.param(0.1, id="u-max-0.1"),
        pytest.param(0.01, id="u-max-0.01"),
    ],
)
def test_target_normalization_reaches_u_max_at_every_overlap_scale(u_max: float) -> None:
    """The best positive's weight is ``u_max`` and the runner-up's falls off by its ``t`` ratio.

    Two anchors sit inside a 10x10 ground truth; each predicts a box nested in it,
    the first with IoU ``u_max`` and the second with IoU ``u_max / 2``, so
    ``t_max = s * u_max**6`` and the second anchor's ``t`` is ``t_max / 64``. R4's
    target normalization scales a ground truth's positives by ``u_max / t_max``,
    which puts the best anchor at exactly ``u_max`` at every overlap scale. An
    *absolute* floor added to that denominator breaks the invariant from below:
    ``t_max`` falls as the sixth power of the IoU, so at IoU 0.01 it is three orders
    of magnitude under a 1e-9 floor and the positive trains as background while
    ``fg_mask`` still reports it foreground.
    """
    side_best = 10.0 * math.sqrt(u_max)
    side_worse = 10.0 * math.sqrt(u_max / 2.0)
    points = torch.tensor([[3.0, 3.0], [7.0, 7.0]])  # both centres inside the ground truth
    scores = torch.full((1, 2, 1), 0.5)
    boxes = torch.tensor([[[0.0, 0.0, side_best, side_best], [0.0, 0.0, side_worse, side_worse]]])
    gt_boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])  # area 100, so IoU == nested area / 100
    gt_labels = torch.tensor([[0]])
    gt_mask = torch.tensor([[True]])

    out = TaskAlignedAssigner(topk=2)(scores, boxes, points, gt_boxes, gt_labels, gt_mask)

    assert torch.equal(out.fg_mask[0], torch.tensor([True, True]))
    expected = torch.tensor([u_max, u_max * 0.5**6])
    assert torch.allclose(out.align_weights[0], expected, atol=1e-6)
