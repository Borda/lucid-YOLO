# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the per-branch detection loss (WP-027).

Covers the three terms against hand-computed values on a crafted
:class:`AssignResult`, the zero-positive batch (box and L1 vanish, classification
still trains, gradients finite), the perfect-prediction floor, linear gain
scaling, the batched scalar-output contract, and gradient flow to both
predictions. Expected values are derived by hand in the test comments; the math
is transcribed from R1 sec. 3.3.2 and R4 (arXiv:2108.07755).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from lucid_yolo.assign.tal import AssignResult
from lucid_yolo.losses import DetectionBranchLoss, DetectionLossOutput

_LN2 = math.log(2.0)


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed every RNG source before each test for deterministic tensors."""
    torch.manual_seed(0)


def _make_assign(
    fg_mask: Tensor,
    target_labels: Tensor,
    target_boxes: Tensor,
    align_weights: Tensor,
) -> AssignResult:
    """Assemble an AssignResult, deriving gt_index from the foreground mask."""
    gt_index = torch.where(fg_mask, torch.zeros_like(target_labels), torch.full_like(target_labels, -1))
    return AssignResult(
        fg_mask=fg_mask,
        gt_index=gt_index,
        target_labels=target_labels,
        target_boxes=target_boxes,
        align_weights=align_weights,
    )


def test_components() -> None:
    """Each term matches a fully hand-derived value on a 3-anchor scenario.

    One image, three anchors, two classes, one ground truth (class 0). Anchors 0
    and 1 are positives for it, anchor 2 is background:

        align_weights = [0.8, 0.4, 0.0]  -> weight_sum = 1.2
        target box (anchors 0, 1) = [0, 0, 2, 2]

    Predictions: anchor 0 box is perfect; anchor 1 box = [0, 0, 1, 1]; every
    class logit is zero (sigmoid 0.5).

    L_box: anchor 0 CIoU loss = 0. Anchor 1: IoU = 1/4, center-dist^2 = 0.5,
    enclosing diag^2 = 8 -> penalty 1/16, aspect term 0 (both square), so
    CIoU = 0.25 - 0.0625 = 0.1875 and its loss = 0.8125. Weighted:
    (0.8*0 + 0.4*0.8125) / 1.2 = 0.325 / 1.2.

    L_l1: anchor 0 = 0; anchor 1 = |1-2| + |1-2| = 2. Weighted:
    (0.8*0 + 0.4*2) / 1.2 = 0.8 / 1.2.

    L_cls: every logit is 0, so each of the 6 BCE elements = ln 2 regardless of
    its soft target. Sum = 6 ln 2, normalized: 6 ln 2 / 1.2.

    total = 7.5*L_box + 0.5*L_cls + 6.0*L_l1.
    """
    logits = torch.zeros(1, 3, 2)
    boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
    target_boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]]])
    assign = _make_assign(
        fg_mask=torch.tensor([[True, True, False]]),
        target_labels=torch.tensor([[0, 0, -1]]),
        target_boxes=target_boxes,
        align_weights=torch.tensor([[0.8, 0.4, 0.0]]),
    )

    out = DetectionBranchLoss()(logits, boxes, assign)

    expected_box = 0.325 / 1.2
    expected_l1 = 0.8 / 1.2
    expected_cls = 6.0 * _LN2 / 1.2
    expected_total = 7.5 * expected_box + 0.5 * expected_cls + 6.0 * expected_l1
    assert math.isclose(out.box.item(), expected_box, abs_tol=1e-4)
    assert math.isclose(out.l1.item(), expected_l1, abs_tol=1e-4)
    assert math.isclose(out.cls.item(), expected_cls, abs_tol=1e-6)
    assert math.isclose(out.total.item(), expected_total, abs_tol=1e-4)


def test_zero_positive_batch_trains_cls_only() -> None:
    """No positives: box and L1 are exactly zero, cls finite, grads finite."""
    logits = torch.zeros(1, 4, 3, requires_grad=True)
    boxes = torch.zeros(1, 4, 4, requires_grad=True)
    assign = _make_assign(
        fg_mask=torch.zeros(1, 4, dtype=torch.bool),
        target_labels=torch.full((1, 4), -1),
        target_boxes=torch.zeros(1, 4, 4),
        align_weights=torch.zeros(1, 4),
    )

    out = DetectionBranchLoss()(logits, boxes, assign)
    out.total.backward()

    assert out.box.item() == 0.0
    assert out.l1.item() == 0.0
    assert torch.isfinite(out.cls).all() and out.cls.item() > 0.0
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert boxes.grad is not None and torch.isfinite(boxes.grad).all()


def test_perfect_predictions_drive_total_to_floor() -> None:
    """Exact boxes and confident logits push the total near zero."""
    logits = torch.tensor([[[12.0, -12.0], [-12.0, -12.0]]])  # anchor 0 sure class 0; anchor 1 background
    boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]]])
    assign = _make_assign(
        fg_mask=torch.tensor([[True, False]]),
        target_labels=torch.tensor([[0, -1]]),
        target_boxes=torch.tensor([[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]]]),
        align_weights=torch.tensor([[1.0, 0.0]]),
    )

    out = DetectionBranchLoss()(logits, boxes, assign)

    assert out.box.item() < 1e-5
    assert out.l1.item() < 1e-5
    assert out.total.item() < 1e-2


def test_gains_scale_total_linearly() -> None:
    """Doubling every gain doubles total; pre-gain components are unchanged."""
    logits = torch.zeros(1, 2, 2)
    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
    assign = _make_assign(
        fg_mask=torch.tensor([[True, False]]),
        target_labels=torch.tensor([[0, -1]]),
        target_boxes=torch.tensor([[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]]]),
        align_weights=torch.tensor([[0.5, 0.0]]),
    )

    base = DetectionBranchLoss()(logits, boxes, assign)
    doubled = DetectionBranchLoss(box_gain=15.0, cls_gain=1.0, l1_gain=12.0)(logits, boxes, assign)

    assert math.isclose(doubled.total.item(), 2.0 * base.total.item(), rel_tol=1e-6)
    assert math.isclose(doubled.box.item(), base.box.item(), rel_tol=1e-6)
    assert math.isclose(doubled.cls.item(), base.cls.item(), rel_tol=1e-6)
    assert math.isclose(doubled.l1.item(), base.l1.item(), rel_tol=1e-6)


def test_batched_output_is_scalar_and_finite() -> None:
    """Batched (B>1) inputs still yield scalar, finite loss terms."""
    logits = torch.randn(2, 5, 3)
    boxes = torch.randn(2, 5, 4).abs()
    boxes = torch.stack([boxes[..., 0], boxes[..., 1], boxes[..., 0] + 1.0, boxes[..., 1] + 1.0], dim=-1)
    fg_mask = torch.tensor([[True, False, True, False, False], [False, True, False, False, True]])
    assign = _make_assign(
        fg_mask=fg_mask,
        target_labels=torch.where(fg_mask, torch.zeros(2, 5, dtype=torch.long), torch.full((2, 5), -1)),
        target_boxes=boxes.detach().clone(),
        align_weights=torch.where(fg_mask, torch.full((2, 5), 0.7), torch.zeros(2, 5)),
    )

    out = DetectionBranchLoss()(logits, boxes, assign)

    assert isinstance(out, DetectionLossOutput)
    for term in (out.total, out.box, out.cls, out.l1):
        assert term.shape == ()
        assert torch.isfinite(term).all()


def test_gradients_flow_to_both_predictions() -> None:
    """backward() populates finite gradients on logits and boxes with positives."""
    logits = torch.randn(1, 3, 2, requires_grad=True)
    boxes_leaf = torch.rand(1, 3, 4, requires_grad=True)
    boxes = torch.stack(
        [boxes_leaf[..., 0], boxes_leaf[..., 1], boxes_leaf[..., 0] + 1.0, boxes_leaf[..., 1] + 1.0], dim=-1
    )
    assign = _make_assign(
        fg_mask=torch.tensor([[True, False, True]]),
        target_labels=torch.tensor([[1, -1, 0]]),
        target_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 1.5, 1.5]]]),
        align_weights=torch.tensor([[0.6, 0.0, 0.9]]),
    )

    out = DetectionBranchLoss()(logits, boxes, assign)
    out.total.backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert boxes_leaf.grad is not None and torch.isfinite(boxes_leaf.grad).all()
    assert not torch.isnan(out.total)
