# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the dual-branch loss composition (WP-028).

DualBranchLoss scores a dense one-to-many branch (SmallTargetAssigner,
``topk = 10``) and a unique one-to-one branch (UniqueAssigner, ``topk = 7 -> 1``)
with shared gains, then combines the totals as
``L = alpha * L_o2m + (1 - alpha) * L_o2o``. These tests cover the assignment
wiring (o2o yields one positive per GT while o2m yields several), the static
alpha combination and its settability (the WP-035 schedule seam), a finite
zero-GT batch, and gradient flow into both branches. The head's raw logits are
mapped through a sigmoid for the assigners; scenes live on the same 4x4 stride-8
grid used by the assigner tests.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from lit_yolo.assign import make_anchor_points
from lit_yolo.losses import DualBranchLoss, DualLossOutput


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed every RNG source before each test for deterministic tensors."""
    torch.manual_seed(0)


def _grid_points() -> Tensor:
    """Return the 16 stride-8 anchor centres of a 4x4 grid; x fastest, row-major."""
    points, _ = make_anchor_points([(4, 4)], [8])
    return points


def _boxes_from_leaf(leaf: Tensor) -> Tensor:
    """Build unit-size ``xyxy`` boxes (x2 > x1, y2 > y1) from a differentiable leaf."""
    return torch.stack([leaf[..., 0], leaf[..., 1], leaf[..., 0] + 1.0, leaf[..., 1] + 1.0], dim=-1)


def test_one_positive_per_gt() -> None:
    """o2o assignment yields exactly one positive for a GT the o2m branch spreads over."""
    gt = torch.tensor([[[0.0, 0.0, 32.0, 32.0]]])  # covers all 16 anchors
    pred_boxes = torch.zeros(1, 16, 4)
    pred_boxes[0, 5] = gt[0, 0]  # anchor (12, 12) aligns best
    scores = torch.full((1, 16, 1), 0.9)
    labels = torch.tensor([[0]])
    mask = torch.tensor([[True]])
    points = _grid_points()
    loss = DualBranchLoss()

    o2m = loss.o2m_assigner(scores, pred_boxes, points, gt, labels, mask)
    o2o = loss.o2o_assigner(scores, pred_boxes, points, gt, labels, mask)

    assert int(o2m.fg_mask.sum()) > 1  # dense branch keeps many positives
    assert int(o2o.fg_mask.sum()) == 1  # unique branch keeps exactly one
    assert int((o2o.gt_index[0] == 0).sum()) == 1


def test_every_gt_covered_by_o2o() -> None:
    """Each real GT on a multi-GT scene keeps exactly one o2o positive."""
    gt = torch.tensor([[[0.0, 0.0, 16.0, 16.0], [16.0, 16.0, 32.0, 32.0]]])
    pred_boxes = torch.zeros(1, 16, 4)
    pred_boxes[0, 0] = gt[0, 0]
    pred_boxes[0, 15] = gt[0, 1]
    scores = torch.full((1, 16, 1), 0.9)
    labels = torch.tensor([[0, 0]])
    mask = torch.tensor([[True, True]])
    loss = DualBranchLoss()

    o2o = loss.o2o_assigner(scores, pred_boxes, _grid_points(), gt, labels, mask)

    assert int(o2o.fg_mask.sum()) == 2
    assert int((o2o.gt_index[0] == 0).sum()) == 1
    assert int((o2o.gt_index[0] == 1).sum()) == 1


def test_alpha_weights_the_two_branch_totals() -> None:
    """total equals alpha*o2m + (1-alpha)*o2o exactly, and tracks a changed alpha."""
    gt = torch.tensor([[[0.0, 0.0, 32.0, 32.0]]])  # covers all 16 anchors
    logits = torch.zeros(1, 16, 1)
    boxes = gt.expand(1, 16, 4).contiguous()  # every pred == the GT box
    labels = torch.tensor([[0]])
    mask = torch.tensor([[True]])
    points = _grid_points()
    loss = DualBranchLoss()

    default = loss(logits, boxes, logits, boxes, points, gt, labels, mask)
    loss.alpha = 0.25
    changed = loss(logits, boxes, logits, boxes, points, gt, labels, mask)

    assert default.alpha == 0.5
    assert math.isclose(default.total.item(), 0.5 * default.o2m.total.item() + 0.5 * default.o2o.total.item())
    assert changed.alpha == 0.25
    assert math.isclose(changed.total.item(), 0.25 * changed.o2m.total.item() + 0.75 * changed.o2o.total.item())
    assert not math.isclose(default.total.item(), changed.total.item())


def test_zero_gt_batch_is_finite() -> None:
    """A batch with no ground truths produces a finite combined loss."""
    logits = torch.zeros(1, 16, 3)
    boxes = torch.zeros(1, 16, 4)
    empty_boxes = torch.zeros(1, 0, 4)
    empty_labels = torch.zeros(1, 0, dtype=torch.long)
    empty_mask = torch.zeros(1, 0, dtype=torch.bool)
    loss = DualBranchLoss()

    out = loss(logits, boxes, logits, boxes, _grid_points(), empty_boxes, empty_labels, empty_mask)

    assert isinstance(out, DualLossOutput)
    assert torch.isfinite(out.total).all()
    assert out.o2m.box.item() == 0.0 and out.o2o.box.item() == 0.0


def test_gradients_flow_through_both_branches() -> None:
    """backward() populates finite gradients on both branches' logits and boxes."""
    gt = torch.tensor([[[2.0, 2.0, 30.0, 30.0]]])
    labels = torch.tensor([[0]])
    mask = torch.tensor([[True]])
    points = _grid_points()
    o2m_logits = torch.randn(1, 16, 1, requires_grad=True)
    o2o_logits = torch.randn(1, 16, 1, requires_grad=True)
    o2m_leaf = torch.rand(1, 16, 4, requires_grad=True)
    o2o_leaf = torch.rand(1, 16, 4, requires_grad=True)
    o2m_boxes = _boxes_from_leaf(o2m_leaf)
    o2o_boxes = _boxes_from_leaf(o2o_leaf)

    out = DualBranchLoss()(o2m_logits, o2m_boxes, o2o_logits, o2o_boxes, points, gt, labels, mask)
    out.total.backward()

    for grad in (o2m_logits.grad, o2o_logits.grad, o2m_leaf.grad, o2o_leaf.grad):
        assert grad is not None and torch.isfinite(grad).all()
    assert not torch.isnan(out.total)
