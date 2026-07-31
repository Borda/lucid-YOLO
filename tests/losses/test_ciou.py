# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Complete-IoU loss (WP-024).

Covers a hand-derived closed-form value, the identity and far-apart boundary
cases, the aspect-ratio ordering property, gradient finiteness for normal and
degenerate boxes, and batched/empty shape handling. The math is transcribed
from R10 (arXiv:1911.08287); expected values are derived by hand in the test
comments rather than lifted from any reference implementation.
"""

import math

import torch

from lit_yolo.losses import box_iou_aligned, ciou_loss, complete_iou


def test_against_closed_form() -> None:
    """CIoU matches a fully hand-derived value to 1e-6.

    Boxes (xyxy):
        pred   = [0, 0, 4, 2]  -> w=4, h=2, center (2, 1), area 8
        target = [1, 0, 3, 4]  -> w=2, h=4, center (2, 2), area 8

    Intersection: x in [1, 3] -> 2, y in [0, 2] -> 2, so inter = 4.
    Union = 8 + 8 - 4 = 12, so IoU = 4 / 12 = 1/3.
    Center distance^2 = (2-2)^2 + (1-2)^2 = 1.
    Enclosing box = [0, 0, 4, 4], diagonal^2 = 4^2 + 4^2 = 32.
    v = (4/pi^2) * (atan(2/4) - atan(4/2))^2.
    alpha = v / ((1 - IoU) + v).
    CIoU = IoU - 1/32 - alpha * v.
    """
    pred = torch.tensor([[0.0, 0.0, 4.0, 2.0]])
    target = torch.tensor([[1.0, 0.0, 3.0, 4.0]])

    iou = 1.0 / 3.0
    distance_penalty = 1.0 / 32.0
    v = (4.0 / math.pi**2) * (math.atan(2.0 / 4.0) - math.atan(4.0 / 2.0)) ** 2
    alpha = v / ((1.0 - iou) + v)
    expected = iou - distance_penalty - alpha * v

    got = complete_iou(pred, target)
    assert got.shape == (1,)
    assert math.isclose(got.item(), expected, abs_tol=1e-6)


def test_iou_aligned_matches_hand_value() -> None:
    """box_iou_aligned returns the paired IoU, not a cross matrix."""
    pred = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
    target = torch.tensor([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 2.0]])
    iou = box_iou_aligned(pred, target)
    assert iou.shape == (2,)
    # First pair: inter 1, union 7 -> 1/7. Second pair: identical -> 1.
    assert torch.allclose(iou, torch.tensor([1.0 / 7.0, 1.0]), atol=1e-6)


def test_identical_boxes_score_one_zero_loss() -> None:
    """Identical boxes give CIoU == 1 and loss == 0."""
    boxes = torch.tensor([[3.0, 5.0, 11.0, 9.0], [-2.0, -2.0, 4.0, 1.0]])
    assert torch.allclose(complete_iou(boxes, boxes), torch.ones(2), atol=1e-6)
    assert torch.allclose(ciou_loss(boxes, boxes), torch.zeros(2), atol=1e-6)


def test_disjoint_far_boxes_go_negative() -> None:
    """Far-apart disjoint boxes push CIoU below 0 and loss above 1."""
    pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    target = torch.tensor([[100.0, 100.0, 101.0, 101.0]])
    ciou = complete_iou(pred, target)
    assert ciou.item() < 0.0
    assert ciou_loss(pred, target).item() > 1.0


def test_matching_aspect_ratio_scores_higher() -> None:
    """Same IoU and centers: the matching-aspect pair scores strictly higher.

    Both candidate preds share the target's center and the same IoU with it
    (equal area, fully contained, symmetric about the center); only the aspect
    ratio differs, so CIoU must rank the aspect-matched candidate above the
    aspect-mismatched one via the ``alpha * v`` term.
    """
    target = torch.tensor([[-2.0, -1.0, 2.0, 1.0]])  # w=4, h=2, center (0, 0)
    matched = torch.tensor([[-1.0, -0.5, 1.0, 0.5]])  # w=2, h=1 -> same 2:1 ratio
    mismatched = torch.tensor([[-0.5, -1.0, 0.5, 1.0]])  # w=1, h=2 -> 1:2 ratio

    # Same area and same center => identical IoU for both candidates.
    assert math.isclose(
        box_iou_aligned(matched, target).item(),
        box_iou_aligned(mismatched, target).item(),
        abs_tol=1e-6,
    )
    assert complete_iou(matched, target).item() > complete_iou(mismatched, target).item()


def test_gradients_finite_normal_and_degenerate() -> None:
    """backward() yields finite grads for normal and zero-width pred boxes."""
    pred = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 1.0, 3.0]],  # second row: zero width
        requires_grad=True,
    )
    target = torch.tensor([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 4.0]])
    loss = ciou_loss(pred, target).sum()
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_batched_and_empty_shapes() -> None:
    """(N,4) inputs give (N,) outputs; N=0 works without crashing."""
    pred = torch.rand(5, 4)
    pred = torch.stack([pred[:, 0], pred[:, 1], pred[:, 0] + 1.0, pred[:, 1] + 1.0], dim=1)
    target = pred.clone() + 0.25
    assert complete_iou(pred, target).shape == (5,)
    assert ciou_loss(pred, target).shape == (5,)

    empty = torch.zeros(0, 4)
    assert complete_iou(empty, empty).shape == (0,)
    assert box_iou_aligned(empty, empty).shape == (0,)
    assert ciou_loss(empty, empty).shape == (0,)
