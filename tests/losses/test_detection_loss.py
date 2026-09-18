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
from lucid_yolo.losses.ciou import ciou_loss

_LN2 = math.log(2.0)

#: Per-anchor strides of the random-assignment scenarios, cycled over the anchor axis.
_LEVEL_STRIDES = (8.0, 16.0, 32.0)


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
    """Assemble an AssignResult, deriving gt_index from the foreground mask.

    Examples:
        >>> assign = _make_assign(
        ...     fg_mask=torch.tensor([[True, False]]),
        ...     target_labels=torch.tensor([[0, -1]]),
        ...     target_boxes=torch.tensor([[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]]]),
        ...     align_weights=torch.tensor([[1.0, 0.0]]),
        ... )
        >>> assign.gt_index
        tensor([[ 0, -1]])
    """
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


def test_l1_stride_normalization_divides_by_anchor_stride() -> None:
    """With strides given, the L1 term is measured in stride units (A13 revision, WP-078).

    Same scenario as test_components (only anchor 1 contributes L1 = 2 pixels,
    weighted 0.8/1.2); per-anchor strides [8, 16, 32] put anchor 1 at stride 16,
    so the stride-unit term is exactly 1/16 of the pixel-frame value. CIoU and
    classification are untouched.
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

    pixel = DetectionBranchLoss()(logits, boxes, assign)
    strided = DetectionBranchLoss()(logits, boxes, assign, strides=torch.tensor([8.0, 16.0, 32.0]))

    assert math.isclose(strided.l1.item(), pixel.l1.item() / 16.0, abs_tol=1e-6)
    assert math.isclose(strided.box.item(), pixel.box.item(), abs_tol=1e-6)
    assert math.isclose(strided.cls.item(), pixel.cls.item(), abs_tol=1e-6)


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


def _masked_box_losses_oracle(
    pred_boxes: Tensor, assign: AssignResult, weight_sum: Tensor, strides: Tensor | None
) -> tuple[Tensor, Tensor]:
    """The pre-WP-183 ``_box_losses``, verbatim: boolean gathers over the positives.

    Kept as the reference the dense form is held against. It is the implementation
    the golden snapshots were captured with, so agreement with it is agreement with
    every frozen value.

    Examples:
        >>> assign = _make_assign(
        ...     fg_mask=torch.tensor([[True, False]]),
        ...     target_labels=torch.tensor([[0, -1]]),
        ...     target_boxes=torch.tensor([[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]]]),
        ...     align_weights=torch.tensor([[1.0, 0.0]]),
        ... )
        >>> box, l1 = _masked_box_losses_oracle(assign.target_boxes, assign, torch.tensor(1.0), None)
        >>> float(box), float(l1)
        (0.0, 0.0)
    """
    fg_mask = assign.fg_mask
    pred_pos = pred_boxes[fg_mask]  # (P, 4)
    target_pos = assign.target_boxes[fg_mask]  # (P, 4)
    weights = assign.align_weights[fg_mask]  # (P,)
    box_terms = ciou_loss(pred_pos, target_pos)  # (P,)
    diffs = pred_pos - target_pos  # (P, 4)
    if strides is not None:
        diffs = diffs / strides.expand(fg_mask.shape)[fg_mask].unsqueeze(-1)
    l1_terms = diffs.abs().sum(dim=-1)  # (P,)
    l_box = (box_terms * weights).sum() / weight_sum
    l_l1 = (l1_terms * weights).sum() / weight_sum
    return l_box, l_l1


def _random_boxes(batch: int, anchors: int) -> Tensor:
    """Random well-formed ``xyxy`` boxes in a pixel frame, ``(batch, anchors, 4)``.

    Examples:
        >>> boxes = _random_boxes(2, 3)
        >>> bool((boxes[..., 2:] > boxes[..., :2]).all())
        True
    """
    corner = torch.rand(batch, anchors, 2) * 100.0
    size = torch.rand(batch, anchors, 2) * 40.0 + 1.0
    return torch.cat([corner, corner + size], dim=-1)


def _random_assign(fg_mask: Tensor) -> AssignResult:
    """A contract-conforming assignment over ``fg_mask``: random targets and weights on positives.

    Background anchors carry the documented sentinels — ``-1`` label and index, a zero
    box, zero weight — which is what makes the dense form's ``torch.where`` load-bearing.

    Examples:
        >>> assign = _random_assign(torch.tensor([[True, False]]))
        >>> bool(assign.align_weights[0, 1] == 0.0), bool((assign.target_boxes[0, 1] == 0.0).all())
        (True, True)
    """
    batch, anchors = fg_mask.shape
    labels = torch.where(fg_mask, torch.randint(0, 3, (batch, anchors)), torch.full((batch, anchors), -1))
    target_boxes = _random_boxes(batch, anchors) * fg_mask.unsqueeze(-1)
    align_weights = torch.rand(batch, anchors) * fg_mask
    return _make_assign(fg_mask=fg_mask, target_labels=labels, target_boxes=target_boxes, align_weights=align_weights)


def _mixed_fg_mask() -> Tensor:
    """``(4, 6)`` mask whose second image has no positive at all.

    Examples:
        >>> mask = _mixed_fg_mask()
        >>> tuple(mask.shape), bool(mask[1].any()), bool(mask.any())
        ((4, 6), False, True)
    """
    mask = torch.rand(4, 6) < 0.5
    mask[1] = False
    mask[0, 0] = True
    return mask


class TestDenseBoxTermsMatchMaskedReference:
    """The dense ``(B, A)`` box terms reproduce the masked-gather implementation (WP-183).

    Each scenario builds one random assignment and scores it twice — once with the
    shipped dense form and once with the pre-change oracle above — on separate leaf
    tensors so the two backward passes cannot accumulate into each other. Values and
    gradients must agree to ``1e-6`` and the gradients must be finite.
    """

    @staticmethod
    def _score_both(
        pred_boxes: Tensor, assign: AssignResult, strides: Tensor | None
    ) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
        """Return ``(box, l1, grad)`` from the dense form and from the oracle, on separate leaves."""
        weight_sum = assign.align_weights.sum().clamp(min=1.0)
        dense_leaf = pred_boxes.clone().requires_grad_(True)
        dense_box, dense_l1 = DetectionBranchLoss._box_losses(dense_leaf, assign, weight_sum, strides)
        (dense_box + dense_l1).backward()
        oracle_leaf = pred_boxes.clone().requires_grad_(True)
        oracle_box, oracle_l1 = _masked_box_losses_oracle(oracle_leaf, assign, weight_sum, strides)
        (oracle_box + oracle_l1).backward()
        assert dense_leaf.grad is not None and oracle_leaf.grad is not None
        return (dense_box, dense_l1, dense_leaf.grad), (oracle_box, oracle_l1, oracle_leaf.grad)

    @pytest.mark.parametrize(
        "strides",
        [
            pytest.param(None, id="pixel-frame"),
            pytest.param(torch.tensor(_LEVEL_STRIDES * 2), id="stride-units"),
        ],
    )
    def test_mixed_batch_with_one_empty_image(self, strides: Tensor | None) -> None:
        """A ``B=4`` batch where one image has no positive scores identically in value and gradient.

        The empty image is the row a masked gather drops outright and the dense form
        carries as exact zeros; the two must sum to the same number either way.
        """
        assign = _random_assign(_mixed_fg_mask())
        pred_boxes = _random_boxes(4, 6)

        (dense_box, dense_l1, dense_grad), (oracle_box, oracle_l1, oracle_grad) = self._score_both(
            pred_boxes, assign, strides
        )

        assert torch.allclose(dense_box, oracle_box, atol=1e-6)
        assert torch.allclose(dense_l1, oracle_l1, atol=1e-6)
        assert torch.allclose(dense_grad, oracle_grad, atol=1e-6)
        assert bool(torch.isfinite(dense_grad).all())

    def test_all_background_batch_is_zero_and_connected(self) -> None:
        """A batch with no positive anywhere gives exact zeros and a finite, all-zero gradient.

        This is the empty-mask contract the module docstring states; the oracle
        reaches it through empty gathers and the dense form through an all-``False``
        selection, and both must land on the same zero.
        """
        assign = _random_assign(torch.zeros(4, 6, dtype=torch.bool))
        pred_boxes = _random_boxes(4, 6)

        (dense_box, dense_l1, dense_grad), (oracle_box, oracle_l1, oracle_grad) = self._score_both(
            pred_boxes, assign, torch.tensor(_LEVEL_STRIDES * 2)
        )

        assert float(dense_box.detach()) == 0.0 and float(dense_l1.detach()) == 0.0
        assert float(oracle_box.detach()) == 0.0 and float(oracle_l1.detach()) == 0.0
        assert torch.equal(dense_grad, torch.zeros_like(pred_boxes))
        assert torch.equal(oracle_grad, torch.zeros_like(pred_boxes))

    def test_degenerate_background_predictions_leak_nothing(self) -> None:
        """Zero-area and inverted predictions on background anchors reach neither the value nor the gradient.

        A background anchor's target is the all-zero box, so a dense CIoU there is
        scored on a degenerate pair the oracle never evaluates. Planting a zero box, a
        box equal to that zero target and an inverted box on background anchors proves
        the ``torch.where`` selection — and the CIoU guards behind it — keep them out.
        """
        fg_mask = torch.tensor([[True, False, False, False], [False, True, False, False]])
        assign = _random_assign(fg_mask)
        pred_boxes = _random_boxes(2, 4)
        pred_boxes[0, 1] = torch.tensor([5.0, 5.0, 5.0, 5.0])  # zero area
        pred_boxes[0, 2] = torch.zeros(4)  # equals the background target exactly
        pred_boxes[1, 2] = torch.tensor([9.0, 9.0, 3.0, 3.0])  # inverted
        pred_boxes[1, 3] = torch.zeros(4)

        (dense_box, dense_l1, dense_grad), (oracle_box, oracle_l1, oracle_grad) = self._score_both(
            pred_boxes, assign, torch.tensor((*_LEVEL_STRIDES, 8.0))
        )

        assert torch.allclose(dense_box, oracle_box, atol=1e-6)
        assert torch.allclose(dense_l1, oracle_l1, atol=1e-6)
        assert bool(torch.isfinite(dense_grad).all())
        assert torch.allclose(dense_grad, oracle_grad, atol=1e-6)
        assert torch.equal(dense_grad[~fg_mask], torch.zeros(6, 4))
