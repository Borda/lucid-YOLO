# SPDX-License-Identifier: Apache-2.0
"""Complete-IoU (CIoU) box regression loss (WP-024).

Implements the Complete-IoU objective for axis-aligned boxes, transcribed by
hand from the Distance-IoU / Complete-IoU paper (R10, arXiv:1911.08287). CIoU
augments the plain IoU overlap with two geometric penalties: a normalized
center-distance term (DIoU) and an aspect-ratio consistency term.

For a predicted box ``b`` and target box ``b_gt`` (both ``xyxy``):

- ``IoU = area(b intersect b_gt) / area(b union b_gt)``.
- DIoU penalty ``= rho^2 / c^2`` where ``rho`` is the Euclidean distance between
  the two box centers and ``c`` is the diagonal length of the smallest box
  enclosing both boxes.
- Aspect term ``v = (4 / pi^2) * (atan2(w_gt, h_gt) - atan2(w, h))^2`` with
  trade-off weight ``alpha = v / ((1 - IoU) + v)``. Per the paper's optimization
  note, ``alpha`` is treated as a constant with respect to gradients (detached).
- ``CIoU = IoU - rho^2 / c^2 - alpha * v`` and ``L_CIoU = 1 - CIoU``.

Every division is epsilon-guarded so that degenerate zero-area boxes produce
neither ``NaN`` nor ``Inf`` in the forward or backward pass. Squared distances are
used directly (no ``sqrt``), which also avoids the infinite gradient of ``sqrt``
at the origin.

The aspect term uses ``atan2(w, h)`` rather than the paper's written
``atan(w / h)``, because the two agree everywhere the ratio is defined and differ
exactly where it is not. Guarding the ratio as ``atan(w / (h + eps))`` keeps the
*value* finite at ``h = 0`` but makes the derivative ``1 / eps``: against a
``[0, 0, 10, 20]`` target, a collapsed prediction ``[5, 5, 5, 5]`` used to
backpropagate ``dL/dx1 = 3.0e5`` — finite, and large enough to move a box across
the image in one step. ``atan2`` is exact on the whole axis (``atan2(w, 0) =
pi/2``), and its derivative at the origin is ``0`` on CPU and MPS alike, so the
collapsed box now contributes nothing through the aspect term instead of
dominating the batch.

An **inverted** box — one whose ``x2 < x1`` or ``y2 < y1`` — is deliberately left
gradient-dead in this loss. Its width and height clamp to zero, so every term
here sees a point box and no derivative points back towards un-inverting it; the
measured ``dL/dx1 = dL/dx2 = 0`` is the intended answer, not an oversight. CIoU
is a similarity between two regions and an inverted description names no region,
so the repair belongs at the boundary that produces the box, not here — see
:meth:`~lucid_yolo.decode.nms_path.NMSDecoder._decode_image`, which drops such
rows rather than scoring them.
"""

import math

import torch
from torch import Tensor

_FOUR_OVER_PI_SQ: float = 4.0 / (math.pi * math.pi)


def box_iou_aligned(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Element-wise IoU between aligned pairs of axis-aligned boxes.

    Computes IoU for each ``(pred[i], target[i])`` pair independently; this is
    the paired (diagonal) IoU, not the ``(N, M)`` cross matrix.

    Args:
        pred: Predicted boxes of shape ``(N, 4)`` in ``xyxy`` order.
        target: Target boxes of shape ``(N, 4)`` in ``xyxy`` order.
        eps: Small constant added to the union to guard the division.

    Returns:
        IoU per pair, shape ``(N,)``, in ``[0, 1]``.

    Examples:
        >>> import torch
        >>> pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        >>> target = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
        >>> box_iou_aligned(pred, target).round(decimals=4)
        tensor([0.1429])
    """
    inter_x1 = torch.maximum(pred[:, 0], target[:, 0])
    inter_y1 = torch.maximum(pred[:, 1], target[:, 1])
    inter_x2 = torch.minimum(pred[:, 2], target[:, 2])
    inter_y2 = torch.minimum(pred[:, 3], target[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    intersection = inter_w * inter_h

    pred_area = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    target_area = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = pred_area + target_area - intersection

    return intersection / (union + eps)


def complete_iou(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Complete-IoU (CIoU) between aligned pairs of axis-aligned boxes.

    Autograd-safe: no in-place operations on the inputs, the aspect-ratio
    trade-off weight ``alpha`` is detached (constant w.r.t. gradients per the
    paper), and every division is epsilon-guarded so that degenerate zero-area
    boxes stay finite in both forward and backward passes. The aspect term needs
    no such guard — ``atan2`` is defined at ``h = 0`` — which is why a collapsed
    box is merely uninformative here rather than a ``1 / eps`` gradient spike
    (module docstring).

    Args:
        pred: Predicted boxes of shape ``(N, 4)`` in ``xyxy`` order.
        target: Target boxes of shape ``(N, 4)`` in ``xyxy`` order.
        eps: Small constant guarding the IoU union, the enclosing-diagonal
            division, and the ``alpha`` denominator. It no longer reaches the
            aspect term, which is unconditionally defined.

    Returns:
        CIoU per pair, shape ``(N,)``. Equals ``1`` for identical boxes and can
        fall below ``0`` when the center-distance penalty dominates.

    Examples:
        >>> import torch
        >>> boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        >>> complete_iou(boxes, boxes).round(decimals=4)
        tensor([1.])
    """
    iou = box_iou_aligned(pred, target, eps=eps)

    pred_cx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_cy = (pred[:, 1] + pred[:, 3]) * 0.5
    target_cx = (target[:, 0] + target[:, 2]) * 0.5
    target_cy = (target[:, 1] + target[:, 3]) * 0.5
    center_dist_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    enclose_x1 = torch.minimum(pred[:, 0], target[:, 0])
    enclose_y1 = torch.minimum(pred[:, 1], target[:, 1])
    enclose_x2 = torch.maximum(pred[:, 2], target[:, 2])
    enclose_y2 = torch.maximum(pred[:, 3], target[:, 3])
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    distance_penalty = center_dist_sq / (enclose_diag_sq + eps)

    pred_w = (pred[:, 2] - pred[:, 0]).clamp(min=0)
    pred_h = (pred[:, 3] - pred[:, 1]).clamp(min=0)
    target_w = (target[:, 2] - target[:, 0]).clamp(min=0)
    target_h = (target[:, 3] - target[:, 1]).clamp(min=0)
    # atan2, not atan(w / (h + eps)): the guarded ratio carries a 1/eps derivative at
    # h = 0, which turned a collapsed box into a 3e5 gradient (module docstring).
    arctan_diff = torch.atan2(target_w, target_h) - torch.atan2(pred_w, pred_h)
    v = _FOUR_OVER_PI_SQ * arctan_diff**2

    # alpha is a constant w.r.t. gradients (R10 optimization note): detach it.
    alpha = (v / ((1 - iou) + v + eps)).detach()

    return iou - distance_penalty - alpha * v


def ciou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Complete-IoU regression loss, ``1 - CIoU``, per aligned box pair.

    Args:
        pred: Predicted boxes of shape ``(N, 4)`` in ``xyxy`` order.
        target: Target boxes of shape ``(N, 4)`` in ``xyxy`` order.
        eps: Small constant forwarded to :func:`complete_iou`.

    Returns:
        Loss per pair, shape ``(N,)``. Equals ``0`` for identical boxes and
        exceeds ``1`` when the center-distance penalty pushes CIoU below ``0``.

    Examples:
        >>> import torch
        >>> boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        >>> ciou_loss(boxes, boxes).round(decimals=4)
        tensor([0.])
    """
    return 1 - complete_iou(pred, target, eps=eps)
