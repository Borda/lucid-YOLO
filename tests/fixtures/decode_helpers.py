# SPDX-License-Identifier: Apache-2.0
"""Anchor-grid, overlap and survivor helpers shared by the ``tests/decode`` decoder modules.

The axis-aligned decoder test modules plant the same four-anchor scenes into raw head outputs, measure the same box
overlap and select the same live rows from a padded detection batch. They live here once instead of being copied per
module. Tests import this module by bare name because ``tests/conftest.py`` puts ``tests/fixtures`` on ``sys.path``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lucid_yolo.assign.grid import make_anchor_points

#: The fixed detection count of a decoder run with its default cap.
DET_CAP = 300

#: Columns of an ``xyxy`` box at the head of a decoded row.
BOX_CORNERS = 4


def grid(height: int, width: int, stride: int = 8) -> tuple[Tensor, Tensor]:
    """Return ``(anchor_points, strides)`` for a single ``(height, width)`` level.

    Examples:
        >>> points, strides = grid(2, 2, stride=8)
        >>> points
        tensor([[ 4.,  4.],
                [12.,  4.],
                [ 4., 12.],
                [12., 12.]])
        >>> strides.tolist()
        [8.0, 8.0, 8.0, 8.0]
    """
    return make_anchor_points([(height, width)], [stride])


def pairwise_iou(box_a: Tensor, box_b: Tensor) -> float:
    """Return the IoU of two ``xyxy`` boxes (test-side, never used by the decoders).

    Examples:
        >>> round(pairwise_iou(torch.tensor([0.0, 0.0, 2.0, 2.0]), torch.tensor([1.0, 1.0, 3.0, 3.0])), 4)
        0.1429
        >>> pairwise_iou(torch.tensor([0.0, 0.0, 2.0, 2.0]), torch.tensor([0.0, 0.0, 2.0, 2.0]))
        1.0
    """
    inter_x1 = torch.maximum(box_a[0], box_b[0])
    inter_y1 = torch.maximum(box_a[1], box_b[1])
    inter_x2 = torch.minimum(box_a[2], box_b[2])
    inter_y2 = torch.minimum(box_a[3], box_b[3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return float(inter / (area_a + area_b - inter))


def surviving_rows(detections: Tensor, score_column: int = 4) -> Tensor:
    """Return the first image's rows with a non-zero score, dropping the padding.

    Args:
        detections: A ``(B, max_det, W)`` padded detection batch.
        score_column: The column holding the score; 4 for an ``xyxy`` row, 5 for a rotated row.

    Examples:
        >>> detections = torch.zeros(1, 3, 6)
        >>> detections[0, 0, 4] = 0.9
        >>> surviving_rows(detections).shape
        torch.Size([1, 6])
        >>> rotated = torch.zeros(1, 3, 7)
        >>> rotated[0, :2, 5] = 0.5
        >>> surviving_rows(rotated, score_column=5).shape
        torch.Size([2, 7])
    """
    return detections[0, detections[0, :, score_column] > 0.0]


def identical_box_pair(first_class: int = 1, second_class: int = 1) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Plant two confident anchors whose distances decode to the identical box ``[-4, -4, 12, 12]``.

    Anchors 0 and 1 of a 2x2 stride-8 grid (centres ``(4, 4)`` and ``(12, 4)``) get ltrb distances that decode to the
    same box. Anchor 0 is confident for ``first_class`` (logit 6) and anchor 1 for ``second_class`` (logit 5); the
    other classes and the other two anchors sit at logit -10, far below any confidence threshold the tests set.

    Args:
        first_class: The class index (of three) that anchor 0 is confident about.
        second_class: The class index (of three) that anchor 1 is confident about.

    Returns:
        ``(cls_logits, raw_ltrb, anchor_points, strides)`` for a batch of one.

    Examples:
        >>> cls_logits, raw_ltrb, points, strides = identical_box_pair(first_class=0, second_class=1)
        >>> cls_logits.shape, int(cls_logits[0, 0].argmax()), int(cls_logits[0, 1].argmax())
        (torch.Size([1, 4, 3]), 0, 1)
        >>> raw_ltrb[0, 1].tolist()
        [2.0, 1.0, 0.0, 1.0]
    """
    points, strides = grid(2, 2, stride=8)
    cls_logits = torch.full((1, 4, 3), -10.0)
    cls_logits[0, 0, first_class] = 6.0
    cls_logits[0, 1, second_class] = 5.0
    raw_ltrb = torch.zeros(1, 4, 4)
    raw_ltrb[0, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])  # anchor 0 -> [-4,-4,12,12]
    raw_ltrb[0, 1] = torch.tensor([2.0, 1.0, 0.0, 1.0])  # anchor 1 -> [-4,-4,12,12]
    return cls_logits, raw_ltrb, points, strides
