# SPDX-License-Identifier: Apache-2.0
"""losses subpackage — see blueprint section 7 layout."""

from lit_yolo.losses.ciou import box_iou_aligned, ciou_loss, complete_iou
from lit_yolo.losses.detection_loss import DetectionBranchLoss, DetectionLossOutput
from lit_yolo.losses.dual_loss import DualBranchLoss, DualLossOutput
from lit_yolo.losses.progressive import ProgressiveLossSchedule, progressive_alpha

__all__ = [
    "DetectionBranchLoss",
    "DetectionLossOutput",
    "DualBranchLoss",
    "DualLossOutput",
    "ProgressiveLossSchedule",
    "box_iou_aligned",
    "ciou_loss",
    "complete_iou",
    "progressive_alpha",
]
