# SPDX-License-Identifier: Apache-2.0
"""losses subpackage — see blueprint section 7 layout."""

from open_yolos.losses.ciou import box_iou_aligned, ciou_loss, complete_iou
from open_yolos.losses.detection_loss import DetectionBranchLoss, DetectionLossOutput
from open_yolos.losses.dual_loss import DualBranchLoss, DualLossOutput
from open_yolos.losses.progressive import ProgressiveLossSchedule, progressive_alpha

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
