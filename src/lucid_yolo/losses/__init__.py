# SPDX-License-Identifier: Apache-2.0
"""losses subpackage — see blueprint section 7 layout."""

from lucid_yolo.losses.ciou import box_iou_aligned, ciou_loss, complete_iou
from lucid_yolo.losses.detection_loss import DetectionBranchLoss, DetectionLossOutput
from lucid_yolo.losses.dual_loss import DualBranchLoss, DualLossOutput
from lucid_yolo.losses.mask_loss import instance_mask_loss
from lucid_yolo.losses.probiou import probabilistic_iou, probiou_bhattacharyya_loss, probiou_hellinger_loss
from lucid_yolo.losses.progressive import ProgressiveLossSchedule, progressive_alpha
from lucid_yolo.losses.semantic_loss import SemanticAuxOutput, semantic_aux_loss

__all__ = [
    "DetectionBranchLoss",
    "DetectionLossOutput",
    "DualBranchLoss",
    "DualLossOutput",
    "ProgressiveLossSchedule",
    "SemanticAuxOutput",
    "box_iou_aligned",
    "ciou_loss",
    "complete_iou",
    "instance_mask_loss",
    "probabilistic_iou",
    "probiou_bhattacharyya_loss",
    "probiou_hellinger_loss",
    "progressive_alpha",
    "semantic_aux_loss",
]
