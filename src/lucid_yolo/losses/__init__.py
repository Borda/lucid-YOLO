# SPDX-License-Identifier: Apache-2.0
"""losses subpackage — see blueprint section 7 layout."""

from lucid_yolo.losses.angle_loss import aspect_ratio_weight, square_angle_loss, wrap_angle_delta
from lucid_yolo.losses.ciou import box_iou_aligned, ciou_loss, complete_iou
from lucid_yolo.losses.detection_loss import DetectionBranchLoss, DetectionLossOutput
from lucid_yolo.losses.dual_loss import DualBranchLoss, DualLossOutput
from lucid_yolo.losses.keypoint_nll_loss import LaplaceNLLLoss
from lucid_yolo.losses.mask_loss import instance_mask_loss
from lucid_yolo.losses.oriented_loss import (
    DEFAULT_ROTATED_IOU_FORM,
    ROTATED_IOU_FORMS,
    OrientedLossOutput,
    oriented_branch_terms,
)
from lucid_yolo.losses.probiou import probabilistic_iou, probiou_bhattacharyya_loss, probiou_hellinger_loss
from lucid_yolo.losses.progressive import ProgressiveLossSchedule, progressive_alpha
from lucid_yolo.losses.rle_loss import RLELoss
from lucid_yolo.losses.semantic_loss import SemanticAuxOutput, semantic_aux_loss

__all__ = [
    "DEFAULT_ROTATED_IOU_FORM",
    "ROTATED_IOU_FORMS",
    "DetectionBranchLoss",
    "DetectionLossOutput",
    "DualBranchLoss",
    "DualLossOutput",
    "LaplaceNLLLoss",
    "OrientedLossOutput",
    "ProgressiveLossSchedule",
    "RLELoss",
    "SemanticAuxOutput",
    "aspect_ratio_weight",
    "box_iou_aligned",
    "ciou_loss",
    "complete_iou",
    "instance_mask_loss",
    "oriented_branch_terms",
    "probabilistic_iou",
    "probiou_bhattacharyya_loss",
    "probiou_hellinger_loss",
    "progressive_alpha",
    "semantic_aux_loss",
    "square_angle_loss",
    "wrap_angle_delta",
]
