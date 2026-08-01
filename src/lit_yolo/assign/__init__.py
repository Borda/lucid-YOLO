# SPDX-License-Identifier: Apache-2.0
"""assign subpackage — see blueprint section 7 layout."""

from lit_yolo.assign.grid import make_anchor_points
from lit_yolo.assign.stal import SmallTargetAssigner, surrogate_boxes
from lit_yolo.assign.tal import AssignResult, TaskAlignedAssigner

__all__ = [
    "AssignResult",
    "SmallTargetAssigner",
    "TaskAlignedAssigner",
    "make_anchor_points",
    "surrogate_boxes",
]
