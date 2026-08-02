# SPDX-License-Identifier: Apache-2.0
"""assign subpackage — see blueprint section 7 layout."""

from lucid_yolo.assign.grid import make_anchor_points
from lucid_yolo.assign.one_to_one import UniqueAssigner
from lucid_yolo.assign.stal import SmallTargetAssigner, surrogate_boxes
from lucid_yolo.assign.tal import AssignResult, TaskAlignedAssigner

__all__ = [
    "AssignResult",
    "SmallTargetAssigner",
    "TaskAlignedAssigner",
    "UniqueAssigner",
    "make_anchor_points",
    "surrogate_boxes",
]
