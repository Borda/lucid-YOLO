# SPDX-License-Identifier: Apache-2.0
"""assign subpackage — see blueprint section 7 layout."""

from open_yolos.assign.grid import make_anchor_points
from open_yolos.assign.one_to_one import UniqueAssigner
from open_yolos.assign.stal import SmallTargetAssigner, surrogate_boxes
from open_yolos.assign.tal import AssignResult, TaskAlignedAssigner

__all__ = [
    "AssignResult",
    "SmallTargetAssigner",
    "TaskAlignedAssigner",
    "UniqueAssigner",
    "make_anchor_points",
    "surrogate_boxes",
]
