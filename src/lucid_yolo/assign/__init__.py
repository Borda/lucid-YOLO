# SPDX-License-Identifier: Apache-2.0
"""assign subpackage — see blueprint section 7 layout."""

from lucid_yolo.assign.grid import (
    HEAD_STRIDES,
    anchor_grid,
    make_anchor_points,
    require_grid_canvas,
    require_grid_side,
)
from lucid_yolo.assign.one_to_one import UniqueAssigner
from lucid_yolo.assign.stal import SmallTargetAssigner, surrogate_boxes, surrogate_rboxes
from lucid_yolo.assign.tal import AssignResult, TaskAlignedAssigner

__all__ = [
    "HEAD_STRIDES",
    "AssignResult",
    "SmallTargetAssigner",
    "TaskAlignedAssigner",
    "UniqueAssigner",
    "anchor_grid",
    "make_anchor_points",
    "require_grid_canvas",
    "require_grid_side",
    "surrogate_boxes",
    "surrogate_rboxes",
]
