# SPDX-License-Identifier: Apache-2.0
"""ptl subpackage — PyTorch Lightning integration (module, callbacks, datamodule, CLI)."""

from __future__ import annotations

from lucid_yolo.ptl.callbacks import CloseMosaicCallback, EMACallback
from lucid_yolo.ptl.datamodule import (
    DetectionDataModule,
    PackedTargets,
    collate_detection,
    pack_targets,
    unpack_batch,
    unpack_targets,
)
from lucid_yolo.ptl.module import (
    DetectionLitModule,
    normalize_keypoints_to_box,
    pad_keypoints,
    pad_rboxes,
    pad_targets,
)

__all__ = [
    "CloseMosaicCallback",
    "DetectionDataModule",
    "DetectionLitModule",
    "EMACallback",
    "PackedTargets",
    "collate_detection",
    "normalize_keypoints_to_box",
    "pack_targets",
    "pad_keypoints",
    "pad_rboxes",
    "pad_targets",
    "unpack_batch",
    "unpack_targets",
]
