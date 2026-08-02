# SPDX-License-Identifier: Apache-2.0
"""ptl subpackage — PyTorch Lightning integration (module, callbacks, datamodule, CLI)."""

from __future__ import annotations

from lucid_yolo.ptl.callbacks import CloseMosaicCallback, EMACallback
from lucid_yolo.ptl.datamodule import DetectionDataModule, collate_detection
from lucid_yolo.ptl.module import DetectionLitModule, pad_targets

__all__ = [
    "CloseMosaicCallback",
    "DetectionDataModule",
    "DetectionLitModule",
    "EMACallback",
    "collate_detection",
    "pad_targets",
]
