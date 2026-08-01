# SPDX-License-Identifier: Apache-2.0
"""ptl subpackage — PyTorch Lightning integration (module, callbacks, datamodule, CLI)."""

from __future__ import annotations

from lit_yolo.ptl.datamodule import DetectionDataModule, collate_detection

__all__ = ["DetectionDataModule", "collate_detection"]
