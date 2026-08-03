# SPDX-License-Identifier: Apache-2.0
"""data subpackage — see blueprint section 7 layout."""

from __future__ import annotations

from lucid_yolo.data.affine import AffineParams, FusedAffineLetterbox, RandomAffine
from lucid_yolo.data.augment import HorizontalFlip, HSVJitter, hsv_to_rgb, rgb_to_hsv
from lucid_yolo.data.coco import CocoDetectionDataset, build_scale_policy
from lucid_yolo.data.download import download_coco
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.mixup import CopyPaste, Mixup
from lucid_yolo.data.mosaic import MosaicAssembly
from lucid_yolo.data.targets import Targets
from lucid_yolo.data.transforms import (
    Compose,
    GeometricTransform,
    apply_affine_to_points,
    boxes_from_polygons,
)
from lucid_yolo.data.verify import VerifyResult, verify_coco_root

__all__ = [
    "AffineParams",
    "CocoDetectionDataset",
    "Compose",
    "CopyPaste",
    "FusedAffineLetterbox",
    "GeometricTransform",
    "HSVJitter",
    "HorizontalFlip",
    "Letterbox",
    "Mixup",
    "MosaicAssembly",
    "RandomAffine",
    "Targets",
    "VerifyResult",
    "apply_affine_to_points",
    "boxes_from_polygons",
    "build_scale_policy",
    "download_coco",
    "hsv_to_rgb",
    "rgb_to_hsv",
    "verify_coco_root",
]
