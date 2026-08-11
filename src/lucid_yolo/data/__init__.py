# SPDX-License-Identifier: Apache-2.0
"""data subpackage — see blueprint section 7 layout."""

from __future__ import annotations

from lucid_yolo.data.affine import AffineParams, FusedAffineLetterbox, RandomAffine
from lucid_yolo.data.augment import HorizontalFlip, HSVJitter, hsv_to_rgb, rgb_to_hsv
from lucid_yolo.data.coco import CocoDetectionDataset, build_scale_policy
from lucid_yolo.data.dota import (
    DOTA_CLASSES,
    DotaObject,
    dota_targets,
    load_dota_targets,
    parse_dota_label_file,
)
from lucid_yolo.data.download import download_coco
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.mixup import CopyPaste, Mixup
from lucid_yolo.data.mosaic import MosaicAssembly
from lucid_yolo.data.rasterize import rasterize_polygon, rasterize_polygons
from lucid_yolo.data.rotated_geom import (
    canonicalize,
    points_in_rboxes,
    polygons_to_rboxes,
    rboxes_to_polygons,
)
from lucid_yolo.data.targets import Targets
from lucid_yolo.data.tiling import (
    CROP_OVERLAP,
    DIFFICULT_AREA_FRACTION,
    PATCH_SIZE,
    TiledTargets,
    crop_image,
    crop_targets,
    tile_image_targets,
    tile_windows,
)
from lucid_yolo.data.transforms import (
    Compose,
    GeometricTransform,
    apply_affine_to_points,
    boxes_from_polygons,
)
from lucid_yolo.data.verify import VerifyResult, verify_coco_root

__all__ = [
    "CROP_OVERLAP",
    "DIFFICULT_AREA_FRACTION",
    "DOTA_CLASSES",
    "PATCH_SIZE",
    "AffineParams",
    "CocoDetectionDataset",
    "Compose",
    "CopyPaste",
    "DotaObject",
    "FusedAffineLetterbox",
    "GeometricTransform",
    "HSVJitter",
    "HorizontalFlip",
    "Letterbox",
    "Mixup",
    "MosaicAssembly",
    "RandomAffine",
    "Targets",
    "TiledTargets",
    "VerifyResult",
    "apply_affine_to_points",
    "boxes_from_polygons",
    "build_scale_policy",
    "canonicalize",
    "crop_image",
    "crop_targets",
    "dota_targets",
    "download_coco",
    "hsv_to_rgb",
    "load_dota_targets",
    "parse_dota_label_file",
    "points_in_rboxes",
    "polygons_to_rboxes",
    "rasterize_polygon",
    "rasterize_polygons",
    "rboxes_to_polygons",
    "rgb_to_hsv",
    "tile_image_targets",
    "tile_windows",
    "verify_coco_root",
]
