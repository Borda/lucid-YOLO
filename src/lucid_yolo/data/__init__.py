# SPDX-License-Identifier: Apache-2.0
"""data subpackage — see blueprint section 7 layout.

This list is an index, not the public API declaration: every module below states its own
``__all__``, and that is where a symbol is published. Re-exported here are the names some
consumer actually reaches for through the package — the ``Targets`` container, the
composition order and the seven transforms, dataset IO, DOTA parsing and tiling, and the
rotated-box conversions the assigner and the OBB head read — which since WP-158 is what the
list is narrowed to. Membership tracks use rather than ownership, so it is not a boundary:
a symbol absent here is neither private nor upstream's, it is imported from the module that
owns it, the way ``rasterize_polygons``, ``rotated_iou`` and ``verify_coco_root`` already
are everywhere they are used.
"""

from __future__ import annotations

from lucid_yolo.data.affine import AffineParams, RandomAffine
from lucid_yolo.data.augment import HorizontalFlip, HSVJitter
from lucid_yolo.data.coco import CocoDetectionDataset, build_scale_policy
from lucid_yolo.data.dota import (
    DOTA_CLASSES,
    DotaObject,
    dota_targets,
    load_dota_targets,
    parse_dota_label_file,
)
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.mixup import CopyPaste, Mixup
from lucid_yolo.data.mosaic import MosaicAssembly
from lucid_yolo.data.rotated_aug import clip_rboxes_to_canvas, warp_rboxes
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
from lucid_yolo.data.yolo import YoloDataConfig, YoloDetectionDataset, load_yolo_targets

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
    "GeometricTransform",
    "HSVJitter",
    "HorizontalFlip",
    "Letterbox",
    "Mixup",
    "MosaicAssembly",
    "RandomAffine",
    "Targets",
    "TiledTargets",
    "YoloDataConfig",
    "YoloDetectionDataset",
    "apply_affine_to_points",
    "boxes_from_polygons",
    "build_scale_policy",
    "canonicalize",
    "clip_rboxes_to_canvas",
    "crop_image",
    "crop_targets",
    "dota_targets",
    "load_dota_targets",
    "load_yolo_targets",
    "parse_dota_label_file",
    "points_in_rboxes",
    "polygons_to_rboxes",
    "rboxes_to_polygons",
    "tile_image_targets",
    "tile_windows",
    "warp_rboxes",
]
