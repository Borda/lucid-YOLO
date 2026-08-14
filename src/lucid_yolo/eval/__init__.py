# SPDX-License-Identifier: Apache-2.0
"""eval subpackage — see blueprint section 7 layout.

The detection evaluation protocol of blueprint sec. 5.11: ``bbox`` mAP50-95
reported for **both** decode paths from one checkpoint in one pass
(:class:`~lucid_yolo.eval.coco_eval.DualPathEvaluator`), with the A9-tuple to
torchmetrics-prediction conversion
(:func:`~lucid_yolo.eval.coco_eval.detections_to_predictions`) and the
:class:`torchmetrics.detection.MeanAveragePrecision` wrapper
(:func:`~lucid_yolo.eval.coco_eval.evaluate_bbox`).

The segmentation half starts at
:func:`~lucid_yolo.eval.segment_decode.decode_instance_masks`, which turns
prototypes and per-detection coefficients into box-cropped binary masks, and
:func:`~lucid_yolo.eval.segment_decode.masks_to_original`, the mask-side twin of
the box inverse-letterbox (A10). Scoring them is
:func:`~lucid_yolo.eval.coco_eval.evaluate_segm` and, for both metrics in one
metric pass, :func:`~lucid_yolo.eval.coco_eval.evaluate_bbox_and_segm`; the
ground-truth masks come from
:func:`~lucid_yolo.eval.annotations.annotation_mask`, which decodes COCO's
polygon and RLE encodings alike.

The oriented half is :mod:`lucid_yolo.eval.dota_eval` (WP-063, A24): exact
polygon-intersection rotated IoU (:func:`~lucid_yolo.data.rotated_geom.rotated_iou`,
defined in the geometry module since WP-091c and re-exported by both)
and the COCO-style accumulator that runs on it
(:func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map`), reached through the
adapters :func:`~lucid_yolo.eval.dota_eval.rotated_detections_to_predictions` and
:func:`~lucid_yolo.eval.dota_eval.tiled_targets_to_ground_truth`. That module
scores one evaluation unit at a time; merging detections across overlapping tiles
back onto whole DOTA images is :mod:`lucid_yolo.eval.tile_merge` (WP-107), whose
core-ownership rule (A59) decides which tile owns a detection before anything is
scored.

:mod:`lucid_yolo.eval.checkpoint` is the step every acceptance script starts with
(:func:`~lucid_yolo.eval.checkpoint.load_eval_module`,
:func:`~lucid_yolo.eval.checkpoint.pick_device`): read a Lightning checkpoint, optionally
overlay the EMA shadow stored inside it, and resolve a device. It lives here rather than
in whichever script needed it first, so the oriented and detection instruments cannot
drift apart on what "the EMA numbers" means. It is deliberately **not** re-exported from
this package: it imports :class:`~lucid_yolo.ptl.module.DetectionLitModule`, which
imports :mod:`lucid_yolo.eval.dota_eval`, so hoisting it here closes an import cycle
through this very module. Import it by its own path.
"""

from lucid_yolo.eval.annotations import (
    EvalImage,
    LazyTargets,
    annotation_mask,
    annotations_to_target,
    empty_target,
    letterboxed_batches,
    load_eval_annotations,
)
from lucid_yolo.eval.coco_eval import (
    DualPathEvaluator,
    detections_to_predictions,
    evaluate_bbox,
    evaluate_bbox_and_segm,
    evaluate_segm,
)
from lucid_yolo.eval.dota_eval import (
    IOU_THRESHOLDS,
    MAX_DETECTIONS,
    RECALL_POINTS,
    evaluate_rotated_map,
    rotated_detections_to_predictions,
    rotated_iou,
    tiled_targets_to_ground_truth,
)
from lucid_yolo.eval.segment_decode import decode_instance_masks, masks_to_original

__all__ = [
    "IOU_THRESHOLDS",
    "MAX_DETECTIONS",
    "RECALL_POINTS",
    "DualPathEvaluator",
    "EvalImage",
    "LazyTargets",
    "annotation_mask",
    "annotations_to_target",
    "decode_instance_masks",
    "detections_to_predictions",
    "empty_target",
    "evaluate_bbox",
    "evaluate_bbox_and_segm",
    "evaluate_rotated_map",
    "evaluate_segm",
    "letterboxed_batches",
    "load_eval_annotations",
    "masks_to_original",
    "rotated_detections_to_predictions",
    "rotated_iou",
    "tiled_targets_to_ground_truth",
]
