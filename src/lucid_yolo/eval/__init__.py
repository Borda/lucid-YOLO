# SPDX-License-Identifier: Apache-2.0
"""eval subpackage — see blueprint section 7 layout.

The detection evaluation protocol of blueprint sec. 5.11: ``bbox`` mAP50-95
reported for **both** decode paths from one checkpoint in one pass
(:class:`~lucid_yolo.eval.coco_eval.DualPathEvaluator`), with the A9-tuple to
torchmetrics-prediction conversion
(:func:`~lucid_yolo.eval.coco_eval.detections_to_predictions`) and the
:class:`torchmetrics.detection.MeanAveragePrecision` wrapper
(:func:`~lucid_yolo.eval.coco_eval.evaluate_bbox`).
"""

from lucid_yolo.eval.annotations import (
    EvalImage,
    annotations_to_target,
    empty_target,
    letterboxed_batches,
    load_eval_annotations,
)
from lucid_yolo.eval.coco_eval import DualPathEvaluator, detections_to_predictions, evaluate_bbox

__all__ = [
    "DualPathEvaluator",
    "EvalImage",
    "annotations_to_target",
    "detections_to_predictions",
    "empty_target",
    "evaluate_bbox",
    "letterboxed_batches",
    "load_eval_annotations",
]
