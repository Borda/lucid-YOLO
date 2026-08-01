# SPDX-License-Identifier: Apache-2.0
"""eval subpackage — see blueprint section 7 layout.

The detection evaluation protocol of blueprint sec. 5.11: pycocotools ``bbox``
mAP50-95 reported for **both** decode paths from one checkpoint in one pass
(:class:`~lit_yolo.eval.coco_eval.DualPathEvaluator`), with the A9-tuple to
pycocotools conversion (:func:`~lit_yolo.eval.coco_eval.detections_to_coco`) and
the COCOeval wrapper (:func:`~lit_yolo.eval.coco_eval.evaluate_bbox`).
"""

from lit_yolo.eval.coco_eval import DualPathEvaluator, detections_to_coco, evaluate_bbox

__all__ = ["DualPathEvaluator", "detections_to_coco", "evaluate_bbox"]
