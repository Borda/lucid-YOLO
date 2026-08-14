# SPDX-License-Identifier: Apache-2.0
"""decode subpackage — see blueprint section 7 layout.

Two inference decoders reduce the dual head's dense outputs to the **same**
fixed-size A9 detection tuple, so a single evaluator can consume either
interchangeably (blueprint sec. 5.5):

- :class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` — the suppression-free
  score-based top-k path over the one-to-one branch: ranks anchors by
  classification confidence with no IoU computation and no NMS (WP-041; R3
  sec. 4, A9);
- :class:`~lucid_yolo.decode.nms_path.NMSDecoder` — the non-E2E path over the dense
  one-to-many branch: confidence threshold then class-wise NMS, for the "mAP
  (non-E2E)" comparison column (WP-042; R1 sec. 3.2.1).

:class:`~lucid_yolo.decode.rotated_nms.RotatedNMSDecoder` (WP-091b) is the oriented
third: the same non-E2E path over the dense branch's *rotated* outputs, suppressing by
the exact rotated overlap of :func:`~lucid_yolo.eval.dota_eval.rotated_iou` rather than
by the boxes' upright envelopes, and emitting the A45 tuple
:func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk` emits so the oriented tier's two
columns compare. It serves the one-to-many branch and exists as that comparison
baseline — the one-to-one branch's freedom from suppression is R1's claim, not a gap this
fills.

:func:`~lucid_yolo.decode.common.to_letterboxed_original` is the eval-time hook that
un-letterboxes either path's boxes back to original-image coordinates (A10), and
:func:`~lucid_yolo.decode.common.rboxes_to_letterboxed_original` (WP-088) is its
oriented twin for the A45 tuple — the same inverse, with the extents scaled and
``theta`` left alone, because a letterbox is isotropic and turns no angle.
"""

from lucid_yolo.decode.common import rboxes_to_letterboxed_original, to_letterboxed_original
from lucid_yolo.decode.nms_path import NMSDecoder
from lucid_yolo.decode.rotated_nms import ROTATED_NMS_IOU_THRESHOLD, RotatedNMSDecoder
from lucid_yolo.decode.topk_e2e import TopKDecoder

__all__ = [
    "ROTATED_NMS_IOU_THRESHOLD",
    "NMSDecoder",
    "RotatedNMSDecoder",
    "TopKDecoder",
    "rboxes_to_letterboxed_original",
    "to_letterboxed_original",
]
