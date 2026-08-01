# SPDX-License-Identifier: Apache-2.0
"""decode subpackage — see blueprint section 7 layout.

Two inference decoders reduce the dual head's dense outputs to the **same**
fixed-size A9 detection tuple, so a single evaluator can consume either
interchangeably (blueprint sec. 5.5):

- :class:`~lit_yolo.decode.topk_e2e.TopKDecoder` — the suppression-free
  score-based top-k path over the one-to-one branch: ranks anchors by
  classification confidence with no IoU computation and no NMS (WP-041; R3
  sec. 4, A9);
- :class:`~lit_yolo.decode.nms_path.NMSDecoder` — the non-E2E path over the dense
  one-to-many branch: confidence threshold then class-wise NMS, for the "mAP
  (non-E2E)" comparison column (WP-042; R1 sec. 3.2.1).

:func:`~lit_yolo.decode.common.to_letterboxed_original` is the eval-time hook that
un-letterboxes either path's boxes back to original-image coordinates (A10).
"""

from lit_yolo.decode.common import to_letterboxed_original
from lit_yolo.decode.nms_path import NMSDecoder
from lit_yolo.decode.topk_e2e import TopKDecoder

__all__ = ["NMSDecoder", "TopKDecoder", "to_letterboxed_original"]
