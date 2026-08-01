# SPDX-License-Identifier: Apache-2.0
"""decode subpackage — see blueprint section 7 layout.

The score-based top-k end-to-end decode of the one-to-one branch (WP-041): a
suppression-free path that ranks anchors by classification confidence with no
IoU computation (R3 sec. 4, A9).
"""

from lit_yolo.decode.topk_e2e import TopKDecoder, to_letterboxed_original

__all__ = ["TopKDecoder", "to_letterboxed_original"]
