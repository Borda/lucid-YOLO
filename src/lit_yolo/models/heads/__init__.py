# SPDX-License-Identifier: Apache-2.0
"""heads subpackage — see blueprint section 7 layout."""

from lit_yolo.models.heads.detect import (
    DualDetectionHead,
    DualHeadOutput,
    decode_ltrb,
    o2o_topk,
)

__all__ = [
    "DualDetectionHead",
    "DualHeadOutput",
    "decode_ltrb",
    "o2o_topk",
]
