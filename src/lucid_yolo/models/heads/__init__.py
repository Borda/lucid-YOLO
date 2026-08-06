# SPDX-License-Identifier: Apache-2.0
"""heads subpackage — see blueprint section 7 layout."""

from lucid_yolo.models.heads.detect import (
    DualDetectionHead,
    DualHeadOutput,
    decode_ltrb,
    o2o_topk,
)
from lucid_yolo.models.heads.proto import ProtoFusion

__all__ = [
    "DualDetectionHead",
    "DualHeadOutput",
    "ProtoFusion",
    "decode_ltrb",
    "o2o_topk",
]
