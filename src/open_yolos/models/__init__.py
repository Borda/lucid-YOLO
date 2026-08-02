# SPDX-License-Identifier: Apache-2.0
"""models subpackage — see blueprint section 7 layout."""

from open_yolos.models.backbone import DetectionBackbone
from open_yolos.models.blocks import (
    C2PSA,
    SPPF,
    Bottleneck,
    C3k,
    C3k2,
    ConvBNAct,
    DepthwiseConv,
    PSABlock,
    SpatialAttention,
)
from open_yolos.models.build import Detector, build_detector, count_flops, count_params
from open_yolos.models.heads import (
    DualDetectionHead,
    DualHeadOutput,
    decode_ltrb,
    o2o_topk,
)
from open_yolos.models.neck import DetectionNeck
from open_yolos.models.registry import VARIANTS, ScaleSpec, scale_spec

__all__ = [
    "C2PSA",
    "SPPF",
    "VARIANTS",
    "Bottleneck",
    "C3k",
    "C3k2",
    "ConvBNAct",
    "DepthwiseConv",
    "DetectionBackbone",
    "DetectionNeck",
    "Detector",
    "DualDetectionHead",
    "DualHeadOutput",
    "PSABlock",
    "ScaleSpec",
    "SpatialAttention",
    "build_detector",
    "count_flops",
    "count_params",
    "decode_ltrb",
    "o2o_topk",
    "scale_spec",
]
