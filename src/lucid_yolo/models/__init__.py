# SPDX-License-Identifier: Apache-2.0
"""models subpackage — see blueprint section 7 layout."""

from lucid_yolo.models.backbone import DetectionBackbone
from lucid_yolo.models.blocks import (
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
from lucid_yolo.models.build import (
    Detector,
    Segmenter,
    SegmentOutput,
    build_detector,
    build_segmenter,
    count_flops,
    count_params,
)
from lucid_yolo.models.heads import (
    DualDetectionHead,
    DualHeadOutput,
    ProtoFusion,
    ProtoNet,
    SemanticAux,
    assemble_masks,
    decode_ltrb,
    o2o_topk,
)
from lucid_yolo.models.neck import DetectionNeck
from lucid_yolo.models.registry import VARIANTS, ScaleSpec, scale_spec

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
    "ProtoFusion",
    "ProtoNet",
    "ScaleSpec",
    "SegmentOutput",
    "Segmenter",
    "SemanticAux",
    "SpatialAttention",
    "assemble_masks",
    "build_detector",
    "build_segmenter",
    "count_flops",
    "count_params",
    "decode_ltrb",
    "o2o_topk",
    "scale_spec",
]
