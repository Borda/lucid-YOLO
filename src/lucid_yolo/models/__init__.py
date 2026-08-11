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
    OrientedDetector,
    Segmenter,
    SegmentOutput,
    build_detector,
    build_obb_detector,
    build_segmenter,
    count_flops,
    count_params,
)
from lucid_yolo.models.heads import (
    RBOX_DET_WIDTH,
    DualDetectionHead,
    DualHeadOutput,
    ProtoFusion,
    ProtoNet,
    SemanticAux,
    assemble_masks,
    decode_ltrb,
    decode_rboxes,
    o2o_rotated_topk,
    o2o_topk,
)
from lucid_yolo.models.neck import DetectionNeck
from lucid_yolo.models.registry import VARIANTS, ScaleSpec, scale_spec

__all__ = [
    "C2PSA",
    "RBOX_DET_WIDTH",
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
    "OrientedDetector",
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
    "build_obb_detector",
    "build_segmenter",
    "count_flops",
    "count_params",
    "decode_ltrb",
    "decode_rboxes",
    "o2o_rotated_topk",
    "o2o_topk",
    "scale_spec",
]
