# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-042 confidence-threshold + class-wise NMS decoder.

Covers :class:`lit_yolo.decode.NMSDecoder`, the non-E2E path over the dense
one-to-many branch, as the suppression contrast to the WP-041 one-to-one path:
two heavily overlapping same-class high-score boxes collapse to exactly one
survivor (``test_overlapping_same_class_collapses_to_one`` — the mirror of the
E2E ``test_no_nms_path`` scene, where both survive); overlapping boxes of
*different* classes both survive class-wise NMS; the confidence threshold drops
low-score anchors; the output is a fixed ``(B, 300, 6)`` shape with score-zero
padding rows and a non-increasing score column; the IoU threshold governs how
aggressively same-class overlaps are suppressed; and decoding is deterministic.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from lit_yolo.assign.grid import make_anchor_points
from lit_yolo.decode import NMSDecoder

_DET_CAP = 300


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed the torch RNG so any random inputs are deterministic."""
    torch.manual_seed(0)


def _grid(height: int, width: int, stride: int = 8) -> tuple[Tensor, Tensor]:
    """Return ``(anchor_points, strides)`` for a single ``(height, width)`` level."""
    return make_anchor_points([(height, width)], [stride])


def _pairwise_iou(box_a: Tensor, box_b: Tensor) -> float:
    """Return the IoU of two ``xyxy`` boxes (test-side helper)."""
    inter_x1 = torch.maximum(box_a[0], box_b[0])
    inter_y1 = torch.maximum(box_a[1], box_b[1])
    inter_x2 = torch.minimum(box_a[2], box_b[2])
    inter_y2 = torch.minimum(box_a[3], box_b[3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return float(inter / (area_a + area_b - inter))


def test_overlapping_same_class_collapses_to_one() -> None:
    """Two heavily overlapping same-class high-score boxes leave exactly one survivor.

    The mirror of the E2E ``test_no_nms_path`` scene: anchors 0 and 1 decode to
    the *identical* box ``[-4, -4, 12, 12]`` and are both confident class-1
    detections, while anchors 2 and 3 fall below the confidence threshold. The
    E2E path keeps both overlapping boxes; class-wise NMS must suppress the
    lower-scoring one and keep exactly one.
    """
    points, strides = _grid(2, 2, stride=8)  # 4 anchors
    cls_logits = torch.full((1, 4, 3), -10.0)
    cls_logits[0, 0, 1] = 6.0
    cls_logits[0, 1, 1] = 5.0
    raw_ltrb = torch.zeros(1, 4, 4)
    raw_ltrb[0, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])  # anchor 0 -> [-4,-4,12,12]
    raw_ltrb[0, 1] = torch.tensor([2.0, 1.0, 0.0, 1.0])  # anchor 1 -> [-4,-4,12,12]

    detections = NMSDecoder()(cls_logits, raw_ltrb, points, strides)

    survivors = detections[0, detections[0, :, 4] > 0.0]
    assert survivors.shape[0] == 1
    assert torch.allclose(survivors[0, :4], torch.tensor([-4.0, -4.0, 12.0, 12.0]))
    assert int(survivors[0, 5]) == 1
    assert float(survivors[0, 4]) == pytest.approx(torch.sigmoid(torch.tensor(6.0)).item())


def test_different_class_overlap_both_survive() -> None:
    """Overlapping boxes of different classes both survive class-wise NMS."""
    points, strides = _grid(2, 2, stride=8)
    cls_logits = torch.full((1, 4, 3), -10.0)
    cls_logits[0, 0, 0] = 6.0  # anchor 0 -> class 0
    cls_logits[0, 1, 1] = 5.0  # anchor 1 -> class 1
    raw_ltrb = torch.zeros(1, 4, 4)
    raw_ltrb[0, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])  # -> [-4,-4,12,12]
    raw_ltrb[0, 1] = torch.tensor([2.0, 1.0, 0.0, 1.0])  # -> [-4,-4,12,12] (identical box)

    detections = NMSDecoder()(cls_logits, raw_ltrb, points, strides)

    survivors = detections[0, detections[0, :, 4] > 0.0]
    assert survivors.shape[0] == 2
    assert _pairwise_iou(survivors[0, :4], survivors[1, :4]) == pytest.approx(1.0)  # same box, kept anyway
    assert set(survivors[:, 5].int().tolist()) == {0, 1}


def test_confidence_threshold_drops_low_scores() -> None:
    """The confidence threshold drops sub-threshold anchors before NMS.

    Two anchors carry distinct classes and non-overlapping boxes, so NMS never
    suppresses either — only the threshold can remove one. With a 0.6 threshold
    the 0.5-score anchor is dropped and the 0.993-score anchor survives.
    """
    points, strides = _grid(2, 2, stride=8)
    cls_logits = torch.full((1, 4, 2), -20.0)
    cls_logits[0, 0, 0] = 6.0  # class 0, sigmoid ~ 0.9975
    cls_logits[0, 1, 1] = 0.0  # class 1, sigmoid = 0.5
    raw_ltrb = torch.zeros(1, 4, 4)  # anchors keep their (distinct) centres as boxes

    detections = NMSDecoder(conf_threshold=0.6)(cls_logits, raw_ltrb, points, strides)

    survivors = detections[0, detections[0, :, 4] > 0.0]
    assert survivors.shape[0] == 1
    assert int(survivors[0, 5]) == 0
    assert float(survivors[0, 4]) == pytest.approx(torch.sigmoid(torch.tensor(6.0)).item())


def test_fixed_shape_with_zero_padding_and_descending_scores() -> None:
    """Output is a fixed ``(B, 300, 6)`` with score-zero padding and sorted scores.

    Five anchors on a 64-anchor grid carry distinct classes (so NMS suppresses
    none) and clear the threshold; the other 59 sit far below it and are dropped.
    Exactly five detections survive, the rest are all-zero padding rows, and the
    score column is non-increasing.
    """
    points, strides = _grid(8, 8, stride=8)  # 64 anchors
    cls_logits = torch.full((1, 64, 5), -20.0)
    for anchor, class_index in enumerate(range(5)):
        cls_logits[0, anchor, class_index] = 3.0 + anchor  # distinct classes, ascending scores
    raw_ltrb = torch.zeros(1, 64, 4)

    detections = NMSDecoder()(cls_logits, raw_ltrb, points, strides)

    assert detections.shape == (1, _DET_CAP, 6)
    scores = detections[0, :, 4]
    assert (scores[:-1] - scores[1:] >= -1e-6).all()  # non-increasing
    assert (scores[:5] > 0.0).all()  # five real detections
    assert torch.equal(detections[0, 5:], torch.zeros(_DET_CAP - 5, 6))  # padding rows are zero


def test_iou_threshold_governs_same_class_suppression() -> None:
    """A loose IoU threshold keeps a moderate same-class overlap; a tight one suppresses it.

    Two same-class boxes overlap with IoU 1/3. A ``0.5`` threshold (loose) leaves
    both, since 1/3 < 0.5; a ``0.2`` threshold (tight) suppresses the lower-scoring
    one, since 1/3 > 0.2.
    """
    points, strides = _grid(1, 2, stride=8)  # anchor centres (4, 4) and (12, 4)
    cls_logits = torch.full((1, 2, 1), -20.0)
    cls_logits[0, 0, 0] = 6.0  # box A, higher score
    cls_logits[0, 1, 0] = 5.0  # box B, lower score
    raw_ltrb = torch.zeros(1, 2, 4)
    raw_ltrb[0, 0] = torch.tensor([0.5, 0.5, 0.75, 0.75])  # -> [0, 0, 10, 10]
    raw_ltrb[0, 1] = torch.tensor([0.875, 1.5, 0.375, 0.75])  # -> [5, 0, 15, 10], IoU 1/3 with A

    loose = NMSDecoder(iou_threshold=0.5)(cls_logits, raw_ltrb, points, strides)
    tight = NMSDecoder(iou_threshold=0.2)(cls_logits, raw_ltrb, points, strides)

    loose_survivors = loose[0, loose[0, :, 4] > 0.0]
    tight_survivors = tight[0, tight[0, :, 4] > 0.0]
    assert _pairwise_iou(loose_survivors[0, :4], torch.tensor([5.0, 0.0, 15.0, 10.0])) == pytest.approx(1 / 3)
    assert loose_survivors.shape[0] == 2
    assert tight_survivors.shape[0] == 1
    assert torch.allclose(tight_survivors[0, :4], torch.tensor([0.0, 0.0, 10.0, 10.0]))  # higher-score box kept


def test_decode_is_deterministic() -> None:
    """Two decodes of the same inputs are bit-for-bit identical."""
    points, strides = _grid(4, 4, stride=8)
    cls_logits = torch.randn(2, 16, 6)
    raw_ltrb = torch.randn(2, 16, 4)
    decoder = NMSDecoder()

    first = decoder(cls_logits, raw_ltrb, points, strides)
    second = decoder(cls_logits, raw_ltrb, points, strides)

    assert torch.equal(first, second)
