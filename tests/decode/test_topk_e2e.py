# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-041 score-based top-k end-to-end decoder (test_topk_e2e.py).

Covers :class:`lit_yolo.decode.TopKDecoder` and
:func:`lit_yolo.decode.to_letterboxed_original`: that the suppression-free path
keeps two heavily overlapping high-score boxes (``test_no_nms_path`` — the
defining property, no IoU, no non-maximum suppression); that the output is a
fixed ``(B, 300, 6)`` shape with score-zero padding rows when fewer anchors
exist; that scores come out sorted descending; that a crafted class spike is
decoded to the right class index; that a confidence threshold zeroes low entries
without changing the shape; that the letterbox inverse round-trips box
coordinates on a known geometry; and that decoding is deterministic. A meta-test
reads the module source and asserts it references no suppression operator or IoU
helper, pinning the no-suppression claim structurally.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from lit_yolo.assign.grid import make_anchor_points
from lit_yolo.data.letterbox import Letterbox
from lit_yolo.data.targets import Targets
from lit_yolo.decode import TopKDecoder, to_letterboxed_original, topk_e2e

_DET_CAP = 300


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed the torch RNG so any random inputs are deterministic."""
    torch.manual_seed(0)


def _grid(height: int, width: int, stride: int = 8) -> tuple[Tensor, Tensor]:
    """Return ``(anchor_points, strides)`` for a single ``(height, width)`` level."""
    return make_anchor_points([(height, width)], [stride])


def _pairwise_iou(box_a: Tensor, box_b: Tensor) -> float:
    """Return the IoU of two ``xyxy`` boxes (test-side, never used by the module)."""
    inter_x1 = torch.maximum(box_a[0], box_b[0])
    inter_y1 = torch.maximum(box_a[1], box_b[1])
    inter_x2 = torch.minimum(box_a[2], box_b[2])
    inter_y2 = torch.minimum(box_a[3], box_b[3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return float(inter / (area_a + area_b - inter))


def test_no_nms_path() -> None:
    """Two heavily overlapping high-score boxes both survive the decode.

    Anchors 0 and 1 (centres ``(4, 4)`` and ``(12, 4)``, stride 8) are given
    ltrb distances crafted so they decode to the *identical* box ``[-4, -4, 12,
    12]``, and both are the confident detections of class 1. A suppression step
    would drop one of the two; the suppression-free path must keep both.
    """
    points, strides = _grid(2, 2, stride=8)  # 4 anchors
    cls_logits = torch.full((1, 4, 3), -10.0)
    cls_logits[0, 0, 1] = 6.0
    cls_logits[0, 1, 1] = 5.0
    raw_ltrb = torch.zeros(1, 4, 4)
    raw_ltrb[0, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])  # anchor 0 -> [-4,-4,12,12]
    raw_ltrb[0, 1] = torch.tensor([2.0, 1.0, 0.0, 1.0])  # anchor 1 -> [-4,-4,12,12]

    detections = TopKDecoder()(cls_logits, raw_ltrb, points, strides)

    top_two = detections[0, :2]
    expected_box = torch.tensor([-4.0, -4.0, 12.0, 12.0])
    assert torch.allclose(top_two[0, :4], expected_box)
    assert torch.allclose(top_two[1, :4], expected_box)
    assert _pairwise_iou(top_two[0, :4], top_two[1, :4]) == pytest.approx(1.0)
    assert (top_two[:, 4] > 0.99).all()  # both high confidence
    assert torch.equal(top_two[:, 5], torch.tensor([1.0, 1.0]))  # both class 1


def test_fixed_shape_with_zero_padding() -> None:
    """Output is a fixed ``(B, 300, 6)`` with score-zero padding rows below k."""
    points, strides = _grid(2, 2, stride=8)  # only 4 anchors < 300
    cls_logits = torch.randn(2, 4, 5)
    raw_ltrb = torch.randn(2, 4, 4)

    detections = TopKDecoder()(cls_logits, raw_ltrb, points, strides)

    assert detections.shape == (2, _DET_CAP, 6)
    assert (detections[:, :4, 4] > 0.0).all()  # the 4 real anchors have score > 0
    assert torch.equal(detections[:, 4:, :], torch.zeros(2, _DET_CAP - 4, 6))  # padding rows are zero


def test_scores_sorted_descending() -> None:
    """The score column is non-increasing along the detection axis."""
    points, strides = _grid(8, 8, stride=8)  # 64 anchors > k below
    cls_logits = torch.randn(3, 64, 4)
    raw_ltrb = torch.randn(3, 64, 4)

    detections = TopKDecoder(k=16)(cls_logits, raw_ltrb, points, strides)

    scores = detections[..., 4]
    assert detections.shape == (3, 16, 6)
    assert (scores[:, :-1] - scores[:, 1:] >= -1e-6).all()


def test_class_index_of_crafted_spike() -> None:
    """A single confident anchor is decoded to its exact class index at the top."""
    points, strides = _grid(2, 2, stride=8)
    cls_logits = torch.full((1, 4, 7), -8.0)
    cls_logits[0, 2, 5] = 9.0  # anchor 2, class 5 is the clear winner
    raw_ltrb = torch.zeros(1, 4, 4)

    detections = TopKDecoder()(cls_logits, raw_ltrb, points, strides)

    assert int(detections[0, 0, 5]) == 5
    assert detections[0, 0, 4] > 0.99


def test_conf_threshold_zeroes_low_entries_without_reshaping() -> None:
    """A confidence threshold zeroes sub-threshold scores but keeps the shape."""
    points, strides = _grid(2, 2, stride=8)
    cls_logits = torch.full((1, 4, 2), -20.0)  # sigmoid ~ 2e-9
    cls_logits[0, 0, 0] = 5.0  # sigmoid ~ 0.993
    cls_logits[0, 1, 0] = 0.0  # sigmoid = 0.5
    raw_ltrb = torch.zeros(1, 4, 4)

    unfiltered = TopKDecoder(conf_threshold=0.0)(cls_logits, raw_ltrb, points, strides)
    filtered = TopKDecoder(conf_threshold=0.1)(cls_logits, raw_ltrb, points, strides)

    assert filtered.shape == unfiltered.shape == (1, _DET_CAP, 6)
    assert filtered[0, 0, 4] == pytest.approx(float(unfiltered[0, 0, 4]))  # 0.993 kept
    assert filtered[0, 1, 4] == pytest.approx(float(unfiltered[0, 1, 4]))  # 0.5 kept
    assert unfiltered[0, 2, 4] > 0.0  # ~2e-9 present without a threshold
    assert filtered[0, 2, 4] == 0.0  # ~2e-9 zeroed by the threshold
    assert torch.equal(filtered[..., :4], unfiltered[..., :4])  # boxes untouched
    assert torch.equal(filtered[..., 5], unfiltered[..., 5])  # classes untouched


def test_inverse_letterbox_known_geometry() -> None:
    """A box in canvas coords maps back to its known original-image coords."""
    # A 2x4 original letterboxed into a 4x4 canvas gains 1px top/bottom pads:
    # the forward transform maps orig box [0,0,4,2] -> canvas box [0,1,4,3].
    detections = torch.tensor([[[0.0, 1.0, 4.0, 3.0, 0.9, 3.0]]])

    mapped = to_letterboxed_original(detections, orig_size=(2, 4), letterboxed_size=(4, 4))

    assert torch.allclose(mapped[0, 0, :4], torch.tensor([0.0, 0.0, 4.0, 2.0]))
    assert mapped[0, 0, 4] == pytest.approx(0.9)  # score preserved
    assert mapped[0, 0, 5] == pytest.approx(3.0)  # class preserved


def test_inverse_letterbox_round_trip() -> None:
    """Forward-letterboxing then un-letterboxing recovers the original box."""
    orig_box = torch.tensor([[0.5, 0.3, 3.5, 1.7]])
    letterbox = Letterbox(4)
    _, warped = letterbox(torch.zeros(3, 2, 4), Targets(boxes=orig_box, labels=torch.tensor([0])))
    detection = torch.cat((warped.boxes, torch.tensor([[0.8, 2.0]])), dim=1).unsqueeze(0)

    mapped = to_letterboxed_original(detection, orig_size=(2, 4), letterboxed_size=(4, 4))

    assert torch.allclose(mapped[0, 0, :4], orig_box[0], atol=1e-5)


def test_decode_is_deterministic() -> None:
    """Two decodes of the same inputs are bit-for-bit identical."""
    points, strides = _grid(4, 4, stride=8)
    cls_logits = torch.randn(2, 16, 6)
    raw_ltrb = torch.randn(2, 16, 4)
    decoder = TopKDecoder(k=8)

    first = decoder(cls_logits, raw_ltrb, points, strides)
    second = decoder(cls_logits, raw_ltrb, points, strides)

    assert torch.equal(first, second)


def test_module_source_has_no_suppression_or_iou_helper() -> None:
    """The module text references no suppression operator or IoU helper (structural pin)."""
    source = Path(topk_e2e.__file__).read_text(encoding="utf-8").lower()

    assert "nms" not in source
    assert "box_iou" not in source
    assert "torchvision" not in source
