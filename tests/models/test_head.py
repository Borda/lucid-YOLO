# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-022 dual detection head (test_head.py).

Covers :class:`lit_yolo.models.DualDetectionHead` and its helpers: that both the
one-to-one and one-to-many branches emit dense ``(B, 8400, nc)`` scores and
``(B, 8400, 4)`` raw ltrb distances at a 640 input; that :func:`o2o_topk` reduces
the one-to-one branch to a valid ``(B, 300, 6)`` detection tuple; that the dense
anchor axis aligns position-for-position with
:func:`~lit_yolo.assign.grid.make_anchor_points`; that :func:`decode_ltrb` matches
a hand-computed box; that the two branches own disjoint parameters and both
receive gradient; and that a small 128-pixel forward runs clean.
"""

from __future__ import annotations

import pytest
import torch

from lit_yolo.assign.grid import make_anchor_points
from lit_yolo.models import DualDetectionHead, DualHeadOutput, decode_ltrb, o2o_topk
from lit_yolo.models.heads.detect import _flatten_level

_STRIDES = [8, 16, 32]
_N_SCALE_CHANNELS = (64, 128, 256)


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed torch RNG so weight init and random inputs are deterministic."""
    torch.manual_seed(0)


def _feature_sizes(input_size: int) -> list[tuple[int, int]]:
    """Return the per-level ``(H, W)`` cell counts for a square input."""
    return [(input_size // stride, input_size // stride) for stride in _STRIDES]


def _make_features(input_size: int, channels: tuple[int, int, int], batch: int = 2) -> tuple[torch.Tensor, ...]:
    """Build random neck-style feature maps at strides 8/16/32."""
    return tuple(
        torch.randn(batch, ch, height, width)
        for ch, (height, width) in zip(channels, _feature_sizes(input_size), strict=True)
    )


def test_dual_head_shapes() -> None:
    """Both branches emit dense (B, 8400, nc) scores and (B, 8400, 4) ltrb boxes."""
    batch, num_classes = 2, 80
    head = DualDetectionHead(in_channels=_N_SCALE_CHANNELS, num_classes=num_classes).eval()
    features = _make_features(640, _N_SCALE_CHANNELS, batch=batch)

    with torch.no_grad():
        out = head(features)

    assert isinstance(out, DualHeadOutput), "forward must return a DualHeadOutput"
    for cls in (out.o2m_cls, out.o2o_cls):
        assert cls.shape == (batch, 8400, num_classes), "dense class logits must be (B, 8400, nc)"
    for box in (out.o2m_box, out.o2o_box):
        assert box.shape == (batch, 8400, 4), "dense ltrb distances must be (B, 8400, 4)"


def test_o2o_topk_detection_tuple() -> None:
    """o2o_topk yields (B, 300, 6) with sigmoid scores in [0, 1] and valid classes."""
    batch, num_classes = 2, 80
    head = DualDetectionHead(in_channels=_N_SCALE_CHANNELS, num_classes=num_classes).eval()
    features = _make_features(640, _N_SCALE_CHANNELS, batch=batch)
    anchor_points, strides = make_anchor_points(_feature_sizes(640), _STRIDES)

    with torch.no_grad():
        out = head(features)
        boxes = decode_ltrb(out.o2o_box, anchor_points, strides)
        detections = o2o_topk(out.o2o_cls, boxes, k=300)

    assert detections.shape == (batch, 300, 6), "o2o_topk must return (B, 300, 6)"
    scores = detections[..., 4]
    assert torch.all((scores >= 0.0) & (scores <= 1.0)), "scores must be sigmoid-bounded to [0, 1]"
    classes = detections[..., 5]
    assert torch.all(classes == classes.floor()), "class ids must be integral-valued floats"
    assert torch.all((classes >= 0) & (classes < num_classes)), "class ids must be valid indices"


def test_o2o_topk_returns_all_when_fewer_than_k() -> None:
    """When the anchor count is below k every anchor is returned."""
    scores = torch.tensor([[[2.0, -1.0], [-3.0, 0.5], [0.1, 0.2]]])  # (1, 3, 2)
    boxes = torch.zeros(1, 3, 4)
    detections = o2o_topk(scores, boxes, k=300)
    assert detections.shape == (1, 3, 6), "keep must clamp to the available anchor count"


def test_anchor_order_agreement() -> None:
    """A planted spike lands at the flattened index make_anchor_points implies."""
    feature_sizes = [(4, 4), (2, 2), (1, 1)]
    level, row, col = 0, 1, 2
    _height, width = feature_sizes[level]

    per_level = [torch.zeros(1, 1, h, w) for h, w in feature_sizes]
    per_level[level][0, 0, row, col] = 1.0
    dense = torch.cat([_flatten_level(m) for m in per_level], dim=1)  # (1, A, 1)

    level_offset = sum(h * w for h, w in feature_sizes[:level])
    expected_index = level_offset + row * width + col
    assert int(dense[0, :, 0].argmax()) == expected_index, "spike must land at the row-major flattened index"

    anchor_points, _ = make_anchor_points(feature_sizes, _STRIDES)
    stride = _STRIDES[level]
    expected_point = torch.tensor([(col + 0.5) * stride, (row + 0.5) * stride])
    assert torch.allclose(anchor_points[expected_index], expected_point), "anchor grid must share the flatten order"


def test_decode_ltrb_hand_case() -> None:
    """decode_ltrb reproduces a hand-computed xyxy box from known ltrb distances."""
    distances = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (1, 1, 4) ltrb
    anchor_points = torch.tensor([[10.0, 10.0]])  # centre (10, 10)
    strides = torch.tensor([2.0])

    boxes = decode_ltrb(distances, anchor_points, strides)

    # x1 = 10 - 1*2, y1 = 10 - 2*2, x2 = 10 + 3*2, y2 = 10 + 4*2
    expected = torch.tensor([[[8.0, 6.0, 16.0, 18.0]]])
    assert torch.allclose(boxes, expected), "decoded corners must match the hand computation"


def test_branch_parameters_disjoint() -> None:
    """The one-to-one and one-to-many branches share no parameters."""
    head = DualDetectionHead(in_channels=(16, 32, 64), num_classes=4)
    o2o_ids = {id(param) for param in head.o2o.parameters()}
    o2m_ids = {id(param) for param in head.o2m.parameters()}

    assert o2o_ids and o2m_ids, "both branches must own parameters"
    assert o2o_ids.isdisjoint(o2m_ids), "branch parameters must be disjoint"


def test_gradient_flows_through_both_branches() -> None:
    """A loss over both branches produces gradient in each branch's stems."""
    head = DualDetectionHead(in_channels=(16, 32, 64), num_classes=4)
    features = _make_features(128, (16, 32, 64), batch=1)

    out = head(features)
    loss = out.o2m_cls.sum() + out.o2m_box.sum() + out.o2o_cls.sum() + out.o2o_box.sum()
    loss.backward()

    o2m_grads = [param.grad for param in head.o2m.parameters() if param.grad is not None]
    o2o_grads = [param.grad for param in head.o2o.parameters() if param.grad is not None]
    assert o2m_grads and any(torch.any(g != 0) for g in o2m_grads), "one-to-many branch must receive gradient"
    assert o2o_grads and any(torch.any(g != 0) for g in o2o_grads), "one-to-one branch must receive gradient"


def test_small_input_forward() -> None:
    """A 128-pixel forward runs clean with A = 16*16 + 8*8 + 4*4 = 336 anchors."""
    head = DualDetectionHead(in_channels=(16, 32, 64), num_classes=4).eval()
    features = _make_features(128, (16, 32, 64), batch=1)

    with torch.no_grad():
        out = head(features)

    assert out.o2o_cls.shape == (1, 336, 4), "small-input dense scores must be (1, 336, nc)"
    assert out.o2o_box.shape == (1, 336, 4), "small-input dense boxes must be (1, 336, 4)"
