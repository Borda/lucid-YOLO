# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the per-task decode compositions in :mod:`lucid_yolo.decode.per_task`.

:func:`~lucid_yolo.decode.per_task.decode_with_coefficients` and
:func:`~lucid_yolo.decode.per_task.decode_with_keypoints` are what prediction, export and evaluation decode a
segmentation or keypoint branch through. Each is pinned against the composition it replaced — the decoder's own
``decode_with_indices`` plus a bare gather by the reported indices — on the real rows, bit for bit, and on the padding
rows, which the shared gather zeroes rather than reading the last anchor's values. Both decoders are exercised, since
predict reads the dense path through the same function.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from lucid_yolo.assign.grid import make_anchor_points
from lucid_yolo.decode.common import PAD_ANCHOR_INDEX
from lucid_yolo.decode.nms_path import NMSDecoder
from lucid_yolo.decode.per_task import decode_with_coefficients, decode_with_keypoints, gather_keypoint_sets
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.models.heads.keypoint import decode_keypoints

_ANCHORS = 6


def _branch() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return one branch's class logits, ltrb, coefficients and raw points over six anchors.

    Each anchor's coefficients and point offsets are distinct, so a row gathered from the wrong anchor is visible.

    Examples:
        >>> cls_logits, raw_ltrb, coefficients, raw_points = _branch()
        >>> cls_logits.shape, raw_ltrb.shape, coefficients.shape, raw_points.shape
        (torch.Size([2, 6, 2]), torch.Size([2, 6, 4]), torch.Size([2, 6, 3]), torch.Size([2, 6, 4, 2]))
    """
    generator = torch.Generator().manual_seed(0)
    cls_logits = torch.randn(2, _ANCHORS, 2, generator=generator) * 3.0
    raw_ltrb = torch.rand(2, _ANCHORS, 4, generator=generator) + 0.5
    coefficients = torch.arange(2 * _ANCHORS * 3, dtype=torch.float32).reshape(2, _ANCHORS, 3)
    raw_points = torch.randn(2, _ANCHORS, 4, 2, generator=generator)
    return cls_logits, raw_ltrb, coefficients, raw_points


def _bare_gather(values: Tensor, anchors: Tensor) -> Tensor:
    """Gather ``values`` by ``anchors`` with no padding rule — the form the shared gather replaced.

    Examples:
        >>> _bare_gather(torch.arange(6.0).reshape(1, 3, 2), torch.tensor([[2, 0]]))
        tensor([[[4., 5.],
                 [0., 1.]]])
    """
    trailing = values.shape[2:]
    index = anchors.clamp(min=0).reshape(*anchors.shape, *([1] * len(trailing))).expand(-1, -1, *trailing)
    return values.gather(1, index)


@pytest.mark.parametrize(
    "decoder",
    [
        pytest.param(TopKDecoder(k=8), id="topk-padded"),
        pytest.param(TopKDecoder(k=4), id="topk-full"),
        pytest.param(NMSDecoder(conf_threshold=0.2, max_det=8), id="dense-path"),
    ],
)
class TestPerTaskDecode:
    """Each composition equals the decoder plus a bare gather on real rows, and zeroes padding rows."""

    def test_coefficients_follow_own_anchor(self, decoder: nn.Module) -> None:
        """Coefficient row ``n`` is the coefficient vector of the anchor row ``n`` came from."""
        points, strides = make_anchor_points([(2, 3)], [8])
        cls_logits, raw_ltrb, coefficients, _ = _branch()

        detections, anchors, gathered = decode_with_coefficients(
            decoder, cls_logits, raw_ltrb, coefficients, points, strides
        )

        expected_detections, expected_anchors = decoder.decode_with_indices(cls_logits, raw_ltrb, points, strides)
        real = anchors >= 0
        assert torch.equal(detections, expected_detections)
        assert torch.equal(anchors, expected_anchors)
        assert torch.equal(gathered[real], _bare_gather(coefficients, anchors)[real])
        assert torch.equal(gathered[~real], torch.zeros_like(gathered[~real]))

    def test_keypoints_decoded_then_gathered(self, decoder: nn.Module) -> None:
        """Point set ``n`` is the decoded set of the anchor row ``n`` came from; padding rows sit at the origin."""
        points, strides = make_anchor_points([(2, 3)], [8])
        cls_logits, raw_ltrb, _, raw_points = _branch()

        detections, anchors, sets = decode_with_keypoints(decoder, cls_logits, raw_ltrb, raw_points, points, strides)

        dense = decode_keypoints(raw_points, points, strides)
        real = anchors >= 0
        assert torch.equal(detections, decoder.decode_with_indices(cls_logits, raw_ltrb, points, strides)[0])
        assert torch.equal(sets[real], _bare_gather(dense, anchors)[real])
        assert torch.equal(sets[~real], torch.zeros_like(sets[~real]))
        assert torch.equal(sets, gather_keypoint_sets(raw_points, anchors, points, strides))


def test_padding_rows_present_below_k() -> None:
    """The padded fixture above really carries padding rows, so its zero-row assertions are not vacuous."""
    points, strides = make_anchor_points([(2, 3)], [8])
    cls_logits, raw_ltrb, coefficients, _ = _branch()

    _, anchors, _ = decode_with_coefficients(TopKDecoder(k=8), cls_logits, raw_ltrb, coefficients, points, strides)

    assert anchors[:, _ANCHORS:].eq(PAD_ANCHOR_INDEX).all()
