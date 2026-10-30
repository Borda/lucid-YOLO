# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-041 score-based top-k end-to-end decoder (test_topk_e2e.py).

Covers :class:`lucid_yolo.decode.TopKDecoder` and :func:`lucid_yolo.decode.to_letterboxed_original`: that the
suppression-free path keeps two heavily overlapping high-score boxes (``test_no_nms_path`` — the defining property, no
IoU, no non-maximum suppression); that the output is a fixed ``(B, 300, 6)`` shape with score-zero padding rows when
fewer anchors exist; that scores come out sorted descending; that a crafted class spike is decoded to the right class
index; that a confidence threshold zeroes low entries without changing the shape; that the letterbox inverse
round-trips box coordinates on a known geometry; and that decoding is deterministic.
:class:`~lucid_yolo.decode.topk_e2e.RotatedTopKDecoder`, the oriented twin, is held to the same fixed ``k`` rows and the
same anchor-index contract. The structural pin -- the module source names no suppression operator or IoU helper -- is
the ``content-pins`` pre-commit hook (``scripts/lint/audit_content_pins.py``).
"""

from __future__ import annotations

import pytest
import torch
from decode_helpers import BOX_CORNERS, DET_CAP, grid, identical_box_pair, pairwise_iou
from torch import Tensor

from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.targets import Targets
from lucid_yolo.decode import RotatedTopKDecoder, TopKDecoder, to_letterboxed_original
from lucid_yolo.decode.common import PAD_ANCHOR_INDEX, RBOX_COLUMNS, gather_by_anchor
from lucid_yolo.models.heads.detect import decode_ltrb
from lucid_yolo.models.heads.obb import RBOX_DET_WIDTH, decode_rboxes, o2o_rotated_topk


def test_no_nms_path() -> None:
    """Two heavily overlapping high-score boxes both survive the decode.

    Anchors 0 and 1 (centres ``(4, 4)`` and ``(12, 4)``, stride 8) are given ltrb distances crafted so they decode to
    the *identical* box ``[-4, -4, 12, 12]``, and both are the confident detections of class 1. A suppression step would
    drop one of the two; the suppression-free path must keep both.
    """
    cls_logits, raw_ltrb, points, strides = identical_box_pair(first_class=1, second_class=1)

    detections = TopKDecoder()(cls_logits, raw_ltrb, points, strides)

    top_two = detections[0, :2]
    expected_box = torch.tensor([-4.0, -4.0, 12.0, 12.0])
    assert torch.allclose(top_two[0, :4], expected_box)
    assert torch.allclose(top_two[1, :4], expected_box)
    assert pairwise_iou(top_two[0, :4], top_two[1, :4]) == pytest.approx(1.0)
    assert (top_two[:, 4] > 0.99).all()  # both high confidence
    assert torch.equal(top_two[:, 5], torch.tensor([1.0, 1.0]))  # both class 1


def test_fixed_shape_with_zero_padding() -> None:
    """Output is a fixed ``(B, 300, 6)`` with score-zero padding rows below k.

    Four anchors are fewer than the default cap of 300, so 296 rows per image are padding. Random logits give every real
    anchor a positive sigmoid score, and the padding rows must be exact zeros rather than leftover box values, so a
    consumer can tell them apart by score alone.
    """
    points, strides = grid(2, 2, stride=8)  # only 4 anchors < 300
    cls_logits = torch.randn(2, 4, 5)
    raw_ltrb = torch.randn(2, 4, 4)

    detections = TopKDecoder()(cls_logits, raw_ltrb, points, strides)

    assert detections.shape == (2, DET_CAP, 6)
    assert (detections[:, :4, 4] > 0.0).all()  # the 4 real anchors have score > 0
    assert torch.equal(detections[:, 4:, :], torch.zeros(2, DET_CAP - 4, 6))  # padding rows are zero


def test_scores_sorted_descending() -> None:
    """The score column is non-increasing along the detection axis.

    Random logits over 64 anchors with ``k=16`` mean the ranking must both pick the 16 best and order them best first
    in every image; a missing sort would still return the right shape.
    """
    points, strides = grid(8, 8, stride=8)  # 64 anchors > k below
    cls_logits = torch.randn(3, 64, 4)
    raw_ltrb = torch.randn(3, 64, 4)

    detections = TopKDecoder(k=16)(cls_logits, raw_ltrb, points, strides)

    scores = detections[..., 4]
    assert detections.shape == (3, 16, 6)
    assert (scores[:, :-1] - scores[:, 1:] >= -1e-6).all()


def test_class_index_of_crafted_spike() -> None:
    """A single confident anchor is decoded to its exact class index at the top.

    Anchor 2 gets one class-5 logit of 9 against -8 everywhere else, so the top row must report class 5, not the anchor
    index or a neighbouring class.
    """
    points, strides = grid(2, 2, stride=8)
    cls_logits = torch.full((1, 4, 7), -8.0)
    cls_logits[0, 2, 5] = 9.0  # anchor 2, class 5 is the clear winner
    raw_ltrb = torch.zeros(1, 4, 4)

    detections = TopKDecoder()(cls_logits, raw_ltrb, points, strides)

    assert int(detections[0, 0, 5]) == 5
    assert detections[0, 0, 4] > 0.99


def test_conf_threshold_zeroes_low_entries_without_reshaping() -> None:
    """A confidence threshold zeroes sub-threshold scores but keeps the shape.

    The logits give sigmoid scores of about 0.993, 0.5 and 2e-9; at a 0.1 threshold the first two are kept and the last
    is zeroed, while boxes and classes stay exactly as the unfiltered decode returned them.
    """
    points, strides = grid(2, 2, stride=8)
    cls_logits = torch.full((1, 4, 2), -20.0)  # sigmoid ~ 2e-9
    cls_logits[0, 0, 0] = 5.0  # sigmoid ~ 0.993
    cls_logits[0, 1, 0] = 0.0  # sigmoid = 0.5
    raw_ltrb = torch.zeros(1, 4, 4)

    unfiltered = TopKDecoder(conf_threshold=0.0)(cls_logits, raw_ltrb, points, strides)
    filtered = TopKDecoder(conf_threshold=0.1)(cls_logits, raw_ltrb, points, strides)

    assert filtered.shape == unfiltered.shape == (1, DET_CAP, 6)
    assert filtered[0, 0, 4] == pytest.approx(float(unfiltered[0, 0, 4]))  # 0.993 kept
    assert filtered[0, 1, 4] == pytest.approx(float(unfiltered[0, 1, 4]))  # 0.5 kept
    assert unfiltered[0, 2, 4] > 0.0  # ~2e-9 present without a threshold
    assert filtered[0, 2, 4] == 0.0  # ~2e-9 zeroed by the threshold
    assert torch.equal(filtered[..., :4], unfiltered[..., :4])  # boxes untouched
    assert torch.equal(filtered[..., 5], unfiltered[..., 5])  # classes untouched


def test_scored_rows_leading_run() -> None:
    """Rows with a live score are always a prefix, whatever zeroed them.

    Three separate mechanisms can leave a zero score in the output, and all three are suffix operations: the ranking is
    score-descending, ``pad_detections`` appends its zero rows after the ranked ones, and a confidence threshold cuts a
    descending column from some point on. A consumer may therefore replace the boolean score mask with a count and a
    slice, which is what :meth:`~lucid_yolo.ptl.module.DetectionLitModule._update_val_segm` does to avoid copying a
    whole mask stack (WP-164). Nothing else states the property, so a reordering of the decode would otherwise break
    that caller silently.
    """
    torch.manual_seed(0)
    points, strides = grid(4, 4, stride=8)  # 16 anchors
    cls_logits = torch.randn(3, points.shape[0], 2)
    raw_ltrb = torch.rand(3, points.shape[0], 4)
    # k above the anchor count, so zero-row padding and the threshold both apply.
    decoder = TopKDecoder(k=24, conf_threshold=0.5)

    detections = decoder(cls_logits, raw_ltrb, points, strides)

    for image in detections:
        live = image[:, 4] > 0.0
        assert 0 < int(live.sum()) < live.shape[0]  # neither all-live nor all-zero
        assert torch.equal(live.nonzero().flatten(), torch.arange(int(live.sum())))


def test_inverse_letterbox_known_geometry() -> None:
    """A box in canvas coords maps back to its known original-image coords.

    A 2x4 image in a 4x4 canvas gains one pixel of padding top and bottom, so the canvas box ``[0, 1, 4, 3]`` must map
    back to ``[0, 0, 4, 2]`` with the score and class columns carried through untouched.
    """
    # A 2x4 original letterboxed into a 4x4 canvas gains 1px top/bottom pads:
    # the forward transform maps orig box [0,0,4,2] -> canvas box [0,1,4,3].
    detections = torch.tensor([[[0.0, 1.0, 4.0, 3.0, 0.9, 3.0]]])

    mapped = to_letterboxed_original(detections, orig_size=(2, 4), letterboxed_size=(4, 4))

    assert torch.allclose(mapped[0, 0, :4], torch.tensor([0.0, 0.0, 4.0, 2.0]))
    assert mapped[0, 0, 4] == pytest.approx(0.9)  # score preserved
    assert mapped[0, 0, 5] == pytest.approx(3.0)  # class preserved


def test_inverse_letterbox_round_trip() -> None:
    """Forward-letterboxing then un-letterboxing recovers the original box.

    The forward step is the real ``Letterbox`` transform rather than hand-computed numbers, so a disagreement between
    the forward and inverse conventions shows even when each looks plausible on its own.
    """
    orig_box = torch.tensor([[0.5, 0.3, 3.5, 1.7]])
    letterbox = Letterbox(4)
    _, warped = letterbox(torch.zeros(3, 2, 4), Targets(boxes=orig_box, labels=torch.tensor([0])))
    detection = torch.cat((warped.boxes, torch.tensor([[0.8, 2.0]])), dim=1).unsqueeze(0)

    mapped = to_letterboxed_original(detection, orig_size=(2, 4), letterboxed_size=(4, 4))

    assert torch.allclose(mapped[0, 0, :4], orig_box[0], atol=1e-5)


def test_decode_deterministic() -> None:
    """Two decodes of the same inputs are bit-for-bit identical.

    Random logits and distances on a 4x4 grid for a batch of two go through the whole ranking and padding chain, so any
    run-to-run variation in it shows up as a differing tensor.
    """
    points, strides = grid(4, 4, stride=8)
    cls_logits = torch.randn(2, 16, 6)
    raw_ltrb = torch.randn(2, 16, 4)
    decoder = TopKDecoder(k=8)

    first = decoder(cls_logits, raw_ltrb, points, strides)
    second = decoder(cls_logits, raw_ltrb, points, strides)

    assert torch.equal(first, second)


def _ranking_scores() -> Tensor:
    """Two images of six per-anchor class-0 logits whose score order is no prefix of anchor order.

    Image 0 ranks ``[2, 5, 0, 4, 1, 3]`` and image 1 ``[3, 1, 5, 2, 0, 4]``; the two differ, so a per-image index base
    that leaked across the batch shows up.

    Examples:
        >>> scores = _ranking_scores()
        >>> scores.shape
        torch.Size([2, 6, 2])
        >>> scores[0, :, 0].argsort(descending=True).tolist()  # image 0's anchor rank
        [2, 5, 0, 4, 1, 3]
    """
    class_zero = torch.tensor([[1.0, -3.0, 5.0, -5.0, 0.0, 3.0], [-1.0, 4.0, 0.5, 6.0, -2.0, 2.0]])
    scores = torch.full((2, 6, 2), -20.0)
    scores[..., 0] = class_zero
    return scores


def test_decode_with_indices_names_ranked_anchors() -> None:
    """Gathering the dense boxes by the returned indices reproduces the detections' own corners (WP-053b).

    The segmentation decode reads its mask coefficients from the dense per-anchor map by these indices, so an index that
    ranked the scores a second time, that reported a row's rank instead of its anchor, or that let the two images share
    a base pairs a correct box with another anchor's coefficients — a plausible mask of the wrong object at a correct
    box with a correct score. Every such index is in range and correctly shaped, so only the corner-level correspondence
    asserted here separates it from a right one. The kept set is deliberately no prefix of the dense order, which a
    wrong base would satisfy by accident.
    """
    points, strides = grid(2, 3, stride=8)  # 6 anchors, no two boxes alike
    cls_logits = _ranking_scores()
    raw_ltrb = torch.full((2, 6, 4), 1.0)  # each anchor decodes a 16px box around its own centre
    decoder = TopKDecoder(k=8)  # above the anchor count, so padding rows are covered too

    detections, anchors = decoder.decode_with_indices(cls_logits, raw_ltrb, points, strides)

    real = anchors >= 0
    dense_boxes = decode_ltrb(raw_ltrb, points, strides)
    gathered = dense_boxes.gather(1, anchors.clamp(min=0).unsqueeze(-1).expand(-1, -1, BOX_CORNERS))
    assert torch.equal(detections, decoder(cls_logits, raw_ltrb, points, strides))  # forward is the same decode
    assert anchors[0, :6].tolist() == [2, 5, 0, 4, 1, 3]  # score order, not anchor order
    assert anchors[1, :6].tolist() == [3, 1, 5, 2, 0, 4]  # and a different order in the second image
    assert real.tolist() == [[True] * 6 + [False] * 2] * 2  # the two padding rows name no anchor
    assert torch.equal(detections[..., :BOX_CORNERS][real], gathered[real])


def _oriented_branch() -> tuple[Tensor, Tensor, Tensor]:
    """Two images of six-anchor one-to-one outputs whose rotated boxes all differ.

    The class logits are :func:`_ranking_scores`'; each anchor's ltrb decodes a 24x8 box around its own centre, and
    every anchor of every image carries its own raw angle, so a row paired with the wrong anchor's box or heading is
    visible.

    Examples:
        >>> cls_logits, raw_ltrb, raw_angles = _oriented_branch()
        >>> cls_logits.shape, raw_ltrb.shape, raw_angles.shape
        (torch.Size([2, 6, 2]), torch.Size([2, 6, 4]), torch.Size([2, 6, 1]))
    """
    raw_ltrb = torch.tensor([1.0, 0.5, 2.0, 0.5]).expand(2, 6, 4).clone()
    raw_angles = torch.linspace(-1.2, 1.3, 12).reshape(2, 6, 1)
    return _ranking_scores(), raw_ltrb, raw_angles


class TestRotatedTopKDecoder:
    """Tests for :class:`~lucid_yolo.decode.topk_e2e.RotatedTopKDecoder` (CS-10).

    The oriented E2E decode used to be the bare ranking, which returns ``min(k, A)`` rows and no anchor indices; the
    decoder pads to ``k`` and reports the indices, exactly as :class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` does for
    the axis-aligned path.
    """

    def test_small_canvas_padded_to_k(self) -> None:
        """Six anchors under ``k = 8`` give eight rows, the last two all-zero and anchorless."""
        points, strides = grid(2, 3, stride=8)
        detections, anchors = RotatedTopKDecoder(k=8).decode_with_indices(*_oriented_branch(), points, strides)

        assert detections.shape == (2, 8, RBOX_DET_WIDTH)
        assert anchors.shape == (2, 8)
        assert torch.equal(detections[:, 6:], torch.zeros(2, 2, RBOX_DET_WIDTH))
        assert anchors[:, 6:].eq(PAD_ANCHOR_INDEX).all()
        assert (anchors[:, :6] >= 0).all()

    def test_leading_rows_match_unpadded_ranking(self) -> None:
        """Below ``k`` anchors, the real rows are the bare ranking's own, bit for bit."""
        points, strides = grid(2, 3, stride=8)
        cls_logits, raw_ltrb, raw_angles = _oriented_branch()
        unpadded = o2o_rotated_topk(cls_logits, decode_rboxes(raw_ltrb, raw_angles, points, strides), k=8)

        detections = RotatedTopKDecoder(k=8)(cls_logits, raw_ltrb, raw_angles, points, strides)

        assert unpadded.shape[1] == 6
        assert torch.equal(detections[:, :6], unpadded)

    def test_equals_unpadded_ranking_when_anchors_reach_k(self) -> None:
        """With at least ``k`` anchors nothing is padded and the output is the bare ranking's."""
        points, strides = grid(2, 3, stride=8)
        cls_logits, raw_ltrb, raw_angles = _oriented_branch()
        unpadded = o2o_rotated_topk(cls_logits, decode_rboxes(raw_ltrb, raw_angles, points, strides), k=4)

        detections, anchors = RotatedTopKDecoder(k=4).decode_with_indices(
            cls_logits, raw_ltrb, raw_angles, points, strides
        )

        assert torch.equal(detections, unpadded)
        assert (anchors >= 0).all()

    def test_indices_gather_same_rows(self) -> None:
        """Gathering the dense rotated boxes by the reported indices reproduces each row's box and heading."""
        points, strides = grid(2, 3, stride=8)
        cls_logits, raw_ltrb, raw_angles = _oriented_branch()
        decoder = RotatedTopKDecoder(k=8)

        detections, anchors = decoder.decode_with_indices(cls_logits, raw_ltrb, raw_angles, points, strides)

        dense = decode_rboxes(raw_ltrb, raw_angles, points, strides)
        real = anchors >= 0
        assert anchors[0, :6].tolist() == [2, 5, 0, 4, 1, 3]  # score order, not anchor order
        assert anchors[1, :6].tolist() == [3, 1, 5, 2, 0, 4]
        assert torch.equal(detections[..., :RBOX_COLUMNS][real], gather_by_anchor(dense, anchors)[real])
        assert torch.equal(detections, decoder(cls_logits, raw_ltrb, raw_angles, points, strides))
