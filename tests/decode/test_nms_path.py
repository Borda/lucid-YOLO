# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-042 confidence-threshold + class-wise NMS decoder.

Covers :class:`lucid_yolo.decode.NMSDecoder`, the non-E2E path over the dense
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

from lucid_yolo.assign.grid import make_anchor_points
from lucid_yolo.decode import NMSDecoder
from lucid_yolo.models.heads.detect import decode_ltrb

_DET_CAP = 300
_BOX_CORNERS = 4


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed the torch RNG so any random inputs are deterministic."""
    torch.manual_seed(0)


def _grid(height: int, width: int, stride: int = 8) -> tuple[Tensor, Tensor]:
    """Return ``(anchor_points, strides)`` for a single ``(height, width)`` level.

    Examples:
        >>> points, strides = _grid(1, 1, stride=8)
        >>> points.tolist()
        [[4.0, 4.0]]
        >>> strides.tolist()
        [8.0]
    """
    return make_anchor_points([(height, width)], [stride])


def _pairwise_iou(box_a: Tensor, box_b: Tensor) -> float:
    """Return the IoU of two ``xyxy`` boxes (test-side helper).

    Examples:
        >>> a = torch.tensor([0.0, 0.0, 10.0, 10.0])
        >>> b = torch.tensor([5.0, 0.0, 15.0, 10.0])  # half-width overlap
        >>> round(_pairwise_iou(a, b), 4)
        0.3333
    """
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
    raw_ltrb = torch.full((1, 4, 4), 0.25)  # 4 px boxes on 8 px centres: real area, still disjoint

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
    raw_ltrb = torch.full((1, 64, 4), 0.25)  # 4 px boxes on 8 px centres: real area, still disjoint

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


def _two_stage_scores() -> Tensor:
    """Two images whose survivors are chosen by the threshold *and* by suppression.

    Six anchors sit in a row 8px apart, so same-class neighbours overlap enough to
    suppress each other at a 0.2 IoU threshold. Image 0 drops anchors 0 and 1
    below the confidence threshold, suppresses anchor 3 against anchor 2, and
    leaves ``[2, 5]``; image 1 leaves ``[4, 1]``, in that score order. Neither
    surviving set is a prefix of the dense anchor order, and neither is a prefix
    of the *thresholded* order, so an index that names a row's position within
    either intermediate set is distinguishable from one that names its anchor.

    Examples:
        >>> _two_stage_scores().shape
        torch.Size([2, 6, 2])
    """
    scores = torch.full((2, 6, 2), -20.0)
    scores[0, 2, 0] = 6.0  # strongest of image 0, class 0
    scores[0, 3, 0] = 4.0  # class 0 again, adjacent to anchor 2 -> suppressed
    scores[0, 5, 1] = 5.0  # class 1 -> survives class-wise suppression
    scores[1, 4, 1] = 6.5  # strongest of image 1
    scores[1, 5, 1] = 3.0  # class 1 again, adjacent to anchor 4 -> suppressed
    scores[1, 1, 0] = 5.5  # class 0, far from anchor 4 -> survives
    return scores


def test_decode_with_indices_names_the_surviving_anchors() -> None:
    """Gathering the dense boxes by the returned indices reproduces the survivors' own corners (WP-053b).

    This path composes two index remappings — the confidence threshold's ``keep``
    mask and then the suppression's ``order`` — and getting either the base or
    the ordering wrong yields indices that are in range, correctly shaped, and
    pointing at the wrong anchors. The mask coefficients of the dense path are
    gathered by exactly these indices, so such an index produces a plausible mask
    of the wrong object beside a correct box and a correct score. Only the
    corner-level correspondence below separates the two: an index left in the
    thresholded subset's own coordinates, or reversed against the detections it
    labels, selects different dense boxes here.
    """
    points, strides = _grid(1, 6, stride=8)  # 6 anchors in a row, centres 8px apart
    cls_logits = _two_stage_scores()
    raw_ltrb = torch.full((2, 6, 4), 1.0)  # every anchor decodes a 16px box around its own centre
    decoder = NMSDecoder(conf_threshold=0.5, iou_threshold=0.2, max_det=4)

    detections, anchors = decoder.decode_with_indices(cls_logits, raw_ltrb, points, strides)

    real = anchors >= 0
    dense_boxes = decode_ltrb(raw_ltrb, points, strides)
    gathered = dense_boxes.gather(1, anchors.clamp(min=0).unsqueeze(-1).expand(-1, -1, _BOX_CORNERS))
    assert torch.equal(detections, decoder(cls_logits, raw_ltrb, points, strides))  # forward is the same decode
    assert anchors[0].tolist() == [2, 5, -1, -1]  # thresholded and suppressed anchors are gone
    assert anchors[1].tolist() == [4, 1, -1, -1]  # a different surviving set in the second image
    assert torch.equal(real, detections[..., 4] > 0.0)  # an index exists exactly where a detection does
    assert torch.equal(detections[..., :_BOX_CORNERS][real], gathered[real])


class TestDegenerateBoxesAreDropped:
    """A row that encloses no region is not a detection and takes no ``max_det`` slot (WP-170).

    ``torchvision.ops.batched_nms`` has no opinion on a box whose ``x2 <= x1``:
    such a box has non-positive area, so its IoU against everything is zero and it
    suppresses nothing and is suppressed by nothing. Three same-class boxes at
    ``iou_threshold=0.7``, two of them inverted descriptions of the one region,
    were therefore all kept. At evaluation an untrained-region anchor cluster
    emitting them consumes the 300-row budget and pushes real detections out of it.

    The drop sits at the confidence threshold rather than inside ``decode_ltrb``:
    that function's values are pinned by the frozen goldens, and this boundary
    reaches the same rows without touching them.
    """

    #: ltrb distances that decode to a real 8 px box on the anchor they ride.
    REAL = (0.5, 0.5, 0.5, 0.5)
    #: The same region written backwards — ``x2`` lands on the far side of ``x1``.
    INVERTED = (-0.5, -0.5, -0.5, -0.5)

    def _decode(self, distances: list[tuple[float, ...]], logits: list[float]) -> Tensor:
        """Decode one image whose anchors carry ``distances`` and ``logits``, one class.

        Every box shares class 0 so class-wise NMS cannot be what keeps a row, and
        the anchors sit on a 1x3 grid so each row rides its own carrier.

        Examples:
            >>> case = TestDegenerateBoxesAreDropped()
            >>> case._decode([case.REAL], [6.0]).shape
            torch.Size([1, 8, 6])
        """
        points, strides = _grid(1, len(distances))
        cls_logits = torch.tensor([[[logit] for logit in logits]])
        raw_ltrb = torch.tensor([list(distances)])
        return NMSDecoder(conf_threshold=0.5, iou_threshold=0.7, max_det=8)(cls_logits, raw_ltrb, points, strides)

    def test_inverted_boxes_no_longer_survive_beside_a_real_one(self) -> None:
        """The finding's own scene: three same-class rows, two inverted, one survivor.

        All three were kept before the drop — the inverted pair suppressed nothing
        and was suppressed by nothing, so each spent a detection slot.
        """
        detections = self._decode([self.REAL, self.INVERTED, self.INVERTED], [6.0, 5.0, 4.0])

        survivors = detections[0, detections[0, :, 4] > 0.0]
        assert survivors.shape[0] == 1
        assert survivors[0, 2] > survivors[0, 0]  # x2 > x1
        assert survivors[0, 3] > survivors[0, 1]  # y2 > y1

    @pytest.mark.parametrize(
        "degenerate",
        [
            pytest.param((0.0, 0.0, 0.0, 0.0), id="zero-area"),
            pytest.param((-0.5, -0.5, -0.5, -0.5), id="inverted-both-axes"),
            pytest.param((-0.5, 0.5, -0.5, 0.5), id="inverted-x-only"),
            pytest.param((0.5, -0.5, 0.5, -0.5), id="inverted-y-only"),
            pytest.param((0.0, 0.5, 0.0, 0.5), id="zero-width"),
            pytest.param((float("nan"),) * 4, id="non-finite"),
            pytest.param((float("inf"),) * 4, id="infinite"),
        ],
    )
    def test_every_degenerate_shape_is_dropped(self, degenerate: tuple[float, ...]) -> None:
        """No non-positive-area row reaches the output, whichever axis collapsed.

        A non-finite row falls out of the same mask for free: it fails both
        strict comparisons rather than needing a rule of its own.
        """
        detections = self._decode([degenerate], [6.0])

        assert torch.equal(detections, torch.zeros_like(detections))

    def test_a_real_detection_beside_them_is_untouched(self) -> None:
        """The drop removes only the degenerate rows, never the honest one it sits beside."""
        detections = self._decode([self.REAL, (0.0, 0.0, 0.0, 0.0)], [6.0, 5.5])

        survivors = detections[0, detections[0, :, 4] > 0.0]
        assert survivors.shape[0] == 1
        assert survivors[0, :4].tolist() == [0.0, 0.0, 8.0, 8.0]

    def test_anchor_indices_stay_aligned_with_the_boxes_they_describe(self) -> None:
        """The index of each survivor still names the anchor its box came from.

        The mask is folded into ``keep`` rather than applied afterwards precisely
        so the boxes and their anchor indices pass through one selection; applying
        it later is how a mask ends up describing another anchor's object.
        """
        points, strides = _grid(1, 3)
        cls_logits = torch.tensor([[[-20.0], [6.0], [5.0]]])
        raw_ltrb = torch.tensor([[list(self.INVERTED), list(self.INVERTED), list(self.REAL)]])

        detections, anchors = NMSDecoder(conf_threshold=0.5, iou_threshold=0.7, max_det=3).decode_with_indices(
            cls_logits, raw_ltrb, points, strides
        )

        survivors = detections[0, detections[0, :, 4] > 0.0]
        assert survivors.shape[0] == 1
        assert int(anchors[0, 0]) == 2  # the only anchor carrying a real box


def test_empty_batch_decodes_to_empty_outputs() -> None:
    """``B = 0`` returns zero-row tensors of the documented widths rather than raising.

    The per-image decode is a list comprehension handed to ``torch.stack``, which has
    no empty case: an empty batch used to die inside stack with a message naming a
    ``TensorList`` rather than the input. Nothing in the return contract excludes
    ``B = 0``, and :class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` already answers it
    with empties, so raising here made this the one path an evaluation loop had to
    special-case for a batch it can legitimately hand either decoder.
    """
    points, strides = make_anchor_points([(1, 2)], [8])
    decoder = NMSDecoder(max_det=4)

    detections, anchors = decoder.decode_with_indices(
        torch.zeros(0, 2, 3), torch.zeros(0, 2, _BOX_CORNERS), points, strides
    )

    assert (detections.shape, anchors.shape, anchors.dtype) == (
        (0, 4, 6),
        (0, 4),
        torch.long,
    )
