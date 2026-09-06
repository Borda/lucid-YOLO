# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-091b rotated suppression decoder.

Covers :class:`lucid_yolo.decode.RotatedNMSDecoder`, the oriented non-E2E path over the
dense one-to-many branch. The gate this suite exists for is the first one: two 20x2 bars
crossing at one centre have **identical** axis-aligned envelopes, so any decoder that
suppresses by upright overlap sees an IoU of exactly 1.0 and deletes one of two objects
that barely touch — their rotated IoU is 0.053. That scene is asserted with both numbers
computed in the test, so it stays self-certifying: a later edit that moved the geometry
somewhere less discriminating would fail on the premise rather than pass on the
conclusion. The claim that an envelope-suppressing decoder *fails* it is executable rather
than argued: :class:`~lucid_yolo.decode.nms_path.NMSDecoder`, handed the identical logits
and distances and simply never shown the angles, answers with one detection where this one
answers with two.

The rest pins what the axis-aligned twin already promises and this one must match: a
same-class duplicate collapses to one survivor while an overlapping *different*-class pair
does not, the threshold is genuinely consulted rather than inlined at either extreme, the
output is the fixed ``(B, max_det, 7)`` A45 batch with score-zero padding and a
non-increasing score column, survivors are canonical (A23, inherited from
:func:`~lucid_yolo.models.heads.obb.decode_rboxes`), each image of a batch is decoded
independently, and the decode is deterministic.

No trained weights anywhere: every scene is planted straight into the head's raw outputs,
the way :mod:`tests.predict.planted` plants its rotated box — through the axis-aligned
envelope, because that is the shape the head actually regresses (A44), with the raw angle
attached beside it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

from lucid_yolo.assign.grid import make_anchor_points
from lucid_yolo.decode import ROTATED_NMS_IOU_THRESHOLD, NMSDecoder, RotatedNMSDecoder
from lucid_yolo.eval.dota_eval import rotated_iou

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

#: One anchor level of four anchors at stride 8 — enough carriers for every scene here,
#: and small enough that a failure prints a readable tensor.
_GRID = (2, 2)
_STRIDE = 8

#: Class count of the planted logits. Three, so a "different class" case has somewhere to
#: go that is neither the planted label nor the argmax of an all-absent row.
_NUM_CLASSES = 3

#: Logit of a class the planted anchor does not have. ``sigmoid(-20)`` is 2e-9, below any
#: confidence threshold a test here sets, so an unplanted anchor never reaches suppression.
_ABSENT_LOGIT = -20.0

#: The canonical angle range (A23), written as the assertions read it rather than imported
#: from :mod:`lucid_yolo.data.rotated_geom`, which would agree with itself if both were wrong.
_THETA_LOW = -math.pi / 4
_THETA_HIGH = 3 * math.pi / 4

#: A thin bar: long enough that a heading change moves its overlap a long way, which is
#: what separates the rotated measure from the upright one.
_BAR_EXTENTS = (20.0, 2.0)

#: Centre every scene shares, in canvas pixels. Away from the anchor points on purpose —
#: the anchor is a carrier for the plant, not part of the geometry under test.
_CENTRE = (32.0, 32.0)


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed the torch RNG so any random input would be deterministic."""
    torch.manual_seed(0)


def _ltrb_from_envelope(envelope: tuple[float, float, float, float], point: Tensor, stride: Tensor) -> Tensor:
    """Return the raw ltrb distances that decode to ``envelope`` from ``point``.

    Examples:
        >>> _ltrb_from_envelope((0.0, 0.0, 8.0, 8.0), torch.tensor([4.0, 4.0]), torch.tensor(4.0)).tolist()
        [1.0, 1.0, 1.0, 1.0]
    """
    x1, y1, x2, y2 = envelope
    return torch.tensor(
        [
            (float(point[0]) - x1) / float(stride),
            (float(point[1]) - y1) / float(stride),
            (x2 - float(point[0])) / float(stride),
            (y2 - float(point[1])) / float(stride),
        ]
    )


def _plant(
    rboxes: Sequence[tuple[float, float, float, float, float]],
    logits: Sequence[float],
    labels: Sequence[int],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Plant rotated boxes into one image's raw head outputs.

    Box ``i`` rides anchor ``i`` — chosen by position in the list rather than by nearest
    anchor, because several scenes here place two boxes at the same centre and a
    nearest-anchor rule would put them on one carrier. The geometry lives entirely in the
    ltrb distances and the angle, both of which decode back exactly.

    Args:
        rboxes: ``(cx, cy, w, h, theta)`` per box, ``theta`` raw as R1 Eq. 13 emits it.
        logits: The class logit each box's label carries; distinct values keep the score
            order unambiguous.
        labels: The class index of each box.

    Returns:
        ``(cls_logits, raw_ltrb, angles, anchor_points, strides)`` for a batch of one.

    Examples:
        >>> cls_logits, raw_ltrb, angles, points, strides = _plant([(32.0, 32.0, 20.0, 2.0, 0.7)], [6.0], [1])
        >>> cls_logits.shape
        torch.Size([1, 4, 3])
        >>> round(float(angles[0, 0, 0]), 4)
        0.7
    """
    points, strides = make_anchor_points([_GRID], [_STRIDE])
    anchors = points.shape[0]
    cls_logits = torch.full((1, anchors, _NUM_CLASSES), _ABSENT_LOGIT)
    raw_ltrb = torch.zeros(1, anchors, 4)
    angles = torch.zeros(1, anchors, 1)
    for index, (rbox, logit, label) in enumerate(zip(rboxes, logits, labels, strict=True)):
        centre_x, centre_y, width, height, theta = rbox
        envelope = (centre_x - width / 2, centre_y - height / 2, centre_x + width / 2, centre_y + height / 2)
        cls_logits[0, index, label] = logit
        raw_ltrb[0, index] = _ltrb_from_envelope(envelope, points[index], strides[index])
        angles[0, index, 0] = theta
    return cls_logits, raw_ltrb, angles, points, strides


def _envelope_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    """Return the IoU of two rotated boxes' axis-aligned envelopes — what upright NMS sees.

    Examples:
        >>> _envelope_iou((0.0, 0.0, 4.0, 4.0, 0.0), (0.0, 0.0, 4.0, 4.0, 0.0))
        1.0
    """
    corners = []
    for centre_x, centre_y, width, height, theta in (first, second):
        half_w = (width * abs(math.cos(theta)) + height * abs(math.sin(theta))) / 2
        half_h = (width * abs(math.sin(theta)) + height * abs(math.cos(theta))) / 2
        corners.append((centre_x - half_w, centre_y - half_h, centre_x + half_w, centre_y + half_h))
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = corners
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - overlap
    return overlap / union


def _survivors(detections: Tensor) -> Tensor:
    """Return one image's rows with a non-zero score, dropping the padding.

    Examples:
        >>> detections = torch.zeros(1, 3, 7)
        >>> detections[0, 0, 5] = 0.9
        >>> _survivors(detections).shape
        torch.Size([1, 7])
    """
    return detections[0, detections[0, :, 5] > 0.0]


class TestRotatedSuppression:
    """The suppression rule itself: what it compares, and what that decides."""

    def test_boxes_overlapping_only_as_upright_rectangles_both_survive(self) -> None:
        """Two bars crossing at one centre both survive, because rotated overlap is what decides.

        The work package's reason for existing. Two 20x2 bars through canvas ``(32, 32)``
        at ``+0.7`` and ``-0.7`` radians have the *same* axis-aligned envelope, so an
        upright-overlap decoder scores them at IoU 1.0 and deletes the lower-scoring one;
        their actual rotated overlap is a small rhombus at the crossing, 0.053. Both
        premises are computed here rather than asserted from memory, so the scene cannot
        quietly stop discriminating: if a later edit made the envelopes disagree or the
        rotated overlap large, this fails on the premise instead of passing.
        """
        first = (*_CENTRE, *_BAR_EXTENTS, 0.7)
        second = (*_CENTRE, *_BAR_EXTENTS, -0.7)
        assert _envelope_iou(first, second) > ROTATED_NMS_IOU_THRESHOLD
        assert float(rotated_iou(torch.tensor([first]), torch.tensor([second]))) < ROTATED_NMS_IOU_THRESHOLD

        detections = RotatedNMSDecoder(max_det=8)(*_plant([first, second], [6.0, 5.0], [1, 1]))

        survivors = _survivors(detections)
        assert survivors.shape[0] == 2
        assert survivors[:, 4].tolist() == pytest.approx([0.7, -0.7])

    def test_the_axis_aligned_decoder_deletes_one_of_that_crossing_pair(self) -> None:
        """The upright decoder, on the very same plant, keeps one box where this one keeps two.

        The contrast that makes the previous test's claim executable rather than argued.
        :class:`~lucid_yolo.decode.nms_path.NMSDecoder` is fed the identical logits and
        distances — it simply never sees the angles — and answers with a single detection.
        That is the WP-091 refusal's whole content: not that the axis-aligned decoder
        errors on oriented input, but that it succeeds, plausibly, on the wrong geometry.
        A future edit routing the oriented ``nms`` flag back to it turns this red.
        """
        planted = _plant([(*_CENTRE, *_BAR_EXTENTS, 0.7), (*_CENTRE, *_BAR_EXTENTS, -0.7)], [6.0, 5.0], [1, 1])
        cls_logits, raw_ltrb, angles, points, strides = planted

        upright = NMSDecoder(max_det=8)(cls_logits, raw_ltrb, points, strides)
        rotated = RotatedNMSDecoder(max_det=8)(cls_logits, raw_ltrb, angles, points, strides)

        assert int((upright[0, :, 4] > 0.0).sum()) == 1
        assert int((rotated[0, :, 5] > 0.0).sum()) == 2

    def test_a_same_class_duplicate_collapses_to_one(self) -> None:
        """Two identical rotated boxes of one class leave exactly one survivor, the higher-scoring.

        The suppression half of the contract, and the mirror of the axis-aligned path's
        ``test_overlapping_same_class_collapses_to_one``: the E2E oriented path returns
        both of these, and this decoder exists precisely to be the column where it does
        not. Identical geometry makes the overlap exactly 1.0, so the case turns on the
        rule rather than on where the threshold sits.
        """
        bar = (*_CENTRE, *_BAR_EXTENTS, 0.7)

        detections = RotatedNMSDecoder(max_det=8)(*_plant([bar, bar], [6.0, 5.0], [1, 1]))

        survivors = _survivors(detections)
        assert survivors.shape[0] == 1
        assert int(survivors[0, 6]) == 1
        assert float(survivors[0, 5]) == pytest.approx(torch.sigmoid(torch.tensor(6.0)).item())

    def test_an_overlapping_pair_of_different_classes_both_survive(self) -> None:
        """The same identical-box pair survives intact when the two carry different labels.

        Class-wise suppression, on the geometry that collapses within one class — so the
        pair isolates the label test from the overlap test. A decoder that suppressed
        globally would answer with one box here and look correct on every other scene in
        this file.
        """
        bar = (*_CENTRE, *_BAR_EXTENTS, 0.7)

        detections = RotatedNMSDecoder(max_det=8)(*_plant([bar, bar], [6.0, 5.0], [1, 2]))

        survivors = _survivors(detections)
        assert survivors.shape[0] == 2
        assert [int(row) for row in survivors[:, 6]] == [1, 2]

    @pytest.mark.parametrize(
        ("iou_threshold", "expected"),
        [pytest.param(0.7, 2, id="above-the-overlap"), pytest.param(0.5, 1, id="below-the-overlap")],
    )
    def test_the_threshold_decides_a_partial_overlap(self, iou_threshold: float, expected: int) -> None:
        """A pair overlapping at 0.667 survives at threshold 0.7 and collapses at 0.5.

        Proof that the threshold is consulted rather than effectively hard-coded at an
        extreme. The two boxes share a heading and are offset along their own long axis by
        a fifth of it, which puts their rotated IoU at 0.667 — between the two thresholds,
        so each answer is available and the parameter is what picks.
        """
        heading = 0.7
        shift = 0.2 * _BAR_EXTENTS[0]
        first = (*_CENTRE, 20.0, 4.0, heading)
        second = (_CENTRE[0] + shift * math.cos(heading), _CENTRE[1] + shift * math.sin(heading), 20.0, 4.0, heading)
        assert float(rotated_iou(torch.tensor([first]), torch.tensor([second]))) == pytest.approx(2 / 3, abs=1e-3)

        decoder = RotatedNMSDecoder(iou_threshold=iou_threshold, max_det=8)
        detections = decoder(*_plant([first, second], [6.0, 5.0], [1, 1]))

        assert _survivors(detections).shape[0] == expected


class TestOutputContract:
    """The A45 batch shape, ordering and padding this decoder promises its consumers."""

    def test_the_batch_is_the_fixed_size_a45_shape_with_zero_padding(self) -> None:
        """Output is ``(B, max_det, 7)`` with score-zero padding rows after the survivors.

        The contract every consumer downstream relies on:
        :func:`~lucid_yolo.eval.dota_eval.rotated_detections_to_predictions` drops rows by
        score alone and would read a short or ragged batch as a different answer. Two
        survivors out of a five-row cap leave three all-zero rows.
        """
        bar = (*_CENTRE, *_BAR_EXTENTS, 0.7)
        crossing = (*_CENTRE, *_BAR_EXTENTS, -0.7)

        detections = RotatedNMSDecoder(max_det=5)(*_plant([bar, crossing], [6.0, 5.0], [1, 1]))

        assert detections.shape == (1, 5, 7)
        assert torch.equal(detections[0, 2:], torch.zeros(3, 7))

    def test_the_score_column_is_non_increasing(self) -> None:
        """Survivors come out in descending score order, padding included.

        Greedy suppression keeps boxes in the order it visits them, which is the score
        order, and the padding rows carry score zero — so the whole column is
        non-increasing and a consumer may stop at the first zero.
        """
        bar = (*_CENTRE, *_BAR_EXTENTS, 0.7)
        crossing = (*_CENTRE, *_BAR_EXTENTS, -0.7)

        detections = RotatedNMSDecoder(max_det=5)(*_plant([bar, crossing], [3.0, 6.0], [1, 1]))

        scores = detections[0, :, 5]
        assert torch.all(scores[:-1] >= scores[1:])

    def test_nothing_above_the_threshold_is_all_padding(self) -> None:
        """An image whose every anchor is below the confidence threshold decodes to zeros.

        The degenerate input the fixed-size contract exists to absorb. Suppression never
        runs, and the answer is still a full ``(1, 4, 7)`` batch rather than an empty
        tensor or an exception — a caller stacking per-image results needs the rank.
        """
        bar = (*_CENTRE, *_BAR_EXTENTS, 0.7)

        detections = RotatedNMSDecoder(conf_threshold=0.999, max_det=4)(*_plant([bar], [6.0], [1]))

        assert detections.shape == (1, 4, 7)
        assert torch.equal(detections, torch.zeros(1, 4, 7))

    def test_each_image_of_a_batch_is_decoded_on_its_own(self) -> None:
        """Two images decode independently: a duplicate in one does not suppress the other's box.

        Suppression is per image by definition, and the batch axis is the one place that
        is easy to lose — a flattened implementation would compare detections across
        images and silently delete objects that share a coordinate, which is common when
        every image is a 1024 px tile on one grid.
        """
        bar = (*_CENTRE, *_BAR_EXTENTS, 0.7)
        duplicate_pair = _plant([bar, bar], [6.0, 5.0], [1, 1])
        single = _plant([bar], [6.0], [1])
        batched = tuple(torch.cat((left, right)) for left, right in zip(duplicate_pair[:3], single[:3], strict=True))

        detections = RotatedNMSDecoder(max_det=8)(*batched, duplicate_pair[3], duplicate_pair[4])

        assert detections.shape == (2, 8, 7)
        assert int((detections[0, :, 5] > 0.0).sum()) == 1
        assert int((detections[1, :, 5] > 0.0).sum()) == 1

    def test_survivors_are_canonical_however_the_head_stated_them(self) -> None:
        """A wild raw angle and short-edge-first extents both come out canonical (A23).

        The decoder inherits canonicalization from
        :func:`~lucid_yolo.models.heads.obb.decode_rboxes` rather than repeating it, and
        that inheritance is worth pinning here because suppression *depends* on it: two
        boxes described a half turn apart are one rectangle, and a path that compared raw
        angles would treat them as two objects and keep both.
        """
        wild = (*_CENTRE, *_BAR_EXTENTS, 0.7 + math.pi)
        short_edge_first = (*_CENTRE, _BAR_EXTENTS[1], _BAR_EXTENTS[0], 0.0)

        detections = RotatedNMSDecoder(max_det=8)(*_plant([wild, short_edge_first], [6.0, 5.0], [1, 2]))

        survivors = _survivors(detections)
        assert survivors.shape[0] == 2
        assert torch.all(survivors[:, 2] >= survivors[:, 3])  # long edge first
        assert torch.all((survivors[:, 4] >= _THETA_LOW) & (survivors[:, 4] < _THETA_HIGH))

    def test_decoding_is_deterministic(self) -> None:
        """The same planted scene decodes to the same tensor twice, bit for bit.

        Greedy suppression is order-sensitive, and the order comes from a sort over scores
        that a scene may tie. The sort is stable for that reason; this is what would catch
        it becoming unstable.
        """
        planted = _plant(
            [(*_CENTRE, *_BAR_EXTENTS, 0.7), (*_CENTRE, *_BAR_EXTENTS, -0.7), (*_CENTRE, *_BAR_EXTENTS, 0.0)],
            [6.0, 6.0, 6.0],
            [1, 1, 1],
        )
        decoder = RotatedNMSDecoder(max_det=8)

        assert torch.equal(decoder(*planted), decoder(*planted))


class TestRegisteredThreshold:
    """Where the suppression threshold comes from."""

    def test_the_default_is_the_registered_constant(self) -> None:
        """The decoder's default IoU threshold is the module constant, not a repeated literal.

        The work package's own definition of done: the threshold is an assumption on the
        register (A61), so the value exists once and the decoder reads it. A literal at the
        constructor would leave the register describing a number the code no longer uses,
        and nothing would report the divergence.
        """
        assert RotatedNMSDecoder().iou_threshold == ROTATED_NMS_IOU_THRESHOLD


class TestNonFiniteRowsAreDropped:
    """A ``NaN`` or ``Inf`` box never reaches the output as a detection (WP-170).

    Suppression cannot remove such a row and never could: ``rotated_iou`` scores
    every pair involving a non-finite box ``0.0`` — every comparison against
    ``NaN`` is false, so the union guard takes the zero branch — and a zero
    overlap clears no threshold. The row therefore survived every round of
    ``_suppress`` and was emitted with a real score beside the honest detections,
    which is why the drop belongs at the confidence threshold, the one boundary
    that sees it as a row rather than as an overlap.
    """

    #: A well-formed bar, planted alongside each poisoned row as the control.
    VALID_BOX = (32.0, 32.0, 20.0, 4.0, 0.3)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_angle_is_not_emitted(self, bad: float) -> None:
        """A box whose ``theta`` is non-finite is dropped, and the valid box beside it stays.

        The realistic entry path: an angle head that has diverged emits ``NaN``
        for one anchor while the rest of the image decodes normally.
        """
        poisoned = (48.0, 48.0, 16.0, 4.0, bad)
        cls_logits, raw_ltrb, angles, points, strides = _plant([self.VALID_BOX, poisoned], [6.0, 5.0], [0, 1])

        detections = RotatedNMSDecoder(conf_threshold=0.5)(cls_logits, raw_ltrb, angles, points, strides)

        survivors = _survivors(detections)
        assert survivors.shape[0] == 1
        assert bool(torch.isfinite(survivors).all())
        assert int(survivors[0, 6]) == 0  # the valid box's class, not the poisoned one's

    def test_a_non_finite_extent_is_not_emitted(self) -> None:
        """A box whose decoded ``w``/``h`` is non-finite is dropped the same way.

        Reaches the decoder through the ltrb distances rather than the angle, so
        the drop is shown to test the decoded row and not one particular input.
        """
        cls_logits, raw_ltrb, angles, points, strides = _plant([self.VALID_BOX], [6.0], [0])
        raw_ltrb[0, 1] = torch.tensor([float("inf")] * 4)
        cls_logits[0, 1, 1] = 5.0

        detections = RotatedNMSDecoder(conf_threshold=0.5)(cls_logits, raw_ltrb, angles, points, strides)

        survivors = _survivors(detections)
        assert survivors.shape[0] == 1
        assert bool(torch.isfinite(survivors).all())
        assert int(survivors[0, 6]) == 0

    def test_a_non_finite_row_does_not_shield_a_duplicate(self) -> None:
        """Dropping the bad row leaves ordinary suppression intact behind it.

        Two same-class copies of one bar sit either side of a ``NaN`` row in score
        order. The ``NaN`` is removed and the duplicate is then suppressed by the
        survivor, so the drop neither hides a duplicate nor takes a real box with it.
        """
        duplicate = (32.0, 32.0, 20.0, 4.0, 0.31)
        poisoned = (48.0, 48.0, 16.0, 4.0, float("nan"))
        cls_logits, raw_ltrb, angles, points, strides = _plant(
            [self.VALID_BOX, poisoned, duplicate], [6.0, 5.0, 4.0], [0, 0, 0]
        )

        detections = RotatedNMSDecoder(conf_threshold=0.5)(cls_logits, raw_ltrb, angles, points, strides)

        survivors = _survivors(detections)
        assert survivors.shape[0] == 1
        assert bool(torch.isfinite(survivors).all())

    def test_an_all_non_finite_image_yields_only_padding(self) -> None:
        """Every row poisoned leaves an all-zero padded output rather than a crash."""
        cls_logits, raw_ltrb, angles, points, strides = _plant([(32.0, 32.0, 20.0, 4.0, float("nan"))], [6.0], [0])

        detections = RotatedNMSDecoder(conf_threshold=0.5, max_det=2)(cls_logits, raw_ltrb, angles, points, strides)

        assert detections.shape == (1, 2, 7)
        assert torch.equal(detections, torch.zeros_like(detections))


def test_empty_batch_decodes_to_an_empty_result() -> None:
    """``B = 0`` returns a zero-row A45 batch rather than raising inside ``torch.stack``.

    The oriented decoder stacks one result per image, and ``torch.stack`` rejects an
    empty list, so an empty batch failed with a message about a ``TensorList``. Both
    other decoders return empties for the same input; matching them keeps the three
    paths interchangeable for an evaluation loop that may legitimately hand any of
    them a batch with no images in it.
    """
    points, strides = make_anchor_points([(1, 2)], [8])
    decoder = RotatedNMSDecoder(max_det=4)

    detections = decoder(torch.zeros(0, 2, 3), torch.zeros(0, 2, 4), torch.zeros(0, 2, 1), points, strides)

    assert detections.shape == (0, 4, 7)
