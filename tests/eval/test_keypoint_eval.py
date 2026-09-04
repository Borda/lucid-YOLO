# SPDX-License-Identifier: Apache-2.0
"""Unit gates for WP-124's COCO OKS keypoint evaluation wiring and WP-134's runner.

The WP-124 cases pin the fixed-size prediction adapter, perfect and missed OKS
matches, the no-predictions contract, and COCO visibility masking without loading a
real dataset.

WP-134 adds three subjects, all of which are wrong in ways that still produce a
plausible-looking report: which **branch** each decode path reads its points from,
what a padding row's anchor index does to the gather, and whether the OKS ground
truth is the annotation's own metadata or a reconstruction of it. There is no COCO
checkout on the gate machine, so each is separated by construction on synthetic
tensors rather than by running a real split.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
import torch
from faster_coco_eval import COCO, COCOeval_faster
from faster_coco_eval.core.cocoeval import Params
from torch import Tensor

from lucid_yolo.assign.grid import HEAD_STRIDES, anchor_grid
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.decode.common import PAD_ANCHOR_INDEX
from lucid_yolo.eval import coco_eval, pose_eval
from lucid_yolo.eval.coco_eval import DualPathEvaluator, evaluate_keypoints, keypoints_to_predictions
from lucid_yolo.models.heads.detect import DualHeadOutput
from lucid_yolo.models.heads.keypoint import decode_keypoints
from lucid_yolo.ptl.module import DetectionLitModule

_KEYPOINT_STATS = (
    "AP_all",
    "AP_50",
    "AP_75",
    "AP_medium",
    "AP_large",
    "AR_all",
    "AR_50",
    "AR_75",
    "AR_medium",
    "AR_large",
)
_TEST_SIGMAS = [0.5, 0.5, 0.5]


def test_keypoints_to_predictions_drops_padding_and_remaps_category() -> None:
    """A score-zero padding row is dropped while shape and category identity survive.

    The adapter must filter every per-instance field with one score mask; otherwise
    keypoints, scores, and remapped labels can silently describe different rows.
    """
    keypoints = torch.tensor(
        [
            [
                [[10.0, 12.0], [20.0, 22.0], [30.0, 32.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        ]
    )
    scores = torch.tensor([[0.9, 0.0]])
    labels = torch.tensor([[1, 0]])

    predictions = keypoints_to_predictions(keypoints, scores, labels, {0: 1, 1: 17})

    assert len(predictions) == 1
    assert predictions[0]["keypoints"].shape == (1, 3, 2)
    assert predictions[0]["keypoints"].tolist() == [[[10.0, 12.0], [20.0, 22.0], [30.0, 32.0]]]
    assert predictions[0]["scores"].tolist() == [pytest.approx(0.9)]
    assert predictions[0]["labels"].dtype == torch.long
    assert predictions[0]["labels"].tolist() == [17]


class TestEvaluateKeypoints:
    """COCO OKS behavior exposed by :func:`evaluate_keypoints`."""

    def test_perfect_match_has_high_average_precision(self) -> None:
        """Predicted keypoints equal to labeled ground truth score near-perfect AP.

        This is the positive end-to-end oracle for the in-memory COCO conversion and
        the backend's OKS matching, accumulation, and summary sequence.
        """
        points = torch.tensor([[[10.0, 10.0], [50.0, 10.0], [10.0, 50.0]]])
        predictions = [{"keypoints": points, "scores": torch.tensor([1.0]), "labels": torch.tensor([1])}]
        targets = [
            {
                "keypoints": points,
                "visibility": torch.tensor([[2, 2, 2]]),
                "labels": torch.tensor([1]),
            }
        ]

        stats = evaluate_keypoints(predictions, targets, sigmas=_TEST_SIGMAS)

        assert stats["AP_all"] > 0.999
        assert stats["AP_50"] > 0.999
        assert stats["AR_all"] > 0.999

    def test_far_prediction_has_zero_average_precision(self) -> None:
        """A prediction far outside the target's OKS tolerance scores zero AP.

        The negative oracle prevents an implementation from reporting the presence
        of a category as a match without evaluating keypoint localization.
        """
        target_points = torch.tensor([[[10.0, 10.0], [50.0, 10.0], [10.0, 50.0]]])
        predicted_points = torch.tensor([[[1010.0, 1010.0], [1050.0, 1010.0], [1010.0, 1050.0]]])
        predictions = [{"keypoints": predicted_points, "scores": torch.tensor([1.0]), "labels": torch.tensor([1])}]
        targets = [
            {
                "keypoints": target_points,
                "visibility": torch.tensor([[2, 2, 2]]),
                "labels": torch.tensor([1]),
            }
        ]

        stats = evaluate_keypoints(predictions, targets, sigmas=_TEST_SIGMAS)

        assert stats["AP_all"] == pytest.approx(0.0)
        assert stats["AP_50"] == pytest.approx(0.0)

    def test_empty_predictions_return_all_zero_statistics(self) -> None:
        """No prediction images yield the exact ten-key all-zero report.

        The explicit terminal contract avoids passing an empty results list into
        ``COCO.loadRes``, whose keypoint loader expects at least one result row.
        """
        stats = evaluate_keypoints([], [])

        assert tuple(stats) == _KEYPOINT_STATS
        assert all(value == 0.0 for value in stats.values())

    def test_unlabeled_target_coordinates_are_excluded(self) -> None:
        """Corrupting only a ``v == 0`` target point leaves every statistic unchanged.

        The two evaluations differ only in a coordinate that COCO marks unlabeled;
        equality proves neither OKS distance nor the visible-point area reads it.
        """
        predicted_points = torch.tensor([[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]])
        target_points = torch.tensor([[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]])
        visibility = torch.tensor([[2, 2, 0]])
        predictions = [{"keypoints": predicted_points, "scores": torch.tensor([1.0]), "labels": torch.tensor([1])}]
        baseline_targets = [{"keypoints": target_points, "visibility": visibility, "labels": torch.tensor([1])}]
        corrupted_points = target_points.clone()
        corrupted_points[0, 2] = torch.tensor([999.0, -999.0])
        corrupted_targets = [{"keypoints": corrupted_points, "visibility": visibility, "labels": torch.tensor([1])}]

        baseline = evaluate_keypoints(predictions, baseline_targets, sigmas=_TEST_SIGMAS)
        corrupted = evaluate_keypoints(predictions, corrupted_targets, sigmas=_TEST_SIGMAS)

        assert baseline == corrupted


def _keypoint_evaluator() -> COCOeval_faster:
    """Build the smallest real keypoint evaluator: one image, one two-point instance, one match.

    Examples:
        >>> _keypoint_evaluator().params.iouType
        'keypoints'
    """
    document = {
        "images": [{"id": 0}],
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "keypoints": [10.0, 10.0, 2.0, 50.0, 50.0, 2.0],
                "num_keypoints": 2,
                "bbox": [10.0, 10.0, 40.0, 40.0],
                "area": 1600.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "1"}],
    }
    results = [{"image_id": 0, "category_id": 1, "keypoints": [10.0, 10.0, 2.0, 50.0, 50.0, 2.0], "score": 0.9}]
    ground_truth = COCO(document)
    return COCOeval_faster(ground_truth, ground_truth.loadRes(results), iouType="keypoints", kpt_oks_sigmas=[1.0, 1.0])


class TestKeypointEvaluationParams:
    """WP-166: the keypoint detection cap and area partition are this project's, not the library's."""

    def test_pinned_values_equal_the_library_defaults(self) -> None:
        """The pinned ``maxDets``/``areaRng`` still equal what ``Params.setKpParams`` sets.

        :func:`evaluate_keypoints` pins both so a report's AR lines and its
        medium/large split are citable numbers rather than whatever the installed
        faster_coco_eval happens to default to. That pin is only worth having if a
        library that moves its own default is *noticed*: without this case the two
        could silently diverge and the comment beside the constants would quietly
        become false, exactly as the OKS sigma comment did.
        """
        defaults = Params(iouType="keypoints")

        assert tuple(defaults.maxDets) == coco_eval._KEYPOINT_MAX_DETS
        assert tuple(tuple(float(bound) for bound in rng) for rng in defaults.areaRng) == (
            coco_eval._KEYPOINT_AREA_RANGES
        )

    def test_evaluator_carries_the_pinned_values(self) -> None:
        """A constructed keypoint evaluator ends up holding the constants, not the defaults.

        The assignment goes through ``evaluator.params``, and hotcoco's own
        equivalent attribute is copy-on-read — an assignment there is a silent
        no-op. This pins that faster_coco_eval's is not, so the pin above is
        actually in force at ``evaluate()`` time rather than merely written down.
        """
        evaluator = _keypoint_evaluator()

        coco_eval._pin_keypoint_params(evaluator)

        assert tuple(evaluator.params.maxDets) == coco_eval._KEYPOINT_MAX_DETS
        assert tuple(tuple(float(bound) for bound in rng) for rng in evaluator.params.areaRng) == (
            coco_eval._KEYPOINT_AREA_RANGES
        )


class TestCanonicalGroundTruthMetadata:
    """WP-134: supplied annotation fields drive OKS, not a visible-point reconstruction."""

    #: Points spanning 50x50 px. Their extent is 2500, which COCO buckets as *medium*;
    #: the annotation below declares an area of 20000, which is *large*. The two readings
    #: therefore disagree about which bucket the one instance belongs to, and exactly one
    #: of the two buckets can be populated.
    _POINTS = torch.tensor([[[10.0, 10.0], [60.0, 10.0], [10.0, 60.0]]])
    _DECLARED_AREA = 20000.0

    def _predictions(self) -> list[dict[str, Tensor]]:
        """One perfect prediction, so OKS is 1.0 under either area and only the bucket moves."""
        return [{"keypoints": self._POINTS, "scores": torch.tensor([1.0]), "labels": torch.tensor([1])}]

    def _target(self, **metadata: Tensor) -> list[dict[str, Tensor]]:
        """One ground-truth instance, optionally carrying real annotation metadata."""
        return [
            {
                "keypoints": self._POINTS,
                "visibility": torch.tensor([[2, 2, 2]]),
                "labels": torch.tensor([1]),
                **metadata,
            }
        ]

    def test_declared_area_decides_the_bucket_over_visible_extent(self) -> None:
        """A supplied ``area`` of 20000 puts the instance in the large bucket, not the medium one.

        The instance's visible points span 2500 px^2 while its annotation declares
        20000, so the two readings land it in different buckets and the report says
        which one was used. WP-124 reconstructed unconditionally and would report
        the medium bucket here; COCO normalizes OKS by the supplied figure and
        buckets on it, which for a person is the segmented area rather than the hull
        of the labeled joints.
        """
        stats = evaluate_keypoints(
            self._predictions(),
            self._target(area=torch.tensor([self._DECLARED_AREA])),
            sigmas=_TEST_SIGMAS,
        )

        assert stats["AP_large"] == pytest.approx(1.0)
        assert stats["AP_medium"] == pytest.approx(-1.0)  # COCO's empty-bucket sentinel

    def test_visible_extent_still_decides_when_no_area_is_supplied(self) -> None:
        """Ground truth with no ``area`` keeps WP-124's reconstruction, and its medium bucket.

        The fallback is what every pre-WP-134 caller and every synthetic fixture with
        no annotation metadata relies on, so the new field must be a preference and
        not a requirement.
        """
        stats = evaluate_keypoints(self._predictions(), self._target(), sigmas=_TEST_SIGMAS)

        assert stats["AP_medium"] == pytest.approx(1.0)
        assert stats["AP_large"] == pytest.approx(-1.0)

    def test_a_zero_num_keypoints_annotation_is_ignored(self) -> None:
        """A supplied ``num_keypoints`` of 0 removes the instance from the evaluation entirely.

        COCO's protocol ignores a person annotated with a box but no joints. That
        rule can only fire on the supplied count: this instance has three visible
        points, so a recount from ``visibility`` would keep it and score 1.0.
        """
        stats = evaluate_keypoints(
            self._predictions(),
            self._target(num_keypoints=torch.tensor([0])),
            sigmas=_TEST_SIGMAS,
        )

        assert stats["AP_all"] == pytest.approx(-1.0)


#: Letterbox canvas of the dual-path cases. Equal to the images' original size, so the
#: A10 inverse is the identity and an assertion reads the decode rather than the resize.
_CANVAS = 64
#: Class count of the stub head; the stub decoders emit label 0 regardless.
_NUM_CLASSES = 2
#: Points per instance in the dual-path cases.
_POINT_COUNT = 3


class _FixedDecoder(torch.nn.Module):
    """A decoder emitting fixed detections and fixed source anchor indices.

    Both decode paths run the same untrained stub head, so a real decoder would give
    the two paths identical output and crossing them would change nothing observable.
    Fixing each path's answer separates them by construction.
    """

    def __init__(self, detections: Tensor, anchor_indices: Tensor) -> None:
        super().__init__()
        self.detections = detections
        self.anchor_indices = anchor_indices

    def forward(self, cls_logits: Tensor, raw_ltrb: Tensor, anchor_points: Tensor, strides: Tensor) -> Tensor:
        """Return the fixed detections, whatever the head predicted."""
        del cls_logits, raw_ltrb, anchor_points, strides
        return self.detections

    def decode_with_indices(
        self, cls_logits: Tensor, raw_ltrb: Tensor, anchor_points: Tensor, strides: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the fixed detections beside the fixed anchor index of each row."""
        del cls_logits, raw_ltrb, anchor_points, strides
        return self.detections, self.anchor_indices


class _KeypointHead(torch.nn.Module):
    """A model whose two branches carry deliberately different raw point offsets.

    ``None`` for both branches is a detection-only head, which is how the absence of
    the keypoint report is asserted against the same stub.
    """

    def __init__(self, anchors: int, o2o_points: Tensor | None, o2m_points: Tensor | None) -> None:
        super().__init__()
        self.anchors = anchors
        self.o2o_points = o2o_points
        self.o2m_points = o2m_points

    def forward(self, images: Tensor) -> DualHeadOutput:
        """Return zero boxes and class logits beside the two branches' own point offsets."""
        del images
        cls_logits, raw_ltrb = torch.zeros(1, self.anchors, _NUM_CLASSES), torch.zeros(1, self.anchors, 4)
        return DualHeadOutput(
            o2m_cls=cls_logits,
            o2m_box=raw_ltrb,
            o2o_cls=cls_logits,
            o2o_box=raw_ltrb,
            o2m_keypoints=self.o2m_points,
            o2o_keypoints=self.o2o_points,
        )


def _distinct_branch_offsets() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build two branches' raw offsets and the anchor grid they decode against.

    Every anchor of the one-to-one branch carries a positive offset and every anchor
    of the one-to-many branch the negative of it; each anchor's magnitude differs from
    its neighbours' and each point's from its siblings', so a gathered point set names
    both the branch it came from and the anchor row it was taken at, and spans a
    non-degenerate extent.

    Examples:
        >>> o2o, o2m, anchors, strides = _distinct_branch_offsets()
        >>> o2o.shape == o2m.shape and bool((o2m == -o2o).all())
        True
        >>> o2o.shape[1] == anchors.shape[0] == strides.shape[0]
        True
    """
    anchor_points, strides = anchor_grid((_CANVAS, _CANVAS), torch.device("cpu"), HEAD_STRIDES)
    anchors = int(anchor_points.shape[0])
    per_anchor = torch.arange(1, anchors + 1, dtype=torch.float32).view(1, anchors, 1, 1)
    per_point = torch.arange(_POINT_COUNT, dtype=torch.float32).view(1, 1, _POINT_COUNT, 1)
    o2o_points = (per_anchor + per_point).expand(1, anchors, _POINT_COUNT, 2).clone()
    return o2o_points, -o2o_points, anchor_points, strides


class TestDualPathKeypoints:
    """WP-134: each decode path's points come from its own branch, at its own anchors.

    The ground truth is set to exactly what the **e2e** path should gather — the
    one-to-one branch's offsets read at the E2E decoder's own anchor. Of the four
    combinations of branch and anchor index the evaluator could take, only that one
    scores OKS 1.0, so a crossed branch and a crossed index are both visible in the
    report and neither can hide behind correct boxes, scores and labels.
    """

    _E2E_ANCHOR = 3
    _NMS_ANCHOR = 11
    _BOX: ClassVar[list[float]] = [8.0, 8.0, 24.0, 24.0]

    def _batches(self) -> list[tuple[Tensor, list[int], list[tuple[int, int]]]]:
        """One batch of one image whose original size equals the canvas."""
        return [(torch.zeros(1, 3, _CANVAS, _CANVAS), [1], [(_CANVAS, _CANVAS)])]

    def _target(self, points: Tensor) -> dict[str, Tensor]:
        """Ground truth for the one image: one instance, at ``points``, all joints labeled."""
        return {
            "boxes": torch.tensor([self._BOX]),
            "labels": torch.tensor([1]),
            "keypoints": points.unsqueeze(0),
            "visibility": torch.full((1, _POINT_COUNT), 2, dtype=torch.long),
        }

    def _evaluator(self, e2e: _FixedDecoder, nms: _FixedDecoder, head: _KeypointHead) -> DualPathEvaluator:
        """Build an evaluator over the stub head and the two fixed decoders."""
        return DualPathEvaluator(head, e2e, nms, {0: 1}, Letterbox(_CANVAS), keypoint_sigmas=_TEST_SIGMAS)

    def test_each_path_reads_its_own_branch_at_its_own_anchor(self) -> None:
        """Only ``o2o_keypoints`` gathered at the E2E anchor scores; the dense branch misses.

        Reading the other branch, or gathering with the other path's anchor indices,
        leaves every box, score and label correct and moves only the poses — a report
        that looks entirely reasonable. The two branches differ in sign here and the
        two anchors in magnitude, so all three wrong combinations land far outside
        OKS tolerance while the right one lands exactly on the target.
        """
        o2o_points, o2m_points, anchor_points, strides = _distinct_branch_offsets()
        detections = torch.tensor([[[*self._BOX, 1.0, 0.0]]])
        e2e_decoder = _FixedDecoder(detections, torch.tensor([[self._E2E_ANCHOR]]))
        nms_decoder = _FixedDecoder(detections, torch.tensor([[self._NMS_ANCHOR]]))
        head = _KeypointHead(int(anchor_points.shape[0]), o2o_points, o2m_points)
        evaluator = self._evaluator(e2e_decoder, nms_decoder, head)
        expected = decode_keypoints(o2o_points, anchor_points, strides)[0, self._E2E_ANCHOR]

        report = evaluator.evaluate(self._batches(), {1: self._target(expected)}, torch.device("cpu"))

        assert report["e2e"]["oks_AP_all"] == pytest.approx(1.0)
        assert report["nms"]["oks_AP_all"] == pytest.approx(0.0)

    def test_a_padding_anchor_index_neither_crashes_nor_invents_a_pose(self) -> None:
        """A ``PAD_ANCHOR_INDEX`` row leaves the real row's score untouched at a perfect OKS.

        The sentinel is ``-1``, which ``Tensor.gather`` would accept as an index and
        answer with the *last* anchor's points. That row also carries score 0, so a
        correct implementation drops it entirely and the surviving prediction is the
        real anchor's — which the perfect OKS below is only reachable through.
        """
        o2o_points, o2m_points, anchor_points, strides = _distinct_branch_offsets()
        detections = torch.tensor([[[*self._BOX, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        decoder = _FixedDecoder(detections, torch.tensor([[self._E2E_ANCHOR, PAD_ANCHOR_INDEX]]))
        head = _KeypointHead(int(anchor_points.shape[0]), o2o_points, o2m_points)
        evaluator = self._evaluator(decoder, decoder, head)
        expected = decode_keypoints(o2o_points, anchor_points, strides)[0, self._E2E_ANCHOR]

        report = evaluator.evaluate(self._batches(), {1: self._target(expected)}, torch.device("cpu"))

        assert report["e2e"]["oks_AP_all"] == pytest.approx(1.0)

    def test_a_detection_checkpoint_reports_no_keypoint_statistics(self) -> None:
        """A head with no point branch produces exactly the box report it always produced.

        The keypoint machinery is reached structurally, off the head output's own
        fields, so a detection checkpoint must not gain an ``oks_`` key — the absence
        is what makes the prefix's presence mean "this checkpoint has points".
        """
        _, _, anchor_points, _ = _distinct_branch_offsets()
        head = _KeypointHead(int(anchor_points.shape[0]), None, None)
        detections = torch.tensor([[[*self._BOX, 1.0, 0.0]]])
        decoder = _FixedDecoder(detections, torch.tensor([[self._E2E_ANCHOR]]))
        evaluator = DualPathEvaluator(head, decoder, decoder, {0: 1}, Letterbox(_CANVAS))
        target = {"boxes": torch.tensor([self._BOX]), "labels": torch.tensor([1])}

        report = evaluator.evaluate(self._batches(), {1: target}, torch.device("cpu"))

        assert report["e2e"]["map"] == pytest.approx(1.0)
        assert not any(key.startswith("oks_") for key in report["e2e"])


class TestPoseProtocolRefusesForeignSchemas:
    """WP-134: the COCO person protocol scores 17 points or refuses to run."""

    def test_a_non_seventeen_point_checkpoint_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A ``K=7`` checkpoint exits non-zero and names both point counts, scoring nothing.

        This protocol's ground truth, point ordering and sigma table are all COCO's
        person schema, so point *i* of a seven-point prediction is not point *i* of
        the annotation and R12's sigmas measure variance on joints that are not being
        predicted. Padding or truncating COCO's table would produce a real-looking
        number for a comparison that does not exist. The refusal fires before any
        annotation file is opened, which is why the data root here need not exist.
        """
        module = DetectionLitModule(
            depth=0.34, width=0.25, max_channels=64, num_classes=1, task="keypoints", num_keypoints=7
        )

        code = pose_eval.run(
            module.eval(),
            {},
            data_root=tmp_path / "absent",
            img_size=640,
            batch_size=1,
            device_name="cpu",
            limit=0,
            output=None,
        )

        assert code == 1
        message = capsys.readouterr().out
        assert "17" in message
        assert "K=7" in message
