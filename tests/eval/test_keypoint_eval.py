# SPDX-License-Identifier: Apache-2.0
"""Unit gates for WP-124's COCO OKS keypoint evaluation wiring.

The cases pin the fixed-size prediction adapter, perfect and missed OKS matches,
the no-predictions contract, and COCO visibility masking without loading a real
dataset.
"""

from __future__ import annotations

import pytest
import torch

from lucid_yolo.eval.coco_eval import evaluate_keypoints, keypoints_to_predictions

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
