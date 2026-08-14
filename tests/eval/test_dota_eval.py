# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the rotated mAP50-95 DOTA instrument (WP-063, A24).

Two oracles carry this file.

The geometric one is **shapely**, which A24 places here rather than in ``src``: the
runtime path must not depend on it, so the exact polygon intersection is implemented in
torch and checked against a mature computational-geometry library instead. It is built
from the *same* :func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons` corners the
kernel consumes, so what is under test is the clipping and area arithmetic alone and not
the long-edge convention WP-055 already gates.

The protocol one is :func:`~lucid_yolo.eval.coco_eval.evaluate_bbox`, the
COCOeval-faithful axis-aligned instrument. At ``theta = 0`` every rotated box is its own
axis-aligned box, so the two are scoring the same overlaps on the same matching problem
and any disagreement is a protocol difference rather than a geometric one. That is the
strongest available check on decision 2 (101-point interpolated recall), whose entire
purpose is that the two reports define mAP identically.

Everything else is written out by hand: the average-precision fixture derives its answer
in its own docstring, and each protocol decision has a case that fails if the decision is
reverted.
"""

from __future__ import annotations

import math

import pytest
import torch
from shapely.geometry import Polygon
from torch import Tensor

from lucid_yolo.data.rotated_geom import rboxes_to_polygons
from lucid_yolo.data.targets import Targets
from lucid_yolo.data.tiling import TiledTargets
from lucid_yolo.eval.coco_eval import evaluate_bbox
from lucid_yolo.eval.dota_eval import (
    IOU_THRESHOLDS,
    MAX_DETECTIONS,
    RECALL_POINTS,
    evaluate_rotated_map,
    rotated_detections_to_predictions,
    rotated_iou,
    tiled_targets_to_ground_truth,
)
from lucid_yolo.models.heads.obb import o2o_rotated_topk

#: Agreement tolerance against the shapely oracle, per the WP-063 acceptance criterion.
_ORACLE_TOLERANCE = 1e-4

#: Shape of the cross-check fixture in :func:`_axis_aligned_fixture`. Large enough that
#: the matching is exercised across many curves rather than one lucky one.
_FIXTURE_IMAGES = 24
_FIXTURE_CLASSES = 4
#: The class skipped in every other image, so at least one curve saturates below recall 1.
_STARVED_CLASS = 3

#: Pairs the kernel must get exactly right, each named for the property it probes rather
#: than sampled from a sweep that might or might not contain it.
_EDGE_CASES = [
    pytest.param([0.0, 0.0, 4.0, 2.0, 0.3], [0.0, 0.0, 4.0, 2.0, 0.3], id="identical"),
    pytest.param([0.0, 0.0, 4.0, 2.0, 0.0], [4.0, 0.0, 4.0, 2.0, 0.0], id="shared-edge"),
    pytest.param([0.0, 0.0, 4.0, 2.0, 0.0], [4.0, 2.0, 4.0, 2.0, 0.0], id="shared-corner"),
    pytest.param([0.0, 0.0, 10.0, 8.0, 0.0], [0.0, 0.0, 3.0, 2.0, 0.4], id="containment"),
    pytest.param([0.0, 0.0, 3.0, 2.0, 0.4], [0.0, 0.0, 10.0, 8.0, 0.0], id="contained-by"),
    pytest.param([0.0, 0.0, 4.0, 2.0, 0.0], [50.0, 50.0, 4.0, 2.0, 0.0], id="disjoint"),
    pytest.param([0.0, 0.0, 300.0, 4.0, 0.30], [1.0, 0.5, 300.0, 4.0, 0.30001], id="near-parallel"),
    pytest.param([0.0, 0.0, 300.0, 4.0, 0.30], [0.0, 0.0, 300.0, 4.0, 0.30], id="near-parallel-coincident"),
    pytest.param([0.0, 0.0, 3.0, 3.0, 0.2], [0.0, 0.0, 3.0, 3.0, 0.2 + math.pi / 2], id="square-fold"),
    pytest.param([0.0, 0.0, 5.0, 5.0, -math.pi / 4], [0.0, 0.0, 5.0, 5.0, math.pi / 4], id="square-fold-bounds"),
    pytest.param([0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 2.0, 0.0], id="zero-area"),
    pytest.param([0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], id="both-zero-area"),
    pytest.param([0.0, 0.0, 6.0, 0.0, 0.2], [0.0, 0.0, 4.0, 2.0, 0.2], id="degenerate-line"),
    pytest.param([0.0, 0.0, 6.0, 0.0, 0.2], [0.0, 0.0, 6.0, 0.0, 0.2], id="both-degenerate-lines"),
    pytest.param([12000.0, 9000.0, 60.0, 30.0, 0.7], [12010.0, 9004.0, 55.0, 33.0, 0.9], id="dota-scale-offset"),
]

#: Angles spanning the canonical ``[-pi/4, 3*pi/4)`` range and both of its bounds, used to
#: sweep the oracle agreement across every orientation the convention admits.
_SWEEP_ANGLES = torch.linspace(-math.pi / 4, 3 * math.pi / 4 - 1e-4, 24)


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed every RNG source before each test for deterministic tensors."""
    torch.manual_seed(0)


def oracle_iou(box_a: Tensor, box_b: Tensor) -> float:
    """Shapely's polygon intersection over union for one pair of rotated boxes.

    Consumes the corners :func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons` emits so
    the comparison isolates the clipping arithmetic, and evaluates them in float64 so the
    oracle is not limited by the precision it is auditing.
    """
    polygon_a = Polygon(rboxes_to_polygons(box_a[None].double())[0].tolist())
    polygon_b = Polygon(rboxes_to_polygons(box_b[None].double())[0].tolist())
    union = polygon_a.union(polygon_b).area
    return 0.0 if union <= 0.0 else polygon_a.intersection(polygon_b).area / union


def _rbox(values: list[float]) -> Tensor:
    """Return a single ``(1, 5)`` float32 rotated box from its five parameters."""
    return torch.tensor([values], dtype=torch.float32)


def _sweep_pairs(count: int) -> tuple[Tensor, Tensor]:
    """Return ``count`` random box pairs whose angles cover the whole canonical range.

    The second box of each pair is a perturbation of the first — its centre displaced on
    the scale of the first's own extent, its sides rescaled — rather than an independent
    draw. Two independent draws are mostly disjoint, and a sweep of disjoint boxes agrees
    with any oracle at zero while testing no clipping at all.

    Half the pairs take an independent angle, which spreads coverage across the canonical
    range; the other half take the first box's angle plus a small offset, which is the
    only way to reach the **near-coincident** regime. That regime is the one worth
    reaching: its edges are near-parallel, so the crossing parameters are the
    worst-conditioned quantities the kernel computes, and its overlaps run to 1.
    """
    angle_index = torch.randint(0, _SWEEP_ANGLES.numel(), (count, 2))
    centres = (torch.rand(count, 2) - 0.5) * 40.0
    extents = torch.rand(count, 2) * 40.0 + 1.0
    first_angle = _SWEEP_ANGLES[angle_index[:, 0]]
    first = torch.cat([centres, extents, first_angle[:, None]], dim=1)

    near_parallel = torch.arange(count) % 2 == 0
    nudged = first_angle + (torch.rand(count) - 0.5) * 0.2
    second_angle = torch.where(near_parallel, nudged, _SWEEP_ANGLES[angle_index[:, 1]])
    displacement = (torch.rand(count, 2) - 0.5) * torch.where(near_parallel[:, None], 0.5, 1.6) * extents
    rescale = torch.where(near_parallel[:, None], 0.85 + torch.rand(count, 2) * 0.3, 0.4 + torch.rand(count, 2) * 1.4)
    second = torch.cat([centres + displacement, extents * rescale, second_angle[:, None]], dim=1)
    return first, second


def _perfect_case(count: int, match_index: int) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Build one image of ``count`` class-0 detections where only ``match_index`` is correct.

    Scores descend with the row index, so ``match_index`` doubles as the rank of the only
    detection that can become a true positive.
    """
    target = torch.tensor([[100.0, 100.0, 20.0, 10.0, 0.2]])
    boxes = torch.tensor([[1000.0 + row, 1000.0, 4.0, 2.0, 0.0] for row in range(count)])
    boxes[match_index] = target[0]
    preds = [
        {
            "rboxes": boxes,
            "scores": torch.linspace(1.0, 0.01, count),
            "labels": torch.zeros(count, dtype=torch.long),
        }
    ]
    targets = [
        {"rboxes": target, "labels": torch.zeros(1, dtype=torch.long), "difficult": torch.zeros(1, dtype=torch.bool)}
    ]
    return preds, targets


class TestRotatedIou:
    """Exact polygon-intersection IoU against the shapely oracle and its own invariants."""

    def test_vs_oracle(self) -> None:
        """Random pairs across the whole canonical angle range agree with shapely to 1e-4."""
        boxes_a, boxes_b = _sweep_pairs(600)

        computed = torch.stack(
            [rotated_iou(boxes_a[row : row + 1], boxes_b[row : row + 1])[0, 0] for row in range(600)]
        )

        expected = torch.tensor([oracle_iou(boxes_a[row], boxes_b[row]) for row in range(600)])
        assert torch.allclose(computed, expected.float(), atol=_ORACLE_TOLERANCE)

    @pytest.mark.parametrize(("first", "second"), _EDGE_CASES)
    def test_vs_oracle_edge_cases(self, first: list[float], second: list[float]) -> None:
        """Each named degenerate, touching, containing or near-parallel pair matches shapely."""
        box_a, box_b = _rbox(first), _rbox(second)

        computed = float(rotated_iou(box_a, box_b))

        assert computed == pytest.approx(oracle_iou(box_a[0], box_b[0]), abs=_ORACLE_TOLERANCE)

    @pytest.mark.parametrize(("first", "second"), _EDGE_CASES)
    def test_edge_cases_stay_in_range(self, first: list[float], second: list[float]) -> None:
        """No degenerate pair produces a NaN or an overlap outside ``[0, 1]``."""
        computed = rotated_iou(_rbox(first), _rbox(second))

        assert torch.isfinite(computed).all()
        assert bool(((computed >= 0.0) & (computed <= 1.0)).all())

    def test_identical_boxes_score_exactly_one(self) -> None:
        """A box against itself is exactly 1.0, not merely close to it."""
        boxes = torch.tensor([[3.0, -2.0, 9.0, 4.0, 0.6], [0.0, 0.0, 5.0, 5.0, -0.7]])

        computed = rotated_iou(boxes, boxes)

        assert torch.equal(torch.diagonal(computed), torch.ones(2))

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            pytest.param([0.0, 0.0, 4.0, 2.0, 0.0], [4.0, 0.0, 4.0, 2.0, 0.0], id="shared-edge"),
            pytest.param([0.0, 0.0, 4.0, 2.0, 0.0], [4.0, 2.0, 4.0, 2.0, 0.0], id="shared-corner"),
        ],
    )
    def test_touching_boxes_enclose_no_area(self, first: list[float], second: list[float]) -> None:
        """Edge-inclusive clipping keeps the shared vertices but they bound zero area."""
        assert float(rotated_iou(_rbox(first), _rbox(second))) == 0.0

    def test_containment_is_the_area_ratio(self) -> None:
        """A box wholly inside another scores exactly the ratio of their areas."""
        outer = _rbox([0.0, 0.0, 10.0, 8.0, 0.3])
        inner = _rbox([0.0, 0.0, 4.0, 2.0, 0.3])

        assert float(rotated_iou(outer, inner)) == pytest.approx(8.0 / 80.0, abs=1e-6)

    def test_symmetric_in_its_arguments(self) -> None:
        """``iou(a, b)`` equals ``iou(b, a)`` transposed."""
        boxes_a, boxes_b = _sweep_pairs(40)

        forward = rotated_iou(boxes_a, boxes_b)

        assert torch.allclose(forward, rotated_iou(boxes_b, boxes_a).T, atol=1e-6)

    def test_pairwise_grid_matches_individual_pairs(self) -> None:
        """The ``(M, N)`` grid holds exactly the overlaps of the pairs it claims to."""
        boxes_a, boxes_b = _sweep_pairs(6)

        grid = rotated_iou(boxes_a, boxes_b)

        singles = torch.tensor(
            [[float(rotated_iou(boxes_a[i : i + 1], boxes_b[j : j + 1])) for j in range(6)] for i in range(6)]
        )
        assert torch.allclose(grid, singles, atol=1e-6)

    @pytest.mark.parametrize(
        "dtype",
        [pytest.param(torch.float32, id="float32"), pytest.param(torch.float64, id="float64")],
    )
    def test_working_dtype_follows_the_input(self, dtype: torch.dtype) -> None:
        """The result keeps the input dtype — no silent promotion to float64 (MPS has none)."""
        boxes = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.3]], dtype=dtype)

        assert rotated_iou(boxes, boxes).dtype == dtype

    @pytest.mark.parametrize(
        ("rows_a", "rows_b"),
        [pytest.param(0, 3, id="no-detections"), pytest.param(3, 0, id="no-targets"), pytest.param(0, 0, id="neither")],
    )
    def test_empty_inputs_give_an_empty_grid(self, rows_a: int, rows_b: int) -> None:
        """An empty box set yields the correctly shaped empty overlap grid, not an error."""
        computed = rotated_iou(torch.zeros(rows_a, 5), torch.zeros(rows_b, 5))

        assert tuple(computed.shape) == (rows_a, rows_b)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((4,), id="one-dimensional"),
            pytest.param((3, 4), id="wrong-width"),
            pytest.param((2, 3, 5), id="three-dimensional"),
        ],
    )
    def test_rejects_malformed_boxes(self, shape: tuple[int, ...]) -> None:
        """A tensor that is not ``(N, 5)`` is rejected rather than silently reinterpreted."""
        with pytest.raises(ValueError, match=r"boxes_a must be \(N, 5\)"):
            rotated_iou(torch.zeros(shape), torch.zeros(1, 5))


class TestProtocolConstants:
    """The five protocol decisions, each pinned where reverting it would change a number."""

    def test_ten_iou_thresholds_from_50_to_95(self) -> None:
        """Decision 1: ten thresholds, 0.50 to 0.95 inclusive, in steps of 0.05."""
        assert IOU_THRESHOLDS == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

    def test_recall_grid_is_the_coco_101_point_one(self) -> None:
        """Decision 2: 101 interpolation points, the COCO convention."""
        assert RECALL_POINTS == 101

    def test_detection_cap_is_300_not_cocos_100(self) -> None:
        """Decision 3: the cap matches what the oriented head actually emits."""

        detections = o2o_rotated_topk(torch.zeros(1, 1000, 3), torch.zeros(1, 1000, 5))
        assert MAX_DETECTIONS == detections.shape[1] == 300


class TestAccumulator:
    """Matching, averaging and the four summary statistics."""

    def test_hand_computed_average_precision(self) -> None:
        """A fixture whose AP is derivable by hand, and derived here.

        One image, one class, three non-difficult ground truths and four detections whose
        overlaps are all exactly 1 or 0, so the result is the same at every threshold:

        =====  =====  ==============================  ====
        rank   score  outcome                         run
        =====  =====  ==============================  ====
        1      0.9    equals ground truth 0            TP
        2      0.8    equals ground truth 0 again      FP
        3      0.7    equals ground truth 1            TP
        4      0.6    far from everything              FP
        =====  =====  ==============================  ====

        Cumulative precision is ``[1/1, 1/2, 2/3, 2/4]`` and recall ``[1/3, 1/3, 2/3,
        2/3]``. The right-to-left running maximum makes precision ``[1, 2/3, 2/3, 1/2]``.
        Of the 101 recall points, the 34 in ``[0.00, 0.33]`` are first reached at rank 1
        and sample precision 1; the 33 in ``[0.34, 0.66]`` are first reached at rank 3 and
        sample 2/3; the 34 in ``[0.67, 1.00]`` are never reached and sample 0. So
        ``AP = (34 * 1 + 33 * 2/3) / 101 = 56/101`` and the attained recall is 2/3.
        """
        first = [10.0, 10.0, 6.0, 4.0, 0.3]
        second = [80.0, 80.0, 6.0, 4.0, -0.2]
        third = [200.0, 200.0, 6.0, 4.0, 0.5]
        elsewhere = [900.0, 900.0, 6.0, 4.0, 0.0]
        preds = [
            {
                "rboxes": torch.tensor([first, first, second, elsewhere]),
                "scores": torch.tensor([0.9, 0.8, 0.7, 0.6]),
                "labels": torch.zeros(4, dtype=torch.long),
            }
        ]
        targets = [
            {
                "rboxes": torch.tensor([first, second, third]),
                "labels": torch.zeros(3, dtype=torch.long),
                "difficult": torch.zeros(3, dtype=torch.bool),
            }
        ]

        stats = evaluate_rotated_map(preds, targets)

        assert stats["map"] == pytest.approx(56 / 101, abs=1e-6)
        assert stats["map_50"] == pytest.approx(56 / 101, abs=1e-6)
        assert stats["map_75"] == pytest.approx(56 / 101, abs=1e-6)
        assert stats["mar_300"] == pytest.approx(2 / 3, abs=1e-6)

    def test_matches_the_axis_aligned_instrument_at_zero_angle(self) -> None:
        """Decision 2's purpose: at ``theta = 0`` this and COCOeval report the same mAP.

        The rotated and axis-aligned instruments then face the same overlaps and the same
        matching problem, so any gap is a difference of protocol — a different recall
        interpolation, a different envelope, a different class-exclusion rule.

        The fixture is deliberately adversarial for the matching: 24 images, four classes,
        ground truths that are never detected, a duplicate detection in every image, two
        outright false positives per image, and one class starved in half the images so
        its recall saturates well below 1.

        What it does **not** test is the recall-grid boundary — no class here attains a
        recall that lands exactly on a grid point, which is the one place the two
        implementations are known to disagree. That case is pinned separately and
        exactly by :meth:`TestRecallGridBoundary.test_boundary_recall_is_sampled_exactly`;
        this fixture would pass with the boundary handled either way, and saying so is the
        difference between a cross-check and a claim it cannot support.
        """
        rotated_preds, rotated_targets, box_preds, box_targets = _axis_aligned_fixture()

        rotated = evaluate_rotated_map(rotated_preds, rotated_targets)

        axis_aligned = evaluate_bbox(box_preds, box_targets)
        for key in ("map", "map_50", "map_75"):
            assert rotated[key] == pytest.approx(axis_aligned[key], abs=1e-6)

    def test_perfect_predictions_score_one(self) -> None:
        """Ground truth returned as its own prediction scores 1.0 at every statistic."""
        boxes = torch.tensor([[10.0, 10.0, 6.0, 4.0, 0.3], [80.0, 40.0, 9.0, 3.0, -0.5]])
        preds = [{"rboxes": boxes, "scores": torch.tensor([0.9, 0.8]), "labels": torch.tensor([0, 1])}]
        targets = [{"rboxes": boxes, "labels": torch.tensor([0, 1]), "difficult": torch.zeros(2, dtype=torch.bool)}]

        stats = evaluate_rotated_map(preds, targets)

        assert stats == {"map": 1.0, "map_50": 1.0, "map_75": 1.0, "mar_300": 1.0}

    def test_greedy_matching_is_one_to_one(self) -> None:
        """A second detection on an already-claimed ground truth is a false positive.

        Two targets, three detections: the first claims target one, the second repeats it
        and cannot claim it again, the third finds target two. Precision runs
        ``[1/1, 1/2, 2/3]`` against recall ``[1/2, 1/2, 1]``, whose envelope is
        ``[1, 2/3, 2/3]``. The 51 recall points up to 0.50 sample 1 and the 50 above it
        sample 2/3, so ``AP = (51 + 50 * 2/3) / 101``. Without the one-to-one rule the
        duplicate would be a second true positive and the answer would be 1.
        """
        found = [10.0, 10.0, 6.0, 4.0, 0.3]
        other = [300.0, 300.0, 6.0, 4.0, -0.1]
        preds = [
            {
                "rboxes": torch.tensor([found, found, other]),
                "scores": torch.tensor([0.9, 0.8, 0.7]),
                "labels": torch.zeros(3, dtype=torch.long),
            }
        ]
        targets = [
            {
                "rboxes": torch.tensor([found, other]),
                "labels": torch.zeros(2, dtype=torch.long),
                "difficult": torch.zeros(2, dtype=torch.bool),
            }
        ]

        stats = evaluate_rotated_map(preds, targets)

        # Both targets are found, so recall saturates; the duplicate ranked between them
        # is a false positive and costs precision at the second target's recall level.
        assert stats["mar_300"] == pytest.approx(1.0)
        assert stats["map"] == pytest.approx((51 + 50 * 2 / 3) / 101, abs=1e-5)

    def test_duplicate_after_full_recall_is_hidden_by_interpolation(self) -> None:
        """A duplicate past the last target costs nothing — COCO's envelope, not a bug.

        Once recall has saturated, every one of the 101 points is first reached at or
        before the true positive, so the right-to-left running maximum never samples the
        halved precision the duplicate produces. ``faster_coco_eval`` reports 1.0 for the
        same fixture; matching that is the point of decision 2, so it is asserted rather
        than worked around.
        """
        box = torch.tensor([[10.0, 10.0, 6.0, 4.0, 0.3]])
        preds = [
            {
                "rboxes": torch.cat([box, box]),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.zeros(2, dtype=torch.long),
            }
        ]
        targets = [
            {"rboxes": box, "labels": torch.zeros(1, dtype=torch.long), "difficult": torch.zeros(1, dtype=torch.bool)}
        ]

        assert evaluate_rotated_map(preds, targets)["map"] == pytest.approx(1.0)

    def test_empty_evaluation_reports_zeros(self) -> None:
        """An evaluation with no ground truth at all yields zeros rather than raising."""
        preds = [{"rboxes": torch.zeros(0, 5), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}]
        targets = [
            {
                "rboxes": torch.zeros(0, 5),
                "labels": torch.zeros(0, dtype=torch.long),
                "difficult": torch.zeros(0, dtype=torch.bool),
            }
        ]

        assert evaluate_rotated_map(preds, targets) == {"map": 0.0, "map_50": 0.0, "map_75": 0.0, "mar_300": 0.0}

    def test_rejects_misaligned_inputs(self) -> None:
        """Prediction and target lists of different lengths are rejected, not zipped short."""
        with pytest.raises(ValueError, match="align by position"):
            evaluate_rotated_map([], [{"rboxes": torch.zeros(0, 5), "labels": torch.zeros(0, dtype=torch.long)}])


class TestThresholdIndependence:
    """The ten thresholds are matched in one pass (WP-104) and must still decide separately."""

    def test_each_threshold_consumes_ground_truth_on_its_own(self) -> None:
        """A ground truth claimed at a loose threshold is still free at a strict one.

        One ground truth and two detections of it: the higher-scoring one overlaps by
        about 0.62, the lower-scoring one exactly. At 0.50 the first claims the target and
        the second is a duplicate, so the ranked run is ``[TP, FP]`` and AP is 1. At 0.75
        the first misses, so the run is ``[FP, TP]``, every recall point is first reached
        at rank 2, and AP is 1/2. Sharing one availability state across thresholds would
        collapse the two into whichever ran last.

        Three thresholds (0.50, 0.55, 0.60) score 1 and the remaining seven score 1/2, so
        the mean is ``(3 + 3.5) / 10``.
        """
        target = [0.0, 0.0, 10.0, 10.0, 0.0]
        partial = [2.345, 0.0, 10.0, 10.0, 0.0]  # IoU ~0.6201 — between two grid points
        preds = [
            {
                "rboxes": torch.tensor([partial, target]),
                "scores": torch.tensor([0.9, 0.5]),
                "labels": torch.zeros(2, dtype=torch.long),
            }
        ]
        targets = [
            {
                "rboxes": torch.tensor([target]),
                "labels": torch.zeros(1, dtype=torch.long),
                "difficult": torch.zeros(1, dtype=torch.bool),
            }
        ]

        stats = evaluate_rotated_map(preds, targets)

        assert stats["map_50"] == pytest.approx(1.0, abs=1e-6)
        assert stats["map_75"] == pytest.approx(0.5, abs=1e-6)
        assert stats["map"] == pytest.approx(0.65, abs=1e-6)
        assert stats["mar_300"] == pytest.approx(1.0, abs=1e-6)

    def test_a_detection_reaching_nothing_is_a_false_positive_and_consumes_nothing(self) -> None:
        """The skipped-walk path: no overlap anywhere means false positive at every threshold.

        WP-104 keeps such a detection out of the matching loop entirely, which is only
        sound if it neither scores nor consumes. The higher-scoring detection is nowhere
        near the target and the lower-scoring one matches it exactly, so the ranked run is
        ``[FP, TP]`` at all ten thresholds and AP is 1/2 throughout — the target still
        available for the detection behind it.
        """
        target = [0.0, 0.0, 10.0, 10.0, 0.0]
        preds = [
            {
                "rboxes": torch.tensor([[900.0, 900.0, 10.0, 10.0, 0.0], target]),
                "scores": torch.tensor([0.9, 0.5]),
                "labels": torch.zeros(2, dtype=torch.long),
            }
        ]
        targets = [
            {
                "rboxes": torch.tensor([target]),
                "labels": torch.zeros(1, dtype=torch.long),
                "difficult": torch.zeros(1, dtype=torch.bool),
            }
        ]

        stats = evaluate_rotated_map(preds, targets)

        assert stats["map"] == pytest.approx(0.5, abs=1e-6)
        assert stats["map_50"] == pytest.approx(0.5, abs=1e-6)
        assert stats["mar_300"] == pytest.approx(1.0, abs=1e-6)


class TestRecallGridBoundary:
    """Decision 2 at the one place floating point cannot decide it: an exact grid hit."""

    @pytest.mark.parametrize(
        ("found", "positives", "grid_index"),
        [
            pytest.param(1, 2, 50, id="half-of-two"),
            pytest.param(3, 4, 75, id="three-of-four"),
            pytest.param(7, 10, 70, id="seven-of-ten"),
            pytest.param(19, 20, 95, id="nineteen-of-twenty"),
            # The three below are boundaries float32 sampling actually gets wrong, at
            # three distinct grid indices — 65, 78 and 84. Without them this test would
            # rest on a single unlucky-for-float32 case.
            pytest.param(13, 20, 65, id="thirteen-of-twenty"),
            pytest.param(39, 50, 78, id="thirty-nine-of-fifty"),
            pytest.param(21, 25, 84, id="twenty-one-of-twenty-five"),
        ],
    )
    def test_boundary_recall_is_sampled_exactly(self, found: int, positives: int, grid_index: int) -> None:
        """A recall landing exactly on grid point ``k`` samples it, giving ``(k+1)/101``.

        Every detection here is a perfect copy of a ground truth, so precision is 1 all
        the way along and the envelope is flat. The average is then purely a count of
        sampled grid points: those at or below the attained recall ``found/positives``,
        which is exactly ``k/100``, so ``k + 1`` of the 101 points sample 1 and the rest
        sample 0.

        This is the case floating point gets wrong, and it is not exotic — any class whose
        found-to-annotated ratio reduces to hundredths hits it. Sampling the grid in
        float32 forfeits the ``k``-th point and returns ``k/101``; that was this module's
        behaviour until the comparison was moved to integers, and was ``evaluate_bbox``'s
        for 36 of the 101 boundaries — torchmetrics builds its recall thresholds with a
        float32 ``torch.linspace`` — until WP-092 handed that metric an exact grid through
        its documented ``rec_thresholds`` argument.
        """
        preds, targets = _boundary_case(found, positives)

        stats = evaluate_rotated_map(preds, targets)

        assert stats["map_50"] == pytest.approx((grid_index + 1) / 101, abs=1e-6)
        assert stats["mar_300"] == pytest.approx(found / positives, abs=1e-6)

    def test_agrees_with_the_axis_aligned_instrument_at_a_float32_boundary(self) -> None:
        """At a boundary float32 forfeits, both instruments keep the point (WP-092).

        This pinned a **divergence** until WP-092. ``evaluate_bbox`` reached 13/20 through
        torchmetrics' float32 recall grid, whose 65th entry is ``0.6500000357627869`` —
        above the ``0.65`` the recall actually attains — so it sampled 65 points where
        this module sampled 66, and the oriented instrument was the higher of the two by
        ``1/101``. That test then failed the moment the axis-aligned reporter was handed
        an exact grid, which is precisely the notification it was written to give.

        Equality is asserted rather than a widened band. The two reach ``66/101`` by
        different arithmetic — int64 rationals here, correctly rounded float64 division
        there — and A46's claim is that those agree *exactly* on every input this project
        can produce, not that they agree closely. A reopened gap means one of the two
        grids regressed, and this is where that surfaces.
        """
        preds, targets = _boundary_case(found=13, positives=20)
        box_preds, box_targets = _boundary_case_axis_aligned(found=13, positives=20)

        rotated = evaluate_rotated_map(preds, targets)

        axis_aligned = evaluate_bbox(box_preds, box_targets)
        assert rotated["map_50"] == pytest.approx(66 / 101, abs=1e-6)
        assert axis_aligned["map_50"] == pytest.approx(66 / 101, abs=1e-6)
        assert rotated["map_50"] == pytest.approx(axis_aligned["map_50"], abs=1e-6)


class TestDifficultGroundTruths:
    """Decision 5 — R18's devkit rule, and what it is worth."""

    def test_detection_on_a_difficult_target_is_discarded(self) -> None:
        """A detection matching a difficult instance is neither a true nor a false positive.

        The contrast is the point. With the difficult instance present and flagged, the
        detection that finds it is discarded and the remaining detection scores a perfect
        1.0. With that instance *deleted* from the ground truth instead — the mistake an
        evaluation loader makes by filtering at load — the very same detection becomes a
        top-ranked false positive and the score falls to 0.5.
        """
        difficult_box = [200.0, 200.0, 8.0, 5.0, 0.4]
        plain_box = [10.0, 10.0, 6.0, 4.0, 0.3]
        preds = [
            {
                "rboxes": torch.tensor([difficult_box, plain_box]),
                "scores": torch.tensor([0.9, 0.5]),
                "labels": torch.zeros(2, dtype=torch.long),
            }
        ]
        flagged = [
            {
                "rboxes": torch.tensor([plain_box, difficult_box]),
                "labels": torch.zeros(2, dtype=torch.long),
                "difficult": torch.tensor([False, True]),
            }
        ]
        deleted = [
            {
                "rboxes": torch.tensor([plain_box]),
                "labels": torch.zeros(1, dtype=torch.long),
                "difficult": torch.tensor([False]),
            }
        ]

        stats = evaluate_rotated_map(preds, flagged)

        assert stats["map"] == pytest.approx(1.0)
        assert evaluate_rotated_map(preds, deleted)["map"] == pytest.approx(0.5, abs=1e-6)

    def test_difficult_targets_leave_the_recall_denominator(self) -> None:
        """Recall is measured against non-difficult instances only."""
        found = [10.0, 10.0, 6.0, 4.0, 0.3]
        missed_difficult = [500.0, 500.0, 6.0, 4.0, 0.0]
        preds = [
            {"rboxes": torch.tensor([found]), "scores": torch.tensor([0.9]), "labels": torch.zeros(1, dtype=torch.long)}
        ]
        targets = [
            {
                "rboxes": torch.tensor([found, missed_difficult]),
                "labels": torch.zeros(2, dtype=torch.long),
                "difficult": torch.tensor([False, True]),
            }
        ]

        stats = evaluate_rotated_map(preds, targets)

        # One of two instances found, but the missed one is difficult, so recall is 1.
        assert stats["mar_300"] == pytest.approx(1.0)

    def test_one_difficult_target_may_absorb_several_detections(self) -> None:
        """Difficult instances are not consumed — every detection on one is discarded."""
        difficult_box = [200.0, 200.0, 8.0, 5.0, 0.4]
        plain_box = [10.0, 10.0, 6.0, 4.0, 0.3]
        preds = [
            {
                "rboxes": torch.tensor([difficult_box, difficult_box, difficult_box, plain_box]),
                "scores": torch.tensor([0.9, 0.8, 0.7, 0.6]),
                "labels": torch.zeros(4, dtype=torch.long),
            }
        ]
        targets = [
            {
                "rboxes": torch.tensor([plain_box, difficult_box]),
                "labels": torch.zeros(2, dtype=torch.long),
                "difficult": torch.tensor([False, True]),
            }
        ]

        stats = evaluate_rotated_map(preds, targets)

        # All three discarded; had the second and third counted as false positives the
        # remaining true positive would have been ranked fourth and precision would drop.
        assert stats["map"] == pytest.approx(1.0)

    def test_absent_difficult_key_reads_as_no_instance_difficult(self) -> None:
        """A target dict without a ``difficult`` entry evaluates as all-plain."""
        box = torch.tensor([[10.0, 10.0, 6.0, 4.0, 0.3]])
        preds = [{"rboxes": box, "scores": torch.tensor([0.9]), "labels": torch.zeros(1, dtype=torch.long)}]
        without = [{"rboxes": box, "labels": torch.zeros(1, dtype=torch.long)}]
        with_flags = [{"rboxes": box, "labels": torch.zeros(1, dtype=torch.long), "difficult": torch.tensor([False])}]

        assert evaluate_rotated_map(preds, without) == evaluate_rotated_map(preds, with_flags)


class TestClassExclusion:
    """Decision 4 — classes without non-difficult ground truth leave the mean."""

    def test_class_with_no_ground_truth_is_excluded(self) -> None:
        """Spurious detections of an unannotated class do not drag the mean to zero."""
        box = torch.tensor([[10.0, 10.0, 6.0, 4.0, 0.3]])
        spurious = torch.tensor([[400.0, 400.0, 6.0, 4.0, 0.0]])
        preds = [
            {
                "rboxes": torch.cat([box, spurious]),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 7]),
            }
        ]
        targets = [{"rboxes": box, "labels": torch.tensor([0]), "difficult": torch.tensor([False])}]

        stats = evaluate_rotated_map(preds, targets)

        # Class 7 is excluded rather than scored 0, so the mean is class 0's perfect 1.0.
        assert stats["map"] == pytest.approx(1.0)

    def test_scorable_class_with_no_detections_scores_zero(self) -> None:
        """An annotated class the model never predicts stays in the mean and scores zero.

        The mirror of the exclusion rule: a class is dropped for having no ground truth,
        never for having no detections. Dropping the latter would let a model raise its
        mAP by declining to predict a class it is bad at.
        """
        found = [10.0, 10.0, 6.0, 4.0, 0.3]
        unpredicted = [400.0, 400.0, 6.0, 4.0, 0.0]
        preds = [{"rboxes": torch.tensor([found]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}]
        targets = [
            {
                "rboxes": torch.tensor([found, unpredicted]),
                "labels": torch.tensor([0, 1]),
                "difficult": torch.tensor([False, False]),
            }
        ]

        stats = evaluate_rotated_map(preds, targets)

        # Class 0 scores 1.0 and class 1 scores 0.0, so the mean over both classes is 0.5.
        assert stats["map"] == pytest.approx(0.5)
        assert stats["mar_300"] == pytest.approx(0.5)

    def test_class_whose_targets_are_all_difficult_is_excluded(self) -> None:
        """An all-difficult class has an empty recall denominator, so it leaves the mean."""
        plain = [10.0, 10.0, 6.0, 4.0, 0.3]
        all_difficult = [400.0, 400.0, 6.0, 4.0, 0.0]
        preds = [
            {
                "rboxes": torch.tensor([plain]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "rboxes": torch.tensor([plain, all_difficult]),
                "labels": torch.tensor([0, 3]),
                "difficult": torch.tensor([False, True]),
            }
        ]

        stats = evaluate_rotated_map(preds, targets)

        assert stats["map"] == pytest.approx(1.0)


class TestDetectionCap:
    """Decision 3 — 300 detections per image, applied across classes."""

    def test_detections_beyond_the_cap_are_dropped(self) -> None:
        """A correct detection ranked below 300 is cut and its recall is lost."""
        preds, targets = _perfect_case(count=400, match_index=350)

        stats = evaluate_rotated_map(preds, targets)

        assert stats["mar_300"] == 0.0
        assert stats["map"] == 0.0

    def test_detections_within_the_cap_survive(self) -> None:
        """The same correct detection ranked inside 300 is kept, so the cap is the cause."""
        preds, targets = _perfect_case(count=400, match_index=299)

        stats = evaluate_rotated_map(preds, targets)

        assert stats["mar_300"] == pytest.approx(1.0)

    def test_cap_keeps_the_highest_scoring_rows(self) -> None:
        """The cap ranks by score, so a low-ranked correct row loses to high-ranked noise."""
        kept, _ = _perfect_case(count=MAX_DETECTIONS, match_index=MAX_DETECTIONS - 1)
        _, targets = _perfect_case(count=MAX_DETECTIONS, match_index=MAX_DETECTIONS - 1)

        assert evaluate_rotated_map(kept, targets)["mar_300"] == pytest.approx(1.0)


class TestAdapters:
    """Conversion from the shipped detection tuple and from WP-057's tiled targets."""

    def test_padding_rows_are_dropped(self) -> None:
        """Score-zero padding rows of the fixed-size batch do not reach the accumulator."""
        detections = torch.zeros(2, 4, 7)
        detections[0, 0] = torch.tensor([5.0, 5.0, 4.0, 2.0, 0.3, 0.9, 1.0])
        detections[1, 0] = torch.tensor([7.0, 7.0, 6.0, 3.0, -0.2, 0.4, 2.0])
        detections[1, 1] = torch.tensor([8.0, 8.0, 6.0, 3.0, 0.1, 0.7, 0.0])

        preds = rotated_detections_to_predictions(detections)

        assert [tuple(image["rboxes"].shape) for image in preds] == [(1, 5), (2, 5)]
        assert preds[1]["labels"].tolist() == [2, 0]

    def test_score_floor_filters_low_confidence_rows(self) -> None:
        """Rows at or below the floor are dropped, matching the axis-aligned adapter."""
        detections = torch.zeros(1, 3, 7)
        detections[0, 0] = torch.tensor([5.0, 5.0, 4.0, 2.0, 0.3, 0.9, 1.0])
        detections[0, 1] = torch.tensor([6.0, 6.0, 4.0, 2.0, 0.3, 0.2, 1.0])

        preds = rotated_detections_to_predictions(detections, score_floor=0.5)

        assert preds[0]["scores"].tolist() == [pytest.approx(0.9)]

    def test_adapter_output_feeds_the_accumulator(self) -> None:
        """The adapter's dicts are exactly what :func:`evaluate_rotated_map` consumes."""
        detections = torch.zeros(1, 3, 7)
        detections[0, 0] = torch.tensor([5.0, 5.0, 4.0, 2.0, 0.3, 0.9, 0.0])
        targets = [
            {
                "rboxes": torch.tensor([[5.0, 5.0, 4.0, 2.0, 0.3]]),
                "labels": torch.zeros(1, dtype=torch.long),
                "difficult": torch.zeros(1, dtype=torch.bool),
            }
        ]

        stats = evaluate_rotated_map(rotated_detections_to_predictions(detections), targets)

        assert stats["map"] == pytest.approx(1.0)

    def test_rejects_a_batch_of_the_wrong_width(self) -> None:
        """An axis-aligned ``(B, N, 6)`` batch is rejected rather than misread."""
        with pytest.raises(ValueError, match=r"must be \(B, N, 7\)"):
            rotated_detections_to_predictions(torch.zeros(1, 4, 6))

    def test_tiled_targets_carry_their_difficult_flags(self) -> None:
        """The R18 flag travels with the box it belongs to, on the shared instance axis."""
        targets = Targets(
            boxes=torch.tensor([[0.0, 0.0, 4.0, 2.0], [10.0, 10.0, 6.0, 4.0]]),
            labels=torch.tensor([3, 5]),
            rboxes=torch.tensor([[2.0, 1.0, 4.0, 2.0, 0.0], [13.0, 12.0, 6.0, 4.0, 0.0]]),
        )
        tiled = TiledTargets(targets, torch.tensor([False, True]), torch.tensor([1.0, 0.4]))

        ground_truth = tiled_targets_to_ground_truth(tiled)

        assert ground_truth["labels"].tolist() == [3, 5]
        assert ground_truth["difficult"].tolist() == [False, True]
        assert torch.equal(ground_truth["rboxes"], targets.rboxes)

    def test_tiled_targets_survive_an_empty_tile(self) -> None:
        """A window containing no instance converts to empty tensors of the right shape."""
        tiled = TiledTargets(Targets.empty(), torch.zeros(0, dtype=torch.bool), torch.zeros(0))

        ground_truth = tiled_targets_to_ground_truth(tiled)

        assert tuple(ground_truth["rboxes"].shape) == (0, 5)
        assert ground_truth["difficult"].dtype == torch.bool


def _axis_aligned_fixture() -> tuple[
    list[dict[str, Tensor]], list[dict[str, Tensor]], list[dict[str, Tensor]], list[dict[str, Tensor]]
]:
    """Build one detection problem in both the rotated and the axis-aligned vocabularies.

    Every angle is zero, so the two descriptions denote the same rectangles. Each image
    detects a strict subset of its ground truths (leaving some unmatched), repeats its
    first detection (a duplicate the one-to-one rule must charge), and adds two detections
    of nothing. Class :data:`_STARVED_CLASS` is skipped entirely in every other image, so
    at least one class saturates well below full recall rather than every curve running to
    1. Randomness comes from the autouse seed, so the fixture is fixed run to run.
    """
    rotated_preds, rotated_targets, box_preds, box_targets = [], [], [], []
    for image in range(_FIXTURE_IMAGES):
        count = int(torch.randint(3, 10, (1,)))
        centres = torch.rand(count, 2) * 400.0 + 50.0
        extents = torch.rand(count, 2) * 60.0 + 20.0
        labels = torch.randint(0, _FIXTURE_CLASSES, (count,))
        ground_truth = torch.cat([centres, extents, torch.zeros(count, 1)], dim=1)

        picked = torch.randperm(count)[: max(1, count - 2)]
        if image % 2 == 0:
            picked = picked[labels[picked] != _STARVED_CLASS]
        if picked.numel() == 0:
            picked = torch.randperm(count)[:1]

        jittered = centres[picked] + (torch.rand(picked.numel(), 2) - 0.5) * 24.0
        resized = extents[picked] * (0.85 + torch.rand(picked.numel(), 2) * 0.3)
        picked_labels = labels[picked].clone()

        noise_centres = torch.rand(2, 2) * 400.0 + 50.0
        noise_extents = torch.rand(2, 2) * 40.0 + 15.0
        noise_labels = torch.randint(0, _FIXTURE_CLASSES, (2,))

        all_centres = torch.cat([jittered, jittered[:1] + 1.0, noise_centres])
        all_extents = torch.cat([resized, resized[:1], noise_extents])
        predicted_labels = torch.cat([picked_labels, picked_labels[:1], noise_labels])
        scores = torch.rand(all_centres.shape[0])
        predicted = torch.cat([all_centres, all_extents, torch.zeros(all_centres.shape[0], 1)], dim=1)

        rotated_preds.append({"rboxes": predicted, "scores": scores, "labels": predicted_labels})
        rotated_targets.append(
            {"rboxes": ground_truth, "labels": labels, "difficult": torch.zeros(count, dtype=torch.bool)}
        )
        box_preds.append({"boxes": _to_xyxy(predicted), "scores": scores, "labels": predicted_labels})
        box_targets.append({"boxes": _to_xyxy(ground_truth), "labels": labels})
    return rotated_preds, rotated_targets, box_preds, box_targets


def _boundary_targets(found: int, positives: int) -> tuple[Tensor, Tensor]:
    """Return ``(ground truth, detections)`` rotated boxes for an exact-recall fixture.

    ``positives`` well-separated ground truths of one class, of which the first ``found``
    are returned as exact copies, so every detection is a true positive and the attained
    recall is exactly ``found / positives``.
    """
    ground_truth = torch.tensor([[100.0 * index, 100.0, 20.0, 10.0, 0.0] for index in range(positives)])
    return ground_truth, ground_truth[:found].clone()


def _boundary_case(found: int, positives: int) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Return rotated prediction and target dicts whose recall is exactly ``found/positives``."""
    ground_truth, detections = _boundary_targets(found, positives)
    preds = [
        {
            "rboxes": detections,
            "scores": torch.linspace(0.9, 0.5, found),
            "labels": torch.zeros(found, dtype=torch.long),
        }
    ]
    targets = [
        {
            "rboxes": ground_truth,
            "labels": torch.zeros(positives, dtype=torch.long),
            "difficult": torch.zeros(positives, dtype=torch.bool),
        }
    ]
    return preds, targets


def _boundary_case_axis_aligned(found: int, positives: int) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Return the same exact-recall fixture in the axis-aligned vocabulary."""
    ground_truth, detections = _boundary_targets(found, positives)
    preds = [
        {
            "boxes": _to_xyxy(detections),
            "scores": torch.linspace(0.9, 0.5, found),
            "labels": torch.zeros(found, dtype=torch.long),
        }
    ]
    targets = [{"boxes": _to_xyxy(ground_truth), "labels": torch.zeros(positives, dtype=torch.long)}]
    return preds, targets


def _to_xyxy(rboxes: Tensor) -> Tensor:
    """Return the ``xyxy`` corners of zero-angle rotated boxes."""
    half = rboxes[:, 2:4] / 2
    return torch.cat([rboxes[:, :2] - half, rboxes[:, :2] + half], dim=1)
