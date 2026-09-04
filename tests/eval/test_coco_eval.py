# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-043/WP-069 torchmetrics bbox evaluator and its WP-044 oracle.

Covers the three pieces of :mod:`lucid_yolo.eval.coco_eval`, now driven by
:class:`torchmetrics.detection.MeanAveragePrecision` on its ``faster_coco_eval``
backend (WP-069):

- :func:`test_dual_path_report` (the WP-043 DoD) — a **real** tiny untrained
  detector is run through :class:`~lucid_yolo.eval.coco_eval.DualPathEvaluator`
  over a few detseg fixture images with ground-truth target dicts built from the
  fixture annotation JSON; the report must carry both ``"e2e"`` and ``"nms"``
  keys, each a full 12-metric dict of finite values (an untrained mAP near zero
  is fine — the contract under test is the plumbing, one pass / one checkpoint /
  both paths, not accuracy).
- :func:`detections_to_predictions` unit cases — padding rows dropped, ``xyxy``
  kept as-is, contiguous-label to COCO-category-id mapping, and the score floor.
- :func:`test_evaluate_bbox_perfect_predictions` — a one-image smoke where the
  predictions are exactly the ground-truth boxes, so mAP is 1.0. It is a
  minimal preview of :class:`TestOracleRoundTrip`, kept deliberately small here.
- :class:`TestOracleRoundTrip` (the WP-044 DoD, :func:`TestOracleRoundTrip.test_oracle`)
  — the full-fixture oracle: perfect predictions round-tripped through
  :func:`detections_to_predictions` score mAP 1.0, shuffled-class predictions
  collapse it near 0, a localization-jitter ladder pins the metric's monotonic
  response to box error, wrong-ranked candidate scores degrade AP even with a
  perfect box present, and score-zero padding rows change nothing.
- :class:`TestRecallGridBoundary` (the WP-092 DoD) — the 101-point recall grid at
  the boundaries floating point decides wrongly, asserted against a hand-derived
  ``(k + 1) / 101`` rather than against another implementation, plus the guard
  that fails loudly if torchmetrics ever stops honouring ``rec_thresholds``.

The oracle bands (perfect -> 1.0, small-jitter floor 0.5, shuffled/offset ceiling
0.05, ranking -> 0.5) are unchanged from the WP-044 gate: the
``faster_coco_eval`` backend reimplements COCOeval faithfully, so only the metric
**key names** move (``map50_95`` -> ``map``, ``recall_*`` -> ``mar_*``) and the
ground truth is a target dict rather than a ``COCO`` object.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from torchvision.io import ImageReadMode, read_image

from lucid_yolo.data.coco import CocoDetectionDataset
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.targets import Targets
from lucid_yolo.decode import NMSDecoder, TopKDecoder
from lucid_yolo.eval import DualPathEvaluator, coco_eval, detections_to_predictions, evaluate_bbox
from lucid_yolo.eval.coco_eval import _METRIC_KEYS, _RECALL_GRID
from lucid_yolo.models.heads.detect import DualHeadOutput
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

_SPLIT = "train"
_ANNOTATION = "_annotations.coco.json"
_CANVAS = 128  # detseg fixtures are 128x128; divisible by 32 for the head grid.
_NUM_IMAGES = 3  # a few fixture images exercise the loop without being slow.
_NUM_CLASSES = 4  # the detseg fixture carries category ids 1..4.
_UINT8_MAX = 255.0

_JITTER_SMALL_FRACTION = 0.05  # 5% of box size, per the WP-044 perturbation ladder.
_JITTER_LARGE_FRACTION = 2.0  # a 2x-box-size offset, i.e. a fully missed detection.
_JITTER_SMALL_FLOOR = 0.5  # small jitter must stay above this map50-95 (observed ~0.70).
_SHUFFLED_MAP_CEILING = 0.05  # shuffled-class predictions must stay under this map50-95.
_RANKING_NUM_IMAGES = 3  # keep the score-ranking case small and fast (spec: 2-3 images).
_PAD_ROWS = 3  # extra score-zero rows appended per image for the padding-invariance case.

# Geometry of the exact-recall boundary fixture. The pitch far exceeds the box size, so
# the targets are pairwise disjoint and each detection can only match its own copy —
# which is what makes the attained recall exactly found/positives at every IoU threshold.
_BOUNDARY_PITCH = 100.0
_BOUNDARY_HALF_WIDTH = 10.0
_BOUNDARY_HALF_HEIGHT = 5.0


def _targets_by_image(doc: dict[str, object]) -> dict[int, dict[str, torch.Tensor]]:
    """Build torchmetrics target dicts (``xyxy`` boxes, category-id labels) keyed by image id.

    Every image in the document gets an entry — images without annotations map to
    empty ``(0, 4)`` box / ``(0,)`` label tensors, so any yielded image id resolves.

    Examples:
        >>> ann = {"image_id": 1, "bbox": [0.0, 0.0, 2.0, 3.0], "category_id": 5}
        >>> doc = {"images": [{"id": 1}], "annotations": [ann]}
        >>> targets = _targets_by_image(doc)
        >>> targets[1]["boxes"].tolist()
        [[0.0, 0.0, 2.0, 3.0]]
        >>> targets[1]["labels"].tolist()
        [5]
    """
    boxes: dict[int, list[list[float]]] = {int(image["id"]): [] for image in doc["images"]}  # type: ignore[union-attr]
    labels: dict[int, list[int]] = {int(image["id"]): [] for image in doc["images"]}  # type: ignore[union-attr]
    for ann in doc["annotations"]:  # type: ignore[union-attr]
        x, y, w, h = ann["bbox"]
        boxes[ann["image_id"]].append([x, y, x + w, y + h])
        labels[ann["image_id"]].append(int(ann["category_id"]))
    return {
        image_id: {
            "boxes": torch.tensor(boxes[image_id], dtype=torch.float32) if boxes[image_id] else torch.zeros((0, 4)),
            "labels": torch.tensor(labels[image_id], dtype=torch.long),
        }
        for image_id in boxes
    }


def _eval_batch(
    fixture_dir: Path,
    letterbox: Letterbox,
    num_images: int,
) -> list[tuple[torch.Tensor, list[int], list[tuple[int, int]]]]:
    """Assemble one ``(images, image_ids, orig_sizes)`` batch from fixture images.

    Reads the first ``num_images`` images named in the fixture COCO JSON, scales
    each to ``[0, 1]``, letterboxes it onto the canvas, and stacks them with their
    COCO image ids and original ``(height, width)`` sizes — the batch contract the
    :class:`~lucid_yolo.eval.coco_eval.DualPathEvaluator` consumes.

    Examples:
        >>> callable(_eval_batch)  # needs a real fixture_dir with COCO images/annotations on disk
        True
    """
    split_dir = fixture_dir / _SPLIT
    doc = json.loads((split_dir / _ANNOTATION).read_text(encoding="utf-8"))
    images: list[torch.Tensor] = []
    image_ids: list[int] = []
    orig_sizes: list[tuple[int, int]] = []
    for record in doc["images"][:num_images]:
        raw = read_image(str(split_dir / record["file_name"]), ImageReadMode.RGB).to(torch.float32) / _UINT8_MAX
        letterboxed, _ = letterbox(raw, Targets.empty())
        images.append(letterboxed)
        image_ids.append(int(record["id"]))
        orig_sizes.append((int(record["height"]), int(record["width"])))
    return [(torch.stack(images), image_ids, orig_sizes)]


def test_dual_path_report(detseg_fixture_dir: Path) -> None:
    """A real tiny detector yields a both-paths report of finite 12-metric dicts (DoD)."""
    torch.manual_seed(0)
    split_dir = detseg_fixture_dir / _SPLIT
    dataset = CocoDetectionDataset(split_dir, split_dir / _ANNOTATION)
    letterbox = Letterbox(_CANVAS)
    model = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=_NUM_CLASSES)
    evaluator = DualPathEvaluator(model, TopKDecoder(), NMSDecoder(), dataset.label_to_category_id, letterbox)
    loader = _eval_batch(detseg_fixture_dir, letterbox, _NUM_IMAGES)
    targets = _targets_by_image(_load_fixture_doc(detseg_fixture_dir))

    report = evaluator.evaluate(loader, targets, torch.device("cpu"))

    assert set(report) == {"e2e", "nms"}
    for stats in report.values():
        assert set(stats) == set(_METRIC_KEYS)
        assert all(isinstance(value, float) for value in stats.values())
        assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())


class TestDetectionsToPredictions:
    """Tests for ``detections_to_predictions``."""

    def test_drops_padding_rows(self) -> None:
        """Score-zero padding rows are dropped; only the scored detection survives."""
        detections = torch.tensor([[[1.0, 2.0, 5.0, 6.0, 0.8, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        preds = detections_to_predictions(detections, label_to_category={0: 5})
        assert len(preds) == 1
        assert preds[0]["boxes"].shape == (1, 4)
        assert preds[0]["scores"].tolist() == [pytest.approx(0.8)]

    def test_xyxy_and_category_mapping(self) -> None:
        """``xyxy`` corners are kept as-is and the contiguous label maps to its COCO category id."""
        detections = torch.tensor([[[10.0, 20.0, 40.0, 80.0, 0.5, 2.0]]])
        preds = detections_to_predictions(detections, label_to_category={2: 42})
        assert preds[0]["boxes"].tolist() == [[10.0, 20.0, 40.0, 80.0]]  # unchanged xyxy corners
        assert preds[0]["labels"].tolist() == [42]

    def test_score_floor(self) -> None:
        """A positive score floor drops detections at or below it."""
        detections = torch.tensor([[[0.0, 0.0, 4.0, 4.0, 0.30, 0.0], [0.0, 0.0, 4.0, 4.0, 0.10, 1.0]]])
        preds = detections_to_predictions(detections, label_to_category={0: 1, 1: 2}, score_floor=0.2)
        assert preds[0]["scores"].tolist() == [pytest.approx(0.30)]
        assert preds[0]["labels"].tolist() == [1]


def _single_image_gt() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A one-box prediction that exactly matches its one-box target (score 1.0).

    Examples:
        >>> pred, target = _single_image_gt()
        >>> pred["boxes"].tolist()
        [[10.0, 12.0, 30.0, 42.0]]
        >>> pred["scores"].tolist()
        [1.0]
        >>> target["labels"].tolist()
        [5]
    """
    box = torch.tensor([[10.0, 12.0, 30.0, 42.0]])  # xyxy
    pred = {"boxes": box, "scores": torch.tensor([1.0]), "labels": torch.tensor([5])}
    target = {"boxes": box, "labels": torch.tensor([5])}
    return pred, target


def test_evaluate_bbox_perfect_predictions() -> None:
    """Predictions equal to the ground truth score mAP 1.0 (a WP-044 oracle preview)."""
    pred, target = _single_image_gt()

    stats = evaluate_bbox([pred], [target])

    assert stats["map"] == pytest.approx(1.0)
    assert stats["map_50"] == pytest.approx(1.0)


class _ConstantDecoder(torch.nn.Module):
    """A decoder that ignores the head output and always emits one fixed detection."""

    def __init__(self, box: list[float]) -> None:
        super().__init__()
        self.detection = torch.tensor([[[*box, 1.0, 0.0]]])  # (1, 1, 6): xyxy, score, label

    def forward(self, cls_logits: Tensor, raw_ltrb: Tensor, anchor_points: Tensor, strides: Tensor) -> Tensor:
        """Return the fixed detection, whatever the head predicted."""
        del cls_logits, raw_ltrb, anchor_points, strides
        return self.detection


class _ZeroHead(torch.nn.Module):
    """A model returning an all-zero :class:`DualHeadOutput` with no coefficients."""

    def forward(self, images: Tensor) -> DualHeadOutput:
        """Return dense zeros shaped for a single anchor."""
        del images
        zeros_cls, zeros_box = torch.zeros(1, 1, _NUM_CLASSES), torch.zeros(1, 1, 4)
        return DualHeadOutput(
            o2m_cls=zeros_cls, o2m_box=zeros_box, o2o_cls=zeros_cls, o2o_box=zeros_box, o2m_coeff=None, o2o_coeff=None
        )


def test_each_path_is_scored_from_its_own_decoder() -> None:
    """The ``e2e`` report comes from the E2E decoder and ``nms`` from the NMS decoder.

    Which predictions reach which path's metric cannot be checked with a real
    untrained model: both paths then emit identical detections and both score
    0.0, so crossing them changes nothing observable. Two constant decoders that
    disagree -- one landing exactly on the ground truth, one missing it entirely
    -- separate the paths by construction, and the report has to show 1.0 against
    0.0 in the right order.
    """
    box = [10.0, 12.0, 30.0, 42.0]
    evaluator = DualPathEvaluator(
        _ZeroHead(),
        _ConstantDecoder(box),
        _ConstantDecoder([100.0, 100.0, 120.0, 130.0]),
        {0: 5},
        Letterbox(_CANVAS),
    )
    target = {"boxes": torch.tensor([box]), "labels": torch.tensor([5])}
    batches = [(torch.zeros(1, 3, _CANVAS, _CANVAS), [1], [(_CANVAS, _CANVAS)])]

    report = evaluator.evaluate(batches, {1: target}, torch.device("cpu"))

    assert report["e2e"]["map"] == pytest.approx(1.0)
    assert report["nms"]["map"] == pytest.approx(0.0)


def test_evaluate_bbox_empty_preds_is_zeroed() -> None:
    """An empty prediction list yields an all-zero stat dict rather than a metric crash."""
    stats = evaluate_bbox([], [])

    assert set(stats) == set(_METRIC_KEYS)
    assert all(value == 0.0 for value in stats.values())


def _load_fixture_doc(fixture_dir: Path) -> dict[str, object]:
    """Load the full detseg fixture COCO annotation document as a dict.

    Examples:
        >>> callable(_load_fixture_doc)  # needs a real fixture_dir; reads a COCO json file from disk
        True
    """
    return json.loads((fixture_dir / _SPLIT / _ANNOTATION).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _image_order(targets: dict[int, dict[str, torch.Tensor]]) -> list[int]:
    """Deterministic image-id order shared by target and prediction lists.

    Examples:
        >>> _image_order({3: {}, 1: {}, 2: {}})
        [1, 2, 3]
    """
    return sorted(targets)


def _targets_list(targets: dict[int, dict[str, torch.Tensor]], order: list[int]) -> list[dict[str, torch.Tensor]]:
    """Materialize the per-image target dicts in ``order``.

    Examples:
        >>> targets = {1: {"boxes": torch.zeros(0, 4)}, 2: {"boxes": torch.ones(1, 4)}}
        >>> _targets_list(targets, [2, 1])[0]["boxes"].shape
        torch.Size([1, 4])
    """
    return [targets[image_id] for image_id in order]


def _perfect_preds(target_list: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Score-1.0 predictions sitting exactly on each image's target boxes.

    Examples:
        >>> target_list = [{"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]), "labels": torch.tensor([1])}]
        >>> preds = _perfect_preds(target_list)
        >>> preds[0]["scores"].tolist()
        [1.0]
        >>> torch.equal(preds[0]["boxes"], target_list[0]["boxes"])
        True
    """
    return [
        {"boxes": target["boxes"], "scores": torch.ones(len(target["boxes"])), "labels": target["labels"]}
        for target in target_list
    ]


def _jittered_preds(target_list: list[dict[str, torch.Tensor]], fraction: float) -> list[dict[str, torch.Tensor]]:
    """Predictions with every box shifted by ``fraction`` of its own width/height.

    Examples:
        >>> target_list = [{"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]), "labels": torch.tensor([1])}]
        >>> _jittered_preds(target_list, 0.5)[0]["boxes"].tolist()
        [[1.0, 1.0, 3.0, 3.0]]
    """
    preds: list[dict[str, torch.Tensor]] = []
    for target in target_list:
        boxes = target["boxes"].clone()
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        boxes[:, 0] += fraction * widths
        boxes[:, 2] += fraction * widths
        boxes[:, 1] += fraction * heights
        boxes[:, 3] += fraction * heights
        preds.append({"boxes": boxes, "scores": torch.ones(len(boxes)), "labels": target["labels"]})
    return preds


def _shuffled_preds(
    target_list: list[dict[str, torch.Tensor]],
    category_ids: list[int],
) -> list[dict[str, torch.Tensor]]:
    """Correct boxes with every label cyclically shifted to the next (wrong) category id.

    Examples:
        >>> target_list = [{"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]), "labels": torch.tensor([5])}]
        >>> _shuffled_preds(target_list, [5, 7])[0]["labels"].tolist()
        [7]
    """
    shift = {cid: category_ids[(index + 1) % len(category_ids)] for index, cid in enumerate(category_ids)}
    preds: list[dict[str, torch.Tensor]] = []
    for target in target_list:
        wrong = torch.tensor([shift[int(label)] for label in target["labels"]], dtype=torch.long)
        assert torch.all(wrong != target["labels"])  # every label actually changes
        preds.append({"boxes": target["boxes"], "scores": torch.ones(len(wrong)), "labels": wrong})
    return preds


def _ranked_preds(target_list: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Two candidates per box: the correct box at 0.9, a one-box-offset wrong box at 0.95.

    The wrong candidate has zero IoU with the target yet outranks the correct one on
    score, so a ranking-aware metric must be dragged down by it even though the correct
    box is present.

    Examples:
        >>> target_list = [{"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]), "labels": torch.tensor([1])}]
        >>> preds = _ranked_preds(target_list)
        >>> preds[0]["boxes"].shape
        torch.Size([2, 4])
        >>> [round(score, 4) for score in preds[0]["scores"].tolist()]
        [0.9, 0.95]
    """
    preds: list[dict[str, torch.Tensor]] = []
    for target in target_list:
        boxes = target["boxes"]
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        wrong = boxes.clone()
        wrong[:, 0] += widths
        wrong[:, 2] += widths
        wrong[:, 1] += heights
        wrong[:, 3] += heights
        preds.append(
            {
                "boxes": torch.cat([boxes, wrong]),
                "scores": torch.cat([torch.full((len(boxes),), 0.9), torch.full((len(wrong),), 0.95)]),
                "labels": torch.cat([target["labels"], target["labels"]]),
            }
        )
    return preds


def _oracle_preds(
    doc: dict[str, object],
    order: list[int],
    category_id_to_label: dict[int, int],
    label_to_category: dict[int, int],
    pad_rows: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """Round-trip every gt annotation through the A9 tensor -> :func:`detections_to_predictions` path.

    Each image's annotations become one perfect-score ``(1, N + pad_rows, 6)`` detections
    tensor (the true box, its true category as a contiguous label, score 1.0), with
    ``pad_rows`` all-zero rows appended to mimic a fixed-size decoder's padding; the batch
    is decoded back into per-image prediction dicts in ``order``.

    Examples:
        >>> doc = {"annotations": [{"image_id": 1, "bbox": [0.0, 0.0, 2.0, 3.0], "category_id": 5}]}
        >>> preds = _oracle_preds(doc, [1], {5: 0}, {0: 5})
        >>> preds[0]["boxes"].tolist()
        [[0.0, 0.0, 2.0, 3.0]]
        >>> preds[0]["labels"].tolist()
        [5]
    """
    anns_by_image: dict[int, list[dict[str, object]]] = {image_id: [] for image_id in order}
    for ann in doc["annotations"]:  # type: ignore[union-attr]
        anns_by_image[ann["image_id"]].append(ann)
    preds: list[dict[str, torch.Tensor]] = []
    for image_id in order:
        rows = []
        for ann in anns_by_image[image_id]:
            x, y, w, h = ann["bbox"]
            label = category_id_to_label[ann["category_id"]]
            rows.append([x, y, x + w, y + h, 1.0, float(label)])
        rows.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(pad_rows))
        detections = torch.tensor([rows], dtype=torch.float32)
        preds.extend(detections_to_predictions(detections, label_to_category))
    return preds


class TestOracleRoundTrip:
    """WP-044 oracle round-trip: predictions of known quality bound the metric predictably."""

    def test_oracle(self, detseg_fixture_dir: Path) -> None:
        """Perfect predictions, round-tripped through the full pipeline, score mAP 1.0 (DoD)."""
        split_dir = detseg_fixture_dir / _SPLIT
        dataset = CocoDetectionDataset(split_dir, split_dir / _ANNOTATION)
        doc = _load_fixture_doc(detseg_fixture_dir)
        targets = _targets_by_image(doc)
        order = _image_order(targets)

        preds = _oracle_preds(doc, order, dataset.category_id_to_label, dataset.label_to_category_id)
        stats = evaluate_bbox(preds, _targets_list(targets, order))

        # Observed: exact 1.0 at this scale/precision (128px canvas, float32) -- no backend
        # quantization drift was seen. Kept as a ">" bound per spec rather than "==", since
        # quantization is a documented caveat of the instrument, not a guarantee to pin down.
        assert stats["map"] > 0.999
        assert stats["map_50"] == pytest.approx(1.0)
        assert stats["map_75"] == pytest.approx(1.0)
        # mar_10/mar_100 saturate. mar_1 does not (observed ~0.50): several fixture images
        # carry more than one annotation, and AR@1 credits only the single highest-scoring
        # detection per image, so it is structurally capped well below 1.0 regardless of
        # prediction quality -- not asserted here for that reason.
        assert stats["mar_10"] == pytest.approx(1.0)
        assert stats["mar_100"] == pytest.approx(1.0)

    def test_oracle_shuffled_classes(self, detseg_fixture_dir: Path) -> None:
        """Correct boxes with every label cyclically shifted to a wrong class collapse mAP."""
        doc = _load_fixture_doc(detseg_fixture_dir)
        targets = _targets_by_image(doc)
        order = _image_order(targets)
        category_ids = sorted({int(category["id"]) for category in doc["categories"]})  # type: ignore[union-attr]

        preds = _shuffled_preds(_targets_list(targets, order), category_ids)
        stats = evaluate_bbox(preds, _targets_list(targets, order))

        assert stats["map"] < _SHUFFLED_MAP_CEILING

    def test_perturbation_ladder_localization_sensitivity(self, detseg_fixture_dir: Path) -> None:
        """mAP degrades monotonically from exact boxes to small jitter to a full offset."""
        doc = _load_fixture_doc(detseg_fixture_dir)
        targets = _targets_by_image(doc)
        order = _image_order(targets)
        target_list = _targets_list(targets, order)

        exact = evaluate_bbox(_jittered_preds(target_list, 0.0), target_list)["map"]
        jittered = evaluate_bbox(_jittered_preds(target_list, _JITTER_SMALL_FRACTION), target_list)["map"]
        offset = evaluate_bbox(_jittered_preds(target_list, _JITTER_LARGE_FRACTION), target_list)["map"]

        assert exact == pytest.approx(1.0)
        assert _JITTER_SMALL_FLOOR < jittered < exact
        assert offset < jittered
        assert offset < _SHUFFLED_MAP_CEILING

    def test_score_ranking_degrades_ap(self, detseg_fixture_dir: Path) -> None:
        """A higher-scored, wrong-location candidate outranks the correct one and halves AP."""
        doc = _load_fixture_doc(detseg_fixture_dir)
        targets = _targets_by_image(doc)
        order = _image_order(targets)[:_RANKING_NUM_IMAGES]
        target_list = _targets_list(targets, order)

        perfect_map = evaluate_bbox(_perfect_preds(target_list), target_list)["map"]
        ranked_map = evaluate_bbox(_ranked_preds(target_list), target_list)["map"]

        assert perfect_map == pytest.approx(1.0)
        assert ranked_map < perfect_map
        assert ranked_map == pytest.approx(0.5)

    def test_detections_padding_invariance(self, detseg_fixture_dir: Path) -> None:
        """Score-zero padding rows appended to each image's detections change nothing."""
        split_dir = detseg_fixture_dir / _SPLIT
        dataset = CocoDetectionDataset(split_dir, split_dir / _ANNOTATION)
        doc = _load_fixture_doc(detseg_fixture_dir)
        targets = _targets_by_image(doc)
        order = _image_order(targets)
        target_list = _targets_list(targets, order)

        unpadded = _oracle_preds(doc, order, dataset.category_id_to_label, dataset.label_to_category_id)
        padded = _oracle_preds(
            doc, order, dataset.category_id_to_label, dataset.label_to_category_id, pad_rows=_PAD_ROWS
        )

        assert evaluate_bbox(padded, target_list) == evaluate_bbox(unpadded, target_list)


def _boundary_case(found: int, positives: int) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Return prediction and target dicts whose attained recall is exactly ``found / positives``.

    ``positives`` well-separated single-class targets, of which the first ``found`` are
    returned as exact copies. Every detection is therefore a true positive at every IoU
    threshold, precision is 1 all the way along, and the interpolated curve is flat — so
    the average is purely a count of sampled grid points and can be derived by hand.

    Examples:
        >>> preds, targets = _boundary_case(2, 3)
        >>> preds[0]["boxes"].shape
        torch.Size([2, 4])
        >>> targets[0]["boxes"].shape
        torch.Size([3, 4])
    """
    centres = torch.tensor([[_BOUNDARY_PITCH * index, _BOUNDARY_PITCH] for index in range(positives)])
    half = torch.tensor([_BOUNDARY_HALF_WIDTH, _BOUNDARY_HALF_HEIGHT])
    boxes = torch.cat([centres - half, centres + half], dim=1)
    preds = [
        {
            "boxes": boxes[:found].clone(),
            "scores": torch.linspace(0.9, 0.5, found),
            "labels": torch.zeros(found, dtype=torch.long),
        }
    ]
    targets = [{"boxes": boxes, "labels": torch.zeros(positives, dtype=torch.long)}]
    return preds, targets


class _IgnoresRecThresholds(MeanAveragePrecision):
    """A metric that silently drops ``rec_thresholds`` — the regression the guard exists for."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.pop("rec_thresholds", None)
        super().__init__(**kwargs)  # type: ignore[arg-type]


class TestRecallGridBoundary:
    """A46 on the axis-aligned instrument: the 101-point grid where float32 decides it wrongly."""

    @pytest.mark.parametrize(
        ("found", "positives", "grid_index"),
        [
            # Boundaries torchmetrics' float32 grid forfeits — the cases this WP fixes.
            # Five distinct indices, so no single unlucky-for-float32 case carries the test.
            pytest.param(7, 50, 14, id="seven-of-fifty"),
            pytest.param(7, 25, 28, id="seven-of-twenty-five"),
            pytest.param(13, 20, 65, id="thirteen-of-twenty"),
            pytest.param(39, 50, 78, id="thirty-nine-of-fifty"),
            pytest.param(21, 25, 84, id="twenty-one-of-twenty-five"),
            # Boundaries float32 already happens to get right — controls that must not move.
            pytest.param(1, 2, 50, id="half-of-two"),
            pytest.param(3, 4, 75, id="three-of-four"),
            pytest.param(19, 20, 95, id="nineteen-of-twenty"),
        ],
    )
    def test_boundary_recall_is_sampled_exactly(self, found: int, positives: int, grid_index: int) -> None:
        """A recall landing exactly on grid point ``k`` samples it, giving ``(k + 1) / 101``.

        The expected value is derived by hand, not read off another implementation: the
        envelope is flat at precision 1, the attained recall is exactly ``k / 100``, so
        ``k + 1`` of the 101 points sample 1 and the remaining ones sample 0.

        The tolerance is set by the metric's **output dtype**, not chosen to make the
        assertion pass: ``MeanAveragePrecision`` returns float32 tensors, in which
        ``66 / 101`` differs from the exact quotient by about 3e-8. The defect being
        detected is a whole grid point, ``1 / 101`` ~ 9.9e-3 — five orders of magnitude
        larger than the tolerance, so it cannot hide inside it.
        """
        preds, targets = _boundary_case(found, positives)

        stats = evaluate_bbox(preds, targets)

        assert stats["map_50"] == pytest.approx((grid_index + 1) / 101, abs=1e-6)
        assert stats["mar_100"] == pytest.approx(found / positives, abs=1e-6)

    def test_the_torchmetrics_default_grid_overshoots_at_thirty_six_boundaries(self) -> None:
        """Pin the dependency defect this module works around, so its removal is noticed.

        ``rec_thresholds=None`` makes torchmetrics build the grid with a **float32**
        ``torch.linspace``; at 36 of the 101 indices the stored value is strictly greater
        than the ``k / 100`` it represents. Should a future torchmetrics build that grid
        in float64 — or exactly — this test fails, which is the notification worth having:
        the workaround in :func:`~lucid_yolo.eval.coco_eval._new_metric` could then go.
        """
        default = torch.linspace(0.0, 1.00, round(1.00 / 0.01) + 1).tolist()

        overshoot = {index for index in range(len(_RECALL_GRID)) if default[index] > _RECALL_GRID[index]}

        assert len(overshoot) == 36
        assert {14, 28, 65, 78, 84} <= overshoot  # the five indices parametrized above
        assert not any(value > index / 100 for index, value in enumerate(_RECALL_GRID))

    def test_guard_fires_when_the_metric_stops_honouring_rec_thresholds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A metric that accepts ``rec_thresholds`` and ignores it must fail loudly, not silently.

        The fix rests on a documented constructor parameter, so a future torchmetrics that
        **removed** it would raise ``TypeError`` at construction on its own. The dangerous
        regression is the quiet one — the parameter still accepted, no longer honoured —
        which would restore the downward bias with nothing to show for it. This substitutes
        exactly that metric and requires the guard to raise.
        """
        monkeypatch.setattr(coco_eval, "MeanAveragePrecision", _IgnoresRecThresholds)
        pred, target = _single_image_gt()

        with pytest.raises(RuntimeError, match="rec_thresholds"):
            evaluate_bbox([pred], [target])


def _synthetic_bbox_case(num_images: int = 6, seed: int = 0) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Build a small, informative bbox case: several images, a few classes, some jitter.

    Neither trivially perfect nor trivially empty — some predictions match their
    ground truth closely, some are offset, some ranked wrong — so the two engines
    being compared actually exercise matching, ranking and the recall grid rather
    than only the degenerate all-1.0/all-0.0 corners.

    Examples:
        >>> preds, targets = _synthetic_bbox_case(num_images=2)
        >>> len(preds), len(targets)
        (2, 2)
        >>> sorted(preds[0])
        ['boxes', 'labels', 'scores']
    """
    generator = torch.Generator().manual_seed(seed)
    preds: list[dict[str, Tensor]] = []
    targets: list[dict[str, Tensor]] = []
    for image_index in range(num_images):
        count = 2 + image_index % 3
        top_left = torch.rand(count, 2, generator=generator) * 400.0
        size = torch.rand(count, 2, generator=generator) * 80.0 + 20.0
        gt_boxes = torch.cat([top_left, top_left + size], dim=1)
        jitter = (torch.rand(count, 4, generator=generator) - 0.5) * 6.0
        pred_boxes = gt_boxes + jitter
        labels = torch.randint(0, 4, (count,), generator=generator)
        scores = torch.rand(count, generator=generator) * 0.5 + 0.5
        targets.append({"boxes": gt_boxes, "labels": labels})
        preds.append({"boxes": pred_boxes, "scores": scores, "labels": labels})
    return preds, targets


def _synthetic_segm_case(num_images: int = 4, seed: int = 0) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """The bbox case's twin with masks: same boxes, plus a dense patch per instance, plus jitter.

    Examples:
        >>> preds, targets = _synthetic_segm_case(num_images=2)
        >>> len(preds), len(targets)
        (2, 2)
        >>> sorted(preds[0])
        ['boxes', 'labels', 'masks', 'scores']
    """
    canvas = 96
    generator = torch.Generator().manual_seed(seed)
    preds: list[dict[str, Tensor]] = []
    targets: list[dict[str, Tensor]] = []
    for image_index in range(num_images):
        count = 2 + image_index % 3
        origins = (torch.rand(count, 2, generator=generator) * (canvas - 24)).to(torch.long)
        labels = torch.randint(0, 3, (count,), generator=generator)
        scores = torch.rand(count, generator=generator) * 0.5 + 0.5
        gt_masks = torch.zeros(count, canvas, canvas, dtype=torch.bool)
        pred_masks = torch.zeros(count, canvas, canvas, dtype=torch.bool)
        boxes = torch.zeros(count, 4)
        for instance, (y0, x0) in enumerate(origins.tolist()):
            gt_masks[instance, y0 : y0 + 20, x0 : x0 + 20] = True
            dy, dx = (
                int(torch.randint(-2, 3, (1,), generator=generator)),
                int(torch.randint(-2, 3, (1,), generator=generator)),
            )
            py0, px0 = max(0, y0 + dy), max(0, x0 + dx)
            pred_masks[instance, py0 : py0 + 20, px0 : px0 + 20] = True
            boxes[instance] = torch.tensor([x0, y0, x0 + 20.0, y0 + 20.0])
        targets.append({"boxes": boxes, "labels": labels, "masks": gt_masks})
        preds.append({"boxes": boxes, "scores": scores, "labels": labels, "masks": pred_masks})
    return preds, targets


def _crowd_and_area_case() -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """One image whose targets carry a crowd instance and an ``area`` that is not the box area.

    The two fields COCO reads off the annotation rather than off the box, in the
    one arrangement where both change the answer: the second instance is
    ``iscrowd=1``, so the detection sitting inside it is ignored rather than
    counted a false positive, and the first instance's ``area`` of 200 puts it in
    the *small* bucket although its box measures 200x200. Rebuilding either field
    from the box moves ``map`` and moves every size bucket.

    Examples:
        >>> preds, targets = _crowd_and_area_case()
        >>> targets[0]["iscrowd"].tolist(), targets[0]["area"].tolist()
        ([0, 1], [200.0, 10000.0])
    """
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 200.0, 200.0], [400.0, 400.0, 500.0, 500.0]]),
            "labels": torch.tensor([1, 1]),
            "iscrowd": torch.tensor([0, 1]),
            "area": torch.tensor([200.0, 10000.0]),
        }
    ]
    preds = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 200.0, 200.0], [410.0, 410.0, 460.0, 460.0]]),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([1, 1]),
        }
    ]
    return preds, targets


def _crowd_and_area_masks_case() -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """The crowd/area case's twin with masks: the same instances, plus a filled patch for each.

    The mask pass reads ``iscrowd`` and ``area`` off the annotation exactly as the
    box pass does, and gets its own fallback (the RLE's own pixel count) when they
    are absent, so it needs its own case rather than inheriting the bbox one's
    coverage.

    Examples:
        >>> preds, targets = _crowd_and_area_masks_case()
        >>> targets[0]["masks"].shape
        torch.Size([2, 500, 500])
    """
    preds, targets = _crowd_and_area_case()
    canvas = 500
    target_masks = torch.zeros(2, canvas, canvas, dtype=torch.bool)
    target_masks[0, :200, :200] = True
    target_masks[1, 400:, 400:] = True
    prediction_masks = torch.zeros(2, canvas, canvas, dtype=torch.bool)
    prediction_masks[0, :200, :200] = True
    prediction_masks[1, 410:460, 410:460] = True
    targets[0]["masks"] = target_masks
    preds[0]["masks"] = prediction_masks
    return preds, targets


class TestHotcocoIsUsableHere:
    """The canary under the parity class's ``skipif`` (WP-174)."""

    def test_hotcoco_is_available_and_working(self) -> None:
        """Hotcoco imports and its compiled extension constructs, unconditionally.

        Every parity case below is guarded by ``skipif`` on this same probe, so if hotcoco
        stops importing the whole class evaporates and the gate stays green — the failure
        mode a guarded suite has by construction. hotcoco is not optional here: it is a
        hard runtime dependency and ``lucid-eval``'s default bbox/segm engine, and three of
        the five compatibility traps recorded in ``pyproject.toml`` "fail silently rather
        than raising". So the probe returning ``True`` is itself a contract, asserted
        without a guard, and the reason string is carried into the failure message because
        "hotcoco is missing" and "hotcoco is built but broken" need different remedies.
        """
        probe = coco_eval.hotcoco_available()

        assert probe == (True, None), f"hotcoco is a hard dependency but is unusable here: {probe[1]}"


@pytest.mark.skipif(not coco_eval.hotcoco_available()[0], reason="hotcoco not installed or not usable here")
class TestHotcocoParity:
    """Hotcoco's streaming scorer must report exactly what faster_coco_eval's does (WP-138).

    hotcoco is opt-in (``DualPathEvaluator(..., backend="hotcoco")``), never the
    silent default of the library path — a caller who does not ask for it gets
    exactly the ``faster_coco_eval`` report this project has always produced. The
    contract this class exists to pin is narrower and stricter: given the
    *identical* accumulated predictions and targets, the two engines must not
    merely agree in spirit, they must report the same numbers under the same
    key names, because a report's ``eval_backend`` field is metadata about how
    it was computed, not part of what it means.
    """

    def test_bbox_only_matches_faster_coco_eval(self) -> None:
        """Every one of the 12 bbox statistics matches exactly, on a mixed-quality case."""
        preds, targets = _synthetic_bbox_case()

        faster = coco_eval._StreamingScorer()
        faster.update(preds, targets)
        hotcoco_scorer = coco_eval._HotcocoStreamingScorer()
        hotcoco_scorer.update(preds, targets)

        faster_stats = faster.compute()
        hotcoco_stats = hotcoco_scorer.compute()
        assert set(hotcoco_stats) == set(_METRIC_KEYS)
        for key in _METRIC_KEYS:
            assert hotcoco_stats[key] == pytest.approx(faster_stats[key], abs=1e-6), key

    def test_bbox_and_segm_matches_faster_coco_eval(self) -> None:
        """All 24 bbox+segm statistics match exactly, including the ``segm_``-prefixed half."""
        preds, targets = _synthetic_segm_case()

        faster = coco_eval._StreamingScorer()
        faster.update(preds, targets)
        hotcoco_scorer = coco_eval._HotcocoStreamingScorer()
        hotcoco_scorer.update(preds, targets)

        faster_stats = faster.compute()
        hotcoco_stats = hotcoco_scorer.compute()
        expected_keys = {*_METRIC_KEYS, *(coco_eval._SEGM_PREFIX + key for key in _METRIC_KEYS)}
        assert set(hotcoco_stats) == expected_keys
        for key in expected_keys:
            assert hotcoco_stats[key] == pytest.approx(faster_stats[key], abs=1e-6), key

    def test_crowd_and_annotation_area_match_faster_coco_eval(self) -> None:
        """A crowd instance and a non-box ``area`` are honoured identically by both engines (WP-166).

        The parity cases above omit ``iscrowd`` and ``area`` entirely, so both
        engines fall back to the same reconstruction and agree by construction on
        the only two fields where they can differ. This case supplies both:
        crowd-ignore decides ``map``, and the annotation's own ``area`` decides
        which of the three size buckets the instance lands in, so a scorer that
        rebuilds either field from the box reports a different protocol here.
        """
        preds, targets = _crowd_and_area_case()

        faster = coco_eval._StreamingScorer()
        faster.update(preds, targets)
        hotcoco_scorer = coco_eval._HotcocoStreamingScorer()
        hotcoco_scorer.update(preds, targets)

        faster_stats = faster.compute()
        hotcoco_stats = hotcoco_scorer.compute()
        for key in _METRIC_KEYS:
            assert hotcoco_stats[key] == pytest.approx(faster_stats[key], abs=1e-6), key

    def test_crowd_and_annotation_area_match_across_masks(self) -> None:
        """The mask pass honours the same two annotation fields the box pass does (WP-166).

        ``_accumulate_masks`` builds its own ground-truth document with its own
        area fallback — the RLE's pixel count rather than the box's extent — so a
        crowd instance or a non-box ``area`` can be dropped on the segm side while
        the bbox side is already correct. This drives the same fixture with masks
        attached and holds all 24 statistics to the parity contract.
        """
        preds, targets = _crowd_and_area_masks_case()

        faster = coco_eval._StreamingScorer()
        faster.update(preds, targets)
        hotcoco_scorer = coco_eval._HotcocoStreamingScorer()
        hotcoco_scorer.update(preds, targets)

        faster_stats = faster.compute()
        hotcoco_stats = hotcoco_scorer.compute()
        for key in (*_METRIC_KEYS, *(coco_eval._SEGM_PREFIX + key for key in _METRIC_KEYS)):
            assert hotcoco_stats[key] == pytest.approx(faster_stats[key], abs=1e-6), key

    def test_empty_preds_is_zeroed_like_the_faster_coco_eval_path(self) -> None:
        """No batch ever fed the scorer -> the same all-zero dict :func:`evaluate_bbox` returns."""
        scorer = coco_eval._HotcocoStreamingScorer()

        assert scorer.compute() == dict.fromkeys(_METRIC_KEYS, 0.0)

    def test_dual_path_report_matches_across_backends(self, detseg_fixture_dir: Path) -> None:
        """The same checkpoint, same batches, scored end to end by each backend: identical report.

        The unit-level parity tests above feed one hand-built batch straight to
        each scorer; this drives the whole :class:`DualPathEvaluator` — the real
        decode, the real letterbox, several batches — the same shape
        :func:`test_dual_path_report` already exercises, to catch anything that
        only surfaces once real decoded detections (not a hand-built fixture)
        reach the RLE/height-width/params machinery above.
        """
        torch.manual_seed(0)
        split_dir = detseg_fixture_dir / _SPLIT
        dataset = CocoDetectionDataset(split_dir, split_dir / _ANNOTATION)
        letterbox = Letterbox(_CANVAS)
        model = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=_NUM_CLASSES)
        targets = _targets_by_image(_load_fixture_doc(detseg_fixture_dir))

        faster_evaluator = DualPathEvaluator(
            model, TopKDecoder(), NMSDecoder(), dataset.label_to_category_id, letterbox
        )
        faster_report = faster_evaluator.evaluate(
            _eval_batch(detseg_fixture_dir, letterbox, _NUM_IMAGES), targets, torch.device("cpu")
        )
        hotcoco_evaluator = DualPathEvaluator(
            model, TopKDecoder(), NMSDecoder(), dataset.label_to_category_id, letterbox, backend="hotcoco"
        )
        hotcoco_report = hotcoco_evaluator.evaluate(
            _eval_batch(detseg_fixture_dir, letterbox, _NUM_IMAGES), targets, torch.device("cpu")
        )

        for path in ("e2e", "nms"):
            for key in _METRIC_KEYS:
                assert hotcoco_report[path][key] == pytest.approx(faster_report[path][key], abs=1e-6), (path, key)

    def test_an_unknown_backend_is_refused_at_construction(self) -> None:
        """A misspelled ``backend`` raises, naming both arms, rather than silently defaulting."""
        module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=2).eval()

        with pytest.raises(ValueError, match="backend must be one of"):
            DualPathEvaluator(module, TopKDecoder(), NMSDecoder(), {0: 1}, Letterbox(64), backend="pycocotools")
