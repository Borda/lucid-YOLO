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
from torchvision.io import ImageReadMode, read_image

from lucid_yolo.data.coco import CocoDetectionDataset
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.targets import Targets
from lucid_yolo.decode import NMSDecoder, TopKDecoder
from lucid_yolo.eval import DualPathEvaluator, detections_to_predictions, evaluate_bbox
from lucid_yolo.eval.coco_eval import _METRIC_KEYS
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


def _targets_by_image(doc: dict[str, object]) -> dict[int, dict[str, torch.Tensor]]:
    """Build torchmetrics target dicts (``xyxy`` boxes, category-id labels) keyed by image id.

    Every image in the document gets an entry — images without annotations map to
    empty ``(0, 4)`` box / ``(0,)`` label tensors, so any yielded image id resolves.
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


def test_detections_to_predictions_drops_padding_rows() -> None:
    """Score-zero padding rows are dropped; only the scored detection survives."""
    detections = torch.tensor([[[1.0, 2.0, 5.0, 6.0, 0.8, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    preds = detections_to_predictions(detections, label_to_category={0: 5})
    assert len(preds) == 1
    assert preds[0]["boxes"].shape == (1, 4)
    assert preds[0]["scores"].tolist() == [pytest.approx(0.8)]


def test_detections_to_predictions_xyxy_and_category_mapping() -> None:
    """``xyxy`` corners are kept as-is and the contiguous label maps to its COCO category id."""
    detections = torch.tensor([[[10.0, 20.0, 40.0, 80.0, 0.5, 2.0]]])
    preds = detections_to_predictions(detections, label_to_category={2: 42})
    assert preds[0]["boxes"].tolist() == [[10.0, 20.0, 40.0, 80.0]]  # unchanged xyxy corners
    assert preds[0]["labels"].tolist() == [42]


def test_detections_to_predictions_score_floor() -> None:
    """A positive score floor drops detections at or below it."""
    detections = torch.tensor([[[0.0, 0.0, 4.0, 4.0, 0.30, 0.0], [0.0, 0.0, 4.0, 4.0, 0.10, 1.0]]])
    preds = detections_to_predictions(detections, label_to_category={0: 1, 1: 2}, score_floor=0.2)
    assert preds[0]["scores"].tolist() == [pytest.approx(0.30)]
    assert preds[0]["labels"].tolist() == [1]


def _single_image_gt() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A one-box prediction that exactly matches its one-box target (score 1.0)."""
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


def test_evaluate_bbox_empty_preds_is_zeroed() -> None:
    """An empty prediction list yields an all-zero stat dict rather than a metric crash."""
    stats = evaluate_bbox([], [])

    assert set(stats) == set(_METRIC_KEYS)
    assert all(value == 0.0 for value in stats.values())


def _load_fixture_doc(fixture_dir: Path) -> dict[str, object]:
    """Load the full detseg fixture COCO annotation document as a dict."""
    return json.loads((fixture_dir / _SPLIT / _ANNOTATION).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _image_order(targets: dict[int, dict[str, torch.Tensor]]) -> list[int]:
    """Deterministic image-id order shared by target and prediction lists."""
    return sorted(targets)


def _targets_list(targets: dict[int, dict[str, torch.Tensor]], order: list[int]) -> list[dict[str, torch.Tensor]]:
    """Materialize the per-image target dicts in ``order``."""
    return [targets[image_id] for image_id in order]


def _perfect_preds(target_list: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Score-1.0 predictions sitting exactly on each image's target boxes."""
    return [
        {"boxes": target["boxes"], "scores": torch.ones(len(target["boxes"])), "labels": target["labels"]}
        for target in target_list
    ]


def _jittered_preds(target_list: list[dict[str, torch.Tensor]], fraction: float) -> list[dict[str, torch.Tensor]]:
    """Predictions with every box shifted by ``fraction`` of its own width/height."""
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
    """Correct boxes with every label cyclically shifted to the next (wrong) category id."""
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
