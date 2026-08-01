# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-043 pycocotools bbox evaluator and its WP-044 oracle.

Covers the three pieces of :mod:`lit_yolo.eval.coco_eval`:

- :func:`test_dual_path_report` (the WP-043 DoD) — a **real** tiny untrained
  detector is run through :class:`~lit_yolo.eval.coco_eval.DualPathEvaluator`
  over a few detseg fixture images with a ground-truth
  :class:`pycocotools.coco.COCO` built from the fixture annotation JSON; the
  report must carry both ``"e2e"`` and ``"nms"`` keys, each a full 12-metric
  dict of finite values (an untrained mAP near zero is fine — the contract
  under test is the plumbing, one pass / one checkpoint / both paths, not
  accuracy).
- :func:`detections_to_coco` unit cases — padding rows dropped, ``xyxy -> xywh``
  conversion, contiguous-label to COCO-category-id mapping, and the score floor.
- :func:`test_evaluate_bbox_perfect_predictions` — a one-image smoke where the
  predictions are exactly the ground-truth boxes, so mAP is 1.0. It is a
  minimal preview of :class:`TestOracleRoundTrip`, kept deliberately small
  here.
- :class:`TestOracleRoundTrip` (the WP-044 DoD, :func:`TestOracleRoundTrip.test_oracle`)
  — the full-fixture oracle: perfect predictions round-tripped through
  :func:`detections_to_coco` score mAP 1.0, shuffled-class predictions collapse
  it near 0, a localization-jitter ladder pins the metric's monotonic response
  to box error, wrong-ranked candidate scores degrade AP even with a perfect
  box present, and score-zero padding rows change nothing.

pycocotools writes progress banners to ``stdout``; the evaluator wrappers
redirect that away, and the ground-truth ``COCO(...)`` construction here is
wrapped in :func:`contextlib.redirect_stdout` so the suite stays quiet.
"""

from __future__ import annotations

import contextlib
import io
import json
from typing import TYPE_CHECKING

import pytest
import torch
from pycocotools.coco import COCO
from torchvision.io import ImageReadMode, read_image

from lit_yolo.data.coco import CocoDetectionDataset
from lit_yolo.data.letterbox import Letterbox
from lit_yolo.data.targets import Targets
from lit_yolo.decode import NMSDecoder, TopKDecoder
from lit_yolo.eval import DualPathEvaluator, detections_to_coco, evaluate_bbox
from lit_yolo.eval.coco_eval import _STAT_NAMES
from lit_yolo.ptl.module import DetectionLitModule

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
_JITTER_SMALL_FLOOR = 0.5  # small jitter must stay above this map50-95.
_SHUFFLED_MAP_CEILING = 0.05  # shuffled-class predictions must stay under this map50-95.
_RANKING_NUM_IMAGES = 3  # keep the score-ranking case small and fast (spec: 2-3 images).
_PAD_ROWS = 3  # extra score-zero rows appended per image for the padding-invariance case.


def _load_coco_gt(annotation_file: Path) -> COCO:
    """Build a ground-truth COCO object from a fixture annotation file, quietly."""
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(annotation_file))


def _eval_batch(
    fixture_dir: Path,
    letterbox: Letterbox,
    num_images: int,
) -> list[tuple[torch.Tensor, list[int], list[tuple[int, int]]]]:
    """Assemble one ``(images, image_ids, orig_sizes)`` batch from fixture images.

    Reads the first ``num_images`` images named in the fixture COCO JSON, scales
    each to ``[0, 1]``, letterboxes it onto the canvas, and stacks them with their
    COCO image ids and original ``(height, width)`` sizes — the batch contract the
    :class:`~lit_yolo.eval.coco_eval.DualPathEvaluator` consumes.
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
    coco_gt = _load_coco_gt(split_dir / _ANNOTATION)

    report = evaluator.evaluate(loader, coco_gt, torch.device("cpu"))

    assert set(report) == {"e2e", "nms"}
    for stats in report.values():
        assert set(stats) == set(_STAT_NAMES)
        assert all(isinstance(value, float) for value in stats.values())
        assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())


def test_detections_to_coco_drops_padding_rows() -> None:
    """Score-zero padding rows are dropped; only the scored detection survives."""
    detections = torch.tensor([[[1.0, 2.0, 5.0, 6.0, 0.8, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    records = detections_to_coco(detections, image_ids=[11], label_to_category={0: 5})
    assert len(records) == 1
    assert records[0]["image_id"] == 11


def test_detections_to_coco_xywh_and_category_mapping() -> None:
    """``xyxy`` becomes ``xywh`` and the contiguous label maps to its COCO category id."""
    detections = torch.tensor([[[10.0, 20.0, 40.0, 80.0, 0.5, 2.0]]])
    records = detections_to_coco(detections, image_ids=[7], label_to_category={2: 42})
    assert records[0]["bbox"] == [10.0, 20.0, 30.0, 60.0]  # x, y, w=x2-x1, h=y2-y1
    assert records[0]["category_id"] == 42


def test_detections_to_coco_score_floor() -> None:
    """A positive score floor drops detections at or below it."""
    detections = torch.tensor([[[0.0, 0.0, 4.0, 4.0, 0.30, 0.0], [0.0, 0.0, 4.0, 4.0, 0.10, 1.0]]])
    records = detections_to_coco(detections, image_ids=[3], label_to_category={0: 1, 1: 2}, score_floor=0.2)
    assert len(records) == 1
    assert records[0]["score"] == pytest.approx(0.30)


def test_detections_to_coco_rejects_length_mismatch() -> None:
    """A batch/image-id length mismatch is a ValueError, not a silent misalignment."""
    detections = torch.zeros(2, 1, 6)
    with pytest.raises(ValueError, match="does not match"):
        detections_to_coco(detections, image_ids=[0], label_to_category={0: 1})


def test_evaluate_bbox_perfect_predictions() -> None:
    """Predictions equal to the ground truth score mAP 1.0 (a WP-044 oracle preview)."""
    box = [10.0, 12.0, 20.0, 30.0]  # COCO xywh
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": [{"id": 1, "width": 64, "height": 64}],
        "categories": [{"id": 5, "name": "widget"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": box, "area": box[2] * box[3], "iscrowd": 0}],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
    results: list[dict[str, object]] = [{"image_id": 1, "category_id": 5, "bbox": box, "score": 1.0}]

    stats = evaluate_bbox(coco_gt, results)

    assert stats["map50_95"] == pytest.approx(1.0)
    assert stats["map50"] == pytest.approx(1.0)


def test_evaluate_bbox_empty_results_is_zeroed() -> None:
    """No predictions yields an all-zero stat dict rather than a pycocotools crash."""
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": [{"id": 1, "width": 64, "height": 64}],
        "categories": [{"id": 5, "name": "widget"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [1.0, 1.0, 2.0, 2.0], "area": 4.0, "iscrowd": 0}
        ],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()

    stats = evaluate_bbox(coco_gt, [])

    assert set(stats) == set(_STAT_NAMES)
    assert all(value == 0.0 for value in stats.values())


def _load_fixture_doc(fixture_dir: Path) -> dict[str, object]:
    """Load the full detseg fixture COCO annotation document as a dict."""
    return json.loads((fixture_dir / _SPLIT / _ANNOTATION).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _build_coco_gt(dataset: dict[str, object]) -> COCO:
    """Build an in-memory ground-truth :class:`COCO` from a dataset dict, quietly."""
    coco_gt = COCO()
    coco_gt.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
    return coco_gt


def _subset_doc(doc: dict[str, object], image_ids: list[int]) -> dict[str, object]:
    """Filter a fixture COCO document down to the given image ids (categories kept whole)."""
    image_id_set = set(image_ids)
    images = [image for image in doc["images"] if image["id"] in image_id_set]  # type: ignore[index,union-attr]
    annotations = [ann for ann in doc["annotations"] if ann["image_id"] in image_id_set]  # type: ignore[index,union-attr]
    return {"images": images, "categories": doc["categories"], "annotations": annotations}  # type: ignore[index]


def _perfect_results(annotations: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build one score-1.0 result dict per annotation, at its exact ground-truth box."""
    return [
        {"image_id": ann["image_id"], "category_id": ann["category_id"], "bbox": ann["bbox"], "score": 1.0}
        for ann in annotations
    ]


def _jittered_results(annotations: list[dict[str, object]], fraction: float) -> list[dict[str, object]]:
    """Build one result dict per annotation, its box shifted by ``fraction`` of its own size."""
    results: list[dict[str, object]] = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]  # type: ignore[misc]
        results.append(
            {
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "bbox": [x + fraction * w, y + fraction * h, w, h],
                "score": 1.0,
            }
        )
    return results


def _ranked_results(annotations: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build two candidates per annotation: the correct box at 0.9, a wrong one scored 0.95.

    The wrong candidate is offset by one full box size and outranks the correct one on score,
    so a ranking-aware metric must be dragged down by it even though the correct box is present.
    """
    results: list[dict[str, object]] = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]  # type: ignore[misc]
        image_id, category_id = ann["image_id"], ann["category_id"]
        results.append({"image_id": image_id, "category_id": category_id, "bbox": [x, y, w, h], "score": 0.9})
        results.append({"image_id": image_id, "category_id": category_id, "bbox": [x + w, y + h, w, h], "score": 0.95})
    return results


def _oracle_detections_to_coco(
    doc: dict[str, object],
    category_id_to_label: dict[int, int],
    label_to_category: dict[int, int],
    pad_rows: int = 0,
) -> list[dict[str, object]]:
    """Round-trip every gt annotation through the A9 tensor -> :func:`detections_to_coco` path.

    Each image's annotations become one perfect-score ``(1, N + pad_rows, 6)`` detections
    tensor (the true box, its true category as a contiguous label, score 1.0), with
    ``pad_rows`` all-zero rows appended to mimic a fixed-size decoder's padding.
    """
    anns_by_image: dict[int, list[dict[str, object]]] = {}
    for ann in doc["annotations"]:  # type: ignore[union-attr]
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    results: list[dict[str, object]] = []
    for image_id, anns in anns_by_image.items():
        rows = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            label = category_id_to_label[ann["category_id"]]
            rows.append([x, y, x + w, y + h, 1.0, float(label)])
        rows.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(pad_rows))
        detections = torch.tensor([rows], dtype=torch.float32)
        results.extend(detections_to_coco(detections, [image_id], label_to_category))
    return results


class TestOracleRoundTrip:
    """WP-044 oracle round-trip: predictions of known quality bound the metric predictably."""

    def test_oracle(self, detseg_fixture_dir: Path) -> None:
        """Perfect predictions, round-tripped through the full pipeline, score mAP 1.0 (DoD)."""
        split_dir = detseg_fixture_dir / _SPLIT
        dataset = CocoDetectionDataset(split_dir, split_dir / _ANNOTATION)
        doc = _load_fixture_doc(detseg_fixture_dir)
        coco_gt = _load_coco_gt(split_dir / _ANNOTATION)

        results = _oracle_detections_to_coco(doc, dataset.category_id_to_label, dataset.label_to_category_id)
        stats = evaluate_bbox(coco_gt, results)

        # Observed: exact 1.0 at this scale/precision (128px canvas, float64 internally in
        # detections_to_coco) -- no pycocotools quantization drift was seen. Kept as a ">"
        # bound per spec rather than "==", since quantization is a documented caveat of the
        # instrument, not a guarantee this test should pin down.
        assert stats["map50_95"] > 0.999
        assert stats["map50"] == pytest.approx(1.0)
        assert stats["map75"] == pytest.approx(1.0)
        # recall_10/recall_100 saturate. recall_1 does not (observed ~0.5): several fixture
        # images carry more than one annotation, and pycocotools' AR@1 credits only the single
        # highest-scoring detection per image, so it is structurally capped well below 1.0
        # regardless of prediction quality -- not asserted here for that reason.
        assert stats["recall_10"] == pytest.approx(1.0)
        assert stats["recall_100"] == pytest.approx(1.0)

    def test_oracle_shuffled_classes(self, detseg_fixture_dir: Path) -> None:
        """Correct boxes with every label cyclically shifted to a wrong class collapse mAP."""
        doc = _load_fixture_doc(detseg_fixture_dir)
        coco_gt = _load_coco_gt(detseg_fixture_dir / _SPLIT / _ANNOTATION)
        category_ids = sorted({int(category["id"]) for category in doc["categories"]})  # type: ignore[union-attr]

        results: list[dict[str, object]] = []
        for ann in doc["annotations"]:  # type: ignore[union-attr]
            index = category_ids.index(ann["category_id"])
            wrong_category = category_ids[(index + 1) % len(category_ids)]
            assert wrong_category != ann["category_id"]  # every label actually changes
            results.append(
                {"image_id": ann["image_id"], "category_id": wrong_category, "bbox": ann["bbox"], "score": 1.0}
            )

        stats = evaluate_bbox(coco_gt, results)

        assert stats["map50_95"] < _SHUFFLED_MAP_CEILING

    def test_perturbation_ladder_localization_sensitivity(self, detseg_fixture_dir: Path) -> None:
        """mAP degrades monotonically from exact boxes to small jitter to a full offset."""
        doc = _load_fixture_doc(detseg_fixture_dir)
        coco_gt = _load_coco_gt(detseg_fixture_dir / _SPLIT / _ANNOTATION)
        annotations = doc["annotations"]  # type: ignore[assignment]

        exact = evaluate_bbox(coco_gt, _jittered_results(annotations, 0.0))["map50_95"]
        jittered = evaluate_bbox(coco_gt, _jittered_results(annotations, _JITTER_SMALL_FRACTION))["map50_95"]
        offset = evaluate_bbox(coco_gt, _jittered_results(annotations, _JITTER_LARGE_FRACTION))["map50_95"]

        assert exact == pytest.approx(1.0)
        assert _JITTER_SMALL_FLOOR < jittered < exact
        assert offset < jittered
        assert offset < _SHUFFLED_MAP_CEILING

    def test_score_ranking_degrades_ap(self, detseg_fixture_dir: Path) -> None:
        """A higher-scored, wrong-location candidate outranks the correct one and halves AP."""
        doc = _load_fixture_doc(detseg_fixture_dir)
        image_ids = sorted({int(image["id"]) for image in doc["images"]})[:_RANKING_NUM_IMAGES]  # type: ignore[union-attr]
        subset = _subset_doc(doc, image_ids)
        coco_gt = _build_coco_gt(subset)
        annotations = subset["annotations"]  # type: ignore[assignment]

        perfect_map = evaluate_bbox(coco_gt, _perfect_results(annotations))["map50_95"]
        ranked_map = evaluate_bbox(coco_gt, _ranked_results(annotations))["map50_95"]

        assert perfect_map == pytest.approx(1.0)
        assert ranked_map < perfect_map
        assert ranked_map == pytest.approx(0.5)

    def test_detections_padding_invariance(self, detseg_fixture_dir: Path) -> None:
        """Score-zero padding rows appended to each image's detections change nothing."""
        split_dir = detseg_fixture_dir / _SPLIT
        dataset = CocoDetectionDataset(split_dir, split_dir / _ANNOTATION)
        doc = _load_fixture_doc(detseg_fixture_dir)
        coco_gt = _load_coco_gt(split_dir / _ANNOTATION)

        unpadded = _oracle_detections_to_coco(doc, dataset.category_id_to_label, dataset.label_to_category_id)
        padded = _oracle_detections_to_coco(
            doc, dataset.category_id_to_label, dataset.label_to_category_id, pad_rows=_PAD_ROWS
        )

        assert padded == unpadded
        assert evaluate_bbox(coco_gt, padded) == evaluate_bbox(coco_gt, unpadded)
