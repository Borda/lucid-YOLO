# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-043 pycocotools bbox evaluator.

Covers the three pieces of :mod:`lit_yolo.eval.coco_eval`:

- :func:`test_dual_path_report` (the DoD) — a **real** tiny untrained detector is
  run through :class:`~lit_yolo.eval.coco_eval.DualPathEvaluator` over a few
  detseg fixture images with a ground-truth :class:`pycocotools.coco.COCO` built
  from the fixture annotation JSON; the report must carry both ``"e2e"`` and
  ``"nms"`` keys, each a full 12-metric dict of finite values (an untrained mAP
  near zero is fine — the contract under test is the plumbing, one pass / one
  checkpoint / both paths, not accuracy).
- :func:`detections_to_coco` unit cases — padding rows dropped, ``xyxy -> xywh``
  conversion, contiguous-label to COCO-category-id mapping, and the score floor.
- :func:`test_evaluate_bbox_perfect_predictions` — a one-image smoke where the
  predictions are exactly the ground-truth boxes, so mAP is 1.0. It is a minimal
  preview of the WP-044 oracle round-trip, kept deliberately small here.

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
