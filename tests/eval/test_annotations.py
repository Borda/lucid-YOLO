# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the COCO annotation reader behind the dual-path evaluator (WP-083).

:mod:`lucid_yolo.eval.annotations` is the one place the evaluator's ground truth is
built, shared by the val2017 checkpoint evaluation and the synthetic-shapes
regression producer. These tests pin the parts a second copy would have been free to
get subtly wrong: the ``xywh`` to ``xyxy`` conversion, which annotations are dropped
as degenerate, that unannotated images still appear as empty targets, and that the
label map follows the sorted-category-id order
:class:`~lucid_yolo.data.coco.CocoDetectionDataset` uses.

The letterboxed batch stream is covered against real image files written to
``tmp_path``, so the loader path (read, scale, letterbox, stack) is exercised rather
than mocked.
"""

import json
from pathlib import Path

import pytest
import torch
from torchvision.io import write_png

from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.eval import annotations
from lucid_yolo.eval.annotations import (
    EvalImage,
    annotations_to_target,
    empty_target,
    letterboxed_batches,
    load_eval_annotations,
)

_TARGET_KEYS = {"boxes", "labels", "iscrowd", "area"}


def _payload() -> dict[str, object]:
    """Return a two-image COCO payload where the higher id is listed first.

    Examples:
        >>> payload = _payload()
        >>> [image["id"] for image in payload["images"]]
        [2, 1]
        >>> len(payload["annotations"])
        1
    """
    return {
        "categories": [{"id": 7}, {"id": 3}],
        "images": [
            {"id": 2, "file_name": "b.png", "height": 8, "width": 12},
            {"id": 1, "file_name": "a.png", "height": 6, "width": 10},
        ],
        "annotations": [
            {"image_id": 1, "bbox": [1.0, 2.0, 3.0, 4.0], "category_id": 3, "iscrowd": 0, "area": 12.0},
        ],
    }


def test_empty_target_has_the_torchmetrics_dtypes() -> None:
    """The no-annotation target carries every key with the dtypes torchmetrics expects."""
    target = empty_target()
    assert set(target) == _TARGET_KEYS
    assert target["boxes"].shape == (0, 4)
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.long
    assert target["iscrowd"].dtype == torch.long
    assert target["area"].dtype == torch.float32


def test_annotations_convert_xywh_to_xyxy() -> None:
    """A COCO ``[x, y, w, h]`` box becomes ``[x1, y1, x2, y2]`` in original coordinates."""
    target = annotations_to_target([{"bbox": [1.0, 2.0, 3.0, 4.0], "category_id": 5}])
    assert target["boxes"].tolist() == [[1.0, 2.0, 4.0, 6.0]]
    assert target["labels"].tolist() == [5]


def test_annotations_default_iscrowd_and_area() -> None:
    """Missing ``iscrowd`` defaults to 0 and missing ``area`` to the box's own area."""
    target = annotations_to_target([{"bbox": [0.0, 0.0, 2.0, 5.0], "category_id": 1}])
    assert target["iscrowd"].tolist() == [0]
    assert target["area"].tolist() == [10.0]


@pytest.mark.parametrize(
    "bbox",
    [
        pytest.param([0.0, 0.0, 0.0, 4.0], id="zero-width"),
        pytest.param([0.0, 0.0, 4.0, 0.0], id="zero-height"),
        pytest.param([0.0, 0.0, -1.0, 4.0], id="negative-width"),
        pytest.param([0.0, 0.0, 4.0], id="wrong-arity"),
        pytest.param("not-a-list", id="not-a-list"),
    ],
)
def test_degenerate_annotations_are_dropped(bbox: object) -> None:
    """Degenerate boxes are skipped, not emitted as unmatchable zero-area targets."""
    target = annotations_to_target([{"bbox": bbox, "category_id": 1}])
    assert target["boxes"].shape == (0, 4)


def test_valid_annotations_survive_alongside_degenerate_ones() -> None:
    """Filtering removes only the bad rows; a good box in the same image is kept."""
    target = annotations_to_target(
        [
            {"bbox": [0.0, 0.0, 0.0, 4.0], "category_id": 1},
            {"bbox": [1.0, 1.0, 2.0, 2.0], "category_id": 9},
        ]
    )
    assert target["boxes"].tolist() == [[1.0, 1.0, 3.0, 3.0]]
    assert target["labels"].tolist() == [9]


def test_load_sorts_images_by_id(tmp_path: Path) -> None:
    """Images are returned in ascending id order regardless of their order in the file."""
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps(_payload()))
    images, _, _ = load_eval_annotations(ann_file)
    assert [image.image_id for image in images] == [1, 2]
    assert images[0] == EvalImage(image_id=1, file_name="a.png", height=6, width=10)


def test_load_gives_unannotated_images_an_empty_target(tmp_path: Path) -> None:
    """Every image gets a target entry, so the evaluator sees the whole split."""
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps(_payload()))
    _, targets, _ = load_eval_annotations(ann_file)
    assert set(targets) == {1, 2}
    assert targets[1]["boxes"].tolist() == [[1.0, 2.0, 4.0, 6.0]]
    assert targets[2]["boxes"].shape == (0, 4)


def test_masked_load_matches_an_eager_decode(tmp_path: Path) -> None:
    """The lazy mapping returns exactly what decoding every annotation up front would.

    Laziness is a memory strategy, not a different ground truth: every key, every
    tensor and every mask must match :func:`annotations_to_target` called directly
    with the image's own size. An off-by-one in the size lookup, or a target built
    against the wrong image's ``(height, width)``, is invisible in the shapes of a
    single-image fixture and shows up here because the two images differ in size.
    """
    payload = _payload()
    polygon = [[1.0, 2.0, 4.0, 2.0, 4.0, 6.0, 1.0, 6.0]]
    payload["annotations"] = [{**_payload()["annotations"][0], "segmentation": polygon}]  # type: ignore[index]
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps(payload))

    _, targets, _ = load_eval_annotations(ann_file, with_masks=True)
    expected = annotations_to_target(payload["annotations"], image_size=(6, 10))  # type: ignore[arg-type]

    assert set(targets) == {1, 2}
    assert set(targets[1]) == _TARGET_KEYS | {"masks"}
    for key, value in expected.items():
        assert torch.equal(targets[1][key], value)
    assert targets[2]["masks"].shape == (0, 8, 12)  # the unannotated image, at its own size


def test_masked_load_decodes_on_lookup_not_up_front(tmp_path: Path) -> None:
    """No mask is decoded until its image is asked for.

    The property that makes val2017 segmentation evaluation runnable at all --
    eager decoding is ~11 GB resident before the first image is read. A mapping
    that decoded everything in the constructor and merely *stored* it would
    satisfy every other test in this file.
    """
    payload = _payload()
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps(payload))

    decoded: list[object] = []
    real_mask = annotations.annotation_mask

    def _counting_mask(segmentation: object, image_size: tuple[int, int]) -> torch.Tensor:
        decoded.append(segmentation)
        return real_mask(segmentation, image_size)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(annotations, "annotation_mask", _counting_mask)
        _, targets, _ = load_eval_annotations(ann_file, with_masks=True)
        assert decoded == []  # loading the file decodes nothing
        _ = targets[1]["masks"]
        assert len(decoded) == 1


def test_load_builds_the_sorted_label_map(tmp_path: Path) -> None:
    """The label map is contiguous-label to category id in sorted-category-id order."""
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps(_payload()))
    _, _, label_to_category = load_eval_annotations(ann_file)
    assert label_to_category == {0: 3, 1: 7}


def test_letterboxed_batches_stack_and_carry_original_sizes(tmp_path: Path) -> None:
    """Batches are stacked to the letterbox side and carry ids plus original sizes."""
    records = []
    for index, (height, width) in enumerate([(6, 10), (8, 12), (5, 5)], start=1):
        name = f"{index}.png"
        write_png(torch.randint(0, 256, (3, height, width), dtype=torch.uint8), str(tmp_path / name))
        records.append(EvalImage(image_id=index, file_name=name, height=height, width=width))

    batches = list(letterboxed_batches(records, tmp_path, Letterbox(32), batch_size=2))

    assert [batch[0].shape[0] for batch in batches] == [2, 1]
    assert [ids for _, ids, _ in batches] == [[1, 2], [3]]
    assert [sizes for _, _, sizes in batches] == [[(6, 10), (8, 12)], [(5, 5)]]
    assert batches[0][0].shape[1:] == (3, 32, 32)
    assert batches[0][0].max() <= 1.0
