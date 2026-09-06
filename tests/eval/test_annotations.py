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


def test_keypoint_load_carries_the_annotations_own_metadata(tmp_path: Path) -> None:
    """A ``with_keypoints`` load hands the OKS scorer the file's fields, not a recount of them.

    Every one of the four is chosen here to disagree with what a reconstruction from
    the points would produce: the declared ``num_keypoints`` is 1 against two visible
    joints, ``area`` is 500 against a 2x2 point extent, ``iscrowd`` is set on an
    instance nothing else marks. If any is recomputed instead of read, this asserts
    the recomputed value and fails.
    """
    payload = _payload()
    payload["annotations"] = [
        {
            "image_id": 1,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "category_id": 3,
            "iscrowd": 1,
            "area": 500.0,
            "num_keypoints": 1,
            "keypoints": [1.0, 2.0, 2, 3.0, 4.0, 1, 0.0, 0.0, 0],
        }
    ]
    ann_file = tmp_path / "person_keypoints.json"
    ann_file.write_text(json.dumps(payload))

    _, targets, _ = load_eval_annotations(ann_file, with_keypoints=True)

    target = targets[1]
    assert target["keypoints"].tolist() == [[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]]
    assert target["visibility"].tolist() == [[2, 1, 0]]
    assert target["num_keypoints"].tolist() == [1]
    assert target["area"].tolist() == [500.0]
    assert target["iscrowd"].tolist() == [1]


def test_keypoint_load_gives_an_unannotated_image_empty_point_tensors(tmp_path: Path) -> None:
    """An image with no annotation still carries the three keypoint keys, all empty.

    Every image the evaluator scores must produce a target dict of one shape: the
    OKS document is built by iterating the instance axis, and a missing key on the
    unannotated images would fail there rather than at the boundary that dropped it.
    Image 2 of this payload carries no annotation at all, which is the majority case
    on a person-keypoints split, where most images hold no person.
    """
    payload = _payload()
    payload["annotations"] = [
        {
            "image_id": 1,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "category_id": 3,
            "keypoints": [1.0, 2.0, 2, 3.0, 4.0, 1, 0.0, 0.0, 0],
        }
    ]
    ann_file = tmp_path / "person_keypoints.json"
    ann_file.write_text(json.dumps(payload))

    _, targets, _ = load_eval_annotations(ann_file, with_keypoints=True)

    unannotated = targets[2]
    assert unannotated["keypoints"].shape == (0, 0, 2)
    assert unannotated["visibility"].shape == (0, 0)
    assert unannotated["num_keypoints"].shape == (0,)


def test_keypoint_load_refuses_an_annotation_file_with_no_points(tmp_path: Path) -> None:
    """A plain ``instances`` file loaded for the pose protocol fails, naming the field.

    The two COCO files differ only in which annotations they carry, so pointing the
    keypoint protocol at the detection one is an easy mistake. It must not resolve
    into an evaluation of zero poses against a full split, which would report a
    perfectly formatted OKS of 0.
    """
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps(_payload()))

    with pytest.raises(ValueError, match="keypoints length"):
        load_eval_annotations(ann_file, with_keypoints=True)


def test_masks_and_keypoints_requested_together_reach_the_same_target(tmp_path: Path) -> None:
    """Both opt-ins compose: one load with both flags carries the mask *and* the points.

    The two options are documented as independent, and asking for both selects the
    lazy mask path — which is where the keypoint flag used to stop, so the OKS keys
    vanished from a target that still looked complete to everything reading its boxes.
    The values are compared against the eager keypoints-only load of the same file, so
    a target that grew the keys but filled them from somewhere else fails too.
    """
    payload = _payload()
    payload["annotations"] = [
        {
            "image_id": 1,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "category_id": 3,
            "num_keypoints": 1,
            "keypoints": [1.0, 2.0, 2, 3.0, 4.0, 1, 0.0, 0.0, 0],
            "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]],
        }
    ]
    ann_file = tmp_path / "person_keypoints.json"
    ann_file.write_text(json.dumps(payload))

    _, combined, _ = load_eval_annotations(ann_file, with_masks=True, with_keypoints=True)

    _, points_only, _ = load_eval_annotations(ann_file, with_keypoints=True)
    both = _TARGET_KEYS | {"masks", "keypoints", "visibility", "num_keypoints"}
    assert set(combined[1]) == both
    assert set(combined[2]) == both  # the unannotated image carries the same key set
    assert combined[1]["masks"].shape == (1, 6, 10)
    assert torch.equal(combined[1]["keypoints"], points_only[1]["keypoints"])
    assert torch.equal(combined[1]["visibility"], points_only[1]["visibility"])
    assert torch.equal(combined[1]["num_keypoints"], points_only[1]["num_keypoints"])


@pytest.mark.parametrize(
    ("annotation", "message"),
    [
        pytest.param(
            {"id": 4, "image_id": 1, "bbox": [float("nan"), 1.0, 2.0, 2.0], "category_id": 3},
            "annotation 4 of image 1: bbox must be finite",
            id="nan-coordinate",
        ),
        pytest.param(
            {"id": 4, "image_id": 1, "bbox": [float("inf"), 1.0, 2.0, 2.0], "category_id": 3},
            "annotation 4 of image 1: bbox must be finite",
            id="infinite-coordinate",
        ),
        pytest.param(
            {"id": 5, "image_id": 1, "bbox": [1.0, 1.0, 2.0, 2.0], "category_id": 3, "area": -1.0},
            "annotation 5 of image 1: area must be finite and non-negative",
            id="negative-area",
        ),
        pytest.param(
            {"id": 6, "image_id": 1, "bbox": [1.0, 1.0, 2.0, 2.0], "category_id": 3, "iscrowd": 2},
            "annotation 6 of image 1: iscrowd must be 0 or 1",
            id="out-of-domain-crowd",
        ),
    ],
)
def test_malformed_numeric_metadata_is_refused_naming_the_annotation(
    annotation: dict[str, object], message: str
) -> None:
    """Each out-of-domain numeric field is refused on its own, naming image, annotation and value.

    Carried through instead, every one of these makes both scorers report ``-1`` for
    every summary — an evaluation run consumed for an undefined metric report with
    nothing in it pointing back at the annotation that caused it. The three fields are
    exercised separately because a combined fixture cannot say which one was diagnosed.
    """
    with pytest.raises(ValueError, match=message):
        annotations_to_target([annotation])


def test_legal_but_unusual_metadata_still_loads() -> None:
    """Off-image coordinates, a zero area and the crowd flag are input, not malformed input.

    The refusal above must land on values with no reading at all, not on the three
    things a real COCO export legitimately contains: a box whose origin sits outside
    the image, an ``area`` of zero a protocol may assign a hairline instance, and
    ``iscrowd = 1``, which is the crowd-ignore rule this reader exists to preserve.
    """
    target = annotations_to_target(
        [{"id": 1, "image_id": 1, "bbox": [-5.0, -5.0, 2.0, 2.0], "category_id": 3, "area": 0.0, "iscrowd": 1}]
    )

    assert target["boxes"].tolist() == [[-5.0, -5.0, -3.0, -3.0]]
    assert target["area"].tolist() == [0.0]
    assert target["iscrowd"].tolist() == [1]


def test_the_lazy_path_refuses_malformed_metadata_at_load_not_mid_run(tmp_path: Path) -> None:
    """A masked load validates the whole file up front rather than at first lookup.

    :class:`~lucid_yolo.eval.annotations.LazyTargets` builds nothing until the
    evaluator asks for an image, so without this pass the same refusal would arrive
    part-way through a scoring run that has already spent its minutes — which is the
    cost the malformed file was meant to avoid in the first place.
    """
    payload = _payload()
    payload["annotations"] = [{"id": 9, "image_id": 1, "bbox": [1.0, 1.0, 2.0, 2.0], "category_id": 3, "iscrowd": 2}]
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="annotation 9 of image 1: iscrowd must be 0 or 1"):
        load_eval_annotations(ann_file, with_masks=True)


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
