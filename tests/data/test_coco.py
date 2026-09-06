# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-014 COCO dataset, datamodule and ``check-data`` script.

All dataset/datamodule tests run offline against the WP-007 synthetic
detection/segmentation fixture (``detseg_fixture_dir``), never real COCO. They
cover: dataset length, sample dtype/range, ``xywh``->``xyxy`` conversion within
image bounds, box/polygon agreement, contiguous label remapping, the size-aware
scale policy, a train/val datamodule smoke (batch stacking + ragged target list),
seeded determinism, and the importable ``check_data`` layout validator (both the
matching and the mismatching path, without spawning a subprocess).

Hand-built COCO payloads cover the parsing rules the fixture cannot exercise, since
its annotations are all single-ring, in-canvas and keypoint-free: which ring a
multi-ring polygon keeps and where that instance's box then comes from (M-38), the
per-image nature of that rule, the drop of an instance whose box clamps to no extent
(L-30), the named error a keypoint run gets from an annotation with no points (L-29),
and the ``image_ids`` a position-pairing consumer reads identity from.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer

from lucid_yolo.cli import data as data_cli
from lucid_yolo.data import Targets, boxes_from_polygons
from lucid_yolo.data import check as check_data
from lucid_yolo.data.coco import CocoDetectionDataset, build_scale_policy
from lucid_yolo.ptl import datamodule as dm
from lucid_yolo.ptl.datamodule import (
    DetectionDataModule,
    PackedTargets,
    collate_detection,
    pack_targets,
    unpack_batch,
    unpack_targets,
)
from lucid_yolo.ptl.seg_targets import instance_mask_targets

#: Image count of the detseg fixture split (tests/fixtures/synthetic.py).
_FIXTURE_IMAGE_COUNT = 16
#: Box/polygon-extent agreement tolerance in pixels (fixture bbox == polygon extent).
_EXTENT_TOL = 1e-3
#: Small square letterbox side keeping the datamodule smoke fast.
_SMOKE_IMG_SIZE = 64


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed the global RNG before each test (pipelines use explicit generators)."""
    torch.manual_seed(0)
    yield


def _dataset(fixture_dir: Path, transforms: object = None) -> CocoDetectionDataset:
    """Build a dataset over the fixture's single ``train`` split.

    Examples:
        >>> callable(_dataset)  # needs a live detseg_fixture_dir fixture
        True
    """
    split = fixture_dir / "train"
    return CocoDetectionDataset(split, split / "_annotations.coco.json", transforms=transforms)  # type: ignore[arg-type]


def _first_nonempty(dataset: CocoDetectionDataset) -> tuple[torch.Tensor, Targets]:
    """Return the first sample carrying at least one box.

    Examples:
        >>> callable(_first_nonempty)  # needs a dataset built from a live fixture
        True
    """
    for index in range(len(dataset)):
        image, targets = dataset[index]
        if targets.boxes.shape[0] > 0:
            return image, targets
    raise AssertionError("fixture unexpectedly has no annotated images")


def test_length_matches_fixture(detseg_fixture_dir: Path) -> None:
    """The dataset length equals the fixture image count."""
    dataset = _dataset(detseg_fixture_dir)
    assert len(dataset) == _FIXTURE_IMAGE_COUNT


def test_sample_dtype_and_range(detseg_fixture_dir: Path) -> None:
    """Each image is a CHW float32 tensor with values in ``[0, 1]``."""
    image, _ = _dataset(detseg_fixture_dir)[0]
    assert image.dtype == torch.float32
    assert image.ndim == 3 and image.shape[0] == 3
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0


def test_boxes_are_xyxy_within_bounds(detseg_fixture_dir: Path) -> None:
    """Boxes are ordered ``xyxy`` and lie inside the image extent."""
    image, targets = _first_nonempty(_dataset(detseg_fixture_dir))
    _, height, width = image.shape
    boxes = targets.boxes
    assert torch.all(boxes[:, 2] >= boxes[:, 0]) and torch.all(boxes[:, 3] >= boxes[:, 1])
    assert torch.all(boxes[:, 0] >= 0.0) and torch.all(boxes[:, 2] <= width)
    assert torch.all(boxes[:, 1] >= 0.0) and torch.all(boxes[:, 3] <= height)


def test_polygons_present_and_agree_with_boxes(detseg_fixture_dir: Path) -> None:
    """One polygon ring per box, whose extent matches the parsed box."""
    _, targets = _first_nonempty(_dataset(detseg_fixture_dir))
    assert len(targets.polygons) == targets.boxes.shape[0]
    derived = boxes_from_polygons(targets.polygons)
    assert torch.allclose(derived, targets.boxes, atol=_EXTENT_TOL)


def test_label_remap_is_contiguous(detseg_fixture_dir: Path) -> None:
    """Labels form a contiguous ``0..K-1`` space and the maps are inverse."""
    dataset = _dataset(detseg_fixture_dir)
    labels = sorted(dataset.category_id_to_label.values())
    assert labels == list(range(len(labels)))
    assert {v: k for k, v in dataset.category_id_to_label.items()} == dataset.label_to_category_id
    _, targets = _first_nonempty(dataset)
    assert torch.all(targets.labels >= 0) and torch.all(targets.labels < len(labels))


def test_an_ordinary_coco_file_reads_as_nothing_difficult(detseg_fixture_dir: Path) -> None:
    """A file with no ``difficult`` key means "no instance is difficult" (A51)."""
    _, targets = _first_nonempty(_dataset(detseg_fixture_dir))
    assert targets.difficult.shape == (targets.boxes.shape[0],)
    assert not bool(targets.difficult.any())


@pytest.mark.parametrize("oriented", [pytest.param(False, id="axis-aligned"), pytest.param(True, id="oriented")])
def test_the_difficult_key_reaches_the_targets_channel(tmp_path: Path, oriented: bool) -> None:
    """R18's flag, written per annotation, lands on the A51 channel on both readings.

    The flag says something about the annotation rather than about the box modality, so
    the oriented and axis-aligned paths must agree on it — the tiled DOTA layout (WP-094)
    is read oriented, and losing the flag there would silently turn every ignorable
    ground truth into a scored one at the metric (A48).
    """
    split = tmp_path / "val"
    split.mkdir()
    from torchvision.io import write_png  # noqa: PLC0415 - test-local, the reader does not need it

    write_png(torch.zeros(3, 8, 8, dtype=torch.uint8), str(split / "tile.png"))
    quad = [1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0]
    (split / "instances.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "tile.png", "height": 8, "width": 8}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2], "segmentation": [quad]},
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [4, 4, 2, 2],
                        "segmentation": [[4.0, 4.0, 6.0, 4.0, 6.0, 6.0, 4.0, 6.0]],
                        "difficult": 1,
                    },
                ],
                "categories": [{"id": 1, "name": "plane"}],
            }
        ),
        encoding="utf-8",
    )

    _, targets = CocoDetectionDataset(split, split / "instances.json", oriented=oriented)[0]

    assert targets.difficult.tolist() == [False, True]


@pytest.fixture
def keypoint_coco(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Create one image and a mutable two-instance COCO keypoint payload."""
    split = tmp_path / "val"
    split.mkdir()
    from torchvision.io import write_png  # noqa: PLC0415 - test-local, the reader does not need it

    write_png(torch.zeros(3, 8, 8, dtype=torch.uint8), str(split / "pose.png"))
    payload: dict[str, object] = {
        "images": [{"id": 1, "file_name": "pose.png", "height": 8, "width": 8}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 2, 2],
                "segmentation": [[1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0]],
                "keypoints": [1.5, 2.0, 2, 2.5, 3.0, 1],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [4, 4, 2, 2],
                "segmentation": [[4.0, 4.0, 6.0, 4.0, 6.0, 6.0, 4.0, 6.0]],
                "keypoints": [4.5, 5.0, 0, 5.5, 6.0, 2],
            },
            {
                "id": 3,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 1, 1],
                "segmentation": [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]],
                "iscrowd": 1,
            },
            {
                "id": 4,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 1, 1],
                "segmentation": {"counts": [1], "size": [8, 8]},
            },
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    return split, payload


class TestKeypointParsing:
    """Tests for opt-in COCO annotation-level keypoint parsing."""

    @pytest.mark.parametrize("oriented", [pytest.param(False, id="axis-aligned"), pytest.param(True, id="oriented")])
    def test_coordinates_and_visibility_reach_targets(
        self, keypoint_coco: tuple[Path, dict[str, object]], oriented: bool
    ) -> None:
        """Retained instances produce aligned coordinate and visibility tensors.

        Two usable annotations carry two keypoints each, while crowd and RLE
        annotations omit the field entirely; only the retained instance axis may
        be indexed and stacked on either box-reading path.
        """
        split, payload = keypoint_coco
        (split / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

        _, targets = CocoDetectionDataset(split, split / "instances.json", oriented=oriented, keypoints=True)[0]

        assert targets.keypoints.shape == (2, 2, 2)
        assert targets.keypoints.dtype == torch.float32
        assert targets.keypoints.tolist() == [[[1.5, 2.0], [2.5, 3.0]], [[4.5, 5.0], [5.5, 6.0]]]
        assert targets.keypoint_vis.shape == (2, 2)
        assert targets.keypoint_vis.dtype == torch.int64
        assert targets.keypoint_vis.tolist() == [[2, 1], [0, 2]]

    def test_default_ignores_source_keypoints(self, keypoint_coco: tuple[Path, dict[str, object]]) -> None:
        """The default reader leaves both keypoint channels canonically empty.

        Source data alone must not opt a detection reader into pose parsing, so
        populated annotation fields still yield the WP-120 dataclass defaults.
        """
        split, payload = keypoint_coco
        (split / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

        _, targets = CocoDetectionDataset(split, split / "instances.json")[0]

        assert targets.keypoints.shape == (0, 0, 2)
        assert targets.keypoint_vis.shape == (0, 0)

    def test_malformed_length_names_the_image(self, keypoint_coco: tuple[Path, dict[str, object]]) -> None:
        """A non-triplet annotation fails with its image name and bad length.

        Four flat values cannot describe repeating ``x, y, v`` triples; raising
        during dataset construction prevents a shifted coordinate/visibility split.
        """
        split, payload = keypoint_coco
        annotations = payload["annotations"]
        assert isinstance(annotations, list)
        annotations[0]["keypoints"] = [1.0, 2.0, 2, 3.0]
        (split / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match=r"pose\.png.*length 4"):
            CocoDetectionDataset(split, split / "instances.json", keypoints=True)

    def test_mismatched_counts_name_both_k_values(self, keypoint_coco: tuple[Path, dict[str, object]]) -> None:
        """Different per-instance K values fail before stacking one image.

        A rectangular ``(N, K, 2)`` target cannot represent one and two points
        together, so the reader identifies the image and both observed counts.
        """
        split, payload = keypoint_coco
        annotations = payload["annotations"]
        assert isinstance(annotations, list)
        annotations[1]["keypoints"] = [4.5, 5.0, 2]
        (split / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match=r"pose\.png.*K values \[1, 2\]"):
            CocoDetectionDataset(split, split / "instances.json", keypoints=True)


#: R21's seven-point symbol schema, whose sides are named as suffixes. Written out rather
#: than imported from fuse-augmentations so this suite pins the *reading*, not the producer.
_SYMBOL_NAMES = ["center", "apex", "tail", "flank_left", "flank_right", "base_left", "base_right"]
#: R12's seventeen-point human pose schema, whose sides are named as prefixes.
_POSE_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _dataset_with_category(
    split: Path, payload: dict[str, object], category: dict[str, object]
) -> CocoDetectionDataset:
    """Write ``payload`` with ``category`` substituted and open it as a keypoint reader.

    Examples:
        >>> callable(_dataset_with_category)  # needs a live keypoint_coco fixture
        True
    """
    payload["categories"] = [category]
    (split / "instances.json").write_text(json.dumps(payload), encoding="utf-8")
    return CocoDetectionDataset(split, split / "instances.json", keypoints=True)


class TestKeypointFlipPairs:
    """The mirror permutation is read off the file's own category schema (A64)."""

    @pytest.mark.parametrize(
        ("names", "expected"),
        [
            pytest.param(_SYMBOL_NAMES, [(3, 4), (5, 6)], id="suffix-sided-symbol-schema"),
            pytest.param(
                _POSE_NAMES,
                [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)],
                id="prefix-sided-human-pose",
            ),
        ],
    )
    def test_sided_names_pair_by_stem(
        self, keypoint_coco: tuple[Path, dict[str, object]], names: list[str], expected: list[tuple[int, int]]
    ) -> None:
        """Both naming conventions in circulation yield their schema's true left/right swap.

        A64 makes the pairing the dataset's to state, and a COCO file states it by naming
        the sides. The two schemas this project reads spell that differently — ``flank_left``
        against ``left_eye`` — so a reader that understood only one would silently return no
        pairs for the other, which is indistinguishable from a genuinely symmetric schema.
        """
        split, payload = keypoint_coco

        dataset = _dataset_with_category(split, payload, {"id": 1, "name": "thing", "keypoints": names})

        assert dataset.keypoint_flip_pairs == expected

    def test_unsided_schema_yields_none(self, keypoint_coco: tuple[Path, dict[str, object]]) -> None:
        """A schema naming no left/right point pairs nothing, so the mirror only reflects x.

        ``None`` is the correct answer here rather than a fallback: with no sided landmark
        there is no identity to swap, and A64 keeps that meaning of ``None`` intact.
        """
        split, payload = keypoint_coco

        dataset = _dataset_with_category(split, payload, {"id": 1, "name": "thing", "keypoints": ["tip", "tail"]})

        assert dataset.keypoint_flip_pairs is None

    def test_detection_reader_derives_nothing(self, keypoint_coco: tuple[Path, dict[str, object]]) -> None:
        """A reader opened without ``keypoints=True`` publishes no pairing even from a named schema.

        The pairing is only meaningful to a run that mirrors points. Deriving it regardless
        would let a schema flaw irrelevant to a detection run stop that run at construction.
        """
        split, payload = keypoint_coco
        payload["categories"] = [{"id": 1, "name": "thing", "keypoints": _SYMBOL_NAMES}]
        (split / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

        dataset = CocoDetectionDataset(split, split / "instances.json")

        assert dataset.keypoint_flip_pairs is None

    def test_half_declared_pair_raises(self, keypoint_coco: tuple[Path, dict[str, object]]) -> None:
        """A name declaring one side with no counterpart is rejected instead of ignored.

        Dropping the unmatched name would mirror that landmark's coordinate while leaving
        its identity in place — the exact silent left/right corruption A64 exists to
        prevent, which trains and reports nothing.
        """
        split, payload = keypoint_coco
        payload["categories"] = [{"id": 1, "name": "thing", "keypoints": ["nose", "left_eye"]}]
        (split / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="right counterpart"):
            CocoDetectionDataset(split, split / "instances.json", keypoints=True)

    def test_conflicting_category_schemas_raise(self, keypoint_coco: tuple[Path, dict[str, object]]) -> None:
        """Two categories declaring different keypoint schemas are rejected, not silently reconciled.

        One permutation is applied to the whole image, so picking either schema's pairing
        would mis-mirror every instance of the other category.
        """
        split, payload = keypoint_coco
        payload["categories"] = [
            {"id": 1, "name": "thing", "keypoints": _SYMBOL_NAMES},
            {"id": 2, "name": "other", "keypoints": _POSE_NAMES},
        ]
        (split / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="differing keypoint schemas"):
            CocoDetectionDataset(split, split / "instances.json", keypoints=True)


@pytest.fixture
def ringless_coco(tmp_path: Path) -> Path:
    """Create one image and a two-instance COCO payload with no ``segmentation`` key at all.

    Mirrors a bare ``Task.DETECTION`` export (fuse-augmentations never writes a
    ``segmentation`` field for that task; WP-121b's own report caught this against
    the fixed ``Task.KEYPOINTS`` export, which has the same shape of gap).
    """
    split = tmp_path / "val"
    split.mkdir()
    from torchvision.io import write_png  # noqa: PLC0415 - test-local, the reader does not need it

    write_png(torch.zeros(3, 8, 8, dtype=torch.uint8), str(split / "plain.png"))
    payload = {
        "images": [{"id": 1, "file_name": "plain.png", "height": 8, "width": 8}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [4, 4, 2, 2]},
        ],
        "categories": [{"id": 1, "name": "square"}],
    }
    (split / "instances.json").write_text(json.dumps(payload), encoding="utf-8")
    return split


class TestRinglessAnnotations:
    """A missing ``segmentation`` ring is fatal only for an oriented reading (WP-121b)."""

    def test_plain_reading_keeps_boxes_without_a_ring(self, ringless_coco: Path) -> None:
        """Both ringless boxes survive and ``polygons`` comes back empty, not one-per-box.

        The fixed-write bug (WP-121b) let a keypoints-only export silently parse to zero
        instances; this pins the corrected contract for the plainer bare-detection case
        the same gap applies to.
        """
        _, targets = CocoDetectionDataset(ringless_coco, ringless_coco / "instances.json")[0]

        assert targets.boxes.shape == (2, 4)
        assert targets.polygons == []

    def test_oriented_reading_still_requires_a_ring(self, ringless_coco: Path) -> None:
        """An oriented reading keeps skipping ringless annotations; it has no other rotated-box source.

        Locks in that relaxing the plain/keypoints path did not loosen ``oriented=True``,
        which derives both ``boxes`` and ``rboxes`` from the same quadrilateral ring.
        """
        _, targets = CocoDetectionDataset(ringless_coco, ringless_coco / "instances.json", oriented=True)[0]

        assert targets.boxes.shape[0] == 0

    def test_mask_targets_still_raises_without_rings(self, ringless_coco: Path) -> None:
        """Rasterising masks off a ring-free image still raises, not silently returns empty masks.

        `instance_mask_targets` is the guard the relaxed parser now depends on to keep a
        detection-only image from training a segmentation head against nothing.
        """
        _, targets = CocoDetectionDataset(ringless_coco, ringless_coco / "instances.json")[0]

        with pytest.raises(ValueError, match="detection-only annotation cannot supervise masks"):
            instance_mask_targets(targets, image_size=(8, 8), grid_size=(4, 4))


def _write_split(split: Path, payload: dict[str, object], *, side: int = 8) -> Path:
    """Write one blank image and its ``instances.json`` into ``split``, returning the split.

    Examples:
        >>> callable(_write_split)  # needs a tmp_path-backed split directory
        True
    """
    split.mkdir(parents=True, exist_ok=True)
    from torchvision.io import write_png  # noqa: PLC0415 - test-local, the reader does not need it

    for image in cast("list[dict[str, object]]", payload["images"]):
        write_png(torch.zeros(3, side, side, dtype=torch.uint8), str(split / str(image["file_name"])))
    (split / "instances.json").write_text(json.dumps(payload), encoding="utf-8")
    return split


class TestMultiRingAnnotations:
    """A multi-ring COCO polygon keeps one ring, and the box is derived from it (M-38)."""

    def test_the_box_is_the_retained_ring_not_the_annotation_bbox(self, tmp_path: Path) -> None:
        """A two-lobe object reads as the lobe whose ring survived, box and mask together.

        COCO writes ``bbox`` as the envelope of *every* ring, so an object split by an
        occluder gets a box spanning both lobes while this reader keeps only the first.
        Left as it was, the mask branch was supervised to predict emptiness over the
        second lobe's pixels while they sat inside that instance's own box — the box and
        the mask described different objects.
        """
        payload = {
            "images": [{"id": 1, "file_name": "lobes.png", "height": 20, "width": 20}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 18, 18],
                    "segmentation": [
                        [0.0, 0.0, 8.0, 0.0, 8.0, 8.0, 0.0, 8.0],
                        [10.0, 10.0, 18.0, 10.0, 18.0, 18.0, 10.0, 18.0],
                    ],
                }
            ],
            "categories": [{"id": 1, "name": "lobed"}],
        }
        split = _write_split(tmp_path / "val", payload, side=20)

        _, targets = CocoDetectionDataset(split, split / "instances.json")[0]

        assert targets.boxes.tolist() == [[0.0, 0.0, 8.0, 8.0]]
        assert torch.equal(boxes_from_polygons(targets.polygons), targets.boxes)

    def test_an_image_with_any_ringless_instance_keeps_its_bbox_boxes(self, tmp_path: Path) -> None:
        """Where no polygons survive, boxes stay COCO's own ``bbox`` — the rule is per image.

        ``Targets`` holds 0 or N rings, never a mix, so one ringless instance collapses
        the whole image's ``polygons``. There is then no mask to stay in lock-step with,
        and the detection reading wants the annotation's own box; deriving one instance's
        box from a ring the image does not carry would mix two sources inside one image.
        """
        payload = {
            "images": [{"id": 1, "file_name": "mixed.png", "height": 20, "width": 20}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 18, 18],
                    "segmentation": [[0.0, 0.0, 8.0, 0.0, 8.0, 8.0, 0.0, 8.0]],
                },
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [12, 12, 4, 4]},
            ],
            "categories": [{"id": 1, "name": "lobed"}],
        }
        split = _write_split(tmp_path / "val", payload, side=20)

        _, targets = CocoDetectionDataset(split, split / "instances.json")[0]

        assert targets.polygons == []
        assert targets.boxes.tolist() == [[0.0, 0.0, 18.0, 18.0], [12.0, 12.0, 16.0, 16.0]]


class TestDegenerateBoxes:
    """An instance whose final box has no extent is dropped at the read (L-30)."""

    def test_a_box_clamped_flat_is_dropped_with_its_whole_instance(self, tmp_path: Path) -> None:
        """A ``bbox`` outside the canvas clamps to zero width and takes its label and flag with it.

        This is the val/train disagreement the drop closes: the train pipeline's
        ``instance_keep_mask`` removes such an instance after every warp, while the val
        path letterboxes and scores it, so the two paths counted different instance sets
        for the same annotation file.
        """
        payload = {
            "images": [{"id": 1, "file_name": "edge.png", "height": 8, "width": 8}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 2, 3, 3], "difficult": 1},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [2, 2, 3, 3]},
            ],
            "categories": [{"id": 1, "name": "square"}],
        }
        split = _write_split(tmp_path / "val", payload)

        _, targets = CocoDetectionDataset(split, split / "instances.json")[0]

        assert targets.boxes.tolist() == [[2.0, 2.0, 5.0, 5.0]]
        assert targets.difficult.tolist() == [False]

    def test_the_keypoint_channels_are_dropped_in_lock_step(self, tmp_path: Path) -> None:
        """The surviving instance keeps its own points, not the dropped instance's.

        Keypoints are stacked onto the instance axis after the drop, so a filter that
        forgot the channel would leave ``keypoints[0]`` describing an instance that no
        longer has a box — a misalignment no shape check would report.
        """
        payload = {
            "images": [{"id": 1, "file_name": "edge.png", "height": 8, "width": 8}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 2, 3, 3], "keypoints": [1.0, 1.0, 2]},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [2, 2, 3, 3], "keypoints": [3.0, 4.0, 1]},
            ],
            "categories": [{"id": 1, "name": "square", "keypoints": ["apex"]}],
        }
        split = _write_split(tmp_path / "val", payload)

        _, targets = CocoDetectionDataset(split, split / "instances.json", keypoints=True)[0]

        assert targets.keypoints is not None and targets.keypoints.tolist() == [[[3.0, 4.0]]]
        assert targets.keypoint_vis is not None and targets.keypoint_vis.tolist() == [[1]]


def test_a_keypoint_run_names_the_image_whose_annotation_has_no_points(tmp_path: Path) -> None:
    """A missing ``keypoints`` field raises the reader's named error, not a bare ``KeyError`` (L-29).

    Every neighbouring read in the parser is a defaulted ``.get``; this one was a
    subscript, so a keypoint run pointed at a detection export died with a key name and
    no file, image or field context to act on.
    """
    payload = {
        "images": [{"id": 1, "file_name": "pointless.png", "height": 8, "width": 8}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 3, 3]}],
        "categories": [{"id": 1, "name": "square", "keypoints": ["apex"]}],
    }
    split = _write_split(tmp_path / "val", payload)

    with pytest.raises(ValueError, match=r"pointless\.png: keypoints length must be a positive multiple of 3"):
        CocoDetectionDataset(split, split / "instances.json", keypoints=True)


def test_image_ids_name_each_loader_position(tmp_path: Path) -> None:
    """``image_ids`` publishes the COCO id at each position, in the reader's sorted order.

    The whole-image tile merge pairs a loader position to a window and today infers the
    tile's identity from the annotations that landed there — two tiles with identical
    annotation sets are indistinguishable that way. The id is what lets that pairing be
    asserted, so it has to be readable off the dataset and aligned with ``__getitem__``.
    """
    payload = {
        "images": [
            {"id": 7, "file_name": "second.png", "height": 8, "width": 8},
            {"id": 3, "file_name": "first.png", "height": 8, "width": 8},
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "square"}],
    }
    split = _write_split(tmp_path / "val", payload)

    dataset = CocoDetectionDataset(split, split / "instances.json")

    assert dataset.image_ids == (3, 7)
    assert len(dataset.image_ids) == len(dataset)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        pytest.param("n", {"scale": 0.5, "mixup": 0.0, "copy_paste": 0.1}, id="n"),
        pytest.param("s", {"scale": 0.9, "mixup": 0.05, "copy_paste": 0.15}, id="s"),
        pytest.param("m", {"scale": 0.9, "mixup": 0.1, "copy_paste": 0.4}, id="m"),
        pytest.param("l", {"scale": 0.9, "mixup": 0.1, "copy_paste": 0.5}, id="l"),
        pytest.param("x", {"scale": 0.9, "mixup": 0.2, "copy_paste": 0.6}, id="x"),
    ],
)
def test_scale_policy_values(variant: str, expected: dict[str, float]) -> None:
    """Each variant returns its Table S3 augmentation strengths."""
    assert build_scale_policy(variant) == expected


def test_scale_policy_rejects_unknown_variant() -> None:
    """An unknown variant letter raises ``ValueError``."""
    with pytest.raises(ValueError, match="unknown variant"):
        build_scale_policy("z")


def _ragged_targets() -> list[Targets]:
    """Build a mixed batch: an image with boxes+polygons+rbox, an empty one, a box-only image.

    Examples:
        >>> targets = _ragged_targets()
        >>> [t.boxes.shape[0] for t in targets]
        [2, 0, 3]
    """
    with_polys = Targets(
        boxes=torch.tensor([[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 2.0, 2.0]]),
        labels=torch.tensor([3, 7]),
        polygons=[torch.rand(5, 2), torch.rand(3, 2)],
        rboxes=torch.tensor([[1.0, 1.0, 3.0, 2.0, 0.1]]),
    )
    box_only = Targets(
        boxes=torch.tensor([[2.0, 2.0, 6.0, 6.0], [3.0, 3.0, 5.0, 5.0], [0.0, 0.0, 1.0, 1.0]]),
        labels=torch.tensor([0, 1, 2]),
        rboxes=torch.tensor([[2.0, 2.0, 4.0, 3.0, 0.2], [5.0, 5.0, 6.0, 4.0, -0.1]]),
    )
    return [with_polys, Targets.empty(), box_only]


def _assert_targets_identical(actual: list[Targets], expected: list[Targets]) -> None:
    """Assert two target lists match tensor-for-tensor (dtype, shape and value).

    Examples:
        >>> targets = _ragged_targets()
        >>> _assert_targets_identical(targets, targets)  # no output means the lists matched
    """
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        for field in ("boxes", "labels", "rboxes"):
            a, b = getattr(got, field), getattr(want, field)
            assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        assert len(got.polygons) == len(want.polygons)
        for ring_a, ring_b in zip(got.polygons, want.polygons, strict=True):
            assert ring_a.dtype == ring_b.dtype and ring_a.shape == ring_b.shape and torch.equal(ring_a, ring_b)


def test_collate_returns_packed_transport() -> None:
    """Collate stacks equal-size images as uint8 and packs the ragged targets into a PackedTargets."""
    images, packed = collate_detection([(torch.zeros(3, 8, 8), target) for target in _ragged_targets()])
    assert images.shape == (3, 3, 8, 8)
    assert images.dtype == torch.uint8  # transport quantizes the float images to uint8
    assert isinstance(packed, PackedTargets)
    assert packed.boxes_per_image.tolist() == [2, 0, 3]
    assert packed.rings_per_image.tolist() == [2, 0, 0]
    assert packed.rboxes_per_image.tolist() == [1, 0, 2]


def test_collate_quantization_round_trips_within_half_step() -> None:
    """Dequantizing the uint8 transport reproduces the pre-collate float image within one half-step."""
    image = torch.linspace(0.0, 1.0, 3 * 48 * 48).reshape(3, 48, 48)  # full [0, 1] range, hits half-steps
    restored, _ = unpack_batch(collate_detection([(image, Targets.empty())]))
    assert restored.dtype == torch.float32
    assert torch.all((restored >= 0.0) & (restored <= 1.0))
    # 1/510 is the exact-arithmetic half-step bound; the float32 x*255 product and the
    # code/255 division each add a sub-ulp of rounding, so allow a small epsilon over it.
    assert (restored - image).abs().max().item() <= 1.0 / 510.0 + 1e-6


def test_unpack_batch_restores_float_images_and_targets() -> None:
    """unpack_batch restores float32 images and the ragged target list from the transport pair."""
    targets = _ragged_targets()
    images, restored = unpack_batch(collate_detection([(torch.zeros(3, 8, 8), target) for target in targets]))
    assert images.shape == (3, 3, 8, 8)
    assert images.dtype == torch.float32
    _assert_targets_identical(restored, targets)


def test_collate_batch_is_a_small_constant_segment_count() -> None:
    """The packed batch is a handful of tensors regardless of instance count (IPC segment cap)."""
    _images, packed = collate_detection([(torch.zeros(3, 8, 8), target) for target in _ragged_targets()])
    packed_tensors = sum(isinstance(getattr(packed, field.name), torch.Tensor) for field in dataclasses.fields(packed))
    segment_count = 1 + packed_tensors  # the stacked images tensor plus the packed target tensors
    # 13 since WP-132 added the keypoint coordinates, visibilities and per-image count
    # (was 10, itself up from 9 when WP-088 added the R18 difficult flags). The number is
    # pinned only to catch a *ragged* modality creeping back in: what matters is that it
    # does not move with the instance count, which the bound below states.
    assert segment_count == 13
    assert segment_count < 15


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param(_ragged_targets(), id="ragged"),
        pytest.param(
            [Targets(boxes=torch.tensor([[0.0, 0.0, 4.0, 4.0]]), labels=torch.tensor([1])), Targets.empty()],
            id="one-empty",
        ),
        pytest.param([Targets.empty(), Targets.empty()], id="all-empty"),
    ],
)
def test_pack_unpack_round_trip_is_byte_identical(targets: list[Targets]) -> None:
    """Unpacking a packed batch reproduces the input targets tensor-for-tensor."""
    _assert_targets_identical(unpack_targets(pack_targets(targets)), targets)


class TestOnAfterBatchTransfer:
    """Tests for ``DetectionDataModule.on_after_batch_transfer``."""

    def test_leaves_unpacked_batch_untouched(self, detseg_fixture_dir: Path) -> None:
        """An already-unpacked (list) batch passes through the hook defensively unchanged."""
        datamodule = _datamodule(detseg_fixture_dir)
        targets = _ragged_targets()
        images, passed, masks = datamodule.on_after_batch_transfer((torch.zeros(2, 3, 8, 8), targets), 0)
        assert images.shape == (2, 3, 8, 8)
        assert passed is targets
        # No transport means no loader-side rasterisation to report, which is the signal
        # the step rasterises for itself.
        assert masks is None

    def test_dequantizes_uint8_images(self, detseg_fixture_dir: Path) -> None:
        """The hook dequantizes a uint8 transport batch to float32 ``[0, 1]`` on the batch's device."""
        datamodule = _datamodule(detseg_fixture_dir)
        transport, packed = collate_detection([(torch.rand(3, 8, 8), target) for target in _ragged_targets()])
        images, _targets, _ = datamodule.on_after_batch_transfer((transport, packed), 0)
        assert images.dtype == torch.float32
        assert images.device == transport.device
        assert torch.all((images >= 0.0) & (images <= 1.0))

    def test_matches_half_module_dtype(self, detseg_fixture_dir: Path) -> None:
        """The hook restores images at the attached module's dtype, so a half module gets half inputs."""
        module = _SingleParamModule().half()
        trainer = Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)
        trainer.strategy.connect(module)
        datamodule = _datamodule(detseg_fixture_dir)
        datamodule.trainer = trainer
        transport, packed = collate_detection([(torch.rand(3, 8, 8), target) for target in _ragged_targets()])
        images, _targets, _ = datamodule.on_after_batch_transfer((transport, packed), 0)
        assert images.dtype == torch.float16
        assert torch.all((images >= 0.0) & (images <= 1.0))


def _datamodule(
    fixture_dir: Path,
    variant: str = "m",
    seed: int = 0,
    num_workers: int | None = 0,
    pin_memory: bool | None = None,
    val_num_workers: int | None = None,
    persistent_workers: bool = False,
) -> DetectionDataModule:
    """Build a datamodule pointing both splits at the fixture's single split.

    Examples:
        >>> callable(_datamodule)  # needs a live detseg_fixture_dir fixture
        True
    """
    split = fixture_dir / "train"
    annotation = split / "_annotations.coco.json"
    return DetectionDataModule(
        data_root=fixture_dir,
        batch_size=2,
        num_workers=num_workers,
        variant=variant,
        img_size=_SMOKE_IMG_SIZE,
        train_images_dir=split,
        train_ann_file=annotation,
        val_images_dir=split,
        val_ann_file=annotation,
        seed=seed,
        pin_memory=pin_memory,
        val_num_workers=val_num_workers,
        persistent_workers=persistent_workers,
    )


def test_datamodule_train_batch_smoke(detseg_fixture_dir: Path) -> None:
    """One training batch, through the transfer hook, stacks images and lists per-image targets."""
    datamodule = _datamodule(detseg_fixture_dir)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    assert isinstance(batch[1], PackedTargets)  # the loader emits the packed transport form
    assert batch[0].dtype == torch.uint8  # transport ships the images as uint8
    images, targets, _ = datamodule.on_after_batch_transfer(batch, 0)
    assert images.shape == (2, 3, _SMOKE_IMG_SIZE, _SMOKE_IMG_SIZE)
    assert images.dtype == torch.float32  # the hook restores float images
    assert torch.all((images >= 0.0) & (images <= 1.0))
    assert len(targets) == 2
    assert all(isinstance(target, Targets) for target in targets)


def test_datamodule_val_batch_letterboxed(detseg_fixture_dir: Path) -> None:
    """One validation batch, through the transfer hook, is letterbox-only and stacks to size."""
    datamodule = _datamodule(detseg_fixture_dir)
    datamodule.setup("validate")
    batch = next(iter(datamodule.val_dataloader()))
    assert isinstance(batch[1], PackedTargets)  # the loader emits the packed transport form
    assert batch[0].dtype == torch.uint8  # the val path quantizes to uint8 through the same collate
    images, targets, _ = datamodule.on_after_batch_transfer(batch, 0)
    assert images.shape == (2, 3, _SMOKE_IMG_SIZE, _SMOKE_IMG_SIZE)
    assert images.dtype == torch.float32  # the hook restores float images
    assert torch.all((images >= 0.0) & (images <= 1.0))
    assert len(targets) == 2
    assert all(isinstance(target, Targets) for target in targets)


class _SingleParamModule(LightningModule):
    """Minimal LightningModule carrying one parameter, so ``.dtype`` follows ``.half()``."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(1, 1)


def test_unpack_batch_dtype_argument_controls_image_precision() -> None:
    """unpack_batch(dtype=...) restores the images at the requested floating precision."""
    images, _targets = unpack_batch(collate_detection([(torch.zeros(3, 8, 8), Targets.empty())]), dtype=torch.float16)
    assert images.dtype == torch.float16


def test_datamodule_train_batch_is_deterministic(detseg_fixture_dir: Path) -> None:
    """Two datamodules with the same seed yield an identical first train batch."""
    first = _datamodule(detseg_fixture_dir, seed=7)
    second = _datamodule(detseg_fixture_dir, seed=7)
    first.setup("fit")
    second.setup("fit")
    images_a, _ = next(iter(first.train_dataloader()))
    images_b, _ = next(iter(second.train_dataloader()))
    assert torch.equal(images_a, images_b)


@pytest.mark.parametrize(
    "loader_name",
    [pytest.param("train_dataloader", id="train"), pytest.param("val_dataloader", id="val")],
)
def test_dataloader_streaming_kwargs_with_workers(detseg_fixture_dir: Path, loader_name: str) -> None:
    """Worker loaders prefetch deeper, cap worker threads, recycle workers per epoch (WP-076)."""
    datamodule = _datamodule(detseg_fixture_dir, num_workers=2)
    datamodule.setup("fit")
    loader = getattr(datamodule, loader_name)()
    assert loader.prefetch_factor == 2
    assert loader.worker_init_fn is not None
    assert not loader.persistent_workers
    assert loader.pin_memory == torch.cuda.is_available()


@pytest.mark.parametrize(
    "loader_name",
    [pytest.param("train_dataloader", id="train"), pytest.param("val_dataloader", id="val")],
)
def test_dataloader_persistent_workers_opt_in(detseg_fixture_dir: Path, loader_name: str) -> None:
    """persistent_workers=True keeps both loaders' worker pools alive across epochs."""
    datamodule = _datamodule(detseg_fixture_dir, num_workers=2, persistent_workers=True)
    datamodule.setup("fit")
    assert getattr(datamodule, loader_name)().persistent_workers


class _StubWorkerInfo:
    """Minimal stand-in for :class:`torch.utils.data.WorkerInfo` (only what ``_init_worker`` reads)."""

    def __init__(self, dataset: object, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed


def _draw_after_init(monkeypatch: pytest.MonkeyPatch, pipeline: object, seed: int) -> float:
    """Seed ``pipeline`` through ``_init_worker`` as torch would, and return its next draw.

    Examples:
        >>> callable(_draw_after_init)  # needs a live pipeline built by a datamodule
        True
    """
    monkeypatch.setattr(dm, "get_worker_info", lambda: _StubWorkerInfo(pipeline, seed))
    dm._init_worker(0)
    return float(torch.rand((), generator=pipeline._generator))


def test_init_worker_reseeds_pipeline_per_worker_seed(
    detseg_fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinct worker seeds give distinct augmentation streams; an equal seed replays one (WP-079)."""
    datamodule = _datamodule(detseg_fixture_dir, num_workers=2)
    datamodule.setup("fit")
    pipeline = datamodule._train
    first = _draw_after_init(monkeypatch, pipeline, seed=11)
    second = _draw_after_init(monkeypatch, pipeline, seed=12)
    replay = _draw_after_init(monkeypatch, pipeline, seed=11)
    assert first != second
    assert first == replay


@pytest.mark.parametrize(
    "worker_info",
    [
        pytest.param(lambda: _StubWorkerInfo(object(), seed=3), id="dataset-without-generator"),
        pytest.param(lambda: None, id="outside-a-worker-process"),
    ],
)
def test_init_worker_caps_threads_without_a_seedable_dataset(
    monkeypatch: pytest.MonkeyPatch, worker_info: object
) -> None:
    """No ``_generator`` to seed (val pipeline, or num_workers=0) still caps the worker's threads."""
    capped: list[int] = []
    monkeypatch.setattr(dm, "get_worker_info", worker_info)
    monkeypatch.setattr(torch, "set_num_threads", capped.append)
    dm._init_worker(0)
    assert capped == [1]
    assert torch.get_num_threads() == 1


def test_worker_loader_augments_differently_each_epoch(detseg_fixture_dir: Path) -> None:
    """Two epochs of a worker-backed train loader must not replay one augmentation stream (WP-079 guard).

    The unit tests above cover ``_init_worker`` in isolation; this one covers the
    wiring, spawning real workers. Before WP-079 every worker inherited the
    parent's generator state and rebuilt from it each epoch, so the run drew a
    single epoch's parameters forever — invisible to every other test.
    """
    datamodule = _datamodule(detseg_fixture_dir, num_workers=2)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    first = [float(images.float().mean()) for images, _ in loader]
    second = [float(images.float().mean()) for images, _ in loader]
    assert first != second


def test_dataloader_zero_workers_keeps_deterministic_path(detseg_fixture_dir: Path) -> None:
    """The num_workers=0 loader skips prefetch and the worker thread cap entirely."""
    datamodule = _datamodule(detseg_fixture_dir)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    assert loader.num_workers == 0
    assert loader.prefetch_factor is None
    assert loader.worker_init_fn is None


def test_dataloader_num_workers_auto_scales_with_batch(detseg_fixture_dir: Path) -> None:
    """num_workers=None resolves to min(batch_size, cpu count) — never oversubscribes cores."""
    datamodule = _datamodule(detseg_fixture_dir, num_workers=None)
    datamodule.setup("fit")
    assert datamodule.train_dataloader().num_workers <= min(2, os.cpu_count() or 1)
    assert datamodule.train_dataloader().num_workers >= 1


def test_a_named_worker_count_reaches_the_val_loader(detseg_fixture_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker count the caller stated is honoured by both loaders when it fits (WP-102).

    The letterbox cap belongs to what this class chose, not to what an operator named:
    someone writing ``--data.num_workers 8`` has said how much of the machine the run may
    use, and quietly validating on half of it surfaces as an idle accelerator with no
    message, which costs more than the host-OOM the cap was avoiding.
    """
    monkeypatch.setattr(dm, "_shm_capped_workers", lambda workers, *args, **kwargs: workers)

    datamodule = _datamodule(detseg_fixture_dir, num_workers=8)
    datamodule.setup("fit")

    assert datamodule.train_dataloader().num_workers == 8
    assert datamodule.val_dataloader().num_workers == 8


def test_an_inherited_worker_count_is_bounded_by_the_memory_budget(
    detseg_fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A count named for training and inherited by validation still fits the host (WP-103).

    ``--data.num_workers 32`` at 1024 px queues about 51 GB for validation alone, beside
    a training queue that is still resident at the epoch boundary. The operator stated
    parallelism; the gigabytes it costs are this class's arithmetic.
    """
    monkeypatch.setattr(dm, "_shm_capped_workers", lambda workers, *args, **kwargs: min(workers, 3))

    datamodule = _datamodule(detseg_fixture_dir, num_workers=8)
    datamodule.setup("fit")

    with pytest.warns(UserWarning, match="val_num_workers"):
        assert datamodule.val_dataloader().num_workers == 3
    assert datamodule.train_dataloader().num_workers == 8


def test_a_named_val_worker_count_is_bounded_by_nothing(
    detseg_fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``val_num_workers`` stated outright outranks the budget — the flag that says so (WP-103)."""
    monkeypatch.setattr(dm, "_shm_capped_workers", lambda workers, *args, **kwargs: 1)

    datamodule = _datamodule(detseg_fixture_dir, num_workers=8, val_num_workers=6)
    datamodule.setup("fit")

    assert datamodule.val_dataloader().num_workers == 6


def test_an_auto_chosen_worker_count_is_still_capped_for_val(
    detseg_fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the count was this class's own guess, the cap still applies (WP-073).

    The library owns the consequences of a number it picked, and the resident worker
    population doubling at every epoch boundary is one of them.
    """
    monkeypatch.setattr(dm.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(dm, "_shm_capped_workers", lambda workers, *args: workers)

    datamodule = _datamodule(detseg_fixture_dir, num_workers=None)
    datamodule.setup("fit")

    assert datamodule.val_dataloader().num_workers == dm._val_worker_target(_SMOKE_IMG_SIZE)
    assert datamodule.val_dataloader().num_workers < datamodule.train_dataloader().num_workers


def test_the_val_worker_target_scales_with_the_letterbox_side() -> None:
    """The cap encodes how many workers saturate a letterbox-only loader, which is pixel-bound.

    Four was reasoned about 640 px samples, where val work was "a fraction of the train
    pipeline's". At 1024 px the decode is the work, and a constant would starve it.
    """
    assert dm._val_worker_target(640) == 4
    assert dm._val_worker_target(1024) > dm._val_worker_target(640)
    assert dm._val_worker_target(64) >= 1


def test_val_loader_workers_explicit_override(detseg_fixture_dir: Path) -> None:
    """An explicit val_num_workers wins over the capped default, including 0 (deterministic path)."""
    datamodule = _datamodule(detseg_fixture_dir, num_workers=2, val_num_workers=0)
    datamodule.setup("fit")
    loader = datamodule.val_dataloader()
    assert loader.num_workers == 0
    assert loader.prefetch_factor is None
    assert loader.worker_init_fn is None


def test_shm_cap_bounds_workers_to_free_shm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shm cap shrinks the worker count so queued batches fit half the free tmpfs."""
    monkeypatch.setattr(dm.Path, "exists", lambda self: True)
    # 1 GiB free (262144 blocks x 4096); batch bytes = 4*3*64*640*640 ~ 315 MB;
    # prefetch 2 -> budget 512 MiB < one queued slot -> floored to 1 worker
    monkeypatch.setattr(dm.os, "statvfs", lambda _: os.statvfs_result((4096, 4096, 0, 262144, 262144, 0, 0, 0, 0, 255)))
    assert dm._shm_capped_workers(48, batch_size=64, img_size=640, prefetch=2) == 1


def test_shm_cap_no_dev_shm_leaves_workers_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hosts without /dev/shm (macOS/Windows) keep the core-based worker count."""
    monkeypatch.setattr(dm.Path, "exists", lambda self: False)
    assert dm._shm_capped_workers(8, batch_size=64, img_size=640, prefetch=2) == 8


def test_dataloader_pin_memory_explicit_override(detseg_fixture_dir: Path) -> None:
    """An explicit pin_memory value wins over the CUDA auto-resolution."""
    datamodule = _datamodule(detseg_fixture_dir, pin_memory=True)
    datamodule.setup("fit")
    assert datamodule.train_dataloader().pin_memory


def _write_fake_split(images_dir: Path, annotation_file: Path, count: int) -> None:
    """Create ``count`` empty image files and a matching annotation JSON.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     images, ann = Path(tmp) / "images", Path(tmp) / "ann.json"
        ...     _write_fake_split(images, ann, 3)
        ...     len(list(images.iterdir())), len(json.loads(ann.read_text())["images"])
        (3, 3)
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (images_dir / f"img_{index:06d}.jpg").touch()
    annotation_file.write_text(json.dumps({"images": [{"id": i} for i in range(count)]}), encoding="utf-8")


def _write_fake_coco_root(root: Path, train: int, val: int) -> None:
    """Materialise a minimal fake COCO 2017 layout under ``root``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _write_fake_coco_root(root, train=2, val=1)
        ...     sorted(p.name for p in root.iterdir())
        ['annotations', 'train2017', 'val2017']
    """
    annotations = root / "annotations"
    _write_fake_split(root / "train2017", annotations / "instances_train2017.json", train)
    _write_fake_split(root / "val2017", annotations / "instances_val2017.json", val)


class TestCheckData:
    """Tests for ``check_data.check_coco_root``."""

    def test_passes_on_matching_counts(self, tmp_path: Path) -> None:
        """A layout whose disk and annotation counts match the expected totals is OK."""
        _write_fake_coco_root(tmp_path, train=2, val=1)
        result = check_data.check_coco_root(tmp_path, expected_train=2, expected_val=1)
        assert result.ok

    def test_fails_on_count_mismatch(self, tmp_path: Path) -> None:
        """Real COCO totals against a tiny layout fail, and the CLI exits 1."""
        _write_fake_coco_root(tmp_path, train=2, val=1)
        result = check_data.check_coco_root(tmp_path)
        assert not result.ok
        assert data_cli.main(["check", "--data_root", str(tmp_path)]) == 1

    def test_fails_on_missing_root(self, tmp_path: Path) -> None:
        """A root without the expected directories is reported invalid."""
        assert not check_data.check_coco_root(tmp_path / "absent").ok
