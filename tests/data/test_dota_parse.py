# SPDX-License-Identifier: Apache-2.0
"""Unit gate for WP-056 DOTA-v1.0 label parsing and long-edge conversion (A39).

Covers the ``labelTxt`` reader (header and blank-line skipping, the 15 categories in
every spelling the files use, the ``difficult`` flag, loud rejection of anything
malformed), the quadrilateral-to-long-edge conversion, the 1:1 rotated-box/instance-axis
invariant WP-058/061/088 rely on, and the ``check-data`` DOTA layout validator.

Real DOTA-v1.0 is not on this machine and is never auto-downloaded (AGENTS.md sec. 3), so
every case here runs against fixtures: hand-written ``labelTxt`` text for the parser, and
a DOTA-shaped root built in ``tmp_path`` from the seeded ``fuse-augmentations`` OBB
micro-set (A26) for the validator. The published 2,806 / 188,282 / 15 totals are
therefore exercised as *parameters* — the code path is proven, the real numbers are not.

No RNG is used beyond the seeded fixture generator: every parser input is written out.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from lucid_yolo.cli import data as data_cli
from lucid_yolo.data import DOTA_CLASSES, DotaObject, dota_targets, load_dota_targets, parse_dota_label_file
from lucid_yolo.data import check as check_data

#: Split of the OBB micro-set, and the Roboflow-style annotation file its generator emits.
_FIXTURE_SPLIT = "train"
_FIXTURE_ANNOTATION = "_annotations.coco.json"
#: Split directory the DOTA-shaped fixture root is built under.
_SPLIT = "train"

#: A rotated rectangle written out as its four corners, clockwise as displayed (y-down).
#: Centre ``(10, 20)``, long edge ``8`` along ``u = (0.8, 0.6)``, short edge ``4`` along
#: ``v = (-0.6, 0.8)``: a 3-4-5 slope, so every corner is an exact decimal and the angle
#: (36.87 deg) sits well clear of the +/-45 deg square tie-break boundary.
_ROTATED_QUAD_LINE = "8.0 16.0 14.4 20.8 12.0 24.0 5.6 19.2 plane 0"
_ROTATED_RBOX = (10.0, 20.0, 8.0, 4.0, 0.6435011)
_ROTATED_ENVELOPE = (5.6, 16.0, 14.4, 24.0)

#: Every category, hyphenated the way the label files write the multi-word names.
_ALL_CATEGORY_LINES = "\n".join(f"0 0 2 0 2 1 0 1 {name.replace(' ', '-')} 0" for name in DOTA_CLASSES)

#: Three objects of distinct size, so an instance can be told from its neighbours by the
#: long edge alone — which is what makes the 1:1 axis pairing observable.
_THREE_OBJECT_LINES = "\n".join(
    (
        "0 0 2 0 2 1 0 1 plane 0",
        "0 0 6 0 6 3 0 3 harbor 1",
        "0 0 10 0 10 5 0 5 bridge 0",
    )
)


def _write_labels(path: Path, text: str) -> Path:
    """Write ``text`` as a ``labelTxt`` file at ``path`` and return the path."""
    path.write_text(text, encoding="utf-8")
    return path


@dataclass(frozen=True)
class _DotaFixture:
    """A DOTA-shaped fixture root and the counts a validator should find in it.

    Attributes:
        root: The dataset root, holding one ``train`` split.
        images: Image files written under ``train/images``.
        instances: Object lines written across ``train/labelTxt``.
        difficult: How many of those lines carry ``difficult = 1``.
        classes: Distinct class ids used.
    """

    root: Path
    images: int
    instances: int
    difficult: int
    classes: int


def _build_dota_root(obb_dir: Path, root: Path) -> _DotaFixture:
    """Rewrite the seeded OBB micro-set into a DOTA-v1.0 directory layout.

    Each COCO annotation of the OBB set stores its rotated box as four corner points, so
    it transcribes directly onto a DOTA object line; category ids are mapped onto
    :data:`DOTA_CLASSES` cyclically and every third object is flagged difficult, so both
    ``difficult`` policies have something to act on.
    """
    document = json.loads((obb_dir / _FIXTURE_SPLIT / _FIXTURE_ANNOTATION).read_text(encoding="utf-8"))
    by_image: dict[int, list[dict[str, object]]] = {}
    for annotation in document["annotations"]:
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    images_dir, labels_dir = root / _SPLIT / "images", root / _SPLIT / "labelTxt"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    instances = difficult = 0
    classes: set[int] = set()
    for image in document["images"]:
        file_name = str(image["file_name"])
        shutil.copyfile(obb_dir / _FIXTURE_SPLIT / file_name, images_dir / file_name)
        lines = ["imagesource:synthetic", "gsd:null"]
        for annotation in by_image.get(int(image["id"]), []):
            label = int(annotation["category_id"]) % len(DOTA_CLASSES)
            corners = " ".join(f"{value:.2f}" for value in annotation["segmentation"][0])
            flag = int(instances % 3 == 0)
            lines.append(f"{corners} {DOTA_CLASSES[label].replace(' ', '-')} {flag}")
            instances += 1
            difficult += flag
            classes.add(label)
        _write_labels(labels_dir / f"{Path(file_name).stem}.txt", "\n".join(lines) + "\n")
    return _DotaFixture(
        root=root,
        images=len(document["images"]),
        instances=instances,
        difficult=difficult,
        classes=len(classes),
    )


@pytest.fixture
def dota_fixture(obb_fixture_dir: Path, tmp_path: Path) -> _DotaFixture:
    """Yield a DOTA-shaped root built from the seeded OBB micro-set (A26)."""
    return _build_dota_root(obb_fixture_dir, tmp_path / "dota")


def test_header_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    """Metadata headers (quoted or bare) and blank lines contribute no objects."""
    text = (
        "imagesource:GoogleEarth\n"
        "gsd:0.146343590398\n"
        "'acquisition dates':2015-01-01\n"
        "\n"
        "0 0 2 0 2 1 0 1 plane 0\n"
        "\n"
        "0 0 4 0 4 2 0 2 ship 0\n"
    )

    objects = parse_dota_label_file(_write_labels(tmp_path / "P0001.txt", text))

    assert [obj.label for obj in objects] == [0, 1]


def test_clockwise_quad_becomes_the_expected_long_edge_box(tmp_path: Path) -> None:
    """A rotated rectangle's four corners convert to its canonical (cx, cy, w, h, theta)."""
    targets = load_dota_targets(_write_labels(tmp_path / "P0002.txt", _ROTATED_QUAD_LINE), keep_difficult=True)

    assert torch.allclose(targets.rboxes[0], torch.tensor(_ROTATED_RBOX), atol=1e-5)
    assert torch.allclose(targets.boxes[0], torch.tensor(_ROTATED_ENVELOPE), atol=1e-5)


def test_every_category_resolves_to_its_index(tmp_path: Path) -> None:
    """All 15 DOTA-v1.0 categories parse, in the hyphenated spelling the files use."""
    objects = parse_dota_label_file(_write_labels(tmp_path / "P0003.txt", _ALL_CATEGORY_LINES))

    assert [obj.label for obj in objects] == list(range(len(DOTA_CLASSES)))


@pytest.mark.parametrize(
    "written",
    [
        pytest.param("storage-tank", id="hyphenated"),
        pytest.param("storage_tank", id="underscored"),
        pytest.param("Storage-Tank", id="mixed-case-hyphenated"),
        pytest.param("STORAGE_TANK", id="upper-case-underscored"),
        pytest.param("storage--tank", id="repeated-separator"),
    ],
)
def test_multi_word_category_spellings_normalize_to_one_class(tmp_path: Path, written: str) -> None:
    """Hyphen, underscore and case variants of a multi-word name resolve to the same id."""
    objects = parse_dota_label_file(_write_labels(tmp_path / "P0004.txt", f"0 0 2 0 2 1 0 1 {written} 0\n"))

    assert [obj.label for obj in objects] == [DOTA_CLASSES.index("storage tank")]


def test_unknown_category_is_rejected(tmp_path: Path) -> None:
    """An unlisted category raises, naming the file, the line and both spellings."""
    text = "gsd:0.1\n0 0 2 0 2 1 0 1 plane 0\n0 0 2 0 2 1 0 1 container-crane 0\n"
    path = _write_labels(tmp_path / "P0005.txt", text)

    with pytest.raises(ValueError, match=r"P0005\.txt:3: unknown category 'container-crane'"):
        parse_dota_label_file(path)


def test_difficult_flag_round_trips_through_the_parse(tmp_path: Path) -> None:
    """The per-object difficult flag survives parsing exactly as written."""
    text = "0 0 2 0 2 1 0 1 plane 0\n0 0 4 0 4 2 0 2 ship 1\n"

    objects = parse_dota_label_file(_write_labels(tmp_path / "P0006.txt", text))

    assert [obj.difficult for obj in objects] == [False, True]


@pytest.mark.parametrize(
    ("keep_difficult", "expected_labels"),
    [
        pytest.param(True, [0, 1], id="kept"),
        pytest.param(False, [0], id="dropped"),
    ],
)
def test_keep_difficult_selects_which_instances_reach_the_targets(
    tmp_path: Path, keep_difficult: bool, expected_labels: list[int]
) -> None:
    """A39: the caller decides the fate of difficult instances, at load time."""
    text = "0 0 2 0 2 1 0 1 plane 0\n0 0 4 0 4 2 0 2 ship 1\n"
    path = _write_labels(tmp_path / "P0007.txt", text)

    targets = load_dota_targets(path, keep_difficult=keep_difficult)

    assert targets.labels.tolist() == expected_labels
    assert targets.rboxes.shape[0] == len(expected_labels)


def test_rboxes_share_the_instance_axis_with_boxes_and_labels(tmp_path: Path) -> None:
    """rboxes[i] is the same instance as boxes[i]/labels[i], so one mask filters both."""
    path = _write_labels(tmp_path / "P0008.txt", _THREE_OBJECT_LINES)

    targets = load_dota_targets(path, keep_difficult=True)
    kept = targets.filter(torch.tensor([True, False, True]), rkeep=torch.tensor([True, False, True]))

    assert targets.boxes.shape[0] == targets.labels.shape[0] == targets.rboxes.shape[0] == 3
    assert targets.rboxes[:, 2].tolist() == [2.0, 6.0, 10.0]
    assert kept.labels.tolist() == [0, 8]
    assert kept.rboxes[:, 2].tolist() == [2.0, 10.0]
    assert kept.boxes[:, 2].tolist() == [2.0, 10.0]


def test_targets_carry_no_polygons(tmp_path: Path) -> None:
    """The quad is carried by rboxes alone; polygons stays empty for the oriented path."""
    targets = load_dota_targets(_write_labels(tmp_path / "P0009.txt", _THREE_OBJECT_LINES), keep_difficult=True)

    assert targets.polygons == []


@pytest.mark.parametrize(
    ("line", "message"),
    [
        pytest.param("0 0 2 0 2 1 0 1 plane", r"expected 10 fields", id="missing-difficult"),
        pytest.param("0 0 2 0 2 1 0 1 plane 0 extra", r"expected 10 fields", id="trailing-field"),
        pytest.param("0 0 2 0 2 1 0 x plane 0", r"coordinates must be numeric", id="non-numeric-coordinate"),
        pytest.param("0 0 2 0 2 1 0 1 plane 2", r"difficult flag must be 0 or 1", id="out-of-range-difficult"),
        pytest.param("0 0 2 0 2 1 0 1 plane yes", r"difficult flag must be 0 or 1", id="worded-difficult"),
    ],
)
def test_malformed_object_line_is_rejected(tmp_path: Path, line: str, message: str) -> None:
    """Every malformed line raises, with the file and the 1-based line number named."""
    path = _write_labels(tmp_path / "P0010.txt", f"gsd:0.1\n{line}\n")

    with pytest.raises(ValueError, match=rf"P0010\.txt:2: {message}"):
        parse_dota_label_file(path)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="zero-byte"),
        pytest.param("imagesource:GoogleEarth\ngsd:null\n", id="headers-only"),
        pytest.param("\n\n", id="blank-lines-only"),
    ],
)
def test_empty_label_file_gives_empty_targets(tmp_path: Path, text: str) -> None:
    """A file with no objects yields a valid, correctly-shaped empty target set."""
    targets = load_dota_targets(_write_labels(tmp_path / "P0011.txt", text), keep_difficult=True)

    assert targets.boxes.shape == (0, 4)
    assert targets.labels.shape == (0,)
    assert targets.rboxes.shape == (0, 5)
    assert targets.polygons == []


def test_objects_can_be_converted_without_a_file() -> None:
    """dota_targets works on hand-built objects, keeping the three axes aligned."""
    quad = torch.tensor([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])

    targets = dota_targets([DotaObject(polygon=quad, label=3, difficult=False)], keep_difficult=False)

    assert targets.labels.tolist() == [3]
    assert targets.rboxes.tolist() == [[2.0, 1.0, 4.0, 2.0, 0.0]]


def test_check_data_passes_on_a_dota_fixture_root(dota_fixture: _DotaFixture) -> None:
    """The DOTA validator accepts a well-formed root when told that root's own counts."""
    result = check_data.check_dota_root(
        dota_fixture.root,
        splits=(_SPLIT,),
        expected_images=dota_fixture.images,
        expected_instances=dota_fixture.instances,
        expected_classes=dota_fixture.classes,
    )

    assert result.ok, result.problems


def test_check_data_counts_difficult_instances_too(dota_fixture: _DotaFixture) -> None:
    """Instance counting is what is on disk: difficult lines count toward the total (A39)."""
    result = check_data.check_dota_root(dota_fixture.root, splits=(_SPLIT,))

    assert dota_fixture.difficult > 0
    assert result.splits[0].instances == dota_fixture.instances


def test_check_data_reports_totals_it_was_not_given_without_failing(dota_fixture: _DotaFixture) -> None:
    """An unstated total is a note, not a problem: a well-formed root passes on its layout alone.

    The published 2,806 / 188,282 / 15 were the defaults until WP-097, which made
    ``lucid-data check --dataset dota`` fail on every correct download: R18 sec. 4
    withholds the testing ground truth, so the annotated root holds about two thirds
    of the published images and can never sum to them.
    """
    result = check_data.check_dota_root(dota_fixture.root, splits=(_SPLIT,))

    assert result.ok, result.problems
    assert any(f"{dota_fixture.instances} instances" in note for note in result.notes)
    assert str(check_data.DOTA_IMAGE_COUNT) not in check_data.format_report(result, dota_fixture.root)


def test_check_data_fails_on_a_total_it_was_given(dota_fixture: _DotaFixture) -> None:
    """A stated expectation is still enforced, and names both the wanted and the found count."""
    result = check_data.check_dota_root(
        dota_fixture.root,
        splits=(_SPLIT,),
        expected_images=check_data.DOTA_IMAGE_COUNT,
    )

    assert not result.ok
    assert any(str(check_data.DOTA_IMAGE_COUNT) in problem for problem in result.problems)
    assert (
        data_cli.main(
            [
                "check",
                "--data_root",
                str(dota_fixture.root),
                "--dataset",
                "dota",
                "--expected_images",
                str(check_data.DOTA_IMAGE_COUNT),
            ]
        )
        == 1
    )


def test_the_annotated_totals_are_the_published_ones_less_the_testing_third() -> None:
    """The measured train-plus-val counts sit where R18's split ratios put them.

    R18 sec. 4 takes half the images as training, a sixth as validation and a third as
    testing, releasing ground truth for the first two: the annotated share is therefore
    about two thirds of the published dataset. About, not exactly: the selection is
    random, so the realised split lands near its ratios rather than on them -- the
    measured images come to 0.666 of 2,806 where the arithmetic would give 1,871 -- and
    instances are not distributed uniformly over images anyway. The band is sized for
    what a transcription error would leave: a count copied from one split, or from
    another dataset version, misses it by far more than the sampling does.
    """
    images = check_data.DOTA_ANNOTATED_IMAGE_COUNT / check_data.DOTA_IMAGE_COUNT
    instances = check_data.DOTA_ANNOTATED_INSTANCE_COUNT / check_data.DOTA_INSTANCE_COUNT

    assert 0.62 < images < 0.71
    assert 0.62 < instances < 0.71


def test_check_data_rejects_a_dota_expectation_aimed_at_coco() -> None:
    """A DOTA-only count passed with --dataset coco is rejected rather than ignored."""
    with pytest.raises(ValueError, match="apply to --dataset dota"):
        check_data.check_dataset(Path("/nonexistent"), dataset="coco", expected_images=1)


def test_check_data_reports_an_image_without_a_label_file(dota_fixture: _DotaFixture) -> None:
    """Pairing is checked by stem in both directions."""
    orphaned = sorted((dota_fixture.root / _SPLIT / "labelTxt").glob("*.txt"))[0]
    orphaned.unlink()

    result = check_data.check_dota_root(dota_fixture.root, splits=(_SPLIT,), expected_images=dota_fixture.images)

    assert any("without a label file" in problem for problem in result.splits[0].problems)


def test_check_data_reports_a_label_file_without_an_image(dota_fixture: _DotaFixture) -> None:
    """A label file whose image is absent is a problem, not a silently extra annotation."""
    _write_labels(dota_fixture.root / _SPLIT / "labelTxt" / "P9999.txt", "0 0 2 0 2 1 0 1 plane 0\n")

    result = check_data.check_dota_root(dota_fixture.root, splits=(_SPLIT,))

    assert any("without an image" in problem for problem in result.splits[0].problems)


def test_check_data_reports_an_unparsable_label_file(dota_fixture: _DotaFixture) -> None:
    """A malformed label file is reported as a problem rather than raising out of the check."""
    broken = sorted((dota_fixture.root / _SPLIT / "labelTxt").glob("*.txt"))[0]
    _write_labels(broken, "0 0 2 0 2 1 0 1 container-crane 0\n")

    result = check_data.check_dota_root(dota_fixture.root, splits=(_SPLIT,))

    assert any("failed to parse" in problem for problem in result.splits[0].problems)


def test_check_data_reports_missing_split_directories(tmp_path: Path) -> None:
    """An empty root names both directories the split is expected to hold."""
    result = check_data.check_dota_root(tmp_path, splits=(_SPLIT,))

    assert not result.ok
    assert any("images directory missing" in problem for problem in result.splits[0].problems)
    assert any("labelTxt directory missing" in problem for problem in result.splits[0].problems)


def test_check_data_coco_path_is_the_default(tmp_path: Path) -> None:
    """The CLI still validates a COCO root unless --dataset says otherwise."""
    assert data_cli.main(["check", "--data_root", str(tmp_path)]) == 1
    assert not check_data.check_coco_root(tmp_path).ok
