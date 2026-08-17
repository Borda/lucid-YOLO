# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-099 YOLO-format reader.

Every root here is written into ``tmp_path`` (A26): a ``data.yaml``, a handful of tiny PNGs
and the label text files beside them. Nothing is downloaded and no real dataset is read —
the format is a directory convention plus a line grammar, and both can be stated exactly in
a fixture.

Covered: the ``data.yaml`` (class names in index order, the published ``../`` split entry,
a contradictory ``nc``), both layout spellings, the normalized-to-pixel conversion asserted
against hand-computed pixel values, the oriented variant's polygon rows and the instance-axis
pairing they produce, the empty-versus-missing label file policy, and every way a row can be
malformed — each rejected with the file *and* the 1-based line named.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torchvision.io import write_png

from lucid_yolo.data.layout import resolve_yolo_split
from lucid_yolo.data.rotated_aug import rbox_envelopes
from lucid_yolo.data.yolo import (
    DATA_YAML_NAME,
    YoloDataConfig,
    YoloDetectionDataset,
    load_yolo_targets,
)

#: Fixture image size, deliberately non-square so an x/y swap cannot pass.
_WIDTH = 100
_HEIGHT = 40
#: Class names the fixture ``data.yaml`` publishes, in index order.
_NAMES = ("car", "truck")
#: Pixel/geometry tolerance for the rotated fit, which goes through a float32 mean of edges.
_TOL = 1e-4

#: One detection row and the pixel ``xyxy`` box it must denormalize to at the fixture size:
#: ``cx=0.5 -> 50``, ``cy=0.5 -> 20``, ``w=0.5 -> 50``, ``h=0.25 -> 10``.
_DETECTION_ROW = "1 0.5 0.5 0.5 0.25"
_DETECTION_BOX = [25.0, 15.0, 75.0, 25.0]
#: A second row whose box runs off the left edge once denormalized, so the canvas clamp shows.
_EDGE_ROW = "0 0.05 0.5 0.2 0.5"
_EDGE_BOX = [0.0, 10.0, 15.0, 30.0]

#: An axis-aligned oriented row: pixel corners (10, 10), (50, 10), (50, 30), (10, 30).
_ORIENTED_AXIS_ROW = "0 0.1 0.25 0.5 0.25 0.5 0.75 0.1 0.75"
_ORIENTED_AXIS_BOX = [10.0, 10.0, 50.0, 30.0]
_ORIENTED_AXIS_RBOX = [30.0, 20.0, 40.0, 20.0, 0.0]
#: A rotated oriented row: pixel corners (0, 10), (20, 30), (30, 20), (10, 0) — a rectangle
#: whose long edge runs at 45 degrees, so the fit cannot be mistaken for an envelope.
_ORIENTED_TILTED_ROW = "1 0.0 0.25 0.2 0.75 0.3 0.5 0.1 0.0"
_ORIENTED_TILTED_BOX = [0.0, 0.0, 30.0, 30.0]
_ORIENTED_TILTED_RBOX = [15.0, 15.0, 28.284271, 14.142136, 0.785398]


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed the global RNG before each test (the reader itself draws nothing)."""
    torch.manual_seed(0)
    yield


def _write_image(path: Path) -> None:
    """Write one fixture-sized PNG at ``path``, creating its directory.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "images" / "frame.png"
        ...     _write_image(path)
        ...     path.is_file()
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(torch.zeros(3, _HEIGHT, _WIDTH, dtype=torch.uint8), str(path))


def _write_labels(path: Path, rows: list[str]) -> Path:
    """Write one label file holding ``rows``, creating its directory.

    Args:
        path: Label file path.
        rows: Label lines without terminators; an empty list writes an empty file.

    Returns:
        The path written.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = _write_labels(Path(tmp) / "labels" / "frame.txt", ["0 0.5 0.5 0.1 0.1"])
        ...     path.read_text()
        '0 0.5 0.5 0.1 0.1\\n'
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
    return path


def _write_data_yaml(root: Path, entry: str = "../train/images", names: tuple[str, ...] = _NAMES) -> Path:
    """Write a ``data.yaml`` in the shape a published export uses.

    Args:
        root: Dataset root to write into.
        entry: The ``train`` split entry, verbatim.
        names: Class names, written as a YAML list.

    Returns:
        The path of the written file.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = _write_data_yaml(Path(tmp))
        ...     "nc: " + str(len(_NAMES)) in path.read_text()
        True
    """
    listed = "".join(f"- {name}\n" for name in names)
    text = f"names:\n{listed}nc: {len(names)}\nroboflow:\n  license: CC BY 4.0\ntrain: {entry}\n"
    path = root / DATA_YAML_NAME
    path.write_text(text, encoding="utf-8")
    return path


def _build_root(root: Path, rows: list[str], *, spelling: str = "per-split") -> Path:
    """Materialise a one-image, one-split YOLO root and return it.

    Args:
        root: Directory to build inside.
        rows: The single image's label rows.
        spelling: ``"per-split"`` for ``train/images``, ``"split-subdir"`` for
            ``images/train``.

    Returns:
        ``root``, now holding a ``data.yaml``, one image and its label file.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = _build_root(Path(tmp), ["0 0.5 0.5 0.1 0.1"])
        ...     (root / "train" / "images" / "frame.png").is_file()
        True
    """
    images, labels = (
        (root / "train" / "images", root / "train" / "labels")
        if spelling == "per-split"
        else (root / "images" / "train", root / "labels" / "train")
    )
    _write_image(images / "frame.png")
    _write_labels(labels / "frame.txt", rows)
    _write_data_yaml(root, entry="../train/images" if spelling == "per-split" else "images/train")
    return root


class TestDataYaml:
    """The dataset's own ``data.yaml``, which supplies the class names and the split paths."""

    def test_names_are_read_in_index_order(self, tmp_path: Path) -> None:
        """The ``names`` list *is* the label space, so its order is preserved verbatim.

        Unlike COCO's sparse category ids, a YOLO row's class field indexes this list
        directly; reordering or remapping it would silently relabel every annotation.
        """
        config = YoloDataConfig.read(_write_data_yaml(tmp_path))

        assert config.names == _NAMES

    def test_a_mapping_of_names_is_accepted_when_it_is_index_keyed(self, tmp_path: Path) -> None:
        """``names`` written as ``{0: ..., 1: ...}`` resolves by its own keys, not by file order.

        The mapping form states each index explicitly, so accepting it needs no convention;
        it is read back in key order regardless of how the file listed the entries.
        """
        path = tmp_path / DATA_YAML_NAME
        path.write_text("names:\n  1: truck\n  0: car\nval: valid/images\n", encoding="utf-8")

        assert YoloDataConfig.read(path).names == _NAMES

    def test_a_mapping_that_is_not_index_keyed_is_rejected(self, tmp_path: Path) -> None:
        """Keys that are not exactly ``0..K-1`` leave the class order unstated, so the file is refused.

        Guessing an order here would assign labels the dataset never claimed — the silent
        mis-parse this reader exists to prevent.
        """
        path = tmp_path / DATA_YAML_NAME
        path.write_text("names:\n  1: truck\n  3: car\nval: valid/images\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"must be keyed by 0\.\.K-1"):
            YoloDataConfig.read(path)

    def test_an_nc_contradicting_names_is_rejected(self, tmp_path: Path) -> None:
        """A file disagreeing with itself about its class count is refused rather than half-trusted.

        ``nc`` is redundant with ``names``; when the two differ, neither can be taken as the
        class space, and a run started on the wrong one only shows up as bad metrics.
        """
        path = tmp_path / DATA_YAML_NAME
        path.write_text("names:\n- car\n- truck\nnc: 3\ntrain: train/images\n", encoding="utf-8")

        with pytest.raises(ValueError, match="nc=3 contradicts 2 names"):
            YoloDataConfig.read(path)

    def test_the_published_parent_prefixed_entry_resolves(self, tmp_path: Path) -> None:
        """``train: ../train/images`` beside a real ``<root>/train/images`` resolves to that tree.

        This is what the published export writes; read literally it escapes the root, so the
        prefix is absorbed after the literal reading is tried and misses.
        """
        (tmp_path / "train" / "images").mkdir(parents=True)
        config = YoloDataConfig.read(_write_data_yaml(tmp_path))

        assert config.images_dir("train") == tmp_path / "train" / "images"

    def test_an_entry_resolving_to_nothing_names_every_path_tried(self, tmp_path: Path) -> None:
        """An unresolvable split entry raises listing its candidates rather than guessing a third.

        A dataset whose split directory is absent is a provisioning problem; the message has
        to be enough to see which spelling was expected.
        """
        config = YoloDataConfig.read(_write_data_yaml(tmp_path))

        with pytest.raises(FileNotFoundError, match="resolves to no directory; tried"):
            config.images_dir("train")

    def test_an_unnamed_split_is_a_key_error_listing_what_the_file_has(self, tmp_path: Path) -> None:
        """Asking for a split the file never named reports which splits it did name.

        An export that ships only ``train`` is legitimate; the caller asking for ``val``
        needs to learn that from the file, not from an empty dataset.
        """
        config = YoloDataConfig.read(_write_data_yaml(tmp_path))

        with pytest.raises(KeyError, match="no 'val' split"):
            config.images_dir("val")


class TestLayoutResolution:
    """The YOLO naming convention, the sibling of the COCO one a root resolves by."""

    @pytest.mark.parametrize(
        ("images", "labels"),
        [
            pytest.param("train/images", "train/labels", id="per-split"),
            pytest.param("images/train", "labels/train", id="split-subdir"),
        ],
    )
    def test_both_published_spellings_resolve_to_themselves(self, tmp_path: Path, images: str, labels: str) -> None:
        """A root written in either spelling is found from ``data_root`` alone.

        The per-split form is what the read export uses and the split-subdirectory form is
        what the roadmap row names; both are conventions in the wild, so both are rows.
        """
        (tmp_path / images).mkdir(parents=True)
        (tmp_path / labels).mkdir(parents=True)

        assert resolve_yolo_split(tmp_path, "train") == (tmp_path / images, tmp_path / labels)

    def test_a_half_present_layout_is_not_a_match(self, tmp_path: Path) -> None:
        """Images without their labels tree does not match, so two spellings cannot be mixed.

        The same both-halves rule the COCO table uses — resolving into one convention's
        images and another's labels would pair every image with the wrong annotations.
        """
        (tmp_path / "images" / "train").mkdir(parents=True)

        assert resolve_yolo_split(tmp_path, "train")[0] == tmp_path / "train" / "images"

    def test_an_unknown_layout_falls_back_to_the_per_split_names(self, tmp_path: Path) -> None:
        """A root matching nothing resolves to the first row, so the reader names the path it wanted.

        Resolution is a naming rule, not an existence check: raising here would replace the
        reader's "no such directory" with this table's, which says less.
        """
        images, labels = resolve_yolo_split(tmp_path, "val")

        assert (images, labels) == (tmp_path / "val" / "images", tmp_path / "val" / "labels")


class TestDetectionReading:
    """Five-field ``cls cx cy w h`` rows, denormalized against the image's own size."""

    def test_rows_denormalize_to_hand_computed_pixel_boxes(self, tmp_path: Path) -> None:
        """``cx cy w h`` fractions become ``xyxy`` pixels using width for x and height for y.

        The expected corners are computed by hand from the 100x40 fixture, not from the
        reader — a reader compared against itself would pass with x and y transposed.
        """
        label_file = _write_labels(tmp_path / "frame.txt", [_DETECTION_ROW])

        targets = load_yolo_targets(label_file, height=_HEIGHT, width=_WIDTH, num_classes=2, oriented=False)

        assert targets.boxes.tolist() == [_DETECTION_BOX]
        assert targets.labels.tolist() == [1]

    def test_a_box_running_past_an_edge_is_clamped_to_the_canvas(self, tmp_path: Path) -> None:
        """A normalized row whose box extends past an image edge is clipped, as the COCO path clips.

        The file's own numbers must be normalized (rejected otherwise), but the geometry they
        describe may still round or reach past a border; the canvas is where that is settled.
        """
        label_file = _write_labels(tmp_path / "frame.txt", [_EDGE_ROW])

        targets = load_yolo_targets(label_file, height=_HEIGHT, width=_WIDTH, num_classes=2, oriented=False)

        assert targets.boxes.tolist() == [_EDGE_BOX]

    def test_an_empty_file_is_a_background_image(self, tmp_path: Path) -> None:
        """A zero-row label file yields empty targets, the format's way of saying "no objects".

        Background images are a normal part of a detection split; the empty file is the
        positive statement that distinguishes them from an image whose labels went missing.
        """
        label_file = _write_labels(tmp_path / "frame.txt", [])

        targets = load_yolo_targets(label_file, height=_HEIGHT, width=_WIDTH, num_classes=2, oriented=False)

        assert targets.boxes.shape == (0, 4)
        assert targets.labels.shape == (0,)

    def test_the_format_carries_no_difficult_flag(self, tmp_path: Path) -> None:
        """Every instance reads non-difficult, which is what A51's default already means.

        The format has no per-instance flag column, so the channel exists and is uniformly
        ``False`` — a consumer may index it beside ``labels`` without asking who produced it.
        """
        label_file = _write_labels(tmp_path / "frame.txt", [_DETECTION_ROW, _EDGE_ROW])

        targets = load_yolo_targets(label_file, height=_HEIGHT, width=_WIDTH, num_classes=2, oriented=False)

        assert targets.difficult.tolist() == [False, False]


class TestOrientedReading:
    """Nine-field rows carrying eight normalized polygon coordinates."""

    def test_polygon_rows_become_paired_boxes_and_rotated_boxes(self, tmp_path: Path) -> None:
        """Each row's quad denormalizes to a rotated box and to that same quad's envelope.

        The tilted row is the load-bearing one: its envelope and its rotated box differ, so a
        reader that quietly returned the envelope as the rotated box would fail here.
        """
        label_file = _write_labels(tmp_path / "frame.txt", [_ORIENTED_AXIS_ROW, _ORIENTED_TILTED_ROW])

        targets = load_yolo_targets(label_file, height=_HEIGHT, width=_WIDTH, num_classes=2, oriented=True)

        assert targets.labels.tolist() == [0, 1]
        assert torch.allclose(targets.boxes, torch.tensor([_ORIENTED_AXIS_BOX, _ORIENTED_TILTED_BOX]), atol=_TOL)
        assert torch.allclose(targets.rboxes, torch.tensor([_ORIENTED_AXIS_RBOX, _ORIENTED_TILTED_RBOX]), atol=_TOL)

    def test_every_modality_shares_one_instance_axis(self, tmp_path: Path) -> None:
        """``boxes[i]``, ``labels[i]`` and ``rboxes[i]`` describe one object, and polygons stay empty.

        This is the invariant the rotated transforms and the assigner rely on: one mask keeps
        all modalities aligned. Both boxes come from the same ring, so the axis-aligned box
        must be the envelope of the rotated one it is listed beside.
        """
        label_file = _write_labels(tmp_path / "frame.txt", [_ORIENTED_AXIS_ROW, _ORIENTED_TILTED_ROW])

        targets = load_yolo_targets(label_file, height=_HEIGHT, width=_WIDTH, num_classes=2, oriented=True)

        assert targets.boxes.shape[0] == targets.labels.shape[0] == targets.rboxes.shape[0] == 2
        assert targets.polygons == []
        assert torch.allclose(rbox_envelopes(targets.rboxes), targets.boxes, atol=_TOL)

    def test_the_variant_is_declared_and_never_sniffed(self, tmp_path: Path) -> None:
        """A nine-field file read as detection is rejected, not silently re-read as oriented.

        Deducing the variant from the field count would make a truncated oriented export
        parse as a plausible detection file; the caller states which dataset it opened.
        """
        label_file = _write_labels(tmp_path / "frame.txt", [_ORIENTED_AXIS_ROW])

        with pytest.raises(ValueError, match="expected 5 fields"):
            load_yolo_targets(label_file, height=_HEIGHT, width=_WIDTH, num_classes=2, oriented=False)


class TestMalformedRows:
    """Every way a row can be wrong, each rejected with its file and line named."""

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            pytest.param("0 0.5 0.5", "expected 5 fields", id="too-few-fields"),
            pytest.param("0 0.5 0.5 0.2 0.2 0.2", "expected 5 fields", id="too-many-fields"),
            pytest.param("0 0.5 nan_x 0.2 0.2", "coordinates must be numeric", id="non-numeric-coordinate"),
            pytest.param("0 0.5 0.5 1.5 0.2", re.escape("outside [0, 1]"), id="unnormalized-coordinate"),
            pytest.param("0 0.5 -0.1 0.2 0.2", re.escape("outside [0, 1]"), id="negative-coordinate"),
            pytest.param("car 0.5 0.5 0.2 0.2", "class index must be an integer", id="non-integer-class"),
            pytest.param("7 0.5 0.5 0.2 0.2", re.escape("class index 7 is outside 0..1"), id="unknown-class"),
        ],
    )
    def test_a_malformed_row_names_its_file_and_line(self, tmp_path: Path, row: str, message: str) -> None:
        """Each distinct mistake raises its own message, prefixed with ``<file>:<line>``.

        The bad row sits third so the line number is a real count rather than a constant, and
        the four mistakes are told apart: a wrong field count, an unparseable coordinate, a
        coordinate that was never normalized, and a class token naming no class.
        """
        label_file = _write_labels(tmp_path / "frame.txt", [_DETECTION_ROW, _DETECTION_ROW, row])

        with pytest.raises(ValueError, match=f"{re.escape(str(label_file))}:3: .*{message}"):
            load_yolo_targets(label_file, height=_HEIGHT, width=_WIDTH, num_classes=2, oriented=False)

    def test_an_oriented_row_with_a_trailing_flag_is_rejected(self, tmp_path: Path) -> None:
        """A ten-field oriented row is refused rather than read as eight coordinates plus noise.

        R18's own label lines carry a trailing ``difficult`` flag; the normalized variant has
        no published spelling for one here, so an extra field is a file this reader cannot
        account for rather than a column to drop.
        """
        label_file = _write_labels(tmp_path / "frame.txt", [f"{_ORIENTED_AXIS_ROW} 1"])

        with pytest.raises(ValueError, match=f"{re.escape(str(label_file))}:1: expected 9 fields"):
            load_yolo_targets(label_file, height=_HEIGHT, width=_WIDTH, num_classes=2, oriented=True)

    def test_blank_lines_are_skipped_without_shifting_the_line_count(self, tmp_path: Path) -> None:
        """Blank lines are ignored, and the reported line number still counts them.

        A trailing newline or a blank separator is not a malformed row, but the number in the
        message has to match what an editor shows or it cannot be acted on.
        """
        label_file = _write_labels(tmp_path / "frame.txt", [_DETECTION_ROW, "", "0 0.5 0.5"])

        with pytest.raises(ValueError, match=f"{re.escape(str(label_file))}:3:"):
            load_yolo_targets(label_file, height=_HEIGHT, width=_WIDTH, num_classes=2, oriented=False)


class TestDatasetRoundTrip:
    """The dataset built from a root, yielding the same pairs the COCO reader does."""

    @pytest.mark.parametrize(
        "spelling",
        [pytest.param("per-split", id="per-split"), pytest.param("split-subdir", id="split-subdir")],
    )
    def test_a_root_round_trips_from_data_root_alone(self, tmp_path: Path, spelling: str) -> None:
        """Both spellings reach their images and labels from the root, with names from the yaml.

        This is the whole point of the layout row: a YOLO root is usable from
        ``--data.data_root`` like every other, without restating four paths.
        """
        root = _build_root(tmp_path, [_DETECTION_ROW], spelling=spelling)

        dataset = YoloDetectionDataset.from_root(root, "train")
        image, targets = dataset[0]

        assert dataset.names == _NAMES
        assert len(dataset) == 1
        assert image.shape == (3, _HEIGHT, _WIDTH)
        assert image.dtype == torch.float32
        assert targets.boxes.tolist() == [_DETECTION_BOX]

    def test_the_oriented_root_round_trips_with_every_modality_aligned(self, tmp_path: Path) -> None:
        """An oriented root yields boxes, labels and rotated boxes on one instance axis.

        The DoD's round trip: the file on disk, through the loader, into the same ``Targets``
        container the COCO path produces, with the pairing the rotated transforms need.
        """
        root = _build_root(tmp_path, [_ORIENTED_AXIS_ROW, _ORIENTED_TILTED_ROW])

        image, targets = YoloDetectionDataset.from_root(root, "train", oriented=True)[0]

        assert image.shape == (3, _HEIGHT, _WIDTH)
        assert targets.labels.tolist() == [0, 1]
        assert torch.allclose(targets.rboxes, torch.tensor([_ORIENTED_AXIS_RBOX, _ORIENTED_TILTED_RBOX]), atol=_TOL)
        assert torch.allclose(rbox_envelopes(targets.rboxes), targets.boxes, atol=_TOL)
        assert targets.difficult.tolist() == [False, False]

    def test_images_are_indexed_in_sorted_name_order(self, tmp_path: Path) -> None:
        """The sorted image listing is the index, since the format publishes no manifest.

        Directory iteration order is filesystem-dependent; sorting is what makes sample ``i``
        the same image on every host and every run.
        """
        root = _build_root(tmp_path, [_DETECTION_ROW])
        for stem in ("aaa", "zzz"):
            _write_image(root / "train" / "images" / f"{stem}.png")
            _write_labels(root / "train" / "labels" / f"{stem}.txt", [_DETECTION_ROW])

        dataset = YoloDetectionDataset.from_root(root, "train")

        assert [path.stem for path in dataset._images] == ["aaa", "frame", "zzz"]

    def test_a_missing_label_file_is_rejected_by_default(self, tmp_path: Path) -> None:
        """An image with no label file raises, naming both the image and the path looked for.

        A YOLO tree has no image manifest, so an absent file is indistinguishable from a
        half-finished export; reading it as a background image would train on it silently.
        """
        root = _build_root(tmp_path, [_DETECTION_ROW])
        (root / "train" / "labels" / "frame.txt").unlink()

        with pytest.raises(FileNotFoundError, match="no label file at"):
            _ = YoloDetectionDataset.from_root(root, "train")[0]

    def test_a_missing_label_file_may_be_opted_into(self, tmp_path: Path) -> None:
        """``allow_missing_labels=True`` reads the absence as a background image instead.

        A dataset that genuinely omits files for its background images stays usable, but the
        caller has to say so — the default keeps the broken export loud.
        """
        root = _build_root(tmp_path, [_DETECTION_ROW])
        (root / "train" / "labels" / "frame.txt").unlink()

        _, targets = YoloDetectionDataset.from_root(root, "train", allow_missing_labels=True)[0]

        assert targets.boxes.shape == (0, 4)
