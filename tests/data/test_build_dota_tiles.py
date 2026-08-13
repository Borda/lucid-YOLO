# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the tiled-layout build (WP-094).

Real DOTA-v1.0 is not on this machine and is never auto-downloaded (AGENTS.md sec. 3),
so every test here builds a tiny synthetic DOTA split on disk — images plus ``labelTxt``
in R18's own line format — and converts it.

The contract under test is a round trip rather than a file format: what the build writes
must be what :class:`~lucid_yolo.data.coco.CocoDetectionDataset` reads back under
``oriented=True``, with the geometry, the class order and R18's ``difficult`` flag intact.
Asserting the JSON's shape alone would pass while the reader saw something else, which is
the failure this build exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lucid_yolo.data import tiles as build
from lucid_yolo.data.coco import CocoDetectionDataset
from lucid_yolo.data.dota import DOTA_CLASSES
from lucid_yolo.data.tiling import tile_windows

#: Patch and overlap used throughout: small enough to tile a 24 px image into four windows.
PATCH = 16
OVERLAP = 8
#: The synthetic source image side.
SIDE = 24


def _write_split(root: Path, split: str, labels: dict[str, list[str]]) -> Path:
    """Write a synthetic DOTA split: one PNG plus one label file per entry."""
    split_dir = root / split
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "labelTxt").mkdir(parents=True)
    for stem, lines in labels.items():
        image = (torch.arange(3 * SIDE * SIDE, dtype=torch.uint8) % 251).reshape(3, SIDE, SIDE)
        build.write_png(image, str(split_dir / "images" / f"{stem}.png"))
        (split_dir / "labelTxt" / f"{stem}.txt").write_text("\n".join(["imagesource:synthetic", *lines]) + "\n")
    return split_dir


def _axis_aligned_line(x0: float, y0: float, x1: float, y1: float, name: str, difficult: int) -> str:
    """One R18 label line for an axis-aligned rectangle, corners clockwise from top-left."""
    return f"{x0} {y0} {x1} {y0} {x1} {y1} {x0} {y1} {name} {difficult}"


@pytest.fixture
def built_root(tmp_path: Path) -> Path:
    """Build a two-image split holding one wholly-contained and one straddling object."""
    _write_split(
        tmp_path / "dota",
        "val",
        {
            "P0001": [_axis_aligned_line(2.0, 2.0, 6.0, 4.0, "plane", 0)],
            "P0002": [_axis_aligned_line(6.0, 6.0, 10.0, 10.0, "harbor", 1)],
        },
    )
    build.convert_split(tmp_path / "dota" / "val", tmp_path / "tiles", "val", patch=PATCH, overlap=OVERLAP)
    return tmp_path / "tiles"


def _payload(root: Path, split: str = "val") -> dict[str, list[dict[str, object]]]:
    """Read a built split's instances JSON."""
    return json.loads((root / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))


def _dataset_index(root: Path, file_name: str, split: str = "val") -> int:
    """Index of one tile in the dataset, which reads images in the JSON's own order."""
    names = [str(record["file_name"]) for record in _payload(root, split)["images"]]
    return names.index(file_name)


def test_every_window_of_every_source_image_is_written(built_root: Path) -> None:
    """The build emits one tile per window of each source image, keeping empty ones (A52)."""
    per_image = len(tile_windows((SIDE, SIDE), patch=PATCH, overlap=OVERLAP))

    payload = _payload(built_root)

    assert len(payload["images"]) == 2 * per_image
    assert all((built_root / "val" / str(record["file_name"])).is_file() for record in payload["images"])


def test_dropping_empty_tiles_keeps_only_annotated_windows(tmp_path: Path) -> None:
    """``keep_empty=False`` writes exactly the tiles that carry an annotation (A52)."""
    _write_split(tmp_path / "dota", "val", {"P0001": [_axis_aligned_line(2.0, 2.0, 6.0, 4.0, "plane", 0)]})

    report = build.convert_split(
        tmp_path / "dota" / "val", tmp_path / "tiles", "val", patch=PATCH, overlap=OVERLAP, keep_empty=False
    )

    payload = _payload(tmp_path / "tiles")
    assert report.tiles == len(payload["images"]) < len(tile_windows((SIDE, SIDE), patch=PATCH, overlap=OVERLAP))
    assert all(record["image_id"] in {r["id"] for r in payload["images"]} for record in payload["annotations"])


def test_the_reader_recovers_the_source_geometry(built_root: Path) -> None:
    """An object wholly inside a window reads back as the same rectangle, window-local.

    The source rectangle spans (2, 2)-(6, 4) of ``P0001``, so in the window at the origin
    it is unmoved: a 4x2 box centred on (4, 3) at angle 0 in canonical long-edge form.
    """
    dataset = CocoDetectionDataset(built_root / "val", built_root / "annotations" / "instances_val.json", oriented=True)
    index = _dataset_index(built_root, "P0001__0_0.png")

    _, targets = dataset[index]

    assert targets.rboxes.shape == (1, 5)
    assert torch.allclose(targets.rboxes[0], torch.tensor([4.0, 3.0, 4.0, 2.0, 0.0]), atol=1e-4)
    assert torch.equal(targets.boxes[0], torch.tensor([2.0, 2.0, 6.0, 4.0]))


def test_the_difficult_flag_survives_the_round_trip(built_root: Path) -> None:
    """R18's flag reaches ``Targets.difficult`` through the layout (A51, A53).

    ``P0002``'s object is flagged difficult in its label file, so every tile that carries
    any part of it must read back flagged — the incoming flag is never cleared by tiling.
    """
    dataset = CocoDetectionDataset(built_root / "val", built_root / "annotations" / "instances_val.json", oriented=True)
    names = [str(record["file_name"]) for record in _payload(built_root)["images"]]
    indices = [index for index, name in enumerate(names) if name.startswith("P0002")]

    flags = [bool(dataset[i][1].difficult.all()) for i in indices if dataset[i][1].boxes.shape[0]]

    assert flags, "no tile of the difficult object carried an annotation"
    assert all(flags)


def test_tiling_flags_a_clipped_part_difficult(tmp_path: Path) -> None:
    """A part below R18's 0.7 area fraction is flagged, though its source object is not."""
    _write_split(tmp_path / "dota", "val", {"P0003": [_axis_aligned_line(6.0, 6.0, 14.0, 14.0, "plane", 0)]})

    build.convert_split(tmp_path / "dota" / "val", tmp_path / "tiles", "val", patch=PATCH, overlap=OVERLAP)

    annotations = _payload(tmp_path / "tiles")["annotations"]
    assert any(int(record["difficult"]) == 1 for record in annotations)
    assert any(float(record["visible_fraction"]) < 1.0 for record in annotations)


def test_category_ids_preserve_the_published_class_order(built_root: Path) -> None:
    """The reader's contiguous label equals the DOTA index, for every class (R18)."""
    dataset = CocoDetectionDataset(built_root / "val", built_root / "annotations" / "instances_val.json", oriented=True)

    labels = {name: dataset.category_id_to_label[index + 1] for index, name in enumerate(DOTA_CLASSES)}

    assert labels == {name: index for index, name in enumerate(DOTA_CLASSES)}


def test_image_records_carry_their_window_provenance(built_root: Path) -> None:
    """Each tile names the source image and the window it came from (A53, WP-064's input)."""
    records = _payload(built_root)["images"]

    windows = {str(record["file_name"]): (record["source_image"], tuple(record["window"])) for record in records}  # type: ignore[arg-type]

    assert windows["P0001__0_0.png"] == ("P0001.png", (0, 0, PATCH, PATCH))
    assert all(name.startswith(str(source).split(".")[0]) for name, (source, _) in windows.items())


def test_a_directory_that_is_not_a_dota_split_is_named_not_guessed(tmp_path: Path) -> None:
    """A missing ``labelTxt/`` fails with the path in the message rather than tiling nothing."""
    (tmp_path / "dota" / "val" / "images").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="labelTxt"):
        build.convert_split(tmp_path / "dota" / "val", tmp_path / "tiles", "val")


def test_the_report_counts_what_was_written(tmp_path: Path) -> None:
    """The printed report's counts agree with the JSON it wrote."""
    _write_split(
        tmp_path / "dota",
        "val",
        {
            "P0001": [
                _axis_aligned_line(2.0, 2.0, 6.0, 4.0, "plane", 0),
                _axis_aligned_line(6.0, 6.0, 10.0, 10.0, "ship", 1),
            ]
        },
    )

    report = build.convert_split(tmp_path / "dota" / "val", tmp_path / "tiles", "val", patch=PATCH, overlap=OVERLAP)

    payload = _payload(tmp_path / "tiles")
    assert (report.tiles, report.instances) == (len(payload["images"]), len(payload["annotations"]))
    assert report.difficult == sum(int(record["difficult"]) for record in payload["annotations"])
    assert report.source_images == 1


def _multi_image_split(root: Path) -> Path:
    """Write a four-image split, enough that a pool's completion order can differ from source order."""
    return _write_split(
        root / "dota",
        "val",
        {
            "P0001": [_axis_aligned_line(2.0, 2.0, 6.0, 4.0, "plane", 0)],
            "P0002": [_axis_aligned_line(6.0, 6.0, 10.0, 10.0, "harbor", 1)],
            "P0003": [_axis_aligned_line(1.0, 1.0, 3.0, 9.0, "ship", 0)],
            "P0004": [_axis_aligned_line(12.0, 12.0, 20.0, 16.0, "bridge", 0)],
        },
    )


def test_a_pooled_build_writes_the_same_bytes_as_a_serial_one(tmp_path: Path) -> None:
    """Workers change the schedule, never the file: ids are assigned by the parent in source order.

    This is the whole risk of the pool. A worker cannot know how many tiles preceded it,
    so numbering anything inside one would make the JSON depend on which process finished
    first -- reproducible on the machine that built it and nowhere else.
    """
    split_dir = _multi_image_split(tmp_path)

    build.convert_split(split_dir, tmp_path / "serial", "val", patch=PATCH, overlap=OVERLAP, progress=False)
    build.convert_split(split_dir, tmp_path / "pooled", "val", patch=PATCH, overlap=OVERLAP, workers=4, progress=False)

    serial = (tmp_path / "serial" / "annotations" / "instances_val.json").read_bytes()
    assert serial == (tmp_path / "pooled" / "annotations" / "instances_val.json").read_bytes()


def test_a_pooled_build_writes_every_tile_image(tmp_path: Path) -> None:
    """The workers do the writing, so the images must be on disk and not only in the JSON."""
    split_dir = _multi_image_split(tmp_path)

    report = build.convert_split(
        split_dir, tmp_path / "tiles", "val", patch=PATCH, overlap=OVERLAP, workers=2, progress=False
    )

    payload = _payload(tmp_path / "tiles")
    assert report.tiles == len(payload["images"])
    assert all((tmp_path / "tiles" / "val" / str(record["file_name"])).is_file() for record in payload["images"])


def test_no_record_keeps_a_worker_placeholder_id(tmp_path: Path) -> None:
    """Every id a worker left unnumbered is overwritten; the placeholder is out of COCO's range."""
    split_dir = _multi_image_split(tmp_path)

    build.convert_split(split_dir, tmp_path / "tiles", "val", patch=PATCH, overlap=OVERLAP, workers=2, progress=False)

    payload = _payload(tmp_path / "tiles")
    assert all(int(record["id"]) > 0 for record in payload["images"])
    assert all(int(record["id"]) > 0 and int(record["image_id"]) > 0 for record in payload["annotations"])


def test_the_worker_count_reaches_the_split_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--workers`` is forwarded rather than accepted and dropped, and defaults to one per CPU."""
    seen: list[int] = []

    def _record(*args: object, **kwargs: object) -> build.SplitReport:
        seen.append(int(kwargs["workers"]))  # type: ignore[call-overload]
        return build.SplitReport("val", 0, 0, 0, 0, 0)

    monkeypatch.setattr(build, "convert_split", _record)
    monkeypatch.setattr(build.os, "cpu_count", lambda: 7)

    assert build.build_tiles(tmp_path, tmp_path / "out", splits="val", workers=3) == 0
    assert build.build_tiles(tmp_path, tmp_path / "out", splits="val") == 0
    assert seen == [3, 7]


def test_jpeg_tiles_are_written_and_read_back(tmp_path: Path) -> None:
    """``--suffix .jpg`` writes JPEG tiles that the reader loads through the same layout."""
    split_dir = _multi_image_split(tmp_path)

    build.convert_split(
        split_dir, tmp_path / "tiles", "val", patch=PATCH, overlap=OVERLAP, progress=False, suffix=".jpg"
    )

    payload = _payload(tmp_path / "tiles")
    names = [str(record["file_name"]) for record in payload["images"]]
    assert names and all(name.endswith(".jpg") for name in names)
    assert all((tmp_path / "tiles" / "val" / name).is_file() for name in names)
    dataset = CocoDetectionDataset(
        tmp_path / "tiles" / "val", tmp_path / "tiles" / "annotations" / "instances_val.json", oriented=True
    )
    assert len(dataset) == len(names)


def test_the_geometry_survives_a_lossy_encoder(tmp_path: Path) -> None:
    """JPEG changes pixels, never annotations: the records are identical to the PNG build's.

    Worth pinning rather than assuming. The encoder sits between the crop and the disk,
    and nothing about an annotation passes through it -- if a suffix ever reached the
    geometry, that is a bug the pixel-level difference would hide.
    """
    split_dir = _multi_image_split(tmp_path)

    build.convert_split(split_dir, tmp_path / "png", "val", patch=PATCH, overlap=OVERLAP, progress=False)
    build.convert_split(split_dir, tmp_path / "jpg", "val", patch=PATCH, overlap=OVERLAP, progress=False, suffix=".jpg")

    assert _payload(tmp_path / "png")["annotations"] == _payload(tmp_path / "jpg")["annotations"]


def test_an_unknown_tile_suffix_is_rejected(tmp_path: Path) -> None:
    """A format the encoder cannot write fails at the flag, not after the first image."""
    with pytest.raises(ValueError, match="unknown tile suffix"):
        build.build_tiles(tmp_path, tmp_path / "out", splits="val", suffix=".gif")
