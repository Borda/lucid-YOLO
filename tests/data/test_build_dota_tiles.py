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

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from lucid_yolo.data.coco import CocoDetectionDataset
from lucid_yolo.data.dota import DOTA_CLASSES
from lucid_yolo.data.tiling import tile_windows

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path) -> ModuleType:
    """Load a ``scripts/`` module by file path (``scripts`` is not a package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered before exec
    spec.loader.exec_module(module)
    return module


build = _load_module("build_dota_tiles", REPO_ROOT / "scripts" / "build_dota_tiles.py")

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
