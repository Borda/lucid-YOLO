# SPDX-License-Identifier: Apache-2.0
"""Unit gate for WP-098 dataset-layout resolution.

Covers each convention :data:`lucid_yolo.data.layout.CANDIDATES` names, the precedence
between them, the requirement that a candidate's directory *and* annotation file both
exist, the COCO 2017 fallback, and the two consumers that stop open-coding the rule:
:class:`~lucid_yolo.ptl.datamodule.DetectionDataModule` and the oriented evaluator's
datamodule builder.

No dataset is read: every root here is an empty directory tree written in ``tmp_path``,
because resolution is a naming question and answering it must not require an image.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lucid_yolo.data.layout import CANDIDATES, resolve_split
from lucid_yolo.eval.rotated_eval import build_datamodule
from lucid_yolo.ptl.datamodule import DetectionDataModule


def _write_layout(root: Path, images: str, annotations: str) -> tuple[Path, Path]:
    """Create one layout's directory and annotation file under ``root``.

    Args:
        root: Dataset root to build inside.
        images: Images directory path relative to ``root``.
        annotations: Annotation file path relative to ``root``.

    Returns:
        The ``(images directory, annotation file)`` pair created.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     images_dir, ann_file = _write_layout(Path(tmp), "val2017", "annotations/instances_val2017.json")
        ...     images_dir.is_dir(), ann_file.read_text()
        (True, '{}')
    """
    images_dir = root / images
    annotation_file = root / annotations
    images_dir.mkdir(parents=True, exist_ok=True)
    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    annotation_file.write_text("{}", encoding="utf-8")
    return images_dir, annotation_file


@pytest.mark.parametrize(
    ("images", "annotations"),
    [pytest.param(images, annotations, id=images.replace("/", "-")) for images, annotations in CANDIDATES],
)
def test_every_named_layout_resolves_to_itself(tmp_path: Path, images: str, annotations: str) -> None:
    """Each convention in the table is found when a root is written in exactly that shape."""
    expected = _write_layout(tmp_path, images.format(split="val"), annotations.format(split="val"))

    assert resolve_split(tmp_path, "val") == expected


def test_an_unknown_layout_falls_back_to_the_coco_2017_names(tmp_path: Path) -> None:
    """A root matching nothing resolves to COCO 2017, so the reader reports the path it wanted.

    Resolution is a naming rule, not an existence check: raising here would turn every
    mistyped root into this module's error instead of the reader's, and the reader's
    names the file it failed to open.
    """
    images_dir, annotation_file = resolve_split(tmp_path, "train")

    assert images_dir == tmp_path / "train2017"
    assert annotation_file == tmp_path / "annotations" / "instances_train2017.json"


def test_a_directory_without_its_annotation_file_is_not_a_match(tmp_path: Path) -> None:
    """Both halves must exist, so a root cannot be resolved into one layout's dir and another's JSON."""
    (tmp_path / "val").mkdir()

    assert resolve_split(tmp_path, "val")[0] == tmp_path / "val2017"


def test_coco_2017_wins_when_a_root_satisfies_two_layouts(tmp_path: Path) -> None:
    """Precedence is the table's order, so a root satisfying two resolves the same way every time."""
    _write_layout(tmp_path, "val", "annotations/instances_val.json")
    expected = _write_layout(tmp_path, "val2017", "annotations/instances_val2017.json")

    assert resolve_split(tmp_path, "val") == expected


def test_the_datamodule_needs_no_overrides_for_a_tiled_root(tmp_path: Path) -> None:
    """A tiled oriented root is reached from ``data_root`` alone — the four overrides were the defect."""
    train = _write_layout(tmp_path, "train", "annotations/instances_train.json")
    val = _write_layout(tmp_path, "val", "annotations/instances_val.json")

    datamodule = DetectionDataModule(data_root=tmp_path, batch_size=2, num_workers=0, variant="n")

    assert (datamodule._train_images_dir, datamodule._train_ann_file) == train
    assert (datamodule._val_images_dir, datamodule._val_ann_file) == val


def test_an_explicit_override_still_wins(tmp_path: Path) -> None:
    """Resolution is a default: a caller naming a path gets that path, matched layout or not."""
    _write_layout(tmp_path, "val", "annotations/instances_val.json")
    elsewhere = tmp_path / "elsewhere"

    datamodule = DetectionDataModule(
        data_root=tmp_path, batch_size=2, num_workers=0, variant="n", val_images_dir=elsewhere
    )

    assert datamodule._val_images_dir == elsewhere


def test_the_oriented_evaluator_resolves_the_same_way(tmp_path: Path) -> None:
    """The evaluator's builder reads the shared rule rather than its own copy of the tiled names."""
    val = _write_layout(tmp_path, "val", "annotations/instances_val.json")

    datamodule = build_datamodule(tmp_path, "val", img_size=1024, batch_size=2, variant="n")

    assert (datamodule._val_images_dir, datamodule._val_ann_file) == val
