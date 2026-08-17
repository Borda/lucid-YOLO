# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-099b datamodule layout dispatch.

Every root here is written into ``tmp_path`` (A26): a two-image COCO root, the same two images
written as a YOLO tree beside a ``data.yaml``, one root holding both and one holding neither.
Nothing is downloaded, and the images are 64x48 PNGs so a letterbox to 32 px still resamples
something real.

Covered: each layout reaching its own reader from ``data_root`` alone — the flag WP-099 left
serving only the COCO one — the refusal to break a tie on a root satisfying both, the error a
root satisfying neither raises (naming both conventions and every path tried), the explicit
``layout`` that settles either case, and the two things the YOLO path cannot serve that the
COCO one does: segmentation mask targets, refused, and copy-paste, suppressed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torchvision.io import write_png

from lucid_yolo.data.coco import CocoDetectionDataset, build_scale_policy
from lucid_yolo.data.layout import DATA_YAML_NAME, DatasetLayout
from lucid_yolo.data.yolo import YoloDetectionDataset
from lucid_yolo.ptl.datamodule import DetectionDataModule, unpack_batch

#: Fixture image size, deliberately non-square so an x/y transposition cannot pass unnoticed.
_WIDTH = 64
_HEIGHT = 48
#: Letterbox side every emitted sample is resampled to.
_IMG_SIZE = 32
#: Images written per split — two, so the mosaic draws from more than one source.
_IMAGE_STEMS = ("frame0", "frame1")
#: Class names the YOLO ``data.yaml`` publishes, in index order.
_NAMES = ("car", "truck")
#: One detection row: a centred box half the width and a quarter the height of the image.
_DETECTION_ROW = "0 0.5 0.5 0.5 0.25"
#: One oriented row: the same box as a quadrilateral, corners in order.
_ORIENTED_ROW = "1 0.25 0.375 0.75 0.375 0.75 0.625 0.25 0.625"
#: Variant whose policy has the strongest copy-paste probability, so a suppression shows.
_COPY_PASTE_VARIANT = "x"


def _write_image(path: Path) -> None:
    """Write one fixture-sized PNG at ``path``, creating its directory.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     png = Path(tmp) / "sub" / "frame0.png"
        ...     _write_image(png)
        ...     png.is_file()
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(torch.zeros(3, _HEIGHT, _WIDTH, dtype=torch.uint8), str(path))


def _coco_payload() -> dict[str, object]:
    """Return a COCO ``instances`` payload for the fixture's two images.

    Each image carries one annotation with a four-point segmentation ring, so the reader keeps
    it on the detection and the segmentation readings alike.

    Returns:
        The payload, ready to serialise.

    Examples:
        >>> payload = _coco_payload()
        >>> sorted(payload.keys())
        ['annotations', 'categories', 'images']
        >>> len(payload["images"])
        2
    """
    return {
        "images": [
            {"id": index, "file_name": f"{stem}.png", "width": _WIDTH, "height": _HEIGHT}
            for index, stem in enumerate(_IMAGE_STEMS, start=1)
        ],
        "annotations": [
            {
                "id": index,
                "image_id": index,
                "category_id": 1,
                "bbox": [16.0, 12.0, 32.0, 12.0],
                "segmentation": [[16.0, 12.0, 48.0, 12.0, 48.0, 24.0, 16.0, 24.0]],
                "iscrowd": 0,
            }
            for index, _ in enumerate(_IMAGE_STEMS, start=1)
        ],
        "categories": [{"id": 1, "name": "car"}],
    }


def _write_coco_root(root: Path) -> Path:
    """Materialise a two-split COCO root: ``<root>/<split>/`` beside ``annotations/``.

    Args:
        root: Directory to build inside.

    Returns:
        ``root``, now holding both splits and their annotation files.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = _write_coco_root(Path(tmp))
        ...     sorted(p.name for p in (root / "train").iterdir())
        ['frame0.png', 'frame1.png']
    """
    for split in ("train", "val"):
        for stem in _IMAGE_STEMS:
            _write_image(root / split / f"{stem}.png")
        annotation_file = root / "annotations" / f"instances_{split}.json"
        annotation_file.parent.mkdir(parents=True, exist_ok=True)
        annotation_file.write_text(json.dumps(_coco_payload()), encoding="utf-8")
    return root


def _write_yolo_root(root: Path, row: str = _DETECTION_ROW) -> Path:
    """Materialise a two-split YOLO root in the shape a published export uses.

    The ``val`` split is written under ``valid/`` and named by the ``data.yaml`` as
    ``../valid/images`` — the published spelling (R32), which no convention table finds — so
    building that split exercises the dataset's own entry rather than the fallback.

    Args:
        root: Directory to build inside.
        row: The label row every image carries.

    Returns:
        ``root``, now holding ``data.yaml``, both image trees and their label trees.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = _write_yolo_root(Path(tmp))
        ...     (root / "data.yaml").is_file()
        True
    """
    for directory in ("train", "valid"):
        for stem in _IMAGE_STEMS:
            _write_image(root / directory / "images" / f"{stem}.png")
            label_file = root / directory / "labels" / f"{stem}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            label_file.write_text(f"{row}\n", encoding="utf-8")
    listed = "".join(f"- {name}\n" for name in _NAMES)
    text = f"names:\n{listed}nc: {len(_NAMES)}\ntrain: ../train/images\nval: ../valid/images\n"
    (root / DATA_YAML_NAME).write_text(text, encoding="utf-8")
    return root


def _datamodule(data_root: Path, **kwargs: object) -> DetectionDataModule:
    """Build a datamodule over ``data_root`` with the offline, deterministic loader settings.

    Args:
        data_root: The dataset root under test.
        **kwargs: Overrides forwarded verbatim to the datamodule.

    Returns:
        The constructed datamodule; no split is built until ``setup``.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dm = _datamodule(Path(tmp))
        ...     type(dm).__name__
        'DetectionDataModule'
    """
    settings: dict[str, object] = {
        "batch_size": 2,
        "num_workers": 0,
        "variant": "n",
        "img_size": _IMG_SIZE,
    }
    settings.update(kwargs)
    return DetectionDataModule(data_root=data_root, **settings)  # type: ignore[arg-type]


class TestLayoutDispatch:
    """Which reader a ``data_root`` reaches, under one flag and with no overrides."""

    def test_a_coco_root_reaches_the_coco_reader(self, tmp_path: Path) -> None:
        """The incumbent layout is unchanged: a COCO root still builds both splits from JSON."""
        datamodule = _datamodule(_write_coco_root(tmp_path))
        datamodule.setup("fit")

        assert datamodule.layout is DatasetLayout.COCO
        assert isinstance(datamodule._train._base, CocoDetectionDataset)
        assert isinstance(datamodule._val, CocoDetectionDataset)

    def test_a_yolo_root_reaches_the_yolo_reader(self, tmp_path: Path) -> None:
        """The same flag on a YOLO root builds both splits from its label trees — the WP's point."""
        datamodule = _datamodule(_write_yolo_root(tmp_path))
        datamodule.setup("fit")

        assert datamodule.layout is DatasetLayout.YOLO
        assert isinstance(datamodule._train._base, YoloDetectionDataset)
        assert isinstance(datamodule._val, YoloDetectionDataset)

    def test_the_yolo_val_split_comes_from_the_datasets_own_entry(self, tmp_path: Path) -> None:
        """``val: ../valid/images`` resolves, so the dataset's spelling outranks the convention."""
        datamodule = _datamodule(_write_yolo_root(tmp_path))
        datamodule.setup("fit")

        assert isinstance(datamodule._val, YoloDetectionDataset)
        assert len(datamodule._val) == len(_IMAGE_STEMS)

    def test_a_root_satisfying_neither_names_both_conventions(self, tmp_path: Path) -> None:
        """The failure WP-098's resolver was written against: an error that says what it looked for."""
        datamodule = _datamodule(tmp_path)

        with pytest.raises(FileNotFoundError) as failure:
            datamodule.setup("fit")

        message = str(failure.value)
        assert "COCO" in message
        assert "YOLO" in message
        for expected in (
            str(tmp_path / "train2017"),
            str(tmp_path / "annotations" / "instances_train2017.json"),
            str(tmp_path / "annotations" / "instances_val.json"),
            str(tmp_path / "images" / "val"),
            str(tmp_path / DATA_YAML_NAME),
            str(tmp_path / "train" / "labels"),
            str(tmp_path / "labels" / "val"),
        ):
            assert expected in message

    def test_a_root_satisfying_both_refuses_to_choose(self, tmp_path: Path) -> None:
        """Across the two conventions the loser is a different label space, so no precedence is taken."""
        _write_coco_root(tmp_path)
        _write_yolo_root(tmp_path)
        datamodule = _datamodule(tmp_path)

        with pytest.raises(ValueError, match="both dataset conventions") as failure:
            datamodule.setup("fit")

        assert "layout='coco'" in str(failure.value)
        assert "layout='yolo'" in str(failure.value)

    @pytest.mark.parametrize(
        ("layout", "reader"),
        [("coco", CocoDetectionDataset), ("yolo", YoloDetectionDataset)],
    )
    def test_an_explicit_layout_settles_an_ambiguous_root(
        self, tmp_path: Path, layout: str, reader: type[object]
    ) -> None:
        """Stating the layout skips the probe, which is the only way to read a root holding both."""
        _write_coco_root(tmp_path)
        _write_yolo_root(tmp_path)
        datamodule = _datamodule(tmp_path, layout=layout)
        datamodule.setup("fit")

        assert isinstance(datamodule._val, reader)

    def test_an_unknown_layout_name_is_rejected_at_construction(self, tmp_path: Path) -> None:
        """A misspelled layout is a configuration error, so it fails where it was written."""
        with pytest.raises(ValueError, match="unknown layout 'yollo'") as failure:
            _datamodule(tmp_path, layout="yollo")

        assert "'coco'" in str(failure.value)
        assert "'yolo'" in str(failure.value)

    def test_a_path_override_states_the_coco_layout(self, tmp_path: Path) -> None:
        """The four overrides name a COCO annotation file, so naming one answers the question.

        Without this the probe would run on roots it has no business judging — the offline
        fixture datamodules point ``data_root`` at a directory whose layout is irrelevant
        because every path is given — and an ambiguous root would fail on a reading nobody asked
        for.
        """
        _write_coco_root(tmp_path)
        _write_yolo_root(tmp_path)
        datamodule = _datamodule(
            tmp_path,
            train_images_dir=tmp_path / "train",
            train_ann_file=tmp_path / "annotations" / "instances_train.json",
            val_images_dir=tmp_path / "val",
            val_ann_file=tmp_path / "annotations" / "instances_val.json",
        )
        datamodule.setup("fit")

        assert datamodule.layout is DatasetLayout.COCO
        assert isinstance(datamodule._val, CocoDetectionDataset)


class TestYoloCapabilities:
    """What the YOLO path serves, and what it refuses rather than serving quietly."""

    def test_mask_targets_are_refused_on_an_inferred_yolo_root(self, tmp_path: Path) -> None:
        """A segmentation run on a format with no rings would supervise on nothing at all."""
        datamodule = _datamodule(_write_yolo_root(tmp_path), mask_targets=True)

        with pytest.raises(ValueError, match="mask_targets=True is unavailable on a YOLO root"):
            datamodule.setup("fit")

    def test_mask_targets_are_refused_at_construction_when_the_layout_is_stated(self, tmp_path: Path) -> None:
        """Stated layout plus stated segmentation is decidable with no filesystem, so it fails there."""
        with pytest.raises(ValueError, match="mask_targets=True is unavailable on a YOLO root"):
            _datamodule(tmp_path, layout="yolo", mask_targets=True)

    def test_coco_shaped_overrides_are_refused_with_an_explicit_yolo_layout(self, tmp_path: Path) -> None:
        """An override the YOLO reader would ignore is a statement it must not silently drop."""
        with pytest.raises(ValueError, match="cannot be combined with the train_images_dir"):
            _datamodule(tmp_path, layout="yolo", val_ann_file=tmp_path / "instances_val.json")

    def test_copy_paste_is_suppressed_on_a_yolo_root(self, tmp_path: Path) -> None:
        """The format carries no rings, so copy-paste would decode a source sample and paste nothing."""
        datamodule = _datamodule(_write_yolo_root(tmp_path), variant=_COPY_PASTE_VARIANT)
        datamodule.setup("fit")

        assert build_scale_policy(_COPY_PASTE_VARIANT)["copy_paste"] > 0.0
        assert datamodule._train._copy_paste_prob == 0.0

    def test_copy_paste_still_runs_on_a_coco_root(self, tmp_path: Path) -> None:
        """The suppression is the YOLO path's alone — the COCO pipeline's draw sequence is untouched."""
        datamodule = _datamodule(_write_coco_root(tmp_path), variant=_COPY_PASTE_VARIANT)
        datamodule.setup("fit")

        assert datamodule._train._copy_paste_prob == build_scale_policy(_COPY_PASTE_VARIANT)["copy_paste"]

    def test_rotated_targets_reach_the_oriented_reading_on_both_splits(self, tmp_path: Path) -> None:
        """``rotated_targets`` is the declaration the YOLO reader needs; the variant is never sniffed."""
        datamodule = _datamodule(_write_yolo_root(tmp_path, row=_ORIENTED_ROW), rotated_targets=True)
        datamodule.setup("fit")

        assert isinstance(datamodule._val, YoloDetectionDataset)
        for source in (datamodule._train._base, datamodule._val):
            _, targets = source[0]
            assert targets.rboxes.shape == (1, 5)
            assert targets.boxes.shape[0] == targets.rboxes.shape[0]


class TestYoloBatches:
    """The whole loader path over a YOLO root, which is what ``lucid-yolo fit`` drives."""

    def test_a_yolo_root_yields_a_full_training_batch(self, tmp_path: Path) -> None:
        """Both readers emit the same ``Targets``, so the collate and the transport are layout-blind."""
        datamodule = _datamodule(_write_yolo_root(tmp_path))
        datamodule.setup("fit")

        images, targets = unpack_batch(next(iter(datamodule.train_dataloader())))

        assert images.shape == (2, 3, _IMG_SIZE, _IMG_SIZE)
        assert images.dtype == torch.float32
        assert len(targets) == 2

    def test_the_yolo_val_loader_letterboxes_to_the_configured_side(self, tmp_path: Path) -> None:
        """Validation is letterbox-only on either layout, so the val samples are square and unaugmented."""
        datamodule = _datamodule(_write_yolo_root(tmp_path))
        datamodule.setup("fit")

        images, targets = unpack_batch(next(iter(datamodule.val_dataloader())))

        assert images.shape == (2, 3, _IMG_SIZE, _IMG_SIZE)
        assert all(target.boxes.shape[0] == 1 for target in targets)
