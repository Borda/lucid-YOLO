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

import dataclasses
import json
import re
import warnings
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torchvision.io import write_png

from lucid_yolo.data import _image as image_data
from lucid_yolo.data.coco import CocoDetectionDataset, build_scale_policy
from lucid_yolo.data.layout import DATA_YAML_NAME, DatasetLayout
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.yolo import YoloDetectionDataset
from lucid_yolo.ptl.datamodule import (
    DetectionDataModule,
    PackedTargets,
    _PreparedTrainingSource,
    pack_targets,
    unpack_batch,
)

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
#: Pipeline seed the determinism runs are built at; arbitrary, but fixed so a failure is one.
_DETERMINISM_SEED = 20260904
#: Silences the image cache's size report on tests about something else; the report itself
#: is asserted by ``test_cached_fit_setup_reports_cache_size``.
_quiet_cache_report = pytest.mark.filterwarnings("ignore:image cache:UserWarning")


def _write_image(path: Path, seed: int | None = None) -> None:
    """Write one fixture-sized PNG at ``path``, creating its directory.

    Args:
        path: Destination file.
        seed: ``None`` (the default) writes an all-zero image, which is what every test that
            does not read pixels wants. An integer writes seeded random 8-bit content
            instead — lossless in PNG, so the pixels read back are the ones written — and two
            seeds give two different images.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     png = Path(tmp) / "sub" / "frame0.png"
        ...     _write_image(png, seed=3)
        ...     png.is_file()
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if seed is None:
        pixels = torch.zeros(3, _HEIGHT, _WIDTH, dtype=torch.uint8)
    else:
        generator = torch.Generator().manual_seed(seed)
        pixels = torch.randint(0, 256, (3, _HEIGHT, _WIDTH), generator=generator, dtype=torch.uint8)
    write_png(pixels, str(path))


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


def _write_coco_root(root: Path, *, distinct: bool = False) -> Path:
    """Materialise a two-split COCO root: ``<root>/<split>/`` beside ``annotations/``.

    Args:
        root: Directory to build inside.
        distinct: Give each image its own seeded random content instead of all zeros.

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
        for index, stem in enumerate(_IMAGE_STEMS, start=1):
            _write_image(root / split / f"{stem}.png", seed=index if distinct else None)
        annotation_file = root / "annotations" / f"instances_{split}.json"
        annotation_file.parent.mkdir(parents=True, exist_ok=True)
        annotation_file.write_text(json.dumps(_coco_payload()), encoding="utf-8")
    return root


def _write_yolo_root(root: Path, row: str = _DETECTION_ROW, *, distinct: bool = False) -> Path:
    """Materialise a two-split YOLO root in the shape a published export uses.

    The ``val`` split is written under ``valid/`` and named by the ``data.yaml`` as
    ``../valid/images`` — the published spelling (R32), which no convention table finds — so
    building that split exercises the dataset's own entry rather than the fallback.

    Args:
        root: Directory to build inside.
        row: The label row every image carries.
        distinct: Give each image its own seeded random content instead of all zeros.

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
        for index, stem in enumerate(_IMAGE_STEMS, start=1):
            _write_image(root / directory / "images" / f"{stem}.png", seed=index if distinct else None)
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


@pytest.fixture
def decode_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every image decode either reader performs, by the path it was asked to read."""
    calls: list[str] = []
    incumbent = image_data.read_image

    def counted_read_image(path: str, mode: object) -> Tensor:
        calls.append(path)
        return incumbent(path, mode)  # type: ignore[arg-type]

    monkeypatch.setattr(image_data, "read_image", counted_read_image)
    return calls


class TestCocoLabelSpaceAgreement:
    """Both COCO splits are checked against each other's label space before either is served (WP-172).

    Each split's contiguous labels are derived from its own file's ``categories`` list, so
    two files that disagree produce two different label spaces and nothing downstream can
    tell: the model trains against one numbering and is scored against another, reporting a
    plausible metric the whole way. The disagreement is a property of the pair, so ``setup``
    is the first and only place it can be seen.
    """

    def test_shared_category_set_builds_both_splits(self, tmp_path: Path) -> None:
        """Splits agreeing on their categories are served, with one label space between them.

        The fixture writes the same ``categories`` list into both files, which is what an
        ordinary export does; the check has to be silent here or it would reject every
        healthy dataset.
        """
        datamodule = _datamodule(_write_coco_root(tmp_path))

        datamodule.setup("fit")

        assert datamodule._train._base.category_id_to_label == datamodule._val.category_id_to_label

    def test_category_only_in_val_named_and_raises(self, tmp_path: Path) -> None:
        """A val-only category raises at ``setup``, naming the ids each side holds alone.

        Adding one category to val alone renumbers every label above it — with ids ``1, 7``
        the val reader maps ``7 -> 1`` while train has no label ``1`` at all — so this is
        the silent-relabelling case, not a cosmetic mismatch.
        """
        root = _write_coco_root(tmp_path)
        payload = _coco_payload()
        payload["categories"] = [{"id": 1, "name": "car"}, {"id": 7, "name": "bus"}]
        (root / "annotations" / "instances_val.json").write_text(json.dumps(payload), encoding="utf-8")
        datamodule = _datamodule(root)

        with pytest.raises(ValueError, match=r"only in val: \[7\]"):
            datamodule.setup("fit")

    def test_category_only_in_train_named_and_raises(self, tmp_path: Path) -> None:
        """A train-only category raises too: the check is symmetric, not a val-side filter.

        Guards the direction a one-sided ``issubset`` test would miss — a train file that
        declares a class the evaluation set never scores.
        """
        root = _write_coco_root(tmp_path)
        payload = _coco_payload()
        payload["categories"] = [{"id": 1, "name": "car"}, {"id": 4, "name": "van"}]
        (root / "annotations" / "instances_train.json").write_text(json.dumps(payload), encoding="utf-8")
        datamodule = _datamodule(root)

        with pytest.raises(ValueError, match=r"only in train: \[4\]"):
            datamodule.setup("fit")

    def test_mismatch_fails_before_train_cache_decodes_anything(self, tmp_path: Path, decode_calls: list[str]) -> None:
        """With ``cache_train_images`` on, a mismatched pair raises before a single image is decoded.

        The comparison needs both readers' category maps, which two JSON reads produce; the
        train cache needs the whole split decoded. Run in that order, a wrong annotation pair
        on a large split fails in seconds instead of after the full decode it would discard.
        """
        root = _write_coco_root(tmp_path)
        payload = _coco_payload()
        payload["categories"] = [{"id": 1, "name": "car"}, {"id": 7, "name": "bus"}]
        (root / "annotations" / "instances_val.json").write_text(json.dumps(payload), encoding="utf-8")
        datamodule = _datamodule(root, cache_train_images=True)

        with pytest.raises(ValueError, match=r"only in val: \[7\]"):
            datamodule.setup("fit")
        assert decode_calls == []


class TestLayoutDispatch:
    """Which reader a ``data_root`` reaches, under one flag and with no overrides."""

    def test_coco_root_reaches_coco_reader(self, tmp_path: Path) -> None:
        """The incumbent layout is unchanged: a COCO root still builds both splits from JSON."""
        datamodule = _datamodule(_write_coco_root(tmp_path))
        datamodule.setup("fit")

        assert datamodule.layout is DatasetLayout.COCO
        assert isinstance(datamodule._train._base, CocoDetectionDataset)
        assert isinstance(datamodule._val, CocoDetectionDataset)

    def test_yolo_root_reaches_yolo_reader(self, tmp_path: Path) -> None:
        """The same flag on a YOLO root builds both splits from its label trees — the WP's point."""
        datamodule = _datamodule(_write_yolo_root(tmp_path))
        datamodule.setup("fit")

        assert datamodule.layout is DatasetLayout.YOLO
        assert isinstance(datamodule._train._base, YoloDetectionDataset)
        assert isinstance(datamodule._val, YoloDetectionDataset)

    @pytest.mark.parametrize(
        ("layout", "write_root"),
        [("coco", _write_coco_root), ("yolo", _write_yolo_root)],
    )
    def test_image_cache_off_unless_asked_for(self, tmp_path: Path, layout: str, write_root: object) -> None:
        """A datamodule built without ``cache_train_images`` decodes on demand and holds no cache.

        The cache costs the uncompressed split in host RAM per process, so it is opt-in: every
        existing run that never names the flag must keep today's memory behaviour.
        """
        datamodule = _datamodule(write_root(tmp_path), layout=layout)  # type: ignore[operator]
        datamodule.setup("fit")

        assert datamodule._cache_train_images is False
        assert datamodule._train._base._image_cache is None
        assert datamodule._val._image_cache is None

    @_quiet_cache_report
    @pytest.mark.parametrize("stage", ["fit", None])
    @pytest.mark.parametrize(
        ("layout", "write_root"),
        [("coco", _write_coco_root), ("yolo", _write_yolo_root)],
    )
    def test_image_cache_train_only_for_both_readers(
        self, tmp_path: Path, layout: str, write_root: object, stage: str | None
    ) -> None:
        """The public switch caches train images without retaining one-use validation pixels.

        ``setup()`` with no stage is Lightning's all-stages call, and it builds the cache as the
        fit stage does.
        """
        root = write_root(tmp_path)  # type: ignore[operator]
        datamodule = _datamodule(root, layout=layout, cache_train_images=True)
        datamodule.setup(stage)

        assert datamodule._train is not None
        assert datamodule._train._base._image_cache is not None
        assert datamodule._val is not None
        assert datamodule._val._image_cache is None

    @pytest.mark.parametrize(
        ("layout", "write_root"),
        [("coco", _write_coco_root), ("yolo", _write_yolo_root)],
    )
    def test_cached_fit_setup_reports_cache_size(self, tmp_path: Path, layout: str, write_root: object) -> None:
        """The datamodule turns the readers' size report on, so a run's log names what the cache costs.

        Both readers stay quiet unless asked; the datamodule is the caller that owns a
        training run's memory, and a decode that silently ends in an out-of-memory kill
        is what the report exists to prevent.
        """
        datamodule = _datamodule(write_root(tmp_path), layout=layout, cache_train_images=True)  # type: ignore[operator]

        with pytest.warns(UserWarning, match=r"image cache: decod.* bytes\) of host RAM per training process"):
            datamodule.setup("fit")

    @pytest.mark.parametrize(
        ("layout", "write_root"),
        [("coco", _write_coco_root), ("yolo", _write_yolo_root)],
    )
    @pytest.mark.parametrize("stage", ["validate", "test", "predict"])
    def test_image_cache_skipped_outside_fit_stage(
        self, tmp_path: Path, layout: str, write_root: object, stage: str
    ) -> None:
        """A validate/test/predict call never decodes the train split it will not read.

        ``setup`` still builds ``_train`` for idempotency, but a stage that cannot reach
        it must not pay to cache it — the flag a training config sets would otherwise
        make a validate-only run decode the whole train split for nothing.
        """
        root = write_root(tmp_path)  # type: ignore[operator]
        datamodule = _datamodule(root, layout=layout, cache_train_images=True)
        datamodule.setup(stage)

        assert datamodule._train is not None
        assert datamodule._train._base._image_cache is None

    @_quiet_cache_report
    @pytest.mark.parametrize(
        ("layout", "write_root"),
        [("coco", _write_coco_root), ("yolo", _write_yolo_root)],
    )
    def test_repeat_fit_setup_reuses_decoded_train_reader(
        self, tmp_path: Path, layout: str, write_root: object, decode_calls: list[str]
    ) -> None:
        """A second fit-stage ``setup`` decodes nothing and serves the reader the first one decoded.

        A manual ``setup("fit")`` followed by ``trainer.fit``, or a second ``trainer.fit`` on
        one datamodule in a notebook, calls it twice. Building a second cached reader there
        held two copies of the split at the peak and paid the whole decode again for an
        identical result; a flat decode count across the two calls rules both out.
        """
        datamodule = _datamodule(write_root(tmp_path), layout=layout, cache_train_images=True)  # type: ignore[operator]
        datamodule.setup("fit")
        first_base = datamodule._train._base
        decoded_by_first_setup = len(decode_calls)

        datamodule.setup("fit")

        assert decoded_by_first_setup == len(_IMAGE_STEMS)
        assert len(decode_calls) == decoded_by_first_setup
        assert datamodule._train._base is first_base

    @_quiet_cache_report
    @pytest.mark.parametrize(
        ("layout", "write_root"),
        [("coco", _write_coco_root), ("yolo", _write_yolo_root)],
    )
    def test_repeat_fit_setup_rebuilds_train_pipeline(self, tmp_path: Path, layout: str, write_root: object) -> None:
        """Reusing the cached reader keeps ``setup``'s reset: the mosaic probability returns to its start.

        The close-mosaic schedule lowers the probability on the live pipeline; a repeat
        ``setup`` has always started a fresh pipeline at the initial value, and keeping
        the decoded reader must not keep the old pipeline along with it.
        """
        datamodule = _datamodule(write_root(tmp_path), layout=layout, cache_train_images=True)  # type: ignore[operator]
        datamodule.setup("fit")
        initial_mosaic_prob = datamodule.mosaic_prob
        datamodule.set_mosaic_prob(0.0)

        datamodule.setup("fit")

        assert datamodule.mosaic_prob == initial_mosaic_prob

    @_quiet_cache_report
    @pytest.mark.parametrize(
        ("layout", "write_root"),
        [("coco", _write_coco_root), ("yolo", _write_yolo_root)],
    )
    def test_validate_setup_after_fit_releases_cache(self, tmp_path: Path, layout: str, write_root: object) -> None:
        """A validate-stage ``setup`` after a fit one keeps no decoded train reader alive.

        A stage that never reads the train split has no use for its pixels, so the reuse
        that saves a repeat fit its decode must not turn into a cache retained for a run
        that only validates.
        """
        datamodule = _datamodule(write_root(tmp_path), layout=layout, cache_train_images=True)  # type: ignore[operator]
        datamodule.setup("fit")

        datamodule.setup("validate")

        assert datamodule._train._base._image_cache is None
        assert datamodule._cached_train_base is None

    def test_yolo_val_split_comes_from_datasets_own_entry(self, tmp_path: Path) -> None:
        """``val: ../valid/images`` resolves, so the dataset's spelling outranks the convention."""
        datamodule = _datamodule(_write_yolo_root(tmp_path))
        datamodule.setup("fit")

        assert isinstance(datamodule._val, YoloDetectionDataset)
        assert len(datamodule._val) == len(_IMAGE_STEMS)

    def test_root_satisfying_neither_names_both_conventions(self, tmp_path: Path) -> None:
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

    def test_root_satisfying_both_refuses_to_choose(self, tmp_path: Path) -> None:
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
    def test_explicit_layout_settles_ambiguous_root(self, tmp_path: Path, layout: str, reader: type[object]) -> None:
        """Stating the layout skips the probe, which is the only way to read a root holding both."""
        _write_coco_root(tmp_path)
        _write_yolo_root(tmp_path)
        datamodule = _datamodule(tmp_path, layout=layout)
        datamodule.setup("fit")

        assert isinstance(datamodule._val, reader)

    def test_unknown_layout_name_rejected_at_construction(self, tmp_path: Path) -> None:
        """A misspelled layout is a configuration error, so it fails where it was written."""
        with pytest.raises(ValueError, match="unknown layout 'yollo'") as failure:
            _datamodule(tmp_path, layout="yollo")

        assert "'coco'" in str(failure.value)
        assert "'yolo'" in str(failure.value)

    def test_path_override_states_coco_layout(self, tmp_path: Path) -> None:
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


def _open_coco_train(root: Path, *, cache_images: bool) -> CocoDetectionDataset:
    """Open the COCO fixture root's train split directly, as the datamodule does.

    Examples:
        >>> callable(_open_coco_train)
        True
    """
    return CocoDetectionDataset(
        root / "train", root / "annotations" / "instances_train.json", cache_images=cache_images
    )


def _open_yolo_train(root: Path, *, cache_images: bool) -> YoloDetectionDataset:
    """Open the YOLO fixture root's train split directly, as the datamodule does.

    Examples:
        >>> callable(_open_yolo_train)
        True
    """
    return YoloDetectionDataset.from_root(root, "train", cache_images=cache_images)


def _corrupt(path: Path) -> None:
    """Replace the image at ``path`` with bytes no PNG decoder accepts.

    Examples:
        >>> callable(_corrupt)
        True
    """
    path.write_bytes(b"not a png")


#: Root writer, train-split opener, the second image's path under the root, and the fault applied.
#: A YOLO image missing at construction is not a case: the reader lists the directory, so a file
#: that is absent then is simply not one of its images (see the removal-after-listing test below).
_UNREADABLE_CASES = [
    pytest.param(_write_coco_root, _open_coco_train, Path("train") / "frame1.png", Path.unlink, id="coco-missing"),
    pytest.param(_write_coco_root, _open_coco_train, Path("train") / "frame1.png", _corrupt, id="coco-corrupt"),
    pytest.param(
        _write_yolo_root, _open_yolo_train, Path("train") / "images" / "frame1.png", _corrupt, id="yolo-corrupt"
    ),
]


@pytest.mark.parametrize(("write_root", "open_train", "image", "fault"), _UNREADABLE_CASES)
class TestUnreadableTrainImage:
    """A train image that is missing or undecodable, with the cache on and off.

    The cache decodes the whole split at construction, so an image that would only have
    failed when its sample was drawn now fails up front; either way the error names the path.
    """

    def test_cache_fails_at_construction_naming_path(
        self, tmp_path: Path, write_root: object, open_train: object, image: Path, fault: object
    ) -> None:
        """With the cache on, building the reader raises ``OSError`` carrying the bad image's path."""
        root = write_root(tmp_path)  # type: ignore[operator]
        fault(root / image)  # type: ignore[operator]

        with pytest.raises(OSError, match=re.escape(str(root / image))):
            open_train(root, cache_images=True)  # type: ignore[operator]

    def test_without_cache_construction_succeeds_and_draw_fails(
        self, tmp_path: Path, write_root: object, open_train: object, image: Path, fault: object
    ) -> None:
        """With the cache off, the same root builds and only the draw of the bad image raises.

        Pins the timing change the cache brings: the failure moves from the draw to the build,
        with no change to which file is named.
        """
        root = write_root(tmp_path)  # type: ignore[operator]
        fault(root / image)  # type: ignore[operator]
        dataset = open_train(root, cache_images=False)  # type: ignore[operator]

        with pytest.raises(OSError, match=re.escape(str(root / image))):
            dataset[1]


def test_yolo_image_removed_after_listing_fails_cache_build_naming_path(tmp_path: Path) -> None:
    """A YOLO image that vanishes between the directory listing and the decode raises ``OSError``.

    The reader lists its images at construction, so this is the one way a listed YOLO image is
    missing; the cache build, run later, reports it by path and leaves the reader uncached.
    """
    root = _write_yolo_root(tmp_path)
    dataset = YoloDetectionDataset.from_root(root, "train")
    gone = root / "train" / "images" / "frame1.png"
    gone.unlink()

    with pytest.raises(OSError, match=re.escape(str(gone))):
        dataset.cache_images()
    assert dataset._image_cache is None


class TestYoloCapabilities:
    """What the YOLO path serves, and what it refuses rather than serving quietly."""

    def test_mask_targets_refused_on_inferred_yolo_root(self, tmp_path: Path) -> None:
        """A segmentation run on a format with no rings would supervise on nothing at all."""
        datamodule = _datamodule(_write_yolo_root(tmp_path), mask_targets=True)

        with pytest.raises(ValueError, match="mask_targets=True is unavailable on a YOLO root"):
            datamodule.setup("fit")

    def test_mask_targets_refused_at_construction_when_layout_stated(self, tmp_path: Path) -> None:
        """Stated layout plus stated segmentation is decidable with no filesystem, so it fails there."""
        with pytest.raises(ValueError, match="mask_targets=True is unavailable on a YOLO root"):
            _datamodule(tmp_path, layout="yolo", mask_targets=True)

    def test_coco_shaped_overrides_refused_with_explicit_yolo_layout(self, tmp_path: Path) -> None:
        """An override the YOLO reader would ignore is a statement it must not silently drop."""
        with pytest.raises(ValueError, match="cannot be combined with the train_images_dir"):
            _datamodule(tmp_path, layout="yolo", val_ann_file=tmp_path / "instances_val.json")

    def test_copy_paste_suppressed_on_yolo_root(self, tmp_path: Path) -> None:
        """The reader parses only detection and oriented rows, so there are no rings for copy-paste to paste.

        Left on, copy-paste would decode a whole source sample per draw and find no paste candidates.
        """
        datamodule = _datamodule(_write_yolo_root(tmp_path), variant=_COPY_PASTE_VARIANT)
        datamodule.setup("fit")

        assert build_scale_policy(_COPY_PASTE_VARIANT)["copy_paste"] > 0.0
        assert datamodule._train._copy_paste_prob == 0.0

    def test_copy_paste_runs_on_coco_root(self, tmp_path: Path) -> None:
        """The suppression is the YOLO path's alone — the COCO pipeline's draw sequence is untouched."""
        datamodule = _datamodule(_write_coco_root(tmp_path), variant=_COPY_PASTE_VARIANT)
        datamodule.setup("fit")

        assert datamodule._train._copy_paste_prob == build_scale_policy(_COPY_PASTE_VARIANT)["copy_paste"]

    def test_rotated_targets_reach_oriented_reading_on_both_splits(self, tmp_path: Path) -> None:
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

    def test_yolo_root_yields_full_training_batch(self, tmp_path: Path) -> None:
        """Both readers emit the same ``Targets``, so the collate and the transport are layout-blind."""
        datamodule = _datamodule(_write_yolo_root(tmp_path))
        datamodule.setup("fit")

        images, targets = unpack_batch(next(iter(datamodule.train_dataloader())))

        assert images.shape == (2, 3, _IMG_SIZE, _IMG_SIZE)
        assert images.dtype == torch.float32
        assert len(targets) == 2

    def test_yolo_val_loader_letterboxes_to_configured_side(self, tmp_path: Path) -> None:
        """Validation is letterbox-only on either layout, so the val samples are square and unaugmented."""
        datamodule = _datamodule(_write_yolo_root(tmp_path))
        datamodule.setup("fit")

        images, targets = unpack_batch(next(iter(datamodule.val_dataloader())))

        assert images.shape == (2, 3, _IMG_SIZE, _IMG_SIZE)
        assert all(target.boxes.shape[0] == 1 for target in targets)


def _fit_datamodule(data_root: Path, global_seed: int, **kwargs: object) -> DetectionDataModule:
    """Build the determinism runs' datamodule and run its fit-stage ``setup``.

    The global RNG is seeded to ``global_seed`` *before* construction on purpose: the claim
    under test is that the pipeline's own ``seed`` determines the epoch, so a run under a
    different ambient torch state must produce the same bytes.

    Args:
        data_root: A root to build both splits from.
        global_seed: Ambient ``torch.manual_seed`` value, deliberately varied between runs.
        **kwargs: Overrides forwarded to the datamodule beyond this helper's own fixed
            ``batch_size``/``variant``/``seed``/``mask_targets`` — ``cache_train_images``
            for the cache-parity check below.

    Returns:
        The datamodule, set up for ``"fit"``.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _fit_datamodule(_write_coco_root(Path(tmp)), 0).layout.value
        'coco'
    """
    torch.manual_seed(global_seed)
    datamodule = _datamodule(
        data_root,
        batch_size=1,
        variant=_COPY_PASTE_VARIANT,
        seed=_DETERMINISM_SEED,
        mask_targets=False,
        **kwargs,  # type: ignore[arg-type]
    )
    datamodule.setup("fit")
    return datamodule


def _epochs_of(datamodule: DetectionDataModule, epochs: int) -> list[list[tuple[Tensor, PackedTargets]]]:
    """Drive ``datamodule``'s training loader for ``epochs`` and return the transport batches.

    Args:
        datamodule: A datamodule already set up for ``"fit"``.
        epochs: How many passes over the loader to collect.

    Returns:
        One list of ``(uint8 images, PackedTargets)`` batches per epoch, in the transport
        form the collate emits — before ``on_after_batch_transfer`` restores floats.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     batches = _epochs_of(_fit_datamodule(_write_coco_root(Path(tmp)), 0), 1)
        ...     len(batches), batches[0][0][0].dtype
        (1, torch.uint8)
    """
    loader = datamodule.train_dataloader()
    return [list(loader) for _ in range(epochs)]


def _train_epochs(
    data_root: Path, global_seed: int, epochs: int = 2, **kwargs: object
) -> list[list[tuple[Tensor, PackedTargets]]]:
    """Drive a freshly-built datamodule's training loader for ``epochs`` and return the transport batches.

    Args:
        data_root: A root to build both splits from.
        global_seed: Ambient ``torch.manual_seed`` value, deliberately varied between runs.
        epochs: How many passes over the loader to collect.
        **kwargs: Overrides forwarded to :func:`_fit_datamodule`.

    Returns:
        One list of ``(uint8 images, PackedTargets)`` batches per epoch.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     epochs = _train_epochs(_write_coco_root(Path(tmp)), 0, epochs=1)
        ...     len(epochs), epochs[0][0][0].dtype
        (1, torch.uint8)
    """
    return _epochs_of(_fit_datamodule(data_root, global_seed, **kwargs), epochs)


def _assert_batches_identical(left: tuple[Tensor, PackedTargets], right: tuple[Tensor, PackedTargets]) -> None:
    """Assert two transport batches agree tensor for tensor.

    Args:
        left: One run's batch.
        right: The other run's batch at the same position.

    Raises:
        AssertionError: On the first image or target tensor that differs.

    Examples:
        >>> import torch
        >>> from lucid_yolo.ptl.datamodule import pack_targets
        >>> from lucid_yolo.data.targets import Targets
        >>> packed = pack_targets([Targets.empty()])
        >>> _assert_batches_identical((torch.zeros(1), packed), (torch.zeros(1), packed))
    """
    left_images, left_targets = left
    right_images, right_targets = right
    assert torch.equal(left_images, right_images), "the transported images differ between runs"
    for field in dataclasses.fields(left_targets):
        got = getattr(left_targets, field.name)
        want = getattr(right_targets, field.name)
        if got is None or want is None:
            assert got is want, f"{field.name} is present in one run and absent in the other"
            continue
        assert torch.equal(got, want), f"{field.name} differs between runs"


class TestTrainingDeterminism:
    """One seed and a deterministic access order reproduce an epoch byte for byte (M-39).

    The module docstring and :class:`_TrainPipeline` both claim this, and every test that
    resembled a check pinned one transform in isolation. The property that actually breaks
    is the composed one: six stateful transforms drawing from a single generator in a fixed
    order, where ``_maybe_mixup`` and ``_maybe_copy_paste`` consume a *variable* number of
    draws depending on their own outcome, so a reordered or added draw shifts everything
    after it. The ``x`` variant is used because its scale policy gives mixup and copy-paste
    non-zero probabilities — under the nano policy those branches never fire and the
    variable-draw path is never exercised.
    """

    def test_two_runs_at_one_seed_produce_identical_epochs(self, tmp_path: Path) -> None:
        """The whole training loader replays byte for byte across two independent builds.

        Both runs are driven for two epochs under deliberately different ambient torch
        seeds. The second epoch is the sharper half: the pipeline's generator has advanced
        by a full pass, so it agrees only if every draw of the first epoch was consumed in
        the same order and the same number.
        """
        data_root = _write_coco_root(tmp_path)

        first = _train_epochs(data_root, global_seed=0)
        second = _train_epochs(data_root, global_seed=999)

        assert [len(epoch) for epoch in first] == [len(epoch) for epoch in second]
        for left_epoch, right_epoch in zip(first, second, strict=True):
            for left, right in zip(left_epoch, right_epoch, strict=True):
                _assert_batches_identical(left, right)

    def test_different_seed_produces_different_epoch(self, tmp_path: Path) -> None:
        """The seed is what determines the epoch, not merely something the pipeline stores.

        Without this, a pipeline that ignored its seed entirely — returning the same
        augmented batch under every configuration — would satisfy the test above perfectly.
        """
        data_root = _write_coco_root(tmp_path)
        torch.manual_seed(0)
        datamodule = _datamodule(data_root, batch_size=1, variant=_COPY_PASTE_VARIANT, seed=_DETERMINISM_SEED)
        other = _datamodule(data_root, batch_size=1, variant=_COPY_PASTE_VARIANT, seed=_DETERMINISM_SEED + 1)
        datamodule.setup("fit")
        other.setup("fit")

        images = next(iter(datamodule.train_dataloader()))[0]
        other_images = next(iter(other.train_dataloader()))[0]

        assert not torch.equal(images, other_images)

    @_quiet_cache_report
    @pytest.mark.parametrize("write_root", [_write_coco_root, _write_yolo_root])
    def test_cache_train_images_not_change_augmented_epoch(self, tmp_path: Path, write_root: object) -> None:
        """The cache swaps where a draw's raw bytes come from, never what mosaic/affine/color/flip do with them.

        ``test_opt_in_cache_decodes_once_and_keeps_returned_images_independent`` (in
        ``test_coco.py``/``test_yolo_parse.py``) proves the reader's own decode count and
        isolation with no ``transforms`` attached. It never drives the actual training
        pipeline, so a cache that fed the same raw bytes through a *different* augmented
        draw would still pass it. Replaying one seeded epoch through the real
        ``_TrainPipeline`` with the cache on and off is the check that would catch that.

        Each image carries its own seeded random content: with all-zero images a cache that
        served the wrong file, or a wrong pixel, would still match, and the assertion that
        the cached arm holds a cache keeps the comparison from quietly running uncached twice.
        """
        data_root = write_root(tmp_path, distinct=True)  # type: ignore[operator]

        uncached = _train_epochs(data_root, global_seed=0, epochs=1, cache_train_images=False)
        cached_datamodule = _fit_datamodule(data_root, global_seed=0, cache_train_images=True)
        assert cached_datamodule._train._base._image_cache is not None
        cached = _epochs_of(cached_datamodule, 1)

        assert [len(epoch) for epoch in uncached] == [len(epoch) for epoch in cached]
        for left, right in zip(uncached[0], cached[0], strict=True):
            _assert_batches_identical(left, right)

    @_quiet_cache_report
    def test_cache_train_images_not_change_epoch_through_worker_process(self, tmp_path: Path) -> None:
        """With one loader worker, a cached epoch equals an uncached epoch drawn the same way.

        A worker reseeds the pipeline per worker and epoch by design, so a one-worker epoch
        is not byte-identical to a ``num_workers=0`` one and is never compared with it; both
        arms here run through a worker under the same ambient seed. What this adds over the
        in-process parity test is the cache surviving the hand-over to the worker (fork or
        spawn), with the pixels it serves there still the ones an uncached reader decodes.
        """
        data_root = _write_coco_root(tmp_path, distinct=True)

        uncached = _train_epochs(data_root, global_seed=0, epochs=1, num_workers=1, cache_train_images=False)
        cached = _train_epochs(data_root, global_seed=0, epochs=1, num_workers=1, cache_train_images=True)

        assert len(uncached[0]) == len(cached[0]) == len(_IMAGE_STEMS)
        for left, right in zip(uncached[0], cached[0], strict=True):
            _assert_batches_identical(left, right)


class TestPreparedCacheMemoryReport:
    """The prepared cache tells the user what it will cost in host RAM, before and after the build.

    Lives outside ``TestPreparedTrainingCache``: that class silences these very warnings so its
    parity tests stay quiet, and a warning asserted under a blanket ignore is not asserted.
    """

    def test_coco_estimate_states_twelve_bytes_per_source_pixel(self, tmp_path: Path) -> None:
        """A COCO root's pre-build estimate is 12 bytes (3 float32 channels) per annotated pixel.

        The annotation sizes are known before a single image is decoded, so the user sees the
        number while they can still abort a run that would not fit.
        """
        expected_bytes = 12 * len(_IMAGE_STEMS) * _WIDTH * _HEIGHT
        module = _datamodule(_write_coco_root(tmp_path), cache_prepared_train_images=True)

        with pytest.warns(
            UserWarning,
            match=re.escape(
                f"prepared image cache: about 0.0 GB ({expected_bytes:,} bytes) from the annotation sizes;"
            ),
        ):
            module.setup("fit")

    def test_yolo_estimate_omits_annotation_size_claim(self, tmp_path: Path) -> None:
        """A YOLO root carries no image sizes in its labels, so the estimate states none.

        Inventing a figure there would be a number the reader cannot trust; the message still
        warns of the four-times cost.
        """
        module = _datamodule(_write_yolo_root(tmp_path), cache_prepared_train_images=True)

        with pytest.warns(UserWarning, match=r"prepared image cache: retaining float32 source images") as record:
            module.setup("fit")

        assert not any("from the annotation sizes" in str(warning.message) for warning in record)

    @pytest.mark.parametrize("layout", ["coco", "yolo"])
    def test_report_after_build_states_retained_bytes_and_record_count(self, tmp_path: Path, layout: str) -> None:
        """The post-build report names the bytes the float cache holds and how many records hold them."""
        write_root = _write_coco_root if layout == "coco" else _write_yolo_root
        module = _datamodule(write_root(tmp_path), cache_prepared_train_images=True)

        with pytest.warns(UserWarning, match=r"prepared image cache: holding") as record:
            module.setup("fit")

        source = module._prepared_train
        assert source is not None
        retained = sum(image.nbytes for image, _ in source)
        expected = (
            f"prepared image cache: holding {retained:,} bytes of float32 pixels for {len(source)} training records"
        )
        assert expected in {str(warning.message) for warning in record}

    def test_both_cache_flags_warn_at_construction_naming_precedence(self, tmp_path: Path) -> None:
        """Setting both train-cache flags warns once, at the caller's line, that the prepared cache wins.

        The flags are alternatives; a config setting both silently builds only one of them, and
        the user who expected the 8-bit cache to be in force should hear which one is.
        """
        root = _write_coco_root(tmp_path)

        with pytest.warns(UserWarning, match=r"both set; the prepared float32 cache takes precedence") as record:
            _datamodule(root, cache_train_images=True, cache_prepared_train_images=True)

        assert [warning.filename for warning in record] == [__file__]

    @pytest.mark.parametrize("flag", ["cache_train_images", "cache_prepared_train_images"])
    def test_single_cache_flag_does_not_warn_about_both(self, tmp_path: Path, flag: str) -> None:
        """Either cache flag alone is a legitimate configuration and raises no precedence warning."""
        root = _write_coco_root(tmp_path)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _datamodule(root, **{flag: True})

        assert not any("both set" in str(warning.message) for warning in record)


@pytest.mark.filterwarnings("ignore:prepared image cache:UserWarning")
class TestPreparedTrainingCache:
    """Moving the affine subtraction to immutable sources preserves every output byte."""

    @pytest.mark.parametrize("layout", ["coco", "yolo"])
    @pytest.mark.parametrize("mosaic_prob", [0.0, 1.0])
    def test_float_pixels_targets_and_draws_match(self, tmp_path: Path, layout: str, mosaic_prob: float) -> None:
        """Compare before transport quantization, with mixed-source blending enabled."""
        write_root = _write_coco_root if layout == "coco" else _write_yolo_root
        root = write_root(tmp_path, distinct=True)
        baseline = _fit_datamodule(root, 0, cache_train_images=True)
        prepared = _fit_datamodule(root, 0, cache_prepared_train_images=True)
        baseline.set_mosaic_prob(mosaic_prob)
        prepared.set_mosaic_prob(mosaic_prob)
        assert baseline._train is not None and prepared._train is not None
        source = prepared._prepared_train
        assert source is not None
        saved = [image.clone() for image, _ in source]
        for index in range(32):
            left, left_targets = baseline._train[index % len(baseline._train)]
            right, right_targets = prepared._train[index % len(prepared._train)]
            assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
            _assert_batches_identical((left, pack_targets([left_targets])), (right, pack_targets([right_targets])))
        assert torch.equal(baseline._train._generator.get_state(), prepared._train._generator.get_state())
        for (image, _), original in zip(source, saved, strict=True):
            assert torch.equal(image.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8))

    @pytest.mark.parametrize("layout", ["coco", "yolo"])
    @pytest.mark.parametrize("stage", ["validate", "test", "predict"])
    def test_reuses_prepared_storage_and_releases_it_outside_fit(
        self, tmp_path: Path, layout: str, stage: str, decode_calls: list[str]
    ) -> None:
        """Repeated setup keeps one float cache; enabling both flags keeps no byte copy."""
        write_root = _write_coco_root if layout == "coco" else _write_yolo_root
        module = _datamodule(write_root(tmp_path), cache_train_images=True, cache_prepared_train_images=True)
        module.setup("fit")
        first = module._prepared_train
        assert first is not None
        assert first.base._image_cache is None
        calls = len(decode_calls)
        module.set_mosaic_prob(0.0)
        module.setup("fit")
        assert module._prepared_train is first
        assert len(decode_calls) == calls == len(_IMAGE_STEMS)
        assert module.mosaic_prob > 0.0
        module.setup(stage)
        assert module._prepared_train is None
        assert module._cached_train_base is None
        assert module._train is not None and not isinstance(module._train._base, _PreparedTrainingSource)

    @pytest.mark.parametrize("layout", ["coco", "yolo"])
    def test_fit_after_release_rebuilds_prepared_cache_by_decoding_again(
        self, tmp_path: Path, layout: str, decode_calls: list[str]
    ) -> None:
        """A fit setup following a release pays the decode again: the cache was dropped, not kept hidden."""
        write_root = _write_coco_root if layout == "coco" else _write_yolo_root
        module = _datamodule(write_root(tmp_path), cache_prepared_train_images=True)
        module.setup("fit")
        module.setup("validate")
        decoded_before_rebuild = len(decode_calls)

        module.setup("fit")

        assert decoded_before_rebuild == len(_IMAGE_STEMS)
        assert len(decode_calls) == 2 * len(_IMAGE_STEMS)
        assert module._prepared_train is not None

    @pytest.mark.parametrize("layout", ["coco", "yolo"])
    def test_stage_none_builds_prepared_cache(self, tmp_path: Path, layout: str) -> None:
        """``setup(None)`` covers every stage, training included, so it builds the cache as ``"fit"`` does."""
        write_root = _write_coco_root if layout == "coco" else _write_yolo_root
        module = _datamodule(write_root(tmp_path), cache_prepared_train_images=True)

        module.setup(None)

        assert module._prepared_train is not None
        assert module._train is not None and module._train._base is module._prepared_train

    def test_rejects_reader_with_transforms(self, tmp_path: Path) -> None:
        """A reader that already transforms its output would be cached transformed, then warped a second time."""
        root = _write_coco_root(tmp_path)
        reader = CocoDetectionDataset(
            root / "train", root / "annotations" / "instances_train.json", transforms=Letterbox(16)
        )

        with pytest.raises(ValueError, match="without transforms"):
            _PreparedTrainingSource(reader)

    def test_failed_prepared_fit_after_validate_setup_leaves_no_train_pipeline(self, tmp_path: Path) -> None:
        """A failed prepared build after a validate setup must not leave the validate pipeline serving training.

        ``setup("validate")`` builds a plain train pipeline for idempotency. When the next fit setup
        fails mid-build, ``train_dataloader`` has to refuse rather than quietly train on that
        uncached, wrong-stage pipeline; the validation split stays the one already built.
        """
        root = _write_coco_root(tmp_path, distinct=True)
        module = _datamodule(root, cache_prepared_train_images=True)
        module.setup("validate")
        validation_before = module._val
        (root / "train" / "frame1.png").unlink()

        with pytest.raises(OSError, match="failed to read image"):
            module.setup("fit")

        assert module._train is None
        assert module._val is validation_before
        with pytest.raises(RuntimeError, match="setup"):
            module.train_dataloader()

    @pytest.mark.parametrize(
        ("context", "persistent_workers", "epochs"),
        [
            pytest.param("spawn", False, 1, id="spawn-recycled-one-epoch"),
            pytest.param(
                "fork",
                True,
                3,
                id="fork-persistent-three-epochs",
                marks=pytest.mark.skipif(
                    "fork" not in torch.multiprocessing.get_all_start_methods(),
                    reason="fork start method unavailable on this platform",
                ),
            ),
        ],
    )
    def test_worker_reads_prepared_pixels(
        self, tmp_path: Path, context: str, persistent_workers: bool, epochs: int
    ) -> None:
        """The source is ordinary picklable state under a spawned worker and shared copy-on-write under a forked one.

        Persistent workers keep one pipeline copy across epochs while recycled ones rebuild it, so
        the multi-epoch fork case also pins that a worker holding the float cache stays byte-equal
        to the decoded-cache run over repeated passes.
        """
        root = _write_coco_root(tmp_path, distinct=True)
        expected = _train_epochs(root, 0, epochs=epochs, num_workers=1, persistent_workers=persistent_workers)
        module = _fit_datamodule(
            root, 0, num_workers=1, persistent_workers=persistent_workers, cache_prepared_train_images=True
        )
        loader = module.train_dataloader()
        loader.multiprocessing_context = context
        strategy = torch.multiprocessing.get_sharing_strategy()
        try:
            # file_descriptor spawn also fails in the baseline on torch 2.14.
            # Exercise prepared-source pickling independently with file_system.
            if context == "spawn":
                torch.multiprocessing.set_sharing_strategy("file_system")
            actual = [list(loader) for _ in range(epochs)]
        finally:
            torch.multiprocessing.set_sharing_strategy(strategy)
        for expected_epoch, actual_epoch in zip(expected, actual, strict=True):
            for left, right in zip(expected_epoch, actual_epoch, strict=True):
                _assert_batches_identical(left, right)

    def test_failed_preparation_installs_no_partial_cache(self, tmp_path: Path) -> None:
        """A failed decode leaves setup retryable after the missing file is restored."""
        root = _write_coco_root(tmp_path, distinct=True)
        missing = root / "train" / "frame1.png"
        missing.unlink()
        module = _datamodule(root, cache_prepared_train_images=True)
        with pytest.raises(OSError, match="failed to read image"):
            module.setup("fit")
        assert module._prepared_train is None
        assert module._cached_train_base is None
        _write_image(missing, seed=1)
        module.setup("fit")
        assert module._prepared_train is not None and len(module._prepared_train) == len(_IMAGE_STEMS)

    def test_draws_reuse_sources_and_do_not_center_mosaic_again(self, tmp_path: Path, decode_calls: list[str]) -> None:
        """A silent return to per-draw reads or a fill-bearing warp must fail this gate."""
        module = _datamodule(_write_coco_root(tmp_path), cache_prepared_train_images=True)
        module.setup("fit")
        assert module._train is not None
        decoded = len(decode_calls)
        for _ in range(8):
            module._train[0]
        assert len(decode_calls) == decoded == len(_IMAGE_STEMS)
        affine = module._train._affine
        assert affine.last_params is not None
        assert affine._segment(affine.last_params).fill is None

    @pytest.mark.parametrize("modality", ["masks", "keypoints", "oriented"])
    def test_preserves_modality_targets_over_repeated_draws(self, tmp_path: Path, modality: str) -> None:
        """Cached targets retain mask geometry, rotated boxes and schema-defined point flips."""
        if modality == "oriented":
            root = _write_yolo_root(tmp_path, row=_ORIENTED_ROW, distinct=True)
        else:
            root = _write_coco_root(tmp_path, distinct=True)
        if modality == "keypoints":
            for split in ("train", "val"):
                path = root / "annotations" / f"instances_{split}.json"
                payload = json.loads(path.read_text())
                payload["categories"][0]["keypoints"] = ["base_left", "base_right"]
                for annotation in payload["annotations"]:
                    annotation["keypoints"] = [20.0, 16.0, 2, 40.0, 20.0, 2]
                    annotation["num_keypoints"] = 2
                path.write_text(json.dumps(payload))
        flags = dict(
            mask_targets=modality == "masks",
            keypoint_targets=modality == "keypoints",
            rotated_targets=modality == "oriented",
            variant="x",
            seed=_DETERMINISM_SEED,
        )
        baseline = _datamodule(root, cache_train_images=True, **flags)
        cached = _datamodule(root, cache_prepared_train_images=True, **flags)
        baseline.setup("fit")
        cached.setup("fit")
        assert baseline._train is not None and cached._train is not None
        left_collate = baseline.train_dataloader().collate_fn
        right_collate = cached.train_dataloader().collate_fn
        for index in range(32):
            left = baseline._train[index % len(baseline._train)]
            right = cached._train[index % len(cached._train)]
            assert torch.equal(left[0].contiguous().view(torch.uint8), right[0].contiguous().view(torch.uint8))
            left_batch, right_batch = left_collate([left]), right_collate([right])
            _assert_batches_identical(left_batch, right_batch)
            for field in dataclasses.fields(left_batch[1]):
                a, b = getattr(left_batch[1], field.name), getattr(right_batch[1], field.name)
                if a is not None and b is not None:
                    assert a.shape == b.shape
                    assert torch.equal(
                        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)
                    )
