# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-014 COCO dataset, datamodule and ``check-data`` script.

All dataset/datamodule tests run offline against the WP-007 synthetic
detection/segmentation fixture (``detseg_fixture_dir``), never real COCO. They
cover: dataset length, sample dtype/range, ``xywh``->``xyxy`` conversion within
image bounds, box/polygon agreement, contiguous label remapping, the size-aware
scale policy, a train/val datamodule smoke (batch stacking + ragged target list),
seeded determinism, and the importable ``check_data`` layout validator (both the
matching and the mismatching path, without spawning a subprocess).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer

from lucid_yolo.data import Targets, boxes_from_polygons
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

_CHECK_DATA_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_data.py"


def _load_check_data() -> ModuleType:
    """Load ``scripts/check_data.py`` as a module (``scripts`` is not a package)."""
    spec = importlib.util.spec_from_file_location("check_data", _CHECK_DATA_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered before exec
    spec.loader.exec_module(module)
    return module


check_data = _load_check_data()

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
    """Build a dataset over the fixture's single ``train`` split."""
    split = fixture_dir / "train"
    return CocoDetectionDataset(split, split / "_annotations.coco.json", transforms=transforms)  # type: ignore[arg-type]


def _first_nonempty(dataset: CocoDetectionDataset) -> tuple[torch.Tensor, Targets]:
    """Return the first sample carrying at least one box."""
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
    """Build a mixed batch: an image with boxes+polygons+rbox, an empty one, a box-only image."""
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
    """Assert two target lists match tensor-for-tensor (dtype, shape and value)."""
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
    assert segment_count == 9
    assert segment_count < 12


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


def test_on_after_batch_transfer_leaves_unpacked_batch_untouched(detseg_fixture_dir: Path) -> None:
    """An already-unpacked (list) batch passes through the hook defensively unchanged."""
    datamodule = _datamodule(detseg_fixture_dir)
    targets = _ragged_targets()
    images, passed = datamodule.on_after_batch_transfer((torch.zeros(2, 3, 8, 8), targets), 0)
    assert images.shape == (2, 3, 8, 8)
    assert passed is targets


def _datamodule(
    fixture_dir: Path,
    variant: str = "m",
    seed: int = 0,
    num_workers: int | None = 0,
    pin_memory: bool | None = None,
    val_num_workers: int | None = None,
) -> DetectionDataModule:
    """Build a datamodule pointing both splits at the fixture's single split."""
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
    )


def test_datamodule_train_batch_smoke(detseg_fixture_dir: Path) -> None:
    """One training batch, through the transfer hook, stacks images and lists per-image targets."""
    datamodule = _datamodule(detseg_fixture_dir)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    assert isinstance(batch[1], PackedTargets)  # the loader emits the packed transport form
    assert batch[0].dtype == torch.uint8  # transport ships the images as uint8
    images, targets = datamodule.on_after_batch_transfer(batch, 0)
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
    images, targets = datamodule.on_after_batch_transfer(batch, 0)
    assert images.shape == (2, 3, _SMOKE_IMG_SIZE, _SMOKE_IMG_SIZE)
    assert images.dtype == torch.float32  # the hook restores float images
    assert torch.all((images >= 0.0) & (images <= 1.0))
    assert len(targets) == 2
    assert all(isinstance(target, Targets) for target in targets)


def test_on_after_batch_transfer_dequantizes_uint8_images(detseg_fixture_dir: Path) -> None:
    """The hook dequantizes a uint8 transport batch to float32 ``[0, 1]`` on the batch's device."""
    datamodule = _datamodule(detseg_fixture_dir)
    transport, packed = collate_detection([(torch.rand(3, 8, 8), target) for target in _ragged_targets()])
    images, _targets = datamodule.on_after_batch_transfer((transport, packed), 0)
    assert images.dtype == torch.float32
    assert images.device == transport.device
    assert torch.all((images >= 0.0) & (images <= 1.0))


class _SingleParamModule(LightningModule):
    """Minimal LightningModule carrying one parameter, so ``.dtype`` follows ``.half()``."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(1, 1)


def test_on_after_batch_transfer_matches_half_module_dtype(detseg_fixture_dir: Path) -> None:
    """The hook restores images at the attached module's dtype, so a half module gets half inputs."""
    module = _SingleParamModule().half()
    trainer = Trainer(accelerator="cpu", logger=False, enable_progress_bar=False)
    trainer.strategy.connect(module)
    datamodule = _datamodule(detseg_fixture_dir)
    datamodule.trainer = trainer
    transport, packed = collate_detection([(torch.rand(3, 8, 8), target) for target in _ragged_targets()])
    images, _targets = datamodule.on_after_batch_transfer((transport, packed), 0)
    assert images.dtype == torch.float16
    assert torch.all((images >= 0.0) & (images <= 1.0))


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
    """Worker loaders prefetch deeper, cap worker threads, and auto-resolve pin_memory."""
    datamodule = _datamodule(detseg_fixture_dir, num_workers=2)
    datamodule.setup("fit")
    loader = getattr(datamodule, loader_name)()
    assert loader.prefetch_factor == 2
    assert loader.worker_init_fn is not None
    assert loader.persistent_workers
    assert loader.pin_memory == torch.cuda.is_available()


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


def test_val_loader_workers_capped_below_train(detseg_fixture_dir: Path) -> None:
    """The val loader defaults to min(num_workers, 4) — never the full train worker pool (WP-073)."""
    datamodule = _datamodule(detseg_fixture_dir, num_workers=8)
    datamodule.setup("fit")
    assert datamodule.train_dataloader().num_workers == 8
    assert datamodule.val_dataloader().num_workers == 4


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
    """Create ``count`` empty image files and a matching annotation JSON."""
    images_dir.mkdir(parents=True, exist_ok=True)
    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (images_dir / f"img_{index:06d}.jpg").touch()
    annotation_file.write_text(json.dumps({"images": [{"id": i} for i in range(count)]}), encoding="utf-8")


def _write_fake_coco_root(root: Path, train: int, val: int) -> None:
    """Materialise a minimal fake COCO 2017 layout under ``root``."""
    annotations = root / "annotations"
    _write_fake_split(root / "train2017", annotations / "instances_train2017.json", train)
    _write_fake_split(root / "val2017", annotations / "instances_val2017.json", val)


def test_check_data_passes_on_matching_counts(tmp_path: Path) -> None:
    """A layout whose disk and annotation counts match the expected totals is OK."""
    _write_fake_coco_root(tmp_path, train=2, val=1)
    result = check_data.check_coco_root(tmp_path, expected_train=2, expected_val=1)
    assert result.ok


def test_check_data_fails_on_count_mismatch(tmp_path: Path) -> None:
    """Real COCO totals against a tiny layout fail, and the CLI exits 1."""
    _write_fake_coco_root(tmp_path, train=2, val=1)
    result = check_data.check_coco_root(tmp_path)
    assert not result.ok
    assert check_data.main(["--data-root", str(tmp_path)]) == 1


def test_check_data_fails_on_missing_root(tmp_path: Path) -> None:
    """A root without the expected directories is reported invalid."""
    assert not check_data.check_coco_root(tmp_path / "absent").ok
