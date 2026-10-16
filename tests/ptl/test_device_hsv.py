# SPDX-License-Identifier: Apache-2.0
"""Deferred HSV retains the host pipeline's pixels, targets and draw sequence."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import CUDAAccelerator
from torchvision.io import write_png

from lucid_yolo.data import augment
from lucid_yolo.data.augment import FlipParams, HorizontalFlip, HSVJitter, HSVParams, _jitter_hsv_batch
from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl.datamodule import (
    DetectionDataModule,
    _DeferredHSVSample,
    _host_sample,
    collate_detection,
    unpack_batch,
    unpack_masks,
)


def _root(path: Path) -> Path:
    """Write two different images and one shared COCO annotation file.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = _root(Path(tmp))
        ...     len(list((root / "images").glob("*.png")))
        2
    """
    (path / "images").mkdir(parents=True)
    images = []
    annotations = []
    for index in range(2):
        name = f"{index}.png"
        generator = torch.Generator().manual_seed(index)
        write_png(
            torch.randint(0, 256, (3, 48, 64), dtype=torch.uint8, generator=generator), str(path / "images" / name)
        )
        images.append({"id": index, "file_name": name, "width": 64, "height": 48})
        annotations.append({"id": index, "image_id": index, "category_id": 1, "bbox": [5, 6, 24, 20], "iscrowd": 0})
    payload = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "object"}]}
    (path / "annotations.json").write_text(json.dumps(payload))
    return path


def _module(root: Path, *, device_hsv: bool = False, workers: int = 0) -> DetectionDataModule:
    """Build the public datamodule against the fixture's explicit layout.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     module = _module(_root(Path(tmp)))
        ...     module.setup("fit")
        ...     len(module.train_dataloader())
        1
    """
    return DetectionDataModule(
        root,
        batch_size=2,
        num_workers=workers,
        variant="n",
        img_size=32,
        train_images_dir=root / "images",
        val_images_dir=root / "images",
        train_ann_file=root / "annotations.json",
        val_ann_file=root / "annotations.json",
        device_hsv=device_hsv,
    )


def _assert_targets(left: list[Targets], right: list[Targets]) -> None:
    """Compare every modality, including an empty image and ragged polygon rings.

    Examples:
        >>> _assert_targets([Targets.empty()], [Targets.empty()])
    """
    assert len(left) == len(right) > 0
    for first, second in zip(left, right, strict=True):
        for field in dataclasses.fields(first):
            a, b = getattr(first, field.name), getattr(second, field.name)
            if isinstance(a, list):
                assert len(a) == len(b)
                assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
            else:
                assert torch.equal(a, b), field.name


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask_targets", [False, True])
def test_transport_matches_host_colour_and_flip(dtype: torch.dtype, mask_targets: bool) -> None:
    """The colour/flip permutation preserves codes, geometry and rasterized masks."""
    images = torch.rand(2, 3, 16, 24)
    ring = torch.tensor([[2.0, 3.0], [10.0, 3.0], [10.0, 11.0], [2.0, 11.0]])
    targets = [
        Targets(boxes=torch.tensor([[2.0, 3.0, 10.0, 11.0]]), labels=torch.tensor([1]), polygons=[ring]),
        Targets.empty(),
    ]
    gains = [HSVParams(0.012, 0.31, -0.27), HSVParams(-0.015, -0.7, 0.4)]
    host, deferred = [], []
    flip = HorizontalFlip()
    for index, (image, target, params) in enumerate(zip(images, targets, gains, strict=True)):
        coloured, _ = HSVJitter().apply(image, target, params)
        flipped, geometry = flip.apply(coloured, target, FlipParams(flipped=index == 0))
        raw, raw_geometry = flip.apply(image, target, FlipParams(flipped=index == 0))
        host.append((flipped, geometry))
        deferred.append(_DeferredHSVSample(raw, raw_geometry, params))
    reference = collate_detection(host, mask_targets=mask_targets)
    transport = collate_detection(deferred, mask_targets=mask_targets)
    assert reference[0].dtype == torch.uint8
    assert transport[0].dtype == torch.float32
    assert transport[1].hsv_gains.dtype == torch.float64
    assert torch.equal(transport[1].hsv_gains, torch.tensor([x.as_tuple() for x in gains], dtype=torch.float64))
    expected_images, expected_targets = unpack_batch(reference, dtype)
    actual_images, actual_targets = unpack_batch(transport, dtype)
    assert torch.equal(actual_images.view(torch.uint8), expected_images.view(torch.uint8))
    _assert_targets(actual_targets, expected_targets)
    if mask_targets:
        expected_masks, actual_masks = unpack_masks(reference[1]), unpack_masks(transport[1])
        assert len(expected_masks) == len(actual_masks) == 2
        assert all(torch.equal(a, b) for a, b in zip(expected_masks, actual_masks, strict=True))


@pytest.mark.parametrize("workers", [0, 2])
def test_pipeline_preserves_rng_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int) -> None:
    """The public loader transports stated gains without running HSV in a worker."""
    root = _root(tmp_path)
    baseline, deferred = _module(root, workers=workers), _module(root, device_hsv=True, workers=workers)
    deferred.trainer = SimpleNamespace(accelerator=CUDAAccelerator(), precision="32-true")
    baseline.setup("fit")
    deferred.setup("fit")
    expected = list(baseline.train_dataloader())

    def refuse_colour(*args: object, **kwargs: object) -> None:
        """Fail if the deferred worker tries to execute the host colour transform.

        Examples:
            >>> callable(refuse_colour)
            True
        """
        raise AssertionError("HSV ran in the data worker")

    monkeypatch.setattr(HSVJitter, "apply", refuse_colour)
    actual = list(deferred.train_dataloader())
    assert len(actual) == len(expected) == 1
    a_images, a_targets = unpack_batch(actual[0])
    b_images, b_targets = unpack_batch(expected[0])
    assert torch.equal(a_images.view(torch.uint8), b_images.view(torch.uint8))
    _assert_targets(a_targets, b_targets)
    assert actual[0][1].hsv_gains is not None
    assert next(iter(deferred.val_dataloader()))[1].hsv_gains is None
    if workers == 0:
        assert torch.equal(baseline._train._generator.get_state(), deferred._train._generator.get_state())
        assert baseline._train._hsv.last_gains == deferred._train._hsv.last_gains
    deferred.setup("validate")
    assert deferred._train._defer_hsv is False


def test_cpu_fallback_warns_once_and_preserves_default(tmp_path: Path) -> None:
    """A CPU Trainer retains the original public transport when the flag is requested."""
    module = _module(_root(tmp_path), device_hsv=True)
    module.trainer = Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    with pytest.warns(UserWarning, match="device_hsv=True is ignored off CUDA"):
        module.setup("fit")
    module.setup("fit")
    images, packed = next(iter(module.train_dataloader()))
    assert images.dtype == torch.uint8
    assert packed.hsv_gains is None


@pytest.mark.parametrize("precision", ["16-true", "bf16-true", "64-true", "transformer-engine"])
def test_casting_precision_keeps_the_original_transport(tmp_path: Path, precision: str) -> None:
    """Plugins that cast before the data hook retain worker-side HSV and its gain precision."""
    module = _module(_root(tmp_path), device_hsv=True)
    module.trainer = SimpleNamespace(accelerator=CUDAAccelerator(), precision=precision)
    with pytest.warns(UserWarning, match="ignored for precision="):
        module.setup("fit")
    images, packed = next(iter(module.train_dataloader()))
    assert images.dtype == torch.uint8
    assert packed.hsv_gains is None


def test_mixed_transport_is_refused() -> None:
    """Mixed producer modes fail before gains can be paired with the wrong image."""
    image, target = torch.zeros(3, 2, 2), Targets.empty()
    deferred = _DeferredHSVSample(image, target, HSVParams(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="cannot collate deferred and already-jittered"):
        collate_detection([(image, target), deferred])


def test_host_diagnostics_resolve_colour_without_quantization() -> None:
    """Dataset-level diagnostics retain the original unquantized image convention."""
    image, target = torch.rand(3, 9, 17), Targets.empty()
    params = HSVParams(0.012, 0.31, -0.27)
    expected, _ = HSVJitter().apply(image, target, params)
    actual, geometry = _host_sample(_DeferredHSVSample(image, target, params))
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert geometry is target
    ordinary = (image, target)
    assert _host_sample(ordinary) is ordinary


def test_consumer_keeps_the_hue_divisor_on_the_image_device() -> None:
    """Offline CI detects a scalar-divisor reversion before a CUDA runner is needed."""
    image = torch.rand(3, 5, 7)
    sample = _DeferredHSVSample(image, Targets.empty(), HSVParams(0.012, 0.31, -0.27))
    with patch("lucid_yolo.data.augment._rgb_to_hsv", wraps=augment._rgb_to_hsv) as converter:
        unpack_batch(collate_detection([sample]))
    assert converter.call_count == 1
    divisor = converter.call_args.args[1]
    assert isinstance(divisor, torch.Tensor), "CUDA parity requires a device-tensor hue divisor"
    assert divisor.device == image.device
    assert divisor.dtype == image.dtype
    assert divisor.shape == ()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression")
@pytest.mark.parametrize("batch_size", [1, 3])
def test_cuda_batch_matches_cpu_transform(batch_size: int) -> None:
    """CUDA runs the CPU division convention before rounding to transport codes."""
    images = torch.rand(batch_size, 3, 61, 97)
    params = [HSVParams(0.012, 0.31, -0.27), HSVParams(-0.015, -0.7, 0.4), HSVParams(0.0, 0.0, 0.0)][:batch_size]
    gains = torch.tensor([p.as_tuple() for p in params], dtype=torch.float64)
    expected = torch.stack([HSVJitter().apply(im, Targets.empty(), p)[0] for im, p in zip(images, params, strict=True)])
    actual = _jitter_hsv_batch(images.cuda(), gains.cuda()).cpu()
    assert torch.equal((expected * 255).round().to(torch.uint8), (actual * 255).round().to(torch.uint8))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression")
def test_cuda_hue_division_preserves_captured_coco_pixel() -> None:
    """The real-data pixel that exposed reciprocal rounding keeps its three codes."""
    image = (
        torch.tensor([1047526623, 1047648800, 1048660522], dtype=torch.int32).view(torch.float32).reshape(1, 3, 1, 1)
    )
    gains = torch.tensor([[-0.011664924521546874, 0.43333633457326814, 0.23118815543879265]], dtype=torch.float64)
    actual = _jitter_hsv_batch(image.cuda(), gains.cuda()).cpu()
    assert (actual * 255).round().to(torch.uint8).flatten().tolist() == [71, 73, 79]
