# SPDX-License-Identifier: Apache-2.0
"""Pin the CUDA restore's pixel boundary, precision and native fallback."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from lucid_yolo.data import Targets, _hsv_cuda
from lucid_yolo.data.augment import _jitter_hsv_batch
from lucid_yolo.ptl import datamodule

_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not _hsv_cuda._AVAILABLE
    or torch.version.hip is not None
    or not _hsv_cuda._host_compiler_available()
    or torch.cuda.get_device_capability()[0] < 8,
    reason="the NVIDIA CUDA/Triton backend is unavailable",
)


def _packed(gains: torch.Tensor) -> datamodule.PackedTargets:
    """Attach stated gains to empty targets without inventing a geometry fixture.

    Examples:
        >>> gains = torch.zeros(2, 3, dtype=torch.float64)
        >>> _packed(gains).boxes_per_image.tolist()
        [0, 0]
    """
    return replace(datamodule.pack_targets([Targets.empty() for _ in range(gains.shape[0])]), hsv_gains=gains)


def test_cpu_restore_keeps_native_path() -> None:
    """Requesting the layout fast path on CPU retains the existing restore."""
    images = torch.full((2, 3, 3, 5), 0.5)
    gains = torch.zeros(2, 3, dtype=torch.float64)
    packed = _packed(gains)
    actual = datamodule._restore_images(images, packed, torch.float32, channels_last=True)
    expected = datamodule._restore_images(images, packed, torch.float32)
    assert _hsv_cuda._restore_hsv_cuda(images, gains) is None
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.gpu
@_CUDA
def test_literal_pixels_and_single_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primaries and the halfway grey pin pixels while a native-call trap pins fusion."""
    images = torch.tensor(
        [[[[0, 1, 0, 0, 1, 0.5]], [[0, 0, 1, 0, 1, 0.5]], [[0, 0, 0, 1, 1, 0.5]]]], device="cuda", dtype=torch.float32
    )
    gains = torch.zeros(1, 3, device="cuda", dtype=torch.float64)
    expected = images.clone()
    expected[:, :, :, -1] = 128 / 255

    def refuse_native(*_args: object) -> torch.Tensor:
        raise AssertionError("the fused entrance must not materialize the native HSV intermediates")

    monkeypatch.setattr(datamodule, "_jitter_hsv_batch", refuse_native)
    actual = datamodule._restore_images(images, _packed(gains), torch.float32, channels_last=True)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert actual.data_ptr() != images.data_ptr()


@pytest.mark.gpu
@_CUDA
def test_all_transport_codes_keep_cuda_scalar_division_bits() -> None:
    """Every grey code pins reciprocal multiplication at the existing dequantization boundary."""
    codes = torch.arange(256, device="cuda", dtype=torch.uint8).reshape(1, 1, 1, 256).expand(1, 3, 1, 256)
    images = codes.to(torch.float32).div(255).contiguous()
    gains = torch.zeros(1, 3, device="cuda", dtype=torch.float64)
    actual = _hsv_cuda._restore_hsv_cuda(images, gains)
    assert actual is not None
    assert torch.equal(actual.view(torch.int32), images.view(torch.int32))


@pytest.mark.gpu
@_CUDA
def test_gain_addition_precedes_float32_cast() -> None:
    """A gain beyond the float32 tie must move an even half-code to the next code.

    ``1 + (2**-24 + 2**-50)`` rounds upward when formed in double; casting
    the gain first loses the final term and rounds the sum back to one. A grey
    at code 126.5 therefore rounds to 127, not the even 126 of the wrong order.
    """
    images = torch.full((1, 3, 1, 1), 126.5 / 255, device="cuda")
    gains = torch.tensor([[0, 0, 2**-24 + 2**-50]], device="cuda", dtype=torch.float64)
    actual = _hsv_cuda._restore_hsv_cuda(images, gains)
    assert actual is not None
    assert actual.mul(255).round().to(torch.uint8).flatten().tolist() == [127, 127, 127]


@pytest.mark.gpu
@_CUDA
@pytest.mark.parametrize("shape", [(1, 3, 1, 1), (2, 3, 13, 17), (3, 3, 7, 31)])
def test_stated_gains_and_tail_blocks_match_native_bits(shape: tuple[int, ...]) -> None:
    """Distinct per-image draws and partial blocks preserve the native post-quantization bits."""
    generator = torch.Generator(device="cuda").manual_seed(31)
    images = torch.rand(shape, device="cuda", generator=generator)
    gains = torch.empty((shape[0], 3), device="cuda", dtype=torch.float64).uniform_(-0.5, 0.5, generator=generator)
    expected = datamodule._dequantize_images(datamodule._quantize_images(_jitter_hsv_batch(images, gains)))
    actual = _hsv_cuda._restore_hsv_cuda(images, gains)
    assert actual is not None
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.gpu
@_CUDA
def test_missing_backend_falls_back_without_changing_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    """An optional compiler's absence retains the original CUDA expressions."""
    images = torch.full((1, 3, 3, 5), 0.5, device="cuda")
    gains = torch.zeros(1, 3, device="cuda", dtype=torch.float64)
    expected = datamodule._restore_images(images, _packed(gains), torch.float32)
    monkeypatch.setattr(_hsv_cuda, "_AVAILABLE", False)
    actual = datamodule._restore_images(images, _packed(gains), torch.float32, channels_last=True)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.gpu
@_CUDA
def test_missing_compiler_keeps_native_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runtime-only CUDA installation keeps working without a host C compiler."""
    images = torch.full((1, 3, 3, 5), 0.5, device="cuda")
    gains = torch.zeros(1, 3, device="cuda", dtype=torch.float64)
    expected = datamodule._restore_images(images, _packed(gains), torch.float32)
    monkeypatch.setattr(_hsv_cuda, "_host_compiler_available", lambda: False)
    actual = datamodule._restore_images(images, _packed(gains), torch.float32, channels_last=True)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.gpu
@_CUDA
def test_special_pixels_and_gain_extremes_keep_native_bits() -> None:
    """Keep clamp, nonfinite and signed-zero behavior at the transport boundary."""
    values = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -1.0, -0.0, 0.0, 0.5, 1.0, 2.0],
        device="cuda",
    )
    r, g, b = torch.meshgrid(values, values, values, indexing="ij")
    image = torch.stack((r, g, b)).reshape(1, 3, 1, -1)
    draws = torch.tensor(
        [
            [0, 0, 0],
            [0.5, -1, 1],
            [-0.5, 1, -1],
            [float("nan"), 0, 0],
            [0, float("nan"), 0],
            [0, 0, float("nan")],
            [float("inf"), 1, 1],
        ],
        device="cuda",
        dtype=torch.float64,
    )
    images = image.expand(len(draws), -1, -1, -1).contiguous()
    original = images.clone()
    expected = datamodule._dequantize_images(datamodule._quantize_images(_jitter_hsv_batch(images, draws)))
    actual = _hsv_cuda._restore_hsv_cuda(images, draws)
    assert actual is not None
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(images.view(torch.int32), original.view(torch.int32))


@pytest.mark.gpu
@_CUDA
def test_noncontiguous_images_keep_native_restore() -> None:
    """A transposed geometric buffer retains native stride handling."""
    images = torch.rand(2, 3, 5, 7, device="cuda").transpose(-1, -2)
    gains = torch.zeros(2, 3, device="cuda", dtype=torch.float64)
    packed = _packed(gains)
    assert _hsv_cuda._restore_hsv_cuda(images, gains) is None
    actual = datamodule._restore_images(images, packed, torch.float32, channels_last=True)
    expected = datamodule._restore_images(images, packed, torch.float32)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.gpu
@_CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_other_consumer_precision_keeps_native_restore(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    """A true-half consumer does not enter the float32 CUDA kernel."""
    images = torch.rand(1, 3, 5, 7, device="cuda")
    gains = torch.zeros(1, 3, device="cuda", dtype=torch.float64)
    packed = _packed(gains)

    def refuse_kernel(*_args: object) -> None:
        raise AssertionError("a different consumer precision must keep its native path")

    monkeypatch.setattr(_hsv_cuda, "_restore_hsv_cuda", refuse_kernel)
    actual = datamodule._restore_images(images, packed, dtype, channels_last=True)
    expected = datamodule._restore_images(images, packed, dtype)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("failure", ["absent", "build-dependency", "binary-import"])
def test_import_without_optional_compiler_keeps_cpu_path(failure: str) -> None:
    """An absent or unusable optional compiler keeps the native restore importable."""
    code = """
import builtins
import sys
import torch
original = builtins.__import__
def without_triton(name, *args, **kwargs):
    if name == "triton" or name.startswith("triton."):
        if sys.argv[1] == "binary-import":
            raise ImportError("the optional compiler binary cannot load")
        missing = "setuptools" if sys.argv[1] == "build-dependency" else "triton"
        raise ModuleNotFoundError("the optional compiler cannot load", name=missing)
    return original(name, *args, **kwargs)
builtins.__import__ = without_triton
from lucid_yolo.data import _hsv_cuda
from lucid_yolo.ptl.datamodule import _restore_images, pack_targets
from lucid_yolo.data.targets import Targets
assert not _hsv_cuda._AVAILABLE
images = torch.zeros(1, 3, 2, 2)
packed = pack_targets([Targets.empty()])
assert torch.equal(_restore_images(images, packed, torch.float32), images)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, failure],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.gpu
@_CUDA
@pytest.mark.parametrize("with_masks", [False, True])
def test_transfer_hook_uses_fusion_and_keeps_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_masks: bool
) -> None:
    """The real consumer entrance requests fusion and preserves geometry and masks."""
    target = Targets(boxes=torch.tensor([[0.0, 0.0, 3.0, 3.0]]), labels=torch.tensor([7]))
    packed = datamodule.pack_targets([target])
    if with_masks:
        packed = replace(packed, masks_cat=torch.tensor([[[True, False], [False, True]]]))
    packed = replace(
        packed,
        **{
            field.name: getattr(packed, field.name).cuda()
            for field in fields(packed)
            if isinstance(getattr(packed, field.name), torch.Tensor)
        },
    )
    gains = torch.zeros(1, 3, device="cuda", dtype=torch.float64)
    packed = replace(packed, hsv_gains=gains)
    images = torch.rand(1, 3, 3, 5, device="cuda")
    expected = datamodule._restore_images(images, packed, torch.float32)

    def refuse_native(*_args: object) -> torch.Tensor:
        raise AssertionError("the transfer hook must reach the fused colour restore")

    monkeypatch.setattr(datamodule, "_jitter_hsv_batch", refuse_native)
    dm = datamodule.DetectionDataModule(tmp_path, 1, 0, "n", img_size=32, device_hsv=True)
    actual, targets, masks = dm.on_after_batch_transfer((images, packed), 0)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert len(targets) == 1
    assert torch.equal(targets[0].boxes.cpu(), target.boxes)
    assert torch.equal(targets[0].labels.cpu(), target.labels)
    if with_masks:
        assert masks is not None and len(masks) == 1
        assert torch.equal(masks[0].cpu(), torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    else:
        assert masks is None


@pytest.mark.gpu
@_CUDA
def test_captured_hue_division_pixel_keeps_transport_codes() -> None:
    """The existing real-data hue-division counterexample also pins the fused consumer."""
    images = (
        torch.tensor([1047526623, 1047648800, 1048660522], dtype=torch.int32, device="cuda")
        .view(torch.float32)
        .reshape(1, 3, 1, 1)
    )
    gains = torch.tensor(
        [[-0.011664924521546874, 0.43333633457326814, 0.23118815543879265]],
        device="cuda",
        dtype=torch.float64,
    )
    actual = _hsv_cuda._restore_hsv_cuda(images, gains)
    assert actual is not None
    assert actual.mul(255).round().to(torch.uint8).flatten().tolist() == [71, 73, 79]


class _ImageSubclass(torch.Tensor):
    """Retain ordinary tensor dispatch in a direct consumer's custom input."""


@pytest.mark.gpu
@_CUDA
@pytest.mark.parametrize("subclass_images", [False, True])
def test_tensor_subclass_keeps_native_restore(subclass_images: bool) -> None:
    """The optional kernel must not bypass a custom tensor's dispatch."""
    images = torch.rand(1, 3, 3, 5, device="cuda")
    gains = torch.zeros(1, 3, device="cuda", dtype=torch.float64)
    if subclass_images:
        images = images.as_subclass(_ImageSubclass)
    else:
        gains = gains.as_subclass(_ImageSubclass)
    packed = _packed(gains)
    assert _hsv_cuda._restore_hsv_cuda(images, gains) is None
    expected = datamodule._restore_images(images, packed, torch.float32)
    actual = datamodule._restore_images(images, packed, torch.float32, channels_last=True)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.gpu
@_CUDA
def test_large_tiny_batch_uses_a_linear_launch() -> None:
    """The image count may exceed CUDA's second launch-axis limit without changing bits."""
    images = torch.linspace(0, 1, 65536 * 3, device="cuda").reshape(65536, 3, 1, 1)
    gains = torch.zeros((65536, 3), dtype=torch.float64, device="cuda")
    actual = _hsv_cuda._restore_hsv_cuda(images, gains)
    assert actual is not None
    expected = datamodule._dequantize_images(datamodule._quantize_images(_jitter_hsv_batch(images, gains)))
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.gpu
@_CUDA
@pytest.mark.parametrize("fallback", ["compile", "capture", "capability"])
def test_execution_scope_keeps_native_backend(fallback: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported execution scopes return before allocating or launching a custom kernel."""
    images = torch.zeros(1, 3, 3, 5, device="cuda")
    gains = torch.zeros(1, 3, device="cuda", dtype=torch.float64)
    if fallback == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif fallback == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    else:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (7, 5))
    assert _hsv_cuda._restore_hsv_cuda(images, gains) is None
