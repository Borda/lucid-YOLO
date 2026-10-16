# SPDX-License-Identifier: Apache-2.0
"""The deferred-HSV guard judges the attached trainer by what its precision plugin does."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.plugins.precision import (
    DeepSpeedPrecision,
    FSDPPrecision,
    HalfPrecision,
    MixedPrecision,
    Precision,
)

from lucid_yolo.data.augment import HSVParams
from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl.datamodule import DetectionDataModule, _DeferredHSVSample, collate_detection, unpack_batch

#: Real plugins that accept an allowlisted precision label yet cast floating batch inputs:
#: FSDP narrows the float64 gains to float32, DeepSpeed's mixed modes also halve the images.
_CASTING_PLUGINS = [
    pytest.param(
        partial(FSDPPrecision, "16-mixed"),
        id="fsdp-16-mixed",
        marks=pytest.mark.filterwarnings("ignore:torch.cuda.amp.GradScaler is enabled"),
    ),
    pytest.param(partial(FSDPPrecision, "bf16-mixed"), id="fsdp-bf16-mixed"),
    pytest.param(partial(FSDPPrecision, "32-true"), id="fsdp-32-true"),
    pytest.param(partial(DeepSpeedPrecision, "16-mixed"), id="deepspeed-16-mixed"),
    pytest.param(partial(DeepSpeedPrecision, "bf16-mixed"), id="deepspeed-bf16-mixed"),
    pytest.param(partial(DeepSpeedPrecision, "32-true"), id="deepspeed-32-true"),
]


def _module(fixture_dir: Path) -> DetectionDataModule:
    """Build a deferred-HSV datamodule pointing both splits at the micro-dataset's one split.

    Examples:
        >>> callable(_module)  # needs a live detseg_fixture_dir fixture
        True
    """
    split = fixture_dir / "train"
    annotation = split / "_annotations.coco.json"
    return DetectionDataModule(
        fixture_dir,
        batch_size=2,
        num_workers=0,
        variant="n",
        img_size=32,
        train_images_dir=split,
        train_ann_file=annotation,
        val_images_dir=split,
        val_ann_file=annotation,
        device_hsv=True,
    )


def _cuda_trainer(plugin: Precision) -> SimpleNamespace:
    """Stand in for a CUDA trainer that exposes a real precision plugin and its label.

    Examples:
        >>> _cuda_trainer(Precision()).precision
        '32-true'
    """
    return SimpleNamespace(accelerator=CUDAAccelerator(), precision=plugin.precision, precision_plugin=plugin)


@pytest.mark.parametrize("make_plugin", _CASTING_PLUGINS)
def test_casting_plugin_keeps_worker_hsv(detseg_fixture_dir: Path, make_plugin: Callable[[], Precision]) -> None:
    """A plugin that casts the transport under an allowlisted label keeps worker-side HSV.

    FSDP and DeepSpeed accept ``32-true``/``16-mixed``/``bf16-mixed`` yet cast floating
    batch inputs before the transfer hook, so trusting the label would run the colour
    transform on rounded gains, and for DeepSpeed mixed on half-precision images.
    """
    plugin = make_plugin()
    module = _module(detseg_fixture_dir)
    module.trainer = _cuda_trainer(plugin)
    with pytest.warns(UserWarning, match=f"under {type(plugin).__name__}"):
        module.setup("fit")
    images, packed = next(iter(module.train_dataloader()))
    assert images.dtype == torch.uint8
    assert packed.hsv_gains is None


@pytest.mark.parametrize(
    "make_plugin",
    [
        pytest.param(Precision, id="base-32-true"),
        pytest.param(partial(MixedPrecision, "bf16-mixed", "cuda"), id="amp-bf16-mixed"),
    ],
)
def test_preserving_plugin_still_defers(detseg_fixture_dir: Path, make_plugin: Callable[[], Precision]) -> None:
    """A plugin whose input conversion is a no-op keeps the deferred transport, silently.

    Single-device float32 and native AMP are the configurations the flag exists for;
    the probe must not refuse them, or the guard would turn the feature off everywhere.
    """
    module = _module(detseg_fixture_dir)
    module.trainer = _cuda_trainer(make_plugin())
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="device_hsv")
        module.setup("fit")
    images, packed = next(iter(module.train_dataloader()))
    assert images.dtype == torch.float32
    assert packed.hsv_gains is not None


@pytest.mark.parametrize("make_plugin", _CASTING_PLUGINS)
def test_cast_deferred_batch_fails_at_restore(make_plugin: Callable[[], Precision]) -> None:
    """A deferred batch passed through a casting plugin's real conversion is refused at restore.

    This is the boundary the precision-label stubs bypass: Lightning calls
    ``convert_input`` before the transfer hook, so a cast that slipped past the guard
    must fail loudly rather than jitter rounded gains or half-precision images.
    """
    sample = _DeferredHSVSample(torch.rand(3, 4, 4), Targets.empty(), HSVParams(0.012, 0.31, -0.27))
    converted = make_plugin().convert_input(collate_detection([sample]))
    with pytest.raises(TypeError, match="deferred HSV batch arrived with"):
        unpack_batch(converted)


def test_unattached_setup_neither_warns_nor_latches(detseg_fixture_dir: Path) -> None:
    """A setup before any trainer is attached stays silent and leaves the warning for later.

    User code may call ``setup("fit")`` before ``trainer.fit``, which runs setup again
    with the trainer attached. The early call cannot know the backend, so saying the
    flag is ignored "off CUDA" would be false, and spending the once-per-module warning
    there would hide a genuine fallback found by a later setup.
    """
    module = _module(detseg_fixture_dir)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="device_hsv")
        module.setup("fit")
        unattached = module._train._defer_hsv
        module.trainer = _cuda_trainer(Precision())
        module.setup("fit")
        attached = module._train._defer_hsv
    module.trainer = _cuda_trainer(HalfPrecision("16-true"))
    with pytest.warns(UserWarning, match="ignored for precision='16-true'"):
        module.setup("fit")
    assert (unattached, attached, module._train._defer_hsv) == (False, True, False)
