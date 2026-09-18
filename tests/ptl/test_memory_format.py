# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``channels_last`` memory-format policy (WP-182).

On a CUDA accelerator :meth:`~lucid_yolo.ptl.module.DetectionLitModule.setup`
converts the parameters to ``channels_last`` and the datamodule's
``on_after_batch_transfer`` hands over images in the same layout. No GPU runs
here, so what is proved on CPU is the part a layout change can break silently:
every task at two variants runs a full ``_shared_step`` over ``channels_last``
weights and images to a finite loss — the one local check that the PSA
attention's ``view`` over the ``qkv`` projection survives non-contiguous strides —
the ``detect``/``n`` loss is numerically the same in either layout for the same
weights and batch, and neither hook touches the layout off CUDA: a CPU batch
leaves the transfer hook contiguous and a CPU trainer leaves the parameters as
built.

Direct ``_shared_step`` calls monkeypatch ``module.log`` to a no-op, as the other
module tests do, because ``self.log`` needs a trainer.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import CUDAAccelerator
from torch import Tensor

from lucid_yolo.data.rotated_geom import canonicalize, rboxes_to_polygons
from lucid_yolo.data.targets import Targets
from lucid_yolo.models.registry import scale_spec
from lucid_yolo.ptl import DetectionDataModule, DetectionLitModule, collate_detection

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

#: Class count of the tiny test head.
_NUM_CLASSES = 4
#: Point count of the keypoint modules; arbitrary, the layout does not depend on it.
_NUM_KEYPOINTS = 3
#: Square input side; divisible by every stride for an integer anchor grid.
_IMG_SIZE = 160
#: Images per synthetic batch.
_BATCH_SIZE = 2
#: The four shipped tasks, each parametrized against both small variants.
_TASKS = ("detect", "segment", "obb", "keypoints")
#: The variants small enough to run a full step on CPU in seconds.
_VARIANTS = ("n", "s")


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed torch before each test so model init and synthetic batches are reproducible."""
    torch.manual_seed(0)
    yield


def _module(task: str, variant: str) -> DetectionLitModule:
    """Build a module for ``task`` at the published multipliers of ``variant``.

    Examples:
        >>> _module("detect", "n").task
        'detect'
        >>> _module("keypoints", "s").task
        'keypoints'
    """
    spec = scale_spec(variant)
    extra = {"num_keypoints": _NUM_KEYPOINTS} if task == "keypoints" else {}
    return DetectionLitModule(
        depth=spec.depth,
        width=spec.width,
        max_channels=spec.max_channels,
        num_classes=_NUM_CLASSES,
        task=task,
        **extra,
    )


def _boxes(num_boxes: int) -> tuple[Tensor, Tensor]:
    """Draw ``num_boxes`` valid ``xyxy`` boxes inside the canvas, with labels.

    Examples:
        >>> boxes, labels = _boxes(2)
        >>> boxes.shape, labels.shape
        (torch.Size([2, 4]), torch.Size([2]))
    """
    top_left = torch.rand(num_boxes, 2) * 80.0
    size = torch.rand(num_boxes, 2) * 40.0 + 10.0
    return torch.cat([top_left, top_left + size], dim=1), torch.randint(0, _NUM_CLASSES, (num_boxes,))


def _targets(task: str, num_boxes: int) -> Targets:
    """Build ``num_boxes`` targets carrying whatever geometry ``task`` supervises.

    ``segment`` gets the rectangular ring of each box, ``obb`` elongated rotated boxes
    with envelopes fitted to their corners, ``keypoints`` one visible point set per box.

    Examples:
        >>> _targets("detect", 2).boxes.shape
        torch.Size([2, 4])
        >>> len(_targets("segment", 2).polygons)
        2
        >>> _targets("obb", 2).rboxes.shape
        torch.Size([2, 5])
        >>> _targets("keypoints", 2).keypoints.shape
        torch.Size([2, 3, 2])
    """
    boxes, labels = _boxes(num_boxes)
    if task == "segment":
        rings = [
            torch.tensor([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=torch.float32)
            for x1, y1, x2, y2 in boxes.tolist()
        ]
        return Targets(boxes=boxes, labels=labels, polygons=rings)
    if task == "obb":
        centre = torch.rand(num_boxes, 2) * 60.0 + 50.0
        long_edge = torch.rand(num_boxes) * 30.0 + 30.0
        short_edge = torch.rand(num_boxes) * 8.0 + 8.0
        theta = torch.rand(num_boxes) * 0.9 + 0.2
        rboxes = canonicalize(torch.stack([centre[:, 0], centre[:, 1], long_edge, short_edge, theta], dim=1))
        corners = rboxes_to_polygons(rboxes)
        envelopes = torch.cat([corners.amin(dim=1), corners.amax(dim=1)], dim=1)
        return Targets(boxes=envelopes, labels=labels, rboxes=rboxes)
    if task == "keypoints":
        points = torch.rand(num_boxes, _NUM_KEYPOINTS, 2) * float(_IMG_SIZE)
        visibility = torch.full((num_boxes, _NUM_KEYPOINTS), 2, dtype=torch.int64)
        return Targets(boxes=boxes, labels=labels, keypoints=points, keypoint_vis=visibility)
    return Targets(boxes=boxes, labels=labels)


def _batch(task: str) -> tuple[Tensor, list[Targets]]:
    """Build a two-image batch with ragged (2 and 1) instance counts for ``task``.

    Examples:
        >>> images, targets = _batch("detect")
        >>> images.shape, [target.boxes.shape[0] for target in targets]
        (torch.Size([2, 3, 160, 160]), [2, 1])
    """
    images = torch.randn(_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)
    return images, [_targets(task, 2), _targets(task, 1)]


def _silence_log(module: DetectionLitModule) -> None:
    """Replace ``module.log`` with a no-op, so a direct step needs no trainer.

    Examples:
        >>> module = _module("detect", "n")
        >>> _silence_log(module)
        >>> _ = module.log("train/loss", torch.tensor(0.0))  # no trainer, no error
    """
    module.log = MagicMock()  # type: ignore[method-assign]


def _step_loss(module: DetectionLitModule, batch: tuple[Tensor, list[Targets]]) -> Tensor:
    """Run one training-stage shared step and return the scalar total.

    Examples:
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> module = _module("detect", "n")
        >>> _silence_log(module)
        >>> _step_loss(module, _batch("detect")).shape
        torch.Size([])
    """
    total, _head_out, _seg_out = module._shared_step(batch, "train")
    return total


def _all_contiguous(module: DetectionLitModule) -> bool:
    """Report whether every parameter keeps the default (NCHW) layout.

    A 4-D ``channels_last`` tensor reports ``is_contiguous()`` false, so this
    discriminates the two layouts on the convolution weights.

    Examples:
        >>> _all_contiguous(_module("detect", "n"))
        True
        >>> _all_contiguous(_module("detect", "n").to(memory_format=torch.channels_last))
        False
    """
    return all(parameter.is_contiguous() for parameter in module.parameters())


class TestChannelsLastStep:
    """Every task at both small variants steps to a finite loss in ``channels_last``."""

    @pytest.mark.parametrize("variant", [pytest.param(variant, id=variant) for variant in _VARIANTS])
    @pytest.mark.parametrize("task", [pytest.param(task, id=task) for task in _TASKS])
    def test_loss_is_finite(self, task: str, variant: str) -> None:
        """``channels_last`` weights over ``channels_last`` images run the whole step.

        The PSA attention views its ``qkv`` projection straight into heads, so this is
        the check that the view survives the strides the layout gives that tensor.
        """
        module = _module(task, variant).to(memory_format=torch.channels_last)
        _silence_log(module)
        images, targets = _batch(task)

        total = _step_loss(module, (images.contiguous(memory_format=torch.channels_last), targets))

        assert total.ndim == 0
        assert torch.isfinite(total)

    def test_loss_matches_contiguous(self) -> None:
        """The same weights and batch score the same loss in either layout (``detect``/``n``)."""
        contiguous = _module("detect", "n")
        channels_last = copy.deepcopy(contiguous).to(memory_format=torch.channels_last)
        _silence_log(contiguous)
        _silence_log(channels_last)
        images, targets = _batch("detect")

        reference = _step_loss(contiguous, (images, targets))
        converted = _step_loss(channels_last, (images.contiguous(memory_format=torch.channels_last), targets))

        assert torch.allclose(converted, reference, rtol=1e-5, atol=1e-6)


class TestLayoutOffCuda:
    """Neither hook touches the layout on CPU, so the CPU/MPS goldens are untouched."""

    def test_transfer_hook_keeps_cpu_images_contiguous(self, tmp_path: Path) -> None:
        """A CPU batch leaves ``on_after_batch_transfer`` in the default layout."""
        datamodule = DetectionDataModule(data_root=tmp_path, batch_size=_BATCH_SIZE, num_workers=0, variant="n")
        samples = [(torch.randint(0, 256, (3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8), _targets("detect", 1))] * 2

        images, targets, masks = datamodule.on_after_batch_transfer(collate_detection(samples), 0)

        assert images.is_contiguous()
        assert not images.is_contiguous(memory_format=torch.channels_last)
        assert images.dtype == torch.float32
        assert len(targets) == _BATCH_SIZE
        assert masks is None

    def test_setup_on_cpu_trainer_keeps_parameters_contiguous(self) -> None:
        """A ``Trainer(accelerator="cpu")`` leaves every parameter as built."""
        module = _module("detect", "n")
        module._trainer = Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)

        module.setup("fit")

        assert _all_contiguous(module)

    def test_setup_without_trainer_keeps_parameters_contiguous(self) -> None:
        """A bare module sets up without a trainer and keeps its layout."""
        module = _module("detect", "n")

        module.setup("fit")

        assert _all_contiguous(module)

    def test_setup_on_cuda_accelerator_converts(self) -> None:
        """A trainer reporting a CUDA accelerator converts the parameters — the branch a GPU run takes.

        :class:`CUDAAccelerator` constructs without a device, so the positive branch is
        reachable here; a typo in the accelerator test would pass every other check.
        """
        module = _module("detect", "n")
        module._trainer = MagicMock(spec=Trainer, world_size=1, accelerator=CUDAAccelerator())

        module.setup("fit")

        assert not _all_contiguous(module)
        assert all(
            parameter.is_contiguous(memory_format=torch.channels_last)
            for parameter in module.parameters()
            if parameter.ndim == 4
        )
