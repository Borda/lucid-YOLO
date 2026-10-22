# SPDX-License-Identifier: Apache-2.0
"""Scaffolding shared by the per-task training tests: a small module, a log stand-in and batch helpers."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from lucid_yolo.data import Targets
from lucid_yolo.ptl import DetectionLitModule, collate_detection, unpack_batch

#: Class count every training test builds its module and targets with.
NUM_CLASSES = 4


class LogRecorder:
    """Stand-in for ``LightningModule.log`` that records every logged scalar.

    Examples:
        >>> import torch
        >>> log = LogRecorder()
        >>> log("train/loss", torch.tensor(2.0), prog_bar=True)
        >>> log.values
        {'train/loss': 2.0}
    """

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def __call__(self, name: str, value: Tensor, **kwargs: object) -> None:
        """Record ``value`` under ``name``, ignoring Lightning's keyword arguments."""
        del kwargs
        self.values[name] = float(value.detach())


def has_gradient(module: nn.Module) -> bool:
    """Return whether any parameter of ``module`` carries a non-zero gradient.

    Examples:
        >>> import torch
        >>> layer = nn.Linear(2, 1)
        >>> has_gradient(layer)
        False
        >>> layer(torch.ones(1, 2)).backward()
        >>> has_gradient(layer)
        True
    """
    return any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in module.parameters())


def tiny_module(task: str = "detect", *, max_channels: int = 256, **kwargs: Any) -> DetectionLitModule:
    """Build an n-scale module with a low channel cap for fast CPU tests.

    ``kwargs`` reaches the constructor untouched, for the task-specific arguments a few
    tasks require (``num_keypoints`` for ``"keypoints"``, which has no default because the
    point count is the dataset's property rather than the model's). A caller that wants
    per-task gains binds them in its own thin wrapper.

    Examples:
        >>> tiny_module().task
        'detect'
        >>> tiny_module(task="segment").task
        'segment'
        >>> tiny_module(task="keypoints", num_keypoints=3).task
        'keypoints'
    """
    return DetectionLitModule(
        depth=0.34, width=0.25, max_channels=max_channels, num_classes=NUM_CLASSES, task=task, **kwargs
    )


def collate_unpacked(batch: list[tuple[Tensor, Targets]]) -> tuple[Tensor, list[Targets]]:
    """Collate, then restore the ``list[Targets]`` the module consumes.

    No datamodule is attached to a bare-loader ``Trainer.fit`` run, so the transfer hook
    that restores the transport form never fires; this wrapper runs the same collate ->
    restore round-trip the datamodule performs in production (dequantizing the uint8
    images and unpacking the targets) so the module receives its float images and ragged
    target list.

    Examples:
        >>> import torch
        >>> boxes = torch.tensor([[8.0, 8.0, 24.0, 24.0]])
        >>> sample = (torch.randn(3, 32, 32), Targets(boxes=boxes, labels=torch.tensor([1])))
        >>> images, targets = collate_unpacked([sample])
        >>> images.shape, len(targets)
        (torch.Size([1, 3, 32, 32]), 1)
    """
    return unpack_batch(collate_detection(batch))
