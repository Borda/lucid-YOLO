# SPDX-License-Identifier: Apache-2.0
"""Scaffolding shared by the per-task training tests: a small module, a log stand-in and batch helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from lucid_yolo.data import Targets
from lucid_yolo.data.rotated_geom import canonicalize, rboxes_to_polygons
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

    ``kwargs`` reaches the constructor untouched, for the task-specific arguments a few tasks require (``num_keypoints``
    for ``"keypoints"``, which has no default because the point count is the dataset's property rather than the
    model's). A caller that wants per-task gains binds them in its own thin wrapper.

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

    No datamodule is attached to a bare-loader ``Trainer.fit`` run, so the transfer hook that restores the transport
    form never fires; this wrapper runs the same collate -> restore round-trip the datamodule performs in production
    (dequantizing the uint8 images and unpacking the targets) so the module receives its float images and ragged target
    list.

    Examples:
        >>> import torch
        >>> boxes = torch.tensor([[8.0, 8.0, 24.0, 24.0]])
        >>> sample = (torch.randn(3, 32, 32), Targets(boxes=boxes, labels=torch.tensor([1])))
        >>> images, targets = collate_unpacked([sample])
        >>> images.shape, len(targets)
        (torch.Size([1, 3, 32, 32]), 1)
    """
    return unpack_batch(collate_detection(batch))


def _rectangle_ring(box: Tensor) -> Tensor:
    """Return the four-point polygon ring tracing an ``xyxy`` box.

    Examples:
        >>> _rectangle_ring(torch.tensor([0.0, 0.0, 4.0, 2.0]))
        tensor([[0., 0.],
                [4., 0.],
                [4., 2.],
                [0., 2.]])
    """
    x1, y1, x2, y2 = box.tolist()
    return torch.tensor([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=torch.float32)


def _random_boxes(num_boxes: int) -> Tensor:
    """Draw ``num_boxes`` valid ``xyxy`` boxes: corners within 80 px, sides of 10 to 50 px.

    Draws the corners first and the sizes second from the global torch RNG, an order the callers' later draws depend on.

    Examples:
        >>> boxes = _random_boxes(3)
        >>> boxes.shape, bool((boxes[:, 2:] > boxes[:, :2]).all())
        (torch.Size([3, 4]), True)
    """
    top_left = torch.rand(num_boxes, 2) * 80.0
    size = torch.rand(num_boxes, 2) * 40.0 + 10.0
    return torch.cat([top_left, top_left + size], dim=1)


def segment_targets(num_boxes: int) -> Targets:
    """Build ``num_boxes`` valid ``xyxy`` targets, each with its own polygon ring.

    Examples:
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> targets = segment_targets(2)
        >>> targets.boxes.shape, len(targets.polygons)
        (torch.Size([2, 4]), 2)
    """
    boxes = _random_boxes(num_boxes)
    labels = torch.randint(0, NUM_CLASSES, (num_boxes,))
    return Targets(boxes=boxes, labels=labels, polygons=[_rectangle_ring(box) for box in boxes])


def oriented_targets(num_boxes: int, difficult: Tensor | None = None) -> Targets:
    """Build elongated, rotated ground truths with envelopes fitted to the same corners.

    Elongation and rotation are load-bearing, not incidental: an axis-aligned or square target sits at the stationary
    point of both angular terms, where a correctly wired angle stem legitimately receives zero gradient.

    Examples:
        >>> targets = oriented_targets(2)
        >>> targets.boxes.shape, targets.rboxes.shape
        (torch.Size([2, 4]), torch.Size([2, 5]))
    """
    centre = torch.rand(num_boxes, 2) * 60.0 + 50.0
    long_edge = torch.rand(num_boxes) * 30.0 + 30.0
    short_edge = torch.rand(num_boxes) * 8.0 + 8.0
    theta = torch.rand(num_boxes) * 0.9 + 0.2
    rboxes = canonicalize(torch.stack([centre[:, 0], centre[:, 1], long_edge, short_edge, theta], dim=1))
    corners = rboxes_to_polygons(rboxes)
    boxes = torch.cat([corners.amin(dim=1), corners.amax(dim=1)], dim=1)
    return Targets(
        boxes=boxes,
        labels=torch.randint(0, NUM_CLASSES, (num_boxes,)),
        rboxes=rboxes,
        difficult=torch.zeros(num_boxes, dtype=torch.bool) if difficult is None else difficult,
    )


def keypoint_targets(num_boxes: int, visibility: Tensor | None = None) -> Targets:
    """Build boxes with one point set per instance, inside the canvas.

    The points are placed independently of the boxes on purpose: nothing in the objective ties a point to its own box's
    interior, and a fixture that put them at box centres would let a decode that ignored the anchor grid look correct.

    Examples:
        >>> targets = keypoint_targets(2)
        >>> targets.boxes.shape, targets.keypoints.shape, targets.keypoint_vis.shape
        (torch.Size([2, 4]), torch.Size([2, 3, 2]), torch.Size([2, 3]))
    """
    boxes = _random_boxes(num_boxes)
    points = torch.rand(num_boxes, 3, 2) * float(160)
    return Targets(
        boxes=boxes,
        labels=torch.randint(0, NUM_CLASSES, (num_boxes,)),
        keypoints=points,
        keypoint_vis=torch.full((num_boxes, 3), 2, dtype=torch.int64) if visibility is None else visibility,
    )
