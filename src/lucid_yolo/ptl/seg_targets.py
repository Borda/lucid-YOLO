# SPDX-License-Identifier: Apache-2.0
"""Segmentation supervision targets built from the batch's polygons (WP-087).

Two dense targets are needed to train the mask side of the model, and both are
derived here from the very rings :class:`~lucid_yolo.data.targets.Targets` carries
through augmentation:

- **per-instance masks** on the prototype grid, which
  :func:`~lucid_yolo.losses.mask_loss.instance_mask_loss` scores the assembled
  Eq. 7 masks against; and
- **the auxiliary semantic map** on the fused-feature grid — for each class, the
  union of that class's instance masks — which
  :func:`~lucid_yolo.losses.semantic_loss.semantic_aux_loss` supervises (A17).

Both grids are **derived from the prediction tensors** by the caller and passed in;
nothing here assumes an input size or a stride. The prototype grid is twice the P3
feature (A15, i.e. a quarter of the input side at stride 8) and the semantic grid is
P3 itself, but neither number is written down anywhere: a model whose ProtoNet
upsample changed would silently train against stale target grids.

Two choices are load-bearing:

1. **Polygons are scaled, then rasterised** — never rasterised at input resolution
   and then resized. A thin instance covering a few input pixels survives the
   coordinate scaling (its ring keeps its shape) but is aliased away by a
   point-sampled downscale of its raster.
2. **The semantic target is max-pooled from the instance masks**, rather than
   rasterised a second time at the coarser grid. One rasterisation pass feeds both
   consumers, so the two targets cannot disagree about where an instance is, and
   the max keeps any instance that occupies a single fine cell — which a second,
   coarser point-sampled rasterisation would drop.

Provenance: R1 Eq. 7-9, R1 sec. 3.4.1. Assumptions: A15, A17, A38.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from lucid_yolo.data.rasterize import rasterize_polygons

if TYPE_CHECKING:
    from torch import Tensor

    from lucid_yolo.data.targets import Targets

__all__ = ["instance_mask_targets", "scale_boxes_to_grid", "semantic_target"]


def _grid_scale(image_size: tuple[int, int], grid_size: tuple[int, int]) -> tuple[float, float]:
    """Return the ``(x, y)`` factors mapping input pixels onto a feature grid.

    The single derivation of the two factors, so box coordinates and polygon
    coordinates cannot be scaled by different numbers. They are separate factors
    rather than one ratio because a letterboxed canvas need not be square.

    Args:
        image_size: Input canvas ``(height, width)`` the coordinates live in.
        grid_size: Target grid ``(height, width)``.

    Returns:
        The ``(x_scale, y_scale)`` pair.
    """
    height, width = image_size
    grid_height, grid_width = grid_size
    return grid_width / width, grid_height / height


def scale_boxes_to_grid(boxes: Tensor, image_size: tuple[int, int], grid_size: tuple[int, int]) -> Tensor:
    """Rescale ``xyxy`` boxes from input pixels into a feature grid's own frame.

    :func:`~lucid_yolo.losses.mask_loss.instance_mask_loss` crops by comparing box
    edges against integer pixel indices of the mask grid, so it documents that its
    boxes must already be expressed in that grid's frame. This is that conversion.

    Args:
        boxes: ``(..., 4)`` boxes in ``xyxy`` input pixels.
        image_size: Input canvas ``(height, width)`` the boxes live in.
        grid_size: Target grid ``(height, width)``.

    Returns:
        A new tensor of the same shape, with ``x`` columns scaled by
        ``grid_width / width`` and ``y`` columns by ``grid_height / height``.

    Examples:
        >>> import torch
        >>> boxes = torch.tensor([[0.0, 0.0, 32.0, 64.0]])
        >>> scale_boxes_to_grid(boxes, image_size=(128, 128), grid_size=(32, 32))
        tensor([[ 0.,  0.,  8., 16.]])
    """
    x_scale, y_scale = _grid_scale(image_size, grid_size)
    scale = torch.tensor([x_scale, y_scale, x_scale, y_scale], dtype=boxes.dtype, device=boxes.device)
    return boxes * scale


def instance_mask_targets(target: Targets, image_size: tuple[int, int], grid_size: tuple[int, int]) -> Tensor:
    """Rasterise one image's instance polygons onto a feature grid.

    The rings are scaled from input pixels into the grid's frame and then
    rasterised there (see the module docstring for why not the other way round).
    Rasterisation runs on CPU — :func:`~lucid_yolo.data.rasterize.rasterize_polygon`
    builds its coordinate grids there — so the result is returned on CPU and the
    caller moves it to the prediction device.

    Args:
        target: One image's targets. Under ``task="segment"`` every instance must
            carry a polygon ring; an image with no instances is fine.
        image_size: Input canvas ``(height, width)`` the polygons live in.
        grid_size: Target grid ``(height, width)``.

    Returns:
        ``(N, grid_height, grid_width)`` float32 masks in ``{0.0, 1.0}``, one per
        instance and in the instance order of ``target.boxes``.

    Raises:
        ValueError: If the image carries boxes without one polygon ring each —
            a detection-only annotation cannot supervise masks, and silently
            skipping it would train the mask branches on a subset nobody chose.

    Examples:
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> ring = torch.tensor([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]])
        >>> target = Targets(boxes=torch.tensor([[0.0, 0.0, 8.0, 8.0]]), labels=torch.tensor([0]), polygons=[ring])
        >>> masks = instance_mask_targets(target, image_size=(16, 16), grid_size=(4, 4))
        >>> masks.shape, float(masks.sum())  # the 8x8 ring covers a 2x2 block of the 4x4 grid
        (torch.Size([1, 4, 4]), 4.0)
    """
    count = int(target.boxes.shape[0])
    if len(target.polygons) != count:
        raise ValueError(
            f"segmentation targets need one polygon ring per instance; got {len(target.polygons)} rings "
            f"for {count} boxes (a detection-only annotation cannot supervise masks)"
        )
    x_scale, y_scale = _grid_scale(image_size, grid_size)
    scale = torch.tensor([x_scale, y_scale], dtype=torch.float32)
    rings = [ring.detach().to(device="cpu", dtype=torch.float32) * scale for ring in target.polygons]
    return rasterize_polygons(rings, grid_size[0], grid_size[1]).to(torch.float32)


def semantic_target(masks: Tensor, labels: Tensor, num_classes: int, grid_size: tuple[int, int]) -> Tensor:
    """Build one image's per-class semantic map: the union of each class's instances.

    The instance masks are max-pooled onto ``grid_size`` and then unioned per class,
    so an instance occupying a single fine cell still marks its coarse cell (see the
    module docstring). Overlapping instances of one class union to ``1.0``, never to
    a count.

    Args:
        masks: ``(N, H, W)`` per-instance masks in ``{0.0, 1.0}``, as returned by
            :func:`instance_mask_targets` (any grid at least as fine as
            ``grid_size``).
        labels: ``(N,)`` int64 class ids, one per mask, all in
            ``[0, num_classes)``.
        num_classes: Channel count ``C`` of the auxiliary branch's logits.
        grid_size: Output grid ``(height, width)`` — the fused feature's own size.

    Returns:
        ``(C, grid_height, grid_width)`` float targets in ``{0.0, 1.0}``, on
        ``masks``' device and dtype. An image with no instances yields all zeros.

    Examples:
        >>> import torch
        >>> masks = torch.zeros(2, 4, 4)
        >>> masks[0, 0, 0] = 1.0  # class 1, one fine cell
        >>> masks[1, 2:, 2:] = 1.0  # class 0, a quarter of the grid
        >>> target = semantic_target(masks, torch.tensor([1, 0]), num_classes=3, grid_size=(2, 2))
        >>> target.shape
        torch.Size([3, 2, 2])
        >>> target[1, 0, 0], target[0, 1, 1]  # the single fine cell survives the pooling
        (tensor(1.), tensor(1.))
    """
    grid_height, grid_width = grid_size
    if masks.shape[0] == 0:
        return masks.new_zeros((num_classes, grid_height, grid_width))
    pooled = F.adaptive_max_pool2d(masks, (grid_height, grid_width))  # (N, gh, gw)
    one_hot = F.one_hot(labels.to(masks.device), num_classes).to(pooled.dtype)  # (N, C)
    return torch.einsum("nc,nhw->chw", one_hot, pooled).clamp(max=1.0)
