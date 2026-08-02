# SPDX-License-Identifier: Apache-2.0
"""Type-generic geometric transform protocol and shared geometry helpers.

Every augmentation in Phase 1 (letterbox, affine, mosaic, mixup, flip —
WP-009…013) and Phase 8 (rotated-aware variants) conforms to
:class:`GeometricTransform`: it maps a CHW image and its :class:`~lucid_yolo.data.targets.Targets`
to a transformed pair. :class:`Compose` chains such transforms.

The two pure helpers here are the single source of geometric truth the concrete
transforms build on: :func:`apply_affine_to_points` applies a homogeneous
``3x3`` matrix to a batch of points (the primitive behind box, polygon and
rotated-box warping), and :func:`boxes_from_polygons` derives axis-aligned boxes
from polygon rings (used whenever a polygon is clipped and its enclosing box must
be recomputed). Keeping them small and exact means the box/polygon/rbox paths
stay consistent by construction rather than by three parallel reimplementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from lucid_yolo.data.targets import Targets

__all__ = ["Compose", "GeometricTransform", "apply_affine_to_points", "boxes_from_polygons"]

_POINT_DIM = 2
_BOX_DIM = 4
_AFFINE_SHAPE = (3, 3)


@runtime_checkable
class GeometricTransform(Protocol):
    """Callable mapping an image and its targets to a transformed pair.

    Implementations must transform ``image`` and ``targets`` jointly so that
    every modality carried by :class:`~lucid_yolo.data.targets.Targets` stays
    geometrically consistent with the returned image.

    Examples:
        ```pycon
        >>> from lucid_yolo.data.transforms import Compose, GeometricTransform
        >>> isinstance(Compose([]), GeometricTransform)
        True

        ```
    """

    def __call__(self, image: Tensor, targets: Targets) -> tuple[Tensor, Targets]:
        """Transform ``image`` and ``targets`` jointly."""
        ...


class Compose:
    """Apply a sequence of :class:`GeometricTransform` in order.

    Args:
        transforms: The transforms to apply left-to-right. An empty sequence is
            the identity.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> identity = Compose([])
        >>> image = torch.zeros(3, 8, 8)
        >>> out_image, out_targets = identity(image, Targets.empty())
        >>> out_image.shape
        torch.Size([3, 8, 8])

        ```
    """

    def __init__(self, transforms: Sequence[GeometricTransform]) -> None:
        self.transforms: list[GeometricTransform] = list(transforms)

    def __call__(self, image: Tensor, targets: Targets) -> tuple[Tensor, Targets]:
        """Run each transform in turn, threading the image/targets pair through.

        Args:
            image: CHW image tensor.
            targets: The targets to carry alongside ``image``.

        Returns:
            The image and targets after all transforms have been applied.
        """
        for transform in self.transforms:
            image, targets = transform(image, targets)
        return image, targets


def apply_affine_to_points(points: Tensor, matrix: Tensor) -> Tensor:
    """Apply a homogeneous ``3x3`` transform to a batch of 2-D points.

    Each point ``(x, y)`` is mapped to ``matrix @ [x, y, 1]`` and de-homogenised
    by its third coordinate, so both affine and projective matrices are handled
    exactly (for affine matrices the divisor is ``1``).

    Args:
        points: ``(K, 2)`` point coordinates.
        matrix: ``(3, 3)`` homogeneous transform matrix.

    Returns:
        ``(K, 2)`` transformed point coordinates, in ``points``'s dtype.

    Raises:
        ValueError: If ``points`` is not ``(K, 2)`` or ``matrix`` is not ``(3, 3)``.

    Examples:
        ```pycon
        >>> import torch
        >>> # rotate +90 degrees about the origin, then translate by (2, 3)
        >>> rot_translate = torch.tensor([[0.0, -1.0, 2.0], [1.0, 0.0, 3.0], [0.0, 0.0, 1.0]])
        >>> apply_affine_to_points(torch.tensor([[1.0, 0.0]]), rot_translate)
        tensor([[2., 4.]])

        ```
    """
    if points.ndim != 2 or points.shape[1] != _POINT_DIM:
        raise ValueError(f"points must be (K, 2); got shape {tuple(points.shape)}")
    if tuple(matrix.shape) != _AFFINE_SHAPE:
        raise ValueError(f"matrix must be (3, 3); got shape {tuple(matrix.shape)}")
    ones = points.new_ones((points.shape[0], 1))
    homogeneous = torch.cat([points, ones], dim=1)
    projected = homogeneous @ matrix.transpose(0, 1)
    return projected[:, :_POINT_DIM] / projected[:, _POINT_DIM : _POINT_DIM + 1]


def boxes_from_polygons(polygons: list[Tensor]) -> Tensor:
    """Derive axis-aligned ``xyxy`` boxes as the extent of each polygon ring.

    Args:
        polygons: List of ``(P_i, 2)`` float32 point rings. An empty list yields
            an empty ``(0, 4)`` box tensor.

    Returns:
        ``(N, 4)`` float32 boxes, one per input ring, as ``(x_min, y_min, x_max,
        y_max)``.

    Raises:
        ValueError: If any ring is not ``(P, 2)`` or has zero points (no extent).

    Examples:
        ```pycon
        >>> import torch
        >>> ring = torch.tensor([[1.0, 2.0], [5.0, 2.0], [5.0, 8.0], [1.0, 8.0]])
        >>> boxes_from_polygons([ring])
        tensor([[1., 2., 5., 8.]])

        ```
    """
    if not polygons:
        return torch.zeros((0, _BOX_DIM), dtype=torch.float32)
    boxes: list[Tensor] = []
    for i, ring in enumerate(polygons):
        if ring.ndim != 2 or ring.shape[1] != _POINT_DIM:
            raise ValueError(f"polygon[{i}] must be (P, 2); got shape {tuple(ring.shape)}")
        if ring.shape[0] == 0:
            raise ValueError(f"polygon[{i}] has no points; cannot derive a box")
        lo = ring.amin(dim=0)
        hi = ring.amax(dim=0)
        boxes.append(torch.cat([lo, hi]))
    return torch.stack(boxes, dim=0)
