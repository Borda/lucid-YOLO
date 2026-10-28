# SPDX-License-Identifier: Apache-2.0
"""Shared CPU routing scaffolds and private-state access for capture tests."""

from __future__ import annotations

import torch

from lucid_yolo.ptl import DetectionLitModule
from lucid_yolo.ptl.loss_capture import _LossCapture, _LossObjective
from lucid_yolo.ptl.network_capture import _NetworkCapture


class CudaFlagged(torch.Tensor):
    """Report CUDA residency while running operations on ordinary CPU tensors.

    Examples:
        >>> tensor = torch.ones(1).as_subclass(CudaFlagged)
        >>> tensor.is_cuda, tensor.device.type
        (True, 'cpu')
        >>> (tensor + 1).is_cuda
        False
    """

    is_cuda = True

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        """Run the operation on the underlying data and return an ordinary tensor."""
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **(kwargs or {}))


def cuda_images() -> torch.Tensor:
    """Make CPU images that pass the CUDA routing predicate.

    Returns:
        A two-image tensor reporting CUDA residency without moving to a GPU.

    Examples:
        >>> images = cuda_images()
        >>> images.is_cuda, images.device.type, tuple(images.shape)
        (True, 'cpu', (2, 3, 64, 64))
    """
    return torch.rand(2, 3, 64, 64).as_subclass(CudaFlagged)


def loss_objective(holder: _LossCapture) -> _LossObjective | None:
    """Expose the recorded objective without spreading its storage name.

    Args:
        holder: Loss replay holder whose recording is inspected.

    Returns:
        The recorded objective, or None before recording or after clearing.

    Examples:
        >>> loss_objective(_LossCapture()) is None
        True
    """
    return holder._objective


def network_capture(module: DetectionLitModule) -> _NetworkCapture:
    """Expose the module's network replay holder.

    Args:
        module: Training module that owns the replay holder.

    Returns:
        The live network replay holder, without copying its state.

    Examples:
        >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=2)
        >>> isinstance(network_capture(module), _NetworkCapture)
        True
    """
    return module._network_capture


def pose_stems(holder: _NetworkCapture) -> dict[int, _NetworkCapture]:
    """Expose the live child holders used for pose stem replay.

    Args:
        holder: Network replay holder that owns the child recordings.

    Returns:
        The live mapping from stem identities to their replay holders.

    Examples:
        >>> pose_stems(_NetworkCapture())
        {}
    """
    return holder._stems


def head_topology(module: DetectionLitModule) -> tuple[tuple[object, ...], ...]:
    """Expose the topology certificate captured at module construction.

    Args:
        module: Training module whose head certificate is inspected.

    Returns:
        The stored topology certificate, empty for tasks without one.

    Examples:
        >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=2)
        >>> head_topology(module)
        ()
    """
    return module._head_topology


def refuse_precision_query() -> tuple[object, ...]:
    """Refuse a precision query with the original runtime diagnostic.

    Raises:
        RuntimeError: Always, to model a rejected precision-policy query.

    Examples:
        >>> refuse_precision_query()
        Traceback (most recent call last):
            ...
        RuntimeError: mix of the legacy and new APIs
    """
    raise RuntimeError("mix of the legacy and new APIs")


def recording_synchronize(order: list[str], device: object = None) -> None:
    """Record the fence before synchronizing the current CUDA stream.

    Args:
        order: Mutable event log shared with buffer-restoration observers.
        device: Ignored argument accepted by the patched synchronization seam.

    Examples:
        >>> from unittest.mock import patch
        >>> order = []
        >>> with patch("torch.cuda.current_stream") as current:
        ...     recording_synchronize(order)
        >>> order, current.return_value.synchronize.call_count
        (['synchronize'], 1)
    """
    order.append("synchronize")
    torch.cuda.current_stream().synchronize()
