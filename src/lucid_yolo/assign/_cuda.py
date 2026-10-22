# SPDX-License-Identifier: Apache-2.0
"""Optional CUDA scheduling for TAL's unchanged dense assignment expressions.

The opt-in scope belongs to one assignment, including nested calls and failures.
Outputs own ordinary PyTorch storage; only compiled-signature metadata survives a
call. First launches finish before becoming capture-ready. A first toolchain failure
warns once and keeps that family native; errors after a verified launch propagate.
"""

from __future__ import annotations

import math
import os
import shutil
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import TypeVar

import torch
from torch import Tensor

try:
    from lucid_yolo.assign import _cuda_kernels as kernels
except ImportError:
    _AVAILABLE = False
else:
    _AVAILABLE = kernels._AVAILABLE

_Result = TypeVar("_Result")
_ENABLED: ContextVar[bool] = ContextVar("lucid_cuda_assignment", default=False)
#: Launches are measured over the public-data shapes, rather than tuned per target count.
_BLOCK = 256
_ANCHOR_BLOCK = 16
#: Validated ground-truth reduction capacity; larger layouts keep native scheduling.
_MAX_GT = 512
#: NVIDIA's y-grid bound, also exposed by the existing runtime's get_max_y_grid.
_MAX_GRID_Y = 65535
#: Runtime integer arguments and per-axis offsets use this compiler index width.
_MAX_INDEX = torch.iinfo(torch.int32).max


@contextmanager
def _assignment_scope(enabled: bool) -> Iterator[None]:
    """Apply one assigner's request and restore its caller even when the assignment fails.

    Examples:
        ```pycon
        >>> with _assignment_scope(True):
        ...     _requested()
        True
        >>> _requested()
        False

        ```
    """
    token = _ENABLED.set(enabled)
    try:
        yield
    finally:
        _ENABLED.reset(token)


def _requested() -> bool:
    """Keep compiler tracing native without reading context-local state in the trace."""
    return (
        not torch.compiler.is_compiling()
        and not vars(torch.jit)["is_tracing"]()
        and not vars(torch.jit)["is_scripting"]()
        and _ENABLED.get()
    )


@dataclass
class _BackendState:
    """Retain compiler readiness and failure decisions, never tensors or graph pools."""

    ready: set[tuple[object, ...]] = field(default_factory=set)
    verified: set[tuple[str, torch.device]] = field(default_factory=set)
    unavailable: set[tuple[str, torch.device]] = field(default_factory=set)


_STATE = _BackendState()


@lru_cache(maxsize=1)
def _host_compiler_available() -> bool:
    """Respect an explicit host compiler before searching the usual launcher toolchains."""
    compiler = os.environ.get("CC")
    return bool(shutil.which(compiler)) if compiler else bool(shutil.which("gcc") or shutil.which("clang"))


@cache
def _device_supported(device: torch.device) -> bool:
    """Cache the documented NVIDIA hardware boundary of the optional compiler."""
    return torch.cuda.get_device_capability(device)[0] >= 8


def _indexable(value: Tensor) -> bool:
    """Keep the kernel's runtime index ABI and every relative strided offset in range."""
    strides = value.stride()
    return (
        value.numel() <= _MAX_INDEX
        and all(0 <= stride <= _MAX_INDEX for stride in strides)
        and sum((size - 1) * stride for size, stride in zip(value.shape, strides, strict=True)) <= _MAX_INDEX
    )


def _supported(kind: str, values: tuple[Tensor, ...]) -> bool:
    """Keep tensor subclasses, other devices/layouts and unavailable toolchains native."""
    if not _AVAILABLE or any(type(value) is not Tensor for value in values):
        return False
    first = values[0]
    if (kind, first.device) in _STATE.unavailable:
        return False
    return (
        first.is_cuda
        and torch.version.hip is None
        and all(
            value.device == first.device
            and value.layout == torch.strided
            and not value.is_conj()
            and not value.is_neg()
            and _indexable(value)
            for value in values
        )
        and not (torch.is_grad_enabled() and any(value.requires_grad for value in values))
        and torch.cuda.current_device() == first.device.index
        and _host_compiler_available()
        and _device_supported(first.device)
    )


def _run(
    kind: str, signature: tuple[object, ...], device: torch.device, produce: Callable[[], _Result]
) -> _Result | None:
    """Publish readiness only after a first eager launch and its completion fence.

    A previously unseen specialization stays native inside capture: its compilation
    belongs to an earlier eager warmup. The same runtime-size kernel can serve many
    target counts, so those counts are deliberately absent from its readiness key.
    """
    family = kind, device
    key = kind, device, *signature
    if family in _STATE.unavailable:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and key not in _STATE.ready:
        return None
    try:
        result = produce()
        if key not in _STATE.ready:
            torch.cuda.synchronize(device)
    except Exception as error:
        if family in _STATE.verified:
            raise
        _STATE.unavailable.add(family)
        warnings.warn(
            f"fused CUDA assignment {kind} failed its first launch ({type(error).__name__}: {error}); "
            "this family keeps the native path for the rest of the process",
            UserWarning,
            stacklevel=3,
        )
        return None
    _STATE.ready.add(key)
    _STATE.verified.add(family)
    return result


def _cuda_metric(
    scores: Tensor, pred: Tensor, gt: Tensor, labels: Tensor, mask: Tensor, alpha: float, beta: float, eps: float
) -> tuple[Tensor, Tensor] | None:
    """Fuse IoU and score/mask products while keeping PyTorch's exact power primitive."""
    values = scores, pred, gt, labels, mask
    if not _requested() or not _supported("metric", values):
        return None
    if (
        scores.ndim != 3
        or pred.shape != (*scores.shape[:2], 4)
        or gt.ndim != 3
        or gt.shape[0] != scores.shape[0]
        or gt.shape[-1] != 4
        or labels.shape != gt.shape[:2]
        or mask.shape != (*gt.shape[:2], scores.shape[1])
        or not all(value.is_contiguous() for value in (pred, gt, labels))
        or pred.dtype != torch.float32
        or gt.dtype != torch.float32
        or scores.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or labels.dtype != torch.int64
        or mask.dtype != torch.bool
        or type(alpha) not in (int, float)
        or alpha != 1.0
        or type(beta) not in (int, float)
        or not math.isfinite(beta)
        or beta < 0
        or type(eps) not in (int, float)
        or not 0 < eps <= torch.finfo(torch.float32).max
        or min(*scores.shape, gt.shape[1]) == 0
    ):
        return None
    batch, anchors, classes = scores.shape
    count = gt.shape[1]
    if (anchors + _BLOCK - 1) // _BLOCK > _MAX_GRID_Y:
        return None
    signature = anchors, classes, scores.dtype, scores.stride()

    def produce() -> tuple[Tensor, Tensor]:
        shape = batch, count, anchors
        iou = torch.empty(shape, device=gt.device, dtype=torch.float32)
        metric = torch.empty_like(iou)
        grid = batch * count, (anchors + _BLOCK - 1) // _BLOCK
        kernels._iou_pass[grid](gt, pred, iou, count, anchors, float(eps), _BLOCK, enable_fp_fusion=False)
        powered = iou.pow(beta)
        kernels._metric_pass[grid](
            scores,
            labels,
            mask,
            powered,
            metric,
            count,
            anchors,
            classes,
            *scores.stride(),
            *mask.stride(),
            _BLOCK,
            enable_fp_fusion=False,
            debug=True,
        )
        return metric, iou

    return _run("metric", signature, scores.device, produce)


def _cuda_inside(points: Tensor, boxes: Tensor) -> Tensor | None:
    """Fuse scalar containment with the native sine/cosine values for each rotated box."""
    if not _requested() or not _supported("inside", (points, boxes)):
        return None
    if (
        points.ndim != 2
        or points.shape[-1] != 2
        or boxes.ndim != 3
        or boxes.shape[-1] != 5
        or points.dtype != torch.float32
        or boxes.dtype != torch.float32
        or not boxes.is_contiguous()
        or points.shape[0] == 0
        or (points.shape[0] + _BLOCK - 1) // _BLOCK > _MAX_GRID_Y
        or boxes.shape[0] == 0
        or boxes.shape[1] == 0
    ):
        return None
    batch, count = boxes.shape[:2]
    anchors = points.shape[0]

    def produce() -> Tensor:
        result = torch.empty((batch, count, anchors), dtype=torch.bool, device=boxes.device)
        theta = boxes[..., 4]
        cosine, sine = torch.cos(theta).contiguous(), torch.sin(theta).contiguous()
        kernels._inside_pass[(batch * count, (anchors + _BLOCK - 1) // _BLOCK)](
            points,
            boxes,
            cosine,
            sine,
            result,
            anchors,
            *points.stride(),
            _BLOCK,
            enable_fp_fusion=False,
        )
        return result

    return _run("inside", (anchors, points.stride()), points.device, produce)


def _dense_rows(kind: str, mask: Tensor, metric: Tensor) -> bool:
    """Bound the register-resident ground-truth reduction and preserve unsupported views."""
    return (
        _requested()
        and _supported(kind, (mask, metric))
        and mask.ndim == 3
        and metric.shape == mask.shape
        and mask.dtype == torch.bool
        and metric.dtype == torch.float32
        and mask.is_contiguous()
        and metric.is_contiguous()
        and min(mask.shape) > 0
        and mask.shape[1] <= _MAX_GT
        and (mask.shape[-1] + _ANCHOR_BLOCK - 1) // _ANCHOR_BLOCK <= _MAX_GRID_Y
    )


def _cuda_conflicts(mask: Tensor, metric: Tensor) -> Tensor | None:
    """Keep the first maximum among selected ground truths, including NaN and tie cases."""
    if not _dense_rows("conflicts", mask, metric):
        return None
    batch, count, anchors = mask.shape
    padded = 1 << (count - 1).bit_length()

    def produce() -> Tensor:
        result = torch.empty_like(mask)
        kernels._conflicts_pass[(batch, (anchors + _ANCHOR_BLOCK - 1) // _ANCHOR_BLOCK)](
            mask,
            metric,
            result,
            count,
            anchors,
            padded,
            _ANCHOR_BLOCK,
            enable_fp_fusion=False,
            num_warps=8,
        )
        return result

    return _run("conflicts", (anchors, padded), mask.device, produce)
