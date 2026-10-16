# SPDX-License-Identifier: Apache-2.0
"""Replay the optimizer's pure orthogonalizations without changing their arithmetic (WP-197).

The optimizer owns this optional cache, not the parameters or their momentum.
Only the two Newton--Schulz functions opt into it. Their input is copied before
replay and their output is copied afterwards, so neither a later call nor a
checkpoint can observe the graph's reusable storage. Capture runs the existing
function; it does not trace, compile or substitute any of its expressions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Any, ParamSpec, cast

import torch
from torch import Tensor

_Orthogonalize = Callable[[Tensor, int, float], Tensor]
_P = ParamSpec("_P")
_ACTIVE: ContextVar[_OrthogonalizationReplay | None] = ContextVar("orthogonalization_replay", default=None)


@dataclass
class _Recorded:
    """Keep a captured function's input, output and graph alive together."""

    source: Tensor
    result: Tensor
    graph: torch.cuda.CUDAGraph


class _OrthogonalizationReplay:
    """Own a bounded set of optional CUDA orthogonalization recordings.

    ``capacity`` is the optimizer's number of matrix parameters, a structural
    upper bound on its ordinary distinct matrix calls. A changed signature can
    fill unused slots, but never grows the cache beyond that bound; overflow
    executes eagerly. Each graph has its own allocator pool, so skipped gradients
    and changing bucket order do not impose a replay-order dependency.

    Only contiguous, zero-offset inputs are recorded. Cloning another layout can
    change the normalization's reduction order; those calls remain eager.

    Attributes:
        capacity: Maximum number of retained recordings.
        entries: Recordings keyed by the function and its execution signature.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.entries: dict[tuple[Any, ...], _Recorded] = {}
        self._streams: dict[torch.device, torch.cuda.Stream] = {}

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Enable this owner for one optimizer group, restoring the previous scope."""
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def clear(self) -> None:
        """Release recordings while keeping the per-device warmup streams reusable."""
        self.entries.clear()

    def run(self, function: _Orthogonalize, value: Tensor, steps: int, eps: float) -> Tensor:
        """Replay one supported signature, or keep the eager function's contract."""
        if (
            type(value) is not Tensor
            or type(steps) is not int
            or type(eps) is not float
            or not value.is_cuda
            or torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
            or torch.is_autocast_enabled()
            or value.layout != torch.strided
            or not value.is_floating_point()
            or not value.is_contiguous()
            or value.storage_offset() != 0
            or value.numel() == 0
            or torch.cuda.is_current_stream_capturing()
        ):
            return function(value, steps, eps)
        key = (
            function,
            value.device,
            value.dtype,
            value.shape,
            value.stride(),
            steps,
            eps,
            torch.get_float32_matmul_precision(),
            getattr(torch.backends, "fp32_precision", None),
            getattr(torch.backends.cuda.matmul, "fp32_precision", None),
            torch.backends.cuda.preferred_blas_library(),
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.cuda.current_stream(value.device).cuda_stream,
        )
        recorded = self.entries.get(key)
        if recorded is None:
            if len(self.entries) >= self.capacity:
                return function(value, steps, eps)
            recorded = self._record(function, value, steps, eps)
            self.entries[key] = recorded
        recorded.source.copy_(value)
        recorded.graph.replay()
        return recorded.result.clone()

    def _record(self, function: _Orthogonalize, value: Tensor, steps: int, eps: float) -> _Recorded:
        """Warm, capture and finish a first replay before publishing its buffers."""
        with torch.cuda.device(value.device):
            source = value.clone()
            stream = self._streams.get(value.device)
            if stream is None:
                stream = torch.cuda.Stream(device=value.device)  # type: ignore[no-untyped-call]
                self._streams[value.device] = stream
            current = torch.cuda.current_stream(value.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                function(source, steps, eps)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = function(source, steps, eps)
            graph.replay()
            current.synchronize()
        return _Recorded(source, result, graph)


def _replay_orthogonalization(function: Callable[_P, Tensor]) -> Callable[_P, Tensor]:
    """Keep an orthogonalization eager unless an optimizer explicitly owns replay."""
    parameters = signature(function)

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> Tensor:
        if torch.compiler.is_compiling():
            return function(*args, **kwargs)
        owner = _ACTIVE.get()
        if owner is None:
            return function(*args, **kwargs)
        bound = parameters.bind(*args, **kwargs)
        bound.apply_defaults()
        value, steps, eps = bound.arguments.values()
        return owner.run(cast("_Orthogonalize", function), value, steps, eps)

    return wrapped
