# SPDX-License-Identifier: Apache-2.0
"""Optional CUDA scheduling for the update preparation that precedes learning-rate application.

The optimizer's own iterator supplies every expression. Private gradient inputs
keep the recording independent of autograd's gradient allocations; parameters
and momentum remain the optimizer's live state. Warmup and recording are rolled
back before the first real replay. Parameter writes and the changing learning
rate stay outside the graph (R42).
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lucid_yolo.optim.replay import _ACTIVE, _capture_open

if TYPE_CHECKING:
    from collections.abc import Callable

    from lucid_yolo.optim.musgd import MuSGD


def _scalar_key(value: int | float) -> tuple[type[int] | type[float], int | bytes]:
    """Keep native scalar types and floating-point bits distinct in a recording."""
    return type(value), struct.pack("!d", value) if isinstance(value, float) else value


def _tensor_key(value: Tensor) -> tuple[Any, ...]:
    """Describe the storage contract a recording copies, without reading its values."""
    return value.device, value.dtype, value.shape, value.stride(), value.layout


def _dense(value: Tensor) -> bool:
    """Whether cloning preserves the layout this path has measured."""
    return value.is_contiguous() or (value.ndim == 4 and value.is_contiguous(memory_format=torch.channels_last))


def _runtime_key(device: torch.device) -> tuple[Any, ...]:
    """Keep a recording tied to its arithmetic settings and submission stream."""
    return (
        torch.get_float32_matmul_precision() if not hasattr(torch.backends.cuda.matmul, "fp32_precision") else None,
        getattr(torch.backends, "fp32_precision", None),
        getattr(torch.backends.cuda.matmul, "fp32_precision", None),
        torch.backends.cuda.preferred_blas_library(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        getattr(torch.utils.deterministic, "fill_uninitialized_memory", False),
        torch.cuda.current_stream(device).cuda_stream,
    )


class _PreparedGroup:
    """Own one group's private inputs, live momentum and recorded update tensors."""

    def __init__(
        self,
        optimizer: MuSGD,
        group: dict[str, Any],
        params: list[Tensor],
        gradients: list[Tensor],
        buffers: list[Tensor],
        key: tuple[Any, ...],
    ) -> None:
        self.group = group
        self.key = key
        self.params = tuple(params)
        self.buffers = buffers
        self.gradients = [gradient.detach().clone() for gradient in gradients]
        self.stream = torch.cuda.Stream(device=params[0].device)  # type: ignore[no-untyped-call]
        self.graph = torch.cuda.CUDAGraph()
        self._record(optimizer, gradients)

    def _record(self, optimizer: MuSGD, gradients: list[Tensor]) -> None:
        """Restore gradients and momentum before warmup can count as a real update."""
        snapshots = [buffer.clone() for buffer in self.buffers]
        current = torch.cuda.current_stream(self.params[0].device)
        token = _ACTIVE.set(None)
        try:
            for param, private in zip(self.params, self.gradients, strict=True):
                param.grad = private
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                list(optimizer._group_updates(self.group))
            current.wait_stream(self.stream)
            torch._foreach_copy_(self.buffers, snapshots)
            self.stream.wait_stream(current)
            with torch.cuda.graph(self.graph, stream=self.stream):
                prepared = list(optimizer._group_updates(self.group))
                updates = []
                for param, update in prepared:
                    memory_format = (
                        torch.channels_last
                        if param.ndim == 4 and param.is_contiguous(memory_format=torch.channels_last)
                        else torch.contiguous_format
                    )
                    updates.append(update.contiguous(memory_format=memory_format))
            self.parameters = [param for param, _ in prepared]
            self.updates = updates
        finally:
            for param, gradient in zip(self.params, gradients, strict=True):
                param.grad = gradient
            try:
                current.wait_stream(self.stream)
                torch._foreach_copy_(self.buffers, snapshots)
                current.synchronize()
            finally:
                _ACTIVE.reset(token)

    def run(self, gradients: list[Tensor]) -> None:
        """Apply the current rate to layout-compatible updates in one native batch.

        A foreach call with one incompatible pair falls back for the entire list.
        Layout copies are recorded with preparation so channel-last parameters do
        not turn this write back into one launch per parameter.
        """
        torch._foreach_copy_(self.gradients, gradients)
        self.graph.replay()
        torch._foreach_add_(self.parameters, self.updates, alpha=-self.group["lr"])


class _UpdateReplay:
    """Retain at most one recording per optimizer group, with native fallbacks.

    Gradient values and learning rates may change on every step. Other signature
    changes, missing gradients, shared storage, other precisions and other devices
    use the native iterator; they never reuse an incompatible recording. Strong
    references keep the group, parameters and momentum alive with their identities.
    """

    def __init__(self, overlaps: Callable[[list[Tensor]], bool]) -> None:
        self.entries: dict[int, _PreparedGroup] = {}
        self._overlaps = overlaps

    def clear(self) -> None:
        """Release all recordings before restored optimizer state can be observed."""
        self.entries.clear()

    def step(self, optimizer: MuSGD, group: dict[str, Any]) -> bool:
        """Run a certified recording, or leave the caller to its native step."""
        params = group["params"]
        if (
            not params
            or any(type(param) is not torch.nn.Parameter or not param.is_cuda for param in params)
            or torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
            or torch.is_autocast_enabled()
            or torch.compiler.is_compiling()
            or _capture_open(params[0].device)
        ):
            return False
        gradients = [param.grad for param in params]
        buffers = [optimizer.state.get(param, {}).get("momentum_buffer") for param in params]
        if any(type(value) is not Tensor for value in [*gradients, *buffers]):
            return False
        if not self._supported(params, gradients, buffers, group):
            return False
        scalars = tuple(
            _scalar_key(group[name]) for name in ("momentum", "weight_decay", "w_muon", "w_sgd", "ns_steps")
        )
        key = (
            tuple(
                (
                    id(param),
                    param.data_ptr(),
                    _tensor_key(param),
                    _tensor_key(gradient),
                    id(buffer),
                    buffer.data_ptr(),
                    _tensor_key(buffer),
                )
                for param, gradient, buffer in zip(params, gradients, buffers, strict=True)
            ),
            scalars,
            _runtime_key(params[0].device),
        )
        entry = self.entries.get(id(group))
        if entry is not None and entry.key != key:
            return False
        if entry is None:
            entry = _PreparedGroup(optimizer, group, params, gradients, buffers, key)
            entry.run(gradients)
            torch.cuda.current_stream(params[0].device).synchronize()
            self.entries[id(group)] = entry
        else:
            entry.run(gradients)
        return True

    def _supported(
        self, params: list[Tensor], gradients: list[Tensor], buffers: list[Tensor], group: dict[str, Any]
    ) -> bool:
        """Require independent dense float32 storage before recording any input."""
        if any(type(group[name]) not in {int, float} for name in ("lr", "momentum", "weight_decay", "w_muon", "w_sgd")):
            return False
        if type(group["ns_steps"]) is not int:
            return False
        values = [*params, *gradients, *buffers]
        device = params[0].device
        if any(
            value.device != device
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or value.numel() == 0
            or not _dense(value)
            for value in values
        ):
            return False
        if any(
            param.shape != gradient.shape or param.shape != buffer.shape
            for param, gradient, buffer in zip(params, gradients, buffers, strict=True)
        ):
            return False
        parameter_storage = {param.untyped_storage().data_ptr() for param in params}
        buffer_storage = {buffer.untyped_storage().data_ptr() for buffer in buffers}
        gradient_storage = {gradient.untyped_storage().data_ptr() for gradient in gradients}
        if len(parameter_storage) != len(params) or len(buffer_storage) != len(buffers):
            return False
        return not (
            parameter_storage & buffer_storage
            or gradient_storage & (parameter_storage | buffer_storage)
            or self._overlaps(values)
        )
