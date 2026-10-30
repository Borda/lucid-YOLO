# SPDX-License-Identifier: Apache-2.0
"""Optional CUDA scheduling for the update preparation that precedes learning-rate application.

The optimizer's own iterator supplies every expression. Private gradient inputs keep the recording independent of
autograd's gradient allocations; parameters and momentum remain the optimizer's live state. Warmup and recording are
rolled back before the first real replay. Parameter writes and the changing learning rate stay outside the graph
(R42).
"""

from __future__ import annotations

import struct
from bisect import bisect_right
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lucid_yolo.optim.replay import _ACTIVE, _capture_open, _runtime_key

if TYPE_CHECKING:
    from collections.abc import Callable

    from lucid_yolo.optim.musgd import MuSGD

#: Consecutive steps whose signature must differ from a group's recording before that recording is replaced. A shorter
#: change, such as momentum set for a single step, runs natively and then returns to the existing recording without
#: paying another capture.
_MISMATCH_TOLERANCE = 3
#: Replacement recordings a group may make after its first one. A signature that keeps changing, such as a per-step
#: momentum schedule, would otherwise pay warmup, capture and a fresh private pool every few steps; past this budget
#: the group stays native.
_RERECORD_BUDGET = 2


def _scalar_key(value: int | float) -> tuple[type[int] | type[float], int | bytes]:
    """Keep native scalar types and floating-point bits distinct in a recording."""
    return type(value), struct.pack("!d", value) if isinstance(value, float) else value


def _tensor_key(value: Tensor) -> tuple[Any, ...]:
    """Describe the storage contract a recording copies, without reading its values."""
    return value.device, value.dtype, value.shape, value.stride(), value.layout


def _dense(value: Tensor) -> bool:
    """Whether cloning preserves the layout this path has measured."""
    return value.is_contiguous() or (value.ndim == 4 and value.is_contiguous(memory_format=torch.channels_last))


def _extent(value: Tensor) -> int:
    """Count the bytes from a non-empty strided tensor's first element through its last, gaps included."""
    return (sum((size - 1) * stride for size, stride in zip(value.shape, value.stride(), strict=True)) + 1) * (
        value.element_size()
    )


def _parameter_layout(param: Tensor, update: Tensor) -> Tensor:
    """Give an update its parameter's exact strides, so one foreach write keeps its fast route.

    A channels-last ``1x1`` or depthwise kernel already counts as channels-last contiguous under strides that differ
    from the parameter's on size-one dimensions. PyTorch versions that compare whole stride vectors would then send
    the entire foreach list down the per-tensor route, so any stride difference is copied away.
    """
    return update if update.stride() == param.stride() else torch.empty_like(param).copy_(update)


class _RecordedSpans:
    """Answer whether new gradients reach a recording's parameter or momentum bytes.

    A matching key fixes each recorded tensor's address, shape, strides and dtype, and therefore its address range,
    which recording certified as pairwise disjoint. Only gradient addresses move between steps; each gradient's extent
    is fixed by its keyed shape, strides and dtype. Ranges are compared as ``_memory_overlaps`` compares them, so
    distinct storage objects wrapping one buffer are still caught.
    """

    def __init__(self, recorded: list[Tensor], gradients: list[Tensor]) -> None:
        spans = sorted((value.data_ptr(), value.data_ptr() + _extent(value)) for value in recorded)
        self.starts = [start for start, _ in spans]
        self.stops = [stop for _, stop in spans]
        self.extents = [_extent(gradient) for gradient in gradients]

    def reached_by(self, gradients: list[Tensor]) -> bool:
        """Whether any gradient, laid out as at recording, shares a byte with a recorded tensor."""
        for gradient, extent in zip(gradients, self.extents, strict=True):
            start = gradient.data_ptr()
            following = bisect_right(self.starts, start)
            if following and self.stops[following - 1] > start:
                return True
            if following < len(self.starts) and self.starts[following] < start + extent:
                return True
        return False


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
        # The key holds this stream's handle, so every replay of the recording runs on it.
        self.replay_stream = torch.cuda.current_stream(params[0].device)
        self.graph = torch.cuda.CUDAGraph()
        self.spans = _RecordedSpans([*params, *buffers], gradients)
        self.mismatches = 0
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
                updates = [_parameter_layout(param, update) for param, update in prepared]
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

        A foreach call with one incompatible pair falls back for the entire list. Layout copies are recorded with
        preparation so channel-last parameters do not turn this write back into one launch per parameter.
        """
        torch._foreach_copy_(self.gradients, gradients)
        self.graph.replay()
        torch._foreach_add_(self.parameters, self.updates, alpha=-self.group["lr"])


class _UpdateReplay:
    """Retain at most one recording per optimizer group, with native fallbacks.

    Gradient values and learning rates may change on every step. Missing gradients, shared storage, other precisions
    and other devices use the native iterator; they never reuse an incompatible recording. A step whose signature
    differs from the group's recording also runs natively and keeps that recording, so a change lasting fewer than
    ``_MISMATCH_TOLERANCE`` consecutive steps returns to it without a new capture. Steps that skip the comparison (a
    missing gradient, say) neither extend nor reset that run. On the ``_MISMATCH_TOLERANCE``-th consecutive differing
    step the stream that replayed the recording is synchronized, the recording is released and the new signature is
    recorded in its place. A group may be re-recorded ``_RERECORD_BUDGET`` times after its first recording; once that
    budget is spent, the next replacement only releases the recording and the group stays native until :meth:`clear`.
    Strong references keep each recorded group, its parameters and momentum alive with their identities.

    Attributes:
        entries: Live recordings keyed by group identity.
        recordings: Recordings published per group identity since the last clear, released ones included. Only
            ``MuSGD.load_state_dict`` replaces groups, and it clears this owner first.
    """

    def __init__(self, overlaps: Callable[[list[Tensor]], bool]) -> None:
        self.entries: dict[int, _PreparedGroup] = {}
        self.recordings: dict[int, int] = {}
        self._overlaps = overlaps

    def clear(self) -> None:
        """Release all recordings and spent budgets before restored optimizer state can be observed."""
        self.entries.clear()
        self.recordings.clear()

    def step(self, optimizer: MuSGD, group: dict[str, Any]) -> bool:
        """Run a certified recording, or leave the caller to its native step.

        The full input check runs only when recording. A matching key already fixes every parameter and momentum
        address, layout and scalar type that check certified, so a replayed step re-checks only what the key leaves
        out: the learning-rate type and layouts (in :meth:`_eligible`) and gradient addresses. Two recording-time
        checks are deliberately not repeated. Gradients overlapping one another are harmless, since both routes only
        read gradients. Parameters or buffers sharing a storage base without sharing bytes are too, since graph replay
        depends on the addresses the key fixes, not on storage objects.
        """
        inputs = self._eligible(optimizer, group)
        if inputs is None:
            return False
        params, gradients, buffers = inputs
        entry = self.entries.get(id(group))
        if entry is None:
            return self._record_group(optimizer, group, params, gradients, buffers)
        if entry.key == self._key(params, gradients, buffers, group):
            entry.mismatches = 0
            if entry.spans.reached_by(gradients):
                return False
            entry.run(gradients)
            return True
        entry.mismatches += 1
        if entry.mismatches < _MISMATCH_TOLERANCE:
            return False
        self._release(group, entry)
        del entry  # free the old pool before a replacement allocates its own
        return self._record_group(optimizer, group, params, gradients, buffers)

    def _record_group(
        self,
        optimizer: MuSGD,
        group: dict[str, Any],
        params: list[Tensor],
        gradients: list[Tensor],
        buffers: list[Tensor],
    ) -> bool:
        """Record and run the group's current signature unless its budget is spent or its inputs are unsupported."""
        if self._exhausted(group) or not self._supported(params, gradients, buffers, group):
            return False
        entry = _PreparedGroup(
            optimizer, group, params, gradients, buffers, self._key(params, gradients, buffers, group)
        )
        entry.run(gradients)
        entry.replay_stream.synchronize()
        self.entries[id(group)] = entry
        self.recordings[id(group)] = self.recordings.get(id(group), 0) + 1
        return True

    def _exhausted(self, group: dict[str, Any]) -> bool:
        """Whether the group has used its first recording and every replacement."""
        return self.recordings.get(id(group), 0) > _RERECORD_BUDGET

    def _release(self, group: dict[str, Any], entry: _PreparedGroup) -> None:
        """Drop a recording once the last replay reading its pool has finished.

        The previous step's parameter write still reads the recorded updates. The
        current stream may already differ from the one that replayed them (the stream
        is part of the signature), so the recording's own stream is synchronized.
        """
        entry.replay_stream.synchronize()
        del self.entries[id(group)]

    def _eligible(
        self, optimizer: MuSGD, group: dict[str, Any]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]] | None:
        """Return the group's parameters, gradients and momentum when this step may replay."""
        params = group["params"]
        if (
            not params
            or (id(group) not in self.entries and self._exhausted(group))
            or any(type(param) is not torch.nn.Parameter or not param.is_cuda for param in params)
            or torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
            or torch.is_autocast_enabled()
            or torch.compiler.is_compiling()
            or _capture_open(params[0].device)
        ):
            return None
        gradients = [param.grad for param in params]
        buffers = [optimizer.state.get(param, {}).get("momentum_buffer") for param in params]
        if any(type(value) is not Tensor for value in [*gradients, *buffers]):
            return None
        # ``lr`` is not keyed, and building the key reads strides and addresses, which
        # sparse and other non-strided tensors do not have.
        if type(group["lr"]) not in {int, float} or any(
            value.layout != torch.strided for value in (*params, *gradients, *buffers)
        ):
            return None
        return params, gradients, buffers

    @staticmethod
    def _key(
        params: list[Tensor], gradients: list[Tensor], buffers: list[Tensor], group: dict[str, Any]
    ) -> tuple[Any, ...]:
        """Describe everything a recording fixes: identities, storage contracts, scalars and runtime."""
        scalars = tuple(
            _scalar_key(group[name]) for name in ("momentum", "weight_decay", "w_muon", "w_sgd", "ns_steps")
        )
        return (
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
            # fill_uninitialized_memory stays unkeyed: it only changes bytes correct preparation never reads.
            _runtime_key(params[0].device),
        )

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
