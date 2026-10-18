# SPDX-License-Identifier: Apache-2.0
"""MuSGD -- the hybrid Muon/SGD optimizer for the detector backbone (WP-032).

MuSGD (R1 sec. 3.3.1) combines two updates per parameter, split by parameter
rank:

* **Matrix parameters** (``ndim >= 2`` -- conv kernels and linear weights) get
  a weighted sum of a Muon-style update and an SGD-momentum update. The Muon
  half orthogonalizes the momentum-adjusted gradient with the Newton-Schulz
  iteration (:func:`lucid_yolo.optim.orthogonalize`) and rescales it by
  ``0.2 * sqrt(max(A, B))`` for the ``(A, B)`` matrix view (R7 Eq. 4's
  update-RMS matching, verbatim ``W_t = W_{t-1} - eta*(0.2*O_t*sqrt(max(A,B)) +
  lambda*W_{t-1})``). A conv kernel of shape ``(out, in, kh, kw)`` is viewed as
  the 2D matrix ``(out, in*kh*kw)`` for the orthogonalization and reshaped back.
  Decoupled weight decay (the ``lambda*W`` term) is applied here and only here.
* **Vector parameters** (``ndim <= 1`` -- biases and norm scales) get a pure
  SGD-momentum update with **no** weight decay ever (A12) and no Muon component.

Single-momentum-buffer design (A27). The papers do not pin down how the two
branches share optimizer state or where Nesterov momentum sits. This module
takes the minimal reading of "hybrid update from one momentum state": **one**
momentum buffer per parameter accumulates the raw gradient
(``m <- momentum*m + g``), and both branches are derived from the same
Nesterov-adjusted gradient ``g + momentum*m`` (R8: "Nesterov-style momentum
works a bit better"). The Muon branch orthogonalizes and rescales that vector;
the SGD branch uses it directly. The vector-parameter path uses the same
Nesterov-adjusted gradient, so a single code path produces every SGD update.

The additive gains ``w_muon`` and ``w_sgd`` are independent and do not sum to 1
(A7). This module is written by hand from the published equations (AGENTS.md
sec. 6-7); no optimizer implementation is consulted.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from typing import TYPE_CHECKING, Any, NamedTuple, overload

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from lucid_yolo.optim.newton_schulz import _orthogonalize_batched, orthogonalize
from lucid_yolo.optim.replay import _OrthogonalizationReplay
from lucid_yolo.optim.update_replay import _UpdateReplay

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from torch.optim.optimizer import ParamsT

#: Update-RMS scaling constant for the Muon branch (R7 Eq. 4: ``0.2 * O_t``).
_MUON_RMS_SCALE: float = 0.2

#: Device types on which :meth:`MuSGD._step_group` batches compatible matrices and
#: updates each stack with one native batched call instead of one call per parameter,
#: and on which :meth:`MuSGD._foreach_nesterov` advances the momentum buffers with
#: foreach operations. Both read this one constant so a test can widen the gate to CPU.
#: Batching trades many small kernel launches for one large one, so it pays exactly
#: where launch overhead dominates. The gate was set on the production n-scale detector
#: while the stack was updated through :func:`torch.vmap`: CUDA (L4) is where the win was
#: found, while MPS ran that vmap path at 108.45 ms per optimizer step against 60.99 ms
#: for the per-parameter loop and CPU moved 39.64 ms to 38.47 ms. The native batch that
#: replaced it (WP-193) was measured on CUDA (L4) and not re-measured on MPS or CPU, so
#: the CUDA-only gate is inherited rather than re-derived. A device outside this set
#: keeps the loop, which is also the arithmetic the tests pin.
_BATCHED_DEVICE_TYPES: frozenset[str] = frozenset({"cuda"})

#: Largest smaller dimension in the measured n-scale workload. Padding above this
#: region regressed the x-scale L4 optimizer screen; those matrices keep exact shapes.
_PADDED_MAX_ROWS = 256

#: Subclasses may implement tensor arithmetic without exposing ordinary storage.
_STORAGE_TYPES = (Tensor, torch.nn.Parameter)


class _BucketKey(NamedTuple):
    """Matrix geometry, both dtypes and device defining one CUDA launch group.

    ``shape`` is a one-element smaller-dimension key for independent nonempty
    matrices within the measured row region, or the original parameter shape for
    larger matrices, tensor subclasses, shared storage and empty views.
    The latter retain exact-shape batching. Parameter and adjusted-gradient dtypes
    remain separate: a restored or mixed-precision gradient can promote the
    Nesterov expression, and another matrix must not inherit that precision.
    """

    shape: torch.Size
    param_dtype: torch.dtype
    grad_dtype: torch.dtype
    device: torch.device


class _PendingUpdate(NamedTuple):
    """A parameter and the Nesterov-adjusted gradient its update has yet to be made from."""

    param: Tensor
    nesterov_grad: Tensor


def _memory_overlaps(tensors: list[Tensor]) -> bool:
    """Whether any two non-empty tensors reach the same bytes, whatever storage wraps them.

    Compares the address range each strided tensor can touch instead of its storage's
    base pointer: ``torch.from_dlpack`` and similar imports wrap one buffer in distinct
    storage objects whose base pointers differ while their ranges intersect.
    """
    spans = sorted(
        (
            tensor.data_ptr(),
            tensor.data_ptr()
            + (sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True)) + 1)
            * tensor.element_size(),
        )
        for tensor in tensors
        if tensor.numel() > 0
    )
    reach = 0
    for start, stop in spans:
        if start < reach:
            return True
        reach = max(reach, stop)
    return False


def _matrix_shape(gradient: Tensor) -> tuple[int, int]:
    """Return the wide orientation used by both Newton--Schulz implementations."""
    rows, columns = gradient.shape[0], math.prod(gradient.shape[1:])
    return min(rows, columns), max(rows, columns)


def _padding_batches(bucket: list[_PendingUpdate]) -> Iterator[list[_PendingUpdate]]:
    """Group widths while keeping each padded gradient stack within its members' own bytes.

    A batch may hold padded gradients no larger than the parameter and gradient bytes
    of the matrices in that batch, the amount an exact-shape stack of the same
    matrices would hold. Only the gradients are padded, so that per-batch sum supplies
    the byte budget rather than an arbitrary aspect-ratio limit; it does not compare
    with any stack the exact-shape path builds for singletons or mixed shapes. Sorting
    by width makes each extension pay for the smallest remaining width; a width that
    exceeds the budget starts another batch. No tensor value is read on the host.
    """
    batch: list[_PendingUpdate] = []
    budget = 0
    for pending in sorted(bucket, key=lambda item: _matrix_shape(item.nesterov_grad)[1]):
        gradient = pending.nesterov_grad
        rows, columns = _matrix_shape(gradient)
        cost = pending.param.numel() * pending.param.element_size() + gradient.numel() * gradient.element_size()
        padded_bytes = (len(batch) + 1) * rows * columns * gradient.element_size()
        if batch and padded_bytes > budget + cost:
            yield batch
            batch, budget = [], 0
        batch.append(pending)
        budget += cost
    if batch:
        yield batch


class MuSGD(Optimizer):
    """Hybrid Muon/SGD optimizer with a rank-based parameter-type split.

    Matrix parameters (``ndim >= 2``) receive ``w_muon * muon + w_sgd * sgd``
    plus decoupled weight decay; vector parameters (``ndim <= 1``) receive pure
    SGD momentum with no weight decay. See the module docstring for the Muon
    scaling (R7 Eq. 4), the single-momentum-buffer design (A27), and the
    independent-gains convention (A7).

    Args:
        params: Iterable of parameters or parameter-group dicts to optimize.
        lr: Learning rate ``eta`` (required, must be non-negative).
        momentum: Momentum coefficient ``mu`` for the shared buffer; in
            ``[0, 1)``. Defaults to 0.95.
        weight_decay: Decoupled weight-decay coefficient ``lambda``, applied to
            matrix parameters only (A12). Must be non-negative. Defaults to
            5e-4.
        w_muon: Additive gain on the Muon branch. Must be non-negative. Defaults
            to 0.5.
        w_sgd: Additive gain on the SGD branch. Must be non-negative. Defaults
            to 0.5.
        ns_steps: Newton-Schulz iteration count for the orthogonalization
            (A5). Must be positive. Defaults to 5.
        capture_orthogonalization: Record and replay the existing CUDA matrix
            iterations. Defaults to ``False`` because capture retains device
            workspace and pays setup per signature; CPU, MPS, autograd and
            autocast calls stay eager. This flag alone captures no momentum or parameter update.
        capture_updates: Record the existing update preparation on independent dense
            float32 CUDA parameters after momentum initialization. Defaults to ``False``.
            Momentum is live optimizer state, while private gradient buffers receive each
            new step. Learning rates and parameter writes stay outside the graph. Changed
            signatures, shared storage, missing gradients, other devices, autograd,
            autocast and inference mode stay native, the same guards ``capture_orthogonalization``
            already keeps.

    Raises:
        ValueError: If ``lr``, ``weight_decay``, ``w_muon``, ``w_sgd`` or
            ``ns_steps`` is out of range, or if ``momentum`` is not in
            ``[0, 1)``. The two gains are checked against NaN explicitly:
            every comparison with NaN is false, so ``w_muon < 0.0`` alone
            admits it, and a NaN gain constructs cleanly and then turns every
            matrix parameter it touches into NaN on the first step (A7).

    Examples:
        ```pycon
        >>> import torch
        >>> weight = torch.nn.Parameter(torch.randn(8, 4))
        >>> bias = torch.nn.Parameter(torch.zeros(8))
        >>> opt = MuSGD([weight, bias], lr=0.01)
        >>> (weight.sum() + bias.sum()).backward()
        >>> opt.step()
        >>> opt.zero_grad()

        ```
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 5e-4,
        w_muon: float = 0.5,
        w_sgd: float = 0.5,
        ns_steps: int = 5,
        *,
        capture_orthogonalization: bool = False,
        capture_updates: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if math.isnan(w_muon) or w_muon < 0.0:
            raise ValueError(f"w_muon must be non-negative, got {w_muon}")
        if math.isnan(w_sgd) or w_sgd < 0.0:
            raise ValueError(f"w_sgd must be non-negative, got {w_sgd}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be positive, got {ns_steps}")
        defaults: dict[str, Any] = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "w_muon": w_muon,
            "w_sgd": w_sgd,
            "ns_steps": ns_steps,
            "capture_orthogonalization": capture_orthogonalization,
            "capture_updates": capture_updates,
        }
        super().__init__(params, defaults)
        self._orthogonalization_replay: _OrthogonalizationReplay | None = None
        self._update_replay: _UpdateReplay | None = None

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step over every parameter group.

        Args:
            closure: Optional callable that re-evaluates the model and returns
                the loss; called under ``torch.enable_grad()`` when provided.

        Returns:
            The loss returned by ``closure``, or ``None`` if no closure is
            given.

        Examples:
            ```pycon
            >>> import torch
            >>> param = torch.nn.Parameter(torch.randn(4, 4))
            >>> opt = MuSGD([param], lr=0.1)
            >>> param.grad = torch.ones_like(param)
            >>> _ = opt.step()

            ```
        """
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("capture_updates", False) and type(self) is MuSGD and not torch.compiler.is_compiling():
                if getattr(self, "_update_replay", None) is None:
                    self._update_replay = _UpdateReplay(_memory_overlaps)
                assert self._update_replay is not None
                if self._update_replay.step(self, group):
                    continue
            if group.get("capture_orthogonalization", False) and not torch.compiler.is_compiling():
                if getattr(self, "_orthogonalization_replay", None) is None:
                    capacity = sum(param.ndim >= 2 for item in self.param_groups for param in item["params"])
                    self._orthogonalization_replay = _OrthogonalizationReplay(capacity)
                assert self._orthogonalization_replay is not None
                with self._orthogonalization_replay.activate():
                    self._step_group(group)
            else:
                self._step_group(group)
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer state and discard recordings from the previous session.

        The replay flags choose how this run executes rather than what it has
        learned, so each group keeps the value it holds now. A checkpoint written with the
        flag on cannot switch replay on for a launch that turned it off, nor the reverse; a
        checkpoint without the key (an older one) needs no separate default for the same reason.
        """
        names = ("capture_orthogonalization", "capture_updates")
        launch_flags = [
            {name: group.get(name, self.defaults.get(name, False)) for name in names} for group in self.param_groups
        ]
        updates = getattr(self, "_update_replay", None)
        if updates is not None:
            updates.clear()
        super().load_state_dict(state_dict)
        for group, flags in zip(self.param_groups, launch_flags, strict=True):
            group.update(flags)
        replay = getattr(self, "_orthogonalization_replay", None)
        if replay is not None:
            replay.clear()

    def _step_group(self, group: dict[str, Any]) -> None:
        """Apply each native update at the point its iterator yields it.

        Keeping the write beside the yield preserves the historical ordering for
        shared storage; only the certified independent CUDA route defers writes.
        """
        lr = group["lr"]
        for param, update in self._group_updates(group):
            param.add_(update, alpha=-lr)

    def _group_updates(self, group: dict[str, Any]) -> Iterator[tuple[Tensor, Tensor]]:
        """Prepare the native updates with CUDA matrices grouped by their smaller matrix dimension.

        For a wide matrix ``X``, appending zero columns preserves its Frobenius norm
        and ``X @ X.T``. Every quintic iterate therefore retains those zero columns;
        removing them recovers the same mathematical update. The scale still uses
        each matrix's original dimensions. This reduces the n detector's 41 exact
        shapes to fewer launch groups without changing the iteration itself.

        ``_padding_batches`` bounds each padded gradient stack by the parameter and
        gradient bytes of the matrices in its own batch. A singleton, or a batch whose
        matrices all share one shape, keeps the original implementation; equal-shape
        matrices that share a smaller dimension with a matrix of another width join
        the padded batch. Parameters whose memory overlaps, shared storage included,
        keep exact-shape keys too: reordering overlapping updates could change
        decoupled decay.
        CPU and MPS retain their existing route. CUDA momentum uses three foreach
        operations when storage is independent; vectors and the zero-Muon arm keep
        their original SGD update formula.
        """
        momentum = group["momentum"]
        matrix_buckets: dict[_BucketKey, list[_PendingUpdate]] = defaultdict(list)
        inspectable = [
            param
            for param in group["params"]
            if type(param) in _STORAGE_TYPES
            and param.layout == torch.strided
            and param.device.type in _BATCHED_DEVICE_TYPES
        ]
        storage_counts = Counter(param.untyped_storage().data_ptr() for param in inspectable)
        independent_storage = len(storage_counts) == len(group["params"]) and not _memory_overlaps(inspectable)
        active = [param for param in group["params"] if param.grad is not None]
        adjusted = self._foreach_nesterov(active, momentum, storage_counts)
        for param in active:
            grad = param.grad
            assert grad is not None
            nesterov_grad = self._nesterov_grad(param, grad, momentum) if adjusted is None else adjusted.popleft()
            if self._is_batchable_matrix(param, group) and param.device.type in _BATCHED_DEVICE_TYPES:
                rows, _ = _matrix_shape(nesterov_grad)
                shape = param.shape
                if (
                    independent_storage
                    and type(param) in _STORAGE_TYPES
                    and type(nesterov_grad) in _STORAGE_TYPES
                    and 0 < rows <= _PADDED_MAX_ROWS
                    and storage_counts[param.untyped_storage().data_ptr()] == 1
                ):
                    shape = torch.Size((rows,))
                key = _BucketKey(shape, param.dtype, nesterov_grad.dtype, param.device)
                matrix_buckets[key].append(_PendingUpdate(param, nesterov_grad))
            else:
                yield param, self._parameter_update(param, nesterov_grad, group)

        for bucket in matrix_buckets.values():
            for batch in _padding_batches(bucket):
                if len(batch) == 1:
                    lone = batch[0]
                    yield lone.param, self._parameter_update(lone.param, lone.nesterov_grad, group)
                elif all(pending.param.shape == batch[0].param.shape for pending in batch):
                    params = torch.stack([pending.param for pending in batch])
                    grads = torch.stack([pending.nesterov_grad for pending in batch])
                    batched = self._batched_parameter_update(params, grads, group)
                    for pending, bucket_update in zip(batch, batched.unbind(), strict=True):
                        yield pending.param, bucket_update
                else:
                    for pending, update in zip(batch, self._padded_parameter_updates(batch, group), strict=True):
                        yield pending.param, update

    def _foreach_nesterov(
        self, params: list[Tensor], momentum: float, storage_counts: Counter[int]
    ) -> deque[Tensor] | None:
        """Batch the same three momentum operations when their storage is independent.

        ``buffer.mul_(momentum).add_(grad)`` stays two rounded operations, followed
        by ``grad.add(buffer, alpha=momentum)``. No interpolation or reassociation
        enters the recurrence. Devices outside ``_BATCHED_DEVICE_TYPES`` (CPU and
        MPS by default) and unusual restored state keep the scalar method; shared
        parameter, gradient or buffer storage does too, because advancing one
        buffer could otherwise change another operation's input.

        The adjusted gradients come back in parameter order as a queue the caller
        pops from, so each one is released once its update has consumed it instead
        of living until the whole group is done.
        """
        on_gated_device = all(
            type(param) in _STORAGE_TYPES and param.device.type in _BATCHED_DEVICE_TYPES for param in params
        )
        if len(params) < 2 or not on_gated_device or any(count > 1 for count in storage_counts.values()):
            return None
        gradients: list[Tensor] = []
        buffers: list[Tensor] = []
        buffer_storage: set[int] = set()
        gradient_storage: set[int] = set()
        buckets: dict[tuple[torch.device, torch.dtype], list[int]] = defaultdict(list)
        for index, param in enumerate(params):
            gradient = param.grad
            if (
                gradient is None
                or type(gradient) not in _STORAGE_TYPES
                or gradient.layout != torch.strided
                or gradient.dtype != param.dtype
            ):
                return None
            state = self.state[param]
            buffer = state.get("momentum_buffer")
            if buffer is None:
                buffer = torch.zeros_like(param)
                state["momentum_buffer"] = buffer
            if (
                type(buffer) not in _STORAGE_TYPES
                or buffer.layout != torch.strided
                or buffer.shape != param.shape
                or buffer.dtype != gradient.dtype
                or buffer.device != gradient.device
            ):
                return None
            pointer = buffer.untyped_storage().data_ptr()
            if pointer in storage_counts or pointer in buffer_storage:
                return None
            buffer_storage.add(pointer)
            gradient_storage.add(gradient.untyped_storage().data_ptr())
            gradients.append(gradient)
            buffers.append(buffer)
            buckets[(gradient.device, gradient.dtype)].append(index)
        if gradient_storage.intersection(buffer_storage) or gradient_storage.intersection(storage_counts):
            return None
        adjusted: dict[int, Tensor] = {}
        for indices in buckets.values():
            group_buffers = [buffers[index] for index in indices]
            group_gradients = [gradients[index] for index in indices]
            torch._foreach_mul_(group_buffers, momentum)
            torch._foreach_add_(group_buffers, group_gradients)
            results = torch._foreach_add(group_gradients, group_buffers, alpha=momentum)
            adjusted.update(zip(indices, results, strict=True))
        return deque(adjusted[index] for index in range(len(params)))

    @staticmethod
    def _padded_parameter_updates(bucket: list[_PendingUpdate], group: dict[str, Any]) -> Iterator[Tensor]:
        """Run the existing quintic over zero-extended, independently normalized matrices.

        Only the common wide orientation is stacked. Each result is cropped back
        before its original ``0.2 * sqrt(max(rows, columns))`` scale, gains and decay
        are applied. Padding is zero work mathematically, but GEMM and reduction
        schedules can differ in floating point; the scalar path remains the oracle.

        The scale, gains and decay here are the third copy of the Muon arm of
        :meth:`_parameter_update`; :meth:`_batched_parameter_update` lists all three copies
        and the tests that hold them together.
        """
        rows = _matrix_shape(bucket[0].nesterov_grad)[0]
        columns = max(_matrix_shape(pending.nesterov_grad)[1] for pending in bucket)
        padded = bucket[0].nesterov_grad.new_zeros((len(bucket), rows, columns))
        for index, pending in enumerate(bucket):
            gradient = pending.nesterov_grad
            matrix = gradient.reshape(gradient.shape[0], -1)
            wide = matrix.T if matrix.shape[0] > matrix.shape[1] else matrix
            padded[index, :, : wide.shape[1]] = wide
        orthogonal = _orthogonalize_batched(padded, steps=group["ns_steps"])
        for index, pending in enumerate(bucket):
            gradient = pending.nesterov_grad
            _, original_columns = _matrix_shape(gradient)
            matrix = orthogonal[index, :, :original_columns]
            if gradient.shape[0] > math.prod(gradient.shape[1:]):
                matrix = matrix.T
            muon = matrix.reshape_as(gradient).mul(_MUON_RMS_SCALE * math.sqrt(original_columns))
            update = muon.mul(group["w_muon"]).add(gradient, alpha=group["w_sgd"])
            if group["weight_decay"] != 0.0:
                update = update.add(pending.param, alpha=group["weight_decay"])
            yield update

    @staticmethod
    def _batched_parameter_update(params: Tensor, grads: Tensor, group: dict[str, Any]) -> Tensor:
        """Evaluate the matrix update over an equal-shape leading batch.

        The scalar path stays the reference for singletons and non-CUDA devices.
        This path only changes how the same Newton--Schulz polynomial is scheduled:
        each product is a native batched product rather than a scalar product inside
        ``vmap``.

        This is the second of three copies of the Muon arm of :meth:`_parameter_update`, kept
        beside it instead of factored out so that the scalar path stays the untouched reference.
        It mirrors :meth:`_muon_branch` (the 2D view and the ``0.2 * sqrt(max(rows, cols))``
        rescale) with :func:`~lucid_yolo.optim.newton_schulz._orthogonalize_batched` in place
        of :func:`~lucid_yolo.optim.newton_schulz.orthogonalize`, then repeats the
        ``w_muon``/``w_sgd`` weighting and the decoupled decay of :meth:`_parameter_update`.
        The third copy is :meth:`_padded_parameter_updates`, which repeats the same scale,
        gains and decay after cropping each padded result. A change to the formula made in one
        place has to be made in all three. Three tests hold the copies together: the first
        compares this path with :meth:`_parameter_update` over 40 optimizer steps, the second
        compares the orthogonalization alone with the scalar loop, the third compares the padded
        path with the scalar formula for mixed widths.

            ``tests/optim/test_musgd.py::TestExactShapeBatching::test_batched_trajectory_matches_unbatched_reference_over_many_steps``

            ``tests/optim/test_newton_schulz.py::TestNativeBatch::test_matches_per_matrix_loop``

            ``tests/optim/test_padded_musgd.py::test_mixed_width_update_matches_scalar_oracle``
        """
        matrices = grads.reshape(grads.shape[0], grads.shape[1], -1)
        rows, cols = matrices.shape[-2:]
        orthogonal = _orthogonalize_batched(matrices, steps=group["ns_steps"])
        muon = orthogonal.reshape_as(grads).mul(_MUON_RMS_SCALE * math.sqrt(max(rows, cols)))
        update = muon.mul(group["w_muon"]).add(grads, alpha=group["w_sgd"])
        if group["weight_decay"] != 0.0:
            update = update.add(params, alpha=group["weight_decay"])
        return update

    @staticmethod
    def _is_batchable_matrix(param: Tensor, group: dict[str, Any]) -> bool:
        """Whether ``param`` reaches the orthogonalizing arm of :meth:`_parameter_update`.

        That arm is the only one carrying a Newton--Schulz product, which makes it
        the only one eligible for a matrix batch -- so "takes the Muon branch" and
        "may join another orthogonalization" are one question with one answer, and
        this is where it is answered. A vector parameter returns before the branch
        and a zero ``w_muon`` takes the arm beside it; neither has a product, so
        neither joins a bucket.

        Both callers read it: :meth:`_step_group` to decide what to stack, and
        :meth:`_parameter_update` to decide which arm to run. Stating the condition
        once is the point. Written out at both call sites it would drift, and the
        two directions of drift are not equally visible -- one silently stops
        batching, the other stacks a parameter whose branch is switched off. The
        native batch raises nothing for that parameter either: it runs the full
        Newton-Schulz iteration and multiplies the result by the zero gain, paying
        for work :meth:`_parameter_update` skips and, should the iteration come out
        non-finite, poisoning the update through ``0 * nan``.

        Args:
            param: The parameter under update.
            group: Its parameter group, read for ``w_muon``.

        Returns:
            ``True`` when ``param`` has rank at least 2 and its group's Muon gain
            is non-zero.
        """
        return param.ndim >= 2 and group["w_muon"] != 0.0

    def _nesterov_grad(self, param: Tensor, grad: Tensor, momentum: float) -> Tensor:
        """Advance the shared momentum buffer and return the Nesterov-adjusted gradient.

        Updates ``m <- momentum*m + g`` in place on the per-parameter buffer and
        returns ``g + momentum*m`` without mutating ``grad`` (A27).
        """
        state = self.state[param]
        buffer = state.get("momentum_buffer")
        if buffer is None:
            buffer = torch.zeros_like(param)
            state["momentum_buffer"] = buffer
        buffer.mul_(momentum).add_(grad)
        return grad.add(buffer, alpha=momentum)

    def _parameter_update(self, param: Tensor, nesterov_grad: Tensor, group: dict[str, Any]) -> Tensor:
        """Return the pre-learning-rate update for ``param`` given its rank.

        Matrix parameters (``ndim >= 2``) get ``w_muon*muon + w_sgd*sgd`` plus
        decoupled weight decay; vector parameters get the SGD update alone.

        ``w_muon == 0`` is the SGD-only arm A7's independent gains admit, and it
        used to pay for the branch it had switched off: every matrix parameter
        still ran its full Newton-Schulz iteration and the result was then
        multiplied by zero. The gain is read here -- through
        :meth:`_is_batchable_matrix`, which owns the condition for both callers --
        rather than inside :meth:`_muon_branch`, because the iteration is reached
        through that one call, and skipping it at the call site is what removes
        the work rather than merely discarding it. The expression inside this
        function remains untouched. Before exact-shape batching, this paragraph
        also promised that a ``w_muon > 0`` run was bit-for-bit the run it was.
        :meth:`_step_group` now batches compatible matrices, padding with zeros when needed, so
        that scheduling promise no longer holds even though the formula does.

        One of the remaining behavioural differences is confined to the arm that
        is switched off: ``0.0 * nan`` is ``nan``, so a non-finite
        orthogonalization used to poison an update whose Muon half carried no
        weight, and now cannot. The zero-gain arm's finite arithmetic is
        unchanged -- ``0 + w_sgd*g`` and ``w_sgd*g`` are the same float.
        """
        if param.ndim < 2:
            return nesterov_grad
        if self._is_batchable_matrix(param, group):
            muon = self._muon_branch(nesterov_grad, group["ns_steps"])
            update = muon.mul(group["w_muon"]).add(nesterov_grad, alpha=group["w_sgd"])
        else:
            update = nesterov_grad.mul(group["w_sgd"])
        weight_decay = group["weight_decay"]
        if weight_decay != 0.0:
            update = update.add(param, alpha=weight_decay)
        return update

    @staticmethod
    def _muon_branch(nesterov_grad: Tensor, ns_steps: int) -> Tensor:
        """Orthogonalize and rescale the matrix view of ``nesterov_grad`` (R7 Equation 4).

        Flattens trailing dimensions to the 2D view ``(A, B)``, orthogonalizes
        it with the Newton-Schulz iteration, rescales by
        ``0.2 * sqrt(max(A, B))``, and reshapes back to the original shape.
        """
        matrix = nesterov_grad.reshape(nesterov_grad.shape[0], -1)
        rows, cols = matrix.shape
        orthogonal = orthogonalize(matrix, steps=ns_steps)
        scaled = orthogonal.mul(_MUON_RMS_SCALE * math.sqrt(max(rows, cols)))
        return scaled.reshape(nesterov_grad.shape)
