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
from collections import defaultdict
from typing import TYPE_CHECKING, Any, NamedTuple, overload

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from lucid_yolo.optim.newton_schulz import _orthogonalize_batched, orthogonalize
from lucid_yolo.optim.replay import _OrthogonalizationReplay

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.optim.optimizer import ParamsT

#: Update-RMS scaling constant for the Muon branch (R7 Eq. 4: ``0.2 * O_t``).
_MUON_RMS_SCALE: float = 0.2

#: Device types on which :meth:`MuSGD._step_group` stacks equal-shape matrices and
#: updates each stack with one native batched call instead of one call per parameter.
#: Batching trades many small kernel launches for one large one, so it pays exactly
#: where launch overhead dominates. The gate was set on the production n-scale detector
#: while the stack was updated through :func:`torch.vmap`: CUDA (L4) is where the win was
#: found, while MPS ran that vmap path at 108.45 ms per optimizer step against 60.99 ms
#: for the per-parameter loop and CPU moved 39.64 ms to 38.47 ms. The native batch that
#: replaced it (WP-193) was measured on CUDA (L4) and not re-measured on MPS or CPU, so
#: the CUDA-only gate is inherited rather than re-derived. A device outside this set
#: keeps the loop, which is also the arithmetic the tests pin.
_BATCHED_DEVICE_TYPES: frozenset[str] = frozenset({"cuda"})


class _BucketKey(NamedTuple):
    """The identity two matrix parameters must share to be stacked into one batch.

    Every operation inside :meth:`MuSGD._batched_parameter_update` is elementwise
    or acts on the final two matrix axes through a native batched product, so a
    leading batch axis cannot mix parameters that agree on these four fields.

    The gradient dtype is one of the four because :func:`torch.stack` is applied
    to the Nesterov-adjusted gradients as well as to the parameters, and those
    two dtypes are not the same field. :meth:`MuSGD._nesterov_grad` returns
    ``grad.add(buffer)``, which promotes, so a parameter whose ``.grad`` carries
    a wider dtype than itself -- reachable once ``param.grad_dtype`` is relaxed
    from its default, which is what a mixed-precision setup does -- yields a
    wider Nesterov gradient than its same-shape, same-dtype neighbour. Stacking
    the two promotes silently rather than raising, and the neighbour would then
    be computed at a precision its own path never chose. Keying on it puts them
    in separate buckets instead, which is the answer the per-parameter path gives
    anyway.
    """

    shape: torch.Size
    param_dtype: torch.dtype
    grad_dtype: torch.dtype
    device: torch.device


class _PendingUpdate(NamedTuple):
    """A parameter and the Nesterov-adjusted gradient its update has yet to be made from."""

    param: Tensor
    nesterov_grad: Tensor


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
            autocast calls stay eager. No momentum or parameter update is captured.

    Raises:
        ValueError: If ``lr``, ``weight_decay``, ``w_muon``, ``w_sgd`` or
            ``ns_steps`` is out of range, or if ``momentum`` is not in
            ``[0, 1)``. The two gains are checked against NaN explicitly:
            every comparison with NaN is false, so ``w_muon < 0.0`` alone
            admits it, and a NaN gain constructs cleanly and then turns every
            matrix parameter it touches into NaN on the first step (A7).

    Examples:
        >>> import torch
        >>> weight = torch.nn.Parameter(torch.randn(8, 4))
        >>> bias = torch.nn.Parameter(torch.zeros(8))
        >>> opt = MuSGD([weight, bias], lr=0.01)
        >>> (weight.sum() + bias.sum()).backward()
        >>> opt.step()
        >>> opt.zero_grad()
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
        }
        super().__init__(params, defaults)
        self._orthogonalization_replay: _OrthogonalizationReplay | None = None

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
            >>> import torch
            >>> param = torch.nn.Parameter(torch.randn(4, 4))
            >>> opt = MuSGD([param], lr=0.1)
            >>> param.grad = torch.ones_like(param)
            >>> _ = opt.step()
        """
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
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
        """Restore optimizer state and discard recordings from the previous session."""
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            group.setdefault("capture_orthogonalization", self.defaults.get("capture_orthogonalization", False))
        replay = getattr(self, "_orthogonalization_replay", None)
        if replay is not None:
            replay.clear()

    def _step_group(self, group: dict[str, Any]) -> None:
        """Apply one MuSGD update, batching matrix parameters with the same exact shape.

        The n-scale detector carries 127 matrix parameters but only 41 exact
        :class:`_BucketKey` values. Native batched matrix products once per key
        replace the :func:`torch.vmap` layer, which had already lowered each product
        to one batched multiply, and fold the scale-and-add into
        :func:`torch.baddbmm`, while keeping the same normalization, requested iteration count, branch
        weights and decay.
        Vector parameters and the ``w_muon == 0`` arm stay on their scalar path;
        :meth:`_is_batchable_matrix` is what says which parameters those are, and
        it says it for :meth:`_parameter_update` too.

        The batched path is taken only where it pays (:data:`_BATCHED_DEVICE_TYPES`). One
        large kernel launch in place of many is a win where launch overhead
        dominates the step and a loss where it does not, and the measurement says
        which is which per backend rather than per shape: MPS ran the
        :func:`torch.vmap` path this native batch replaced at 108.45 ms per optimizer
        step against 60.99 ms for the per-parameter loop. The native batch has not
        been re-measured on MPS or CPU, so the CUDA-only gate is inherited from that
        measurement. Every parameter on a device outside that set keeps the loop.

        A bucket holding one parameter is called directly rather than batched. There
        is nothing to combine, the stack-batch-unbind round trip costs more than the
        call it wraps, and the direct call is the one that stays bit-for-bit what an
        unbatched run produced -- a single-element batch ``(4, 21, 1, 1)`` differed from it
        by 4.77e-7. Ten of the n-scale detector's buckets hold one parameter.

        Each update is applied where it resolves rather than collected for a final
        pass. Every update reads its own parameter and nothing else, so the order is
        free, while holding one update per parameter alive alongside one gradient per
        parameter is a transient peak that grows with the model.
        """
        momentum = group["momentum"]
        lr = group["lr"]
        matrix_buckets: dict[_BucketKey, list[_PendingUpdate]] = defaultdict(list)
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            nesterov_grad = self._nesterov_grad(param, grad, momentum)
            if self._is_batchable_matrix(param, group) and param.device.type in _BATCHED_DEVICE_TYPES:
                key = _BucketKey(param.shape, param.dtype, nesterov_grad.dtype, param.device)
                matrix_buckets[key].append(_PendingUpdate(param, nesterov_grad))
            else:
                param.add_(self._parameter_update(param, nesterov_grad, group), alpha=-lr)

        for bucket in matrix_buckets.values():
            if len(bucket) == 1:
                lone = bucket[0]
                lone.param.add_(self._parameter_update(lone.param, lone.nesterov_grad, group), alpha=-lr)
                continue
            params = torch.stack([pending.param for pending in bucket])
            grads = torch.stack([pending.nesterov_grad for pending in bucket])
            batched = self._batched_parameter_update(params, grads, group)
            for pending, bucket_update in zip(bucket, batched.unbind(), strict=True):
                pending.param.add_(bucket_update, alpha=-lr)

    @staticmethod
    def _batched_parameter_update(params: Tensor, grads: Tensor, group: dict[str, Any]) -> Tensor:
        """Evaluate the matrix update over an equal-shape leading batch.

        The scalar path stays the reference for singletons and non-CUDA devices.
        This path only changes how the same Newton--Schulz polynomial is scheduled:
        each product is a native batched product rather than a scalar product inside
        ``vmap``.

        This is a second copy of the Muon arm of :meth:`_parameter_update`, kept beside
        it instead of factored out so that the scalar path stays the untouched reference.
        It mirrors :meth:`_muon_branch` (the 2D view and the ``0.2 * sqrt(max(rows, cols))``
        rescale) with :func:`~lucid_yolo.optim.newton_schulz._orthogonalize_batched` in place
        of :func:`~lucid_yolo.optim.newton_schulz.orthogonalize`, then repeats the
        ``w_muon``/``w_sgd`` weighting and the decoupled decay of :meth:`_parameter_update`.
        A change to the formula made in one place has to be made in the other. Two tests hold
        the copies together: the first compares this path with :meth:`_parameter_update` over
        40 optimizer steps, the second compares the orthogonalization alone with the scalar loop.

            ``tests/optim/test_musgd.py::TestExactShapeBatching::test_batched_trajectory_matches_unbatched_reference_over_many_steps``

            ``tests/optim/test_newton_schulz.py::TestNativeBatch::test_matches_per_matrix_loop``
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
        the only one with anything to batch -- so "takes the Muon branch" and "may
        be stacked with another parameter" are one question with one answer, and
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
        :meth:`_step_group` now batches the expression over equal-shape matrices, so
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
