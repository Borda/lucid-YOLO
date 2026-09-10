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
from typing import TYPE_CHECKING, Any, overload

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from lucid_yolo.optim.newton_schulz import orthogonalize

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.optim.optimizer import ParamsT

#: Update-RMS scaling constant for the Muon branch (R7 Eq. 4: ``0.2 * O_t``).
_MUON_RMS_SCALE: float = 0.2


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
        }
        super().__init__(params, defaults)

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
            self._step_group(group)
        return loss

    def _step_group(self, group: dict[str, Any]) -> None:
        """Apply one MuSGD update, batching matrix parameters with the same exact shape.

        The n-scale detector carries 127 matrix parameters but only 20 exact
        ``(shape, dtype, device)`` keys. Calling :meth:`_parameter_update` through
        :func:`torch.vmap` once per key turns each Newton--Schulz product into a
        batched product, while the function inside the map keeps the same
        normalization, requested iteration count, branch weights and decay.
        Vector parameters and the ``w_muon == 0`` arm stay on their scalar path;
        neither has an orthogonalization to combine.
        """
        momentum = group["momentum"]
        lr = group["lr"]
        active: list[tuple[Tensor, Tensor]] = []
        updates: list[Tensor | None] = []
        matrix_buckets: dict[tuple[torch.Size, torch.dtype, torch.device], list[int]] = defaultdict(list)
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            nesterov_grad = self._nesterov_grad(param, grad, momentum)
            index = len(active)
            active.append((param, nesterov_grad))
            if param.ndim < 2 or group["w_muon"] == 0.0:
                updates.append(self._parameter_update(param, nesterov_grad, group))
            else:
                updates.append(None)
                matrix_buckets[(param.shape, param.dtype, param.device)].append(index)

        for indices in matrix_buckets.values():
            params = torch.stack([active[index][0] for index in indices])
            grads = torch.stack([active[index][1] for index in indices])
            batched = torch.vmap(lambda param, grad: self._parameter_update(param, grad, group))(params, grads)
            for index, bucket_update in zip(indices, batched.unbind(), strict=True):
                updates[index] = bucket_update

        for (param, _), parameter_update in zip(active, updates, strict=True):
            if parameter_update is None:  # pragma: no cover - every matrix bucket is filled above
                raise RuntimeError("missing MuSGD matrix update")
            param.add_(parameter_update, alpha=-lr)

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
        multiplied by zero. The gain is read here rather than inside
        :meth:`_muon_branch` because the iteration is reached through that one
        call, and skipping it at the call site is what removes the work rather
        than merely discarding it. The expression inside this function remains
        untouched. Before exact-shape batching, this paragraph also promised that
        a ``w_muon > 0`` run was bit-for-bit the run it was. :meth:`_step_group`
        now maps the expression over equal-shape matrices, so that scheduling
        promise no longer holds even though the formula does.

        The one behavioural difference is confined to the arm that is switched
        off: ``0.0 * nan`` is ``nan``, so a non-finite orthogonalization used to
        poison an update whose Muon half carried no weight, and now cannot. The
        zero-gain arm's finite arithmetic is unchanged -- ``0 + w_sgd*g`` and
        ``w_sgd*g`` are the same float.
        """
        if param.ndim < 2:
            return nesterov_grad
        if group["w_muon"] == 0.0:
            update = nesterov_grad.mul(group["w_sgd"])
        else:
            muon = self._muon_branch(nesterov_grad, group["ns_steps"])
            update = muon.mul(group["w_muon"]).add(nesterov_grad, alpha=group["w_sgd"])
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
