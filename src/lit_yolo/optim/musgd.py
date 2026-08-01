# SPDX-License-Identifier: Apache-2.0
"""MuSGD -- the hybrid Muon/SGD optimizer for the detector backbone (WP-032).

MuSGD (R1 sec. 3.3.1) combines two updates per parameter, split by parameter
rank:

* **Matrix parameters** (``ndim >= 2`` -- conv kernels and linear weights) get
  a weighted sum of a Muon-style update and an SGD-momentum update. The Muon
  half orthogonalizes the momentum-adjusted gradient with the Newton-Schulz
  iteration (:func:`lit_yolo.optim.orthogonalize`) and rescales it by
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
from typing import TYPE_CHECKING, Any, overload

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from lit_yolo.optim.newton_schulz import orthogonalize

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
        w_muon: Additive gain on the Muon branch. Defaults to 0.5.
        w_sgd: Additive gain on the SGD branch. Defaults to 0.5.
        ns_steps: Newton-Schulz iteration count for the orthogonalization
            (A5). Must be positive. Defaults to 5.

    Raises:
        ValueError: If ``lr``, ``weight_decay``, or ``ns_steps`` is out of
            range, or if ``momentum`` is not in ``[0, 1)``.

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
        """Apply the MuSGD update to every parameter with a gradient in ``group``."""
        momentum = group["momentum"]
        lr = group["lr"]
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            nesterov_grad = self._nesterov_grad(param, grad, momentum)
            update = self._parameter_update(param, nesterov_grad, group)
            param.add_(update, alpha=-lr)

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
        """
        if param.ndim < 2:
            return nesterov_grad
        muon = self._muon_branch(nesterov_grad, group["ns_steps"])
        update = muon.mul(group["w_muon"]).add(nesterov_grad, alpha=group["w_sgd"])
        weight_decay = group["weight_decay"]
        if weight_decay != 0.0:
            update = update.add(param, alpha=weight_decay)
        return update

    @staticmethod
    def _muon_branch(nesterov_grad: Tensor, ns_steps: int) -> Tensor:
        """Orthogonalize and rescale the matrix view of ``nesterov_grad`` (R7 Eq. 4).

        Flattens trailing dimensions to the 2D view ``(A, B)``, orthogonalizes
        it with the Newton-Schulz iteration, rescales by
        ``0.2 * sqrt(max(A, B))``, and reshapes back to the original shape.
        """
        matrix = nesterov_grad.reshape(nesterov_grad.shape[0], -1)
        rows, cols = matrix.shape
        orthogonal = orthogonalize(matrix, steps=ns_steps)
        scaled = orthogonal.mul(_MUON_RMS_SCALE * math.sqrt(max(rows, cols)))
        return scaled.reshape(nesterov_grad.shape)
