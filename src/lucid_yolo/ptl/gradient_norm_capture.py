# SPDX-License-Identifier: Apache-2.0
"""Opt-in replay of PyTorch's global L2 gradient norm on private CUDA inputs.

Only the reduction is recorded. Gradient scaling stays in the dependency's native
``clip_grads_with_norm_`` call, so aliases, duplicate parameters and changing clipping
limits keep their original update semantics. No model or optimizer arithmetic is rewritten.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
from packaging.version import Version
from pytorch_lightning import Callback
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pytorch_lightning import LightningModule, Trainer

_SUPPORTED = Version(torch.__version__) >= Version("2.9")
_GET_NORM = getattr(torch.nn.utils, "get_total_norm", None)
_SCALE = getattr(torch.nn.utils, "clip_grads_with_norm_", None)
_CLIP = torch.nn.utils.clip_grad_norm_


class _GradientNormCapture:
    """Own one reduction signature per fit and retire on incompatible inputs.

    Private inputs are overwritten before every replay. A norm is consumed by native
    scaling before the next call can overwrite it; it is never exposed to a caller.
    Reset and serialization drop all recordings and retained tensors.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Release this fit's graph and its private inputs."""
        self._graph: torch.cuda.CUDAGraph | None = None
        self._inputs: list[Tensor] = []
        self._norm: Tensor | None = None
        self._signature: tuple[object, ...] | None = None
        self._disabled = False

    def __getstate__(self) -> dict[str, object]:
        """Exclude device resources from copying and checkpoints."""
        return {}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore an owner that may record in a new fit."""
        del state
        self.clear()

    def retire(self, reason: str) -> None:
        """Keep subsequent calls native and issue one actionable diagnostic."""
        if not self._disabled:
            self.clear()
            self._disabled = True
            warnings.warn(f"capture_gradient_norm=True stays eager: {reason}", UserWarning, stacklevel=3)

    def run(self, parameters: Iterable[Tensor], maximum: float, eager: Callable[[], None]) -> None:
        """Replay a certified reduction and apply the current limit through native scaling."""
        if self._disabled:
            eager()
            return
        params = list(parameters)
        grads = [grad for parameter in params if (grad := parameter.grad) is not None]
        reason = None
        if not _SUPPORTED:
            reason = "the norm decomposition requires torch>=2.9"
        elif (
            torch.nn.utils.clip_grad_norm_ is not _CLIP
            or getattr(torch.nn.utils, "get_total_norm", None) is not _GET_NORM
            or getattr(torch.nn.utils, "clip_grads_with_norm_", None) is not _SCALE
        ):
            reason = "a PyTorch clipping implementation was customized"
        elif not grads:
            reason = "no parameter has a gradient"
        elif any(
            type(grad) is not Tensor
            or not grad.is_cuda
            or grad.layout != torch.strided
            or grad.storage_offset() != 0
            or grad.is_conj()
            or grad.is_neg()
            or not (grad.is_contiguous() or (grad.ndim == 4 and grad.is_contiguous(memory_format=torch.channels_last)))
            for grad in grads
        ):
            reason = "gradients require ordinary dense zero-offset contiguous or channels-last CUDA tensors"
        elif any(grad.device != grads[0].device for grad in grads):
            reason = "gradients span more than one device"
        if reason is not None:
            self.retire(reason)
            eager()
            return
        signature = (
            tuple((tuple(grad.shape), grad.stride(), grad.dtype, grad.device) for grad in grads),
            torch.cuda.current_stream(grads[0].device).cuda_stream,
            torch.is_autocast_enabled(),
            torch.get_autocast_dtype("cuda"),
            torch.are_deterministic_algorithms_enabled(),
        )
        if self._signature is not None and signature != self._signature:
            self.retire("the gradient signature, precision policy or submission stream changed")
            eager()
            return
        assert _GET_NORM is not None and _SCALE is not None
        with torch.no_grad():
            if self._graph is None:
                inputs = [grad.detach().clone() for grad in grads]
                stream = torch.cuda.Stream(device=grads[0].device)  # type: ignore[no-untyped-call]
                stream.wait_stream(torch.cuda.current_stream(grads[0].device))
                try:
                    with torch.cuda.stream(stream):
                        _GET_NORM(inputs)
                finally:
                    # Inputs were allocated on the caller's stream. Order reuse there
                    # after every warmup read, including a partially submitted failure.
                    torch.cuda.current_stream(grads[0].device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        norm = _GET_NORM(inputs)
                except Exception:
                    self.clear()
                    raise
                self._inputs, self._graph, self._norm, self._signature = inputs, graph, norm, signature
            torch._foreach_copy_(self._inputs, grads)
            assert self._graph is not None
            self._graph.replay()
            assert self._norm is not None
            _SCALE(params, maximum, self._norm)


class _GradientNormLifecycle(Callback):
    """Clear gradient-norm recording at each successful or failed fit boundary."""

    @staticmethod
    def _clear(module: LightningModule) -> None:
        """Clear this model's owner, or ignore a callback retained for another model."""
        owner = getattr(module, "_gradient_norm_capture", None)
        if isinstance(owner, _GradientNormCapture):
            owner.clear()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start a new bounded recording lifetime."""
        del trainer
        self._clear(pl_module)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Release successful-fit resources."""
        del trainer
        self._clear(pl_module)

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        """Release resources when a fit fails after recording."""
        del trainer, exception
        self._clear(pl_module)
