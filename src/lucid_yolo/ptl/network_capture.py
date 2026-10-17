# SPDX-License-Identifier: Apache-2.0
"""Opt-in replay of the detection network's CUDA forward and backward work.

The network has a fixed image signature while its ground-truth count changes per
batch. Recording only backbone, neck and head leaves assignment, losses, schedules,
optimizer and EMA in their existing loop. PyTorch's graphed-callable API owns the
autograd bridge; no network arithmetic is rewritten. The recording is transient,
bounded to one input signature and discarded at the fit boundary.

Examples:
    Copying the transient holder starts with no recording, even without CUDA:

    ```pycon
    >>> import copy
    >>> holder = _NetworkCapture()
    >>> copy.deepcopy(holder)._network is None
    True

    ```
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import torch
from packaging.version import Version
from pytorch_lightning import Callback
from torch import Tensor, nn

from lucid_yolo.models.heads.detect import DualHeadOutput

#: First verified BF16 replay runtime; 2.4's path does not meet the exact eager oracle.
_BF16_REPLAY_SUPPORTED = Version(torch.__version__) >= Version("2.9")

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytorch_lightning import LightningModule, Trainer


class _DetectionNetwork(nn.Module):
    """Expose the existing detection stages as the tensor tuple the recorder accepts."""

    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone, self.neck, self.head = backbone, neck, head

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the four detection tensors without changing stage composition."""
        output = cast("DualHeadOutput", self.head(self.neck(self.backbone(images))))
        return output.o2m_cls, output.o2m_box, output.o2o_cls, output.o2o_box


def _precision_signature() -> tuple[object, ...] | None:
    """Read the active CUDA policy, declining legacy queries the new precision API refuses."""
    try:
        return (
            torch.is_autocast_enabled(),
            torch.get_autocast_dtype("cuda"),
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.enabled,
            torch.are_deterministic_algorithms_enabled(),
            torch.get_default_dtype(),
        )
    except RuntimeError:
        return None


def _network_signature(network: _DetectionNetwork) -> tuple[object, ...] | None:
    """Certify the storage, modes and hook-free stages a recording will keep reading."""
    runtime = nn.modules.module
    if any(
        getattr(runtime, name, {})
        for name in (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
        )
    ):
        return None
    modules = tuple(network.modules())
    if any(
        module._forward_hooks
        or module._forward_pre_hooks
        or module._backward_hooks
        or module._backward_pre_hooks
        or (module is not network and "forward" in vars(module))
        or getattr(module, "_compiled_call_impl", None) is not None
        for module in modules
    ):
        return None
    params = tuple(network.parameters())
    buffers = tuple(network.buffers())
    if not params or any(param._backward_hooks for param in params) or any(buffer.requires_grad for buffer in buffers):
        return None
    tensors = params + buffers
    if any(
        type(tensor) not in (Tensor, nn.Parameter) or not tensor.is_cuda or tensor.layout != torch.strided
        for tensor in tensors
    ):
        return None
    return (
        tuple(
            (
                id(module),
                id(type(module).forward),
                module.training,
                tuple(
                    (name, value)
                    for name, value in vars(module).items()
                    if value is None
                    or isinstance(value, (bool, int, float, str))
                    or (isinstance(value, tuple) and all(isinstance(part, (bool, int, float, str)) for part in value))
                ),
            )
            for module in modules
        ),
        tuple(
            (
                id(tensor),
                tensor.data_ptr(),
                tuple(tensor.shape),
                tensor.stride(),
                tensor.dtype,
                tensor.device,
                tensor.requires_grad,
            )
            for tensor in tensors
        ),
    )


class _NetworkCapture:
    """Own one fit's recording without registering another copy of the model's parameters.

    The holder is an ordinary Python object, so Lightning checkpoints keep the original
    parameter names. Pickle and deepcopy discard its recording too. An input-size change
    takes the eager route; a change to model storage, modes, hooks or precision retires
    the recording for the rest of the fit instead of replaying stale work. Anomaly
    detection stays eager so its backward NaN checks remain observable.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Release the recording and start a fresh fit's eligibility decision."""
        previous = getattr(self, "_network", None)
        if previous is not None:
            vars(previous).pop("forward", None)
        self._network: _DetectionNetwork | None = None
        self._input: tuple[object, ...] | None = None
        self._state: tuple[object, ...] | None = None
        self._owners: tuple[int, ...] | None = None
        self._disabled = False

    def __getstate__(self) -> dict[str, object]:
        """Leave CUDA recordings out of pickle and deepcopy."""
        return {}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore an empty holder that may record in its next fit."""
        del state
        self.clear()

    def retire(self) -> None:
        """Retire a recording whose static execution contract no longer holds."""
        if self._disabled:
            return
        self.clear()
        self._disabled = True
        warnings.warn(
            "capture_network=True stays eager: network storage, modes, hooks, precision or anomaly detection "
            "do not match the recording contract",
            UserWarning,
            stacklevel=3,
        )

    def run(
        self,
        images: Tensor,
        stages: tuple[nn.Module, nn.Module, nn.Module],
        eager: Callable[[Tensor], DualHeadOutput],
        *,
        preserve_gradients: bool = False,
    ) -> DualHeadOutput:
        """Replay an eligible training network, leaving other image signatures eager."""
        if self._disabled:
            return eager(images)
        precision = _precision_signature()
        if precision is None or images.requires_grad or torch.is_anomaly_enabled():
            self.retire()
            return eager(images)
        if not _BF16_REPLAY_SUPPORTED and (
            images.dtype == torch.bfloat16 or (precision[0] and precision[1] == torch.bfloat16)
        ):
            self.retire()
            return eager(images)
        owners = tuple(id(stage) for stage in stages)
        network = self._network if self._network is not None else _DetectionNetwork(*stages)
        state = _network_signature(network)
        if state is None or (self._state is not None and (state != self._state or owners != self._owners)):
            self.retire()
            return eager(images)
        signature = tuple(images.shape), images.stride(), images.dtype, images.device, precision
        if self._input is not None and signature != self._input:
            if signature[-1] != self._input[-1]:
                self.retire()
            return eager(images)
        with torch.autocast(
            "cuda", enabled=torch.is_autocast_enabled(), dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False
        ):
            if self._network is None:
                snapshots = [(buffer, buffer.detach().clone()) for buffer in network.buffers()]
                with torch.random.fork_rng(devices=[images.device.index]):
                    try:
                        sample = images.detach().clone()
                        graphed = cast("_DetectionNetwork", torch.cuda.make_graphed_callables(network, (sample,)))
                        torch.cuda.synchronize(images.device)
                    finally:
                        with torch.no_grad():
                            for buffer, snapshot in snapshots:
                                buffer.copy_(snapshot)
                self._network, self._input, self._state, self._owners = graphed, signature, state, owners
            if preserve_gradients:
                # Graphed backward reuses its gradient storage. Keep a live accumulated
                # gradient out of that storage before the next backward overwrites it.
                for parameter in network.parameters():
                    if parameter.grad is not None:
                        parameter.grad = parameter.grad.clone()
            return DualHeadOutput(*self._network(images))


class _NetworkCaptureLifecycle(Callback):
    """Discard transient recordings on fit entry, completion and failure."""

    @staticmethod
    def _clear(module: LightningModule) -> None:
        holder = getattr(module, "_network_capture", None)
        if isinstance(holder, _NetworkCapture):
            holder.clear()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start the fit without recordings from an earlier run."""
        del trainer
        self._clear(pl_module)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Release the fit's graph pool after the last training step."""
        del trainer
        self._clear(pl_module)

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        """Release recordings when the trainer aborts."""
        del trainer, exception
        self._clear(pl_module)
