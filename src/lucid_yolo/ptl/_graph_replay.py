# SPDX-License-Identifier: Apache-2.0
"""Primitives the three CUDA replay holders share: recording, live gradients, release and lifecycle.

The network, segmentation-head and loss holders each record existing work once with
PyTorch's graphed-callable API and replay it while their own certificates hold. What they
certify differs, so each keeps its own guards and warning texts; this module holds only
the steps that are the same in all of them. A holder that needs one of these names under
its historical module-level name re-binds it there, so ``patch.object(<holder module>, ...)``
keeps reaching the value that holder reads.

Examples:
    A fresh process has no global module hook, so no holder is refused for one:

    ```pycon
    >>> global_hook_breach()
    False

    ```
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

import torch
from packaging.version import Version
from pytorch_lightning import Callback
from torch import Tensor, nn

if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer

#: First verified BF16 replay runtime; 2.4's path does not meet the exact eager oracle.
BF16_REPLAY_SUPPORTED = Version(torch.__version__) >= Version("2.9")

#: The process-wide module hook registries; any entry runs code a recording would not replay.
_GLOBAL_HOOKS = (
    "_global_forward_hooks",
    "_global_forward_pre_hooks",
    "_global_backward_hooks",
    "_global_backward_pre_hooks",
)


def _fp32_policy() -> tuple[object, ...]:
    """Read the float32 matmul and cuDNN policy through the API this torch build exposes.

    Once any ``fp32_precision`` attribute has been set, torch refuses the legacy getters
    (``get_float32_matmul_precision``, ``allow_tf32``) with a :class:`RuntimeError`. A
    build that has the new attributes is therefore read only through them; they also
    reflect values written through the legacy setters.
    """
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    if not hasattr(matmul, "fp32_precision"):
        return torch.get_float32_matmul_precision(), matmul.allow_tf32, cudnn.allow_tf32
    return (
        getattr(torch.backends, "fp32_precision", None),
        matmul.fp32_precision,
        getattr(cudnn, "fp32_precision", None),
        getattr(getattr(cudnn, "conv", None), "fp32_precision", None),
        getattr(getattr(cudnn, "rnn", None), "fp32_precision", None),
    )


def precision_signature() -> tuple[object, ...]:
    """Read the autocast and CUDA numeric policy a recording keeps for its whole lifetime.

    Raises:
        RuntimeError: When torch refuses one of the policy queries.

    Examples:
        The first two fields are the autocast state the BF16 gates read:

        ```pycon
        >>> precision_signature()[:2]
        (False, torch.float16)

        ```
    """
    return (
        torch.is_autocast_enabled(),
        torch.get_autocast_dtype("cuda"),
        *_fp32_policy(),
        torch.backends.cuda.preferred_blas_library(),
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.enabled,
        torch.are_deterministic_algorithms_enabled(),
        torch.get_default_dtype(),
    )


def tensor_signature(tensors: Iterable[Tensor]) -> tuple[object, ...]:
    """Describe the mutable tensor identities, storage and gradient contract without reading values.

    Examples:
        ```pycon
        >>> tensor_signature((torch.zeros(2, 3),))[0][2:]
        ((2, 3), (3, 1), torch.float32, device(type='cpu'), False)

        ```
    """
    return tuple(
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
    )


def global_hook_breach() -> bool:
    """Tell whether a process-wide module hook is registered, which no recording would replay.

    Examples:
        ```pycon
        >>> with nn.modules.module.register_module_forward_hook(lambda *args: None):
        ...     global_hook_breach()
        True

        ```
    """
    runtime = nn.modules.module
    return any(getattr(runtime, name, {}) for name in _GLOBAL_HOOKS)


def module_hook_breach(module: nn.Module) -> bool:
    """Tell whether one module carries a forward, forward-pre, backward or backward-pre hook.

    Examples:
        ```pycon
        >>> layer = nn.Linear(1, 1)
        >>> module_hook_breach(layer)
        False
        >>> handle = layer.register_forward_hook(lambda *args: None)
        >>> module_hook_breach(layer)
        True

        ```
    """
    return bool(
        module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks
    )


def replay_autocast() -> torch.autocast:
    """Re-enter the caller's CUDA autocast state with the cache off, as recording and replay both need.

    Examples:
        ```pycon
        >>> with replay_autocast():
        ...     torch.is_autocast_enabled()
        False

        ```
    """
    return torch.autocast(
        "cuda", enabled=torch.is_autocast_enabled(), dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False
    )


def record_graphed(module: nn.Module, samples: tuple[Tensor, ...], device: torch.device) -> nn.Module:
    """Record ``module`` once on ``samples`` and leave its buffers and the RNG as they were.

    The recorder's warmup runs the module, which advances BatchNorm statistics and may
    draw random numbers; both are restored, so recording alone changes no training state.
    This is the one call through which the network and segmentation-head holders record,
    so a test can replace it to drive their whole lifecycle without CUDA.

    Args:
        module: The module to record; the recorder installs the graphed forward on it.
        samples: Static inputs of the recorded signature, owned by the recording.
        device: The CUDA device the recording runs on.

    Returns:
        The graphed module the recorder returns.

    Examples:
        ```pycon
        >>> layer = nn.Linear(2, 2).cuda()  # needs a CUDA device  # doctest: +SKIP
        >>> sample = torch.zeros(4, 2, device="cuda", requires_grad=True)  # needs a CUDA device  # doctest: +SKIP
        >>> record_graphed(layer, (sample,), sample.device) is layer  # needs a CUDA device  # doctest: +SKIP
        True

        ```
    """
    snapshots = [(buffer, buffer.detach().clone()) for buffer in module.buffers()]
    with torch.random.fork_rng(devices=[device.index]):
        try:
            return cast("nn.Module", torch.cuda.make_graphed_callables(module, samples))
        finally:
            # Warmup runs on a side stream the recorder owns; a failure there can leave
            # BatchNorm writes queued, so drain the device before the restore below
            # copies onto the current stream.
            torch.cuda.synchronize(device)
            with torch.no_grad():
                for buffer, snapshot in snapshots:
                    buffer.copy_(snapshot)


def copy_out_live_gradients(parameters: Iterable[Tensor]) -> None:
    """Move live parameter gradients out of the storage the next replayed backward overwrites.

    A parameter whose ``.grad`` was adopted from a recording's static gradient buffers
    still aliases them, so the next replayed backward would overwrite it and then
    accumulate onto itself. Any ``.grad`` alive before a replay (gradient accumulation, an
    in-place ``zero_grad``, a callback keeping gradients) is copied out first. Lightning's
    automatic optimization clears gradients after the training step's forward, so the
    previous step's gradients are still alive at this point and are copied too, at every
    accumulation setting; the clones are then dropped by ``zero_grad``.

    The caller makes the copy at forward time, before the replay, and holds every later
    replay of the same recording until this one's backward has run. Nothing can then
    alias the static buffers again before that backward writes them. The same copy made
    from a pre-hook inside the replayed backward left every accumulated gradient wrong on
    CUDA (WP-206).

    Examples:
        ```pycon
        >>> weight = nn.Parameter(torch.zeros(2))
        >>> weight.grad = live = torch.ones(2)
        >>> copy_out_live_gradients([weight])
        >>> weight.grad is live, torch.equal(weight.grad, live)
        (False, True)

        ```
    """
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad = parameter.grad.clone()


def drain_device(device: torch.device | None) -> None:
    """Wait for all queued work on a recording's device, for a holder that does not record its streams.

    The whole device is synchronized rather than one stream, so a replay launched from any
    stream has finished before the caller drops its graphs (D6). Nothing runs without a
    device, for a device that is not CUDA, or before CUDA is initialized, so a CPU run never
    touches the CUDA runtime.

    Raises:
        RuntimeError: When the device reports an error while draining.

    Examples:
        ```pycon
        >>> drain_device(None), drain_device(torch.device("cpu"))
        (None, None)

        ```
    """
    if device is None or device.type != "cuda" or not torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        return
    torch.cuda.synchronize(device)


def release_graphs(streams: Iterable[torch.cuda.Stream]) -> None:
    """Wait for every replay queued on ``streams`` before the caller drops its graphs.

    Each distinct stream is synchronized once, in first-seen order. The caller keeps its
    recordings referenced until this returns; dropping a graph's private pool while a
    replay on it may still be queued is the hazard this rules out (D6). It runs only when
    a holder is cleared, never per step.

    Raises:
        RuntimeError: When a stream reports a device error while draining.

    Examples:
        ```pycon
        >>> class Stream:
        ...     def synchronize(self):
        ...         print("drained")
        >>> stream = Stream()
        >>> release_graphs([stream, stream])
        drained

        ```
    """
    for stream in dict.fromkeys(streams):
        stream.synchronize()


class _ReplayHolder(Protocol):
    """The two calls the fit lifecycle makes on any replay holder."""

    def clear(self) -> None:
        """Drain and release every recording and restart the next fit's eligibility decision."""

    def start_batch(self) -> None:
        """Drop holds whose backward never ran, so the next training batch may replay again."""


class ReplayLifecycle(Callback):
    """Bound one replay holder's recordings to a fit, through skipped backwards and failure.

    Subclass hook: set :attr:`attribute`, :attr:`holder_type` and :attr:`flag`. A module
    without that attribute, or holding something other than :attr:`holder_type`, makes
    every hook a no-op, so a trainer reused on another module class does not fail here.
    """

    #: Name of the module attribute that owns the holder.
    attribute: ClassVar[str]
    #: The holder class the attribute must hold.
    holder_type: ClassVar[type[_ReplayHolder]]
    #: Public option name the holder's warnings start with, e.g. ``"capture_network"``.
    flag: ClassVar[str]

    def _holder(self, module: LightningModule) -> _ReplayHolder | None:
        holder = getattr(module, self.attribute, None)
        return holder if isinstance(holder, self.holder_type) else None

    def _clear(self, module: LightningModule) -> None:
        holder = self._holder(module)
        if holder is not None:
            holder.clear()

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: object, batch_idx: int) -> None:
        """Let each training batch replay even when the previous one skipped its backward."""
        del trainer, batch, batch_idx
        holder = self._holder(pl_module)
        if holder is not None:
            holder.start_batch()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start the fit without recordings from an earlier model or trainer."""
        del trainer
        self._clear(pl_module)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Drain and release the fit's recordings after the last training step; a failing drain is raised."""
        del trainer
        self._clear(pl_module)

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        """Release recordings when the trainer fails, without replacing its exception or skipping teardown.

        A failed CUDA step usually leaves the device in an error state that the drain raises again.
        Raising it here would hide the trainer's own exception and skip the trainer's teardown, so a
        drain failure becomes a warning; the holder is already empty by then.
        """
        del trainer, exception
        try:
            self._clear(pl_module)
        except RuntimeError as error:
            # Warning filters must not replace the trainer's original exception.
            with contextlib.suppress(UserWarning):
                warnings.warn(
                    f"{self.flag}=True could not drain its recordings after the fit failed ({error})",
                    UserWarning,
                    stacklevel=2,
                )
