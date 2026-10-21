# SPDX-License-Identifier: Apache-2.0
"""Opt-in replay of fixed network stages' CUDA forward and backward work.

The network has a fixed image signature while its ground-truth count changes per
batch. Detection and oriented detection record backbone, neck and head; pose records
backbone, neck and each prediction stem separately. All three leave assignment, losses,
schedules, optimizer and EMA in their existing loop. PyTorch's graphed-callable API owns the
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
from typing import TYPE_CHECKING, TypeVar, cast

import torch
from packaging.version import Version
from pytorch_lightning import Callback
from torch import Tensor, nn

from lucid_yolo.models.heads.detect import DualHeadOutput, _DetectionBranch

_Output = TypeVar("_Output")
_Features = tuple[Tensor, Tensor, Tensor]

#: First verified BF16 replay runtime; 2.4's path does not meet the exact eager oracle.
_BF16_REPLAY_SUPPORTED = Version(torch.__version__) >= Version("2.9")

#: Method names through which a module's call reaches its arithmetic; the same name set on
#: an instance shadows the class method a recording replays or a certificate fingerprints.
_DISPATCH = ("forward", "forward_branch", "_keypoints", "_base_predictions")

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pytorch_lightning import LightningModule, Trainer


def _detection_stages(backbone: nn.Module, neck: nn.Module, head: nn.Module, images: Tensor) -> DualHeadOutput:
    """Run backbone, neck and dual head in the one order eager and replayed forwards share."""
    return cast("DualHeadOutput", head(neck(backbone(images))))


class _DetectionNetwork(nn.Module):
    """Expose the existing detection stages as the tensor tuple the recorder accepts."""

    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone, self.neck, self.head = backbone, neck, head

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the four detection tensors of the shared stage composition, in rebuild order."""
        output = _detection_stages(self.backbone, self.neck, self.head, images)
        return output.o2m_cls, output.o2m_box, output.o2o_cls, output.o2o_box

    @staticmethod
    def rebuild(tensors: tuple[Tensor, ...]) -> DualHeadOutput:
        """Name the replayed tensors by field, in the order :meth:`forward` returns them."""
        o2m_cls, o2m_box, o2o_cls, o2o_box = tensors
        return DualHeadOutput(o2m_cls=o2m_cls, o2m_box=o2m_box, o2o_cls=o2o_cls, o2o_box=o2o_box)


class _OrientedNetwork(nn.Module):
    """Expose both native angle tensors beside the four dense detection fields.

    Examples:
        ```pycon
        >>> from lucid_yolo.ptl import DetectionLitModule
        >>> model = DetectionLitModule(0.34, 0.25, 64, 2, task="obb").eval()
        >>> network = _OrientedNetwork(model.backbone, model.neck, model.head)
        >>> with torch.no_grad():
        ...     tensors = network(torch.zeros(2, 3, 64, 64))
        >>> len(tensors), network.rebuild(tensors).o2m_angle.shape
        (6, torch.Size([2, 84, 1]))

        ```
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone, self.neck, self.head = backbone, neck, head

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run the existing stages and retain the six tensors the native oriented objective reads."""
        output = _detection_stages(self.backbone, self.neck, self.head, images)
        assert output.o2m_angle is not None and output.o2o_angle is not None
        return (
            output.o2m_cls,
            output.o2m_box,
            output.o2o_cls,
            output.o2o_box,
            output.o2m_angle,
            output.o2o_angle,
        )

    @staticmethod
    def rebuild(tensors: tuple[Tensor, ...]) -> DualHeadOutput:
        """Rebuild every recorded field by name, leaving absent modalities absent."""
        o2m_cls, o2m_box, o2o_cls, o2o_box, o2m_angle, o2o_angle = tensors
        return DualHeadOutput(
            o2m_cls=o2m_cls,
            o2m_box=o2m_box,
            o2o_cls=o2o_cls,
            o2o_box=o2o_box,
            o2m_angle=o2m_angle,
            o2o_angle=o2o_angle,
        )


class _FeatureNetwork(nn.Module):
    """Expose backbone and neck without fixing which optional head gradients exist.

    A pose batch may have no assigned positives. Its point stems then keep
    ``grad=None``; including those stems in one fixed backward would materialize
    zero gradients and advance optimizer state that the native loop skips.
    The shared features still feed the dense classification objective, while
    the stems have independent autograd nodes and variable supervision stays native.
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module) -> None:
        super().__init__()
        self.backbone, self.neck = backbone, neck

    def forward(self, images: Tensor) -> _Features:
        """Return the three existing neck features in their original order."""
        return cast("_Features", self.neck(self.backbone(images)))

    @staticmethod
    def rebuild(tensors: tuple[Tensor, ...]) -> _Features:
        """Keep the three feature levels separate for the native dual head."""
        first, second, third = tensors
        return first, second, third


class _StemNetwork(nn.Module):
    """Keep one native stem's backward separate from conditionally unused stems."""

    def __init__(self, stem: nn.Module) -> None:
        super().__init__()
        self.stem = stem

    def forward(self, feature: Tensor) -> tuple[Tensor]:
        """Expose one native stem output to PyTorch's graphed-callable bridge."""
        return (cast("Tensor", self.stem(feature)),)


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


def _precision_signature() -> tuple[object, ...]:
    """Read the autocast and CUDA numeric policy a recording keeps for its whole lifetime.

    Raises:
        RuntimeError: When torch refuses one of the policy queries.
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


def _input_breach(images: Tensor, precision: tuple[object, ...], *, allow_input_grad: bool = False) -> str | None:
    """Name why this call's inputs or autograd mode cannot use a recording, if they cannot."""
    if images.requires_grad and not allow_input_grad:
        return "the input images require gradients"
    if torch.is_anomaly_enabled():
        return "anomaly detection is enabled, so its backward NaN checks must stay observable"
    if not _BF16_REPLAY_SUPPORTED and (
        images.dtype == torch.bfloat16 or (precision[0] and precision[1] == torch.bfloat16)
    ):
        return (
            "BF16 replay requires torch>=2.9, the first release verified against the exact eager oracle, "
            f"but this runtime is torch {torch.__version__}"
        )
    return None


def _module_breach(
    modules: Iterable[nn.Module], dispatch: tuple[str, ...] = _DISPATCH, exclude: nn.Module | None = None
) -> str | None:
    """Name the first way a module's call runs code its class methods do not show, if any.

    A recording replays what the class methods computed when it was made, and the pose
    head certificate trusts the class methods it fingerprinted. A module hook, an
    instance attribute shadowing a dispatch method or a compiled call breaks both, so
    the recorder and the certificate share this one predicate.

    Args:
        modules: The modules whose calls must stay what their class methods define.
        dispatch: Method names through which a call reaches its arithmetic.
        exclude: A module whose instance dispatch attributes do not count: the recorder's
            own root, whose ``forward`` the graphed callable replaces. Its hooks and
            compiled call still count.

    Returns:
        The breach as the predicate of a sentence about one module ("carries a forward or
        backward hook"), checked in the order hooks, dispatch overrides, compiled calls;
        ``None`` when no module breaches.
    """
    members = tuple(modules)
    if any(
        module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks
        for module in members
    ):
        return "carries a forward or backward hook"
    overridden = next(
        (name for name in dispatch for module in members if module is not exclude and name in vars(module)), None
    )
    if overridden is not None:
        return f"overrides {overridden} on its instance"
    if any(getattr(module, "_compiled_call_impl", None) is not None for module in members):
        return "is compiled"
    return None


def _ineligibility(network: nn.Module) -> str | None:
    """Name the first property of the stages a recording could not keep honouring, if any."""
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
        return "a global module hook is registered"
    modules = tuple(network.modules())
    breach = _module_breach(modules, exclude=network)
    if breach is not None:
        return f"a recorded stage module {breach}"
    params = tuple(network.parameters())
    buffers = tuple(network.buffers())
    tensors = params + buffers
    checks: tuple[tuple[str, Callable[[], bool]], ...] = (
        (
            # A cumulative-average BatchNorm divides by its batch count read on the host,
            # which a stream capture forbids; frozen (eval-mode) BatchNorm reads nothing.
            "a training BatchNorm with momentum=None reads its batch count on the host",
            lambda: any(
                isinstance(module, nn.modules.batchnorm._BatchNorm)
                and module.training
                and module.track_running_stats
                and module.momentum is None
                for module in modules
            ),
        ),
        ("the network has no parameters", lambda: not params),
        ("a network parameter carries a gradient hook", lambda: any(param._backward_hooks for param in params)),
        ("a network buffer requires gradients", lambda: any(buffer.requires_grad for buffer in buffers)),
        (
            "a network parameter or buffer is a tensor subclass",
            lambda: any(type(tensor) not in (Tensor, nn.Parameter) for tensor in tensors),
        ),
        (
            "a network parameter or buffer is not a strided CUDA tensor",
            lambda: any(not tensor.is_cuda or tensor.layout != torch.strided for tensor in tensors),
        ),
    )
    return next((reason for reason, breached in checks if breached()), None)


def _network_signature(network: nn.Module) -> tuple[object, ...]:
    """Fingerprint the storage, modes and scalar attributes a recording will keep reading."""
    tensors = tuple(network.parameters()) + tuple(network.buffers())
    modules = tuple(network.modules())
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
        for holder in getattr(self, "_stems", {}).values():
            holder.clear()
        self._stems: dict[int, _NetworkCapture] = {}
        self._stem_owners: tuple[int, ...] | None = None
        previous = getattr(self, "_network", None)
        if previous is not None:
            vars(previous).pop("forward", None)
        self._network: nn.Module | None = None
        self._input: tuple[object, ...] | None = None
        self._state: tuple[object, ...] | None = None
        self._owners: tuple[int, ...] | None = None
        self._disabled = False
        self._pending_backward = False

    def __getstate__(self) -> dict[str, object]:
        """Leave CUDA recordings out of pickle and deepcopy."""
        return {}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore an empty holder that may record in its next fit."""
        del state
        self.clear()

    def retire(self, reason: str, *, stacklevel: int = 3) -> None:
        """Retire a recording whose static execution contract no longer holds.

        Only the first cause of a fit is reported: the holder stays eager afterwards, so
        later causes change nothing a user could act on.

        Args:
            reason: Which part of the recording contract broke, completing the warning
                that starts ``capture_network=True stays eager:``.
            stacklevel: The frame the warning names, counted as :func:`warnings.warn`
                counts from this method. The default names the caller of whoever called
                ``retire``; :meth:`_run` passes 4 so a retirement it decides still names
                the code that called :meth:`run`, :meth:`run_features` or :meth:`run_oriented`.
        """
        if self._disabled:
            return
        self.clear()
        self._disabled = True
        warnings.warn(f"capture_network=True stays eager: {reason}", UserWarning, stacklevel=stacklevel)

    def _drift(self, state: tuple[object, ...], owners: tuple[int, ...], precision: tuple[object, ...]) -> str | None:
        """Name a change since recording that a replay would silently ignore, if any."""
        if self._input is None:
            return None
        if owners != self._owners:
            return "a recorded stage module was replaced after recording"
        if state != self._state:
            return "parameter or buffer storage, a module mode or a scalar module attribute changed after recording"
        if precision != self._input[-1]:
            return "the autocast or CUDA precision policy changed after recording"
        return None

    def run(
        self,
        images: Tensor,
        stages: tuple[nn.Module, nn.Module, nn.Module],
        eager: Callable[[Tensor], DualHeadOutput],
    ) -> DualHeadOutput:
        """Replay an eligible training network, leaving other image signatures eager."""
        return self._run(images, stages, eager, lambda: _DetectionNetwork(*stages), _DetectionNetwork.rebuild)

    def run_features(self, images: Tensor, stages: tuple[nn.Module, nn.Module]) -> _Features | None:
        """Replay shared pose features, or return no features so the full eager call stays observable."""
        if not any(parameter.requires_grad for stage in stages for parameter in stage.parameters()):
            self.retire("the shared pose features are frozen")
            return None

        def eager(_: Tensor) -> _Features | None:
            """Leave the full native call to the module that owns its hooks."""
            return None

        rebuild: Callable[[tuple[Tensor, ...]], _Features | None] = _FeatureNetwork.rebuild
        return self._run(images, stages, eager, lambda: _FeatureNetwork(*stages), rebuild)

    def run_oriented(
        self,
        images: Tensor,
        stages: tuple[nn.Module, nn.Module, nn.Module],
        eager: Callable[[Tensor], DualHeadOutput],
    ) -> DualHeadOutput:
        """Replay all six network outputs consumed by the unchanged native oriented objective."""
        return self._run(images, stages, eager, lambda: _OrientedNetwork(*stages), _OrientedNetwork.rebuild)

    def check_stem_owners(self, head: nn.Module) -> None:
        """Retire all recordings before changed head ownership can retain obsolete pools.

        The owner checks this before shared-feature replay, so replacing an otherwise
        identical stem keeps the complete forward native and the holder count bounded.
        Fit cleanup and serialization reset this transient identity signature.
        """
        if self._disabled:
            return
        owners = tuple(id(module) for module in head.modules())
        if self._stem_owners is not None and owners != self._stem_owners:
            self.retire("a pose head module was replaced after recording")
        else:
            self._stem_owners = owners

    def stem(self, stage: nn.Module, feature: Tensor) -> Tensor:
        """Record each pose stem independently so unused point outputs keep absent gradients.

        The parent owns these holders: fit boundaries and skipped backwards reset them
        together with the shared-feature recording. Each stem retains its own input
        signature, state guards and autograd node, with no fixed joint backward that
        would advance unused point-stem optimizer state.
        """
        if self._disabled or not feature.is_cuda or not torch.is_grad_enabled() or not stage.training:
            return cast("Tensor", stage(feature))
        holder = self._stems.get(id(stage))
        if holder is None:
            holder = _NetworkCapture()
            self._stems[id(stage)] = holder
        eager = cast("Callable[[Tensor], Tensor]", stage)
        return holder._run(
            feature, (stage,), eager, lambda: _StemNetwork(stage), lambda outputs: outputs[0], allow_input_grad=True
        )

    @staticmethod
    def coefficients(branch: _DetectionBranch, features: _Features) -> Tensor | None:
        """Keep the shared prediction-runner protocol's unused mask path native.

        Pose builds no coefficient stems. This adapter composes the existing private
        runner seam without extending the recording into an optional mask objective.
        """
        return branch._coefficients(features)

    def _run(
        self,
        images: Tensor,
        stages: tuple[nn.Module, ...],
        eager: Callable[[Tensor], _Output],
        build: Callable[[], nn.Module],
        rebuild: Callable[[tuple[Tensor, ...]], _Output],
        *,
        allow_input_grad: bool = False,
    ) -> _Output:
        """Keep recording, eligibility and lifecycle identical for both stage compositions.

        Args:
            images: The training batch; its shape, strides, dtype and device key the recording.
            stages: The caller's stage modules, whose identities the recording keeps.
            eager: Produces the value returned whenever this call does not replay. It is a
                fallback value, not necessarily the native result: :meth:`run` passes the
                native forward, and so does :meth:`run_oriented`, while :meth:`run_features`
                returns ``None`` so its caller makes the full native call itself.
            build: Composes the stages into the module checked for eligibility until a
                recording exists, and captured by the first one.
            rebuild: Turns the replayed output tensors back into the caller's value.
            allow_input_grad: Permit differentiable feature inputs for an independent stem.

        Returns:
            The rebuilt replay output, or whatever ``eager`` returns when this call stays eager.
        """
        if self._disabled:
            return eager(images)
        try:
            precision = _precision_signature()
        except RuntimeError as error:
            self.retire(f"the CUDA precision policy cannot be read ({error})", stacklevel=4)
            return eager(images)
        owners = tuple(id(stage) for stage in stages)
        network = self._network if self._network is not None else build()
        reason = _input_breach(images, precision, allow_input_grad=allow_input_grad) or _ineligibility(network)
        state = _network_signature(network) if reason is None else None
        if state is not None:
            reason = self._drift(state, owners, precision)
        if reason is not None:
            self.retire(reason, stacklevel=4)
            return eager(images)
        signature = tuple(images.shape), images.stride(), images.dtype, images.device, images.requires_grad, precision
        if (self._input is not None and signature != self._input) or self._pending_backward:
            # A second forward before the previous replay's backward would overwrite the
            # static activations that backward still has to read, so it runs eagerly.
            return eager(images)
        with torch.autocast(
            "cuda", enabled=torch.is_autocast_enabled(), dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False
        ):
            if self._network is None:
                snapshots = [(buffer, buffer.detach().clone()) for buffer in network.buffers()]
                with torch.random.fork_rng(devices=[images.device.index]):
                    try:
                        sample = images.detach().clone().requires_grad_(images.requires_grad)
                        graphed = cast("nn.Module", torch.cuda.make_graphed_callables(network, (sample,)))
                    finally:
                        # Warmup runs on a side stream the recorder owns; a failure there
                        # can leave BatchNorm writes queued, so drain the device before the
                        # restore below copies onto the current stream.
                        torch.cuda.synchronize(images.device)
                        with torch.no_grad():
                            for buffer, snapshot in snapshots:
                                buffer.copy_(snapshot)
                self._network, self._input, self._state, self._owners = graphed, signature, state, owners
            outputs = cast("tuple[Tensor, ...]", self._network(images))
        self._await_backward(outputs)
        return rebuild(outputs)

    def _await_backward(self, outputs: tuple[Tensor, ...]) -> None:
        """Guard the static buffers this replay's backward will read and write.

        The replay's outputs share the graphed function's backward node. Its pre-hook runs
        just before the replayed backward writes the static gradient buffers; its post-hook
        runs once that backward has consumed the static activations, and releases the hold
        that keeps later forwards eager. An output without a backward node (a fully frozen
        network) leaves nothing to guard.
        """
        node = outputs[0].grad_fn
        if node is None:
            return
        self._pending_backward = True
        node.register_prehook(self._before_replayed_backward)
        node.register_hook(self._backward_replayed)

    def _before_replayed_backward(self, grad_outputs: object) -> None:
        """Move live parameter gradients out of the storage the replayed backward overwrites.

        A parameter whose ``.grad`` was adopted from the static gradient buffers still
        aliases them, so the next replayed backward would overwrite it and then accumulate
        onto itself. Any ``.grad`` alive at this point (gradient accumulation, an in-place
        ``zero_grad``, a callback keeping gradients) is copied out first; after
        ``zero_grad(set_to_none=True)`` there is none, and nothing is copied.
        """
        del grad_outputs
        network = self._network
        if network is None:
            return
        for parameter in network.parameters():
            if parameter.grad is not None:
                parameter.grad = parameter.grad.clone()

    def _backward_replayed(self, grad_inputs: object, grad_outputs: object) -> None:
        """Release the hold once a replayed backward has consumed the static activations."""
        del grad_inputs, grad_outputs
        self._pending_backward = False

    def start_batch(self) -> None:
        """Drop a hold whose backward never ran, so the next training batch may replay again.

        A ``training_step`` returning ``None`` skips the backward, so the post-hook that
        normally releases the hold never fires; a new batch makes that replay unreachable.
        """
        self._pending_backward = False
        for holder in self._stems.values():
            holder.start_batch()


class _NetworkCaptureLifecycle(Callback):
    """Discard transient recordings on fit entry, completion and failure."""

    @staticmethod
    def _holder(module: LightningModule) -> _NetworkCapture | None:
        holder = getattr(module, "_network_capture", None)
        return holder if isinstance(holder, _NetworkCapture) else None

    @classmethod
    def _clear(cls, module: LightningModule) -> None:
        holder = cls._holder(module)
        if holder is not None:
            holder.clear()

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: object, batch_idx: int) -> None:
        """Let each training batch replay even when the previous one skipped its backward."""
        del trainer, batch, batch_idx
        holder = self._holder(pl_module)
        if holder is not None:
            holder.start_batch()

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
