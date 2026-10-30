# SPDX-License-Identifier: Apache-2.0
"""Opt-in replay of fixed network stages' CUDA forward and backward work.

The network has a fixed image signature while its ground-truth count changes per
batch. Detection and oriented detection record backbone, neck and head; pose records
backbone, neck and each prediction stem separately; segmentation records backbone and neck.
All four leave assignment, losses, schedules, optimizer and EMA in their existing loop.
PyTorch's graphed-callable API owns the
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

import sys
import warnings
from operator import is_
from typing import TYPE_CHECKING, TypeVar, cast

import torch
from torch import Tensor, nn

from lucid_yolo.models.heads.detect import DualHeadOutput, _DetectionBranch
from lucid_yolo.ptl import _graph_replay
from lucid_yolo.ptl._graph_replay import (
    BF16_REPLAY_SUPPORTED,
    ReplayLifecycle,
    copy_out_live_gradients,
    drain_device,
    global_hook_breach,
    module_hook_breach,
    replay_autocast,
    tensor_signature,
)
from lucid_yolo.ptl._graph_replay import precision_signature as _precision_signature

_Output = TypeVar("_Output")
_Features = tuple[Tensor, Tensor, Tensor]
_Members = tuple[tuple[nn.Module, ...], tuple[Tensor, ...], tuple[Tensor, ...]]

#: First verified BF16 replay runtime, bound here so a patch of this module reaches the gate below.
_BF16_REPLAY_SUPPORTED = BF16_REPLAY_SUPPORTED

#: Method names through which a module's call reaches its arithmetic; the same name set on
#: an instance shadows the class method a recording replays or a certificate fingerprints.
_DISPATCH = ("forward", "forward_branch", "_keypoints", "_base_predictions")
_NATIVE_ENUMERATION = {
    name: getattr(nn.Module, name) for name in ("modules", "parameters", "buffers", "named_parameters", "named_buffers")
}
_NATIVE_NAMED_MODULES = nn.Module.named_modules

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


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

    A pose or segmentation batch may have no assigned positives. Its optional
    branches (pose point stems; segmentation coefficient, prototype and semantic
    predictions) then keep ``grad=None``; including them in one fixed backward
    would materialize zero gradients and advance optimizer state that the native
    loop skips. The shared features still feed the dense classification objective,
    while those branches keep independent backward nodes outside the recording and
    variable supervision stays native.
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
    if any(module_hook_breach(module) for module in members):
        return "carries a forward or backward hook"
    if any(module is not exclude and not vars(module).keys().isdisjoint(dispatch) for module in members):
        overridden = next(
            name for name in dispatch for module in members if module is not exclude and name in vars(module)
        )
        return f"overrides {overridden} on its instance"
    if any(getattr(module, "_compiled_call_impl", None) is not None for module in members):
        return "is compiled"
    return None


def _members(owner: nn.Module) -> _Members:
    """Enumerate current modules and both tensor registries in one traversal.

    The direct registries retain ``named_parameters`` and ``named_buffers`` preorder
    and identity deduplication. This is shared work within one check, never a cached
    topology: replacement, aliases and registration order are read on every call.
    """
    if any(
        getattr(type(owner), name) is not method or name in vars(owner) for name, method in _NATIVE_ENUMERATION.items()
    ):
        return tuple(owner.modules()), tuple(owner.parameters()), tuple(owner.buffers())
    pending = [owner]
    visited: set[nn.Module] = set()
    ordered: list[nn.Module] = []
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        if type(module).named_modules is not _NATIVE_NAMED_MODULES or "named_modules" in vars(module):
            modules = tuple(owner.modules())
            break
        visited.add(module)
        ordered.append(module)
        pending.extend(child for child in reversed(module._modules.values()) if child is not None)
    else:
        modules = tuple(ordered)
    parameters = {id(value): value for module in modules for value in module._parameters.values() if value is not None}
    buffers = {id(value): value for module in modules for value in module._buffers.values() if value is not None}
    return modules, tuple(parameters.values()), tuple(buffers.values())


#: Training mode's position in each module entry returned by `_module_signature`.
_MODULE_TRAINING_INDEX = 2


def _module_signature(modules: tuple[nn.Module, ...]) -> tuple[object, ...]:
    """Read the scalar attributes and class dispatch fixed by a recording."""
    return tuple(
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
    )


class _ModuleMetadata:
    """Reuse scalar metadata only while every binding and class forward is unchanged.

    Stored objects stay alive. Identity comparisons never call tensor equality, and
    unchanged dictionary identities alone certify nothing: keys, their order and every
    value are reread. The public training mode is read too: a descriptor can change it
    without rebinding an instance scalar. Hooks, live registration and tensor storage remain independent
    checks. A miss computes the original signature, including equal-value replacements.
    Recording pairs post-warmup bindings with a fresh scalar snapshot while the owner
    keeps its original drift stamp; warmup side effects must still retire the graph.
    """

    def __init__(self, modules: tuple[nn.Module, ...], signature: tuple[object, ...]) -> None:
        self.modules = modules
        self.signature = signature
        self.bindings = tuple(dict(vars(module)) for module in modules)
        self.names = tuple(tuple(saved) for saved in self.bindings)
        self.forwards = tuple(type(module).forward for module in modules)
        self.training = tuple(cast("tuple[object, ...]", entry)[_MODULE_TRAINING_INDEX] for entry in signature)

    def read(self, modules: tuple[nn.Module, ...]) -> tuple[object, ...]:
        """Return certified metadata, or perform the unchanged full attribute read."""
        if len(modules) != len(self.modules) or not all(map(is_, modules, self.modules)):
            return _module_signature(modules)
        for module, saved, names, forward, training in zip(
            modules, self.bindings, self.names, self.forwards, self.training, strict=True
        ):
            current = vars(module)
            if (
                type(module).forward is not forward
                or module.training is not training
                or tuple(current) != names
                or not all(map(is_, current.values(), saved.values()))
            ):
                return _module_signature(modules)
        return self.signature


def _ineligibility(network: nn.Module, members: _Members | None = None) -> str | None:
    """Name the first property of the stages a recording could not keep honouring, if any."""
    if global_hook_breach():
        return "a global module hook is registered"
    if members is None:
        members = tuple(network.modules()), tuple(network.parameters()), tuple(network.buffers())
    modules, params, buffers = members
    breach = _module_breach(modules, exclude=network)
    if breach is not None:
        return f"a recorded stage module {breach}"
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


def _network_signature(
    network: nn.Module, members: _Members | None = None, metadata: _ModuleMetadata | None = None
) -> tuple[object, ...]:
    """Fingerprint live storage and reuse attribute inspection only with an identity proof."""
    if members is None:
        members = tuple(network.modules()), tuple(network.parameters()), tuple(network.buffers())
    modules, parameters, buffers = members
    tensors = parameters + buffers
    return (
        _module_signature(modules) if metadata is None else metadata.read(modules),
        tensor_signature(tensors),
    )


def _caller_stacklevel() -> int:
    """Name the first frame outside the replay stack in a warning that a pose stem holder raises.

    A stem holder is reached through the head's dispatch and the module call machinery,
    whose depth no fixed ``stacklevel`` can follow. Counting starts at the frame that calls
    :func:`warnings.warn`, which is the caller of this function.
    """
    internal = {
        __file__,
        _DetectionBranch._keypoints.__code__.co_filename,
        nn.Module._call_impl.__code__.co_filename,
    }
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and frame.f_code.co_filename in internal:
        frame = frame.f_back
        level += 1
    return level


class _NetworkCapture:
    """Own one fit's recording without registering another copy of the model's parameters.

    The holder is an ordinary Python object, so Lightning checkpoints keep the original
    parameter names. Pickle and deepcopy discard its recording too. An input-size change
    takes the eager route; a change to model storage, modes, hooks or precision retires
    the recording for the rest of the fit instead of replaying stale work. Anomaly
    detection stays eager so its backward NaN checks remain observable.
    """

    def __init__(self, label: str | None = None) -> None:
        # A pose stem holder names its stem in warnings; the parent holder carries no label.
        self._label = label
        self.clear()

    def clear(self) -> None:
        """Reset the holder, then drain the recorded device before the recordings are released.

        The reset comes first, so a failing drain still leaves an empty holder; the stem holders
        and this holder's recording stay alive in local references until the drain has finished.
        Every stem holder is cleared even when an earlier one fails, and the first failure is
        raised once all are. Without a recording nothing is drained, so a CPU run and
        :meth:`__setstate__` never touch the CUDA runtime.
        """
        stems = tuple(getattr(self, "_stems", {}).values())
        previous, device = getattr(self, "_network", None), getattr(self, "_device", None)
        self._stems: dict[int, _NetworkCapture] = {}
        self._stem_owners: tuple[int, ...] | None = None
        self._stem_names: dict[int, str] = {}
        self._network: nn.Module | None = None
        self._metadata: _ModuleMetadata | None = None
        self._recorded_parameters: tuple[Tensor, ...] = ()
        self._device: torch.device | None = None
        self._input: tuple[object, ...] | None = None
        self._state: tuple[object, ...] | None = None
        self._owners: tuple[int, ...] | None = None
        self._disabled = False
        self._pending_backward = False
        failure: RuntimeError | None = None
        for holder in stems:
            try:
                holder.clear()
            except RuntimeError as error:
                failure = failure or error
        if previous is not None:
            try:
                drain_device(device)
            except RuntimeError as error:
                failure = failure or error
            finally:
                vars(previous).pop("forward", None)
        if failure is not None:
            raise failure

    def __getstate__(self) -> dict[str, object]:
        """Leave CUDA recordings out of pickle and deepcopy."""
        return {}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore an empty holder that may record in its next fit."""
        del state
        self._label = None
        self.clear()

    @property
    def retired(self) -> bool:
        """Report whether this fit's recording was retired, so every later call takes the eager route.

        Callers read it to skip their own replay checks once no cause they could find would
        change the route; :meth:`clear` at the next fit resets it.

        Examples:
            ```pycon
            >>> import warnings
            >>> holder = _NetworkCapture()
            >>> holder.retired
            False
            >>> with warnings.catch_warnings():
            ...     warnings.simplefilter("ignore")
            ...     holder.retire("a doctest cause")
            >>> holder.retired
            True

            ```
        """
        return self._disabled

    def retire(self, reason: str, *, stacklevel: int = 3) -> None:
        """Retire a recording whose static execution contract no longer holds.

        Only the first cause of a fit is reported: the holder stays eager afterwards, so
        later causes change nothing a user could act on. A pose stem holder retires only
        its own stem, so its warning names that stem and says the shared features and the
        other stems keep replaying; each stem reports its own first cause.

        Args:
            reason: Which part of the recording contract broke, completing the warning
                that starts ``capture_network=True stays eager:``, or, for a stem holder,
                ``capture_network=True keeps pose stem <name> eager ...:``.
            stacklevel: The frame the warning names, counted as :func:`warnings.warn`
                counts from this method. The default names the caller of whoever called
                ``retire``; :meth:`_run` passes 4 so a retirement it decides still names
                the code that called :meth:`run`, :meth:`run_features` or :meth:`run_oriented`.
                A stem holder ignores it and names the first frame outside the head and
                this module, the training code that called the head.
        """
        if self._disabled:
            return
        self.clear()
        self._disabled = True
        if self._label is None:
            message = f"capture_network=True stays eager: {reason}"
        else:
            message = (
                f"capture_network=True keeps pose stem {self._label} eager while the shared features "
                f"and the other stems keep replaying: {reason}"
            )
            stacklevel = _caller_stacklevel()
        warnings.warn(message, UserWarning, stacklevel=stacklevel)

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
        """Replay shared features, or return no features so the full eager call stays observable."""
        if not any(parameter.requires_grad for stage in stages for parameter in stage.parameters()):
            self.retire("the shared network features are frozen")
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
        if self._stem_owners is None:
            self._stem_owners = owners
            self._stem_names = {id(module): name for name, module in head.named_modules()}
        elif owners != self._stem_owners:
            self.retire("a pose head module was replaced after recording")

    def stem(self, stage: nn.Module, feature: Tensor) -> Tensor:
        """Record each pose stem independently so unused point outputs keep absent gradients.

        The parent owns these holders: fit boundaries and skipped backwards reset them
        together with the shared-feature recording. Each stem retains its own input
        signature, state guards and autograd node, with no fixed joint backward that
        would advance unused point-stem optimizer state. An eval-mode stem never starts a
        recording; a recorded stem switched to eval reaches the drift check and retires,
        as a mode change of a recorded feature stage does.
        """
        if self._disabled or not feature.is_cuda or not torch.is_grad_enabled():
            return cast("Tensor", stage(feature))
        holder = self._stems.get(id(stage))
        if not stage.training and (holder is None or holder._network is None):
            return cast("Tensor", stage(feature))
        if holder is None:
            holder = _NetworkCapture(self._stem_names.get(id(stage), type(stage).__name__))
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
        reason = _input_breach(images, precision, allow_input_grad=allow_input_grad)
        members = _members(network) if reason is None else None
        if reason is None:
            reason = _ineligibility(network, members)
        state = _network_signature(network, members, self._metadata) if reason is None else None
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
        with replay_autocast():
            if self._network is None:
                sample = images.detach().clone().requires_grad_(images.requires_grad)
                # Looked up through the module: the one seam a CPU test replaces for both holders.
                graphed = _graph_replay.record_graphed(network, (sample,), images.device)
                self._network, self._input, self._state, self._owners = graphed, signature, state, owners
                assert members is not None and state is not None
                self._metadata = _ModuleMetadata(members[0], _module_signature(members[0]))
                self._recorded_parameters = members[1]
                self._device = images.device
            self._copy_out_live_gradients()
            outputs = cast("tuple[Tensor, ...]", self._network(images))
        self._await_backward(outputs)
        return rebuild(outputs)

    def _await_backward(self, outputs: tuple[Tensor, ...]) -> None:
        """Hold later forwards eager until this replay's backward has read the static buffers.

        The replay's outputs share the graphed function's backward node; its post-hook runs
        once that backward has consumed the static activations, and releases the hold. An
        output without a backward node (a fully frozen network) leaves nothing to guard.
        """
        node = outputs[0].grad_fn
        if node is None:
            return
        self._pending_backward = True
        node.register_hook(self._backward_replayed)

    def _copy_out_live_gradients(self) -> None:
        """Move live parameter gradients out of the storage the next replayed backward overwrites.

        A parameter whose ``.grad`` was adopted from the static gradient buffers still
        aliases them, so the next replayed backward would overwrite it and then accumulate
        onto itself. Any ``.grad`` alive before a replay (gradient accumulation, an in-place
        ``zero_grad``, a callback keeping gradients) is copied out first. Lightning clears
        gradients after the training step's forward, so the previous step's gradients are
        copied too, at every accumulation setting, and ``zero_grad`` then drops the clones.

        The copy runs at forward time, before the replay. The hold in :meth:`_await_backward`
        keeps any other replay from running between this forward and its backward, so no
        gradient can alias the static buffers again before that backward writes them. The
        same copy made from a pre-hook inside the replayed backward left every accumulated
        gradient wrong on CUDA (WP-206).
        """
        network = self._network
        if network is None:
            return
        copy_out_live_gradients(self._recorded_parameters)

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


class _NetworkCaptureLifecycle(ReplayLifecycle):
    """Discard transient recordings on fit entry, completion and failure."""

    attribute, holder_type, flag = "_network_capture", _NetworkCapture, "capture_network"
