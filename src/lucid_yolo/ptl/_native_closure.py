# SPDX-License-Identifier: Apache-2.0
"""Lightning and PyTorch identities that certify the native automatic-optimization closure (WP-193h).

:class:`~lucid_yolo.ptl.module.DetectionLitModule` omits the forward-time gradient
snapshots that Lightning's automatic-optimization closure clears before backward. The
omission is safe only while the closure, its zeroing function and every hook that can
observe the old gradients are the ones Lightning and PyTorch ship. This module holds those
third-party identities, captured when it is imported, and the checks that compare them at
run time. Most name a private Lightning or PyTorch detail, so re-inspect every one of them
on each Lightning or PyTorch minor before widening :data:`_ZEROING_RUNTIME_SUPPORTED`.

The checks are four named predicates, each owning one hazard: the trainer contract, the
forward-time observers, the zeroing closure and the optimizer. :func:`_first_refusal`
evaluates them in that order and names the first that fails with a constant
:class:`_Refusal` token, so a caller can report why the omission is off without formatting
anything per step.

The module never imports :mod:`lucid_yolo.ptl.module` at runtime. Identities bound to
``DetectionLitModule`` itself (its methods and its objective) stay beside the class, and
the module runs its own checks before it delegates here. The Lightning clipping identities
the module's gradient-norm capture compares live here as well, as the same kind of record.
The PyTorch step-wrapper code objects are read on first use rather than at import, so an
importer that never trains pays nothing for them.
"""

from __future__ import annotations

import contextlib
import functools
import weakref
from enum import StrEnum
from types import CodeType
from typing import TYPE_CHECKING, Any, TypeGuard, cast

import pytorch_lightning
import torch
from packaging.version import Version
from pytorch_lightning import Callback, LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.plugins.precision import MixedPrecision, Precision
from pytorch_lightning.profilers import PassThroughProfiler
from pytorch_lightning.strategies import SingleDeviceStrategy
from torch.optim import optimizer as optimizer_api

from lucid_yolo.optim.musgd import MuSGD

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytorch_lightning import Trainer
    from torch import Tensor
    from torch.optim import Optimizer

_automatic_api: Any = None
with contextlib.suppress(ImportError):
    from pytorch_lightning.loops.optimization import automatic as _automatic_api
_trainer_call: Any = None
with contextlib.suppress(ImportError):
    from pytorch_lightning.trainer import call as _trainer_call

_ClosureType: Any = getattr(_automatic_api, "Closure", None)
_AutomaticOptimizationType: Any = getattr(_automatic_api, "_AutomaticOptimization", None)

_NATIVE_MODULE_CLIPPING = LightningModule.clip_gradients
_NATIVE_CONFIGURE_CLIPPING = LightningModule.configure_gradient_clipping
_NATIVE_PRECISION_CLIPPING = {
    kind: {name: getattr(kind, name) for name in ("clip_gradients", "clip_grad_by_norm", "main_params")}
    for kind in (Precision, MixedPrecision)
}

_NATIVE_CALLBACK_ATTRIBUTE_ACCESS: object = object.__getattribute__
_NATIVE_CALLBACK_ZEROING = Callback.on_before_zero_grad
_NATIVE_LIGHTNING_STEP = LightningOptimizer.step
_NATIVE_CLOSURE_CALL = getattr(_ClosureType, "__call__", None)  # noqa: B004 — retain the method identity
_NATIVE_CLOSURE_BODY = getattr(_ClosureType, "closure", None)
_NATIVE_MUSGD_STEP = MuSGD.step
_NATIVE_ZERO_GRAD = torch.optim.Optimizer.zero_grad
_NATIVE_TRAINING_LOOP_STEP = getattr(_AutomaticOptimizationType, "_training_step", None)
_ZEROING_FACTORY = getattr(_AutomaticOptimizationType, "_make_zero_grad_fn", None)
_NATIVE_ZEROING_CODE = next(
    (
        value
        for value in getattr(getattr(_ZEROING_FACTORY, "__code__", None), "co_consts", ())
        if getattr(value, "co_name", None) == "zero_grad_fn"
    ),
    None,
)
_ZEROING_RUNTIME_SUPPORTED = Version(getattr(pytorch_lightning, "__version__", "0")).release[:2] in {
    (2, 4),
    (2, 5),
    (2, 6),
}
_NATIVE_STRATEGY_METHODS = {
    "training_step": getattr(SingleDeviceStrategy, "training_step", None),
    "optimizer_step": getattr(SingleDeviceStrategy, "optimizer_step", None),
    "post_training_step": getattr(SingleDeviceStrategy, "post_training_step", None),
}
_NATIVE_PRECISION_METHODS = {
    kind: {
        name: getattr(kind, name, None)
        for name in (
            "optimizer_step",
            "_wrap_closure",
            "train_step_context",
            "forward_context",
            "autocast_context_manager",
        )
    }
    for kind in (Precision, MixedPrecision)
}
#: The dispatchers the closure's forward and zeroing hooks run through. The loop resolves each
#: on the ``call`` module at every call, so the module attribute is what is compared.
_NATIVE_TRAINER_CALLS = {
    name: getattr(_trainer_call, name, None)
    for name in ("_call_strategy_hook", "_call_lightning_module_hook", "_call_callback_hooks")
}
_NATIVE_FORWARD_PROFILER_METHODS = {
    name: getattr(PassThroughProfiler, name, None) for name in ("profile", "start", "stop")
}
_NATIVE_ZEROING_METHODS = {
    name: getattr(_AutomaticOptimizationType, name, None) for name in ("_on_before_zero_grad", "_optimizer_zero_grad")
}


class _Refusal(StrEnum):
    """The certificate predicate that first refused the native closure, as a constant token."""

    #: The module's own methods, objective, hooks or task (checked by the module itself).
    MODULE = "module"
    #: :func:`_trainer_contract`.
    TRAINER_CONTRACT = "trainer_contract"
    #: :func:`_forward_observers`.
    FORWARD_OBSERVERS = "forward_observers"
    #: :func:`_zeroing_closure`.
    ZEROING_CLOSURE = "zeroing_closure"
    #: :func:`_optimizer_contract`.
    OPTIMIZER = "optimizer"


@functools.cache
def _resolve_profile_wrapper_code() -> CodeType | None:
    """Read the code of the wrapper PyTorch's optimizer profiler puts around ``step``, once, on first use.

    Returns:
        The wrapper's code object, or ``None`` when this PyTorch has no such factory or building
        a wrapper with it fails; a profiler-wrapped step is then not recognized as native.
    """
    factory = getattr(torch.optim.Optimizer, "profile_hook_step", None)
    if factory is None:
        return None
    try:
        wrapper = factory(_NATIVE_MUSGD_STEP)
    except Exception:  # a private PyTorch factory whose contract changed only withdraws the omission
        return None
    return getattr(wrapper, "__code__", None)


@functools.cache
def _resolve_scheduler_wrapper_code() -> CodeType | None:
    """Read the code of the wrapper a PyTorch scheduler installs on ``optimizer.step``, once, on first use.

    The wrapper is a nested function of ``LRScheduler.__init__``; its name differs across
    PyTorch versions, so each inspected nesting level is searched in turn.

    Returns:
        The wrapper's code object, or ``None`` when this PyTorch nests it differently.
    """
    code: CodeType | None = getattr(torch.optim.lr_scheduler.LRScheduler.__init__, "__code__", None)
    for wrapper_name in ("patch_track_step_called", "wrap_step", "wrapper"):
        if code is None:
            return None
        code = next(
            (value for value in code.co_consts if isinstance(value, CodeType) and value.co_name == wrapper_name),
            None,
        )
    return code


def _native_update_step(optimizer: MuSGD) -> bool:
    """Recognize native profiler and scheduler wrappers without trusting custom step wrappers."""
    profile_code, scheduler_code = _resolve_profile_wrapper_code(), _resolve_scheduler_wrapper_code()
    root = type(optimizer).step
    if root is not _NATIVE_MUSGD_STEP and not (
        profile_code is not None
        and getattr(root, "__code__", None) is profile_code
        and getattr(root, "__wrapped__", None) is _NATIVE_MUSGD_STEP
    ):
        return False
    pre_hooks = getattr(optimizer, "_optimizer_step_pre_hooks", None)
    if pre_hooks is None or pre_hooks or getattr(optimizer_api, "_global_optimizer_pre_hooks", None) != {}:
        return False
    step = optimizer.step
    if "step" not in vars(optimizer):
        return getattr(step, "__func__", None) is root
    if scheduler_code is None or getattr(step, "__code__", None) is not scheduler_code:
        return False
    cells = dict(zip(step.__code__.co_freevars, step.__closure__ or (), strict=True))
    return (
        "func" in cells
        and cells["func"].cell_contents is root
        and "opt_ref" in cells
        and cells["opt_ref"].cell_contents() is optimizer
    )


def _native_closure_shape(closure: object) -> bool:
    """Tell whether ``closure`` is Lightning's own closure object, its ``closure`` method unshadowed.

    ``optimizer_step`` swaps the step function only of a closure of this shape, and the
    zeroing-closure predicate requires the same shape, so both read this one definition.

    Args:
        closure: The closure ``optimizer_step`` received.

    Returns:
        Whether ``closure`` is an instance of Lightning's ``Closure`` without an instance-level
        ``closure`` method.
    """
    return type(closure) is _ClosureType and "closure" not in vars(closure)


def _dispatches_to(optimizer: object, strategy: object) -> bool:
    """Tell whether a Lightning optimizer wrapper sends its step through ``strategy``.

    Lightning stores a weak proxy of the trainer's strategy on the wrapper, and a proxy is
    never the object it refers to, so the stored value is matched against the strategy's own
    weak references as well as against the strategy itself.

    Args:
        optimizer: The Lightning optimizer wrapper.
        strategy: The trainer's strategy.

    Returns:
        Whether the wrapper's ``_strategy`` is ``strategy`` or a weak proxy of it.
    """
    held = getattr(optimizer, "_strategy", None)
    return held is strategy or any(reference is held for reference in weakref.getweakrefs(strategy))


def _trainer_contract(trainer: Trainer | None, module: LightningModule) -> TypeGuard[Trainer]:
    """Hazard: another loop, strategy, accumulation or hook dispatcher routes forward and zeroing elsewhere."""
    return not (
        not _ZEROING_RUNTIME_SUPPORTED
        or trainer is None
        or not trainer.training
        or trainer.accumulate_grad_batches != 1
        or len(trainer.optimizers) != 1
        or type(trainer.precision_plugin) not in (Precision, MixedPrecision)
        or type(trainer.strategy) is not SingleDeviceStrategy
        or trainer.strategy.model is not module
        or any(
            native is None or getattr(_trainer_call, name, None) is not native
            for name, native in _NATIVE_TRAINER_CALLS.items()
        )
    )


def _forward_observers(trainer: Trainer) -> bool:
    """Hazard: code running between the forward-time copy-out and zeroing holds an old gradient."""
    profiler = getattr(trainer, "profiler", None)
    callbacks = getattr(trainer, "callbacks", None)
    return not (
        type(profiler) is not PassThroughProfiler
        or PassThroughProfiler.__getattribute__ is not object.__getattribute__
        or any(
            getattr(PassThroughProfiler, name, None) is not native or name in vars(profiler)
            for name, native in _NATIVE_FORWARD_PROFILER_METHODS.items()
        )
        or any(
            getattr(type(trainer.strategy), name, None) is not native or name in vars(trainer.strategy)
            for name, native in _NATIVE_STRATEGY_METHODS.items()
        )
        or any(
            getattr(type(trainer.precision_plugin), name, None) is not native or name in vars(trainer.precision_plugin)
            for name, native in _NATIVE_PRECISION_METHODS[type(trainer.precision_plugin)].items()
        )
        or callbacks is None
        or any(
            type(callback).__getattribute__ is not _NATIVE_CALLBACK_ATTRIBUTE_ACCESS
            or type(callback).on_before_zero_grad is not _NATIVE_CALLBACK_ZEROING
            or "on_before_zero_grad" in vars(callback)
            for callback in callbacks
        )
    )


def _zeroing_closure(trainer: Trainer, closure: Callable[[], Any] | None, step_fn: Callable[[], Any]) -> bool:
    """Hazard: a closure, loop or zeroing function other than Lightning's skips or misdirects zeroing."""
    loop = getattr(getattr(trainer.fit_loop, "epoch_loop", None), "automatic_optimization", None)
    if (
        loop is None
        or type(loop) is not _AutomaticOptimizationType
        or any(
            getattr(type(loop), name, None) is not native or name in vars(loop)
            for name, native in _NATIVE_ZEROING_METHODS.items()
        )
        or loop._skip_backward
        or not _native_closure_shape(closure)
        or _ClosureType.__call__ is not _NATIVE_CLOSURE_CALL
        or _ClosureType.closure is not _NATIVE_CLOSURE_BODY
    ):
        return False
    zeroing = cast("Any", closure)._zero_grad_fn
    step = getattr(step_fn, "func", None)
    if (
        zeroing is None
        or _NATIVE_ZEROING_CODE is None
        or getattr(zeroing, "__code__", None) is not _NATIVE_ZEROING_CODE
        or getattr(step, "__func__", None) is not _NATIVE_TRAINING_LOOP_STEP
        or getattr(step, "__self__", None) is not loop
    ):
        return False
    bindings = dict(zip(zeroing.__code__.co_freevars, zeroing.__closure__ or (), strict=True))
    return (
        "self" in bindings
        and bindings["self"].cell_contents is loop
        and "optimizer" in bindings
        and bindings["optimizer"].cell_contents is trainer.optimizers[0]
    )


def _optimizer_contract(trainer: Trainer, optimizer: Optimizer | LightningOptimizer) -> bool:
    """Hazard: a foreign, rewrapped or hooked optimizer steps or zeroes outside the certified path."""
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    native = trainer.optimizers[0]
    return not (
        raw_optimizer is not native
        or (
            raw_optimizer is not optimizer
            and (
                getattr(optimizer.step, "__func__", None) is not _NATIVE_LIGHTNING_STEP
                or "step" in vars(optimizer)
                or not _dispatches_to(optimizer, trainer.strategy)
            )
        )
        or type(native) is not MuSGD
        or not _native_update_step(native)
        or type(native).zero_grad is not _NATIVE_ZERO_GRAD
        or "zero_grad" in vars(native)
    )


def _first_refusal(
    trainer: Trainer | None,
    module: LightningModule,
    closure: Callable[[], Any] | None,
    step_fn: Callable[[], Any],
    optimizer: Optimizer | LightningOptimizer,
) -> _Refusal | None:
    """Name the first Lightning or PyTorch predicate the current step fails.

    The caller has already certified its own methods, objective and hooks.

    Args:
        trainer: The module's attached trainer, or ``None``.
        module: The module whose training step the closure runs.
        closure: The closure ``optimizer_step`` received.
        step_fn: The closure's original step function.
        optimizer: The optimizer ``optimizer_step`` received, wrapped or raw.

    Returns:
        The refusing predicate's token, or ``None`` when every predicate holds.
    """
    if not _trainer_contract(trainer, module):
        return _Refusal.TRAINER_CONTRACT
    if not _forward_observers(trainer):
        return _Refusal.FORWARD_OBSERVERS
    if not _zeroing_closure(trainer, closure, step_fn):
        return _Refusal.ZEROING_CLOSURE
    if not _optimizer_contract(trainer, optimizer):
        return _Refusal.OPTIMIZER
    return None


def _cleared_parameters(trainer: Trainer) -> tuple[Tensor, ...]:
    """Name the parameters a certified native closure clears before backward.

    Args:
        trainer: The trainer whose single optimizer the closure zeroes.

    Returns:
        Every parameter in the live optimizer's groups, in group order.
    """
    return tuple(parameter for group in trainer.optimizers[0].param_groups for parameter in group["params"])
