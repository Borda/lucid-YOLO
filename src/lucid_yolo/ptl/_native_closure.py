# SPDX-License-Identifier: Apache-2.0
"""Lightning and PyTorch identities that certify the native automatic-optimization closure (WP-193h).

:class:`~lucid_yolo.ptl.module.DetectionLitModule` omits the forward-time gradient
snapshots that Lightning's automatic-optimization closure clears before backward. The
omission is safe only while the closure, its zeroing function and every hook that can
observe the old gradients are the ones Lightning and PyTorch ship. This module holds those
third-party identities, captured when it is imported, and the checks that compare them at
run time. Most name a private Lightning or PyTorch detail, so re-inspect every one of them
on each Lightning or PyTorch minor before widening :data:`_ZEROING_RUNTIME_SUPPORTED`.

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
from types import CodeType
from typing import TYPE_CHECKING, Any, cast

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
        for name in ("optimizer_step", "_wrap_closure", "train_step_context", "forward_context")
    }
    for kind in (Precision, MixedPrecision)
}
_NATIVE_FORWARD_PROFILER_METHODS = {
    name: getattr(PassThroughProfiler, name, None) for name in ("profile", "start", "stop")
}
_NATIVE_ZEROING_METHODS = {
    name: getattr(_AutomaticOptimizationType, name, None) for name in ("_on_before_zero_grad", "_optimizer_zero_grad")
}


@functools.cache
def _native_wrapper_codes() -> tuple[CodeType | None, CodeType | None]:
    """Read the code of PyTorch's optimizer-profiler and scheduler step wrappers, once, on first use.

    Returns:
        The ``(profile, scheduler)`` code objects, each ``None`` when this PyTorch does not
        expose it where it was inspected.
    """
    factory = getattr(torch.optim.Optimizer, "profile_hook_step", None)
    profile = getattr(factory(_NATIVE_MUSGD_STEP), "__code__", None) if factory else None
    scheduler: CodeType | None = getattr(torch.optim.lr_scheduler.LRScheduler.__init__, "__code__", None)
    for wrapper_name in ("patch_track_step_called", "wrap_step", "wrapper"):
        if scheduler is not None:
            scheduler = next(
                (
                    value
                    for value in scheduler.co_consts
                    if isinstance(value, CodeType) and value.co_name == wrapper_name
                ),
                None,
            )
    return profile, scheduler


def _native_update_step(optimizer: MuSGD) -> bool:
    """Recognize native profiler and scheduler wrappers without trusting custom step wrappers."""
    profile_code, scheduler_code = _native_wrapper_codes()
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


def _native_closure_shape(
    trainer: Trainer | None, optimizer: Optimizer | LightningOptimizer, closure: Callable[[], Any] | None
) -> bool:
    """Tell whether ``optimizer_step`` received Lightning's own closure and optimizer wrapper.

    Args:
        trainer: The module's attached trainer, or ``None``.
        optimizer: The optimizer ``optimizer_step`` received, wrapped or raw.
        closure: The closure ``optimizer_step`` received.

    Returns:
        Whether the native closure may have its step function wrapped for this step.
    """
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    return not (
        not _ZEROING_RUNTIME_SUPPORTED
        or type(closure) is not _ClosureType
        or _ClosureType.__call__ is not _NATIVE_CLOSURE_CALL
        or _ClosureType.closure is not _NATIVE_CLOSURE_BODY  # type: ignore[union-attr]
        or "closure" in vars(closure)
        or trainer is None
        or len(trainer.optimizers) != 1
        or raw_optimizer is not trainer.optimizers[0]
        or (
            raw_optimizer is not optimizer
            and (getattr(optimizer.step, "__func__", None) is not _NATIVE_LIGHTNING_STEP or "step" in vars(optimizer))
        )
    )


def _runtime_cleared_parameters(
    trainer: Trainer | None, module: LightningModule, closure: Callable[[], Any] | None, step_fn: Callable[[], Any]
) -> tuple[Tensor, ...]:
    """Name the parameters the native closure clears before backward, or none when unproven.

    The caller has already certified its own methods, objective and hooks; this checks the
    trainer, the forward-time observers, the zeroing closure and the optimizer against the
    identities captured at import.

    Args:
        trainer: The module's attached trainer, or ``None``.
        module: The module whose training step the closure runs.
        closure: The closure ``optimizer_step`` received.
        step_fn: The closure's original step function.

    Returns:
        The live optimizer's parameters, or ``()`` when any contract is not the native one.
    """
    callbacks = getattr(trainer, "callbacks", None)
    profiler = getattr(trainer, "profiler", None)
    if (
        not _ZEROING_RUNTIME_SUPPORTED
        or trainer is None
        or not trainer.training
        or trainer.accumulate_grad_batches != 1
        or type(trainer.precision_plugin) not in (Precision, MixedPrecision)
        or type(trainer.strategy) is not SingleDeviceStrategy
        or trainer.strategy.model is not module
        or type(profiler) is not PassThroughProfiler
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
        or len(trainer.optimizers) != 1
    ):
        return ()
    loop = getattr(getattr(trainer.fit_loop, "epoch_loop", None), "automatic_optimization", None)
    if (
        loop is None
        or type(loop) is not _AutomaticOptimizationType
        or any(
            getattr(type(loop), name, None) is not native or name in vars(loop)
            for name, native in _NATIVE_ZEROING_METHODS.items()
        )
        or loop._skip_backward
        or type(closure) is not _ClosureType
        or "closure" in vars(closure)
    ):
        return ()
    zeroing = cast("Any", closure)._zero_grad_fn
    step = getattr(step_fn, "func", None)
    if (
        zeroing is None
        or _NATIVE_ZEROING_CODE is None
        or getattr(zeroing, "__code__", None) is not _NATIVE_ZEROING_CODE
        or getattr(step, "__func__", None) is not _NATIVE_TRAINING_LOOP_STEP
        or getattr(step, "__self__", None) is not loop
    ):
        return ()
    optimizer = trainer.optimizers[0]
    bindings = dict(zip(zeroing.__code__.co_freevars, zeroing.__closure__ or (), strict=True))
    if (
        type(optimizer) is not MuSGD
        or not _native_update_step(optimizer)
        or type(optimizer).zero_grad is not _NATIVE_ZERO_GRAD
        or "zero_grad" in vars(optimizer)
        or "self" not in bindings
        or bindings["self"].cell_contents is not loop
        or "optimizer" not in bindings
        or bindings["optimizer"].cell_contents is not optimizer
    ):
        return ()
    return tuple(parameter for group in optimizer.param_groups for parameter in group["params"])
