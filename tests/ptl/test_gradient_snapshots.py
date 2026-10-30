# SPDX-License-Identifier: Apache-2.0
"""Old replay gradients survive unless a native closure is about to clear them."""

from __future__ import annotations

import functools
import types
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loops.optimization.automatic import Closure, ClosureResult
from pytorch_lightning.plugins.precision import MixedPrecision
from pytorch_lightning.profilers import PassThroughProfiler
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.trainer import call as trainer_call
from pytorch_lightning.trainer.states import RunningStage

from lucid_yolo.optim.musgd import MuSGD
from lucid_yolo.ptl._graph_replay import _gradients_cleared_after_forward, copy_out_live_gradients
from lucid_yolo.ptl._native_closure import (
    _NATIVE_TRAINER_CALLS,
    _native_update_step,
    _Refusal,
    _resolve_profile_wrapper_code,
)
from lucid_yolo.ptl.module import DetectionLitModule


@pytest.fixture
def native() -> Iterator[tuple[DetectionLitModule, Trainer, object]]:
    """Build the real CPU closure; stand in only for the already-verified CUDA activation."""
    module = DetectionLitModule(0.5, 0.25, 64, 2, capture_network=True)
    trainer = Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.strategy._lightning_module = module
    trainer.strategy.model = module
    module._trainer = trainer
    trainer.optimizers = [MuSGD(module.parameters(), 0.01)]
    trainer.state.stage = RunningStage.TRAINING
    module._capture_network_active = True
    closure = trainer.fit_loop.epoch_loop.automatic_optimization._make_closure(OrderedDict(), trainer.optimizers[0], 0)
    yield module, trainer, closure
    module._trainer = None


@pytest.fixture
def certify(native) -> Callable[[], tuple[torch.Tensor, ...]]:
    """Evaluate the certificate as ``optimizer_step`` does, with the strategy's own optimizer wrapper."""
    module, trainer, closure = native
    return lambda: module._parameters_cleared_after_forward(
        closure, closure._step_fn, trainer.strategy._lightning_optimizers[0]
    )


def test_native_closure_names_live_optimizer_parameters(native, certify) -> None:
    """The native plan follows group membership rather than every registered parameter."""
    module, trainer, _ = native
    parameters = tuple(module.parameters())
    removed = trainer.optimizers[0].param_groups[0]["params"].pop()
    clearing = certify()
    assert len(clearing) == len(parameters) - 1
    assert all(parameter is not removed for parameter in clearing)
    first = clearing[0]
    first.grad = first_live = torch.ones_like(first)
    removed.grad = removed_live = torch.full_like(removed, 2)
    with _gradients_cleared_after_forward(clearing):
        copy_out_live_gradients([first, removed])
    assert first.grad is first_live
    assert removed.grad is not removed_live
    torch.testing.assert_close(removed.grad, torch.full_like(removed, 2), rtol=0, atol=0)


@pytest.mark.parametrize("fails", [False, True])
def test_copy_permission_ends_after_forward(fails: bool) -> None:
    """Nested and failing forwards cannot suppress a later accumulation snapshot."""
    parameter = torch.nn.Parameter(torch.zeros(3))
    parameter.grad = live = torch.tensor([1.0, -0.0, 3.0])
    try:
        with _gradients_cleared_after_forward([parameter]):
            with _gradients_cleared_after_forward([]):
                copy_out_live_gradients([parameter])
            assert parameter.grad is not live
            held = parameter.grad
            copy_out_live_gradients([parameter])
            assert parameter.grad is held
            if fails:
                raise RuntimeError("forward failed")
    except RuntimeError as error:
        assert str(error) == "forward failed"
    copy_out_live_gradients([parameter])
    assert parameter.grad is not held
    assert torch.equal(parameter.grad.view(torch.int32), live.view(torch.int32))


@pytest.mark.parametrize(
    "change",
    [
        "accumulation",
        "manual",
        "inactive",
        "validation",
        "task",
        "zero_hook",
        "callback",
        "custom_step",
        "custom_zero",
        "missing_zero",
    ],
)
def test_custom_zeroing_keeps_snapshots(native, certify, change: str) -> None:
    """Each unsupported zeroing contract independently selects the original clone route."""
    module, trainer, closure = native
    assert certify()
    optimizer = trainer.optimizers[0]
    if change == "accumulation":
        trainer.accumulate_grad_batches = 2
    elif change == "manual":
        module.automatic_optimization = False
    elif change == "inactive":
        module._capture_network_active = False
    elif change == "validation":
        trainer.state.stage = RunningStage.VALIDATING
    elif change == "task":
        module._task = "keypoints"
    elif change == "zero_hook":
        module.on_before_zero_grad = lambda optimizer: None
    elif change == "callback":

        class _Observer(Callback):
            def on_before_zero_grad(self, trainer, module, optimizer):
                raise AssertionError("not executed by the certificate")

        trainer.callbacks.append(_Observer())
    elif change == "custom_step":
        optimizer.step = lambda closure=None: None
    elif change == "custom_zero":
        optimizer.zero_grad = lambda: None
    elif change == "missing_zero":
        closure._zero_grad_fn = None
    assert certify() == ()


def test_shadowed_closure_keeps_snapshots(native, certify) -> None:
    """A closure whose own ``closure`` method was replaced on the instance keeps the clone route.

    ``type(closure)`` and the class's ``closure``/``__call__`` stay the native ones, so this is
    not caught by the class-identity checks alone; only the instance-dict check is.
    """
    _, _, closure = native
    assert certify()
    closure.closure = types.MethodType(lambda self, *args, **kwargs: self._step_fn(), closure)
    assert certify() == ()


def test_scheduler_wrapper_keeps_native_update_step(native, certify) -> None:
    """LambdaLR's real wrapper qualifies; an arbitrary functools wrapper does not."""
    _, trainer, _ = native
    optimizer = trainer.optimizers[0]
    torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    assert _native_update_step(optimizer)
    assert certify()
    native_step = optimizer.step

    @functools.wraps(native_step)
    def custom_step(*args, **kwargs):
        return native_step(*args, **kwargs)

    optimizer.step = custom_step
    assert not _native_update_step(optimizer)


@pytest.fixture
def fresh_profile_code() -> Iterator[None]:
    """Drop the cached profiler-wrapper code around a test that changes how PyTorch builds it."""
    _resolve_profile_wrapper_code.cache_clear()
    yield
    _resolve_profile_wrapper_code.cache_clear()


@pytest.mark.usefixtures("fresh_profile_code")
def test_failing_profile_factory_keeps_snapshots(native, certify, monkeypatch: pytest.MonkeyPatch) -> None:
    """A profiler factory that raises withdraws the omission instead of failing the training step.

    The wrapper code is read on the first certificate evaluation, mid-fit, so a changed private
    PyTorch factory must leave the profiler-wrapped MuSGD step unrecognized rather than raise.
    """
    _, trainer, _ = native

    def broken(func):
        raise RuntimeError("factory contract changed")

    monkeypatch.setattr(torch.optim.Optimizer, "profile_hook_step", staticmethod(broken))
    assert _resolve_profile_wrapper_code() is None
    assert not _native_update_step(trainer.optimizers[0])
    assert certify() == ()


def test_direct_training_forward_has_no_clear_permission(native) -> None:
    """A manually invoked training_step must retain snapshots despite an attached training trainer."""
    module, _, _ = native
    parameter = next(module.parameters())
    parameter.grad = live = torch.ones_like(parameter)

    def shared_step(batch, stage):
        copy_out_live_gradients([parameter])
        return torch.tensor(1.0), None, None

    with patch.object(module, "_shared_step", shared_step):
        assert module.training_step(None, 0).item() == 1.0
    assert parameter.grad is not live


@pytest.mark.parametrize("fails", [False, True])
def test_optimizer_scope_restores_closure_after_forward(native, fails: bool) -> None:
    """The real MuSGD/Closure bridge clears its permission on success and on failure."""
    module, trainer, _ = native
    parameter = torch.nn.Parameter(torch.zeros(3))
    parameter.grad = live = torch.ones(3)
    optimizer = MuSGD([parameter], 0.01)
    trainer.optimizers = [optimizer]
    observed = []

    def step():
        copy_out_live_gradients([parameter])
        observed.append(parameter.grad is live)
        if fails:
            raise RuntimeError("forward failed")
        return ClosureResult.from_training_step_output(torch.tensor(1.0), normalize=1)

    closure = Closure(step_fn=step, backward_fn=None, zero_grad_fn=optimizer.zero_grad)
    with patch.object(module, "_parameters_cleared_after_forward", return_value=(parameter,)):
        if fails:
            with pytest.raises(RuntimeError, match="forward failed"):
                module.optimizer_step(0, 0, optimizer, closure)
        else:
            module.optimizer_step(0, 0, optimizer, closure)
    assert observed == [True]
    assert closure._step_fn is step
    parameter.grad = live
    copy_out_live_gradients([parameter])
    assert parameter.grad is not live


def test_shadowed_closure_never_enters_fast_path(native) -> None:
    """A closure whose own ``closure`` method was replaced on the instance keeps the clone route.

    ``type(closure)`` stays the native one, so only the instance-dict check of the shared closure
    shape catches it. The certificate is patched to accept, so ``optimizer_step``'s pre-filter
    alone must refuse to wrap the step function.
    """
    module, trainer, _ = native
    parameter = torch.nn.Parameter(torch.zeros(3))
    parameter.grad = live = torch.ones(3)
    optimizer = MuSGD([parameter], 0.01)
    trainer.optimizers = [optimizer]

    def step():
        copy_out_live_gradients([parameter])
        return ClosureResult.from_training_step_output(torch.tensor(1.0), normalize=1)

    closure = Closure(step_fn=step, backward_fn=None, zero_grad_fn=optimizer.zero_grad)
    closure.closure = types.MethodType(lambda self, *args, **kwargs: self._step_fn(), closure)
    with patch.object(module, "_parameters_cleared_after_forward", return_value=(parameter,)):
        module.optimizer_step(0, 0, optimizer, closure)
    assert closure._step_fn is step
    assert parameter.grad is not live


def test_foreign_optimizer_zeroing_keeps_snapshots(native, certify) -> None:
    """A genuine zeroing function for another optimizer cannot clear this owner's gradients."""
    module, trainer, closure = native
    other = MuSGD(module.parameters(), 0.01)
    closure._zero_grad_fn = trainer.fit_loop.epoch_loop.automatic_optimization._make_zero_grad_fn(0, other)
    assert certify() == ()


def test_custom_objective_keeps_snapshots(native, certify) -> None:
    """A replacement objective can observe old gradients and retains their snapshots."""
    module, _, _ = native
    module._objective = lambda *args: None
    assert certify() == ()


@pytest.mark.parametrize("name", ["_alpha_tensor", "_anchor_grid", "_mark_dynamic", "_network_replay_eligible"])
@pytest.mark.parametrize("replaced", ["class", "instance"])
def test_replaced_module_helper_keeps_snapshots(
    native, certify, monkeypatch: pytest.MonkeyPatch, name: str, replaced: str
) -> None:
    """Any helper the module class defines keeps snapshots when replaced, without being listed.

    These helpers run inside ``training_step`` and once were missing from a hand-kept table; the
    certificate now reads every member of the class itself, so a class patch or an instance
    shadow of each one refuses the omission.
    """
    module, _, _ = native
    assert certify()
    if replaced == "class":
        monkeypatch.setattr(DetectionLitModule, name, lambda self, *args, **kwargs: None)
    else:
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None, raising=False)
    assert certify() == ()


def test_replaced_callback_base_keeps_snapshots(native, certify, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing the shared callback hook is still a new old-gradient observer."""
    _, trainer, _ = native
    trainer.callbacks.append(Callback())
    assert certify()
    monkeypatch.setattr(Callback, "on_before_zero_grad", lambda *args: None)
    assert certify() == ()


@pytest.mark.parametrize(
    "change", ["strategy_post", "precision_context", "precision_forward_context", "loop_zero", "log", "model_wrapper"]
)
def test_forward_observers_keep_snapshots(native, certify, change: str) -> None:
    """Public forward-time hooks and replaced zeroing helpers cannot retain static gradients."""
    module, trainer, _ = native
    assert certify()
    if change == "strategy_post":
        trainer.strategy.post_training_step = lambda: None
    elif change == "precision_context":
        trainer.precision_plugin.train_step_context = lambda: None
    elif change == "precision_forward_context":
        trainer.precision_plugin.forward_context = lambda: None
    elif change == "loop_zero":
        trainer.fit_loop.epoch_loop.automatic_optimization._optimizer_zero_grad = lambda *args: None
    elif change == "log":
        module.log = lambda *args, **kwargs: None
    elif change == "model_wrapper":
        trainer.strategy.model = torch.nn.Identity()
    assert certify() == ()


@pytest.mark.parametrize("replaced", ["class", "instance"])
def test_replaced_autocast_keeps_snapshots(native, certify, monkeypatch: pytest.MonkeyPatch, replaced: str) -> None:
    """A replaced autocast context, entered by the native mixed-precision forward context, keeps snapshots.

    ``MixedPrecision.forward_context`` enters ``autocast_context_manager()`` after the forward-time
    copy-out. A replacement that holds ``parameter.grad`` on exit would keep the static buffer the
    next replayed backward overwrites, so a class patch and an instance shadow both refuse.
    """
    _, trainer, _ = native
    trainer.strategy.precision_plugin = MixedPrecision("bf16-mixed", "cpu")
    assert certify()
    if replaced == "class":
        monkeypatch.setattr(MixedPrecision, "autocast_context_manager", lambda self: nullcontext())
    else:
        trainer.precision_plugin.autocast_context_manager = nullcontext
    assert certify() == ()


@pytest.mark.parametrize("dispatcher", ["_call_strategy_hook", "_call_lightning_module_hook", "_call_callback_hooks"])
def test_replaced_hook_dispatcher_keeps_snapshots(certify, monkeypatch: pytest.MonkeyPatch, dispatcher: str) -> None:
    """A replaced Lightning hook dispatcher keeps snapshots.

    The closure runs its forward through ``_call_strategy_hook`` and its zeroing hooks through
    ``_call_callback_hooks`` and ``_call_lightning_module_hook``, each looked up on the module at
    call time. A replacement there can observe the old gradients or skip zeroing.
    """
    assert certify()
    monkeypatch.setattr(trainer_call, dispatcher, lambda *args, **kwargs: None)
    assert certify() == ()


@pytest.mark.parametrize("change", ["strategy", "step", "optimizer"])
def test_foreign_optimizer_wrapper_keeps_snapshots(native, change: str) -> None:
    """An optimizer wrapper that can step outside the certified strategy keeps snapshots.

    ``LightningOptimizer.step`` routes through its own ``_strategy``, which Lightning sets to a weak
    proxy of the trainer's strategy. The trainer's own wrapper qualifies; one bound to a second
    strategy, one whose ``step`` is shadowed on the instance, and one over a different optimizer
    do not.
    """
    module, trainer, closure = native
    wrapper = trainer.strategy._lightning_optimizers[0]
    assert module._parameters_cleared_after_forward(closure, closure._step_fn, wrapper)
    other = SingleDeviceStrategy()
    if change == "strategy":
        wrapper = LightningOptimizer._to_lightning_optimizer(trainer.optimizers[0], other)
    elif change == "step":
        wrapper.step = lambda closure=None: None
    else:
        wrapper = LightningOptimizer(MuSGD(module.parameters(), 0.01))
        wrapper._strategy = trainer.strategy
    assert module._parameters_cleared_after_forward(closure, closure._step_fn, wrapper) == ()


@pytest.mark.parametrize("method", ["__call__", "closure"])
def test_replaced_closure_class_keeps_snapshots(certify, monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    """A class-level replacement of Lightning's closure call or body keeps snapshots."""
    assert certify()
    monkeypatch.setattr(Closure, method, lambda self, *args, **kwargs: None)
    assert certify() == ()


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ("native", None),
        ("inactive", _Refusal.MODULE),
        ("accumulation", _Refusal.TRAINER_CONTRACT),
        ("callback", _Refusal.FORWARD_OBSERVERS),
        ("missing_zero", _Refusal.ZEROING_CLOSURE),
        ("custom_zero", _Refusal.OPTIMIZER),
    ],
)
def test_refusal_names_first_failed_predicate(native, change: str, refusal: _Refusal | None) -> None:
    """The certificate reports the first refusing predicate as its constant token.

    Each change breaks exactly one predicate, so the token names the hazard the step was refused
    for; an unchanged native step reports none.
    """
    module, trainer, closure = native
    if change == "inactive":
        module._capture_network_active = False
    elif change == "accumulation":
        trainer.accumulate_grad_batches = 2
    elif change == "callback":

        class _Observer(Callback):
            def on_before_zero_grad(self, trainer, module, optimizer):
                raise AssertionError("not executed by the certificate")

        trainer.callbacks.append(_Observer())
    elif change == "missing_zero":
        closure._zero_grad_fn = None
    elif change == "custom_zero":
        trainer.optimizers[0].zero_grad = lambda: None
    wrapper = trainer.strategy._lightning_optimizers[0]
    assert module._native_closure_refusal(closure, closure._step_fn, wrapper) == refusal


def test_delegating_callback_keeps_snapshots(native, certify) -> None:
    """A callback proxy cannot hide its observer behind an inherited default hook."""
    _, trainer, _ = native

    class _DelegatingObserver(Callback):
        def __getattribute__(self, name):
            if name == "on_before_zero_grad":
                return lambda *args: None
            return super().__getattribute__(name)

    trainer.callbacks.append(_DelegatingObserver())
    assert certify() == ()


@pytest.mark.parametrize("observer", ["custom", "instance", "class"])
def test_profiler_observer_retains_old_gradient_values(native, monkeypatch: pytest.MonkeyPatch, observer: str) -> None:
    """A profiler observing the forward must retain an independent old-gradient snapshot.

    The forward stand-in replaces Lightning's strategy-hook dispatcher, so it is also registered as
    the certified dispatcher; otherwise the certificate refuses on it and never reaches the profiler.
    """
    module, trainer, closure = native
    parameter = next(module.parameters())
    parameter.grad = static = torch.ones_like(parameter)
    held = []

    def observe(action_name):
        if action_name == "training_step":
            held.append(parameter.grad)

    if observer == "custom":

        class _Observer(PassThroughProfiler):
            def stop(self, action_name):
                observe(action_name)

        trainer.profiler = _Observer()
    elif observer == "instance":
        trainer.profiler.stop = observe
    else:
        monkeypatch.setattr(PassThroughProfiler, "stop", lambda self, action_name: observe(action_name))

    def forward(*args, **kwargs):
        with trainer.profiler.profile("training_step"):
            copy_out_live_gradients([parameter])
        return torch.tensor(1.0, requires_grad=True)

    closure._backward_fn = lambda loss: static.fill_(2)
    with (
        patch("pytorch_lightning.trainer.call._call_strategy_hook", side_effect=forward) as dispatch,
        patch.dict(_NATIVE_TRAINER_CALLS, {"_call_strategy_hook": dispatch}),
    ):
        module.optimizer_step(0, 0, trainer.optimizers[0], closure)
    assert len(held) == 1
    assert held[0] is not static
    assert torch.equal(held[0].view(torch.uint8), torch.ones_like(static).view(torch.uint8))


def test_custom_precision_closure_keeps_snapshots(native) -> None:
    """A precision wrapper that bypasses zeroing cannot suppress the forward snapshot.

    The forward stand-in is registered as the certified strategy-hook dispatcher, so the replaced
    ``_wrap_closure`` alone decides the refusal.
    """
    module, trainer, closure = native
    parameter = next(module.parameters())
    parameter.grad = live = torch.ones_like(parameter)
    optimizer = trainer.optimizers[0]
    wrapped = LightningOptimizer._to_lightning_optimizer(optimizer, trainer.strategy)
    trainer.precision_plugin._wrap_closure = lambda model, optimizer, closure: closure._step_fn().closure_loss

    def forward(*args, **kwargs):
        copy_out_live_gradients([parameter])
        return torch.tensor(1.0)

    step = closure._step_fn
    with (
        patch("pytorch_lightning.trainer.call._call_strategy_hook", side_effect=forward) as dispatch,
        patch.dict(_NATIVE_TRAINER_CALLS, {"_call_strategy_hook": dispatch}),
    ):
        module.optimizer_step(0, 0, wrapped, closure)
    assert closure._step_fn is step
    assert parameter.grad is not live
