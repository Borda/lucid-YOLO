# SPDX-License-Identifier: Apache-2.0
"""Old replay gradients survive unless a native closure is about to clear them."""

from __future__ import annotations

import functools
import types
from collections import OrderedDict
from collections.abc import Iterator
from unittest.mock import patch

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loops.optimization.automatic import Closure, ClosureResult
from pytorch_lightning.profilers import PassThroughProfiler
from pytorch_lightning.trainer.states import RunningStage

from lucid_yolo.optim.musgd import MuSGD
from lucid_yolo.ptl._graph_replay import _gradients_cleared_after_forward, copy_out_live_gradients
from lucid_yolo.ptl.module import DetectionLitModule, _native_update_step


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


def test_native_closure_names_live_optimizer_parameters(native) -> None:
    """The native plan follows group membership rather than every registered parameter."""
    module, trainer, closure = native
    parameters = tuple(module.parameters())
    removed = trainer.optimizers[0].param_groups[0]["params"].pop()
    clearing = module._parameters_cleared_after_forward(closure, closure._step_fn)
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
def test_custom_zeroing_keeps_snapshots(native, change: str) -> None:
    """Each unsupported zeroing contract independently selects the original clone route."""
    module, trainer, closure = native
    assert module._parameters_cleared_after_forward(closure, closure._step_fn)
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
    assert module._parameters_cleared_after_forward(closure, closure._step_fn) == ()


def test_shadowed_closure_keeps_snapshots(native) -> None:
    """A closure whose own ``closure`` method was replaced on the instance keeps the clone route.

    ``type(closure)`` and the class's ``closure``/``__call__`` stay the native ones, so this is
    not caught by the class-identity checks alone; only the instance-dict check is.
    """
    module, _, closure = native
    assert module._parameters_cleared_after_forward(closure, closure._step_fn)
    closure.closure = types.MethodType(lambda self, *args, **kwargs: self._step_fn(), closure)
    assert module._parameters_cleared_after_forward(closure, closure._step_fn) == ()


def test_scheduler_wrapper_keeps_native_update_step(native) -> None:
    """LambdaLR's real wrapper qualifies; an arbitrary functools wrapper does not."""
    module, trainer, closure = native
    optimizer = trainer.optimizers[0]
    torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    assert _native_update_step(optimizer)
    assert module._parameters_cleared_after_forward(closure, closure._step_fn)
    native_step = optimizer.step

    @functools.wraps(native_step)
    def custom_step(*args, **kwargs):
        return native_step(*args, **kwargs)

    optimizer.step = custom_step
    assert not _native_update_step(optimizer)


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

    ``type(closure)`` and the class's ``closure``/``__call__`` stay the native ones, so this is
    not caught by the class-identity checks alone; only the instance-dict check is.
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


def test_foreign_optimizer_zeroing_keeps_snapshots(native) -> None:
    """A genuine zeroing function for another optimizer cannot clear this owner's gradients."""
    module, trainer, closure = native
    other = MuSGD(module.parameters(), 0.01)
    closure._zero_grad_fn = trainer.fit_loop.epoch_loop.automatic_optimization._make_zero_grad_fn(0, other)
    assert module._parameters_cleared_after_forward(closure, closure._step_fn) == ()


def test_custom_objective_keeps_snapshots(native) -> None:
    """A replacement objective can observe old gradients and retains their snapshots."""
    module, _, closure = native
    module._objective = lambda *args: None
    assert module._parameters_cleared_after_forward(closure, closure._step_fn) == ()


def test_replaced_callback_base_keeps_snapshots(native, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing the shared callback hook is still a new old-gradient observer."""
    module, trainer, closure = native
    trainer.callbacks.append(Callback())
    assert module._parameters_cleared_after_forward(closure, closure._step_fn)
    monkeypatch.setattr(Callback, "on_before_zero_grad", lambda *args: None)
    assert module._parameters_cleared_after_forward(closure, closure._step_fn) == ()


@pytest.mark.parametrize(
    "change", ["strategy_post", "precision_context", "precision_forward_context", "loop_zero", "log", "model_wrapper"]
)
def test_forward_observers_keep_snapshots(native, change: str) -> None:
    """Public forward-time hooks and replaced zeroing helpers cannot retain static gradients."""
    module, trainer, closure = native
    assert module._parameters_cleared_after_forward(closure, closure._step_fn)
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
    assert module._parameters_cleared_after_forward(closure, closure._step_fn) == ()


def test_delegating_callback_keeps_snapshots(native) -> None:
    """A callback proxy cannot hide its observer behind an inherited default hook."""
    module, trainer, closure = native

    class _DelegatingObserver(Callback):
        def __getattribute__(self, name):
            if name == "on_before_zero_grad":
                return lambda *args: None
            return super().__getattribute__(name)

    trainer.callbacks.append(_DelegatingObserver())
    assert module._parameters_cleared_after_forward(closure, closure._step_fn) == ()


@pytest.mark.parametrize("observer", ["custom", "instance", "class"])
def test_profiler_observer_retains_old_gradient_values(native, monkeypatch: pytest.MonkeyPatch, observer: str) -> None:
    """A profiler observing the forward must retain an independent old-gradient snapshot."""
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
    with patch("pytorch_lightning.trainer.call._call_strategy_hook", side_effect=forward):
        module.optimizer_step(0, 0, trainer.optimizers[0], closure)
    assert len(held) == 1
    assert held[0] is not static
    assert torch.equal(held[0].view(torch.uint8), torch.ones_like(static).view(torch.uint8))


def test_custom_precision_closure_keeps_snapshots(native) -> None:
    """A precision wrapper that bypasses zeroing cannot suppress the forward snapshot."""
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
    with patch("pytorch_lightning.trainer.call._call_strategy_hook", side_effect=forward):
        module.optimizer_step(0, 0, wrapped, closure)
    assert closure._step_fn is step
    assert parameter.grad is not live
