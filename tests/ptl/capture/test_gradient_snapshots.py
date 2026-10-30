# SPDX-License-Identifier: Apache-2.0
"""Old replay gradients survive unless a native closure is about to clear them."""

from __future__ import annotations

import functools
import types
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning
import torch
from markers import skip_unless_native_zeroing
from packaging.version import Version
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loops.optimization.automatic import Closure, ClosureResult
from pytorch_lightning.plugins.precision import MixedPrecision, Precision
from pytorch_lightning.profilers import PassThroughProfiler
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.trainer import call as trainer_call
from pytorch_lightning.trainer.states import RunningStage

from lucid_yolo.data.targets import Targets
from lucid_yolo.optim.musgd import MuSGD
from lucid_yolo.ptl import _native_closure
from lucid_yolo.ptl._graph_replay import (
    _CLEARING_GRADIENTS,
    _gradients_cleared_after_forward,
    copy_out_live_gradients,
)
from lucid_yolo.ptl._native_closure import (
    _NATIVE_TRAINER_CALLS,
    _native_update_step,
    _Refusal,
    _resolve_profile_wrapper_code,
)
from lucid_yolo.ptl.module import DetectionLitModule
from lucid_yolo.tasks import Task


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


@pytest.fixture
def stepping(native) -> Iterator[tuple[DetectionLitModule, MuSGD, Closure, list[frozenset[int]]]]:
    """Build Lightning's own closure over a real batch, with a probe on the forward's clear permission.

    The probe is a forward hook on a submodule, so it runs inside the real ``training_step`` without
    shadowing any method the certificate inspects; each forward appends the parameter ids whose
    snapshot is currently omitted.
    """
    module, trainer, _ = native
    optimizer = trainer.optimizers[0]
    targets = [
        Targets(boxes=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long)),
        Targets(boxes=torch.tensor([[8.0, 6.0, 48.0, 52.0]]), labels=torch.tensor([1])),
    ]
    kwargs = OrderedDict(batch=(torch.rand(2, 3, 64, 64), targets), batch_idx=0)
    closure = trainer.fit_loop.epoch_loop.automatic_optimization._make_closure(kwargs, optimizer, 0)
    permissions: list[frozenset[int]] = []
    handle = module.backbone.register_forward_hook(lambda *_: permissions.append(_CLEARING_GRADIENTS.get()))
    yield module, optimizer, closure, permissions
    handle.remove()


@pytest.fixture
def reported() -> Iterator[MagicMock]:
    """Stand in for the rank-zero info logger, patched where the module looks it up."""
    with patch("lucid_yolo.ptl.module.rank_zero_info") as log:
        yield log


@skip_unless_native_zeroing
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


class _ZeroingObserver(Callback):
    """A callback that overrides the zeroing hook the certificate refuses to skip past."""

    def on_before_zero_grad(self, trainer, module, optimizer):
        raise AssertionError("not executed by the certificate")


class _SubclassedModule(DetectionLitModule):
    """A subclass whose methods the certificate has not inspected."""


class _CustomPrecision(Precision):
    """A precision plugin of a type other than the two native ones."""


#: Every change that must withdraw the omission, by the contract it breaks; each takes ``(module, trainer, closure)``.
_ZEROING_CHANGES: dict[str, Callable[..., object]] = {
    "accumulation": lambda module, trainer, closure: setattr(trainer, "accumulate_grad_batches", 2),
    "manual": lambda module, trainer, closure: setattr(module, "automatic_optimization", False),
    "inactive": lambda module, trainer, closure: setattr(module, "_capture_network_active", False),
    "validation": lambda module, trainer, closure: setattr(trainer.state, "stage", RunningStage.VALIDATING),
    "zero_hook": lambda module, trainer, closure: setattr(module, "on_before_zero_grad", lambda optimizer: None),
    "zero_override": lambda module, trainer, closure: setattr(
        module, "optimizer_zero_grad", lambda epoch, batch_idx, optimizer: None
    ),
    "callback": lambda module, trainer, closure: trainer.callbacks.append(_ZeroingObserver()),
    "subclass": lambda module, trainer, closure: setattr(module, "__class__", _SubclassedModule),
    "custom_step": lambda module, trainer, closure: setattr(trainer.optimizers[0], "step", lambda closure=None: None),
    "step_pre_hook": lambda module, trainer, closure: trainer.optimizers[0].register_step_pre_hook(
        lambda optimizer, args, kwargs: None
    ),
    "custom_zero": lambda module, trainer, closure: setattr(trainer.optimizers[0], "zero_grad", lambda: None),
    "missing_zero": lambda module, trainer, closure: setattr(closure, "_zero_grad_fn", None),
    "foreign_zero_code": lambda module, trainer, closure: setattr(closure, "_zero_grad_fn", lambda: None),
    "skip_backward": lambda module, trainer, closure: setattr(
        trainer.fit_loop.epoch_loop.automatic_optimization, "_skip_backward", True
    ),
    "second_optimizer": lambda module, trainer, closure: setattr(
        trainer, "optimizers", [*trainer.optimizers, MuSGD(module.parameters(), 0.01)]
    ),
    "precision_type": lambda module, trainer, closure: setattr(
        trainer.strategy, "precision_plugin", _CustomPrecision()
    ),
    "precision_optimizer_step": lambda module, trainer, closure: setattr(
        trainer.precision_plugin, "optimizer_step", lambda *args, **kwargs: None
    ),
    "module_forward_hook": lambda module, trainer, closure: module.register_forward_hook(lambda *args: None),
    "module_forward_pre_hook": lambda module, trainer, closure: module.register_forward_pre_hook(lambda *args: None),
    "module_backward_hook": lambda module, trainer, closure: module.register_full_backward_hook(lambda *args: None),
}


@pytest.mark.parametrize("change", [pytest.param(change, id=name) for name, change in _ZEROING_CHANGES.items()])
@skip_unless_native_zeroing
def test_custom_zeroing_keeps_snapshots(native, certify, change: Callable[..., None]) -> None:
    """Each unsupported zeroing contract independently selects the original clone route."""
    assert certify()
    change(*native)
    assert certify() == ()


@skip_unless_native_zeroing
def test_shadowed_closure_keeps_snapshots(native, certify) -> None:
    """A closure whose own ``closure`` method was replaced on the instance keeps the clone route.

    ``type(closure)`` and the class's ``closure``/``__call__`` stay the native ones, so this is
    not caught by the class-identity checks alone; only the instance-dict check is.
    """
    _, _, closure = native
    assert certify()
    closure.closure = types.MethodType(lambda self, *args, **kwargs: self._step_fn(), closure)
    assert certify() == ()


@skip_unless_native_zeroing
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
@skip_unless_native_zeroing
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
    """The ``optimizer_step`` gate never wraps ``_step_fn`` of a closure with a shadowed ``closure`` method.

    This is the gate layer, not the certificate: ``type(closure)`` stays the native one, so only
    the instance-dict check of the shared closure shape catches it. The certificate is patched to
    accept, so the pre-filter in ``optimizer_step`` alone must leave the step function unwrapped
    and the snapshot copy in place.
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


@skip_unless_native_zeroing
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
@skip_unless_native_zeroing
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


@skip_unless_native_zeroing
def test_replaced_callback_base_keeps_snapshots(native, certify, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing the shared callback hook is still a new old-gradient observer."""
    _, trainer, _ = native
    trainer.callbacks.append(Callback())
    assert certify()
    monkeypatch.setattr(Callback, "on_before_zero_grad", lambda *args: None)
    assert certify() == ()


#: Public forward-time hooks and replaced zeroing helpers, by name; each takes ``(module, trainer, closure)``.
_OBSERVER_CHANGES: dict[str, Callable[..., object]] = {
    "strategy_post": lambda module, trainer, closure: setattr(trainer.strategy, "post_training_step", lambda: None),
    "precision_context": lambda module, trainer, closure: setattr(
        trainer.precision_plugin, "train_step_context", lambda: None
    ),
    "precision_forward_context": lambda module, trainer, closure: setattr(
        trainer.precision_plugin, "forward_context", lambda: None
    ),
    "loop_zero": lambda module, trainer, closure: setattr(
        trainer.fit_loop.epoch_loop.automatic_optimization, "_optimizer_zero_grad", lambda *args: None
    ),
    "log": lambda module, trainer, closure: setattr(module, "log", lambda *args, **kwargs: None),
    "model_wrapper": lambda module, trainer, closure: setattr(trainer.strategy, "model", torch.nn.Identity()),
}


@pytest.mark.parametrize("change", [pytest.param(change, id=name) for name, change in _OBSERVER_CHANGES.items()])
@skip_unless_native_zeroing
def test_forward_observers_keep_snapshots(native, certify, change: Callable[..., None]) -> None:
    """Public forward-time hooks and replaced zeroing helpers cannot retain static gradients."""
    assert certify()
    change(*native)
    assert certify() == ()


@pytest.mark.parametrize("replaced", ["class", "instance"])
@skip_unless_native_zeroing
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
@skip_unless_native_zeroing
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
@skip_unless_native_zeroing
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
@skip_unless_native_zeroing
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
@skip_unless_native_zeroing
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


@skip_unless_native_zeroing
def test_active_snapshot_skip_logs_once_per_fit(certify, reported: MagicMock) -> None:
    """Many native steps of one fit log the active verdict once, at the first.

    The certificate is evaluated at every step's forward; the line must not repeat per step, or a
    long fit would flood the log with a fact that did not change.
    """
    for _ in range(3):
        assert certify()
    reported.assert_called_once()
    assert "skip active" in reported.call_args.args[0]


@skip_unless_native_zeroing
def test_certificate_flip_after_first_step_stays_silent(native, certify, reported: MagicMock) -> None:
    """A later step that falls back does not add a second line to the fit.

    The first step reported active; accumulation changing afterwards still keeps the original
    snapshots, but the one-line-per-fit contract holds and the hot path formats nothing.
    """
    _, trainer, _ = native
    assert certify()
    trainer.accumulate_grad_batches = 2
    assert certify() == ()
    reported.assert_called_once()
    assert "skip active" in reported.call_args.args[0]


@pytest.mark.parametrize(
    ("break_certificate", "refusal"),
    [
        pytest.param(
            lambda module, trainer, closure: setattr(trainer, "accumulate_grad_batches", 2),
            _Refusal.TRAINER_CONTRACT,
            id="trainer-contract",
        ),
        pytest.param(
            lambda module, trainer, closure: setattr(closure, "_zero_grad_fn", None),
            _Refusal.ZEROING_CLOSURE,
            id="zeroing-closure",
        ),
        pytest.param(
            lambda module, trainer, closure: setattr(trainer.optimizers[0], "zero_grad", lambda: None),
            _Refusal.OPTIMIZER,
            id="optimizer",
        ),
        pytest.param(
            lambda module, trainer, closure: (
                setattr(trainer, "accumulate_grad_batches", 2),
                setattr(trainer.optimizers[0], "zero_grad", lambda: None),
            ),
            _Refusal.TRAINER_CONTRACT,
            id="first-of-two",
        ),
    ],
)
@skip_unless_native_zeroing
def test_refused_snapshot_skip_logs_first_failed_check(
    native, certify, reported: MagicMock, break_certificate: Callable[..., None], refusal: _Refusal
) -> None:
    """A refused certificate logs the constant token of the first predicate that failed.

    The last case breaks two predicates; only the one evaluated first is named, so a user fixing
    the report sees the next refusal on the following run instead of a misleading list.
    """
    break_certificate(*native)
    assert certify() == ()
    reported.assert_called_once()
    assert reported.call_args.args[0].endswith(f"skip inactive; first failed check: {refusal.value}.")


@pytest.mark.parametrize("stage", ["fit", "validate"])
@pytest.mark.filterwarnings("ignore:capture_network=True is ignored")
@skip_unless_native_zeroing
def test_setup_rearms_snapshot_skip_log_for_fit_only(native, certify, reported: MagicMock, stage: str) -> None:
    """Setting up ``fit`` again starts a new report; another stage leaves the fit's report spent.

    ``setup`` clears the CPU fixture's activation, as it would on any non-CUDA run, so the
    activation the fixture stands in for is restored before the next step.
    """
    module, _, _ = native
    assert certify()
    module.setup(stage)
    module._capture_network_active = True
    assert certify()
    assert reported.call_count == (2 if stage == "fit" else 1)


@pytest.mark.parametrize(
    "silence",
    [
        pytest.param(lambda module: setattr(module, "_capture_network_active", False), id="capture-off"),
        pytest.param(lambda module: setattr(module, "_task", "keypoints"), id="non-detect-task"),
    ],
)
def test_inapplicable_snapshot_skip_logs_nothing(
    native, certify, reported: MagicMock, silence: Callable[[DetectionLitModule], None]
) -> None:
    """A module the fast path cannot serve never logs a verdict about it.

    Network replay off, or a task other than detection, means the certificate has nothing to report
    and a line would only suggest a fast path the run never had.
    """
    module, _, _ = native
    silence(module)
    assert certify() == ()
    reported.assert_not_called()


@skip_unless_native_zeroing
def test_foreign_closure_logs_zeroing_check_once(native, reported: MagicMock) -> None:
    """A closure whose own ``closure`` method was replaced is reported by the pre-filter.

    The shape check sits in ``optimizer_step`` before the certificate runs, so it must report on
    its own; the zeroing-closure predicate owns this shape and is the first that fails here.
    """
    module, trainer, _ = native
    parameter = torch.nn.Parameter(torch.zeros(3))
    optimizer = MuSGD([parameter], 0.01)
    trainer.optimizers = [optimizer]
    step = lambda: ClosureResult.from_training_step_output(torch.tensor(1.0), normalize=1)  # noqa: E731
    closure = Closure(step_fn=step, backward_fn=None, zero_grad_fn=optimizer.zero_grad)
    closure.closure = types.MethodType(lambda self, *args, **kwargs: self._step_fn(), closure)
    module.optimizer_step(0, 0, optimizer, closure)
    module.optimizer_step(0, 1, optimizer, closure)
    reported.assert_called_once()
    assert reported.call_args.args[0].endswith(f"first failed check: {_Refusal.ZEROING_CLOSURE.value}.")


@skip_unless_native_zeroing
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


def test_unsupported_lightning_minor_refuses_certificate(native, certify, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Lightning minor outside the inspected 2.4-2.6 contract yields no clearing set.

    Lightning is not capped, so a newer minor reaches users with the runtime gate off. The
    certificate must name the trainer contract as the first failed check and clear nothing.
    """
    module, trainer, closure = native
    monkeypatch.setattr(_native_closure, "_ZEROING_RUNTIME_SUPPORTED", False)
    wrapper = trainer.strategy._lightning_optimizers[0]
    assert module._native_closure_refusal(closure, closure._step_fn, wrapper) == _Refusal.TRAINER_CONTRACT
    assert certify() == ()


def test_unsupported_lightning_minor_steps_with_snapshots(stepping, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real ``optimizer_step`` on an unsupported Lightning minor reaches the forward with no omission permission.

    The closure, optimizer and certificate are all unpatched; only the runtime gate is off, so
    the forward must run through the original snapshot route.
    """
    module, optimizer, closure, permissions = stepping
    monkeypatch.setattr(_native_closure, "_ZEROING_RUNTIME_SUPPORTED", False)
    step = closure._step_fn
    module.optimizer_step(0, 0, optimizer, closure)
    assert permissions == [frozenset()]
    assert closure._step_fn is step


def test_runtime_gate_tracks_lightning_2_4_to_2_6() -> None:
    """The snapshot-skip gate is open exactly on the Lightning minors whose closure was inspected.

    The expectation is spelled out here from the installed version, not read from the gate, so a
    changed production set fails this test instead of silently skipping every fast-path case.
    """
    minor = Version(pytorch_lightning.__version__).release[:2]
    assert (minor in {(2, 4), (2, 5), (2, 6)}) == _native_closure._ZEROING_RUNTIME_SUPPORTED


@skip_unless_native_zeroing
def test_real_closure_scopes_permission_to_optimizer_parameters(stepping) -> None:
    """An unpatched ``optimizer_step`` over Lightning's own closure omits snapshots for the live optimizer parameters.

    Nothing is patched: the certificate, the closure, the optimizer and the strategy are the
    native ones, so this joins ``optimizer_step`` to the real certificate. A parameter dropped
    from the optimizer's group keeps its snapshot, and the step function is restored afterwards.
    """
    module, optimizer, closure, permissions = stepping
    removed = optimizer.param_groups[0]["params"].pop()
    step = closure._step_fn
    module.optimizer_step(0, 0, optimizer, closure)
    assert permissions == [frozenset(id(parameter) for parameter in optimizer.param_groups[0]["params"])]
    assert id(removed) not in permissions[0]
    assert closure._step_fn is step
    assert _CLEARING_GRADIENTS.get() == frozenset()


@pytest.mark.parametrize("task", [task.value for task in Task if task is not Task.DETECT])
@skip_unless_native_zeroing
def test_non_detect_task_keeps_snapshots(native, certify, task: str) -> None:
    """Every task other than detection keeps the original snapshots.

    The fast path is certified for the detection objective alone, so segmentation, oriented boxes
    and keypoints replay with their snapshots. The cases come from the ``Task`` enum, so a task
    added later is held to the same guard without editing this test.
    """
    module, _, _ = native
    assert certify()
    module._task = task
    assert certify() == ()
