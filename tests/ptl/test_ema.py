# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`~lucid_yolo.ptl.callbacks.EMACallback` (WP-037).

Covers the DoD ``test_shadow_updates`` — after a run of manual
:meth:`~lucid_yolo.ptl.callbacks.EMACallback.on_train_batch_end` calls against a
hand-mutated tiny module the shadow equals the EMA recursion recomputed in the
test with the same warmup-ramp formula — plus the supporting contracts: the
warmup ramp starts far below the nominal decay; the validation swap loads the
shadow for the duration of validation and restores the live weights after;
floating-point buffers (batch-norm running stats) are tracked while integer
buffers are skipped; ``state_dict``/``load_state_dict`` survive a save/mutate/
load round trip; ``update_every`` gates updates; and a two-epoch Lightning run
with the callback attached completes with the shadow diverged from the live
weights.

:class:`TestOptimizerStepCadence` pins the WP-158 fix for the audit's A05/M-14:
the blend follows *optimizer steps*, so ``accumulate_grad_batches`` changes how
many batches a run takes but not how many times the shadow moves. Its expected
values are folded over the post-step weights a real
:class:`~pytorch_lightning.Trainer` produced, recorded through the module's own
``optimizer_step`` hook so the expectation does not reuse the callback's notion
of when a step happened. :class:`TestRestoreOnFailure` pins A06 (an exception
inside validation must not leave the shadow installed) and :class:`TestSavedWeights`
pins L-48 (a checkpoint written mid-validation holds raw weights, not the shadow).

The module under test is a tiny ``Linear``-free stub carrying one flat parameter
and a :class:`~torch.nn.BatchNorm1d` so both the parameter and buffer paths are
exercised; the unit tests drive the callback hooks directly with a stub trainer
whose ``global_step`` the test advances, and the integration tests wire a real
:class:`~pytorch_lightning.Trainer` over a synthetic in-memory dataset.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from lucid_yolo.ptl import EMACallback

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

#: Feature width of the tiny module's parameter and batch-norm layer.
_FEATURES = 4
#: A dummy trainer for direct hook calls that discard it (the validation swap pair).
_DUMMY_TRAINER = MagicMock(spec=Trainer)
#: Message of the deliberate validation failure the A06 restoration tests drive.
_VALIDATION_FAILURE = "deliberate validation failure"
#: Training batches the cadence tests run: four, matching the audit's own reproduction, so
#: accumulation 2 halves them to two optimizer steps and update_every 2 halves those again.
_TRAIN_BATCHES = 4


class _StepTrainer:
    """A trainer stub exposing the one attribute the update hook reads: ``global_step``.

    :meth:`~lucid_yolo.ptl.callbacks.EMACallback.on_train_batch_end` blends only when
    ``global_step`` has moved since the previous batch, so a unit test that wants to drive
    N updates has to advance a counter rather than call the hook N times. A ``MagicMock``
    would not do: its ``global_step`` compares unequal to everything, which would make
    every batch look like an optimizer step and hide exactly the regression under test.
    """

    def __init__(self) -> None:
        self.global_step = 0

    def step(self) -> None:
        """Mark one optimizer step as completed."""
        self.global_step += 1


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed torch before each test so module init and synthetic data are reproducible."""
    torch.manual_seed(0)
    yield


class _TinyEMAModule(LightningModule):
    """One flat parameter plus a ``BatchNorm1d`` — exercises the param and buffer paths.

    The batch-norm layer contributes the floating-point ``running_mean``/``running_var``
    buffers the EMA must track and the integer ``num_batches_tracked`` buffer it must skip;
    ``training_step`` couples the parameter to a real gradient so a live run moves it.
    """

    def __init__(self, fail_validation: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(_FEATURES))
        self.bn = nn.BatchNorm1d(_FEATURES)
        self.fail_validation = fail_validation
        #: Every EMA-tracked tensor as it stood immediately after each optimizer step,
        #: recorded from Lightning's own ``optimizer_step`` hook so the cadence tests can
        #: fold the EMA recursion over a trajectory the callback had no part in defining.
        self.post_step_tensors: list[dict[str, Tensor]] = []

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Any, optimizer_closure: Any = None) -> None:
        """Take the optimizer step, then snapshot every floating tensor it just moved."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        snapshot = {name: param.detach().clone() for name, param in self.named_parameters()}
        snapshot |= {
            name: buffer.detach().clone() for name, buffer in self.named_buffers() if buffer.is_floating_point()
        }
        self.post_step_tensors.append(snapshot)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Return a scalar loss driving the parameter toward one while training batch-norm.

        The batch-norm term keeps its parameters (and, through the forward pass, its running
        buffers) in play; the squared term gives ``weight`` a non-zero gradient from its zero
        initialisation, so a live run visibly moves it away from the EMA shadow.
        """
        del batch_idx
        images, _labels = batch
        return self.bn(images).pow(2).mean() + (self.weight - 1.0).pow(2).sum()

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Return a scalar so the val loop runs, or raise when the module was built to fail.

        The failing variant models the A06 scenario: something inside the validation loop
        throws *after* the callback has installed the shadow, so the run unwinds through a
        path the ordinary ``on_validation_end`` restore never reaches.
        """
        del batch_idx
        if self.fail_validation:
            raise RuntimeError(_VALIDATION_FAILURE)
        images, _labels = batch
        return (self.weight * self.bn(images).mean()).sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return a non-trivial-LR SGD so a live run perturbs the weights."""
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _SyntheticVectorDataset(Dataset[tuple[Tensor, int]]):
    """Tiny in-memory dataset of ``(feature_vector, label)`` samples for the smoke run."""

    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        """Return the sample count."""
        return self._size

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Return one ``(feature_vector, label)`` sample."""
        del index
        return torch.randn(_FEATURES), 0


def _fitted_callback(module: LightningModule, **kwargs: float) -> EMACallback:
    """Build a callback and run ``on_fit_start`` so its shadow is initialised.

    Examples:
        >>> module = _TinyEMAModule()
        >>> callback = _fitted_callback(module, decay=0.9, tau=1)
        >>> callback._shadow is not None
        True
    """
    callback = EMACallback(**kwargs)  # type: ignore[arg-type]
    callback.on_fit_start(_DUMMY_TRAINER, module)
    return callback


def _ema_recursion(callback: EMACallback, start: Tensor, trajectory: list[Tensor]) -> Tensor:
    """Fold the warmup-ramped EMA over ``trajectory``, honoring the callback's ``update_every``.

    The trajectory is the sequence of values the tracked tensor held immediately after each
    optimizer step. Only every ``update_every``-th entry blends, and the ramp is evaluated at
    the *update* count rather than the step count — the same two facts the implementation
    encodes, restated here in the flat form a reader can check by eye.

    Examples:
        >>> callback = EMACallback(decay=0.9, tau=1)
        >>> start = torch.zeros(1)
        >>> float(_ema_recursion(callback, start, [torch.ones(1)])[0]) > 0.0
        True
    """
    expected = start.clone()
    updates = 0
    for index, value in enumerate(trajectory, start=1):
        if index % callback.update_every != 0:
            continue
        updates += 1
        decay = callback._decay_at(updates)
        expected = decay * expected + (1.0 - decay) * value
    return expected


def _batch_end(
    callback: EMACallback,
    module: LightningModule,
    trainer: _StepTrainer,
    *,
    optimizer_stepped: bool = True,
) -> None:
    """Drive one training batch through the callback, with or without an optimizer step.

    ``optimizer_stepped=False`` is the accumulation case — a batch whose gradients were
    only accumulated, leaving ``trainer.global_step`` where it was.

    Examples:
        >>> module = _TinyEMAModule()
        >>> callback, trainer = _fitted_callback(module, decay=0.9, tau=1), _StepTrainer()
        >>> _batch_end(callback, module, trainer)
        >>> callback._num_updates
        1
        >>> _batch_end(callback, module, trainer, optimizer_stepped=False)
        >>> callback._num_updates
        1
    """
    if optimizer_stepped:
        trainer.step()
    callback.on_train_batch_end(trainer, module, None, None, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_steps", [pytest.param(1, id="one-step"), pytest.param(5, id="five-steps")])
def test_shadow_updates(num_steps: int) -> None:
    """The shadow equals the warmup-ramped EMA recursion recomputed step by step."""
    module = _TinyEMAModule()
    callback = _fitted_callback(module, decay=0.99, tau=50)
    trainer = _StepTrainer()
    expected = module.weight.detach().clone()
    for step in range(1, num_steps + 1):
        with torch.no_grad():
            module.weight.copy_(torch.full((_FEATURES,), float(step)))
        _batch_end(callback, module, trainer)
        decay = callback._decay_at(step)
        expected = decay * expected + (1.0 - decay) * module.weight.detach()
    assert callback._shadow is not None
    assert torch.allclose(callback._shadow["weight"], expected)


def test_warmup_ramp_starts_far_below_nominal_decay() -> None:
    """The first-step effective decay is a small fraction of the nominal decay."""
    callback = EMACallback(decay=0.9999, tau=2000)
    first_step_decay = callback._decay_at(1)
    assert first_step_decay == pytest.approx(0.9999 * (1.0 - math.exp(-1.0 / 2000)))
    assert first_step_decay * 100 < callback.decay


def test_validation_swap_loads_shadow_then_restores_live() -> None:
    """During validation the module holds the shadow; afterwards the live weights return."""
    module = _TinyEMAModule()
    callback = _fitted_callback(module)
    assert callback._shadow is not None
    shadow_value = torch.full((_FEATURES,), 7.0)
    callback._shadow["weight"].copy_(shadow_value)
    with torch.no_grad():
        module.weight.copy_(torch.full((_FEATURES,), -3.0))
    live_before = module.weight.detach().clone()
    callback.on_validation_start(_DUMMY_TRAINER, module)
    assert torch.allclose(module.weight.detach(), shadow_value)
    callback.on_validation_end(_DUMMY_TRAINER, module)
    assert torch.allclose(module.weight.detach(), live_before)


def test_float_buffers_tracked_integer_buffers_skipped() -> None:
    """A batch-norm running mean is EMA-tracked; the integer step counter is not shadowed."""
    module = _TinyEMAModule()
    callback = _fitted_callback(module, decay=0.9, tau=1)
    assert callback._shadow is not None
    assert "bn.num_batches_tracked" not in callback._shadow
    before = callback._shadow["bn.running_mean"].clone()
    new_mean = torch.full((_FEATURES,), 5.0)
    with torch.no_grad():
        module.bn.running_mean.copy_(new_mean)
    _batch_end(callback, module, _StepTrainer())
    decay = callback._decay_at(1)
    expected = decay * before + (1.0 - decay) * new_mean
    assert torch.allclose(callback._shadow["bn.running_mean"], expected)


def test_state_dict_round_trip_restores_shadow_and_counters() -> None:
    """Saving, mutating, then loading restores the shadow tensors and update counters."""
    module = _TinyEMAModule()
    callback = _fitted_callback(module)
    _batch_end(callback, module, _StepTrainer())
    saved = callback.state_dict()
    assert callback._shadow is not None
    original_weight = callback._shadow["weight"].clone()
    callback._shadow["weight"].add_(100.0)
    callback._num_updates = 999
    callback._steps_seen = 999
    callback.load_state_dict(saved)
    assert callback._num_updates == 1
    assert callback._steps_seen == 1
    assert torch.allclose(callback._shadow["weight"], original_weight)


def test_legacy_batches_seen_counter_is_read_as_steps_seen() -> None:
    """A checkpoint predating the optimizer-step cadence still resumes its ``update_every`` phase.

    Before WP-158 the ``update_every`` counter was written under ``batches_seen``. Resuming
    such a run must continue the gate where it stood rather than silently restart it at
    zero, which would shift every subsequent update by one step.
    """
    module = _TinyEMAModule()
    callback = _fitted_callback(module, update_every=2)
    legacy = {"shadow": None, "num_updates": 3, "batches_seen": 7}
    callback.load_state_dict(legacy)
    assert callback._steps_seen == 7
    assert callback._num_updates == 3


def test_update_every_skips_intermediate_steps() -> None:
    """With ``update_every=2`` the first step is skipped and the second applies an update."""
    module = _TinyEMAModule()
    callback = _fitted_callback(module, decay=0.9, tau=1, update_every=2)
    assert callback._shadow is not None
    initial = callback._shadow["weight"].clone()
    trainer = _StepTrainer()
    _batch_end(callback, module, trainer)
    assert callback._num_updates == 0
    assert torch.allclose(callback._shadow["weight"], initial)
    with torch.no_grad():
        module.weight.copy_(torch.full((_FEATURES,), 9.0))
    _batch_end(callback, module, trainer)
    assert callback._num_updates == 1
    assert not torch.allclose(callback._shadow["weight"], initial)


def test_two_epoch_run_diverges_shadow_from_live() -> None:
    """A two-epoch Trainer run with the callback attached leaves the shadow off the live weights."""
    module = _TinyEMAModule()
    callback = EMACallback(decay=0.9, tau=1)
    loader: DataLoader[tuple[Tensor, int]] = DataLoader(_SyntheticVectorDataset(8), batch_size=2)
    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        callbacks=[callback],
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.state.finished
    assert callback._shadow is not None
    assert callback._num_updates > 0
    assert not torch.allclose(callback._shadow["weight"], module.weight.detach())


def _cadence_trainer(
    callback: EMACallback,
    accumulate: int,
    extra_callbacks: list[Callback] | None = None,
    max_epochs: int = 1,
    **kwargs: Any,
) -> Trainer:
    """Build a one-epoch CPU trainer over ``_TRAIN_BATCHES`` batches at a given accumulation.

    Checkpointing is off unless ``extra_callbacks`` brings a
    :class:`~pytorch_lightning.callbacks.ModelCheckpoint`, so the cadence runs write nothing
    to disk while the L-48 test can still ask for a real saved file.

    Examples:
        >>> _cadence_trainer(EMACallback(), accumulate=2).accumulate_grad_batches
        2
    """
    extras = extra_callbacks or []
    return Trainer(
        max_epochs=max_epochs,
        limit_train_batches=_TRAIN_BATCHES,
        accumulate_grad_batches=accumulate,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=bool(extras),
        callbacks=[callback, *extras],
        **kwargs,
    )


class TestOptimizerStepCadence:
    """The blend follows optimizer steps, so gradient accumulation does not multiply updates.

    The audit (A05/M-14) reproduced the opposite with a real Trainer: four batches at
    ``accumulate_grad_batches=2`` gave two optimizer steps but four EMA updates, because the
    callback counted batches. ``tau`` and ``update_every`` are both documented in optimizer
    steps, so that silently re-tuned the warmup ramp of every accumulated run.
    """

    @pytest.mark.parametrize(
        ("accumulate", "expected_steps"),
        [pytest.param(1, 4, id="no-accumulation"), pytest.param(2, 2, id="accumulate-two")],
    )
    @pytest.mark.parametrize("update_every", [pytest.param(1, id="every-step"), pytest.param(2, id="every-other")])
    def test_matches_hand_computed_recursion(self, accumulate: int, expected_steps: int, update_every: int) -> None:
        """Update count equals optimizer steps, and the shadow equals the folded recursion.

        Four batches run at accumulation 1 and 2, each with ``update_every`` 1 and 2. The
        expectation is folded over the weights the module itself recorded after each real
        optimizer step, so it is independent of the callback's own step accounting; both the
        parameter and the batch-norm running buffer are checked, since the callback blends
        floating buffers on the same schedule.
        """
        module = _TinyEMAModule()
        callback = EMACallback(decay=0.9, tau=1, update_every=update_every)
        loader: DataLoader[tuple[Tensor, int]] = DataLoader(_SyntheticVectorDataset(8), batch_size=2)
        initial = {name: tensor.detach().clone() for name, tensor in callback._ema_tensors(module)}

        _cadence_trainer(callback, accumulate).fit(module, train_dataloaders=loader)

        assert len(module.post_step_tensors) == expected_steps
        assert callback._steps_seen == expected_steps
        assert callback._num_updates == expected_steps // update_every
        assert callback._shadow is not None
        for name in ("weight", "bn.running_mean"):
            trajectory = [snapshot[name] for snapshot in module.post_step_tensors]
            expected = _ema_recursion(callback, initial[name], trajectory)
            assert torch.allclose(callback._shadow[name], expected), name

    def test_resume_continues_the_same_average_under_accumulation(self, tmp_path: Path) -> None:
        """A save/resume round trip at accumulation 2 continues the update count, not restarts it.

        Resuming re-seeds the optimizer-step watermark from the restored
        ``trainer.global_step``; getting that wrong would either replay the first batch of the
        resumed epoch as a spurious update or stall the counter entirely.
        """
        module = _TinyEMAModule()
        callback = EMACallback(decay=0.9, tau=1)
        loader: DataLoader[tuple[Tensor, int]] = DataLoader(_SyntheticVectorDataset(8), batch_size=2)
        checkpoint = tmp_path / "resume.ckpt"
        first_leg = _cadence_trainer(callback, accumulate=2)
        first_leg.fit(module, train_dataloaders=loader)
        first_leg.save_checkpoint(checkpoint)
        first_leg_updates = callback._num_updates

        resumed = EMACallback(decay=0.9, tau=1)
        _cadence_trainer(resumed, accumulate=2, max_epochs=2).fit(
            _TinyEMAModule(), train_dataloaders=loader, ckpt_path=str(checkpoint)
        )

        assert first_leg_updates == 2
        assert resumed._num_updates == 2 * first_leg_updates
        assert resumed._steps_seen == 2 * first_leg_updates


class TestRestoreOnFailure:
    """A validation failure must not leave the EMA shadow wearing the live model's identity.

    Restoration used to live only in ``on_validation_end``, which an exception raised inside
    the validation loop never reaches (audit A06): the module was left holding the shadow
    while the trained weights survived only in the callback's private backup, so any
    handler that inspected, saved or resumed from the module got the average instead.
    """

    def test_caught_validation_failure_leaves_raw_weights_installed(self) -> None:
        """Every floating tensor equals its pre-swap raw value and the backup is cleared.

        The run trains first so the shadow has genuinely diverged from the live weights —
        against an untouched shadow the assertion would hold for the wrong reason.
        """
        module = _TinyEMAModule(fail_validation=True)
        callback = EMACallback(decay=0.9, tau=1)
        loader: DataLoader[tuple[Tensor, int]] = DataLoader(_SyntheticVectorDataset(8), batch_size=2)
        trainer = _cadence_trainer(callback, accumulate=1, limit_val_batches=1)

        with pytest.raises(RuntimeError, match=_VALIDATION_FAILURE):
            trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

        assert callback._shadow is not None
        assert callback._backup is None
        raw = module.post_step_tensors[-1]
        for name, live in callback._ema_tensors(module):
            assert torch.allclose(live.detach(), raw[name]), name
        assert not torch.allclose(callback._shadow["weight"], raw["weight"])

    def test_restoration_is_idempotent(self) -> None:
        """A second restore after the normal one does not re-install stale weights.

        ``on_exception`` can fire after ``on_validation_end`` already ran — a failure later
        in the epoch, say. Clearing the backup is what makes that second call a no-op rather
        than a copy of weights the training loop has since moved on from.
        """
        module = _TinyEMAModule()
        callback = _fitted_callback(module, decay=0.9, tau=1)
        assert callback._shadow is not None
        callback._shadow["weight"].copy_(torch.full((_FEATURES,), 7.0))
        with torch.no_grad():
            module.weight.copy_(torch.full((_FEATURES,), -3.0))
        callback.on_validation_start(_DUMMY_TRAINER, module)
        callback.on_validation_end(_DUMMY_TRAINER, module)
        with torch.no_grad():
            module.weight.copy_(torch.full((_FEATURES,), 5.0))

        callback.on_exception(_DUMMY_TRAINER, module, RuntimeError(_VALIDATION_FAILURE))

        assert callback._backup is None
        assert torch.allclose(module.weight.detach(), torch.full((_FEATURES,), 5.0))


class TestSavedWeights:
    """A checkpoint written during validation holds the raw weights, not the EMA shadow.

    ``ModelCheckpoint`` also saves from ``on_validation_end``, and Lightning reorders it to
    the end of the callback list, so within one validation the EMA restore runs first. That
    ordering is Lightning's to change and nothing asserted it (audit L-48); the EMA reaches a
    checkpoint only through this callback's own ``state_dict``, which is the split
    ``lucid_yolo.eval.checkpoint`` relies on when it overlays the shadow on request.
    """

    def test_state_dict_is_raw_weights_with_the_shadow_kept_separately(self, tmp_path: Path) -> None:
        """The saved model ``state_dict`` matches the live weights and differs from the shadow."""
        module = _TinyEMAModule()
        callback = EMACallback(decay=0.9, tau=1)
        checkpointer = ModelCheckpoint(dirpath=str(tmp_path), filename="last", save_top_k=1, monitor=None)
        loader: DataLoader[tuple[Tensor, int]] = DataLoader(_SyntheticVectorDataset(8), batch_size=2)
        trainer = _cadence_trainer(callback, accumulate=1, limit_val_batches=1, extra_callbacks=[checkpointer])

        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

        saved = torch.load(checkpointer.best_model_path, weights_only=False)
        assert callback._shadow is not None
        assert torch.allclose(saved["state_dict"]["weight"], module.weight.detach())
        assert not torch.allclose(saved["state_dict"]["weight"], callback._shadow["weight"])
        shadow_state = saved["callbacks"][callback.state_key]["shadow"]
        assert torch.allclose(shadow_state["weight"], callback._shadow["weight"])
