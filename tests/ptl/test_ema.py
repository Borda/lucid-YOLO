# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`~open_yolos.ptl.callbacks.EMACallback` (WP-037).

Covers the DoD ``test_shadow_updates`` — after a run of manual
:meth:`~open_yolos.ptl.callbacks.EMACallback.on_train_batch_end` calls against a
hand-mutated tiny module the shadow equals the EMA recursion recomputed in the
test with the same warmup-ramp formula — plus the supporting contracts: the
warmup ramp starts far below the nominal decay; the validation swap loads the
shadow for the duration of validation and restores the live weights after;
floating-point buffers (batch-norm running stats) are tracked while integer
buffers are skipped; ``state_dict``/``load_state_dict`` survive a save/mutate/
load round trip; ``update_every`` gates updates; and a two-epoch Lightning run
with the callback attached completes with the shadow diverged from the live
weights.

The module under test is a tiny ``Linear``-free stub carrying one flat parameter
and a :class:`~torch.nn.BatchNorm1d` so both the parameter and buffer paths are
exercised; the unit tests drive the callback hooks directly with a dummy trainer
(the hooks discard the trainer argument), and the integration test wires a real
:class:`~pytorch_lightning.Trainer` over a synthetic in-memory dataset.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from open_yolos.ptl import EMACallback

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Feature width of the tiny module's parameter and batch-norm layer.
_FEATURES = 4
#: A dummy trainer for direct hook calls — every EMACallback hook discards it.
_DUMMY_TRAINER = MagicMock(spec=Trainer)


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

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(_FEATURES))
        self.bn = nn.BatchNorm1d(_FEATURES)

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
        """Return a scalar so the val loop runs, driving the EMA swap in the real loop."""
        del batch_idx
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
    """Build a callback and run ``on_fit_start`` so its shadow is initialised."""
    callback = EMACallback(**kwargs)  # type: ignore[arg-type]
    callback.on_fit_start(_DUMMY_TRAINER, module)
    return callback


def _batch_end(callback: EMACallback, module: LightningModule, batch_idx: int) -> None:
    """Invoke ``on_train_batch_end`` with the dummy trainer and unused step payloads."""
    callback.on_train_batch_end(_DUMMY_TRAINER, module, None, None, batch_idx)


@pytest.mark.parametrize("num_steps", [pytest.param(1, id="one-step"), pytest.param(5, id="five-steps")])
def test_shadow_updates(num_steps: int) -> None:
    """The shadow equals the warmup-ramped EMA recursion recomputed step by step."""
    module = _TinyEMAModule()
    callback = _fitted_callback(module, decay=0.99, tau=50)
    expected = module.weight.detach().clone()
    for step in range(1, num_steps + 1):
        with torch.no_grad():
            module.weight.copy_(torch.full((_FEATURES,), float(step)))
        _batch_end(callback, module, step - 1)
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
    _batch_end(callback, module, 0)
    decay = callback._decay_at(1)
    expected = decay * before + (1.0 - decay) * new_mean
    assert torch.allclose(callback._shadow["bn.running_mean"], expected)


def test_state_dict_round_trip_restores_shadow_and_counters() -> None:
    """Saving, mutating, then loading restores the shadow tensors and update counters."""
    module = _TinyEMAModule()
    callback = _fitted_callback(module)
    _batch_end(callback, module, 0)
    saved = callback.state_dict()
    assert callback._shadow is not None
    original_weight = callback._shadow["weight"].clone()
    callback._shadow["weight"].add_(100.0)
    callback._num_updates = 999
    callback._batches_seen = 999
    callback.load_state_dict(saved)
    assert callback._num_updates == 1
    assert callback._batches_seen == 1
    assert torch.allclose(callback._shadow["weight"], original_weight)


def test_update_every_skips_intermediate_steps() -> None:
    """With ``update_every=2`` the first step is skipped and the second applies an update."""
    module = _TinyEMAModule()
    callback = _fitted_callback(module, decay=0.9, tau=1, update_every=2)
    assert callback._shadow is not None
    initial = callback._shadow["weight"].clone()
    _batch_end(callback, module, 0)
    assert callback._num_updates == 0
    assert torch.allclose(callback._shadow["weight"], initial)
    with torch.no_grad():
        module.weight.copy_(torch.full((_FEATURES,), 9.0))
    _batch_end(callback, module, 1)
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
