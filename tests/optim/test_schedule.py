# SPDX-License-Identifier: Apache-2.0
"""Tests for the A8 warmup + linear-decay LR factor (WP-072).

Covers the pure :func:`~lucid_yolo.optim.schedule.warmup_decay_factor` contract
(warmup ramp endpoints, decay endpoints, the ``lrf`` floor, the no-warmup form)
and the module wiring: ``configure_optimizers`` pairs MuSGD with a step-interval
``LambdaLR`` when a trainer with an epoch budget is attached, and degrades to
the bare constant-LR optimizer when the schedule is disabled or no trainer is
attached (the overfit-100 recipe and direct tool calls rely on both fallbacks).
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from pytorch_lightning import Trainer

from lucid_yolo.optim.musgd import MuSGD
from lucid_yolo.optim.schedule import warmup_decay_factor
from lucid_yolo.ptl.module import DetectionLitModule


class TestWarmupDecayFactor:
    """Tests for ``warmup_decay_factor``."""

    @pytest.mark.parametrize(
        ("step", "total", "warmup", "lrf", "expected"),
        [
            pytest.param(0, 100, 10, 0.01, 0.1, id="warmup-first-step"),
            pytest.param(4, 100, 10, 0.01, 0.5, id="warmup-midpoint"),
            pytest.param(9, 100, 10, 0.01, 1.0, id="warmup-peak"),
            pytest.param(10, 100, 10, 0.01, 1.0, id="decay-start-full-lr"),
            pytest.param(100, 100, 10, 0.01, 0.01, id="decay-end-hits-lrf"),
            pytest.param(1000, 100, 10, 0.01, 0.01, id="beyond-total-clamps-at-lrf"),
            pytest.param(0, 100, 0, 0.5, 1.0, id="no-warmup-starts-at-one"),
            pytest.param(50, 100, 0, 0.5, 0.75, id="no-warmup-linear-midpoint"),
        ],
    )
    def test_endpoints(self, step: int, total: int, warmup: int, lrf: float, expected: float) -> None:
        """The factor ramps 1/w..1 across warmup, then decays linearly to the lrf floor."""
        assert warmup_decay_factor(step, total, warmup, lrf) == pytest.approx(expected)

    def test_is_monotonic_after_warmup(self) -> None:
        """After the warmup peak the factor never increases and never dips below lrf."""
        values = [warmup_decay_factor(step, 200, 20, 0.05) for step in range(20, 220)]
        deltas = [after - before for before, after in pairwise(values)]
        assert all(delta <= 0 for delta in deltas)
        assert min(values) >= 0.05


def _module(**kwargs: float) -> DetectionLitModule:
    """Build a minimal n-scale-ish module with the given schedule overrides.

    Examples:
        >>> module = _module(lr=0.02)
        >>> module.hparams.lr
        0.02
    """
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4, **kwargs)


class TestConfigureOptimizers:
    """Tests for ``DetectionLitModule.configure_optimizers``."""

    def test_without_trainer_returns_bare_musgd(self) -> None:
        """No attached trainer (direct tool/test calls) degrades to the constant-LR optimizer."""
        optimizer = _module().configure_optimizers()
        assert isinstance(optimizer, MuSGD)

    def test_with_trainer_pairs_step_lambda_lr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With an epoch-budgeted trainer attached, MuSGD comes wrapped with a step-interval LambdaLR."""
        module = _module()
        trainer = Trainer(max_epochs=5, limit_train_batches=10, logger=False, enable_progress_bar=False)
        trainer.strategy._lightning_module = module
        module.trainer = trainer
        # The real property loads a train dataloader to count batches; the schedule
        # only needs the resulting integer.
        monkeypatch.setattr(Trainer, "estimated_stepping_batches", property(lambda self: 50))
        config = module.configure_optimizers()
        assert isinstance(config, dict)
        scheduler_config = config["lr_scheduler"]
        assert scheduler_config["interval"] == "step"
        assert type(scheduler_config["scheduler"]).__name__ == "LambdaLR"

    def test_schedule_disabled_returns_bare_musgd(self) -> None:
        """lrf >= 1 with warmup 0 (the overfit-100 recipe) keeps the bare optimizer under a trainer."""
        module = _module(lrf=1.0, warmup_epochs=0.0)
        trainer = Trainer(max_epochs=5, limit_train_batches=10, logger=False, enable_progress_bar=False)
        trainer.strategy._lightning_module = module
        module.trainer = trainer
        optimizer = module.configure_optimizers()
        assert isinstance(optimizer, MuSGD)
