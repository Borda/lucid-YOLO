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

import warnings
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
            pytest.param(99, 100, 10, 0.01, 0.01, id="last-step-hits-lrf"),
            pytest.param(100, 100, 10, 0.01, 0.01, id="past-total-clamps-at-lrf"),
            pytest.param(1000, 100, 10, 0.01, 0.01, id="beyond-total-clamps-at-lrf"),
            pytest.param(0, 100, 0, 0.5, 1.0, id="no-warmup-starts-at-one"),
            # Halfway along a 100-step run is step 50 of the 99 decay steps 0..99,
            # not 50 of 100: the run's last index is total-1. The 0.75 this asserted
            # before was the old off-by-one's answer, not a loosened expectation.
            pytest.param(50, 100, 0, 0.5, 1.0 - 0.5 * 50 / 99, id="no-warmup-linear-midpoint"),
        ],
    )
    def test_endpoints(self, step: int, total: int, warmup: int, lrf: float, expected: float) -> None:
        """The factor ramps 1/w..1 across warmup, then decays linearly to the lrf floor."""
        assert warmup_decay_factor(step, total, warmup, lrf) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("total", "warmup", "lrf"),
        [
            pytest.param(100, 10, 0.01, id="doctested-shape"),
            pytest.param(50, 3, 0.01, id="smoke-tier-shape"),
            pytest.param(12, 6, 0.05, id="half-the-run-is-warmup"),
            pytest.param(1000, 0, 0.01, id="no-warmup-long-run"),
            pytest.param(3, 1, 0.2, id="three-step-run"),
        ],
    )
    def test_reaches_lrf_on_the_runs_last_step(self, total: int, warmup: int, lrf: float) -> None:
        """The floor is hit at ``total_steps - 1``, the last step a run of that length takes.

        ``LambdaLR`` is called with zero-based step indices, so a run of ``total``
        steps never evaluates index ``total``. Anchoring the decay there left the
        schedule short of its own floor — at the 100-step shape the final factor was
        0.021, twice the requested 0.01 — and the miss widens as runs get shorter,
        which is the wrong direction for this project's short smoke tiers.
        """
        assert warmup_decay_factor(total - 1, total, warmup, lrf) == pytest.approx(lrf)

    def test_warmup_covering_all_but_one_step_never_decays(self) -> None:
        """``warmup_steps == total_steps - 1`` peaks at 1.0 and never reaches the floor.

        The boundary the caller-side clamp lands on, pinned here so it is a stated
        limit rather than a surprise. Such a run has exactly one post-warmup step,
        so the decay's own span is empty and its single sample sits at progress 0.
        The floor is unreachable for this shape by construction — no schedule
        arithmetic recovers it, only leaving fewer warmup steps does.
        """
        assert warmup_decay_factor(1, 2, 1, 0.2) == pytest.approx(1.0)

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

    def test_warmup_longer_than_the_run_is_clamped_below_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A warmup budget exceeding the run is clamped to ``total_steps - 1``, with a warning.

        ``warmup_epochs`` is a fraction of the run, so the 3-epoch default against a
        1-3 epoch smoke budget asks for more warmup steps than the run has — here 5
        warmup epochs of a 2-epoch run over 50 steps requests 125. Unclamped, every
        step takes the ramp branch: the LR climbs for the whole run, peaks below
        ``lr``, and never decays, silently. Clamped, the ramp completes exactly at
        the final step. It reaches 1.0 there rather than ``lrf`` because a run that
        is all warmup has no step left to decay on — the clamp buys back the peak,
        not the floor.
        """
        module = _module(warmup_epochs=5.0)
        trainer = Trainer(max_epochs=2, limit_train_batches=25, logger=False, enable_progress_bar=False)
        trainer.strategy._lightning_module = module
        module.trainer = trainer
        monkeypatch.setattr(Trainer, "estimated_stepping_batches", property(lambda self: 50))

        with pytest.warns(UserWarning, match="leave room for the decay"):
            config = module.configure_optimizers()

        assert isinstance(config, dict)
        factor = config["lr_scheduler"]["scheduler"].lr_lambdas[0]
        assert factor(49) == pytest.approx(1.0)  # unclamped this would still be mid-ramp at 0.4
        assert factor(0) == pytest.approx(1 / 49)  # the ramp is 49 steps long, not the requested 125

    def test_warmup_fitting_the_run_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A warmup budget that fits the run is untouched and warns nothing.

        The guard against over-clamping: the common case (3 warmup epochs of a
        50-epoch run) must keep its exact requested ramp length, and the decay must
        still reach the ``lrf`` floor on the run's last step.
        """
        module = _module(warmup_epochs=3.0, lrf=0.01)
        trainer = Trainer(max_epochs=50, limit_train_batches=10, logger=False, enable_progress_bar=False)
        trainer.strategy._lightning_module = module
        module.trainer = trainer
        monkeypatch.setattr(Trainer, "estimated_stepping_batches", property(lambda self: 500))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config = module.configure_optimizers()

        assert isinstance(config, dict)
        factor = config["lr_scheduler"]["scheduler"].lr_lambdas[0]
        assert factor(29) == pytest.approx(1.0)  # 3/50 of 500 steps = a 30-step ramp, peaking at its end
        assert factor(499) == pytest.approx(0.01)  # and the decay lands on the floor at the last step
