# SPDX-License-Identifier: Apache-2.0
"""Gate on resuming training from the command line: ``lucid-yolo fit --ckpt_path``.

``tests/ptl/test_resume.py`` drives :class:`~pytorch_lightning.Trainer` directly; this file drives the shipped command,
on the packaged ``det_nano_smoke`` recipe, the way an operator resumes an interrupted run. The interruption is modelled
as a real one is: both runs are configured for the same three epochs, and the first is stopped after two by
``max_steps``, which is trainer configuration and not checkpointed state, so the resumed run continues to the
configured end.

What a resume must carry across, and what each test asserts:

- the step and epoch counters continue rather than restart;
- the :class:`~lucid_yolo.ptl.callbacks.EMACallback` shadow and update count are the checkpoint's when training
  restarts, so the average continues rather than re-seeding from the resumed weights;
- the optimizer's momentum state is the checkpoint's;
- the progressive-loss weight ``alpha`` is the schedule's value for the resumed epoch. The schedule holds no state of
  its own -- ``alpha`` is a function of the current and final epoch (WP-035) -- so restoring the epoch counter is what
  restores it, and the test asserts that consequence rather than a stored value.

The checkpoint the command writes is also the realistic input for :func:`~lucid_yolo.eval.checkpoint.load_eval_module`,
whose hyper-parameters carry LightningCLI's ``_instantiator`` entry beside the module's arguments; the last test reads
it back.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
import torch

from lucid_yolo.cli import train as train_cli
from lucid_yolo.eval.checkpoint import load_eval_module
from lucid_yolo.losses.progressive import ProgressiveLossSchedule
from lucid_yolo.ptl.callbacks import EMACallback
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

#: Epochs both runs are configured for.
_MAX_EPOCHS = 3
#: Batches per epoch, capped so the n-scale model trains in about a second per run.
_STEPS_PER_EPOCH = 2
#: Optimizer steps after which the first run stops: the end of its second epoch.
_INTERRUPT_STEP = 4
#: Category ids 1..4 in the detseg fixture.
_NUM_CLASSES = 4


@dataclass
class _Observed:
    """What the two runs left behind: their checkpoints and what training saw at each start."""

    interrupted: Path
    resumed: Path
    #: ``(current_epoch, alpha)`` at every epoch start, across both runs, in order.
    alphas: list[tuple[int, float]] = field(default_factory=list)
    #: EMA and optimizer state at each run's ``on_train_start``, in run order.
    starts: list[dict[str, Any]] = field(default_factory=list)


def _fit(root: Path, out: Path, *extra: str) -> Path:
    """Run ``lucid-yolo fit`` on ``root`` and return the one checkpoint it wrote.

    Examples:
        >>> callable(_fit)  # a real call trains a model
        True
    """
    train_cli.main(
        [
            "fit",
            "--data.data_root", str(root),
            "--data.batch_size", "2",
            "--data.num_workers", "0",
            "--data.img_size", "64",
            "--model.num_classes", str(_NUM_CLASSES),
            "--model.warmup_epochs", "1",
            "--trainer.max_epochs", str(_MAX_EPOCHS),
            "--trainer.accelerator", "cpu",
            "--trainer.devices", "1",
            "--trainer.logger", "false",
            "--trainer.limit_train_batches", str(_STEPS_PER_EPOCH),
            "--trainer.limit_val_batches", "0",
            "--trainer.default_root_dir", str(out),
            "--progress_bar", "none",
            *extra,
        ]
    )  # fmt: skip
    (checkpoint,) = out.rglob("*.ckpt")
    return checkpoint


@pytest.fixture(scope="module")
def observed(detseg_fixture_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Observed]:
    """Train, interrupt, resume, and record what each run started from.

    Module-scoped: the two fits are the expensive part, and every test reads the same pair. The recording hooks wrap the
    real ones, so nothing about training changes.
    """
    tmp_path = tmp_path_factory.mktemp("resume")
    root = tmp_path / "root"
    root.mkdir()
    for split in ("train", "val"):
        (root / split).symlink_to(detseg_fixture_dir / "train", target_is_directory=True)
    alphas: list[tuple[int, float]] = []
    starts: list[dict[str, Any]] = []
    epoch_start = DetectionLitModule.on_train_epoch_start
    train_start = EMACallback.on_train_start

    def recording_epoch_start(self: DetectionLitModule) -> None:
        epoch_start(self)
        alphas.append((self.current_epoch, self.alpha))

    def recording_train_start(self: EMACallback, trainer: Any, pl_module: Any) -> None:
        train_start(self, trainer, pl_module)
        state = self.state_dict()
        starts.append(
            {
                "shadow": {name: tensor.detach().cpu().clone() for name, tensor in state["shadow"].items()},
                "num_updates": state["num_updates"],
                # Deep-copied: the optimizer's state dict holds its live buffers, which the
                # steps after this hook go on to update in place.
                "optimizer": copy.deepcopy(trainer.optimizers[0].state_dict()["state"]),
            }
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DetectionLitModule, "on_train_epoch_start", recording_epoch_start)
        patch.setattr(EMACallback, "on_train_start", recording_train_start)
        interrupted = _fit(root, tmp_path / "interrupted", "--trainer.max_steps", str(_INTERRUPT_STEP))
        resumed = _fit(root, tmp_path / "resumed", "--ckpt_path", str(interrupted))
    yield _Observed(interrupted, resumed, alphas, starts)


def _load(path: Path) -> dict[str, Any]:
    """Read a checkpoint written by the command, restricted.

    Examples:
        >>> callable(_load)
        True
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def _ema_entry(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return the checkpoint's single EMA callback entry.

    Examples:
        >>> _ema_entry({"callbacks": {"EMACallback": {"num_updates": 3}}})["num_updates"]
        3
    """
    (key,) = [name for name in checkpoint["callbacks"] if "EMACallback" in name]
    return checkpoint["callbacks"][key]


def test_step_and_epoch_continue(observed: _Observed) -> None:
    """The resumed run ends one epoch and two steps past the interrupted one, not from zero."""
    interrupted, resumed = _load(observed.interrupted), _load(observed.resumed)

    assert (interrupted["epoch"], interrupted["global_step"]) == (1, _INTERRUPT_STEP)
    assert (resumed["epoch"], resumed["global_step"]) == (_MAX_EPOCHS - 1, _MAX_EPOCHS * _STEPS_PER_EPOCH)


def test_ema_continues_from_checkpoint(observed: _Observed) -> None:
    """Training resumes with the checkpoint's EMA shadow and count, then keeps counting.

    A shadow re-seeded from the resumed weights would differ from the stored one after four blends, and a reset counter
    would end at two rather than six.
    """
    stored = _ema_entry(_load(observed.interrupted))

    torch.testing.assert_close(observed.starts[1]["shadow"], stored["shadow"], rtol=0, atol=0)
    assert observed.starts[1]["num_updates"] == stored["num_updates"] == _INTERRUPT_STEP
    assert _ema_entry(_load(observed.resumed))["num_updates"] == _MAX_EPOCHS * _STEPS_PER_EPOCH


def test_optimizer_state_restored(observed: _Observed) -> None:
    """The resumed run's optimizer starts from the checkpoint's momentum, not from empty."""
    stored = _load(observed.interrupted)["optimizer_states"][0]["state"]

    assert stored
    torch.testing.assert_close(observed.starts[1]["optimizer"], stored, rtol=0, atol=0)


def test_alpha_follows_schedule_at_resumed_epoch(observed: _Observed) -> None:
    """Epoch two, the first after resume, trains at the schedule's epoch-two weight.

    The interrupted run saw epochs zero and one, the resumed run only epoch two; a counter restarted from zero would
    have recorded a third epoch-zero weight instead.
    """
    schedule = ProgressiveLossSchedule(alpha_init=0.8, alpha_final=0.1)

    assert [epoch for epoch, _ in observed.alphas] == [0, 1, 2]
    assert observed.alphas[2][1] == schedule.alpha_at(2, _MAX_EPOCHS)


def test_command_checkpoint_loads_for_evaluation(observed: _Observed) -> None:
    """The command's checkpoint, ``_instantiator`` entry and all, loads as Lightning loads it."""
    hyper_parameters = _load(observed.resumed)["hyper_parameters"]

    loaded, info = load_eval_module(observed.resumed, use_ema=True)
    reference = DetectionLitModule.load_from_checkpoint(observed.resumed, map_location="cpu")

    assert "_instantiator" in hyper_parameters
    assert dict(loaded.hparams) == dict(reference.hparams)
    assert info["ema_updates"] == _MAX_EPOCHS * _STEPS_PER_EPOCH
