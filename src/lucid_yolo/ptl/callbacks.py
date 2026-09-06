# SPDX-License-Identifier: Apache-2.0
"""Training-schedule Lightning callbacks for the detection loop (WP-036, WP-037).

This module hosts the :class:`~pytorch_lightning.callbacks.Callback` s that shape
the training *schedule* rather than the model or the data:

- :class:`CloseMosaicCallback` disables mosaic augmentation for the final
  ``close_mosaic`` epochs (blueprint sec. 5.9; [R1] Table S3 recipe context),
  letting the network settle on un-composited images before training ends.
- :class:`EMACallback` maintains an exponential-moving-average shadow of the
  model weights and evaluates with it (blueprint D4: "EMA is a callback";
  "used for eval"), the standard YOLO-lineage evaluation convention.

Both share the same rank-zero-safe, side-effect-scoped conventions.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_info

from lucid_yolo.ptl.datamodule import DetectionDataModule

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytorch_lightning import LightningModule, Trainer
    from torch import Tensor

__all__ = ["CloseMosaicCallback", "EMACallback"]

#: Default number of final epochs over which mosaic is disabled (blueprint sec. 5.9:
#: ``close_mosaic`` = 8-10; the 10-epoch end of that range is the conventional default).
_DEFAULT_CLOSE_MOSAIC = 10
#: Mosaic probability the schedule flips to once the close-mosaic window is entered.
_MOSAIC_DISABLED = 0.0


@runtime_checkable
class _SupportsMosaicProb(Protocol):
    """Duck-typed datamodule seam the callback drives to disable mosaic.

    Any datamodule exposing :meth:`set_mosaic_prob` (notably
    :class:`~lucid_yolo.ptl.datamodule.DetectionDataModule`, but also a test stub)
    satisfies this protocol, so the callback works without importing a concrete type.
    """

    def set_mosaic_prob(self, prob: float) -> None:
        """Set the training pipeline's mosaic probability."""
        ...


class CloseMosaicCallback(Callback):
    """Disable mosaic augmentation for the final ``close_mosaic`` training epochs.

    Mosaic (four-image compositing) runs at ``p=1.0`` for most of training but is
    switched off for the last ``close_mosaic`` epochs so the network finishes on
    natural, un-composited images (blueprint sec. 5.9; [R1] Tables S3/S6). At the
    start of every training epoch this callback checks whether the current epoch has
    reached the close-mosaic window (``current_epoch >= max_epochs - close_mosaic``)
    and, if so, sets the training pipeline's mosaic probability to ``0`` via the
    datamodule's :meth:`~lucid_yolo.ptl.datamodule.DetectionDataModule.set_mosaic_prob`
    seam. The flip is idempotent — re-setting ``0`` each subsequent epoch is harmless
    — and is logged exactly once, on the epoch it first takes effect (rank zero only).

    Args:
        close_mosaic: Number of final epochs to train with mosaic disabled. ``0``
            never disables mosaic; a value ``>= max_epochs`` disables it from the
            first epoch. Must be non-negative.

    Raises:
        ValueError: If ``close_mosaic`` is negative.

    Examples:
        ```pycon
        >>> from lucid_yolo.ptl.callbacks import CloseMosaicCallback
        >>> callback = CloseMosaicCallback(close_mosaic=10)
        >>> callback.close_mosaic
        10

        ```
    """

    def __init__(self, close_mosaic: int = _DEFAULT_CLOSE_MOSAIC) -> None:
        super().__init__()
        if close_mosaic < 0:
            raise ValueError(f"close_mosaic must be non-negative, got {close_mosaic}")
        self.close_mosaic = int(close_mosaic)
        self._closed = False

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Disable mosaic once the close-mosaic window is reached (idempotent).

        Args:
            trainer: The active trainer, read for ``current_epoch``, ``max_epochs``,
                and the attached ``datamodule``.
            pl_module: The module being trained (unused; part of the hook signature).
        """
        if self._closed or not self._window_reached(trainer):
            return
        datamodule = self._mosaic_datamodule(trainer)
        if datamodule is None:
            return
        datamodule.set_mosaic_prob(_MOSAIC_DISABLED)
        self._closed = True
        rank_zero_info(
            f"CloseMosaicCallback: disabling mosaic from epoch {trainer.current_epoch} "
            f"(final {self.close_mosaic} of {trainer.max_epochs} epochs)."
        )

    def _window_reached(self, trainer: Trainer) -> bool:
        """Return whether the current epoch has entered the close-mosaic window.

        ``close_mosaic=0`` yields a threshold of ``max_epochs``, which the last
        training epoch (``max_epochs - 1``) never reaches, so mosaic is never
        disabled; a ``close_mosaic`` at least ``max_epochs`` yields a non-positive
        threshold, so the window is entered at epoch ``0``.
        """
        if self.close_mosaic == 0:
            return False
        max_epochs = trainer.max_epochs or 0
        return trainer.current_epoch >= max_epochs - self.close_mosaic

    @staticmethod
    def _mosaic_datamodule(trainer: Trainer) -> _SupportsMosaicProb | None:
        """Return the attached datamodule if it exposes the mosaic seam, else ``None``.

        Prefers a concrete :class:`~lucid_yolo.ptl.datamodule.DetectionDataModule`
        (the type check also narrows for the type checker); falls back to any object
        that duck-types the :class:`_SupportsMosaicProb` seam so a test stub works.
        """
        # ``Trainer.datamodule`` is set at runtime by the data connector and is not
        # declared on the class, so it is reached via ``getattr`` for the type checker.
        datamodule = getattr(trainer, "datamodule", None)
        if isinstance(datamodule, DetectionDataModule):
            return datamodule
        if isinstance(datamodule, _SupportsMosaicProb):
            return datamodule
        return None


#: Default EMA decay (blueprint D4 names EMA as a callback but leaves the constant
#: unpublished; ``0.9999`` is the conventional YOLO-lineage per-step decay).
_DEFAULT_DECAY = 0.9999
#: Default warmup time constant (updates) of the decay ramp, again the conventional value.
_DEFAULT_TAU = 2000
#: Default update stride: refresh the shadow every optimizer step.
_DEFAULT_UPDATE_EVERY = 1


class EMACallback(Callback):
    """Track an exponential-moving-average shadow of the weights and eval with it.

    A shadow copy of the module's floating-point parameters **and** buffers (batch-norm
    running statistics included; integer buffers such as ``num_batches_tracked`` are left
    out — they carry no meaningful average) is created on :meth:`on_fit_start` on the
    module's device. After each *optimizer step* (:meth:`on_train_batch_end`, honoring
    ``update_every``) the shadow is blended toward the live weights under
    :func:`torch.no_grad` with a *warmup-ramped* decay

    ``d_t = decay * (1 - exp(-t / tau))``     ``shadow <- d_t * shadow + (1 - d_t) * live``

    where ``t`` is the number of updates performed so far. The ramp starts the effective
    decay near zero so the early shadow tracks the fast-moving weights closely, then
    relaxes toward the nominal ``decay`` as training settles (blueprint D4). Around the
    validation loop the callback swaps the shadow in (:meth:`on_validation_start`) and the
    live weights back out (:meth:`on_validation_end`), so validation metrics are EMA
    metrics — the "used for eval" contract of D4. The shadow and update counters are
    checkpointed via :meth:`state_dict`/:meth:`load_state_dict`, so a save/resume round
    trip continues the same average rather than restarting it.

    **Cadence is optimizer steps, not batches.** ``update_every`` and ``tau`` are both
    counted in optimizer steps, so ``accumulate_grad_batches > 1`` does not multiply the
    number of blends: :meth:`on_train_batch_end` blends only on a batch that actually
    advanced ``trainer.global_step``, which under accumulation is the last batch of each
    accumulation window. That hook rather than ``on_before_zero_grad`` because Lightning
    calls the latter at the *start* of an accumulation window, before that window's
    ``training_step`` — it would blend pre-step weights and never see the state left by the
    final optimizer step of a run.

    **Callback ordering.** The weight swap is deliberately hook-scoped, not run-scoped, so
    it does not depend on how Lightning happens to order the callback list: the shadow is
    installed in :meth:`on_validation_start` and taken out again in
    :meth:`on_validation_end` (and in :meth:`on_exception`, so a failed validation cannot
    leave EMA weights wearing the live model's identity). Checkpoints written by
    :class:`~pytorch_lightning.callbacks.ModelCheckpoint` therefore hold the *raw* trained
    weights in ``state_dict``, with the EMA available separately under this callback's
    ``state_dict`` — the split :func:`~lucid_yolo.eval.checkpoint.load_checkpoint_model`
    relies on when it overlays the shadow on request. ``ModelCheckpoint`` saves from
    ``on_validation_end`` too, and Lightning moves it to the end of the callback list, so
    within a single validation the restore runs first; the two facts are recorded together
    because only the second is Lightning's to change.

    The exact ``decay``/``tau`` constants are unpublished in the source papers; the
    defaults are the conventional YOLO-lineage values and are documented here as a small,
    self-contained assumption (D4 names EMA explicitly, so no ``ASSUMPTIONS.md`` row is
    warranted).

    Args:
        decay: Nominal per-update decay the warmup ramp relaxes toward. Must be in
            ``[0, 1)``. Defaults to ``0.9999``.
        tau: Warmup time constant (in updates, and so in optimizer steps) of the decay
            ramp; larger values ramp in more slowly. Must be positive. Defaults to ``2000``.
        update_every: Refresh the shadow every ``update_every`` optimizer steps — not every
            ``update_every`` batches. Must be at least ``1``. Defaults to ``1``.

    Raises:
        ValueError: If ``decay`` is outside ``[0, 1)``, ``tau`` is not positive, or
            ``update_every`` is less than ``1``.

    Examples:
        ```pycon
        >>> from lucid_yolo.ptl.callbacks import EMACallback
        >>> callback = EMACallback(decay=0.99, tau=100)
        >>> round(callback._decay_at(1), 6)  # ramped far below the nominal decay
        0.009851
        >>> callback._decay_at(1) < callback.decay
        True

        ```
    """

    def __init__(
        self,
        decay: float = _DEFAULT_DECAY,
        tau: int = _DEFAULT_TAU,
        update_every: int = _DEFAULT_UPDATE_EVERY,
    ) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if update_every < 1:
            raise ValueError(f"update_every must be >= 1, got {update_every}")
        self.decay = float(decay)
        self.tau = int(tau)
        self.update_every = int(update_every)
        self._shadow: dict[str, Tensor] | None = None
        self._backup: dict[str, Tensor] | None = None
        self._num_updates = 0
        self._steps_seen = 0
        #: ``trainer.global_step`` as of the last batch this callback looked at, so a batch
        #: that did not advance the optimizer (an accumulation window's non-final batch) is
        #: told apart from one that did. Re-seeded from the trainer at every
        #: :meth:`on_train_start`, which on resume already carries the restored step count,
        #: so it is derived rather than checkpointed.
        self._last_global_step = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Create (or, on resume, relocate) the shadow on the module's device.

        A shadow restored from a checkpoint via :meth:`load_state_dict` is kept — only its
        device is refreshed — so resuming continues the saved average; otherwise a fresh
        shadow is cloned from the current floating-point parameters and buffers.

        Args:
            trainer: The active trainer (unused; part of the hook signature).
            pl_module: The module whose weights are shadowed.
        """
        del trainer
        device = pl_module.device
        if self._shadow is None:
            self._shadow = {name: tensor.detach().clone() for name, tensor in self._ema_tensors(pl_module)}
        self._shadow = {name: tensor.to(device) for name, tensor in self._shadow.items()}

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Seed the optimizer-step watermark from the trainer, so a resume does not double-count.

        Args:
            trainer: The active trainer, read for its ``global_step``.
            pl_module: The module being trained (unused; part of the hook signature).
        """
        del pl_module
        self._last_global_step = trainer.global_step

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Blend the shadow toward the live weights once per optimizer step.

        A batch that did not advance ``trainer.global_step`` — every batch of an
        ``accumulate_grad_batches`` window but its last — leaves the shadow untouched, so
        ``update_every`` and the ``tau`` ramp are counted in optimizer steps as documented
        rather than in batches. By this hook the step has already been applied, so the
        blend reads post-step weights.

        Args:
            trainer: The active trainer, read for its ``global_step``.
            pl_module: The module whose weights are shadowed.
            outputs: The step outputs (unused; part of the hook signature).
            batch: The batch just processed (unused; part of the hook signature).
            batch_idx: Index of the batch within the epoch (unused).
        """
        del outputs, batch, batch_idx
        if self._shadow is None:
            return
        global_step = trainer.global_step
        if global_step == self._last_global_step:
            return
        self._last_global_step = global_step
        self._steps_seen += 1
        if self._steps_seen % self.update_every != 0:
            return
        self._num_updates += 1
        decay = self._decay_at(self._num_updates)
        with torch.no_grad():
            for name, live in self._ema_tensors(pl_module):
                self._shadow[name].mul_(decay).add_(live.detach(), alpha=1.0 - decay)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Stash the live weights and load the shadow so validation runs on the EMA.

        Args:
            trainer: The active trainer (unused; part of the hook signature).
            pl_module: The module to load the shadow into.
        """
        del trainer
        if self._shadow is None:
            return
        self._backup = {name: tensor.detach().clone() for name, tensor in self._ema_tensors(pl_module)}
        with torch.no_grad():
            for name, live in self._ema_tensors(pl_module):
                live.detach().copy_(self._shadow[name])

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore the live weights stashed in :meth:`on_validation_start`.

        Args:
            trainer: The active trainer (unused; part of the hook signature).
            pl_module: The module to restore the live weights into.
        """
        del trainer
        self._restore_live_weights(pl_module)

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        """Take the shadow back out of the module when a run dies mid-validation.

        Without this the swap of :meth:`on_validation_start` is undone only by the normal
        completion path, so an exception raised inside the validation loop propagates with
        the EMA weights still installed under the raw model's identity — anything the
        failure handler then inspects, saves or resumes from would be the shadow rather
        than the trained weights. Restoration is idempotent, so the ordinary path running
        first (or this hook firing outside validation) is harmless.

        Args:
            trainer: The active trainer (unused; part of the hook signature).
            pl_module: The module to restore the live weights into.
            exception: The exception being propagated (unused; part of the hook signature).
        """
        del trainer, exception
        self._restore_live_weights(pl_module)

    def _restore_live_weights(self, pl_module: LightningModule) -> None:
        """Copy the stashed live weights back into ``pl_module`` and drop the stash.

        A no-op when no backup is held, which is what makes it safe to call from both the
        normal and the exceptional path; clearing the backup afterwards is what makes a
        second call a no-op rather than a re-restore of stale weights.
        """
        if self._backup is None:
            return
        with torch.no_grad():
            for name, live in self._ema_tensors(pl_module):
                live.detach().copy_(self._backup[name])
        self._backup = None

    def state_dict(self) -> dict[str, Any]:
        """Return the shadow and update counters for Lightning to checkpoint.

        Returns:
            A dict with the ``shadow`` tensor mapping (or ``None`` before ``on_fit_start``)
            and the ``num_updates`` / ``steps_seen`` counters. The shadow tensors are
            cloned so a checkpoint captures the snapshot at call time, not a later state.
        """
        shadow = None if self._shadow is None else {name: tensor.clone() for name, tensor in self._shadow.items()}
        return {
            "shadow": shadow,
            "num_updates": self._num_updates,
            "steps_seen": self._steps_seen,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the shadow and update counters from a checkpoint.

        A checkpoint written before the cadence was fixed to optimizer steps carries the
        counter under ``batches_seen``; it is read as the ``update_every`` phase it in fact
        was, so resuming such a run continues rather than restarts. The two counts differ
        only when that run used ``accumulate_grad_batches > 1``.

        Args:
            state_dict: A mapping produced by :meth:`state_dict`.
        """
        shadow = state_dict.get("shadow")
        self._shadow = None if shadow is None else {name: tensor.clone() for name, tensor in shadow.items()}
        self._num_updates = int(state_dict.get("num_updates", 0))
        legacy_steps = state_dict.get("batches_seen", 0)
        self._steps_seen = int(state_dict.get("steps_seen", legacy_steps))

    def _decay_at(self, step: int) -> float:
        """Return the warmup-ramped decay ``decay * (1 - exp(-step / tau))`` at ``step``."""
        return self.decay * (1.0 - math.exp(-step / self.tau))

    @staticmethod
    def _ema_tensors(pl_module: LightningModule) -> Iterator[tuple[str, Tensor]]:
        """Yield ``(name, tensor)`` for every floating-point parameter and buffer.

        Integer buffers (e.g. batch-norm ``num_batches_tracked``) are skipped: an
        exponential average of a step counter is meaningless. Parameter and buffer names
        share one namespace, so the yielded names are unique across both.
        """
        for name, param in pl_module.named_parameters():
            if param.is_floating_point():
                yield name, param
        for name, buffer in pl_module.named_buffers():
            if buffer.is_floating_point():
                yield name, buffer
