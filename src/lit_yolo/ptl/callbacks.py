# SPDX-License-Identifier: Apache-2.0
"""Training-schedule Lightning callbacks for the detection loop (WP-036).

This module hosts the :class:`~lightning.pytorch.callbacks.Callback` s that shape
the training *schedule* rather than the model or the data:

- :class:`CloseMosaicCallback` disables mosaic augmentation for the final
  ``close_mosaic`` epochs (blueprint sec. 5.9; [R1] Table S3 recipe context),
  letting the network settle on un-composited images before training ends.

The EMA callback (WP-037) lands here next, sharing the same rank-zero-safe,
datamodule-seam conventions established below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_info

from lit_yolo.ptl.datamodule import DetectionDataModule

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer

__all__ = ["CloseMosaicCallback"]

#: Default number of final epochs over which mosaic is disabled (blueprint sec. 5.9:
#: ``close_mosaic`` = 8-10; the 10-epoch end of that range is the conventional default).
_DEFAULT_CLOSE_MOSAIC = 10
#: Mosaic probability the schedule flips to once the close-mosaic window is entered.
_MOSAIC_DISABLED = 0.0


@runtime_checkable
class _SupportsMosaicProb(Protocol):
    """Duck-typed datamodule seam the callback drives to disable mosaic.

    Any datamodule exposing :meth:`set_mosaic_prob` (notably
    :class:`~lit_yolo.ptl.datamodule.DetectionDataModule`, but also a test stub)
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
    datamodule's :meth:`~lit_yolo.ptl.datamodule.DetectionDataModule.set_mosaic_prob`
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
        >>> from lit_yolo.ptl.callbacks import CloseMosaicCallback
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

        Prefers a concrete :class:`~lit_yolo.ptl.datamodule.DetectionDataModule`
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
