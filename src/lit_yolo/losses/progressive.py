# SPDX-License-Identifier: Apache-2.0
"""Progressive dual-branch loss schedule (WP-035).

Transcribed by hand from the reproduction's technical specification (R1 Eq. 2-3,
§4.1; blueprint sec. 5.7). The dual detection loss combines the two branch
totals as::

    L = alpha(t) * L_o2m + (1 - alpha(t)) * L_o2o

where the one-to-many weight ``alpha`` is *ramped down* once per epoch on a
linear schedule::

    alpha(t) = max(1 - t / max(E - 1, 1), 0) * (alpha_init - alpha_final) + alpha_final

with ``t`` the 0-based epoch index and ``E`` the total epoch count. The defaults
``(alpha_init, alpha_final) = (0.8, 0.1)`` place the branch weights at
``(0.8, 0.2)`` on the first epoch (dense o2m supervision dominates, driving fast
high-recall learning) and ramp them to ``(0.1, 0.9)`` on the last epoch (the
NMS-free o2o branch takes over). The ``max(E - 1, 1)`` guard keeps the single
epoch case (``E == 1``) finite: the ramp evaluates to ``alpha_init``.

The schedule writes to the bare ``alpha`` attribute exposed by
:class:`~lit_yolo.losses.dual_loss.DualBranchLoss` (via
:attr:`~lit_yolo.ptl.module.DetectionLitModule.alpha`); the Lightning
``on_train_epoch_start`` hook applies :func:`progressive_alpha` per epoch.

Provenance: R1 Eq. 2-3, R1 §4.1. Assumptions: none.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ProgressiveLossSchedule", "progressive_alpha"]

#: Default one-to-many weight on the first epoch (branch weights ``(0.8, 0.2)``).
_DEFAULT_ALPHA_INIT = 0.8
#: Default one-to-many weight on the last epoch (branch weights ``(0.1, 0.9)``).
_DEFAULT_ALPHA_FINAL = 0.1


def progressive_alpha(
    epoch: int,
    total_epochs: int,
    alpha_init: float = _DEFAULT_ALPHA_INIT,
    alpha_final: float = _DEFAULT_ALPHA_FINAL,
) -> float:
    """Linear once-per-epoch ramp of the one-to-many branch weight (R1 Eq. 2-3).

    Implements ``alpha(t) = max(1 - t / max(E - 1, 1), 0) * (alpha_init -
    alpha_final) + alpha_final`` verbatim, with ``t = epoch`` (0-based) and
    ``E = total_epochs``. The ``max(E - 1, 1)`` denominator guard keeps the
    ``E == 1`` degenerate case finite (the ramp factor is ``1``, so the result
    is ``alpha_init``), and the outer ``max(..., 0)`` clamps the ramp to zero at
    and beyond the final epoch so ``alpha`` never falls below ``alpha_final``.

    Args:
        epoch: Current 0-based epoch index ``t``. Values ``>= total_epochs - 1``
            all yield ``alpha_final`` (the ramp is clamped at zero).
        total_epochs: Total epoch count ``E`` of the training run.
        alpha_init: One-to-many weight at ``epoch == 0``. Defaults to ``0.8``.
        alpha_final: One-to-many weight at ``epoch == total_epochs - 1``.
            Defaults to ``0.1``.

    Returns:
        The one-to-many branch weight ``alpha`` for this epoch; the one-to-one
        branch receives ``1 - alpha``.

    Examples:
        >>> progressive_alpha(0, 100)  # first epoch: dense o2m dominates
        0.8
        >>> round(progressive_alpha(99, 100), 4)  # last epoch: o2o takes over
        0.1
        >>> round(progressive_alpha(5, 11), 4)  # midpoint of an 11-epoch run
        0.45
        >>> progressive_alpha(0, 1)  # single-epoch run: ramp guard -> alpha_init
        0.8
    """
    ramp = max(1.0 - epoch / max(total_epochs - 1, 1), 0.0)
    return ramp * (alpha_init - alpha_final) + alpha_final


@dataclass(frozen=True)
class ProgressiveLossSchedule:
    """Stateless linear schedule for the dual-loss one-to-many branch weight.

    Holds the ramp endpoints ``(alpha_init, alpha_final)`` and evaluates the
    per-epoch weight through :func:`progressive_alpha`. The object carries no
    epoch state of its own — the caller supplies the current epoch and total
    epoch count each call — so a single instance is reused across the run and
    the :class:`~lit_yolo.ptl.module.DetectionLitModule` ``on_train_epoch_start``
    hook drives it from ``self.current_epoch`` / ``trainer.max_epochs``.

    Attributes:
        alpha_init: One-to-many weight at ``epoch == 0``. Defaults to ``0.8``.
        alpha_final: One-to-many weight at the final epoch. Defaults to ``0.1``.

    Examples:
        >>> schedule = ProgressiveLossSchedule()
        >>> schedule.alpha_at(0, 10)
        0.8
        >>> round(schedule.alpha_at(9, 10), 4)
        0.1
    """

    alpha_init: float = _DEFAULT_ALPHA_INIT
    alpha_final: float = _DEFAULT_ALPHA_FINAL

    def alpha_at(self, epoch: int, total_epochs: int) -> float:
        """Return the one-to-many branch weight for a given epoch.

        Args:
            epoch: Current 0-based epoch index ``t``.
            total_epochs: Total epoch count ``E`` of the training run.

        Returns:
            The one-to-many branch weight from :func:`progressive_alpha` using
            this schedule's ``(alpha_init, alpha_final)`` endpoints.

        Examples:
            >>> round(ProgressiveLossSchedule().alpha_at(5, 11), 4)
            0.45
        """
        return progressive_alpha(epoch, total_epochs, self.alpha_init, self.alpha_final)
