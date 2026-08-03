# SPDX-License-Identifier: Apache-2.0
"""Linear warmup + linear-decay learning-rate factor (WP-072, A8).

The papers name ``lr0`` and ``lrf`` but never restate the schedule shape (A8);
the convention widely restated in third-party YOLO-application literature is a
linear decay from ``lr0`` to ``lr0 * lrf`` over the run. A short linear warmup
from near zero is prepended — generic from-scratch stabilization practice (the
same family as the A31 gradient clip): the RetinaNet-prior head init (A30)
bounds the opening loss, and warmup keeps the very first MuSGD updates from
overshooting while batch-norm statistics are still garbage.

:func:`warmup_decay_factor` is the pure per-step multiplier applied to ``lr0``
by a :class:`torch.optim.lr_scheduler.LambdaLR` (stepped per optimizer step,
``interval: "step"``):

- steps ``0 .. warmup_steps-1``: linear ramp ``(step + 1) / warmup_steps``;
- remaining steps: linear decay from ``1`` at the end of warmup to ``lrf`` at
  ``total_steps``, clamped at ``lrf`` beyond.

Provenance: A8 (gap; third-party convention), A31 lineage (generic practice).
"""

from __future__ import annotations

__all__ = ["warmup_decay_factor"]


def warmup_decay_factor(step: int, total_steps: int, warmup_steps: int, lrf: float) -> float:
    """Return the LR multiplier for ``step`` of a warmup + linear-decay schedule.

    Args:
        step: Zero-based optimizer step.
        total_steps: Total optimizer steps of the run (the decay reaches ``lrf``
            here); values below 1 are treated as 1.
        warmup_steps: Steps of the opening linear ramp; ``0`` disables warmup.
        lrf: Final LR fraction — the multiplier decays from ``1.0`` to ``lrf``.

    Returns:
        The factor in ``[min(lrf, 1/warmup_steps), 1.0]`` to multiply ``lr0`` by.

    Examples:
        >>> warmup_decay_factor(0, 100, 10, 0.01)  # first step: 1/10 of lr0
        0.1
        >>> warmup_decay_factor(9, 100, 10, 0.01)  # warmup peak
        1.0
        >>> round(warmup_decay_factor(55, 100, 10, 0.01), 4)  # mid-decay
        0.505
        >>> warmup_decay_factor(100, 100, 10, 0.01)  # floor
        0.01
        >>> warmup_decay_factor(0, 100, 0, 0.5)  # no warmup: decay starts at 1.0
        1.0
    """
    total = max(1, total_steps)
    if step < warmup_steps:
        return float(step + 1) / warmup_steps
    decay_span = max(1, total - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_span)
    return 1.0 - (1.0 - lrf) * progress
