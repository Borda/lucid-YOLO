# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loading and device selection shared by the acceptance scripts (WP-095).

``lucid_yolo.eval.detect_eval`` grew these two steps first — read a Lightning checkpoint,
optionally overlay the EMA shadow the :class:`~lucid_yolo.ptl.callbacks.EMACallback`
stored inside it, and resolve ``auto`` to the fastest available backend. The oriented
instrument needs both, unchanged, and a second copy of a "find the shadow inside the
callbacks dict" walk is the defect class WP-089's row names: two copies of one piece of
knowledge that can drift apart silently, because each has its own tests passing.

The EMA overlay is worth stating plainly, since it is the part that is easy to get
subtly wrong. A shadow covers parameters *and* buffers, so both are collected into one
name-keyed mapping before the copy; a missing shadow raises rather than quietly
evaluating raw weights, because "the EMA numbers" and "the raw numbers" are different
claims and a run that silently reports the second under the first name is unfalsifiable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

__all__ = ["load_eval_module", "pick_device"]


def load_eval_module(checkpoint: Path, *, use_ema: bool) -> tuple[DetectionLitModule, dict[str, object]]:
    """Load an eval-mode module from a Lightning checkpoint, optionally with EMA weights.

    Args:
        checkpoint: Path to the Lightning ``.ckpt`` file.
        use_ema: When ``True``, overlay the :class:`EMACallback` shadow stored in the
            checkpoint onto the module's parameters and buffers.

    Returns:
        The eval-mode module and a small provenance dict (checkpoint, epoch, step, EMA).

    Raises:
        ValueError: If ``use_ema`` is requested but the checkpoint carries no EMA shadow.

    Examples:
        >>> from pathlib import Path
        >>> load_eval_module(Path("/nonexistent.ckpt"), use_ema=False)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        FileNotFoundError: ...
    """
    module = DetectionLitModule.load_from_checkpoint(checkpoint, map_location="cpu")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    info: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "ema": use_ema,
    }
    if use_ema:
        _overlay_ema(module, payload)
        info["ema_updates"] = int(payload["callbacks"]["EMACallback"]["num_updates"])
    module.eval()
    return module, info


def pick_device(requested: str) -> torch.device:
    """Resolve ``auto`` to the fastest available backend, else pass ``requested`` through.

    Args:
        requested: ``"auto"`` or any string :class:`torch.device` accepts.

    Returns:
        The chosen device.

    Examples:
        >>> pick_device("cpu")
        device(type='cpu')
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _overlay_ema(module: DetectionLitModule, payload: dict[str, object]) -> None:
    """Copy the checkpoint's EMA shadow over the module's parameters and buffers.

    Raises:
        ValueError: If the checkpoint carries no EMA shadow.
    """
    callbacks = cast("dict[str, dict[str, object]]", payload.get("callbacks", {}))
    shadow: dict[str, Tensor] | None = None
    for name, state in callbacks.items():
        if "EMACallback" in name:
            shadow = cast("dict[str, Tensor] | None", state.get("shadow"))
    if not shadow:
        raise ValueError(f"EMA weights requested but {payload.get('checkpoint', 'the checkpoint')} carries no shadow")
    tensors: dict[str, Tensor] = dict(module.named_parameters())
    tensors.update(module.named_buffers())
    with torch.no_grad():
        for name, value in shadow.items():
            tensors[name].copy_(value)
