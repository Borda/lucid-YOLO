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

WP-174 closed the two ways that argument leaked. The callback's key inside the
checkpoint is resolved **once** and reused for both the shadow and the update counter --
resolving it by substring here and by literal name there is how a checkpoint written by a
parametrized callback passed the overlay and then died on a bare ``KeyError``, with the
module already mutated -- and an ambiguous key is refused rather than resolved by "last
one wins". The shadow is also checked to cover every floating-point parameter and buffer
*before* any copy happens: a partial shadow otherwise leaves the module half EMA and half
raw, which is neither of the two claims a report can make, and leaves it that way silently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

__all__ = ["available_devices", "load_eval_module", "pick_device"]

#: Class name identifying the EMA callback's entry in a checkpoint's ``callbacks`` dict.
#: The key Lightning writes is the callback's ``state_key``, which carries the callback's
#: arguments for a parametrized instance, so the entry is found by this substring and then
#: required to be unique -- see :func:`_ema_state_key`.
_EMA_CALLBACK_MARKER = "EMACallback"


def load_eval_module(checkpoint: Path, *, use_ema: bool) -> tuple[DetectionLitModule, dict[str, object]]:
    """Load an eval-mode module from a Lightning checkpoint, optionally with EMA weights.

    Args:
        checkpoint: Path to the Lightning ``.ckpt`` file.
        use_ema: When ``True``, overlay the :class:`EMACallback` shadow stored in the
            checkpoint onto the module's parameters and buffers.

    Returns:
        The eval-mode module and a small provenance dict (checkpoint, epoch, step, EMA).

    Raises:
        ValueError: If ``use_ema`` is requested and the checkpoint carries no EMA shadow,
            names the EMA callback in more than one ``callbacks`` entry, or carries a
            shadow that does not cover every floating-point parameter and buffer.

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
        # One resolution of the callback's key, used for both lookups: the shadow and the
        # counter that says how many updates produced it must come from the same entry.
        state_key = _overlay_ema(module, payload)
        info["ema_updates"] = int(payload["callbacks"][state_key]["num_updates"])
    module.eval()
    return module, info


def pick_device(requested: str) -> torch.device:
    """Resolve ``auto`` to the fastest available backend, else check ``requested`` exists here.

    A named backend is held to the same availability question ``auto`` asks, rather than
    passed through (WP-171). ``--device cuda`` on a machine with no CUDA used to be
    accepted and then fail several minutes later, inside the first ``.to(device)`` — after
    the checkpoint was loaded and, for a long run, after the annotations were parsed —
    with a message about a tensor rather than about the flag. The refusal names the
    request and what this machine does have, because the useful next command is the one
    the caller cannot guess from a stack trace.

    Args:
        requested: ``"auto"`` or any string :class:`torch.device` accepts.

    Returns:
        The chosen device.

    Raises:
        ValueError: If ``requested`` names a backend this machine cannot run on.

    Examples:
        >>> pick_device("cpu")
        device(type='cpu')
        >>> pick_device("auto").type in available_devices()
        True
    """
    if requested != "auto":
        device = torch.device(requested)
        if not _is_available(device):
            raise ValueError(
                f"device={requested!r} names a {device.type} backend this machine does not have; "
                f"available here: {available_devices()}"
            )
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def available_devices() -> tuple[str, ...]:
    """Name the backends this machine can actually run on, fastest first.

    Probed on every call rather than resolved once at import: a device string is checked
    where it enters, and a module-level snapshot would answer for the machine that
    imported this module rather than for the one running now.

    Returns:
        The usable ``--device`` spellings, ending in ``"cpu"`` — always available — and
        ``"auto"``, which is what a caller who does not want to know should pass.

    Examples:
        >>> available_devices()[-2:]
        ('cpu', 'auto')
    """
    names = ["cuda"] if torch.cuda.is_available() else []
    if torch.backends.mps.is_available():
        names.append("mps")
    return (*names, "cpu", "auto")


def _is_available(device: torch.device) -> bool:
    """Report whether ``device``'s backend can be run on here.

    Only the two accelerators this project ships support for are probed. Anything else
    — ``cpu``, and whatever other type :class:`torch.device` accepts — passes through
    unjudged, because a check that guessed at a backend nobody here tests would refuse
    working configurations rather than protect any.

    Examples:
        >>> import torch
        >>> _is_available(torch.device("cpu"))
        True
    """
    if device.type == "cuda":
        return bool(torch.cuda.is_available())
    if device.type == "mps":
        return bool(torch.backends.mps.is_available())
    return True


def _ema_state_key(callbacks: dict[str, dict[str, object]]) -> str:
    """Resolve the one ``callbacks`` entry that belongs to the EMA callback.

    Lightning keys the ``callbacks`` dict by each callback's ``state_key``, which for a
    parametrized callback is the class name followed by its arguments -- so the match is on
    the class name rather than on the whole key. Exactly one entry may match: two would
    make "the EMA shadow" ambiguous, and picking one (the previous behaviour, where the
    last match silently won) reports numbers under a name that does not identify them.

    Args:
        callbacks: The checkpoint's ``callbacks`` mapping.

    Returns:
        The single key naming the EMA callback.

    Raises:
        ValueError: If no entry names the EMA callback, or if more than one does.
    """
    matches = [name for name in callbacks if _EMA_CALLBACK_MARKER in name]
    if len(matches) > 1:
        raise ValueError(
            f"EMA weights requested but the checkpoint carries {len(matches)} callback entries naming "
            f"{_EMA_CALLBACK_MARKER}: {matches}. Which of them is 'the' EMA shadow is not something this "
            f"loader may decide by iteration order."
        )
    if not matches:
        raise ValueError(
            f"EMA weights requested but no callback entry names {_EMA_CALLBACK_MARKER}; "
            f"the checkpoint carries {sorted(callbacks)}"
        )
    return matches[0]


def _float_tensors(module: DetectionLitModule) -> dict[str, Tensor]:
    """Name every floating-point parameter and buffer -- the exact set a shadow covers.

    Mirrors :class:`~lucid_yolo.ptl.callbacks.EMACallback`'s own rule on the writing side:
    integer buffers (batch-norm's ``num_batches_tracked``) are excluded, because an
    exponential average of a step counter is meaningless and the writer never shadows one.
    Parameter and buffer names share one namespace, so the merged mapping loses nothing.

    Args:
        module: The module whose tensors are named.

    Returns:
        A name-keyed mapping of the module's floating-point parameters and buffers.
    """
    named = [*module.named_parameters(), *module.named_buffers()]
    return {name: tensor for name, tensor in named if tensor.is_floating_point()}


def _overlay_ema(module: DetectionLitModule, payload: dict[str, object]) -> str:
    """Copy the checkpoint's EMA shadow over the module's parameters and buffers.

    The callback's key is resolved once and handed back, so the caller's ``num_updates``
    lookup reads the same entry the shadow came from rather than a literal name that may
    not be the one that matched.

    Coverage is checked **before** anything is copied. Raising afterwards would leave the
    module half EMA and half raw -- neither of the two claims a report can make -- and a
    shadow that merely omits a tensor would otherwise leave that tensor at its raw value
    with nothing said.

    Args:
        module: The module to overlay in place.
        payload: The raw checkpoint dict.

    Returns:
        The ``callbacks`` key the shadow was taken from.

    Raises:
        ValueError: If the checkpoint carries no EMA shadow, if more than one callback
            entry names the EMA callback, or if the shadow omits any floating-point
            parameter or buffer of ``module``.
    """
    callbacks = cast("dict[str, dict[str, object]]", payload.get("callbacks", {}))
    state_key = _ema_state_key(callbacks)
    shadow = cast("dict[str, Tensor] | None", callbacks[state_key].get("shadow"))
    if not shadow:
        raise ValueError(f"EMA weights requested but {payload.get('checkpoint', 'the checkpoint')} carries no shadow")
    tensors = _float_tensors(module)
    missing = sorted(set(tensors) - set(shadow))
    if missing:
        raise ValueError(
            f"EMA shadow in {payload.get('checkpoint', 'the checkpoint')} covers {len(shadow)} tensors but omits "
            f"{len(missing)} of the module's floating-point parameters and buffers: {missing}. Overlaying it "
            f"would report a mixture of EMA and raw weights under the EMA name."
        )
    with torch.no_grad():
        for name, tensor in tensors.items():
            tensor.copy_(shadow[name])
    return state_key
