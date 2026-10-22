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

The file is read **once**, restricted. ``torch.load(..., weights_only=True)`` refuses any
global the file names outside torch's allowlist while it is still parsing the pickle
stream, so a checkpoint carrying an arbitrary reducer is rejected before that reducer runs,
and the module is then built from that same payload rather than handed to Lightning's
``load_from_checkpoint`` for a second, unrestricted read. Earlier releases read the file
twice -- restricted first, as the gate, then through Lightning --
which deserialized every tensor twice (twice the peak host memory) and checked the file in
its first form while loading it in its second if it was replaced between the reads. One
read closes both.

Building the module needs no private Lightning API, because the class is simple to
build: :class:`~lucid_yolo.ptl.module.DetectionLitModule` takes primitive keyword
arguments only, so its saved ``hyper_parameters``, filtered to the ``__init__``
signature, construct it directly. The filter is what a ``lucid-yolo fit`` checkpoint
needs: LightningCLI saves an ``_instantiator`` entry beside the arguments, which Lightning
uses to rebuild the class through jsonargparse, and which, for this class, builds the
same object a plain call does. Then come the public steps Lightning's loader takes
after construction: the ``on_load_checkpoint`` hook and a ``load_state_dict`` at the
module's own ``strict_loading``. What this skips is Lightning's legacy-checkpoint
migration, which rewrites loop and callback state from Lightning 1.x layouts. The
fields read here, ``hyper_parameters``, ``state_dict``, ``epoch``, ``global_step`` and the
EMA callback's entry, are written in the same layout by every Lightning release in
``pyproject.toml``'s supported range.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import torch

from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from torch import Tensor

__all__ = ["available_devices", "load_eval_module", "pick_device"]

#: Class name identifying the EMA callback's entry in a checkpoint's ``callbacks`` dict.
#: The key Lightning writes is the callback's ``state_key``, which carries the callback's
#: arguments for a parametrized instance, so the entry is found by this substring and then
#: required to be unique -- see :func:`_ema_state_key`.
_EMA_CALLBACK_MARKER = "EMACallback"

#: Keyword arguments :class:`DetectionLitModule` accepts, so saved hyper-parameters can be
#: filtered to them: a ``lucid-yolo fit`` checkpoint saves an ``_instantiator`` entry
#: beside them, which is Lightning's own bookkeeping, not an argument.
_INIT_ARGUMENTS = frozenset(inspect.signature(DetectionLitModule.__init__).parameters) - {"self"}


def load_eval_module(checkpoint: Path, *, use_ema: bool) -> tuple[DetectionLitModule, dict[str, object]]:
    """Load an eval-mode module from a Lightning checkpoint, optionally with EMA weights.

    Args:
        checkpoint: Path to the Lightning ``.ckpt`` file.
        use_ema: When ``True``, overlay the :class:`EMACallback` shadow stored in the
            checkpoint onto the module's parameters and buffers.

    Returns:
        The eval-mode module and a small provenance dict (checkpoint, epoch, step, EMA).

    Raises:
        pickle.UnpicklingError: If the checkpoint names anything restricted loading refuses.
            Raised by the one read, before any reducer the file carries is called -- see
            the module docstring.
        ValueError: If the checkpoint carries no ``hyper_parameters`` or ``state_dict``
            mapping, no ``epoch`` or ``global_step``, or -- under
            ``use_ema`` -- no ``num_updates`` beside the shadow; and if ``use_ema`` is
            requested and the checkpoint carries no EMA shadow, names the EMA callback in
            more than one ``callbacks`` entry, or carries a shadow that does not cover
            every floating-point parameter and buffer.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> load_eval_module(Path("/nonexistent.ckpt"), use_ema=False)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        FileNotFoundError: ...

        ```
    """
    # Restricted, unconditionally, and the only read: the module is built from this payload.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    module = _module_from_payload(payload, checkpoint)
    info: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "epoch": _required_int(payload, "epoch", str(checkpoint)),
        "global_step": _required_int(payload, "global_step", str(checkpoint)),
        "ema": use_ema,
    }
    if use_ema:
        # One resolution of the callback's key, used for both lookups: the shadow and the
        # counter that says how many updates produced it must come from the same entry.
        state_key = _overlay_ema(module, payload, checkpoint)
        entry = cast("Mapping[str, Any]", payload["callbacks"][state_key])
        info["ema_updates"] = _required_int(entry, "num_updates", f"{checkpoint} callbacks entry {state_key!r}")
    module.eval()
    return module, info


def _module_from_payload(payload: Mapping[str, Any], checkpoint: Path) -> DetectionLitModule:
    """Build the module a checkpoint describes from its already-read payload.

    The public steps of Lightning's own loader, in its order: construct from the saved
    hyper-parameters, give the module its ``on_load_checkpoint`` hook, then load the
    weights at the module's ``strict_loading`` -- so a missing or unexpected key raises
    PyTorch's own ``load_state_dict`` error, as it did through ``load_from_checkpoint``.

    Args:
        payload: The checkpoint dict, read with ``weights_only=True``.
        checkpoint: Path to name when a required entry is missing.

    Returns:
        The module, in training mode, with the checkpoint's raw weights.

    Raises:
        ValueError: If ``hyper_parameters`` or ``state_dict`` is absent or not a mapping.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=2)
        >>> payload = {"hyper_parameters": dict(module.hparams), "state_dict": module.state_dict()}
        >>> _module_from_payload(payload, Path("run.ckpt")).task
        'detect'
        >>> _module_from_payload({"state_dict": {}}, Path("run.ckpt"))
        Traceback (most recent call last):
        ValueError: run.ckpt carries no 'hyper_parameters' mapping; it holds ['state_dict']...

        ```
    """
    hparams = _required_mapping(payload, "hyper_parameters", str(checkpoint))
    state_dict = _required_mapping(payload, "state_dict", str(checkpoint))
    module = DetectionLitModule(**{name: value for name, value in hparams.items() if name in _INIT_ARGUMENTS})
    module.on_load_checkpoint(dict(payload))
    module.load_state_dict(state_dict, strict=module.strict_loading)
    return module


def _required_mapping(payload: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    """Read ``key`` as a mapping, or refuse naming the key, the file, and what is there.

    Args:
        payload: The checkpoint dict.
        key: The key that must hold a mapping.
        where: The checkpoint path to name in the refusal.

    Returns:
        The mapping at ``key``.

    Raises:
        ValueError: If ``key`` is absent or holds something other than a mapping.

    Examples:
        ```pycon
        >>> _required_mapping({"state_dict": {}}, "state_dict", "run.ckpt")
        {}

        ```
    """
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{where} carries no {key!r} mapping; it holds {sorted(payload)}. Every Lightning checkpoint of "
            f"this project is written with one, so the file is truncated, hand-built, or not a checkpoint "
            f"of this project."
        )
    return value


def _required_int(payload: Mapping[str, Any], key: str, where: str) -> int:
    """Read ``key`` as an integer, or refuse naming the key, the file, and what is there.

    Indexing the payload raw turns a truncated or hand-built checkpoint into ``KeyError:
    'epoch'`` -- which names the dict lookup that failed and nothing an operator can act
    on, on a path ``lucid-eval`` and ``lucid-predict`` surface directly.

    Args:
        payload: The checkpoint dict, or one entry inside it.
        key: The key that must be present.
        where: What to name in the refusal -- the checkpoint path, plus the entry when the
            lookup is a nested one.

    Returns:
        The value at ``key``, as an ``int``.

    Raises:
        ValueError: If ``key`` is absent.

    Examples:
        ```pycon
        >>> _required_int({"epoch": 3}, "epoch", "run.ckpt")
        3
        >>> _required_int({}, "epoch", "run.ckpt")
        Traceback (most recent call last):
        ValueError: run.ckpt carries no 'epoch'; it holds []...

        ```
    """
    if key not in payload:
        raise ValueError(
            f"{where} carries no {key!r}; it holds {sorted(payload)}. Every Lightning checkpoint is written "
            f"with this key, so the file is truncated, hand-built, or not a checkpoint of this project."
        )
    return int(payload[key])


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
        ```pycon
        >>> pick_device("cpu")
        device(type='cpu')
        >>> pick_device("auto").type in available_devices()
        True

        ```
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
        ```pycon
        >>> available_devices()[-2:]
        ('cpu', 'auto')

        ```
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
        ```pycon
        >>> import torch
        >>> _is_available(torch.device("cpu"))
        True

        ```
    """
    if device.type == "cuda":
        return bool(torch.cuda.is_available())
    if device.type == "mps":
        return bool(torch.backends.mps.is_available())
    return True


def _ema_state_key(callbacks: dict[str, dict[str, object]], checkpoint: Path) -> str:
    """Resolve the one ``callbacks`` entry that belongs to the EMA callback.

    Lightning keys the ``callbacks`` dict by each callback's ``state_key``, which for a
    parametrized callback is the class name followed by its arguments -- so the match is on
    the class name rather than on the whole key. Exactly one entry may match: two would
    make "the EMA shadow" ambiguous, and picking one (the previous behaviour, where the
    last match silently won) reports numbers under a name that does not identify them.

    Args:
        callbacks: The checkpoint's ``callbacks`` mapping.
        checkpoint: Path to name when no entry, or more than one, names the EMA callback.

    Returns:
        The single key naming the EMA callback.

    Raises:
        ValueError: If no entry names the EMA callback, or if more than one does.
    """
    matches = [name for name in callbacks if _EMA_CALLBACK_MARKER in name]
    if len(matches) > 1:
        raise ValueError(
            f"EMA weights requested but {checkpoint} carries {len(matches)} callback entries naming "
            f"{_EMA_CALLBACK_MARKER}: {matches}. Which of them is 'the' EMA shadow is not something this "
            f"loader may decide by iteration order."
        )
    if not matches:
        raise ValueError(
            f"EMA weights requested but no callback entry names {_EMA_CALLBACK_MARKER}; "
            f"{checkpoint} carries {sorted(callbacks)}"
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


def _overlay_ema(module: DetectionLitModule, payload: dict[str, object], checkpoint: Path) -> str:
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
        checkpoint: Path to name in every refusal.

    Returns:
        The ``callbacks`` key the shadow was taken from.

    Raises:
        ValueError: If the checkpoint carries no EMA shadow, if more than one callback
            entry names the EMA callback, or if the shadow omits any floating-point
            parameter or buffer of ``module``.
    """
    callbacks = cast("dict[str, dict[str, object]]", payload.get("callbacks", {}))
    state_key = _ema_state_key(callbacks, checkpoint)
    shadow = cast("dict[str, Tensor] | None", callbacks[state_key].get("shadow"))
    if not shadow:
        raise ValueError(f"EMA weights requested but {checkpoint} carries no shadow")
    tensors = _float_tensors(module)
    missing = sorted(set(tensors) - set(shadow))
    if missing:
        raise ValueError(
            f"EMA shadow in {checkpoint} covers {len(shadow)} tensors but omits "
            f"{len(missing)} of the module's floating-point parameters and buffers: {missing}. Overlaying it "
            f"would report a mixture of EMA and raw weights under the EMA name."
        )
    with torch.no_grad():
        for name, tensor in tensors.items():
            tensor.copy_(shadow[name])
    return state_key
