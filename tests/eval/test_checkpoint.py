# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the EMA overlay every acceptance script loads through (WP-095, WP-174).

:func:`~lucid_yolo.eval.checkpoint.load_eval_module` is the one door ``lucid-eval``,
``lucid-predict`` and the export path all read a checkpoint through, and ``--ema`` is on by
default, so ``_overlay_ema`` decides which weights every published number was measured
with. It had no test of its own: the module sat at 47% with the whole overlay unexecuted.

Nothing here needs a checkpoint file. The payload ``torch.load`` returns is a plain dict,
so it is built by hand — which is what makes the awkward shapes reachable at all: a
callback key that is not the literal class name, two entries that both claim to be the EMA
one, a shadow that covers most of the module. Each of those is a checkpoint a real
training run can write, and each used to resolve into either a silently wrong overlay or a
bare ``KeyError`` raised after the module had already been mutated.

The three claims under test:

- **The key is resolved once.** The shadow and the ``num_updates`` counter must come from
  the same ``callbacks`` entry. Reading one by substring and the other by literal name is
  what made a parametrized callback's checkpoint die after the overlay succeeded.
- **An ambiguous key is refused.** Two matching entries used to resolve by iteration
  order, which is not a decision this loader is entitled to make.
- **A partial shadow is refused before anything is copied.** Half EMA and half raw is
  neither of the two claims a report can make, and the failure has to arrive while the
  module is still intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from lucid_yolo.eval import checkpoint as loader
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

#: The value every shadow tensor is planted at: not a plausible initialization, so a
#: tensor still holding its raw value after an overlay is visible rather than arguable.
_SHADOW_VALUE = 0.5

#: A ``state_key`` Lightning writes for a *parametrized* callback — the class name
#: followed by its arguments. This is the shape that used to pass the substring match and
#: then fail the literal-name lookup, after the overlay had already been applied.
_PARAMETRIZED_KEY = "EMACallback{'decay': 0.9999, 'update_every': 1}"

#: Update count carried beside the shadow, read back through the resolved key.
_NUM_UPDATES = 37


def _module() -> DetectionLitModule:
    """Return a tiny detection module: floating-point parameters, buffers, and an int buffer.

    Batch-norm is what makes this fixture worth building rather than faking — it supplies
    floating-point buffers (``running_mean``, ``running_var``) that a parameters-only
    overlay would miss, and an integer ``num_batches_tracked`` the shadow must *not* carry.

    Returns:
        The module, untrained and in eval mode.

    Examples:
        >>> module = _module()
        >>> module.task
        'detect'
        >>> any(name.endswith("num_batches_tracked") for name, _ in module.named_buffers())
        True
    """
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=2).eval()


def _full_shadow(module: torch.nn.Module, value: float = _SHADOW_VALUE) -> dict[str, Tensor]:
    """Build the shadow ``EMACallback`` would write: every floating-point tensor, at ``value``.

    Integer buffers are excluded, exactly as the writing side excludes them — an
    exponential average of a step counter is meaningless — so this is the *complete*
    shadow the overlay must accept without complaint.

    Args:
        module: The module whose tensors are shadowed.
        value: Constant every shadow tensor is filled with.

    Returns:
        A name-keyed mapping of floating-point shadow tensors.

    Examples:
        >>> import torch
        >>> shadow = _full_shadow(torch.nn.BatchNorm1d(2))
        >>> sorted(shadow)  # num_batches_tracked is integer, so it is not shadowed
        ['bias', 'running_mean', 'running_var', 'weight']
        >>> float(shadow["weight"][0])
        0.5
    """
    named = [*module.named_parameters(), *module.named_buffers()]
    return {name: torch.full_like(tensor, value) for name, tensor in named if tensor.is_floating_point()}


def _payload(shadow: dict[str, Tensor] | None, state_key: str = "EMACallback") -> dict[str, object]:
    """Wrap ``shadow`` in the checkpoint dict shape ``torch.load`` hands the overlay.

    Args:
        shadow: The shadow mapping, or ``None`` for a checkpoint that carries none.
        state_key: The ``callbacks`` key the entry is filed under. Defaults to the bare
            class name, which is what an unparametrized callback writes.

    Returns:
        A checkpoint payload holding exactly the keys the overlay reads.

    Examples:
        >>> import torch
        >>> payload = _payload({"weight": torch.zeros(2)})
        >>> sorted(payload["callbacks"]["EMACallback"])
        ['num_updates', 'shadow']
        >>> _payload(None, state_key="EMACallback{'decay': 0.9}")["callbacks"]["EMACallback{'decay': 0.9}"]["shadow"]
    """
    return {"callbacks": {state_key: {"shadow": shadow, "num_updates": _NUM_UPDATES}}}


def _float_snapshot(module: torch.nn.Module) -> dict[str, Tensor]:
    """Clone every floating-point tensor of ``module``, to prove a failed overlay changed none.

    Args:
        module: The module to snapshot.

    Returns:
        Detached clones, name-keyed.

    Examples:
        >>> import torch
        >>> layer = torch.nn.BatchNorm1d(2)
        >>> before = _float_snapshot(layer)
        >>> bool(torch.equal(before["weight"], layer.weight.detach()))
        True
    """
    named = [*module.named_parameters(), *module.named_buffers()]
    return {name: tensor.detach().clone() for name, tensor in named if tensor.is_floating_point()}


class TestOverlayEma:
    """The overlay itself, against hand-built payloads — no checkpoint file involved."""

    def test_every_floating_point_tensor_takes_the_shadows_value(self) -> None:
        """A complete shadow lands on parameters and float buffers alike, leaving counters alone.

        The buffers are the half a parameters-only overlay silently misses: batch-norm
        statistics that stay raw while the weights are EMA produce a number that is neither
        of the two claims the report can make. The integer ``num_batches_tracked`` must be
        untouched for the same reason it is never shadowed.
        """
        module = _module()
        integer_buffers = {
            name: buffer.clone() for name, buffer in module.named_buffers() if not buffer.is_floating_point()
        }

        state_key = loader._overlay_ema(module, _payload(_full_shadow(module)))

        assert state_key == "EMACallback"
        overlaid = _float_snapshot(module)
        assert overlaid, "the module must carry floating-point tensors for this to mean anything"
        for name, tensor in overlaid.items():
            assert bool((tensor == _SHADOW_VALUE).all()), f"{name} kept its raw value"
        assert integer_buffers, "batch-norm should give this module an integer buffer"
        for name, before in integer_buffers.items():
            assert torch.equal(module.get_buffer(name), before), f"{name} is not an EMA quantity"

    def test_a_parametrized_state_key_is_resolved_and_handed_back(self) -> None:
        """A key carrying the callback's arguments resolves, and the overlay names it to its caller.

        Returning the resolved key is the whole fix: the caller's ``num_updates`` lookup
        reads this entry rather than a literal ``"EMACallback"`` that is not present, which
        is how a parametrized callback's checkpoint used to raise a bare ``KeyError`` with
        the module already overlaid.
        """
        module = _module()

        state_key = loader._overlay_ema(module, _payload(_full_shadow(module), state_key=_PARAMETRIZED_KEY))

        assert state_key == _PARAMETRIZED_KEY

    @pytest.mark.parametrize(
        "shadow",
        [pytest.param(None, id="absent"), pytest.param({}, id="empty")],
    )
    def test_a_checkpoint_with_no_shadow_raises_rather_than_evaluating_raw_weights(
        self, shadow: dict[str, Tensor] | None
    ) -> None:
        """A shadow that is missing or empty refuses, because raw numbers are a different claim.

        "The EMA numbers" and "the raw numbers" are separate statements about a run, and
        quietly reporting the second under the first name is unfalsifiable — nothing
        downstream can tell which it received.
        """
        module = _module()

        with pytest.raises(ValueError, match="carries no shadow"):
            loader._overlay_ema(module, _payload(shadow))

    def test_no_entry_naming_the_callback_raises_and_lists_what_is_there(self) -> None:
        """A checkpoint saved without the EMA callback names the callbacks it does carry.

        The remedy differs by cause — a run trained without the callback needs ``--no-ema``,
        a renamed callback needs the loader taught about it — so the message has to carry
        enough to tell them apart.
        """
        module = _module()

        with pytest.raises(ValueError, match="no callback entry names"):
            loader._overlay_ema(module, {"callbacks": {"ModelCheckpoint": {}}})

    def test_two_entries_naming_the_callback_raise_rather_than_the_last_one_winning(self) -> None:
        """Two EMA-ish entries are ambiguous, and ambiguity is refused, not resolved by order.

        The previous loop kept overwriting ``shadow`` as it walked the dict, so whichever
        entry came last silently won. Dict order is insertion order, which here means
        callback registration order — not a fact anyone reading the report could see, and
        not a basis for choosing which weights the numbers describe.
        """
        module = _module()
        shadow = _full_shadow(module)
        payload = {
            "callbacks": {
                "EMACallback": {"shadow": shadow, "num_updates": 1},
                _PARAMETRIZED_KEY: {"shadow": shadow, "num_updates": 2},
            }
        }

        with pytest.raises(ValueError, match="2 callback entries naming"):
            loader._overlay_ema(module, payload)

    def test_a_partial_shadow_names_what_is_missing_and_leaves_the_module_untouched(self) -> None:
        """A shadow short of one tensor refuses **before** copying, so no half-overlaid module exists.

        This is the failure the old code could not report at all: it copied the shadow's own
        keys, so any module tensor the shadow lacked simply kept its raw value and the run
        reported a mixture under the EMA name. Checking coverage after the copy would name
        the problem but leave the module in the state the message says is invalid, so the
        order matters as much as the check — the snapshot comparison is what pins it.
        """
        module = _module()
        shadow = _full_shadow(module)
        dropped = sorted(shadow)[0]
        del shadow[dropped]
        before = _float_snapshot(module)

        with pytest.raises(ValueError, match=f"omits 1 of the module's floating-point.*{dropped}"):
            loader._overlay_ema(module, _payload(shadow))

        after = _float_snapshot(module)
        assert set(after) == set(before)
        for name, tensor in after.items():
            assert torch.equal(tensor, before[name]), f"{name} was overlaid before the refusal"


class TestLoadEvalModuleWithEma:
    """The overlay reached the way every command reaches it: through a real checkpoint file."""

    def test_a_parametrized_callback_key_survives_the_whole_load(self, tmp_path: Path) -> None:
        """``--ema`` on a checkpoint whose callback carries arguments loads and reports its counter.

        The end-to-end shape of the WP-174 defect. The substring match accepted this
        checkpoint and the overlay ran; the ``num_updates`` lookup then went to a literal
        ``"EMACallback"`` that is not a key of this dict and raised ``KeyError`` — an
        unhandled crash, from an operator's point of view, on a checkpoint that is entirely
        well-formed. ``ema_updates`` reaching the provenance dict is what says both lookups
        landed on the same entry.
        """
        module = _module()
        path = tmp_path / "ema.ckpt"
        torch.save(
            {
                "state_dict": module.state_dict(),
                "hyper_parameters": dict(module.hparams),
                "epoch": 3,
                "global_step": 120,
                "pytorch-lightning_version": "2.4.0",
                "loops": {},
                "callbacks": {_PARAMETRIZED_KEY: {"shadow": _full_shadow(module), "num_updates": _NUM_UPDATES}},
                "optimizer_states": [],
                "lr_schedulers": [],
            },
            path,
        )

        loaded, info = loader.load_eval_module(path, use_ema=True)

        assert info["ema"] is True
        assert info["ema_updates"] == _NUM_UPDATES
        assert info["epoch"] == 3
        assert not loaded.training
        for name, tensor in _float_snapshot(loaded).items():
            assert bool((tensor == _SHADOW_VALUE).all()), f"{name} did not take the shadow"
