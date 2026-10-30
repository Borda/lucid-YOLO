# SPDX-License-Identifier: Apache-2.0
"""CPU checks for replay resource lifetime and customized tensor enumeration."""

from __future__ import annotations

import weakref
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
from torch import nn

from lucid_yolo.ptl import gradient_norm_capture as implementation
from lucid_yolo.ptl.gradient_norm_capture import _GradientNormCapture
from lucid_yolo.ptl.network_capture import _members


@pytest.mark.parametrize("drain_fails", [False, True])
def test_norm_clear_keeps_resources_alive_during_drain(drain_fails: bool) -> None:
    """Reset precedes draining, but graph-pool resources survive until the drain finishes."""
    owner = _GradientNormCapture()
    # Only ownership is exercised here; an ordinary weak-referenceable object
    # stands in for the opaque graph without constructing a CUDA resource.
    owner._graph = nn.Module()
    owner._inputs = [torch.ones(3)]
    owner._norm = torch.ones(())
    owner._signature = ("recorded",)
    owner._disabled = True
    device = owner._device = torch.device("cuda:3")
    resources = tuple(weakref.ref(value) for value in (owner._graph, owner._inputs[0], owner._norm))

    def drain(recorded_device: torch.device | None) -> None:
        """Observe ownership at the CUDA synchronization boundary without invoking CUDA."""
        assert recorded_device == device
        assert owner._graph is None and owner._inputs == [] and owner._norm is None
        assert owner._signature is None and owner._device is None and not owner._disabled
        assert all(resource() is not None for resource in resources)
        if drain_fails:
            raise RuntimeError("drain failure")

    with patch.object(implementation, "drain_device", side_effect=drain, create=True) as drained:
        with pytest.raises(RuntimeError, match="drain failure") if drain_fails else nullcontext():
            owner.clear()
        drained.assert_called_once_with(device)
    assert all(resource() is None for resource in resources)
    assert owner._graph is None and owner._inputs == [] and owner._norm is None
    assert owner._signature is None and owner._device is None and not owner._disabled


def test_norm_clear_without_graph_never_drains() -> None:
    """Construction and repeated empty cleanup cannot synchronize the CUDA runtime."""
    with patch.object(implementation, "drain_device", create=True) as drained:
        owner = _GradientNormCapture()
        owner._device = torch.device("cuda:3")
        owner.clear()
        owner.clear()
    drained.assert_not_called()
    assert owner._device is None


@pytest.mark.parametrize("class_override", [False, True])
def test_members_respect_named_members_override(class_override: bool) -> None:
    """A private native enumeration hook controls both parameter and buffer visibility."""
    owner = nn.Sequential(nn.Linear(3, 2))
    owner.register_buffer("sample", torch.ones(3))
    with patch.object(type(owner) if class_override else owner, "_named_members", return_value=iter(())):
        modules, parameters, buffers = _members(owner)
        assert modules == tuple(owner.modules())
        assert parameters == tuple(owner.parameters()) == ()
        assert buffers == tuple(owner.buffers()) == ()
