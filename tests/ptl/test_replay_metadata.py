# SPDX-License-Identifier: Apache-2.0
"""Replay metadata reuse preserves live storage and attribute drift checks."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from operator import is_
from unittest.mock import patch

import pytest
import torch
from torch import nn

from lucid_yolo.ptl.network_capture import (
    _members,
    _module_breach,
    _ModuleMetadata,
    _network_signature,
    _NetworkCapture,
)


def _network() -> nn.Sequential:
    """Build asymmetric stages with scalar attributes and a real registered buffer.

    Examples:
        >>> len(_network())
        2
    """
    network = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 2))
    network.register_buffer("sample", torch.zeros(3))
    return network


def test_members_preserve_preorder_and_identity_deduplication() -> None:
    """Aliases retain their first position without losing either tensor namespace."""
    first, second = nn.Linear(3, 5), nn.Linear(5, 2)
    owner = nn.Module()
    root = nn.Parameter(torch.ones(4))
    owner.register_parameter("root", root)
    owner.register_parameter("alias", root)
    owner.layers = nn.ModuleList((first, second, first))
    buffer = torch.ones(7)
    owner.register_buffer("buffer", buffer)
    owner.register_buffer("buffer_alias", buffer)
    modules, parameters, buffers = _members(owner)
    expected_modules = owner, owner.layers, first, second
    expected_parameters = root, first.weight, first.bias, second.weight, second.bias
    assert len(modules) == 4 and all(map(is_, modules, expected_modules))
    assert len(parameters) == 5 and all(map(is_, parameters, expected_parameters))
    assert len(buffers) == 1 and buffers[0] is buffer


def test_unchanged_bindings_skip_full_scalar_inspection() -> None:
    """The optimization must remain observable even when all numerical outputs agree."""
    network = _network()
    members = _members(network)
    before = _network_signature(network)
    metadata = _ModuleMetadata(members[0], before[0])
    with patch("lucid_yolo.ptl.network_capture._module_signature", side_effect=AssertionError("rescanned")):
        assert _network_signature(network, _members(network), metadata) == before
    network[0].weight.data.add_(1)
    network.sample.add_(1)
    with patch(
        "lucid_yolo.ptl.network_capture._module_signature", side_effect=AssertionError("value update rescanned")
    ):
        assert _network_signature(network, _members(network), metadata) == before


@pytest.mark.parametrize("mutation", ["training", "scalar", "tuple", "new_scalar", "class_forward", "child"])
def test_changed_module_contract_reaches_original_signature(mutation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A certificate miss rereads the changed contract instead of returning stale metadata."""
    network = _network()
    before = _network_signature(network)
    metadata = _ModuleMetadata(_members(network)[0], before[0])
    if mutation == "training":
        network[0].eval()
    elif mutation == "scalar":
        network[0].in_features = 7
    elif mutation == "tuple":
        network[0].layout_marker = (2, 3)
    elif mutation == "new_scalar":
        network[0].generation = 3
    elif mutation == "class_forward":
        monkeypatch.setattr(nn.Linear, "forward", lambda self, inputs: inputs)
    else:
        network[1] = nn.Linear(5, 2)
    after = _network_signature(network, _members(network), metadata)
    assert after != before
    assert after == _network_signature(network)
    fields = dict(after[0][1][3])
    if mutation == "scalar":
        assert fields["in_features"] == 7
    elif mutation == "tuple":
        assert fields["layout_marker"] == (2, 3)
    elif mutation == "new_scalar":
        assert fields["generation"] == 3


@pytest.mark.parametrize("mutation", ["parameter", "buffer", "storage", "dtype", "trainability", "stride", "shape"])
def test_live_tensor_contract_is_not_cached(mutation: str) -> None:
    """Unchanged Python attributes cannot certify a tensor's current storage or layout."""
    network = _network()
    before = _network_signature(network)
    metadata = _ModuleMetadata(_members(network)[0], before[0])
    if mutation == "parameter":
        network[0].weight = nn.Parameter(network[0].weight.detach().clone())
    elif mutation == "buffer":
        network.sample = network.sample.clone()
    elif mutation == "storage":
        network[0].weight.data = network[0].weight.detach().clone()
    elif mutation == "dtype":
        weight = network[0].weight
        weight.data = weight.detach().view(torch.float16).as_strided(weight.shape, weight.stride())
    elif mutation == "trainability":
        network[0].weight.requires_grad_(False)
    elif mutation == "stride":
        network[0].weight.data = network[0].weight.detach().as_strided((5, 3), (1, 5))
    else:
        network[0].weight.data = network[0].weight.detach().as_strided((3, 3), (3, 1))
    after = _network_signature(network, _members(network), metadata)
    assert after != before
    assert after == _network_signature(network)


def test_tensor_attribute_replacement_never_compares_tensor_values() -> None:
    """An equal-valued vector is a cache miss whose fallback keeps the original scalar signature."""
    network = _network()
    network.marker = torch.ones(4)
    before = _network_signature(network)
    metadata = _ModuleMetadata(_members(network)[0], before[0])
    network.marker = torch.ones(4)
    assert _network_signature(network, _members(network), metadata) == before


def test_equal_scalar_rebinding_keeps_original_value_semantics() -> None:
    """A new float object may miss the identity certificate without retiring valid scalar metadata."""
    network = _network()
    network[0].scale_marker = float("1.25")
    before = _network_signature(network)
    metadata = _ModuleMetadata(_members(network)[0], before[0])
    network[0].scale_marker = float("1.25")
    assert _network_signature(network, _members(network), metadata) == before


def test_clear_and_copy_discard_metadata_owner() -> None:
    """A plan cannot retain the previous fit through reset or serialization."""
    network = _network()
    holder = _NetworkCapture()
    holder._metadata = _ModuleMetadata(_members(network)[0], _network_signature(network)[0])
    assert copy.deepcopy(holder)._metadata is None
    holder.clear()
    assert holder._metadata is None


def test_members_follow_custom_module_enumeration() -> None:
    """An overridden traversal keeps its native order instead of using the iterative route."""

    class Reordered(nn.Sequential):
        def named_modules(
            self, memo: set[nn.Module] | None = None, prefix: str = "", remove_duplicate: bool = True
        ) -> Iterator[tuple[str, nn.Module]]:
            yield prefix, self
            yield prefix + ".1", self[1]
            yield prefix + ".0", self[0]

    owner = nn.Sequential(Reordered(nn.Linear(3, 5), nn.Linear(5, 2)))
    modules, parameters, buffers = _members(owner)
    assert all(map(is_, modules, tuple(owner.modules())))
    assert len(modules) == 4
    assert len(parameters) == 4 and all(map(is_, parameters, tuple(owner.parameters())))
    assert buffers == ()


def test_members_read_live_registration_and_null_children() -> None:
    """Topology replacement and newly registered aliases preserve the native enumeration."""
    owner = _network()
    owner.add_module("absent", None)
    owner[1] = nn.Linear(5, 4)
    owner.register_parameter("shared", owner[0].weight)
    owner.register_buffer("fresh", torch.zeros(11))
    modules, parameters, buffers = _members(owner)
    assert len(modules) == len(tuple(owner.modules()))
    assert all(map(is_, modules, tuple(owner.modules())))
    assert len(parameters) == len(tuple(owner.parameters()))
    assert all(map(is_, parameters, tuple(owner.parameters())))
    assert len(buffers) == len(tuple(owner.buffers()))
    assert all(map(is_, buffers, tuple(owner.buffers())))


def test_live_gradient_copy_uses_recorded_parameters_without_traversal() -> None:
    """A captured parameter's live gradient is detached from its static replay storage once."""
    network = _network()
    holder = _NetworkCapture()
    holder._network = network
    holder._recorded_parameters = tuple(network.parameters())
    parameter = network[0].weight
    parameter.grad = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
    before = parameter.grad
    with patch.object(network, "parameters", side_effect=AssertionError("traversed during backward")):
        holder._copy_out_live_gradients()
    assert parameter.grad is not before
    assert torch.equal(parameter.grad.view(torch.int32), before.view(torch.int32))
    assert network[1].weight.grad is None
    holder.clear()
    assert holder._recorded_parameters == ()


def test_dispatch_fast_negative_preserves_original_warning_priority() -> None:
    """A fast absence test must retain name-first diagnostics when different modules drift."""
    first, second = nn.Linear(3, 5), nn.Linear(5, 2)
    first._keypoints = lambda: None
    second.forward = lambda inputs: inputs
    assert _module_breach((first, second)) == "overrides forward on its instance"
    assert _module_breach((first, second), exclude=second) == "overrides _keypoints on its instance"


@pytest.mark.parametrize("method", ["parameters", "buffers", "named_parameters", "named_buffers", "modules"])
def test_owner_enumeration_override_keeps_original_calls(method: str) -> None:
    """An instance traversal override is observable through the conservative enumeration route."""
    network = _network()
    original = getattr(network, method)
    with patch.object(network, method, wraps=original) as overridden:
        _members(network)
    assert overridden.called


def test_scalar_binding_removal_and_key_order_are_not_reused() -> None:
    """Reordering two scalar bindings still reaches the original ordered signature."""
    network = _network()
    network.first = 3
    network.second = 7
    before = _network_signature(network)
    metadata = _ModuleMetadata(_members(network)[0], before[0])
    value = vars(network).pop("first")
    network.first = value
    after = _network_signature(network, _members(network), metadata)
    assert after != before
    assert after == _network_signature(network)
    del network.second
    assert _network_signature(network, _members(network), metadata) == _network_signature(network)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_stem_owner_reuses_metadata_then_retires_on_mode_change() -> None:
    """An actual recording uses the certificate and releases it on a changed module mode."""
    stage = nn.Linear(3, 5, device="cuda")
    owner = _NetworkCapture()
    inputs = torch.ones(2, 3, device="cuda", requires_grad=True)
    owner.stem(stage, inputs).sum().backward()
    child = owner._stems[id(stage)]
    assert child._metadata is not None and child._network is not None
    assert len(child._recorded_parameters) == 2
    assert all(map(is_, child._recorded_parameters, tuple(stage.parameters())))
    with patch("lucid_yolo.ptl.network_capture._module_signature", side_effect=AssertionError("rescanned")):
        owner.stem(stage, inputs).sum().backward()
        owner.stem(stage, inputs).sum().backward()
    for parameter in stage.parameters():
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, torch.full_like(parameter, 6.0))
    stage.eval()
    with pytest.warns(UserWarning, match="a module mode"):
        owner.stem(stage, inputs).sum().backward()
    assert child._disabled and child._metadata is None and child._recorded_parameters == ()


class _DescriptorModeLinear(nn.Linear):
    """Expose the public mode through mutable descriptor state rather than an instance scalar."""

    @property
    def training(self) -> bool:
        """Read the public mode without adding a scalar binding."""
        return bool(self.__dict__["_mode"][0])

    @training.setter
    def training(self, mode: bool) -> None:
        """Update the descriptor's state through the normal Module.train call."""
        self.__dict__.setdefault("_mode", [True])[0] = mode

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Make a mode change numerically observable with an independent integer-valued oracle."""
        outputs = super().forward(inputs)
        return outputs if self.training else outputs * 2


def test_descriptor_training_mode_is_read_independently_of_attribute_bindings() -> None:
    """Module.eval can change the explicit training field without rebinding an instance attribute."""
    network = nn.Sequential(_DescriptorModeLinear(3, 5))
    before = _network_signature(network)
    metadata = _ModuleMetadata(_members(network)[0], before[0])
    network[0].eval()
    actual = _network_signature(network, _members(network), metadata)
    assert actual[0][1][2] is False
    assert actual == _network_signature(network) and actual != before


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_stem_retires_when_eval_changes_a_training_descriptor() -> None:
    """The actual recorded owner must stop replaying its training branch after selective eval."""
    layer = _DescriptorModeLinear(3, 5, device="cuda")
    with torch.no_grad():
        layer.weight.fill_(1)
        layer.bias.zero_()
    inputs = torch.ones(2, 3, device="cuda", requires_grad=True)
    owner = _NetworkCapture()
    owner.stem(layer, inputs).sum().backward()
    child = owner._stems[id(layer)]
    assert child._network is not None and child._metadata is not None
    layer.eval()
    with pytest.warns(UserWarning, match="a module mode"):
        actual = owner.stem(layer, inputs)
    assert child._disabled and child._metadata is None
    assert torch.equal(actual, torch.full((2, 5), 6.0, device="cuda"))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_scalar_drift_retires_against_original_recording_stamp() -> None:
    """Caching a post-record binding must not hide the scalar drift the original guard observes."""

    class CountingLinear(nn.Linear):
        def __init__(self) -> None:
            super().__init__(3, 5, device="cuda")
            self.calls = 0

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Expose every Python forward call through a scalar and an exact integer-valued output."""
            self.calls += 1
            return super().forward(inputs) * self.calls

    layer = CountingLinear()
    with torch.no_grad():
        layer.weight.fill_(1)
        layer.bias.zero_()
    inputs = torch.ones(2, 3, device="cuda", requires_grad=True)
    owner = _NetworkCapture()
    owner.stem(layer, inputs).sum().backward()
    child = owner._stems[id(layer)]
    assert child._network is not None and layer.calls > 0
    calls = layer.calls
    with pytest.warns(UserWarning, match="scalar module attribute"):
        actual = owner.stem(layer, inputs)
    assert child._disabled and child._metadata is None
    assert layer.calls == calls + 1
    assert torch.equal(actual, torch.full((2, 5), 3.0 * (calls + 1), device="cuda"))
