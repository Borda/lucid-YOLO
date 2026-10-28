# SPDX-License-Identifier: Apache-2.0
"""Shared density batching retains separate branch selections, means and gradients."""

from __future__ import annotations

import copy
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch
from markers import requires_cuda
from torch import nn

from lucid_yolo.losses.rle_loss import RLELoss


def _inputs(count: int, dtype: torch.dtype = torch.float64, device: str = "cpu") -> tuple[torch.Tensor, ...]:
    """Build varied labelled subsets, with an unlabelled non-finite annotation.

    Examples:
        >>> _inputs(2)[0].shape
        torch.Size([2, 3, 2])
    """
    generator = torch.Generator(device=device).manual_seed(31 + count)
    predicted = torch.rand(count, 3, 2, generator=generator, dtype=dtype, device=device).requires_grad_()
    sigma = torch.randn(count, 3, 2, generator=generator, dtype=dtype, device=device).requires_grad_()
    annotated = torch.rand(count, 3, 2, generator=generator, dtype=dtype, device=device)
    visible = torch.randint(0, 3, (count, 3), generator=generator, device=device)
    if count:
        visible[0, 0] = 0
        annotated[0, 0] = float("nan")
    return predicted, sigma, annotated, visible


@pytest.mark.parametrize("counts", [(0, 0), (0, 2), (3, 0), (1, 4), (5, 2)])
def test_pair_keeps_separate_visible_means_and_gradients(counts: tuple[int, int]) -> None:
    """Asymmetric labelled counts catch splitting by instances or one combined mean."""
    torch.manual_seed(11)
    native = RLELoss().double()
    paired = copy.deepcopy(native)
    left, right = (_inputs(n) for n in counts)
    actual_inputs = [tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in row) for row in (left, right)]
    expected = native(*left), native(*right)
    actual = paired._forward_pair(*actual_inputs)
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-12)
    (0.8 * expected[0] + 0.2 * expected[1]).backward()
    (0.8 * actual[0] + 0.2 * actual[1]).backward()
    for x, y in zip(native.parameters(), paired.parameters(), strict=True):
        assert x.grad is not None and y.grad is not None
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-10, atol=1e-11)
    for before, after in zip((left, right), actual_inputs, strict=True):
        for x, y in zip(before[:2], after[:2], strict=True):
            assert x.grad is not None and y.grad is not None
            torch.testing.assert_close(x.grad, y.grad, rtol=1e-10, atol=1e-11)


def test_pair_halves_native_linear_calls() -> None:
    """The batched scorer performs eighteen linear calls instead of thirty-six."""
    loss = RLELoss().double()
    left, right = _inputs(3), _inputs(1)
    counts = []
    for batched in (False, True):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            if batched:
                loss._forward_pair(left, right)
            else:
                loss(*left), loss(*right)
        counts.append(sum(event.count for event in profiler.key_averages() if event.key == "aten::linear"))
    assert counts == [36, 18]


class _AlternateLinear(nn.Linear):
    """A subclass whose forward the certificate cannot vouch for."""


class _AlternateParameter(nn.Parameter):
    """A parameter subclass whose gradient behaviour the certificate cannot vouch for."""


def _first_linear(loss: RLELoss) -> nn.Linear:
    """Return the first conditioner layer, a built-in `nn.Linear` in the shipped loss.

    Examples:
        >>> type(_first_linear(RLELoss())) is nn.Linear
        True
    """
    layer = loss.flow.layers[0].conditioner.net[0]
    assert isinstance(layer, nn.Linear)
    return layer


_ALTERATIONS: dict[str, Callable[[RLELoss], object]] = {
    "root": lambda loss: loss.register_forward_hook(lambda *args: None),
    "forward_pre": lambda loss: loss.register_forward_pre_hook(lambda *args: None),
    "backward_pre": lambda loss: loss.register_full_backward_pre_hook(lambda *args: None),
    "linear": lambda loss: _first_linear(loss).register_full_backward_hook(lambda *args: None),
    "parameter": lambda loss: next(loss.parameters()).register_hook(lambda gradient: gradient),
    "post_accumulate": lambda loss: next(loss.parameters()).register_post_accumulate_grad_hook(lambda parameter: None),
    "parameter_subclass": lambda loss: setattr(
        _first_linear(loss), "weight", _AlternateParameter(_first_linear(loss).weight.detach().clone())
    ),
    "instance": lambda loss: setattr(loss.flow, "invert", lambda value: (value, value.sum(-1))),
    "subclass": lambda loss: loss.flow.layers[0].conditioner.net.__setitem__(0, _AlternateLinear(1, 64)),
    "compiled": lambda loss: setattr(loss.flow, "_compiled_call_impl", lambda *args: None),
    "call_impl": lambda loss: setattr(loss, "_call_impl", lambda *args: torch.tensor(123.0)),
}


@pytest.mark.parametrize("location", list(_ALTERATIONS))
def test_pair_certificate_refuses_altered_dispatch(location: str) -> None:
    """A user hook, replacement or compiler keeps the original public loss calls."""
    loss = RLELoss()
    assert loss._can_pair()
    _ALTERATIONS[location](loss)
    assert not loss._can_pair()


@pytest.mark.parametrize(
    "owner,name",
    [pytest.param(nn.Linear, "forward", id="class"), pytest.param(RLELoss, "__call__", id="class_call")],
)
def test_pair_certificate_refuses_replaced_class_method(owner: type[nn.Module], name: str) -> None:
    """Replacing a built-in stage's class method keeps the original public loss calls."""
    loss = RLELoss()
    with patch.object(owner, name, lambda *args: torch.tensor(123.0)):
        assert not loss._can_pair()


@pytest.mark.parametrize(
    "register",
    [
        "register_module_forward_hook",
        "register_module_forward_pre_hook",
        "register_module_full_backward_hook",
        "register_module_full_backward_pre_hook",
    ],
)
def test_pair_certificate_refuses_global_hook(register: str) -> None:
    """Even an observer outside this module tree preserves the two-call boundary.

    Each of the four process-wide hook registries observes every module call, so a
    hook in any of them must keep the loss on its original public calls.
    """
    loss = RLELoss()
    with getattr(nn.modules.module, register)(lambda *args: None):
        assert not loss._can_pair()
    assert loss._can_pair()


def test_pair_mixed_sigma_dtypes_keep_native_log_precision() -> None:
    """Concatenation must not promote one branch's sigmoid/logarithm to another dtype."""
    loss = RLELoss()
    left, right = _inputs(3, torch.float32), _inputs(1, torch.float32)
    left = left[0], left[1].detach().to(torch.bfloat16).requires_grad_(), left[2], left[3]
    expected = loss(*left), loss(*right)
    with patch("torch.cat", side_effect=AssertionError("mixed precision concatenated")):
        actual = loss._forward_pair(left, right)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)


def test_pair_tensor_subclass_keeps_native_calls() -> None:
    """An input override is outside the row-independent built-in tensor certificate."""

    class AlternateTensor(torch.Tensor):
        pass

    loss = RLELoss().double()
    left, right = _inputs(1), _inputs(2)
    left = left[0].as_subclass(AlternateTensor), *left[1:]
    expected = loss(*left), loss(*right)
    with patch("torch.cat", side_effect=AssertionError("tensor subclass concatenated")):
        actual = loss._forward_pair(left, right)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)


_POSITIONS = ("predicted", "sigma", "annotated", "visibility")


@pytest.mark.parametrize("position", _POSITIONS)
def test_pair_dtype_mismatch_at_any_input_keeps_native_calls(position: str) -> None:
    """A branch whose input dtype differs from the other branch's is never concatenated with it.

    Every one of the four inputs is checked, visibility included: its dtype never reaches
    the concatenation, so only the guard itself keeps the branches apart.
    """
    loss = RLELoss().double()
    left, right = _inputs(1), _inputs(2)
    index = _POSITIONS.index(position)
    changed = list(left)
    changed[index] = left[index].detach().to(torch.int32 if position == "visibility" else torch.float32)
    left = tuple(changed)
    expected = loss(*left), loss(*right)
    with patch("torch.cat", side_effect=AssertionError("mixed dtypes concatenated")):
        actual = loss._forward_pair(left, right)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("position", _POSITIONS)
def test_pair_tensor_subclass_at_any_input_keeps_native_calls(position: str, side: str) -> None:
    """A tensor subclass in any of the eight input slots keeps both branches on their own calls."""

    class AlternateTensor(torch.Tensor):
        pass

    loss = RLELoss().double()
    branches = {"left": _inputs(1), "right": _inputs(2)}
    changed = list(branches[side])
    index = _POSITIONS.index(position)
    changed[index] = changed[index].as_subclass(AlternateTensor)
    branches[side] = tuple(changed)
    expected = loss(*branches["left"]), loss(*branches["right"])
    with patch("torch.cat", side_effect=AssertionError("tensor subclass concatenated")):
        actual = loss._forward_pair(branches["left"], branches["right"])
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)


@requires_cuda
def test_pair_device_mismatch_keeps_native_calls() -> None:
    """Branches on different devices are never concatenated; each is scored by its own call.

    The CPU branch then meets a CUDA-resident loss and the native call refuses it, which
    is the observable proof that the pair stepped aside instead of moving or merging data.
    """
    loss = RLELoss().double().to("cuda")
    left, right = _inputs(1, device="cuda"), _inputs(2)
    with patch("torch.cat", side_effect=AssertionError("cross-device concatenated")), pytest.raises(RuntimeError):
        loss._forward_pair(left, right)
