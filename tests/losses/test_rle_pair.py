# SPDX-License-Identifier: Apache-2.0
"""Shared density batching retains separate branch selections, means and gradients."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch
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


@pytest.mark.parametrize(
    "location", ["root", "linear", "parameter", "class", "instance", "subclass", "compiled", "call_impl", "class_call"]
)
def test_pair_certificate_refuses_altered_dispatch(location: str) -> None:
    """A user hook, replacement or compiler keeps the original public loss calls."""
    loss = RLELoss()
    assert loss._can_pair()
    if location == "root":
        loss.register_forward_hook(lambda *args: None)
    elif location == "linear":
        loss.flow.layers[0].conditioner.net[0].register_full_backward_hook(lambda *args: None)
    elif location == "parameter":
        next(loss.parameters()).register_hook(lambda gradient: gradient)
    elif location == "class":
        with patch.object(nn.Linear, "forward", lambda *args: None):
            assert not loss._can_pair()
        return
    elif location == "instance":
        loss.flow.invert = lambda value: (value, value.sum(-1))
    elif location == "call_impl":
        loss._call_impl = lambda *args: torch.tensor(123.0)
    elif location == "class_call":
        with patch.object(RLELoss, "__call__", lambda *args: torch.tensor(123.0)):
            assert not loss._can_pair()
        return
    elif location == "compiled":
        loss.flow._compiled_call_impl = lambda *args: None
    else:

        class AlternateLinear(nn.Linear):
            pass

        loss.flow.layers[0].conditioner.net[0] = AlternateLinear(1, 64)
    assert not loss._can_pair()


def test_pair_certificate_refuses_global_hook() -> None:
    """Even an observer outside this module tree preserves the two-call boundary."""
    loss = RLELoss()
    with nn.modules.module.register_module_forward_hook(lambda *args: None):
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
