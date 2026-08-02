# SPDX-License-Identifier: Apache-2.0
"""Tests for the MuSGD hybrid Muon/SGD optimizer (WP-032).

Covers the parameter-type split (the DoD test): vector parameters take a pure
SGD-momentum step while matrix parameters take the orthogonalized Muon update.
Also covers weight-decay routing (matrix params only), conv-kernel flattening,
shape/finiteness preservation, state-dict round-trip, and skipping grad-less
parameters.
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterator

import pytest
import torch

from open_yolos.optim import MuSGD, orthogonalize


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed torch before every test so random parameters and grads are reproducible."""
    torch.manual_seed(0)
    yield


def _param_with_grad(*shape: int) -> torch.nn.Parameter:
    """Return a parameter of ``shape`` with a random gradient already attached."""
    param = torch.nn.Parameter(torch.randn(*shape))
    param.grad = torch.randn(*shape)
    return param


def _expected_muon_update(nesterov_grad: torch.Tensor, ns_steps: int) -> torch.Tensor:
    """Recompute the Muon branch by hand: scaled orthogonalization of the 2D view."""
    matrix = nesterov_grad.reshape(nesterov_grad.shape[0], -1)
    rows, cols = matrix.shape
    orthogonal = orthogonalize(matrix, steps=ns_steps)
    return (0.2 * math.sqrt(max(rows, cols)) * orthogonal).reshape(nesterov_grad.shape)


def test_param_split() -> None:
    """A 1D param takes a pure SGD-momentum step; a 2D param takes the Muon update.

    With ``w_sgd=0`` and ``weight_decay=0`` the matrix branch is exactly the
    orthogonalized-and-scaled Muon term, and the vector branch is exactly the
    Nesterov SGD step ``-lr*(1+mu)*g`` -- isolating the split cleanly.
    """
    lr, momentum, w_muon, ns_steps = 0.1, 0.9, 0.7, 5
    vector = _param_with_grad(8)
    matrix = _param_with_grad(6, 4)
    vector0, matrix0 = vector.detach().clone(), matrix.detach().clone()
    vector_grad, matrix_grad = vector.grad.clone(), matrix.grad.clone()
    opt = MuSGD(
        [vector, matrix], lr=lr, momentum=momentum, weight_decay=0.0, w_muon=w_muon, w_sgd=0.0, ns_steps=ns_steps
    )

    opt.step()

    # First step: buffer = g, Nesterov-adjusted grad = g + mu*g = (1+mu)*g.
    vector_expected = vector0 - lr * (1.0 + momentum) * vector_grad
    matrix_nesterov = (1.0 + momentum) * matrix_grad
    matrix_expected = matrix0 - lr * w_muon * _expected_muon_update(matrix_nesterov, ns_steps)
    torch.testing.assert_close(vector.detach(), vector_expected)
    torch.testing.assert_close(matrix.detach(), matrix_expected)


def test_step_shapes() -> None:
    """A step over a mixed vector/matrix/conv parameter set preserves shapes and finiteness."""
    params = [_param_with_grad(8), _param_with_grad(6, 4), _param_with_grad(5, 3, 3, 3)]
    shapes = [tuple(p.shape) for p in params]
    opt = MuSGD(params, lr=0.05)

    opt.step()

    assert [tuple(p.shape) for p in params] == shapes
    assert all(torch.isfinite(p).all() for p in params)
    assert all(not torch.equal(p.grad, torch.zeros_like(p)) for p in params)  # grads left intact


def test_weight_decay_applies_to_matrix_params_only() -> None:
    """Decoupled weight decay shifts a matrix param by ``-lr*wd*w`` and never touches a vector param."""
    lr, weight_decay = 0.1, 0.2
    vector_a, vector_b = _param_with_grad(8), None
    matrix_a, matrix_b = _param_with_grad(6, 4), None
    vector_b = torch.nn.Parameter(vector_a.detach().clone())
    vector_b.grad = vector_a.grad.clone()
    matrix_b = torch.nn.Parameter(matrix_a.detach().clone())
    matrix_b.grad = matrix_a.grad.clone()
    matrix0 = matrix_a.detach().clone()
    opt_no_wd = MuSGD([vector_a, matrix_a], lr=lr, weight_decay=0.0)
    opt_wd = MuSGD([vector_b, matrix_b], lr=lr, weight_decay=weight_decay)

    opt_no_wd.step()
    opt_wd.step()

    torch.testing.assert_close(vector_a.detach(), vector_b.detach())  # vector identical: wd excluded
    matrix_delta = matrix_b.detach() - matrix_a.detach()
    torch.testing.assert_close(matrix_delta, -lr * weight_decay * matrix0)  # only the wd term differs


def test_conv_kernel_flattens_and_reshapes() -> None:
    """A 4D conv kernel is orthogonalized via its ``(out, in*kh*kw)`` view and reshaped back."""
    lr, momentum, w_muon, ns_steps = 0.1, 0.9, 0.6, 5
    kernel = _param_with_grad(5, 3, 3, 3)
    kernel0, kernel_grad = kernel.detach().clone(), kernel.grad.clone()
    opt = MuSGD([kernel], lr=lr, momentum=momentum, weight_decay=0.0, w_muon=w_muon, w_sgd=0.0, ns_steps=ns_steps)

    opt.step()

    expected = kernel0 - lr * w_muon * _expected_muon_update((1.0 + momentum) * kernel_grad, ns_steps)
    assert kernel.shape == (5, 3, 3, 3)
    torch.testing.assert_close(kernel.detach(), expected)


def test_state_dict_roundtrip() -> None:
    """A serialized checkpoint reproduces the momentum buffer: the next step matches.

    Uses a real ``torch.save``/``torch.load`` round-trip (the actual checkpoint
    use case); this also decouples the resumed optimizer's buffer from the
    original's in-place updates.
    """
    original = [_param_with_grad(8), _param_with_grad(6, 4)]
    opt_a = MuSGD(original, lr=0.1)
    opt_a.step()
    checkpoint = io.BytesIO()
    torch.save(opt_a.state_dict(), checkpoint)
    resumed = [torch.nn.Parameter(p.detach().clone()) for p in original]
    opt_b = MuSGD(resumed, lr=0.1)
    checkpoint.seek(0)
    opt_b.load_state_dict(torch.load(checkpoint, weights_only=True))

    next_grads = [torch.randn_like(p) for p in original]
    for param, grad in zip(original, next_grads, strict=True):
        param.grad = grad.clone()
    for param, grad in zip(resumed, next_grads, strict=True):
        param.grad = grad.clone()
    opt_a.step()
    opt_b.step()

    for param_a, param_b in zip(original, resumed, strict=True):
        torch.testing.assert_close(param_a.detach(), param_b.detach())


def test_zero_grad_param_skipped() -> None:
    """A parameter with no gradient is left unchanged and accrues no optimizer state."""
    active = _param_with_grad(6, 4)
    idle = torch.nn.Parameter(torch.randn(8))  # grad stays None
    idle0 = idle.detach().clone()
    opt = MuSGD([active, idle], lr=0.1)

    opt.step()

    torch.testing.assert_close(idle.detach(), idle0)
    assert idle not in opt.state


class TestConstructorValidation:
    """Out-of-range hyperparameters are rejected at construction."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param({"lr": -0.1}, "lr must be non-negative", id="negative-lr"),
            pytest.param({"lr": 0.1, "momentum": 1.0}, r"momentum must be in \[0, 1\)", id="momentum-too-high"),
            pytest.param({"lr": 0.1, "momentum": -0.1}, r"momentum must be in \[0, 1\)", id="negative-momentum"),
            pytest.param({"lr": 0.1, "weight_decay": -1.0}, "weight_decay must be non-negative", id="negative-wd"),
            pytest.param({"lr": 0.1, "ns_steps": 0}, "ns_steps must be positive", id="zero-ns-steps"),
        ],
    )
    def test_invalid_hyperparameters_raise(self, kwargs: dict[str, float], match: str) -> None:
        """Each out-of-range hyperparameter raises ValueError with a descriptive message."""
        param = torch.nn.Parameter(torch.randn(4, 4))

        with pytest.raises(ValueError, match=match):
            MuSGD([param], **kwargs)
