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

from lucid_yolo.optim import MuSGD, orthogonalize


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed torch before every test so random parameters and grads are reproducible."""
    torch.manual_seed(0)
    yield


def _param_with_grad(*shape: int) -> torch.nn.Parameter:
    """Return a parameter of ``shape`` with a random gradient already attached.

    Examples:
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> param = _param_with_grad(2, 3)
        >>> param.shape, param.grad.shape
        (torch.Size([2, 3]), torch.Size([2, 3]))
        >>> isinstance(param, torch.nn.Parameter)
        True
    """
    param = torch.nn.Parameter(torch.randn(*shape))
    param.grad = torch.randn(*shape)
    return param


def _expected_muon_update(nesterov_grad: torch.Tensor, ns_steps: int) -> torch.Tensor:
    """Recompute the Muon branch by hand: scaled orthogonalization of the 2D view.

    "By hand" is bounded, and the boundary is deliberate. The reshape to a 2D view and
    the ``0.2 * sqrt(max(rows, cols))`` scale are re-derived here, so the assertions
    below genuinely check what ``_matrix_update`` does with them; the Newton-Schulz
    iteration itself is delegated to :func:`orthogonalize`, which makes that one factor
    a comparison of the code against itself. It is left that way on purpose. Re-deriving
    a five-step quintic iteration in the test would duplicate the algorithm rather than
    check it, and a hand-copy that drifts from the production one is a false failure
    waiting to happen; meanwhile ``orthogonalize`` carries thirteen tests of its own,
    including orthogonality against known singular vectors, so it is verified elsewhere
    rather than assumed here. The parts this helper could get wrong are the parts it
    computes.

    Examples:
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> update = _expected_muon_update(torch.randn(4, 4), ns_steps=5)
        >>> update.shape
        torch.Size([4, 4])
        >>> bool(torch.isfinite(update).all())
        True
    """
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


class TestExactShapeBatching:
    """Equal-shape matrices share one Newton--Schulz call without sharing state.

    Shape, dtype and device are the complete batching key: every operation inside
    :meth:`MuSGD._parameter_update` is elementwise or acts on the final two matrix
    axes, so a leading batch axis cannot mix parameters. These tests compare the
    result with separate per-parameter calls and count the production call
    boundary, since values alone cannot prove that the separate launches are gone.
    """

    def _count_orthogonalize_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.Size, torch.dtype]]:
        """Install a recording wrapper around the real orthogonalization."""
        calls: list[tuple[torch.Size, torch.dtype]] = []
        real = orthogonalize

        def recording(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
            calls.append((matrix.shape, matrix.dtype))
            return real(matrix, steps=steps, eps=eps)

        monkeypatch.setattr("lucid_yolo.optim.musgd.orthogonalize", recording)
        return calls

    def test_equal_shapes_share_one_call_and_match_individual_updates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three equal conv kernels become one call and retain three independent answers."""
        lr, momentum, w_muon, w_sgd, weight_decay, ns_steps = 0.03, 0.8, 0.6, 0.4, 0.02, 5
        kernels = [_param_with_grad(5, 3, 3, 3) for _ in range(3)]
        starts = [param.detach().clone() for param in kernels]
        grads = [param.grad.clone() for param in kernels]
        calls = self._count_orthogonalize_calls(monkeypatch)
        opt = MuSGD(
            kernels,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            w_muon=w_muon,
            w_sgd=w_sgd,
            ns_steps=ns_steps,
        )

        opt.step()

        assert calls == [(torch.Size([5, 27]), torch.float32)]
        for param, start, grad in zip(kernels, starts, grads, strict=True):
            nesterov = (1.0 + momentum) * grad
            expected_update = w_muon * _expected_muon_update(nesterov, ns_steps)
            expected_update.add_(nesterov, alpha=w_sgd).add_(start, alpha=weight_decay)
            torch.testing.assert_close(param.detach(), start - lr * expected_update)

    def test_shape_and_dtype_boundaries_form_separate_batches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Equal dimensions batch, an opposite orientation and float64 do not, and every value still matches unbatched.

        Values are checked against the same `_expected_muon_update` construction the
        equal-shape test uses, so the boundary is proven on outcomes, not only on the
        recorded call sites.
        """
        lr, momentum, w_muon, w_sgd, weight_decay, ns_steps = 0.05, 0.95, 0.5, 0.5, 5e-4, 5
        double_param = torch.nn.Parameter(torch.randn(6, 4, dtype=torch.float64))
        double_param.grad = torch.randn_like(double_param)
        params = [
            _param_with_grad(6, 4),
            _param_with_grad(6, 4),
            _param_with_grad(4, 6),
            double_param,
        ]
        starts = [param.detach().clone() for param in params]
        grads = [param.grad.clone() for param in params]
        calls = self._count_orthogonalize_calls(monkeypatch)
        opt = MuSGD(params, lr=lr)

        opt.step()

        assert calls == [
            (torch.Size([6, 4]), torch.float32),
            (torch.Size([4, 6]), torch.float32),
            (torch.Size([6, 4]), torch.float64),
        ]
        assert [param.dtype for param in params] == [torch.float32, torch.float32, torch.float32, torch.float64]
        for param, start, grad in zip(params, starts, grads, strict=True):
            nesterov = (1.0 + momentum) * grad
            expected_update = w_muon * _expected_muon_update(nesterov, ns_steps)
            expected_update.add_(nesterov, alpha=w_sgd).add_(start, alpha=weight_decay)
            torch.testing.assert_close(param.detach(), start - lr * expected_update)

    def test_parameter_groups_never_share_a_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Equal matrices with different group options keep separate update calls and separate values."""
        lr, momentum, w_sgd, weight_decay, ns_steps = 0.05, 0.95, 0.5, 5e-4, 5
        left = _param_with_grad(6, 4)
        right = _param_with_grad(6, 4)
        starts = [left.detach().clone(), right.detach().clone()]
        grads = [left.grad.clone(), right.grad.clone()]
        calls = self._count_orthogonalize_calls(monkeypatch)
        opt = MuSGD(
            [
                {"params": [left], "w_muon": 0.2},
                {"params": [right], "w_muon": 0.8},
            ],
            lr=lr,
        )

        opt.step()

        assert calls == [
            (torch.Size([6, 4]), torch.float32),
            (torch.Size([6, 4]), torch.float32),
        ]
        for param, start, grad, w_muon in zip([left, right], starts, grads, [0.2, 0.8], strict=True):
            nesterov = (1.0 + momentum) * grad
            expected_update = w_muon * _expected_muon_update(nesterov, ns_steps)
            expected_update.add_(nesterov, alpha=w_sgd).add_(start, alpha=weight_decay)
            torch.testing.assert_close(param.detach(), start - lr * expected_update)

    @pytest.mark.gpu
    def test_device_boundary_forms_a_separate_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Equal shape and dtype on different devices never share a batch."""
        lr, momentum, w_muon, w_sgd, weight_decay, ns_steps = 0.05, 0.95, 0.5, 0.5, 5e-4, 5
        cpu_param = _param_with_grad(6, 4)
        cuda_param = torch.nn.Parameter(torch.randn(6, 4, device="cuda"))
        cuda_param.grad = torch.randn_like(cuda_param)
        params = [cpu_param, cuda_param]
        starts = [param.detach().clone() for param in params]
        grads = [param.grad.clone() for param in params]
        calls = self._count_orthogonalize_calls(monkeypatch)
        opt = MuSGD(params, lr=lr)

        opt.step()

        assert calls == [
            (torch.Size([6, 4]), torch.float32),
            (torch.Size([6, 4]), torch.float32),
        ]
        for param, start, grad in zip(params, starts, grads, strict=True):
            nesterov = (1.0 + momentum) * grad
            expected_update = w_muon * _expected_muon_update(nesterov, ns_steps)
            expected_update.add_(nesterov, alpha=w_sgd).add_(start, alpha=weight_decay)
            torch.testing.assert_close(param.detach(), start - lr * expected_update)


class TestZeroMuonGain:
    """The ``w_muon = 0`` arm skips the Muon branch instead of computing and discarding it.

    A7's gains are independent, so ``w_muon = 0`` is a usable SGD-only arm rather
    than a degenerate configuration. It used to run the full Newton-Schulz
    iteration for every matrix parameter and then multiply the result by zero --
    on the n-scale detector, 127 orthogonalizations per step for nothing.
    """

    def _count_orthogonalize_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """Install a counting stand-in for ``orthogonalize`` and return its one-element tally."""
        tally = [0]
        real = orthogonalize

        def counting(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
            tally[0] += 1
            return real(matrix, steps=steps, eps=eps)

        monkeypatch.setattr("lucid_yolo.optim.musgd.orthogonalize", counting)
        return tally

    @pytest.mark.parametrize(
        ("w_muon", "expected_calls"),
        [
            pytest.param(0.0, 0, id="zero-gain-runs-no-iteration"),
            pytest.param(0.5, 1, id="nonzero-gain-still-iterates"),
        ],
    )
    def test_newton_schulz_runs_only_for_a_nonzero_gain(
        self, monkeypatch: pytest.MonkeyPatch, w_muon: float, expected_calls: int
    ) -> None:
        """One matrix parameter triggers one orthogonalization at ``w_muon > 0`` and none at zero.

        The zero case is the regression: before the short-circuit the iteration
        ran and its result was scaled to nothing, so the call count -- not the
        parameter value -- is what distinguishes the two implementations.
        """
        tally = self._count_orthogonalize_calls(monkeypatch)
        matrix = _param_with_grad(6, 4)
        opt = MuSGD([matrix], lr=0.1, w_muon=w_muon)

        opt.step()

        assert tally[0] == expected_calls

    def test_zero_gain_update_is_the_pure_sgd_step_with_weight_decay(self) -> None:
        """At ``w_muon = 0`` a matrix parameter moves by exactly ``-lr*(w_sgd*nesterov + wd*w)``.

        The skipped branch must be worth exactly zero, which is what makes
        skipping it a performance change and not a numerical one: the surviving
        terms are the SGD half and the decoupled decay, and both are asserted
        against hand-written arithmetic rather than against the old code path.
        """
        lr, momentum, w_sgd, weight_decay = 0.1, 0.9, 0.4, 0.2
        matrix = _param_with_grad(6, 4)
        matrix0, matrix_grad = matrix.detach().clone(), matrix.grad.clone()
        opt = MuSGD([matrix], lr=lr, momentum=momentum, weight_decay=weight_decay, w_muon=0.0, w_sgd=w_sgd, ns_steps=5)

        opt.step()

        nesterov = (1.0 + momentum) * matrix_grad  # first step: buffer = g
        expected = matrix0 - lr * (w_sgd * nesterov + weight_decay * matrix0)
        torch.testing.assert_close(matrix.detach(), expected)

    def test_zero_gain_leaves_a_vector_parameter_on_its_own_path(self) -> None:
        """A 1D parameter takes the same pure Nesterov SGD step whatever ``w_muon`` says.

        The short-circuit sits in the matrix branch, and the rank split above it
        is unchanged: a vector parameter never reached the Muon branch and must
        still move by ``-lr*(1+mu)*g``, with no weight decay (A12).
        """
        lr, momentum = 0.1, 0.9
        vector = _param_with_grad(8)
        vector0, vector_grad = vector.detach().clone(), vector.grad.clone()
        opt = MuSGD([vector], lr=lr, momentum=momentum, weight_decay=0.2, w_muon=0.0, w_sgd=1.0)

        opt.step()

        torch.testing.assert_close(vector.detach(), vector0 - lr * (1.0 + momentum) * vector_grad)


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
            pytest.param({"lr": 0.1, "w_muon": -0.5}, "w_muon must be non-negative", id="negative-w-muon"),
            pytest.param({"lr": 0.1, "w_sgd": -0.5}, "w_sgd must be non-negative", id="negative-w-sgd"),
            pytest.param({"lr": 0.1, "w_muon": math.nan}, "w_muon must be non-negative", id="nan-w-muon"),
            pytest.param({"lr": 0.1, "w_sgd": math.nan}, "w_sgd must be non-negative", id="nan-w-sgd"),
        ],
    )
    def test_invalid_hyperparameters_raise(self, kwargs: dict[str, float], match: str) -> None:
        """Each out-of-range hyperparameter raises ValueError with a descriptive message.

        The two gains joined the four originals at WP-171: they were the only
        constructor arguments unchecked, so a negative gain -- which subtracts its
        branch's update instead of adding it -- and a NaN gain -- which turns every
        matrix parameter it touches into NaN on the first step -- both constructed
        cleanly and failed, if at all, as a loss curve rather than as an error. NaN
        needs the explicit case because ``w_muon < 0.0`` is false for it.
        """
        param = torch.nn.Parameter(torch.randn(4, 4))

        with pytest.raises(ValueError, match=match):
            MuSGD([param], **kwargs)
