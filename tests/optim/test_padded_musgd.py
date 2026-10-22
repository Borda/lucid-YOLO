# SPDX-License-Identifier: Apache-2.0
"""Mixed-width MuSGD batches retain each matrix's own normalization and scale."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from markers import requires_cuda

from lucid_yolo.optim import MuSGD, musgd, orthogonalize


class _StorageOpaqueParameter(torch.nn.Parameter):
    """Keep tensor arithmetic available while refusing direct storage inspection."""

    def untyped_storage(self) -> torch.UntypedStorage:
        """Reject a metadata operation ordinary SGD never needs."""
        raise RuntimeError("storage access unavailable")


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=requires_cuda),
    ]
)
def device(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Exercise the production CUDA gate and widen it only for CPU compatibility."""
    value = str(request.param)
    if value == "cpu":
        monkeypatch.setattr(musgd, "_BATCHED_DEVICE_TYPES", frozenset({"cpu"}))
    return value


def _parameters(device: str, dtype: torch.dtype) -> list[torch.nn.Parameter]:
    """Build asymmetric wide, tall, square and convolutional matrices.

    Examples:
        >>> [tuple(p.shape) for p in _parameters("cpu", torch.float32)]
        [(4, 9), (6, 4), (4, 4), (4, 1, 2, 3)]
    """
    generator = torch.Generator(device=device).manual_seed(31)
    return [
        torch.nn.Parameter(torch.randn(shape, generator=generator, device=device, dtype=dtype))
        for shape in [(4, 9), (6, 4), (4, 4), (4, 1, 2, 3)]
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mixed_width_update_matches_scalar_oracle(
    device: str, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One native batch preserves independent norms, transpose, scale, gains and decay."""
    params = _parameters(device, dtype)
    for index, param in enumerate(params):
        param.grad = torch.arange(param.numel(), device=device, dtype=dtype).reshape_as(param) / (index + 7)
    starts = [param.detach().clone() for param in params]
    calls: list[tuple[int, ...]] = []
    native = musgd._orthogonalize_batched

    def record(matrices: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        calls.append(tuple(matrices.shape))
        return native(matrices, steps=steps, eps=eps)

    monkeypatch.setattr(musgd, "_orthogonalize_batched", record)
    optimizer = MuSGD(params, lr=0.03, momentum=0.8, w_muon=0.7, w_sgd=0.2, weight_decay=0.04, ns_steps=3)
    optimizer.step()

    assert calls == [(4, 4, 9)]
    for param, start in zip(params, starts, strict=True):
        assert param.grad is not None
        gradient = param.grad + 0.8 * param.grad
        matrix = gradient.reshape(gradient.shape[0], -1)
        muon = orthogonalize(matrix, steps=3).reshape_as(param) * (0.2 * math.sqrt(max(matrix.shape)))
        update = muon * 0.7 + gradient * 0.2 + start * 0.04
        tolerance = 1e-11 if dtype == torch.float64 else 2e-6
        torch.testing.assert_close(param, start - 0.03 * update, rtol=tolerance * 10, atol=tolerance)
        torch.testing.assert_close(optimizer.state[param]["momentum_buffer"], param.grad, rtol=0, atol=0)


def test_padding_storage_uses_existing_stack_budget(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An extreme width starts another batch instead of padding every small matrix to it."""
    params = [torch.nn.Parameter(torch.ones(shape, device=device)) for shape in [(4, 5), (4, 6), (4, 1000)]]
    for param in params:
        param.grad = torch.ones_like(param)
    calls: list[tuple[int, ...]] = []
    native = musgd._orthogonalize_batched

    def record(matrices: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        calls.append(tuple(matrices.shape))
        return native(matrices, steps=steps, eps=eps)

    monkeypatch.setattr(musgd, "_orthogonalize_batched", record)
    MuSGD(params, lr=0.01).step()
    assert calls == [(2, 4, 6)]
    assert math.prod(calls[0]) <= 2 * sum(param.numel() for param in params[:2])
    assert all(bool(torch.isfinite(param).all()) for param in params)


@pytest.mark.parametrize(("rows", "expected_batches"), [(255, 1), (256, 1), (257, 0)])
def test_padding_stays_inside_measured_row_region(
    device: str, monkeypatch: pytest.MonkeyPatch, rows: int, expected_batches: int
) -> None:
    """The measured boundary includes 256 rows; the next row retains exact shapes."""
    params = [torch.nn.Parameter(torch.ones((rows, rows + extra), device=device)) for extra in (1, 2)]
    for param in params:
        param.grad = torch.ones_like(param)
    calls = []
    native = musgd._orthogonalize_batched

    def record(matrices: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        calls.append(tuple(matrices.shape))
        return native(matrices, steps=steps, eps=eps)

    monkeypatch.setattr(musgd, "_orthogonalize_batched", record)
    MuSGD(params, lr=0.01).step()
    assert len(calls) == expected_batches
    assert all(bool(torch.isfinite(param).all()) for param in params)


def test_mixed_width_resume_preserves_momentum(device: str) -> None:
    """Resuming with different shapes and skipped gradients keeps per-parameter state."""
    params = _parameters(device, torch.float32)
    reference = [param.detach().clone() for param in params]
    buffers = [torch.zeros_like(param) for param in params]
    optimizer = MuSGD(params, lr=0.01, momentum=0.9, weight_decay=0.02)
    for step in range(40):
        for index, (param, expected, buffer) in enumerate(zip(params, reference, buffers, strict=True)):
            param.grad = torch.full_like(param, (index + 1) / (step + 3)) if step % 7 != index else None
            if param.grad is None:
                continue
            buffer.mul_(0.9).add_(param.grad)
            gradient = param.grad.add(buffer, alpha=0.9)
            matrix = gradient.reshape(gradient.shape[0], -1)
            muon = orthogonalize(matrix).reshape_as(param).mul(0.2 * math.sqrt(max(matrix.shape)))
            update = muon.mul(0.5).add(gradient, alpha=0.5).add(expected, alpha=0.02)
            expected.add_(update, alpha=-0.01)
        optimizer.step()
        if step == 17:
            restored = MuSGD(params, lr=0.9)
            restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            optimizer = restored
    for param, expected, buffer in zip(params, reference, buffers, strict=True):
        torch.testing.assert_close(param, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(optimizer.state[param]["momentum_buffer"], buffer, rtol=0, atol=0)


def test_shared_storage_keeps_exact_shape_updates(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Overlapping matrix views retain sequential decay rather than sharing a padded update."""
    storage = torch.arange(24.0, device=device)
    params = [torch.nn.Parameter(storage[:20].view(4, 5)), torch.nn.Parameter(storage.view(4, 6))]
    for param in params:
        param.grad = torch.ones_like(param)
    expected = storage.clone()
    for width in (5, 6):
        view = expected[: 4 * width].view(4, width)
        gradient = torch.full_like(view, 1.95)
        muon = orthogonalize(gradient).mul(0.2 * math.sqrt(width))
        update = muon.mul(0.5).add(gradient, alpha=0.5).add(view, alpha=0.2)
        view.add_(update, alpha=-0.1)
    monkeypatch.setattr(musgd, "_orthogonalize_batched", lambda *args, **kwargs: pytest.fail("shared storage batched"))
    MuSGD(params, lr=0.1, weight_decay=0.2).step()
    torch.testing.assert_close(storage, expected)


def test_dlpack_overlap_keeps_scalar_order(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Views of one buffer under distinct storage objects keep the sequential decay order.

    ``torch.from_dlpack`` wraps the same memory in another storage object with its own base pointer,
    so a base-pointer comparison calls the two parameters independent while their bytes overlap.
    Padding would then reorder their decoupled-decay updates and diverge from the scalar path.
    """
    base = torch.linspace(-1.0, 1.0, 40, device=device)

    def run() -> torch.Tensor:
        flat = base.clone()
        params = [
            torch.nn.Parameter(flat[:24].view(4, 6)),
            torch.nn.Parameter(torch.from_dlpack(flat[2:22]).view(4, 5)),
        ]
        for param in params:
            param.grad = torch.ones_like(param)
        MuSGD(params, lr=0.1, weight_decay=0.2).step()
        return flat

    with monkeypatch.context() as scalar:
        scalar.setattr(musgd, "_BATCHED_DEVICE_TYPES", frozenset())
        expected = run()
    monkeypatch.setattr(
        musgd, "_orthogonalize_batched", lambda *args, **kwargs: pytest.fail("overlapping views batched")
    )
    assert torch.equal(run(), expected)


@pytest.mark.parametrize("shape", [(3,), (4, 6)])
def test_parameter_subclass_keeps_legacy_route(device: str, shape: tuple[int, ...]) -> None:
    """A valid parameter subclass need not expose storage to keep its old update."""
    params = [_StorageOpaqueParameter(torch.ones(shape, device=device)) for _ in range(2)]
    for param in params:
        param.grad = torch.ones_like(param)
    gradient = torch.full(shape, 1.95, device=device)
    update = gradient
    if len(shape) > 1:
        muon = orthogonalize(gradient).mul(0.2 * math.sqrt(max(shape)))
        update = muon.mul(0.5).add(gradient, alpha=0.5).add(torch.ones(shape, device=device), alpha=5e-4)
    expected = torch.ones(shape, device=device).add(update, alpha=-0.1)
    MuSGD(params, lr=0.1).step()
    for param in params:
        torch.testing.assert_close(param, expected)


def test_mixed_subclass_group_keeps_alias_order(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opaque storage can overlap an ordinary parameter, so the whole group keeps its old keys."""
    storage = torch.arange(32.0, device=device).view(4, 8)
    params = [
        torch.nn.Parameter(torch.ones((4, 6), device=device)),
        _StorageOpaqueParameter(storage),
        torch.nn.Parameter(storage[:, :7]),
    ]
    for index, param in enumerate(params):
        param.grad = torch.full_like(param, (index + 1) / 4)
    expected = storage.clone()
    for width, value in ((8, 0.5), (7, 0.75)):
        view = expected[:, :width]
        gradient = torch.full_like(view, value)
        nesterov = gradient.add(gradient, alpha=0.95)
        muon = orthogonalize(nesterov).mul(0.2 * math.sqrt(width))
        update = muon.mul(0.5).add(nesterov, alpha=0.5).add(view, alpha=0.5)
        view.add_(update, alpha=-0.2)
    monkeypatch.setattr(musgd, "_orthogonalize_batched", lambda *args, **kwargs: pytest.fail("opaque group padded"))
    MuSGD(params, lr=0.2, weight_decay=0.5).step()
    torch.testing.assert_close(storage, expected)


@pytest.mark.parametrize(
    ("device_name", "dtype"),
    [
        pytest.param("cpu", torch.float32, id="cpu-float32"),
        pytest.param("cpu", torch.float64, id="cpu-float64"),
        pytest.param("cuda", torch.float16, marks=requires_cuda, id="cuda-float16"),
        pytest.param("cuda", torch.bfloat16, marks=requires_cuda, id="cuda-bfloat16"),
        pytest.param("cuda", torch.float32, marks=requires_cuda, id="cuda-float32"),
        pytest.param("cuda", torch.float64, marks=requires_cuda, id="cuda-float64"),
    ],
)
def test_momentum_batches_without_reassociation(
    device_name: str, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three foreach calls retain the scalar recurrence and its updates bit for bit across resume.

    Zero Muon gain, unit SGD gain and no decay make every update the Nesterov gradient itself, so the
    parameters expose that gradient after a nonzero-lr step instead of the buffers alone. The gate is
    widened to CPU unconditionally: half precision stays on CUDA, where the recurrence was measured.
    """
    monkeypatch.setattr(musgd, "_BATCHED_DEVICE_TYPES", frozenset({"cpu", "cuda"}))
    shapes = [(), (3,), (2, 3), (4,)]
    params = [torch.nn.Parameter(torch.zeros(shape, device=device_name, dtype=dtype)) for shape in shapes]
    settings = {"lr": 0.01, "w_muon": 0.0, "w_sgd": 1.0, "weight_decay": 0.0}
    optimizer = MuSGD(params, **settings)
    buffers = [torch.zeros_like(param) for param in params]
    expected = [torch.zeros_like(param) for param in params]
    calls: list[int] = []
    original = torch._foreach_mul_

    def record(tensors: list[torch.Tensor], scalar: float) -> tuple[torch.Tensor, ...]:
        calls.append(len(tensors))
        return original(tensors, scalar)

    monkeypatch.setattr(torch, "_foreach_mul_", record)
    for step in range(20):
        for index, param in enumerate(params):
            param.grad = torch.full_like(param, (index + 1) / (step + 3))
            buffers[index].mul_(0.95).add_(param.grad)
            expected[index].add_(param.grad.add(buffers[index], alpha=0.95), alpha=-0.01)
        optimizer.step()
        if step == 7:
            saved = copy.deepcopy(optimizer.state_dict())
            optimizer = MuSGD(params, **settings)
            optimizer.load_state_dict(saved)
        for param, buffer, value in zip(params, buffers, expected, strict=True):
            actual = optimizer.state[param]["momentum_buffer"]
            assert torch.equal(actual.reshape(-1).view(torch.uint8), buffer.reshape(-1).view(torch.uint8))
            assert torch.equal(param.detach().reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8))
    assert calls == [4] * 20


def test_default_gate_keeps_scalar_momentum_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU stays outside the foreach gate, so the narrow default never calls a foreach kernel."""
    params = [torch.nn.Parameter(torch.ones(3)) for _ in range(2)]
    for param in params:
        param.grad = torch.ones_like(param)
    monkeypatch.setattr(torch, "_foreach_mul_", lambda *args: pytest.fail("cpu momentum batched"))
    MuSGD(params, lr=0.1).step()


def test_gradient_parameter_alias_preserves_sequential_updates(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A vector update that changes another gradient retains the original ordering."""
    params = [torch.nn.Parameter(torch.tensor([1.0, 2.0], device=device)) for _ in range(2)]
    params[0].grad = params[1].detach()
    params[1].grad = params[0].detach()
    expected = [param.detach().clone() for param in params]
    expected[0].add_(expected[1] * 1.95, alpha=-0.1)
    expected[1].add_(expected[0] * 1.95, alpha=-0.1)
    monkeypatch.setattr(torch, "_foreach_mul_", lambda *args: pytest.fail("aliased gradients batched"))
    MuSGD(params, lr=0.1).step()
    for param, value in zip(params, expected, strict=True):
        torch.testing.assert_close(param, value)


def test_restored_mixed_buffer_dtype_keeps_scalar_route(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wider saved buffer keeps the scalar promotion instead of joining a foreach batch."""
    param = torch.nn.Parameter(torch.ones(3, device=device))
    param.grad = torch.full_like(param, 0.3)
    other = torch.nn.Parameter(torch.ones(3, device=device))
    other.grad = torch.ones_like(other)
    optimizer = MuSGD([param, other], lr=0.1)
    buffer = torch.ones(3, device=device, dtype=torch.float64)
    optimizer.state[param]["momentum_buffer"] = buffer
    expected_buffer = buffer * 0.95 + param.grad
    expected = param.detach() - 0.1 * param.grad.add(expected_buffer, alpha=0.95)
    monkeypatch.setattr(torch, "_foreach_mul_", lambda *args: pytest.fail("mixed buffer dtype batched"))
    optimizer.step()
    torch.testing.assert_close(param, expected.to(param.dtype))
    torch.testing.assert_close(buffer, expected_buffer, rtol=0, atol=0)


def test_shared_momentum_buffer_preserves_sequential_recurrence(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A shared restored buffer advances once per parameter in the original order."""
    params = [torch.nn.Parameter(torch.ones(3, device=device)) for _ in range(2)]
    optimizer = MuSGD(params, lr=0.1)
    shared = torch.ones(3, device=device)
    reference = shared.clone()
    expected = []
    for index, param in enumerate(params):
        param.grad = torch.full_like(param, (index + 1) / 3)
        optimizer.state[param]["momentum_buffer"] = shared
        reference.mul_(0.95).add_(param.grad)
        expected.append(param.detach() - 0.1 * param.grad.add(reference, alpha=0.95))
    monkeypatch.setattr(torch, "_foreach_mul_", lambda *args: pytest.fail("shared momentum batched"))
    optimizer.step()
    for param, value in zip(params, expected, strict=True):
        torch.testing.assert_close(param, value)
    torch.testing.assert_close(shared, reference, rtol=0, atol=0)


@pytest.mark.skipif(not hasattr(torch.Tensor, "grad_dtype"), reason="grad_dtype postdates torch>=2.4")
def test_wider_gradient_keeps_scalar_promotion(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional wider gradient does not inherit the parameter's precision."""
    param = torch.nn.Parameter(torch.ones(3, device=device))
    param.grad_dtype = None
    param.grad = torch.full((3,), 0.3, device=device, dtype=torch.float64)
    buffer = param.grad.float()
    expected = param.detach() - 0.1 * param.grad.add(buffer, alpha=0.95)
    monkeypatch.setattr(torch, "_foreach_mul_", lambda *args: pytest.fail("wider gradient batched"))
    other = torch.nn.Parameter(torch.ones(3, device=device))
    other.grad = torch.ones_like(other)
    MuSGD([param, other], lr=0.1).step()
    torch.testing.assert_close(param, expected.float())
