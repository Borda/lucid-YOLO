# SPDX-License-Identifier: Apache-2.0
"""Check replay ownership, eager compatibility and the unchanged CUDA arithmetic."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor

from lucid_yolo.optim.musgd import MuSGD
from lucid_yolo.optim.newton_schulz import _orthogonalize_batched, orthogonalize
from lucid_yolo.optim.replay import _ACTIVE, _OrthogonalizationReplay, _replay_orthogonalization
from lucid_yolo.ptl.module import DetectionLitModule


def assert_bytes(left: Tensor, right: Tensor) -> None:
    """Compare storage values without hiding NaNs or signed zero.

    Examples:
        >>> assert_bytes(torch.tensor([1.0]), torch.tensor([1.0]))
    """
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert torch.equal(
        left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8)
    )


class TestEagerReplay:
    def test_cpu_keeps_keywords_values_and_gradients(self) -> None:
        value = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
        expected = orthogonalize(matrix=value, steps=2, eps=1e-6)
        owner = _OrthogonalizationReplay(2)
        with owner.activate():
            actual = orthogonalize(matrix=value, steps=2, eps=1e-6)
        assert_bytes(actual, expected)
        assert_bytes(torch.autograd.grad(actual.sum(), value)[0], torch.autograd.grad(expected.sum(), value)[0])
        assert not owner.entries

    def test_failed_scope_restores_previous_owner(self) -> None:
        outer, inner = _OrthogonalizationReplay(1), _OrthogonalizationReplay(1)
        with outer.activate():
            with pytest.raises(ValueError, match="needs steps >= 1"), inner.activate():
                orthogonalize(torch.eye(2), steps=0)
            assert _ACTIVE.get() is outer
            assert_bytes(orthogonalize(torch.eye(2)), orthogonalize(matrix=torch.eye(2)))
        assert _ACTIVE.get() is None
        assert not outer.entries
        assert not inner.entries

    def test_module_forwards_opt_in_and_old_checkpoint_remains_loadable(self) -> None:
        module = DetectionLitModule(0.5, 0.25, 1024, 4, capture_orthogonalization=True)
        optimizer = module.configure_optimizers()
        assert isinstance(optimizer, MuSGD)
        assert optimizer.param_groups[0]["capture_orthogonalization"]
        legacy = copy.deepcopy(optimizer.state_dict())
        for group in legacy["param_groups"]:
            group.pop("capture_orthogonalization")
        optimizer.load_state_dict(legacy)
        assert optimizer.param_groups[0]["capture_orthogonalization"]
        assert optimizer._orthogonalization_replay is None

    def test_cpu_optimizer_runs_with_flag(self) -> None:
        reference = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12)
        actual = torch.nn.Parameter(reference.detach().clone())
        baseline = MuSGD([reference], lr=0.01)
        candidate = MuSGD([actual], lr=0.01, capture_orthogonalization=True)
        for step in range(3):
            reference.grad = torch.full_like(reference, (step + 1) / 10)
            actual.grad = reference.grad.clone()
            baseline.step()
            candidate.step()
        assert_bytes(reference, actual)
        assert candidate._orthogonalization_replay is not None
        assert not candidate._orthogonalization_replay.entries


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA replay needs a CUDA device")
class TestCUDAReplay:
    @pytest.mark.parametrize("shape", [(16, 32), (32, 16), (16, 16), (3, 16, 32), (2, 32, 16)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
    def test_new_values_keep_outputs_and_inputs_independent(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        function = orthogonalize if len(shape) == 2 else _orthogonalize_batched
        value = torch.randn(shape, device="cuda", dtype=dtype)
        original = value.clone()
        expected = function(value, steps=3)
        owner = _OrthogonalizationReplay(1)
        with torch.no_grad(), owner.activate():
            first = function(value, steps=3)
            second_input = value * 0.37
            second = function(second_input, steps=3)
        assert_bytes(first, expected)
        assert_bytes(second, function(second_input, steps=3))
        assert_bytes(value, original)
        assert len(owner.entries) == 1
        assert first.data_ptr() != second.data_ptr()
        assert first.data_ptr() != next(iter(owner.entries.values())).result.data_ptr()

    def test_replay_removes_python_work_with_bounded_capacity(self) -> None:
        calls = []

        @_replay_orthogonalization
        def counted(value: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
            calls.append(steps)
            return value * steps + eps

        owner = _OrthogonalizationReplay(1)
        value = torch.arange(8, device="cuda", dtype=torch.float32)
        with torch.no_grad(), owner.activate():
            first = counted(value)
            capture_calls = len(calls)
            actual = counted(value + 1)
            assert len(calls) == capture_calls
            assert_bytes(actual, (value + 1) * 5 + 1e-7)
            assert_bytes(first, value * 5 + 1e-7)
            changed = counted(value, steps=2)
            assert len(calls) == capture_calls + 1  # new signature uses eager at capacity
            assert_bytes(changed, value * 2 + 1e-7)
        assert len(owner.entries) == 1
        owner.clear()
        assert not owner.entries

    def test_optimizer_resume_and_skipped_gradient_keep_trajectory(self) -> None:
        torch.manual_seed(19)
        original = [torch.nn.Parameter(torch.randn(12, 8, device="cuda")) for _ in range(3)]
        actual = [torch.nn.Parameter(p.detach().clone()) for p in original]
        baseline = MuSGD(original, lr=0.013, ns_steps=3)
        candidate = MuSGD(actual, lr=0.013, ns_steps=3, capture_orthogonalization=True)
        for step in range(8):
            for index, (left, right) in enumerate(zip(original, actual, strict=True)):
                grad = torch.full_like(left, (step + index + 1) / 17)
                skipped = (step == 3 and index == 1) or (step == 4 and index > 0)
                left.grad = None if skipped else grad
                right.grad = None if left.grad is None else grad.clone()
            for optimizer in (baseline, candidate):
                optimizer.param_groups[0]["lr"] = 0.013 / (step + 1)
                optimizer.step()
            for left, right in zip(original, actual, strict=True):
                assert_bytes(left, right)
                assert_bytes(baseline.state[left]["momentum_buffer"], candidate.state[right]["momentum_buffer"])
            if step == 4:
                state = copy.deepcopy(candidate.state_dict())
                candidate.load_state_dict(state)
                assert candidate._orthogonalization_replay is not None
                assert not candidate._orthogonalization_replay.entries

    def test_autograd_and_autocast_keep_eager_route(self) -> None:
        owner = _OrthogonalizationReplay(2)
        value = torch.randn(8, 12, device="cuda", requires_grad=True)
        with owner.activate():
            actual = orthogonalize(value)
            actual.sum().backward()
        assert value.grad is not None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), owner.activate():
            actual = orthogonalize(value.detach())
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = orthogonalize(value.detach())
        assert_bytes(actual, expected)
        assert not owner.entries

    def test_strided_zero_inputs_and_precision_changes(self) -> None:
        owner = _OrthogonalizationReplay(4)
        value = torch.randn(3, 16, 32, device="cuda")[:, :, ::2]
        previous = torch.get_float32_matmul_precision()
        try:
            for precision in ("highest", "high"):
                torch.set_float32_matmul_precision(precision)
                with torch.no_grad():
                    expected = _orthogonalize_batched(value, steps=2)
                    with owner.activate():
                        actual = _orthogonalize_batched(value, steps=2)
                        zero = _orthogonalize_batched(torch.zeros_like(value), steps=2)
                assert_bytes(actual, expected)
                assert_bytes(zero, torch.zeros_like(zero))
            assert len(owner.entries) == 2  # strided inputs keep their original reduction layout
        finally:
            torch.set_float32_matmul_precision(previous)

    def test_failed_capture_publishes_no_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        owner = _OrthogonalizationReplay(1)
        value = torch.randn(8, 16, device="cuda")

        def fail(*args: object, **kwargs: object) -> None:
            raise RuntimeError("recording failed")

        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "graph", fail)
            with torch.no_grad(), owner.activate(), pytest.raises(RuntimeError, match="recording failed"):
                orthogonalize(value)
        assert not owner.entries
        with torch.no_grad(), owner.activate():
            actual = orthogonalize(value)
        assert_bytes(actual, orthogonalize(value))

    def test_cached_integer_steps_preserve_float_rejection(self) -> None:
        owner = _OrthogonalizationReplay(1)
        value = torch.randn(8, 16, device="cuda")
        with torch.no_grad(), owner.activate():
            orthogonalize(value, steps=5)
            with pytest.raises(TypeError, match="integer"):
                orthogonalize(value, steps=5.0)  # type: ignore[arg-type]
        assert len(owner.entries) == 1

    def test_mutable_epsilon_keeps_eager_route(self) -> None:
        owner = _OrthogonalizationReplay(1)
        value = torch.randn(8, 16, device="cuda")
        epsilon = torch.tensor(0.5)
        with torch.no_grad(), owner.activate():
            first = orthogonalize(value, eps=epsilon)  # type: ignore[arg-type]
            epsilon.fill_(0.1)
            second = orthogonalize(value, eps=epsilon)  # type: ignore[arg-type]
        assert_bytes(first, orthogonalize(value, eps=0.5))
        assert_bytes(second, orthogonalize(value, eps=0.1))
        assert not owner.entries

    def test_inference_mode_never_seeds_mutable_replay_storage(self) -> None:
        owner = _OrthogonalizationReplay(1)
        value = torch.randn(8, 16, device="cuda")
        with torch.inference_mode(), owner.activate():
            first = orthogonalize(value)
        assert first.is_inference()
        assert not owner.entries
        with torch.no_grad(), owner.activate():
            second = orthogonalize(value)
        assert not second.is_inference()
        with torch.inference_mode(), owner.activate():
            third = orthogonalize(value * 0.37)
        assert third.is_inference()
        assert_bytes(first, second)
        assert_bytes(third, orthogonalize(value * 0.37))
        assert len(owner.entries) == 1

    def test_enclosing_cuda_graph_keeps_original_operations(self) -> None:
        owner = _OrthogonalizationReplay(1)
        value = torch.randn(8, 16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(stream):
            expected = orthogonalize(value)
        torch.cuda.current_stream().wait_stream(stream)
        outer = torch.cuda.CUDAGraph()
        with torch.no_grad(), owner.activate(), torch.cuda.graph(outer, stream=stream):
            actual = orthogonalize(value)
        outer.replay()
        assert_bytes(actual, expected)
        assert not owner.entries

    def test_contiguous_offset_input_keeps_eager_route(self) -> None:
        owner = _OrthogonalizationReplay(1)
        value = torch.randn(9, 16, device="cuda")[1:]
        assert value.is_contiguous() and value.storage_offset() != 0
        with torch.no_grad():
            expected = orthogonalize(value)
            with owner.activate():
                actual = orthogonalize(value)
        assert_bytes(actual, expected)
        assert not owner.entries
