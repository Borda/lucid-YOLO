# SPDX-License-Identifier: Apache-2.0
"""Check replay ownership, eager compatibility and the unchanged CUDA arithmetic."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

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


@contextmanager
def _other_eps() -> Iterator[dict[str, Any]]:
    """Ask for a different epsilon, one part of the replay key that is an argument.

    Examples:
        >>> with _other_eps() as extra:
        ...     extra
        {'eps': 1e-06}
    """
    yield {"eps": 1e-6}


@contextmanager
def _other_stream() -> Iterator[dict[str, Any]]:
    """Run on a side stream, ordered after and before the caller's stream.

    Examples:
        >>> with _other_stream() as extra:  # needs a CUDA device  # doctest: +SKIP
        ...     extra
        {}
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield {}
    torch.cuda.current_stream().wait_stream(stream)


@contextmanager
def _deterministic_warn_only() -> Iterator[dict[str, Any]]:
    """Enable deterministic algorithms in warn-only mode, restoring both flags afterwards.

    Examples:
        >>> with _deterministic_warn_only() as extra:
        ...     (torch.are_deterministic_algorithms_enabled(), extra)
        (True, {})
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield {}
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@contextmanager
def _other_blas_library() -> Iterator[dict[str, Any]]:
    """Run under a cuBLAS front end different from the current one, restoring the previous choice.

    Examples:
        >>> with _other_blas_library() as extra:  # needs a CUDA build of PyTorch  # doctest: +SKIP
        ...     extra
        {}
    """
    previous = torch.backends.cuda.preferred_blas_library()
    if torch.backends.cuda.preferred_blas_library("cublaslt") == previous:
        torch.backends.cuda.preferred_blas_library("cublas")
    try:
        yield {}
    finally:
        torch.backends.cuda.preferred_blas_library(previous)


class TestEagerReplay:
    def test_default_optimizer_keeps_replay_off(self) -> None:
        """Without the opt-in flag no group asks for replay and no owner is ever created."""
        param = torch.nn.Parameter(torch.ones(3, 4))
        optimizer = MuSGD([param], lr=0.01)
        param.grad = torch.ones_like(param)
        optimizer.step()
        assert optimizer.param_groups[0]["capture_orthogonalization"] is False
        assert optimizer._orthogonalization_replay is None

    @pytest.mark.parametrize(
        ("default_flag", "group_options", "group_flag", "capacity"),
        [
            pytest.param(True, {}, True, 2, id="group-inherits-on"),
            pytest.param(False, {}, False, None, id="group-inherits-off"),
            pytest.param(False, {"capture_orthogonalization": True}, True, 2, id="group-overrides-off"),
        ],
    )
    def test_added_group_follows_its_own_flag(
        self, default_flag: bool, group_options: dict[str, bool], group_flag: bool, capacity: int | None
    ) -> None:
        """A group added later inherits the constructor flag unless it names its own flag.

        The owner is sized from the matrix parameters of every group when the first replaying group steps.
        """
        first = torch.nn.Parameter(torch.ones(3, 4))
        second = torch.nn.Parameter(torch.ones(2, 5))
        optimizer = MuSGD([first], lr=0.01, capture_orthogonalization=default_flag)
        optimizer.add_param_group({"params": [second], **group_options})
        first.grad, second.grad = torch.ones_like(first), torch.ones_like(second)
        optimizer.step()
        replay = optimizer._orthogonalization_replay
        assert optimizer.param_groups[1]["capture_orthogonalization"] is group_flag
        assert (None if replay is None else replay.capacity) == capacity

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

    @pytest.mark.parametrize(
        ("checkpoint_flag", "launch_flag"),
        [
            pytest.param(True, False, id="checkpoint-on-launch-off"),
            pytest.param(False, True, id="checkpoint-off-launch-on"),
        ],
    )
    def test_launch_flag_wins_over_checkpoint(self, checkpoint_flag: bool, launch_flag: bool) -> None:
        """Resuming keeps the flag the new run was launched with, not the one the checkpoint recorded.

        A run resumed with replay switched off must stay eager even when the checkpoint came
        from a replaying run, and a run launched with replay on must replay from an eager checkpoint.
        """
        saved_param = torch.nn.Parameter(torch.randn(3, 4))
        saved = MuSGD([saved_param], lr=0.01, capture_orthogonalization=checkpoint_flag)
        saved_param.grad = torch.ones_like(saved_param)
        saved.step()
        resumed_param = torch.nn.Parameter(torch.randn(3, 4))
        resumed = MuSGD([resumed_param], lr=0.01, capture_orthogonalization=launch_flag)
        resumed.load_state_dict(saved.state_dict())
        resumed_param.grad = torch.ones_like(resumed_param)
        resumed.step()
        assert resumed.param_groups[0]["capture_orthogonalization"] is launch_flag
        assert (resumed._orthogonalization_replay is not None) is launch_flag

    @pytest.mark.parametrize(
        ("args", "kwargs", "steps", "eps"),
        [
            pytest.param((), {}, 5, 1e-7, id="defaults"),
            pytest.param((3,), {}, 3, 1e-7, id="positional-steps"),
            pytest.param((3, 0.5), {}, 3, 0.5, id="positional-steps-and-eps"),
            pytest.param((), {"eps": 0.25}, 5, 0.25, id="keyword-eps"),
            pytest.param((), {"steps": 2, "eps": 0.125}, 2, 0.125, id="keyword-steps-and-eps"),
        ],
    )
    def test_wrapper_hands_arguments_to_owner_in_order(
        self, args: tuple[object, ...], kwargs: dict[str, object], steps: int, eps: float
    ) -> None:
        """The wrapper binds defaults, positional and keyword arguments to the owner's matrix, steps and eps slots.

        A crossed or dropped slot would reach every recording's key and every replayed call, yet nothing
        else runs the wrapper off a CUDA device: the owner returns eagerly for any other tensor.
        """
        seen: list[tuple[Tensor, int, float]] = []

        @_replay_orthogonalization
        def record(matrix: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
            seen.append((matrix, steps, eps))
            return matrix * steps + eps

        value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        with _OrthogonalizationReplay(1).activate():
            actual = record(value, *args, **kwargs)  # type: ignore[arg-type]
        assert_bytes(actual, value * steps + eps)
        assert len(seen) == 1
        assert seen[0][0] is value
        assert seen[0][1:] == (steps, eps)

    @pytest.mark.parametrize(
        "clone",
        [
            pytest.param(copy.deepcopy, id="deepcopy"),
            pytest.param(lambda optimizer: pickle.loads(pickle.dumps(optimizer)), id="pickle"),
        ],
    )
    def test_copied_flag_on_optimizer_steps_like_twin(self, clone: Callable[[MuSGD], MuSGD]) -> None:
        """A deep-copied or unpickled flag-on optimizer has no replay owner yet and still continues like its twin.

        ``Optimizer.__getstate__`` keeps only the defaults, state and groups, so the copy starts without the
        owner attribute; both the restore and the next step must cope with that and give the twin's parameters.
        """
        grads = [torch.linspace(-1, 1, 12).reshape(3, 4), torch.linspace(1, -2, 12).reshape(3, 4)]
        start = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
        original_param, twin_param = torch.nn.Parameter(start.clone()), torch.nn.Parameter(start.clone())
        original = MuSGD([original_param], lr=0.01, capture_orthogonalization=True)
        twin = MuSGD([twin_param], lr=0.01, capture_orthogonalization=True)
        original_param.grad, twin_param.grad = grads[0].clone(), grads[0].clone()
        original.step()
        twin.step()
        copied = clone(original)
        assert "_orthogonalization_replay" in vars(original)
        assert "_orthogonalization_replay" not in vars(copied)
        copied.load_state_dict(copy.deepcopy(copied.state_dict()))
        copied_param = copied.param_groups[0]["params"][0]
        copied_param.grad, twin_param.grad = grads[1].clone(), grads[1].clone()
        copied.step()
        twin.step()
        assert copied.param_groups[0]["capture_orthogonalization"] is True
        assert_bytes(copied_param, twin_param)
        assert_bytes(copied.state[copied_param]["momentum_buffer"], twin.state[twin_param]["momentum_buffer"])

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
            second_input = torch.randn_like(value)
            second = function(second_input, steps=3)
        assert_bytes(first, expected)
        assert_bytes(second, function(second_input, steps=3))
        assert not torch.equal(first, second)
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
                grad = torch.randn_like(left)
                skipped = (step == 3 and index == 1) or (step == 4 and index > 0)
                left.grad = None if skipped else grad
                right.grad = None if left.grad is None else grad.clone()
            for optimizer in (baseline, candidate):
                optimizer.param_groups[0]["lr"] = 0.013 / (step + 1)
                optimizer.step()
            for left, right in zip(original, actual, strict=True):
                assert_bytes(left, right)
                assert_bytes(baseline.state[left]["momentum_buffer"], candidate.state[right]["momentum_buffer"])
            replay = candidate._orthogonalization_replay
            assert replay is not None
            if step == 4:
                assert len(replay.entries) == 3  # the stack of 3, the stack of 2 and the lone matrix
                state = copy.deepcopy(candidate.state_dict())
                candidate.load_state_dict(state)
                assert not replay.entries
        assert len(replay.entries) == 1  # steps 5 to 7 hold every gradient again: only the stack of 3 is recorded

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

    def test_failure_inside_capture_publishes_no_entry(self) -> None:
        """A function that raises while its stream is capturing leaves no entry, no open capture and a usable owner.

        The stand-in passes its warmup call and fails only on the capture call, so the exception is raised
        with the stream capturing. The graph context may replace the original error with its own, hence the
        loose ``RuntimeError``; the state left behind is what matters. The owner has one slot, so the real
        call afterwards records only if the failed attempt kept none.
        """
        owner = _OrthogonalizationReplay(1)
        value = torch.randn(8, 16, device="cuda")
        stream_before = torch.cuda.current_stream()

        @_replay_orthogonalization
        def fail_in_capture(matrix: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("raised inside capture")
            return matrix * steps + eps

        with torch.no_grad(), owner.activate(), pytest.raises(RuntimeError):
            fail_in_capture(value)
        assert not owner.entries
        assert not torch.cuda.is_current_stream_capturing()
        assert torch.cuda.current_stream() == stream_before
        assert _ACTIVE.get() is None
        with torch.no_grad(), owner.activate():
            actual = orthogonalize(value)
        assert_bytes(actual, orthogonalize(value))
        assert len(owner.entries) == 1

    @pytest.mark.parametrize("shape", [(32, 64), (2, 32, 64)])
    @pytest.mark.parametrize("setting", ["global", "cuda"])
    def test_new_precision_api_invalidates_recordings_without_legacy_getter(
        self, shape: tuple[int, ...], setting: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            pytest.skip("Per-backend precision settings require newer PyTorch")
        previous_global = torch.backends.fp32_precision
        previous_cuda = torch.backends.cuda.matmul.fp32_precision
        function = orthogonalize if len(shape) == 2 else _orthogonalize_batched
        owner = _OrthogonalizationReplay(2)
        value = torch.randn(shape, device="cuda")

        def fail_legacy_getter() -> str:
            raise AssertionError("Replay must not query legacy precision with the new API")

        monkeypatch.setattr(torch, "get_float32_matmul_precision", fail_legacy_getter)
        try:
            torch.backends.fp32_precision = "ieee"
            torch.backends.cuda.matmul.fp32_precision = "none"
            backend = torch.backends if setting == "global" else torch.backends.cuda.matmul
            for index, precision in enumerate(("tf32", "ieee", "tf32")):
                backend.fp32_precision = precision
                current = value * (index + 1)
                with torch.no_grad():
                    expected = function(current, steps=2)
                    with owner.activate():
                        actual = function(current, steps=2)
                assert_bytes(actual, expected)
                assert len(owner.entries) == min(index + 1, 2)
        finally:
            torch.backends.fp32_precision = previous_global
            torch.backends.cuda.matmul.fp32_precision = previous_cuda

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

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
    def test_enclosing_capture_on_value_device_keeps_original_operations(self) -> None:
        """A capture open on the tensor's device is found even while another device is current.

        Device 1 holds the tensor and the outer capture, device 0 is made current inside it. Looking only
        at the current device's stream would miss the capture and start a recording inside it.
        """
        owner = _OrthogonalizationReplay(1)
        with torch.cuda.device(1):
            value = torch.randn(8, 16, device="cuda:1")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.no_grad(), torch.cuda.stream(stream):
                expected = orthogonalize(value)
            torch.cuda.current_stream().wait_stream(stream)
            outer = torch.cuda.CUDAGraph()
            with torch.no_grad(), owner.activate(), torch.cuda.graph(outer, stream=stream), torch.cuda.device(0):
                actual = orthogonalize(value)
            outer.replay()
        assert_bytes(actual, expected)
        assert not owner.entries

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(_other_eps, id="eps"),
            pytest.param(_other_stream, id="stream"),
            pytest.param(_deterministic_warn_only, id="deterministic-warn-only"),
            pytest.param(_other_blas_library, id="blas-library"),
        ],
    )
    def test_changed_execution_setting_records_another_entry(
        self, change: Callable[[], AbstractContextManager[dict[str, Any]]]
    ) -> None:
        """Each setting named in the replay key sends the same input to a new recording.

        A repeat of the first call must keep the count at one, and the call under the changed setting must
        match the eager result under that setting, so a recording reused across the change would fail.
        """
        owner = _OrthogonalizationReplay(3)
        value = torch.randn(16, 32, device="cuda")
        with torch.no_grad():
            with owner.activate():
                orthogonalize(value, steps=2)
                orthogonalize(value, steps=2)
            assert len(owner.entries) == 1
            with change() as extra:
                with owner.activate():
                    actual = orthogonalize(value, steps=2, **extra)
                expected = orthogonalize(value, steps=2, **extra)
        assert len(owner.entries) == 2
        assert_bytes(actual, expected)

    def test_empty_input_keeps_eager_route(self) -> None:
        """An input with no elements is never recorded; the eager function answers it."""
        owner = _OrthogonalizationReplay(1)
        value = torch.empty(0, 16, device="cuda")
        with torch.no_grad():
            expected = orthogonalize(value)
            with owner.activate():
                actual = orthogonalize(value)
        assert_bytes(actual, expected)
        assert not owner.entries

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((32, 16), id="tall"),
            pytest.param((16, 32), id="wide"),
            pytest.param((2, 32, 16), id="tall-batch"),
        ],
    )
    def test_replayed_output_keeps_eager_strides(self, shape: tuple[int, ...]) -> None:
        """The copy handed back has the strides the eager function returns, tall inputs included.

        Tall inputs are iterated transposed and returned as a view of the iteration's layout, so a copy that
        made the result contiguous would match the values yet not the layout later code may rely on.
        """
        function = orthogonalize if len(shape) == 2 else _orthogonalize_batched
        owner = _OrthogonalizationReplay(1)
        value = torch.randn(shape, device="cuda")
        with torch.no_grad():
            expected = function(value, steps=2)
            with owner.activate():
                function(value, steps=2)
                actual = function(value, steps=2)
        assert actual.stride() == expected.stride()
        assert_bytes(actual, expected)
        assert len(owner.entries) == 1

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
