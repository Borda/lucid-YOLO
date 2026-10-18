# SPDX-License-Identifier: Apache-2.0
"""Check the live-state boundary around optional update preparation replay."""

from __future__ import annotations

import copy
import pickle
from typing import Any

import pytest
import torch
from torch import Tensor

from lucid_yolo.optim.musgd import MuSGD, _memory_overlaps
from lucid_yolo.optim.replay import _ACTIVE
from lucid_yolo.optim.update_replay import _UpdateReplay


def assert_bytes(left: Tensor, right: Tensor) -> None:
    """Compare numerical storage without rounding or hiding signed zero.

    Examples:
        >>> assert_bytes(torch.tensor([-0.0]), torch.tensor([-0.0]))
    """
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert torch.equal(
        left.detach().contiguous().reshape(-1).view(torch.uint8),
        right.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def parameters(device: str) -> list[torch.nn.Parameter]:
    """Interleave matrices and a vector so native yield order differs from group order.

    Examples:
        >>> [tuple(param.shape) for param in parameters("cpu")]
        [(4, 4), (4,), (8, 4), (4, 4, 3, 3)]
    """
    result = []
    for shape in ((4, 4), (4,), (8, 4), (4, 4, 3, 3)):
        value = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape) / 127
        if value.ndim == 4:
            value = value.contiguous(memory_format=torch.channels_last)
        result.append(torch.nn.Parameter(value))
    return result


def gradients(params: list[torch.nn.Parameter], step: int, *, skip: bool = False) -> None:
    """Supply new gradient allocations without consuming an RNG stream.

    Examples:
        >>> params = parameters("cpu")
        >>> gradients(params, 0)
        >>> params[0].grad.shape
        torch.Size([4, 4])
    """
    for index, param in enumerate(params):
        value = torch.arange(param.numel(), dtype=param.dtype, device=param.device).reshape_as(param)
        value = (value + index + step + 1) / 100
        if value.ndim == 4:
            value = value.contiguous(memory_format=torch.channels_last)
        param.grad = None if skip and index == 1 else value


def assert_state(left: MuSGD, right: MuSGD) -> None:
    """Compare parameters and every learned optimizer tensor independently of execution flags.

    Examples:
        >>> a, b = parameters("cpu"), parameters("cpu")
        >>> assert_state(MuSGD(a, lr=0.01), MuSGD(b, lr=0.01))
    """
    for first, second in zip(left.param_groups[0]["params"], right.param_groups[0]["params"], strict=True):
        assert_bytes(first, second)
        assert left.state.get(first, {}).keys() == right.state.get(second, {}).keys()
        for name, value in left.state.get(first, {}).items():
            assert_bytes(value, right.state[second][name])


class TestNativeUpdates:
    def test_cpu_flag_keeps_native_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No CUDA runtime is entered when the optional flag is used on CPU."""

        def forbidden(*args: Any, **kwargs: Any) -> None:
            """Refuse an unexpected capture.

            Examples:
                >>> forbidden()  # doctest: +SKIP
            """
            raise AssertionError("CPU attempted CUDA capture")

        monkeypatch.setattr(torch.cuda, "CUDAGraph", forbidden)
        a, b = parameters("cpu"), parameters("cpu")
        reference = MuSGD(a, lr=0.01)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        for step in range(4):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
            assert_state(reference, candidate)
        assert candidate._update_replay is not None
        assert not candidate._update_replay.entries

    @pytest.mark.parametrize("saved", [False, True])
    @pytest.mark.parametrize("launch", [False, True])
    def test_reload_preserves_launch_flag(self, saved: bool, launch: bool) -> None:
        """A checkpoint controls learned state rather than the next launch's scheduling."""
        source = MuSGD(parameters("cpu"), lr=0.01, capture_updates=saved)
        target = MuSGD(parameters("cpu"), lr=0.01, capture_updates=launch)
        state = copy.deepcopy(source.state_dict())
        target.load_state_dict(state)
        assert target.param_groups[0]["capture_updates"] is launch
        del state["param_groups"][0]["capture_updates"]
        target.load_state_dict(state)
        assert target.param_groups[0]["capture_updates"] is launch

    def test_float_ns_steps_is_refused_rather_than_cached_as_its_int_value(self) -> None:
        """A float Newton-Schulz count is refused instead of compared as equal to its integer value.

        ``ns_steps`` only reaches ``range()`` inside the Newton-Schulz iteration, which raises
        ``TypeError`` for a float no matter its value. The cache key compares scalars for equality,
        and ``5 == 5.0``, so admitting both types would let a recording captured under the int
        survive unnoticed once the group holds the equal-valued float, hiding the native rejection
        instead of returning to it.

        Examples:
            >>> params = parameters("cpu")
            >>> group = {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0, "w_muon": 0.5, "w_sgd": 0.5, "ns_steps": 5}
            >>> replay = _UpdateReplay(_memory_overlaps)
            >>> gradient, buffer = torch.ones_like(params[0]), torch.zeros_like(params[0])
            >>> replay._supported([params[0]], [gradient], [buffer], group)
            True
            >>> group["ns_steps"] = 5.0
            >>> replay._supported([params[0]], [gradient], [buffer], group)
            False
        """
        params = parameters("cpu")
        group: dict[str, Any] = {
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "w_muon": 0.5,
            "w_sgd": 0.5,
            "ns_steps": 5,
        }
        replay = _UpdateReplay(_memory_overlaps)
        gradient, buffer = torch.ones_like(params[0]), torch.zeros_like(params[0])
        assert replay._supported([params[0]], [gradient], [buffer], group)
        group["ns_steps"] = 5.0
        assert not replay._supported([params[0]], [gradient], [buffer], group)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="update replay needs CUDA")
class TestCUDAUpdates:
    @pytest.mark.parametrize("capture_orthogonalization", [False, True])
    def test_new_gradients_rates_and_native_fallbacks_match_bytes(self, capture_orthogonalization: bool) -> None:
        """Missing gradients and changed momentum return to a recording without losing state."""
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a, lr=0.01, weight_decay=0.03, capture_orthogonalization=capture_orthogonalization)
        candidate = MuSGD(
            b, lr=0.01, weight_decay=0.03, capture_updates=True, capture_orthogonalization=capture_orthogonalization
        )
        for step in range(10):
            gradients(a, step, skip=step == 4)
            gradients(b, step, skip=step == 4)
            original_gradients = [param.grad for param in b]
            for optimizer in (reference, candidate):
                optimizer.param_groups[0]["lr"] = (0.0, 0.01, 0.003)[step % 3]
                optimizer.param_groups[0]["momentum"] = 0.8 if step == 6 else 0.95
            reference.step()
            candidate.step()
            assert all(param.grad is gradient for param, gradient in zip(b, original_gradients, strict=True))
            assert_state(reference, candidate)
        assert candidate._update_replay is not None
        assert len(candidate._update_replay.entries) == 1

    def test_replay_removes_native_preparation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A silent return to the native iterator fails even if timing and values look plausible."""
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a, lr=0.01)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        for step in range(2):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
        gradients(a, 2)
        gradients(b, 2)
        reference.step()
        with monkeypatch.context() as patch:
            patch.setattr(MuSGD, "_group_updates", lambda *args: pytest.fail("native preparation ran after recording"))
            candidate.step()
        assert_state(reference, candidate)

    def test_warmup_failure_restores_gradient_identity_and_momentum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A Python failure after native preparation cannot count an extra state update."""
        params = parameters("cuda")
        optimizer = MuSGD(params, lr=0.01, capture_updates=True)
        gradients(params, 0)
        optimizer.step()
        before = [param.detach().clone() for param in params]
        buffers = [optimizer.state[param]["momentum_buffer"].clone() for param in params]
        originals = [param.grad for param in params]
        active = _ACTIVE.get()
        native = MuSGD._group_updates

        def broken(self: MuSGD, group: dict[str, Any]) -> Any:
            """Advance native momentum, then fail before a recording is published.

            Examples:
                >>> broken(None, {})  # doctest: +SKIP
            """
            next(native(self, group))
            raise RuntimeError("warmup stopped")

        with monkeypatch.context() as patch:
            patch.setattr(MuSGD, "_group_updates", broken)
            with pytest.raises(RuntimeError, match="warmup stopped"):
                optimizer.step()
        for param, weight, buffer, gradient in zip(params, before, buffers, originals, strict=True):
            assert_bytes(param, weight)
            assert_bytes(optimizer.state[param]["momentum_buffer"], buffer)
            assert param.grad is gradient
        assert _ACTIVE.get() is active
        assert optimizer._update_replay is not None
        assert not optimizer._update_replay.entries

    def test_inference_mode_never_seeds_mutable_replay_storage(self) -> None:
        """A recording seeded under inference mode would crash the next normal-mode reuse."""
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a, lr=0.01)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        gradients(a, 0)
        gradients(b, 0)
        reference.step()
        candidate.step()
        assert candidate._update_replay is not None
        assert not candidate._update_replay.entries
        gradients(a, 1)
        gradients(b, 1)
        with torch.inference_mode():
            reference.step()
            candidate.step()
        assert not candidate._update_replay.entries
        assert_state(reference, candidate)
        gradients(a, 2)
        gradients(b, 2)
        reference.step()
        candidate.step()
        assert_state(reference, candidate)
        assert len(candidate._update_replay.entries) == 1

    def test_reload_discards_recordings(self) -> None:
        """A restored momentum buffer is never replaced by an old captured address."""
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a, lr=0.01)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        for step in range(3):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
        assert candidate._update_replay is not None
        state = copy.deepcopy(candidate.state_dict())
        candidate.load_state_dict(state)
        assert not candidate._update_replay.entries
        for step in range(3, 6):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
            assert_state(reference, candidate)

    def test_scaler_overflow_skips_momentum_and_recovers(self) -> None:
        """A skipped AMP update leaves the recording and live momentum at the same step."""
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a, lr=0.01)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        scalers = [torch.amp.GradScaler("cuda", init_scale=16.0) for _ in range(2)]
        for step in range(5):
            for params, optimizer, scaler in zip((a, b), (reference, candidate), scalers, strict=True):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = sum(param.square().mean() for param in params)
                scaler.scale(loss).backward()
                if step == 2:
                    params[0].grad.fill_(float("inf"))
                scaler.step(optimizer)
                scaler.update()
            assert_state(reference, candidate)
            assert scalers[0].state_dict() == scalers[1].state_dict()
        assert candidate._update_replay is not None
        assert len(candidate._update_replay.entries) == 1

    def test_pickle_reconstructs_recordings(self) -> None:
        """Optimizer's pickle protocol omits runtime owners and rebuilds them on demand."""
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a, lr=0.01)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        for step in range(2):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
        candidate = pickle.loads(pickle.dumps(candidate))
        b = candidate.param_groups[0]["params"]
        for step in range(2, 5):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
            assert_state(reference, candidate)
        assert candidate._update_replay is not None
        assert len(candidate._update_replay.entries) == 1

    def test_layout_keeps_parameter_write_on_foreach_route(self) -> None:
        """Removing the captured layout copy restores per-parameter additions and fails here."""
        params = parameters("cuda")
        optimizer = MuSGD(params, lr=0.01, capture_updates=True)
        for step in range(2):
            gradients(params, step)
            optimizer.step()
        gradients(params, 2)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            optimizer.step()
            torch.cuda.synchronize()
        counts = {event.key: event.count for event in profile.key_averages()}
        assert counts.get("aten::_foreach_add_", 0) == 1
        assert counts.get("aten::add_", 0) == 0

    def test_gradient_alias_keeps_ordered_native_writes(self) -> None:
        """A later gradient may read a parameter an earlier native update writes."""
        a = [torch.nn.Parameter(torch.arange(4, device="cuda", dtype=torch.float32)) for _ in range(2)]
        b = [torch.nn.Parameter(param.detach().clone()) for param in a]
        reference = MuSGD(a, lr=0.01)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        for _ in range(4):
            for params in (a, b):
                params[0].grad = torch.ones_like(params[0])
                params[1].grad = params[0].detach()
            reference.step()
            candidate.step()
            assert_state(reference, candidate)
        assert candidate._update_replay is not None
        assert not candidate._update_replay.entries

    def test_signed_zero_setting_preserves_momentum_bits(self) -> None:
        """Numerically equal scalar keys cannot hide the native sign of a zero buffer."""
        a = [torch.nn.Parameter(torch.ones(1, device="cuda"))]
        b = [torch.nn.Parameter(a[0].detach().clone())]
        reference = MuSGD(a, lr=0.01, momentum=0.0, w_muon=0.0)
        candidate = MuSGD(b, lr=0.01, momentum=0.0, w_muon=0.0, capture_updates=True)
        for _ in range(2):
            for params, optimizer in ((a, reference), (b, candidate)):
                params[0].grad = torch.zeros_like(params[0])
                optimizer.step()
        for params, optimizer in ((a, reference), (b, candidate)):
            optimizer.param_groups[0]["momentum"] = -0.0
            params[0].grad = -torch.zeros_like(params[0])
            optimizer.step()
        assert_state(reference, candidate)
        assert reference.state[a[0]]["momentum_buffer"].view(torch.int32).item() == -(2**31)

    def test_scalar_integer_conversion_error_remains_native(self) -> None:
        """A float gain and an equal oversized integer have different native contracts."""
        a = [torch.nn.Parameter(torch.ones(2, 2, device="cuda"))]
        b = [torch.nn.Parameter(a[0].detach().clone())]
        reference = MuSGD(a, lr=0.01, w_muon=0.0, w_sgd=float(2**64))
        candidate = MuSGD(b, lr=0.01, w_muon=0.0, w_sgd=float(2**64), capture_updates=True)
        for _ in range(2):
            for params, optimizer in ((a, reference), (b, candidate)):
                params[0].grad = torch.full_like(params[0], 1e-30)
                optimizer.step()
        for params, optimizer in ((a, reference), (b, candidate)):
            optimizer.param_groups[0]["w_sgd"] = 2**64
            params[0].grad = torch.full_like(params[0], 1e-30)
            with pytest.raises(OverflowError, match="int too big"):
                optimizer.step()
        assert_state(reference, candidate)
