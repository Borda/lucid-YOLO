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


def assert_bytes_except_nan(left: Tensor, right: Tensor) -> None:
    """Compare NaN positions, then the remaining storage bytes, ignoring which NaN payload was produced.

    Examples:
        >>> assert_bytes_except_nan(torch.tensor([float("nan"), 1.0]), torch.tensor([float("nan"), 1.0]))
    """
    assert torch.equal(left.isnan(), right.isnan())
    assert_bytes(left.masked_fill(left.isnan(), 0.0), right.masked_fill(right.isnan(), 0.0))


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


def twin_matrix_parameters(device: str) -> list[torch.nn.Parameter]:
    """Hold two equal-shape matrices so the stacked update path runs.

    Kept apart from ``parameters`` because the profiler-count test relies on that list having no equal shapes.

    Examples:
        >>> [tuple(param.shape) for param in twin_matrix_parameters("cpu")]
        [(4, 4), (4, 4), (4,)]
    """
    result = []
    for shape in ((4, 4), (4, 4), (4,)):
        value = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape) / 127
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
    for first_group, second_group in zip(left.param_groups, right.param_groups, strict=True):
        for first, second in zip(first_group["params"], second_group["params"], strict=True):
            assert_bytes(first, second)
            assert left.state.get(first, {}).keys() == right.state.get(second, {}).keys()
            for name, value in left.state.get(first, {}).items():
                assert_bytes(value, right.state[second][name])


class TestNativeUpdates:
    def test_default_optimizer_keeps_update_replay_off(self) -> None:
        """Without the opt-in flag the group asks for no replay and no owner is ever created.

        The CUDA reference optimizers in this file name ``capture_updates=False`` explicitly, so this
        is the case that fails if the constructor default flips.
        """
        param = torch.nn.Parameter(torch.ones(3, 4))
        optimizer = MuSGD([param], lr=0.01)
        param.grad = torch.ones_like(param)
        optimizer.step()
        assert optimizer.param_groups[0]["capture_updates"] is False
        assert optimizer._update_replay is None

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
        reference = MuSGD(a, lr=0.01, capture_updates=False)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        for step in range(4):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
            assert_state(reference, candidate)
        assert candidate._update_replay is not None
        assert not candidate._update_replay.entries

    def test_subclass_keeps_native_route(self) -> None:
        """A MuSGD subclass never builds a replay owner, because an override could write parameters mid-recording."""

        class Derived(MuSGD):
            """MuSGD subclass with no overrides.

            Examples:
                >>> Derived([torch.nn.Parameter(torch.ones(2))], lr=0.1).param_groups[0]["capture_updates"]
                False
            """

            __slots__ = ()

        a, b = parameters("cpu"), parameters("cpu")
        reference = MuSGD(a, lr=0.01, capture_updates=False)
        candidate = Derived(b, lr=0.01, capture_updates=True)
        for step in range(3):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
            assert_state(reference, candidate)
        assert candidate._update_replay is None

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

    def test_float_ns_steps_refused_not_cached_as_int(self) -> None:
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
        reference = MuSGD(
            a,
            lr=0.01,
            weight_decay=0.03,
            capture_updates=False,
            capture_orthogonalization=capture_orthogonalization,
        )
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

    def test_groups_with_own_rates_match_bytes(self) -> None:
        """Each group keeps its own recording, learning rate and momentum."""
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(
            [{"params": a[:2]}, {"params": a[2:], "lr": 0.003, "momentum": 0.9}], lr=0.01, capture_updates=False
        )
        candidate = MuSGD(
            [{"params": b[:2]}, {"params": b[2:], "lr": 0.003, "momentum": 0.9}], lr=0.01, capture_updates=True
        )
        for step in range(5):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
            assert_state(reference, candidate)
        assert candidate._update_replay is not None
        assert len(candidate._update_replay.entries) == 2

    def test_group_added_after_recording_is_recorded_too(self) -> None:
        """A group added once another is already replaying warms up natively, then replays on its own."""
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a[:2], lr=0.01, capture_updates=False)
        candidate = MuSGD(b[:2], lr=0.01, capture_updates=True)
        for step in range(3):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
        assert candidate._update_replay is not None
        assert len(candidate._update_replay.entries) == 1
        for optimizer, params in ((reference, a), (candidate, b)):
            optimizer.add_param_group({"params": params[2:], "lr": 0.003})
        for step in range(3, 7):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
            assert_state(reference, candidate)
        assert len(candidate._update_replay.entries) == 2

    def test_equal_shape_matrices_replay_stacked_update(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two equal-shape matrices take the stacked update inside the recording and match native bytes."""
        a, b = twin_matrix_parameters("cuda"), twin_matrix_parameters("cuda")
        reference = MuSGD(a, lr=0.01, capture_updates=False)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        native = MuSGD._batched_parameter_update
        captured: list[tuple[int, ...]] = []

        def spy(params: Tensor, grads: Tensor, group: dict[str, Any]) -> Tensor:
            """Note each stacked update issued while a graph is capturing, then run it unchanged.

            Examples:
                >>> spy  # doctest: +ELLIPSIS
                <function ...spy at ...>
            """
            if torch.cuda.is_current_stream_capturing():
                captured.append(tuple(params.shape))
            return native(params, grads, group)

        monkeypatch.setattr(MuSGD, "_batched_parameter_update", staticmethod(spy))
        for step in range(5):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
            assert_state(reference, candidate)
        assert captured == [(2, 4, 4)]
        assert candidate._update_replay is not None
        assert len(candidate._update_replay.entries) == 1

    def test_replay_removes_native_preparation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A silent return to the native iterator fails even if timing and values look plausible."""
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a, lr=0.01, capture_updates=False)
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

    def test_failure_inside_capture_publishes_no_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failure raised while the stream is capturing leaves no entry, no open capture and a usable optimizer.

        The stand-in iterator delegates to the native one during warmup and fails only on the capture call.
        The graph context may replace the original error with its own, hence the loose ``RuntimeError``; the
        state left behind is what matters. The next real step records only if the failed attempt kept nothing.
        """
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a, lr=0.01, capture_updates=False)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        gradients(a, 0)
        gradients(b, 0)
        reference.step()
        candidate.step()
        gradients(a, 1)
        gradients(b, 1)
        before = [param.detach().clone() for param in b]
        buffers = [candidate.state[param]["momentum_buffer"].clone() for param in b]
        originals = [param.grad for param in b]
        active, stream = _ACTIVE.get(), torch.cuda.current_stream()
        native = MuSGD._group_updates

        def fail_in_capture(self: MuSGD, group: dict[str, Any]) -> Any:
            """Run natively unless the stream is capturing, then fail.

            Examples:
                >>> fail_in_capture(None, {})  # doctest: +SKIP
            """
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("raised inside capture")
            yield from native(self, group)

        with monkeypatch.context() as patch:
            patch.setattr(MuSGD, "_group_updates", fail_in_capture)
            with pytest.raises(RuntimeError):
                candidate.step()
        for param, weight, buffer, gradient in zip(b, before, buffers, originals, strict=True):
            assert_bytes(param, weight)
            assert_bytes(candidate.state[param]["momentum_buffer"], buffer)
            assert param.grad is gradient
        assert not torch.cuda.is_current_stream_capturing()
        assert torch.cuda.current_stream() == stream
        assert _ACTIVE.get() is active
        assert candidate._update_replay is not None
        assert not candidate._update_replay.entries
        reference.step()
        candidate.step()
        assert_state(reference, candidate)
        assert len(candidate._update_replay.entries) == 1

    def test_failure_in_first_run_publishes_no_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failure copying gradients into a fresh recording leaves the optimizer where it started.

        The recording is built and rolled back before its first run, and the failure precedes the graph launch,
        so parameters and momentum are untouched and nothing is published. The stand-in fails at the first
        ``_foreach_copy_`` whose target is not a momentum buffer, which is the copy into the recording's
        private gradients. A failure after the launch would leave momentum advanced and is not covered here.
        """
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a, lr=0.01, capture_updates=False)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        gradients(a, 0)
        gradients(b, 0)
        reference.step()
        candidate.step()
        gradients(a, 1)
        gradients(b, 1)
        before = [param.detach().clone() for param in b]
        buffers = [candidate.state[param]["momentum_buffer"].clone() for param in b]
        originals = [param.grad for param in b]
        momentum = {id(candidate.state[param]["momentum_buffer"]) for param in b}
        active, stream = _ACTIVE.get(), torch.cuda.current_stream()
        native = torch._foreach_copy_
        failures: list[int] = []

        def fail_first_input_copy(targets: list[Tensor], sources: list[Tensor], *args: Any, **kwargs: Any) -> Any:
            """Fail once on the copy into recorded gradients; pass every other copy through.

            Examples:
                >>> fail_first_input_copy([], [])  # doctest: +SKIP
            """
            if id(targets[0]) not in momentum and not failures:
                failures.append(len(targets))
                raise RuntimeError("first run stopped")
            return native(targets, sources, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch, "_foreach_copy_", fail_first_input_copy)
            with pytest.raises(RuntimeError, match="first run stopped"):
                candidate.step()
        assert failures == [len(b)]
        for param, weight, buffer, gradient in zip(b, before, buffers, originals, strict=True):
            assert_bytes(param, weight)
            assert_bytes(candidate.state[param]["momentum_buffer"], buffer)
            assert param.grad is gradient
        assert not torch.cuda.is_current_stream_capturing()
        assert torch.cuda.current_stream() == stream
        assert _ACTIVE.get() is active
        assert candidate._update_replay is not None
        assert not candidate._update_replay.entries
        reference.step()
        candidate.step()
        assert_state(reference, candidate)
        assert len(candidate._update_replay.entries) == 1

    def test_inference_mode_never_seeds_mutable_replay_storage(self) -> None:
        """A recording seeded under inference mode would crash the next normal-mode reuse."""
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a, lr=0.01, capture_updates=False)
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
        reference = MuSGD(a, lr=0.01, capture_updates=False)
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
        reference = MuSGD(a, lr=0.01, capture_updates=False)
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

    @pytest.mark.parametrize("value", ["inf", "nan"])
    def test_nonfinite_gradient_replays_like_native(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """A step called directly with an infinite or NaN gradient propagates it exactly as native does.

        ``GradScaler`` skips such steps, so only a direct call reaches the recording with them. NaN payload
        bits are not part of the contract, hence the comparison keeps the NaN positions and the finite bytes.
        """
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a, lr=0.01, capture_updates=False)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        for step in range(3):
            gradients(a, step)
            gradients(b, step)
            reference.step()
            candidate.step()
        assert candidate._update_replay is not None
        assert len(candidate._update_replay.entries) == 1
        for params in (a, b):
            gradients(params, 3)
            params[0].grad[0, 0] = float(value)
        reference.step()
        with monkeypatch.context() as patch:
            patch.setattr(MuSGD, "_group_updates", lambda *args: pytest.fail("native preparation ran after recording"))
            candidate.step()
        for first, second in zip(a, b, strict=True):
            assert_bytes_except_nan(first, second)
            assert_bytes_except_nan(
                reference.state[first]["momentum_buffer"], candidate.state[second]["momentum_buffer"]
            )
        assert torch.isnan(a[0]).any() or torch.isinf(a[0]).any()

    def test_pickle_reconstructs_recordings(self) -> None:
        """Optimizer's pickle protocol omits runtime owners and rebuilds them on demand."""
        a, b = parameters("cuda"), parameters("cuda")
        reference = MuSGD(a, lr=0.01, capture_updates=False)
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
        reference = MuSGD(a, lr=0.01, capture_updates=False)
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
        reference = MuSGD(a, lr=0.01, momentum=0.0, w_muon=0.0, capture_updates=False)
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
        reference = MuSGD(a, lr=0.01, w_muon=0.0, w_sgd=float(2**64), capture_updates=False)
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
