# SPDX-License-Identifier: Apache-2.0
"""Check replay's Python ownership and failure contracts without a CUDA device.

CPU tensors exercise storage, signatures and state restoration. CUDA submission objects are stubs: these tests do not
certify graph execution or device arithmetic. The real CUDA trajectories remain in test_replay.py and
test_update_replay.py.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
from torch import Tensor

from lucid_yolo.optim import replay as orthogonalization
from lucid_yolo.optim import update_replay as preparation
from lucid_yolo.optim.musgd import MuSGD, _memory_overlaps


@pytest.fixture
def submission(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace device submission while retaining real CPU tensor storage.

    Examples:
        >>> submission(None)  # doctest: +SKIP
    """
    current, stream, graph = Mock(), Mock(), Mock()
    current.cuda_stream = 17
    stream_factory = Mock(return_value=stream)
    monkeypatch.setattr(torch.cuda, "Stream", stream_factory)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Mock(return_value=graph))
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=current))
    monkeypatch.setattr(torch.cuda, "device", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", Mock(return_value=False))
    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", lambda: "cublas")
    return SimpleNamespace(current=current, stream=stream, graph=graph, factory=stream_factory)


@pytest.fixture
def inputs() -> tuple[MuSGD, list[torch.nn.Parameter], list[Tensor], list[Tensor]]:
    """Give each parameter, gradient and momentum buffer independent storage.

    Examples:
        >>> inputs()  # doctest: +SKIP
    """
    params = [torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3)) for _ in range(2)]
    params.append(torch.nn.Parameter(torch.ones(2, 2, 2, 2).contiguous(memory_format=torch.channels_last)))
    optimizer = MuSGD(params, lr=0.125, momentum=0.0, ns_steps=2)
    gradients, buffers = [], []
    for param in params:
        param.grad = torch.ones_like(param)
        buffer = torch.full_like(param, 0.25)
        optimizer.state[param]["momentum_buffer"] = buffer
        gradients.append(param.grad)
        buffers.append(buffer)
    return optimizer, params, gradients, buffers


class TestUpdateSignatures:
    @pytest.mark.parametrize(
        "edits",
        [
            # Each edit is (target, key, build): ``target[key]`` is replaced by ``build(params, grads, bufs)``.
            pytest.param([("group", "lr", lambda params, grads, bufs: True)], id="bool-rate"),
            pytest.param([("group", "ns_steps", lambda params, grads, bufs: 2.0)], id="float-steps"),
            pytest.param([("gradients", 0, lambda params, grads, bufs: grads[0].double())], id="dtype"),
            pytest.param(
                [("buffers", 0, lambda params, grads, bufs: torch.empty_like(bufs[0], device="meta"))], id="device"
            ),
            pytest.param(
                [
                    ("params", 0, lambda params, grads, bufs: torch.nn.Parameter(torch.empty(0, 3))),
                    ("gradients", 0, lambda params, grads, bufs: torch.empty(0, 3)),
                    ("buffers", 0, lambda params, grads, bufs: torch.empty(0, 3)),
                ],
                id="empty",
            ),
            pytest.param([("gradients", 0, lambda params, grads, bufs: torch.ones(2, 6)[:, ::2])], id="strides"),
            pytest.param([("gradients", 0, lambda params, grads, bufs: grads[0].to_sparse())], id="layout"),
            pytest.param([("buffers", 0, lambda params, grads, bufs: torch.zeros(3, 2))], id="shape"),
            pytest.param(
                [("params", 1, lambda params, grads, bufs: torch.nn.Parameter(params[0].detach()))], id="parameters"
            ),
            pytest.param([("buffers", 1, lambda params, grads, bufs: bufs[0])], id="buffers"),
            pytest.param([("gradients", 0, lambda params, grads, bufs: params[0].detach())], id="gradient"),
        ],
    )
    def test_unsupported_storage_or_scalar_keeps_native_route(
        self, inputs: Any, edits: list[tuple[str, int | str, Callable[..., Any]]]
    ) -> None:
        """An unsupported input cannot acquire live-state ownership through replay."""
        optimizer, params, gradients, buffers = inputs
        group = optimizer.param_groups[0]
        owner = preparation._UpdateReplay(_memory_overlaps)
        assert owner._supported(params, gradients, buffers, group)
        targets = {"group": group, "params": params, "gradients": gradients, "buffers": buffers}
        for target, key, build in edits:
            targets[target][key] = build(params, gradients, buffers)
        assert not owner._supported(params, gradients, buffers, group)

    def test_cross_allocation_overlap_keeps_native_route(self, inputs: Any) -> None:
        """An overlap detector can reject aliases even when storage objects differ."""
        optimizer, params, gradients, buffers = inputs
        overlap = Mock(return_value=True)
        assert not preparation._UpdateReplay(overlap)._supported(params, gradients, buffers, optimizer.param_groups[0])
        overlap.assert_called_once()


class TestUpdateRecordingState:
    @pytest.mark.parametrize("failure", [None, 1, 2])
    def test_recording_restores_live_state_after_success_or_failure(
        self, inputs: Any, submission: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: int | None
    ) -> None:
        """Warmup and recording cannot advance momentum or replace autograd gradients."""
        optimizer, params, gradients, buffers = inputs
        snapshots = [value.clone() for value in buffers]
        weights = [param.detach().clone() for param in params]
        native = optimizer._group_updates
        calls = []

        def updates(group: dict[str, Any]) -> Any:
            """Advance real native preparation before injecting a submission failure.

            Examples:
                >>> list(updates({}))  # doctest: +SKIP
            """
            assert orthogonalization._ACTIVE.get() is None
            calls.append(None)
            yield from native(group)
            if len(calls) == failure:
                raise RuntimeError("recording stopped")

        monkeypatch.setattr(optimizer, "_group_updates", updates)
        previous = orthogonalization._OrthogonalizationReplay(1)
        with previous.activate(), torch.no_grad():
            if failure is None:
                recorded = preparation._PreparedGroup(
                    optimizer, optimizer.param_groups[0], params, gradients, buffers, ()
                )
                assert len(recorded.updates) == len(params)
                assert recorded.updates[-1].is_contiguous(memory_format=torch.channels_last)
                assert all(
                    private.data_ptr() != original.data_ptr()
                    for private, original in zip(recorded.gradients, gradients, strict=True)
                )
            else:
                with pytest.raises(RuntimeError, match="recording stopped"):
                    preparation._PreparedGroup(optimizer, optimizer.param_groups[0], params, gradients, buffers, ())
            assert orthogonalization._ACTIVE.get() is previous
        for param, gradient, buffer, snapshot, weight in zip(
            params, gradients, buffers, snapshots, weights, strict=True
        ):
            assert param.grad is gradient
            assert torch.equal(buffer, snapshot)
            assert torch.equal(param, weight)
        submission.current.synchronize.assert_called_once()

    def test_replay_copies_fresh_gradients_before_applying_current_rate(
        self, inputs: Any, submission: SimpleNamespace
    ) -> None:
        """The Python boundary copies inputs before submission and writes with the live rate."""
        optimizer, params, gradients, buffers = inputs
        group = optimizer.param_groups[0]
        with torch.no_grad():
            recorded = preparation._PreparedGroup(optimizer, group, params, gradients, buffers, ())
            fresh = [torch.full_like(value, 2.0) for value in gradients]
            weights = [param.clone() for param in params]

            def submit() -> None:
                """Stand in for device execution with a known output, not a CUDA oracle.

                Examples:
                    >>> submit()  # doctest: +SKIP
                """
                assert all(
                    torch.equal(value, expected) for value, expected in zip(recorded.gradients, fresh, strict=True)
                )
                for value in recorded.updates:
                    value.fill_(3.0)

            submission.graph.replay.side_effect = submit
            group["lr"] = 0.25
            recorded.run(fresh)
        submission.graph.replay.assert_called_once()
        for param, weight in zip(params, weights, strict=True):
            assert torch.equal(param, weight - 0.75)


class TestUpdateCacheOwnership:
    @pytest.mark.parametrize(
        "change", ["values", "rate", "signed-zero", "scalar-type", "buffer", "storage", "stream", "missing"]
    )
    @patch.object(torch.nn.Parameter, "is_cuda", property(lambda self: True))
    def test_only_values_and_rates_reuse_cached_group(
        self, inputs: Any, submission: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, change: str
    ) -> None:
        """A cache hit may accept fresh allocations without accepting a different state owner."""
        optimizer, params, _, _ = inputs
        group = optimizer.param_groups[0]
        factory = Mock(
            side_effect=lambda *args: SimpleNamespace(
                key=args[-1],
                run=Mock(),
                replay_stream=submission.current,
                spans=SimpleNamespace(reached_by=Mock(return_value=False)),
                mismatches=0,
            )
        )
        monkeypatch.setattr(preparation, "_PreparedGroup", factory)
        owner = preparation._UpdateReplay(_memory_overlaps)
        with torch.no_grad():
            assert owner.step(optimizer, group)
            entry = owner.entries[id(group)]
            if change == "values":
                for param in params:
                    param.grad = torch.full_like(param, 2.0)
            elif change == "rate":
                group["lr"] = 0.03125
            elif change == "signed-zero":
                group["momentum"] = -0.0
            elif change == "scalar-type":
                group["momentum"] = 0
            elif change == "buffer":
                optimizer.state[params[0]]["momentum_buffer"] = torch.zeros_like(params[0])
            elif change == "storage":
                params[0].data = params[0].detach().clone()
            elif change == "stream":
                submission.current.cuda_stream += 1
            else:
                params[0].grad = None
            reusable = change in {"values", "rate"}
            assert owner.step(optimizer, group) is reusable
        assert factory.call_count == 1
        assert entry.run.call_count == (2 if reusable else 1)
        assert len(owner.entries) == 1
        owner.clear()
        assert not owner.entries

    @pytest.mark.parametrize("failure", ["construct", "submit", "synchronize"])
    @patch.object(torch.nn.Parameter, "is_cuda", property(lambda self: True))
    def test_failed_first_submission_never_enters_cache(
        self, inputs: Any, submission: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str
    ) -> None:
        """A partially initialized owner is never reused after an unsuccessful first call."""
        optimizer, _, _, _ = inputs
        entry = SimpleNamespace(run=Mock(), replay_stream=submission.current)
        factory = Mock(return_value=entry)
        if failure == "construct":
            factory.side_effect = RuntimeError("submission stopped")
        elif failure == "submit":
            entry.run.side_effect = RuntimeError("submission stopped")
        else:
            submission.current.synchronize.side_effect = RuntimeError("submission stopped")
        monkeypatch.setattr(preparation, "_PreparedGroup", factory)
        owner = preparation._UpdateReplay(_memory_overlaps)
        with torch.no_grad(), pytest.raises(RuntimeError, match="submission stopped"):
            owner.step(optimizer, optimizer.param_groups[0])
        assert not owner.entries


class TestSharedRecordingState:
    @patch.object(Tensor, "is_cuda", property(lambda self: True))
    def test_returned_results_own_storage_and_capacity_limits_recordings(self, submission: SimpleNamespace) -> None:
        """Returned tensors cannot mutate recordings, and a changed stream cannot grow a full cache."""
        operation = Mock(side_effect=lambda value, steps, eps: value * steps + eps)
        value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        owner = orthogonalization._OrthogonalizationReplay(1)
        with torch.no_grad():
            first = owner.run(operation, value, 1, 0.0)
            assert len(owner.entries) == 1
            recorded = next(iter(owner.entries.values()))
            first.fill_(-1.0)
            assert torch.equal(recorded.result, value)
            submission.graph.replay.side_effect = lambda: recorded.result.fill_(7.0)
            fresh = value + 2
            second = owner.run(operation, fresh, 1, 0.0)
            assert torch.equal(recorded.source, fresh)
            assert torch.equal(second, torch.full_like(second, 7.0))
            recorded.result.fill_(8.0)
            assert torch.equal(second, torch.full_like(second, 7.0))
            submission.current.cuda_stream += 1
            fallback = owner.run(operation, fresh, 1, 0.0)
            assert torch.equal(fallback, fresh)
        assert len(owner.entries) == 1
        assert operation.call_count == 3  # warmup, recording, then native overflow
        submission.factory.assert_called_once()
        owner.clear()
        assert not owner.entries

    @pytest.mark.parametrize("capturing", [[True], [False, True], [False, False]])
    def test_capture_query_checks_requested_device(
        self, submission: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capturing: list[bool]
    ) -> None:
        """An existing capture on either stream prevents a nested recording."""
        query = Mock(side_effect=capturing)
        device = Mock(return_value=nullcontext())
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", query)
        monkeypatch.setattr(torch.cuda, "device", device)
        assert orthogonalization._capture_open(torch.device("cuda:1")) is any(capturing)
        assert query.call_count == len(capturing)
        assert device.call_count == (len(capturing) - 1)
        if len(capturing) > 1:
            device.assert_called_once_with(torch.device("cuda:1"))
