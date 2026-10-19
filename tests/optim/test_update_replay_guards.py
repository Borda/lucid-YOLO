# SPDX-License-Identifier: Apache-2.0
"""Check the guards, recording key and recording lifecycle of optional update preparation replay."""

from __future__ import annotations

import copy
import gc
import weakref
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from lucid_yolo.optim.musgd import MuSGD, _memory_overlaps
from lucid_yolo.optim.update_replay import (
    _MISMATCH_TOLERANCE,
    _RERECORD_BUDGET,
    _parameter_layout,
    _RecordedSpans,
    _UpdateReplay,
)

_SHAPES = ((4, 4), (4,), (8, 4), (4, 4, 3, 3))


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


def parameters(device: str, shapes: tuple[tuple[int, ...], ...] = _SHAPES) -> list[torch.nn.Parameter]:
    """Build float32 parameters, holding four-dimensional ones channels-last as a converted convolution does.

    Examples:
        >>> [param.stride() for param in parameters("cpu", ((4, 4), (4, 4, 1, 1)))]
        [(4, 1), (4, 1, 4, 4)]
    """
    result = []
    for shape in shapes:
        value = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape) / 127
        memory_format = torch.channels_last if len(shape) == 4 else torch.contiguous_format
        result.append(torch.nn.Parameter(torch.empty(shape, device=device, memory_format=memory_format).copy_(value)))
    return result


def gradients(params: list[torch.nn.Parameter], step: int) -> None:
    """Supply new gradient allocations with each parameter's own strides, without consuming an RNG stream.

    Examples:
        >>> params = parameters("cpu", ((4, 4, 1, 1),))
        >>> gradients(params, 0)
        >>> params[0].grad.stride()
        (4, 1, 4, 4)
    """
    for index, param in enumerate(params):
        value = torch.arange(param.numel(), dtype=param.dtype, device=param.device).reshape(param.shape)
        param.grad = torch.empty_like(param).copy_((value + index + step + 1) / 100)


def assert_state(left: MuSGD, right: MuSGD) -> None:
    """Compare parameters and every learned optimizer tensor independently of execution flags.

    Examples:
        >>> assert_state(MuSGD(parameters("cpu"), lr=0.01), MuSGD(parameters("cpu"), lr=0.01))
    """
    for first, second in zip(left.param_groups[0]["params"], right.param_groups[0]["params"], strict=True):
        assert_bytes(first, second)
        assert left.state.get(first, {}).keys() == right.state.get(second, {}).keys()
        for name, value in left.state.get(first, {}).items():
            assert_bytes(value, right.state[second][name])


def advance(
    pairs: tuple[tuple[list[torch.nn.Parameter], MuSGD], ...], step: int, calls: list[MuSGD], **values: float
) -> int:
    """Step every optimizer on equal inputs and return how often the last one started native preparation.

    The first optimizer is the native reference; the last is the candidate whose state
    must match it byte for byte after the step. ``values`` overwrite group entries of
    every optimizer before the step, and ``calls`` lists the optimizer of each native
    preparation the ``native_calls`` fixture has seen.

    Examples:
        >>> a, b = parameters("cpu"), parameters("cpu")
        >>> pairs = ((a, MuSGD(a, lr=0.01)), (b, MuSGD(b, lr=0.01, capture_updates=True)))
        >>> advance(pairs, 0, [], momentum=0.9)
        0
    """
    for params, optimizer in pairs:
        gradients(params, step)
        optimizer.param_groups[0].update(values)
    candidate = pairs[-1][1]
    before = calls.count(candidate)
    for _, optimizer in pairs:
        optimizer.step()
    assert_state(pairs[0][1], candidate)
    return calls.count(candidate) - before


@contextmanager
def high_matmul_precision() -> Iterator[None]:
    """Allow TF32 float32 matrix products, restoring every precision setting afterwards.

    The legacy setter also moves the per-backend settings newer PyTorch reads, and
    restoring through it alone leaves those at other values than before.

    Examples:
        >>> with high_matmul_precision():
        ...     torch.get_float32_matmul_precision()
        'high'
    """
    owners = [owner for owner in (torch.backends, torch.backends.cuda.matmul) if hasattr(owner, "fp32_precision")]
    legacy, backends = torch.get_float32_matmul_precision(), [owner.fp32_precision for owner in owners]
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(legacy)
        for owner, value in zip(owners, backends, strict=True):
            owner.fp32_precision = value


@contextmanager
def deterministic_warn_only() -> Iterator[None]:
    """Enable deterministic algorithms in warn-only mode, restoring both flags afterwards.

    Examples:
        >>> with deterministic_warn_only():
        ...     torch.are_deterministic_algorithms_enabled()
        True
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@contextmanager
def other_blas_library() -> Iterator[None]:
    """Run under a cuBLAS front end different from the current one, restoring the previous choice.

    Examples:
        >>> with other_blas_library():  # needs a CUDA build of PyTorch  # doctest: +SKIP
        ...     pass
    """
    previous = torch.backends.cuda.preferred_blas_library()
    if torch.backends.cuda.preferred_blas_library("cublaslt") == previous:
        torch.backends.cuda.preferred_blas_library("cublas")
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(previous)


@contextmanager
def side_stream() -> Iterator[None]:
    """Run on a new CUDA stream ordered after the caller's stream and before its next work.

    Examples:
        >>> with side_stream():  # needs a CUDA device  # doctest: +SKIP
        ...     pass
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield
    torch.cuda.current_stream().wait_stream(stream)


@contextmanager
def flipped_fill() -> Iterator[None]:
    """Invert whether deterministic mode fills new allocations, restoring the flag afterwards.

    Examples:
        >>> before = torch.utils.deterministic.fill_uninitialized_memory
        >>> with flipped_fill():
        ...     torch.utils.deterministic.fill_uninitialized_memory is not before
        True
    """
    previous = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = not previous
    try:
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous


@dataclass(frozen=True)
class RecordingInputs:
    """Parameters, gradients, momentum buffers and group scalars as the recording check receives them."""

    params: tuple[Tensor, ...]
    gradients: tuple[Tensor, ...]
    buffers: tuple[Tensor, ...]
    group: dict[str, Any]

    @classmethod
    def supported(cls) -> RecordingInputs:
        """Build one independent dense float32 set the check accepts."""
        params = parameters("cpu")
        gradients(params, 0)
        return cls(
            params=tuple(params),
            gradients=tuple(param.grad for param in params if param.grad is not None),
            buffers=tuple(torch.zeros_like(param) for param in params),
            group={"lr": 0.01, "momentum": 0.95, "weight_decay": 5e-4, "w_muon": 0.5, "w_sgd": 0.5, "ns_steps": 5},
        )

    def with_tensor(self, field: str, index: int, value: Tensor) -> RecordingInputs:
        """Replace one tensor of ``params``, ``gradients`` or ``buffers``."""
        values = list(getattr(self, field))
        values[index] = torch.nn.Parameter(value) if field == "params" else value
        return replace(self, **{field: tuple(values)})

    def with_shared(self, *places: tuple[str, int]) -> RecordingInputs:
        """Move the named two-dimensional or vector tensors into disjoint views of one new storage."""
        shapes = [getattr(self, field)[index].shape for field, index in places]
        storage = torch.zeros(sum(shape.numel() for shape in shapes))
        result = self
        for (field, index), shape, view in zip(
            places, shapes, storage.split([shape.numel() for shape in shapes]), strict=True
        ):
            result = result.with_tensor(field, index, view.view(shape))
        return result

    def with_imported_overlap(self) -> RecordingInputs:
        """Let the ``(8, 4)`` gradient import bytes of its buffer through DLPack, under another storage base."""
        owner = torch.zeros(36)
        buffer = owner[4:].view(8, 4)
        imported = torch.from_dlpack(owner[2:34]).view(8, 4)
        return self.with_tensor("buffers", 2, buffer).with_tensor("gradients", 2, imported)

    def with_scalars(self, **values: object) -> RecordingInputs:
        """Replace group scalars."""
        return replace(self, group={**self.group, **values})

    def accepted(self) -> bool:
        """Run the full recording check on these inputs."""
        replay = _UpdateReplay(_memory_overlaps)
        return replay._supported(list(self.params), list(self.gradients), list(self.buffers), self.group)


@pytest.fixture
def native_calls(monkeypatch: pytest.MonkeyPatch) -> list[MuSGD]:
    """Record the optimizer behind every call of the native preparation iterator.

    A step that replays a recording adds nothing; a native step adds one call, and a
    new recording adds two (its warmup and its capture).
    """
    calls: list[MuSGD] = []
    native = MuSGD._group_updates

    def counted(self: MuSGD, group: dict[str, object]) -> object:
        """Note the caller, then hand back the unchanged native iterator.

        Examples:
            >>> counted(None, {})  # needs the fixture's patched optimizer  # doctest: +SKIP
        """
        calls.append(self)
        return native(self, group)

    monkeypatch.setattr(MuSGD, "_group_updates", counted)
    return calls


class TestRecordingCheck:
    def test_accepts_independent_dense_float32(self) -> None:
        """The unchanged control set passes, so every refusal below is caused by its own change."""
        assert RecordingInputs.supported().accepted()

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(lambda x: x.with_tensor("gradients", 0, x.gradients[0].double()), id="float64-gradient"),
            pytest.param(lambda x: x.with_tensor("buffers", 1, x.buffers[1].half()), id="float16-buffer"),
            pytest.param(lambda x: x.with_tensor("params", 2, x.params[2].detach().double()), id="float64-parameter"),
            pytest.param(lambda x: x.with_tensor("gradients", 0, x.gradients[0].to("meta")), id="other-device"),
            pytest.param(lambda x: x.with_tensor("gradients", 0, x.gradients[0].to_sparse()), id="sparse-gradient"),
            pytest.param(
                lambda x: x.with_tensor("gradients", 0, x.gradients[0].to_sparse_csr()),
                id="compressed-sparse-gradient",
                marks=pytest.mark.filterwarnings("ignore:Sparse CSR tensor support is in beta"),
            ),
            pytest.param(
                lambda x: (
                    x.with_tensor("params", 1, torch.zeros(1)[1:])
                    .with_tensor("gradients", 1, torch.zeros(1)[1:])
                    .with_tensor("buffers", 1, torch.zeros(1)[1:])
                ),
                id="zero-size",
            ),
            pytest.param(lambda x: x.with_tensor("gradients", 0, x.gradients[0].t()), id="transposed-gradient"),
            pytest.param(lambda x: x.with_tensor("params", 0, x.params[0].detach().t()), id="transposed-parameter"),
            pytest.param(lambda x: x.with_tensor("gradients", 2, x.gradients[2].reshape(4, 8)), id="gradient-shape"),
            pytest.param(lambda x: x.with_tensor("buffers", 0, x.buffers[0].reshape(16)), id="buffer-shape"),
            pytest.param(lambda x: x.with_shared(("params", 0), ("params", 1)), id="parameters-share-storage"),
            pytest.param(lambda x: x.with_shared(("params", 0), ("buffers", 2)), id="parameter-shares-buffer-storage"),
            pytest.param(lambda x: x.with_shared(("buffers", 0), ("buffers", 1)), id="buffers-share-storage"),
            pytest.param(lambda x: x.with_shared(("params", 2), ("gradients", 0)), id="gradient-shares-storage"),
            pytest.param(lambda x: x.with_imported_overlap(), id="imported-gradient-overlaps-buffer"),
            pytest.param(lambda x: x.with_scalars(lr=True), id="bool-lr"),
            pytest.param(lambda x: x.with_scalars(momentum=np.float64(0.95)), id="numpy-momentum"),
            pytest.param(lambda x: x.with_scalars(weight_decay=torch.tensor(5e-4)), id="tensor-weight-decay"),
            pytest.param(lambda x: x.with_scalars(w_sgd=None), id="missing-gain"),
            pytest.param(lambda x: x.with_scalars(ns_steps=True), id="bool-ns-steps"),
        ],
    )
    def test_refuses_change(self, change: Callable[[RecordingInputs], RecordingInputs]) -> None:
        """Each input the recording cannot certify is refused before anything is captured.

        One change at a time is applied to the accepted control set: another precision
        or device, a sparse, empty, transposed or reshaped tensor, storage shared between
        parameters, buffers or gradients (also through a DLPack import with its own
        storage base), and group scalars of another type. Deleting any single guard of
        the recording check lets its case through.
        """
        assert not change(RecordingInputs.supported()).accepted()


class TestRecordedSpans:
    @pytest.mark.parametrize(
        ("probe", "expected"),
        [
            pytest.param(lambda base: torch.zeros(4, 4), False, id="other-allocation"),
            pytest.param(lambda base: base[16:32].view(4, 4), False, id="same-storage-gap"),
            pytest.param(lambda base: base[0:16].view(4, 4), True, id="recorded-bytes"),
            pytest.param(lambda base: base[8:24].view(4, 4), True, id="straddles-recorded-end"),
            pytest.param(lambda base: base[24:40].view(4, 4), True, id="straddles-recorded-start"),
            pytest.param(lambda base: base[20:36:2], True, id="strided-reach"),
            pytest.param(lambda base: torch.from_dlpack(base[4:20]).view(4, 4), True, id="imported-view"),
        ],
    )
    def test_gradient_reach_matches_memory_overlaps(self, probe: Callable[[Tensor], Tensor], expected: bool) -> None:
        """A moved gradient is caught by address range exactly when the full overlap scan would catch it.

        The spans are built with a gradient of the probe's layout at an unrelated address,
        as at recording, and then asked about the probe at its own address, as on a later
        replayed step. The recorded parameter and buffer sit in one buffer with a gap, so
        same-storage neighbours, partial overlaps at either end, a strided reach and a
        DLPack import with its own storage base are all decided by bytes, not storage.
        """
        base = torch.zeros(64)
        recorded = [base[0:16].view(4, 4), base[32:48]]
        gradient = probe(base)
        spans = _RecordedSpans(recorded, [torch.empty_strided(gradient.shape, gradient.stride())])
        assert spans.reached_by([gradient]) is expected
        assert _memory_overlaps([*recorded, gradient]) is expected

    def test_later_gradient_reach_is_caught(self) -> None:
        """Every gradient slot is checked, not only the first one.

        The first gradient is an unrelated allocation and the second reads the recorded
        parameter, so a check that stopped after one slot would replay a step whose
        native ordering differs.
        """
        base = torch.zeros(32)
        recorded = [base[0:16], base[16:32]]
        spans = _RecordedSpans(recorded, [torch.zeros(16), torch.zeros(16)])
        assert spans.reached_by([torch.zeros(16), base[0:16]])


class TestRecordingBudget:
    def test_reload_forgets_spent_budget(self) -> None:
        """Loading optimizer state lets a group whose re-record budget was spent record again.

        A reload replaces every parameter group, so recording counts kept for the old
        groups describe nothing that still exists. The CPU route creates the same owner
        and reaches the same ``load_state_dict`` clear as a CUDA run.
        """
        params = parameters("cpu")
        optimizer = MuSGD(params, lr=0.01, capture_updates=True)
        gradients(params, 0)
        optimizer.step()
        replay = optimizer._update_replay
        assert replay is not None
        replay.recordings[id(optimizer.param_groups[0])] = _RERECORD_BUDGET + 1
        optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        assert not replay.recordings


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="update replay needs CUDA")
class TestCUDARecordingLifecycle:
    def test_brief_changes_return_to_recording(self, native_calls: list[MuSGD]) -> None:
        """Signature changes shorter than the tolerance run natively, then replay the first recording.

        Momentum differs for one step fewer than ``_MISMATCH_TOLERANCE``, twice, with a
        single matching step between the two blips. Each differing step runs the native
        iterator once and leaves the recording in place; the matching step replays it
        and restarts the count, so the second blip does not inherit the first and no
        blip pays a capture.
        """
        a, b = parameters("cuda"), parameters("cuda")
        pairs = ((a, MuSGD(a, lr=0.01)), (b, MuSGD(b, lr=0.01, capture_updates=True)))
        blip = [0.8] * (_MISMATCH_TOLERANCE - 1)
        schedule = [0.95, 0.95, *blip, 0.95, *blip, 0.95]
        prepared = [advance(pairs, step, native_calls, momentum=value) for step, value in enumerate(schedule)]
        replay = pairs[1][1]._update_replay
        assert replay is not None
        native = [1] * (_MISMATCH_TOLERANCE - 1)
        assert prepared == [1, 2, *native, 0, *native, 0]
        assert replay.recordings == {id(pairs[1][1].param_groups[0]): 1}

    def test_lasting_change_records_again(self, native_calls: list[MuSGD]) -> None:
        """A change lasting the tolerance releases the old recording and replays a new one.

        Momentum changes for good after the first recording. The tolerated steps run
        natively; the next one records the new signature (warmup plus capture) and the
        old recording must be unreachable, so its private pool is free rather than held
        until a reload.
        """
        a, b = parameters("cuda"), parameters("cuda")
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        pairs = ((a, MuSGD(a, lr=0.01)), (b, candidate))
        prepared = [advance(pairs, step, native_calls, momentum=0.95) for step in range(2)]
        replay = candidate._update_replay
        assert replay is not None
        first = weakref.ref(replay.entries[id(candidate.param_groups[0])])
        prepared += [advance(pairs, step, native_calls, momentum=0.8) for step in range(2, 3 + _MISMATCH_TOLERANCE)]
        gc.collect()
        assert prepared == [1, 2, *[1] * (_MISMATCH_TOLERANCE - 1), 2, 0]
        assert first() is None
        assert replay.recordings == {id(candidate.param_groups[0]): 2}
        assert len(replay.entries) == 1

    def test_spent_budget_releases_recording(self, native_calls: list[MuSGD]) -> None:
        """A signature that never settles stops recording after the budget and frees the last recording.

        Momentum takes a new value on every step after the first recording, so each
        recording meets ``_MISMATCH_TOLERANCE`` consecutive mismatches. After
        ``_RERECORD_BUDGET`` replacements the next release leaves the group native for
        good: no entry, no further capture, and every step matches the native reference.
        """
        a, b = parameters("cuda"), parameters("cuda")
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        pairs = ((a, MuSGD(a, lr=0.01)), (b, candidate))
        changing = 2 + _RERECORD_BUDGET * _MISMATCH_TOLERANCE
        prepared = [advance(pairs, step, native_calls, momentum=0.95) for step in range(2)]
        prepared += [advance(pairs, step, native_calls, momentum=0.9 - 0.01 * step) for step in range(2, changing)]
        replay = candidate._update_replay
        assert replay is not None
        last = weakref.ref(replay.entries[id(candidate.param_groups[0])])
        prepared += [
            advance(pairs, step, native_calls, momentum=0.9 - 0.01 * step)
            for step in range(changing, changing + _MISMATCH_TOLERANCE + 1)
        ]
        gc.collect()
        cycle = [*[1] * (_MISMATCH_TOLERANCE - 1), 2]
        assert prepared == [1, 2, *cycle * _RERECORD_BUDGET, *[1] * (_MISMATCH_TOLERANCE + 1)]
        assert last() is None
        assert not replay.entries
        assert replay.recordings == {id(candidate.param_groups[0]): 1 + _RERECORD_BUDGET}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="update replay needs CUDA")
class TestCUDAReplayedStepChecks:
    def test_replay_skips_full_input_check(self, native_calls: list[MuSGD], monkeypatch: pytest.MonkeyPatch) -> None:
        """A step matching its recording replays without the full input scan.

        The scan sorts every parameter, gradient and buffer span; the key already fixes
        what it certified, so running it again on each replayed step only costs time.
        """
        a, b = parameters("cuda"), parameters("cuda")
        pairs = ((a, MuSGD(a, lr=0.01)), (b, MuSGD(b, lr=0.01, capture_updates=True)))
        prepared = [advance(pairs, step, native_calls) for step in range(2)]
        monkeypatch.setattr(_UpdateReplay, "_supported", lambda *args: pytest.fail("full input scan on replay"))
        prepared.append(advance(pairs, 2, native_calls))
        assert prepared == [1, 2, 0]

    def test_gradient_reaching_parameter_after_recording_runs_native(self, native_calls: list[MuSGD]) -> None:
        """A gradient that moves onto a parameter's bytes after recording takes the native route.

        The aliasing gradient keeps the recorded shape, strides and dtype, so the key
        still matches; only the per-step address check can keep that step native. The
        next step, with an independent gradient again, replays the same recording.
        """
        a = [torch.nn.Parameter(torch.arange(4, device="cuda", dtype=torch.float32)) for _ in range(2)]
        b = [torch.nn.Parameter(param.detach().clone()) for param in a]
        reference, candidate = MuSGD(a, lr=0.01), MuSGD(b, lr=0.01, capture_updates=True)
        pairs = ((a, reference), (b, candidate))
        prepared = [advance(pairs, step, native_calls) for step in range(2)]
        before = native_calls.count(candidate)
        for params, optimizer in pairs:
            gradients(params, 2)
            params[1].grad = params[0].detach()
            optimizer.step()
        assert_state(reference, candidate)
        prepared.append(native_calls.count(candidate) - before)
        prepared.append(advance(pairs, 3, native_calls))
        assert prepared == [1, 2, 1, 0]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="update replay needs CUDA")
class TestCUDAStepGuards:
    def test_autocast_keeps_native_step(self, native_calls: list[MuSGD]) -> None:
        """A step under CUDA autocast runs the native iterator and records nothing.

        Autocast changes the precision native preparation computes in, which a recording
        made outside it would not follow. Both optimizers take their first recordable
        step inside the same autocast scope, so their states agree byte for byte; the
        candidate must have prepared natively and kept no recording.
        """
        a, b = parameters("cuda"), parameters("cuda")
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        pairs = ((a, MuSGD(a, lr=0.01)), (b, candidate))
        prepared = [advance(pairs, 0, native_calls)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prepared.append(advance(pairs, 1, native_calls))
        assert prepared == [1, 1]
        assert candidate._update_replay is not None
        assert not candidate._update_replay.entries

    def test_enclosing_capture_keeps_native_step(self, native_calls: list[MuSGD]) -> None:
        """A step captured into a caller's CUDA graph runs the native iterator and records nothing.

        A recording started inside another capture would synchronize and allocate inside
        it. After one eager step initializes momentum, the candidate's next step is
        captured into an outer graph on a warmed side stream; replaying that graph once
        must match one native reference step, with no recording kept.
        """
        a, b, warm = parameters("cuda"), parameters("cuda"), parameters("cuda")
        reference, candidate, warmup = MuSGD(a, lr=0.01), MuSGD(b, lr=0.01, capture_updates=True), MuSGD(warm, lr=0.01)
        prepared = [advance(((a, reference), (b, candidate)), 0, native_calls)]
        gradients(a, 1)
        gradients(b, 1)
        reference.step()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gradients(warm, 0)
            warmup.step()
            gradients(warm, 1)
            warmup.step()
        outer = torch.cuda.CUDAGraph()
        before = native_calls.count(candidate)
        with torch.cuda.graph(outer, stream=stream):
            candidate.step()
        prepared.append(native_calls.count(candidate) - before)
        outer.replay()
        assert_state(reference, candidate)
        assert prepared == [1, 1]
        assert candidate._update_replay is not None
        assert not candidate._update_replay.entries


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="update replay needs CUDA")
class TestCUDARecordingKey:
    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(high_matmul_precision, id="matmul-precision"),
            pytest.param(deterministic_warn_only, id="deterministic-warn-only"),
            pytest.param(other_blas_library, id="blas-library"),
            pytest.param(side_stream, id="stream"),
        ],
    )
    def test_changed_runtime_setting_runs_native(
        self, change: Callable[[], AbstractContextManager[None]], native_calls: list[MuSGD]
    ) -> None:
        """A step under another arithmetic setting or stream prepares natively instead of replaying.

        Each setting is part of the recording key. Both optimizers take the step under
        the change, so their bytes agree with native arithmetic under it; matching bytes
        alone cannot tell a reused recording apart where the setting changes no result
        (no TF32 hardware, say), so the candidate must also have run the native iterator.
        """
        a, b = parameters("cuda"), parameters("cuda")
        pairs = ((a, MuSGD(a, lr=0.01)), (b, MuSGD(b, lr=0.01, capture_updates=True)))
        prepared = [advance(pairs, step, native_calls) for step in range(2)]
        with change():
            prepared.append(advance(pairs, 2, native_calls))
        assert prepared == [1, 2, 1]

    def test_fill_setting_keeps_replay(self, native_calls: list[MuSGD]) -> None:
        """Flipping the uninitialized-memory fill flag keeps replaying the recording.

        The flag only decides what new memory holds before it is written, which correct
        preparation never reads, so it is deliberately left out of the recording key.
        """
        a, b = parameters("cuda"), parameters("cuda")
        pairs = ((a, MuSGD(a, lr=0.01)), (b, MuSGD(b, lr=0.01, capture_updates=True)))
        prepared = [advance(pairs, step, native_calls) for step in range(2)]
        with flipped_fill():
            prepared.append(advance(pairs, 2, native_calls))
        assert prepared == [1, 2, 0]

    @pytest.mark.parametrize(
        ("name", "changed", "default"),
        [
            pytest.param("momentum", 0.8, 0.95, id="momentum"),
            pytest.param("weight_decay", 0.0, 5e-4, id="weight-decay"),
            pytest.param("w_muon", 0.25, 0.5, id="muon-gain"),
            pytest.param("w_sgd", 0.75, 0.5, id="sgd-gain"),
            pytest.param("ns_steps", 4, 5, id="newton-schulz-steps"),
        ],
    )
    def test_changed_scalar_runs_native_once(
        self, name: str, changed: float, default: float, native_calls: list[MuSGD]
    ) -> None:
        """A group scalar changed for one step prepares natively, then the recording replays again.

        Every scalar the recording bakes in is keyed by type and bits. The step after the
        change restores the default and must replay with no preparation.
        """
        a, b = parameters("cuda"), parameters("cuda")
        pairs = ((a, MuSGD(a, lr=0.01)), (b, MuSGD(b, lr=0.01, capture_updates=True)))
        prepared = [advance(pairs, step, native_calls) for step in range(2)]
        prepared.append(advance(pairs, 2, native_calls, **{name: changed}))
        prepared.append(advance(pairs, 3, native_calls, **{name: default}))
        assert prepared == [1, 2, 1, 0]

    def test_moved_parameter_runs_native_once(self, native_calls: list[MuSGD]) -> None:
        """A parameter whose storage moves for one step prepares natively, then replays once it is back.

        The recording reads and writes the parameter at its captured address. Assigning
        ``param.data`` to a copy keeps the parameter's identity but not that address.
        """
        a, b = parameters("cuda"), parameters("cuda")
        pairs = ((a, MuSGD(a, lr=0.01)), (b, MuSGD(b, lr=0.01, capture_updates=True)))
        prepared = [advance(pairs, step, native_calls) for step in range(2)]
        original = b[0].data
        b[0].data = original.clone()
        prepared.append(advance(pairs, 2, native_calls))
        original.copy_(b[0].data)
        b[0].data = original
        prepared.append(advance(pairs, 3, native_calls))
        assert prepared == [1, 2, 1, 0]

    def test_replaced_momentum_buffer_runs_native_once(self, native_calls: list[MuSGD]) -> None:
        """A momentum buffer replaced for one step prepares natively, then replays once it is back.

        The recording updates momentum in place at the captured buffer. A new buffer
        object, as a state load or a manual edit would leave, is a different address.
        """
        a, b = parameters("cuda"), parameters("cuda")
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        pairs = ((a, MuSGD(a, lr=0.01)), (b, candidate))
        prepared = [advance(pairs, step, native_calls) for step in range(2)]
        original = candidate.state[b[0]]["momentum_buffer"]
        candidate.state[b[0]]["momentum_buffer"] = original.clone()
        prepared.append(advance(pairs, 2, native_calls))
        original.copy_(candidate.state[b[0]]["momentum_buffer"])
        candidate.state[b[0]]["momentum_buffer"] = original
        prepared.append(advance(pairs, 3, native_calls))
        assert prepared == [1, 2, 1, 0]

    def test_changed_gradient_layout_runs_native_once(self, native_calls: list[MuSGD]) -> None:
        """A gradient arriving in another memory layout prepares natively, then replays with the usual layout.

        The recording copies each gradient into a private buffer with the recorded
        strides; a contiguous gradient for the channels-last kernel has other strides.
        """
        a, b = parameters("cuda"), parameters("cuda")
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        pairs = ((a, MuSGD(a, lr=0.01)), (b, candidate))
        prepared = [advance(pairs, step, native_calls) for step in range(2)]
        before = native_calls.count(candidate)
        for params, optimizer in pairs:
            gradients(params, 2)
            params[3].grad = params[3].grad.contiguous()
            optimizer.step()
        assert_state(pairs[0][1], candidate)
        prepared.append(native_calls.count(candidate) - before)
        prepared.append(advance(pairs, 3, native_calls))
        assert prepared == [1, 2, 1, 0]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="update replay needs CUDA")
class TestCUDAGradientInputs:
    def test_recorded_gradients_keep_caller_values(self) -> None:
        """The gradients a recording was made from keep their values while later steps replay.

        Each replay copies the new gradients into the recording's private inputs. Were
        those inputs the caller's step-1 tensors rather than copies, every later step
        would overwrite them while the optimizer's own bytes still matched native, so
        the retained step-1 tensors are compared with snapshots taken right after it.
        """
        a, b = parameters("cuda"), parameters("cuda")
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        pairs = ((a, MuSGD(a, lr=0.01)), (b, candidate))
        advance(pairs, 0, [])
        advance(pairs, 1, [])
        recorded = [param.grad for param in b if param.grad is not None]
        snapshots = [gradient.clone() for gradient in recorded]
        advance(pairs, 2, [])
        advance(pairs, 3, [])
        assert candidate._update_replay is not None
        assert len(candidate._update_replay.entries) == 1
        assert len(recorded) == len(b)
        for gradient, snapshot in zip(recorded, snapshots, strict=True):
            assert_bytes(gradient, snapshot)


_KERNEL_SHAPES = ((4, 4, 1, 1), (4, 1, 3, 3), (4, 4, 3, 3), (4, 4), (4,))


class TestParameterLayout:
    def test_native_updates_take_parameter_strides(self) -> None:
        """Native updates for channels-last ``1x1``, depthwise and full kernels leave with their parameter's strides.

        A ``1x1`` or depthwise update already counts as channels-last contiguous while
        its strides differ from the parameter's on size-one dimensions, so a memory
        format conversion leaves it untouched. The parameters are built as a converted
        convolution holds them, and the values must survive the restride unchanged.
        """
        params = parameters("cpu", _KERNEL_SHAPES)
        optimizer = MuSGD(params, lr=0.01)
        gradients(params, 0)
        optimizer.step()
        gradients(params, 1)
        with torch.no_grad():
            prepared = list(optimizer._group_updates(optimizer.param_groups[0]))
        assert len(prepared) == len(params)
        for param, update in prepared:
            restrided = _parameter_layout(param, update)
            assert restrided.stride() == param.stride()
            assert_bytes(restrided, update)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="update replay needs CUDA")
class TestCUDAParameterLayout:
    def test_recorded_updates_take_parameter_strides(self) -> None:
        """Every recorded update has its parameter's exact strides, so one foreach call writes them all.

        Counting foreach launches in a profiler cannot show this on PyTorch versions that
        ignore size-one strides, so the recorded pairs are compared directly; the replayed
        steps must still match the native reference.
        """
        a, b = parameters("cuda", _KERNEL_SHAPES), parameters("cuda", _KERNEL_SHAPES)
        candidate = MuSGD(b, lr=0.01, capture_updates=True)
        pairs = ((a, MuSGD(a, lr=0.01)), (b, candidate))
        for step in range(3):
            advance(pairs, step, [])
        assert candidate._update_replay is not None
        entry = candidate._update_replay.entries[id(candidate.param_groups[0])]
        assert len(entry.updates) == len(b)
        for param, update in zip(entry.parameters, entry.updates, strict=True):
            assert update.stride() == param.stride()
