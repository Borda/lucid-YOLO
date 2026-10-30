# SPDX-License-Identifier: Apache-2.0
"""CUDA assignment keeps values, ties, target ownership and native fallbacks."""

from __future__ import annotations

import math
from dataclasses import fields
from unittest.mock import Mock, patch

import pytest
import torch
from isolated_interpreter_helpers import run_isolated
from markers import requires_fused_cuda
from torch import Tensor

from lucid_yolo.assign import _cuda, tal
from lucid_yolo.assign.one_to_one import UniqueAssigner
from lucid_yolo.assign.stal import SmallTargetAssigner
from lucid_yolo.assign.tal import AssignResult, TaskAlignedAssigner
from lucid_yolo.losses.dual_loss import DualBranchLoss
from lucid_yolo.ptl.loss_capture import _source_signature
from lucid_yolo.ptl.module import DetectionLitModule


@pytest.fixture(autouse=True)
def _restore_backend(monkeypatch: pytest.MonkeyPatch):
    """Keep launch decisions and deterministic settings independent of test order."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setattr(_cuda, "_STATE", _cuda._BackendState())
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _bits(actual: Tensor, expected: Tensor) -> None:
    """Compare all stored bits, shapes and dtypes rather than rounded prints.

    Examples:
        >>> _bits(torch.tensor([1.0]), torch.tensor([1.0]))
    """
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.detach().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def _same(actual: AssignResult, expected: AssignResult) -> None:
    """Compare the independently specified five-field assignment contract.

    Examples:
        >>> value = TaskAlignedAssigner._empty_result(1, 2, torch.float32, torch.device("cpu"))
        >>> _same(value, value)
    """
    names = ("fg_mask", "gt_index", "target_labels", "target_boxes", "align_weights")
    assert tuple(field.name for field in fields(actual)) == names
    for name in names:
        _bits(getattr(actual, name), getattr(expected, name))


def _inputs(device: str = "cpu", count: int = 7, anchors: int = 129, dtype: torch.dtype = torch.float32):
    """Build an asymmetric two-image batch, including inverted predictions and padding.

    Examples:
        >>> _inputs(count=0)[3].shape
        torch.Size([2, 0, 4])
    """
    generator = torch.Generator(device=device).manual_seed(739 + count)
    scores = torch.rand((2, 3, anchors), generator=generator, device=device, dtype=dtype).transpose(1, 2)
    pred = torch.rand((2, anchors, 4), generator=generator, device=device) * 64
    points = torch.rand((anchors, 2), generator=generator, device=device) * 64
    gt = torch.rand((2, count, 4), generator=generator, device=device) * 24
    gt[..., 2:] += gt[..., :2] + 8
    labels = torch.arange(count, device=device).expand(2, -1).remainder(3).contiguous()
    mask = torch.ones((2, count), dtype=torch.bool, device=device)
    if count:
        labels[0, -1] = -1
        mask[0, -1] = False
    return scores, pred, points, gt, labels, mask


def _skewed_boxes(gt: Tensor, angle: float = 0.37) -> Tensor:
    """Turn each axis-aligned box into a rotated ``(cx, cy, w, h, angle)`` box with one shared angle.

    Examples:
        >>> _skewed_boxes(torch.tensor([[[0.0, 0.0, 4.0, 2.0]]])).tolist()[0][0][:4]
        [2.0, 1.0, 4.0, 2.0]
    """
    theta = torch.full((*gt.shape[:2], 1), angle, device=gt.device)
    return torch.cat(((gt[..., :2] + gt[..., 2:]) / 2, gt[..., 2:] - gt[..., :2], theta), -1)


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record the result of every ``_cuda._run`` call, to prove a captured graph holds fused kernels.

    A fallen-back native pass returns ``None`` under capture, so a list of non-``None`` results is the only direct
    evidence that the captured kernels are the fused ones, not a silent fallback.
    """
    results: list[object] = []
    original_run = _cuda._run

    def _spy(*args, **kwargs):
        result = original_run(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(_cuda, "_run", _spy)
    return results


class _Marked(Tensor):
    """A tensor subclass with no behaviour of its own, to meet the exact-type input guard."""


def _sweep_inputs(seed: int, batch: int, count: int, device: str = "cpu", anchors: int = 301):
    """Build seeded inputs whose scores, IoUs and edge contacts tie on purpose.

    Every value sits on a coarse grid, so equal alignment metrics, shared box edges and anchors
    exactly on a boundary occur often. The anchor count is not a multiple of any launch tile.

    Examples:
        >>> [tuple(value.shape) for value in _sweep_inputs(5, 2, 3)]
        [(2, 301, 3), (2, 301, 4), (301, 2), (2, 3, 4), (2, 3), (2, 3)]
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    def grid(*shape: int, scale: float, step: float) -> Tensor:
        return torch.rand(shape, generator=generator, device=device).mul(scale).div(step).round().mul(step)

    scores = grid(batch, anchors, 3, scale=1.0, step=0.25)
    pred = grid(batch, anchors, 4, scale=64.0, step=8.0)
    points = grid(anchors, 2, scale=64.0, step=4.0)
    gt = grid(batch, count, 4, scale=48.0, step=8.0)
    gt[..., 2:] += gt[..., :2] + 8.0
    mask = torch.rand((batch, count), generator=generator, device=device) > 0.2
    labels = torch.where(mask, torch.arange(count, device=device).remainder(3).expand(batch, -1), -1)
    return scores, pred, points, gt, labels, mask


def _right_angle_boxes(gt: Tensor) -> Tensor:
    """Rotate each axis-aligned box about its centre, cycling quarter turns of both signs and a skew.

    Examples:
        >>> [round(angle, 2) for angle in _right_angle_boxes(torch.zeros(1, 4, 4))[0, :, 4].tolist()]
        [1.57, -1.57, 0.0, 0.37]
    """
    angles = torch.tensor([math.pi / 2, -math.pi / 2, 0.0, 0.37], device=gt.device)
    theta = angles[torch.arange(gt.shape[1], device=gt.device) % 4].expand(gt.shape[0], -1).unsqueeze(-1)
    return torch.cat(((gt[..., :2] + gt[..., 2:]) / 2, gt[..., 2:] - gt[..., :2], theta), -1)


def _bits_except_nan_payload(actual: Tensor, expected: Tensor) -> None:
    """Compare bits everywhere but the payload of NaNs, whose positions must still agree.

    IEEE 754 leaves a NaN's sign and payload unspecified, so only its position is a contract.

    Examples:
        >>> _bits_except_nan_payload(torch.tensor([1.0, float("nan")]), torch.tensor([1.0, -float("nan")]))
    """
    assert torch.equal(actual.isnan(), expected.isnan())
    _bits(actual.masked_fill(actual.isnan(), 0.0), expected.masked_fill(expected.isnan(), 0.0))


@pytest.mark.parametrize("assigner", [TaskAlignedAssigner, SmallTargetAssigner, UniqueAssigner])
@pytest.mark.parametrize("count", [0, 1, 7])
def test_cpu_request_preserves_native_assignments(assigner, count: int) -> None:
    """The production constructor forwards the request while CPU retains every native field."""
    native, fused = assigner(topk=3), assigner(topk=3, fused_assignment=True)
    with patch.object(_cuda, "_run", side_effect=AssertionError("CPU launch")):
        _same(fused(*_inputs(count=count)), native(*_inputs(count=count)))
    assert not _cuda._requested()


@pytest.mark.parametrize("bad", [None, 1, "true"])
def test_request_requires_bool(bad) -> None:
    """A new keyword rejects non-boolean intent before any assignment is created."""
    with pytest.raises(TypeError, match="fused_assignment must be bool"):
        TaskAlignedAssigner(3, fused_assignment=bad)


def test_nested_scope_and_failure_restore_request() -> None:
    """A nested native request and an exception cannot leak an outer CUDA request."""
    with pytest.raises(RuntimeError, match="interrupted"), _cuda._assignment_scope(True):
        assert _cuda._requested()
        with _cuda._assignment_scope(False):
            assert not _cuda._requested()
        assert _cuda._requested()
        raise RuntimeError("interrupted")
    assert not _cuda._requested()


def test_module_and_criterion_forward_request() -> None:
    """The real model constructor owns the checkpointed option and both branch switches."""
    module = DetectionLitModule(0.5, 0.25, 64, 3, fused_assignment=True)
    assert module.hparams.fused_assignment is True
    assert module.loss.o2m_assigner.fused_assignment is True
    assert module.loss.o2o_assigner.fused_assignment is True
    assert DetectionLitModule(0.5, 0.25, 64, 3).hparams.fused_assignment is False
    source = DualBranchLoss()
    before = _source_signature(source)
    source.o2m_assigner.fused_assignment = True
    assert _source_signature(source) is not None and _source_signature(source) != before
    source.o2m_assigner.fused_assignment = 1
    assert _source_signature(source) is None


@pytest.mark.parametrize("helper", ["_cuda_metric", "_cuda_conflicts", "_assignment_scope"])
def test_replaced_recorded_helper_invalidates_source(monkeypatch: pytest.MonkeyPatch, helper: str) -> None:
    """A graph certificate includes each newly delegated operation, independently of its table."""
    assert _source_signature(DualBranchLoss()) is not None
    original = getattr(tal, helper)
    monkeypatch.setattr(tal, helper, Mock(wraps=original))
    assert _source_signature(DualBranchLoss()) is None


def test_replaced_assign_impl_invalidates_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """The implementation method belongs to the certificate: replacing it refuses the recording.

    Dropping the implementation method from the certificate makes this regression fail.
    """
    source = DualBranchLoss()
    assert _source_signature(source) is not None
    monkeypatch.setattr(source.o2m_assigner, "_assign_impl", lambda *args, **kwargs: None)
    assert _source_signature(source) is None


@pytest.mark.parametrize("failure", ["produce"])
def test_first_launch_failure_warns_once_and_stays_native(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """A partially prepared result is never published and a disabled family skips later work."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fence = Mock(side_effect=RuntimeError("fence failure") if failure == "fence" else None)
    monkeypatch.setattr(torch.cuda, "synchronize", fence)
    produce = Mock(
        side_effect=RuntimeError("compile failure") if failure == "produce" else None, return_value=torch.tensor([7.0])
    )
    with pytest.warns(UserWarning, match="failed its first launch"):
        assert _cuda._run("test", (), torch.device("cpu"), produce) is None
    assert _cuda._run("test", (), torch.device("cpu"), produce) is None
    assert produce.call_count == 1
    assert not _cuda._STATE.ready and not _cuda._STATE.verified


def test_verified_launch_error_propagates_and_unseen_capture_stays_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """Readiness is signature-specific; an execution error after verification is not hidden."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    device = torch.device("cpu")
    result = torch.tensor([3.0])
    assert _cuda._run("test", (1,), device, lambda: result) is result
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    never = Mock(side_effect=AssertionError("compile inside capture"))
    assert _cuda._run("test", (2,), device, never) is None
    assert never.call_count == 0
    with pytest.raises(RuntimeError, match="execution"):
        _cuda._run("test", (1,), device, Mock(side_effect=RuntimeError("execution")))


@requires_fused_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("assigner", [TaskAlignedAssigner, SmallTargetAssigner, UniqueAssigner])
@pytest.mark.parametrize("rotated", [False, True])
def test_cuda_assignments_match_native_fields(assigner, dtype: torch.dtype, rotated: bool) -> None:
    """All three assigners preserve their own ranking and all coupled targets on CUDA."""
    native, fused = assigner(topk=3), assigner(topk=3, fused_assignment=True)
    for count in (7, 1, 0, 9, 2):
        inputs = _inputs("cuda", count=count, dtype=dtype)
        boxes = _skewed_boxes(inputs[3]) if rotated else None
        expected = native(*inputs, boxes)
        actual = fused(*inputs, boxes)
        _same(actual, expected)
        assert all(not getattr(actual, field.name).requires_grad for field in fields(actual))
    assert {kind for kind, _ in _cuda._STATE.verified} >= {"metric", "conflicts"}
    assert ("inside" in {kind for kind, _ in _cuda._STATE.verified}) is rotated


@requires_fused_cuda
@pytest.mark.parametrize("seed", [5, 23])
@pytest.mark.parametrize("batch", [1, 2, 16])
@pytest.mark.parametrize("count", [63, 64, 65, 512, 513])
@pytest.mark.parametrize("rotated", [False, True])
def test_seeded_tie_sweep_matches_native_assignment_bits(seed: int, batch: int, count: int, rotated: bool) -> None:
    """Tied metrics, boundary anchors and padded rows keep native owners across reduction widths.

    Ground-truth counts straddle the 64-wide power-of-two row padding and the 512 capacity; the anchor count fits no
    launch tile. Rotated boxes cycle quarter turns of both signs and a skew.
    """
    values = _sweep_inputs(seed, batch, count, "cuda")
    boxes = _right_angle_boxes(values[3]) if rotated else None
    expected = TaskAlignedAssigner(3)(*values, boxes)
    actual = TaskAlignedAssigner(3, fused_assignment=True)(*values, boxes)
    _same(actual, expected)
    launched = {"metric"} | ({"conflicts"} if count <= 512 else set()) | ({"inside"} if rotated else set())
    assert {kind for kind, _ in _cuda._STATE.verified} == launched


@requires_fused_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_nonfinite_and_zero_extent_boxes_match_native_metric_bits(dtype: torch.dtype) -> None:
    """NaN, infinite and zero-extent corners keep native IoU and metric values and NaN positions."""
    nan, inf = float("nan"), float("inf")
    pred_rows = [[nan, 0, 1, 1], [0, 0, inf, inf], [-inf, -inf, inf, inf], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, nan]]
    gt_rows = [[0, 0, 2, 2], [nan, nan, nan, nan], [0, 0, inf, inf], [1, 1, 1, 1], [-inf, 0, 0, inf]]
    generator = torch.Generator(device="cuda").manual_seed(739)
    scores = torch.rand((2, 301, 3), generator=generator, device="cuda", dtype=dtype)
    pred = torch.tensor(pred_rows, device="cuda").repeat(51, 1)[:301].expand(2, -1, -1).contiguous()
    gt = torch.tensor(gt_rows, device="cuda").expand(2, -1, -1).contiguous()
    labels = torch.arange(5, device="cuda").remainder(3).expand(2, -1).contiguous()
    candidates = torch.ones((2, 5, 301), dtype=torch.bool, device="cuda")
    expected = TaskAlignedAssigner(3)._alignment_metric(scores, pred, gt, labels, candidates)
    with _cuda._assignment_scope(True), torch.no_grad():
        actual = _cuda._cuda_metric(scores, pred, gt, labels, candidates, 1.0, 6.0, 1e-9)
    assert actual is not None
    for value, oracle in zip(actual, expected, strict=True):
        _bits_except_nan_payload(value, oracle)


@requires_fused_cuda
@pytest.mark.parametrize("count,anchors", [(1, 17), (33, 129), (151, 21504), (512, 257), (3, 32769), (1, 131073)])
def test_cuda_conflict_reduction_keeps_native_bits_and_owned_output(count: int, anchors: int) -> None:
    """Tail tiles, variable row counts and maxima retain first owners and independent storage."""
    generator = torch.Generator(device="cuda").manual_seed(739)
    shape = 2, count, anchors
    mask = torch.rand(shape, generator=generator, device="cuda") > 0.97
    metric = torch.rand(shape, generator=generator, device="cuda")
    expected = TaskAlignedAssigner(3)._resolve_conflicts(mask, metric)
    with _cuda._assignment_scope(True), torch.no_grad():
        actual = _cuda._cuda_conflicts(mask, metric)
    assert actual is not None
    _bits(actual, expected)
    saved = actual.clone()
    mask.zero_()
    metric.zero_()
    _bits(actual, saved)


@requires_fused_cuda
def test_conflict_ties_nan_and_background_keep_first_owner() -> None:
    """A last-index tie policy or ignored selected NaN fails the independent index oracle."""
    mask = torch.tensor(
        [[[True, True, False, True], [True, True, False, True], [False, True, False, False]]], device="cuda"
    )
    metric = torch.tensor(
        [[[0.5, 0.5, 0.0, float("nan")], [0.5, float("nan"), 0.0, 0.9], [0.0, 0.9, 0.0, 0.0]]], device="cuda"
    )
    expected = torch.tensor(
        [[[True, False, False, True], [False, True, False, False], [False, False, False, False]]], device="cuda"
    )
    with _cuda._assignment_scope(True), torch.no_grad():
        actual = _cuda._cuda_conflicts(mask, metric)
    assert actual is not None
    _bits(actual, expected)


@requires_fused_cuda
def test_unsupported_layout_precision_and_capacity_decline() -> None:
    """Another dtype, negative view, non-contiguous target or oversized reduction stays native."""
    inputs = _inputs("cuda")
    s, p, points, gt, labels, _mask = inputs
    candidates = torch.ones((2, 7, 129), dtype=torch.bool, device="cuda")
    with _cuda._assignment_scope(True), torch.no_grad():
        assert _cuda._cuda_metric(s, p.double(), gt, labels, candidates, 1.0, 6.0, 1e-9) is None
        assert _cuda._cuda_metric(s, p, gt, labels, candidates, 0.5, 6.0, 1e-9) is None
        assert _cuda._cuda_inside(points, gt) is None
        neg = torch._neg_view(p)
        assert _cuda._cuda_metric(s, neg, gt, labels, candidates, 1.0, 6.0, 1e-9) is None
        large_mask = torch.zeros((1, 513, 3), dtype=torch.bool, device="cuda")
        assert _cuda._cuda_conflicts(large_mask, large_mask.float()) is None
    assert not _cuda._STATE.ready


@requires_fused_cuda
@pytest.mark.parametrize(
    "degrade",
    [
        pytest.param(lambda v: (v[0].as_subclass(_Marked), *v[1:]), id="tensor-subclass-scores"),
        pytest.param(lambda v: (*v[:4], v[4].int(), v[5]), id="int32-labels"),
        pytest.param(lambda v: (v[0], v[1].half(), *v[2:]), id="fp16-pred-boxes"),
        pytest.param(lambda v: (v[0], v[1].bfloat16(), *v[2:]), id="bf16-pred-boxes"),
        pytest.param(
            lambda v: (v[0], v[1].transpose(1, 2).contiguous().transpose(1, 2), *v[2:]), id="non-contiguous-pred-boxes"
        ),
        pytest.param(lambda v: (*v[:3], v[3][:, :0], v[4][:, :0], v[5][:, :0]), id="zero-targets"),
    ],
)
def test_unsupported_public_inputs_skip_metric_launch_and_keep_native_bits(
    monkeypatch: pytest.MonkeyPatch, degrade
) -> None:
    """An input the fused kernels cannot read exactly is assigned natively, with no metric launch.

    Each row degrades one public input of an otherwise launchable call: a Tensor subclass, 32-bit labels,
    half-precision or non-contiguous predicted boxes, or an empty target set.
    """
    values = degrade(_inputs("cuda"))
    expected = TaskAlignedAssigner(3)(*values)
    launched: list[str] = []
    original_run = _cuda._run

    def _spy(kind: str, *args, **kwargs):
        launched.append(kind)
        return original_run(kind, *args, **kwargs)

    monkeypatch.setattr(_cuda, "_run", _spy)
    actual = TaskAlignedAssigner(3, fused_assignment=True)(*values)
    _same(actual, expected)
    assert "metric" not in launched


@pytest.fixture
def _forget_device_support():
    """Drop the cached compute-capability verdict around a test that fakes the device's capability."""
    _cuda._device_supported.cache_clear()
    yield
    _cuda._device_supported.cache_clear()


@requires_fused_cuda
@pytest.mark.usefixtures("_forget_device_support")
@pytest.mark.parametrize(
    ("target", "value"),
    [
        pytest.param("torch.version.hip", "6.0", id="rocm-runtime"),
        pytest.param("torch.cuda.get_device_capability", lambda *args, **kwargs: (7, 5), id="pre-ampere-device"),
        pytest.param("lucid_yolo.assign._cuda._host_compiler_available", lambda: False, id="no-host-compiler"),
    ],
)
def test_unsupported_runtime_keeps_public_native_route(monkeypatch: pytest.MonkeyPatch, target: str, value) -> None:
    """A ROCm runtime, a pre-Ampere device or a missing host compiler never launches a fused kernel."""
    values = _inputs("cuda")
    expected = TaskAlignedAssigner(3)(*values)
    monkeypatch.setattr(target, value)
    with patch.object(_cuda, "_run", side_effect=AssertionError("unsupported runtime launched")):
        actual = TaskAlignedAssigner(3, fused_assignment=True)(*values)
    _same(actual, expected)
    assert not _cuda._STATE.ready


@requires_fused_cuda
@pytest.mark.parametrize(
    ("grad_mode", "degrade"),
    [
        pytest.param(True, lambda s, p, g, labels: (s, p.requires_grad_(), g, labels), id="requires-grad-in-grad-mode"),
        pytest.param(False, lambda s, p, g, labels: (s, p, g, labels.int()), id="int32-labels"),
    ],
)
def test_metric_guards_decline_without_launching(grad_mode: bool, degrade) -> None:
    """Autograd-tracked boxes and 32-bit labels are refused by the metric entry before any launch.

    The assigner itself runs without autograd, so the grad-mode guard is reachable only by calling the metric entry
    directly.
    """
    scores, pred, _points, gt, labels, _mask = _inputs("cuda")
    candidates = torch.ones((2, 7, 129), dtype=torch.bool, device="cuda")
    args = degrade(scores, pred, gt, labels)
    with (
        torch.set_grad_enabled(grad_mode),
        _cuda._assignment_scope(True),
        patch.object(_cuda, "_run", side_effect=AssertionError("guarded input launched")),
    ):
        assert _cuda._cuda_metric(*args, candidates, 1.0, 6.0, 1e-9) is None


@requires_fused_cuda
def test_graph_replays_new_values_after_eager_readiness(launched: list[object]) -> None:
    """A real recording reuses compiled code while reading the current scores and targets."""
    inputs = _inputs("cuda", count=3)
    source = TaskAlignedAssigner(3, fused_assignment=True)
    expected_source = TaskAlignedAssigner(3)
    source(*inputs)
    launched.clear()  # keep only the launches made during capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = source._assign(*inputs, check_labels=False)
    assert launched and all(result is not None for result in launched)
    inputs[0].mul_(0.5)
    inputs[3].add_(1.0)
    graph.replay()
    torch.cuda.synchronize()
    _same(actual, expected_source(*inputs))


@requires_fused_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_metric_and_rotated_edges_match_native_at_public_scale(dtype: torch.dtype) -> None:
    """Real anchor counts, nonstandard score strides and exact edges keep native bits."""
    scores, pred, _points, gt, labels, _mask = _inputs("cuda", count=151, anchors=21504, dtype=dtype)
    candidates = torch.ones((2, 151, 21504), dtype=torch.bool, device="cuda")
    # Degenerate and signed-zero boxes exercise clamps and division independently of random interiors.
    pred[:, :3] = torch.tensor([[0.0, -0.0, 0.0, -0.0], [2.0, 3.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]], device="cuda")
    gt[:, :3] = pred[:, :3]
    source = TaskAlignedAssigner(3)
    expected = source._alignment_metric(scores, pred, gt, labels, candidates)
    with _cuda._assignment_scope(True), torch.no_grad():
        actual = _cuda._cuda_metric(scores, pred, gt, labels, candidates, 1.0, 6.0, 1e-9)
    assert actual is not None
    for value, oracle in zip(actual, expected, strict=True):
        _bits(value, oracle)
    boxes = torch.tensor(
        [[[0.0, 0.0, 4.0, 2.0, 0.0], [1.0, -1.0, 0.0, 0.0, 1e8], [0.0, 0.0, 4.0, 2.0, -3.1415927]]], device="cuda"
    )
    edges = torch.tensor([[2.0, 1.0], [-2.0, -1.0], [2.000001, 1.0], [1.0, -1.0], [0.0, -0.0]], device="cuda")
    expected_inside = tal._rotated_inside(edges, boxes)
    with _cuda._assignment_scope(True), torch.no_grad():
        inside = _cuda._cuda_inside(edges, boxes)
    assert inside is not None
    _bits(inside, expected_inside)


@patch.object(_cuda, "_ENABLED", new=Mock(get=Mock(side_effect=AssertionError("traced context"))))
@patch.object(torch.compiler, "is_compiling", new=lambda: True)
def test_compiler_never_reads_scope_state() -> None:
    """Compiler tracing rejects the optional route before accessing context-local state."""
    assert not _cuda._requested()


@pytest.mark.parametrize("helper", ["_run", "_supported", "_dense_rows", "_requested", "_indexable"])
def test_replaced_backend_helper_invalidates_recording(monkeypatch: pytest.MonkeyPatch, helper: str) -> None:
    """A recorded fused operation cannot hide a changed backend implementation."""
    assert _source_signature(DualBranchLoss(fused_assignment=True)) is not None
    monkeypatch.setattr(_cuda, helper, Mock(wraps=getattr(_cuda, helper)))
    assert _source_signature(DualBranchLoss(fused_assignment=True)) is None


@pytest.mark.parametrize("kernel", ["_iou_pass", "_divide"])
def test_replaced_device_kernel_invalidates_recording(monkeypatch: pytest.MonkeyPatch, kernel: str) -> None:
    """Kernel identity, rather than only its Python launcher, belongs to the certificate."""
    if _cuda._kernels() is None:
        pytest.skip("optional compiler unavailable")
    assert _source_signature(DualBranchLoss(fused_assignment=True)) is not None
    monkeypatch.setattr(_cuda._kernels(), kernel, Mock())
    assert _source_signature(DualBranchLoss(fused_assignment=True)) is None


@requires_fused_cuda
def test_other_current_device_declines_before_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The allocator's current-device precondition is enforced by the public input gate."""
    values = _inputs("cuda")
    expected = TaskAlignedAssigner(3)(*values)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    with patch.object(_cuda, "_run", side_effect=AssertionError("wrong-device allocation")):
        _same(TaskAlignedAssigner(3, fused_assignment=True)(*values), expected)
    assert not _cuda._STATE.ready


@requires_fused_cuda
def test_side_stream_outputs_keep_current_stream_order() -> None:
    """Fresh independent output storage remains ordered on a non-default consumer stream."""
    values = _inputs("cuda")
    expected = TaskAlignedAssigner(3)(*values)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = TaskAlignedAssigner(3, fused_assignment=True)(*values)
    stream.synchronize()
    _same(actual, expected)
    assert {kind for kind, _ in _cuda._STATE.verified} == {"metric", "conflicts"}


def test_missing_optional_compiler_keeps_import_and_public_constructor() -> None:
    """A fresh interpreter without Triton still imports, constructs and assigns on CPU."""
    code = """
import sys
sys.modules["triton"] = None
import torch
from lucid_yolo.assign import _cuda
from lucid_yolo.assign.tal import TaskAlignedAssigner
from lucid_yolo.ptl.module import DetectionLitModule
assert _cuda._kernels() is None
import lucid_yolo.assign._cuda_kernels as optional_kernels
assert not optional_kernels._IS_TRITON_3_AVAILABLE
boxes = torch.tensor([[[0., 0., 2., 2.]]])
result = TaskAlignedAssigner(1, fused_assignment=True)(
    torch.tensor([[[.8]]]), boxes.clone(), torch.tensor([[1., 1.]]),
    boxes, torch.tensor([[0]]), torch.tensor([[True]]),
)
assert result.fg_mask.tolist() == [[True]] and result.target_labels.tolist() == [[0]]
assert torch.equal(result.align_weights, torch.ones(1, 1))
model = DetectionLitModule(.5, .25, 64, 3, fused_assignment=True)
assert model.loss.o2m_assigner.fused_assignment and model.loss.o2o_assigner.fused_assignment
assert not _cuda._STATE.verified
print("OPTIONAL_COMPILER_ABSENT_NATIVE_PASS")
"""
    assert "OPTIONAL_COMPILER_ABSENT_NATIVE_PASS" in run_isolated(code)


@pytest.mark.parametrize("shape,strides", [((1, 3, 1), (0, 2**30, 1)), ((1,), (2**31,))])
def test_wide_metadata_declines_32_bit_kernel_abi(shape, strides) -> None:
    """A metadata-only tensor proves both offset overflow and unused-wide-stride rejection."""
    assert _cuda._indexable(torch.empty((3,), device="meta"))
    assert not _cuda._indexable(torch.empty_strided(shape, strides, device="meta"))


@requires_fused_cuda
def test_wide_singleton_stride_declines_before_launch() -> None:
    """A real CUDA view needs no large allocation to exercise the runtime ABI boundary."""
    scores, pred, _points, gt, labels, _mask = _inputs("cuda", count=1, anchors=3)
    mask = torch.ones((2, 1, 3), dtype=torch.bool, device="cuda")
    # N=1 makes this stride unused by storage, but it would change the launch argument's width.
    mask = torch.as_strided(mask, (2, 1, 3), (3, 2**31, 1))
    assert mask.is_contiguous()
    with _cuda._assignment_scope(True), torch.no_grad(), patch.object(_cuda, "_run") as launch:
        assert _cuda._cuda_metric(scores, pred, gt, labels, mask, 1.0, 6.0, 1e-9) is None
    launch.assert_not_called()


@pytest.mark.parametrize("family", ["metric", "inside", "conflicts"])
@patch.object(_cuda, "_run")
@patch.object(_cuda, "_supported", new=lambda *args: True)
def test_oversized_y_grid_declines_before_verified_launcher(launch: Mock, family: str) -> None:
    """The host shape gate prevents invalid configurations rather than swallowing launch errors."""
    sentinel = launch.return_value

    def call(anchors: int):
        scores = torch.empty((2, anchors, 3), device="meta")
        pred = torch.empty((2, anchors, 4), device="meta")
        points = torch.empty((anchors, 2), device="meta")
        gt = torch.empty((2, 1, 4), device="meta")
        labels = torch.empty((2, 1), device="meta", dtype=torch.int64)
        mask = torch.ones((2, 1, anchors), device="meta", dtype=torch.bool)
        metric = torch.ones((2, 1, anchors), device="meta")
        with _cuda._assignment_scope(True):
            if family == "metric":
                return _cuda._cuda_metric(scores, pred, gt, labels, mask, 1.0, 6.0, 1e-9)
            if family == "inside":
                return _cuda._cuda_inside(points, torch.empty((2, 1, 5), device="meta"))
            return _cuda._cuda_conflicts(mask, metric)

    assert call(3) is sentinel
    launch.assert_called_once()
    launch.reset_mock()
    block = _cuda._BLOCK if family in ("metric", "inside") else _cuda._ANCHOR_BLOCK
    assert call(65535 * block + 1) is None
    launch.assert_not_called()


@requires_fused_cuda
def test_jit_trace_keeps_native_data_dependencies() -> None:
    """Tracing records tensor expressions instead of unobserved external kernel writes."""
    assigner = TaskAlignedAssigner(3, fused_assignment=True)
    values = _inputs("cuda", count=3, anchors=17)

    def forward(scores, predictions, boxes, labels, mask):
        with _cuda._assignment_scope(True):
            return assigner._alignment_metric(scores, predictions, boxes, labels, mask)

    metric_inputs = values[0], values[1], values[3], values[4], values[5].unsqueeze(-1).expand(-1, -1, 17)
    with patch.object(_cuda, "_run", side_effect=AssertionError("external launch during trace")):
        traced = torch.jit.trace(forward, metric_inputs, check_trace=False)
    assert not _cuda._STATE.verified
    changed = tuple(value.clone() for value in metric_inputs)
    changed[0].mul_(0.5)
    changed[2].add_(1.0)
    expected = TaskAlignedAssigner(3)._alignment_metric(*changed)
    for actual, reference in zip(traced(*changed), expected, strict=True):
        _bits(actual, reference)
    assigner(*values)
    assert {kind for kind, _ in _cuda._STATE.verified} == {"metric", "conflicts"}


@requires_fused_cuda
@pytest.mark.parametrize("eps", [1e-9, 0.5, 2.0**-149])
def test_subnormal_iou_division_keeps_native_bits(eps: float) -> None:
    """Tiny intersections, quotients and denominators cannot be silently flushed to zero."""
    side = 2.0**-70
    scores = torch.ones((1, 17, 3), device="cuda")
    boxes = torch.tensor([0.0, 0.0, side, side], device="cuda").expand(1, 17, 4).contiguous()
    gt = boxes[:, :1].contiguous()
    labels = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
    mask = torch.ones((1, 1, 17), dtype=torch.bool, device="cuda")
    expected = TaskAlignedAssigner(3, beta=1.0, eps=eps)._alignment_metric(scores, boxes, gt, labels, mask)
    assert bool((expected[1] > 0).all())
    with _cuda._assignment_scope(True), torch.no_grad():
        actual = _cuda._cuda_metric(scores, boxes, gt, labels, mask, 1.0, 1.0, eps)
    assert actual is not None and {kind for kind, _ in _cuda._STATE.verified} == {"metric"}
    for value, oracle in zip(actual, expected, strict=True):
        _bits(value, oracle)


@requires_fused_cuda
def test_graph_replays_unaligned_views_after_aligned_warmup(launched: list[object]) -> None:
    """New pointer alignment preserves current values inside a real CUDA recording."""
    values = _inputs("cuda", count=7)
    boxes = _skewed_boxes(values[3])
    source = TaskAlignedAssigner(3, fused_assignment=True)
    source(*values, boxes)
    ready = set(_cuda._STATE.ready)
    shifted = []
    for value in (*values, boxes):
        storage = torch.empty(value.numel() + 1, dtype=value.dtype, device="cuda")
        view = storage.as_strided(value.shape, value.stride(), storage_offset=1)
        view.copy_(value)
        assert view.storage_offset() == 1 and view.data_ptr() % 16 != 0
        shifted.append(view)
    launched.clear()  # keep only the launches made during capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = source._assign(*shifted, check_labels=False)
    # Same direct evidence as the aligned-warmup case: every launch during capture
    # returned a real result, so the kernels inside the graph are the fused ones.
    assert launched and all(result is not None for result in launched)
    shifted[0].mul_(0.5)
    shifted[3].add_(1.0)
    shifted[6][..., :2].add_(1.0)
    graph.replay()
    torch.cuda.synchronize()
    expected = TaskAlignedAssigner(3)(*shifted)
    assert bool(expected.fg_mask.any())
    _same(actual, expected)
    assert _cuda._STATE.ready == ready
    assert {kind for kind, _ in _cuda._STATE.verified} == {"metric", "inside", "conflicts"}


@requires_fused_cuda
@pytest.mark.parametrize("assigner", [TaskAlignedAssigner, SmallTargetAssigner, UniqueAssigner])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_public_signed_zero_weights_keep_native_bits(assigner, dtype: torch.dtype) -> None:
    """Zero-score classes preserve the native reduction's exact zero-weight sign bits."""
    patterns = torch.arange(64, device="cuda")[:, None].bitwise_right_shift(torch.arange(6, device="cuda"))
    signs = torch.where(patterns.bitwise_and(1).bool(), 0.0, -0.0).to(dtype)
    scores = signs.reshape(64, 2, 3).transpose(1, 2)
    pred = torch.tensor([0.0, 0.0, 1.0, 1.0], device="cuda").expand(64, 3, 4).contiguous()
    points = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], device="cuda")
    gt = pred[:, :2].contiguous()
    labels = torch.tensor([[0, 1]], device="cuda").expand(64, -1).contiguous()
    mask = torch.ones((64, 2), device="cuda", dtype=torch.bool)
    args = scores, pred, points, gt, labels, mask
    expected = assigner(3)(*args)
    actual = assigner(3, fused_assignment=True)(*args)
    assert not bool(expected.align_weights.count_nonzero())
    assert bool((expected.align_weights.view(torch.int32) < 0).any())
    _same(actual, expected)
