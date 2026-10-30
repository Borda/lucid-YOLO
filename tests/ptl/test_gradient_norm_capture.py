# SPDX-License-Identifier: Apache-2.0
"""Gradient-norm recording preserves native scaling and owns no checkpoint state."""

from __future__ import annotations

import copy
import re
import warnings
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from markers import NORM_REPLAY_EXPECTED, requires_cuda, skip_unless_norm_replay
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.plugins.precision import DoublePrecision, HalfPrecision, MixedPrecision, Precision
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.utils._pytree import tree_flatten
from torch.utils.data import DataLoader

from lucid_yolo.ptl import gradient_norm_capture as implementation
from lucid_yolo.ptl.gradient_norm_capture import _GradientNormCapture, _GradientNormLifecycle
from lucid_yolo.ptl.module import DetectionLitModule


def _parameters(device: str, dtype: torch.dtype) -> list[nn.Parameter]:
    """Create asymmetric gradients that exercise both clipping and an unchanged update.

    Examples:
        ```pycon
        >>> [tuple(p.shape) for p in _parameters("cpu", torch.float32)]
        [(3, 5), (7,)]

        ```
    """
    result = [
        nn.Parameter(torch.zeros(3, 5, device=device, dtype=dtype)),
        nn.Parameter(torch.zeros(7, device=device, dtype=dtype)),
    ]
    for index, parameter in enumerate(result):
        parameter.grad = torch.arange(parameter.numel(), device=device, dtype=dtype).reshape(parameter.shape) + index
    return result


def _bits_equal(first: list[nn.Parameter], second: list[nn.Parameter]) -> None:
    """Compare every gradient byte, without tolerances or rounded decimal prints.

    Examples:
        ```pycon
        >>> params = _parameters("cpu", torch.float32)
        >>> _bits_equal(params, params)

        ```
    """
    for a, b in zip(first, second, strict=True):
        assert a.grad is not None and b.grad is not None
        assert torch.equal(a.grad.contiguous().view(torch.uint8), b.grad.contiguous().view(torch.uint8))


def test_cpu_refusal_is_native_and_retirement_skips_scans() -> None:
    """An unsupported owner warns once and never scans gradients again during that fit."""
    first, second = _parameters("cpu", torch.float32), _parameters("cpu", torch.float32)
    owner = _GradientNormCapture()
    # Isolate the CPU-tensor diagnostic from the installed torch version: `run` checks
    # `_SUPPORTED` before the dense-CUDA-tensor check, and pyproject.toml's floor is
    # torch>=2.4, so a real pre-2.9 install would otherwise retire with the unrelated
    # version-verification reason and never reach the message this test asserts.
    with (
        patch("lucid_yolo.ptl.gradient_norm_capture._SUPPORTED", True),
        pytest.warns(UserWarning, match="ordinary dense.*CUDA"),
    ):
        owner.run(first, 2.0, lambda: torch.nn.utils.clip_grad_norm_(first, 2.0))
    torch.nn.utils.clip_grad_norm_(second, 2.0)
    _bits_equal(first, second)
    assert owner.retired and owner._graph is None
    with patch("lucid_yolo.ptl.gradient_norm_capture._SUPPORTED", False):
        owner.run(None, 2.0, lambda: None)
    owner.clear()
    assert not owner.retired and owner._graph is None and owner._inputs == []


@pytest.mark.parametrize(
    ("refusal", "reason"),
    [
        ("unsupported-torch", "the norm decomposition is verified only on torch>=2.9"),
        ("clip_grad_norm_", "a PyTorch clipping implementation was customized"),
        ("get_total_norm", "a PyTorch clipping implementation was customized"),
        ("clip_grads_with_norm_", "a PyTorch clipping implementation was customized"),
        ("no-gradients", "no parameter has a gradient"),
    ],
)
def test_norm_refusal_retires_and_runs_native(refusal: str, reason: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each refusal reaches its diagnostic on a fresh holder and preserves native clipping."""
    owner = _GradientNormCapture()
    first, second = _parameters("cpu", torch.float32), _parameters("cpu", torch.float32)
    monkeypatch.setattr(implementation, "_SUPPORTED", refusal != "unsupported-torch")
    if refusal == "no-gradients":
        for parameter in first + second:
            parameter.grad = None
    elif refusal != "unsupported-torch":
        original = getattr(torch.nn.utils, refusal, None)
        monkeypatch.setattr(torch.nn.utils, refusal, MagicMock(wraps=original), raising=False)
    eager = MagicMock(side_effect=lambda: torch.nn.utils.clip_grad_norm_(first, 2.0))

    with pytest.warns(UserWarning, match=re.escape(reason)) as recorded:
        owner.run(first, 2.0, eager)

    eager.assert_called_once_with()
    assert len(recorded) == 1
    assert owner.retired and owner._graph is None and owner._inputs == []
    torch.nn.utils.clip_grad_norm_(second, 2.0)
    if refusal == "no-gradients":
        assert all(parameter.grad is None for parameter in first)
    else:
        _bits_equal(first, second)


def test_copy_and_fit_boundaries_drop_norm_resources() -> None:
    """Reset, deep copy and failure release the actual holder, including a recorded signature."""
    owner = _GradientNormCapture()
    owner._inputs = [torch.ones(4)]
    owner._signature = ("old fit",)
    assert copy.deepcopy(owner)._inputs == []
    module = DetectionLitModule(0.34, 0.25, 64, 4, capture_gradient_norm=True)
    module._gradient_norm_capture = owner
    lifecycle = _GradientNormLifecycle()
    for event in ("on_fit_start", "on_fit_end", "on_exception"):
        owner._inputs = [torch.ones(4)]
        owner._signature = ("old fit",)
        args = (None, module, RuntimeError("failure")) if event == "on_exception" else (None, module)
        getattr(lifecycle, event)(*args)
        assert owner._inputs == [] and owner._signature is None


@pytest.mark.parametrize("supported", [False, True])
def test_new_option_preserves_default_state_and_cpu_setup(supported: bool) -> None:
    """The constructor adds no tensor keys and CPU setup keeps clipping native."""
    plain = DetectionLitModule(0.34, 0.25, 64, 4)
    enabled = DetectionLitModule(0.34, 0.25, 64, 4, capture_gradient_norm=True)
    assert plain.state_dict().keys() == enabled.state_dict().keys()
    assert not plain.hparams.capture_gradient_norm
    with (
        patch("lucid_yolo.ptl.module._NORM_CAPTURE_SUPPORTED", supported),
        pytest.warns(UserWarning, match="capture_gradient_norm=True is ignored outside single-device CUDA") as recorded,
    ):
        enabled.setup("fit")
    assert len(recorded) == 1 and recorded[0].filename == __file__
    assert not enabled._capture_gradient_norm_active
    assert any(isinstance(callback, _GradientNormLifecycle) for callback in enabled.configure_callbacks())


def test_failed_norm_cleanup_preserves_trainer_exception() -> None:
    """A cleanup error cannot replace the fit failure, even when warnings are errors."""
    module = DetectionLitModule(0.34, 0.25, 64, 4, capture_gradient_norm=True)
    lifecycle = _GradientNormLifecycle()
    with patch.object(module._gradient_norm_capture, "clear", side_effect=RuntimeError("cleanup failure")):
        with pytest.warns(UserWarning, match="capture_gradient_norm=True could not drain.*cleanup failure"):
            lifecycle.on_exception(None, module, RuntimeError("fit failure"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lifecycle.on_exception(None, module, RuntimeError("fit failure"))


@pytest.mark.parametrize(
    "plugin",
    [
        pytest.param(HalfPrecision("16-true"), id="half"),
        pytest.param(HalfPrecision("bf16-true"), id="bfloat"),
        pytest.param(DoublePrecision(), id="double"),
    ],
)
def test_inherited_stock_clipping_reaches_norm_holder(plugin: Precision) -> None:
    """Stock precision subclasses reach the holder's real CPU refusal and native scaling."""
    module = DetectionLitModule(0.34, 0.25, 64, 4, capture_gradient_norm=True)
    module._trainer = SimpleNamespace(precision_plugin=plugin, gradient_clip_val=2.0, gradient_clip_algorithm=None)
    module._capture_gradient_norm_active = True
    first, second = _parameters("cpu", torch.float32), _parameters("cpu", torch.float32)
    optimizer = torch.optim.SGD(first, lr=0.1)
    with patch.object(implementation, "_SUPPORTED", True), pytest.warns(UserWarning, match="ordinary dense.*CUDA"):
        module.configure_gradient_clipping(optimizer)
    torch.nn.utils.clip_grad_norm_(second, 2.0)
    _bits_equal(first, second)


@pytest.mark.parametrize(
    ("outcome", "maximum", "algorithm"),
    [
        ("unset", None, "norm"),
        ("zero", 0, "norm"),
        ("value", 2, "value"),
        ("amp-scaling", 2, "norm"),
        ("module-override", 2, "norm"),
        ("unknown-plugin", 2, "norm"),
        ("option-off", 2, "norm"),
        ("capture", 2, "norm"),
    ],
)
def test_clipping_routes_capture_and_preserves_native_gradients(
    outcome: str, maximum: int | None, algorithm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clipping routes preserve native results, retire once when required, and forward capture inputs."""
    enabled = outcome != "option-off"
    module = DetectionLitModule(0.34, 0.25, 64, 4, capture_gradient_norm=enabled)
    plugin = MagicMock(wraps=Precision()) if outcome == "unknown-plugin" else Precision()
    module._trainer = SimpleNamespace(
        precision_plugin=plugin, gradient_clip_val=maximum, gradient_clip_algorithm=algorithm
    )
    module._capture_gradient_norm_active = enabled
    first, second = _parameters("cpu", torch.float32), _parameters("cpu", torch.float32)
    optimizer = torch.optim.SGD(first, lr=0.1)
    native = MagicMock(wraps=LightningModule.configure_gradient_clipping.__get__(module))
    override = MagicMock(wraps=module.clip_gradients)
    if outcome == "amp-scaling":
        optimizer._step_supports_amp_scaling = True
    elif outcome == "module-override":
        monkeypatch.setattr(module, "clip_gradients", override)
    elif outcome == "option-off":
        monkeypatch.setattr(LightningModule, "configure_gradient_clipping", native)

    def capture(parameters: Iterable[torch.Tensor], limit: float, eager: Callable[[], None]) -> None:
        """Check the CPU forwarding seam, then apply native clipping through its callback."""
        assert all(
            actual is expected for actual, expected in zip(parameters, plugin.main_params(optimizer), strict=True)
        )
        assert type(limit) is float and limit == float(maximum)
        eager()

    run = MagicMock(side_effect=capture)
    monkeypatch.setattr(module._gradient_norm_capture, "run", run)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        module.configure_gradient_clipping(optimizer)
        module.configure_gradient_clipping(optimizer)
    retired = outcome in {"unset", "zero", "value", "module-override", "unknown-plugin"}
    assert module._gradient_norm_capture.retired is retired
    assert len(recorded) == int(retired)
    if retired:
        reason = (
            "customized clipping" if outcome in {"module-override", "unknown-plugin"} else "disabled or not by norm"
        )
        assert reason in str(recorded[0].message)
    assert run.call_count == (2 if outcome == "capture" else 0)
    assert native.call_args_list == ([call(optimizer, None, None)] * 2 if outcome == "option-off" else [])
    assert override.call_count == (2 if outcome == "module-override" else 0)
    for _ in range(2):
        if algorithm == "value":
            torch.nn.utils.clip_grad_value_(second, maximum)
        elif maximum:
            torch.nn.utils.clip_grad_norm_(second, maximum)
    _bits_equal(first, second)


def test_norm_replay_gate_tracks_torch_2_9() -> None:
    """Norm replay accepts only the independently specified verified torch 2.9 runtime."""
    assert implementation._SUPPORTED == NORM_REPLAY_EXPECTED


@requires_cuda
@skip_unless_norm_replay
@pytest.mark.parametrize("nonfinite", ["nan", "inf", "-inf"])
def test_cuda_nonfinite_norm_replay_matches_native(nonfinite: str) -> None:
    """Replaying fresh non-finite gradients preserves native NaN and infinity scaling semantics."""
    first, second = _parameters("cuda", torch.float32), _parameters("cuda", torch.float32)
    owner = _GradientNormCapture()
    owner.run(first, 2.0, lambda: pytest.fail("initial call stayed native"))
    graph = owner._graph
    assert graph is not None
    for params in (first, second):
        for index, parameter in enumerate(params):
            parameter.grad = torch.full_like(parameter, index + 1.0)
        params[0].grad[0, 0] = float(nonfinite)
    owner.run(first, 2.0, lambda: pytest.fail("non-finite call stayed native"))
    torch.nn.utils.clip_grad_norm_(second, 2.0)
    torch.cuda.synchronize()
    for actual, expected in zip(first, second, strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0, equal_nan=True)
    assert torch.isnan(first[0].grad).any()
    assert owner._graph is graph and not owner._disabled


@requires_cuda
@skip_unless_norm_replay
def test_cuda_clipping_hook_tracks_trainer_limit_changes() -> None:
    """Changing Trainer's clipping limit between hooks reuses the norm graph with current native scaling."""
    module = DetectionLitModule(0.34, 0.25, 64, 4, capture_gradient_norm=True)
    module._trainer = SimpleNamespace(precision_plugin=Precision(), gradient_clip_val=2.0, gradient_clip_algorithm=None)
    module._capture_gradient_norm_active = True
    first, second = _parameters("cuda", torch.float32), _parameters("cuda", torch.float32)
    optimizer = torch.optim.SGD(first, lr=0.1)
    graph = None
    for limit in (2.0, 200.0, 0.5):
        for params in (first, second):
            for index, parameter in enumerate(params):
                parameter.grad = torch.full_like(parameter, index + 1.0)
        module.trainer.gradient_clip_val = limit
        with patch.object(module._gradient_norm_capture, "retire", side_effect=AssertionError("unexpected retirement")):
            module.configure_gradient_clipping(optimizer)
        torch.nn.utils.clip_grad_norm_(second, limit)
        torch.cuda.synchronize()
        _bits_equal(first, second)
        assert module._gradient_norm_capture._graph is not None
        if graph is not None:
            assert module._gradient_norm_capture._graph is graph
        graph = module._gradient_norm_capture._graph


@requires_cuda
@skip_unless_norm_replay
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_cuda_norm_replay_and_changing_limits_are_byte_exact(dtype: torch.dtype) -> None:
    """The recorded route really runs; limits and fresh gradient storage remain live."""
    first, second = _parameters("cuda", dtype), _parameters("cuda", dtype)
    owner = _GradientNormCapture()
    graph = None
    for step, limit in enumerate((2.0, 200.0, 0.5, 4.0)):
        for params in (first, second):
            for index, parameter in enumerate(params):
                parameter.grad = (
                    torch.arange(parameter.numel(), device="cuda", dtype=dtype).reshape(parameter.shape) + step + index
                )
        with patch.object(owner, "retire", side_effect=AssertionError("unexpected retirement")):
            owner.run(first, limit, lambda: pytest.fail("eligible call stayed native"))
        torch.nn.utils.clip_grad_norm_(second, limit)
        torch.cuda.synchronize()
        _bits_equal(first, second)
        assert owner._graph is not None and not owner._disabled
        if graph is not None:
            assert owner._graph is graph
        graph = owner._graph
    owner.clear()
    assert owner._graph is None and owner._inputs == []


@requires_cuda
@skip_unless_norm_replay
def test_cuda_aliases_and_duplicate_parameters_keep_native_scaling() -> None:
    """Only the read-only reduction is captured, so native repeated alias writes remain observable."""
    first, second = _parameters("cuda", torch.float32), _parameters("cuda", torch.float32)
    first[1].grad = first[0].grad.flatten()[:7]
    second[1].grad = second[0].grad.flatten()[:7]
    left, right = [first[0], first[1], first[0]], [second[0], second[1], second[0]]
    owner = _GradientNormCapture()
    owner.run(left, 2.0, lambda: pytest.fail("eligible alias case stayed native"))
    torch.nn.utils.clip_grad_norm_(right, 2.0)
    torch.cuda.synchronize()
    _bits_equal(first, second)
    assert owner._graph is not None


@requires_cuda
@skip_unless_norm_replay
@pytest.mark.parametrize(
    "mutation", ["shape", "dtype", "none", "stream", "strided", "offset", "autocast", "autocast_dtype", "determinism"]
)
def test_cuda_drift_retires_real_owner_and_keeps_native(mutation: str) -> None:
    """Incompatible calls release a live recording and keep all subsequent calls native."""
    first = _parameters("cuda", torch.float32)
    owner = _GradientNormCapture()
    owner.run(first, 2.0, lambda: pytest.fail("initial call stayed native"))
    if mutation == "shape":
        first[0] = nn.Parameter(torch.zeros(2, 5, device="cuda"))
        first[0].grad = torch.ones_like(first[0])
    elif mutation == "dtype":
        first = _parameters("cuda", torch.float64)
    elif mutation == "none":
        first[0].grad = None
    elif mutation == "strided":
        first[0].grad = torch.ones(3, 10, device="cuda")[:, ::2]
    elif mutation == "offset":
        first[0].grad = torch.ones(16, device="cuda")[1:].reshape(3, 5)
    torch.cuda.synchronize()
    context = torch.cuda.stream(torch.cuda.Stream()) if mutation == "stream" else torch.no_grad()
    if mutation == "autocast":
        context = torch.autocast("cuda")
    elif mutation == "autocast_dtype":
        dtype = torch.float16 if torch.get_autocast_dtype("cuda") == torch.bfloat16 else torch.bfloat16
        context = torch.autocast("cuda", dtype=dtype, enabled=False)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if mutation == "determinism":
            torch.use_deterministic_algorithms(not enabled)
        with context, pytest.warns(UserWarning, match="capture_gradient_norm=True stays eager"):
            owner.run(first, 3.0, lambda: torch.nn.utils.clip_grad_norm_(first, 3.0))
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.cuda.synchronize()
    assert owner._disabled and owner._graph is None and owner._inputs == []
    owner.run(None, 3.0, lambda: None)


def test_custom_precision_clipping_retires_and_keeps_native_calls() -> None:
    """A real custom plugin remains observable, and retirement bypasses future certificates."""

    class CustomPrecision(Precision):
        def __init__(self) -> None:
            self.calls = 0

        def clip_grad_by_norm(self, optimizer: torch.optim.Optimizer, clip_val: float) -> None:
            self.calls += 1
            torch.nn.utils.clip_grad_norm_(self.main_params(optimizer), clip_val)

    module = DetectionLitModule(0.34, 0.25, 64, 4, capture_gradient_norm=True)
    plugin = CustomPrecision()
    module._trainer = SimpleNamespace(precision_plugin=plugin, gradient_clip_val=2.0, gradient_clip_algorithm=None)
    module._capture_gradient_norm_active = True
    parameters = _parameters("cpu", torch.float32)
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    with pytest.warns(UserWarning, match="customized clipping"):
        module.configure_gradient_clipping(optimizer, 2.0)
    assert plugin.calls == 1 and module._gradient_norm_capture._disabled
    with patch("lucid_yolo.ptl.module._NATIVE_PRECISION_CLIPPING", None):
        module.configure_gradient_clipping(optimizer, 2.0)
    assert plugin.calls == 2


def test_conflicting_trainer_and_argument_limits_raise_like_native() -> None:
    """A caller's explicit limit contradicting `Trainer(gradient_clip_val=...)` must still refuse.

    Dispatch only ever sees this exact combination through a direct call: the one real
    caller, the precision plugin's own hook dispatch, always passes the trainer's own value
    back in, so the two can never disagree there. A subclass or a test calling the hook
    directly with a different value must still get native's rejection, not a silently
    recorded clip at the wrong limit.
    """
    module = DetectionLitModule(0.34, 0.25, 64, 4, capture_gradient_norm=True)
    module._trainer = SimpleNamespace(precision_plugin=Precision(), gradient_clip_val=2.0, gradient_clip_algorithm=None)
    module._capture_gradient_norm_active = True
    parameters = _parameters("cpu", torch.float32)
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    with (
        patch.object(module._gradient_norm_capture, "run", side_effect=AssertionError("must not record")) as run_mock,
        pytest.raises(MisconfigurationException, match="Please use only one of them"),
    ):
        module.configure_gradient_clipping(optimizer, 1.0)
    assert not run_mock.called


def test_older_runtime_keeps_new_option_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new flag can be loaded on older runtimes without touching the newer norm API."""
    monkeypatch.setattr("lucid_yolo.ptl.module._NORM_CAPTURE_SUPPORTED", False)
    module = DetectionLitModule(0.34, 0.25, 64, 4, capture_gradient_norm=True)
    module._trainer = SimpleNamespace(accelerator=CUDAAccelerator(), strategy=SingleDeviceStrategy(device="cuda"))
    with pytest.warns(UserWarning, match="verified only on torch>=2.9") as recorded:
        module.setup("fit")
    assert len(recorded) == 1 and torch.__version__ in str(recorded[0].message)
    assert recorded[0].filename == __file__
    assert not module._capture_gradient_norm_active
    old = dict(module.hparams)
    old.pop("capture_gradient_norm")
    restored = DetectionLitModule(**old)
    assert not restored.hparams.capture_gradient_norm


class _OverflowTrainingModule(DetectionLitModule):
    """Use bounded linear gradients to isolate the norm/AMP hook from network replay."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(0.34, 0.25, 64, 4, **kwargs)
        self.probe = nn.Linear(3, 2)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Inject one overflow; finite linear batches then exercise actual clipping updates."""
        loss = self.probe(batch).square().sum()
        return loss * float("inf") if batch_idx == 0 else loss


@requires_cuda
@skip_unless_norm_replay
@pytest.mark.parametrize("accumulation", [1, 2])
def test_fp16_fit_unscales_before_norm_replay_and_recovers_from_overflow(accumulation: int) -> None:
    """Actual Lightning clipping, AMP skip/recovery and fit cleanup preserve every state byte."""

    class Observe(Callback):
        def __init__(self) -> None:
            self.replays = 0

        def on_train_batch_end(
            self, trainer: object, module: DetectionLitModule, outputs: object, batch: object, index: int
        ) -> None:
            del trainer, outputs, batch, index
            self.replays += int(module._gradient_norm_capture._graph is not None)

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    try:
        torch.manual_seed(3)
        plain = _OverflowTrainingModule()
        replay = _OverflowTrainingModule(capture_gradient_norm=True)
        replay.load_state_dict(plain.state_dict())
        items = [torch.ones(2, 3) * (index + 1) for index in range(4)]
        states = []
        for model in (plain, replay):
            observe = Observe()
            scaler = torch.amp.GradScaler("cuda", init_scale=1.0)
            trainer = Trainer(
                accelerator="cuda",
                devices=1,
                plugins=[MixedPrecision("16-mixed", "cuda", scaler=scaler)],
                max_epochs=1,
                accumulate_grad_batches=accumulation,
                gradient_clip_val=2.0,
                deterministic=True,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
                callbacks=[observe],
            )
            with patch.object(
                model._gradient_norm_capture, "run", wraps=model._gradient_norm_capture.run
            ) as norm_calls:
                trainer.fit(model, train_dataloaders=DataLoader(items, batch_size=None, num_workers=0))
            assert norm_calls.call_count == (4 // accumulation if model is replay else 0)
            assert scaler.get_scale() == 0.5
            assert trainer.optimizers[0].state_dict()["state"]
            assert observe.replays >= 2 if model is replay else observe.replays == 0
            assert model._gradient_norm_capture._graph is None and model._gradient_norm_capture._inputs == []
            states.append((model.state_dict(), trainer.optimizers[0].state_dict(), scaler.state_dict()))
        left, left_spec = tree_flatten(states[0])
        right, right_spec = tree_flatten(states[1])
        assert left_spec == right_spec
        for a, b in zip(left, right, strict=True):
            if isinstance(a, torch.Tensor):
                assert torch.equal(
                    a.detach().reshape(-1).contiguous().view(torch.uint8),
                    b.detach().reshape(-1).contiguous().view(torch.uint8),
                )
            else:
                assert a == b
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, deterministic


@requires_cuda
@skip_unless_norm_replay
@pytest.mark.parametrize("failure", ["warmup", "capture"])
def test_failed_recording_orders_private_reads_and_leaves_live_gradients_untouched(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queued warmup read is fenced even when recording fails before ownership is committed."""
    parameters = _parameters("cuda", torch.float32)
    before = [parameter.grad.clone() for parameter in parameters]
    native_norm = torch.nn.utils.get_total_norm
    native_wait = torch.cuda.Stream.wait_stream
    waits = []
    calls = 0
    caller = torch.cuda.current_stream().cuda_stream

    def fail_norm(inputs: list[torch.Tensor]) -> torch.Tensor:
        nonlocal calls
        calls += 1
        result = native_norm(inputs)
        if calls == (1 if failure == "warmup" else 2):
            raise RuntimeError("controlled norm recording failure")
        return result

    def observe_wait(stream: torch.cuda.Stream, other: torch.cuda.Stream) -> None:
        waits.append((stream.cuda_stream, other.cuda_stream))
        native_wait(stream, other)

    monkeypatch.setattr(torch.nn.utils, "get_total_norm", fail_norm)
    monkeypatch.setattr(implementation, "_GET_NORM", fail_norm)
    monkeypatch.setattr(torch.cuda.Stream, "wait_stream", observe_wait)
    owner = _GradientNormCapture()
    with pytest.raises(RuntimeError, match="controlled norm recording failure"):
        owner.run(parameters, 2.0, lambda: pytest.fail("eligible call stayed native"))
    torch.cuda.synchronize()
    assert len(waits) >= 2 and waits[0][1] == caller and waits[1] == (caller, waits[0][0])
    assert owner._graph is None and owner._inputs == [] and owner._signature is None
    for parameter, expected in zip(parameters, before, strict=True):
        assert torch.equal(parameter.grad.view(torch.uint8), expected.view(torch.uint8))


def test_public_trainer_reuse_ignores_foreign_model_missing_norm_holder() -> None:
    """A model-contributed callback survives Trainer reuse and must not mask a later fit's exception."""

    class Finite(_OverflowTrainingModule):
        def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
            """Keep this first fit finite; its norm option deliberately takes the CPU fallback."""
            del batch_idx
            return self.probe(batch).square().sum()

    class Foreign(LightningModule):
        def __init__(self, fail: bool = False) -> None:
            super().__init__()
            self.probe = nn.Linear(3, 2)
            self.fail, self.calls = fail, 0

        def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
            """Make the next model's actual update, or its original controlled failure, observable."""
            del batch_idx
            self.calls += 1
            if self.fail:
                raise RuntimeError("foreign fit failure")
            return self.probe(batch).square().sum()

        def configure_optimizers(self) -> torch.optim.Optimizer:
            """Use the dependency's basic optimizer without a norm holder."""
            return torch.optim.SGD(self.parameters(), lr=0.01)

    loader = DataLoader([torch.ones(2, 3)], batch_size=None, num_workers=0)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        gradient_clip_val=2.0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    with pytest.warns(UserWarning, match="capture_gradient_norm=True is ignored outside single-device CUDA"):
        trainer.fit(Finite(capture_gradient_norm=True), train_dataloaders=loader)
    assert any(isinstance(callback, _GradientNormLifecycle) for callback in trainer.callbacks)
    first_steps = trainer.global_step
    trainer.fit_loop.max_epochs = 2
    foreign = Foreign()
    trainer.fit(foreign, train_dataloaders=loader)
    assert foreign.calls > 0 and trainer.global_step > first_steps
    assert not hasattr(foreign, "_gradient_norm_capture")
    trainer.fit_loop.max_epochs = 3
    with pytest.raises(RuntimeError, match="foreign fit failure"):
        trainer.fit(Foreign(fail=True), train_dataloaders=loader)
