# SPDX-License-Identifier: Apache-2.0
"""Opt-in CUDA network replay preserves training state and has a bounded lifetime.

Real CUDA tests compare eager and replayed forwards, gradients and BatchNorm buffers
across optimizer updates. CPU tests cover the default, unsupported configurations and
checkpoint ownership without treating CPU execution as performance evidence.
"""

from __future__ import annotations

import copy
import gc
import pickle
import warnings
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest
import torch
from packaging.version import Version
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.plugins.precision import MixedPrecision
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy
from torch.utils.data import DataLoader

from lucid_yolo.cli.train import DetectionCLI, packaged_config
from lucid_yolo.data.targets import Targets
from lucid_yolo.optim.musgd import MuSGD
from lucid_yolo.ptl import DetectionLitModule, EMACallback
from lucid_yolo.ptl import network_capture as capture_module
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.module import StepBatch
from lucid_yolo.ptl.network_capture import _NetworkCapture, _NetworkCaptureLifecycle


@pytest.fixture(autouse=True)
def _deterministic_kernels() -> Iterator[None]:
    """Use the deterministic policy of the public-data trajectory comparison."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _module(**kwargs: object) -> DetectionLitModule:
    """Build a small real detection network with the normal dual head.

    Examples:
        >>> _module(capture_network=True).hparams.capture_network
        True
    """
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=4, **kwargs)


def _run(holder: _NetworkCapture, model: DetectionLitModule, images: torch.Tensor) -> object:
    """Exercise the same stages and eager fallback the training objective uses.

    Examples:
        >>> holder = _NetworkCapture()
        >>> holder._disabled = True  # the retired path is also usable on CPU
        >>> result = _run(holder, _module(), torch.zeros(2, 3, 64, 64))
        >>> result.o2m_cls.shape[0]
        2
    """
    return holder.run(images, (model.backbone, model.neck, model.head), model)


def test_default_has_no_callback_or_registered_state() -> None:
    """The off path retains the old checkpoint keys and never requests a recording."""
    baseline = _module()
    candidate = _module(capture_network=True)
    assert baseline.configure_callbacks() == []
    assert len(candidate.configure_callbacks()) == 1
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    assert candidate.hparams.capture_network is True
    with patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("CPU must stay eager")):
        with pytest.warns(UserWarning, match="single-device CUDA"):
            candidate.setup("fit")
        assert not candidate._capture_network_active


@pytest.mark.parametrize("kwargs", [{"task": "segment"}, {"compile_step": True}])
def test_incompatible_options_are_explicit(kwargs: dict[str, object]) -> None:
    """Unsupported tasks and the separate compiler path fail before training begins."""
    with pytest.raises(ValueError, match="requires task='detect' or task='keypoints', and compile_step=False"):
        _module(capture_network=True, **kwargs)


def test_copy_and_pickle_discard_transient_recordings() -> None:
    """Neither copying nor serialization retains graph ownership or disabled-fit state."""
    holder = _NetworkCapture()
    holder._disabled = True
    holder._input = ("old fit",)
    for restored in (copy.deepcopy(holder), pickle.loads(pickle.dumps(holder))):
        assert restored._network is None
        assert restored._input is None
        assert not restored._disabled


@pytest.mark.parametrize("hook", ["on_fit_start", "on_fit_end", "on_exception"])
def test_fit_boundaries_clear_recording(hook: str) -> None:
    """Success, failure and reuse of the same module all discard the previous recording."""
    model = _module(capture_network=True)
    model._network_capture._disabled = True
    model._network_capture._input = ("old fit",)
    args = (MagicMock(spec=Trainer), model)
    getattr(_NetworkCaptureLifecycle(), hook)(*args, *([RuntimeError("abort")] if hook == "on_exception" else []))
    assert model._network_capture._input is None
    assert not model._network_capture._disabled


@contextmanager
def _bf16_unsupported() -> Iterator[None]:
    """Report a runtime without verified BF16 replay.

    Examples:
        >>> with _bf16_unsupported():
        ...     capture_module._BF16_REPLAY_SUPPORTED
        False
    """
    with patch.object(capture_module, "_BF16_REPLAY_SUPPORTED", False):
        yield


@contextmanager
def _bf16_autocast_unsupported() -> Iterator[None]:
    """Report BF16 autocast on a runtime without verified BF16 replay.

    Examples:
        >>> with _bf16_autocast_unsupported():
        ...     torch.get_autocast_dtype("cuda")
        torch.bfloat16
    """
    with (
        _bf16_unsupported(),
        patch("torch.is_autocast_enabled", return_value=True),
        patch("torch.get_autocast_dtype", return_value=torch.bfloat16),
    ):
        yield


@pytest.mark.parametrize(
    ("context", "dtype", "requires_grad"),
    [
        pytest.param(_bf16_unsupported, torch.bfloat16, False, id="bf16-images-unsupported"),
        pytest.param(_bf16_autocast_unsupported, torch.float32, False, id="bf16-autocast-unsupported"),
        pytest.param(nullcontext, torch.float32, True, id="images-require-grad"),
        pytest.param(lambda: torch.autograd.set_detect_anomaly(True), torch.float32, False, id="anomaly-detection"),
        pytest.param(
            lambda: patch("torch.get_autocast_dtype", side_effect=RuntimeError("legacy query refused")),
            torch.float32,
            False,
            id="precision-query-refused",
        ),
        pytest.param(nullcontext, torch.float32, False, id="cpu-tensors"),
    ],
)
def test_cpu_call_outside_recording_contract_runs_eager_and_retires(
    context: Callable[[], AbstractContextManager[object]], dtype: torch.dtype, requires_grad: bool
) -> None:
    """Every CPU-reachable ineligible call returns the eager result, never records and retires the holder.

    The BF16 gate and the non-CUDA storage check are indistinguishable without matching reason text, so
    this pins what both promise a CPU-only merge gate: an unsupported call stays eager and builds no graph.
    """
    model = _module()
    holder = _NetworkCapture()
    images = torch.zeros(2, 3, 64, 64, dtype=dtype, requires_grad=requires_grad)
    sentinel = object()
    eager = MagicMock(return_value=sentinel)
    with (
        patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("ineligible call must not record")),
        context(),
        pytest.warns(UserWarning, match="stays eager"),
    ):
        result = holder.run(images, (model.backbone, model.neck, model.head), eager)
    assert result is sentinel
    assert eager.call_args.args[0] is images
    assert holder._disabled
    assert holder._network is None


def test_retirement_warns_once_and_later_calls_stay_eager() -> None:
    """After a retirement the rest of the fit is silent and eager: the user is told once, not per batch."""
    model = _module()
    holder = _NetworkCapture()
    images = torch.zeros(2, 3, 64, 64)
    stages = (model.backbone, model.neck, model.head)
    eager = MagicMock(return_value=object())
    with pytest.warns(UserWarning, match="stays eager"):
        holder.run(images, stages, eager)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        holder.run(images, stages, eager)
    assert eager.call_count == 2


def test_cpu_images_bypass_recording_when_capture_is_active() -> None:
    """A step on CPU images never enters the recorder even when the module believes capture is active."""
    model = _module(capture_network=True)
    model._capture_network_active = True
    model._trainer = MagicMock(spec=Trainer, training=True)
    images = torch.rand(2, 3, 64, 64)
    points, strides = model._anchor_grid(64, 64, images.device)
    with patch.object(model._network_capture, "run", side_effect=AssertionError("CPU images must stay eager")):
        head_out, _ = model._detect_objective(
            images,
            points,
            strides,
            torch.empty(2, 0, 4),
            torch.empty(2, 0, dtype=torch.long),
            torch.empty(2, 0, dtype=torch.bool),
            model._alpha_tensor(images.device),
        )
    assert head_out.o2m_cls.shape[0] == 2


@pytest.mark.parametrize(
    ("strategy", "automatic_optimization"),
    [
        pytest.param(DDPStrategy, True, id="distributed-strategy"),
        pytest.param(lambda: SingleDeviceStrategy(device="cuda"), False, id="manual-optimization"),
    ],
)
def test_setup_ignores_capture_outside_single_device_automatic_optimization(
    strategy: Callable[[], object], automatic_optimization: bool
) -> None:
    """Asking for capture under DDP or manual optimization warns and leaves the module eager."""
    model = _module(capture_network=True)
    model.automatic_optimization = automatic_optimization
    model._trainer = MagicMock(spec=Trainer, accelerator=CUDAAccelerator(), strategy=strategy())
    with pytest.warns(UserWarning, match="single-device CUDA"):
        model.setup("fit")
    assert not model._capture_network_active


def test_setup_activates_capture_on_single_device_cuda_automatic_optimization() -> None:
    """The supported trainer shape turns capture on without a warning (the module is not moved off CPU here)."""
    model = _module(capture_network=True)
    model._trainer = MagicMock(
        spec=Trainer, accelerator=CUDAAccelerator(), strategy=SingleDeviceStrategy(device="cuda")
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="capture_network=True is ignored")
        model.setup("fit")
    assert model._capture_network_active


def test_both_capture_flags_reach_optimizer_and_callbacks() -> None:
    """Network and orthogonalization replay are independent opt-ins: combining them drops neither."""
    model = _module(capture_network=True, capture_orthogonalization=True)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, MuSGD)
    assert optimizer.param_groups[0]["capture_orthogonalization"] is True
    assert any(isinstance(callback, _NetworkCaptureLifecycle) for callback in model.configure_callbacks())


_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _targets(count: int) -> Targets:
    """Build ``count`` nested boxes of class ids cycling through the four classes.

    Examples:
        >>> _targets(2).boxes.shape
        torch.Size([2, 4])
        >>> _targets(0).labels.shape
        torch.Size([0])
    """
    boxes = torch.tensor([[8.0 + i, 8.0 + i, 40.0 + i, 40.0 + i] for i in range(count)]).reshape(count, 4)
    return Targets(boxes=boxes, labels=torch.arange(count) % 4)


def _loader(batches: Sequence[Sequence[int]]) -> DataLoader[Any]:
    """Build a loader whose batches list one box count per image; an inner length is that batch's size.

    Examples:
        >>> loader = _loader([[1, 2], [0]])
        >>> [images.shape[0] for images, _ in loader]
        [2, 1]
    """
    items = [(torch.rand(len(counts), 3, 64, 64), [_targets(count) for count in counts]) for counts in batches]
    return DataLoader(items, batch_size=None, num_workers=0)


def _fit(
    model: DetectionLitModule, loader: DataLoader[Any], *, val: DataLoader[Any] | None = None, **kwargs: Any
) -> Trainer:
    """Fit ``model`` on one CUDA device with logging and checkpointing off unless ``kwargs`` override.

    Examples:
        >>> _fit(_module(), _loader([[1]]), accelerator="cpu").global_step
        1
    """
    options: dict[str, Any] = {
        "accelerator": "cuda",
        "devices": 1,
        "max_epochs": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
    }
    ckpt_path = kwargs.pop("ckpt_path", None)
    trainer = Trainer(**{**options, **kwargs})
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=val, ckpt_path=ckpt_path)
    return trainer


def _assert_identical(left: object, right: object) -> None:
    """Require bitwise-equal tensors and equal scalars through arbitrarily nested state.

    Examples:
        >>> _assert_identical({"a": [torch.ones(2)], "b": 1}, {"a": [torch.ones(2)], "b": 1})
    """
    torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.fixture
def replay_spy() -> Iterator[MagicMock]:
    """Count graph recordings and turn a retirement warning into a failure, so replay cannot silently lapse."""
    with (
        warnings.catch_warnings(),
        patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record,
    ):
        warnings.filterwarnings("error", message="capture_network=True stays eager")
        yield record


@pytest.mark.gpu
@_cuda
@pytest.mark.parametrize("dtype", [None, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("accumulation", [1, 2])
def test_cuda_replay_matches_eager_updates_and_buffers(dtype: torch.dtype | None, accumulation: int) -> None:
    """Recording warmup must not advance BatchNorm, RNG or the optimizer trajectory."""
    torch.manual_seed(1)
    eager = _module().cuda().to(memory_format=torch.channels_last)
    replay = copy.deepcopy(eager)
    holder = _NetworkCapture()
    eager_opt = torch.optim.SGD(eager.parameters(), lr=0.001)
    replay_opt = torch.optim.SGD(replay.parameters(), lr=0.001)
    with patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record:
        for _ in range(3):
            eager_opt.zero_grad(set_to_none=accumulation == 1)
            replay_opt.zero_grad(set_to_none=accumulation == 1)
            for _microbatch in range(accumulation):
                images = torch.randn(2, 3, 64, 64, device="cuda").contiguous(memory_format=torch.channels_last)
                for model, capture in ((eager, False), (replay, True)):
                    with torch.autocast("cuda", enabled=dtype is not None, dtype=dtype, cache_enabled=False):
                        output = _run(holder, model, images) if capture else model(images)
                        loss = (
                            sum(
                                t.float().square().mean()
                                for t in (output.o2m_cls, output.o2m_box, output.o2o_cls, output.o2o_box)
                            )
                            / accumulation
                        )
                    loss.backward()
            eager_opt.step()
            replay_opt.step()
            for name, tensor in eager.state_dict().items():
                assert torch.equal(tensor, replay.state_dict()[name]), name
            for left, right in zip(eager.parameters(), replay.parameters(), strict=True):
                assert torch.equal(left.grad, right.grad)
        expected = 0 if dtype == torch.bfloat16 and Version(torch.__version__) < Version("2.9") else 1
        assert record.call_count == expected
        record.reset_mock()  # the spy itself owns the recorded module through its call arguments
    recorded = weakref.ref(holder._network) if holder._network is not None else lambda: None
    holder.clear()
    assert holder._network is None
    assert recorded() is None


@pytest.mark.gpu
@_cuda
def test_partial_batch_stays_eager_and_reuses_single_recording() -> None:
    """A short last batch does not grow a cache or replace the full-batch graph."""
    model = _module().cuda()
    holder = _NetworkCapture()
    with patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record:
        for batch in (2, 1, 2):
            images = torch.randn(batch, 3, 64, 64, device="cuda")
            output = _run(holder, model, images)
            output.o2m_cls.sum().backward()
            model.zero_grad(set_to_none=True)
        assert record.call_count == 1
    holder.clear()


@pytest.mark.gpu
@_cuda
@pytest.mark.parametrize("change", ["hook", "storage", "precision", "batchnorm"])
def test_mutated_contract_retires_recording(change: str) -> None:
    """Changes hidden from a static graph take the ordinary forward for the rest of the fit."""
    model = _module().cuda()
    holder = _NetworkCapture()
    images = torch.randn(2, 3, 64, 64, device="cuda")
    _run(holder, model, images).o2m_cls.sum().backward()
    model.zero_grad(set_to_none=True)
    old_precision = torch.get_float32_matmul_precision()
    handles = []
    try:
        if change == "hook":
            handles.append(model.backbone.register_forward_hook(lambda _module, _inputs, output: output))
        elif change == "storage":
            param = next(model.parameters())
            param.data = param.detach().clone()
        elif change == "precision":
            torch.set_float32_matmul_precision("medium" if old_precision == "highest" else "highest")
        else:
            next(module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)).momentum = 0.2
        with pytest.warns(UserWarning, match="stays eager"):
            _run(holder, model, images).o2m_cls.sum().backward()
        assert holder._disabled
        assert holder._network is None
    finally:
        torch.set_float32_matmul_precision(old_precision)
        for handle in handles:
            handle.remove()
        holder.clear()


@pytest.mark.gpu
@_cuda
def test_failed_recording_restores_rng_and_buffers() -> None:
    """A failed CUDA setup cannot leave warmup's BatchNorm or RNG mutations behind."""
    model = _module().cuda()
    holder = _NetworkCapture()
    images = torch.randn(2, 3, 64, 64, device="cuda")
    buffers = {name: tensor.clone() for name, tensor in model.named_buffers()}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()

    def fail(network: torch.nn.Module, _inputs: object) -> None:
        network(images)
        torch.rand(1)
        torch.rand(1, device="cuda")
        raise RuntimeError("injected capture failure")

    with (
        patch("torch.cuda.make_graphed_callables", side_effect=fail),
        pytest.raises(RuntimeError, match="injected capture failure"),
    ):
        _run(holder, model, images)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    for name, tensor in model.named_buffers():
        assert torch.equal(tensor, buffers[name]), name
    assert holder._network is None


@pytest.mark.gpu
@_cuda
@pytest.mark.parametrize("custom", ["root_hook", "forward_override"])
def test_training_objective_preserves_root_customisations(custom: str) -> None:
    """The tensor-tuple view cannot bypass a hook or an overridden public forward."""
    model = _module(capture_network=True).cuda()
    model._trainer = MagicMock(
        spec=Trainer, accelerator=CUDAAccelerator(), strategy=SingleDeviceStrategy(device="cuda"), training=True
    )
    model.setup("fit")
    images = torch.randn(2, 3, 64, 64, device="cuda").contiguous(memory_format=torch.channels_last)
    points, strides = model._anchor_grid(64, 64, images.device)
    calls = MagicMock()
    if custom == "root_hook":
        handle = model.register_forward_hook(lambda *_args: calls())
        # Forward hooks returning None preserve the output.
        calls.return_value = None
    else:
        handle = None
        original = model.forward
        model.forward = MagicMock(side_effect=original)
    try:
        with pytest.warns(UserWarning, match="stays eager"):
            model._detect_objective(
                images,
                points,
                strides,
                torch.empty(2, 0, 4, device="cuda"),
                torch.empty(2, 0, dtype=torch.long, device="cuda"),
                torch.empty(2, 0, dtype=torch.bool, device="cuda"),
                model._alpha_tensor(images.device),
            )
        assert model._network_capture._disabled
        if custom == "root_hook":
            calls.assert_called_once()
        else:
            model.forward.assert_called_once()
    finally:
        if handle is not None:
            handle.remove()


@pytest.mark.gpu
@_cuda
def test_public_forward_never_records() -> None:
    """Prediction and direct callers keep ordinary tensor ownership even when the flag is on."""
    model = _module(capture_network=True).cuda()
    model._capture_network_active = True
    with patch.object(model._network_capture, "run", side_effect=AssertionError("public forward must stay eager")):
        output = model(torch.randn(2, 3, 64, 64, device="cuda"))
        assert output.o2m_cls.shape[0] == 2


@pytest.mark.gpu
@_cuda
def test_anomaly_detection_trainer_stays_eager() -> None:
    """The public debug flag keeps backward NaN checks outside CUDA recordings."""
    model = _module(capture_network=True)
    initial = next(model.parameters()).detach().clone()
    images = torch.rand(2, 3, 64, 64)
    targets = [Targets(boxes=torch.tensor([[8.0, 8.0, 40.0, 40.0]]), labels=torch.tensor([1])) for _ in range(2)]
    loader = DataLoader([(images, targets)] * 2, batch_size=None, num_workers=0)
    trainer = Trainer(
        accelerator="cuda",
        devices=1,
        detect_anomaly=True,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    with (
        patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("anomaly detection must stay eager")),
        pytest.warns(UserWarning, match="stays eager"),
    ):
        trainer.fit(model, train_dataloaders=loader)
    assert trainer.global_step == 2
    assert trainer.optimizers[0].state
    assert not torch.equal(initial, next(model.parameters()).detach().cpu())
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    assert model._network_capture._network is None


@pytest.mark.gpu
@_cuda
def test_anomaly_detection_retires_existing_recording() -> None:
    """Enabling debug checks after capture releases the graph before the next call."""
    model = _module().cuda()
    holder = _NetworkCapture()
    images = torch.randn(2, 3, 64, 64, device="cuda")
    with patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record:
        _run(holder, model, images).o2m_cls.sum().backward()
        model.zero_grad(set_to_none=True)
        with torch.autograd.set_detect_anomaly(True), pytest.warns(UserWarning, match="stays eager"):
            _run(holder, model, images).o2m_cls.sum().backward()
        assert record.call_count == 1
    assert holder._disabled
    assert holder._network is None
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


@pytest.mark.gpu
@_cuda
@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed"])
@pytest.mark.parametrize("accumulation", [1, 2])
def test_trainer_activates_once_per_fit_and_cleans_up(
    accumulation: int, precision: Literal["bf16-mixed", "16-mixed"]
) -> None:
    """The public automatic-optimisation loop records once, releases and records afresh on reuse.

    ``16-mixed`` exercises Lightning's GradScaler-wrapped backward, the precision the
    accepted detection recipe actually trains with (docs/model_cards/detection.md); the
    scaler lives outside the recorded backbone/neck/head region, so it only changes the
    magnitude of the incoming gradient the graph replays, never its shape or structure.
    """
    eager = _module()
    replay = _module(capture_network=True)
    replay.load_state_dict(eager.state_dict())
    images = torch.rand(2, 3, 64, 64)
    targets = [Targets(boxes=torch.tensor([[8.0, 8.0, 40.0, 40.0]]), labels=torch.tensor([1])) for _ in range(2)]
    loader = DataLoader([(images, targets), (images.flip(-1), targets)], batch_size=None, num_workers=0)
    with patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record:
        for fit_index in range(2):
            optimizers = []
            for model in (eager, replay):
                trainer = Trainer(
                    accelerator="cuda",
                    devices=1,
                    precision=precision,
                    accumulate_grad_batches=accumulation,
                    max_epochs=1,
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    num_sanity_val_steps=0,
                )
                trainer.fit(model, train_dataloaders=loader)
                assert trainer.global_step == 2 // accumulation
                optimizers.append(trainer.optimizers[0].state_dict()["state"])
            bf16_gated = precision == "bf16-mixed" and Version(torch.__version__) < Version("2.9")
            expected = 0 if bf16_gated else fit_index + 1
            assert record.call_count == expected
            assert replay._network_capture._network is None
            for name, tensor in eager.state_dict().items():
                assert torch.equal(tensor, replay.state_dict()[name]), name
            assert optimizers[0].keys() == optimizers[1].keys()
            for index, fields in optimizers[0].items():
                for name, tensor in fields.items():
                    actual = optimizers[1][index][name]
                    if isinstance(tensor, torch.Tensor):
                        assert torch.equal(tensor, actual), (index, name)
                    else:
                        assert tensor == actual


@pytest.mark.parametrize("capture_network", [False, True])
@pytest.mark.parametrize("capture_orthogonalization", [False, True])
def test_cli_accepts_network_capture_option(capture_network: bool, capture_orthogonalization: bool) -> None:
    """The two replay flags independently reach the model and optimizer through the public recipe."""
    cli = DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        run=False,
        args=[
            "--config",
            str(packaged_config("det_nano_smoke")),
            "--model.capture_network",
            str(capture_network).lower(),
            "--model.capture_orthogonalization",
            str(capture_orthogonalization).lower(),
            "--trainer.accelerator",
            "cpu",
            "--trainer.logger",
            "false",
        ],
    )
    assert cli.model.hparams.capture_network is capture_network
    assert cli.model.hparams.capture_orthogonalization is capture_orthogonalization
    callbacks = cli.model.configure_callbacks()
    if capture_network:
        assert isinstance(callbacks[0], _NetworkCaptureLifecycle)
    else:
        assert callbacks == []
    optimizer = cli.model.configure_optimizers()
    assert all(group["capture_orthogonalization"] is capture_orthogonalization for group in optimizer.param_groups)


class _OverflowModule(DetectionLitModule):
    """Inject one nonfinite gradient without changing the detection network."""

    def training_step(self, batch: StepBatch, batch_idx: int) -> torch.Tensor:
        """Overflow the first microbatch; later batches must make finite updates."""
        loss = super().training_step(batch, batch_idx)
        return loss * float("inf") if batch_idx == 0 else loss


@pytest.mark.gpu
@_cuda
@pytest.mark.parametrize("accumulation", [1, 2])
def test_fp16_scaler_recovers_after_overflow(accumulation: int, replay_spy: MagicMock) -> None:
    """An overflow skips one update, then eager and replay recover identical optimizer state.

    The batches carry 0, 1 and many ground-truth boxes per image, and one is empty throughout: the count
    varies per step while the recorded network input does not, which is why only the network replays.
    """
    torch.manual_seed(1)
    eager = _OverflowModule(depth=0.34, width=0.25, max_channels=64, num_classes=4)
    replay = _OverflowModule(depth=0.34, width=0.25, max_channels=64, num_classes=4, capture_network=True)
    replay.load_state_dict(eager.state_dict())
    loader = _loader([[0, 1], [1, 4], [0, 0], [4, 0]])
    states = []
    for model in (eager, replay):
        scaler = torch.amp.GradScaler("cuda", init_scale=1.0)
        trainer = Trainer(
            accelerator="cuda",
            devices=1,
            plugins=[MixedPrecision("16-mixed", "cuda", scaler=scaler)],
            accumulate_grad_batches=accumulation,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_dataloaders=loader)
        optimizer_state = trainer.optimizers[0].state_dict()["state"]
        assert optimizer_state  # at least one real update; skipping every step cannot pass
        assert scaler.get_scale() == 0.5  # exactly one controlled overflow from the unit initial scale
        states.append((optimizer_state, scaler.state_dict()))
    assert replay_spy.call_count == 1  # recorded once and never retired, so equality is not eager-vs-eager
    assert states[0][1] == states[1][1]
    assert states[0][0].keys() == states[1][0].keys()
    for index, fields in states[0][0].items():
        for name, expected in fields.items():
            actual = states[1][0][index][name]
            if isinstance(expected, torch.Tensor):
                assert torch.equal(expected, actual), (index, name)
            else:
                assert expected == actual
    for name, expected in eager.state_dict().items():
        assert torch.isfinite(expected).all(), name
        assert torch.equal(expected, replay.state_dict()[name]), name


@pytest.mark.gpu
@_cuda
def test_two_epoch_replay_with_validation_and_ema_matches_eager(replay_spy: MagicMock) -> None:
    """Train, validate under the EMA shadow, train again: replay stays engaged and tracks eager bit for bit.

    Validation swaps the shadow weights in and the live ones back in place, so the recording must survive
    the eval phase and the second epoch's training must still replay it instead of retiring.
    """
    eager = _module()
    replay = _module(capture_network=True)
    replay.load_state_dict(eager.state_dict())
    loader = _loader([[1, 3], [0, 2]])
    outcomes = []
    for model in (eager, replay):
        ema = EMACallback(decay=0.9, tau=1)
        trainer = _fit(model, loader, val=loader, max_epochs=2, callbacks=[ema])
        optimizer_state = trainer.optimizers[0].state_dict()
        outcomes.append((model.state_dict(), ema.state_dict(), optimizer_state, dict(trainer.callback_metrics)))
    assert replay_spy.call_count == 1
    _assert_identical(outcomes[0], outcomes[1])


@pytest.mark.gpu
@_cuda
def test_resume_from_checkpoint_with_replay_matches_eager(tmp_path: Path, replay_spy: MagicMock) -> None:
    """A fit resumed from a saved checkpoint records afresh and finishes where an eager resume does.

    The fresh module restores weights, BatchNorm statistics and optimizer momentum from the file, none of
    which may still alias a graph from the first fit.
    """
    initial = _module().state_dict()
    loader = _loader([[1, 2], [0, 3]])
    outcomes = []
    for capture in (False, True):
        first = _module(capture_network=capture)
        first.load_state_dict(initial)
        checkpoint = tmp_path / f"capture-{capture}.ckpt"
        _fit(first, loader).save_checkpoint(checkpoint)
        resumed = _module(capture_network=capture)
        trainer = _fit(resumed, loader, max_epochs=2, ckpt_path=checkpoint)
        outcomes.append((resumed.state_dict(), trainer.optimizers[0].state_dict()))
    assert replay_spy.call_count == 2  # one recording per fit of the replay twin
    _assert_identical(outcomes[0], outcomes[1])


@pytest.mark.gpu
@_cuda
def test_load_state_dict_assign_retires_recording_and_matches_eager_gradients() -> None:
    """Assigning checkpoint tensors after recording leaves the graph on stale storage, so it must retire.

    ``assign=True`` swaps every parameter for a new tensor; a surviving graph would keep reading and
    writing the old ones, leaving the live parameters without gradients.
    """
    eager = _module().cuda()
    replay = copy.deepcopy(eager)
    holder = _NetworkCapture()
    images = torch.randn(2, 3, 64, 64, device="cuda")
    eager(images).o2m_cls.sum().backward()
    _run(holder, replay, images).o2m_cls.sum().backward()
    for model in (eager, replay):
        model.zero_grad(set_to_none=True)
        model.load_state_dict(copy.deepcopy(eager.state_dict()), assign=True)
    eager(images).o2m_cls.sum().backward()
    with pytest.warns(UserWarning, match="stays eager"):
        _run(holder, replay, images).o2m_cls.sum().backward()
    assert holder._network is None
    assert all(parameter.grad is not None for parameter in replay.parameters())
    _assert_identical([p.grad for p in eager.parameters()], [p.grad for p in replay.parameters()])


@pytest.mark.gpu
@_cuda
@pytest.mark.parametrize("algorithm", ["norm", "value"])
def test_gradient_clipping_with_replay_matches_eager(
    algorithm: Literal["norm", "value"], replay_spy: MagicMock
) -> None:
    """Lightning clips the replayed gradients in place; the update must equal the eager clipped update."""
    eager = _module()
    replay = _module(capture_network=True)
    replay.load_state_dict(eager.state_dict())
    loader = _loader([[1, 2], [0, 3]])
    outcomes = []
    for model in (eager, replay):
        trainer = _fit(model, loader, gradient_clip_val=0.01, gradient_clip_algorithm=algorithm)
        outcomes.append((model.state_dict(), trainer.optimizers[0].state_dict()))
    assert replay_spy.call_count == 1
    _assert_identical(outcomes[0], outcomes[1])


@pytest.mark.gpu
@_cuda
@pytest.mark.parametrize(
    "window",
    [
        pytest.param([[1, 2], [3], [0, 1]], id="partial-between-full"),
        pytest.param([[3], [1, 2], [0, 1]], id="partial-first"),
        pytest.param([[1, 2], [0, 1], [3]], id="partial-last"),
    ],
)
def test_partial_batch_inside_accumulation_window_matches_eager(window: list[list[int]], replay_spy: MagicMock) -> None:
    """A short batch takes the eager route mid-window without corrupting the gradient already accumulated.

    The eager route skips the gradient clone a replayed batch makes, so the window's running gradient must
    still end up equal to the all-eager sum, wherever the short batch falls.
    """
    eager = _module()
    replay = _module(capture_network=True)
    replay.load_state_dict(eager.state_dict())
    loader = _loader([*window, [1, 2], [0, 1], [2, 3]])
    outcomes = []
    for model in (eager, replay):
        trainer = _fit(model, loader, accumulate_grad_batches=3)
        assert trainer.global_step == 2
        outcomes.append((model.state_dict(), trainer.optimizers[0].state_dict()))
    assert replay_spy.call_count == 1
    _assert_identical(outcomes[0], outcomes[1])


class _RaiseOnLaterBatch(DetectionLitModule):
    """Fail on a batch after the first, so a recording exists when the fit aborts."""

    def training_step(self, batch: StepBatch, batch_idx: int) -> torch.Tensor:
        """Run normally except for the third batch, which raises."""
        if batch_idx == 2:
            raise RuntimeError("injected later-batch failure")
        return super().training_step(batch, batch_idx)


class _RecordingProbe(Callback):
    """Keep only a weak reference to the recorded network, so the probe never extends its lifetime."""

    def __init__(self) -> None:
        self.network: weakref.ReferenceType[torch.nn.Module] | None = None

    def on_train_batch_end(self, trainer: Trainer, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Remember the recording once a training step has created it."""
        recorded = pl_module._network_capture._network
        if recorded is not None:
            self.network = weakref.ref(recorded)


#: Allocator rounding between two settled measurements; far below the pool of a recorded network pair.
_RESERVED_TOLERANCE = 8 * 2**20


def _settled_reserved() -> int:
    """Report the reserved CUDA bytes once garbage is collected and the caching allocator is emptied.

    Examples:
        >>> _settled_reserved() >= 0
        True
    """
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.memory_reserved()


def _fit_and_discard(
    module_class: type[DetectionLitModule],
    expectation: Callable[[], AbstractContextManager[object]],
    accelerator: str = "cuda",
) -> weakref.ReferenceType[torch.nn.Module] | None:
    """Fit a capture module inside ``expectation``, drop everything and return a weak ref to its recording.

    Examples:
        >>> _fit_and_discard(DetectionLitModule, nullcontext, accelerator="cpu") is None  # nothing records on CPU
        True
    """
    probe = _RecordingProbe()
    model = module_class(depth=0.34, width=0.25, max_channels=64, num_classes=4, capture_network=True)
    with expectation():
        _fit(model, _loader([[1, 2]] * 4), accelerator=accelerator, callbacks=[probe])
    return probe.network


@pytest.mark.gpu
@_cuda
@pytest.mark.parametrize(
    ("module_class", "expectation"),
    [
        pytest.param(DetectionLitModule, nullcontext, id="completed-fit"),
        pytest.param(
            _RaiseOnLaterBatch,
            lambda: pytest.raises(RuntimeError, match="injected later-batch"),
            id="exception-on-later-batch",
        ),
    ],
)
def test_fit_exit_releases_recorded_network_and_graph_memory(
    module_class: type[DetectionLitModule], expectation: Callable[[], AbstractContextManager[object]]
) -> None:
    """Completing or aborting a fit frees the recorded network and returns graph memory to the pre-fit level.

    A first identical fit settles allocator workspaces that outlive any fit, so the baseline measures only what
    a fit could leak: the weak reference must die and reserved bytes must not grow beyond allocator rounding.
    """
    _fit_and_discard(module_class, expectation)
    baseline = _settled_reserved()
    recorded = _fit_and_discard(module_class, expectation)
    reserved = _settled_reserved()
    assert recorded is not None  # a recording existed to be released
    assert recorded() is None
    assert reserved <= baseline + _RESERVED_TOLERANCE
