# SPDX-License-Identifier: Apache-2.0
"""Opt-in CUDA network replay preserves training state and has a bounded lifetime.

Real CUDA tests compare eager and replayed forwards, gradients and BatchNorm buffers
across optimizer updates. CPU tests cover the default, unsupported configurations and
checkpoint ownership without treating CPU execution as performance evidence.
"""

from __future__ import annotations

import copy
import pickle
import weakref
from collections.abc import Iterator
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
import torch
from packaging.version import Version
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.plugins.precision import MixedPrecision
from pytorch_lightning.strategies import SingleDeviceStrategy
from torch.utils.data import DataLoader

from lucid_yolo.cli.train import DetectionCLI, packaged_config
from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl import DetectionLitModule
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


def _run(
    holder: _NetworkCapture, model: DetectionLitModule, images: torch.Tensor, *, preserve_gradients: bool = False
) -> object:
    """Exercise the same stages and eager fallback the training objective uses.

    Examples:
        >>> holder = _NetworkCapture()
        >>> holder._disabled = True  # the retired path is also usable on CPU
        >>> result = _run(holder, _module(), torch.zeros(2, 3, 64, 64))
        >>> result.o2m_cls.shape[0]
        2
    """
    return holder.run(images, (model.backbone, model.neck, model.head), model, preserve_gradients=preserve_gradients)


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
    with pytest.raises(ValueError, match="requires task='detect' and compile_step=False"):
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


_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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
                        output = (
                            _run(holder, model, images, preserve_gradients=accumulation > 1)
                            if capture
                            else model(images)
                        )
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


def test_cli_accepts_network_capture_option() -> None:
    """The documented flag reaches the model through the public packaged recipe."""
    cli = DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        run=False,
        args=[
            "--config",
            str(packaged_config("det_nano_smoke")),
            "--model.capture_network",
            "true",
            "--trainer.accelerator",
            "cpu",
            "--trainer.logger",
            "false",
        ],
    )
    assert cli.model.hparams.capture_network is True
    assert isinstance(cli.model.configure_callbacks()[0], _NetworkCaptureLifecycle)


class _OverflowModule(DetectionLitModule):
    """Inject one nonfinite gradient without changing the detection network."""

    def training_step(self, batch: StepBatch, batch_idx: int) -> torch.Tensor:
        """Overflow the first microbatch; later batches must make finite updates."""
        loss = super().training_step(batch, batch_idx)
        return loss * float("inf") if batch_idx == 0 else loss


@pytest.mark.gpu
@_cuda
@pytest.mark.parametrize("accumulation", [1, 2])
def test_fp16_scaler_recovers_after_overflow(accumulation: int) -> None:
    """An overflow skips one update, then eager and replay recover identical optimizer state."""
    torch.manual_seed(1)
    eager = _OverflowModule(depth=0.34, width=0.25, max_channels=64, num_classes=4)
    replay = _OverflowModule(depth=0.34, width=0.25, max_channels=64, num_classes=4, capture_network=True)
    replay.load_state_dict(eager.state_dict())
    images = torch.rand(2, 3, 64, 64)
    targets = [Targets(boxes=torch.tensor([[8.0, 8.0, 40.0, 40.0]]), labels=torch.tensor([1])) for _ in range(2)]
    loader = DataLoader([(images, targets), (images.flip(-1), targets)] * 2, batch_size=None, num_workers=0)
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
