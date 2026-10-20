# SPDX-License-Identifier: Apache-2.0
"""Oriented replay retains angle fields, native gradients and validation transitions."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterator
from dataclasses import fields
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.plugins.precision import MixedPrecision
from torch.utils.data import DataLoader

from lucid_yolo.cli.train import DetectionCLI, packaged_config
from lucid_yolo.data.rotated_geom import rboxes_to_polygons
from lucid_yolo.data.targets import Targets
from lucid_yolo.models.heads.detect import DualHeadOutput
from lucid_yolo.ptl import DetectionLitModule, EMACallback
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.module import _oriented_replay_breach
from lucid_yolo.ptl.network_capture import _OrientedNetwork


@pytest.fixture(autouse=True)
def _deterministic_kernels() -> Iterator[None]:
    """Keep native comparisons deterministic and restore the caller's policy afterwards."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _oriented(**kwargs: Any) -> DetectionLitModule:
    """Build a small ordinary oriented model whose optimizer fits in a focused test.

    Examples:
        >>> _oriented().task
        'obb'
    """
    options = {"lr": 1e-4, "warmup_epochs": 0.0, "lrf": 1.0, "w_muon": 0.0, "w_sgd": 1.0}
    options.update(kwargs)
    return DetectionLitModule(0.34, 0.25, 64, 2, task="obb", **options)


def _same(left: Any, right: Any) -> None:
    """Compare tensor bytes and recursive training state, including signed zero.

    Examples:
        >>> _same({"counter": torch.tensor(2)}, {"counter": torch.tensor(2)})
    """
    if isinstance(left, torch.Tensor):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key, value in left.items():
            _same(value, right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _same(first, second)
    else:
        assert left == right


def _batches(layout: str) -> list[tuple[torch.Tensor, list[Targets]]]:
    """Keep full, empty and temporary partial batches in one native loader contract.

    Examples:
        >>> [len(targets) for _, targets in _batches("partial")]
        [2, 1, 2]
    """
    image = torch.rand(2, 3, 64, 64)
    rboxes = torch.tensor([[32.0, 32.0, 32.0, 12.0, 0.3]])
    corners = rboxes_to_polygons(rboxes)
    target = Targets(
        boxes=torch.cat((corners.amin(dim=1), corners.amax(dim=1)), dim=1),
        labels=torch.tensor([1]),
        rboxes=rboxes,
    )
    empty = Targets(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.int64), rboxes=torch.empty(0, 5))
    batches = [(image.clone(), [copy.deepcopy(target), copy.deepcopy(target)]) for _ in range(3)]
    if layout == "empty":
        batches[1] = (image.clone(), [copy.deepcopy(empty), copy.deepcopy(empty)])
    elif layout == "partial":
        batches[1] = (image[:1].clone(), [copy.deepcopy(target)])
    return batches


class _ReplayProbe(Callback):
    """Observe real graph launches and angle-gradient presence after every training batch."""

    def __init__(self, launches: list[int]) -> None:
        self.launches = launches
        self.deltas: list[int] = []
        self.angle_gradients: list[bool] = []
        self.losses: list[torch.Tensor] = []
        self.start = 0

    def on_train_batch_start(self, *args: Any, **kwargs: Any) -> None:
        """Mark the real replay count immediately before this batch's forward."""
        self.start = len(self.launches)

    def on_train_batch_end(self, trainer: Trainer, module: DetectionLitModule, outputs: Any, *args: Any) -> None:
        """Retain launched work, live gradients and loss bytes after backward and the optimizer."""
        self.deltas.append(len(self.launches) - self.start)
        angles = [parameter for name, parameter in module.named_parameters() if "angle_stems" in name]
        assert angles
        self.angle_gradients.append(all(parameter.grad is not None for parameter in angles))
        self.losses.append(outputs["loss"].detach().clone())


class _OverflowFirst(Callback):
    """Make one backward non-finite, then require the scaler to resume real updates."""

    def __init__(self) -> None:
        self.pending = True

    def on_before_backward(self, trainer: Trainer, module: DetectionLitModule, loss: torch.Tensor) -> None:
        """Inject through the public callback without changing which network outputs are consumed."""
        if self.pending:
            loss.mul_(float("inf"))
            self.pending = False


def test_oriented_adapter_preserves_fields() -> None:
    """The six-field adapter cannot swap the two angle branches or lose optional fields."""
    model = _oriented().eval()
    with torch.no_grad():
        assert model.head.o2o.angle_stems is not None
        for parameter in model.head.o2o.angle_stems.parameters():
            parameter.add_(0.1)
        image = torch.rand(2, 3, 64, 64)
        expected = model(image)
        network = _OrientedNetwork(model.backbone, model.neck, model.head)
        actual = network.rebuild(network(image))
    assert not torch.equal(expected.o2m_angle, expected.o2o_angle)
    for field in fields(DualHeadOutput):
        _same(getattr(expected, field.name), getattr(actual, field.name))


def test_oriented_default_and_cpu_fallback() -> None:
    """Enabling the existing flag changes no checkpoint keys and stays native without CUDA."""
    native, replayed = _oriented(), _oriented(capture_network=True)
    assert native.state_dict().keys() == replayed.state_dict().keys()
    assert not native.hparams.capture_network and replayed.hparams.capture_network
    assert _oriented_replay_breach(replayed) is None
    replayed.trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
    with pytest.warns(UserWarning, match="single-device CUDA"):
        replayed.setup("fit")
    assert not replayed._capture_network_active


def test_oriented_cli_accepts_network_replay() -> None:
    """The packaged oriented recipe builds the flag and keeps rotated-target transport enabled."""
    cli = DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        args=[
            "--config",
            str(packaged_config("obb_nano_smoke")),
            "--model.capture_network",
            "true",
            "--trainer.accelerator",
            "cpu",
            "--trainer.logger",
            "false",
        ],
        run=False,
    )
    assert cli.model.task == "obb" and cli.model.hparams.capture_network
    assert cli.datamodule._rotated_targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph replay")
@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
@pytest.mark.parametrize("layout", ["uniform", "empty", "partial"])
@pytest.mark.parametrize("accumulation", [1, 2])
def test_oriented_trainer_preserves_native_state(
    precision: str,
    layout: str,
    accumulation: int,
    graph_recordings: MagicMock,
    bf16_replay_expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual replay survives EMA validation and preserves empty-target gradients and temporary fallback."""
    torch.manual_seed(42)
    batches = _batches(layout)
    initial = _oriented().state_dict()
    real_replay = torch.cuda.CUDAGraph.replay
    launches: list[int] = []

    def observe(graph: torch.cuda.CUDAGraph) -> None:
        """Count actual graph execution while retaining the real runtime call."""
        launches.append(id(graph))
        real_replay(graph)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", observe)
    results = []
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        for capture in (False, True):
            torch.manual_seed(42)
            model = _oriented(capture_network=capture)
            model.load_state_dict(initial)
            probe, ema = _ReplayProbe(launches), EMACallback()
            callbacks: list[Callback] = [probe, ema]
            precision_options: dict[str, Any] = {"precision": precision}
            if precision == "16-mixed":
                # The native short fixture skips every update at the default scale.
                # Unit scale plus one explicit overflow proves both skipping and recovery.
                scaler = torch.amp.GradScaler("cuda", init_scale=1.0)
                precision_options = {"plugins": [MixedPrecision("16-mixed", "cuda", scaler=scaler)]}
                callbacks.append(_OverflowFirst())
            trainer = Trainer(
                accelerator="cuda",
                devices=1,
                max_epochs=2,
                gradient_clip_val=1.0,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
                deterministic=True,
                callbacks=callbacks,
                accumulate_grad_batches=accumulation,
                **precision_options,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                trainer.fit(
                    model,
                    train_dataloaders=DataLoader(copy.deepcopy(batches), batch_size=None),
                    val_dataloaders=DataLoader(copy.deepcopy(batches[:1]), batch_size=None),
                )
            recording = capture and accumulation == 1 and (precision != "bf16-mixed" or bf16_replay_expected)
            retired = [
                str(item.message) for item in caught if str(item.message).startswith("capture_network=True stays")
            ]
            assert len(retired) == int(capture and not recording), retired
            assert graph_recordings.call_count == int(recording)
            assert len(probe.deltas) == 6 and all(probe.angle_gradients)
            assert trainer.global_step == (3 + accumulation - 1) // accumulation * 2
            for index, delta in enumerate(probe.deltas):
                active = recording and not (layout == "partial" and index % 3 == 1)
                assert delta >= 2 if active else delta == 0
            assert trainer.optimizers[0].state
            assert model._network_capture._network is None
            scaler = getattr(trainer.precision_plugin, "scaler", None)
            if scaler is not None:
                assert scaler.get_scale() < 1.0
            results.append(
                copy.deepcopy(
                    (
                        model.state_dict(),
                        trainer.optimizers[0].state_dict(),
                        ema.state_dict(),
                        probe.losses,
                        None if scaler is None else scaler.state_dict(),
                        trainer.callback_metrics,
                    )
                )
            )
        _same(*results)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("method", ["training_step", "_task_extra_loss", "_branch_oriented_terms"])
def test_oriented_objective_override_stays_native(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed consumer cannot silently replace absent gradients with recorded zeros."""
    model = _oriented(capture_network=True)
    monkeypatch.setattr(model, method, lambda *args: torch.tensor(0.0))
    assert _oriented_replay_breach(model) == "the native oriented objective is overridden"


def test_oriented_classification_override_stays_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unchanged model does not certify a replaced classification consumer."""
    model = _oriented(capture_network=True)
    monkeypatch.setattr(model.loss._o2m_loss, "_classification_loss", lambda *args: torch.tensor(0.0))
    assert _oriented_replay_breach(model) == "the native oriented classification objective changed"


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("lucid_yolo.ptl.module.decode_ltrb", id="decode-ltrb"),
        pytest.param("lucid_yolo.ptl.module.decode_rboxes", id="decode-rboxes"),
        pytest.param("lucid_yolo.ptl.module.oriented_branch_terms", id="oriented-branch-terms"),
        pytest.param("lucid_yolo.losses.oriented_loss.square_angle_loss", id="square-angle-loss"),
        pytest.param("lucid_yolo.losses.oriented_loss._no_ground_truth", id="no-ground-truth"),
    ],
)
def test_oriented_helper_replacement_stays_native(target: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A swapped decode or loss helper retires replay, the oriented twin of WP-199's delegated-stage check."""
    model = _oriented(capture_network=True)
    monkeypatch.setattr(target, lambda *args, **kwargs: None)
    assert _oriented_replay_breach(model) == "a native oriented objective helper changed"


class _ClassificationOnly(DetectionLitModule):
    """A supported native customization whose objective leaves angle gradients absent."""

    def _task_extra_loss(self, context: Any, out: Any, stage: str) -> torch.Tensor:
        """Keep classification connected while declining oriented supervision."""
        return torch.zeros_like(out.total)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA training")
def test_oriented_override_keeps_absent_angle_gradients(graph_recordings: MagicMock) -> None:
    """The public trainer fallback skips angle state instead of materializing zero gradients."""
    batches = _batches("uniform")
    initial = _oriented().state_dict()
    results = []
    for capture in (False, True):
        model = _ClassificationOnly(
            0.34,
            0.25,
            64,
            2,
            task="obb",
            capture_network=capture,
            w_muon=0.0,
            w_sgd=1.0,
            lr=1e-4,
            warmup_epochs=0.0,
            lrf=1.0,
        )
        model.load_state_dict(initial)
        trainer = Trainer(
            accelerator="cuda",
            devices=1,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            gradient_clip_val=1.0,
            deterministic=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.fit(model, train_dataloaders=DataLoader(copy.deepcopy(batches), batch_size=None))
        retired = [str(item.message) for item in caught if str(item.message).startswith("capture_network=True stays")]
        assert len(retired) == int(capture)
        assert not graph_recordings.call_count
        angles = [parameter for name, parameter in model.named_parameters() if "angle_stems" in name]
        assert angles and all(parameter.grad is None for parameter in angles)
        assert all(parameter not in trainer.optimizers[0].state for parameter in angles)
        results.append(copy.deepcopy((model.state_dict(), trainer.optimizers[0].state_dict())))
    _same(*results)
