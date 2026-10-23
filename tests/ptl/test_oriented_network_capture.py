# SPDX-License-Identifier: Apache-2.0
"""Oriented replay retains angle fields, native gradients and validation transitions."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from dataclasses import fields
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from markers import requires_cuda
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.plugins.precision import MixedPrecision
from torch.utils.data import DataLoader

from lucid_yolo.cli.train import DetectionCLI, packaged_config
from lucid_yolo.data.rotated_geom import rboxes_to_polygons
from lucid_yolo.data.targets import Targets
from lucid_yolo.models.heads.detect import DualHeadOutput
from lucid_yolo.ptl import DetectionLitModule, EMACallback
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.loss_capture import _source_signature
from lucid_yolo.ptl.module import _oriented_replay_breach
from lucid_yolo.ptl.network_capture import _OrientedNetwork


@pytest.fixture(autouse=True)
def _deterministic(deterministic_kernels: None) -> None:
    """Keep native comparisons deterministic and restore the caller's policy afterwards."""


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


@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
def test_oriented_default_and_cpu_fallback(stage: str) -> None:
    """Enabling the existing flag changes no checkpoint keys and stays native without CUDA.

    Every Lightning stage sets the module up, and only ``fit`` trains, so only ``fit``
    names the ignored flag: an evaluation run reporting a replay it never would use is
    noise, and a CPU setup that turned recording on would be a silent regression. The
    recorder entry point raises if anything reaches it, so the CPU setup is proven inert.
    """
    native, replayed = _oriented(), _oriented(capture_network=True)
    assert native.state_dict().keys() == replayed.state_dict().keys()
    assert not native.hparams.capture_network and replayed.hparams.capture_network
    assert _oriented_replay_breach(replayed) is None
    replayed.trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
    with (
        patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("CPU must stay eager")),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        replayed.setup(stage)
    ignored = [str(item.message) for item in caught if str(item.message).startswith("capture_network=True is ignored")]
    assert len(ignored) == int(stage == "fit"), ignored
    assert not replayed._capture_network_active


def test_oriented_cli_accepts_network_replay() -> None:
    """The packaged oriented recipe builds the flag, keeps rotated-target transport and passes the replay guard.

    A shipped setting that tripped the guard would leave every user of the recipe on the
    eager path with only a retirement warning to show for it.
    """
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
    assert _oriented_replay_breach(cli.model) is None


@requires_cuda
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
    """Actual replay survives EMA validation and preserves empty-target gradients and temporary fallback.

    Both accumulation settings record and replay; accumulation 2 also carries live gradients
    into each second replay. A replaying arm that never records names exactly one cause, the
    precision policy, so a retirement for an unrelated cause cannot satisfy the count while
    hiding the regression behind it.
    """
    torch.manual_seed(42)
    batches = _batches(layout)
    retirements = ["BF16 replay requires"] if precision == "bf16-mixed" and not bf16_replay_expected else []
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
            # A decaying LR keeps a real scheduler on both arms, so its state joins the oracle.
            model = _oriented(capture_network=capture, lrf=0.1)
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
            recording = capture and (precision != "bf16-mixed" or bf16_replay_expected)
            retired = [
                str(item.message) for item in caught if str(item.message).startswith("capture_network=True stays")
            ]
            assert len(retired) == int(capture and not recording), retired
            expected = retirements if capture else []
            assert all(fragment in message for fragment, message in zip(expected, retired, strict=True)), retired
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
                        trainer.lr_scheduler_configs[0].scheduler.state_dict(),
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


@pytest.mark.parametrize(
    "method", ["training_step", "_shared_step", "_task_extra_loss", "_oriented_extra_loss", "_branch_oriented_terms"]
)
def test_oriented_objective_override_stays_native(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed consumer cannot silently replace absent gradients with recorded zeros."""
    model = _oriented(capture_network=True)
    monkeypatch.setattr(model, method, lambda *args: torch.tensor(0.0))
    assert _oriented_replay_breach(model) == "the native oriented objective is overridden"


def test_oriented_classification_override_stays_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unchanged model does not certify a replaced classification consumer."""
    model = _oriented(capture_network=True)
    monkeypatch.setattr(model.loss._o2m_loss, "_classification_loss", lambda *args: torch.tensor(0.0))
    assert _oriented_replay_breach(model) == "the native loss or an assignment implementation changed"


def test_oriented_loss_setting_change_keeps_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scalar loss setting outside the network recording leaves oriented replay eligible.

    The oriented loss always runs eagerly after the recorded network, so an assigner epsilon
    the detection loss recording refuses cannot change which outputs carry gradients. Both
    halves are asserted: the loss certificate still refuses the value, and the network
    guard does not inherit that refusal.
    """
    model = _oriented(capture_network=True)
    monkeypatch.setattr(model.loss.o2m_assigner, "eps", 1e-40)
    assert _source_signature(model.loss) is None
    assert _oriented_replay_breach(model) is None


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


class _CudaFlagged(torch.Tensor):
    """CPU tensor that reports CUDA residency so the CUDA-gated oriented routing is reachable offline.

    Operations unwrap to plain tensors, so the real model still runs on the CPU. A copy of
    the pose suite's class: ``tests/`` is not a package, so one test module cannot import
    another's helpers.
    """

    is_cuda = True

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        """Run the operation on the underlying data and return an ordinary tensor."""
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **(kwargs or {}))


def _routed_oriented(accumulation: int = 1) -> DetectionLitModule:
    """Build an oriented module that satisfies every training-time gate of the full-network route.

    Examples:
        >>> _routed_oriented(2).trainer.accumulate_grad_batches
        2
    """
    model = _oriented(capture_network=True)
    model._trainer = SimpleNamespace(training=True, accumulate_grad_batches=accumulation)
    model._capture_network_active = True
    return model


def _cuda_flagged_images() -> torch.Tensor:
    """Images that pass the ``is_cuda`` gate while living on the CPU.

    Examples:
        >>> _cuda_flagged_images().is_cuda
        True
    """
    return torch.rand(2, 3, 64, 64).as_subclass(_CudaFlagged)


def _override_objective(model: DetectionLitModule) -> None:
    """Replace one native oriented consumer on this instance, as a user customization would.

    Examples:
        >>> model = _oriented()
        >>> _override_objective(model)
        >>> "_task_extra_loss" in vars(model)
        True
    """
    model._task_extra_loss = lambda *args: torch.tensor(0.0)  # type: ignore[method-assign]


def test_oriented_routing_dispatches_full_network_to_recorder() -> None:
    """An unbroken oriented module hands all three stages to the oriented recorder and returns its output.

    The CUDA-only route cannot execute on CPU runners, so images spoof ``is_cuda`` and the
    recorder is replaced: what is pinned is the dispatch itself, the oriented entry point
    with the complete stage tuple and the module as its eager fallback, never the
    detection route, and no retirement warning.
    """
    model = _routed_oriented()
    images = _cuda_flagged_images()
    replayed = object()
    with (
        patch.object(model._network_capture, "run_oriented", return_value=replayed) as oriented,
        patch.object(model._network_capture, "run") as detect,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        output = model._training_network(images)
    oriented.assert_called_once_with(images, (model.backbone, model.neck, model.head), model)
    detect.assert_not_called()
    assert output is replayed


def test_oriented_routing_leaves_cpu_images_native() -> None:
    """Images that are not on CUDA never reach the oriented recorder, however many gates pass.

    Every other training-time gate is satisfied, so the image residency check alone
    keeps the route native: the recorder is never called, no retirement is warned, and
    all six fields equal the module's own eager forward byte for byte.
    """
    model = _routed_oriented()
    images = torch.rand(2, 3, 64, 64)
    expected = model(images)
    with (
        patch.object(model._network_capture, "run_oriented") as oriented,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        output = model._training_network(images)
    oriented.assert_not_called()
    _same(
        [getattr(expected, field.name) for field in fields(DualHeadOutput)],
        [getattr(output, field.name) for field in fields(DualHeadOutput)],
    )


@pytest.mark.parametrize(
    ("accumulation", "mutate", "recorder_calls", "reason"),
    [
        pytest.param(1, _override_objective, 0, "the native oriented objective is overridden", id="objective-override"),
    ],
)
def test_oriented_routing_retires_then_runs_native(
    accumulation: int, mutate: Callable[[DetectionLitModule], None], recorder_calls: int, reason: str
) -> None:
    """A broken oriented contract retires the recording once and returns the full native call.

    An overridden consumer retires before the recorder is reached at all, and the output
    is the module's own forward, run exactly once.
    """
    model = _routed_oriented(accumulation)
    mutate(model)
    images = _cuda_flagged_images()
    expected = model(images)
    calls: list[None] = []
    model.backbone.register_forward_hook(lambda *_: calls.append(None))
    with (
        patch.object(model._network_capture, "run_oriented", wraps=model._network_capture.run_oriented) as spy,
        pytest.warns(UserWarning, match=f"stays eager: {reason}"),
    ):
        output = model._training_network(images)
    assert spy.call_count == recorder_calls
    assert model._network_capture._disabled and model._network_capture._network is None
    assert len(calls) == 1
    _same(
        [getattr(expected, field.name) for field in fields(DualHeadOutput)],
        [getattr(output, field.name) for field in fields(DualHeadOutput)],
    )


class _ClassificationOnly(DetectionLitModule):
    """A supported native customization whose objective leaves angle gradients absent."""

    def _task_extra_loss(self, context: Any, out: Any, stage: str) -> torch.Tensor:
        """Keep classification connected while declining oriented supervision."""
        return torch.zeros_like(out.total)


def test_oriented_subclass_override_stays_native() -> None:
    """A consumer overridden in a subclass retires replay, as an instance attribute does.

    The guard unwraps bound methods through ``__func__`` before comparing them. A subclass
    method is a bound method whose function is not the native one, the path the
    instance-level lambdas above never reach; this case runs it without a GPU.
    """
    model = _ClassificationOnly(0.34, 0.25, 64, 2, task="obb", capture_network=True)
    assert _oriented_replay_breach(model) == "the native oriented objective is overridden"


@requires_cuda
def test_oriented_override_keeps_absent_angle_gradients(graph_recordings: MagicMock) -> None:
    """The public trainer fallback skips angle state instead of materializing zero gradients.

    The replaying arm retires once, and for the overridden objective specifically: a
    retirement for any other cause would leave the same absent gradients behind it.
    """
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
        assert all("the native oriented objective is overridden" in message for message in retired), retired
        assert not graph_recordings.call_count
        angles = [parameter for name, parameter in model.named_parameters() if "angle_stems" in name]
        assert angles and all(parameter.grad is None for parameter in angles)
        assert all(parameter not in trainer.optimizers[0].state for parameter in angles)
        results.append(copy.deepcopy((model.state_dict(), trainer.optimizers[0].state_dict())))
    _same(*results)
