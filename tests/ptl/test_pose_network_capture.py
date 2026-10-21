# SPDX-License-Identifier: Apache-2.0
"""Pose replay retains every point/sigma tensor and its eager training gradients."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Iterator
from dataclasses import fields
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from lucid_yolo.cli.train import DetectionCLI, packaged_config
from lucid_yolo.data.targets import Targets
from lucid_yolo.models.heads.detect import DualHeadOutput
from lucid_yolo.ptl import DetectionLitModule
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.module import _POSE_STEM_DISPATCH, _pose_head_breach
from lucid_yolo.ptl.network_capture import _FeatureNetwork, _input_breach, _NetworkCapture, _NetworkCaptureLifecycle


@pytest.fixture(autouse=True)
def _deterministic() -> Iterator[None]:
    """Match the deterministic kernel policy of the public-data fits."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _pose(**kwargs: object) -> DetectionLitModule:
    """Build a real small pose network with a dataset-specified point count.

    Examples:
        >>> _pose().head.num_keypoints
        5
    """
    return DetectionLitModule(
        depth=0.34,
        width=0.25,
        max_channels=64,
        num_classes=2,
        task="keypoints",
        num_keypoints=5,
        **kwargs,
    )


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare raw tensor bytes, including signed zero.

    Examples:
        >>> _bits_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
        False
    """
    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        return False
    return torch.equal(
        left.detach().contiguous().reshape(-1).view(torch.uint8),
        right.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def test_pose_features_preserve_fields() -> None:
    """The recorder's tensor tuple reconstructs native point and uncertainty fields."""
    torch.manual_seed(42)
    model = _pose().eval()
    images = torch.rand(2, 3, 64, 64)
    network = _FeatureNetwork(model.backbone, model.neck)
    with torch.no_grad():
        expected = model(images)
        tensors = network(images)
        actual = model.head(network.rebuild(tensors))
    assert len(tensors) == 3
    for field in fields(DualHeadOutput):
        left, right = getattr(expected, field.name), getattr(actual, field.name)
        if left is None:
            assert right is None
        else:
            assert right is not None and _bits_equal(left, right), field.name


def _stem_gradients(model: DetectionLitModule) -> list[str]:
    """Name the point-stem parameters that hold a gradient.

    Examples:
        >>> _stem_gradients(_pose())
        []
    """
    return [
        name for name, parameter in model.named_parameters() if "keypoint_stems" in name and parameter.grad is not None
    ]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph replay")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("loss_fields", "stems_trained"),
    [
        pytest.param(None, True, id="all-fields"),
        pytest.param(("o2m_cls", "o2o_cls"), False, id="classification-only"),
    ],
)
def test_pose_replay_keeps_outputs_gradients_buffers(
    dtype: torch.dtype, loss_fields: tuple[str, ...] | None, stems_trained: bool, bf16_replay_expected: bool
) -> None:
    """Changing images and repeated SGD updates follow the exact native trajectory.

    The replayed arm runs through ``_training_network`` so the production eligibility,
    breach and root-hook gates apply. A classification-only loss never reaches the point
    stems, and their gradients must stay absent on both arms rather than become zeros.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    native = _pose().cuda().to(memory_format=torch.channels_last)
    replayed = _routed_pose().cuda().to(memory_format=torch.channels_last)
    replayed.load_state_dict(native.state_dict())
    names = [field.name for field in fields(DualHeadOutput)] if loss_fields is None else list(loss_fields)
    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-5) for model in (native, replayed)]
    for _ in range(4):
        image = torch.rand(2, 3, 64, 64, device="cuda").contiguous(memory_format=torch.channels_last)
        outputs = []
        for model, optimizer in zip((native, replayed), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
                output = model(image) if model is native else model._training_network(image)
                loss = sum(
                    value.float().square().mean() for name in names if (value := getattr(output, name)) is not None
                )
            loss.backward()
            outputs.append([getattr(output, field.name) for field in fields(DualHeadOutput)])
            optimizer.step()
            assert bool(_stem_gradients(model)) is stems_trained
        assert (replayed._network_capture._network is not None) == (dtype != torch.bfloat16 or bf16_replay_expected)
        for left, right in zip(*outputs, strict=True):
            assert (left is None) == (right is None)
            if left is not None:
                assert _bits_equal(left, right)
        _assert_same_gradients_and_state(native, replayed)


def _assert_same_gradients_and_state(native: DetectionLitModule, replayed: DetectionLitModule) -> None:
    """Require equal gradient presence and bytes, and equal parameters and buffers, after a step.

    Examples:
        >>> model = _pose()
        >>> _assert_same_gradients_and_state(model, copy.deepcopy(model))
    """
    for (name, left), (_, right) in zip(native.named_parameters(), replayed.named_parameters(), strict=True):
        if left.grad is None:
            assert right.grad is None, name
        else:
            assert right.grad is not None and _bits_equal(left.grad, right.grad), name
    for name, value in native.state_dict().items():
        assert _bits_equal(value, replayed.state_dict()[name]), name


def _frozen_pair(freeze: tuple[str, ...]) -> tuple[DetectionLitModule, DetectionLitModule]:
    """Build a native and a replaying CUDA pose module with equal weights and the same stages frozen.

    Examples:
        >>> native, replayed = _frozen_pair(("backbone",))  # needs CUDA  # doctest: +SKIP
    """
    torch.manual_seed(42)
    native = _pose().cuda().to(memory_format=torch.channels_last)
    replayed = _routed_pose().cuda().to(memory_format=torch.channels_last)
    replayed.load_state_dict(native.state_dict())
    for model in (native, replayed):
        for stage in freeze:
            getattr(model, stage).requires_grad_(False)
    return native, replayed


def _step_pair(
    native: DetectionLitModule, replayed: DetectionLitModule, optimizers: list[torch.optim.Optimizer]
) -> None:
    """Run one classification-plus-points SGD step on a fresh batch through each arm and require equal results.

    Examples:
        >>> pair = _frozen_pair(())  # needs CUDA  # doctest: +SKIP
        >>> optimizers = [torch.optim.SGD(m.parameters(), lr=0.1) for m in pair]  # needs CUDA  # doctest: +SKIP
        >>> _step_pair(*pair, optimizers)  # needs CUDA  # doctest: +SKIP
    """
    image = torch.rand(2, 3, 64, 64, device="cuda").contiguous(memory_format=torch.channels_last)
    outputs = []
    for model, optimizer in zip((native, replayed), optimizers, strict=True):
        optimizer.zero_grad(set_to_none=True)
        output = model(image) if model is native else model._training_network(image)
        loss = sum(
            value.float().square().mean()
            for field in fields(DualHeadOutput)
            if (value := getattr(output, field.name)) is not None
        )
        loss.backward()
        outputs.append(output)
        optimizer.step()
    for field in fields(DualHeadOutput):
        left, right = (getattr(output, field.name) for output in outputs)
        assert (left is None) == (right is None), field.name
        if left is not None:
            assert _bits_equal(left, right), field.name
    _assert_same_gradients_and_state(native, replayed)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph replay")
@pytest.mark.parametrize(
    "freeze",
    [
        pytest.param(("backbone",), id="backbone-only"),
        pytest.param(("neck",), id="neck-only"),
        pytest.param((), id="none"),
    ],
)
def test_pose_replay_with_partial_freeze_matches_native(freeze: tuple[str, ...]) -> None:
    """Fine-tuning with only some shared stages frozen replays and keeps native gradient presence.

    Frozen stages must stay without gradients on both arms, trainable ones must receive
    identical bytes, and the recording must exist so the parity is not two native runs.
    """
    native, replayed = _frozen_pair(freeze)
    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-5) for model in (native, replayed)]
    for _ in range(3):
        _step_pair(native, replayed, optimizers)
    assert replayed._network_capture._network is not None
    for stage in freeze:
        assert all(parameter.grad is None for parameter in getattr(replayed, stage).parameters())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph replay")
def test_pose_unfreeze_mid_fit_warns_once_and_stays_native() -> None:
    """Unfreezing a stage after recording retires the replay with one warning and keeps native results.

    The frozen backbone is recorded for two steps, then trained: the recording would keep
    producing frozen-stage gradients, so the signature change must send every later step
    through the full native call without repeating the warning.
    """
    native, replayed = _frozen_pair(("backbone",))
    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-5) for model in (native, replayed)]
    for _ in range(2):
        _step_pair(native, replayed, optimizers)
    assert replayed._network_capture._network is not None
    for model in (native, replayed):
        model.backbone.requires_grad_(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            _step_pair(native, replayed, optimizers)
    retired = [str(item.message) for item in caught if str(item.message).startswith("capture_network=True stays")]
    assert len(retired) == 1 and "changed after recording" in retired[0], retired
    assert replayed._network_capture._disabled and replayed._network_capture._network is None


def _same_state(left: object, right: object) -> None:
    """Compare optimizer, scaler and model state by tensor bytes and exact metadata.

    Examples:
        >>> _same_state({"count": 2}, {"count": 2})
    """
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and _bits_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _same_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _same_state(a, b)
    else:
        assert left == right


def _pose_batches(layout: str) -> list[tuple[torch.Tensor, list[Targets]]]:
    """Three loader batches of a realistic shape pattern for a five-point pose fit.

    ``uniform`` repeats one ``(2, 3, 64, 64)`` batch; ``empty`` makes the middle batch
    supervision-free; ``short-last`` ends on a one-image batch (``[2, 2, 1]``); ``mixed``
    puts an image with a person and one without in the middle batch; ``invisible`` gives
    the middle batch boxes whose points are all unlabelled (``keypoint_vis`` zero).

    Examples:
        >>> [len(targets) for _, targets in _pose_batches("short-last")]
        [2, 2, 1]
    """
    image = torch.rand(2, 3, 64, 64)
    person = Targets(
        boxes=torch.tensor([[8.0, 8.0, 48.0, 48.0]]),
        labels=torch.tensor([1]),
        keypoints=torch.tensor([[[16.0, 16.0]] * 5]),
        keypoint_vis=torch.ones(1, 5, dtype=torch.int64),
    )
    empty = Targets(
        boxes=torch.empty(0, 4),
        labels=torch.empty(0, dtype=torch.int64),
        keypoints=torch.empty(0, 5, 2),
        keypoint_vis=torch.empty(0, 5, dtype=torch.int64),
    )
    invisible = Targets(
        boxes=person.boxes.clone(),
        labels=person.labels.clone(),
        keypoints=person.keypoints.clone(),
        keypoint_vis=torch.zeros(1, 5, dtype=torch.int64),
    )
    batches = [(image, [copy.deepcopy(person), copy.deepcopy(person)]) for _ in range(3)]
    middle = {
        "uniform": [person, person],
        "empty": [empty, empty],
        "short-last": [person, person],
        "mixed": [person, empty],
        "invisible": [invisible, invisible],
    }[layout]
    batches[1] = (image, [copy.deepcopy(target) for target in middle])
    if layout == "short-last":
        batches[2] = (image[:1], [copy.deepcopy(person)])
    return batches


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires public CUDA training")
@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("layout", ["uniform", "empty", "short-last", "mixed", "invisible"])
def test_pose_trainer_matches_native(
    precision: str, accumulation: int, layout: str, graph_recordings: MagicMock, bf16_replay_expected: bool
) -> None:
    """Public pose training keeps flow/optimizer/scaler state through clipped accumulation.

    Accumulation 1 records shared features and eighteen independent stems once and replays every later batch,
    the empty-supervision batch included, without a retirement warning. Accumulation 2
    never records and reports exactly the accumulation reason, so a lapse after the first
    batch cannot pass as parity of two native runs.
    """
    torch.manual_seed(42)
    batches = _pose_batches(layout)
    retirements = (
        ["accumulate_grad_batches=1"]
        if accumulation == 2
        else ["BF16 replay requires"]
        if precision == "bf16-mixed" and not bf16_replay_expected
        else []
    )
    initial = _pose().state_dict()
    states = []
    for capture in (False, True):
        torch.manual_seed(42)
        model = _pose(capture_network=capture, lr=1e-4)
        model.load_state_dict(initial)
        trainer = Trainer(
            accelerator="cuda",
            devices=1,
            max_epochs=1,
            precision=precision,
            accumulate_grad_batches=accumulation,
            gradient_clip_val=1.0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.fit(model, train_dataloaders=DataLoader(batches, batch_size=None))
        retired = [str(item.message) for item in caught if str(item.message).startswith("capture_network=True stays")]
        expected = retirements if capture else []
        assert len(retired) == len(expected), retired
        assert all(fragment in message for fragment, message in zip(expected, retired, strict=True)), retired
        assert graph_recordings.call_count == 19 * int(capture and not retirements)
        assert trainer.global_step == (3 + accumulation - 1) // accumulation
        assert model._network_capture._network is None
        assert model._network_capture._stems == {}
        assert model._network_capture._stem_owners is None
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        states.append(
            (
                copy.deepcopy(model.state_dict()),
                copy.deepcopy(trainer.optimizers[0].state_dict()),
                copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
            )
        )
    _same_state(*states)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires public CUDA training")
def test_pose_trainer_with_optimizer_captures_matches_native(graph_recordings: MagicMock) -> None:
    """The benchmarked combination of network replay, orthogonalization and update capture stays native-exact.

    One FP32, accumulation-1 fit enables all three CUDA replay options together, the way the
    WP-193d measurement did, and must end on the model weights and per-parameter optimizer
    state of a fit with none of them. Group flags legitimately differ, so only ``state`` is compared.
    """
    batches = _pose_batches("uniform")
    initial = _pose().state_dict()
    results = []
    for options in ({}, {"capture_network": True, "capture_orthogonalization": True, "capture_updates": True}):
        torch.manual_seed(42)
        model = _pose(lr=1e-4, **options)
        model.load_state_dict(initial)
        trainer = Trainer(
            accelerator="cuda",
            devices=1,
            max_epochs=1,
            precision="32-true",
            gradient_clip_val=1.0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="capture_network=True stays eager")
            trainer.fit(model, train_dataloaders=DataLoader(batches, batch_size=None))
        results.append((copy.deepcopy(model.state_dict()), copy.deepcopy(trainer.optimizers[0].state_dict()["state"])))
    assert graph_recordings.call_count == 19
    _same_state(*results)


@pytest.mark.parametrize("capture", [False, True])
def test_pose_cli_accepts_network_capture(capture: bool) -> None:
    """The shipped pose recipe exposes the existing recording flag without another knob."""
    cli = DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        run=False,
        args=[
            "--config",
            str(packaged_config("pose_nano_smoke")),
            "--model.capture_network",
            str(capture).lower(),
            "--trainer.accelerator",
            "cpu",
            "--trainer.logger",
            "false",
        ],
    )
    assert cli.model.hparams.capture_network is capture
    assert cli.model.head.num_keypoints == 17
    assert len(cli.model.configure_callbacks()) == int(capture)


def test_pose_head_certificate_survives_copy_and_rejects_overrides() -> None:
    """A native head stays eligible through deepcopy; changed dispatch cannot detach shared features."""
    model = copy.deepcopy(_pose())
    assert _pose_head_breach(model.head, model._head_topology) is None
    with patch.object(model.head, "forward", wraps=model.head.forward):
        assert _pose_head_breach(model.head, model._head_topology) is not None
    assert _pose_head_breach(model.head, model._head_topology) is None
    with patch.object(type(model.head), "forward", lambda self, values: values):
        assert _pose_head_breach(model.head, model._head_topology) is not None


def test_pose_head_certificate_rejects_forward_branch_overrides() -> None:
    """``forward`` only reaches submodules through ``forward_branch``; both dispatch points count."""
    model = copy.deepcopy(_pose())
    assert _pose_head_breach(model.head, model._head_topology) is None
    with patch.object(model.head, "forward_branch", wraps=model.head.forward_branch):
        assert _pose_head_breach(model.head, model._head_topology) is not None
    assert _pose_head_breach(model.head, model._head_topology) is None
    with patch.object(type(model.head), "forward_branch", lambda self, features, branch: features):
        assert _pose_head_breach(model.head, model._head_topology) is not None


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda head: head.register_forward_pre_hook(lambda *_: None), id="forward-pre-hook"),
        pytest.param(lambda head: head.register_full_backward_hook(lambda *_: None), id="backward-hook"),
        pytest.param(lambda head: next(head.children()).register_forward_hook(lambda *_: None), id="submodule-hook"),
        pytest.param(
            lambda head: setattr(next(head.children()), "forward", next(head.children()).forward),
            id="submodule-forward-override",
        ),
        pytest.param(lambda head: setattr(head, next(head.named_children())[0], torch.nn.Identity()), id="child-swap"),
        pytest.param(lambda head: setattr(head, "_compiled_call_impl", lambda *_: None), id="compiled-call"),
        pytest.param(lambda head: setattr(head.o2m, "_keypoints", head.o2m._keypoints), id="keypoints-override"),
        pytest.param(
            lambda head: setattr(head.o2m, "_base_predictions", head.o2m._base_predictions),
            id="base-predictions-override",
        ),
    ],
)
def test_altered_pose_head_dispatch_fails_certificate(mutate: Callable[[torch.nn.Module], object]) -> None:
    """Every way to change how the head dispatches is refused, so shared features cannot detach from it.

    Starts from a fresh deepcopy certified clean, applies one mutation, and expects a named breach.
    """
    model = copy.deepcopy(_pose())
    assert _pose_head_breach(model.head, model._head_topology) is None
    mutate(model.head)
    assert _pose_head_breach(model.head, model._head_topology) is not None


def test_frozen_features_leave_full_native_call_to_owner() -> None:
    """A fully frozen feature extractor needs no recorded backward, even on CPU."""
    model = _pose()
    model.backbone.requires_grad_(False)
    model.neck.requires_grad_(False)
    holder = _NetworkCapture()
    with pytest.warns(UserWarning, match="shared network features are frozen"):
        features = holder.run_features(torch.zeros(2, 3, 64, 64), (model.backbone, model.neck))
    assert features is None and holder._disabled and holder._network is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pose dispatch")
@pytest.mark.parametrize(
    ("location", "reason"),
    [
        pytest.param("head", "a pose head module is hooked", id="head"),
        pytest.param("global", "a global module hook is registered", id="global"),
    ],
)
def test_pose_hook_retires_features_and_observes_native_root(location: str, reason: str) -> None:
    """After recording, hooks keep their full native call boundary and retire the graph.

    The warning must name the hook that caused the retirement, so a retirement for an
    unrelated reason cannot satisfy the test.
    """
    model = _pose(capture_network=True).cuda()
    model._trainer = SimpleNamespace(training=True, accumulate_grad_batches=1)
    model._capture_network_active = True
    image = torch.rand(2, 3, 64, 64, device="cuda")
    model._training_network(image).o2m_cls.sum().backward()
    model.zero_grad(set_to_none=True)
    assert model._network_capture._network is not None
    calls = []

    def observe(module: torch.nn.Module, args: object, output: object) -> None:
        """Observe only the intended public module in a global hook's larger call tree."""
        del args, output
        if module is (model if location == "global" else model.head):
            calls.append(module)

    handle = (
        model.head.register_forward_hook(observe)
        if location == "head"
        else torch.nn.modules.module.register_module_forward_hook(observe)
    )
    try:
        with pytest.warns(UserWarning, match=f"stays eager: {reason}"):
            model._training_network(image)
        assert calls == [model if location == "global" else model.head]
        assert model._network_capture._disabled and model._network_capture._network is None
    finally:
        handle.remove()


class _CudaFlagged(torch.Tensor):
    """CPU tensor that reports CUDA residency so the CUDA-gated pose routing is reachable offline.

    Operations unwrap to plain tensors, so the real model still runs on the CPU.
    """

    is_cuda = True

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        """Run the operation on the underlying data and return an ordinary tensor."""
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **(kwargs or {}))


def _routed_pose(accumulation: int = 1) -> DetectionLitModule:
    """Build a pose module that satisfies every training-time gate of the shared-feature route.

    Examples:
        >>> _routed_pose(2).trainer.accumulate_grad_batches
        2
    """
    model = _pose(capture_network=True)
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


def _count_backbone_calls(model: DetectionLitModule) -> list[None]:
    """Record each full native forward through the backbone, which only a native fallback performs.

    Examples:
        >>> model = _pose()
        >>> calls = _count_backbone_calls(model)
        >>> _ = model(torch.rand(2, 3, 64, 64))
        >>> len(calls)
        1
    """
    calls: list[None] = []
    model.backbone.register_forward_hook(lambda *_: calls.append(None))
    return calls


@pytest.mark.parametrize(
    ("accumulation", "mutate", "reason"),
    [
        pytest.param(2, lambda head: None, "keypoint replay requires accumulate_grad_batches=1", id="accumulation"),
        pytest.param(
            1,
            lambda head: head.register_forward_hook(lambda *_: None),
            "a pose head module is hooked, overridden or compiled",
            id="head-breach",
        ),
    ],
)
def test_pose_routing_retires_then_runs_native(
    accumulation: int, mutate: Callable[[torch.nn.Module], object], reason: str
) -> None:
    """A broken pose contract retires the recording once and trains through the full native call.

    The CUDA-only route cannot execute on CPU runners, so images spoof ``is_cuda`` and the
    real retirement logic runs; the native fallback shows as one backbone forward.
    """
    torch.manual_seed(42)
    model = _routed_pose(accumulation)
    mutate(model.head)
    images = _cuda_flagged_images()
    calls = _count_backbone_calls(model)
    with (
        patch.object(model._network_capture, "run_features", wraps=model._network_capture.run_features) as spy,
        pytest.warns(UserWarning, match=f"stays eager: {reason}"),
    ):
        output = model._training_network(images)
    spy.assert_called_once_with(images, (model.backbone, model.neck))
    assert model._network_capture._disabled and model._network_capture._network is None
    assert len(calls) == 1
    assert output.o2m_cls.shape[0] == 2


def test_retired_holder_skips_pose_replay_predicates() -> None:
    """A retired recording sends later steps through the full native call without re-checking the head.

    Once retired, the owner check and every later ``retire`` are no-ops, so the stem-owner
    walk and the head certificate could change nothing; a step after retirement skips both
    and never reaches the recorder.
    """
    torch.manual_seed(42)
    model = _routed_pose()
    with pytest.warns(UserWarning, match="stays eager: a test cause"):
        model._network_capture.retire("a test cause")
    images = _cuda_flagged_images()
    with (
        patch("lucid_yolo.ptl.module._pose_head_breach", autospec=True) as breach,
        patch.object(model._network_capture, "check_stem_owners") as owners,
        patch.object(model._network_capture, "run_features") as run,
    ):
        output = model._training_network(images)
    breach.assert_not_called()
    owners.assert_not_called()
    run.assert_not_called()
    assert output.o2m_cls.shape[0] == 2


def test_pose_routing_falls_back_when_features_are_absent() -> None:
    """When the recorder declines to return features, the full native call runs without a warning."""
    torch.manual_seed(42)
    model = _routed_pose()
    images = _cuda_flagged_images()
    calls = _count_backbone_calls(model)
    with patch.object(model._network_capture, "run_features", return_value=None) as spy, warnings.catch_warnings():
        warnings.simplefilter("error")
        output = model._training_network(images)
    spy.assert_called_once_with(images, (model.backbone, model.neck))
    assert len(calls) == 1
    assert output.o2m_cls.shape[0] == 2


def test_pose_routing_dispatches_shared_features_to_native_head() -> None:
    """Returned shared features feed the native head directly, with no second backbone pass."""
    torch.manual_seed(42)
    model = _routed_pose()
    images = _cuda_flagged_images()
    network = _FeatureNetwork(model.backbone, model.neck)
    features = network.rebuild(network(images))
    expected = model.head(features)
    calls = _count_backbone_calls(model)
    with (
        patch.object(model._network_capture, "run_features", return_value=features) as spy,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        output = model._training_network(images)
    spy.assert_called_once_with(images, (model.backbone, model.neck))
    assert calls == []
    for field in fields(DualHeadOutput):
        left, right = getattr(expected, field.name), getattr(output, field.name)
        assert (left is None) == (right is None), field.name
        if left is not None:
            assert _bits_equal(left, right), field.name


def test_pose_routing_leaves_cpu_images_native() -> None:
    """Images that are not on CUDA never reach the recorder, however many gates pass."""
    torch.manual_seed(42)
    model = _routed_pose()
    with patch.object(model._network_capture, "run_features") as spy:
        output = model._training_network(torch.rand(2, 3, 64, 64))
    spy.assert_not_called()
    assert output.o2m_cls.shape[0] == 2


def test_native_stem_executor_keeps_every_field_and_call_order() -> None:
    """The private scheduling seam calls the same eighteen native stems in order."""
    model = _pose().eval()
    features = _FeatureNetwork(model.backbone, model.neck)(torch.rand(2, 3, 64, 64))
    expected = model.head(features)
    visited = []

    def execute(stem: torch.nn.Module, feature: torch.Tensor) -> torch.Tensor:
        visited.append(stem)
        return stem(feature)

    holder = _NetworkCapture()
    with patch.object(holder, "stem", side_effect=execute):
        actual = model.head(features, _runner=holder)
    assert len(visited) == 2 * 3 * 3
    assert len({id(stem) for stem in visited}) == len(visited)
    assert visited[0] is model.head.o2m.box_stems[0]
    assert visited[1] is model.head.o2m.cls_stems[0]
    assert visited[-1] is model.head.o2o.keypoint_stems[-1]
    for field in fields(DualHeadOutput):
        left, right = getattr(expected, field.name), getattr(actual, field.name)
        assert (left is None) == (right is None)
        if left is not None:
            assert _bits_equal(left, right), field.name


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires independent CUDA stem replay")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pose_stems_replay_through_changing_supervision(
    dtype: torch.dtype, graph_recordings: MagicMock, bf16_replay_expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unused point backwards stay absent and resume replay when supervision returns."""
    if dtype == torch.bfloat16 and not bf16_replay_expected:
        pytest.skip("BF16 graph replay requires the verified torch 2.9 runtime")
    launches: list[int] = []
    native_replay = torch.cuda.CUDAGraph.replay

    def observe(graph: torch.cuda.CUDAGraph) -> None:
        """Count real forward/backward execution without replacing device work."""
        launches.append(id(graph))
        native_replay(graph)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", observe)
    torch.manual_seed(42)
    native = _pose().cuda().to(memory_format=torch.channels_last)
    replayed = _routed_pose().cuda().to(memory_format=torch.channels_last)
    replayed.load_state_dict(native.state_dict())
    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9) for model in (native, replayed)]
    for iteration, point_branches in enumerate((("o2m", "o2o"), (), ("o2m",), ("o2m", "o2o"))):
        previous_launches = len(launches)
        replayed._network_capture.start_batch()
        image = torch.rand(2, 3, 64, 64, device="cuda").contiguous(memory_format=torch.channels_last)
        names = ["o2m_cls", "o2o_cls"] + [f"{branch}_keypoints" for branch in point_branches]
        for model, optimizer in zip((native, replayed), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32, cache_enabled=False):
                output = model(image) if model is native else model._training_network(image)
                loss = sum(getattr(output, name).float().square().mean() for name in names)
            loss.backward()
            optimizer.step()
        if iteration:
            # Nineteen forwards; one feature backward, six class backwards,
            # and three point backwards per supervised branch. Box outputs are unused.
            assert len(launches) - previous_launches >= 19 + 1 + 6 + 3 * len(point_branches)
        _assert_same_gradients_and_state(native, replayed)
        _same_state(optimizers[0].state_dict(), optimizers[1].state_dict())
        holders = replayed._network_capture._stems
        assert len(holders) == 18
        assert all(holder._network is not None for holder in holders.values())
        assert graph_recordings.call_count == 19
        for branch in ("o2m", "o2o"):
            stem = getattr(replayed.head, branch).keypoint_stems[0]
            assert all((parameter.grad is not None) == (branch in point_branches) for parameter in stem.parameters())
            assert holders[id(stem)]._pending_backward == (branch not in point_branches)
    restored = copy.deepcopy(replayed._network_capture)
    assert restored._network is None and restored._stems == {}
    replayed._network_capture.clear()
    assert replayed._network_capture._stems == {}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires differentiable CUDA stem inputs")
def test_stem_replay_preserves_input_gradients_and_accumulation() -> None:
    """A stem propagates feature gradients without overwriting adopted parameter gradients."""
    torch.manual_seed(42)
    native = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).cuda()
    replayed = copy.deepcopy(native)
    holder = _NetworkCapture()
    for _ in range(3):
        left = torch.randn(3, 4, device="cuda", requires_grad=True)
        right = left.detach().clone().requires_grad_()
        native(left).square().sum().backward()
        holder.stem(replayed, right).square().sum().backward()
        assert _bits_equal(left.grad, right.grad)
        for expected, actual in zip(native.parameters(), replayed.parameters(), strict=True):
            assert _bits_equal(expected.grad, actual.grad)
        assert holder._stems[id(replayed)]._network is not None
    actual = holder.stem(replayed, right.detach())
    assert _bits_equal(native(right.detach()), actual)


@pytest.mark.parametrize(
    ("owner", "name"),
    [pytest.param(owner, name, id=f"{owner.__name__}.{name}") for owner, name, _ in _POSE_STEM_DISPATCH],
)
def test_stem_dispatch_override_before_construction_stays_observable(owner: type, name: str) -> None:
    """A class-level replacement of any native dispatch method is refused, even before construction.

    Each dispatch-table entry is patched alone with a forwarding wrapper that cannot be
    the native function object, so dropping one entry from the table fails exactly its case.
    """
    native = getattr(owner, name)

    def custom(self: torch.nn.Module, *args: object, **kwargs: object) -> object:
        """Forward to the native method while being a different function object."""
        return native(self, *args, **kwargs)

    with patch.object(owner, name, custom):
        model = _pose()
        assert _pose_head_breach(model.head, model._head_topology) == "the native pose stem dispatch changed"


def test_stem_holder_dispatch_and_lifetime_without_cuda() -> None:
    """CPU CI checks host ownership while actual replay numerics are checked on CUDA."""
    parent = _NetworkCapture()
    stem = torch.nn.Identity()
    feature = _cuda_flagged_images()
    with patch.object(_NetworkCapture, "_run", autospec=True, return_value=feature) as run:
        assert parent.stem(stem, feature) is feature
        child = parent._stems[id(stem)]
        assert parent.stem(stem, feature) is feature
        assert len(parent._stems) == 1
        assert run.call_args.args[0] is child
        assert run.call_args.kwargs == {"allow_input_grad": True}
        _, stages, eager, build, rebuild = run.call_args.args[1:]
        assert stages == (stem,)
        assert eager(feature) is feature
        assert build()(feature) == (feature,)
        assert rebuild((feature,)) is feature
    child._pending_backward = True
    parent.start_batch()
    assert not child._pending_backward
    assert copy.deepcopy(parent)._stems == {}
    parent.clear()
    assert parent._stems == {}


def test_stem_feature_gradient_permission_does_not_change_image_guard() -> None:
    """Only the explicit internal stem route permits differentiable inputs."""
    feature = torch.ones(2, 3, requires_grad=True)
    assert _NetworkCapture().stem(torch.nn.Identity(), feature) is feature

    precision = (False, torch.float16)
    assert _input_breach(feature, precision) == "the input images require gradients"
    assert _input_breach(feature, precision, allow_input_grad=True) is None


@pytest.mark.parametrize(
    ("training", "disabled", "make_feature", "grad"),
    [
        pytest.param(False, False, _cuda_flagged_images, True, id="eval-stage"),
        pytest.param(True, True, _cuda_flagged_images, True, id="disabled-parent"),
        pytest.param(True, False, lambda: torch.rand(2, 3, 64, 64), True, id="cpu-feature"),
        pytest.param(True, False, _cuda_flagged_images, False, id="no-grad"),
    ],
)
def test_stem_bypass_runs_native_without_holder(
    training: bool, disabled: bool, make_feature: Callable[[], torch.Tensor], grad: bool
) -> None:
    """Each native-bypass clause of the stem route returns the native result and starts no recording.

    Every case differs from a recordable call in one clause only: an eval-mode stage, a
    retired parent, a CPU feature or disabled gradients. Dropping that clause would record
    graphs during validation, under ``no_grad`` or for an eval-mode stem.
    """
    parent = _NetworkCapture()
    parent._disabled = disabled
    stage = torch.nn.Identity().train(training)
    feature = make_feature()
    with patch.object(_NetworkCapture, "_run", autospec=True) as run, torch.set_grad_enabled(grad):
        result = parent.stem(stage, feature)
    assert result is feature
    assert parent._stems == {}
    run.assert_not_called()


def test_recorded_stem_switched_to_eval_reaches_drift_check() -> None:
    """A recorded stem switched to eval goes through its holder, whose drift check retires it.

    The holder is marked recorded without CUDA and the patched replay path stands in for the
    drift check that a CUDA run applies; only the routing decision is exercised here.
    """
    parent = _NetworkCapture()
    stage = torch.nn.Identity()
    feature = _cuda_flagged_images()
    with patch.object(_NetworkCapture, "_run", autospec=True, return_value=feature) as run:
        parent.stem(stage, feature)
        parent._stems[id(stage)]._network = torch.nn.Identity()
        stage.eval()
        parent.stem(stage, feature)
    assert run.call_count == 2


def test_stem_retirement_warning_names_stem_and_caller() -> None:
    """A retired stem holder names its stem, says the rest keeps replaying, and points past internal frames.

    The child is created through the parent's normal route after the owner check recorded
    the head's module names; the warning must name this test as its caller, the first frame
    outside the replay stack, never the head's dispatch code.
    """
    model = _pose()
    parent = _NetworkCapture()
    parent.check_stem_owners(model.head)
    stage = model.head.o2m.keypoint_stems[0]
    feature = _cuda_flagged_images()
    with patch.object(_NetworkCapture, "_run", autospec=True, return_value=feature):
        parent.stem(stage, feature)
    match = r"keeps pose stem o2m\.keypoint_stems\.0 eager while the shared features and the other stems keep replaying"
    with pytest.warns(UserWarning, match=match) as caught:
        parent._stems[id(stage)].retire("a test cause")
    assert caught[0].filename == __file__


@pytest.mark.parametrize(
    "finish",
    [
        pytest.param(lambda callback, module: callback.on_fit_end(cast("Trainer", None), module), id="fit-end"),
        pytest.param(
            lambda callback, module: callback.on_exception(cast("Trainer", None), module, RuntimeError("abort")),
            id="exception",
        ),
    ],
)
def test_lifecycle_releases_stem_holders(finish: Callable[[_NetworkCaptureLifecycle, object], None]) -> None:
    """Fit completion and failure both release every stem holder together with the shared recording.

    A child holder stands in for a recorded stem and the callback is driven directly, so the
    clearing path is checked on CPU while recording numerics stay on the CUDA matrix.
    """
    holder = _NetworkCapture()
    holder.check_stem_owners(_pose().head)
    holder._stems[1] = _NetworkCapture("stem")
    finish(_NetworkCaptureLifecycle(), SimpleNamespace(_network_capture=holder))
    assert holder._stems == {}
    assert holder._stem_owners is None


def test_changed_pose_stem_ownership_retires_all_holders() -> None:
    """An identical replacement cannot keep adding unreachable graph owners to a fit."""
    model = _pose()
    holder = _NetworkCapture()
    holder.check_stem_owners(model.head)
    holder._stems[1] = _NetworkCapture()
    holder.check_stem_owners(model.head)
    assert not holder._disabled and len(holder._stems) == 1
    model.head.o2m.keypoint_stems[0] = copy.deepcopy(model.head.o2m.keypoint_stems[0])
    with pytest.warns(UserWarning, match="pose head module was replaced"):
        holder.check_stem_owners(model.head)
    assert holder._disabled and holder._stems == {} and holder._stem_owners is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires retirement before CUDA feature replay")
def test_replaced_pose_stem_keeps_complete_forward_native() -> None:
    """Replacement is detected before recording can leave an obsolete graph pool alive."""
    native, replayed = _frozen_pair(())
    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-5) for model in (native, replayed)]
    _step_pair(native, replayed, optimizers)
    assert len(replayed._network_capture._stems) == 18
    replacement = copy.deepcopy(native.head.o2m.keypoint_stems[0])
    native.head.o2m.keypoint_stems[0] = copy.deepcopy(replacement)
    replayed.head.o2m.keypoint_stems[0] = copy.deepcopy(replacement)
    with pytest.warns(UserWarning, match="pose head module was replaced"):
        _step_pair(native, replayed, optimizers)
    assert replayed._network_capture._disabled
    assert replayed._network_capture._network is None and replayed._network_capture._stems == {}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires changed CUDA stem trainability")
def test_changed_stem_trainability_retires_only_its_recording() -> None:
    """Freezing and unfreezing one point stem preserves native gradients and optimizer state."""
    native, replayed = _frozen_pair(())
    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9) for model in (native, replayed)]
    _step_pair(native, replayed, optimizers)
    for model in (native, replayed):
        model.head.o2m.keypoint_stems[0].requires_grad_(False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _step_pair(native, replayed, optimizers)
    retired = [item for item in caught if str(item.message).startswith("capture_network=True")]
    assert len(retired) == 1, [str(item.message) for item in retired]
    assert "o2m.keypoint_stems.0" in str(retired[0].message)
    assert "changed after recording" in str(retired[0].message)
    assert not retired[0].filename.endswith("detect.py")
    capture = replayed._network_capture
    frozen = id(replayed.head.o2m.keypoint_stems[0])
    assert capture._network is not None
    assert capture._stems[frozen]._disabled and capture._stems[frozen]._network is None
    assert sum(child._disabled for child in capture._stems.values()) == 1
    assert sum(child._network is not None for child in capture._stems.values()) == len(capture._stems) - 1
    for model in (native, replayed):
        model.head.o2m.keypoint_stems[0].requires_grad_(True)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="capture_network=True")
        _step_pair(native, replayed, optimizers)
    assert capture._stems[frozen]._disabled
    _same_state(optimizers[0].state_dict(), optimizers[1].state_dict())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires changed CUDA stem mode")
def test_changed_stem_mode_retires_only_its_recording() -> None:
    """Switching one recorded point stem to eval retires that stem alone and keeps native parity.

    The mode change reaches the stem holder's drift check, as a recorded feature stage's
    would, so the stem runs eager while the shared features keep replaying.
    """
    native, replayed = _frozen_pair(())
    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9) for model in (native, replayed)]
    _step_pair(native, replayed, optimizers)
    for model in (native, replayed):
        model.head.o2m.keypoint_stems[0].eval()
    with pytest.warns(UserWarning, match=r"keeps pose stem o2m\.keypoint_stems\.0 eager"):
        _step_pair(native, replayed, optimizers)
    capture = replayed._network_capture
    assert capture._network is not None
    assert capture._stems[id(replayed.head.o2m.keypoint_stems[0])]._disabled
    _same_state(optimizers[0].state_dict(), optimizers[1].state_dict())
