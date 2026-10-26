# SPDX-License-Identifier: Apache-2.0
"""Pose loss execution reuses native assignment and independent branch reductions."""

from __future__ import annotations

import copy
import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch
from markers import requires_cuda
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.plugins.precision import MixedPrecision
from torch.utils.data import DataLoader

from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl.loss_capture import _BF16_SUPPORTED, _LossCaptureLifecycle
from lucid_yolo.ptl.module import DetectionLitModule


def _pose(
    capture: bool, keypoint_loss: str = "rle", *, alpha: float = 0.5, oks_gain: float = 0.0
) -> DetectionLitModule:
    """Construct the public pose option with a small matching annotation schema.

    Examples:
        >>> _pose(True).hparams.capture_loss
        True
        >>> _pose(False, alpha=0.8, oks_gain=1.0).loss.alpha
        0.8
    """
    return DetectionLitModule(
        0.5,
        0.25,
        64,
        2,
        task="keypoints",
        num_keypoints=3,
        capture_loss=capture,
        keypoint_loss=keypoint_loss,
        alpha=alpha,
        oks_gain=oks_gain,
    )


@pytest.fixture
def true_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run cuDNN convolutions in IEEE float32 rather than TF32, the arithmetic a ``32-true`` bound assumes.

    cuDNN defaults to TF32 for float32 convolutions on Ampere and newer. There it rounds operands to a
    10-bit mantissa, which turns ulp-level differences in a loss gradient into relative gradient
    distances above 1e-5: an A100 shows 4.7e-5 between replay and native at ``32-true``, and 1e-5 or
    less with TF32 off (WP-209). Only the legacy flag is set, because a partial set through the newer
    ``fp32_precision`` attributes makes the legacy getters Lightning reads raise.

    Examples:
        >>> callable(true_fp32)
        True
    """
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)


def _batch(count: int, visible: bool = True) -> tuple[torch.Tensor, list[Targets]]:
    """Build one empty image beside an asymmetric point-annotated image.

    Examples:
        >>> [len(target.boxes) for target in _batch(2)[1]]
        [0, 2]
    """
    boxes = torch.tensor([[4.0, 6.0, 36.0, 42.0], [24.0, 18.0, 59.0, 62.0]])[:count]
    points = torch.tensor([[[10.0, 12.0], [18.0, 20.0], [28.0, 30.0]], [[30.0, 24.0], [40.0, 32.0], [50.0, 52.0]]])
    targets = [
        Targets(
            boxes=boxes[:n].clone(),
            labels=torch.arange(n),
            keypoints=points[:n].clone(),
            keypoint_vis=torch.full((n, 3), 2 if visible else 0, dtype=torch.long),
        )
        for n in (0, count)
    ]
    return torch.rand(2, 3, 64, 64), targets


@pytest.mark.parametrize("keypoint_loss", ["rle", "laplace_nll"])
@pytest.mark.parametrize("count,visible", [(0, False), (1, False), (2, True)])
@pytest.mark.parametrize("alpha,oks_gain", [(0.5, 0.0), (0.8, 1.0)])
def test_pose_host_execution_keeps_loss_and_gradient_values(
    alpha: float, oks_gain: float, count: int, visible: bool, keypoint_loss: str
) -> None:
    """On CPU the routed pose step matches the native step in loss value and gradients.

    This proves wiring only: the loss recorder is entered and answers with its native call, and the paired
    density scoring reproduces the per-branch result. Nothing is recorded off CUDA, so recorder coverage
    lives in the gpu-marked tests below.

    The default branch weights (alpha 0.5, OKS gain 0) hide a swapped branch blend, since both branches weigh
    the same, and a dropped OKS term, since its gain is zero. The second weight pair makes both visible.
    """
    seed_everything(19)
    native = _pose(False, keypoint_loss, alpha=alpha, oks_gain=oks_gain)
    executed = copy.deepcopy(native)
    executed._capture_loss_active = True
    executed._trainer = MagicMock(spec=Trainer, training=True)
    images, targets = _batch(count, visible)
    with (
        patch.object(native, "log"),
        patch.object(executed, "log"),
        patch.object(executed, "_paired_keypoint_terms", wraps=executed._paired_keypoint_terms) as paired,
        patch.object(executed._loss_capture, "run", wraps=executed._loss_capture.run) as replay,
        patch("torch.cuda.make_graphed_callables", side_effect=AssertionError("CPU recording")),
    ):
        expected = native.training_step((images.clone(), targets), 0)
        actual = executed.training_step((images.clone(), targets), 0)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        expected.backward()
        actual.backward()
    assert replay.call_count == 1
    assert paired.call_count == int(keypoint_loss == "rle")
    for (name, left), (other, right) in zip(native.named_parameters(), executed.named_parameters(), strict=True):
        assert name == other and (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert right.grad is not None
            assert (left.grad - right.grad).norm() <= 1e-5 * left.grad.norm() + 1e-6, name


@pytest.mark.parametrize("custom", ["_score_branch", "_rle_on", "_oks_on", "_gather_keypoint_positives"])
def test_pose_custom_scoring_keeps_native_dispatch(custom: str) -> None:
    """An overridden scoring boundary cannot be bypassed by the faster composition."""
    model = _pose(True)
    model._capture_loss_active = True
    model._trainer = MagicMock(spec=Trainer, training=True)
    original = getattr(model, custom)
    with (
        patch.object(model, "log"),
        patch.object(model, custom, wraps=original) as wrapped,
        patch.object(model, "_paired_keypoint_terms", side_effect=AssertionError("custom scoring paired")),
    ):
        model.training_step(_batch(1), 0)
    assert wrapped.call_count == 2


def test_pose_pairing_continues_after_recorder_retires() -> None:
    """Density batching keys on the fit's capture flag, not on the loss recorder's state.

    A retired recorder answers every later detection loss with its native call for the rest of
    the fit. The pose density batch has a gate of its own, so it still scores both branches in one
    call — the documented behaviour, pinned so a later change to it is a decision, not a drift.
    """
    model = _pose(True)
    model._capture_loss_active = True
    model._trainer = MagicMock(spec=Trainer, training=True)
    with pytest.warns(UserWarning, match="^capture_loss=True stays eager: a test cause$"):
        model._loss_capture.retire("a test cause")
    with (
        patch.object(model, "log"),
        patch.object(model, "_paired_keypoint_terms", wraps=model._paired_keypoint_terms) as paired,
        patch.object(model._loss_capture, "run", wraps=model._loss_capture.run) as replay,
    ):
        model.training_step(_batch(2), 0)
    assert replay.call_count == 1 and model._loss_capture._disabled
    assert paired.call_count == 1


def test_pose_validation_keeps_native_dispatch_with_certified_criterion() -> None:
    """A validation step keeps native loss dispatch although the stock criterion could be recorded.

    The stock criterion certifies for recording, so only the training-stage gate keeps a validation step
    off the detection objective the recorder serves. A customized criterion also takes the native route,
    but by failing certification, so it would leave the gate itself unproven; the custom-criterion tests
    own that route and its keyword signature.
    """
    model = _pose(True).eval()
    model._capture_loss_active = True
    with (
        patch.object(model, "log"),
        patch.object(model, "_objective", side_effect=AssertionError("validation objective")),
        patch.object(model._loss_capture, "run", side_effect=AssertionError("validation recording")),
        patch.object(model, "_paired_keypoint_terms", side_effect=AssertionError("validation pairing")),
    ):
        loss, _, _ = model._shared_step(_batch(1), "val")
    assert torch.isfinite(loss) and not model._loss_capture._disabled


@requires_cuda
@pytest.mark.parametrize(
    "precision",
    [
        "32-true",
        "16-mixed",
        pytest.param(
            "bf16-mixed", marks=pytest.mark.skipif(not _BF16_SUPPORTED, reason="BF16 replay requires torch 2.9")
        ),
    ],
)
@pytest.mark.parametrize("count,visible,records", [(2, True, True), (2, False, True), (0, False, False)])
@pytest.mark.usefixtures("true_fp32")
def test_pose_cuda_execution_keeps_objective_and_gradient_direction(
    precision: str, count: int, visible: bool, records: bool
) -> None:
    """A real automatic step records assignment and batches the same point objective.

    Loss and gradients must agree with the native step within the precision's epsilon: over the whole gradient in
    direction, as cosine similarity, and in relative L2 distance; in float32 also per parameter. Under autocast the
    distance bound is 8 epsilon, set from the 1-ulp noise floor measured on an A100 (WP-209).

    Besides a batch of visible points, the step runs one whose points are all invisible and one with no
    instances. An empty batch is answered natively, so the recorder only records for the first two.
    """
    seed_everything(23)
    batch = _batch(count, visible)
    results = []

    class Observe(Callback):
        def __init__(self) -> None:
            self.gradients = {}
            self.loss = None
            self.recorded = False

        def on_after_backward(self, trainer: Trainer, pl_module: DetectionLitModule) -> None:
            self.gradients = {
                name: None if value.grad is None else value.grad.detach().cpu().clone()
                for name, value in pl_module.named_parameters()
            }

        def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: DetectionLitModule,
            outputs: dict[str, torch.Tensor],
            batch: object,
            batch_idx: int,
        ) -> None:
            self.loss = outputs["loss"].detach().cpu().clone()
            self.recorded = pl_module._loss_capture._objective is not None

    for capture in (False, True):
        seed_everything(17)
        model = _pose(capture)
        observed = Observe()
        plugin = (
            MixedPrecision("16-mixed", "cuda", scaler=torch.amp.GradScaler("cuda", init_scale=32.0))
            if precision == "16-mixed"
            else None
        )
        trainer = Trainer(
            accelerator="cuda",
            devices=1,
            precision=None if plugin is not None else precision,
            plugins=plugin,
            max_steps=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            callbacks=[observed],
            deterministic=True,
        )
        with patch.object(model, "_paired_keypoint_terms", wraps=model._paired_keypoint_terms) as paired:
            trainer.fit(model, train_dataloaders=DataLoader([batch], batch_size=None))
        assert paired.call_count == int(capture)
        assert observed.recorded is (capture and records)
        assert model._loss_capture._objective is None
        results.append(observed)
    epsilon = {
        "32-true": 1e-5,
        "16-mixed": 2 * torch.finfo(torch.float16).eps,
        "bf16-mixed": 2 * torch.finfo(torch.bfloat16).eps,
    }[precision]
    torch.testing.assert_close(results[0].loss, results[1].loss, rtol=epsilon, atol=1e-6)
    assert results[0].gradients.keys() == results[1].gradients.keys()
    for name, left in results[0].gradients.items():
        right = results[1].gradients[name]
        assert (left is None) == (right is None), name
        if left is not None:
            assert right is not None and torch.isfinite(left).all() and torch.isfinite(right).all()
            # Per parameter only in float32. Under autocast a gradient that is zero in exact arithmetic (the
            # attention BatchNorm biases) holds residue as large as itself: on an A100 a 1-ulp change to the head
            # output gradients moves one such parameter by up to 922 times this bound, so no per-parameter bound
            # that admits it would still test anything. The float32 case carries the per-parameter claim.
            if precision == "32-true":
                assert (left - right).norm() <= epsilon * left.norm() + 1e-5, name
    # One vector per arm, in float64: a per-parameter cosine on a near-zero gradient is rounding noise.
    native, executed = (
        torch.cat([value.flatten().double() for value in arm.gradients.values() if value is not None])
        for arm in results
    )
    assert torch.cosine_similarity(native, executed, dim=0) >= 1 - epsilon
    # Whole-gradient distance under autocast: five 1-ulp noise draws on an A100 reach 3.4 epsilon (16-mixed) and
    # 3.2 epsilon (bf16-mixed), replay 1.3 and 1.2; the bound is 8 epsilon, about 2.5 times the largest draw.
    assert (native - executed).norm() <= (epsilon if precision == "32-true" else 8 * epsilon) * native.norm()


@requires_cuda
@pytest.mark.usefixtures("true_fp32")
def test_pose_cuda_execution_composes_with_network_replay() -> None:
    """Pose loss dispatch stays correct alongside the independent backbone/neck network replay.

    `capture_loss` reaches the network through `_training_network`, which runs
    `_network_capture.run_features` for pose when `capture_network` is also set — a
    second, independent recorder with its own graph pool. Nothing before this test ran
    the two together for `task="keypoints"`.

    The four steps differ in content: full instances, no instances, instances without a visible point, then
    full instances again. The empty step is answered natively by the loss recorder, and the last step proves
    both recorders still serve the fit afterwards. Loss and gradients are compared at every step.
    """
    seed_everything(23)
    batches = [_batch(2), _batch(0), _batch(1, visible=False), _batch(2)]
    results = []

    class Observe(Callback):
        def __init__(self) -> None:
            self.gradients: list[dict[str, torch.Tensor | None]] = []
            self.losses: list[torch.Tensor] = []
            self.active: list[tuple[bool, bool]] = []

        def on_after_backward(self, trainer: Trainer, pl_module: DetectionLitModule) -> None:
            self.gradients.append(
                {
                    name: None if value.grad is None else value.grad.detach().cpu().clone()
                    for name, value in pl_module.named_parameters()
                }
            )

        def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: DetectionLitModule,
            outputs: dict[str, torch.Tensor],
            batch: object,
            batch_idx: int,
        ) -> None:
            self.losses.append(outputs["loss"].detach().cpu().clone())
            self.active.append(
                (pl_module._network_capture._network is not None, pl_module._loss_capture._objective is not None)
            )

    for capture_network, capture_loss in ((False, False), (True, True)):
        seed_everything(17)
        model = DetectionLitModule(
            0.5,
            0.25,
            64,
            2,
            task="keypoints",
            num_keypoints=3,
            capture_network=capture_network,
            capture_loss=capture_loss,
        )
        observed = Observe()
        trainer = Trainer(
            accelerator="cuda",
            devices=1,
            precision="32-true",
            max_steps=len(batches),
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            callbacks=[observed],
            deterministic=True,
        )
        trainer.fit(model, train_dataloaders=DataLoader(batches, batch_size=None))
        if capture_network:
            assert observed.active[-1] == (True, True)
        results.append(observed)

    assert len(results[0].losses) == len(results[0].gradients) == len(batches)
    assert len(results[1].losses) == len(results[1].gradients) == len(batches)
    # Step 0 starts both arms from the same weights; every later step starts from weights one optimizer update
    # apart, and that update carries the previous step's rounding. Five 1-ulp noise draws on the head output
    # gradients on an A100 move a later step's loss by up to 1.7e-5 and its whole gradient by up to 4.0e-4
    # relative, and replay by 6.9e-6 and 2.3e-4; the later-step bounds are 5e-5 and 1e-3, about 2.5 times the
    # largest draw. Step 0 keeps the per-parameter claim.
    for step, (left, right) in enumerate(zip(results[0].losses, results[1].losses, strict=True)):
        torch.testing.assert_close(
            left, right, rtol=1e-5 if step == 0 else 5e-5, atol=1e-6, msg=lambda text, step=step: f"step {step}: {text}"
        )
    for step, (native, executed) in enumerate(zip(results[0].gradients, results[1].gradients, strict=True)):
        assert native.keys() == executed.keys(), step
        for name, left in native.items():
            right = executed[name]
            assert (left is None) == (right is None), (step, name)
            if left is not None:
                assert right is not None and torch.isfinite(right).all()
                # The absolute floor is the one the single-step test uses. Some parameters have a gradient that is
                # zero in exact arithmetic (an attention projection's BatchNorm bias reads 2e-15 in float64), so in
                # float32 both routes hold rounding residue in steps of 2**-23, which a 1e-6 floor cannot resolve.
                if step == 0:
                    assert (left - right).norm() <= 1e-5 * left.norm() + 1e-5, (step, name)
        left, right = (
            torch.cat([value.flatten().double() for value in arm.values() if value is not None])
            for arm in (native, executed)
        )
        assert (left - right).norm() <= 1e-3 * left.norm(), step


#: The warning a customized criterion raises, for detection and pose alike, when it retires the fit's recorder.
_CUSTOMIZED = "capture_loss=True stays eager: the loss or an assignment implementation was customized"


def test_pose_custom_criterion_keeps_native_keywords() -> None:
    """A customized pose criterion need not accept the recorder's alpha keyword, and its fallback is reported.

    The routing gate sends a criterion the recorder cannot certify to its native call, which never reaches
    the recorder's own source check; the step still names the cause, as detection does for the same setup.
    """
    model = _pose(True)
    model._capture_loss_active = True
    model._trainer = MagicMock(spec=Trainer, training=True)
    original = model.loss
    calls = []

    def criterion(
        ml: torch.Tensor,
        mb: torch.Tensor,
        ol: torch.Tensor,
        ob: torch.Tensor,
        points: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        *,
        strides: torch.Tensor,
        gt_rboxes: torch.Tensor | None,
    ) -> object:
        calls.append(gt_rboxes)
        return original(ml, mb, ol, ob, points, boxes, labels, mask, strides=strides, gt_rboxes=gt_rboxes)

    model.loss = criterion
    with (
        patch.object(model, "log"),
        patch.object(model._loss_capture, "run", side_effect=AssertionError("custom criterion recorded")),
        pytest.warns(UserWarning, match=f"^{_CUSTOMIZED}$"),
    ):
        result = model.training_step(_batch(1), 0)
    assert len(calls) == 1 and calls[0] is None and torch.isfinite(result)


def test_pose_custom_criterion_retires_recorder_once_per_fit() -> None:
    """The first customized-criterion step retires the recorder; later steps of the fit stay quiet.

    Without a once-per-fit latch, every training step would repeat the same fallback warning. The latch is
    the recorder's own retirement, which the fit lifecycle clears at the next fit start.
    """
    model = _pose(True)
    model._capture_loss_active = True
    model._trainer = MagicMock(spec=Trainer, training=True)
    model.loss = model.loss.__call__  # the same arithmetic behind a callable the recorder cannot certify
    with patch.object(model, "log"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.training_step(_batch(1), 0)
        model.training_step(_batch(2), 1)
    retirements = [str(warning.message) for warning in caught if "stays eager" in str(warning.message)]
    assert retirements == [_CUSTOMIZED]
    assert model._loss_capture._disabled


#: The `_keypoint_terms` refusal a 4-point head raises for the 3-point annotations of `_batch`.
_POINT_MISMATCH = "the head predicts 4 keypoints but the batch annotates 3"


def _mismatched_pose() -> DetectionLitModule:
    """Construct a capture-enabled pose module whose head predicts one point more than `_batch` annotates.

    Examples:
        >>> _mismatched_pose().hparams.num_keypoints
        4
    """
    return DetectionLitModule(0.5, 0.25, 64, 2, task="keypoints", num_keypoints=4, capture_loss=True)


def _single_step_trainer(accelerator: str, callback: Callback) -> Trainer:
    """Build a one-step automatic trainer with no logging, checkpointing or validation.

    Examples:
        >>> _single_step_trainer("cpu", Callback()).max_steps
        1
    """
    return Trainer(
        accelerator=accelerator,
        devices=1,
        precision="32-true",
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[callback],
    )


@pytest.mark.parametrize("capture", [False, True])
def test_pose_capture_installs_loss_lifecycle(capture: bool) -> None:
    """A pose module opting into capture_loss carries the fit lifecycle that releases the recorder.

    The lifecycle clears the recorder at fit start, at fit end and on failure. A pose step can fail after
    the detection objective already ran, so pose needs that owner exactly as detection does.
    """
    callbacks = _pose(capture).configure_callbacks()
    assert [type(callback) for callback in callbacks] == [_LossCaptureLifecycle] * capture


def test_pose_point_mismatch_clears_recorder_on_fit_exception() -> None:
    """A pose fit aborted by the point-count check leaves its loss recorder cleared.

    `_keypoint_terms` raises after the detection objective ran. CPU records nothing, so a callback marks
    the recorder retired at batch start; only the lifecycle's `on_exception` can reset that mark before
    the error reaches the caller, which is the release a CUDA recording relies on.
    """

    class Retire(Callback):
        def on_train_batch_start(
            self, trainer: Trainer, pl_module: DetectionLitModule, batch: object, batch_idx: int
        ) -> None:
            pl_module._loss_capture._disabled = True

    model = _mismatched_pose()
    trainer = _single_step_trainer("cpu", Retire())
    with (
        pytest.warns(UserWarning, match="capture_loss=True is ignored outside single-device CUDA"),
        pytest.raises(ValueError, match=_POINT_MISMATCH),
    ):
        trainer.fit(model, train_dataloaders=DataLoader([_batch(2)], batch_size=None))
    assert not model._loss_capture._disabled


@requires_cuda
def test_pose_cuda_point_mismatch_releases_recording() -> None:
    """A recording made before the point-count check raises is released by the fit's exception hook.

    The detection objective records on the first CUDA step, then `_keypoint_terms` rejects the batch. A
    trainer callback runs before the model's own lifecycle, so it sees the recording in `on_exception`;
    after the error propagates, the graph and its target workspace are gone.
    """

    class Observe(Callback):
        def __init__(self) -> None:
            self.recorded: bool | None = None

        def on_exception(self, trainer: Trainer, pl_module: DetectionLitModule, exception: BaseException) -> None:
            self.recorded = pl_module._loss_capture._objective is not None

    model = _mismatched_pose()
    observed = Observe()
    trainer = _single_step_trainer("cuda", observed)
    with pytest.raises(ValueError, match=_POINT_MISMATCH):
        trainer.fit(model, train_dataloaders=DataLoader([_batch(2)], batch_size=None))
    assert observed.recorded is True
    assert model._loss_capture._objective is None and model._loss_capture._workspace is None
