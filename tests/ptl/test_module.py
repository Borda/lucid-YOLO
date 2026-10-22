# SPDX-License-Identifier: Apache-2.0
"""Tests for the detection LightningModule and its helpers (WP-034).

Covers the DoD ``test_training_step`` (a forward + dual loss over a synthetic
batch returns a finite scalar carrying gradients and logs every per-component
term), the :func:`~lucid_yolo.ptl.module.pad_targets` ragged-to-dense contract
(including empty and all-empty batches), ``configure_optimizers`` building a
:class:`~lucid_yolo.optim.musgd.MuSGD` over every parameter, the constant-LR /
automatic-optimization wiring proved by a ``fast_dev_run`` Lightning smoke run,
the ``alpha`` delegation seam (WP-035), and the task conditioning: an invalid
task is rejected, the ``_task_extra_loss`` stub is an inert zero for every task,
and a ``segment`` module trains identically to a ``detect`` module. ``setup``'s
warning for an opted-in recording it cannot activate names the line that called it.

The module is built at n-scale multipliers with a low channel cap and a 160-px
input so the real backbone/neck/head stack runs in a couple of seconds on CPU.
Batches are synthetic tensors plus hand-built :class:`~lucid_yolo.data.targets.Targets`
(no dataset fixture needed). Direct ``training_step`` calls monkeypatch
``module.log`` because Lightning's ``self.log`` requires trainer attachment; the
``fast_dev_run`` test exercises the real logging path.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from markers import requires_cuda
from pytorch_lightning import Trainer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from training import collate_unpacked, tiny_module

from lucid_yolo.data.targets import Targets
from lucid_yolo.losses.dual_loss import DualLossOutput
from lucid_yolo.models.heads.detect import decode_ltrb
from lucid_yolo.optim.musgd import MuSGD
from lucid_yolo.ptl import DetectionLitModule, pad_targets
from lucid_yolo.ptl.module import _host_split, _StepContext

#: Frozen pre-WP-087 state-dict key sequence of a default detection module.
_PRECHANGE_KEYS_FILE = Path(__file__).parent / "prechange_detect_state_dict_keys.txt"
#: Class count of the tiny test head.
_NUM_CLASSES = 4
#: Point count for the keypoint modules the distributed guard test builds. Small and
#: arbitrary: the guard fires on the task name, so the schema only has to be valid.
_GUARD_NUM_KEYPOINTS = 3
#: Square input side; divisible by every stride for an integer anchor grid.
_IMG_SIZE = 160
#: Images per synthetic batch.
_BATCH_SIZE = 2
#: The full set of metric keys a train step must log.
_TRAIN_LOG_KEYS = {
    "train/loss",
    "train/o2m_box",
    "train/o2m_cls",
    "train/o2m_l1",
    "train/o2o_box",
    "train/o2o_cls",
    "train/o2o_l1",
}


def _synthetic_targets(num_boxes: int) -> Targets:
    """Build ``num_boxes`` valid ``xyxy`` targets inside the test image.

    Examples:
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> targets = _synthetic_targets(2)
        >>> targets.boxes.shape, targets.labels.shape
        (torch.Size([2, 4]), torch.Size([2]))
    """
    top_left = torch.rand(num_boxes, 2) * 80.0
    size = torch.rand(num_boxes, 2) * 40.0 + 10.0
    boxes = torch.cat([top_left, top_left + size], dim=1)
    labels = torch.randint(0, _NUM_CLASSES, (num_boxes,))
    return Targets(boxes=boxes, labels=labels)


def _synthetic_batch() -> tuple[Tensor, list[Targets]]:
    """Build a two-image batch with ragged (2 and 1) instance counts.

    Examples:
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> images, targets = _synthetic_batch()
        >>> images.shape
        torch.Size([2, 3, 160, 160])
        >>> [target.boxes.shape[0] for target in targets]
        [2, 1]
    """
    images = torch.randn(_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)
    targets = [_synthetic_targets(2), _synthetic_targets(1)]
    return images, targets


def _with_polygons(target: Targets) -> Targets:
    """Attach the rectangular ring of each box, so a segment module can supervise masks.

    Examples:
        >>> boxes = torch.tensor([[0.0, 0.0, 4.0, 2.0]])
        >>> target = Targets(boxes=boxes, labels=torch.tensor([0]))
        >>> polygoned = _with_polygons(target)
        >>> len(polygoned.polygons), polygoned.polygons[0]
        (1, tensor([[0., 0.],
                [4., 0.],
                [4., 2.],
                [0., 2.]]))
    """
    rings = [
        torch.tensor([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=torch.float32)
        for x1, y1, x2, y2 in target.boxes.tolist()
    ]
    return Targets(boxes=target.boxes, labels=target.labels, polygons=rings)


def _synthetic_batch_with_polygons() -> tuple[Tensor, list[Targets]]:
    """Build the same ragged batch with one polygon ring per instance.

    Examples:
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> images, targets = _synthetic_batch_with_polygons()
        >>> [len(target.polygons) for target in targets]  # one ring per box
        [2, 1]
    """
    images, targets = _synthetic_batch()
    return images, [_with_polygons(target) for target in targets]


def _dual_loss_output(module: DetectionLitModule, images: Tensor, targets: list[Targets]) -> DualLossOutput:
    """Score the module's own dual detection loss over a batch (the reference total).

    Examples:
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> module = tiny_module()
        >>> images, targets = _synthetic_batch()
        >>> out = _dual_loss_output(module, images, targets)
        >>> isinstance(out, DualLossOutput)
        True
        >>> bool(torch.isfinite(out.total))
        True
    """
    head_out = module(images)
    points, strides = module._anchor_grid(images.shape[-2], images.shape[-1], images.device)
    gt_boxes, gt_labels, gt_mask = pad_targets(targets)
    return module.loss(
        head_out.o2m_cls,
        decode_ltrb(head_out.o2m_box, points, strides),
        head_out.o2o_cls,
        decode_ltrb(head_out.o2o_box, points, strides),
        points,
        gt_boxes,
        gt_labels,
        gt_mask,
        strides=strides,
    )


class _SyntheticDetectionDataset(Dataset[tuple[Tensor, Targets]]):
    """Two-sample synthetic detection dataset for the Lightning smoke run."""

    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        """Return the sample count."""
        return self._size

    def __getitem__(self, index: int) -> tuple[Tensor, Targets]:
        """Return one ``(image, Targets)`` sample with two instances."""
        return torch.randn(3, _IMG_SIZE, _IMG_SIZE), _synthetic_targets(2)


def test_training_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """A training step returns a finite scalar requiring grad and logs every component."""
    module = tiny_module()
    recorder = MagicMock()
    monkeypatch.setattr(module, "log", recorder)
    loss = module.training_step(_synthetic_batch(), 0)
    assert loss.ndim == 0
    assert bool(torch.isfinite(loss))
    assert loss.requires_grad
    logged_keys = {call.args[0] for call in recorder.call_args_list}
    assert logged_keys == _TRAIN_LOG_KEYS


def test_validation_step_logs_under_val_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A validation step returns a finite scalar and logs its total under ``val/``."""
    module = tiny_module()
    recorder = MagicMock()
    monkeypatch.setattr(module, "log", recorder)
    loss = module.validation_step(_synthetic_batch(), 0)
    assert bool(torch.isfinite(loss))
    logged_keys = {call.args[0] for call in recorder.call_args_list}
    assert "val/loss" in logged_keys


def test_configure_optimizers_musgd_over_all_parameters() -> None:
    """configure_optimizers returns a MuSGD covering every module parameter."""
    module = tiny_module()
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, MuSGD)
    optim_param_count = sum(len(group["params"]) for group in optimizer.param_groups)
    assert optim_param_count == len(list(module.parameters()))
    assert optim_param_count > 0


def test_alpha_property_delegates_to_loss() -> None:
    """The alpha property reads and writes the underlying DualBranchLoss weight."""
    module = tiny_module()
    module.alpha = 0.8
    assert module.alpha == 0.8
    assert module.loss.alpha == 0.8


class TestPadTargets:
    """Tests for ``pad_targets``."""

    def test_pads_ragged_batch_with_mask(self) -> None:
        """A ragged batch pads to N_max with a mask marking the real rows."""
        first = Targets(boxes=torch.tensor([[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 5.0, 5.0]]), labels=torch.tensor([1, 2]))
        second = Targets(boxes=torch.tensor([[2.0, 2.0, 6.0, 6.0]]), labels=torch.tensor([3]))
        boxes, labels, mask = pad_targets([first, second])
        assert boxes.shape == (2, 2, 4)
        assert mask.tolist() == [[True, True], [True, False]]
        assert labels.tolist() == [[1, 2], [3, 0]]
        assert torch.equal(boxes[1, 1], torch.zeros(4))

    def test_handles_empty_image(self) -> None:
        """An image with no instances contributes an all-False mask row."""
        annotated = Targets(boxes=torch.tensor([[0.0, 0.0, 4.0, 4.0]]), labels=torch.tensor([1]))
        boxes, _labels, mask = pad_targets([annotated, Targets.empty()])
        assert boxes.shape == (2, 1, 4)
        assert mask.tolist() == [[True], [False]]

    def test_all_empty_batch_zero_width(self) -> None:
        """A batch whose every image is empty yields N_max = 0 tensors."""
        boxes, labels, mask = pad_targets([Targets.empty(), Targets.empty()])
        assert boxes.shape == (2, 0, 4)
        assert labels.shape == (2, 0)
        assert mask.shape == (2, 0)


def test_invalid_task_raises() -> None:
    """An unsupported task name is rejected at construction."""
    with pytest.raises(ValueError, match="task must be one of"):
        tiny_module(task="pose")


def test_task_extra_loss_inert_zero_for_detection() -> None:
    """The task extra-loss dispatch returns a zero scalar for the one task with no extra term.

    ``segment`` and ``obb`` are excluded: WP-087 and WP-088 made their contributions
    live, and ``test_seg_training.py`` / ``test_obb_training.py`` cover them. Detection
    is the task whose total must stay *exactly* the dual detection loss, so the zero is
    asserted rather than assumed.
    """
    module = tiny_module(task="detect")
    images, targets = _synthetic_batch()
    gt_boxes, _, _ = pad_targets(targets)
    head_out = module(images)
    anchor_points, strides = module._anchor_grid(_IMG_SIZE, _IMG_SIZE, images.device)
    context = _StepContext(
        head_out=head_out,
        seg_out=None,
        targets=targets,
        gt_boxes=gt_boxes,
        gt_rboxes=None,
        gt_keypoints=None,
        gt_keypoint_vis=None,
        anchor_points=anchor_points,
        strides=strides,
        image_size=(_IMG_SIZE, _IMG_SIZE),
        masks=None,
    )

    extra = module._task_extra_loss(context, _dual_loss_output(module, images, targets), "train")

    assert extra.ndim == 0
    assert float(extra) == 0.0


def test_segment_task_with_zero_gains_trains_identically_to_detect() -> None:
    """With both segmentation gains at zero, a segment module reproduces the detection loss bit for bit.

    A ``segment`` module carries the head's mask coefficient stems and the three
    prototype/auxiliary branches, so its state dict is a strict superset of the
    ``detect`` one — hence ``strict=False``. Zeroing only the two WP-087 gains
    must leave *nothing* else different: any segmentation quantity that reached
    the detection terms by another route — a shared forward that mutates them, a
    re-run assignment — would break this equality while the ordinary gains hid it
    inside a larger total.
    """
    detect = tiny_module(task="detect")
    segment = DetectionLitModule(
        depth=0.34,
        width=0.25,
        max_channels=256,
        num_classes=_NUM_CLASSES,
        task="segment",
        mask_gain=0.0,
        semantic_gain=0.0,
    )
    missing, unexpected = segment.load_state_dict(detect.state_dict(), strict=False)
    assert not unexpected
    assert all(
        key.startswith(("head.o2m.coeff", "head.o2o.coeff", "proto_fusion.", "protonet.", "semantic."))
        for key in missing
    )
    batch = _synthetic_batch_with_polygons()
    detect.log = MagicMock()  # type: ignore[method-assign]
    segment.log = MagicMock()  # type: ignore[method-assign]
    detect_loss = detect.training_step(batch, 0)
    segment_loss = segment.training_step(batch, 0)
    assert torch.equal(detect_loss, segment_loss)


def test_fast_dev_run_smoke() -> None:
    """A fast_dev_run Trainer drives one train and one val batch end to end."""
    module = tiny_module()
    loader: DataLoader[tuple[Tensor, Targets]] = DataLoader(
        _SyntheticDetectionDataset(_BATCH_SIZE), batch_size=_BATCH_SIZE, collate_fn=collate_unpacked
    )
    trainer = Trainer(fast_dev_run=True, accelerator="cpu", logger=False, enable_progress_bar=False)
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.state.finished
    assert "train/loss" in trainer.callback_metrics
    assert "val/mAP" in trainer.callback_metrics


def test_val_map_metric_leaves_state_dict_unchanged() -> None:
    """The WP-077 val mAP metric adds no state_dict entries, so older checkpoints still load."""
    module = tiny_module()
    assert not [key for key in module.state_dict() if key.startswith("_val_")]


def test_compile_step_leaves_state_dict_keys_unchanged() -> None:
    """``compile_step=True`` keeps the pre-WP-087 key sequence: the flag names no submodule (WP-184)."""
    expected = tuple(_PRECHANGE_KEYS_FILE.read_text().split())

    keys = tuple(tiny_module(compile_step=True).state_dict())

    assert keys == expected
    assert len(keys) == 714


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        pytest.param("cpu", "cpu", id="cpu-stays-on-cpu"),
        pytest.param(
            "mps",
            "cpu",
            id="mps-has-no-float64-so-host",
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS backend not available"),
        ),
        pytest.param(
            "cuda",
            "cuda",
            id="cuda-goes-to-cuda",
            marks=requires_cuda,
        ),
    ],
)
def test_alpha_tensor_device_follows_what_can_hold_float64(device: str, expected: str) -> None:
    """The 0-d ``float64`` branch weight lands on ``device``, except MPS, where it stays on the host.

    ``torch.tensor(..., dtype=float64, device="mps")`` raises ``TypeError`` -- the MPS
    framework has no double -- which made every MPS training step fail at the first
    loss after WP-184 moved the weight from a Python float to a tensor input. A 0-d CPU
    tensor multiplies a device tensor through scalar promotion, so the host is the
    answer for MPS.
    """
    module = tiny_module()

    alpha = module._alpha_tensor(torch.device(device))

    assert alpha.device.type == expected
    assert alpha.dtype == torch.float64
    assert alpha.item() == module.alpha


def test_compile_step_on_cpu_trainer_warns_once_and_stays_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off CUDA the flag warns at the first ``setup`` only, keeps the bare method and the same loss."""
    reference = tiny_module()
    torch.manual_seed(0)  # same weights as ``reference``, the flag being the only difference
    module = tiny_module(compile_step=True)
    module._trainer = Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
    monkeypatch.setattr(reference, "log", MagicMock())
    monkeypatch.setattr(module, "log", MagicMock())
    images, targets = _synthetic_batch()

    with pytest.warns(UserWarning, match="ignored off CUDA") as record:
        module.setup("fit")
        module.setup("validate")
    total, _head_out, _seg_out = module._shared_step((images, targets), "train")
    expected, _head_out, _seg_out = reference._shared_step((images, targets), "train")

    assert len(record) == 1
    assert module._objective == module._detect_objective
    assert torch.equal(total, expected)


@pytest.mark.parametrize("flag", ["capture_network", "capture_loss"])
def test_ignored_capture_warning_names_setup_caller(flag: str) -> None:
    """An opted-in recording ``setup`` cannot activate warns at the line that called ``setup``.

    The activation check runs in a helper one frame below ``setup``, so its ``stacklevel``
    has to skip that frame too: pointing into the module instead would break a user's
    warning filter keyed on their own calling module. A detached module is never on
    CUDA, so either flag warns at ``fit``.
    """
    module = tiny_module(**{flag: True})

    with pytest.warns(UserWarning, match=f"{flag}=True is ignored outside") as record:
        module.setup("fit")

    assert [warning.filename for warning in record] == [__file__]


@pytest.mark.parametrize("flag", ["capture_network", "capture_loss"])
def test_ignored_capture_clears_activation_before_warning(flag: str) -> None:
    """A recording ``setup`` refuses is switched off even when its warning is raised as an error.

    A module reused across trainers can arrive at ``setup`` with the flag still on from a
    run that activated it. The activation is decided and stored before the warning is
    emitted, so a filter promoting that warning to an exception cannot leave the stale
    ``True`` behind for the next step to record under. A detached module is never on
    CUDA, so either flag is refused at ``fit``.
    """
    module = tiny_module(**{flag: True})
    setattr(module, f"_{flag}_active", True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match=f"{flag}=True is ignored outside"):
            module.setup("fit")

    assert getattr(module, f"_{flag}_active") is False


class TestHostSplit:
    """Tests for ``_host_split``, WP-163's batched device-to-host target transfer."""

    def test_returns_same_pieces_per_tensor_transfer_would(self) -> None:
        """One concatenated transfer splits back into exactly the per-image tensors."""
        tensors = [torch.rand(3, 4), torch.rand(0, 4), torch.rand(5, 4)]

        split = _host_split(tensors)

        assert len(split) == len(tensors)
        assert all(torch.equal(got, want.cpu()) for got, want in zip(split, tensors, strict=True))

    def test_preserves_dtype_and_trailing_shape(self) -> None:
        """Labels stay integral and keypoints keep their point and coordinate axes."""
        labels = [torch.tensor([1, 2]), torch.tensor([3])]
        keypoints = [torch.rand(2, 7, 3), torch.rand(1, 7, 3)]

        assert [piece.dtype for piece in _host_split(labels)] == [torch.int64, torch.int64]
        assert [tuple(piece.shape) for piece in _host_split(keypoints)] == [(2, 7, 3), (1, 7, 3)]

    def test_empty_batch_returns_no_pieces(self) -> None:
        """No targets means no transfer to make, rather than a ``torch.cat`` on an empty list."""
        assert _host_split([]) == ()

    def test_every_image_empty_splits(self) -> None:
        """A batch where no image carries a box yields one empty piece per image."""
        split = _host_split([torch.zeros(0, 4), torch.zeros(0, 4)])

        assert [tuple(piece.shape) for piece in split] == [(0, 4), (0, 4)]


def test_validation_ground_truth_matches_per_target_transfer() -> None:
    """WP-163's batched transfer feeds the mAP metric exactly what the per-target one did."""
    module = tiny_module()
    images, targets = _synthetic_batch()
    recorded: list[list[dict[str, Tensor]]] = []
    module._val_map.update = lambda preds, ground_truth: recorded.append(ground_truth)  # type: ignore[method-assign]

    module.validation_step((images, targets), 0)

    assert len(recorded) == 1
    for entry, target in zip(recorded[0], targets, strict=True):
        assert torch.equal(entry["boxes"], target.boxes.cpu())
        assert torch.equal(entry["labels"], target.labels.cpu())


def _guarded_module(task: str) -> DetectionLitModule:
    """Build a tiny module for any of the four tasks, filling in the keypoint schema.

    Examples:
        >>> _guarded_module("keypoints").task
        'keypoints'
    """
    extra = {"num_keypoints": _GUARD_NUM_KEYPOINTS} if task == "keypoints" else {}
    return tiny_module(task=task, **extra)


class TestDistributedValidationGuard:
    """Tasks whose epoch metric accumulates rank-local lists refuse to run multi-process.

    ``val/rotated_mAP`` and ``val/oks_mAP`` are scored off plain Python lists that nothing
    gathers, so on more than one process each rank would score its own shard and log it as
    the split's metric (audit A07/M-04). Average precision ranks every detection of the
    split against every other by confidence, so per-rank values cannot be averaged back into
    the right number — the run is refused at ``setup`` rather than allowed to log a
    plausible wrong one. ``detect``/``segment`` use torchmetrics metrics, which synchronise.
    """

    @pytest.mark.parametrize("task", [pytest.param("obb", id="obb"), pytest.param("keypoints", id="keypoints")])
    @pytest.mark.parametrize("stage", [pytest.param("fit", id="fit"), pytest.param("validate", id="validate")])
    def test_rejects_multi_process_runs(self, task: str, stage: str) -> None:
        """A two-process trainer is refused for both guarded tasks, naming the limitation.

        Both ``fit`` and ``validate`` are checked: the metric is computed in the validation
        epoch end either way, so guarding only the training entry point would leave
        ``lucid-yolo validate`` free to produce the same rank-local number.
        """
        module = _guarded_module(task)
        module._trainer = MagicMock(spec=Trainer, world_size=2)

        with pytest.raises(RuntimeError, match="single-process only"):
            module.setup(stage)

    @pytest.mark.parametrize(
        "task",
        [
            pytest.param("detect", id="detect"),
            pytest.param("segment", id="segment"),
            pytest.param("obb", id="obb"),
            pytest.param("keypoints", id="keypoints"),
        ],
    )
    def test_accepts_single_process_runs(self, task: str) -> None:
        """Every task sets up normally on one process, so the guard costs accepted runs nothing."""
        module = _guarded_module(task)
        module._trainer = MagicMock(spec=Trainer, world_size=1)

        module.setup("fit")

    @pytest.mark.parametrize("task", [pytest.param("detect", id="detect"), pytest.param("segment", id="segment")])
    def test_leaves_gathering_tasks_free_to_scale(self, task: str) -> None:
        """A multi-process ``detect``/``segment`` run is untouched: torchmetrics gathers for it."""
        module = _guarded_module(task)
        module._trainer = MagicMock(spec=Trainer, world_size=4)

        module.setup("fit")

    @pytest.mark.parametrize("task", [pytest.param("obb", id="obb"), pytest.param("keypoints", id="keypoints")])
    def test_predict_stage_exempt(self, task: str) -> None:
        """Prediction runs no validation protocol, so it keeps the multi-process path open."""
        module = _guarded_module(task)
        module._trainer = MagicMock(spec=Trainer, world_size=2)

        module.setup("predict")
