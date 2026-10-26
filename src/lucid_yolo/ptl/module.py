# SPDX-License-Identifier: Apache-2.0
"""Detection :class:`~pytorch_lightning.LightningModule` with task-conditional losses (WP-034).

:class:`DetectionLitModule` composes the WP-020…022 model stack (backbone, neck,
dual detection head) with the WP-028 :class:`~lucid_yolo.losses.dual_loss.DualBranchLoss`
and the WP-032 :class:`~lucid_yolo.optim.musgd.MuSGD` optimizer into the Phase 5
training loop. It runs under Lightning **automatic optimization** (blueprint D4):
``training_step`` returns the scalar total and Lightning owns the
backward/step/zero-grad cycle.

Batch contract:
    Every step consumes ``(images, list[Targets], masks)`` — the images stacked into
    one ``(B, C, H, W)`` float32 tensor, the ragged per-image
    :class:`~lucid_yolo.data.targets.Targets` as a length-``B`` list, and the
    per-image instance masks a segmentation loader rasterised in its workers
    (``None`` otherwise, which makes the step rasterise them itself). The
    datamodule ships the batch across the DataLoader worker boundary in a packed
    uint8 transport form and restores the float images, this list and those masks in
    its ``on_after_batch_transfer`` hook
    (see :class:`~lucid_yolo.ptl.datamodule.DetectionDataModule`), so the module
    never sees the packed form — it always receives the ragged list unchanged. The
    two-element ``(images, list[Targets])`` form is still accepted, so a hand-built
    feed needs no mask stack.

Forward and loss wiring:
    The head emits raw ``ltrb`` distances per anchor; the module derives the
    anchor grid for the batch's feature sizes (image size divided by the level
    strides ``(8, 16, 32)``, cached per size), decodes both branches' boxes with
    :func:`~lucid_yolo.models.heads.detect.decode_ltrb`, pads the ragged
    ``list[Targets]`` into dense ``(B, N_max, ...)`` ground-truth tensors with
    :func:`pad_targets`, and scores the pair with :class:`DualBranchLoss`. Every
    per-component term (``o2m``/``o2o`` box/cls/l1 and the combined total) is
    logged.

Task conditioning:
    ``task`` selects the active supervision, and every task's extra contribution
    flows through :meth:`DetectionLitModule._task_extra_loss`. ``"detect"`` adds
    nothing — a zero scalar, so the total is exactly the dual detection loss.

    ``"obb"`` (WP-088) is the one task that does not merely *add*. Term by term
    against the detection objective, with the enumeration
    :mod:`lucid_yolo.losses.oriented_loss` states in full:

    ================  ==========  ==================================================
    term              fate        how it is arranged here
    ================  ==========  ==================================================
    ``L_cls``         kept        Unchanged, at ``cls_gain``, over the same
                                  assignment.
    ``L_box``         replaced    :class:`DualBranchLoss` is constructed with
                                  ``box_gain = 0``, so its Complete-IoU term is
                                  computed (and logged, as a diagnostic) but enters
                                  no total; ``box_gain`` is spent instead on the
                                  rotated ProbIoU term of the assembled A44 box.
    ``L_l1``          replaced    Likewise ``l1_gain = 0`` inside the dual loss and
                                  ``l1_gain`` spent on the stride-normalized L1
                                  retargeted onto the rotated box's own
                                  ``(cx, cy, w, h)`` — the axis-aligned envelope
                                  target fights the rotated term at every non-zero
                                  ``theta`` (A50).
    ``L_angle``       added       R1 Eq. 15 at ``angle_gain`` (A22's ``0.25``).
    assignment        kept        One assignment, with rotated **candidacy** only
                                  (A25): ``gt_rboxes`` reaches the assigners and
                                  nothing else.
    ================  ==========  ==================================================

    Zeroing the two gains inside the dual loss rather than subtracting its box
    terms afterwards is deliberate: ``(c + b) - b`` is not ``c`` in floating point,
    so a subtraction would leave the classification term carrying the rounding of a
    number that is meant not to be there. The consequence to know about is that
    ``self.loss.box_gain`` reads ``0.0`` under ``task="obb"`` while
    ``hparams["box_gain"]`` carries the gain the rotated term actually uses — the
    hyperparameter names the *slot*, not the formula in it.

    ``"segment"`` (WP-087) adds ``mask_gain * mask + semantic_gain * semantic``
    (A38). :meth:`DetectionLitModule.forward_segmentation` produces the prototype
    maps and auxiliary logits alongside the detection output;
    :mod:`lucid_yolo.ptl.seg_targets` rasterizes the batch's polygons once, onto
    the prototype grid — in the loader workers when the loader was built for it,
    in the step otherwise, and checked against the prototypes either way; the
    assembled Eq. 7 masks
    of the positives are scored by
    :func:`~lucid_yolo.losses.mask_loss.instance_mask_loss` and the pooled
    per-class union by :func:`~lucid_yolo.losses.semantic_loss.semantic_aux_loss`.

    ``"keypoints"`` (WP-132) adds ``keypoint_gain * rle + oks_gain * oks`` — R14's
    residual log-likelihood over the head's point predictions, or whichever
    objective ``keypoint_loss`` put in that slot (WP-135), beside the
    Object-Keypoint-Similarity term R1 sec. 3.4.2 names as YOLO26's own pose
    objective (:class:`~lucid_yolo.losses.oks_loss.OKSLoss`, WP-178, A75; at
    ``oks_gain=0.0`` by default, so the shipped objective is the RLE term alone).
    Structurally it is the
    ``"segment"`` shape rather than the ``"obb"`` one: nothing is replaced, every
    detection term keeps its gain, and the pose terms ride on top. That follows
    from what the two tasks *are*. An oriented box is the same object the
    axis-aligned box describes, told better, so the two descriptions compete for
    the same slot; a keypoint is a second, independent thing to say about an
    object whose box is still wanted exactly as it was. So ``val/mAP`` also stays
    on (unlike under ``"obb"``, WP-102): the box figure a keypoints run logs is
    the box figure it is still training for.

    :class:`~lucid_yolo.losses.rle_loss.RLELoss` is the one loss in the project
    that carries **parameters** (R14's RealNVP flow), so it is held as a submodule
    — ``self.rle_loss`` — and reaches the optimizer, the checkpoint and the device
    placement through the ordinary ``nn.Module`` tree. It is constructed **after**
    the detection stages so that a same-seed ``"detect"`` module and a same-seed
    ``"keypoints"`` module draw identical backbone, neck, box-stem and class-stem
    weights: the flow's ``Linear`` layers consume RNG, and drawing them first
    would perturb every parameter that already existed. This is the same
    construction-order rule :class:`~lucid_yolo.models.heads.detect._DetectionBranch`
    states for its own optional stems, applied one level up.

    Which keypoint objective fills that slot is the ``keypoint_loss`` argument
    (WP-135). Its default, ``"rle"``, is the term just described and is what every
    existing config and checkpoint carries; ``"laplace_nll"`` substitutes
    :class:`~lucid_yolo.losses.keypoint_nll_loss.LaplaceNLLLoss`, R14 Table 7's
    flow-free control, which shares the call signature exactly so the step's own
    call site is unchanged. It is an experimental arm for WP-125's mechanism
    claim rather than a tuning choice, and it holds no parameters at all — so an
    ablation module's weights are bit-for-bit a same-seed detection module's,
    where a RLE module's diverge from the flow's first ``Linear``.

    ``num_keypoints`` is required for this task and has no default, because ``K``
    is a property of the dataset's annotation schema rather than of the method:
    COCO person is 17 points, and a hand or a vehicle-keypoint set is not. A
    default would be a silent claim about data the module has never seen, and the
    failure it buys is a head built for the wrong number of points, which
    broadcasts cleanly against nothing and raises far from its cause.

    The positives are **not** re-assigned, under any of these tasks:
    :class:`DualLossOutput`
    carries the two :class:`~lucid_yolo.assign.tal.AssignResult` values the box terms
    were scored against, and the mask, rotated, angle and keypoint terms alike gather
    their targets by those. A second assignment
    would be a second selection path, free to pair an anchor's mask with a
    different instance than its box — a defect no loss value reveals. Both
    branches' coefficients are supervised, each against its own assignment and
    weighted by the same ``alpha`` split :class:`DualBranchLoss` applies to the
    box terms, because the one-to-one coefficients are the ones the segmentation
    decode reads at inference.

Learning-rate schedule (A8, WP-072):
    :meth:`DetectionLitModule.configure_optimizers` pairs :class:`MuSGD` with a
    per-step :class:`~torch.optim.lr_scheduler.LambdaLR` running
    :func:`~lucid_yolo.optim.schedule.warmup_decay_factor` — a linear warmup
    over the first ``warmup_epochs`` epochs followed by a linear decay from
    ``lr`` to ``lr * lrf`` at the end of the run (A8; the Det-smoke attempt-1
    diagnosis showed the constant-LR deferral plateauing val loss). The
    schedule needs the trainer's step budget
    (``trainer.estimated_stepping_batches``), so a module with **no trainer
    attached** — direct ``configure_optimizers()`` calls in tests and tools —
    falls back to the bare constant-LR optimizer, as does an explicitly
    disabled schedule (``lrf >= 1`` with ``warmup_epochs <= 0``, the overfit
    recipe's setting) or a step-bounded run without ``max_epochs``.

Progressive-loss schedule (WP-035):
    :attr:`DetectionLitModule.alpha` delegates to the underlying
    :class:`DualBranchLoss` branch weight. The constructor ``alpha`` seeds it, and
    :meth:`DetectionLitModule.on_train_epoch_start` overwrites it once per epoch
    from a :class:`~lucid_yolo.losses.progressive.ProgressiveLossSchedule` — the
    linear ramp ``(0.8, 0.2) -> (0.1, 0.9)`` of R1 Eq. 2-3 (endpoints
    ``alpha_init``/``alpha_final``). When ``trainer.max_epochs`` is unset
    (``None`` or ``<= 0``, e.g. a step-bounded or ``fast_dev_run`` run) the ramp
    denominator is undefined, so the hook leaves ``alpha`` at its seeded value.

Provenance: R1 sec. 3.2, R1 Eq. 2-3, R1 Tables S2/S5. Assumptions: A8.
"""

from __future__ import annotations

import copy
import inspect
import warnings
from typing import TYPE_CHECKING, Any, cast

import torch
from pytorch_lightning import Callback, LightningModule
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.strategies import SingleDeviceStrategy
from torch import Tensor

from lucid_yolo.assign import make_anchor_points
from lucid_yolo.decode.common import BOX_CORNERS, MAX_DETECTIONS, SCORE_COLUMN
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.coco_eval import gather_keypoints
from lucid_yolo.eval.dota_eval import rotated_detections_to_predictions
from lucid_yolo.eval.segment_decode import decode_instance_masks
from lucid_yolo.losses import oriented_loss
from lucid_yolo.losses.dual_loss import BranchPrediction, DualBranchLoss, DualLossOutput
from lucid_yolo.losses.keypoint_nll_loss import LaplaceNLLLoss
from lucid_yolo.losses.mask_loss import instance_mask_loss
from lucid_yolo.losses.oks_loss import OKSLoss
from lucid_yolo.losses.oriented_loss import (
    DEFAULT_ROTATED_IOU_FORM,
    ROTATED_IOU_FORMS,
    OrientedLossOutput,
    oriented_branch_terms,
)
from lucid_yolo.losses.progressive import ProgressiveLossSchedule
from lucid_yolo.losses.rle_loss import RLELoss
from lucid_yolo.losses.semantic_loss import semantic_aux_loss
from lucid_yolo.models.build import SegmentOutput, build_detection_stages, build_segmentation_stages
from lucid_yolo.models.heads.detect import (
    DEFAULT_NUM_COEFFS,
    BranchName,
    BranchOutput,
    DualDetectionHead,
    DualHeadOutput,
    _DetectionBranch,
    decode_ltrb,
)
from lucid_yolo.models.heads.keypoint import decode_keypoints
from lucid_yolo.models.heads.obb import decode_rboxes, o2o_rotated_topk
from lucid_yolo.models.heads.proto import assemble_masks
from lucid_yolo.optim.musgd import MuSGD
from lucid_yolo.optim.schedule import warmup_decay_factor
from lucid_yolo.ptl._padding import (
    _BOX_DIM,
    _KEYPOINT_DIM,
    StepBatch,
    _host_split,
    _pad_mask_targets,
    _split_batch,
    pad_keypoints,
    pad_rboxes,
    pad_targets,
)
from lucid_yolo.ptl._task_strategies import _STRATEGIES
from lucid_yolo.ptl._task_terms import (
    _KEYPOINT_LOSSES,
    _blend,
    _keypoint_oks_sigmas,
    _KeypointPositives,
    _StepContext,
    normalize_keypoints_to_box,
)
from lucid_yolo.ptl._validation import _KeypointAP, _RotatedMAP, _ValidationMetric, _ValidationTargets
from lucid_yolo.ptl.loss_capture import (
    _LossCapture,
    _LossCaptureLifecycle,
    _LossInputs,
    _native_loss,
    _source_signature,
)
from lucid_yolo.ptl.network_capture import (
    _detection_stages,
    _module_breach,
    _NetworkCapture,
    _NetworkCaptureLifecycle,
)
from lucid_yolo.ptl.seg_targets import instance_mask_targets, scale_boxes_to_grid, semantic_target
from lucid_yolo.ptl.segment_head_capture import _head_breach as _segment_head_breach
from lucid_yolo.ptl.segment_head_capture import _SegmentHeadCapture, _SegmentHeadCaptureLifecycle
from lucid_yolo.tasks import TASK_NAMES, Task

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import FrameType

    from pytorch_lightning.utilities.types import OptimizerLRScheduler

    from lucid_yolo.assign.tal import AssignResult
    from lucid_yolo.data.targets import Targets

__all__ = ["DetectionLitModule", "pad_keypoints", "pad_rboxes", "pad_targets"]

#: Feature-level input-pixel strides of the P3/P4/P5 detection head (8, 16, 32).
_STRIDES: tuple[int, int, int] = (8, 16, 32)

#: Task names the module accepts, read off :class:`~lucid_yolo.tasks.Task` in registry
#: order. All four are wired: ``"detect"`` is the dual detection objective alone,
#: ``"segment"`` adds the WP-087 mask terms, ``"obb"`` swaps the two box terms for the
#: WP-088 rotated ones and adds the angle term, and ``"keypoints"`` adds the WP-132
#: residual log-likelihood term and the WP-178 OKS term.
_TASKS: tuple[str, ...] = TASK_NAMES

#: Column index of the integral class label within the A9 detection tuple.
_LABEL_COLUMN = 5

#: Detections per image whose masks are decoded for the epoch ``val/segm_mAP``.
#: COCO's own ``maxDets`` cap is 100, and the metric applies it anyway, so
#: decoding the fixed 300-row decoder output in full would cost two thirds of the
#: mask work to produce rows the evaluation then discards.
_VAL_SEGM_MAX_DET = 100


def _box_predictions(detections: Tensor) -> list[dict[str, Tensor]]:
    """Keep each image's scored detections, on the host, in the per-image form ``val/mAP`` reads.

    Args:
        detections: The ``(B, k, 6)`` decoded A9 batch; a row scoring zero is padding.

    Returns:
        One ``{"boxes", "scores", "labels"}`` dict per image, holding its kept rows.

    Examples:
        ```pycon
        >>> detections = torch.tensor([[[0.0, 0.0, 4.0, 4.0, 0.9, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        >>> _box_predictions(detections)[0]["labels"]
        tensor([2])

        ```
    """
    preds = []
    for image_detections in detections.cpu():
        kept = image_detections[image_detections[:, SCORE_COLUMN] > 0.0]
        preds.append(
            {
                "boxes": kept[:, :BOX_CORNERS],
                "scores": kept[:, SCORE_COLUMN],
                "labels": kept[:, _LABEL_COLUMN].long(),
            }
        )
    return preds


def _validate_task_arguments(task: str, num_keypoints: int | None, rotated_iou_form: str, keypoint_loss: str) -> None:
    """Refuse a :class:`DetectionLitModule` argument set that names nothing buildable.

    The constructor's own checks, held apart from it so the flat hyperparameter
    surface WP-038 configures from YAML stays one readable block of assignments.
    Every rejection is a misconfiguration that would otherwise surface far from its
    cause — a head built for the wrong point count broadcasts cleanly against nothing.

    Args:
        task: The supervision task; must be one of ``_TASKS``.
        num_keypoints: The point count, required under ``"keypoints"`` and refused
            under every other task (the class docstring argues both directions).
        rotated_iou_form: Must name one of R17's two rotated-IoU losses.
        keypoint_loss: Must name one of the ``_KEYPOINT_LOSSES`` objectives.

    Raises:
        ValueError: On any of the five rejections the class docstring lists.

    Examples:
        ```pycon
        >>> _validate_task_arguments("detect", None, "hellinger", "rle")
        >>> _validate_task_arguments("keypoints", None, "hellinger", "rle")
        Traceback (most recent call last):
            ...
        ValueError: task='keypoints' needs an explicit num_keypoints: ...

        ```
    """
    if task not in _TASKS:
        raise ValueError(f"task must be one of {_TASKS}, got {task!r}")
    if rotated_iou_form not in ROTATED_IOU_FORMS:
        raise ValueError(f"rotated_iou_form must be one of {sorted(ROTATED_IOU_FORMS)}, got {rotated_iou_form!r}")
    if keypoint_loss not in _KEYPOINT_LOSSES:
        raise ValueError(f"keypoint_loss must be one of {tuple(_KEYPOINT_LOSSES)}, got {keypoint_loss!r}")
    if task == "keypoints" and num_keypoints is None:
        raise ValueError(
            "task='keypoints' needs an explicit num_keypoints: the point count is a property of the "
            "dataset's annotation schema (COCO person is 17), so there is no default to fall back on"
        )
    if task != "keypoints" and num_keypoints is not None:
        raise ValueError(
            f"num_keypoints={num_keypoints} is meaningless for task={task!r}: only the 'keypoints' "
            f"task builds the point stem, so this module would predict no points at all while "
            f"save_hyperparameters records a point schema its checkpoint cannot honour. Set "
            f"task='keypoints' to train points, or drop num_keypoints"
        )


# The private executor keyword belongs to these native dispatch implementations.
_POSE_STEM_DISPATCH = (
    (DualDetectionHead, "forward", DualDetectionHead.forward),
    (DualDetectionHead, "forward_branch", DualDetectionHead.forward_branch),
    (_DetectionBranch, "forward", _DetectionBranch.forward),
    (_DetectionBranch, "_keypoints", _DetectionBranch._keypoints),
    (_DetectionBranch, "_base_predictions", _DetectionBranch._base_predictions),
)


def _head_topology(head: torch.nn.Module) -> tuple[tuple[object, ...], ...]:
    """Identify the head composition as constructed, whose dense outputs use every shared feature.

    The fingerprint is taken when the module is built, so the certificate it feeds holds
    the head unchanged since construction: a class patched before then is the reference.
    Pose and segmentation both replay shared features into this head, so both compare
    against the same fingerprint.
    """
    return tuple(
        (name, type(module), type(module).forward, getattr(type(module), "forward_branch", None))
        for name, module in head.named_modules()
    )


def _head_topology_breach(head: torch.nn.Module, topology: tuple[tuple[object, ...], ...], label: str) -> str | None:
    """Name a head changed or hooked since construction, which replayed shared features cannot feed.

    Args:
        head: The prediction head consuming the replayed shared features.
        topology: The head's :func:`_head_topology` fingerprint taken at construction.
        label: The task named in the reason, such as ``"pose"`` or ``"segmentation"``.

    Returns:
        The retirement reason, or ``None`` while the head is still the one constructed.
    """
    if _head_topology(head) != topology:
        return f"the {label} head topology or class forward changed"
    if _module_breach(head.modules()) is not None:
        return f"a {label} head module is hooked, overridden or compiled"
    return None


def _pose_head_breach(head: torch.nn.Module, topology: tuple[tuple[object, ...], ...]) -> str | None:
    """Keep altered head control flow outside a recording of its shared inputs."""
    if any(getattr(cls, name) is not native for cls, name, native in _POSE_STEM_DISPATCH):
        return "the native pose stem dispatch changed"
    return _head_topology_breach(head, topology, "pose")


def _validate_replay_options(
    task: str, compile_step: bool, capture_network: bool, capture_loss: bool, capture_segment_heads: bool
) -> None:
    """Reject replay options outside the objectives they record.

    Network replay records the detection, pose, oriented or segmentation stages; loss replay
    records the detection objective shared by detection and pose; segment-head replay records
    the segmentation prediction groups only. Segment-head replay refuses the compiler although
    ``compile_step`` leaves the segment objective eager today, so an accepted configuration
    keeps its meaning once that objective is compiled.

    Raises:
        ValueError: If network replay is combined with a task other than detection,
            keypoints, oriented detection or segmentation, loss replay with a task other than
            detection or keypoints, segment-head replay with a task other than segmentation, or
            any of them with the compiler (segment-head replay included, for forward
            compatibility with a compiled segment objective).
    """
    if capture_network and (task not in ("detect", "keypoints", "obb", "segment") or compile_step):
        raise ValueError(
            "capture_network=True requires task='detect', 'keypoints', 'obb' or 'segment', and compile_step=False"
        )
    if capture_loss and (task not in ("detect", "keypoints") or compile_step):
        raise ValueError("capture_loss=True requires task='detect' or 'keypoints', and compile_step=False")
    if capture_segment_heads and (task != "segment" or compile_step):
        raise ValueError(
            "capture_segment_heads=True requires task='segment' and compile_step=False; the compiler is refused "
            "even while the segment objective stays eager, so this configuration keeps its meaning once it is compiled"
        )


def _oriented_helpers() -> tuple[object, ...]:
    """Read the decode, oriented-loss and strategy helpers the native objective calls.

    Resolved at call time, so the import-time call that builds :data:`_ORIENTED_HELPERS`
    keeps the originals and every later call sees the current bindings.
    """
    return (
        decode_ltrb,
        decode_rboxes,
        oriented_branch_terms,
        vars(oriented_loss)["square_angle_loss"],
        oriented_loss._no_ground_truth,
        _STRATEGIES[Task.OBB],
        type(_STRATEGIES[Task.OBB]).extra_loss,
    )


def _oriented_replay_breach(module: DetectionLitModule) -> str | None:
    """Keep full-network replay on the native objective that connects all six outputs.

    The native empty-target path retains zero gradients for the angle stems. An
    overridden objective may leave them absent instead, which a recorded backward
    would materialize. Changed consumers therefore keep their complete native call.
    """
    for name, expected in _ORIENTED_METHODS:
        method = getattr(module, name)
        if getattr(method, "__func__", method) is not expected:
            return "the native oriented objective is overridden"
    if module._strategy is not _STRATEGIES[Task.OBB]:
        return "the native oriented objective is overridden"
    if _oriented_helpers() != _ORIENTED_HELPERS:
        return "a native oriented objective helper changed"
    return None if _native_loss(module.loss) else "the native loss or an assignment implementation changed"


class DetectionLitModule(LightningModule):
    """Lightning detector: model stack, dual-branch loss, and MuSGD (WP-034).

    Composes :class:`~lucid_yolo.models.backbone.DetectionBackbone`,
    :class:`~lucid_yolo.models.neck.DetectionNeck`, and
    :class:`~lucid_yolo.models.heads.detect.DualDetectionHead` through
    :func:`~lucid_yolo.models.build.build_detection_stages` — the same factory
    :class:`~lucid_yolo.models.build.Detector` uses, so the model the fidelity
    gate measures and the model training runs cannot drift apart (WP-087). It
    scores their dense predictions with
    :class:`~lucid_yolo.losses.dual_loss.DualBranchLoss`, and optimizes under
    Lightning automatic optimization (D4) with
    :class:`~lucid_yolo.optim.musgd.MuSGD`. See the module docstring for the A8
    constant-LR deferral, the ``task`` conditioning, and the :attr:`alpha` seam.

    Args:
        depth: Depth multiplier scaling per-stage repeat counts of the backbone
            and neck.
        width: Width multiplier scaling channel counts.
        max_channels: Channel cap applied before the width multiply.
        num_classes: Number of object classes the head predicts.
        task: Supervision task; one of ``"detect"``, ``"segment"``, ``"obb"`` or
            ``"keypoints"``, all four active. ``"segment"`` additionally builds the
            head's mask-coefficient stems and the three
            :func:`~lucid_yolo.models.build.build_segmentation_stages` branches as
            ``proto_fusion``/``protonet``/``semantic``, and supervises them with the
            two mask terms below. ``"obb"`` instead builds both head branches'
            orientation stems (``predict_angle``, A20) and swaps the two box terms
            for their rotated counterparts (see the module docstring's table).
            ``"keypoints"`` builds both branches' point stems (``num_keypoints``,
            WP-122) and adds the R14 residual log-likelihood term at
            ``keypoint_gain`` and the OKS term at ``oks_gain``, leaving every
            detection term as it was. Defaults to ``"detect"``.
        num_keypoints: Point count ``K`` both head branches predict under
            ``task="keypoints"``; **required** for that task and refused for every
            other one. It has no default on purpose — ``K`` describes the dataset's
            annotation schema, not the method (the module docstring argues the case).
            The two directions are both errors on purpose: a missing count under
            ``"keypoints"`` has nothing to guess, and a count under any other task
            names a schema the head builds no stem for, so the module would train
            no points while ``save_hyperparameters`` recorded ``K`` for a reader to
            trust later.
        lr: Base learning rate for MuSGD (``lr0``; the A8 schedule decays from
            it). Defaults to ``0.01``.
        lrf: Final LR fraction of the A8 linear decay — the LR ends at
            ``lr * lrf``. ``>= 1`` together with ``warmup_epochs <= 0`` disables
            the schedule (constant LR). Defaults to ``0.01``.
        warmup_epochs: Length of the opening linear LR warmup, in epochs
            (fractions allowed). ``0`` disables warmup. Defaults to ``3.0``.
        momentum: MuSGD momentum coefficient. Defaults to ``0.95``.
        weight_decay: Decoupled weight decay (matrix parameters only). Defaults to
            ``5e-4``.
        w_muon: Additive gain on the MuSGD Muon branch. Defaults to ``0.5``.
        w_sgd: Additive gain on the MuSGD SGD branch. Defaults to ``0.5``.
        box_gain: Gain on the **IoU term** of both loss branches — the axis-aligned
            Complete-IoU under ``"detect"``/``"segment"``, and under ``"obb"`` the
            rotated ProbIoU that replaces it in the same slot (A49; the module
            docstring explains why the slot rather than the formula is what the name
            refers to). Defaults to ``7.5``.
        cls_gain: Classification-term gain shared by both branches. Defaults to ``0.5``.
        l1_gain: Gain on the **L1 box term** of both branches: against the
            axis-aligned target under ``"detect"``/``"segment"``, and under ``"obb"``
            against the rotated box's own ``(cx, cy, w, h)`` (A50). Defaults to ``6.0``.
        alpha: Initial one-to-many branch weight seeding the dual loss (used
            before training and whenever the epoch schedule is inactive).
            Defaults to ``0.5``.
        alpha_init: One-to-many branch weight the progressive schedule sets on the
            first epoch (branch weights ``(0.8, 0.2)``). Defaults to ``0.8``.
        alpha_final: One-to-many branch weight the schedule ramps to on the last
            epoch (branch weights ``(0.1, 0.9)``). Defaults to ``0.1``.
        mask_gain: Weight on the instance-mask term under ``task="segment"``,
            ignored otherwise (A38). Defaults to ``2.5``.
        semantic_gain: Weight on the auxiliary semantic term under
            ``task="segment"``, ignored otherwise (A38). Defaults to ``0.5``.
        angle_gain: Weight on R1 Eq. 15's square-object angle term under
            ``task="obb"``, ignored otherwise. Defaults to ``0.25`` (A22 — R1 does
            not state one, and :func:`~lucid_yolo.losses.angle_loss.square_angle_loss`
            returns the term pre-gain precisely so this caller owns it). Lowered
            from ``1.0`` by WP-093, which measured the term destabilizing angle
            regression on **elongated** targets: at ``1.0`` the oriented overfit
            cleared its floor on 1 of 5 seeds with a 0.667 spread, at ``0.25`` on
            4 of 5 with 0.096. See A22 for the dose-response and why the remedy is
            this gain rather than ``lambda``, which R1 does state.
        rotated_iou_form: Which of R17's two rotated-IoU losses the ``"obb"`` box
            term uses, ``"hellinger"`` (bounded, the A49 default) or
            ``"bhattacharyya"`` (unbounded). Ignored otherwise.
        keypoint_gain: Weight on the R14 residual-log-likelihood term under
            ``task="keypoints"``, ignored otherwise. Defaults to ``1.0`` (A68), which
            is **not a measured value**: R14 trains RLE as an objective in its own
            right and never states a weight for it beside a detection loss, and no
            allowlisted source covers the pair. ``1.0`` is therefore the neutral
            placeholder — the term as the paper writes it — pending the same
            dose-response treatment WP-093 gave ``angle_gain`` under A22. Expect it
            to want lowering, and by orders rather than factors: the term is a
            negative log-likelihood of residuals standardized by a sigmoid ``sigma``
            in ``(0, 1)``, while
            :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints` works in
            absolute input pixels, so at initialisation it is measured two orders
            above the whole detection total rather than beside it. A68 cites this
            module's ``keypoint_gain=0`` bit-exactness test as its validation: at
            zero the term must leave the detection objective untouched, whatever
            scale it takes at one.
        keypoint_loss: Which keypoint objective ``task="keypoints"`` supervises with,
            ignored otherwise. Defaults to ``"rle"``, R14's full residual
            log-likelihood (:class:`~lucid_yolo.losses.rle_loss.RLELoss`) — every
            config, checkpoint and accepted figure this project has produced was
            produced under it, and the default is what keeps that true. The
            alternative, ``"laplace_nll"``
            (:class:`~lucid_yolo.losses.keypoint_nll_loss.LaplaceNLLLoss`), drops the
            normalizing flow and keeps everything else, which is R14 Table 7's own
            published ablation: the two rows are "RLE" at ``70.5`` AP and "Laplace,
            learnable variance" at ``67.4`` AP on COCO. It exists for WP-125, whose
            acceptance is RLE's *mechanism* claim rather than an absolute pose
            number — a claim of that shape needs a paired run differing only in the
            mechanism, and one figure on its own cannot settle it. The switch is
            therefore an experimental control, not a tuning knob: a production pose
            run has no reason to leave the default.
        oks_gain: Weight on the Object-Keypoint-Similarity term
            (:class:`~lucid_yolo.losses.oks_loss.OKSLoss`, WP-178) under
            ``task="keypoints"``, ignored otherwise. Defaults to ``0.0`` (A75), which
            is a deliberate *off* rather than a measured value. R1 sec. 3.4.2 states
            that YOLO26's pose training uses an OKS-based loss extended with RLE, and
            R1 Table 9 ablates the pair's weights ``(w_OKS, w_RLE)`` on COCO
            keypoints val for YOLO26s — ``(48, 0)`` 61.5, ``(48, 1)`` 60.8,
            ``(24, 1)`` 63.0, ``(24, 2)`` 62.5, ``(12, 1)`` 62.6, ``(0, 1)`` 61.9,
            ``(0, 2)`` 62.4 mAP — so the published target is ``(24, 1)``. But R1
            gives no formula for its OKS loss and its citation is off this project's
            allowlist, so the term here is derived from R12's metric definition
            (A75), and a weight measured under a loss this project had to derive is
            not a measured value here. Shipping ``24.0`` as the default would make
            an untested number look like evidence while being none — A68's own
            reasoning for ``keypoint_gain`` — and would silently move every existing
            keypoints config off the objective its accepted figures were produced
            under. ``0.0`` keeps that objective bit-exact (the module's
            ``oks_gain=0`` bit-exactness test is its validation); the pre-gain term
            is still computed and logged as ``train/oks`` so a dose-response run
            against ``(24, 1)`` has its reading before it moves the gain.
        compile_step: Whether to :func:`torch.compile` the detection objective —
            forward, both decodes, assignment and the dual loss as **one** Inductor
            region (WP-184, D24). Defaults to ``False``. Honoured on a CUDA
            accelerator under ``task="detect"`` only; any other accelerator or task
            warns once at :meth:`setup` and stays eager (MPS Inductor crashes in
            ``convolution_backward``, and the three other tasks' extra terms still
            read tensors on the host). Measured on an RTX PRO 6000 at variant ``n``,
            COCO, batch 16, deterministic: the training step goes from 74.7 ms to
            60.9 ms (x1.23) against eager, for 110-140 s of compilation per process
            plus a handful of recompiles per run (the ``no_grad`` evaluation graph, and
            one each for a batch with no ground truths, one per image, or one
            image — Dynamo specialises sizes 0 and 1 whatever is marked). It
            compiles the bound method rather than the module, so parameter names,
            the state dict, EMA, checkpoints and export are untouched. Whether the
            compiled objective trains to the same mAP as eager is A78, still open —
            which is why the default is off.
        capture_orthogonalization: Enable MuSGD's optional CUDA graph replay of
            its existing Newton--Schulz calls. Defaults to ``False``. Shapes,
            strides, dtype, device, step count and precision select each recording;
            this flag alone leaves learning rates, momentum and parameters outside the graphs.
            Extra workspace and setup can outweigh the saving on short runs.
        capture_network: Replay fixed detection, keypoint, oriented or segmentation network stages' CUDA
            forward and backward with :func:`torch.cuda.make_graphed_callables`. Defaults to ``False``
            because capture pays setup and retains a graph pool. Requires ``task="detect"``,
            ``task="keypoints"``, ``task="obb"`` or ``task="segment"``, ``compile_step=False`` and
            single-device CUDA automatic optimization.
            Pose records backbone, neck and each prediction stem independently, preserving
            absent point-stem gradients when a batch supplies no positives. Pose training stays eager
            with a fully frozen backbone and neck, or when
            the head's topology or class forward changed since construction or a head
            module is hooked, overridden or compiled. The trainable set is fixed per fit
            at the first recording: a fully frozen start, or a later freeze or unfreeze
            of a recorded parameter, keeps the rest of that fit eager after one warning.
            Each pose stem separately retires its recording if its trainable set or mode changes,
            and replacing a head module retires every recording before shared features run.
            Oriented training records the complete network, including both angle stems,
            and retains the native classification and oriented objective. An overridden
            objective keeps it eager. Detection, pose and oriented training
            stay eager when the LightningModule carries a hook, overrides
            ``forward`` or is compiled. Segmentation records only backbone and neck,
            leaving all predictions independent; ``capture_segment_heads`` can replay
            those predictions separately. Segmentation retires replay when the head
            topology/dispatch changes, a head hook is registered, or ``forward_segmentation``
            is overridden or compiled.
            Assignment, losses, schedules, optimizer, EMA and evaluation stay eager.
            One image signature is recorded per fit; partial batches stay eager and
            changed storage, modes, hooks or precision retire the recording. Warmup
            restores buffers and RNG, and the fit callbacks discard recordings on
            completion or failure. Higher-order differentiation is unsupported.
        capture_loss: Replay the existing detection assignment and loss forward/backward,
            for detection or pose, on single-device CUDA automatic training. Default off:
            recording pays setup and retains a graph pool. Masked target capacity grows
            geometrically; the method, label checks, live branch schedule and optimizer stay
            unchanged. Requires ``task="detect"`` or ``task="keypoints"``, and
            ``compile_step=False``. Pose also batches the existing RLE densities, keeping
            visibility, separate means and branch weights. Batched parameter-gradient
            summation can change rounding. Can combine with ``capture_network``,
            ``capture_orthogonalization`` and ``capture_updates``; it records no optimizer
            update itself, so the optimizer step is whatever those optimizer options select.
        capture_updates: Enable MuSGD's optional replay of the existing update preparation,
            including momentum, followed by a native batched parameter write at the current
            learning rate. Defaults to ``False``. Uses private gradient storage and live
            parameters and momentum; unsupported signatures stay native. Like
            ``capture_orthogonalization``, this has setup and retained-workspace costs.
        capture_segment_heads: Record independent CUDA segmentation prediction groups, including first-order backward.
            Defaults to ``False``. Requires ``task="segment"`` and ``compile_step=False``, PyTorch 2.9 or newer, BF16
            mixed precision and single-device automatic optimization; gradient accumulation replays too. CPU, MPS,
            FP32, FP16 and validation keep native execution. ``compile_step=True`` is refused rather than ignored,
            although it leaves the segment objective eager today, so an accepted configuration keeps its meaning once
            that is compiled.

    Raises:
        ValueError: If ``task`` is not one of ``"detect"``, ``"segment"``, ``"obb"``,
            ``"keypoints"``, if ``rotated_iou_form`` is not a known rotated-IoU form,
            if ``capture_network`` is combined with a task other than detection, keypoints,
            oriented detection or segmentation, ``capture_loss`` with a task other than detection
            or keypoints, ``capture_segment_heads`` with a non-segmentation task, or any of them
            with ``compile_step=True`` (for ``capture_segment_heads``, forward compatibility
            with a compiled segment objective), if ``keypoint_loss`` is not a known keypoint
            objective, if ``task="keypoints"`` was asked for without a ``num_keypoints``, or if
            a ``num_keypoints`` was given for a task that builds no point stem.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4)
        >>> images = torch.zeros(1, 3, 160, 160)
        >>> targets = [Targets(boxes=torch.tensor([[8.0, 8.0, 40.0, 40.0]]), labels=torch.tensor([1]))]
        >>> loss = module.training_step((images, targets), 0)  # doctest: +SKIP
        >>> bool(torch.isfinite(loss))  # doctest: +SKIP
        True

        ```
    """

    def __init__(  # noqa: PLR0913 — flat hyperparameter surface is deliberate: WP-038 LightningCLI configures each knob directly from YAML (only 4 are required positionals)
        self,
        depth: float,
        width: float,
        max_channels: int,
        num_classes: int,
        task: str = "detect",
        num_keypoints: int | None = None,
        *,
        lr: float = 0.01,
        lrf: float = 0.01,
        warmup_epochs: float = 3.0,
        momentum: float = 0.95,
        weight_decay: float = 5e-4,
        w_muon: float = 0.5,
        w_sgd: float = 0.5,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        l1_gain: float = 6.0,
        alpha: float = 0.5,
        alpha_init: float = 0.8,
        alpha_final: float = 0.1,
        mask_gain: float = 2.5,
        semantic_gain: float = 0.5,
        angle_gain: float = 0.25,
        rotated_iou_form: str = DEFAULT_ROTATED_IOU_FORM,
        keypoint_gain: float = 1.0,
        keypoint_loss: str = "rle",
        oks_gain: float = 0.0,
        compile_step: bool = False,
        capture_orthogonalization: bool = False,
        capture_network: bool = False,
        capture_updates: bool = False,
        capture_loss: bool = False,
        capture_segment_heads: bool = False,
    ) -> None:
        super().__init__()
        _validate_task_arguments(task, num_keypoints, rotated_iou_form, keypoint_loss)
        _validate_replay_options(task, compile_step, capture_network, capture_loss, capture_segment_heads)
        self.save_hyperparameters()
        task = Task(task).value
        self._capture_loss, self._capture_loss_active = capture_loss, False
        self._capture_network, self._capture_network_active = capture_network, False
        self._network_capture, self._loss_capture = _NetworkCapture(), _LossCapture()
        self._strategy, self._task, self._num_keypoints = _STRATEGIES[Task(task)], task, num_keypoints
        self._compile_step: bool = compile_step
        self._capture_orthogonalization, self._capture_updates = capture_orthogonalization, capture_updates
        #: The detection objective `_shared_step` calls: the bare bound method until
        #: `setup` swaps in its `torch.compile` wrapper (WP-184), and the bare method
        #: for good on every backend and task the wrapper is refused for. Function-level
        #: rather than `torch.compile(self)`: a compiled module registers as
        #: `_orig_mod.<name>`, which would rename every state-dict key.
        self._objective: Callable[..., tuple[DualHeadOutput, DualLossOutput]] = self._detect_objective
        #: Whether `setup` has already decided on `compile_step` — `fit` then `validate`
        #: calls it twice, and the decision (compile, or warn and stay eager) is made once.
        self._compile_step_resolved: bool = False
        self._objective_compiled: bool = False
        self._mask_gain: float = mask_gain
        self._semantic_gain: float = semantic_gain
        self._angle_gain: float = angle_gain
        self._keypoint_gain: float = keypoint_gain
        self._keypoint_loss: str = keypoint_loss
        self._oks_gain: float = oks_gain
        self._rotated_iou_form: str = rotated_iou_form
        #: Under ``"obb"`` the two box gains move out of the dual detection loss and
        #: onto the rotated terms; the dual loss is then constructed with zeros in
        #: their place so its axis-aligned box terms enter no total (module docstring).
        self._rbox_gain: float = box_gain
        self._rl1_gain: float = l1_gain
        self._lr = lr
        self._lrf = lrf
        self._warmup_epochs = warmup_epochs
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._w_muon, self._w_sgd = w_muon, w_sgd

        #: Mask coefficients are built only for the segmentation task, so a
        #: ``"detect"`` module's head — and therefore its state dict — is exactly
        #: what it was before the segmentation branches existed (the Det-smoke
        #: checkpoint still loads).
        num_coeffs = DEFAULT_NUM_COEFFS if task == "segment" else None
        #: The oriented task opts into the A20 angle stems through the **same** factory
        #: the detection and segmentation tasks use, and the stages stay flat attributes.
        #: Holding an `OrientedDetector` here instead would prefix every state-dict key
        #: with its attribute name and invalidate the accepted checkpoints — the WP-087
        #: finding, which `test_module_composition.py` pins for detection and
        #: `test_obb_training.py` now pins for the oriented head's own keys.
        self.backbone, self.neck, self.head = build_detection_stages(
            depth,
            width,
            max_channels,
            num_classes,
            num_coeffs=num_coeffs,
            predict_angle=task == "obb",
            num_keypoints=num_keypoints if task == "keypoints" else None,
        )
        #: The head as constructed, for the two tasks that replay shared features into it.
        self._head_topology = _head_topology(self.head) if task in ("keypoints", "segment") else ()
        self._init_segmentation(task, num_classes, capture_segment_heads)
        #: The keypoint objective, held as a submodule because R14's flow is the only
        #: loss in the project carrying parameters — hence a submodule rather than a
        #: call. Constructed **after** the stages above so those ``Linear`` layers draw
        #: their RNG last and leave every already-existing parameter with the value a
        #: same-seed detection module gives it; ``None`` for any other task, whose
        #: state dict must stay exactly what it was. WP-135's ``LaplaceNLLLoss``
        #: alternative has no parameters and draws no RNG at all, so the ordering is
        #: moot for it and kept only because one construction site serves both.
        #:
        #: The attribute keeps the name ``rle_loss`` under either objective: it is an
        #: ``nn.Module`` attribute, so it prefixes this loss's state-dict keys, and
        #: renaming it would stop every keypoints checkpoint produced before WP-135
        #: loading — the same key-stability rule the stages above are kept flat for.
        self.rle_loss: RLELoss | LaplaceNLLLoss | None = (
            _KEYPOINT_LOSSES[keypoint_loss]() if task == "keypoints" else None
        )
        #: The OKS term (WP-178, A75), beside the RLE term rather than in its slot: R1
        #: sec. 3.4.2 composes the two. Its sigma table is a **non-persistent** buffer
        #: and it holds no parameters, so a keypoints module's ``state_dict`` — and
        #: every pose checkpoint produced before the term existed — is exactly what it
        #: was; the sigmas are the ``val/oks_mAP`` evaluator's own, through
        #: :func:`_keypoint_oks_sigmas`, so the loss pulls towards what the metric scores.
        self.oks_loss: OKSLoss | None = OKSLoss(_keypoint_oks_sigmas(num_keypoints)) if task == "keypoints" else None
        oriented = task == "obb"
        self.loss = DualBranchLoss(
            box_gain=0.0 if oriented else box_gain,
            cls_gain=cls_gain,
            l1_gain=0.0 if oriented else l1_gain,
            alpha=alpha,
        )
        self._loss_schedule = ProgressiveLossSchedule(alpha_init=alpha_init, alpha_final=alpha_final)

        #: E2E decoder + epoch mAP over the one-to-one branch (WP-077). Neither
        #: carries parameters and the metric's states are non-persistent, so the
        #: module's ``state_dict`` — and older checkpoints — are unaffected. Both
        #: metrics come from :func:`~lucid_yolo.ptl.coco_backend.build_mean_average_precision`,
        #: which is the stock metric with its epoch-to-COCO conversion reading
        #: scores once per image rather than once per detection (WP-165). Both also
        #: take :data:`~lucid_yolo.eval.coco_eval.COCO_RECALL_GRID` rather than
        #: torchmetrics' own float32 ``linspace``, for the reason that constant
        #: states and because the alternative is worse than a small bias: the report
        #: path pins the grid, so leaving it unpinned here would have the epoch
        #: figure and the acceptance figure integrate over different recall points
        #: while both were logged as ``mAP`` (audit M-03). The two remain proxies of
        #: each other for the reasons at :meth:`on_validation_epoch_end` — different
        #: coordinate frame, different detection cap — but no longer for a third,
        #: silent, arithmetic reason.
        self._val_decoder = TopKDecoder()
        self._val_map = self._strategy.build_box_metric()
        self._val_map.warn_on_many_detections = False

        #: Epoch mask mAP, for ``"segment"`` only (WP-087). A second metric rather
        #: than ``iou_type=("bbox", "segm")`` on :attr:`_val_map`, because the two
        #: are scored in **different frames** — boxes in letterbox pixels, masks on
        #: the prototype grid — and one metric holding both would report each
        #: instance's area under whichever frame torchmetrics picked, silently
        #: mis-bucketing the small/medium/large splits. Split in two, each metric's
        #: inputs are self-consistent. ``None`` for a detection module, whose
        #: validation must not pay for mask machinery it has no branch for.
        self._val_segm = self._strategy.build_mask_metric()
        if self._val_segm is not None:
            self._val_segm.warn_on_many_detections = False
        #: Whether any batch fed :attr:`_val_segm` this epoch — ``compute`` on an
        #: untouched metric raises, and a loader without mask targets never feeds it.
        self._val_segm_seen = False

        #: Whole-epoch metrics gather complete images before applying the existing protocols.
        #: Dataset-position order matches the sequential validation loader, including score ties.
        self._val_rotated = _RotatedMAP(dataset_order=True) if task == "obb" else None
        self._val_keypoints = (
            _KeypointAP(_keypoint_oks_sigmas(num_keypoints), dataset_order=True) if task == "keypoints" else None
        )

        #: Per-image-size cache of ``(anchor_points, stride_per_anchor)`` on CPU.
        self._anchor_cache: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}

    def save_hyperparameters(
        self,
        *args: Any,
        ignore: Sequence[str] | str | None = None,
        frame: FrameType | None = None,
        logger: bool = True,
    ) -> None:
        """Record Lightning hyperparameters, with ``task`` in its plain spelling, on every save.

        The plain spelling, even when a :class:`~lucid_yolo.tasks.Task` member was passed:
        a checkpoint loads under ``torch.load(weights_only=True)``, whose restricted
        unpickler refuses a ``Task`` in ``hparams["task"]``. Normalizing the constructor's
        local before the call is not enough, because the save merges every subclass
        ``__init__`` frame's arguments over this one's, ``**kwargs`` contents included, and
        correcting the record once after the constructor's own call is not enough either,
        because a subclass may save again after ``super().__init__`` returns. So every save
        corrects both records it wrote: ``hparams``, which the checkpoint stores, and the
        initial snapshot a logger writes. The arguments keep Lightning's meaning; with no
        ``frame``, the caller's frame is passed on, since Lightning would otherwise inspect
        this override's. A hyperparameter dict given without ``task`` (LightningCLI's path)
        stays as given.

        Args:
            *args: Lightning's positional hyperparameters: argument names, or one dict or namespace.
            ignore: Argument names Lightning leaves out of the record.
            frame: The frame whose ``__init__`` arguments are recorded; the caller's when ``None``.
            logger: Whether the initial snapshot is offered to the logger.

        Examples:
            ```pycon
            >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=3)
            >>> module.save_hyperparameters({"task": Task.DETECT}, logger=False)
            >>> type(module.hparams["task"]) is str
            True

            ```
        """
        if frame is None:
            # Lightning must inspect the original caller, rather than this wrapper.
            frame = inspect.currentframe()
            if frame is not None:
                frame = frame.f_back
        super().save_hyperparameters(*args, ignore=ignore, frame=frame, logger=logger)
        recorded_task = self.hparams.get("task")
        if isinstance(recorded_task, Task):
            self.hparams["task"] = recorded_task.value
            self._hparams_initial = copy.deepcopy(self.hparams)

    def _init_segmentation(self, task: str, num_classes: int, capture_segment_heads: bool) -> None:
        """Build segmentation in its existing initialization order and retain its scheduling request."""
        self._segment_head_capture = _SegmentHeadCapture(capture_segment_heads)
        if task == "segment":
            self.proto_fusion, self.protonet, self.semantic = build_segmentation_stages(
                self.neck.channels, num_classes, DEFAULT_NUM_COEFFS
            )

    @property
    def task(self) -> str:
        """Supervision task this module was built for: ``"detect"``, ``"segment"``, ``"obb"`` or ``"keypoints"``.

        Read-only, and the supported way for a consumer to ask whether a loaded
        checkpoint has a mask branch, an angle branch or a keypoint branch —
        :meth:`forward_segmentation` gates on this
        same value, so the caller's question and the module's own behaviour cannot
        answer differently. Reading ``hparams["task"]`` instead would be a
        stringly-typed lookup into a bag that is only as complete as the
        checkpoint that filled it, and it raises rather than defaulting when a
        checkpoint predates the hyperparameter.

        Returns:
            The task string given at construction.

        Examples:
            ```pycon
            >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4)
            >>> module.task
            'detect'

            ```
        """
        return self._task

    def setup(self, stage: str) -> None:
        """Pick the CUDA memory layout and objective, then refuse multi-process runs the protocols cannot gather.

        With ``compile_step`` on, :meth:`_compile_objective` decides once whether the
        detection objective is compiled (CUDA accelerator, ``"detect"`` task) or stays
        eager with a warning naming why; a second ``setup`` — ``fit`` then ``validate``
        — changes nothing.

        On a CUDA accelerator the parameters are converted to ``channels_last`` (NHWC),
        the layout cuDNN's convolution kernels prefer: eager training of the ``n`` model
        stepped 8% faster on an RTX PRO 6000 with no other change, and the datamodule
        hands over its images in the same layout so no op has to permute. This hook is
        where the conversion belongs because it runs before device placement and before
        every callback's ``on_fit_start`` — so :class:`EMACallback`'s shadow, cloned
        there, inherits the layout rather than averaging across a permute. The decision
        reads the trainer's accelerator rather than :func:`torch.cuda.is_available`,
        which is true on a CUDA box that was asked for ``accelerator="cpu"``, and it is
        not a flag: CPU and MPS keep their default layout and their bit-exact goldens,
        and there is nothing to configure on the one backend where the layout pays. A
        module with no trainer attached is left as built.

        All task metrics gather whole-epoch records across ranks before scoring.
        Rotated and keypoint metrics also drop sampler padding by dataset identity,
        retaining the original sequential validation order and scoring protocols.

        Args:
            stage: The Trainer stage being set up.

        Examples:
            ```pycon
            >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4)
            >>> module.setup("fit")  # "detect" scales freely; no trainer attachment needed

            ```
        """
        # `_trainer` rather than the `trainer` property: the property raises when nothing
        # is attached, and a bare module (every direct-call test) must stay set-up-able.
        self._segment_head_capture.setup(self, stage)
        on_cuda = self._trainer is not None and isinstance(self._trainer.accelerator, CUDAAccelerator)
        if on_cuda:
            self.to(memory_format=torch.channels_last)
        self._capture_network_active = self._replay_active(self._capture_network, on_cuda)
        self._warn_ignored_replay(self._capture_network, self._capture_network_active, "capture_network", stage)
        self._capture_loss_active = self._replay_active(self._capture_loss, on_cuda)
        self._warn_ignored_replay(self._capture_loss, self._capture_loss_active, "capture_loss", stage)
        if self._compile_step and not self._compile_step_resolved:
            self._compile_objective(on_cuda)

    def _replay_active(self, requested: bool, on_cuda: bool) -> bool:
        """Decide whether one opted-in CUDA recording runs for the stage being set up.

        ``capture_network`` and ``capture_loss`` share the condition: single-device CUDA
        automatic optimization.

        Args:
            requested: The constructor flag.
            on_cuda: Whether the attached trainer's accelerator is CUDA.

        Returns:
            Whether the recording is active.
        """
        return (
            requested
            and on_cuda
            and self._trainer is not None
            and isinstance(self.trainer.strategy, SingleDeviceStrategy)
            and self.automatic_optimization
        )

    @staticmethod
    def _warn_ignored_replay(requested: bool, active: bool, flag: str, stage: str) -> None:
        """Warn at ``fit`` that an opted-in recording :meth:`_replay_active` refused will not run.

        :meth:`setup` calls this only after storing the activation, the order it kept while
        the check was inline, so a filter raising the warning as an error cannot leave a
        flag still on from an earlier run. ``stacklevel=3`` names the caller of
        :meth:`setup`, the frame the warning named while it was raised in :meth:`setup`
        itself.

        Args:
            requested: The constructor flag.
            active: What :meth:`_replay_active` decided for it.
            flag: The flag's name, as the warning spells it.
            stage: The Trainer stage being set up.
        """
        if requested and not active and stage == "fit":
            warnings.warn(
                f"{flag}=True is ignored outside single-device CUDA automatic optimization",
                UserWarning,
                stacklevel=3,
            )

    def configure_callbacks(self) -> list[Callback]:
        """Keep opted-in CUDA recordings scoped to one fit."""
        callbacks: list[Callback] = [_NetworkCaptureLifecycle()] if self._capture_network else []
        if self._capture_loss:
            callbacks.append(_LossCaptureLifecycle())
        if self._segment_head_capture.requested:
            callbacks.append(_SegmentHeadCaptureLifecycle())
        return callbacks

    def _compile_objective(self, on_cuda: bool) -> None:
        """Swap the compiled detection objective in, or warn once and stay eager (WP-184, D24).

        The region is :meth:`_detect_objective` — forward, both decodes, assignment
        and the dual loss — compiled as one Inductor graph through the **bound
        method**, so the module's parameters keep their names (``torch.compile`` on
        the module itself would prefix every state-dict key with ``_orig_mod.``).
        ``capture_scalar_outputs`` is set process-wide, only here: the CUDA bench that
        earned the flag needed it, while on CPU the region traces without it once
        :func:`~lucid_yolo.assign.tal._check_labels` is skipped under tracing.

        Refused, with a :class:`UserWarning` naming why, off CUDA (the MPS Inductor
        backend crashes in ``convolution_backward``; CPU is not the point) and for
        every task but ``"detect"`` — the other three tasks' extra terms still read
        tensors on the host inside :meth:`_task_extra_loss`, which the region does not
        contain and would not trace.

        Args:
            on_cuda: Whether the attached trainer's accelerator is CUDA.
        """
        self._compile_step_resolved = True
        if not self._strategy.compile_objective:
            warnings.warn(
                f"compile_step=True is ignored for task={self._task!r}: only the 'detect' objective is "
                "compiled (WP-184); the segment, obb and keypoint terms stay eager",
                UserWarning,
                stacklevel=3,
            )
            return
        if not on_cuda:
            warnings.warn(
                "compile_step=True is ignored off CUDA: the Inductor backend is only exercised there "
                "(MPS crashes in convolution_backward, WP-184); the step stays eager",
                UserWarning,
                stacklevel=3,
            )
            return
        torch._dynamo.config.capture_scalar_outputs = True
        self._objective = torch.compile(self._detect_objective, backend="inductor")
        self._objective_compiled = True

    def _detect_objective(
        self,
        images: Tensor,
        anchor_points: Tensor,
        strides: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        gt_mask: Tensor,
        alpha: Tensor,
    ) -> tuple[DualHeadOutput, DualLossOutput]:
        """Forward, decode both branches and score the dual detection loss — the compiled region.

        Everything the ``"detect"`` task computes between the padded targets and the
        scalar total, and nothing that reads a tensor on the host: the anchor grid, the
        padding and the logging stay in :meth:`_shared_step`, on either side of this
        call, so the whole of it traces as one graph (``tests/ptl/test_compile.py``).
        Nor does it read a Python scalar that training rewrites: the branch weight
        arrives as a tensor because Dynamo guards on a float attribute's value, and
        :attr:`alpha` changes every epoch under the WP-035 schedule — read inside the
        region it recompiled the whole of it per epoch until ``cache_size_limit`` (8)
        dropped the step to eager, the defect WP-187 found on an A100.

        Args:
            images: ``(B, 3, H, W)`` input batch.
            anchor_points: ``(A, 2)`` anchor centres in input pixels.
            strides: ``(A,)`` per-anchor level stride.
            gt_boxes: ``(B, N, 4)`` padded ground-truth boxes in input pixels.
            gt_labels: ``(B, N)`` padded class ids.
            gt_mask: ``(B, N)`` bool; ``True`` marks a real ground truth.
            alpha: 0-d ``float64`` tensor holding :attr:`alpha` for this step, built
                by :meth:`_alpha_tensor` outside the region.

        Returns:
            The dense head output and the dual loss output.
        """
        head_out = self._training_network(images)
        o2m_boxes = decode_ltrb(head_out.o2m_box, anchor_points, strides)
        o2o_boxes = decode_ltrb(head_out.o2o_box, anchor_points, strides)
        if (
            self._capture_loss_active
            and self._trainer is not None
            and self._trainer.training
            and self.training
            and self.automatic_optimization
            and torch.is_grad_enabled()
        ):
            out = self._loss_capture.run(
                self.loss,
                _LossInputs(
                    head_out.o2m_cls,
                    o2m_boxes,
                    head_out.o2o_cls,
                    o2o_boxes,
                    anchor_points,
                    gt_boxes,
                    gt_labels,
                    gt_mask,
                    strides,
                    alpha,
                ),
            )
        else:
            out = self.loss(
                BranchPrediction(head_out.o2m_cls, o2m_boxes),
                BranchPrediction(head_out.o2o_cls, o2o_boxes),
                anchor_points,
                gt_boxes,
                gt_labels,
                gt_mask,
                strides=strides,
                alpha=alpha,
            )
        return head_out, out

    def _network_replay_eligible(self, images: Tensor) -> bool:
        """Decide whether a call may reach the network recording, before any task-specific contract.

        Every task's training route shares this, so a change to when replay applies lands
        once: only a training step of an activated fit, with automatic optimisation, enabled
        gradients and CUDA images, qualifies. Everything else, evaluation included, stays native.
        A retired recording answers every call eagerly and ignores later causes, so the
        task-specific checks are skipped for the rest of that fit.
        """
        return (
            self._capture_network_active
            and not self._network_capture.retired
            and self._trainer is not None
            and self._trainer.training
            and self.training
            and self.automatic_optimization
            and torch.is_grad_enabled()
            and images.is_cuda
        )

    def _training_network(self, images: Tensor) -> DualHeadOutput:
        """Run the dual head through the training recording when its contract holds.

        Detection and oriented detection record all three stages; pose records shared
        features and each stem independently. Unused point outputs have no backward
        invocation, so batches with no assigned positives retain absent point-stem gradients.
        All tasks keep changing instance counts and loss arithmetic outside capture.
        """
        head_out: DualHeadOutput
        if self._network_replay_eligible(images):
            self._strategy.check_replay_head(self)
            root_hooks = any(
                (self._forward_hooks, self._forward_pre_hooks, self._backward_hooks, self._backward_pre_hooks)
            )
            custom_forward = type(self).forward is not _EAGER_NETWORK_FORWARD or "forward" in vars(self)
            if root_hooks or custom_forward or self._compiled_call_impl is not None:
                if root_hooks:
                    reason = "the LightningModule carries a forward or backward hook"
                elif custom_forward:
                    reason = "the LightningModule overrides forward"
                else:
                    reason = "the LightningModule call is compiled"
                self._network_capture.retire(reason)
                head_out = self(images)
            else:
                head_out = self._strategy.training_network(self, images)
        else:
            head_out = self(images)
        return head_out

    def _check_pose_replay_head(self) -> None:
        """Check pose owners before root-hook guards, retaining the module's certificate binding."""
        self._network_capture.check_stem_owners(self.head)
        breach = _pose_head_breach(self.head, self._head_topology)
        if breach is not None:
            self._network_capture.retire(breach)

    def _training_oriented_network(self, images: Tensor) -> DualHeadOutput:
        """Record the complete oriented network only while its native consumers stay intact."""
        reason = _oriented_replay_breach(self)
        if reason is not None:
            self._network_capture.retire(reason)
            return cast("DualHeadOutput", self(images))
        eager = cast("Callable[[Tensor], DualHeadOutput]", self)
        return self._network_capture.run_oriented(images, (self.backbone, self.neck, self.head), eager)

    def _alpha_tensor(self, device: torch.device) -> Tensor:
        """Lift the current :attr:`alpha` to a 0-d ``float64`` tensor on ``device`` for the compiled region.

        Built once per step on the host, before :meth:`_detect_objective` is entered,
        so the traced region takes the weight as a tensor input — a dtype and shape
        guard, never a value guard. ``float64`` so that :class:`DualBranchLoss` can
        form ``1 - alpha`` in double and cast once, the rounding the Python-float
        blend does; a ``float32`` scalar would put the eager and compiled totals one
        ulp apart at ``0.8``.

        Args:
            device: Where the images, and so the branch totals, live.

        Returns:
            A ``()``-shaped ``float64`` tensor equal to :attr:`alpha`, on ``device`` --
            except for MPS, which has no ``float64``: there it stays on the host, where
            torch's scalar promotion lets a 0-d CPU tensor weight a device tensor, and
            no compiled region is involved (the compiled step is CUDA-only).

        Examples:
            ```pycon
            >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4)
            >>> module.alpha = 0.8
            >>> module._alpha_tensor(torch.device("cpu"))
            tensor(0.8000, dtype=torch.float64)

            ```
        """
        host = torch.device("cpu") if device.type == "mps" else device
        return torch.tensor(self.loss.alpha, dtype=torch.float64, device=host)

    def _mark_dynamic(self, images: Tensor, gt_boxes: Tensor, gt_labels: Tensor, gt_mask: Tensor) -> None:
        """Mark the batch and ground-truth counts dynamic before a compiled objective sees them.

        :func:`pad_targets` pads to the batch's largest instance count and the last
        batch of an epoch is smaller, so without the marks Dynamo would specialise a
        graph per distinct ``(B, N)`` until ``cache_size_limit`` and then silently fall
        back to eager. ``B`` is marked on the ground truths as well as the images: the
        assigner broadcasts the two against each other, and a static ``B`` on either
        side would pin the symbolic one to it. A no-op while the objective is eager.
        """
        if not self._objective_compiled:
            return
        torch._dynamo.mark_dynamic(images, 0)
        for tensor in (gt_boxes, gt_labels, gt_mask):
            torch._dynamo.mark_dynamic(tensor, 0)
            torch._dynamo.mark_dynamic(tensor, 1)

    @property
    def alpha(self) -> float:
        """One-to-many branch weight of the dual loss (the WP-035 schedule seam).

        Returns:
            The current :attr:`DualBranchLoss.alpha`; setting it rewrites the
            underlying loss attribute so a scheduler can update the ramp per epoch.

        Examples:
            ```pycon
            >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4)
            >>> module.alpha = 0.8
            >>> module.loss.alpha
            0.8

            ```
        """
        return self.loss.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self.loss.alpha = value

    def on_train_epoch_start(self) -> None:
        """Ramp the dual-loss branch weight for the epoch about to start (WP-035).

        Sets :attr:`alpha` to
        :meth:`~lucid_yolo.losses.progressive.ProgressiveLossSchedule.alpha_at`
        evaluated at the current 0-based epoch and the trainer's total epoch
        count — the linear R1 Eq. 2-3 ramp updated once per epoch. When
        ``trainer.max_epochs`` is unset (``None`` or ``<= 0``, as for a
        step-bounded or ``fast_dev_run`` run) the ramp denominator is undefined,
        so ``alpha`` is left at its seeded value.

        Examples:
            ```pycon
            >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4)
            >>> module.alpha  # seeded value before any epoch starts
            0.5

            ```
        """
        max_epochs = self.trainer.max_epochs
        if max_epochs is None or max_epochs <= 0:
            return
        self.alpha = self._loss_schedule.alpha_at(self.current_epoch, max_epochs)

    def forward(self, images: Tensor) -> DualHeadOutput:
        """Run the backbone, neck, and dual head over an image batch.

        Args:
            images: Input batch of shape ``(B, 3, H, W)`` with ``H`` and ``W``
                divisible by 32.

        Returns:
            The :class:`~lucid_yolo.models.heads.detect.DualHeadOutput` dense
            predictions (raw class logits and raw ``ltrb`` distances) of both
            branches.

        Examples:
            ```pycon
            >>> import torch
            >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4).eval()
            >>> with torch.no_grad():
            ...     out = module(torch.zeros(1, 3, 160, 160))
            >>> out.o2o_cls.shape[-1]
            4

            ```
        """
        return _detection_stages(self.backbone, self.neck, self.head, images)

    def forward_segmentation(self, images: Tensor) -> SegmentOutput:
        """Run the detection **and** mask branches over an image batch (WP-087).

        :meth:`forward` returns the detection head's output alone, because that is
        what every detection consumer — the E2E decode, the mAP metric, the export
        path — asks for. Segmentation needs the prototype maps and the auxiliary
        logits from the *same* neck features, so this is the second entry point,
        and it is the module's only composition of the mask side: the training
        step and the overfit gate both call it rather than re-running
        ``backbone -> neck -> proto_fusion -> protonet`` themselves, which is how
        the two would drift apart.

        The auxiliary semantic branch is training-only (A17) and returns ``None``
        in eval mode; the prototypes are produced in both modes.

        Args:
            images: Input batch of shape ``(B, 3, H, W)`` with ``H`` and ``W``
                divisible by 32.

        Returns:
            A :class:`~lucid_yolo.models.build.SegmentOutput` holding the dual
            head's dense predictions (including both branches' mask
            coefficients), the raw prototype maps, and the auxiliary logits.

        Raises:
            ValueError: If this module's task is not ``"segment"`` — the mask
                branches are only built for that task, so any other task has no
                prototypes to return.

        Examples:
            ```pycon
            >>> import torch
            >>> module = DetectionLitModule(
            ...     depth=0.34, width=0.25, max_channels=256, num_classes=4, task="segment"
            ... ).eval()
            >>> with torch.no_grad():
            ...     out = module.forward_segmentation(torch.zeros(1, 3, 64, 64))
            >>> out.prototypes.shape  # twice the P3 grid (A15): 64 / 8 * 2
            torch.Size([1, 32, 16, 16])
            >>> out.detect.o2o_coeff.shape[-1], out.semantic is None  # aux is training-only (A17)
            (32, True)

            ```
        """
        if not self._strategy.segmentation:
            raise ValueError(f"forward_segmentation requires task='segment'; this module's task is {self._task!r}")
        features = self._segmentation_features(images)
        runner = self._segment_head_capture.runner(self, features)
        detect = cast("DualHeadOutput", self.head(features) if runner is None else self.head(features, _runner=runner))
        fused: Tensor = self.proto_fusion(features)
        if runner is None:
            return SegmentOutput(detect=detect, prototypes=self.protonet(fused), semantic=self.semantic(fused))
        return SegmentOutput(
            detect=detect, prototypes=runner.mask(self.protonet, fused), semantic=runner.mask(self.semantic, fused)
        )

    def _segmentation_features(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Replay shared segmentation features while keeping mask supervision outside capture.

        The three feature levels always feed the dense classification objective.
        Coefficients, prototypes and semantic predictions keep their independent
        backward nodes, so an unused mask branch retains its absent gradients.
        The existing holder owns precision/storage checks, buffer rollback and
        the one-signature-per-fit lifecycle, as for shared pose features.
        """
        if self._network_replay_eligible(images):
            breach = _segment_head_breach(self.head) or _head_topology_breach(
                self.head, self._head_topology, "segmentation"
            )
            if breach is not None:
                self._network_capture.retire(breach)
            # Training calls `forward_segmentation` directly, never `self(...)`, so a compiled
            # LightningModule call (`_compiled_call_impl`) never reaches this route; a compiled
            # `forward_segmentation` assigned on the instance is caught by the `vars` check.
            if type(self).forward_segmentation is not _EAGER_SEGMENT_FORWARD or "forward_segmentation" in vars(self):
                self._network_capture.retire("the segmentation forward is overridden or compiled")
            features = self._network_capture.run_features(images, (self.backbone, self.neck))
            if features is not None:
                return features
        return cast("tuple[Tensor, Tensor, Tensor]", self.neck(self.backbone(images)))

    def forward_branch(self, images: Tensor, branch: BranchName) -> BranchOutput:
        """Run the backbone, neck, and **one** head branch over an image batch.

        :meth:`forward` returns both branches because training reads both. Every
        inference decode reads exactly one: the top-k path the one-to-one branch, the
        suppression path the one-to-many. Called through :meth:`forward`, the other
        branch's stems run over all three levels and their outputs are discarded, so this
        is the entry point for a caller that already knows which half it will read.

        The composition is deliberately the same three calls :meth:`forward` makes, with
        only the head's own branch selection narrowed -- the backbone and neck are shared
        by both branches and are not run twice or run differently here.

        Args:
            images: Input batch of shape ``(B, 3, H, W)`` with ``H`` and ``W``
                divisible by 32.
            branch: Which head branch to run, one of
                :data:`~lucid_yolo.models.heads.detect.BRANCH_NAMES`.

        Returns:
            That branch's
            :class:`~lucid_yolo.models.heads.detect.BranchOutput` dense predictions.

        Raises:
            ValueError: If ``branch`` is not a known branch name.

        Examples:
            ```pycon
            >>> import torch
            >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4).eval()
            >>> with torch.no_grad():
            ...     one_to_one = module.forward_branch(torch.zeros(1, 3, 160, 160), "o2o")
            >>> one_to_one.cls.shape[-1]
            4

            ```
        """
        return self.head.forward_branch(self.neck(self.backbone(images)), branch)

    def forward_segmentation_branch(self, images: Tensor, branch: BranchName) -> tuple[BranchOutput, Tensor]:
        """Run **one** head branch and the prototype maps over an image batch.

        The branch-limited twin of :meth:`forward_segmentation`, and the mask side's
        reason for existing is the same as the detection side's: Eq. 7 reads the
        coefficients of the branch its decoder read, never both.

        Two things are dropped rather than returned. The other branch's stems -- including
        its coefficient stem, which is the widest of the optional ones -- are never run.
        The auxiliary semantic head is not run either: it is training-only (A17) and
        already returns ``None`` in eval mode, so a caller decoding masks has nothing to
        read from it. What remains is exactly the pair Eq. 7 multiplies.

        Args:
            images: Input batch of shape ``(B, 3, H, W)`` with ``H`` and ``W``
                divisible by 32.
            branch: Which head branch to run, one of
                :data:`~lucid_yolo.models.heads.detect.BRANCH_NAMES`.

        Returns:
            A ``(branch_output, prototypes)`` pair: that branch's dense predictions, and
            the raw prototype maps ``(B, num_coeffs, H/4, W/4)``.

        Raises:
            ValueError: If this module's task is not ``"segment"`` -- the mask branches
                are only built for that task, so any other task has no prototypes to
                return; or if ``branch`` is not a known branch name.

        Examples:
            ```pycon
            >>> import torch
            >>> module = DetectionLitModule(
            ...     depth=0.34, width=0.25, max_channels=256, num_classes=4, task="segment"
            ... ).eval()
            >>> with torch.no_grad():
            ...     one_to_one, prototypes = module.forward_segmentation_branch(torch.zeros(1, 3, 64, 64), "o2o")
            >>> prototypes.shape  # twice the P3 grid (A15): 64 / 8 * 2
            torch.Size([1, 32, 16, 16])
            >>> one_to_one.coeff.shape[-1]
            32

            ```
        """
        if not self._strategy.segmentation:
            raise ValueError(
                f"forward_segmentation_branch requires task='segment'; this module's task is {self._task!r}"
            )
        features: tuple[Tensor, Tensor, Tensor] = self.neck(self.backbone(images))
        detect = self.head.forward_branch(features, branch)
        fused: Tensor = self.proto_fusion(features)
        prototypes: Tensor = self.protonet(fused)
        return detect, prototypes

    def training_step(self, batch: StepBatch, batch_idx: int) -> Tensor:
        """Run one training step under automatic optimization (D4).

        Args:
            batch: The datamodule batch ``(images, list[Targets], masks)``; the
                two-element ``(images, list[Targets])`` form rasterises in the step.
            batch_idx: Index of the batch within the epoch (unused).

        Returns:
            The scalar total loss for Lightning to backpropagate.
        """
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch: StepBatch, batch_idx: int) -> Tensor:
        """Run one validation step: shared forward and loss, plus the mAP update.

        Beyond the ``val/``-logged loss, the one-to-one branch is decoded with
        the E2E :class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` from the same
        forward and accumulated into the epoch's
        :class:`~torchmetrics.detection.MeanAveragePrecision` (WP-077), logged
        as ``val/mAP`` by :meth:`on_validation_epoch_end`. Scoring runs in
        letterbox coordinates — IoU is invariant to each image's uniform
        letterbox scaling, so the number tracks the original-coordinate
        protocol closely; the acceptance figure remains ``lucid-eval``
        (original coordinates, both paths).

        A ``"segment"`` module additionally decodes the kept detections' masks
        and accumulates ``val/segm_mAP`` (:meth:`_update_val_segm`), so a
        segmentation run reports the quality of the branch it exists for rather
        than of its boxes alone. That needs the ground-truth masks, which only a
        loader built with ``mask_targets=True`` supplies — the CLI links that
        flag to ``model.task``, and a batch without them is scored on boxes only
        rather than rasterised a second time here.

        An ``"obb"`` module accumulates the WP-063 rotated mAP
        (:meth:`_update_val_rotated`) **instead of** ``val/mAP``, not beside it
        (WP-102): the axis-aligned figure reads the A44 composition's
        *pre-rotation* rectangle, so a run whose orientations were random and one
        whose orientations were right log the identical curve, and the acceptance
        number would otherwise first appear hours after the run.

        A ``"keypoints"`` module additionally decodes the kept detections' points
        and accumulates ``val/oks_mAP`` (:meth:`_update_val_keypoints`, WP-137),
        the same "beside ``val/mAP``" relationship ``"segment"`` has: the box
        branch and the point branch answer different questions, and a run whose
        boxes are fine but whose points have not yet wired up should not read as
        healthy on the one number that was already there.

        Args:
            batch: The datamodule batch ``(images, list[Targets], masks)``; the
                two-element ``(images, list[Targets])`` form rasterises in the step.
            batch_idx: Index of the batch within the epoch (unused).

        Returns:
            The scalar total validation loss.
        """
        images, targets, masks = _split_batch(batch)
        loss, head_out, seg_out = self._shared_step(batch, "val")
        anchor_points, strides = self._anchor_grid(images.shape[-2], images.shape[-1], images.device)
        detections, anchor_indices = self._val_decoder.decode_with_indices(
            head_out.o2o_cls, head_out.o2o_box, anchor_points, strides
        )
        self._strategy.update_box_validation(self, detections, targets)
        if self._val_segm is not None and seg_out is not None and masks is not None:
            image_size = (int(images.shape[-2]), int(images.shape[-1]))
            self._update_val_segm(seg_out, detections, anchor_indices, targets, masks, image_size)
        self._strategy.update_extra_validation(
            self, head_out, targets, anchor_points, strides, detections, anchor_indices
        )
        return loss

    def _update_val_map(self, detections: Tensor, targets: list[Targets]) -> None:
        """Accumulate one batch's kept boxes and its ground truth, moved to the host, for the epoch ``val/mAP``.

        The host copy of the detections is made here rather than by the caller, so a strategy
        that keeps no axis-aligned metric never pays for it.

        Args:
            detections: The ``(B, k, 6)`` decoded A9 batch, reduced by :func:`_box_predictions`.
            targets: The batch's per-image ground truth, carrying ``boxes`` and ``labels``.
        """
        host_boxes = _host_split([t.boxes for t in targets])
        host_labels = _host_split([t.labels for t in targets])
        ground_truth = [
            {"boxes": boxes, "labels": labels} for boxes, labels in zip(host_boxes, host_labels, strict=True)
        ]
        self._val_map.update(_box_predictions(detections), ground_truth)

    def _update_val_rotated(
        self, head_out: DualHeadOutput, targets: list[Targets], anchor_points: Tensor, strides: Tensor
    ) -> None:
        """Accumulate one batch's oriented detections and ground truth for the epoch mAP.

        The decode is the **deployed** one: :func:`~lucid_yolo.models.heads.obb.decode_rboxes`
        assembles the one-to-one branch's boxes (A44) and
        :func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk` ranks them, gathering each
        angle by the anchor index its own box was ranked by. A second ranking here would
        be free to pair one anchor's heading with another's box — a detection with the
        right centre, the right score and a silently wrong orientation.

        Scoring stays in **letterbox** coordinates, exactly as ``val/mAP`` does (WP-077).
        A letterbox is one isotropic scale plus a translation, so it maps every rotated
        box and every ground truth by the same similarity and leaves rotated IoU — a
        ratio of areas — unchanged; un-letterboxing both sides with
        :func:`~lucid_yolo.decode.common.rboxes_to_letterboxed_original` would divide the
        same number by itself. That helper is what an *original-coordinate* report needs,
        and it is where the acceptance figure is measured.

        The R18 difficult flags come from the targets rather than being defaulted here
        (A48): an ignorable ground truth that arrives as an ordinary one turns every
        detection of it into a false positive.

        Args:
            head_out: This batch's dense dual-head output.
            targets: The batch's per-image ground truth, carrying ``rboxes``, ``labels``
                and the difficult flags.
            anchor_points: ``(A, 2)`` anchor centres in input pixels.
            strides: ``(A,)`` per-anchor level stride.
        """
        angles = head_out.o2o_angle
        assert angles is not None  # an "obb" module always builds the angle stems
        rboxes = decode_rboxes(head_out.o2o_box, angles, anchor_points, strides)
        detections = o2o_rotated_topk(head_out.o2o_cls, rboxes, k=MAX_DETECTIONS)
        preds = rotated_detections_to_predictions(detections.detach())
        host_rboxes = _host_split([target.rboxes.detach() for target in targets])
        host_labels = _host_split([target.labels.detach() for target in targets])
        host_difficult = _host_split([target.difficult.detach() for target in targets])
        ground_truth = [
            {"rboxes": rboxes, "labels": labels.to(torch.long), "difficult": difficult}
            for rboxes, labels, difficult in zip(host_rboxes, host_labels, host_difficult, strict=True)
        ]
        assert self._val_rotated is not None
        self._val_rotated.update(preds, ground_truth, self._validation_ids(targets, self._val_rotated))

    def _update_val_keypoints(
        self,
        head_out: DualHeadOutput,
        detections: Tensor,
        anchor_indices: Tensor,
        targets: list[Targets],
        anchor_points: Tensor,
        strides: Tensor,
    ) -> None:
        """Accumulate one batch's decoded keypoints and ground truth for the epoch OKS.

        The decode is the **deployed** one, the same shape :meth:`_update_val_rotated`
        and :meth:`_update_val_segm` already use: :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints`
        composes the one-to-one branch's dense point offsets, and
        :func:`~lucid_yolo.eval.coco_eval.gather_keypoints` takes each kept
        detection's own set by the anchor index the *same* top-k selection
        already returned — never a second ranking computed here, which could
        pair one anchor's points with another's box.

        Scoring stays in **letterbox** coordinates, exactly as ``val/mAP`` does
        (WP-077): a letterbox is one isotropic scale plus a translation, and OKS
        normalizes by the ground-truth box's own area, which that same scaling
        leaves proportionate. Un-letterboxing first would cost a second geometry
        pass to answer a question this ratio already answers unscaled.

        No category remap is needed here (unlike
        :func:`~lucid_yolo.eval.coco_eval.keypoints_to_predictions`): predictions
        and targets both already live in this module's own contiguous label
        space, the same one :attr:`_val_map` scores boxes in.

        Args:
            head_out: This batch's dense dual-head output.
            detections: The ``(B, k, 6)`` decoded A9 batch already computed for
                ``val/mAP``, on the model's device.
            anchor_indices: The ``(B, k)`` source anchor of each detection row,
                from the same decode.
            targets: The batch's per-image ground truth, carrying ``keypoints``,
                ``keypoint_vis`` and ``labels``.
            anchor_points: ``(A, 2)`` anchor centres in input pixels.
            strides: ``(A,)`` per-anchor level stride.
        """
        raw_points = head_out.o2o_keypoints
        assert raw_points is not None  # a "keypoints" module always builds the point stem
        dense_points = decode_keypoints(raw_points, anchor_points, strides)
        gathered_points = gather_keypoints(dense_points, anchor_indices).cpu()
        cpu_detections = detections.cpu()
        host_keypoints = _host_split([target.keypoints.detach() for target in targets])
        host_visibility = _host_split([target.keypoint_vis.detach() for target in targets])
        host_labels = _host_split([target.labels.detach() for target in targets])
        preds = []
        ground_truth = []
        for index in range(len(targets)):
            keep = cpu_detections[index, :, SCORE_COLUMN] > 0.0
            preds.append(
                {
                    "keypoints": gathered_points[index, keep],
                    "scores": cpu_detections[index, keep, SCORE_COLUMN],
                    "labels": cpu_detections[index, keep, _LABEL_COLUMN].long(),
                }
            )
            ground_truth.append(
                {
                    "keypoints": host_keypoints[index],
                    "visibility": host_visibility[index],
                    "labels": host_labels[index].to(torch.long),
                }
            )

        assert self._val_keypoints is not None
        self._val_keypoints.update(preds, ground_truth, self._validation_ids(targets, self._val_keypoints))

    def _validation_ids(self, targets: list[Targets], metric: _ValidationMetric) -> list[int]:
        """Read transported identities; direct single-device callers use running positions."""
        if all(isinstance(target, _ValidationTargets) for target in targets):
            return [target._image_id for target in targets if isinstance(target, _ValidationTargets)]
        if self._trainer is not None and self.trainer.world_size > 1:
            raise RuntimeError("Distributed task validation requires the datamodule's indexed validation targets")
        return list(range(metric.image_count, metric.image_count + len(targets)))

    def _update_val_segm(
        self,
        seg_out: SegmentOutput,
        detections: Tensor,
        anchor_indices: Tensor,
        targets: list[Targets],
        masks: list[Tensor],
        image_size: tuple[int, int],
    ) -> None:
        """Decode the kept detections' masks and accumulate them into ``val/segm_mAP``.

        The decode is the **deployed** one (A37): the one-to-one branch's boxes and
        coefficients paired by the anchor indices the same top-k selection returned,
        then assembled, cropped and binarised by
        :func:`~lucid_yolo.eval.segment_decode.decode_instance_masks`. Pairing a
        coefficient row with a differently ranked box would yield a plausible mask of
        the wrong object, which no metric value reveals.

        Scoring happens **on the prototype grid**, not on the letterboxed canvas: the
        ground-truth masks arrive from the loader already rasterised there, and both
        sides upsampled by four would cost sixteen times the memory to compare the
        same two fields at a finer sampling of the same boundary. That makes
        ``val/segm_mAP`` a proxy in exactly the sense ``val/mAP`` already is — the
        acceptance figure is ``lucid-eval``, which scores masks at original
        resolution.

        Masks are decoded one image at a time: every kept detection materialises a
        full-grid float map, and a whole batch at once is a needless memory spike.
        Only the surviving detections are decoded — the padding rows a fixed-length
        decoder pads with are excluded before the decode rather than discarded after
        it, which is sound because they are always a suffix (see the loop's comment).

        Args:
            seg_out: The segmentation forward output of this batch.
            detections: The ``(B, k, 6)`` decoded A9 batch, on the model's device.
            anchor_indices: The ``(B, k)`` source anchor of each detection row.
            targets: Per-image ground truth, for the labels.
            masks: Per-image ground-truth masks on the prototype grid.
            image_size: The letterboxed canvas ``(height, width)`` the boxes are in.
        """
        assert self._val_segm is not None  # the caller gates on it
        prototypes = seg_out.prototypes
        proto_grid = (int(prototypes.shape[-2]), int(prototypes.shape[-1]))
        coefficients = seg_out.detect.o2o_coeff
        assert coefficients is not None  # a "segment" module always builds the coefficient stems
        limit = min(_VAL_SEGM_MAX_DET, int(detections.shape[1]))
        detections = detections[:, :limit]
        # Padding rows carry PAD_ANCHOR_INDEX; they are clamped into range so the
        # gather is legal and then dropped by the score filter below.
        gather_index = anchor_indices[:, :limit].clamp_min(0).unsqueeze(-1).expand(-1, -1, coefficients.shape[-1])
        kept_coefficients = coefficients.gather(1, gather_index)
        grid_boxes = scale_boxes_to_grid(detections[..., :BOX_CORNERS], image_size, proto_grid)
        cpu_detections = detections.cpu()
        preds = []
        ground_truth = []
        host_labels = _host_split([target.labels for target in targets])
        host_masks = _host_split([mask.to(torch.bool) for mask in masks])
        for index in range(len(targets)):
            # The surviving rows are a prefix, so a count is all the decode needs.
            # `TopKDecoder` ranks score-descending, `pad_detections` appends its zero
            # rows after the ranked ones, and `_zero_below_threshold` zeroes a
            # descending column from some point on -- three suffix operations, so
            # `score > 0` can only ever be a leading run. Counting it on the host
            # copy that is already here costs nothing, and slicing to that count
            # decodes the real detections alone rather than decoding a hundred rows
            # to discard the padding immediately afterwards (WP-164).
            kept = int((cpu_detections[index, :, SCORE_COLUMN] > 0.0).sum())
            decoded = decode_instance_masks(
                prototypes[index : index + 1],
                kept_coefficients[index : index + 1, :kept],
                grid_boxes[index : index + 1, :kept],
                image_size=proto_grid,
            )[0].cpu()
            preds.append(
                {
                    "scores": cpu_detections[index, :kept, SCORE_COLUMN],
                    "labels": cpu_detections[index, :kept, _LABEL_COLUMN].long(),
                    "masks": decoded,
                }
            )
            ground_truth.append({"labels": host_labels[index], "masks": host_masks[index]})
        self._val_segm.update(preds, ground_truth)
        self._val_segm_seen = True

    def on_validation_epoch_end(self) -> None:
        """Compute and log the epoch's E2E ``val/mAP`` (progress-bar metric), then reset.

        A ``"segment"`` run additionally logs ``val/segm_mAP``, whenever any batch
        of the epoch carried ground-truth masks; an ``"obb"`` run logs the WP-063
        ``val/rotated_mAP`` and ``val/rotated_mAP50`` over everything the epoch saw
        and logs **no** ``val/mAP`` at all (WP-102): that figure reads the A44
        composition's pre-rotation rectangle, so it cannot separate a run whose
        orientations were right from one whose orientations were random, and its
        accumulator costs a CPU pass over every detection of every batch to say so.
        The rotated metric is the one an oriented run is asking for; carrying a
        second one that answers a different question is not a free comparison.

        A ``"keypoints"`` run additionally logs ``val/oks_mAP`` (WP-137,
        :meth:`_log_oks_map`), beside ``val/mAP`` rather than instead of it, for
        the same reason ``val/segm_mAP`` sits beside ``val/mAP`` and not the
        other way ``val/rotated_mAP`` does.

        The ``val/segm_mAP`` guard is reduced across processes before it is read.
        :meth:`~torchmetrics.Metric.compute` synchronises collectively, so a rank
        that happened to draw a shard without mask targets must still enter it —
        deciding to skip on that rank alone would leave the others waiting inside
        the collective for a call that never comes. The reduction is ``any``: one
        rank with masks makes the metric well-defined for all of them, and on a
        single process it is the identity, so the accepted single-device behaviour
        is unchanged.
        """
        self._strategy.log_box_validation(self)
        if self._val_segm is not None and self._any_rank_saw_masks():
            self.log("val/segm_mAP", self._val_segm.compute()["map"].to(torch.float32), prog_bar=True)
            self._val_segm.reset()
            self._val_segm_seen = False
        if self._val_rotated is not None and self._any_rank_saw_images(self._val_rotated):
            self._log_rotated_map()
        if self._val_keypoints is not None and self._any_rank_saw_images(self._val_keypoints):
            self._log_oks_map()

    def _any_rank_saw_images(self, metric: _ValidationMetric) -> bool:
        """Make every rank enter compute when any rank has task records."""
        seen = metric.image_count > 0
        if self._trainer is None:
            return seen
        return bool(self.trainer.strategy.reduce_boolean_decision(seen, all=False))

    def _any_rank_saw_masks(self) -> bool:
        """Return whether *any* process fed the segmentation metric this epoch.

        :attr:`_val_segm_seen` is set per rank, but the metric's ``compute`` is a collective
        call, so the decision to make it has to be one every rank agrees on. Lightning's
        strategy reduces the flag with ``all=False`` (an ``any`` across the world); on a
        single process the base strategy returns the flag untouched. A detached module —
        every doctest here, and any direct call outside a run — reads its own flag.
        """
        if self._trainer is None:
            return self._val_segm_seen
        return bool(self.trainer.strategy.reduce_boolean_decision(self._val_segm_seen, all=False))

    def _log_rotated_map(self) -> None:
        """Score the epoch's accumulated oriented detections and clear the buffers.

        ``val/rotated_mAP50`` is put on the progress bar rather than ``val/rotated_mAP``
        because the DoD of the oriented path is stated at IoU 0.50; both are logged, so
        the stricter figure is in ``metrics.csv`` either way. The buffers are cleared
        unconditionally — a metric that silently carried last epoch's detections into
        this one would improve monotonically for reasons that have nothing to do with
        the model.
        """
        assert self._val_rotated is not None
        stats = self._val_rotated.compute()
        self.log("val/rotated_mAP50", torch.tensor(stats["map_50"], dtype=torch.float32), prog_bar=True)
        self.log("val/rotated_mAP", torch.tensor(stats["map"], dtype=torch.float32))
        self._val_rotated.reset()

    def _log_oks_map(self) -> None:
        """Score the epoch's accumulated keypoints and clear the buffers.

        The sigma vector is picked by point count through
        :func:`_keypoint_oks_sigmas` when the metric is built — the same rule
        the :class:`~lucid_yolo.losses.oks_loss.OKSLoss` training term is built
        with (WP-178), so the metric and the loss agree on every point's
        tolerance. The buffers are cleared unconditionally, the
        same reason :meth:`_log_rotated_map` does: a metric that silently
        carried last epoch's detections into this one would improve
        monotonically for reasons that have nothing to do with the model.
        """
        assert self._val_keypoints is not None
        stats = self._val_keypoints.compute()
        self.log("val/oks_mAP", torch.tensor(stats["AP_all"], dtype=torch.float32), prog_bar=True)
        self._val_keypoints.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return MuSGD, paired with the A8 warmup + linear-decay LR schedule.

        The schedule (WP-072) is a per-step
        :class:`~torch.optim.lr_scheduler.LambdaLR` over
        :func:`~lucid_yolo.optim.schedule.warmup_decay_factor`: a linear warmup
        across the first ``warmup_epochs`` epochs, then a linear decay from
        ``lr`` down to ``lr * lrf`` at the trainer's estimated final step. The
        bare constant-LR optimizer is returned instead when the schedule is
        explicitly disabled (``lrf >= 1`` and ``warmup_epochs <= 0``), when no
        trainer is attached (direct calls in tests and tools), or when the run
        has no positive ``max_epochs`` to anchor the warmup fraction.

        The warmup length is a **fraction of the run**, so a short run can ask
        for more warmup steps than it has steps: ``warmup_epochs=3`` against a
        two-epoch smoke budget is config-reachable and is exactly the shape the
        short tiers get launched at. Left alone, every step would take the ramp
        branch — the LR would climb for the whole run, peak below ``lr``, and
        never decay, with nothing saying so. The count is clamped to
        ``total_steps - 1``, which recovers the peak but not the decay: a run
        whose requested warmup covers it has no steps left to decay over, so the
        clamp buys back only the ramp reaching full ``lr`` on the final step.
        That is the ceiling of what a clamp can do here, which is why it warns
        rather than applying silently — the recipe cannot be honoured, and the
        honest signal is to say which schedule actually ran. Clamping rather
        than raising keeps the launchable run launchable: a crash here would
        fail a run over a schedule detail on the tier where the schedule matters
        least.

        Returns:
            A :class:`~lucid_yolo.optim.musgd.MuSGD` over ``self.parameters()``,
            alone or inside a Lightning optimizer/scheduler config dict with the
            step-interval LambdaLR.
        """
        optimizer = MuSGD(
            self.parameters(),
            lr=self._lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
            w_muon=self._w_muon,
            w_sgd=self._w_sgd,
            capture_orthogonalization=self._capture_orthogonalization,
            capture_updates=self._capture_updates,
        )
        schedule_off = self._lrf >= 1.0 and self._warmup_epochs <= 0
        trainer = self._trainer
        max_epochs = None if trainer is None else trainer.max_epochs
        if schedule_off or trainer is None or max_epochs is None or max_epochs <= 0:
            return optimizer
        total_steps = max(1, int(trainer.estimated_stepping_batches))
        requested_warmup = round(total_steps * self._warmup_epochs / max_epochs)
        warmup_steps = min(requested_warmup, total_steps - 1)
        if warmup_steps < requested_warmup:
            warnings.warn(
                f"warmup_epochs={self._warmup_epochs} of max_epochs={max_epochs} asks for "
                f"{requested_warmup} warmup steps, but the run is only {total_steps} steps long: "
                f"the whole run is warmup and the LR never decays to lr*lrf. "
                f"Clamping warmup to {warmup_steps} steps so the ramp at least reaches lr on the "
                f"final step; lower --model.warmup_epochs to leave room for the decay.",
                stacklevel=2,
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: warmup_decay_factor(step, total_steps, warmup_steps, self._lrf),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _shared_step(self, batch: StepBatch, stage: str) -> tuple[Tensor, DualHeadOutput, SegmentOutput | None]:
        """Forward, decode both branches, score the dual loss, and log every term.

        Returns:
            The scalar total loss, the dense head output, and the segmentation
            output (``None`` for a detection module) — so callers can decode from
            the same forward instead of running a second one. The mask metric needs
            the prototypes, which the head output alone does not carry.
        """
        images, targets, masks = _split_batch(batch)
        anchor_points, strides = self._anchor_grid(images.shape[-2], images.shape[-1], images.device)
        gt_boxes, gt_labels, gt_mask = pad_targets(targets)
        gt_boxes = gt_boxes.to(images.device)
        gt_labels = gt_labels.to(images.device)
        gt_mask = gt_mask.to(images.device)
        gt_rboxes, gt_keypoints, gt_keypoint_vis = self._strategy.pad_extras(targets, images.device)
        seg_out = self._strategy.segmentation_output(self, images)
        # `_objective` is the detection region `compile_step` may compile (WP-184). Detection
        # always calls it; pose calls it only for training steps `_pose_records_loss` routes to
        # the loss recorder, and there it is always the bare method, since `_compile_objective`
        # compiles detection alone and `_validate_replay_options` refuses `capture_loss` with
        # `compile_step`. Oriented and segmentation steps keep the inline call below: the
        # oriented assignment reads `gt_rboxes`, which the region does not take, and
        # segmentation already ran its network in `forward_segmentation`, which the region
        # would run a second time.
        if self._strategy.uses_objective(self, stage):
            # Pose shares the unrotated detection objective, but its ragged point
            # terms remain outside recording and consume the returned assignments.
            # Its native and validation routes retain their existing call signature.
            self._mark_dynamic(images, gt_boxes, gt_labels, gt_mask)
            alpha = self._alpha_tensor(images.device)
            head_out, out = self._objective(images, anchor_points, strides, gt_boxes, gt_labels, gt_mask, alpha)
        else:
            head_out = self._training_network(images) if seg_out is None else seg_out.detect
            o2m_boxes = decode_ltrb(head_out.o2m_box, anchor_points, strides)
            o2o_boxes = decode_ltrb(head_out.o2o_box, anchor_points, strides)
            out = self.loss(
                BranchPrediction(head_out.o2m_cls, o2m_boxes),
                BranchPrediction(head_out.o2o_cls, o2o_boxes),
                anchor_points,
                gt_boxes,
                gt_labels,
                gt_mask,
                strides=strides,
                gt_rboxes=gt_rboxes,
            )
        context = _StepContext(
            head_out=head_out,
            seg_out=seg_out,
            targets=targets,
            gt_boxes=gt_boxes,
            gt_rboxes=gt_rboxes,
            gt_keypoints=gt_keypoints,
            gt_keypoint_vis=gt_keypoint_vis,
            anchor_points=anchor_points,
            strides=strides,
            image_size=(int(images.shape[-2]), int(images.shape[-1])),
            masks=masks,
        )
        total = out.total + self._task_extra_loss(context, out, stage)
        self._log_loss(out, total, stage, images.shape[0])
        return total, head_out, seg_out

    def _pose_records_loss(self, stage: str) -> bool:
        """Tell whether a pose step sends its unrotated detection objective to the loss recorder.

        Only a training step of a pose fit whose ``capture_loss`` is active qualifies;
        validation keeps the criterion's original call signature. A criterion
        :func:`~lucid_yolo.ptl.loss_capture._source_signature` cannot certify keeps its
        native call, and the first such step retires the fit's recorder with the warning
        detection emits for the same configuration, so the fallback is reported once per
        fit rather than silently on every step.

        Args:
            stage: Metric prefix of the step, ``"train"`` or ``"val"``.

        Returns:
            ``True`` when the step should call :attr:`_objective`, else ``False``.
        """
        if not self._strategy.pose_loss_capture or not self._capture_loss_active or stage != "train":
            return False
        if _source_signature(self.loss) is not None:
            return True
        if not self._loss_capture._disabled:
            self._loss_capture.retire("the loss or an assignment implementation was customized")
        return False

    def _task_extra_loss(self, context: _StepContext, out: DualLossOutput, stage: str) -> Tensor:
        """Return the task-conditional extra loss: the segment, oriented or pose terms, else zero.

        For ``"segment"`` this is ``mask_gain * mask + semantic_gain * semantic``
        (A38); for ``"obb"`` it is ``box_gain * rbox + l1_gain * rl1 + angle_gain *
        angle``, which together with the dual loss's classification term — the only
        term left with a non-zero gain under that task — *is* the whole oriented
        objective rather than an addition to a detection one (module docstring); for
        ``"keypoints"`` it is ``keypoint_gain * rle + oks_gain * oks``, an addition in
        the segment sense, with every detection term left at its own gain; both pose
        terms are computed and logged whatever their gain, so a run at the default
        ``oks_gain=0`` still reads the OKS term it is not yet training on.
        ``"detect"`` contributes a zero scalar, so its total stays exactly the dual
        detection loss. Every pre-gain term is logged.

        Args:
            context: The step's shared quantities (see :class:`_StepContext`).
            out: The dual-branch loss output, read for its ``alpha`` and — the
                point of the WP-087 plumbing — for the two assignments the box
                terms were scored against.
            stage: Metric prefix (``"train"``/``"val"``) for the logged terms.

        Returns:
            A scalar tensor on the prediction device.
        """
        return self._strategy.extra_loss(self, context, out, stage)

    def _oriented_extra_loss(self, context: _StepContext, out: DualLossOutput, stage: str) -> Tensor:
        """Weight and log the three oriented terms, blended by the WP-035 ``alpha``.

        Each branch is scored against **its own** assignment and the two are combined
        by the same ``alpha * o2m + (1 - alpha) * o2o`` split
        :class:`~lucid_yolo.losses.dual_loss.DualBranchLoss` applies to the box terms,
        for the reason WP-087 gives for the mask term: the one-to-one branch is the
        one the NMS-free oriented decode reads, so supervising the one-to-many angle
        stems alone would ship an untrained orientation, and blending with the same
        ramp moves orientation supervision *with* box supervision rather than against
        it. Both branches' stems therefore receive gradient at every step, which is
        exactly what ``test_obb_training.py`` asserts.

        Args:
            context: The step's shared quantities.
            out: The dual-branch loss output supplying both assignments and ``alpha``.
            stage: Metric prefix for the logged terms.

        Returns:
            The gain-weighted oriented contribution, a scalar on the prediction device.
        """
        head_out, alpha = context.head_out, out.alpha
        o2m_angle, o2o_angle = head_out.o2m_angle, head_out.o2o_angle
        assert o2m_angle is not None  # an "obb" module always builds both angle stems
        assert o2o_angle is not None
        assert context.gt_rboxes is not None  # the step pads them for exactly this task
        o2m = self._branch_oriented_terms(head_out.o2m_box, o2m_angle, out.o2m_assign, context)
        o2o = self._branch_oriented_terms(head_out.o2o_box, o2o_angle, out.o2o_assign, context)
        rbox = _blend(alpha, o2m.rbox, o2o.rbox)
        rl1 = _blend(alpha, o2m.rl1, o2o.rl1)
        angle = _blend(alpha, o2m.angle, o2o.angle)
        batch_size = len(context.targets)
        self.log(f"{stage}/rbox", rbox, batch_size=batch_size)
        self.log(f"{stage}/rl1", rl1, batch_size=batch_size)
        self.log(f"{stage}/angle", angle, batch_size=batch_size)
        return self._rbox_gain * rbox + self._rl1_gain * rl1 + self._angle_gain * angle

    def _branch_oriented_terms(
        self, distances: Tensor, angles: Tensor, assign: AssignResult, context: _StepContext
    ) -> OrientedLossOutput:
        """Assemble one branch's oriented boxes (A44) and score them against its assignment.

        The assembly is :func:`~lucid_yolo.models.heads.obb.decode_rboxes` — the
        deployed composition, not a training-only copy of it — so what the loss pulls
        towards the ground truth is precisely what inference will emit. Its
        canonicalization is invisible to both box terms (a canonical box is the same
        rectangle, hence the same Gaussian and the same ``(cx, cy, w, h)`` up to the
        long-edge swap the target underwent too), which is why the raw Eq. 13 angle is
        passed separately for the Eq. 14 residual rather than read back off the
        canonical box.
        """
        assert context.gt_rboxes is not None  # gated by the caller
        rboxes = decode_rboxes(distances, angles, context.anchor_points, context.strides)
        return oriented_branch_terms(
            rboxes,
            angles.squeeze(-1),
            context.gt_rboxes,
            assign,
            context.strides,
            form=self._rotated_iou_form,
        )

    def _keypoint_terms(self, context: _StepContext, out: DualLossOutput) -> tuple[Tensor, Tensor]:
        """Return the pre-gain ``(rle, oks)`` pose terms for a batch, blended over both branches (WP-132).

        Both branches are scored and blended by the WP-035 ``alpha``, exactly as the
        mask and oriented terms are, and for the same reason: the one-to-one branch is
        the one a NMS-free pose decode reads, so supervising the one-to-many point
        stems alone would ship an untrained keypoint head, and the shared ramp moves
        point supervision *with* box supervision rather than against it.

        Both terms — the R14 residual log-likelihood and the WP-178 OKS term — come
        out of one :meth:`_score_branch` call per branch, so the point-count check, the
        decode, the assignment gather and the blend are written once and both terms
        read the same positives in the same order.

        The two branches share **one** loss instance either way; for the RLE term that
        means one flow learns one residual density from both branches' positives. R14
        fits a single density to a single regressor's errors; whether two branches
        that disagree early in training are better served by one shared density or by
        two independent ones is a question this project has no measurement for, and
        one flow is both the smaller claim and the smaller parameter count. It is
        normally called twice. With ``capture_loss`` active for the fit — decided once
        at :meth:`setup`, for single-device CUDA automatic optimization — a step in
        training mode scores both branches' row-independent density calls as one batch
        through :meth:`_paired_keypoint_terms`, while selection, each branch's mean and
        its ``alpha`` share stay separate. That batching has a gate of its own: the
        density must be the built-in :class:`~lucid_yolo.losses.rle_loss.RLELoss`
        passing ``_can_pair`` (no hooked, altered or compiled density module), and
        ``_score_branch``, ``_rle_on``, ``_oks_on`` and ``_gather_keypoint_positives``
        must keep their class implementations. It never reads the loss recorder's state,
        so batching continues after the recorder retires and on every step whose
        detection loss takes its native call — a customized criterion, or BF16 on a
        torch older than 2.9. Batched parameter-gradient summation can differ in
        floating-point rounding; this opt-in path promises the same objective rather
        than identical training bytes.

        Args:
            context: The step's shared quantities, carrying the padded point targets.
            out: The dual-branch loss output supplying both assignments and ``alpha``.

        Returns:
            The ``(rle, oks)`` pair of pre-gain blended scalars, on the prediction device.

        Raises:
            ValueError: If the batch's point count differs from the ``K`` the head was
                built for — a head predicting 17 points supervised by a 5-point
                annotation set is a misconfiguration no broadcast should paper over.
        """
        head_out, alpha = context.head_out, out.alpha
        gt_keypoints, gt_vis = context.gt_keypoints, context.gt_keypoint_vis
        assert gt_keypoints is not None  # the step pads them for exactly this task
        assert gt_vis is not None
        o2m_points, o2o_points = head_out.o2m_keypoints, head_out.o2o_keypoints
        o2m_sigma, o2o_sigma = head_out.o2m_keypoint_sigma, head_out.o2o_keypoint_sigma
        assert o2m_points is not None  # a "keypoints" module always builds both point stems
        assert o2o_points is not None
        assert o2m_sigma is not None
        assert o2o_sigma is not None
        predicted_points = int(o2o_points.shape[2])
        annotated_points = int(gt_keypoints.shape[2])
        if annotated_points and annotated_points != predicted_points:
            raise ValueError(
                f"the head predicts {predicted_points} keypoints but the batch annotates {annotated_points}; "
                f"num_keypoints must match the dataset's annotation schema"
            )
        if (
            self._capture_loss_active
            and self.training
            and type(self.rle_loss) is RLELoss
            and self.rle_loss._can_pair()
            and all(
                getattr(getattr(self, name), "__func__", None) is function for name, function in _EAGER_POSE_SCORING
            )
        ):
            return self._paired_keypoint_terms(context, out)
        o2m_rle, o2m_oks = self._score_branch(o2m_points, o2m_sigma, out.o2m_assign, context)
        o2o_rle, o2o_oks = self._score_branch(o2o_points, o2o_sigma, out.o2o_assign, context)
        return _blend(alpha, o2m_rle, o2o_rle), _blend(alpha, o2m_oks, o2o_oks)

    def _paired_keypoint_terms(self, context: _StepContext, out: DualLossOutput) -> tuple[Tensor, Tensor]:
        """Score the two existing pose branches with one shared density batch.

        Gathers, box frames, visibility and OKS stay per branch. Only the row-independent
        RLE density calls are batched; each selected subset retains its own mean and
        its original alpha weight. This is an opt-in part of CUDA pose loss execution.
        """
        head = context.head_out
        assert head.o2m_keypoints is not None and head.o2m_keypoint_sigma is not None
        assert head.o2o_keypoints is not None and head.o2o_keypoint_sigma is not None
        left = self._gather_keypoint_positives(head.o2m_keypoints, head.o2m_keypoint_sigma, out.o2m_assign, context)
        right = self._gather_keypoint_positives(head.o2o_keypoints, head.o2o_keypoint_sigma, out.o2o_assign, context)
        arguments = tuple(
            (
                normalize_keypoints_to_box(value.predicted, value.frame),
                value.raw_sigma,
                normalize_keypoints_to_box(value.annotated, value.frame),
                value.visibility,
            )
            for value in (left, right)
        )
        assert isinstance(self.rle_loss, RLELoss)
        left_rle, right_rle = self.rle_loss._forward_pair(*arguments)
        alpha = out.alpha
        return (
            _blend(alpha, left_rle, right_rle),
            _blend(alpha, self._oks_on(left), self._oks_on(right)),
        )

    def _score_branch(
        self, raw_points: Tensor, raw_sigma: Tensor, assign: AssignResult, context: _StepContext
    ) -> tuple[Tensor, Tensor]:
        """Gather one branch's positives once and score them with both pose terms (WP-178).

        The decode and the assignment gather are the shared cost; the two losses read
        the same :class:`_KeypointPositives` — RLE in the A71 box frame, OKS in input
        pixels — so a step pays for them once per branch, not once per term.

        Args:
            raw_points: ``(B, A, K, 2)`` raw point offsets of this branch.
            raw_sigma: ``(B, A, K, 2)`` raw, unactivated per-axis sigma of this branch.
            assign: This branch's assignment (``fg_mask`` and ``gt_index``).
            context: The step's shared quantities, for the anchor grid and the targets.

        Returns:
            The ``(rle, oks)`` pair of pre-gain scalars for this branch.
        """
        positives = self._gather_keypoint_positives(raw_points, raw_sigma, assign, context)
        return self._rle_on(positives), self._oks_on(positives)

    def _rle_on(self, positives: _KeypointPositives) -> Tensor:
        """Score gathered positives with the RLE term in the A71 box frame."""
        assert self.rle_loss is not None  # gated by the caller's task
        scored = self.rle_loss(
            normalize_keypoints_to_box(positives.predicted, positives.frame),
            positives.raw_sigma,
            normalize_keypoints_to_box(positives.annotated, positives.frame),
            positives.visibility,
        )
        return cast("Tensor", scored)

    def _oks_on(self, positives: _KeypointPositives) -> Tensor:
        """Score gathered positives with the OKS term in input pixels, the assigned box's area as scale (A75)."""
        assert self.oks_loss is not None  # gated by the caller's task
        extent = positives.frame[:, 2:] - positives.frame[:, :2]  # (P, 2) width, height
        scored = self.oks_loss(positives.predicted, positives.annotated, extent.prod(dim=-1), positives.visibility)
        return cast("Tensor", scored)

    def _branch_keypoint_loss(
        self, raw_points: Tensor, raw_sigma: Tensor, assign: AssignResult, context: _StepContext
    ) -> Tensor:
        """Decode one branch's points and score its assigned positives with the RLE loss.

        The positives come from :meth:`_gather_keypoint_positives`, the one gather both
        pose terms read. Sigma is passed on raw —
        :class:`~lucid_yolo.losses.rle_loss.RLELoss` owns the A65 sigmoid.

        Both point tensors are mapped into the assigned instance's box frame by
        :func:`normalize_keypoints_to_box` before they are scored (A71). The decode
        emits absolute input pixels and R14's ``sigma_hat`` is sigmoid-bounded into
        ``(0, 1)``; scoring the two together unmodified is what makes the residual
        ``O(10^3)`` and the flow non-finite. The normalization is the *loss's* frame
        only — nothing downstream of it is normalized, so what inference emits is
        unchanged.

        Args:
            raw_points: ``(B, A, K, 2)`` raw point offsets of this branch.
            raw_sigma: ``(B, A, K, 2)`` raw, unactivated per-axis sigma of this branch.
            assign: This branch's assignment (``fg_mask`` and ``gt_index``).
            context: The step's shared quantities, for the anchor grid and the targets.

        Returns:
            The scalar :class:`~lucid_yolo.losses.rle_loss.RLELoss` over every positive
            of the batch; a finite zero when there are none.
        """
        return self._rle_on(self._gather_keypoint_positives(raw_points, raw_sigma, assign, context))

    def _branch_oks_loss(
        self, raw_points: Tensor, raw_sigma: Tensor, assign: AssignResult, context: _StepContext
    ) -> Tensor:
        """Decode one branch's points and score its assigned positives with the OKS loss (WP-178).

        The same positives, in the same order, as :meth:`_branch_keypoint_loss` reads —
        through :meth:`_gather_keypoint_positives` — but scored **in input pixels**
        rather than the A71 box frame: OKS normalizes by the object scale itself, so
        the assigned box supplies its area ``w * h`` (A75) instead of a frame, and a
        box-normalized coordinate would be divided by that scale twice. ``raw_sigma``
        is accepted for signature parity with the RLE scorer and unread: the OKS term
        has no uncertainty to weight by.

        Args:
            raw_points: ``(B, A, K, 2)`` raw point offsets of this branch.
            raw_sigma: ``(B, A, K, 2)`` raw per-axis sigma of this branch; unread.
            assign: This branch's assignment (``fg_mask`` and ``gt_index``).
            context: The step's shared quantities, for the anchor grid and the targets.

        Returns:
            The scalar :class:`~lucid_yolo.losses.oks_loss.OKSLoss` over every positive
            of the batch; a finite zero when there are none.
        """
        return self._oks_on(self._gather_keypoint_positives(raw_points, raw_sigma, assign, context))

    def _gather_keypoint_positives(
        self, raw_points: Tensor, raw_sigma: Tensor, assign: AssignResult, context: _StepContext
    ) -> _KeypointPositives:
        """Decode one branch's points and gather everything its positives are scored with.

        The decode is the deployed one
        (:func:`~lucid_yolo.models.heads.keypoint.decode_keypoints`), so what a loss
        pulls towards the ground truth is what inference will emit.

        Every per-positive quantity is gathered by the **same** assignment, mirroring
        :meth:`_branch_mask_loss`: the anchor rows by ``fg_mask`` and the instances by
        ``gt_index``. Pairing positive ``k`` with instance ``k`` instead would supervise
        an anchor towards another object's pose — a perfectly finite number that no loss
        value reveals. The gather is the whole batch at once for that method's stated
        reason: a per-image boolean index makes the device report a data-dependent
        element count back to the host, once per image per branch. It is one method
        because two pose terms read it (WP-178): a second copy of the pairing logic
        would be a second place for the pairing to go wrong.

        Args:
            raw_points: ``(B, A, K, 2)`` raw point offsets of this branch.
            raw_sigma: ``(B, A, K, 2)`` raw, unactivated per-axis sigma of this branch.
            assign: This branch's assignment (``fg_mask`` and ``gt_index``).
            context: The step's shared quantities, for the anchor grid and the targets.

        Returns:
            The batch's positives, in input pixels, with the assigned instance's box
            as each one's frame; shaped-but-empty tensors when there are none.
        """
        gt_keypoints, gt_vis = context.gt_keypoints, context.gt_keypoint_vis
        assert gt_keypoints is not None
        assert gt_vis is not None
        decoded = decode_keypoints(raw_points, context.anchor_points, context.strides)
        num_points = decoded.shape[2]
        fg_mask, gt_index = assign.fg_mask, assign.gt_index
        batch = fg_mask.shape[0]
        max_positives = int(fg_mask.sum(dim=1).max()) if fg_mask.numel() else 0
        if max_positives == 0:
            # An empty gather would be legal but the *targets* would not: a batch with no
            # instances pads to K = 0, which broadcasts against the head's K only by
            # accident. Empties shaped like the predictions keep the zero honest.
            empty = decoded.new_zeros((0, num_points, _KEYPOINT_DIM))
            return _KeypointPositives(
                predicted=empty,
                raw_sigma=empty,
                annotated=empty,
                frame=decoded.new_zeros((0, _BOX_DIM)),
                visibility=gt_vis.new_zeros((0, num_points)),
            )

        # Positives first, ascending anchor index within each image (stable sort).
        order = torch.argsort(fg_mask.to(torch.uint8), dim=1, descending=True, stable=True)[:, :max_positives]
        valid = torch.gather(fg_mask, 1, order)  # (B, P) — padding rows are False
        # Padding rows carry the -1 "unassigned" sentinel, a legal but wrong index here;
        # clamped into range and then discarded by `keep`, so the target is arbitrary.
        rows = torch.gather(gt_index, 1, order).clamp(min=0)  # (B, P) instance ids
        anchor_index = order.view(batch, -1, 1, 1).expand(-1, -1, num_points, _KEYPOINT_DIM)
        keep = valid.reshape(-1)
        image_ids = torch.arange(batch, device=decoded.device).unsqueeze(-1).expand_as(rows)
        # One box per positive — the assigned instance's, so prediction and target are
        # mapped by the *same* frame and the difference a loss reads stays meaningful.
        return _KeypointPositives(
            predicted=decoded.gather(1, anchor_index).flatten(0, 1)[keep],
            raw_sigma=raw_sigma.gather(1, anchor_index).flatten(0, 1)[keep],
            annotated=gt_keypoints[image_ids, rows].flatten(0, 1)[keep],
            frame=context.gt_boxes[image_ids, rows].flatten(0, 1)[keep],
            visibility=gt_vis[image_ids, rows].flatten(0, 1)[keep],
        )

    def _segment_terms(self, context: _StepContext, out: DualLossOutput) -> tuple[Tensor, Tensor]:
        """Return the pre-gain ``(mask, semantic)`` terms for a segmentation batch.

        The instance masks are rasterized **once** per batch, on the prototype grid
        read off ``seg_out.prototypes``, and feed both terms (the semantic map is
        pooled down from them). A segmentation loader rasterises them in its workers
        and passes them in; this rasterises them here only when it did not, which is
        the hand-built-batch path. Either way the grid is checked against the
        prototypes actually produced, so a transport built for a different input size
        — or a topology that moved the prototype stride off A15 — fails loudly rather
        than supervising at the wrong resolution. The mask term mirrors
        :class:`~lucid_yolo.losses.dual_loss.DualBranchLoss`'s own composition —
        ``alpha * o2m + (1 - alpha) * o2o``, each branch against its own assignment
        — so the coefficient stem of the branch that survives into deployment (the
        one-to-one branch, the only one the segmentation decode reads) is trained,
        and the WP-035 ramp moves the mask supervision with the box supervision
        rather than against it.
        """
        seg_out, targets = context.seg_out, context.targets
        assert seg_out is not None  # the caller reaches this only for "segment"
        image_size, gt_boxes = context.image_size, context.gt_boxes
        prototypes = seg_out.prototypes
        proto_grid = (int(prototypes.shape[-2]), int(prototypes.shape[-1]))
        masks = context.masks
        if masks is None:
            masks = [instance_mask_targets(target, image_size, proto_grid) for target in targets]
        masks = [image_masks.to(prototypes.device) for image_masks in masks]
        grids = {tuple(image_masks.shape[-2:]) for image_masks in masks}
        if grids - {proto_grid}:
            raise ValueError(
                f"instance mask targets are on grid(s) {sorted(grids)} but the prototypes are {proto_grid}; "
                f"the loader rasterised for a different input size or prototype stride"
            )
        grid_boxes = scale_boxes_to_grid(gt_boxes, image_size, proto_grid)
        o2m_coeff, o2o_coeff = seg_out.detect.o2m_coeff, seg_out.detect.o2o_coeff
        assert o2m_coeff is not None  # a "segment" module always builds both coefficient stems
        assert o2o_coeff is not None
        # Densified once and shared by both branches: the ragged per-image stacks are
        # padded to a single (B, N, Hp, Wp) tensor here rather than inside the branch
        # loss, which would rebuild it for the second branch from the same input.
        padded_masks = _pad_mask_targets(masks, targets, prototypes)
        o2m_mask = self._branch_mask_loss(prototypes, o2m_coeff, out.o2m_assign, padded_masks, grid_boxes)
        o2o_mask = self._branch_mask_loss(prototypes, o2o_coeff, out.o2o_assign, padded_masks, grid_boxes)
        mask_term = _blend(out.alpha, o2m_mask, o2o_mask)
        return mask_term, self._semantic_term(seg_out, masks, targets)

    @staticmethod
    def _branch_mask_loss(
        prototypes: Tensor,
        coefficients: Tensor,
        assign: AssignResult,
        masks: Tensor,
        grid_boxes: Tensor,
    ) -> Tensor:
        """Score one branch's assembled masks against the ground truth it was assigned.

        Every per-positive quantity — the coefficient row, the target mask, and the
        crop box — is gathered by the **same** assignment: the anchor rows by
        ``fg_mask`` and the instances by ``gt_index``. Pairing positive ``k`` with
        instance ``k`` instead would produce a perfectly plausible mask of the wrong
        object, which no loss value reveals.

        Args:
            prototypes: ``(B, K, Hp, Wp)`` raw prototype maps.
            coefficients: ``(B, A, K)`` tanh mask coefficients of this branch.
            assign: This branch's assignment (``fg_mask`` and ``gt_index``).
            masks: ``(B, N, Hp, Wp)`` instance-mask targets, padded to the batch's
                largest instance count by the caller so both branches share one
                densification.
            grid_boxes: ``(B, N, 4)`` padded ground-truth boxes, already in the
                prototype grid's frame.

        Returns:
            The scalar :func:`~lucid_yolo.losses.mask_loss.instance_mask_loss` over
            every positive of the batch; a finite zero when there are none.

        Note:
            The whole batch is gathered at once, deliberately. The obvious form of
            this function loops over images and selects each one's positives with
            ``coefficients[i][fg_mask[i]]`` — but a boolean-mask index produces a
            tensor whose **shape depends on the data**, so the device must report
            its element count back to the host. Inside a per-image loop, run once
            per branch, that is ``2 * B`` forced device syncs per step, each
            draining the queue: measured at 1.2 s per step of pure stall at batch
            32, four times the cost of the entire detector. Padding to the batch's
            largest positive count and selecting once costs two syncs per branch
            regardless of batch size. The positives keep their anchor order within
            each image and their image order across the batch, so the summation
            order — and therefore the value, bit for bit — is the one the looped
            form produced.
        """
        fg_mask, gt_index = assign.fg_mask, assign.gt_index
        batch, _ = fg_mask.shape
        proto_grid = prototypes.shape[-2:]
        max_positives = int(fg_mask.sum(dim=1).max())  # one host read, not one per image
        if max_positives == 0:
            empty = prototypes.new_zeros((0, *proto_grid))
            return instance_mask_loss(empty, empty, prototypes.new_zeros((0, 4)))

        # Positives first, ascending anchor index within each image (stable sort).
        order = torch.argsort(fg_mask.to(torch.uint8), dim=1, descending=True, stable=True)[:, :max_positives]
        valid = torch.gather(fg_mask, 1, order)  # (B, P) — padding rows are False
        # Padding rows carry gt_index -1 (the "unassigned" sentinel), which is a valid
        # negative index in Python but out of bounds here; clamp them into range. Their
        # gathered values are discarded by `keep` below, so the clamp target is arbitrary.
        rows = torch.gather(gt_index, 1, order).clamp(min=0)  # (B, P) instance ids
        coefficient_rows = torch.gather(coefficients, 1, order.unsqueeze(-1).expand(-1, -1, coefficients.shape[-1]))

        image_ids = torch.arange(batch, device=prototypes.device).unsqueeze(-1).expand_as(rows)
        keep = valid.reshape(-1)
        return instance_mask_loss(
            assemble_masks(prototypes, coefficient_rows).flatten(0, 1)[keep],
            masks[image_ids, rows].flatten(0, 1)[keep],
            torch.gather(grid_boxes, 1, rows.unsqueeze(-1).expand(-1, -1, 4)).flatten(0, 1)[keep],
        )

    def _semantic_term(self, seg_out: SegmentOutput, masks: list[Tensor], targets: list[Targets]) -> Tensor:
        """Score the auxiliary semantic branch, or return zero when it is absent.

        The branch is training-only and returns ``None`` outside training mode
        (A17), so a validation step's segmentation loss carries the mask term
        alone — the auxiliary objective exists to shape the shared features during
        training and has nothing to report at eval.
        """
        logits = seg_out.semantic
        if logits is None:
            return torch.zeros((), device=seg_out.prototypes.device, dtype=seg_out.prototypes.dtype)
        grid = (int(logits.shape[-2]), int(logits.shape[-1]))
        num_classes = self.head.num_classes
        dense = torch.stack(
            [
                semantic_target(image_masks, target.labels, num_classes, grid)
                for image_masks, target in zip(masks, targets, strict=True)
            ]
        )
        return semantic_aux_loss(logits, dense).total

    def _log_loss(self, out: DualLossOutput, total: Tensor, stage: str, batch_size: int) -> None:
        """Log the combined total and every per-branch box/cls/l1 component."""
        self.log(f"{stage}/loss", total, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/o2m_box", out.o2m.box, batch_size=batch_size)
        self.log(f"{stage}/o2m_cls", out.o2m.cls, batch_size=batch_size)
        self.log(f"{stage}/o2m_l1", out.o2m.l1, batch_size=batch_size)
        self.log(f"{stage}/o2o_box", out.o2o.box, batch_size=batch_size)
        self.log(f"{stage}/o2o_cls", out.o2o.cls, batch_size=batch_size)
        self.log(f"{stage}/o2o_l1", out.o2o.l1, batch_size=batch_size)

    def _anchor_grid(self, height: int, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return the cached ``(anchor_points, stride_per_anchor)`` for a feature size.

        The grid depends only on the image height/width (feature sizes are the
        image size divided by each level stride), so it is computed once per size
        and moved onto ``device`` on retrieval.
        """
        key = (int(height), int(width))
        cached = self._anchor_cache.get(key)
        if cached is None:
            feature_sizes = [(key[0] // stride, key[1] // stride) for stride in _STRIDES]
            cached = make_anchor_points(feature_sizes, list(_STRIDES))
            self._anchor_cache[key] = cached
        points, strides = cached
        return points.to(device), strides.to(device)


#: The unmodified public forward; instance/class overrides retain their eager dispatch.
_EAGER_NETWORK_FORWARD = DetectionLitModule.forward

#: Segment composition overrides keep their native execution.
_EAGER_SEGMENT_FORWARD = DetectionLitModule.forward_segmentation

#: Consumers whose native calls retain gradients to every oriented output, even on empty batches.
_ORIENTED_METHODS = tuple(
    (name, getattr(DetectionLitModule, name))
    for name in ("training_step", "_shared_step", "_task_extra_loss", "_oriented_extra_loss", "_branch_oriented_terms")
)
#: The oriented objective helpers as imported, against which :func:`_oriented_replay_breach` compares.
_ORIENTED_HELPERS = _oriented_helpers()

_EAGER_POSE_SCORING = tuple(
    (name, getattr(DetectionLitModule, name))
    for name in ("_score_branch", "_rle_on", "_oks_on", "_gather_keypoint_positives")
)
