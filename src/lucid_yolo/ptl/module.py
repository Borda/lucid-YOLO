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

    ``"keypoints"`` (WP-132) adds ``keypoint_gain * rle`` — R14's residual
    log-likelihood over the head's point predictions, or whichever objective
    ``keypoint_loss`` put in that slot (WP-135). Structurally it is the
    ``"segment"`` shape rather than the ``"obb"`` one: nothing is replaced, every
    detection term keeps its gain, and the pose term rides on top. That follows
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision

from lucid_yolo.assign import make_anchor_points
from lucid_yolo.decode.common import BOX_CORNERS, SCORE_COLUMN
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.dota_eval import MAX_DETECTIONS, evaluate_rotated_map, rotated_detections_to_predictions
from lucid_yolo.eval.segment_decode import decode_instance_masks
from lucid_yolo.losses.dual_loss import DualBranchLoss, DualLossOutput
from lucid_yolo.losses.keypoint_nll_loss import LaplaceNLLLoss
from lucid_yolo.losses.mask_loss import instance_mask_loss
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
from lucid_yolo.models.heads.detect import DEFAULT_NUM_COEFFS, DualHeadOutput, decode_ltrb
from lucid_yolo.models.heads.keypoint import decode_keypoints
from lucid_yolo.models.heads.obb import decode_rboxes, o2o_rotated_topk
from lucid_yolo.models.heads.proto import assemble_masks
from lucid_yolo.optim.musgd import MuSGD
from lucid_yolo.optim.schedule import warmup_decay_factor
from lucid_yolo.ptl.seg_targets import instance_mask_targets, scale_boxes_to_grid, semantic_target

if TYPE_CHECKING:
    from pytorch_lightning.utilities.types import OptimizerLRScheduler

    from lucid_yolo.assign.tal import AssignResult
    from lucid_yolo.data.targets import Targets

__all__ = ["DetectionLitModule", "pad_keypoints", "pad_rboxes", "pad_targets"]

#: Feature-level input-pixel strides of the P3/P4/P5 detection head (8, 16, 32).
_STRIDES: tuple[int, int, int] = (8, 16, 32)

#: Task names the module accepts. All four are wired: ``"detect"`` is the dual
#: detection objective alone, ``"segment"`` adds the WP-087 mask terms, ``"obb"``
#: swaps the two box terms for the WP-088 rotated ones and adds the angle term,
#: and ``"keypoints"`` adds the WP-132 residual log-likelihood term.
_TASKS: tuple[str, ...] = ("detect", "segment", "obb", "keypoints")

#: Keypoint objectives a ``"keypoints"`` run may select between, each mapped to the
#: class that implements it (WP-135). ``"rle"`` is R14's full residual
#: log-likelihood and the default every existing config, checkpoint and accepted
#: result was produced with; ``"laplace_nll"`` is R14 Table 7's own flow-free
#: control, present so WP-125 can state a direction of effect rather than a bare
#: number. Keyed by plain strings for the reason ``_TASKS`` and
#: ``ROTATED_IOU_FORMS`` are: the value arrives from YAML. One mapping rather than a
#: name tuple beside a constructor branch, so the accepted set and the thing each
#: name builds cannot drift apart.
_KEYPOINT_LOSSES: dict[str, type[RLELoss] | type[LaplaceNLLLoss]] = {
    "rle": RLELoss,
    "laplace_nll": LaplaceNLLLoss,
}

#: Column count of an ``xyxy`` axis-aligned box.
_BOX_DIM = 4

#: Column count of a long-edge rotated box ``(cx, cy, w, h, theta)``.
_RBOX_DIM = 5

#: Coordinate count of a keypoint ``(x, y)``.
_KEYPOINT_DIM = 2

#: Floor, in input pixels, on the per-axis box extent :func:`normalize_keypoints_to_box`
#: divides by. One pixel: a box thinner than that carries no pose worth a frame, and the
#: unclamped division would answer a finite annotation with an infinity.
_MIN_BOX_EXTENT = 1.0

#: Column index of the integral class label within the A9 detection tuple.
_LABEL_COLUMN = 5

#: Detections per image whose masks are decoded for the epoch ``val/segm_mAP``.
#: COCO's own ``maxDets`` cap is 100, and the metric applies it anyway, so
#: decoding the fixed 300-row decoder output in full would cost two thirds of the
#: mask work to produce rows the evaluation then discards.
_VAL_SEGM_MAX_DET = 100


def pad_targets(targets: list[Targets]) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a ragged batch of per-image targets into dense ground-truth tensors.

    The batch's :class:`~lucid_yolo.data.targets.Targets` each carry their own
    instance count, so they are padded to the batch maximum ``N_max`` and paired
    with a boolean mask marking the real (non-padding) rows — the dense form the
    :class:`~lucid_yolo.losses.dual_loss.DualBranchLoss` assigners consume. An image
    with no instances contributes an all-``False`` mask row (no positives), and a
    batch in which *every* image is empty yields ``N_max = 0`` tensors, which the
    loss handles as its finite no-ground-truth case.

    Args:
        targets: Length-``B`` list of per-image targets (the datamodule batch's
            second element). All tensors must share one device.

    Returns:
        A triple ``(gt_boxes, gt_labels, gt_mask)`` with shapes ``(B, N_max, 4)``
        float32 ``xyxy`` boxes, ``(B, N_max)`` int64 class ids, and ``(B, N_max)``
        bool mask. Padding rows are zero boxes / zero labels / ``False`` mask.

    Examples:
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> a = Targets(boxes=torch.tensor([[0.0, 0.0, 4.0, 4.0]]), labels=torch.tensor([3]))
        >>> b = Targets.empty()
        >>> boxes, labels, mask = pad_targets([a, b])
        >>> boxes.shape, labels.shape, mask.shape
        (torch.Size([2, 1, 4]), torch.Size([2, 1]), torch.Size([2, 1]))
        >>> mask.tolist()
        [[True], [False]]
    """
    batch_size = len(targets)
    counts = [int(target.boxes.shape[0]) for target in targets]
    max_n = max(counts) if counts else 0
    device = targets[0].boxes.device if targets else torch.device("cpu")
    gt_boxes = torch.zeros((batch_size, max_n, _BOX_DIM), dtype=torch.float32, device=device)
    gt_labels = torch.zeros((batch_size, max_n), dtype=torch.int64, device=device)
    gt_mask = torch.zeros((batch_size, max_n), dtype=torch.bool, device=device)
    for index, (target, count) in enumerate(zip(targets, counts, strict=True)):
        if count == 0:
            continue
        gt_boxes[index, :count] = target.boxes
        gt_labels[index, :count] = target.labels
        gt_mask[index, :count] = True
    return gt_boxes, gt_labels, gt_mask


def pad_rboxes(targets: list[Targets]) -> Tensor:
    """Pad a ragged batch's rotated boxes into one dense ``(B, N_max, 5)`` tensor (WP-088).

    The rotated companion of :func:`pad_targets`, kept as its own function rather than
    a fourth element of that tuple: only ``task="obb"`` needs it, and widening a return
    every caller unpacks would put an oriented concern in the detection path.

    The padding is by the **instance** axis, not by a rotated-box axis of its own. WP-056
    pins ``rboxes[i]`` to the same object as ``boxes[i]`` and ``labels[i]``, and that
    pairing is the whole reason the assignment computed on the axis-aligned envelopes may
    be reused to gather rotated targets. An image whose two axes disagree is therefore a
    hard error here rather than something to reconcile: whichever way it were resolved,
    some anchor would be supervised towards another instance's orientation.

    Args:
        targets: The length-``B`` per-image target list, each carrying one rotated box
            per instance. All tensors must share one device.

    Returns:
        ``(B, N_max, 5)`` float32 long-edge rotated boxes on the targets' device, padded
        with zero rows exactly where :func:`pad_targets` pads with ``False``.

    Raises:
        ValueError: If any image's rotated-box count differs from its instance count —
            including the all-important case of *zero* rotated boxes beside real ones,
            which is what a detection loader hands an oriented run.

    Examples:
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> one = Targets(
        ...     boxes=torch.tensor([[0.0, 0.0, 4.0, 2.0]]),
        ...     labels=torch.tensor([3]),
        ...     rboxes=torch.tensor([[2.0, 1.0, 4.0, 2.0, 0.0]]),
        ... )
        >>> pad_rboxes([one, Targets.empty()]).shape
        torch.Size([2, 1, 5])
        >>> pad_rboxes([Targets(boxes=torch.zeros(1, 4), labels=torch.zeros(1, dtype=torch.int64))])
        Traceback (most recent call last):
        ...
        ValueError: image 0 carries 1 instances but 0 rotated boxes; task='obb' needs one per instance
    """
    counts = [int(target.boxes.shape[0]) for target in targets]
    for index, (target, count) in enumerate(zip(targets, counts, strict=True)):
        if int(target.rboxes.shape[0]) != count:
            raise ValueError(
                f"image {index} carries {count} instances but {int(target.rboxes.shape[0])} rotated boxes; "
                f"task='obb' needs one per instance"
            )
    max_n = max(counts) if counts else 0
    device = targets[0].rboxes.device if targets else torch.device("cpu")
    padded = torch.zeros((len(targets), max_n, _RBOX_DIM), dtype=torch.float32, device=device)
    for index, (target, count) in enumerate(zip(targets, counts, strict=True)):
        if count:
            padded[index, :count] = target.rboxes
    return padded


def pad_keypoints(targets: list[Targets]) -> tuple[Tensor, Tensor]:
    """Pad a ragged batch's keypoints and visibilities onto the instance axis (WP-132).

    The pose companion of :func:`pad_targets`, and a sibling of :func:`pad_rboxes` in
    both shape and argument. Like the rotated boxes, keypoints are padded by the
    **instance** axis rather than by an axis of their own: WP-120 pins
    ``keypoints[i]`` to the same object as ``boxes[i]`` and ``labels[i]``, and that
    pairing is the only reason the assignment computed on the boxes may be reused to
    gather point targets. An image whose two axes disagree is a hard error for the
    same reason it is one there — whichever way it were reconciled, some anchor would
    be supervised towards another instance's pose.

    The visibility padding is zeros, which is not merely a filler value: A66 reads
    ``v == 0`` as "no annotation exists", so
    :class:`~lucid_yolo.losses.rle_loss.RLELoss` already excludes exactly those
    points. The padding rows are therefore inert in the loss by the same rule that
    excludes a genuinely unlabeled point, rather than by a second mechanism that
    could disagree with it.

    Args:
        targets: The length-``B`` per-image target list, each carrying one point set
            per instance. All tensors must share one device.

    Returns:
        A pair ``(keypoints, visibility)`` of shapes ``(B, N_max, K, 2)`` float32 and
        ``(B, N_max, K)`` int64, padded with zero coordinates and zero (unlabeled)
        visibility exactly where :func:`pad_targets` pads with ``False``. ``K`` is
        the batch's shared point count, and ``0`` for a batch holding no instances
        at all.

    Raises:
        ValueError: If any image's keypoint-set count differs from its instance
            count — including the case of *zero* keypoints beside real instances,
            which is what a plain detection loader hands a keypoints run — or if two
            images disagree on ``K``, which no single head could predict for both.

    Examples:
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> one = Targets(
        ...     boxes=torch.tensor([[0.0, 0.0, 4.0, 2.0]]),
        ...     labels=torch.tensor([3]),
        ...     keypoints=torch.tensor([[[1.0, 1.0], [3.0, 1.0]]]),
        ...     keypoint_vis=torch.tensor([[2, 1]]),
        ... )
        >>> coords, visibility = pad_keypoints([one, Targets.empty()])
        >>> coords.shape, visibility.shape
        (torch.Size([2, 1, 2, 2]), torch.Size([2, 1, 2]))
        >>> visibility.tolist()  # the padded image is unlabeled everywhere (A66)
        [[[2, 1]], [[0, 0]]]
        >>> pad_keypoints([Targets(boxes=torch.zeros(1, 4), labels=torch.zeros(1, dtype=torch.int64))])
        Traceback (most recent call last):
        ...
        ValueError: image 0 carries 1 instances but 0 keypoint sets; task='keypoints' needs one per instance
    """
    counts = [int(target.boxes.shape[0]) for target in targets]
    for index, (target, count) in enumerate(zip(targets, counts, strict=True)):
        if int(target.keypoints.shape[0]) != count:
            raise ValueError(
                f"image {index} carries {count} instances but {int(target.keypoints.shape[0])} keypoint sets; "
                f"task='keypoints' needs one per instance"
            )
    point_counts = {int(target.keypoints.shape[1]) for target in targets if target.keypoints.shape[0]}
    if len(point_counts) > 1:
        raise ValueError(f"the batch's images disagree on the keypoint count K: {sorted(point_counts)}")
    num_points = point_counts.pop() if point_counts else 0
    max_n = max(counts) if counts else 0
    device = targets[0].keypoints.device if targets else torch.device("cpu")
    coords = torch.zeros((len(targets), max_n, num_points, _KEYPOINT_DIM), dtype=torch.float32, device=device)
    visibility = torch.zeros((len(targets), max_n, num_points), dtype=torch.int64, device=device)
    for index, (target, count) in enumerate(zip(targets, counts, strict=True)):
        if count:
            coords[index, :count] = target.keypoints
            visibility[index, :count] = target.keypoint_vis
    return coords, visibility


def normalize_keypoints_to_box(points: Tensor, boxes: Tensor) -> Tensor:
    """Map absolute point pixels into their instance's box frame (A71, WP-132).

    The frame :class:`~lucid_yolo.losses.rle_loss.RLELoss` needs and this project's
    decode does not produce. R14 predicts ``sigma_hat`` through a sigmoid, so
    ``sigma_hat`` lies in ``(0, 1)`` by construction (R14 sec. 3.3), and the residual
    it scales is ``x_bar = (mu_g - mu_hat) / sigma_hat``. A scale bounded above by 1
    is only expressive in a frame whose errors are ``O(1)``: fed absolute pixels, the
    *smallest* residual the model can express for a 40 px error is 40, and an early
    prediction on the far side of a 256 px canvas reaches into the hundreds. R14 never
    states the frame — it works top-down on a person crop resized to a fixed input, so
    the crop *is* the normalization and the paper has no occasion to name it. This
    function supplies the dense-detector equivalent: the assigned ground-truth box
    plays the part R14's crop plays.

    Measured, not reasoned: run un-normalized, the keypoint term enters at 1300 of a
    1342 total loss and the flow's coupling layers overflow to non-finite on the
    second step. See A71 and the ``#wp-132`` log entry.

    Normalization is per axis — ``x`` by the box width, ``y`` by its height — which is
    exactly what resizing a crop to a fixed input does, rather than the single
    ``sqrt(area)`` scalar OKS uses. The two agree on a square object and differ only in
    how an elongated one distributes its tolerance; per-axis is the one that makes the
    frame an actual crop.

    The translation term cancels: the loss reads ``mu_hat`` and ``mu_gt`` only through
    their difference, and both are mapped by the same box. It is applied anyway, so the
    returned values mean what the name says — ``0`` at the box's top-left corner, ``1``
    at its bottom-right, a point outside the box outside ``[0, 1]`` — rather than being
    a bare division that happens to be sufficient.

    Args:
        points: ``(..., K, 2)`` absolute point coordinates in input pixels, as
            :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints` returns them.
        boxes: ``(..., 4)`` ``xyxy`` boxes in input pixels, aligned with ``points`` on
            every leading axis — one box per point set.

    Returns:
        The points in their box's frame, shaped as given. A box narrower than one pixel
        on an axis is treated as one pixel wide there: a degenerate box carries no pose
        to normalize by, and the alternative is a division that returns infinity for a
        finite annotation.

    Examples:
        >>> import torch
        >>> points = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
        >>> boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        >>> normalize_keypoints_to_box(points, boxes)  # corner to corner
        tensor([[[0., 0.],
                 [1., 1.]]])
    """
    origin = boxes[..., :2].unsqueeze(-2)
    extent = (boxes[..., 2:] - boxes[..., :2]).clamp(min=_MIN_BOX_EXTENT).unsqueeze(-2)
    return (points - origin) / extent


#: A step batch. The datamodule's transfer hook produces the three-element form; the
#: two-element form is the hand-built feed a direct caller passes to
#: :meth:`DetectionLitModule.training_step`.
StepBatch = tuple[Tensor, list["Targets"]] | tuple[Tensor, list["Targets"], list[Tensor] | None]


def _split_batch(batch: StepBatch) -> tuple[Tensor, list[Targets], list[Tensor] | None]:
    """Split a step batch into images, targets and the loader's mask stacks.

    :meth:`~lucid_yolo.ptl.datamodule.DetectionDataModule.on_after_batch_transfer`
    hands over a third element — the per-image instance masks a segmentation loader
    rasterised in its workers, or ``None``. A two-element batch is still accepted so
    that a hand-built ``(images, targets)`` feed (every direct-call test, every
    script that drives the module without the datamodule) keeps working and simply
    rasterises in the step, as it always did.

    Args:
        batch: The ``(images, targets)`` or ``(images, targets, masks)`` step batch.

    Returns:
        The ``(images, targets, masks)`` triple, with ``masks`` ``None`` when the
        batch carried none.

    Examples:
        >>> import torch
        >>> images, targets, masks = _split_batch((torch.zeros(1, 3, 4, 4), []))
        >>> masks is None
        True
    """
    images, targets, *rest = batch
    return images, targets, rest[0] if rest else None


@dataclass(frozen=True)
class _StepContext:
    """Everything one step computed once and its task-conditional terms may read.

    The three tasks need overlapping but different slices of a step: the mask terms
    want the prototypes and the ground-truth boxes in pixels, the oriented terms want
    the raw angles and the anchor grid, and neither wants the other's. Passing them as
    positional arguments put :meth:`DetectionLitModule._task_extra_loss` at seven
    required parameters before the oriented path added any, and the eighth would have
    been a lint failure standing in for a design one — a dispatch point whose signature
    grows with every task is a dispatch point that will eventually be typed wrongly.

    Attributes:
        head_out: The dense dual-head output of this step's forward.
        seg_out: The segmentation forward output, or ``None`` for a task with no
            mask branches.
        targets: The batch's ragged per-image targets.
        gt_boxes: ``(B, N, 4)`` padded ground-truth boxes in input pixels.
        gt_rboxes: ``(B, N, 5)`` padded rotated ground truths, or ``None`` when the
            task does not supervise orientation.
        gt_keypoints: ``(B, N, K, 2)`` padded point ground truths in input pixels,
            or ``None`` when the task does not supervise pose.
        gt_keypoint_vis: ``(B, N, K)`` padded COCO visibilities paired with
            ``gt_keypoints``, or ``None`` alongside it.
        anchor_points: ``(A, 2)`` anchor centres in input pixels.
        strides: ``(A,)`` per-anchor level stride.
        image_size: The batch's ``(height, width)`` in input pixels.
        masks: The loader's per-image instance masks, or ``None`` to rasterise them
            in the step.
    """

    head_out: DualHeadOutput
    seg_out: SegmentOutput | None
    targets: list[Targets]
    gt_boxes: Tensor
    gt_rboxes: Tensor | None
    gt_keypoints: Tensor | None
    gt_keypoint_vis: Tensor | None
    anchor_points: Tensor
    strides: Tensor
    image_size: tuple[int, int]
    masks: list[Tensor] | None


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
            ``keypoint_gain``, leaving every detection term as it was.
            Defaults to ``"detect"``.
        num_keypoints: Point count ``K`` both head branches predict under
            ``task="keypoints"``; **required** for that task and ignored otherwise.
            It has no default on purpose — ``K`` describes the dataset's annotation
            schema, not the method (the module docstring argues the case).
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

    Raises:
        ValueError: If ``task`` is not one of ``"detect"``, ``"segment"``, ``"obb"``,
            ``"keypoints"``, if ``rotated_iou_form`` is not a known rotated-IoU form,
            if ``keypoint_loss`` is not a known keypoint objective, or if
            ``task="keypoints"`` was asked for without a ``num_keypoints``.

    Examples:
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4)
        >>> images = torch.zeros(1, 3, 160, 160)
        >>> targets = [Targets(boxes=torch.tensor([[8.0, 8.0, 40.0, 40.0]]), labels=torch.tensor([1]))]
        >>> loss = module.training_step((images, targets), 0)  # doctest: +SKIP
        >>> bool(torch.isfinite(loss))  # doctest: +SKIP
        True
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
    ) -> None:
        super().__init__()
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
        self.save_hyperparameters()
        self._task = task
        self._mask_gain: float = mask_gain
        self._semantic_gain: float = semantic_gain
        self._angle_gain: float = angle_gain
        self._keypoint_gain: float = keypoint_gain
        self._keypoint_loss: str = keypoint_loss
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
        self._w_muon = w_muon
        self._w_sgd = w_sgd

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
        if task == "segment":
            self.proto_fusion, self.protonet, self.semantic = build_segmentation_stages(
                self.neck.channels, num_classes, DEFAULT_NUM_COEFFS
            )
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
        #: module's ``state_dict`` — and older checkpoints — are unaffected.
        self._val_decoder = TopKDecoder()
        self._val_map = MeanAveragePrecision(backend="faster_coco_eval", box_format="xyxy")
        self._val_map.warn_on_many_detections = False

        #: Epoch mask mAP, for ``"segment"`` only (WP-087). A second metric rather
        #: than ``iou_type=("bbox", "segm")`` on :attr:`_val_map`, because the two
        #: are scored in **different frames** — boxes in letterbox pixels, masks on
        #: the prototype grid — and one metric holding both would report each
        #: instance's area under whichever frame torchmetrics picked, silently
        #: mis-bucketing the small/medium/large splits. Split in two, each metric's
        #: inputs are self-consistent. ``None`` for a detection module, whose
        #: validation must not pay for mask machinery it has no branch for.
        self._val_segm = (
            MeanAveragePrecision(backend="faster_coco_eval", iou_type="segm") if task == "segment" else None
        )
        if self._val_segm is not None:
            self._val_segm.warn_on_many_detections = False
        #: Whether any batch fed :attr:`_val_segm` this epoch — ``compute`` on an
        #: untouched metric raises, and a loader without mask targets never feeds it.
        self._val_segm_seen = False

        #: Epoch rotated mAP accumulators, for ``"obb"`` only (WP-088). Plain lists
        #: rather than a :class:`~torchmetrics.Metric`: the WP-063 protocol is not a
        #: torchmetrics implementation, it fixes four constants that library defaults
        #: get wrong (A46-A48), and mAP is not a per-batch quantity that can be
        #: averaged — the whole epoch's ranked detections have to meet at once. Both
        #: lists hold CPU tensors and are cleared at every epoch end.
        self._val_rotated_preds: list[dict[str, Tensor]] = []
        self._val_rotated_targets: list[dict[str, Tensor]] = []

        #: Per-image-size cache of ``(anchor_points, stride_per_anchor)`` on CPU.
        self._anchor_cache: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}

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
            >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4)
            >>> module.task
            'detect'
        """
        return self._task

    @property
    def alpha(self) -> float:
        """One-to-many branch weight of the dual loss (the WP-035 schedule seam).

        Returns:
            The current :attr:`DualBranchLoss.alpha`; setting it rewrites the
            underlying loss attribute so a scheduler can update the ramp per epoch.

        Examples:
            >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4)
            >>> module.alpha = 0.8
            >>> module.loss.alpha
            0.8
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
            >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4)
            >>> module.alpha  # seeded value before any epoch starts
            0.5
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
            >>> import torch
            >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=4).eval()
            >>> with torch.no_grad():
            ...     out = module(torch.zeros(1, 3, 160, 160))
            >>> out.o2o_cls.shape[-1]
            4
        """
        return cast("DualHeadOutput", self.head(self.neck(self.backbone(images))))

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
        """
        if self._task != "segment":
            raise ValueError(f"forward_segmentation requires task='segment'; this module's task is {self._task!r}")
        features: tuple[Tensor, Tensor, Tensor] = self.neck(self.backbone(images))
        detect = cast("DualHeadOutput", self.head(features))
        fused: Tensor = self.proto_fusion(features)
        return SegmentOutput(detect=detect, prototypes=self.protonet(fused), semantic=self.semantic(fused))

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
        if self.task != "obb":
            ground_truth = [{"boxes": t.boxes.cpu(), "labels": t.labels.cpu()} for t in targets]
            self._val_map.update(preds, ground_truth)
        if self._val_segm is not None and seg_out is not None and masks is not None:
            image_size = (int(images.shape[-2]), int(images.shape[-1]))
            self._update_val_segm(seg_out, detections, anchor_indices, targets, masks, image_size)
        if self._task == "obb":
            self._update_val_rotated(head_out, targets, anchor_points, strides)
        return loss

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
        self._val_rotated_preds.extend(rotated_detections_to_predictions(detections.detach()))
        self._val_rotated_targets.extend(
            {
                "rboxes": target.rboxes.detach().cpu(),
                "labels": target.labels.detach().cpu().to(torch.long),
                "difficult": target.difficult.detach().cpu(),
            }
            for target in targets
        )

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
        for index, target in enumerate(targets):
            decoded = decode_instance_masks(
                prototypes[index : index + 1],
                kept_coefficients[index : index + 1],
                grid_boxes[index : index + 1],
                image_size=proto_grid,
            )[0].cpu()
            keep = cpu_detections[index, :, SCORE_COLUMN] > 0.0
            preds.append(
                {
                    "scores": cpu_detections[index, keep, SCORE_COLUMN],
                    "labels": cpu_detections[index, keep, _LABEL_COLUMN].long(),
                    "masks": decoded[keep],
                }
            )
            ground_truth.append({"labels": target.labels.cpu(), "masks": masks[index].to(torch.bool).cpu()})
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
        """
        if self.task != "obb":
            computed = self._val_map.compute()
            self.log("val/mAP", computed["map"].to(torch.float32), prog_bar=True)
            self._val_map.reset()
        if self._val_segm is not None and self._val_segm_seen:
            self.log("val/segm_mAP", self._val_segm.compute()["map"].to(torch.float32), prog_bar=True)
            self._val_segm.reset()
            self._val_segm_seen = False
        if self._val_rotated_preds:
            self._log_rotated_map()

    def _log_rotated_map(self) -> None:
        """Score the epoch's accumulated oriented detections and clear the buffers.

        ``val/rotated_mAP50`` is put on the progress bar rather than ``val/rotated_mAP``
        because the DoD of the oriented path is stated at IoU 0.50; both are logged, so
        the stricter figure is in ``metrics.csv`` either way. The buffers are cleared
        unconditionally — a metric that silently carried last epoch's detections into
        this one would improve monotonically for reasons that have nothing to do with
        the model.
        """
        stats = evaluate_rotated_map(self._val_rotated_preds, self._val_rotated_targets)
        self.log("val/rotated_mAP50", torch.tensor(stats["map_50"], dtype=torch.float32), prog_bar=True)
        self.log("val/rotated_mAP", torch.tensor(stats["map"], dtype=torch.float32))
        self._val_rotated_preds.clear()
        self._val_rotated_targets.clear()

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
        )
        schedule_off = self._lrf >= 1.0 and self._warmup_epochs <= 0
        trainer = self._trainer
        max_epochs = None if trainer is None else trainer.max_epochs
        if schedule_off or trainer is None or max_epochs is None or max_epochs <= 0:
            return optimizer
        total_steps = max(1, int(trainer.estimated_stepping_batches))
        warmup_steps = round(total_steps * self._warmup_epochs / max_epochs)
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
        seg_out = self.forward_segmentation(images) if self._task == "segment" else None
        head_out = self(images) if seg_out is None else seg_out.detect
        anchor_points, strides = self._anchor_grid(images.shape[-2], images.shape[-1], images.device)
        o2m_boxes = decode_ltrb(head_out.o2m_box, anchor_points, strides)
        o2o_boxes = decode_ltrb(head_out.o2o_box, anchor_points, strides)
        gt_boxes, gt_labels, gt_mask = pad_targets(targets)
        gt_boxes = gt_boxes.to(images.device)
        gt_labels = gt_labels.to(images.device)
        gt_mask = gt_mask.to(images.device)
        gt_rboxes = pad_rboxes(targets).to(images.device) if self._task == "obb" else None
        gt_keypoints, gt_keypoint_vis = None, None
        if self._task == "keypoints":
            gt_keypoints, gt_keypoint_vis = pad_keypoints(targets)
            gt_keypoints = gt_keypoints.to(images.device)
            gt_keypoint_vis = gt_keypoint_vis.to(images.device)
        out = self.loss(
            head_out.o2m_cls,
            o2m_boxes,
            head_out.o2o_cls,
            o2o_boxes,
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

    def _task_extra_loss(self, context: _StepContext, out: DualLossOutput, stage: str) -> Tensor:
        """Return the task-conditional extra loss: the segment, oriented or pose terms, else zero.

        For ``"segment"`` this is ``mask_gain * mask + semantic_gain * semantic``
        (A38); for ``"obb"`` it is ``box_gain * rbox + l1_gain * rl1 + angle_gain *
        angle``, which together with the dual loss's classification term — the only
        term left with a non-zero gain under that task — *is* the whole oriented
        objective rather than an addition to a detection one (module docstring); for
        ``"keypoints"`` it is ``keypoint_gain * rle``, an addition in the segment
        sense, with every detection term left at its own gain.
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
        if self._task == "obb":
            return self._oriented_extra_loss(context, out, stage)
        if self._task == "keypoints":
            keypoint_term = self._keypoint_term(context, out)
            self.log(f"{stage}/keypoint", keypoint_term, batch_size=len(context.targets))
            return self._keypoint_gain * keypoint_term
        if context.seg_out is None:
            return torch.zeros_like(out.total)
        mask_term, semantic_term = self._segment_terms(context, out)
        self.log(f"{stage}/mask", mask_term, batch_size=len(context.targets))
        self.log(f"{stage}/semantic", semantic_term, batch_size=len(context.targets))
        return self._mask_gain * mask_term + self._semantic_gain * semantic_term

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
        rbox = alpha * o2m.rbox + (1.0 - alpha) * o2o.rbox
        rl1 = alpha * o2m.rl1 + (1.0 - alpha) * o2o.rl1
        angle = alpha * o2m.angle + (1.0 - alpha) * o2o.angle
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

    def _keypoint_term(self, context: _StepContext, out: DualLossOutput) -> Tensor:
        """Return the pre-gain R14 residual-log-likelihood term for a pose batch (WP-132).

        Both branches are scored and blended by the WP-035 ``alpha``, exactly as the
        mask and oriented terms are, and for the same reason: the one-to-one branch is
        the one a NMS-free pose decode reads, so supervising the one-to-many point
        stems alone would ship an untrained keypoint head, and the shared ramp moves
        point supervision *with* box supervision rather than against it.

        The two branches share **one** :class:`~lucid_yolo.losses.rle_loss.RLELoss`,
        so one flow learns one residual density from both branches' positives. R14
        fits a single density to a single regressor's errors; whether two branches
        that disagree early in training are better served by one shared density or by
        two independent ones is a question this project has no measurement for, and
        one flow is both the smaller claim and the smaller parameter count. It is
        called twice rather than once over concatenated positives so each branch's
        term can be weighted by its own ``alpha`` share.

        Args:
            context: The step's shared quantities, carrying the padded point targets.
            out: The dual-branch loss output supplying both assignments and ``alpha``.

        Returns:
            The pre-gain blended scalar, on the prediction device.

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
        o2m = self._branch_keypoint_loss(o2m_points, o2m_sigma, out.o2m_assign, context)
        o2o = self._branch_keypoint_loss(o2o_points, o2o_sigma, out.o2o_assign, context)
        return alpha * o2m + (1.0 - alpha) * o2o

    def _branch_keypoint_loss(
        self, raw_points: Tensor, raw_sigma: Tensor, assign: AssignResult, context: _StepContext
    ) -> Tensor:
        """Decode one branch's points and score its assigned positives with the RLE loss.

        The decode is the deployed one
        (:func:`~lucid_yolo.models.heads.keypoint.decode_keypoints`), so what the loss
        pulls towards the ground truth is what inference will emit, and sigma is passed
        on raw — :class:`~lucid_yolo.losses.rle_loss.RLELoss` owns the A65 sigmoid.

        Both point tensors are then mapped into the assigned instance's box frame by
        :func:`normalize_keypoints_to_box` before they are scored (A71). The decode
        emits absolute input pixels and R14's ``sigma_hat`` is sigmoid-bounded into
        ``(0, 1)``; scoring the two together unmodified is what makes the residual
        ``O(10^3)`` and the flow non-finite. The normalization is the *loss's* frame
        only — nothing downstream of it is normalized, so what inference emits is
        unchanged.

        Every per-positive quantity is gathered by the **same** assignment, mirroring
        :meth:`_branch_mask_loss`: the anchor rows by ``fg_mask`` and the instances by
        ``gt_index``. Pairing positive ``k`` with instance ``k`` instead would supervise
        an anchor towards another object's pose — a perfectly finite number that no loss
        value reveals. The gather is the whole batch at once for that method's stated
        reason: a per-image boolean index makes the device report a data-dependent
        element count back to the host, once per image per branch.

        Args:
            raw_points: ``(B, A, K, 2)`` raw point offsets of this branch.
            raw_sigma: ``(B, A, K, 2)`` raw, unactivated per-axis sigma of this branch.
            assign: This branch's assignment (``fg_mask`` and ``gt_index``).
            context: The step's shared quantities, for the anchor grid and the targets.

        Returns:
            The scalar :class:`~lucid_yolo.losses.rle_loss.RLELoss` over every positive
            of the batch; a finite zero when there are none.
        """
        assert self.rle_loss is not None  # gated by the caller's task
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
            return cast("Tensor", self.rle_loss(empty, empty, empty, gt_vis.new_zeros((0, num_points))))

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
        # mapped by the *same* frame and the difference the loss reads stays meaningful.
        frame = context.gt_boxes[image_ids, rows].flatten(0, 1)[keep]
        predicted = decoded.gather(1, anchor_index).flatten(0, 1)[keep]
        annotated = gt_keypoints[image_ids, rows].flatten(0, 1)[keep]
        scored = self.rle_loss(
            normalize_keypoints_to_box(predicted, frame),
            raw_sigma.gather(1, anchor_index).flatten(0, 1)[keep],
            normalize_keypoints_to_box(annotated, frame),
            gt_vis[image_ids, rows].flatten(0, 1)[keep],
        )
        return cast("Tensor", scored)

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
        instances = max(*(int(image_masks.shape[0]) for image_masks in masks), 1)
        padded_masks = prototypes.new_zeros((len(masks), instances, *proto_grid))
        for index, image_masks in enumerate(masks):
            padded_masks[index, : image_masks.shape[0]] = image_masks
        o2m_mask = self._branch_mask_loss(prototypes, o2m_coeff, out.o2m_assign, padded_masks, grid_boxes)
        o2o_mask = self._branch_mask_loss(prototypes, o2o_coeff, out.o2o_assign, padded_masks, grid_boxes)
        mask_term = out.alpha * o2m_mask + (1.0 - out.alpha) * o2o_mask
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
