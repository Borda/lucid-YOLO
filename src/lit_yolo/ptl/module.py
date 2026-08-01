# SPDX-License-Identifier: Apache-2.0
"""Detection :class:`~lightning.pytorch.LightningModule` with task-conditional losses (WP-034).

:class:`DetectionLitModule` composes the WP-020…022 model stack (backbone, neck,
dual detection head) with the WP-028 :class:`~lit_yolo.losses.dual_loss.DualBranchLoss`
and the WP-032 :class:`~lit_yolo.optim.musgd.MuSGD` optimizer into the Phase 5
training loop. It runs under Lightning **automatic optimization** (blueprint D4):
``training_step`` returns the scalar total and Lightning owns the
backward/step/zero-grad cycle.

Forward and loss wiring:
    The head emits raw ``ltrb`` distances per anchor; the module derives the
    anchor grid for the batch's feature sizes (image size divided by the level
    strides ``(8, 16, 32)``, cached per size), decodes both branches' boxes with
    :func:`~lit_yolo.models.heads.detect.decode_ltrb`, pads the ragged
    ``list[Targets]`` into dense ``(B, N_max, ...)`` ground-truth tensors with
    :func:`pad_targets`, and scores the pair with :class:`DualBranchLoss`. Every
    per-component term (``o2m``/``o2o`` box/cls/l1 and the combined total) is
    logged.

Task conditioning:
    ``task`` selects the active supervision. ``"detect"`` is fully wired here;
    ``"segment"`` and ``"obb"`` are accepted so their configs parse, but their
    extra-loss contribution flows through :meth:`DetectionLitModule._task_extra_loss`,
    an **inert stub** returning a zero scalar. That method is the seam the Phase 7
    (segmentation) and Phase 8 (oriented-box) work packages fill; until then all
    three tasks train identically on the detection objective.

Learning-rate schedule (A8 deferral):
    :meth:`DetectionLitModule.configure_optimizers` returns :class:`MuSGD` with a
    **constant** learning rate. The linear-decay schedule shape (A8) and the
    ~1-epoch warmup land with the WP-038/WP-039 experiment configs; the
    constructor carries ``lr``/``momentum``/``weight_decay`` as the base
    hyperparameters (blueprint sec. 5.7-5.8: ``lr0 = 0.01``, ``momentum ~ 0.95``,
    ``weight_decay = 5e-4``) but the schedule itself is out of scope for this WP.

Progressive-loss schedule (WP-035):
    :attr:`DetectionLitModule.alpha` delegates to the underlying
    :class:`DualBranchLoss` branch weight. The constructor ``alpha`` seeds it, and
    :meth:`DetectionLitModule.on_train_epoch_start` overwrites it once per epoch
    from a :class:`~lit_yolo.losses.progressive.ProgressiveLossSchedule` — the
    linear ramp ``(0.8, 0.2) -> (0.1, 0.9)`` of R1 Eq. 2-3 (endpoints
    ``alpha_init``/``alpha_final``). When ``trainer.max_epochs`` is unset
    (``None`` or ``<= 0``, e.g. a step-bounded or ``fast_dev_run`` run) the ramp
    denominator is undefined, so the hook leaves ``alpha`` at its seeded value.

Provenance: R1 sec. 3.2, R1 Eq. 2-3, R1 Tables S2/S5. Assumptions: A8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from lightning.pytorch import LightningModule
from torch import Tensor

from lit_yolo.assign import make_anchor_points
from lit_yolo.losses.dual_loss import DualBranchLoss, DualLossOutput
from lit_yolo.losses.progressive import ProgressiveLossSchedule
from lit_yolo.models.backbone import DetectionBackbone
from lit_yolo.models.heads.detect import DualDetectionHead, DualHeadOutput, decode_ltrb
from lit_yolo.models.neck import DetectionNeck
from lit_yolo.optim.musgd import MuSGD

if TYPE_CHECKING:
    from lit_yolo.data.targets import Targets

__all__ = ["DetectionLitModule", "pad_targets"]

#: Feature-level input-pixel strides of the P3/P4/P5 detection head (8, 16, 32).
_STRIDES: tuple[int, int, int] = (8, 16, 32)

#: Task names the module accepts; only ``"detect"`` carries active extra losses.
_TASKS: tuple[str, ...] = ("detect", "segment", "obb")

#: Column count of an ``xyxy`` axis-aligned box.
_BOX_DIM = 4


def pad_targets(targets: list[Targets]) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a ragged batch of per-image targets into dense ground-truth tensors.

    The batch's :class:`~lit_yolo.data.targets.Targets` each carry their own
    instance count, so they are padded to the batch maximum ``N_max`` and paired
    with a boolean mask marking the real (non-padding) rows — the dense form the
    :class:`~lit_yolo.losses.dual_loss.DualBranchLoss` assigners consume. An image
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
        >>> from lit_yolo.data.targets import Targets
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


class DetectionLitModule(LightningModule):
    """Lightning detector: model stack, dual-branch loss, and MuSGD (WP-034).

    Composes :class:`~lit_yolo.models.backbone.DetectionBackbone`,
    :class:`~lit_yolo.models.neck.DetectionNeck`, and
    :class:`~lit_yolo.models.heads.detect.DualDetectionHead` (built from the same
    compound-scaling multipliers), scores their dense predictions with
    :class:`~lit_yolo.losses.dual_loss.DualBranchLoss`, and optimizes under
    Lightning automatic optimization (D4) with
    :class:`~lit_yolo.optim.musgd.MuSGD`. See the module docstring for the A8
    constant-LR deferral, the ``task`` conditioning, and the :attr:`alpha` seam.

    Args:
        depth: Depth multiplier scaling per-stage repeat counts of the backbone
            and neck.
        width: Width multiplier scaling channel counts.
        max_channels: Channel cap applied before the width multiply.
        num_classes: Number of object classes the head predicts.
        task: Supervision task; one of ``"detect"`` (active), ``"segment"`` or
            ``"obb"`` (accepted, inert extra-loss stub). Defaults to ``"detect"``.
        lr: Base learning rate for MuSGD (constant here; A8 schedule is WP-038/039).
            Defaults to ``0.01``.
        momentum: MuSGD momentum coefficient. Defaults to ``0.95``.
        weight_decay: Decoupled weight decay (matrix parameters only). Defaults to
            ``5e-4``.
        w_muon: Additive gain on the MuSGD Muon branch. Defaults to ``0.5``.
        w_sgd: Additive gain on the MuSGD SGD branch. Defaults to ``0.5``.
        box_gain: CIoU-term gain shared by both loss branches. Defaults to ``7.5``.
        cls_gain: Classification-term gain shared by both branches. Defaults to ``0.5``.
        l1_gain: L1-box-term gain shared by both branches. Defaults to ``6.0``.
        alpha: Initial one-to-many branch weight seeding the dual loss (used
            before training and whenever the epoch schedule is inactive).
            Defaults to ``0.5``.
        alpha_init: One-to-many branch weight the progressive schedule sets on the
            first epoch (branch weights ``(0.8, 0.2)``). Defaults to ``0.8``.
        alpha_final: One-to-many branch weight the schedule ramps to on the last
            epoch (branch weights ``(0.1, 0.9)``). Defaults to ``0.1``.

    Raises:
        ValueError: If ``task`` is not one of ``"detect"``, ``"segment"``, ``"obb"``.

    Examples:
        >>> import torch
        >>> from lit_yolo.data.targets import Targets
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
        *,
        lr: float = 0.01,
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
    ) -> None:
        super().__init__()
        if task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}, got {task!r}")
        self.save_hyperparameters()
        self._task = task
        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._w_muon = w_muon
        self._w_sgd = w_sgd

        self.backbone = DetectionBackbone(depth=depth, width=width, max_channels=max_channels)
        self.neck = DetectionNeck(self.backbone.channels, depth=depth, width=width, max_channels=max_channels)
        self.head = DualDetectionHead(self.neck.channels, num_classes=num_classes)
        self.loss = DualBranchLoss(box_gain=box_gain, cls_gain=cls_gain, l1_gain=l1_gain, alpha=alpha)
        self._loss_schedule = ProgressiveLossSchedule(alpha_init=alpha_init, alpha_final=alpha_final)

        #: Per-image-size cache of ``(anchor_points, stride_per_anchor)`` on CPU.
        self._anchor_cache: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}

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
        :meth:`~lit_yolo.losses.progressive.ProgressiveLossSchedule.alpha_at`
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
            The :class:`~lit_yolo.models.heads.detect.DualHeadOutput` dense
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

    def training_step(self, batch: tuple[Tensor, list[Targets]], batch_idx: int) -> Tensor:
        """Run one training step under automatic optimization (D4).

        Args:
            batch: The datamodule batch ``(images, list[Targets])``.
            batch_idx: Index of the batch within the epoch (unused).

        Returns:
            The scalar total loss for Lightning to backpropagate.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, list[Targets]], batch_idx: int) -> Tensor:
        """Run one validation step: identical forward and loss, logged under ``val/``.

        Evaluation decode and mAP scoring land in WP-042/WP-043; this step reports
        the training loss on the validation split only.

        Args:
            batch: The datamodule batch ``(images, list[Targets])``.
            batch_idx: Index of the batch within the epoch (unused).

        Returns:
            The scalar total validation loss.
        """
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> MuSGD:
        """Return the MuSGD optimizer over every parameter with a constant LR.

        The learning-rate schedule shape (A8) and warmup are deferred to the
        WP-038/WP-039 configs; this returns a bare optimizer so the LR stays
        constant across the run.

        Returns:
            A :class:`~lit_yolo.optim.musgd.MuSGD` over ``self.parameters()`` built
            from the constructor hyperparameters.
        """
        return MuSGD(
            self.parameters(),
            lr=self._lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
            w_muon=self._w_muon,
            w_sgd=self._w_sgd,
        )

    def _shared_step(self, batch: tuple[Tensor, list[Targets]], stage: str) -> Tensor:
        """Forward, decode both branches, score the dual loss, and log every term."""
        images, targets = batch
        head_out = self(images)
        anchor_points, strides = self._anchor_grid(images.shape[-2], images.shape[-1], images.device)
        o2m_boxes = decode_ltrb(head_out.o2m_box, anchor_points, strides)
        o2o_boxes = decode_ltrb(head_out.o2o_box, anchor_points, strides)
        gt_boxes, gt_labels, gt_mask = pad_targets(targets)
        gt_boxes = gt_boxes.to(images.device)
        gt_labels = gt_labels.to(images.device)
        gt_mask = gt_mask.to(images.device)
        out = self.loss(
            head_out.o2m_cls, o2m_boxes, head_out.o2o_cls, o2o_boxes, anchor_points, gt_boxes, gt_labels, gt_mask
        )
        total = out.total + self._task_extra_loss(head_out, targets)
        self._log_loss(out, total, stage, images.shape[0])
        return total

    def _task_extra_loss(self, head_out: DualHeadOutput, targets: list[Targets]) -> Tensor:
        """Return the task-conditional extra loss; an inert zero for detection.

        This is the seam Phase 7 (segmentation mask loss) and Phase 8 (oriented-box
        angle loss) fill. For ``"detect"`` — and, until those phases land, for
        ``"segment"``/``"obb"`` too — it contributes a zero scalar on the
        prediction device so the total is exactly the dual detection loss.

        Args:
            head_out: The dense head predictions (used only for device/dtype here).
            targets: The batch targets (unused by the detection stub).

        Returns:
            A zero scalar tensor matching the head output's device and dtype.
        """
        del targets  # inert stub: no extra supervision for the detection task
        return torch.zeros((), device=head_out.o2m_cls.device, dtype=head_out.o2m_cls.dtype)

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
