# SPDX-License-Identifier: Apache-2.0
"""The private training and validation routes for each supervision task.

One strategy per :class:`~lucid_yolo.tasks.Task` owns target padding, extra-loss
composition and logging, replay selection and validation behavior. The exhaustive
:data:`_STRATEGIES` table lets :class:`~lucid_yolo.ptl.module.DetectionLitModule`
resolve its strategy once, after validating the task, rather than select behavior
from the task string at every step.

Strategies hold no tensors or per-fit state. Registered losses, replay holders and
validation accumulators stay on the Lightning module, so checkpoint keys and
module-method override checks keep their existing owners. Copying or unpickling a
strategy retains the process's registered instance: the oriented replay certificate
checks that identity as well as the extra-loss implementation.

``module.py`` is imported only for typing because it imports this table at runtime;
importing it here at runtime would create a cycle. The strategies call the existing
module methods for task loss terms and validation accumulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from lucid_yolo.eval.coco_eval import COCO_RECALL_GRID
from lucid_yolo.ptl._padding import pad_keypoints, pad_rboxes
from lucid_yolo.ptl.coco_backend import build_mean_average_precision
from lucid_yolo.tasks import Task

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torchmetrics.detection import MeanAveragePrecision

    from lucid_yolo.data.targets import Targets
    from lucid_yolo.losses.dual_loss import DualLossOutput
    from lucid_yolo.models.build import SegmentOutput
    from lucid_yolo.models.heads.detect import DualHeadOutput
    from lucid_yolo.ptl._task_terms import _StepContext
    from lucid_yolo.ptl.module import DetectionLitModule


class _TaskStrategy:
    """Provide shared box behavior and inactive routes for optional supervision."""

    __slots__ = ()
    _task: Task
    segmentation = False
    compile_objective = False
    pose_loss_capture = False

    def __copy__(self) -> _TaskStrategy:
        """Keep the registered stateless strategy when its module is copied."""
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _TaskStrategy:
        """Share the strategy while the module's tensors and fit state are copied."""
        return self

    def __reduce__(self) -> tuple[Callable[[Task], _TaskStrategy], tuple[Task]]:
        """Resolve unpickling through the receiving process's strategy table."""
        return _strategy_for_task, (self._task,)

    def pad_extras(
        self, targets: list[Targets], device: torch.device
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Return rotated boxes, keypoints and visibility absent from box supervision."""
        return None, None, None

    def segmentation_output(self, module: DetectionLitModule, images: Tensor) -> SegmentOutput | None:
        """Leave the ordinary training forward responsible for the network."""
        return None

    def uses_objective(self, module: DetectionLitModule, stage: str) -> bool:
        """Keep task-specific inputs outside the detection-only objective region."""
        return False

    def check_replay_head(self, module: DetectionLitModule) -> None:
        """Leave head certification to the complete-network recording route."""

    def training_network(self, module: DetectionLitModule, images: Tensor) -> DualHeadOutput:
        """Replay the complete detection network through the existing holder."""
        eager = cast("Callable[[Tensor], DualHeadOutput]", module)
        return module._network_capture.run(images, (module.backbone, module.neck, module.head), eager)

    def extra_loss(self, module: DetectionLitModule, context: _StepContext, out: DualLossOutput, stage: str) -> Tensor:
        """Contribute a scalar zero without changing the detection total's dtype."""
        return torch.zeros_like(out.total)

    def build_box_metric(self) -> MeanAveragePrecision:
        """Build the existing box accumulator with its pinned recall grid."""
        return build_mean_average_precision(
            backend="faster_coco_eval", box_format="xyxy", rec_thresholds=list(COCO_RECALL_GRID)
        )

    def build_mask_metric(self) -> MeanAveragePrecision | None:
        """Avoid constructing a mask accumulator without mask supervision."""
        return None

    def update_box_validation(self, module: DetectionLitModule, detections: Tensor, targets: list[Targets]) -> None:
        """Accumulate the batch's box predictions and ground truth."""
        module._update_val_map(detections, targets)

    def update_extra_validation(
        self,
        module: DetectionLitModule,
        head_out: DualHeadOutput,
        targets: list[Targets],
        anchor_points: Tensor,
        strides: Tensor,
        detections: Tensor,
        anchor_indices: Tensor,
    ) -> None:
        """Leave specialized accumulators untouched for ordinary box supervision."""

    def log_box_validation(self, module: DetectionLitModule) -> None:
        """Log and reset the epoch's box metric with its original dtype and key."""
        computed = module._val_map.compute()
        module.log("val/mAP", computed["map"].to(torch.float32), prog_bar=True)
        module._val_map.reset()


class _DetectionStrategy(_TaskStrategy):
    """Use the detection objective region for every detection step."""

    __slots__ = ()
    _task = Task.DETECT
    compile_objective = True

    def uses_objective(self, module: DetectionLitModule, stage: str) -> bool:
        """Enter the shared eager or compiled detection objective."""
        return True


class _SegmentationStrategy(_TaskStrategy):
    """Add prototype-mask and semantic supervision to the shared box objective."""

    __slots__ = ()
    _task = Task.SEGMENT
    segmentation = True

    def segmentation_output(self, module: DetectionLitModule, images: Tensor) -> SegmentOutput:
        """Run the segmentation forward once before scoring its detection output."""
        return module.forward_segmentation(images)

    def extra_loss(self, module: DetectionLitModule, context: _StepContext, out: DualLossOutput, stage: str) -> Tensor:
        """Weight and log the existing mask and semantic terms in their original order."""
        if context.seg_out is None:
            return torch.zeros_like(out.total)
        mask_term, semantic_term = module._segment_terms(context, out)
        module.log(f"{stage}/mask", mask_term, batch_size=len(context.targets))
        module.log(f"{stage}/semantic", semantic_term, batch_size=len(context.targets))
        return module._mask_gain * mask_term + module._semantic_gain * semantic_term

    def build_mask_metric(self) -> MeanAveragePrecision:
        """Build a separate mask accumulator on the prototype-grid frame."""
        return build_mean_average_precision(
            backend="faster_coco_eval", iou_type="segm", rec_thresholds=list(COCO_RECALL_GRID)
        )


class _OrientedStrategy(_TaskStrategy):
    """Replace the box terms through the existing rotated loss and validation routes."""

    __slots__ = ()
    _task = Task.OBB

    def pad_extras(
        self, targets: list[Targets], device: torch.device
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Pad rotated boxes and transfer them before the task's network forward."""
        return pad_rboxes(targets).to(device), None, None

    def training_network(self, module: DetectionLitModule, images: Tensor) -> DualHeadOutput:
        """Keep the oriented route's native-consumer certificate and fallback."""
        return module._training_oriented_network(images)

    def extra_loss(self, module: DetectionLitModule, context: _StepContext, out: DualLossOutput, stage: str) -> Tensor:
        """Retain the module method whose identity the replay certificate checks."""
        return module._oriented_extra_loss(context, out, stage)

    def update_box_validation(self, module: DetectionLitModule, detections: Tensor, targets: list[Targets]) -> None:
        """Omit the axis-aligned metric, and its host copy of the detections, for oriented supervision."""

    def update_extra_validation(
        self,
        module: DetectionLitModule,
        head_out: DualHeadOutput,
        targets: list[Targets],
        anchor_points: Tensor,
        strides: Tensor,
        detections: Tensor,
        anchor_indices: Tensor,
    ) -> None:
        """Accumulate rotated predictions through the deployed oriented decoder."""
        module._update_val_rotated(head_out, targets, anchor_points, strides)

    def log_box_validation(self, module: DetectionLitModule) -> None:
        """Leave the unused axis-aligned box accumulator untouched."""


class _KeypointStrategy(_TaskStrategy):
    """Add pose supervision and route feature/stem replay beside box supervision."""

    __slots__ = ()
    _task = Task.KEYPOINTS
    pose_loss_capture = True

    def pad_extras(
        self, targets: list[Targets], device: torch.device
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Pad point coordinates and visibility, retaining transfer order and dtype."""
        gt_keypoints, gt_keypoint_vis = pad_keypoints(targets)
        gt_keypoints = gt_keypoints.to(device)
        gt_keypoint_vis = gt_keypoint_vis.to(device)
        return None, gt_keypoints, gt_keypoint_vis

    def uses_objective(self, module: DetectionLitModule, stage: str) -> bool:
        """Enter the detection region only when the pose loss recorder qualifies."""
        return module._pose_records_loss(stage)

    def check_replay_head(self, module: DetectionLitModule) -> None:
        """Certify pose stem ownership before common root-hook checks."""
        module._check_pose_replay_head()

    def training_network(self, module: DetectionLitModule, images: Tensor) -> DualHeadOutput:
        """Replay shared features and native pose stems, preserving eager fallback."""
        features = module._network_capture.run_features(images, (module.backbone, module.neck))
        return (
            module(images)
            if features is None
            else cast("DualHeadOutput", module.head(features, _runner=module._network_capture))
        )

    def extra_loss(self, module: DetectionLitModule, context: _StepContext, out: DualLossOutput, stage: str) -> Tensor:
        """Weight and log the point and OKS terms without regrouping their sum."""
        keypoint_term, oks_term = module._keypoint_terms(context, out)
        module.log(f"{stage}/keypoint", keypoint_term, batch_size=len(context.targets))
        module.log(f"{stage}/oks", oks_term, batch_size=len(context.targets))
        return module._keypoint_gain * keypoint_term + module._oks_gain * oks_term

    def update_extra_validation(
        self,
        module: DetectionLitModule,
        head_out: DualHeadOutput,
        targets: list[Targets],
        anchor_points: Tensor,
        strides: Tensor,
        detections: Tensor,
        anchor_indices: Tensor,
    ) -> None:
        """Accumulate point predictions beside the task's ordinary box metric."""
        module._update_val_keypoints(head_out, detections, anchor_indices, targets, anchor_points, strides)


_STRATEGIES: dict[Task, _TaskStrategy] = {
    Task.DETECT: _DetectionStrategy(),
    Task.SEGMENT: _SegmentationStrategy(),
    Task.OBB: _OrientedStrategy(),
    Task.KEYPOINTS: _KeypointStrategy(),
}


def _strategy_for_task(task: Task) -> _TaskStrategy:
    """Return the registered strategy when restoring a pickled task strategy.

    Examples:
        ```pycon
        >>> import copy
        >>> import pickle
        >>> strategy = _strategy_for_task(Task.OBB)
        >>> copy.copy(strategy) is strategy, copy.deepcopy(strategy) is strategy
        (True, True)
        >>> pickle.loads(pickle.dumps(strategy)) is strategy
        True

        ```
    """
    return _STRATEGIES[task]
