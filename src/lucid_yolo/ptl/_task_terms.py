# SPDX-License-Identifier: Apache-2.0
"""Task loss-term helpers of :class:`~lucid_yolo.ptl.module.DetectionLitModule`.

The module-level pieces the segment, oriented and pose terms are built from: the per-step records those terms read
(:class:`_StepContext`, :class:`_KeypointPositives`), the keypoint objective table and OKS sigma choice, the A71 box
frame the RLE term scores in (:func:`normalize_keypoints_to_box`), and :func:`_blend`, the WP-035 branch blend every
extra term applies. The terms themselves stay methods of the module: network replay certifies them by method
identity and by the helper bindings the module itself looks up. Every name here stays importable from
:mod:`lucid_yolo.ptl.module`, where it was defined until the module was split by responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

from lucid_yolo.eval.coco_eval import COCO_KEYPOINT_OKS_SIGMAS, SYMBOL_KEYPOINT_OKS_SIGMA
from lucid_yolo.losses.keypoint_nll_loss import LaplaceNLLLoss
from lucid_yolo.losses.rle_loss import RLELoss

if TYPE_CHECKING:
    from lucid_yolo.data.targets import Targets
    from lucid_yolo.models.build import SegmentOutput
    from lucid_yolo.models.heads.detect import DualHeadOutput


def _keypoint_oks_sigmas(num_keypoints: int | None) -> tuple[float, ...]:
    """Pick the per-keypoint OKS sigma vector for a point count (A67, WP-137).

    :data:`~lucid_yolo.eval.coco_eval.COCO_KEYPOINT_OKS_SIGMAS` only when the count matches COCO's own 17-point human
    schema exactly, and :data:`~lucid_yolo.eval.coco_eval.SYMBOL_KEYPOINT_OKS_SIGMA` (A67's uniform sigma) repeated
    for every other count — the choice ``scripts/overfit_micro.py``'s synthetic gate makes for its 7-point symbol
    schema, shared from one place. One function serves both consumers on purpose: the ``val/oks_mAP`` evaluator and
    the :class:`~lucid_yolo.losses.oks_loss.OKSLoss` training term (WP-178) must score against the **same**
    tolerances, or the loss would pull towards a target the metric does not measure.

    Args:
        num_keypoints: The schema's point count, or ``None`` for a module with no
            point stem (an empty vector).

    Returns:
        A length-``K`` tuple of sigmas.

    Examples:
        ```pycon
        >>> _keypoint_oks_sigmas(3)
        (0.072, 0.072, 0.072)
        >>> len(_keypoint_oks_sigmas(17)), _keypoint_oks_sigmas(17)[0]
        (17, 0.026)
        >>> _keypoint_oks_sigmas(None)
        ()

        ```
    """
    if num_keypoints == len(COCO_KEYPOINT_OKS_SIGMAS):
        return COCO_KEYPOINT_OKS_SIGMAS
    return (SYMBOL_KEYPOINT_OKS_SIGMA,) * (num_keypoints or 0)


#: Keypoint objectives a ``"keypoints"`` run may select between, each mapped to the class that implements it (WP-135).
#: ``"rle"`` is R14's full residual log-likelihood and the default every existing config, checkpoint and accepted
#: result was produced with; ``"laplace_nll"`` is R14 Table 7's own flow-free control, present so WP-125 can state
#: a direction of effect rather than a bare number. Keyed by plain strings for the reason ``_TASKS`` and
#: ``ROTATED_IOU_FORMS`` are: the value arrives from YAML. One mapping rather than a name tuple beside a constructor
#: branch, so the accepted set and the thing each name builds cannot drift apart.
_KEYPOINT_LOSSES: dict[str, type[RLELoss] | type[LaplaceNLLLoss]] = {
    "rle": RLELoss,
    "laplace_nll": LaplaceNLLLoss,
}

#: Floor, in input pixels, on the per-axis box extent :func:`normalize_keypoints_to_box`
#: divides by. One pixel: a box thinner than that carries no pose worth a frame, and the
#: unclamped division would answer a finite annotation with an infinity.
_MIN_BOX_EXTENT = 1.0


def normalize_keypoints_to_box(points: Tensor, boxes: Tensor) -> Tensor:
    """Map absolute point pixels into their instance's box frame (A71, WP-132).

    The frame :class:`~lucid_yolo.losses.rle_loss.RLELoss` needs and this project's decode does not produce. R14
    predicts ``sigma_hat`` through a sigmoid, so ``sigma_hat`` lies in ``(0, 1)`` by construction (R14 sec. 3.3), and
    the residual it scales is ``x_bar = (mu_g - mu_hat) / sigma_hat``. A scale bounded above by 1 is only expressive
    in a frame whose errors are ``O(1)``: fed absolute pixels, the *smallest* residual the model can express for a 40
    px error is 40, and an early prediction on the far side of a 256 px canvas reaches into the hundreds. R14 never
    states the frame — it works top-down on a person crop resized to a fixed input, so the crop *is* the normalization
    and the paper has no occasion to name it. This function supplies the dense-detector equivalent: the assigned
    ground-truth box plays the part R14's crop plays.

    Measured, not reasoned: run un-normalized, the keypoint term enters at 1300 of a 1342 total loss and the flow's
    coupling layers overflow to non-finite on the second step. See A71 and the ``#wp-132`` log entry.

    Normalization is per axis — ``x`` by the box width, ``y`` by its height — which is exactly what resizing a crop to
    a fixed input does, rather than the single ``sqrt(area)`` scalar OKS uses. The two agree on a square object and
    differ only in how an elongated one distributes its tolerance; per-axis is the one that makes the frame
    an actual crop.

    The translation term cancels: the loss reads ``mu_hat`` and ``mu_gt`` only through their difference, and both are
    mapped by the same box. It is applied anyway, so the returned values mean what the name says — ``0`` at the box's
    top-left corner, ``1`` at its bottom-right, a point outside the box outside ``[0, 1]`` — rather than being a bare
    division that happens to be sufficient.

    Args:
        points: ``(..., K, 2)`` absolute point coordinates in input pixels, as
            :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints` returns them.
        boxes: ``(..., 4)`` ``xyxy`` boxes in input pixels, aligned with ``points`` on
            every leading axis — one box per point set.

    Returns:
        The points in their box's frame, shaped as given. A box narrower than one pixel on an axis is treated as one
        pixel wide there: a degenerate box carries no pose to normalize by, and the alternative is a division that
        returns infinity for a finite annotation.

    Examples:
        ```pycon
        >>> import torch
        >>> points = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
        >>> boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        >>> normalize_keypoints_to_box(points, boxes)  # corner to corner
        tensor([[[0., 0.],
                 [1., 1.]]])

        ```
    """
    origin = boxes[..., :2].unsqueeze(-2)
    extent = (boxes[..., 2:] - boxes[..., :2]).clamp(min=_MIN_BOX_EXTENT).unsqueeze(-2)
    return (points - origin) / extent


@dataclass(frozen=True)
class _StepContext:
    """Everything one step computed once and its task-conditional terms may read.

    The three tasks need overlapping but different slices of a step: the mask terms want the prototypes and the
    ground-truth boxes in pixels, the oriented terms want the raw angles and the anchor grid, and neither wants the
    other's. Passing them as positional arguments put :meth:`DetectionLitModule._task_extra_loss` at seven required
    parameters before the oriented path added any, and the eighth would have been a lint failure standing in for a
    design one — a dispatch point whose signature grows with every task is a dispatch point that will eventually be
    typed wrongly.

    Attributes:
        head_out: The dense dual-head output of this step's forward.
        seg_out: The segmentation forward output, or ``None`` for a task with no mask branches.
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
        masks: The loader's per-image instance masks, or ``None`` to rasterise them in the step.
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


@dataclass(frozen=True)
class _KeypointPositives:
    """One branch's assigned positives, gathered once for every pose term to score (WP-178).

    Attributes:
        predicted: ``(P, K, 2)`` decoded points of the positives, in input pixels.
        raw_sigma: ``(P, K, 2)`` raw, unactivated per-axis sigma of the positives.
        annotated: ``(P, K, 2)`` ground-truth points of each positive's assigned
            instance, in input pixels.
        frame: ``(P, 4)`` ``xyxy`` box of each positive's assigned instance, in
            input pixels — the A71 frame for the RLE term, the A75 area for OKS.
        visibility: ``(P, K)`` int64 visibility of the assigned instance's points.
    """

    predicted: Tensor
    raw_sigma: Tensor
    annotated: Tensor
    frame: Tensor
    visibility: Tensor


def _blend(alpha: float | Tensor, o2m: Tensor, o2o: Tensor) -> Tensor:
    """Blend a term's one-to-many and one-to-one values by the WP-035 branch weight.

    ``alpha * o2m + (1.0 - alpha) * o2o``, the split :class:`~lucid_yolo.losses.dual_loss.DualBranchLoss` applies to
    the box terms and every task-conditional term repeats, written once. The expression and its operand order are the
    ones each call site carried, so the result is the same value bit for bit; reordering it would not be, because
    floating-point addition and multiplication do not reassociate. A tensor weight is cast separately for each
    branch, after the complement is formed in double precision, to preserve the float weight's rounding and avoid
    promoting scalar task losses to float64.

    Args:
        alpha: The one-to-many branch weight, as the dual loss output carries it.
        o2m: The term scored on the one-to-many branch.
        o2o: The same term scored on the one-to-one branch.

    Returns:
        The blended term.

    Examples:
        ```pycon
        >>> import torch
        >>> _blend(0.25, torch.tensor(4.0), torch.tensor(8.0))
        tensor(7.)

        ```
    """
    if isinstance(alpha, Tensor):
        return alpha.to(o2m.dtype) * o2m + (1.0 - alpha).to(o2o.dtype) * o2o
    return alpha * o2m + (1.0 - alpha) * o2o
