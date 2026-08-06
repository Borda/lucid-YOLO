# SPDX-License-Identifier: Apache-2.0
"""Instance mask loss over assembled prototype masks (WP-051).

A16 fixes the objective: per-pixel binary cross-entropy on **box-cropped** masks,
normalized by the ground-truth box area. Cropping is what makes the term local —
a prototype combination is global by construction (Eq. 7 mixes ``K`` full-image
maps), so without the crop every instance would be penalized for prototype
content living far outside its own box, and the loss would fight the linear
combination instead of shaping it. Box-area normalization is what makes a large
instance and a small one carry the same weight: the cropped sum grows with the
box area, so dividing it out leaves a per-pixel average.

The crop is a hard boolean window, not a soft weighting, and its membership test
reuses the repository's existing pixel-centre convention (A11: centres at
``(i + 0.5) * stride``, mirrored here at unit stride on the mask grid). A pixel
belongs to the box when its centre falls inside ``[x1, x2) x [y1, y2)``; the
half-open interval keeps two boxes that share an edge from both claiming the
boundary column.

Degenerate boxes are the one case the papers leave open (A36): a zero-width,
zero-height, or sub-pixel box has an area at or near zero and would otherwise
divide the loss by it. The area is therefore clamped to a minimum of ``1.0`` —
one pixel — which leaves every box of at least unit area untouched and turns the
degenerate case into a finite (and, since the crop selects no pixel centre,
exactly zero) contribution rather than a ``NaN``.

Like :class:`~lucid_yolo.losses.detection_loss.DetectionBranchLoss`, this term is
coordinate-scale agnostic in the sense that it makes no assumption about the
input image size — but it is **not** frame-agnostic: the boxes must already be
expressed in the mask grid's own coordinate frame, because the crop compares
them against integer pixel indices. Scaling boxes from input pixels to the
prototype grid is the caller's responsibility.

When there are no positive instances the loss is exactly zero, finite, and still
connected to ``mask_logits`` with zero gradient — the same no-positives handling
:class:`~lucid_yolo.losses.detection_loss.DetectionBranchLoss` gives its box
terms, rather than a ``0 / 0`` mean over an empty batch.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = ["instance_mask_loss"]

#: Minimum box area used as the per-instance normalizer (A36 degenerate-box clamp).
_MIN_BOX_AREA: float = 1.0


def instance_mask_loss(mask_logits: Tensor, target_masks: Tensor, boxes: Tensor) -> Tensor:
    """Box-cropped, box-area-normalized BCE over assembled instance masks (A16).

    Args:
        mask_logits: Raw assembled mask logits ``(N, H, W)`` for ``N`` positive
            instances, as produced by
            :func:`~lucid_yolo.models.heads.proto.assemble_masks`.
        target_masks: Binary ground-truth masks ``(N, H, W)`` as floats in
            ``{0.0, 1.0}``, on the same grid as ``mask_logits``.
        boxes: Ground-truth boxes ``(N, 4)`` in ``xyxy``, expressed in **the mask
            grid's own coordinate frame**. Rescaling boxes from input pixels to
            the prototype grid is the caller's job; this function only compares
            them against that grid's pixel centres.

    Returns:
        A scalar (zero-dimensional) tensor: the mean over instances of each
        instance's box-cropped BCE sum divided by its box area. ``N == 0`` gives
        a finite zero on ``mask_logits``' device and dtype.

    Examples:
        >>> import torch
        >>> logits = torch.zeros(1, 4, 4)
        >>> targets = torch.ones(1, 4, 4)
        >>> boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0]])  # whole grid: loss is ln 2
        >>> instance_mask_loss(logits, targets, boxes).round(decimals=4)
        tensor(0.6931)
    """
    per_pixel = F.binary_cross_entropy_with_logits(mask_logits, target_masks, reduction="none")  # (N, H, W)

    height, width = mask_logits.shape[-2], mask_logits.shape[-1]
    centres_x = torch.arange(width, device=mask_logits.device, dtype=mask_logits.dtype) + 0.5  # (W,)
    centres_y = torch.arange(height, device=mask_logits.device, dtype=mask_logits.dtype) + 0.5  # (H,)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)  # each (N,)
    inside_x = (centres_x.unsqueeze(0) >= x1.unsqueeze(-1)) & (centres_x.unsqueeze(0) < x2.unsqueeze(-1))  # (N, W)
    inside_y = (centres_y.unsqueeze(0) >= y1.unsqueeze(-1)) & (centres_y.unsqueeze(0) < y2.unsqueeze(-1))  # (N, H)
    crop = inside_y.unsqueeze(-1) & inside_x.unsqueeze(-2)  # (N, H, W)

    cropped = (per_pixel * crop.to(per_pixel.dtype)).sum(dim=(-2, -1))  # (N,)
    areas = ((x2 - x1) * (y2 - y1)).clamp(min=_MIN_BOX_AREA)  # (N,)
    per_instance = cropped / areas  # (N,)
    # Summing then dividing by max(N, 1) is the mean for N > 0 and, for N == 0,
    # a zero still attached to mask_logits — an empty .mean() would be NaN.
    return per_instance.sum() / max(per_instance.numel(), 1)
