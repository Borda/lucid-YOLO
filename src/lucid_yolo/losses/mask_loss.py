# SPDX-License-Identifier: Apache-2.0
"""Instance mask loss over assembled prototype masks (WP-051).

A16 fixes the objective: per-pixel binary cross-entropy on **box-cropped** masks,
normalized by the ground-truth box area. Cropping is what makes the term local —
a prototype combination is global by construction (Eq. 7 mixes ``K`` full-image
maps), so without the crop every instance would be penalized for prototype
content living far outside its own box, and the loss would fight the linear
combination instead of shaping it. Box-area normalization is what makes a large
instance and a small one carry the same weight: the cropped sum grows with the
box area, so dividing it out leaves a per-instance value of per-pixel scale.

It is a per-pixel *average* only in the limit. The numerator counts whole pixels —
the crop is a hard boolean window on the pixel lattice — while the denominator is
the box's continuous area, and the two agree only when the box edges fall on pixel
boundaries. A box spanning 2.5 pixels selects either 2 or 3 pixel centres depending
on where it sits, against a divisor of 2.5, so the quotient of an instance whose
per-pixel loss is uniform still swings by roughly ±20% at that size. The swing
shrinks as the box grows and is negligible at the box sizes the segmentation head
is trained on; it is A16's normalizer that is being followed here, and A16 names
the box area, not the selected-pixel count.

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

**Inverted** boxes are the same case wearing a disguise, and the clamp alone does
not catch them: a box inverted on *both* axes has two negative extents whose
product is positive, so ``(x2 - x1) * (y2 - y1)`` reports a healthy area for a
window that selects nothing. :func:`_box_areas` therefore clamps each extent at
zero before multiplying, which is what makes the normalizer a statement about the
region the crop actually selects. No loss value moves: the crop of an inverted box
is empty, so its numerator is exactly zero and the quotient was zero under either
divisor. The expression is corrected because a normalizer that disagrees with its
own crop is a trap for the next reader, not because a number is wrong today.

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
        instance's box-cropped BCE sum divided by its box area — a per-pixel
        average up to the lattice/continuum mismatch the module docstring
        quantifies. ``N == 0`` gives a finite zero on ``mask_logits``' device and
        dtype.

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
    per_instance = cropped / _box_areas(x1, y1, x2, y2)  # (N,)
    # Summing then dividing by max(N, 1) is the mean for N > 0 and, for N == 0,
    # a zero still attached to mask_logits — an empty .mean() would be NaN.
    return per_instance.sum() / max(per_instance.numel(), 1)


def _box_areas(x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor) -> Tensor:
    """Per-instance normalizer: box area, extents clamped at zero, area at one pixel.

    Each extent is clamped **before** the product, not after (L-07). Clamping only
    the product lets a box inverted on both axes multiply two negative extents into
    a positive area, so a crop that selects no pixel centre would still be
    normalized as though it had selected some — see the module docstring for why
    this moves no loss value and is corrected anyway.

    Args:
        x1: Left edges ``(N,)`` in the mask grid's frame.
        y1: Top edges ``(N,)``.
        x2: Right edges ``(N,)``.
        y2: Bottom edges ``(N,)``.

    Returns:
        ``(N,)`` areas, each at least :data:`_MIN_BOX_AREA`.

    Examples:
        >>> import torch
        >>> corners = (torch.tensor([0.0, 4.0]), torch.tensor([0.0, 4.0]))
        >>> far = (torch.tensor([3.0, 1.0]), torch.tensor([2.0, 1.0]))
        >>> _box_areas(*corners, *far)  # a 3x2 box, then one inverted on both axes
        tensor([6., 1.])
    """
    widths = (x2 - x1).clamp(min=0)
    heights = (y2 - y1).clamp(min=0)
    return (widths * heights).clamp(min=_MIN_BOX_AREA)
