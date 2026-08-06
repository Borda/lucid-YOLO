# SPDX-License-Identifier: Apache-2.0
"""Instance-mask decode: assemble, crop, binarize, un-letterbox (WP-053a).

The inference-side counterpart of the WP-051 mask loss. Eq. 7 gives per-instance
mask logits as a linear combination of the ``K`` prototype maps
(:func:`~lucid_yolo.models.heads.proto.assemble_masks`); turning that into a
binary mask an evaluator can score takes four further steps, and their order is
the whole content of this module:

1. **sigmoid at the prototype grid** — the probabilities are defined there, and
   it is the cheapest place to compute them (the prototype grid is a quarter of
   the input's area per A15).
2. **bilinear upsample of the probabilities** to the letterboxed input
   resolution. Interpolating probabilities and thresholding afterwards places
   each boundary where the probability field actually crosses ``0.5``, at
   sub-prototype-cell precision; thresholding first and upsampling a binary mask
   would snap every boundary onto the prototype grid, and no amount of later
   filtering recovers the lost position.
3. **crop to the predicted box**, using the *same* half-open pixel-centre
   membership rule (A11) that
   :func:`~lucid_yolo.losses.mask_loss.instance_mask_loss` applies to the
   ground-truth box at train time. The boxes differ — predicted here, ground
   truth there — but the membership rule must not, or the model is evaluated
   under a geometry it was never trained under.
4. **threshold to bool** at :data:`_MASK_THRESHOLD`.

:func:`masks_to_original` then lands the binary masks in original-image
coordinates (A10), the frame evaluation scores in. It reads its scale and
padding from :meth:`~lucid_yolo.data.letterbox.Letterbox.forward_affine` — the
exact affine whose inverse :func:`~lucid_yolo.decode.common.to_letterboxed_original`
applies to the boxes — so masks and boxes cannot disagree about the transform.
That disagreement is the failure this module is most exposed to: it leaves every
bbox score right and every segm score quietly wrong.

Provenance: R1 Eq. 7, R16. Assumptions: A10, A11, A15, A16, A37.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.models.heads.proto import assemble_masks

__all__ = ["decode_instance_masks", "masks_to_original"]

#: Probability above which a decoded mask pixel is foreground (A37). Unspecified
#: by the papers; ``0.5`` is the neutral cut of a sigmoid trained with BCE.
_MASK_THRESHOLD: float = 0.5


def decode_instance_masks(
    prototypes: Tensor,
    coefficients: Tensor,
    boxes: Tensor,
    image_size: tuple[int, int],
    threshold: float = _MASK_THRESHOLD,
) -> Tensor:
    """Assemble, upsample, box-crop and binarize per-instance masks (A16, A37).

    Runs the four decode steps of the module docstring in order. Every step is
    batched — there is no Python loop over images or instances — and every
    coordinate tensor is built from ``prototypes``, so the function runs wherever
    its inputs live.

    ``coefficients`` must **already** be gathered for the kept detections, by the
    very indices ``boxes`` was gathered by
    (:func:`~lucid_yolo.models.heads.detect.o2o_topk_with_indices`); this function
    pairs row ``n`` of one with row ``n`` of the other and cannot detect a
    mismatch.

    Args:
        prototypes: Raw prototype maps ``(B, K, Hp, Wp)`` from
            :class:`~lucid_yolo.models.heads.proto.ProtoNet`.
        coefficients: Tanh mask coefficients ``(B, N, K)`` for the ``N`` kept
            detections of each image.
        boxes: Predicted ``xyxy`` boxes ``(B, N, 4)`` in **letterboxed input
            pixels** — the frame the detection head decodes into, and the frame
            ``image_size`` describes.
        image_size: The letterboxed canvas ``(height, width)`` to decode onto.
        threshold: Probability above which a pixel is foreground. Defaults to
            :data:`_MASK_THRESHOLD` (``0.5``).

    Returns:
        Boolean instance masks of shape ``(B, N, height, width)``, zero outside
        each detection's own box.

    Examples:
        >>> import torch
        >>> prototypes = torch.full((1, 1, 2, 2), 10.0)  # one saturated prototype
        >>> coefficients = torch.ones(1, 1, 1)  # (B=1, N=1, K=1)
        >>> boxes = torch.tensor([[[0.0, 0.0, 2.0, 4.0]]])  # left half of the canvas
        >>> masks = decode_instance_masks(prototypes, coefficients, boxes, image_size=(4, 4))
        >>> masks.shape, masks.dtype
        (torch.Size([1, 1, 4, 4]), torch.bool)
        >>> masks[0, 0, 0]  # only the boxed columns survive the crop
        tensor([ True,  True, False, False])
    """
    probabilities = assemble_masks(prototypes, coefficients).sigmoid()  # (B, N, Hp, Wp)
    upsampled = F.interpolate(probabilities, size=image_size, mode="bilinear", align_corners=False)
    crop = _box_crop(boxes, image_size, prototypes)  # (B, N, H, W)
    return (upsampled * crop.to(upsampled.dtype)) > threshold


def _box_crop(boxes: Tensor, image_size: tuple[int, int], reference: Tensor) -> Tensor:
    """Return the ``(B, N, H, W)`` half-open pixel-centre box window (A11).

    The membership test is the one
    :func:`~lucid_yolo.losses.mask_loss.instance_mask_loss` uses: a pixel belongs
    to a box when its centre ``(i + 0.5, j + 0.5)`` falls inside
    ``[x1, x2) x [y1, y2)``. Coordinate tensors are built on ``reference``'s
    device and dtype.
    """
    height, width = image_size
    centres_x = torch.arange(width, device=reference.device, dtype=reference.dtype) + 0.5  # (W,)
    centres_y = torch.arange(height, device=reference.device, dtype=reference.dtype) + 0.5  # (H,)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)  # each (B, N)
    inside_x = (centres_x >= x1.unsqueeze(-1)) & (centres_x < x2.unsqueeze(-1))  # (B, N, W)
    inside_y = (centres_y >= y1.unsqueeze(-1)) & (centres_y < y2.unsqueeze(-1))  # (B, N, H)
    return inside_y.unsqueeze(-1) & inside_x.unsqueeze(-2)


def masks_to_original(masks: Tensor, letterbox: Letterbox, orig_size: tuple[int, int]) -> Tensor:
    """Un-letterbox binary instance masks back to original-image coordinates (A10).

    The mask-side twin of
    :func:`~lucid_yolo.decode.common.to_letterboxed_original`: it removes the
    letterbox padding and undoes the resize for one image's masks, landing them
    in the frame the ground truth (and therefore the segm metric) lives in.

    The geometry comes from
    :meth:`~lucid_yolo.data.letterbox.Letterbox.forward_affine` evaluated for the
    mask's own canvas — the same ``r``, ``pad_left`` and ``pad_top`` the box path
    inverts, taken from the same resolver rather than recomputed here, so the two
    paths cannot drift. Only ``letterbox.allow_upscale`` is read off the passed
    transform; the canvas is taken from ``masks`` exactly as the box path takes
    it from the image batch.

    Resampling is nearest-neighbour, since the input is already binary and
    interpolating it would only reintroduce a threshold. It is applied by mapping
    each *output* pixel centre through the forward affine and reading the canvas
    pixel that contains it — the same point correspondence the box inverse uses,
    rather than a crop-then-resize whose implicit ratio is the **rounded** content
    size over the original size instead of ``r`` (A37).

    Args:
        masks: Boolean (or numeric) masks ``(N, H, W)`` for one image, at the
            letterboxed canvas size.
        letterbox: The validation transform whose ``allow_upscale`` setting
            defines the geometry.
        orig_size: Original image ``(height, width)``.

    Returns:
        Masks of shape ``(N, orig_height, orig_width)`` in original-image
        coordinates, in ``masks``' own dtype.

    Examples:
        >>> import torch
        >>> from lucid_yolo.data.letterbox import Letterbox
        >>> # A 2x4 image letterboxed into a 4x4 canvas gains 1px top/bottom pads.
        >>> masks = torch.zeros(1, 4, 4, dtype=torch.bool)
        >>> masks[0, 1:3] = True  # the content rows carry the whole instance
        >>> recovered = masks_to_original(masks, Letterbox(4), orig_size=(2, 4))
        >>> recovered.shape
        torch.Size([1, 2, 4])
        >>> bool(recovered.all())  # the padding is gone, the content survives
        True
    """
    orig_h, orig_w = orig_size
    canvas_h, canvas_w = int(masks.shape[-2]), int(masks.shape[-1])
    canvas = Letterbox((canvas_h, canvas_w), allow_upscale=letterbox.allow_upscale)
    matrix, _, _ = canvas.forward_affine(orig_h, orig_w)
    rows = _source_indices(orig_h, float(matrix[1, 1]), float(matrix[1, 2]), canvas_h)
    cols = _source_indices(orig_w, float(matrix[0, 0]), float(matrix[0, 2]), canvas_w)
    gathered = masks.index_select(-2, rows.to(masks.device))
    return gathered.index_select(-1, cols.to(masks.device))


def _source_indices(length: int, ratio: float, offset: float, limit: int) -> Tensor:
    """Map ``length`` original pixel centres onto the canvas rows/columns holding them.

    Centre ``i + 0.5`` of the original image sits at ``(i + 0.5) * ratio +
    offset`` on the letterboxed canvas, and the pixel containing that coordinate
    is its floor. The arithmetic runs in ``float64`` on the CPU — the affine is
    float64 by construction and MPS has no double — and the result is clamped
    into the canvas as a guard against a half-pixel overshoot at the far edge.
    """
    centres = torch.arange(length, dtype=torch.float64) + 0.5
    return (centres * ratio + offset).floor().long().clamp(0, limit - 1)
