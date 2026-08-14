# SPDX-License-Identifier: Apache-2.0
"""Path-agnostic decode building blocks shared by both detection paths.

Both inference decoders emit the **same** fixed-size A9 detection tuple so a
single evaluator can consume either interchangeably: the suppression-free
score-based top-k path over the one-to-one branch (WP-041) and the
confidence-threshold path over the dense one-to-many branch (WP-042). This module
holds the pieces they share — the A9 tuple column layout, the fixed-length
zero-padding step, and the eval-time inverse-letterbox hook.

The A9 detection tuple is ``[x1, y1, x2, y2, score, class]`` with ``score`` in
``[0, 1]`` and ``class`` the integral class index stored as a float. Rows beyond
the available detections carry ``score == 0`` and are ignored by an evaluation
loop that filters on score.

This module deliberately names no path-specific operator, so the WP-041 meta-test
that reads the one-to-one decoder's source can keep pinning its suppression-free
property structurally.

Provenance: R1 sec. 3.2.1, R3 sec. 4. Assumptions: A9, A10.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lucid_yolo.data.letterbox import Letterbox

__all__ = [
    "BOX_CORNERS",
    "DET_WIDTH",
    "LABEL_COLUMN",
    "PAD_ANCHOR_INDEX",
    "RBOX_COLUMNS",
    "SCORE_COLUMN",
    "pad_anchor_indices",
    "pad_detections",
    "rboxes_to_letterboxed_original",
    "to_letterboxed_original",
]

#: Width of the A9 detection tuple ``[x1, y1, x2, y2, score, class]``.
DET_WIDTH = 6

#: Number of box-coordinate columns (the ``xyxy`` corners) of a detection tuple.
BOX_CORNERS = 4

#: Number of box columns of the A45 oriented tuple ``(cx, cy, w, h, theta)``.
RBOX_COLUMNS = 5

#: Column index of the confidence score within the A9 detection tuple.
SCORE_COLUMN = 4

#: Column index of the integral class label within the A9 detection tuple. Stored as a
#: float like every other column — the tuple is one tensor — and narrowed to ``int`` by
#: whichever boundary reads it (a category-id lookup, a written report).
LABEL_COLUMN = 5

#: Anchor index reported for a padding detection row, which has no source anchor.
#: Negative so it can never be mistaken for a real row and cannot silently index
#: the last anchor the way ``-1`` would if it were used as a gather index directly.
PAD_ANCHOR_INDEX = -1


def pad_detections(detections: Tensor, max_det: int) -> Tensor:
    """Pad the detection axis to a fixed length with score-zero rows.

    Appends all-zero rows (score 0) after the ranked detections so the output
    length along the detection axis (the second-to-last axis) is always
    ``max_det``, regardless of how many detections a given image produced. The
    fixed length is the export-friendly contract: the shape does not depend on
    the anchor count or the survivor count, so a traced graph has a static output
    shape. The score-descending ordering of the input is preserved because the
    appended rows all carry score 0. When the input already has at least
    ``max_det`` rows it is returned unchanged (callers cap before padding).

    Args:
        detections: Ranked detections with the detection count on the
            second-to-last axis, e.g. shape ``(B, N, 6)`` or ``(N, 6)``.
        max_det: Fixed output length along the detection axis.

    Returns:
        Detections whose detection axis has length ``max_det`` (unchanged when
        the input already has at least ``max_det`` rows).

    Examples:
        >>> import torch
        >>> dets = torch.ones(1, 2, 6)
        >>> pad_detections(dets, max_det=4).shape
        torch.Size([1, 4, 6])
        >>> pad_detections(dets, max_det=4)[0, 2:]  # appended rows are zero
        tensor([[0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.]])
    """
    kept = detections.shape[-2]
    if kept >= max_det:
        return detections
    pad_shape = list(detections.shape)
    pad_shape[-2] = max_det - kept
    padding = detections.new_zeros(pad_shape)
    return torch.cat((detections, padding), dim=-2)


def pad_anchor_indices(indices: Tensor, max_det: int) -> Tensor:
    """Pad the anchor-index axis to a fixed length with :data:`PAD_ANCHOR_INDEX`.

    The index-side twin of :func:`pad_detections`, and it must stay that: the two
    are padded to the same length by the same callers so that row ``n`` of a
    decoder's detections and entry ``n`` of its anchor indices describe the same
    detection for every ``n``, padding rows included. Anything gathered per anchor
    and **not** carried in the A9 tuple — the mask coefficients of the
    segmentation decode — is selected by these indices, so a length or ordering
    mismatch here pairs a box with another anchor's coefficients: a plausible mask
    of the wrong object, at a correct box, with a correct score.

    Args:
        indices: Source anchor index per kept detection, shape ``(N,)`` or
            ``(B, N)`` with the detection count last.
        max_det: Fixed output length along the detection axis.

    Returns:
        Indices whose last axis has length ``max_det``, the shortfall filled with
        :data:`PAD_ANCHOR_INDEX` (unchanged when already at least that long).

    Examples:
        >>> import torch
        >>> pad_anchor_indices(torch.tensor([3, 7]), max_det=4)
        tensor([ 3,  7, -1, -1])
        >>> pad_anchor_indices(torch.tensor([[3, 7]]), max_det=2)  # already full
        tensor([[3, 7]])
    """
    kept = indices.shape[-1]
    if kept >= max_det:
        return indices
    pad_shape = list(indices.shape)
    pad_shape[-1] = max_det - kept
    padding = torch.full(pad_shape, PAD_ANCHOR_INDEX, dtype=indices.dtype, device=indices.device)
    return torch.cat((indices, padding), dim=-1)


def to_letterboxed_original(
    detections: Tensor,
    orig_size: tuple[int, int],
    letterboxed_size: tuple[int, int],
    allow_upscale: bool = True,
) -> Tensor:
    """Un-letterbox detection boxes back to original-image coordinates (A10).

    Eval-time hook shared by both decode paths: maps the ``xyxy`` box corners of
    each detection from the letterboxed canvas the model saw back to the original
    image via the exact analytic inverse of
    :class:`~lucid_yolo.data.letterbox.Letterbox`. The score and class columns pass
    through untouched. Padding / sub-threshold rows (score 0) are mapped like any
    other row; they are identified downstream by their zero score, not by their
    coordinates.

    Args:
        detections: Detections of shape ``(B, N, 6)`` with the A9 tuple
            ``[x1, y1, x2, y2, score, class]`` in letterboxed-canvas pixels.
        orig_size: Original image ``(height, width)``.
        letterboxed_size: Letterboxed canvas ``(height, width)`` the boxes live
            in.
        allow_upscale: The letterbox ``allow_upscale`` setting used at resize
            time; must match so the inverse recovers the exact geometry.
            Defaults to ``True`` (the transform default).

    Returns:
        Detections of shape ``(B, N, 6)`` with box corners in original-image
        coordinates and score/class unchanged.

    Examples:
        >>> import torch
        >>> # A 2x4 image letterboxed into a 4x4 canvas gains 1px top/bottom pads.
        >>> boxes = torch.tensor([[[0.0, 1.0, 4.0, 3.0, 0.9, 0.0]]])
        >>> mapped = to_letterboxed_original(boxes, orig_size=(2, 4), letterboxed_size=(4, 4))
        >>> mapped[0, 0, :4]
        tensor([0., 0., 4., 2.])
        >>> mapped[0, 0, 4:]  # score and class survive the round trip
        tensor([0.9000, 0.0000])
    """
    batch, num_det, _ = detections.shape
    letterbox = Letterbox(letterboxed_size, allow_upscale=allow_upscale)
    corner_points = detections[..., :BOX_CORNERS].reshape(-1, 2)
    mapped_points = letterbox.inverse_map(corner_points, orig_size, letterboxed_size)
    mapped_boxes = mapped_points.reshape(batch, num_det, BOX_CORNERS)
    return torch.cat((mapped_boxes, detections[..., BOX_CORNERS:]), dim=-1)


def rboxes_to_letterboxed_original(
    detections: Tensor,
    orig_size: tuple[int, int],
    letterboxed_size: tuple[int, int],
    allow_upscale: bool = True,
) -> Tensor:
    """Un-letterbox **oriented** detections back to original-image coordinates (A10, A45).

    The rotated twin of :func:`to_letterboxed_original`, for the A45 tuple
    ``[cx, cy, w, h, theta, score, class]``. A letterbox is one isotropic scale plus a
    translation, so its inverse maps a rectangle onto a rectangle exactly: the centre
    goes through the same point inverse the axis-aligned corners use, the two extents
    divide by the same ratio, and **``theta`` is unchanged** — an isotropic map turns
    no angle. That is why this is an exact inverse rather than a re-fit, and why the
    long-edge convention survives it: ``w >= h`` is preserved by scaling both by one
    positive number.

    The ratio is not passed in but recovered from the two sizes through
    :class:`~lucid_yolo.data.letterbox.Letterbox`'s own geometry, so the forward
    transform stays the single definition of what a letterbox is (the WP-053a rule
    against a second copy of that arithmetic). Score and class pass through untouched,
    and score-zero padding rows are mapped like any other row — they are identified
    downstream by their score, never by their coordinates.

    Args:
        detections: Oriented detections of shape ``(B, N, 7)`` in letterboxed-canvas
            pixels, as :func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk` emits them.
        orig_size: Original image ``(height, width)``.
        letterboxed_size: Letterboxed canvas ``(height, width)`` the boxes live in.
        allow_upscale: The letterbox ``allow_upscale`` setting used at resize time;
            must match so the inverse recovers the exact geometry. Defaults to ``True``.

    Returns:
        Detections of shape ``(B, N, 7)`` with centres and extents in original-image
        coordinates, ``theta``, ``score`` and ``class`` unchanged.

    Raises:
        ValueError: If ``detections`` is not a 3-D tensor with at least
            :data:`RBOX_COLUMNS` trailing columns.

    Examples:
        >>> import torch
        >>> # A 2x4 image letterboxed into a 4x4 canvas: ratio 1, 1px top/bottom pads.
        >>> dets = torch.tensor([[[2.0, 2.0, 4.0, 2.0, 0.3, 0.9, 1.0]]])
        >>> mapped = rboxes_to_letterboxed_original(dets, orig_size=(2, 4), letterboxed_size=(4, 4))
        >>> mapped[0, 0, :5]  # the centre drops by the top pad; theta is untouched
        tensor([2.0000, 1.0000, 4.0000, 2.0000, 0.3000])
        >>> mapped[0, 0, 5:]  # score and class survive the round trip
        tensor([0.9000, 1.0000])
    """
    if detections.ndim != 3 or detections.shape[-1] < RBOX_COLUMNS:
        raise ValueError(f"detections must be (B, N, >={RBOX_COLUMNS}); got shape {tuple(detections.shape)}")
    batch, num_det, _ = detections.shape
    letterbox = Letterbox(letterboxed_size, allow_upscale=allow_upscale)
    centres = letterbox.inverse_map(detections[..., :2].reshape(-1, 2), orig_size, letterboxed_size)
    # The extents scale by the same ratio the centres do; read it off the inverse rather
    # than recomputing `min(out_h / H, out_w / W)` here, which would be that second copy.
    origin = letterbox.inverse_map(detections.new_zeros((2, 2)), orig_size, letterboxed_size)[0]
    unit = letterbox.inverse_map(detections.new_ones((2, 2)), orig_size, letterboxed_size)[0]
    inverse_ratio = unit[0] - origin[0]
    extents = detections[..., 2:4] * inverse_ratio
    return torch.cat((centres.reshape(batch, num_det, 2), extents, detections[..., 4:]), dim=-1)
