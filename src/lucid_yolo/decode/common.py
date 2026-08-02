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

__all__ = ["BOX_CORNERS", "DET_WIDTH", "SCORE_COLUMN", "pad_detections", "to_letterboxed_original"]

#: Width of the A9 detection tuple ``[x1, y1, x2, y2, score, class]``.
DET_WIDTH = 6

#: Number of box-coordinate columns (the ``xyxy`` corners) of a detection tuple.
BOX_CORNERS = 4

#: Column index of the confidence score within the A9 detection tuple.
SCORE_COLUMN = 4


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
