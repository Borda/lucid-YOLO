# SPDX-License-Identifier: Apache-2.0
"""Anchor-and-stride decode for raw keypoint coordinate offsets (WP-122).

This module is the inference-side counterpart of
:func:`~lucid_yolo.models.heads.detect._build_keypoint_stem`. The head emits a
generic ``K`` points per anchor as raw ``(x, y)`` offsets beside raw per-axis
uncertainty. :func:`decode_keypoints` composes only those coordinate offsets
with the existing anchor grid, returning absolute input-pixel positions without
introducing another geometric convention.

**Composition principle.** The detection head already establishes that raw
regression values are offsets from an anchor centre in stride units:
:func:`~lucid_yolo.models.heads.detect.decode_ltrb` scales each ltrb value by the
anchor's stride before displacing the centre. Keypoints use that same frame,
generalized from four directional box offsets to a direct two-value offset for
each point. Anchor points and strides broadcast over the new ``K`` axis; no box
geometry or task-specific point semantics enter the calculation.

**Uncertainty is deliberately outside this module.** The decoder neither
accepts nor transforms sigma. Callers carry the head's raw, unbounded sigma
tensor alongside the decoded coordinates unchanged. WP-123's hand-written RLE
loss owns the equation-tied positivity mapping recorded by A65, so applying an
activation, absolute value, exponential, or clamp here would cross that work
package boundary and change the future likelihood.

The function is pure: it owns no parameters, mutates no input, performs no
selection, and has no side effects. Shape incompatibilities follow PyTorch's
broadcasting errors rather than being repaired into a plausible but misaligned
point tensor.

Provenance: R14. Assumptions: A65 (sigma stays raw).
"""

from __future__ import annotations

from torch import Tensor

__all__ = ["decode_keypoints"]


def decode_keypoints(raw_coords: Tensor, anchor_points: Tensor, strides: Tensor) -> Tensor:
    """Decode raw per-point offsets into absolute input-pixel coordinates.

    This is the anchor-centre, stride-scaled-offset convention established by
    :func:`~lucid_yolo.models.heads.detect.decode_ltrb`, generalized from one
    ltrb quadruple to ``K`` direct ``(x, y)`` offsets per anchor::

        x = anchor_x + raw_x * stride
        y = anchor_y + raw_y * stride

    Sigma is intentionally not accepted or transformed here. Callers pass it
    through raw until WP-123's RLE loss applies its equation-defined mapping
    (A65).

    Args:
        raw_coords: Raw point offsets of shape ``(B, A, K, 2)``.
        anchor_points: Anchor-centre ``(x, y)`` coordinates of shape ``(A, 2)``
            in input pixels, as returned by
            :func:`~lucid_yolo.assign.grid.make_anchor_points`.
        strides: Per-anchor level stride of shape ``(A,)``.

    Returns:
        Absolute point coordinates of shape ``(B, A, K, 2)`` in input pixels.

    Examples:
        >>> import torch
        >>> raw = torch.tensor([[[[1.0, 2.0], [-1.0, 0.5]], [[0.25, -0.5], [2.0, 1.0]]]])
        >>> anchors = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        >>> strides = torch.tensor([2.0, 4.0])
        >>> decode_keypoints(raw, anchors, strides)  # each anchor uses its own stride
        tensor([[[[12., 24.],
                  [ 8., 21.]],
        <BLANKLINE>
                 [[31., 38.],
                  [38., 44.]]]])
    """
    centres = anchor_points.view(1, anchor_points.shape[0], 1, 2)
    scales = strides.view(1, strides.shape[0], 1, 1)
    return centres + raw_coords * scales
