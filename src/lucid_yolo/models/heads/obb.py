# SPDX-License-Identifier: Apache-2.0
"""NMS-free oriented decode for the angle branch (WP-062).

The inference-side counterpart of the head's orientation stems
(:func:`~lucid_yolo.models.heads.detect._build_angle_stem`, A20). Two pure
helpers turn the one-to-one branch's dense outputs into oriented detections, and
they are the exact rotated analogues of
:func:`~lucid_yolo.models.heads.detect.decode_ltrb` and
:func:`~lucid_yolo.models.heads.detect.o2o_topk`, deliberately mirroring them
step for step so the two paths cannot drift apart.

**How an oriented box is assembled.** R1 gives Eq. 13 — ``theta_hat = z``, the
angle read straight off the branch with the Eq. 12 sigmoid squashing of the
previous versions removed — and says nothing about how that scalar combines with
the ltrb regression. The composition used here is the minimal one the "separate
branch" wording implies: the ltrb distances are decoded **exactly** as the
axis-aligned path decodes them, the resulting ``xyxy`` box supplies the centre and
the two extents, and ``theta`` then rotates that box about its own centre. The
angle branch therefore adds orientation to the existing box regression rather than
reinterpreting it, and a predicted ``theta`` of 0 reproduces the axis-aligned
decode exactly — a property :mod:`tests.models.test_obb_head` pins, and the
cheapest available check that the two paths still agree. This composition is not
stated by R1; it is registered as A44, together with the readings it forecloses.

**Why the range is fixed here and not in the head.** Eq. 13 leaves the emitted
angle unbounded — the stem ends at a raw 1x1 and nothing squashes it — so the
head can and will emit any real number, at any magnitude, throughout early
training. A23 places the normalization after the decode:
:func:`decode_rboxes` therefore ends in
:func:`~lucid_yolo.data.rotated_geom.canonicalize`, which returns the unique
long-edge representative with ``w >= h`` and ``theta`` in the half-open
``[-pi/4, 3*pi/4)``. Canonicalizing the **dense** output rather than only the
selected detections is what makes the guarantee unconditional: a caller that
decodes without ranking (a loss, an assigner, an export graph) cannot end up with
an un-normalized angle by taking a shorter path.

**Selection happens exactly once.** :func:`o2o_rotated_topk` does not rank
anything itself — it delegates to
:func:`~lucid_yolo.models.heads.detect.o2o_topk_with_indices` and gathers the
angle by the anchor indices that helper reports. A second ranking computed here
would be free to disagree with the first, pairing one anchor's orientation with
another anchor's box: a detection with a correct centre, a correct score, and a
silently wrong heading.

The oriented detection tuple is ``[cx, cy, w, h, theta, score, class]`` — the
five-column long-edge rotated box of
:mod:`~lucid_yolo.data.rotated_geom` followed by the score and class columns the
axis-aligned A9 tuple ends with, so the two layouts differ only by the box
columns they carry (A45).

Provenance: R1 sec. 3.4.3, R1 Eq. 13, R3 sec. 4, R13.
Assumptions: A9, A20, A23, A44 (the ltrb-plus-theta composition), A45 (the tuple).
"""

from __future__ import annotations

import torch
from torch import Tensor

from lucid_yolo.data.rotated_geom import canonicalize
from lucid_yolo.models.heads.detect import decode_ltrb, o2o_topk_with_indices

__all__ = ["RBOX_DET_WIDTH", "decode_rboxes", "o2o_rotated_topk"]

#: Width of the oriented detection tuple ``[cx, cy, w, h, theta, score, class]``.
RBOX_DET_WIDTH = 7

#: Column count of a long-edge rotated box ``(cx, cy, w, h, theta)``.
_RBOX_COLUMNS = 5

#: Default per-image detection cap, shared with the axis-aligned path (A9).
_DEFAULT_TOPK = 300


def decode_rboxes(distances: Tensor, angles: Tensor, anchor_points: Tensor, strides: Tensor) -> Tensor:
    """Decode raw ltrb distances and raw angles into canonical rotated boxes.

    Runs :func:`~lucid_yolo.models.heads.detect.decode_ltrb` unchanged — the
    oriented path shares the axis-aligned path's box regression rather than
    reinterpreting it — converts the resulting ``xyxy`` corners to a centre and
    two extents, attaches the raw angle of R1 Eq. 13, and returns the
    :func:`~lucid_yolo.data.rotated_geom.canonicalize`\\ d result (A23).

    The angle arrives unbounded, and canonicalization is what makes the output
    usable regardless: whatever real number the branch emits, the returned
    ``theta`` lies in ``[-pi/4, 3*pi/4)`` and the returned ``w`` is the long edge.
    Because ``theta`` and ``theta + pi`` describe the same rectangle, the decode
    is also continuous across that identification — two raw values a half turn
    apart yield the same box, not two different ones.

    Extents are passed through as decoded and are **not** clamped, matching the
    axis-aligned decode: an untrained head emitting ``r < -l`` produces a
    negative extent there and here alike, and clamping only in the oriented path
    would hide that divergence rather than fix it.

    Args:
        distances: Raw ltrb distances of shape ``(B, A, 4)`` (order l, t, r, b).
        angles: Raw orientation angles in radians, shape ``(B, A, 1)`` or
            ``(B, A)``, aligned with ``distances`` on the anchor axis.
        anchor_points: Anchor-centre ``(x, y)`` coordinates of shape ``(A, 2)``
            in input pixels, as returned by
            :func:`~lucid_yolo.assign.grid.make_anchor_points`.
        strides: Per-anchor level stride of shape ``(A,)``.

    Returns:
        Canonical long-edge rotated boxes of shape ``(B, A, 5)``, each row
        ``(cx, cy, w, h, theta)`` with ``w >= h`` and ``theta`` in
        ``[-pi/4, 3*pi/4)``.

    Examples:
        >>> import torch
        >>> distances = torch.tensor([[[3.0, 1.0, 4.0, 2.0]]])  # (1, 1, 4)
        >>> anchor_points = torch.tensor([[10.0, 10.0]])
        >>> strides = torch.tensor([2.0])
        >>> angles = torch.zeros(1, 1, 1)  # theta = 0 reproduces the axis-aligned box
        >>> decode_rboxes(distances, angles, anchor_points, strides)
        tensor([[[11., 11., 14.,  6.,  0.]]])
        >>> # A raw angle far outside the canonical range still lands inside it.
        >>> wild = torch.full((1, 1, 1), 500.0)
        >>> theta = float(decode_rboxes(distances, wild, anchor_points, strides)[0, 0, 4])
        >>> -torch.pi / 4 <= theta < 3 * torch.pi / 4
        True
    """
    boxes = decode_ltrb(distances, anchor_points, strides)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    theta = angles.squeeze(-1) if angles.ndim == boxes.ndim else angles
    rboxes = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, theta), dim=-1)
    return canonicalize(rboxes.reshape(-1, _RBOX_COLUMNS)).reshape(rboxes.shape)


def o2o_rotated_topk(scores: Tensor, rboxes: Tensor, k: int = _DEFAULT_TOPK) -> Tensor:
    """Reduce the one-to-one branch to the top-k score-ranked oriented detections.

    The oriented twin of :func:`~lucid_yolo.models.heads.detect.o2o_topk`: score
    each anchor by its maximum class confidence after a sigmoid, keep the ``k``
    highest-scoring anchors per image, and emit the oriented tuple
    ``[cx, cy, w, h, theta, score, class]``. There is no IoU computation and no
    suppression — two overlapping high-score boxes both survive, which is the
    defining property of the end-to-end path (R3 sec. 4). When the anchor count
    ``A`` is below ``k`` every anchor is returned.

    The ranking itself is
    :func:`~lucid_yolo.models.heads.detect.o2o_topk_with_indices`', not a second
    copy: the four centre-and-extent columns ride through that helper as its box
    argument and ``theta`` is gathered by the anchor indices it reports, so the
    orientation of row ``n`` provably comes from the same anchor as its box.

    Args:
        scores: Raw class logits of shape ``(B, A, C)``.
        rboxes: Canonical rotated boxes of shape ``(B, A, 5)`` from
            :func:`decode_rboxes`, aligned with ``scores`` on the anchor axis.
        k: Maximum detections kept per image. Defaults to 300 (A9).

    Returns:
        A tensor of shape ``(B, min(k, A), 7)`` whose last axis is
        ``[cx, cy, w, h, theta, score, class]``; ``score`` lies in ``[0, 1]`` and
        ``class`` holds the integral class index as a float.

    Examples:
        >>> import torch
        >>> scores = torch.tensor([[[2.0, -1.0], [-3.0, 0.5]]])  # (1, 2, 2)
        >>> rboxes = torch.tensor([[[0.0, 0.0, 4.0, 2.0, 0.3], [9.0, 9.0, 2.0, 1.0, -0.2]]])
        >>> detections = o2o_rotated_topk(scores, rboxes, k=1)
        >>> detections.shape
        torch.Size([1, 1, 7])
        >>> detections[0, 0, :5]  # anchor 0 outranks anchor 1, orientation included
        tensor([0.0000, 0.0000, 4.0000, 2.0000, 0.3000])
        >>> int(detections[0, 0, 6])  # class index of the top detection
        0
    """
    partial, anchor_index = o2o_topk_with_indices(scores, rboxes[..., :4], k=k)
    theta = rboxes[..., 4].gather(1, anchor_index).unsqueeze(-1)
    return torch.cat((partial[..., :4], theta, partial[..., 4:]), dim=-1)
