# SPDX-License-Identifier: Apache-2.0
"""Rotated-box arithmetic shared by the geometric augmentations (WP-058, A40).

WP-055 established the long-edge form and WP-057 the crop-time geometry; this module is
what the *training-time* transforms — :class:`~lucid_yolo.data.augment.HorizontalFlip`,
:class:`~lucid_yolo.data.affine.RandomAffine`, :class:`~lucid_yolo.data.mosaic.MosaicAssembly`
— call so the rotated modality moves with the image instead of being rejected. One
implementation, four call sites: the alternative is four subtly different angle
conventions.

A general affine does not map a rectangle to a rectangle:
    Only a **similarity** — rotation, uniform scale, translation — preserves
    rectangularity. The random affine also samples per-axis shear, which sends a rectangle
    to a parallelogram, and no ``(cx, cy, w, h, theta)`` describes a parallelogram. So
    :func:`warp_rboxes` expands the box to the four corners the image warp actually moves,
    pushes those corners through the very same matrix, and re-fits with
    :func:`~lucid_yolo.data.rotated_geom.polygons_to_rboxes` — R18's own prescription for
    its cropped parts ("we need to ensure they can be described as an oriented bounding
    box with 4 vertices in the clockwise order with a fitting method"). The residual is
    stated rather than hidden: under a similarity the fit is **exact** (corners map to
    corners), under shear it is a **fit** whose corners sit up to ``h * sin(s / 2)`` from
    the warped parallelogram's, where ``s`` is the angle by which the shear tilted the two
    edge directions off perpendicular.

Canonical on the way out, everywhere:
    Every function here returns canonical long-edge boxes (``w >= h``, ``theta`` in
    ``[-pi/4, 3*pi/4)``). :func:`~lucid_yolo.data.rotated_geom.polygons_to_rboxes` already
    canonicalizes; the flip and shift paths, which are exact analytic maps rather than
    fits, call :func:`~lucid_yolo.data.rotated_geom.canonicalize` explicitly. That closes a
    live defect: WP-013's flip negated ``theta`` and stopped, so every box with
    ``theta > pi/4`` left the transform outside the canonical range. Negation is the
    *right* mirror (``-theta`` and ``pi - theta`` differ by ``pi``, which a rectangle is
    invariant under) — it was only ever the re-wrap that was missing.

Clipping, and the one place augmentation diverges from WP-057:
    :func:`clip_rboxes_to_canvas` clips against the canvas with **WP-057's** Sutherland-
    Hodgman clipper and re-fits with WP-057's orientation-preserving fit, rather than
    growing a second clipper. What it does *not* inherit is R18's 0.7 rule: tiling is
    dataset preparation, where a clipped part is **flagged** difficult and kept, whereas
    augmentation is training-time and :class:`~lucid_yolo.data.targets.Targets` carries no
    ``difficult`` field to flag into. Augmentation therefore **drops** instances, by
    exactly the visibility rule the axis-aligned path already applies (``min_box_size`` on
    the clipped envelope's sides and ``min_visibility`` on clipped-over-pre-clip envelope
    area) — one policy for both modalities. This module supplies the two envelopes; the
    threshold lives with the caller that owns those two numbers (A40).

Instance axis:
    WP-056's invariant holds throughout: ``rboxes[i]`` is the same instance as
    ``boxes[i]``/``labels[i]``, so a transform that drops one drops all three via
    ``Targets.filter(keep, rkeep=keep)``. :func:`check_rotated_pairing` is the guard that
    makes a caller violating it fail loudly rather than emit crossed modalities.

An instance wholly inside the canvas takes no clip at all — it is returned bit for bit —
so the common case costs one vectorized containment test, and only boxes that actually
cross an edge enter the per-instance Python loop the variable clipped-vertex count forces.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lucid_yolo.data.rotated_geom import canonicalize, polygons_to_rboxes, rboxes_to_polygons
from lucid_yolo.data.targets import Targets

# WP-057's clipper, area and orientation-preserving fit, imported deliberately rather than
# reimplemented: the geometry of "rotated box meets axis-aligned rectangle" is settled
# there, and a second copy would be a second set of edge cases. They stay private to
# `tiling` because emitting *tiles* is that module's public job, not this one's.
from lucid_yolo.data.tiling import _clip_to_window, _fit_corners, _polygon_area
from lucid_yolo.data.transforms import apply_affine_to_points

__all__ = [
    "check_rotated_pairing",
    "clip_rboxes_to_canvas",
    "mirror_rboxes",
    "rbox_envelopes",
    "shift_rboxes",
    "warp_rboxes",
]

#: Column count of a long-edge rotated box ``(cx, cy, w, h, theta)``.
_RBOX_DIM = 5
#: Column count of an ``xyxy`` axis-aligned box.
_BOX_DIM = 4
#: Column count of a point ``(x, y)``.
_POINT_DIM = 2
#: Corner count of the quadrilateral form of a rotated box.
_QUAD_CORNERS = 4
#: Minimum vertex count for a clipped ring to enclose any area (matches WP-057).
_MIN_AREA_CORNERS = 3


def check_rotated_pairing(targets: Targets) -> None:
    """Assert the oriented path's instance-axis invariant before a filtering transform.

    A transform that drops instances applies one keep mask to every modality, which is
    only meaningful when ``rboxes[i]`` is the same instance as ``boxes[i]`` (WP-056). This
    is the guard for that: with rotated boxes present the two axes must have equal length,
    and polygons must be absent — the oriented path carries none, and a transform cannot
    invent the ring a mask-carrying instance would need. Empty ``rboxes`` pass through
    silently, so the axis-aligned pipeline never meets this check.

    Args:
        targets: The targets a rotated-aware transform is about to process.

    Raises:
        ValueError: If ``rboxes`` is non-empty and either does not share the instance axis
            with ``boxes`` or arrives alongside polygon rings.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> check_rotated_pairing(Targets.empty())  # nothing rotated, nothing to check
        >>> paired = Targets(
        ...     boxes=torch.tensor([[0.0, 0.0, 4.0, 2.0]]),
        ...     labels=torch.tensor([0]),
        ...     rboxes=torch.tensor([[2.0, 1.0, 4.0, 2.0, 0.0]]),
        ... )
        >>> check_rotated_pairing(paired)

        ```
    """
    if targets.rboxes.shape[0] == 0:
        return
    if targets.rboxes.shape[0] != targets.boxes.shape[0]:
        raise ValueError(
            f"rboxes must share the instance axis with boxes: {targets.boxes.shape[0]} boxes, "
            f"{targets.rboxes.shape[0]} rboxes (WP-056)"
        )
    if targets.polygons:
        raise ValueError(f"the oriented path carries no polygons (WP-056); got {len(targets.polygons)} rings")


def mirror_rboxes(rboxes: Tensor, width: float) -> Tensor:
    """Mirror rotated boxes about the vertical axis at ``x = (width - 1) / 2``.

    The centre reflects (``cx' = (width - 1) - cx``) and the long-edge direction reflects with
    it: ``u = (cos theta, sin theta)`` maps to ``(-cos theta, sin theta)``, the direction
    of ``pi - theta``, which is ``-theta`` plus a half turn — and a rectangle is invariant
    under a half turn. Extents are unchanged, a mirror being an isometry. The result is
    canonicalized, which is what keeps a box with ``theta > pi/4`` in range: its bare
    negation would not be (the WP-013 defect this closes).

    Args:
        rboxes: ``(M, 5)`` rotated boxes ``(cx, cy, w, h, theta)``, canonical or not.
        width: Canvas width in pixels; the mirror line sits at ``(width - 1) / 2``, the
            axis the image's own column reversal reflects about (WP-154b).

    Returns:
        ``(M, 5)`` canonical rotated boxes describing the mirrored rectangles.

    Examples:
        ```pycon
        >>> import torch
        >>> box = torch.tensor([[3.0, 5.0, 8.0, 4.0, 1.2]])  # theta above pi/4
        >>> [round(v, 4) for v in mirror_rboxes(box, width=10.0)[0].tolist()]
        [6.0, 5.0, 8.0, 4.0, 1.9416]

        ```
    """
    mirrored = rboxes.clone()
    mirrored[:, 0] = (width - 1) - rboxes[:, 0]
    mirrored[:, 4] = -rboxes[:, 4]
    return canonicalize(mirrored)


def shift_rboxes(rboxes: Tensor, off_x: float, off_y: float) -> Tensor:
    """Translate rotated boxes by a pixel offset.

    A translation moves the centre and touches nothing else, so unlike :func:`warp_rboxes`
    this needs no corner round trip and introduces no fitting residual — which is why the
    mosaic placement, a pure translation, calls this instead. The output is canonicalized,
    a bit-for-bit no-op on canonical input.

    Args:
        rboxes: ``(M, 5)`` rotated boxes ``(cx, cy, w, h, theta)``, canonical or not.
        off_x: Horizontal offset in pixels, added to ``cx``.
        off_y: Vertical offset in pixels, added to ``cy``.

    Returns:
        ``(M, 5)`` canonical rotated boxes translated by ``(off_x, off_y)``.

    Examples:
        ```pycon
        >>> import torch
        >>> box = torch.tensor([[2.0, 3.0, 6.0, 4.0, 0.5]])
        >>> [round(v, 4) for v in shift_rboxes(box, off_x=10.0, off_y=-1.0)[0].tolist()]
        [12.0, 2.0, 6.0, 4.0, 0.5]

        ```
    """
    shifted = rboxes.clone()
    shifted[:, 0] = rboxes[:, 0] + off_x
    shifted[:, 1] = rboxes[:, 1] + off_y
    return canonicalize(shifted)


def warp_rboxes(rboxes: Tensor, matrix: Tensor) -> Tensor:
    """Push rotated boxes through the same affine the image goes through.

    The box is expanded to the four corners the warp actually moves, those corners are
    mapped by ``matrix``, and a canonical box is re-fitted to them. Under a similarity the
    fit reproduces the warped rectangle exactly; under shear the warped quad is a
    parallelogram and the returned box is the fit described in the module docstring, not a
    lossless re-parameterization.

    Args:
        rboxes: ``(M, 5)`` rotated boxes ``(cx, cy, w, h, theta)``, canonical or not.
        matrix: ``(3, 3)`` homogeneous forward affine, in the dtype the caller carries its
            geometry in (``float64`` for every transform in this package).

    Returns:
        ``(M, 5)`` canonical rotated boxes fitted to the warped corners, in ``rboxes``'s
        dtype.

    Examples:
        ```pycon
        >>> import torch
        >>> double = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        >>> box = torch.tensor([[1.0, 2.0, 4.0, 2.0, 0.0]])
        >>> [round(v, 4) for v in warp_rboxes(box, double)[0].tolist()]
        [2.0, 4.0, 8.0, 4.0, 0.0]

        ```
    """
    corners = rboxes_to_polygons(rboxes)
    flat = corners.reshape(-1, _POINT_DIM).to(matrix.dtype)
    warped = apply_affine_to_points(flat, matrix).to(rboxes.dtype)
    return polygons_to_rboxes(warped.reshape(-1, _QUAD_CORNERS, _POINT_DIM))


def rbox_envelopes(rboxes: Tensor) -> Tensor:
    """Return each rotated box's tight axis-aligned envelope.

    This is the ``boxes[i]`` a rotated-aware transform emits alongside ``rboxes[i]``: the
    envelope of the rotated geometry itself, so the two modalities describe one instance
    rather than drifting apart (A40).

    Args:
        rboxes: ``(M, 5)`` rotated boxes ``(cx, cy, w, h, theta)``, canonical or not.

    Returns:
        ``(M, 4)`` ``xyxy`` boxes enclosing each rotated box's four corners.

    Examples:
        ```pycon
        >>> import torch
        >>> box = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]])
        >>> rbox_envelopes(box).tolist()
        [[-2.0, -1.0, 2.0, 1.0]]

        ```
    """
    corners = rboxes_to_polygons(rboxes)
    return torch.cat([corners.amin(dim=1), corners.amax(dim=1)], dim=1)


def clip_rboxes_to_canvas(rboxes: Tensor, height: float, width: float) -> tuple[Tensor, Tensor]:
    """Clip rotated boxes to a ``[0, width] x [0, height]`` canvas and re-fit.

    A box wholly inside the canvas is returned untouched (bit for bit, after
    canonicalization) — the round trip a transform composed with its inverse depends on.
    A box crossing an edge is clipped by WP-057's Sutherland-Hodgman clipper and re-fitted
    at its **own** orientation, so the clip never invents an angle the annotation did not
    carry; its envelope is the envelope of the clipped polygon, exactly as WP-057 emits
    for a below-threshold part. A box with nothing left inside collapses to zeros, which
    the caller's visibility rule drops (a zero-extent envelope fails any positive
    ``min_box_size`` and scores zero visibility) — augmentation drops where tiling flags,
    per the module docstring.

    Args:
        rboxes: ``(M, 5)`` rotated boxes ``(cx, cy, w, h, theta)``, canonical or not.
        height: Canvas height in pixels.
        width: Canvas width in pixels.

    Returns:
        A ``(clipped, envelopes)`` pair: ``(M, 5)`` canonical rotated boxes and the
        ``(M, 4)`` ``xyxy`` envelopes of the clipped regions, both on the input's instance
        axis with no instance dropped (dropping is the caller's decision).

    Raises:
        ValueError: If ``rboxes`` is not a 2-D ``(M, 5)`` tensor.

    Examples:
        ```pycon
        >>> import torch
        >>> box = torch.tensor([[4.0, 1.0, 4.0, 2.0, 0.0]])  # spans x in [2, 6]
        >>> clipped, envelopes = clip_rboxes_to_canvas(box, height=4.0, width=5.0)
        >>> [round(v, 4) for v in clipped[0].tolist()]
        [3.5, 1.0, 3.0, 2.0, 0.0]
        >>> envelopes[0].tolist()
        [2.0, 0.0, 5.0, 2.0]

        ```
    """
    if rboxes.ndim != 2 or rboxes.shape[1] != _RBOX_DIM:
        raise ValueError(f"rboxes must be (M, 5); got shape {tuple(rboxes.shape)}")
    corners = rboxes_to_polygons(rboxes)
    lower, upper = corners.amin(dim=1), corners.amax(dim=1)
    inside = (lower[:, 0] >= 0.0) & (lower[:, 1] >= 0.0) & (upper[:, 0] <= width) & (upper[:, 1] <= height)
    clipped = canonicalize(rboxes)
    envelopes = torch.cat([lower, upper], dim=1)
    for index in (~inside).nonzero(as_tuple=False).flatten().tolist():
        part_rbox, part_box = _clip_one(
            corners[index].to(torch.float64), clipped[index].to(torch.float64), height, width
        )
        clipped[index] = part_rbox.to(clipped.dtype)
        envelopes[index] = part_box.to(envelopes.dtype)
    return clipped, envelopes


def _clip_one(ring: Tensor, rbox: Tensor, height: float, width: float) -> tuple[Tensor, Tensor]:
    """Clip one rotated box's corner ring to the canvas and re-fit it.

    Args:
        ring: ``(4, 2)`` float64 corners of the rotated box.
        rbox: ``(5,)`` float64 rotated box supplying the fit's orientation.
        height: Canvas height in pixels.
        width: Canvas width in pixels.

    Returns:
        The ``(5,)`` re-fitted rotated box and its ``(4,)`` ``xyxy`` envelope, both
        float32; zeros when nothing of the box survives the clip.

    Examples:
        >>> import torch
        >>> quad = torch.tensor([[2.0, 0.0], [6.0, 0.0], [6.0, 2.0], [2.0, 2.0]], dtype=torch.float64)
        >>> rb = torch.tensor([4.0, 1.0, 4.0, 2.0, 0.0], dtype=torch.float64)
        >>> _clip_one(quad, rb, height=4.0, width=5.0)[1].tolist()
        [2.0, 0.0, 5.0, 2.0]
    """
    part = _clip_to_window(ring, 0.0, 0.0, width, height)
    if part.shape[0] < _MIN_AREA_CORNERS or float(_polygon_area(part)) <= 0.0:
        return torch.zeros(_RBOX_DIM), torch.zeros(_BOX_DIM)
    fitted = polygons_to_rboxes(_fit_corners(part, rbox).to(torch.float32).unsqueeze(0))[0]
    envelope = torch.cat([part.amin(dim=0), part.amax(dim=0)]).to(torch.float32)
    return fitted, envelope
