# SPDX-License-Identifier: Apache-2.0
"""Rotated-box arithmetic shared by the geometric augmentations (WP-058, A40).

WP-055 established the long-edge form and WP-057 the crop-time geometry; this module is
what the *training-time* transforms — :class:`~lucid_yolo.data.augment.HorizontalFlip`,
:class:`~lucid_yolo.data.affine.RandomAffine`, :class:`~lucid_yolo.data.mosaic.MosaicAssembly`
— call so the rotated modality moves with the image instead of being rejected. Since
WP-157 the arithmetic itself belongs to :mod:`fuse_augmentations`; what stays here is the
task convention that arithmetic has to be parameterized by, and the one operation upstream
declines to own.

Delegated, with this project's angle convention passed in (A22):
    :func:`mirror_rboxes`, :func:`shift_rboxes`, :func:`warp_rboxes` and
    :func:`rbox_envelopes` are adapters over :func:`~fuse_augmentations.mirror_rboxes`,
    :func:`~fuse_augmentations.shift_rboxes`, :func:`~fuse_augmentations.transform_rboxes`
    and :func:`~fuse_augmentations.rbox_envelopes`. Upstream imposes no angle convention of
    its own and its ``canonicalize`` argument defaults to ``None``, which returns the box
    exactly as the arithmetic left it — so each of the three transforms is handed
    :func:`~lucid_yolo.data.rotated_geom.canonicalize` explicitly. That is deliberate and
    not defensive: the long-edge range is what the assigner, the rotated NMS and the OBB
    head read, it is a task convention rather than a resampling concern, and a mirror test
    would not catch its absence, because the two forms a missing wrap leaves behind differ
    by a half turn that a rectangle is invariant under.

    Upstream's mirror returns ``pi - theta`` where WP-058 returned ``-theta``. Those differ
    by exactly ``pi``, so canonicalization collapses them to one box and only the rounding
    of the wrap differs — measured below ``3e-7`` on float32 angles, against the ``1e-4``
    the frozen geometric expectations are compared at. No frozen value moved for this row.

A general affine does not map a rectangle to a rectangle:
    Only a **similarity** — rotation, uniform scale, translation — preserves
    rectangularity. The random affine also samples per-axis shear, which sends a rectangle
    to a parallelogram, and no ``(cx, cy, w, h, theta)`` describes a parallelogram. So
    :func:`~fuse_augmentations.transform_rboxes` expands the box to the four corners the
    image warp actually moves, pushes those corners through the very same matrix, and
    re-fits — R18's own prescription for its cropped parts ("we need to ensure they can be
    described as an oriented bounding box with 4 vertices in the clockwise order with a
    fitting method"). Under a similarity that fit is exact; under shear it is a fit, whose
    residual upstream's own docstring states in closed form rather than hiding.

Clipping, the operation upstream refuses by design:
    Clipping a rotated box yields a polygon, not a rotated box, so :mod:`fuse_augmentations`
    supplies :func:`~fuse_augmentations.rbox_envelopes` and a plain-box clip and declines to
    invent the rectangle — which is why :func:`clip_rboxes_to_canvas` and its helper are the
    two bodies that stay here rather than becoming adapters. It clips against the canvas
    with **WP-057's** Sutherland-Hodgman clipper and re-fits with WP-057's
    orientation-preserving fit, rather than growing a second clipper. What it does *not*
    inherit is R18's 0.7 rule: tiling is dataset preparation, where a clipped part is
    **flagged** difficult and kept, whereas augmentation is training-time and
    :class:`~lucid_yolo.data.targets.Targets` carries no ``difficult`` field to flag into.
    Augmentation therefore **drops** instances, by exactly the visibility rule the
    axis-aligned path already applies (``min_box_size`` on the clipped envelope's sides and
    ``min_visibility`` on clipped-over-pre-clip envelope area) — one policy for both
    modalities. This module supplies the two envelopes; the threshold lives with the caller
    that owns those two numbers (A40).

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
from fuse_augmentations import (  # type: ignore[import-untyped]
    mirror_rboxes as _fuse_mirror_rboxes,
)
from fuse_augmentations import (
    rbox_envelopes as _fuse_rbox_envelopes,
)
from fuse_augmentations import (
    shift_rboxes as _fuse_shift_rboxes,
)
from fuse_augmentations import (
    transform_rboxes as _fuse_transform_rboxes,
)
from torch import Tensor

from lucid_yolo.data.rotated_geom import canonicalize, polygons_to_rboxes, rboxes_to_polygons
from lucid_yolo.data.targets import Targets

# WP-057's clipper, area and orientation-preserving fit, imported deliberately rather than
# reimplemented: the geometry of "rotated box meets axis-aligned rectangle" is settled
# there, and a second copy would be a second set of edge cases. They stay private to
# `tiling` because emitting *tiles* is that module's public job, not this one's.
from lucid_yolo.data.tiling import _clip_to_window, _fit_corners, _polygon_area

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
#: Minimum vertex count for a clipped ring to enclose any area (matches WP-057).
_MIN_AREA_CORNERS = 3


def _canonicalize_batched(rboxes: Tensor) -> Tensor:
    """Adapt the long-edge canonicalizer to upstream's batched box layout.

    :func:`~lucid_yolo.data.rotated_geom.canonicalize` takes a flat ``(M, 5)`` and rejects
    anything else; :func:`~fuse_augmentations.transform_rboxes` hands its ``canonicalize``
    callback the ``(batch_size, num_boxes, 5)`` it works in. Flattening and restoring is
    the whole adapter — canonicalization is per box and carries no cross-box state, so the
    round trip changes nothing about the result.

    Args:
        rboxes: Rotated boxes ``(cx, cy, w, h, theta)`` in any leading-dimension layout
            whose trailing dimension is 5.

    Returns:
        Canonical long-edge boxes in ``rboxes``'s own shape.

    Examples:
        ```pycon
        >>> import torch
        >>> tall = torch.tensor([[[10.0, 10.0, 4.0, 8.0, 0.0]]])  # short edge stored first
        >>> [round(v, 4) for v in _canonicalize_batched(tall)[0, 0].tolist()]
        [10.0, 10.0, 8.0, 4.0, 1.5708]

        ```
    """
    return canonicalize(rboxes.reshape(-1, _RBOX_DIM)).reshape(rboxes.shape)


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

    :func:`~fuse_augmentations.mirror_rboxes` does the reflection: the centre reflects
    (``cx' = (width - 1) - cx``) and the long-edge direction reflects with it, since
    ``u = (cos theta, sin theta)`` maps to ``(-cos theta, sin theta)``, the direction of
    ``pi - theta``. Extents are unchanged, a mirror being an isometry. This project's
    :func:`~lucid_yolo.data.rotated_geom.canonicalize` is passed in as the callback, which
    is what keeps a box with ``theta > pi/4`` in range — upstream returns the raw
    ``pi - theta`` when no callback is supplied, and the WP-013 defect this closes was
    exactly a missing re-wrap.

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
    mirrored: Tensor = _fuse_mirror_rboxes(rboxes, width, canonicalize=canonicalize)
    return mirrored


def shift_rboxes(rboxes: Tensor, off_x: float, off_y: float) -> Tensor:
    """Translate rotated boxes by a pixel offset.

    A translation moves the centre and touches nothing else, so unlike :func:`warp_rboxes`
    this needs no corner round trip and introduces no fitting residual — which is why
    :func:`~fuse_augmentations.shift_rboxes` exists as its own upstream path, and why the
    mosaic placement calls this instead. This project's canonicalizer is passed in as the
    callback, a bit-for-bit no-op on canonical input.

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
    shifted: Tensor = _fuse_shift_rboxes(rboxes, off_x, off_y, canonicalize=canonicalize)
    return shifted


def warp_rboxes(rboxes: Tensor, matrix: Tensor) -> Tensor:
    """Push rotated boxes through the same affine the image goes through.

    :func:`~fuse_augmentations.transform_rboxes` expands the box to the four corners the
    warp actually moves, maps them by ``matrix`` and re-fits; this project's canonicalizer
    is passed in so the fit lands in the long-edge range. Under a similarity the fit
    reproduces the warped rectangle exactly; under shear the warped quad is a parallelogram
    and the returned box is a fit, not a lossless re-parameterization.

    Upstream works batched, so the single instance axis is wrapped and unwrapped here, and
    the corner round trip runs in ``matrix``'s dtype — the geometry dtype the caller warps
    its image by, ``float64`` throughout this package. The narrowing back to ``rboxes``'s
    own dtype happens **inside** the ``canonicalize`` callback rather than after it: a
    canonical ``float64`` angle sitting a hair inside ``[-pi/4, 3*pi/4)`` can round out of
    that half-open range on the way to ``float32``, and the range is a postcondition the
    assigner and the OBB head rely on in the dtype :class:`~lucid_yolo.data.targets.Targets`
    actually carries, not in the one the warp was computed in.

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

    def _canonical_in_box_dtype(fitted: Tensor) -> Tensor:
        """Narrow to the box dtype, *then* canonicalize, so the range holds where it is read."""
        return _canonicalize_batched(fitted.to(rboxes.dtype))

    warped: Tensor = _fuse_transform_rboxes(
        rboxes.to(matrix.dtype).unsqueeze(0), matrix.unsqueeze(0), canonicalize=_canonical_in_box_dtype
    )[0]
    return warped


def rbox_envelopes(rboxes: Tensor) -> Tensor:
    """Return each rotated box's tight axis-aligned envelope.

    This is the ``boxes[i]`` a rotated-aware transform emits alongside ``rboxes[i]``: the
    envelope of the rotated geometry itself, so the two modalities describe one instance
    rather than drifting apart (A40). :func:`~fuse_augmentations.rbox_envelopes` computes
    it, and takes no ``canonicalize`` argument because an envelope carries no angle to
    canonicalize — it is the bridge *out* of the rotated convention, not a box in it.

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
    envelopes: Tensor = _fuse_rbox_envelopes(rboxes)
    return envelopes


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
