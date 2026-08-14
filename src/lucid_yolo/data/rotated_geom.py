# SPDX-License-Identifier: Apache-2.0
"""Long-edge rotated-box primitives for the oriented-detection path (WP-055, A23).

A rotated box is ``(cx, cy, w, h, theta)`` float32 in the **long-edge** convention R1
sec. 3.4.3 adopts from MMRotate: ``w >= h``, and ``theta`` in radians on
``[-pi/4, 3*pi/4)`` — the ``[-45, 135)`` degree range whose stated purpose is that it
"alleviates the boundary ambiguity near 0 or 90deg and reduces the instability caused by
edge swapping" (R1 Table 10 measures it at +1.3 mAP over the ``(0, 90]`` definition).
:class:`~lucid_yolo.data.targets.Targets` already *carries* rotated boxes in this form;
this module is where the form is established and enforced.

Conventions, all load-bearing for the work packages that consume this module:

Rotation:
    Image coordinates are y-down. ``theta`` is the angle of the long (``w``) edge
    measured from ``+x`` **towards ``+y``** — clockwise as the image is displayed. The
    box's local frame is ``u = (cos theta, sin theta)`` along ``w`` and
    ``v = (-sin theta, cos theta)`` along ``h``, so a corner is
    ``centre +- (w/2) u +- (h/2) v``.

Canonical form:
    Two moves preserve the rectangle: ``theta += pi`` (a rectangle is invariant under a
    180-degree rotation — the premise R1 Eq. 14 measures its angular residual modulo
    ``pi`` on) and swapping ``w``/``h`` together with ``theta += pi/2`` (which maps
    ``u -> v`` and ``v -> -u``). :func:`canonicalize` applies them so the output always
    satisfies ``w >= h`` and the angle range, and is exactly idempotent: a box already in
    canonical form is returned bit for bit, not merely to tolerance.

Square tie-break:
    When ``w == h`` exactly, ``theta`` and ``theta + pi/2`` both land in range and
    describe the same square, so the range alone does not pin a unique answer. The tie is
    broken **towards zero**: a square's canonical angle lies in ``[-pi/4, pi/4)``.

Winding:
    :func:`rboxes_to_polygons` emits corners from local ``(-w/2, -h/2)`` and steps
    ``(+w/2, -h/2)``, ``(+w/2, +h/2)``, ``(-w/2, +h/2)`` — clockwise as displayed, which
    is a positive shoelace area in the raw ``(x, y)`` values. The order is defined on the
    *canonical* box, so ``theta`` and ``theta + pi`` yield the same corners in the same
    slots rather than one ring rolled by two.

Containment:
    :func:`points_in_rboxes` is edge-**inclusive**: a point exactly on a boundary counts
    as inside. WP-061's rotated TAL/STAL candidate selection (A25) inherits that choice.

Overlap:
    :func:`rotated_iou` is the **exact** pairwise intersection over union (A24):
    Sutherland-Hodgman clipping of one quadrilateral against the other — valid because
    both are convex — then the shoelace area of what survives, with no sampling and no
    Gaussian surrogate. :mod:`lucid_yolo.losses.probiou` approximates a rotated box by its
    uniform-density Gaussian because a loss needs a smooth gradient; a decoder suppressing
    by overlap and a metric scoring one both need the area itself, so it is computed here
    rather than separately in either of them. That single home is what makes
    :class:`~lucid_yolo.decode.rotated_nms.RotatedNMSDecoder` and
    :func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map` agree on what "overlap" means
    by construction (WP-091c). Clipping is edge-inclusive, matching
    :func:`points_in_rboxes`, so boxes sharing only an edge or a corner intersect in a
    zero-area polygon.

Precision, and why there is no float64 here:
    :mod:`lucid_yolo.data.tiling` clips in float64 because DOTA coordinates reach 10^4 px
    and a float32 shoelace difference loses the precision its 0.7 threshold is compared
    at. That escape is not available to :func:`rotated_iou`: it runs on the evaluation and
    the decode paths alike, and both run on MPS, which has no float64 — the same
    constraint :mod:`lucid_yolo.losses.probiou` restructured its algebra for. The working
    dtype there follows the input instead, and the conditioning is fixed structurally:
    intersection over union is **translation invariant**, so every pair is re-centred on
    its own midpoint and only then expanded to corners. A pair of 50 px boxes at
    ``x = 12000`` is clipped at coordinates near zero rather than near 12000, which is
    where a float32 shoelace has its precision, and the remaining operations are all
    like-signed sums over small numbers.

    The order of those two steps is the whole of it, and it is not a detail. Worst
    absolute IoU error over 300 overlapping 20-80 px pairs per row, measured against a
    float64 shapely evaluation, as the pair's common offset from the origin grows::

        common offset   no shift at all   shift after   shift before (shipped)
        0               1.5e-07           2.2e-07       1.4e-07
        1e3             5.1e-05           1.8e-06       1.7e-07
        1e4             6.3e-03           2.7e-05       1.6e-07
        1e5             1.7e+00           1.4e-04       1.4e-07
        1e6             1.5e+00           1.4e-03       1.4e-07

    Shifting *after* the corners are expanded still leaves them rounded at absolute scale
    — a cliff that merely starts later. Shifting *before* makes the kernel scale-free, at
    the cost of canonicalizing ``M * N`` boxes rather than ``M + N``. That price is worth
    paying: tiles are 1024 px local, so the tiled path never exercises the cliff, but
    WP-064 evaluates on whole DOTA images whose coordinates reach 10^4, and a metric that
    quietly loses three digits at that scale would be found by nobody.

Everything here is plain vectorized torch — no numpy, and no Python loop over boxes
anywhere: the only loop is :func:`rotated_iou`'s pass over the four edges of a
quadrilateral, whose count is fixed by the geometry. That matters because
:func:`points_in_rboxes` sits inside the assigner's per-level work and
:func:`rotated_iou` inside both the oriented accumulator and the rotated decoder's
suppression loop. Box parameters are float32 as A23 stores them; :func:`rotated_iou`
alone works in whatever dtype :func:`torch.result_type` gives its two arguments, so a
float64 caller is not silently narrowed.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["canonicalize", "points_in_rboxes", "polygons_to_rboxes", "rboxes_to_polygons", "rotated_iou"]

#: Column count of a long-edge rotated box ``(cx, cy, w, h, theta)``.
_RBOX_DIM = 5
#: Column count of a point ``(x, y)``.
_POINT_DIM = 2
#: Corner count of the quadrilateral form of a rotated box.
_QUAD_CORNERS = 4
#: Vertex bound on the intersection of two convex quadrilaterals. Clipping a 4-gon by
#: ``j`` half-planes yields at most ``4 + j`` vertices, so 8 bounds every intermediate
#: stage of the four-edge pass as well as its result.
_MAX_INTERSECTION_CORNERS = 8
#: Minimum vertex count for a polygon to enclose any area.
_MIN_AREA_CORNERS = 3

_PI = math.pi
_HALF_PI = math.pi / 2
_QUARTER_PI = math.pi / 4
#: Inclusive lower bound of the canonical angle range, ``-45`` degrees.
_THETA_LOW = -math.pi / 4
#: Exclusive upper bound of the canonical angle range, ``135`` degrees.
_THETA_HIGH = 3 * math.pi / 4


def canonicalize(rboxes: Tensor) -> Tensor:
    """Map rotated boxes to their unique long-edge representative.

    Applies the two geometry-preserving moves the module docstring describes: ``w``/``h``
    are swapped (with ``theta += pi/2``) when the short edge was stored first, and
    ``theta`` is shifted by whole multiples of ``pi`` into ``[-pi/4, 3*pi/4)``. An exact
    square, whose two in-range representatives describe the same rectangle, is folded
    towards zero so its angle lands in ``[-pi/4, pi/4)``.

    The function is exactly idempotent: an already-canonical box short-circuits the angle
    wrap and is returned bit for bit, so ``canonicalize(canonicalize(x))`` equals
    ``canonicalize(x)`` under :func:`torch.equal`, not merely under a tolerance.

    Args:
        rboxes: ``(M, 5)`` rotated boxes ``(cx, cy, w, h, theta)``; ``w``, ``h`` and
            ``theta`` may be in any orientation or angle range.

    Returns:
        ``(M, 5)`` canonical rotated boxes with ``w >= h`` and ``theta`` in
        ``[-pi/4, 3*pi/4)``, describing exactly the same rectangles.

    Raises:
        ValueError: If ``rboxes`` is not a 2-D ``(M, 5)`` tensor.

    Examples:
        ```pycon
        >>> import torch
        >>> tall = torch.tensor([[10.0, 10.0, 4.0, 8.0, 0.0]])  # short edge stored first
        >>> [round(v, 4) for v in canonicalize(tall)[0].tolist()]
        [10.0, 10.0, 8.0, 4.0, 1.5708]
        >>> turned = torch.tensor([[0.0, 0.0, 6.0, 3.0, 0.3 + 3.14159265]])
        >>> [round(v, 4) for v in canonicalize(turned)[0].tolist()]
        [0.0, 0.0, 6.0, 3.0, 0.3]

        ```
    """
    _check_2d(rboxes, _RBOX_DIM, "rboxes")
    width, height, theta = rboxes[:, 2], rboxes[:, 3], rboxes[:, 4]
    swap = width < height
    long_edge = torch.where(swap, height, width)
    short_edge = torch.where(swap, width, height)
    angle = _wrap_theta(torch.where(swap, theta + _HALF_PI, theta))
    # An exact square has two in-range representatives a quarter turn apart; fold the
    # upper half onto the lower one so the answer is unique (and stays idempotent).
    fold = (long_edge == short_edge) & (angle >= _QUARTER_PI)
    angle = torch.where(fold, angle - _HALF_PI, angle)
    return torch.stack([rboxes[:, 0], rboxes[:, 1], long_edge, short_edge, angle], dim=1)


def rboxes_to_polygons(rboxes: Tensor) -> Tensor:
    """Expand rotated boxes into their four corner coordinates.

    The boxes are canonicalized first, so the corner slots are a function of the
    *rectangle* rather than of the parameterization it arrived in: ``theta`` and
    ``theta + pi`` produce identical output rather than the same ring rolled by two.
    Corners run from the local ``(-w/2, -h/2)`` corner clockwise as displayed (see the
    module docstring's winding note).

    Args:
        rboxes: ``(M, 5)`` rotated boxes ``(cx, cy, w, h, theta)``, canonical or not.

    Returns:
        ``(M, 4, 2)`` corner coordinates ``(x, y)`` in the fixed winding order.

    Raises:
        ValueError: If ``rboxes`` is not a 2-D ``(M, 5)`` tensor.

    Examples:
        ```pycon
        >>> import torch
        >>> box = torch.tensor([[5.0, 3.0, 4.0, 2.0, 0.0]])
        >>> rboxes_to_polygons(box)[0].tolist()
        [[3.0, 2.0], [7.0, 2.0], [7.0, 4.0], [3.0, 4.0]]

        ```
    """
    canonical = canonicalize(rboxes)
    centre = canonical[:, :2]
    cos, sin = torch.cos(canonical[:, 4]), torch.sin(canonical[:, 4])
    half_w, half_h = canonical[:, 2] / 2, canonical[:, 3] / 2
    along_w = torch.stack([cos * half_w, sin * half_w], dim=1)
    along_h = torch.stack([-sin * half_h, cos * half_h], dim=1)
    return torch.stack(
        [
            centre - along_w - along_h,
            centre + along_w - along_h,
            centre + along_w + along_h,
            centre - along_w + along_h,
        ],
        dim=1,
    )


def polygons_to_rboxes(polygons: Tensor) -> Tensor:
    """Fit canonical rotated boxes to quadrilaterals that are rotated rectangles.

    This is the load path for DOTA's eight-coordinate labels (WP-056), whose quads are
    hand-drawn and therefore only *approximately* rectangular. The fit is chosen so no
    single vertex decides an output: the centre is the vertex centroid, each extent is
    the mean of its pair of opposite side lengths, and the long-edge direction is the
    mean of that pair's two (antiparallel) edge vectors. On an exact rectangle all three
    reduce to the exact values, and a quad whose vertices sit within ``eps`` of a
    rectangle returns extents within ``eps`` of that rectangle's. The result is
    :func:`canonicalize`\\ d, so the corner order and winding of the input quad do not
    matter — any cyclic rotation or reversal of the same ring gives the same box.

    Args:
        polygons: ``(M, 4, 2)`` quadrilateral corners ``(x, y)``, each ring in order
            around its quad (either winding).

    Returns:
        ``(M, 5)`` canonical rotated boxes ``(cx, cy, w, h, theta)``.

    Raises:
        ValueError: If ``polygons`` is not a 3-D ``(M, 4, 2)`` tensor.

    Examples:
        ```pycon
        >>> import torch
        >>> quad = torch.tensor([[[3.0, 2.0], [7.0, 2.0], [7.0, 4.0], [3.0, 4.0]]])
        >>> [round(v, 4) for v in polygons_to_rboxes(quad)[0].tolist()]
        [5.0, 3.0, 4.0, 2.0, 0.0]

        ```
    """
    _check_polygons(polygons)
    edges = polygons.roll(-1, dims=1) - polygons
    lengths = edges.norm(dim=2)
    # Opposite sides of a rectangle are the (0, 2) and (1, 3) edge pairs under any cyclic
    # ordering, and are antiparallel -- hence the difference, not the sum, for direction.
    first_len = (lengths[:, 0] + lengths[:, 2]) / 2
    second_len = (lengths[:, 1] + lengths[:, 3]) / 2
    first_dir = (edges[:, 0] - edges[:, 2]) / 2
    theta = torch.atan2(first_dir[:, 1], first_dir[:, 0])
    # `canonicalize` swaps the pair round when the second one turned out to be longer,
    # which is why the first pair may be handed over as `w` unconditionally.
    return canonicalize(torch.stack([*polygons.mean(dim=1).unbind(dim=1), first_len, second_len, theta], dim=1))


def points_in_rboxes(points: Tensor, rboxes: Tensor) -> Tensor:
    """Test every point against every rotated box, edge-inclusive.

    Each point is expressed in each box's local frame — projected onto ``u`` and ``v``
    (module docstring, rotation note) — where containment is the pair of interval tests
    ``|local_x| <= w/2`` and ``|local_y| <= h/2``. This is the containment primitive
    WP-061 selects rotated TAL/STAL candidates with (A25), so it is a single broadcast
    expression over the ``(P, M)`` pair grid with no Python loop over boxes. Boxes are
    used as given: containment is invariant under the canonical moves, so no
    canonicalization is needed or performed.

    Args:
        points: ``(P, 2)`` query points ``(x, y)``.
        rboxes: ``(M, 5)`` rotated boxes ``(cx, cy, w, h, theta)``, canonical or not.

    Returns:
        ``(P, M)`` bool tensor, ``True`` where the point lies inside or exactly on the
        boundary of the box.

    Raises:
        ValueError: If ``points`` is not ``(P, 2)`` or ``rboxes`` is not ``(M, 5)``.

    Examples:
        ```pycon
        >>> import torch
        >>> box = torch.tensor([[0.0, 0.0, 10.0, 2.0, 0.7854]])  # 45 deg, long and thin
        >>> pts = torch.tensor([[0.0, 0.0], [3.0, -3.0]])  # centre; inside its envelope
        >>> points_in_rboxes(pts, box).tolist()
        [[True], [False]]

        ```
    """
    _check_2d(points, _POINT_DIM, "points")
    _check_2d(rboxes, _RBOX_DIM, "rboxes")
    delta = points[:, None, :] - rboxes[None, :, :2]
    cos, sin = torch.cos(rboxes[:, 4]), torch.sin(rboxes[:, 4])
    local_x = delta[..., 0] * cos + delta[..., 1] * sin
    local_y = delta[..., 1] * cos - delta[..., 0] * sin
    return (local_x.abs() <= rboxes[:, 2] / 2) & (local_y.abs() <= rboxes[:, 3] / 2)


def rotated_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Compute exact pairwise intersection-over-union between rotated boxes.

    Each box is expanded to its four corners by :func:`rboxes_to_polygons` — which
    canonicalizes first, so ``theta`` and ``theta + pi`` give identical overlaps — and
    every pair's intersection is obtained by clipping one quadrilateral against the
    other's four edges (Sutherland-Hodgman, valid because both are convex) and taking the
    shoelace area of what survives. No sampling, no Gaussian surrogate, no float64: see
    the module docstring on the per-pair midpoint shift that makes float32 sufficient.

    Conventions and degenerate cases:

    - Clipping is **edge-inclusive** (a vertex exactly on a clip edge counts as inside),
      matching :func:`points_in_rboxes` and WP-055's containment choice. Boxes sharing
      only an edge or a corner therefore intersect in a zero-area polygon and score
      exactly ``0.0`` either way — inclusivity changes which vertices are kept, never the
      area.
    - A box with zero or negative extent encloses no area. Its polygon has zero or
      reversed winding, both of which clamp to zero area, so every IoU involving it is
      ``0.0``. The union is guarded against division by zero, so a degenerate pair yields
      ``0.0`` rather than ``NaN`` — the same refusal :mod:`lucid_yolo.losses.probiou`
      makes.

    Args:
        boxes_a: ``(M, 5)`` rotated boxes ``(cx, cy, w, h, theta)``, canonical or not.
        boxes_b: ``(N, 5)`` rotated boxes in the same form.

    Returns:
        ``(M, N)`` IoU in ``[0, 1]``, in the dtype
        :func:`torch.result_type` gives the two inputs.

    Raises:
        ValueError: If either argument is not a 2-D five-column tensor. The message
            names the offending argument and spells the row count ``N``, the letter
            :func:`_check_2d` uses for every shape it rejects.

    Examples:
        ```pycon
        >>> import torch
        >>> box = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]])
        >>> float(rotated_iou(box, box))  # a box against itself
        1.0
        >>> shifted = torch.tensor([[2.0, 0.0, 4.0, 2.0, 0.0]])  # half its width along +x
        >>> round(float(rotated_iou(box, shifted)), 4)
        0.3333
        >>> touching = torch.tensor([[4.0, 0.0, 4.0, 2.0, 0.0]])  # shares one edge only
        >>> float(rotated_iou(box, touching))
        0.0
        >>> square = torch.tensor([[0.0, 0.0, 3.0, 3.0, 0.2]])
        >>> turned = torch.tensor([[0.0, 0.0, 3.0, 3.0, 0.2 + torch.pi / 2]])
        >>> round(float(rotated_iou(square, turned)), 5)  # the same square, folded
        1.0

        ```
    """
    _check_2d(boxes_a, _RBOX_DIM, "boxes_a")
    _check_2d(boxes_b, _RBOX_DIM, "boxes_b")
    dtype = torch.result_type(boxes_a, boxes_b)
    left, right = boxes_a.to(dtype), boxes_b.to(dtype)
    if left.shape[0] == 0 or right.shape[0] == 0:
        return torch.zeros((left.shape[0], right.shape[0]), dtype=dtype, device=left.device)

    # Translation invariance is what buys float32 the headroom float64 would otherwise be
    # needed for: each pair is re-centred on its own midpoint *before* its corners are
    # expanded, so no coordinate in the clipping arithmetic ever carries the absolute
    # offset. Re-centring after the expansion would leave the corners themselves rounded
    # at absolute scale, which is a cliff rather than a constant (see the module docstring).
    shape = (left.shape[0], right.shape[0], _RBOX_DIM)
    midpoint = (left[:, None, :2] + right[None, :, :2]) / 2
    subject = _local_polygons(left[:, None].expand(shape), midpoint)
    clip = _local_polygons(right[None, :].expand(shape), midpoint)

    area_a = _shoelace(subject).clamp_min(0)
    area_b = _shoelace(clip).clamp_min(0)
    intersection = _intersection_area(subject, clip)
    union = area_a + area_b - intersection
    tiny = torch.finfo(union.dtype).tiny
    return torch.where(union > tiny, intersection / union.clamp_min(tiny), torch.zeros_like(union)).clamp(0.0, 1.0)


def _wrap_theta(theta: Tensor) -> Tensor:
    """Shift angles by whole multiples of ``pi`` into ``[-pi/4, 3*pi/4)``.

    An angle already in range is passed through untouched rather than recomputed. That
    short-circuit is what makes :func:`canonicalize` exactly idempotent: at float32
    precision the round trip through :func:`torch.remainder` and back need not reproduce
    its own input near the range boundary, so the second call must not attempt it.

    Shifting by ``pi`` cannot land an out-of-range angle back in range at float32 when the
    input sits within an ulp of a boundary: an input one ulp below ``-pi/4`` has
    ``theta - _THETA_LOW`` round to a value whose remainder is ``pi``, so the sum comes
    back to the input itself and the second guard's subtraction returns it unchanged. The
    two guards therefore do not close the boundary case on their own, and the final clamp
    does. It moves such an angle by at most one ulp — under 1e-7 radians — which is the
    price of the range being a guarantee callers may rely on rather than a near-certainty.

    Examples:
        >>> import torch
        >>> [round(v, 4) for v in _wrap_theta(torch.tensor([3.5, -0.9, 0.3])).tolist()]
        [0.3584, 2.2416, 0.3]
    """
    in_range = (theta >= _THETA_LOW) & (theta < _THETA_HIGH)
    wrapped = torch.remainder(theta - _THETA_LOW, _PI) + _THETA_LOW
    wrapped = torch.where(wrapped < _THETA_LOW, wrapped + _PI, wrapped)
    wrapped = torch.where(wrapped >= _THETA_HIGH, wrapped - _PI, wrapped)
    low = torch.tensor(_THETA_LOW, dtype=wrapped.dtype, device=wrapped.device)
    wrapped = wrapped.clamp(min=low, max=_greatest_below_high(wrapped))
    return torch.where(in_range, theta, wrapped)


def _greatest_below_high(like: Tensor) -> Tensor:
    """Return the largest value below ``3*pi/4`` representable in ``like``'s dtype.

    Clamping to ``3*pi/4`` itself would leave the angle on the excluded end of the
    half-open range, so the clamp needs the neighbour below it in the working precision —
    which differs between float32 and float64 and cannot be written as one literal.

    Examples:
        >>> import torch
        >>> float(_greatest_below_high(torch.zeros(1))) < 3 * math.pi / 4
        True
    """
    high = torch.tensor(_THETA_HIGH, dtype=like.dtype, device=like.device)
    return torch.nextafter(high, torch.full_like(high, -math.inf))


def _check_2d(tensor: Tensor, columns: int, name: str) -> None:
    """Raise :class:`ValueError` unless ``tensor`` is 2-D with ``columns`` columns.

    The row count is spelled ``N`` in every message this raises, :func:`rotated_iou`'s
    two operands included. Their signature distinguishes ``M`` from ``N`` because the
    output is ``(M, N)`` — but the *predicate* each input has to satisfy is one and the
    same, so a second letter in the rejection named a difference the check never tested.

    Examples:
        >>> import torch
        >>> _check_2d(torch.zeros((0, 5)), 5, "rboxes")
        >>> _check_2d(torch.zeros((0, 5)), 5, "boxes_a")
    """
    if tensor.ndim != 2 or tensor.shape[1] != columns:
        raise ValueError(f"{name} must be (N, {columns}); got shape {tuple(tensor.shape)}")


def _check_polygons(polygons: Tensor) -> None:
    """Raise :class:`ValueError` unless ``polygons`` is a ``(M, 4, 2)`` quad tensor.

    Examples:
        >>> import torch
        >>> _check_polygons(torch.zeros((0, 4, 2)))
    """
    if polygons.ndim != 3 or tuple(polygons.shape[1:]) != (_QUAD_CORNERS, _POINT_DIM):
        raise ValueError(f"polygons must be (M, 4, 2); got shape {tuple(polygons.shape)}")


def _local_polygons(pairs: Tensor, midpoint: Tensor) -> Tensor:
    """Expand per-pair rotated boxes to corners in the frame centred on ``midpoint``.

    The subtraction happens on the **centre**, before :func:`rboxes_to_polygons` adds the
    half-extents, so the corner coordinates are born small instead of being made small
    afterwards. That is the whole precision story of :func:`rotated_iou`: a corner
    expanded at absolute DOTA scale is already rounded to that scale's float32 spacing,
    and no later shift recovers it.

    Args:
        pairs: ``(M, N, 5)`` rotated boxes, broadcast to the pair grid.
        midpoint: ``(M, N, 2)`` centre each pair is re-expressed about.

    Returns:
        ``(M, N, 4, 2)`` corner coordinates in the per-pair local frame.

    Examples:
        >>> import torch
        >>> box = torch.tensor([[[10.0, 10.0, 4.0, 2.0, 0.0]]])
        >>> _local_polygons(box, torch.tensor([[[10.0, 10.0]]]))[0, 0].tolist()
        [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]]
    """
    shifted = torch.cat([pairs[..., :2] - midpoint, pairs[..., 2:]], dim=-1)
    corners = rboxes_to_polygons(shifted.reshape(-1, _RBOX_DIM))
    return corners.reshape(*pairs.shape[:2], _QUAD_CORNERS, corners.shape[-1])


def _intersection_area(subject: Tensor, clip: Tensor) -> Tensor:
    """Clip ``subject`` against every edge of ``clip`` and return the surviving area.

    One Sutherland-Hodgman pass per clip edge, each followed by a compaction back to
    :data:`_MAX_INTERSECTION_CORNERS` slots — which loses nothing, since an intermediate
    result cannot exceed that bound (see the constant's note).

    Args:
        subject: ``(M, N, 4, 2)`` corners of the clipped quadrilateral, per pair.
        clip: ``(M, N, 4, 2)`` corners of the clipping quadrilateral, per pair.

    Returns:
        ``(M, N)`` intersection area, never negative.

    Examples:
        >>> import torch
        >>> unit = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        >>> pair = unit[None, None]
        >>> float(_intersection_area(pair, pair))
        4.0
    """
    polygon = subject
    valid = torch.ones(subject.shape[:-1], dtype=torch.bool, device=subject.device)
    for corner in range(_QUAD_CORNERS):
        start = clip[..., corner, :]
        end = clip[..., (corner + 1) % _QUAD_CORNERS, :]
        polygon, valid = _clip_by_edge(polygon, valid, start, end)
        polygon, valid = _compact(polygon, valid)
    enclosed = valid.sum(dim=-1) >= _MIN_AREA_CORNERS
    return torch.where(enclosed, _shoelace(polygon), torch.zeros_like(enclosed, dtype=polygon.dtype)).clamp_min(0)


def _clip_by_edge(polygon: Tensor, valid: Tensor, start: Tensor, end: Tensor) -> tuple[Tensor, Tensor]:
    """Run one Sutherland-Hodgman step against the directed edge ``start -> end``.

    Emits two slots per input vertex — the vertex itself when it is inside, and the
    crossing point when the edge to its successor changes side — so the output shape is a
    pure function of the input shape and no data-dependent resize is needed. The interior
    test is ``cross >= 0``, which is **edge-inclusive** and matches the positive winding
    :func:`rboxes_to_polygons` guarantees.

    Args:
        polygon: ``(M, N, K, 2)`` vertices, valid ones compacted to the front.
        valid: ``(M, N, K)`` prefix mask of live vertices.
        start: ``(M, N, 2)`` first endpoint of the clip edge.
        end: ``(M, N, 2)`` second endpoint of the clip edge.

    Returns:
        ``(M, N, 2K, 2)`` vertices and their ``(M, N, 2K)`` validity mask.

    Examples:
        >>> import torch
        >>> square = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])[None, None]
        >>> live = torch.ones(square.shape[:-1], dtype=torch.bool)
        >>> _, mask = _clip_by_edge(square, live, torch.zeros(1, 1, 2), torch.tensor([[[1.0, 0.0]]]))
        >>> int(mask.sum())  # the whole square lies on the inside of the x axis
        4
    """
    edge = (end - start)[..., None, :]
    offset = polygon - start[..., None, :]
    distance = edge[..., 0] * offset[..., 1] - edge[..., 1] * offset[..., 0]
    successor = _successor(polygon, valid)
    next_distance = _successor(distance[..., None], valid)[..., 0]

    inside = distance >= 0
    crossing = inside != (next_distance >= 0)
    denominator = distance - next_distance
    step = distance / torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    crossed = polygon + step[..., None] * (successor - polygon)

    vertices = torch.stack([polygon, crossed], dim=-2).flatten(-3, -2)
    kept = torch.stack([valid & inside, valid & crossing], dim=-1).flatten(-2, -1)
    return vertices, kept


def _successor(values: Tensor, valid: Tensor) -> Tensor:
    """Return each slot's cyclic successor among the live vertices.

    ``valid`` is a prefix mask, so the successor of slot ``i`` is slot ``i + 1`` when that
    slot is live and slot ``0`` otherwise — the ring closes at the first vertex rather
    than wandering into the padding.

    Args:
        values: ``(M, N, K, C)`` per-vertex values.
        valid: ``(M, N, K)`` prefix mask of live vertices.

    Returns:
        ``(M, N, K, C)`` values of each slot's successor.

    Examples:
        >>> import torch
        >>> values = torch.tensor([[[[1.0], [2.0], [3.0]]]])
        >>> mask = torch.tensor([[[True, True, False]]])
        >>> _successor(values, mask).flatten().tolist()  # slot 1 wraps to slot 0
        [2.0, 1.0, 1.0]
    """
    rolled = values.roll(-1, dims=-2)
    rolled_valid = valid.roll(-1, dims=-1)
    return torch.where(rolled_valid[..., None], rolled, values[..., :1, :])


def _compact(polygon: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Move live vertices to the front, truncate to the vertex bound, and pad with vertex 0.

    Restores the prefix-mask invariant :func:`_successor` relies on, and replaces every
    dead slot with the first live vertex so :func:`_shoelace` may run mask-free: the
    padding edges are zero-length and contribute nothing to the area.

    Args:
        polygon: ``(M, N, K, 2)`` vertices in emission order.
        valid: ``(M, N, K)`` mask of live vertices, in any arrangement.

    Returns:
        ``(M, N, 8, 2)`` compacted vertices and their ``(M, N, 8)`` prefix mask.

    Examples:
        >>> import torch
        >>> pts = torch.tensor([[[[9.0, 9.0], [1.0, 1.0], [2.0, 2.0]]]])
        >>> mask = torch.tensor([[[False, True, True]]])
        >>> kept, live = _compact(pts, mask)
        >>> kept[0, 0, :3].tolist(), live[0, 0, :3].tolist()
        ([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]], [True, True, False])
    """
    order = torch.argsort(valid.logical_not().to(torch.uint8), dim=-1, stable=True)[..., :_MAX_INTERSECTION_CORNERS]
    gathered = polygon.gather(-2, order[..., None].expand(*order.shape, polygon.shape[-1]))
    kept = valid.gather(-1, order)
    return torch.where(kept[..., None], gathered, gathered[..., :1, :]), kept


def _shoelace(polygon: Tensor) -> Tensor:
    """Return the signed area of each polygon by the shoelace formula.

    Positive for the winding :func:`rboxes_to_polygons` emits. Repeated vertices
    contribute zero, which is what lets padded rings be measured without a mask.

    Args:
        polygon: ``(..., K, 2)`` vertices in ring order.

    Returns:
        ``(...)`` signed area.

    Examples:
        >>> import torch
        >>> square = torch.tensor([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]])
        >>> float(_shoelace(square))
        6.0
    """
    successor = polygon.roll(-1, dims=-2)
    cross = polygon[..., 0] * successor[..., 1] - successor[..., 0] * polygon[..., 1]
    return 0.5 * cross.sum(dim=-1)
