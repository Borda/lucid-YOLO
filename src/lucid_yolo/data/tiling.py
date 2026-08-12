# SPDX-License-Identifier: Apache-2.0
"""Overlapping 1024 px crop tiling for the oriented path (WP-057, A21).

DOTA images run to several thousand pixels a side, and R18 sec. 4 states the reason
plainly: "Images in DOTA are so large that they cannot be directly sent to CNN-based
detectors. Therefore, we crop a series of 1024x1024 patches from the original images with
a stride set to 512." R1 restates only the shape of the step — "We split the images into
overlapping 1024x1024 crops" — without an overlap. This module is the geometry of that
split and of the target re-mapping it forces; the tiles themselves are a **build
artifact** (AGENTS.md sec. 3), so nothing here writes to disk on its own.

The overlap (A21):
    ``overlap`` is a parameter, defaulting to the 200 px A21 records. That default is
    **not attested by either source A21 cites**: R18's own published protocol is stride
    512 on a 1024 patch, i.e. a 512 px overlap, and R13 (MMRotate, arXiv:2204.13317)
    fixes the long-edge-135 angle definition but names no crop overlap at all. The
    register row says so, and the value is a sensitivity item for the OBB tier — which is
    why it is a parameter here rather than a constant, and why ``CROP_OVERLAP`` is
    exported so a sweep can name what it is varying.

Window placement:
    Windows are laid down at ``patch - overlap`` stride and the last window of each axis
    is pulled **flush against the right/bottom edge** instead of running past it, so a
    trailing window may overlap its predecessor by more than the nominal amount. An axis
    shorter than the patch yields a single window of the axis's own length. The
    consequence relied on by ``test_coverage_no_gaps``: every pixel of the source lies in
    at least one window, for any size, patch and overlap. :func:`tile_windows` is a pure
    function of ``(image size, patch, overlap)`` and touches no pixels, so the placement
    can be tested without images.

The 0.7 rule, and where difficult instances come from (A39):
    R18 continues: "we denote the area of the original object as A_o, and the area of
    divided parts P_i, (i=1,2) as a_i... Then we compute the parts areas over the original
    object area: U_i = a_i / A_o. Finally, we label the part P_i with U_i < 0.7 as
    *difficult* and for the other one, we keep it the same as the original annotation."
    So the threshold **flags, it does not drop**: an instance is dropped only when nothing
    of it lands in the window (``U == 0``). Tiling is therefore the step that *creates*
    difficult instances — a label file may contain none and its tiles still will — which
    is what makes A39 (the deferral of what ``difficult`` means for training and
    evaluation, WP-088 and WP-063) live here rather than only at load. Incoming flags are
    never cleared: a part of an already-difficult object stays difficult whatever its
    ``U``.

The fit:
    "For the vertices of the newly generated parts, we need to ensure they can be
    described as an oriented bounding box with 4 vertices in the clockwise order with a
    fitting method." The clipped region is a convex polygon of up to eight vertices; the
    fit used here keeps the **original object's orientation** and takes the tight extents
    of the clipped vertices in that box's own ``(u, v)`` frame. It is exact when the
    object is wholly inside (it reproduces the original rectangle), it never invents an
    orientation the annotation did not carry, and its four corners go through
    :func:`~lucid_yolo.data.rotated_geom.polygons_to_rboxes`, so the emitted box is
    canonical long-edge form with R18's winding. A part at or above the threshold is not
    re-fitted at all: R18 keeps *the original annotation* there, so it is translated and
    otherwise untouched — bar a :func:`~lucid_yolo.data.rotated_geom.canonicalize` pass
    that is bit-for-bit identity on WP-056's output and makes "every emitted box is
    canonical" a property of this module rather than of its caller.

Axis-aligned boxes:
    ``boxes[i]`` stays the axis-aligned envelope of the geometry ``rboxes[i]`` describes,
    per WP-056. Above the threshold that is the original ``boxes[i]`` translated (the
    original annotation, envelope of the hand-drawn quad, which may hang outside the
    window exactly as WP-056 leaves it unclipped); below it, it is the envelope of the
    **clipped polygon** — the tight axis-aligned reading of the part that actually landed
    in the tile, and for an axis-aligned object exactly the intersection of its box with
    the window.

Instance axis:
    WP-056's invariant is preserved end to end: ``boxes``, ``labels`` and ``rboxes`` stay
    1:1 in the emitted :class:`~lucid_yolo.data.targets.Targets`, and
    :class:`TiledTargets` carries ``difficult`` and ``visible_fraction`` on that same
    axis. ``polygons`` stays empty, and a caller that supplies polygons is rejected rather
    than having a modality silently dropped.

Precision and shape of the code:
    The clipping (Sutherland-Hodgman against the four window edges — sufficient because
    the window is a convex axis-aligned rectangle) and the shoelace areas are computed in
    **float64** and cast back to float32 on the way out, following
    :class:`~lucid_yolo.data.letterbox.Letterbox`'s float64 affine: DOTA coordinates reach
    10^4 px, where a float32 shoelace difference loses the precision the ``U`` threshold
    is compared at. Unlike :mod:`lucid_yolo.data.rotated_geom`, this module does loop over
    instances in Python — clipping produces a different vertex count per instance, and
    this is offline build-time code, not the assigner's inner loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lucid_yolo.data.rotated_geom import canonicalize, polygons_to_rboxes, rboxes_to_polygons
from lucid_yolo.data.targets import Targets

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "CROP_OVERLAP",
    "DIFFICULT_AREA_FRACTION",
    "PATCH_SIZE",
    "TiledTargets",
    "crop_image",
    "crop_targets",
    "tile_image_targets",
    "tile_windows",
]

#: Crop side in pixels: R18 sec. 4 and R1 sec. 4.4 both name 1024.
PATCH_SIZE = 1024
#: Default crop overlap in pixels (A21). Not attested by R13 or R18 — see the module
#: docstring; R18's own protocol is stride 512, i.e. a 512 px overlap.
CROP_OVERLAP = 200
#: R18's ``U_i`` threshold: a part below this fraction of the original area is *difficult*.
DIFFICULT_AREA_FRACTION = 0.7

#: Column count of a window ``(x0, y0, x1, y1)``.
_WINDOW_DIM = 4
#: Column count of an ``xyxy`` axis-aligned box.
_BOX_DIM = 4
#: Corner count of the quadrilateral form of a rotated box.
_QUAD_CORNERS = 4
#: Minimum vertex count for a polygon to enclose any area.
_MIN_AREA_CORNERS = 3


@dataclass(frozen=True)
class TiledTargets:
    """One window's targets, with the R18 bookkeeping the crop produced.

    Attributes:
        targets: The window-local targets, ``boxes``/``labels``/``rboxes`` on one instance
            axis (WP-056) and ``polygons`` empty.
        difficult: ``(N,)`` bool, the R18 flag after the crop — the incoming flag OR-ed
            with ``visible_fraction < 0.7`` (A39: carried, not acted on).
        visible_fraction: ``(N,)`` float32, R18's ``U_i`` — clipped area over original
            area. Kept because the fitted box's area is not the clipped area, so the
            quantity the rule turns on is otherwise unrecoverable downstream.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> empty = TiledTargets(Targets.empty(), torch.zeros(0, dtype=torch.bool), torch.zeros(0))
        >>> empty.difficult.shape
        torch.Size([0])

        ```
    """

    targets: Targets
    difficult: Tensor
    visible_fraction: Tensor

    def __post_init__(self) -> None:
        """Validate that every per-instance field shares one axis with the targets.

        Raises:
            TypeError: If ``difficult`` is not bool or ``visible_fraction`` not float32.
            ValueError: If any field's length disagrees with the instance axis.
        """
        count = self.targets.boxes.shape[0]
        if self.targets.rboxes.shape[0] != count:
            raise ValueError(
                f"boxes/rboxes length mismatch: {count} boxes, {self.targets.rboxes.shape[0]} rboxes; "
                "the oriented path keeps them 1:1"
            )
        if self.difficult.dtype != torch.bool:
            raise TypeError(f"difficult must be bool; got {self.difficult.dtype}")
        if self.visible_fraction.dtype != torch.float32:
            raise TypeError(f"visible_fraction must be float32; got {self.visible_fraction.dtype}")
        for name, tensor in (("difficult", self.difficult), ("visible_fraction", self.visible_fraction)):
            if tensor.ndim != 1 or tensor.shape[0] != count:
                raise ValueError(f"{name} must be 1-D of length {count}; got shape {tuple(tensor.shape)}")


def tile_windows(image_size: tuple[int, int], patch: int = PATCH_SIZE, overlap: int = CROP_OVERLAP) -> Tensor:
    """Lay overlapping crop windows over an image of arbitrary size.

    Windows advance by ``patch - overlap`` and the last one on each axis is placed flush
    against the far edge, so the whole image is covered with no gap and the trailing
    window may overlap its predecessor by more than ``overlap``. An axis shorter than
    ``patch`` yields one window of that axis's length, so a small image still tiles.

    Args:
        image_size: Source ``(height, width)`` in pixels, both positive.
        patch: Crop side in pixels. Defaults to 1024 (R18 sec. 4, R1 sec. 4.4).
        overlap: Nominal overlap between neighbouring windows, ``0 <= overlap < patch``.
            Defaults to 200 (A21) — see the module docstring on why that is a parameter.

    Returns:
        ``(K, 4)`` int64 windows ``(x0, y0, x1, y1)``, row-major (all windows of the top
        row first). ``x1``/``y1`` are exclusive, as slice bounds.

    Raises:
        ValueError: If either image dimension is not positive, ``patch`` is not positive,
            or ``overlap`` is negative or not smaller than ``patch``.

    Examples:
        ```pycon
        >>> tile_windows((10, 10), patch=6, overlap=2).tolist()
        [[0, 0, 6, 6], [4, 0, 10, 6], [0, 4, 6, 10], [4, 4, 10, 10]]
        >>> tile_windows((3, 5), patch=4, overlap=1).tolist()  # short axis; flush edge
        [[0, 0, 4, 3], [1, 0, 5, 3]]

        ```
    """
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive (height, width); got {(height, width)}")
    if patch <= 0:
        raise ValueError(f"patch must be positive; got {patch}")
    if not 0 <= overlap < patch:
        raise ValueError(f"overlap must satisfy 0 <= overlap < patch={patch}; got {overlap}")
    tile_h, tile_w = min(patch, height), min(patch, width)
    rows, columns = _axis_starts(height, patch, overlap), _axis_starts(width, patch, overlap)
    windows = [[x, y, x + tile_w, y + tile_h] for y in rows for x in columns]
    return torch.tensor(windows, dtype=torch.int64)


def crop_image(image: Tensor, window: Tensor) -> Tensor:
    """Cut one window out of a CHW image.

    The result is a **view** into ``image``, not a copy: the caller decides whether the
    tile outlives the source. Nothing is written to disk here — the derived tile set is a
    build artifact (AGENTS.md sec. 3), so emitting it is the caller's choice of directory.

    Args:
        image: ``(C, H, W)`` image tensor.
        window: ``(4,)`` window ``(x0, y0, x1, y1)`` with exclusive far bounds, e.g. a row
            of :func:`tile_windows`.

    Returns:
        The ``(C, y1 - y0, x1 - x0)`` crop.

    Raises:
        ValueError: If ``image`` is not 3-D, or ``window`` is malformed.

    Examples:
        ```pycon
        >>> import torch
        >>> image = torch.arange(12.0).reshape(1, 3, 4)
        >>> crop_image(image, torch.tensor([1, 0, 3, 2]))
        tensor([[[1., 2.],
                 [5., 6.]]])

        ```
    """
    if image.ndim != 3:
        raise ValueError(f"image must be (C, H, W); got shape {tuple(image.shape)}")
    x0, y0, x1, y1 = _window_bounds(window)
    return image[..., int(y0) : int(y1), int(x0) : int(x1)]


def crop_targets(targets: Targets, window: Tensor, *, difficult: Tensor) -> TiledTargets:
    """Re-map one image's targets into one window, applying R18's 0.7 rule.

    Each rotated box is expanded to its corners, clipped to the window, and scored by
    ``U = clipped area / original area``. ``U == 0`` drops the instance; ``U >= 0.7``
    keeps the original annotation, translated into window-local pixels; ``U < 0.7`` keeps
    the instance too but flags it *difficult* and re-fits the clipped region to a canonical
    long-edge box at the original orientation (module docstring, "The fit"). Flags already
    set on input survive regardless of ``U``.

    ``difficult`` has no default, for the same reason ``keep_difficult`` has none in
    :mod:`lucid_yolo.data.dota` (A39): the flag's fate is undecided, so it is carried
    explicitly rather than defaulted away.

    Args:
        targets: One image's targets in source-image pixels, with ``rboxes`` 1:1 with
            ``boxes`` and ``labels`` (WP-056) and no polygons.
        window: ``(4,)`` window ``(x0, y0, x1, y1)``, far bounds exclusive.
        difficult: ``(N,)`` bool incoming R18 flags, one per instance. Required.

    Returns:
        The window-local :class:`TiledTargets`; empty (but correctly shaped) when no
        instance intersects the window.

    Raises:
        TypeError: If ``difficult`` is not a bool tensor.
        ValueError: If ``window`` is malformed, ``targets`` carries polygons or a
            ``rboxes`` axis that does not match its instance axis, or ``difficult`` has
            the wrong length.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> t = Targets(
        ...     boxes=torch.tensor([[3.0, 0.0, 7.0, 2.0]]),
        ...     labels=torch.tensor([1]),
        ...     rboxes=torch.tensor([[5.0, 1.0, 4.0, 2.0, 0.0]]),
        ... )
        >>> tiled = crop_targets(t, torch.tensor([0, 0, 5, 4]), difficult=torch.tensor([False]))
        >>> [round(v, 4) for v in tiled.targets.rboxes[0].tolist()]  # half of it survived
        [4.0, 1.0, 2.0, 2.0, 0.0]
        >>> tiled.difficult.tolist(), round(tiled.visible_fraction.item(), 4)
        ([True], 0.5)

        ```
    """
    _validate_inputs(targets, difficult)
    x0, y0, x1, y1 = _window_bounds(window)
    origin = torch.tensor([x0, y0], dtype=torch.float64)
    rboxes = targets.rboxes.to(torch.float64)
    polygons = rboxes_to_polygons(targets.rboxes).to(torch.float64)
    parts: list[_Part] = []
    for index in range(rboxes.shape[0]):
        clipped = _clip_to_window(polygons[index], x0, y0, x1, y1)
        fraction = _visible_fraction(clipped, rboxes[index])
        if fraction <= 0.0:
            continue
        rbox, box = _fit_part(clipped, rboxes[index], targets.boxes[index], origin, fraction)
        parts.append(
            _Part(
                rbox=rbox,
                box=box,
                label=int(targets.labels[index]),
                difficult=bool(difficult[index]) or fraction < DIFFICULT_AREA_FRACTION,
                fraction=fraction,
            )
        )
    return _assemble(parts)


def tile_image_targets(
    image: Tensor,
    targets: Targets,
    *,
    difficult: Tensor,
    patch: int = PATCH_SIZE,
    overlap: int = CROP_OVERLAP,
) -> Iterator[tuple[Tensor, Tensor, TiledTargets]]:
    """Iterate the tiles of one image together with their re-mapped targets.

    A thin composition of :func:`tile_windows`, :func:`crop_image` and
    :func:`crop_targets`, in window order. It deliberately yields rather than returns a
    list, and writes nothing: where the derived tiles land is the caller's decision
    (AGENTS.md sec. 3 — never the repository).

    Args:
        image: ``(C, H, W)`` source image.
        targets: The source image's targets, in source-image pixels.
        difficult: ``(N,)`` bool incoming R18 flags, one per instance. Required (A39).
        patch: Crop side in pixels. Defaults to 1024.
        overlap: Nominal overlap in pixels. Defaults to 200 (A21).

    Yields:
        ``(window, tile, tiled_targets)`` per window: the ``(4,)`` int64 window, the image
        crop (a view) and its :class:`TiledTargets`.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> image = torch.zeros(3, 8, 8)
        >>> t = Targets(
        ...     boxes=torch.tensor([[1.0, 1.0, 3.0, 3.0]]),
        ...     labels=torch.tensor([0]),
        ...     rboxes=torch.tensor([[2.0, 2.0, 2.0, 2.0, 0.0]]),
        ... )
        >>> tiles = list(tile_image_targets(image, t, difficult=torch.tensor([False]), patch=6, overlap=2))
        >>> len(tiles), tiles[0][1].shape
        (4, torch.Size([3, 6, 6]))
        >>> tiles[0][2].visible_fraction.tolist()  # whole object in the first window
        [1.0]
        >>> tiles[3][2].visible_fraction.tolist(), tiles[3][2].difficult.tolist()  # a corner of it
        ([0.25], [True])

        ```
    """
    if image.ndim != 3:
        raise ValueError(f"image must be (C, H, W); got shape {tuple(image.shape)}")
    _, height, width = image.shape
    for window in tile_windows((height, width), patch=patch, overlap=overlap):
        yield window, crop_image(image, window), crop_targets(targets, window, difficult=difficult)


@dataclass(frozen=True)
class _Part:
    """One instance's contribution to one window, before assembly.

    Attributes:
        rbox: ``(5,)`` float32 window-local canonical rotated box.
        box: ``(4,)`` float32 window-local ``xyxy`` envelope.
        label: Class id, carried through unchanged.
        difficult: The R18 flag after the crop.
        fraction: R18's ``U_i`` for this part.

    Examples:
        >>> import torch
        >>> _Part(torch.zeros(5), torch.zeros(4), label=3, difficult=True, fraction=0.5).label
        3
    """

    rbox: Tensor
    box: Tensor
    label: int
    difficult: bool
    fraction: float


def _assemble(parts: list[_Part]) -> TiledTargets:
    """Stack per-instance parts into one window's :class:`TiledTargets`.

    Args:
        parts: The surviving parts, in source instance order.

    Returns:
        The window's targets; correctly-shaped empties when ``parts`` is empty.

    Examples:
        >>> _assemble([]).targets.rboxes.shape
        torch.Size([0, 5])
    """
    if not parts:
        return TiledTargets(
            targets=Targets.empty(),
            difficult=torch.zeros((0,), dtype=torch.bool),
            visible_fraction=torch.zeros((0,), dtype=torch.float32),
        )
    # The flag is written into *both* places on purpose: `TiledTargets.difficult` is what
    # WP-063's adapter reads, while `Targets.difficult` (WP-088) is what travels through a
    # loader. Filling one and defaulting the other would leave two answers to one question.
    difficult = torch.tensor([part.difficult for part in parts], dtype=torch.bool)
    targets = Targets(
        boxes=torch.stack([part.box for part in parts], dim=0),
        labels=torch.tensor([part.label for part in parts], dtype=torch.int64),
        rboxes=torch.stack([part.rbox for part in parts], dim=0),
        difficult=difficult,
    )
    return TiledTargets(
        targets=targets,
        difficult=difficult,
        visible_fraction=torch.tensor([part.fraction for part in parts], dtype=torch.float32),
    )


def _fit_part(clipped: Tensor, rbox: Tensor, box: Tensor, origin: Tensor, fraction: float) -> tuple[Tensor, Tensor]:
    """Produce one part's window-local rotated box and axis-aligned envelope.

    At or above the threshold R18 keeps the original annotation, so both are translated
    and otherwise untouched. Below it the part is newly generated: the rotated box is the
    tight fit of the clipped vertices in the original box's frame, and the envelope is the
    envelope of the clipped polygon itself.

    Args:
        clipped: ``(Q, 2)`` float64 clipped polygon, in source-image pixels.
        rbox: ``(5,)`` float64 original rotated box.
        box: ``(4,)`` float32 original axis-aligned envelope.
        origin: ``(2,)`` float64 window origin ``(x0, y0)``.
        fraction: R18's ``U_i`` for this part.

    Returns:
        The ``(5,)`` rotated box and ``(4,)`` ``xyxy`` envelope, both float32 and
        window-local.

    Examples:
        >>> import torch
        >>> rb = torch.tensor([5.0, 1.0, 4.0, 2.0, 0.0], dtype=torch.float64)
        >>> quad = torch.tensor([[3.0, 0.0], [7.0, 0.0], [7.0, 2.0], [3.0, 2.0]], dtype=torch.float64)
        >>> whole = _fit_part(quad, rb, torch.tensor([3.0, 0.0, 7.0, 2.0]), torch.zeros(2, dtype=torch.float64), 1.0)
        >>> whole[0].tolist()
        [5.0, 1.0, 4.0, 2.0, 0.0]
    """
    if fraction >= DIFFICULT_AREA_FRACTION:
        kept = rbox.clone()
        kept[:2] -= origin
        # `canonicalize` is exactly idempotent, so this is a bit-for-bit no-op on the
        # canonical boxes WP-056 emits; it is here so the module's own output is
        # canonical whatever a caller hands in, not because translation could break it.
        return canonicalize(kept.to(torch.float32).unsqueeze(0))[0], (box.to(torch.float64) - origin.repeat(2)).to(
            torch.float32
        )
    corners = _fit_corners(clipped, rbox) - origin
    fitted = polygons_to_rboxes(corners.to(torch.float32).unsqueeze(0))[0]
    local = clipped - origin
    envelope = torch.cat([local.min(dim=0).values, local.max(dim=0).values])
    return fitted, envelope.to(torch.float32)


def _fit_corners(ring: Tensor, rbox: Tensor) -> Tensor:
    """Fit a rectangle at ``rbox``'s orientation tightly around ``ring``'s vertices.

    R18 asks only that the new part be describable "as an oriented bounding box with 4
    vertices"; keeping the original orientation and taking the extents of the clipped
    vertices in that frame is the fit that invents nothing and reproduces the original
    rectangle exactly when the part is the whole object.

    Args:
        ring: ``(Q, 2)`` float64 polygon vertices.
        rbox: ``(5,)`` float64 rotated box supplying the centre and orientation.

    Returns:
        ``(4, 2)`` float64 corners of the fitted rectangle, in source-image pixels.

    Examples:
        >>> import torch
        >>> rb = torch.tensor([0.0, 0.0, 4.0, 2.0, 0.0], dtype=torch.float64)
        >>> half = torch.tensor([[-2.0, -1.0], [0.0, -1.0], [0.0, 1.0], [-2.0, 1.0]], dtype=torch.float64)
        >>> _fit_corners(half, rb).tolist()
        [[-2.0, -1.0], [0.0, -1.0], [0.0, 1.0], [-2.0, 1.0]]
    """
    cos, sin = torch.cos(rbox[4]), torch.sin(rbox[4])
    along_w = torch.stack([cos, sin])
    along_h = torch.stack([-sin, cos])
    delta = ring - rbox[:2]
    local_w, local_h = delta @ along_w, delta @ along_h
    low_w, high_w = local_w.min(), local_w.max()
    low_h, high_h = local_h.min(), local_h.max()
    local = torch.stack(
        [
            torch.stack([low_w, low_h]),
            torch.stack([high_w, low_h]),
            torch.stack([high_w, high_h]),
            torch.stack([low_w, high_h]),
        ]
    )
    return rbox[:2] + local[:, :1] * along_w + local[:, 1:] * along_h


def _visible_fraction(clipped: Tensor, rbox: Tensor) -> float:
    """Compute R18's ``U = clipped area / original area`` for one instance.

    Args:
        clipped: ``(Q, 2)`` float64 clipped polygon; fewer than three vertices means no
            area survived.
        rbox: ``(5,)`` float64 original rotated box, whose area is ``w * h``.

    Returns:
        The visible fraction; ``0.0`` when nothing survives or the annotation is
        degenerate (zero-area boxes cannot be scored and are dropped).

    Examples:
        >>> import torch
        >>> rb = torch.tensor([0.0, 0.0, 4.0, 2.0, 0.0], dtype=torch.float64)
        >>> half = torch.tensor([[-2.0, -1.0], [0.0, -1.0], [0.0, 1.0], [-2.0, 1.0]], dtype=torch.float64)
        >>> _visible_fraction(half, rb)
        0.5
    """
    original = float(rbox[2] * rbox[3])
    if original <= 0.0:
        return 0.0
    return float(_polygon_area(clipped)) / original


def _polygon_area(ring: Tensor) -> Tensor:
    """Return the shoelace area of a ring, winding-agnostic.

    Args:
        ring: ``(Q, 2)`` polygon vertices in order around the ring.

    Returns:
        The scalar absolute area; zero for a ring too short to enclose any.

    Examples:
        >>> import torch
        >>> _polygon_area(torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]])).item()
        6.0
    """
    if ring.shape[0] < _MIN_AREA_CORNERS:
        return ring.new_zeros(())
    following = ring.roll(-1, dims=0)
    return 0.5 * (ring[:, 0] * following[:, 1] - following[:, 0] * ring[:, 1]).sum().abs()


def _clip_to_window(ring: Tensor, x0: float, y0: float, x1: float, y1: float) -> Tensor:
    """Clip a convex ring to an axis-aligned window (Sutherland-Hodgman).

    The window is convex, so clipping against its four edges in turn is exact and needs no
    general polygon-intersection machinery — and no shapely, which is a dev-only
    dependency and never enters ``src/``.

    Args:
        ring: ``(Q, 2)`` float64 convex polygon vertices in order.
        x0: Window left bound.
        y0: Window top bound.
        x1: Window right bound.
        y1: Window bottom bound.

    Returns:
        The clipped ring, ``(0, 2)`` when the polygon lies wholly outside.

    Examples:
        >>> import torch
        >>> quad = torch.tensor([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
        >>> _clip_to_window(quad, 2.0, 0.0, 6.0, 4.0).tolist()
        [[2.0, 0.0], [4.0, 0.0], [4.0, 4.0], [2.0, 4.0]]
    """
    for axis, bound, keep_upper in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        if ring.shape[0] == 0:
            break
        ring = _clip_halfplane(ring, axis, bound, keep_upper)
    return ring


def _clip_halfplane(ring: Tensor, axis: int, bound: float, keep_upper: bool) -> Tensor:
    """Clip a ring to one half-plane, keeping vertex order.

    Each vertex contributes itself when inside, followed by the edge's intersection with
    the bound when that edge crosses it — the Sutherland-Hodgman step, written as one
    masked gather over an interleaved ``(2Q, 2)`` candidate array so the order falls out
    of the layout instead of an append loop.

    Args:
        ring: ``(Q, 2)`` polygon vertices in order.
        axis: ``0`` for a vertical bound (``x``), ``1`` for a horizontal one (``y``).
        bound: The coordinate of the clipping line.
        keep_upper: Keep the ``coord >= bound`` side when ``True``, ``coord <= bound``
            otherwise.

    Returns:
        The clipped ring.

    Examples:
        >>> import torch
        >>> square = torch.tensor([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
        >>> _clip_halfplane(square, axis=0, bound=2.0, keep_upper=False).tolist()
        [[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [0.0, 4.0]]
    """
    coordinate = ring[:, axis]
    inside = coordinate >= bound if keep_upper else coordinate <= bound
    following = ring.roll(-1, dims=0)
    crossing = inside != inside.roll(-1, dims=0)
    delta = following[:, axis] - coordinate
    # Only crossing edges are gathered, and a crossing edge has a non-zero delta; the
    # guard keeps the unused divisions finite rather than seeding NaNs into the stack.
    safe = torch.where(delta == 0, torch.ones_like(delta), delta)
    ratio = ((bound - coordinate) / safe).unsqueeze(1)
    intersections = ring + ratio * (following - ring)
    candidates = torch.stack([ring, intersections], dim=1).reshape(-1, 2)
    return candidates[torch.stack([inside, crossing], dim=1).reshape(-1)]


def _axis_starts(length: int, patch: int, overlap: int) -> list[int]:
    """Place window starts along one axis, the last flush against the far edge.

    Args:
        length: Axis length in pixels.
        patch: Crop side in pixels.
        overlap: Nominal overlap in pixels.

    Returns:
        Ascending window start coordinates; ``[0]`` when the axis is no longer than the
        patch.

    Examples:
        >>> _axis_starts(10, 6, 2), _axis_starts(11, 6, 2), _axis_starts(4, 6, 2)
        ([0, 4], [0, 4, 5], [0])
    """
    if length <= patch:
        return [0]
    stride = patch - overlap
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] + patch < length:
        starts.append(length - patch)
    return starts


def _window_bounds(window: Tensor) -> tuple[float, float, float, float]:
    """Unpack and validate a window tensor into ``(x0, y0, x1, y1)`` floats.

    Args:
        window: ``(4,)`` window ``(x0, y0, x1, y1)``, far bounds exclusive.

    Returns:
        The four bounds as Python floats.

    Raises:
        ValueError: If the shape is wrong or either extent is not positive.

    Examples:
        >>> import torch
        >>> _window_bounds(torch.tensor([0, 1, 4, 5]))
        (0.0, 1.0, 4.0, 5.0)
    """
    if window.ndim != 1 or window.shape[0] != _WINDOW_DIM:
        raise ValueError(f"window must be (4,) as (x0, y0, x1, y1); got shape {tuple(window.shape)}")
    x0, y0, x1, y1 = (float(value) for value in window.tolist())
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"window must have positive extent; got {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def _validate_inputs(targets: Targets, difficult: Tensor) -> None:
    """Check the instance-axis invariant and the ``difficult`` companion tensor.

    Args:
        targets: The source targets.
        difficult: The incoming R18 flags.

    Raises:
        TypeError: If ``difficult`` is not a bool tensor.
        ValueError: If polygons are present, ``rboxes`` does not share the instance axis,
            or ``difficult`` has the wrong shape.

    Examples:
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> _validate_inputs(Targets.empty(), torch.zeros(0, dtype=torch.bool))
    """
    count = targets.boxes.shape[0]
    if targets.polygons:
        raise ValueError(f"the oriented path carries no polygons (WP-056); got {len(targets.polygons)} rings")
    if targets.rboxes.shape[0] != count:
        raise ValueError(
            f"rboxes must share the instance axis: {count} boxes, {targets.rboxes.shape[0]} rboxes (WP-056)"
        )
    if difficult.dtype != torch.bool:
        raise TypeError(f"difficult must be bool; got {difficult.dtype}")
    if difficult.ndim != 1 or difficult.shape[0] != count:
        raise ValueError(f"difficult must be 1-D of length {count}; got shape {tuple(difficult.shape)}")
