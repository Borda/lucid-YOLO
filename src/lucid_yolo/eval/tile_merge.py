# SPDX-License-Identifier: Apache-2.0
"""Merge overlapping-tile detections back onto whole source images (WP-107).

The step WP-063 deferred, WP-088's scope never took up and 0.3.0 shipped without. Until
this module existed every oriented figure this project published was **per tile**: an
object crossing a 1024 px seam was counted once in each tile that saw it, and a per-tile
score never pays the duplicate-detection cost a whole-image score charges. A per-tile
number is therefore comparable to nothing outside this repository, and R1 Tables 10-11
are whole-image numbers.

The merge rule: **core ownership**
    Each tile owns a *core* — the rectangle of source-image coordinates lying closer to
    that tile's window than to any neighbouring window. A detection survives the merge
    if and only if its centre, mapped into source-image coordinates, lies in the core of
    the tile that produced it. Everything else is discarded, before anything is scored.

    The cores are built from the recorded windows alone (:func:`core_bounds`): on each
    axis the boundary between two neighbouring windows is placed at the **midpoint of
    their overlap band**, and the outer bounds run to infinity. They therefore *partition
    the plane*: every point of every source image lies in the core of exactly one tile,
    so an object contributes at most one detection however many tiles saw it.

Why this rule, given that the architecture is suppression-free
    R1's whole claim for the one-to-one branch is that it needs no NMS. Two overlapping
    tiles that both see one object emit two detections at full confidence and there is no
    suppression stage to remove either, so the merge is where a duplicate rule has to be
    stated — and stating it as a suppression step would quietly reintroduce, at the seam,
    exactly what the architecture exists to avoid.

    Core ownership is not suppression, in the strict sense that matters here. A detection
    is kept or dropped by **where it is**, decided by the tiling geometry before the model
    runs: it is never compared against another detection, never ranked by confidence, and
    is dropped whether or not the neighbouring tile detected anything at all. Run the
    model twice with different weights and the same detection is owned by the same tile.
    That is an ownership assignment over a partition of the image, and it composes with a
    suppression-free detector without borrowing anything from one.

    It also matches the protocol that produced the tiles. With overlap ``v``, a tile's
    core ends ``v / 2`` inside its own window on every side (the boundary sits at the
    middle of the overlap band, and the band is ``v`` wide), so **every object whose
    circumradius is at most ``v / 2`` is wholly visible to the tile that owns its
    centre**. R18 chose stride 512 on a 1024 patch precisely so that objects are seen
    whole somewhere; core ownership is the merge that cashes that guarantee in.

What it costs, stated plainly
    1. **One shot per object.** Whole-image recall is the *owning* tile's recall, not the
       union of every tile's. If the owner misses an object its neighbour detected, the
       image misses it. That is the price of refusing a confidence-ranked union, and it
       is a real cost: this merge can only lower the number a per-tile score reports.
    2. **Objects wider than the overlap are unreliable under any tile-local rule.** They
       are truncated in every tile that sees them, so no ownership assignment recovers
       the whole object, and the same width bound is what admits the two residual failure
       modes below. DOTA has such objects — bridges, harbours, soccer fields.
    3. **Localization jitter at a core boundary can double a detection.** Two tiles
       detecting one object whose centre sits within a pixel or two of a core boundary
       may place their centres on opposite sides of it and both survive. The boundary
       lies ``v / 2`` from either seam, where both tiles see the object with the most
       context they will ever have, so the two centres agree closely; the mode is
       bounded and rare, not absent.

Ground truth, and why it is reconstructed by the same rule
    Whole-image ground truth is assembled from the tiles' own annotations — each mapped
    into source coordinates by its window origin, then passed through the identical
    ownership filter. No second policy, no IoU threshold, no dedup heuristic.

    That is exact rather than approximate for the instances that matter, and the argument
    is R18's own 0.7 rule. A part at or above the threshold keeps *the original
    annotation, translated*: every tile holding an object at ``U >= 0.7`` reports the same
    source-coordinate centre, the cores partition the plane, so exactly one of those
    copies is kept — double-counting a non-difficult ground truth is structurally
    impossible, not merely unlikely. A clipped copy has a centre displaced towards its own
    tile's interior and is by construction flagged *difficult* (``U < 0.7``), which under
    A48 puts it in the discard class: it stays out of the recall denominator, and A48's
    non-difficult-first precedence stops it shadowing a real target. So such a straggler
    can never cost a true positive or a recall point.

    It is **not** neutral, and this is the merge's one flattering residual. A detection
    that finds a straggler is discarded rather than counted, where against the true
    whole-image annotation set — which has no straggler in it — the same detection would
    have been a false positive. Discarding false positives raises precision. The effect is
    bounded to objects wider than the overlap (nothing narrower can displace a clipped
    centre across a core boundary — the same width bound as the two costs above), it is
    the discard R18's devkit applies to difficult instances by design rather than
    something invented here, and it is measured in
    ``tests/eval/test_tile_merge.py::TestGroundTruthReconstruction``: on that fixture the
    absorption is worth 0.165 of ``map_50``. A report quoting a whole-image figure quotes
    one that errs upward by that mechanism and by no other.

    The alternative — reading the original whole-image label files — is more faithful and
    was rejected on coupling: the evaluation would need the source dataset root beside the
    tiled one, and a dataset-specific label parser on a path whose whole layout (WP-094)
    is deliberately dataset-agnostic. A53 put ``source_image`` and ``window`` on every
    tile record for exactly this reason, and they are sufficient.

Alternatives considered and rejected
    - **Centre-in-tile ownership** (keep a detection whose centre lies inside its own
      window). Ambiguous where it matters: in the overlap band a centre lies inside two or
      four windows, so it needs a tie-break, and every tie-break is either this module's
      partition or a suppression rule wearing a different name.
    - **Interior-only acceptance by a fixed margin** (drop detections within ``m`` pixels
      of a tile edge). Not a partition: a band of the source image is owned by no tile
      when ``m`` exceeds the overlap and by two when it does not, so it either invents
      blind strips or leaves the duplicates it was meant to remove.
    - **Confidence-ranked dedup by rotated IoU.** This is NMS at the seam. It is a
      defensible engineering answer and it would score higher than this module does, by
      taking the union of the tiles' recalls rather than the owner's. It is not taken
      because the number is meant to be an honest measurement of a suppression-free
      pipeline, and a pipeline that runs NMS on its output is not one. It also needs an
      IoU threshold no source states.
    - **Weighted box fusion.** Averages duplicate boxes into a consensus box. It needs the
      same clustering threshold NMS does, and then emits geometry no forward pass
      produced — a box whose provenance is the merge rather than the model.

Detection cap (A47, and the deviation this path makes)
    ``MAX_DETECTIONS = 300`` is per *evaluation unit*, and it is 300 because that is what
    :func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk` emits per forward pass. A whole
    source image is many forward passes, so the merged pipeline can emit ``300 x tiles``
    detections for it and re-capping the merged image at 300 would measure a truncation
    rather than the model — a dense DOTA image carries far more than 300 instances. The
    whole-image path therefore scores with the cap **off**
    (``evaluate_rotated_map(..., max_detections=None)``); the per-tile cap still applies,
    at emission, where A47 put it. The per-tile path is untouched.

Coordinates
    Detections arrive in the letterboxed frame the model saw. They are returned to tile
    pixels by :func:`~lucid_yolo.decode.common.rboxes_to_letterboxed_original` — the
    existing exact inverse, which is the only definition of the letterbox geometry this
    path touches — and the tile's window origin is then added as a plain translation.
    Nothing here computes a pad, a ratio or a corner. Ground truth needs no inverse at
    all: the annotations are in tile pixels already, so only the translation applies.

Provenance: R18 sec. 4 (the crop protocol and the 0.7 rule the reconstruction rests on),
R1 sec. 3.2.1 (the suppression-free branch this must not undo), R1 Tables 10-11 (the
whole-image figures a merged number is finally comparable to).
Assumptions: A45, A46, A47, A48, A53.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import torch

from lucid_yolo.data.rotated_geom import polygons_to_rboxes
from lucid_yolo.decode.common import rboxes_to_letterboxed_original
from lucid_yolo.eval.dota_eval import rotated_detections_to_predictions

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from torch import Tensor

__all__ = [
    "TileIndex",
    "TileWindow",
    "core_bounds",
    "in_core",
    "load_tile_index",
    "merge_whole_images",
    "tile_detections_to_source",
]

#: Column count of a window ``(x0, y0, x1, y1)`` in a tile's ``window`` record (A53).
_WINDOW_VALUES = 4
#: Corner count of the quadrilateral ring a tile annotation carries as ``segmentation``.
_QUAD_CORNERS = 4
#: Coordinates per ring point.
_POINT_STRIDE = 2
#: Column count of a core rectangle ``(x_lo, y_lo, x_hi, y_hi)``.
_CORE_VALUES = 4


@dataclass(frozen=True)
class TileWindow:
    """Where one tile sat in its source image (A53), and which tile it is.

    Attributes:
        source_image: File name of the source image the tile was cut from.
        origin: The window's ``(x0, y0)`` top-left corner in source-image pixels.
        size: The tile's own ``(height, width)`` in pixels, before any letterbox.
        tile_id: The tile's own COCO image id, carried so a consumer pairing this
            window with a loader position can *assert* the pairing rather than infer
            it from whatever annotations landed there
            (:func:`~lucid_yolo.eval.rotated_eval._require_matching_tile`). Two tiles
            holding byte-identical annotation sets are indistinguishable by their
            annotations and are distinct by id.

    Examples:
        >>> TileWindow("P0007.png", (824, 0), (1024, 1024), 7).origin
        (824, 0)
    """

    source_image: str
    origin: tuple[int, int]
    size: tuple[int, int]
    tile_id: int


@dataclass(frozen=True)
class TileIndex:
    """One split's tiles: where each sat, and what it was annotated with.

    The two tuples share one axis, and that axis is the order
    :class:`~lucid_yolo.data.coco.CocoDetectionDataset` reads images in — ascending COCO
    image id — so entry ``i`` describes the same tile the unshuffled val loader yields
    ``i``-th. That positional correspondence is the whole coupling between this module and
    the loader; nothing else about the loader is assumed.

    It is also checked rather than assumed, at the one place that consumes it:
    :func:`~lucid_yolo.eval.rotated_eval.score_split` compares each position's
    :attr:`TileWindow.tile_id` against the val dataset's own
    :attr:`~lucid_yolo.data.coco.CocoDetectionDataset.image_ids` at that position, and
    each yielded tile's labels and difficult flags against :attr:`ground_truth` there.
    The two catch different things and neither subsumes the other: the ids compare the
    index against the dataset it is meant to describe, which a re-sorted or differently
    built reader breaks even when the tiles involved carry identical annotations, while
    the annotation sets are the only trace of the *loader's* own yield order, which
    carries no ids. A split that fails either is refused instead of translating
    detections by another tile's window origin.

    Attributes:
        windows: One :class:`TileWindow` per tile.
        ground_truth: One ``{"rboxes", "labels", "difficult"}`` dict per tile, in
            **tile-local** pixels — the same per-image shape
            :func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map` consumes.

    Examples:
        >>> TileIndex((), ()).windows
        ()
    """

    windows: tuple[TileWindow, ...]
    ground_truth: tuple[dict[str, Tensor], ...]


def load_tile_index(ann_file: Path) -> TileIndex | None:
    """Read a tiled split's window provenance and per-tile ground truth.

    Returns ``None`` — rather than raising — when the file is an ordinary COCO container
    whose image records carry no ``source_image``/``window`` keys. Such a layout was not
    written by ``lucid-data build-tiles``, so it has no tiles to merge and no whole-image
    figure to report; the caller says so and reports the per-tile number alone.

    Args:
        ann_file: A COCO ``instances_<split>.json`` written by
            :func:`~lucid_yolo.data.tiles.build_tiles`.

    Returns:
        The split's :class:`TileIndex`, or ``None`` when the layout carries no window
        provenance.

    Raises:
        ValueError: If some but not all image records carry window provenance — a
            half-annotated layout would silently merge part of a split and drop the rest.

    Examples:
        >>> load_tile_index  # doctest: +ELLIPSIS
        <function load_tile_index at ...>
    """
    payload = json.loads(ann_file.read_text(encoding="utf-8"))
    records = sorted(payload["images"], key=lambda record: int(record["id"]))
    provenanced = [record for record in records if "source_image" in record and "window" in record]
    if not provenanced:
        return None
    if len(provenanced) != len(records):
        raise ValueError(
            f"{ann_file} carries window provenance on {len(provenanced)} of {len(records)} images; "
            "a partially tiled layout cannot be merged"
        )
    # Dense labels by *sorted category id*, the rule `CocoDetectionDataset` applies, so a
    # whole-image score and a per-tile score speak the same label space by construction.
    labels = {
        int(category["id"]): index  # type: ignore[call-overload]
        for index, category in enumerate(_by_id(payload["categories"]))
    }
    grouped = _group_annotations(payload.get("annotations", []))
    return TileIndex(
        windows=tuple(_window(record) for record in records),
        ground_truth=tuple(_tile_ground_truth(grouped.get(int(record["id"]), []), labels) for record in records),
    )


def core_bounds(windows: Sequence[TileWindow]) -> Tensor:
    """Build the core rectangle of every tile of **one** source image.

    On each axis the boundary between neighbouring window strips is the midpoint of their
    overlap band, and the outermost bounds are infinite. The result therefore partitions
    the plane: every point lies in exactly one tile's core, including points outside the
    source image, so a detection whose centre falls off the edge is owned rather than
    silently dropped.

    A single window yields a single infinite core, which is what makes the whole-image
    path an identity on a one-tile image: nothing can be disowned when one tile owns
    everything.

    Args:
        windows: The tiles of one source image, in any order. All are assumed to come from
            the same image; the caller groups them.

    Returns:
        ``(K, 4)`` float32 cores ``(x_lo, y_lo, x_hi, y_hi)``, aligned with ``windows``.
        The low bounds are inclusive and the high bounds exclusive (:func:`in_core`).

    Raises:
        ValueError: If ``windows`` is empty, or if two of them share an origin on one
            axis while spanning different lengths along it — a tiling this rule cannot
            partition (:func:`_axis_cores`), refused rather than mis-partitioned. Two
            windows sharing an origin *and* a span are the ordinary uniform-tiler case
            and are unaffected.

    Examples:
        >>> tiles = [
        ...     TileWindow("a.png", (0, 0), (6, 6), 1),
        ...     TileWindow("a.png", (4, 0), (6, 6), 2),
        ... ]
        >>> core_bounds(tiles)[:, 0].tolist()  # the shared boundary is the band midpoint
        [-inf, 5.0]
        >>> core_bounds(tiles)[:, 2].tolist()
        [5.0, inf]
        >>> core_bounds(tiles[:1]).tolist()  # one tile owns the plane
        [[-inf, -inf, inf, inf]]
    """
    if not windows:
        raise ValueError("core_bounds needs at least one window")
    horizontal = _axis_cores([(window.origin[0], window.origin[0] + window.size[1]) for window in windows], "x")
    vertical = _axis_cores([(window.origin[1], window.origin[1] + window.size[0]) for window in windows], "y")
    rows: list[list[float]] = []
    for window in windows:
        x_low, x_high = horizontal[window.origin[0]]
        y_low, y_high = vertical[window.origin[1]]
        rows.append([x_low, y_low, x_high, y_high])
    return torch.tensor(rows, dtype=torch.float32)


def in_core(centres: Tensor, core: Tensor) -> Tensor:
    """Test which centres a tile's core owns.

    The low bounds are inclusive and the high bounds exclusive, which is what makes
    neighbouring cores meet without overlapping: a centre exactly on a boundary belongs to
    the tile on the far side, once.

    Args:
        centres: ``(N, 2)`` ``(x, y)`` centres in source-image pixels.
        core: ``(4,)`` core rectangle ``(x_lo, y_lo, x_hi, y_hi)``.

    Returns:
        ``(N,)`` bool mask of the centres this core owns.

    Raises:
        ValueError: If ``core`` is not a ``(4,)`` tensor or ``centres`` not ``(N, 2)``.

    Examples:
        >>> import torch
        >>> core = torch.tensor([0.0, 0.0, 10.0, 10.0])
        >>> in_core(torch.tensor([[5.0, 5.0], [10.0, 5.0], [-1.0, 5.0]]), core).tolist()
        [True, False, False]
    """
    if core.ndim != 1 or core.shape[0] != _CORE_VALUES:
        raise ValueError(f"core must be (4,) as (x_lo, y_lo, x_hi, y_hi); got shape {tuple(core.shape)}")
    if centres.ndim != 2 or centres.shape[1] != _POINT_STRIDE:
        raise ValueError(f"centres must be (N, 2); got shape {tuple(centres.shape)}")
    low, high = core[:_POINT_STRIDE].to(centres), core[_POINT_STRIDE:].to(centres)
    return ((centres >= low) & (centres < high)).all(dim=1)


def tile_detections_to_source(
    detections: Tensor,
    window: TileWindow,
    letterboxed_size: tuple[int, int],
    *,
    score_floor: float = 0.0,
) -> dict[str, Tensor]:
    """Map one tile's oriented detections from the letterboxed canvas into source pixels.

    Two steps, in this order and no others. The letterbox is undone by
    :func:`~lucid_yolo.decode.common.rboxes_to_letterboxed_original`, the exact analytic
    inverse that already owns that geometry; the window origin is then **added**, because
    a tile offset is a translation and nothing more. Score-zero padding rows are dropped
    before either step reaches the caller, so they cannot survive an ownership test as
    zero-confidence detections parked at the window origin.

    Args:
        detections: ``(N, 7)`` A45 tuples ``[cx, cy, w, h, theta, score, class]`` for one
            tile, in letterboxed-canvas pixels.
        window: That tile's :class:`TileWindow`.
        letterboxed_size: The ``(height, width)`` canvas the detections live in.
        score_floor: Rows at or below this score are dropped. Defaults to ``0.0``, which
            drops the padding rows and nothing else.

    Returns:
        A ``{"rboxes", "scores", "labels"}`` prediction dict in source-image pixels.

    Raises:
        ValueError: If ``detections`` is not a 2-D tensor.

    Examples:
        >>> import torch
        >>> dets = torch.tensor([[2.0, 2.0, 4.0, 2.0, 0.3, 0.9, 1.0], [0.0] * 7])
        >>> window = TileWindow("a.png", (100, 200), (4, 4), 1)
        >>> merged = tile_detections_to_source(dets, window, (4, 4))
        >>> merged["rboxes"].tolist()  # identity letterbox, then + (100, 200)
        [[102.0, 202.0, 4.0, 2.0, 0.30000001192092896]]
        >>> merged["labels"].tolist()
        [1]
    """
    if detections.ndim != 2:
        raise ValueError(f"detections must be (N, 7) for one tile; got shape {tuple(detections.shape)}")
    mapped = rboxes_to_letterboxed_original(detections.unsqueeze(0), window.size, letterboxed_size)
    prediction = rotated_detections_to_predictions(mapped, score_floor)[0]
    prediction["rboxes"] = _translate(prediction["rboxes"], window.origin)
    return prediction


def merge_whole_images(
    index: TileIndex,
    tile_predictions: Sequence[Mapping[str, Tensor]],
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]], list[str]]:
    """Assemble per-source-image predictions and ground truth from per-tile ones.

    Both sides go through the same core-ownership filter (module docstring), so a
    detection and the ground truth it should match are kept or dropped by one rule.
    Every source image contributes an entry even when nothing of it survived, because
    :func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map` aligns its two sequences by
    position: an image dropped from one side and not the other would shift the pairing.

    A short ``tile_predictions`` is accepted, for the ``--limit`` path that stops
    part-way through a split. Its prefix is merged and the trailing source image is
    **dropped when incomplete** — scoring an image assembled from some of its tiles would
    report a recall the pipeline never had a chance to attain.

    Args:
        index: The split's :class:`TileIndex`.
        tile_predictions: Per-tile prediction dicts in **source-image** pixels, as
            :func:`tile_detections_to_source` returns them, aligned by position with
            ``index.windows`` (a prefix of it is allowed).

    Returns:
        A ``(predictions, ground_truth, names)`` triple: two position-aligned lists of
        per-source-image dicts and the source image names they belong to, in first-seen
        tile order.

    Raises:
        ValueError: If ``tile_predictions`` is longer than the index.

    Examples:
        >>> import torch
        >>> box = torch.tensor([[3.0, 3.0, 2.0, 1.0, 0.0]])
        >>> index = TileIndex(
        ...     windows=(TileWindow("a.png", (0, 0), (8, 8), 1),),
        ...     ground_truth=({"rboxes": box, "labels": torch.tensor([0]), "difficult": torch.tensor([False])},),
        ... )
        >>> prediction = {"rboxes": box, "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}
        >>> predictions, ground_truth, names = merge_whole_images(index, [prediction])
        >>> names, predictions[0]["rboxes"].shape, ground_truth[0]["labels"].tolist()
        (['a.png'], torch.Size([1, 5]), [0])
    """
    if len(tile_predictions) > len(index.windows):
        raise ValueError(f"got {len(tile_predictions)} predictions for {len(index.windows)} tiles")
    complete = _complete_prefix(index, len(tile_predictions))
    groups: dict[str, list[int]] = {}
    for tile in range(complete):
        groups.setdefault(index.windows[tile].source_image, []).append(tile)

    predictions: list[dict[str, Tensor]] = []
    ground_truth: list[dict[str, Tensor]] = []
    for tiles in groups.values():
        cores = core_bounds([index.windows[tile] for tile in tiles])
        predictions.append(
            _concatenate(
                [_own(tile_predictions[tile], cores[position]) for position, tile in enumerate(tiles)],
                ("rboxes", "scores", "labels"),
            )
        )
        ground_truth.append(
            _concatenate(
                [_own(_source_ground_truth(index, tile), cores[position]) for position, tile in enumerate(tiles)],
                ("rboxes", "labels", "difficult"),
            )
        )
    return predictions, ground_truth, list(groups)


def _complete_prefix(index: TileIndex, scored: int) -> int:
    """Return how many leading tiles form whole source images.

    Args:
        index: The split's tile index.
        scored: How many leading tiles have predictions.

    Returns:
        ``scored`` when it ends on a source-image boundary, otherwise the tile count of
        the last complete source image before it.

    Examples:
        >>> windows = (TileWindow("a.png", (0, 0), (4, 4), 1), TileWindow("b.png", (0, 0), (4, 4), 2))
        >>> index = TileIndex(windows, ({}, {}))
        >>> _complete_prefix(index, 2), _complete_prefix(index, 1)
        (2, 1)
    """
    if scored == len(index.windows):
        return scored
    trailing = index.windows[scored - 1].source_image if scored else ""
    if scored < len(index.windows) and index.windows[scored].source_image != trailing:
        return scored
    while scored and index.windows[scored - 1].source_image == trailing:
        scored -= 1
    return scored


def _own(entry: Mapping[str, Tensor], core: Tensor) -> dict[str, Tensor]:
    """Keep the rows of one tile's entry whose rotated-box centre lies in ``core``.

    Args:
        entry: A per-tile prediction or ground-truth dict in source-image pixels.
        core: That tile's ``(4,)`` core rectangle.

    Returns:
        The same keys, restricted to the owned rows.

    Examples:
        >>> import torch
        >>> entry = {"rboxes": torch.tensor([[1.0, 1.0, 2.0, 1.0, 0.0]]), "labels": torch.tensor([0])}
        >>> _own(entry, torch.tensor([0.0, 0.0, 10.0, 10.0]))["labels"].tolist()
        [0]
    """
    keep = in_core(entry["rboxes"][:, :_POINT_STRIDE], core)
    return {key: value[keep] for key, value in entry.items()}


def _concatenate(entries: Sequence[Mapping[str, Tensor]], keys: tuple[str, ...]) -> dict[str, Tensor]:
    """Concatenate one source image's surviving per-tile rows into a single dict.

    Args:
        entries: The owned rows of each tile of one source image.
        keys: The keys to carry through, which fixes the output shape even when every
            tile contributed nothing.

    Returns:
        One dict per key, concatenated along the instance axis.

    Examples:
        >>> import torch
        >>> parts = [{"labels": torch.tensor([1])}, {"labels": torch.tensor([2])}]
        >>> _concatenate(parts, ("labels",))["labels"].tolist()
        [1, 2]
    """
    return {key: torch.cat([entry[key] for entry in entries], dim=0) for key in keys}


def _source_ground_truth(index: TileIndex, tile: int) -> dict[str, Tensor]:
    """Translate one tile's ground truth into source-image pixels.

    No inverse letterbox is involved: the annotations are stored in tile pixels, so the
    window origin is the whole of the map.

    Args:
        index: The split's tile index.
        tile: Position of the tile within it.

    Returns:
        The tile's ground-truth dict with ``rboxes`` centres translated.

    Examples:
        >>> import torch
        >>> box = torch.tensor([[1.0, 2.0, 4.0, 2.0, 0.0]])
        >>> target = {"rboxes": box, "labels": torch.tensor([0]), "difficult": torch.tensor([False])}
        >>> index = TileIndex((TileWindow("a.png", (10, 20), (4, 4), 1),), (target,))
        >>> _source_ground_truth(index, 0)["rboxes"][0, :2].tolist()
        [11.0, 22.0]
    """
    target = index.ground_truth[tile]
    return {**target, "rboxes": _translate(target["rboxes"], index.windows[tile].origin)}


def _translate(rboxes: Tensor, origin: tuple[int, int]) -> Tensor:
    """Shift rotated-box centres by a window origin, leaving extents and angle alone.

    Args:
        rboxes: ``(N, 5)`` long-edge rotated boxes.
        origin: The window's ``(x0, y0)`` in source-image pixels.

    Returns:
        ``(N, 5)`` boxes with translated centres.

    Examples:
        >>> import torch
        >>> _translate(torch.tensor([[1.0, 1.0, 4.0, 2.0, 0.5]]), (10, 20)).tolist()
        [[11.0, 21.0, 4.0, 2.0, 0.5]]
    """
    offset = torch.tensor([*origin, 0.0, 0.0, 0.0], dtype=rboxes.dtype, device=rboxes.device)
    return rboxes + offset


def _axis_cores(spans: Sequence[tuple[int, int]], axis: str) -> dict[int, tuple[float, float]]:
    """Place one axis's core boundaries at the midpoints of the overlap bands.

    The intervals are keyed on the window **start**, which is what makes the boundary a
    property of the strip rather than of the individual tile — every tile of a row shares
    one horizontal strip, and a uniform tiler emits exactly that. Two windows sharing a
    start with *different* ends are a different tiling: one strip start would then need
    two boundaries, and keying on the start alone would silently keep whichever span was
    read last and partition the axis by it. That is refused rather than approximated.

    Args:
        spans: One ``(start, end)`` pair per window on this axis, repeats included.
        axis: ``"x"`` or ``"y"``, named in the refusal message.

    Returns:
        Each distinct start mapped to its ``(low, high)`` core interval; the first low
        and the last high are infinite, so the intervals tile the whole axis.

    Raises:
        ValueError: If two windows share a start on this axis but end differently.

    Examples:
        >>> _axis_cores([(0, 6), (4, 10)], "x")  # windows [0, 6) and [4, 10) overlap on [4, 6)
        {0: (-inf, 5.0), 4: (5.0, inf)}
        >>> _axis_cores([(0, 6), (0, 6)], "x")  # a shared strip is the uniform-tiler case
        {0: (-inf, inf)}
        >>> _axis_cores([(0, 6), (0, 9)], "x")
        Traceback (most recent call last):
        ValueError: two windows share the x origin 0 with different spans (end 6 and end 9); ...
    """
    ends: dict[int, int] = {}
    for start, end in spans:
        kept = ends.setdefault(start, end)
        if kept != end:
            raise ValueError(
                f"two windows share the {axis} origin {start} with different spans "
                f"(end {kept} and end {end}); core_bounds cannot partition that tiling"
            )
    ordered = sorted(ends)
    boundaries = [-math.inf]
    boundaries += [(start + ends[previous]) / 2.0 for previous, start in pairwise(ordered)]
    boundaries.append(math.inf)
    return {start: (boundaries[index], boundaries[index + 1]) for index, start in enumerate(ordered)}


def _window(record: Mapping[str, object]) -> TileWindow:
    """Read one COCO image record's A53 window provenance.

    Args:
        record: A tile's ``images`` entry.

    Returns:
        Its :class:`TileWindow`.

    Raises:
        ValueError: If ``window`` is not four values.

    Examples:
        >>> record = {"id": 3, "source_image": "a.png", "window": [1, 2, 5, 6], "height": 4, "width": 4}
        >>> _window(record).origin, _window(record).tile_id
        ((1, 2), 3)
    """
    window = record["window"]
    if not isinstance(window, list) or len(window) != _WINDOW_VALUES:
        raise ValueError(f"window must be [x0, y0, x1, y1]; got {window!r}")
    return TileWindow(
        source_image=str(record["source_image"]),
        origin=(int(window[0]), int(window[1])),
        size=(int(record["height"]), int(record["width"])),  # type: ignore[call-overload]
        tile_id=int(record["id"]),  # type: ignore[call-overload]
    )


def _by_id(categories: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Return the category records sorted by id, the reader's own label order.

    Args:
        categories: The container's ``categories`` list.

    Returns:
        The same records, ascending by ``id``.

    Examples:
        >>> [record["id"] for record in _by_id([{"id": 3}, {"id": 1}])]
        [1, 3]
    """
    return sorted(categories, key=lambda category: int(category["id"]))  # type: ignore[call-overload]


def _group_annotations(annotations: Sequence[Mapping[str, object]]) -> dict[int, list[Mapping[str, object]]]:
    """Group a container's annotations by image id.

    Args:
        annotations: The container's ``annotations`` list.

    Returns:
        Image id mapped to its annotations, in file order.

    Examples:
        >>> _group_annotations([{"image_id": 2}, {"image_id": 2}])[2]  # doctest: +ELLIPSIS
        [{'image_id': 2}, {'image_id': 2}]
    """
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for annotation in annotations:
        grouped.setdefault(int(annotation["image_id"]), []).append(annotation)  # type: ignore[call-overload]
    return grouped


def _tile_ground_truth(annotations: Sequence[Mapping[str, object]], labels: Mapping[int, int]) -> dict[str, Tensor]:
    """Fit one tile's annotation rings back to rotated boxes, in tile-local pixels.

    The ring is the load-bearing field of the oriented layout (WP-094), so the box is
    fitted from it by :func:`~lucid_yolo.data.rotated_geom.polygons_to_rboxes` rather than
    read from the advisory ``bbox`` — the same choice
    :class:`~lucid_yolo.data.coco.CocoDetectionDataset` makes, so the two paths cannot
    disagree about what a tile's ground truth is.

    Args:
        annotations: The tile's annotation records.
        labels: Category id mapped to dense label.

    Returns:
        A ``{"rboxes", "labels", "difficult"}`` dict on one instance axis.

    Examples:
        >>> ring = [[3.0, 2.0, 7.0, 2.0, 7.0, 4.0, 3.0, 4.0]]
        >>> target = _tile_ground_truth([{"segmentation": ring, "category_id": 1}], {1: 0})
        >>> [round(v, 4) for v in target["rboxes"][0].tolist()], target["difficult"].tolist()
        ([5.0, 3.0, 4.0, 2.0, 0.0], [False])
    """
    rings: list[Tensor] = []
    kept: list[int] = []
    difficult: list[bool] = []
    for annotation in annotations:
        ring = _quad(annotation.get("segmentation"))
        if ring is None or int(annotation.get("iscrowd", 0)) == 1:  # type: ignore[call-overload]
            continue
        rings.append(ring)
        kept.append(labels[int(annotation["category_id"])])  # type: ignore[call-overload]
        difficult.append(bool(int(annotation.get("difficult", 0))))  # type: ignore[call-overload]
    stacked = torch.stack(rings) if rings else torch.zeros((0, _QUAD_CORNERS, _POINT_STRIDE))
    return {
        "rboxes": polygons_to_rboxes(stacked),
        "labels": torch.tensor(kept, dtype=torch.int64),
        "difficult": torch.tensor(difficult, dtype=torch.bool),
    }


def _quad(segmentation: object) -> Tensor | None:
    """Parse a ``segmentation`` field into a ``(4, 2)`` ring, or ``None`` when unusable.

    Args:
        segmentation: The annotation's ``segmentation`` value.

    Returns:
        The quadrilateral, or ``None`` for an RLE dict, an empty list or a ring that is
        not four points.

    Examples:
        >>> _quad([[0.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0]]).shape
        torch.Size([4, 2])
        >>> _quad({"counts": []}) is None
        True
    """
    if not isinstance(segmentation, list) or not segmentation:
        return None
    flat = segmentation[0]
    if not isinstance(flat, list) or len(flat) != _QUAD_CORNERS * _POINT_STRIDE:
        return None
    return torch.tensor(flat, dtype=torch.float32).reshape(_QUAD_CORNERS, _POINT_STRIDE)
