# SPDX-License-Identifier: Apache-2.0
"""``lucid-predict`` — detections for one image, from one checkpoint (WP-089, WP-090, WP-091, WP-152).

One command, no task flag::

    lucid-predict --checkpoint runs/det.ckpt --image street.jpg
    lucid-predict --checkpoint runs/det.ckpt --image street.jpg --decoder nms --output dets.json
    lucid-predict --checkpoint runs/seg.ckpt --image street.jpg --output masks.json
    lucid-predict --checkpoint runs/obb.ckpt --image aerial.png --output rboxes.json
    lucid-predict --checkpoint runs/pose.ckpt --image street.jpg --output poses.json

The task is read from the checkpoint, exactly as ``lucid-eval`` reads it: a caller who
names the wrong task gets a wrong answer, and a caller who names none cannot. A
``detect`` checkpoint answers with boxes, a ``segment`` one additionally with each
detection's instance mask, an ``obb`` one with rotated boxes, and a ``keypoints`` one with
each detection's point set, through the four entry points of :mod:`lucid_yolo.predict`.
Which of the four runs is decided here, once, from the checkpoint's own task; each entry
point additionally refuses the other three by name, because
:meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward` would happily return
axis-aligned boxes for any of them. The keypoint branch arrived at WP-152, a release
after the task it serves: until then the dispatch's final ``else`` was detection, written
before a fourth task existed.

``--ema``, ``--device`` and ``--output`` keep ``lucid-eval``'s spellings and semantics,
so the two commands cannot disagree about what a flag means. ``--img_size`` defaults from
the checkpoint's task through the same :data:`~lucid_yolo.cli.eval.DEFAULT_IMG_SIZE`
table, rather than restating ``640`` here.

Assumptions:
    The roadmap row says nothing about output shape, so: stdout gets one line per
    detection, and ``--output`` writes a JSON object ``{"info", "image", "decoder",
    "conf_threshold", "masks", "detections"}``, plus an optional ``boxes`` or
    ``keypoints`` key on the reports that have one (below), whose ``detections`` are records of ``box``
    (``xyxy``, original-image pixels), ``score`` and ``label`` — mirroring
    ``lucid-eval``'s report, which also nests the checkpoint provenance under ``info``.
    Labels are contiguous class indices; a single image carries no category map. The
    parent directory of ``--output`` is created, which is WP-105's fix and not a
    convenience: the expensive part has already run by the time the file is written.

    Masks ride in that same file as **COCO RLE** — each segmentation record gains a
    ``segmentation`` of ``{"size": [height, width], "counts": "..."}`` — and the report's
    top-level ``masks`` key names the encoding (``"coco-rle"``, or ``null`` for a
    detection checkpoint), so a reader learns what it is holding from the file rather
    than from this docstring. The encoder is
    :func:`faster_coco_eval.mask.encode`, already a direct dependency and already the
    encoder every ``segm_`` statistic is measured through, so a predicted mask on disk
    and a scored mask are the same object in the same format. Nothing here invents one.

    The alternatives were weighed and lost: a polygon contour needs a contour tracer and
    is lossy on masks with holes, which the box crop routinely produces; a sidecar
    ``.npz`` splits one prediction across two files, so a report can be moved or archived
    into a state where it silently describes masks that are no longer there; and
    carrying only mask-derived scalars answers a smaller question than the one a caller
    running a segmentation checkpoint asked. RLE keeps one file, adds no dependency, and
    round-trips exactly.

    A keypoint report keeps ``box`` and adds ``keypoints`` to each record: the
    detection's points as ``[[x, y], ...]`` in the same original-image pixels, in the
    head's own point order. The report's top-level ``keypoints`` key names that layout
    (``"xy-pairs"``) for the reason ``masks`` names the mask encoding, and the layout is
    worth naming because COCO's own keypoint field is a **flat triplet** list carrying a
    visibility flag. This one is neither flat nor triplets: nothing in this project
    predicts visibility, so a third column would be an invented number. ``info`` gains
    ``num_keypoints``, since ``K`` is a property of whichever schema supplied the points
    (A64) and a reader should not have to re-load the checkpoint to tell a 17-point human
    pose from a 15-point letter one. Both keys appear on keypoint reports only.

    An oriented report replaces ``box`` with **two** derived geometries of the same
    object: ``rbox``, the A45 five-tuple exactly as
    :func:`~lucid_yolo.predict.predict_oriented` returned it, and ``polygon``, its four
    corners as ``[x1, y1, ..., x4, y4]``. Neither is a second arithmetic — the corners
    come from :func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons`, the same function
    :func:`~lucid_yolo.data.tiles.annotation_records` writes rotated ground truth with,
    and that record likewise carries one object as two derived geometries. Both are
    written because each answers a question the other cannot: the tuple is what
    reproduces the library's own answer and carries the canonical angle A23 guarantees,
    while the ring is what DOTA's protocol states an oriented object in and what a reader
    can draw without knowing this project's angle convention. The report's top-level
    ``boxes`` key names that convention (``"obb-longedge-rad"``) for the same reason
    ``masks`` names the mask encoding: the file should say what it holds. It appears on
    oriented reports only — the ``box`` of the other two is plain ``xyxy`` and was
    already shipped without it.

Provenance: R1 sec. 3.2.1, R1 Eq. 7, R1 Eq. 13, R3 sec. 4, R1 sec. 4.4, R18 sec. 4, R14.
Assumptions: A9, A10, A23, A37, A45, A64.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
from faster_coco_eval import mask as mask_api
from jsonargparse import auto_cli

from lucid_yolo.cli.eval import DEFAULT_IMG_SIZE
from lucid_yolo.data.rotated_geom import rboxes_to_polygons
from lucid_yolo.decode.common import BOX_CORNERS, LABEL_COLUMN, RBOX_COLUMNS, SCORE_COLUMN
from lucid_yolo.eval.checkpoint import load_eval_module, pick_device

# ``DecodePath`` is imported at runtime, not under TYPE_CHECKING: jsonargparse resolves
# the signature's annotations through ``get_type_hints`` to build the parser, and a name
# only the type checker can see is not in the module globals it resolves against.
from lucid_yolo.predict import (
    DEFAULT_CONF_THRESHOLD,
    DecodePath,
    predict_image,
    predict_keypoints,
    predict_oriented,
    predict_segmentation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

__all__ = [
    "DetectionRecord",
    "KeypointDetectionRecord",
    "OrientedDetectionRecord",
    "RleMask",
    "SegmentedDetectionRecord",
    "main",
    "predict",
]

#: Value of the report's ``masks`` key when the checkpoint has a mask branch: the name of
#: the encoding the ``segmentation`` records carry, so the file says what it holds.
#: A detection report carries ``null`` there rather than omitting the key, which would
#: make "this checkpoint has no masks" and "this report predates masks" the same reading.
_MASK_FORMAT = "coco-rle"

#: The task whose checkpoints additionally produce masks (WP-090).
_SEGMENT_TASK = "segment"

#: The task whose checkpoints produce rotated boxes instead of corners (WP-091).
_OBB_TASK = "obb"

#: The task whose checkpoints additionally produce point sets (WP-152).
_KEYPOINTS_TASK = "keypoints"

#: Value of a keypoint report's ``keypoints`` key: the layout its ``keypoints`` records
#: are written in -- one ``[x, y]`` pair per point, in the head's own point order, in
#: original-image pixels. Named because COCO's own keypoint field is a flat triplet list
#: carrying a visibility flag, and this one is neither flat nor triplets: nothing here
#: predicts visibility, so writing a third column would be inventing a number.
_KEYPOINT_FORMAT = "xy-pairs"

#: Coordinate count of a rotated box's corner ring, ``[x1, y1, ..., x4, y4]``. Named so
#: the flattening reads as "one ring per row" rather than as an inferred dimension, which
#: an image with nothing above the threshold would leave ambiguous.
_RING_VALUES = 8

#: Value of an oriented report's ``boxes`` key: the convention its ``rbox`` records are
#: written in — the long-edge form of :mod:`lucid_yolo.data.rotated_geom` with ``theta``
#: in **radians**. Named in the file because angle conventions are the thing oriented
#: formats silently disagree about (degrees, and a ``(0, 90]`` range, are both common).
_OBB_BOX_FORMAT = "obb-longedge-rad"


class DetectionRecord(TypedDict):
    """One detection as the report and the stdout summary carry it.

    A ``TypedDict`` rather than a dataclass because this *is* the JSON object — it goes
    to :func:`json.dumps` unchanged — while still naming its keys, so a renamed field is
    a type error here instead of a ``KeyError`` in whatever reads the report.

    Attributes:
        box: The ``xyxy`` corners in original-image pixels.
        score: Confidence in ``[0, 1]``.
        label: Contiguous class index (not a dataset category id — see the module
            docstring).
    """

    box: list[float]
    score: float
    label: int


class RleMask(TypedDict):
    """One instance mask in COCO run-length encoding, as JSON carries it.

    Attributes:
        size: The ``[height, width]`` of the **original** image the mask lives on, which
            is also what makes the record self-contained: a reader decodes it without
            knowing the letterbox side the model ran at.
        counts: The compressed RLE string. :func:`faster_coco_eval.mask.encode` returns
            these as ``bytes``; JSON has no such type, so they are decoded as ASCII here,
            exactly as the COCO annotation format stores them.
    """

    size: list[int]
    counts: str


class OrientedDetectionRecord(TypedDict):
    """One oriented detection as the report and the stdout summary carry it.

    Deliberately **not** a subclass of :class:`DetectionRecord`: it has no ``box``,
    because its geometry is not corners, and inheriting one to leave it out is not
    something a ``TypedDict`` can express. A reader therefore learns which report it is
    holding from the key it finds, before it has parsed a number.

    Attributes:
        rbox: The A45 box columns ``[cx, cy, w, h, theta]`` in original-image pixels,
            canonical per A23 — ``w >= h`` and ``theta`` in radians on
            ``[-pi/4, 3*pi/4)``, measured from ``+x`` towards ``+y`` on the y-down grid.
        polygon: The same rectangle's four corners, ``[x1, y1, ..., x4, y4]``, in the
            clockwise-as-displayed winding
            :func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons` defines. Redundant
            with ``rbox`` by construction and useful for exactly that reason: it is
            drawable and comparable without this project's angle convention.
        score: Confidence in ``[0, 1]``.
        label: Contiguous class index (not a dataset category id — see the module
            docstring).
    """

    rbox: list[float]
    polygon: list[float]
    score: float
    label: int


class KeypointDetectionRecord(DetectionRecord):
    """A detection record that also carries its point set.

    Inherits rather than restates, for the reason
    :class:`SegmentedDetectionRecord` does: a reader of either report parses ``box``,
    ``score`` and ``label`` the same way, and the added field is visibly the one thing
    that differs.

    Attributes:
        keypoints: The detection's points as ``[[x, y], ...]`` in the same
            original-image coordinates as ``box``, in the head's own point order. The
            length is the checkpoint's ``K``, never assumed (A64).
    """

    keypoints: list[list[float]]


class SegmentedDetectionRecord(DetectionRecord):
    """A detection record that also carries its instance mask.

    Inherits rather than restates, so a reader of either report parses ``box``, ``score``
    and ``label`` the same way and the mask is visibly the one added field.

    Attributes:
        segmentation: The detection's mask as :class:`RleMask`, in the same
            original-image coordinates as ``box``.
    """

    segmentation: RleMask


def predict(
    checkpoint: Path,
    image: Path,
    ema: bool = True,
    decoder: DecodePath = "e2e",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    img_size: int | None = None,
    device: str = "auto",
    output: Path | None = None,
) -> int:
    """Detect objects in one image, and segment or orient them when the checkpoint can.

    Which of the four happens is the checkpoint's ``task``, not a flag: a ``segment``
    checkpoint run as a detector would answer plausibly with its mask branch unread, an
    ``obb`` one with its angle branch unread, and that is precisely the mistake a flag
    lets a caller make.

    Args:
        checkpoint: Lightning ``.ckpt`` to predict with; its task must be ``detect``,
            ``segment`` or ``obb``.
        image: Image file to run on.
        ema: Predict with the EMA shadow stored in the checkpoint rather than raw
            weights.
        decoder: ``e2e`` for the suppression-free top-k path over the one-to-one branch,
            ``nms`` for the confidence-threshold plus class-wise suppression path over
            the dense branch. Defaults to ``e2e`` — it is the path this architecture
            exists to demonstrate, and the one a deployment would ship; ``nms`` is the
            comparison column, run on an ``obb`` checkpoint by the rotated suppression
            decoder (WP-091b), which compares rotated overlaps rather than upright ones.
        conf_threshold: Detections at or below this score are dropped, with their masks.
            In ``[0, 1]``: a negative cut admits the decoders' score-zero padding rows,
            which are not detections at any threshold and would reach the report as
            phantom objects at plausible coordinates.
        img_size: Letterbox side. Defaults per the checkpoint's task, and must be a
            positive multiple of every head stride.
        device: ``auto``, ``cpu``, ``mps`` or ``cuda``; a named backend this machine does
            not have is refused rather than failing later inside ``.to(device)``.
        output: Write the JSON report here; the parent directory is created if absent.

    Returns:
        ``0``; a failure here raises rather than returning a code.

    Raises:
        ValueError: If ``device`` names a backend that is not available
            (:func:`~lucid_yolo.eval.checkpoint.pick_device`), or if ``decoder``,
            ``conf_threshold`` or ``img_size`` is outside its domain — the four entry
            points of :mod:`lucid_yolo.predict` own that check, so this command and a
            direct library call refuse the same values with the same message.

    Examples:
        >>> predict(Path("/nonexistent.ckpt"), Path("/none.jpg"))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        FileNotFoundError: ...
    """
    module, info = load_eval_module(checkpoint, use_ema=ema)
    task = str(module.task)
    info["task"] = task
    info["decoder"] = decoder
    resolved_img_size = DEFAULT_IMG_SIZE.get(task, 640) if img_size is None else img_size
    run_on = pick_device(device)
    records: list[DetectionRecord] | list[OrientedDetectionRecord] | list[KeypointDetectionRecord]
    box_format: str | None = None
    if task == _OBB_TASK:
        rotated = predict_oriented(
            module,
            image,
            img_size=resolved_img_size,
            decoder=decoder,
            conf_threshold=conf_threshold,
            device=run_on,
        )
        records = _to_oriented_records(rotated)
        lines = _oriented_lines(records)
        mask_format: str | None = None
        keypoint_format: str | None = None
        box_format = _OBB_BOX_FORMAT
    elif task == _KEYPOINTS_TASK:
        posed = predict_keypoints(
            module,
            image,
            img_size=resolved_img_size,
            decoder=decoder,
            conf_threshold=conf_threshold,
            device=run_on,
        )
        records = _to_keypoint_records(posed.detections, posed.keypoints)
        lines = _keypoint_lines(records)
        mask_format = None
        keypoint_format = _KEYPOINT_FORMAT
        info["num_keypoints"] = posed.num_keypoints
    elif task == _SEGMENT_TASK:
        prediction = predict_segmentation(
            module,
            image,
            img_size=resolved_img_size,
            decoder=decoder,
            conf_threshold=conf_threshold,
            device=run_on,
        )
        records = _to_segmented_records(prediction.detections, prediction.masks)
        lines = _detection_lines(records)
        mask_format = _MASK_FORMAT
        keypoint_format = None
    else:
        # Not an `elif task == "detect"`: an unknown task must be refused, and the refusal
        # belongs to the library (see `lucid_yolo.predict`), which is where it names the
        # task the checkpoint actually carries. Restating the test here would give this
        # command a second opinion about which checkpoints it accepts.
        detections = predict_image(
            module,
            image,
            img_size=resolved_img_size,
            decoder=decoder,
            conf_threshold=conf_threshold,
            device=run_on,
        )
        records = _to_records(detections)
        lines = _detection_lines(records)
        mask_format = None
        keypoint_format = None

    print(
        f"predict: {image} -> {len(records)} detections, path={decoder}, "
        f"img_size={resolved_img_size}, masks={mask_format or 'none'}"
    )
    for line in lines:
        print(line)

    if output:
        payload: dict[str, object] = {
            "info": info,
            "image": str(image),
            "decoder": decoder,
            "conf_threshold": conf_threshold,
            "masks": mask_format,
            "detections": records,
        }
        # Added rather than always present, so the two shipped report shapes are untouched:
        # `box` has meant `xyxy` since WP-089 and needs no announcement, while `rbox` is
        # new and its angle convention is the thing oriented formats disagree about.
        if box_format is not None:
            payload["boxes"] = box_format
        # Added rather than always present, for the same reason `boxes` is: the other
        # shipped report shapes stay byte-identical, and a reader finds this key exactly
        # when the records carry a point set.
        if keypoint_format is not None:
            payload["keypoints"] = keypoint_format
        # Created rather than required, for the reason `detect_eval.run` states: the
        # forward pass is the expensive part and this file is its only durable form.
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"report -> {output}")
    return 0


def _to_records(detections: Tensor) -> list[DetectionRecord]:
    """Turn an A9 detection tensor into JSON-serialisable per-detection records.

    Args:
        detections: Detections of shape ``(N, 6)`` in original-image coordinates, as
            :func:`~lucid_yolo.predict.predict_image` returns them.

    Returns:
        One :class:`DetectionRecord` per row. The class column is stored as a float in
        the A9 tuple and is narrowed back to ``int`` here, at the boundary where the
        report is written.
    """
    return [
        DetectionRecord(
            box=[float(value) for value in row[:BOX_CORNERS]],
            score=float(row[SCORE_COLUMN]),
            label=int(row[LABEL_COLUMN]),
        )
        for row in detections
    ]


def _to_oriented_records(detections: Tensor) -> list[OrientedDetectionRecord]:
    """Turn an A45 oriented detection tensor into JSON-serialisable per-detection records.

    Args:
        detections: Oriented detections of shape ``(N, 7)`` in original-image coordinates,
            as :func:`~lucid_yolo.predict.predict_oriented` returns them.

    Returns:
        One :class:`OrientedDetectionRecord` per row. The corners come from
        :func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons` applied to the whole
        stack at once, so the ``polygon`` of row ``n`` is that row's own ``rbox`` and the
        two cannot describe different rectangles.
    """
    rings = rboxes_to_polygons(detections[:, :RBOX_COLUMNS]).reshape(-1, _RING_VALUES)
    return [
        OrientedDetectionRecord(
            rbox=[float(value) for value in row[:RBOX_COLUMNS]],
            polygon=[float(value) for value in ring],
            score=float(row[RBOX_COLUMNS]),
            label=int(row[RBOX_COLUMNS + 1]),
        )
        for row, ring in zip(detections, rings, strict=True)
    ]


def _to_keypoint_records(detections: Tensor, keypoints: Tensor) -> list[KeypointDetectionRecord]:
    """Turn a row-aligned detection tensor and point stack into records carrying both.

    Args:
        detections: Detections of shape ``(N, 6)`` in original-image coordinates.
        keypoints: Point coordinates ``(N, K, 2)``, row ``n`` belonging to detection
            ``n``, as :class:`~lucid_yolo.predict.KeypointPrediction` pairs them.

    Returns:
        One :class:`KeypointDetectionRecord` per row. ``strict=True`` on the zip is the
        point, exactly as it is for the mask records: the pairing is what
        :class:`~lucid_yolo.predict.KeypointPrediction` exists to protect, and a length
        mismatch that silently truncated would attach every pose to the wrong box from
        the mismatch onward.
    """
    return [
        KeypointDetectionRecord(**record, keypoints=[[float(x), float(y)] for x, y in points])
        for record, points in zip(_to_records(detections), keypoints, strict=True)
    ]


def _keypoint_lines(records: Sequence[KeypointDetectionRecord]) -> list[str]:
    """Render one stdout line per posed detection.

    The point count is printed rather than the points: ``K`` is often 17 and a terminal
    line carrying 34 coordinates is not read by anyone. The report file carries them all.
    """
    return [
        f"  class={record['label']} score={record['score']:.3f} "
        f"box=[{' '.join(f'{value:.1f}' for value in record['box'])}] "
        f"keypoints={len(record['keypoints'])}"
        for record in records
    ]


def _detection_lines(records: Sequence[DetectionRecord]) -> list[str]:
    """Render one stdout line per axis-aligned detection."""
    return [
        f"  class={record['label']} score={record['score']:.3f} "
        f"box=[{' '.join(f'{value:.1f}' for value in record['box'])}]"
        for record in records
    ]


def _oriented_lines(records: Sequence[OrientedDetectionRecord]) -> list[str]:
    """Render one stdout line per oriented detection.

    The angle is printed in degrees beside the tuple it is stored in radians in: a
    heading is what a reader of this line is checking by eye, and ``2.6`` radians is not
    a number anyone recognises as pointing anywhere.
    """
    return [
        f"  class={record['label']} score={record['score']:.3f} "
        f"rbox=[{' '.join(f'{value:.1f}' for value in record['rbox'][:-1])} "
        f"{math.degrees(record['rbox'][-1]):.1f}deg]"
        for record in records
    ]


def _to_segmented_records(detections: Tensor, masks: Tensor) -> list[DetectionRecord]:
    """Turn a row-aligned detection tensor and mask stack into records carrying both.

    Args:
        detections: Detections of shape ``(N, 6)`` in original-image coordinates.
        masks: Boolean masks ``(N, orig_height, orig_width)``, row ``n`` belonging to
            detection ``n``, as :class:`~lucid_yolo.predict.SegmentedPrediction` pairs
            them.

    Returns:
        One :class:`SegmentedDetectionRecord` per row. ``strict=True`` on the zip is the
        point: the pairing is what
        :class:`~lucid_yolo.predict.SegmentedPrediction` exists to protect, and a length
        mismatch that silently truncated would attach every mask to the wrong box from
        the mismatch onward.
    """
    return [
        SegmentedDetectionRecord(**record, segmentation=rle)
        for record, rle in zip(_to_records(detections), _encode_masks(masks), strict=True)
    ]


def _encode_masks(masks: Tensor) -> list[RleMask]:
    """Run-length encode a mask stack into the JSON-carryable COCO form.

    The whole stack goes through :func:`faster_coco_eval.mask.encode` in one call, which
    wants the instance axis **last** and the array Fortran-ordered — the layout COCO's
    RLE is defined over, column-major within each instance. An empty stack encodes to an
    empty list, which is the report of an image with nothing above the threshold.
    """
    stack = np.asfortranarray(masks.permute(1, 2, 0).to(torch.uint8).numpy())
    return [
        RleMask(size=[int(value) for value in rle["size"]], counts=rle["counts"].decode("ascii"))
        for rle in mask_api.encode(stack)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``lucid-predict`` from the command line.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The command's exit code.

    Examples:
        >>> main(["--help"])  # argparse exits the process on --help  # doctest: +SKIP
        0
    """
    return int(auto_cli(predict, args=None if argv is None else list(argv), as_positional=False))


if __name__ == "__main__":  # pragma: no cover - `python -m lucid_yolo.cli.predict`
    raise SystemExit(main())
