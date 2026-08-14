# SPDX-License-Identifier: Apache-2.0
"""``lucid-predict`` — detections for one image, from one checkpoint (WP-089, WP-090).

One command, no task flag::

    lucid-predict --checkpoint runs/det.ckpt --image street.jpg
    lucid-predict --checkpoint runs/det.ckpt --image street.jpg --decoder nms --output dets.json
    lucid-predict --checkpoint runs/seg.ckpt --image street.jpg --output masks.json

The task is read from the checkpoint, exactly as ``lucid-eval`` reads it: a caller who
names the wrong task gets a wrong answer, and a caller who names none cannot. A
``detect`` checkpoint answers with boxes and a ``segment`` one additionally with each
detection's instance mask, through the two entry points of :mod:`lucid_yolo.predict`; an
``obb`` checkpoint is refused by name rather than run as a detector, because
:meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward` would happily return boxes for
it. Oriented inference is WP-091.

``--ema``, ``--device`` and ``--output`` keep ``lucid-eval``'s spellings and semantics,
so the two commands cannot disagree about what a flag means. ``--img_size`` defaults from
the checkpoint's task through the same :data:`~lucid_yolo.cli.eval.DEFAULT_IMG_SIZE`
table, rather than restating ``640`` here.

Assumptions:
    The roadmap row says nothing about output shape, so: stdout gets one line per
    detection, and ``--output`` writes a JSON object ``{"info", "image", "decoder",
    "conf_threshold", "masks", "detections"}`` whose ``detections`` are records of ``box``
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

Provenance: R1 sec. 3.2.1, R1 Eq. 7, R3 sec. 4, R1 sec. 4.4. Assumptions: A9, A10, A37.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
from faster_coco_eval import mask as mask_api
from jsonargparse import auto_cli

from lucid_yolo.cli.eval import DEFAULT_IMG_SIZE
from lucid_yolo.decode.common import BOX_CORNERS, LABEL_COLUMN, SCORE_COLUMN
from lucid_yolo.eval.checkpoint import load_eval_module, pick_device

# ``DecodePath`` is imported at runtime, not under TYPE_CHECKING: jsonargparse resolves
# the signature's annotations through ``get_type_hints`` to build the parser, and a name
# only the type checker can see is not in the module globals it resolves against.
from lucid_yolo.predict import DEFAULT_CONF_THRESHOLD, DecodePath, predict_image, predict_segmentation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

__all__ = ["DetectionRecord", "RleMask", "SegmentedDetectionRecord", "main", "predict"]

#: Value of the report's ``masks`` key when the checkpoint has a mask branch: the name of
#: the encoding the ``segmentation`` records carry, so the file says what it holds.
#: A detection report carries ``null`` there rather than omitting the key, which would
#: make "this checkpoint has no masks" and "this report predates masks" the same reading.
_MASK_FORMAT = "coco-rle"

#: The task whose checkpoints additionally produce masks (WP-090).
_SEGMENT_TASK = "segment"


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
    """Detect objects in one image, and segment them when the checkpoint can.

    Which of the two happens is the checkpoint's ``task``, not a flag: a ``segment``
    checkpoint run as a detector would answer plausibly with its mask branch unread, and
    that is precisely the mistake a flag lets a caller make.

    Args:
        checkpoint: Lightning ``.ckpt`` to predict with; its task must be ``detect`` or
            ``segment``.
        image: Image file to run on.
        ema: Predict with the EMA shadow stored in the checkpoint rather than raw
            weights.
        decoder: ``e2e`` for the suppression-free top-k path over the one-to-one branch,
            ``nms`` for the confidence-threshold plus class-wise suppression path over
            the dense branch. Defaults to ``e2e`` — it is the path this architecture
            exists to demonstrate, and the one a deployment would ship; ``nms`` is the
            comparison column.
        conf_threshold: Detections at or below this score are dropped, with their masks.
        img_size: Letterbox side. Defaults per the checkpoint's task.
        device: ``auto``, ``cpu``, ``mps`` or ``cuda``.
        output: Write the JSON report here; the parent directory is created if absent.

    Returns:
        ``0``; a failure here raises rather than returning a code.

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
    records: list[DetectionRecord]
    if task == _SEGMENT_TASK:
        prediction = predict_segmentation(
            module,
            image,
            img_size=resolved_img_size,
            decoder=decoder,
            conf_threshold=conf_threshold,
            device=run_on,
        )
        records = _to_segmented_records(prediction.detections, prediction.masks)
        mask_format: str | None = _MASK_FORMAT
    else:
        # Not an `elif task == "detect"`: an `obb` checkpoint must be refused, and the
        # refusal belongs to the library (see `lucid_yolo.predict`), which is where it
        # names the task and points at WP-091. Restating the test here would give this
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
        mask_format = None

    print(
        f"predict: {image} -> {len(records)} detections, path={decoder}, "
        f"img_size={resolved_img_size}, masks={mask_format or 'none'}"
    )
    for record in records:
        corners = " ".join(f"{value:.1f}" for value in record["box"])
        print(f"  class={record['label']} score={record['score']:.3f} box=[{corners}]")

    if output:
        payload = {
            "info": info,
            "image": str(image),
            "decoder": decoder,
            "conf_threshold": conf_threshold,
            "masks": mask_format,
            "detections": records,
        }
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
        >>> main(["--help"])  # doctest: +SKIP
        0
    """
    return int(auto_cli(predict, args=None if argv is None else list(argv), as_positional=False))


if __name__ == "__main__":  # pragma: no cover - `python -m lucid_yolo.cli.predict`
    raise SystemExit(main())
