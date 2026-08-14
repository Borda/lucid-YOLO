# SPDX-License-Identifier: Apache-2.0
"""``lucid-predict`` — detections for one image, from one checkpoint (WP-089).

One command, no task flag::

    lucid-predict --checkpoint runs/det.ckpt --image street.jpg
    lucid-predict --checkpoint runs/det.ckpt --image street.jpg --decoder nms --output dets.json

The task is read from the checkpoint, exactly as ``lucid-eval`` reads it: a caller who
names the wrong task gets a wrong answer, and a caller who names none cannot. Here that
reading has only one accepted outcome — ``detect``. A ``segment`` or ``obb`` checkpoint
is refused by name rather than run as a detector, because
:meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward` would happily return boxes for
either (see :mod:`lucid_yolo.predict`). Oriented and segmentation inference are WP-091
and WP-090.

``--ema``, ``--device`` and ``--output`` keep ``lucid-eval``'s spellings and semantics,
so the two commands cannot disagree about what a flag means. ``--img_size`` defaults from
the checkpoint's task through the same :data:`~lucid_yolo.cli.eval.DEFAULT_IMG_SIZE`
table, rather than restating ``640`` here.

Assumptions:
    The roadmap row says nothing about output shape, so: stdout gets one line per
    detection, and ``--output`` writes a JSON object ``{"info", "image", "decoder",
    "conf_threshold", "detections"}`` whose ``detections`` are records of ``box``
    (``xyxy``, original-image pixels), ``score`` and ``label`` — mirroring
    ``lucid-eval``'s report, which also nests the checkpoint provenance under ``info``.
    Labels are contiguous class indices; a single image carries no category map. The
    parent directory of ``--output`` is created, which is WP-105's fix and not a
    convenience: the expensive part has already run by the time the file is written.

Provenance: R1 sec. 3.2.1, R3 sec. 4, R1 sec. 4.4. Assumptions: A9, A10.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from jsonargparse import auto_cli

from lucid_yolo.cli.eval import DEFAULT_IMG_SIZE
from lucid_yolo.decode.common import BOX_CORNERS, LABEL_COLUMN, SCORE_COLUMN
from lucid_yolo.eval.checkpoint import load_eval_module, pick_device

# ``DecodePath`` is imported at runtime, not under TYPE_CHECKING: jsonargparse resolves
# the signature's annotations through ``get_type_hints`` to build the parser, and a name
# only the type checker can see is not in the module globals it resolves against.
from lucid_yolo.predict import DEFAULT_CONF_THRESHOLD, DecodePath, predict_image

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

__all__ = ["DetectionRecord", "main", "predict"]


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
    """Detect objects in one image with a detection checkpoint.

    Args:
        checkpoint: Lightning ``.ckpt`` to predict with; its task must be ``detect``.
        image: Image file to run on.
        ema: Predict with the EMA shadow stored in the checkpoint rather than raw
            weights.
        decoder: ``e2e`` for the suppression-free top-k path over the one-to-one branch,
            ``nms`` for the confidence-threshold plus class-wise suppression path over
            the dense branch. Defaults to ``e2e`` — it is the path this architecture
            exists to demonstrate, and the one a deployment would ship; ``nms`` is the
            comparison column.
        conf_threshold: Detections at or below this score are dropped.
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
    detections = predict_image(
        module,
        image,
        img_size=resolved_img_size,
        decoder=decoder,
        conf_threshold=conf_threshold,
        device=pick_device(device),
    )

    records = _to_records(detections)
    print(f"predict: {image} -> {len(records)} detections, path={decoder}, img_size={resolved_img_size}")
    for record in records:
        corners = " ".join(f"{value:.1f}" for value in record["box"])
        print(f"  class={record['label']} score={record['score']:.3f} box=[{corners}]")

    if output:
        payload = {
            "info": info,
            "image": str(image),
            "decoder": decoder,
            "conf_threshold": conf_threshold,
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
