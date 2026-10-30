# SPDX-License-Identifier: Apache-2.0
"""Dual-path evaluation of a training checkpoint on COCO val2017 (WP-045).

Drives :class:`lucid_yolo.eval.coco_eval.DualPathEvaluator` end to end: loads a Lightning checkpoint (optionally
overlaying the EMA shadow the :class:`~lucid_yolo.ptl.callbacks.EMACallback` stored inside it), letterboxes val2017,
runs the model once per batch, decodes both the suppression-free E2E path and the NMS path, and prints the twelve
COCO summary statistics for each.

The ground truth is read straight from ``instances_val2017.json`` in original image coordinates, including ``iscrowd``
flags and annotation areas so the crowd-ignore rule and the small/medium/large breakdown follow the COCO protocol.

A **segmentation** checkpoint additionally reports the twelve ``segm_`` statistics per path, against ground-truth
masks decoded from the same annotation file. That is driven by the checkpoint's own ``task``, not by a flag the
caller has to remember: a detection checkpoint takes exactly the path it always took and pays nothing for the mask
machinery, and a segmentation checkpoint cannot be evaluated on boxes alone by accident. ``--no-masks`` forces the
detection-only reading when only the box numbers are wanted.

Two properties make the segmentation path affordable at val2017 scale, both in the library rather than here: the
ground truth is a :class:`~lucid_yolo.eval.annotations.LazyTargets` that decodes an image's masks when that image
is reached, and the evaluator folds each batch into the metric immediately (masks are RLE-encoded into the metric
state on update). Neither the targets nor the predictions are ever dense for the whole split at once.

The scoring stages themselves live in :mod:`lucid_yolo.eval.coco_score`, shared with the keypoint runner and with
:func:`~lucid_yolo.eval.coco_score.score_coco`, which returns the same report to a Python caller instead of printing
it; this module is the command's wrapper around them: the banner, the progress bar, the per-path lines and the
JSON file.

This is the path :func:`lucid_yolo.eval.cli.main` takes for a ``detect`` or ``segment`` checkpoint (WP-096); the
oriented one is :mod:`lucid_yolo.eval.rotated_eval`. Which of the two runs is decided by the checkpoint, in the
same spirit as the mask decision above.

Usage::

    lucid-eval CHECKPOINT --data-root ~/data/coco2017 \
        [--no-ema] [--no-masks] [--batch-size 32] [--img-size 640] \
        [--device mps] [--output report.json]

Provenance: R1 Table 7, R1 sec. 4.4 (dual-path protocol), R1 Table S9 (segmentation protocol). Assumptions: A9, A10,
A37.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tqdm.auto import tqdm

from lucid_yolo.eval.coco_score import (
    _coco_paths,
    _prepare_coco_run,
    _print_timing,
    _score_run,
    _write_report,
)

if TYPE_CHECKING:
    from pathlib import Path

    from lucid_yolo.ptl.module import DetectionLitModule

__all__ = ["run"]


def run(
    module: DetectionLitModule,
    info: dict[str, object],
    *,
    data_root: Path,
    img_size: int,
    batch_size: int,
    masks: bool,
    device_name: str,
    limit: int,
    output: Path | None,
    eval_backend: str = "faster_coco_eval",
    ann_file: Path | None = None,
    images_dir: Path | None = None,
) -> int:
    """Run the dual-path COCO evaluation for an already-loaded checkpoint.

    Args:
        module: The eval-mode module, loaded by :func:`lucid_yolo.cli.eval.evaluate`.
        info: That loader's provenance dict, extended here and written to the report.
        data_root: COCO root holding ``val2017/`` and ``annotations/``.
        img_size: Letterbox side.
        batch_size: Images per forward pass.
        masks: Whether a segmentation checkpoint also scores masks.
        device_name: Device string, or ``auto``.
        limit: Score only the first N images; ``0`` scores all.
        output: Optional path for the JSON report.
        eval_backend: The already-resolved bbox/segm scoring engine (WP-138),
            ``"faster_coco_eval"`` or ``"hotcoco"`` — resolved once by
            :func:`lucid_yolo.cli.eval.evaluate`, not re-probed here.
        ann_file: Annotation JSON override; defaults to annotations/instances_val2017.json under data_root.
        images_dir: Image directory override; defaults to val2017 under data_root.

    Returns:
        ``0``; a failure here raises rather than returning a code.

    Examples:
        ```pycon
        >>> callable(run)
        True

        ```
    """
    ann_file, images_dir = _coco_paths("detect", data_root, ann_file, images_dir)
    # Whether the ground truth needs masks at all is the checkpoint's property, not the caller's: --masks false can
    # decline them, but nothing opts a detector *into* them.
    segmentation = module.task == "segment" and masks
    info["masks"] = segmentation
    info["eval_backend"] = eval_backend
    coco_run = _prepare_coco_run(
        module,
        ann_file=ann_file,
        images_dir=images_dir,
        img_size=img_size,
        batch_size=batch_size,
        device_name=device_name,
        limit=limit,
        eval_backend=eval_backend,
        masks=segmentation,
        keypoints=False,
    )

    print(
        f"eval: {len(coco_run.images)} images, device={coco_run.device.type}, ema={info.get('ema')}, "
        f"img_size={img_size}, masks={segmentation}"
    )
    # The bar wraps the batch generator rather than living inside DualPathEvaluator: the evaluator is library code
    # that tests drive in-process, and a progress bar is a property of being run from a command line, which is here.
    batches = tqdm(coco_run.batches(), total=coco_run.num_batches, desc="eval", unit="batch")
    score = _score_run(coco_run, batches)
    _print_timing(score)

    report = score.report
    for path in ("e2e", "nms"):
        stats = report[path]
        print(f"[{path}] mAP50-95={stats['map']:.4f} mAP50={stats['map_50']:.4f} mAP75={stats['map_75']:.4f}")
        if "segm_map" in stats:
            print(
                f"[{path}] segm mAP50-95={stats['segm_map']:.4f} "
                f"segm mAP50={stats['segm_map_50']:.4f} segm mAP75={stats['segm_map_75']:.4f}"
            )
    delta = report["nms"]["map"] - report["e2e"]["map"]
    print(f"E2E deficit vs NMS: {delta * 100:.2f} AP")

    if output:
        _write_report(output, info, score)
    return 0
