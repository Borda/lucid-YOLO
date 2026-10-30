# SPDX-License-Identifier: Apache-2.0
"""Dual-path keypoint evaluation of a training checkpoint on COCO val2017 (WP-134).

The pose counterpart of :mod:`lucid_yolo.eval.detect_eval`, and structurally its twin: load the letterboxed split,
run the model once per batch, decode both the suppression-free E2E path and the NMS path, and report each. What
it adds is the half WP-124 built and nothing called — OKS keypoint mAP — so the acceptance metric of the keypoint
tier is finally producible by a shipped command instead of only by a library function with no caller.

**Two things a keypoint report must not inherit from the detection one.**

The ground truth is ``person_keypoints_val2017.json``, not ``instances_val2017.json``. A keypoint checkpoint is
trained on one category, and COCO's mAP averages average precision over every category the ground truth contains:
scored against the 80-category instances file, a person detector's 79 empty categories each contribute ``AP = 0``
and the reported ``map`` is its true person AP divided by 80. That is not an approximation of the right number,
it is a different number — a checkpoint whose training loop logged ``val/mAP`` at 0.48 reported ``map = 0.0063``
through the detection path, and ``0.0063 * 80 = 0.50``. The person-keypoints file names one category, so the average
is over that one.

The comparable figure is the **e2e** entry. Training's own ``val/mAP`` is logged from the one-to-one branch alone
(:mod:`lucid_yolo.ptl.module`), against the datamodule's already single-class targets, so it is the E2E row of this
report that an epoch metric can be read beside; the ``nms`` row is the dense branch's own answer and is expected
to sit above it by the R1 sec. 4.4 deficit.

Box AP and OKS come from one pass over one set of decoded detections, per path, so the two columns cannot describe
differently filtered predictions. The report keeps the box statistics under their bare names and prefixes the ten
keypoint ones with ``oks_``, exactly as the segmentation report prefixes its mask ones.

``lucid-eval`` reaches this path when the checkpoint's own ``task`` is ``"keypoints"`` (WP-096's rule: the protocol
is the checkpoint's property, not a flag the caller has to remember). A checkpoint whose ``K`` is not COCO's 17
is refused here rather than scored: this protocol's ground truth, its sigma table and its point ordering are all
the person schema's, and nothing maps some other schema's point ``i`` onto it (A67 makes the same argument from
the other side, for the synthetic symbol schema's own gate).

Usage::

    lucid-eval --checkpoint runs/pose.ckpt --data_root ~/data/coco2017 \
        [--no-ema] [--batch_size 32] [--img_size 640] \
        [--device mps] [--limit 200] [--output report.json]

Provenance: R1 Table 7, R1 sec. 4.4 (dual-path protocol), R12 (COCO keypoint protocol and the OKS sigma table), R14.
Assumptions: A9, A10, A37.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tqdm.auto import tqdm

from lucid_yolo.eval.coco_score import (
    _coco_paths,
    _prepare_coco_run,
    _print_timing,
    _require_coco_person_schema,
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
    device_name: str,
    limit: int,
    output: Path | None,
    eval_backend: str = "faster_coco_eval",
    ann_file: Path | None = None,
    images_dir: Path | None = None,
) -> int:
    """Run the dual-path keypoint evaluation for an already-loaded checkpoint.

    Args:
        module: The eval-mode module, loaded by :func:`lucid_yolo.cli.eval.evaluate`.
        info: That loader's provenance dict, extended here and written to the report.
        data_root: COCO root holding ``val2017/`` and ``annotations/``.
        img_size: Letterbox side.
        batch_size: Images per forward pass.
        device_name: Device string, or ``auto``.
        limit: Score only the first N images; ``0`` scores all.
        output: Optional path for the JSON report.
        eval_backend: The already-resolved **box-branch** scoring engine (WP-138),
            ``"faster_coco_eval"`` or ``"hotcoco"``. The ten ``oks_`` keypoint
            statistics are unaffected either way — they stay on hand-driven
            ``faster_coco_eval`` (A73), the box branch's own protocol, not this
            report's.
        ann_file: Annotation JSON override; defaults to annotations/person_keypoints_val2017.json under data_root.
        images_dir: Image directory override; defaults to val2017 under data_root.

    Returns:
        ``0`` on success; ``1`` when the checkpoint's point count is not COCO's.

    Examples:
        ```pycon
        >>> callable(run)
        True

        ```
    """
    try:
        # Refused rather than scored on an invented sigma table; the reason is the helper's docstring. A library
        # caller gets the ValueError, the command a FAIL line.
        num_keypoints = _require_coco_person_schema(module)
    except ValueError as refusal:
        print(f"FAIL: {refusal}")
        return 1
    ann_file, images_dir = _coco_paths("keypoints", data_root, ann_file, images_dir)
    info["keypoints"] = num_keypoints
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
        masks=False,
        keypoints=True,
    )

    print(
        f"eval-pose: {len(coco_run.images)} images, device={coco_run.device.type}, ema={info.get('ema')}, "
        f"img_size={img_size}, K={num_keypoints}"
    )
    batches = tqdm(coco_run.batches(), total=coco_run.num_batches, desc="eval-pose", unit="batch")
    score = _score_run(coco_run, batches)
    _print_timing(score)

    for path in ("e2e", "nms"):
        stats = score.report[path]
        print(f"[{path}] box mAP50-95={stats['map']:.4f} mAP50={stats['map_50']:.4f} mAP75={stats['map_75']:.4f}")
        print(f"[{path}] OKS AP={stats['oks_AP_all']:.4f} AP50={stats['oks_AP_50']:.4f} AP75={stats['oks_AP_75']:.4f}")
    print("the e2e row is the one comparable to a training run's own val/mAP; nms is the dense branch")

    if output:
        _write_report(output, info, score)
    return 0
