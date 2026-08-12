# SPDX-License-Identifier: Apache-2.0
"""Dual-path evaluation of a training checkpoint on COCO val2017 (WP-045).

Drives :class:`lucid_yolo.eval.coco_eval.DualPathEvaluator` end to end: loads a
Lightning checkpoint (optionally overlaying the EMA shadow the
:class:`~lucid_yolo.ptl.callbacks.EMACallback` stored inside it), letterboxes
val2017, runs the model once per batch, decodes both the suppression-free E2E
path and the NMS path, and prints the twelve COCO summary statistics for each.

The ground truth is read straight from ``instances_val2017.json`` in original
image coordinates, including ``iscrowd`` flags and annotation areas so the
crowd-ignore rule and the small/medium/large breakdown follow the COCO protocol.

A **segmentation** checkpoint additionally reports the twelve ``segm_``
statistics per path, against ground-truth masks decoded from the same annotation
file. That is driven by the checkpoint's own ``task``, not by a flag the caller
has to remember: a detection checkpoint takes exactly the path it always took and
pays nothing for the mask machinery, and a segmentation checkpoint cannot be
evaluated on boxes alone by accident. ``--no-masks`` forces the detection-only
reading when only the box numbers are wanted.

Two properties make the segmentation path affordable at val2017 scale, both in
the library rather than here: the ground truth is a
:class:`~lucid_yolo.eval.annotations.LazyTargets` that decodes an image's masks
when that image is reached, and the evaluator folds each batch into the metric
immediately (masks are RLE-encoded into the metric state on update). Neither the
targets nor the predictions are ever dense for the whole split at once.

Usage::

    python scripts/eval_det.py CHECKPOINT --data-root ~/data/coco2017 \
        [--no-ema] [--no-masks] [--batch-size 32] [--img-size 640] \
        [--device mps] [--output report.json]

Provenance: R1 Table 7, R1 sec. 4.4 (dual-path protocol), R1 Table S9
(segmentation protocol). Assumptions: A9, A10, A37.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from torch import Tensor, nn

from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.decode.nms_path import NMSDecoder
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.annotations import letterboxed_batches, load_eval_annotations
from lucid_yolo.eval.checkpoint import load_eval_module, pick_device
from lucid_yolo.eval.coco_eval import DualPathEvaluator
from lucid_yolo.models.build import SegmentOutput
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from collections.abc import Sequence


class _SegmentationForward(nn.Module):
    """Present a segmentation checkpoint under the forward contract the evaluator expects.

    :meth:`DetectionLitModule.forward` returns the detection head's output alone
    — deliberately, since that is what the E2E decode, the epoch metric and the
    export path all ask for. The evaluator decides to score masks structurally,
    by whether the forward returns a
    :class:`~lucid_yolo.models.build.SegmentOutput`, so handing it the module
    unwrapped would silently produce a *detection* report from a segmentation
    checkpoint: no prototypes, no predicted masks, twelve keys, no error, and the
    cost of the ground-truth masks paid for nothing.

    The wrapped module is registered as a child, so ``.to()`` and ``.eval()``
    reach it, and the forward delegates to the module's own
    :meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward_segmentation` rather
    than recomposing the backbone, neck and mask branches here — that composition
    exists once, on purpose.

    Args:
        module: A checkpoint-loaded module whose ``task`` is ``"segment"``.

    Examples:
        >>> import torch
        >>> from lucid_yolo.ptl.module import DetectionLitModule
        >>> module = DetectionLitModule(
        ...     depth=0.34, width=0.25, max_channels=1024, num_classes=4, task="segment"
        ... ).eval()
        >>> with torch.no_grad():
        ...     out = _SegmentationForward(module)(torch.zeros(1, 3, 160, 160))
        >>> out.prototypes.shape[1]
        32
    """

    def __init__(self, module: DetectionLitModule) -> None:
        super().__init__()
        self.module = module

    def forward(self, images: Tensor) -> SegmentOutput:
        """Run the detection and mask branches over ``images``."""
        return self.module.forward_segmentation(images)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the dual-path COCO evaluation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="Lightning .ckpt to evaluate")
    parser.add_argument("--data-root", type=Path, required=True, help="COCO root with val2017/ and annotations/")
    parser.add_argument("--no-ema", dest="ema", action="store_false", help="evaluate raw weights, not the EMA shadow")
    parser.add_argument(
        "--no-masks",
        dest="masks",
        action="store_false",
        help="score boxes only, even for a segmentation checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (0 = all)")
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    ann_file = args.data_root / "annotations" / "instances_val2017.json"
    images_dir = args.data_root / "val2017"
    # The checkpoint loads first: whether the ground truth needs masks at all is
    # the checkpoint's property, not the caller's.
    module, info = load_eval_module(args.checkpoint, use_ema=args.ema)
    segmentation = module.task == "segment" and args.masks
    info["masks"] = segmentation
    images, targets, label_to_category = load_eval_annotations(ann_file, with_masks=segmentation)
    if args.limit:
        images = images[: args.limit]
    device = pick_device(args.device)
    letterbox = Letterbox(args.img_size)
    model: nn.Module = _SegmentationForward(module) if segmentation else module
    evaluator = DualPathEvaluator(model, TopKDecoder(), NMSDecoder(), label_to_category, letterbox)

    print(
        f"eval: {len(images)} images, device={device.type}, ema={args.ema}, "
        f"img_size={args.img_size}, masks={segmentation}"
    )
    start = time.perf_counter()
    report = evaluator.evaluate(letterboxed_batches(images, images_dir, letterbox, args.batch_size), targets, device)
    elapsed = time.perf_counter() - start
    print(f"done in {elapsed:.1f}s ({len(images) / elapsed:.1f} img/s)")

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

    if args.output:
        payload = {"info": info, "images": len(images), "seconds": round(elapsed, 1), "report": report}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"report -> {args.output}")


if __name__ == "__main__":
    main()
