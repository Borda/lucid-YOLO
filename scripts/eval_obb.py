# SPDX-License-Identifier: Apache-2.0
"""Per-tile rotated evaluation of an oriented checkpoint (WP-095).

The oriented counterpart of ``eval_det.py``. It loads an ``obb`` checkpoint, runs it over
a split of the tiled layout ``build_dota_tiles.py`` writes, decodes the NMS-free
one-to-one branch into rotated boxes and scores them with the WP-063 accumulator
(:func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map`), which brings A46's exact
101-point recall grid, A47's 300-detection cap and A48's difficult rule with it.

**Per tile, not per image.** This reports the number over 1024 px crops, and a crop
number is not comparable to a published one: an object crossing a tile boundary is
counted twice, once as a clipped part in each tile. Merging tiles back into whole images
needs a duplicate rule, and in an NMS-free path there is nothing to suppress the
duplicate with — so that rule is a decision (WP-064 owns it), not an implementation
detail this script may quietly pick. What this script is for is the acceptance figure
that the tier run itself gates on, and the regression instrument for it; the layout
carries ``source_image`` and ``window`` on every tile so the merge can be built on top
without re-tiling anything.

Coordinate frame:
    Scoring runs in **letterbox** coordinates, with the ground truth letterboxed by the
    same transform as the image rather than the predictions being mapped back. A
    letterbox is a uniform scale plus a translation, matching is per image, and rotated
    IoU is invariant under both — so this is an equality, not the approximation the
    epoch-level ``val/rotated_mAP`` proxy makes. The one thing it does *not* preserve is
    any absolute-pixel reading of a box, which this report does not make.

The epoch metric already exists (WP-088 logs ``val/rotated_mAP`` and
``val/rotated_mAP50`` from the validation loop) and is deliberately not what this is:
that one runs on the training recipe's own loader, on whatever split the run configured,
and reports at epoch granularity. This script takes a finished checkpoint, an explicit
split, and the EMA weights a release is actually evaluated on.

Usage::

    python scripts/eval_obb.py CHECKPOINT --data-root /data/dota_tiles \
        [--split val] [--no-ema] [--batch-size 8] [--img-size 1024] \
        [--device mps] [--limit 200] [--output report.json]

Provenance: R18 sec. 4 (the tiled protocol), R1 sec. 4.4. Assumptions: A46, A47, A48, A53.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from lucid_yolo.assign import make_anchor_points
from lucid_yolo.eval.checkpoint import load_eval_module, pick_device
from lucid_yolo.eval.dota_eval import MAX_DETECTIONS, evaluate_rotated_map, rotated_detections_to_predictions
from lucid_yolo.models.heads.obb import decode_rboxes, o2o_rotated_topk
from lucid_yolo.ptl.datamodule import DetectionDataModule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lucid_yolo.ptl.module import DetectionLitModule

#: Feature-map strides of the three detection levels (blueprint sec. 5.4).
STRIDES = (8, 16, 32)


def build_datamodule(
    data_root: Path, split: str, *, img_size: int, batch_size: int, variant: str
) -> DetectionDataModule:
    """Point a datamodule's **val** loader at one split of the tiled layout.

    Both loaders are configured, because the datamodule builds them as a pair, but only
    the val one is read: it applies the letterbox and nothing else, which is what an
    acceptance figure is scored through.

    Args:
        data_root: Root of the tiled layout.
        split: Split name, e.g. ``"val"``.
        img_size: Letterbox side; the tier's tiles are 1024 px.
        batch_size: Images per forward pass.
        variant: Scale letter, which selects the augmentation policy the val loader
            does not use.

    Returns:
        A datamodule set up for the oriented val split.

    Examples:
        >>> datamodule = build_datamodule(Path("/data/tiles"), "val", img_size=1024, batch_size=8, variant="n")
        >>> type(datamodule).__name__
        'DetectionDataModule'
    """
    images_dir = data_root / split
    ann_file = data_root / "annotations" / f"instances_{split}.json"
    return DetectionDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=0,
        variant=variant,
        img_size=img_size,
        train_images_dir=images_dir,
        train_ann_file=ann_file,
        val_images_dir=images_dir,
        val_ann_file=ann_file,
        rotated_targets=True,
    )


def score_split(
    module: DetectionLitModule,
    datamodule: DetectionDataModule,
    device: torch.device,
    *,
    img_size: int,
    limit: int = 0,
) -> tuple[dict[str, float], int, int]:
    """Run the one-to-one branch over the split and score it per tile.

    Args:
        module: The eval-mode oriented module.
        datamodule: A datamodule whose val loader yields the split, letterboxed.
        device: Device to run on.
        img_size: The letterbox side the anchor grid is built for.
        limit: Stop after this many tiles; ``0`` scores the whole split.

    Returns:
        A ``(metrics, tiles, instances)`` triple: the
        :func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map` dict, the number of
        tiles scored, and the number of ground-truth instances they carried.

    Raises:
        ValueError: If the module's task is not ``"obb"`` — a detection checkpoint has
            no angle stem, so there is no oriented number to report.

    Examples:
        >>> from lucid_yolo.ptl.module import DetectionLitModule
        >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=1)
        >>> score_split(module, None, torch.device("cpu"), img_size=64)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: ...
    """
    if module.task != "obb":
        raise ValueError(f"eval_obb needs an oriented checkpoint; this one has task={module.task!r}")
    module = module.to(device)
    anchor_points, strides = _anchor_grid(img_size, device)
    predictions: list[dict[str, Tensor]] = []
    ground_truth: list[dict[str, Tensor]] = []
    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            images, targets, _ = datamodule.on_after_batch_transfer(batch, 0)
            head_out = module(images.to(device))
            rboxes = decode_rboxes(head_out.o2o_box, head_out.o2o_angle, anchor_points, strides)
            detections = o2o_rotated_topk(head_out.o2o_cls, rboxes, k=MAX_DETECTIONS)
            predictions.extend(rotated_detections_to_predictions(detections))
            ground_truth.extend(
                {
                    "rboxes": target.rboxes.cpu(),
                    "labels": target.labels.cpu().to(torch.long),
                    "difficult": target.difficult.cpu(),
                }
                for target in targets
            )
            if limit and len(predictions) >= limit:
                break
    if limit:
        predictions, ground_truth = predictions[:limit], ground_truth[:limit]
    instances = sum(int(target["labels"].numel()) for target in ground_truth)
    return evaluate_rotated_map(predictions, ground_truth), len(ground_truth), instances


def _anchor_grid(img_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Return the ``(anchor_points, strides)`` grid for a square input on ``device``."""
    feature_sizes = [(img_size // stride, img_size // stride) for stride in STRIDES]
    points, strides = make_anchor_points(feature_sizes, list(STRIDES))
    return points.to(device), strides.to(device)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the per-tile oriented evaluation from the command line.

    Returns:
        ``0`` on success; ``1`` when the checkpoint is not an oriented one.

    Examples:
        >>> callable(main)
        True
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="Lightning .ckpt to evaluate")
    parser.add_argument("--data-root", type=Path, required=True, help="tiled layout root (build_dota_tiles.py)")
    parser.add_argument("--split", default="val", help="split to score (default: val)")
    parser.add_argument("--no-ema", dest="ema", action="store_false", help="evaluate raw weights, not the EMA shadow")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--variant", default="n", help="scale letter of the trained model")
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N tiles (0 = all)")
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    module, info = load_eval_module(args.checkpoint, use_ema=args.ema)
    device = pick_device(args.device)
    datamodule = build_datamodule(
        args.data_root, args.split, img_size=args.img_size, batch_size=args.batch_size, variant=args.variant
    )
    datamodule.setup("validate")

    print(f"eval-obb: split={args.split}, device={device.type}, ema={args.ema}, img_size={args.img_size}")
    start = time.perf_counter()
    try:
        metrics, tiles, instances = score_split(module, datamodule, device, img_size=args.img_size, limit=args.limit)
    except ValueError as error:
        print(f"FAIL: {error}")
        return 1
    elapsed = time.perf_counter() - start

    print(f"scored {tiles} tiles ({instances} instances) in {elapsed:.1f}s")
    print(f"[per-tile] rotated mAP50-95={metrics['map']:.4f} mAP50={metrics['map_50']:.4f}")
    print("whole-image merge is WP-064's decision; this number is per tile")

    if args.output:
        payload = {
            "info": {**info, "split": args.split, "img_size": args.img_size, "per_tile": True},
            "tiles": tiles,
            "instances": instances,
            "seconds": round(elapsed, 1),
            "metrics": metrics,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"report -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
