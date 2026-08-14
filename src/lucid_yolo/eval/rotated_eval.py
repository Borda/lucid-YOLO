# SPDX-License-Identifier: Apache-2.0
"""Rotated evaluation of an oriented checkpoint, per tile and per whole image (WP-095, WP-107).

The oriented counterpart of :mod:`lucid_yolo.eval.detect_eval`. It takes an ``obb``
checkpoint, runs it over a split of the tiled layout ``lucid-data build-tiles`` writes,
decodes the NMS-free
one-to-one branch into rotated boxes and scores them with the WP-063 accumulator
(:func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map`), which brings A46's exact
101-point recall grid, A47's 300-detection cap and A48's difficult rule with it.

**Two figures, never one.** The per-tile number is the acceptance figure the tier run
gates on, and the regression instrument for it; it is computed over 1024 px crops, where
an object crossing a tile boundary is counted once in each tile that saw it, and it is
comparable to nothing published. The whole-image number is the one R1 Tables 10-11 can be
read beside: :mod:`lucid_yolo.eval.tile_merge` maps each tile's detections back into
source-image coordinates through the A53 window provenance and resolves the seam
duplicates by core ownership — the rule and its argument from the NMS-free constraint
live in that module. Both are printed, both are written to the report, and each is
labelled, because a report that does not say which figure it quotes is how the two got
confused in the first place (0.3.0 shipped the per-tile one).

The whole-image figure is unavailable — and said to be, rather than silently omitted —
when the split's annotations carry no window provenance, which is any layout not written
by ``lucid-data build-tiles``.

Coordinate frame:
    Scoring runs in **letterbox** coordinates, with the ground truth letterboxed by the
    same transform as the image rather than the predictions being mapped back. A
    letterbox is a uniform scale plus a translation, matching is per image, and rotated
    IoU is invariant under both — so this is an equality, not the approximation the
    epoch-level ``val/rotated_mAP`` proxy makes. The one thing it does *not* preserve is
    any absolute-pixel reading of a box, which the per-tile report does not make. The
    whole-image path does need one, so it takes the other route: detections go back
    through the exact letterbox inverse into tile pixels and are then translated by the
    window origin, and its ground truth is read from the annotations in tile pixels
    rather than through the loader's transform.

The epoch metric already exists (WP-088 logs ``val/rotated_mAP`` and
``val/rotated_mAP50`` from the validation loop) and is deliberately not what this is:
that one runs on the training recipe's own loader, on whatever split the run configured,
and reports at epoch granularity. This takes a finished checkpoint, an explicit split,
and the EMA weights a release is actually evaluated on.

``lucid-eval`` reaches this path when the checkpoint's own ``task`` is ``"obb"``
(WP-096) — the caller names no task, for the same reason the detection path decides
masks from the checkpoint rather than from a flag.

Usage::

    lucid-eval CHECKPOINT --data-root /data/dota_tiles \
        [--split val] [--no-ema] [--batch-size 8] [--img-size 1024] \
        [--device mps] [--limit 200] [--output report.json]

Provenance: R18 sec. 4 (the tiled protocol), R1 sec. 4.4. Assumptions: A46, A47, A48, A53.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from lucid_yolo.assign import anchor_grid
from lucid_yolo.data.layout import resolve_split
from lucid_yolo.eval.checkpoint import pick_device
from lucid_yolo.eval.dota_eval import MAX_DETECTIONS, evaluate_rotated_map, rotated_detections_to_predictions
from lucid_yolo.eval.tile_merge import TileIndex, load_tile_index, merge_whole_images, tile_detections_to_source
from lucid_yolo.models.heads.obb import decode_rboxes, o2o_rotated_topk
from lucid_yolo.ptl.datamodule import DetectionDataModule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lucid_yolo.eval.tile_merge import TileWindow
    from lucid_yolo.ptl.module import DetectionLitModule


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
    images_dir, ann_file = resolve_split(data_root, split)
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


@dataclass(frozen=True)
class SplitScoring:
    """What one scoring pass produced, per tile and (when merged) per source image.

    Attributes:
        per_tile: The per-tile :func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map`
            dict, scored in letterbox coordinates over 1024 px crops.
        tiles: Tiles scored.
        instances: Ground-truth instances those tiles carried.
        source_predictions: Per-tile predictions in **source-image** pixels, ready for
            :func:`~lucid_yolo.eval.tile_merge.merge_whole_images`. Empty when the split
            carries no window provenance and no whole-image figure is available.

    Examples:
        >>> SplitScoring({"map": 0.5}, tiles=4, instances=9).tiles
        4
    """

    per_tile: dict[str, float]
    tiles: int
    instances: int
    source_predictions: list[dict[str, Tensor]] = field(default_factory=list)


def score_split(
    module: DetectionLitModule,
    datamodule: DetectionDataModule,
    device: torch.device,
    *,
    img_size: int,
    limit: int = 0,
    windows: Sequence[TileWindow] | None = None,
) -> SplitScoring:
    """Run the one-to-one branch over the split, scoring per tile and mapping to source.

    The forward pass is walked once. Each batch's detections are scored per tile in the
    letterbox frame they arrive in, and — when ``windows`` is supplied — additionally
    mapped back into source-image pixels for the merge, which needs each tile's own
    pre-letterbox size and its window origin. The two readings come from the same
    detections, so a difference between the reported figures is the merge and nothing
    else.

    Args:
        module: The eval-mode oriented module.
        datamodule: A datamodule whose val loader yields the split, letterboxed.
        device: Device to run on.
        img_size: The letterbox side the anchor grid is built for.
        limit: Stop after this many tiles; ``0`` scores the whole split.
        windows: The split's tile windows, in loader order (the unshuffled val loader
            yields images in ascending COCO image id, which is the order
            :func:`~lucid_yolo.eval.tile_merge.load_tile_index` returns). ``None`` skips
            the source-coordinate mapping entirely.

    Returns:
        The :class:`SplitScoring`.

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
    anchor_points, strides = anchor_grid((img_size, img_size), device)
    predictions: list[dict[str, Tensor]] = []
    ground_truth: list[dict[str, Tensor]] = []
    source_predictions: list[dict[str, Tensor]] = []
    loader = datamodule.val_dataloader()
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="eval-obb", unit="batch"):
            images, targets, _ = datamodule.on_after_batch_transfer(batch, 0)
            head_out = module(images.to(device))
            rboxes = decode_rboxes(head_out.o2o_box, head_out.o2o_angle, anchor_points, strides)
            detections = o2o_rotated_topk(head_out.o2o_cls, rboxes, k=MAX_DETECTIONS).cpu()
            source_predictions.extend(_to_source(detections, windows, len(predictions), img_size))
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
        source_predictions = source_predictions[:limit]
    instances = sum(int(target["labels"].numel()) for target in ground_truth)
    return SplitScoring(
        per_tile=evaluate_rotated_map(predictions, ground_truth),
        tiles=len(ground_truth),
        instances=instances,
        source_predictions=source_predictions,
    )


def _to_source(
    detections: Tensor, windows: Sequence[TileWindow] | None, first: int, img_size: int
) -> list[dict[str, Tensor]]:
    """Map one batch's oriented detections into source-image pixels, tile by tile.

    Args:
        detections: ``(B, N, 7)`` A45 detections in letterboxed-canvas pixels.
        windows: The split's tile windows in loader order, or ``None`` to skip.
        first: Loader position of this batch's first tile.
        img_size: The letterbox side, which is the canvas both axes were padded to.

    Returns:
        One prediction dict per tile of the batch, in source-image pixels; empty when
        ``windows`` is ``None``.

    Examples:
        >>> import torch
        >>> _to_source(torch.zeros(2, 3, 7), None, 0, 64)
        []
    """
    if windows is None:
        return []
    return [
        tile_detections_to_source(detections[position], windows[first + position], (img_size, img_size))
        for position in range(detections.shape[0])
    ]


def run(
    module: DetectionLitModule,
    info: dict[str, object],
    *,
    data_root: Path,
    split: str,
    variant: str,
    img_size: int,
    batch_size: int,
    device_name: str,
    limit: int,
    output: Path | None,
) -> int:
    """Run the per-tile oriented evaluation for an already-loaded checkpoint.

    Args:
        module: The eval-mode module, loaded by :func:`lucid_yolo.cli.eval.evaluate`.
        info: That loader's provenance dict, extended here and written to the report.
        data_root: Root of the tiled layout.
        split: Split of that layout to score.
        variant: Scale letter of the trained model.
        img_size: Letterbox side; the tier's tiles are 1024 px.
        batch_size: Tiles per forward pass.
        device_name: Device string, or ``auto``.
        limit: Score only the first N tiles; ``0`` scores all.
        output: Optional path for the JSON report.

    Returns:
        ``0`` on success; ``1`` when the checkpoint is not an oriented one.

    Examples:
        >>> callable(run)
        True
    """
    device = pick_device(device_name)
    _, ann_file = resolve_split(data_root, split)
    index = load_tile_index(ann_file)
    datamodule = build_datamodule(data_root, split, img_size=img_size, batch_size=batch_size, variant=variant)
    datamodule.setup("validate")

    print(f"eval-obb: split={split}, device={device.type}, ema={info.get('ema')}, img_size={img_size}")
    start = time.perf_counter()
    try:
        scoring = score_split(
            module,
            datamodule,
            device,
            img_size=img_size,
            limit=limit,
            windows=None if index is None else index.windows,
        )
    except ValueError as error:
        print(f"FAIL: {error}")
        return 1
    elapsed = time.perf_counter() - start
    whole_image = None if index is None else _whole_image(index, scoring)

    print(f"scored {scoring.tiles} tiles ({scoring.instances} instances) in {elapsed:.1f}s")
    print(f"[per-tile] rotated mAP50-95={scoring.per_tile['map']:.4f} mAP50={scoring.per_tile['map_50']:.4f}")
    if whole_image is None:
        print("[whole-image] unavailable: this split carries no A53 window provenance to merge on")
    else:
        metrics, images = whole_image
        print(f"[whole-image] rotated mAP50-95={metrics['map']:.4f} mAP50={metrics['map_50']:.4f} ({images} images)")

    if output:
        # The scoring pass is minutes of work and the report is its only durable form, so
        # the directory is created rather than required: a run that computed the number
        # and then raised on a missing parent has lost exactly what it was asked for.
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_report(info, split, img_size, elapsed, scoring, whole_image), indent=2) + "\n")
        print(f"report -> {output}")
    return 0


def _whole_image(index: TileIndex, scoring: SplitScoring) -> tuple[dict[str, float], int]:
    """Merge the pass's per-tile detections onto source images and score them.

    The cap is lifted (:func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map`'s
    ``max_detections=None``): each tile's 300 were already applied at emission, and a
    merged source image is many tiles, so re-capping it at 300 would report a truncation.

    Args:
        index: The split's tile index.
        scoring: The pass's result, carrying per-tile predictions in source pixels.

    Returns:
        The whole-image metrics and how many source images they cover.

    Examples:
        >>> from lucid_yolo.eval.tile_merge import TileIndex
        >>> _whole_image(TileIndex((), ()), SplitScoring({}, 0, 0))[1]
        0
    """
    predictions, ground_truth, names = merge_whole_images(index, scoring.source_predictions)
    return evaluate_rotated_map(predictions, ground_truth, max_detections=None), len(names)


def _report(
    info: dict[str, object],
    split: str,
    img_size: int,
    elapsed: float,
    scoring: SplitScoring,
    whole_image: tuple[dict[str, float], int] | None,
) -> dict[str, object]:
    """Build the JSON report, with each figure named rather than merely present.

    ``metrics`` stays where every previous report put it — the per-tile number, which the
    tier acceptance gate reads — and the whole-image number is its own block carrying the
    detection cap it was scored at, because ``mar_300`` keeps A47's name whatever the cap.

    Args:
        info: The checkpoint loader's provenance dict.
        split: The split scored.
        img_size: The letterbox side.
        elapsed: Seconds the scoring pass took.
        scoring: The pass's result.
        whole_image: The merged metrics and source-image count, or ``None``.

    Returns:
        The report payload.

    Examples:
        >>> _report({}, "val", 64, 1.0, SplitScoring({"map": 0.5}, 2, 3), None)["whole_image"] is None
        True
    """
    return {
        "info": {**info, "split": split, "img_size": img_size, "per_tile": True},
        "tiles": scoring.tiles,
        "instances": scoring.instances,
        "seconds": round(elapsed, 1),
        "metrics": scoring.per_tile,
        "whole_image": None
        if whole_image is None
        else {
            "source_images": whole_image[1],
            "merge": "core-ownership (WP-107)",
            "max_detections": None,
            "metrics": whole_image[0],
        },
    }
