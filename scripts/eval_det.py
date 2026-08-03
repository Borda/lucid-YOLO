# SPDX-License-Identifier: Apache-2.0
"""Dual-path detection evaluation of a training checkpoint on COCO val2017 (WP-045).

Drives :class:`lucid_yolo.eval.coco_eval.DualPathEvaluator` end to end: loads a
Lightning checkpoint (optionally overlaying the EMA shadow the
:class:`~lucid_yolo.ptl.callbacks.EMACallback` stored inside it), letterboxes
val2017, runs the model once per batch, decodes both the suppression-free E2E
path and the NMS path, and prints the twelve COCO summary statistics for each.

The ground truth is read straight from ``instances_val2017.json`` in original
image coordinates, including ``iscrowd`` flags and annotation areas so the
crowd-ignore rule and the small/medium/large breakdown follow the COCO protocol.

Usage::

    python scripts/eval_det.py CHECKPOINT --data-root ~/data/coco2017 \
        [--no-ema] [--batch-size 32] [--img-size 640] [--device mps] \
        [--output report.json]

Provenance: R1 Table 7, R1 sec. 4.4 (dual-path protocol). Assumptions: A9, A10.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torchvision.io import ImageReadMode, read_image

from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.targets import Targets
from lucid_yolo.decode.nms_path import NMSDecoder
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.coco_eval import DualPathEvaluator
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from torch import Tensor

_UINT8_MAX = 255.0
_XYWH_LEN = 4


@dataclass(frozen=True)
class _EvalImage:
    """One val-split image: identity plus original geometry."""

    image_id: int
    file_name: str
    height: int
    width: int


def _load_annotations(ann_file: Path) -> tuple[list[_EvalImage], dict[int, dict[str, Tensor]], dict[int, int]]:
    """Parse a COCO instances file into eval images, target dicts, and the label map.

    Args:
        ann_file: Path to ``instances_val2017.json`` (or a compatible subset).

    Returns:
        A triple of the image records sorted by ascending image id, the
        torchmetrics ground-truth dict per image id (``boxes`` ``xyxy`` in
        original coordinates, ``labels`` as category ids, plus ``iscrowd`` and
        ``area``), and the contiguous-label -> category-id mapping (matching
        :class:`~lucid_yolo.data.coco.CocoDetectionDataset`'s sorted-id order).
    """
    payload = json.loads(ann_file.read_text())
    sorted_ids = sorted(int(cat["id"]) for cat in payload["categories"])
    label_to_category = dict(enumerate(sorted_ids))
    images = sorted(
        (
            _EvalImage(
                image_id=int(img["id"]),
                file_name=str(img["file_name"]),
                height=int(img["height"]),
                width=int(img["width"]),
            )
            for img in payload["images"]
        ),
        key=lambda record: record.image_id,
    )
    targets: dict[int, dict[str, Tensor]] = {
        record.image_id: {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
            "iscrowd": torch.zeros((0,), dtype=torch.long),
            "area": torch.zeros((0,), dtype=torch.float32),
        }
        for record in images
    }
    grouped: dict[int, list[dict[str, object]]] = {}
    for ann in payload["annotations"]:
        grouped.setdefault(int(ann["image_id"]), []).append(ann)
    for image_id, anns in grouped.items():
        boxes, labels, iscrowd, area = [], [], [], []
        for ann in anns:
            raw_bbox = ann["bbox"]
            if not isinstance(raw_bbox, list):
                continue
            bbox = [float(value) for value in raw_bbox]
            if len(bbox) != _XYWH_LEN or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]))  # type: ignore[call-overload]
            iscrowd.append(int(ann.get("iscrowd", 0)))  # type: ignore[call-overload]
            area.append(float(ann.get("area", w * h)))  # type: ignore[arg-type]
        if boxes:
            targets[image_id] = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
                "iscrowd": torch.tensor(iscrowd, dtype=torch.long),
                "area": torch.tensor(area, dtype=torch.float32),
            }
    return images, targets, label_to_category


def _batches(
    images: Sequence[_EvalImage],
    images_dir: Path,
    letterbox: Letterbox,
    batch_size: int,
) -> Iterator[tuple[Tensor, list[int], list[tuple[int, int]]]]:
    """Yield ``(images, image_ids, orig_sizes)`` batches for the evaluator.

    Args:
        images: The eval image records to iterate, in order.
        images_dir: Directory holding the image files.
        letterbox: The validation letterbox applied to every image.
        batch_size: Number of images per yielded batch.

    Yields:
        Batches matching the :class:`DualPathEvaluator` dataloader contract.
    """
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        tensors = []
        for record in chunk:
            raw = read_image(str(images_dir / record.file_name), ImageReadMode.RGB)
            boxed, _ = letterbox(raw.to(torch.float32) / _UINT8_MAX, Targets.empty())
            tensors.append(boxed)
        yield (
            torch.stack(tensors),
            [record.image_id for record in chunk],
            [(record.height, record.width) for record in chunk],
        )


def _load_module(checkpoint: Path, use_ema: bool) -> tuple[DetectionLitModule, dict[str, object]]:
    """Load the LightningModule from ``checkpoint``, optionally with EMA weights.

    Args:
        checkpoint: Path to the Lightning ``.ckpt`` file.
        use_ema: When ``True``, overlay the :class:`EMACallback` shadow stored in
            the checkpoint onto the module's parameters and buffers.

    Returns:
        The eval-mode module and a small provenance dict (epoch, step, EMA use).

    Raises:
        ValueError: If ``use_ema`` is requested but the checkpoint carries no
            EMA shadow.
    """
    module = DetectionLitModule.load_from_checkpoint(checkpoint, map_location="cpu")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    info: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "ema": use_ema,
    }
    if use_ema:
        shadow = None
        for name, state in payload.get("callbacks", {}).items():
            if "EMACallback" in name:
                shadow = state.get("shadow")
        if not shadow:
            raise ValueError(f"--ema requested but {checkpoint} carries no EMACallback shadow")
        tensors: dict[str, Tensor] = {name: param for name, param in module.named_parameters()}
        tensors.update(module.named_buffers())
        with torch.no_grad():
            for name, value in shadow.items():
                tensors[name].copy_(value)
        info["ema_updates"] = int(payload["callbacks"]["EMACallback"]["num_updates"])
    module.eval()
    return module, info


def _pick_device(requested: str) -> torch.device:
    """Resolve ``auto`` to the fastest available backend, else pass through."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the dual-path COCO evaluation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="Lightning .ckpt to evaluate")
    parser.add_argument("--data-root", type=Path, required=True, help="COCO root with val2017/ and annotations/")
    parser.add_argument("--no-ema", dest="ema", action="store_false", help="evaluate raw weights, not the EMA shadow")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (0 = all)")
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    ann_file = args.data_root / "annotations" / "instances_val2017.json"
    images_dir = args.data_root / "val2017"
    images, targets, label_to_category = _load_annotations(ann_file)
    if args.limit:
        images = images[: args.limit]
    module, info = _load_module(args.checkpoint, use_ema=args.ema)
    device = _pick_device(args.device)
    letterbox = Letterbox(args.img_size)
    evaluator = DualPathEvaluator(module, TopKDecoder(), NMSDecoder(), label_to_category, letterbox)

    print(f"eval: {len(images)} images, device={device.type}, ema={args.ema}, img_size={args.img_size}")
    start = time.perf_counter()
    report = evaluator.evaluate(_batches(images, images_dir, letterbox, args.batch_size), targets, device)
    elapsed = time.perf_counter() - start
    print(f"done in {elapsed:.1f}s ({len(images) / elapsed:.1f} img/s)")

    for path in ("e2e", "nms"):
        stats = report[path]
        print(f"[{path}] mAP50-95={stats['map']:.4f} mAP50={stats['map_50']:.4f} mAP75={stats['map_75']:.4f}")
    delta = report["nms"]["map"] - report["e2e"]["map"]
    print(f"E2E deficit vs NMS: {delta * 100:.2f} AP")

    if args.output:
        payload = {"info": info, "images": len(images), "seconds": round(elapsed, 1), "report": report}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"report -> {args.output}")


if __name__ == "__main__":
    main()
