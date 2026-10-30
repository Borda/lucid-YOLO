# SPDX-License-Identifier: Apache-2.0
"""Compare packed training input restoration and 300-mask decode across revisions.

Measures the same input work on two revisions, without a model, so a before/after pair on one device shows what
WP-214 changed. Synthetic host collation is untimed; each timed sample covers the device transfer, the datamodule
restore hook and the public box, rotated and point padding. The decode sample always scores 300 instances, the
evaluation cap. Every entry point it calls exists on WP-213 as well, so the same file runs on both revisions.

It prints one JSON line: median wall times in milliseconds and the CUDA peak allocated memory in MiB, ``null`` on
CPU and MPS. Invalid arguments and an unavailable device raise normally. Usage::

    python scripts/bench_input_path.py --device cpu --iterations 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch import Tensor

from lucid_yolo.data.targets import Targets
from lucid_yolo.eval.segment_decode import decode_instance_masks
from lucid_yolo.ptl import DetectionDataModule, pad_keypoints, pad_rboxes, pad_targets
from lucid_yolo.ptl.datamodule import collate_detection


def _synchronize(device: torch.device) -> None:
    """Include completion of accelerator work in wall-clock measurements."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _samples(batch_size: int, instances: int, size: int) -> list[tuple[Tensor, Targets]]:
    """Build a ragged host batch with every transported training modality."""
    samples = []
    for index in range(batch_size):
        count = instances if index % 3 else 0
        box = torch.tensor([[0.0, 0.0, float(size), float(size)]])
        ring = torch.tensor([[0.0, 0.0], [float(size), 0.0], [float(size), float(size)], [0.0, float(size)]])
        target = (
            Targets(
                boxes=box.repeat(count, 1),
                labels=torch.arange(count) % 4,
                rboxes=torch.tensor([[size / 2, size / 2, float(size), float(size), 0.0]]).repeat(count, 1),
                polygons=[ring.clone() for _ in range(count)],
                keypoints=torch.zeros(count, 17, 2),
                keypoint_vis=torch.ones(count, 17, dtype=torch.int64),
            )
            if count
            else Targets.empty()
        )
        samples.append((torch.rand(3, size, size), target))
    return samples


def _measure(args: argparse.Namespace, device: torch.device) -> dict[str, str | float | None]:
    """Measure synchronized input iterations and independent full-canvas decodes."""
    torch.manual_seed(214)
    dm = DetectionDataModule(Path("."), args.batch_size, 0, "n")
    batch = collate_detection(_samples(args.batch_size, args.instances, args.img_size), mask_targets=True)
    transfer_times = []
    for iteration in range(args.warmup + args.iterations):
        _synchronize(device)
        start = time.perf_counter()
        moved = dm.transfer_batch_to_device(batch, device, 0)
        restored = dm.on_after_batch_transfer(moved, 0)
        targets = restored[1]
        pad_targets(targets)
        pad_rboxes(targets)
        pad_keypoints(targets)
        _synchronize(device)
        if iteration >= args.warmup:
            transfer_times.append((time.perf_counter() - start) * 1000)
        del moved, restored, targets
    prototypes = torch.randn(1, 32, args.img_size // 4, args.img_size // 4, device=device)
    coefficients = torch.randn(1, 300, 32, device=device)
    boxes = torch.tensor([0.0, 0.0, float(args.img_size), float(args.img_size)], device=device).repeat(1, 300, 1)
    decode_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(args.warmup + args.decode_iterations):
        _synchronize(device)
        start = time.perf_counter()
        masks = decode_instance_masks(prototypes, coefficients, boxes, (args.img_size, args.img_size))
        _synchronize(device)
        if iteration >= args.warmup:
            decode_times.append((time.perf_counter() - start) * 1000)
        del masks
    return {
        "device": str(device),
        "transfer_ms_median": statistics.median(transfer_times),
        "decode_ms_median": statistics.median(decode_times),
        "decode_peak_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
    }


def main(argv: list[str] | None = None) -> int:
    """Print a JSON benchmark record using runtime APIs shared with the parent tree.

    Examples:
        >>> main(["--device", "cpu", "--iterations", "1", "--decode_iterations", "1",
        ...       "--warmup", "0", "--batch_size", "2", "--instances", "1", "--img_size", "4"])  # doctest: +ELLIPSIS
        {"device": "cpu", ...}
        0
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--iterations", type=int, default=20, help="Timed training input iterations (default: 20)")
    parser.add_argument("--decode_iterations", type=int, default=5, help="Timed 300-mask decodes (default: 5)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=640)
    args = parser.parse_args(argv)
    if min(args.iterations, args.decode_iterations, args.batch_size) < 1 or min(args.warmup, args.instances) < 0:
        parser.error("iterations, decode_iterations and batch_size must be positive; warmup and instances nonnegative")
    if args.img_size < 4 or args.img_size % 4:
        parser.error("img_size must be a positive multiple of 4")
    selected = args.device
    if selected == "auto":
        selected = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    with torch.no_grad():
        result = _measure(args, torch.device(selected))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
