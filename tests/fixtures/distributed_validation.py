# SPDX-License-Identifier: Apache-2.0
"""Synthetic whole-epoch records and a spawn-safe gloo validation worker."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data.distributed import DistributedSampler

from lucid_yolo.eval.coco_eval import evaluate_keypoints
from lucid_yolo.eval.dota_eval import evaluate_rotated_map
from lucid_yolo.ptl._validation import _KeypointAP, _RotatedMAP

SIGMAS = (0.1, 0.1, 0.1)

#: The geometry entry each task's prediction and target records carry.
_GEOMETRY_KEY = {"obb": "rboxes", "keypoints": "keypoints"}


def validation_records(task: str) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Build hits, misses, a tied false positive, difficult GT and an empty image.

    Args:
        task: ``"obb"`` for rotated boxes, anything else for keypoints.

    Returns:
        Five prediction records and the five target records they are scored against.

    Examples:
        >>> predictions, targets = validation_records("obb")
        >>> len(predictions), len(targets), [len(p["labels"]) for p in predictions]
        (5, 5, [1, 1, 1, 0, 1])
    """
    predictions, targets = [], []
    for index in range(5):
        count = 0 if index == 3 else 1
        labels = torch.zeros(count, dtype=torch.int64)
        scores = torch.full((count,), 0.9 if index == 0 else 0.6)
        if task == "obb":
            geometry = torch.tensor([[20.0, 20.0, 10.0, 6.0, 0.2]])[:count]
            target = {"rboxes": geometry, "labels": labels, "difficult": torch.tensor([index == 4])[:count]}
            prediction = {"rboxes": geometry.clone(), "labels": labels, "scores": scores}
        else:
            geometry = torch.tensor([[[10.0, 10.0], [30.0, 30.0], [50.0, 50.0]]])[:count]
            target = {
                "keypoints": geometry if count else torch.empty(0, 0, 2),
                "visibility": torch.full((count, 3 if count else 0), 2, dtype=torch.int64),
                "labels": labels,
            }
            prediction = {"keypoints": geometry.clone(), "labels": labels, "scores": scores}
        if index == 2:
            prediction[_GEOMETRY_KEY.get(task, "keypoints")] += 200
        predictions.append(prediction)
        targets.append(target)
    return predictions, targets


def list_accumulator_score(
    task: str, preds: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]
) -> dict[str, float]:
    """Score exactly as the former module's append/extend then epoch-end evaluator.

    Args:
        task: ``"obb"`` for rotated mean average precision, anything else for keypoint AP.
        preds: Prediction records for the whole epoch.
        targets: The matching target records.

    Returns:
        The metric dictionary the evaluator for ``task`` reports.

    Examples:
        >>> preds, targets = validation_records("obb")
        >>> sorted(list_accumulator_score("obb", preds, targets)) == sorted(evaluate_rotated_map(preds, targets))
        True
    """
    if task == "obb":
        return evaluate_rotated_map(preds, targets)
    return evaluate_keypoints(preds, targets, sigmas=SIGMAS)


def gloo_validation_worker(rank: int, rendezvous: str, output_dir: str) -> None:
    """Compare padded shards and an empty rank with full-set scoring on each process.

    Args:
        rank: This process's rank in a two-process gloo group.
        rendezvous: ``init_method`` URL all ranks use to find each other.
        output_dir: Directory receiving this rank's ``rank-<n>.json`` result.

    Examples:
        >>> gloo_validation_worker.__name__
        'gloo_validation_worker'
    """
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    results = {}
    try:
        for task in ("obb", "keypoints"):
            preds, targets = validation_records(task)
            metric = _RotatedMAP(dataset_order=True) if task == "obb" else _KeypointAP(SIGMAS, dataset_order=True)
            positions = list(DistributedSampler(range(len(preds)), num_replicas=2, rank=rank, shuffle=False))
            shard = [{name: value.clone() for name, value in preds[index].items()} for index in positions]
            # Rank 1's padded copy of image 0 is deliberately wrong: rank 0's first wins.
            if rank == 1:
                shard[-1][_GEOMETRY_KEY[task]] += 200
            metric.update(shard, [targets[index] for index in positions], positions)
            expected = list_accumulator_score(task, preds, targets)
            actual = metric.compute()
            assert actual == expected, (task, rank, actual, expected)
            assert metric.image_count == len(positions)  # sync must restore local state
            results[task] = actual
            metric.reset()
            metric.update(
                preds if rank == 0 else [], targets if rank == 0 else [], list(range(len(preds))) if rank == 0 else []
            )
            assert metric.compute() == expected, (task, rank, "empty rank")
            results[task + "_empty_rank"] = metric.compute()
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()
