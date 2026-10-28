# SPDX-License-Identifier: Apache-2.0
"""Exact whole-epoch scoring, sampler deduplication and CPU distributed reduction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
import torch.multiprocessing as mp
from distributed_validation import SIGMAS, gloo_validation_worker, list_accumulator_score, validation_records
from lightning_fabric.utilities.apply_func import move_data_to_device
from training import tiny_module

from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl._transport import collate_detection, unpack_targets
from lucid_yolo.ptl._validation import _KeypointAP, _RotatedMAP, _ValidationDataset, _ValidationTargets

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("task", ["obb", "keypoints"])
def test_single_process_matches_list_accumulator(task: str) -> None:
    """The old accumulator's exact floats survive multiple batches and epoch resets."""
    preds, targets = validation_records(task)
    metric = _RotatedMAP() if task == "obb" else _KeypointAP(SIGMAS)
    image_ids = [41, 3, 17, 2, 9]
    for start, end in ((0, 2), (2, 5)):
        metric.update(preds[start:end], targets[start:end], image_ids[start:end])
    assert metric.compute() == list_accumulator_score(task, preds, targets)
    assert 0.0 < metric.compute()["map" if task == "obb" else "AP_all"] < 1.0
    stored_preds, stored_targets = metric._images()
    for original, stored in zip(preds + targets, stored_preds + stored_targets, strict=True):
        for name, value in original.items():
            assert torch.equal(value, stored[name])
            assert value.dtype == stored[name].dtype
    metric.reset()
    assert metric.image_count == 0
    assert metric._images() == ([], [])
    metric.update(preds, targets, image_ids)
    assert metric.compute() == list_accumulator_score(task, preds, targets)


@pytest.mark.parametrize("task", ["obb", "keypoints"])
def test_deduplication_keeps_first_image_and_input_order(task: str) -> None:
    """Arbitrary IDs never reorder tied scores, and a conflicting later copy is discarded."""
    preds, targets = validation_records(task)
    metric = _RotatedMAP() if task == "obb" else _KeypointAP(SIGMAS)
    metric.update(preds, targets, [41, 3, 17, 2, 9])
    metric.update([preds[2]], [targets[0]], [41])
    assert metric.compute() == list_accumulator_score(task, preds, targets)
    assert len(metric._images()[0]) == len(preds)
    assert torch.equal(metric._images()[0][0]["scores"], preds[0]["scores"])


@pytest.mark.parametrize("task", ["obb", "keypoints"])
def test_local_module_metric_keeps_supplied_order(task: str) -> None:
    """Dataset-order restoration applies only to gathered shards, never a single-device batch."""
    preds, targets = validation_records(task)
    metric = _RotatedMAP(dataset_order=True) if task == "obb" else _KeypointAP(SIGMAS, dataset_order=True)
    metric.update(preds, targets, [41, 3, 17, 2, 9])
    assert metric.compute() == list_accumulator_score(task, preds, targets)
    stored = metric._images()[0]
    assert [entry["scores"].tolist() for entry in stored] == [entry["scores"].tolist() for entry in preds]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_record_transport_preserves_tensor_precision(dtype: torch.dtype) -> None:
    """Byte packing preserves original dtype and values rather than promoting precision."""
    preds, targets = validation_records("obb")
    preds[0]["rboxes"] = preds[0]["rboxes"].to(dtype)
    metric = _RotatedMAP()
    metric.update(preds, targets, list(range(len(preds))))
    restored = metric._images()[0][0]["rboxes"]
    assert restored.dtype == dtype
    assert torch.equal(restored, preds[0]["rboxes"])


def test_gloo_matches_full_validation_set(tmp_path: Path) -> None:
    """Two real CPU processes gather padded shards and tolerate an empty rank for both tasks."""
    rendezvous = (tmp_path / "gloo-init").as_uri()
    mp.spawn(gloo_validation_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True)
    first = json.loads((tmp_path / "rank-0.json").read_text())
    second = json.loads((tmp_path / "rank-1.json").read_text())
    assert first == second
    for task in ("obb", "keypoints"):
        expected = list_accumulator_score(task, *validation_records(task))
        assert first[task] == expected
        assert first[task + "_empty_rank"] == expected


def test_validation_identity_survives_worker_and_device_transport() -> None:
    """Positions survive packing and Lightning dataclass reconstruction without mutating cached targets."""
    original = Targets.empty()
    dataset = _ValidationDataset([(torch.zeros(3, 8, 8), original)] * 2)
    _, packed = collate_detection([dataset[1], dataset[0], dataset[1]])
    moved = move_data_to_device(packed, torch.device("cpu"))
    restored = unpack_targets(moved)
    assert [target._image_id for target in restored] == [1, 0, 1]
    assert all(isinstance(target, _ValidationTargets) for target in restored)
    assert not isinstance(original, _ValidationTargets)
    assert not hasattr(original, "_image_id")


@pytest.mark.parametrize("task", ["obb", "keypoints"])
def test_module_metric_retains_validation_identity(task: str) -> None:
    """The module passes transported IDs to the task metric rather than assigning batch positions."""
    module = tiny_module(task=task, **({"num_keypoints": 3} if task == "keypoints" else {})).eval()
    metric = module._val_rotated if task == "obb" else module._val_keypoints
    preds, targets = validation_records(task)
    identities = [_ValidationTargets(**vars(Targets.empty()), _image_id=position) for position in (4, 2)]
    metric.update(preds[:2], targets[:2], module._validation_ids(identities, metric))
    assert metric._metadata[-1][0] == 2
    assert module._validation_ids([Targets.empty()], metric) == [2]


@pytest.mark.parametrize("task", ["obb", "keypoints"])
@pytest.mark.parametrize("empty_rank", [False, True])
def test_cat_reduction_restores_dataset_order_and_local_state(task: str, empty_rank: bool) -> None:
    """Exercise torchmetrics' cat reduction with a supplied collective boundary, then unsync."""
    preds, targets = validation_records(task)
    first = _RotatedMAP(dataset_order=True) if task == "obb" else _KeypointAP(SIGMAS, dataset_order=True)
    second = _RotatedMAP(dataset_order=True) if task == "obb" else _KeypointAP(SIGMAS, dataset_order=True)
    positions = [] if empty_rank else [0, 2, 4]
    other_positions = list(range(5)) if empty_rank else [1, 3, 0]
    first.update([preds[index] for index in positions], [targets[index] for index in positions], positions)
    second.update(
        [preds[index] for index in other_positions], [targets[index] for index in other_positions], other_positions
    )

    def _gather(value: torch.Tensor, group: object = None) -> list[torch.Tensor]:
        del group
        state = second._metadata if value.dtype == torch.int64 else second._values
        return [value, torch.cat(state)]

    first.sync(dist_sync_fn=_gather, distributed_available=lambda: True)
    assert list_accumulator_score(task, *first._images()) == list_accumulator_score(task, preds, targets)
    first.unsync()
    assert first.image_count == len(positions)
    assert len(first._images()[0]) == len(positions)
