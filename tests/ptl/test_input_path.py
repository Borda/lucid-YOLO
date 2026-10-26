# SPDX-License-Identifier: Apache-2.0
"""Byte identity and synchronization boundaries of packed input restoration."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from input_path import input_samples
from training import tiny_module

from lucid_yolo.ptl import DetectionDataModule, pad_keypoints, pad_rboxes, pad_targets
from lucid_yolo.ptl._padding import _pad_mask_targets
from lucid_yolo.ptl.datamodule import collate_detection, unpack_masks


def test_transfer_hook_reads_no_tensor_counts() -> None:
    """Host metadata must survive collate without reading the transported counts."""
    images, packed = collate_detection(input_samples((0, 2, 1)), mask_targets=True)
    count_ids = {
        id(getattr(packed, name))
        for name in ("boxes_per_image", "rboxes_per_image", "keypoints_per_image", "rings_per_image", "points_per_ring")
    }
    original_tolist, original_item = torch.Tensor.tolist, torch.Tensor.item
    calls = []

    def tolist(tensor):
        if tensor.device.type != "cpu" or id(tensor) in count_ids:
            calls.append("tolist")
        return original_tolist(tensor)

    def item(tensor, *args):
        if tensor.device.type != "cpu" or id(tensor) in count_ids:
            calls.append("item")
        return original_item(tensor, *args)

    with patch.object(torch.Tensor, "tolist", tolist), patch.object(torch.Tensor, "item", item):
        DetectionDataModule(Path("."), 2, 0, "n").on_after_batch_transfer((images, packed), 0)
    assert calls == []


@pytest.mark.parametrize("counts", [pytest.param((0, 2, 1, 0), id="mixed"), pytest.param((0, 0), id="empty")])
def test_packed_padding_equals_ragged_padding(counts: tuple[int, ...]) -> None:
    """All fields retain their old values, shapes, padding and dtypes."""
    samples = input_samples(counts)
    raw = [target for _, target in samples]
    images, packed = collate_detection(samples, mask_targets=True)
    _, restored, masks = DetectionDataModule(Path("."), 2, 0, "n").on_after_batch_transfer((images, packed), 0)
    expected = (*pad_targets(raw), pad_rboxes(raw), *pad_keypoints(raw))
    actual = (*pad_targets(restored), pad_rboxes(restored), *pad_keypoints(restored))
    for value, reference in zip(actual, expected, strict=True):
        assert value.dtype == reference.dtype
        assert torch.equal(value, reference)
    assert masks is not None
    for value, reference in zip(masks, unpack_masks(packed), strict=True):
        assert torch.equal(value, reference)
    reference = torch.zeros(1, 1, 8, 8)
    assert torch.equal(_pad_mask_targets(masks, restored, reference), _pad_mask_targets(masks, raw, reference))


@pytest.mark.parametrize("mutation", ["reverse", "replace", "in_place"])
def test_packed_padding_keeps_current_targets(mutation: str) -> None:
    """List edits fall back; edits through a tensor view update flat storage."""
    images, packed = collate_detection(input_samples((1, 2)), mask_targets=True)
    _, targets, masks = DetectionDataModule(Path("."), 2, 0, "n").on_after_batch_transfer((images, packed), 0)
    if mutation == "reverse":
        targets.reverse()
    elif mutation == "replace":
        targets[0].boxes = torch.full_like(targets[0].boxes, 5.0)
        targets[0].keypoints = torch.ones_like(targets[0].keypoints)
    else:
        targets[0].boxes.add_(1)
        targets[0].keypoints.add_(1)
    raw = list(targets)
    for actual, expected in zip(
        (*pad_targets(targets), pad_rboxes(targets), *pad_keypoints(targets)),
        (*pad_targets(raw), pad_rboxes(raw), *pad_keypoints(raw)),
        strict=True,
    ):
        assert torch.equal(actual, expected)
    assert masks is not None
    masks[0].fill_(0.25)
    assert torch.equal(_pad_mask_targets(masks, targets, masks[0]), _pad_mask_targets(masks, raw, masks[0]))


def test_manual_packed_transport_keeps_ragged_fallback() -> None:
    """Public transports without private collate metadata keep ordinary padding."""
    images, packed = collate_detection(input_samples((0, 2)))
    manual = replace(packed, _host_counts={}, _padding_indices=None)
    _, targets, _ = DetectionDataModule(Path("."), 2, 0, "n").on_after_batch_transfer((images, manual), 0)
    for actual, expected in zip(pad_targets(targets), pad_targets(list(targets)), strict=True):
        assert torch.equal(actual, expected)


def test_packed_padding_scatters_each_field_once() -> None:
    """Each flat field reaches its dense tensor in one device write, regardless of batch size."""
    images, packed = collate_detection(input_samples((0, 2, 1, 0)), mask_targets=True)
    _, targets, masks = DetectionDataModule(Path("."), 2, 0, "n").on_after_batch_transfer((images, packed), 0)
    assert masks is not None
    calls = []
    original = torch.Tensor.index_copy_

    def index_copy(tensor, dim, indices, values):
        calls.append(tuple(values.shape))
        return original(tensor, dim, indices, values)

    with patch.object(torch.Tensor, "index_copy_", index_copy):
        pad_targets(targets)
        pad_rboxes(targets)
        pad_keypoints(targets)
        _pad_mask_targets(masks, targets, masks[0])
    assert calls == [(3, 4), (3,), (3, 5), (3, 17, 2), (3, 17), (3, 8, 8)]


@pytest.mark.parametrize("task", ["detect", "segment", "obb", "keypoints"])
def test_packed_step_loss_equals_ragged_step(task: str) -> None:
    """The real per-task objective sees identical padded inputs and mask targets."""
    torch.manual_seed(214)
    module = tiny_module(task, **({"num_keypoints": 17} if task == "keypoints" else {})).eval()
    module.log = lambda *args, **kwargs: None
    samples = input_samples((0, 2), size=64)
    images, packed = collate_detection(samples, mask_targets=task == "segment")
    batch = DetectionDataModule(Path("."), 2, 0, "n").on_after_batch_transfer((images, packed), 0)
    with torch.no_grad():
        actual = module._shared_step(batch, "val")[0]
        expected = module._shared_step((batch[0], list(batch[1]), batch[2]), "val")[0]
    assert torch.equal(actual, expected)
