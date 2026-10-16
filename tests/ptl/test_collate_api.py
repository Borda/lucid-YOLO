# SPDX-License-Identifier: Apache-2.0
"""The public collate keeps its pair-only contract while the training loader routes the private union."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import get_type_hints

import torch
from torch import Tensor

from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl.datamodule import (
    DetectionDataModule,
    _collate_training,
    _TrainingSample,
    collate_detection,
)


def test_public_collate_names_only_ordinary_pairs() -> None:
    """The exported signature carries no private sample type; the training twin carries the union."""
    assert get_type_hints(collate_detection)["batch"] == Sequence[tuple[Tensor, Targets]]
    assert get_type_hints(_collate_training)["batch"] == Sequence[_TrainingSample]


def test_ordinary_pairs_collate_identically_through_either_entry() -> None:
    """Delegation keeps the public transport identical to the training path for ordinary pairs."""
    batch = [(torch.rand(3, 8, 8), Targets.empty()) for _ in range(2)]
    public_images, public_packed = collate_detection(batch)
    training_images, training_packed = _collate_training(batch)
    assert public_images.dtype == torch.uint8
    assert torch.equal(public_images, training_images)
    assert torch.equal(public_packed.boxes_per_image, training_packed.boxes_per_image)
    assert public_packed.hsv_gains is None
    assert training_packed.hsv_gains is None


def test_training_loader_routes_through_the_private_collate(tmp_path: Path) -> None:
    """Only the training loader can see deferred samples, so only it uses the widened collate."""
    datamodule = DetectionDataModule(data_root=tmp_path, batch_size=2, num_workers=0, variant="n", img_size=64)
    # Stand-in one-item splits: building a loader reads only a split's length, never its
    # items, and the shuffled train sampler refuses an empty one.
    datamodule._train = [None]  # type: ignore[assignment]
    datamodule._val = [None]  # type: ignore[assignment]
    train_collate = datamodule.train_dataloader().collate_fn
    val_collate = datamodule.val_dataloader().collate_fn
    assert train_collate.func is _collate_training
    assert val_collate.func is collate_detection
    assert train_collate.keywords == {"mask_targets": False}
    assert val_collate.keywords == {"mask_targets": False}
