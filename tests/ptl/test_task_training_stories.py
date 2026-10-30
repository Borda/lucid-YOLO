# SPDX-License-Identifier: Apache-2.0
"""Shared training stories whose task-specific differences are inputs and expected branches.

Assignment-order oracles and OBB's empty-batch backward connectivity stay in the
per-task suites because their actions and assertions differ, not just their data.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from training import LogRecorder, has_gradient, keypoint_targets, oriented_targets, segment_targets, tiny_module

from lucid_yolo.data.targets import Targets

_GAINS = {"box_gain": 7.5, "cls_gain": 0.5, "l1_gain": 6.0, "alpha": 0.5}
_KEYPOINT_OPTIONS = {"num_keypoints": 3, "keypoint_gain": 1.0, "keypoint_loss": "rle"}
_OBB_OPTIONS = {"angle_gain": 1.0}


@pytest.mark.parametrize(
    ("task", "options", "target_builder", "branches"),
    [
        pytest.param(
            "segment",
            {},
            segment_targets,
            ("protonet", "proto_fusion", "semantic", "head.o2o.coeff_stems", "head.o2m.coeff_stems"),
            id="segment",
        ),
        pytest.param(
            "obb",
            _OBB_OPTIONS,
            oriented_targets,
            ("head.o2o.angle_stems", "head.o2m.angle_stems"),
            id="obb",
        ),
        pytest.param(
            "keypoints",
            _KEYPOINT_OPTIONS,
            keypoint_targets,
            ("head.o2o.keypoint_stems", "head.o2m.keypoint_stems", "rle_loss"),
            id="keypoints",
        ),
    ],
)
def test_task_gradients_reach_every_branch(
    task: str, options: dict[str, Any], target_builder: Callable[[int], Targets], branches: tuple[str, ...]
) -> None:
    """Backward reaches every original task branch, including the non-null pose flow.

    Missing mask coefficients, angle stems or point stems leave deployed branches untrained. The pose flow must also
    exist and receive a non-zero gradient. The original two-image batch has ragged instance counts of two and one;
    oriented targets remain elongated and rotated to avoid stationary angles.
    """
    module = tiny_module(task, **_GAINS, **options)
    module.log = LogRecorder()  # type: ignore[method-assign]
    images = torch.randn(2, 3, 160, 160)
    targets = [target_builder(2), target_builder(1)]

    module.training_step((images, targets), 0).backward()

    for name in branches:
        branch = module.get_submodule(name)
        assert branch is not None, name
        assert has_gradient(branch), name


@pytest.mark.parametrize(
    ("task", "options", "term"),
    [
        pytest.param("segment", {}, "train/mask", id="segment"),
        pytest.param("keypoints", _KEYPOINT_OPTIONS, "train/keypoint", id="keypoints"),
    ],
)
def test_task_image_without_instances_gives_finite_zero_term(task: str, options: dict[str, Any], term: str) -> None:
    """Empty targets yield a finite total and an exactly-zero task term.

    An empty instance axis must not introduce a mean-of-empty NaN or attempt a
    gather from length-zero padded ground truth.
    """
    module = tiny_module(task, **_GAINS, **options)
    recorder = LogRecorder()
    module.log = recorder  # type: ignore[method-assign]

    loss = module.training_step((torch.randn(1, 3, 160, 160), [Targets.empty()]), 0)

    assert bool(torch.isfinite(loss))
    assert recorder.values[term] == 0.0


@pytest.mark.parametrize(
    ("task", "options", "stem", "extra_prefixes"),
    [
        pytest.param("obb", _OBB_OPTIONS, ".angle_stems.", (), id="obb"),
        pytest.param("keypoints", _KEYPOINT_OPTIONS, ".keypoint_stems.", ("rle_loss.",), id="keypoints"),
    ],
)
def test_task_state_dict_keeps_detection_keys(
    task: str, options: dict[str, Any], stem: str, extra_prefixes: tuple[str, ...]
) -> None:
    """Removing task keys leaves detection keys in exactly their original order.

    Task stages must stay flat so accepted detection checkpoints keep loading;
    each task must also retain its own branch stems.
    """
    module = tiny_module(task, **_GAINS, **options)
    detection = tiny_module("detect", **_GAINS)

    keys = [key for key in module.state_dict() if stem not in key and not key.startswith(extra_prefixes)]

    assert keys == list(detection.state_dict())
    assert any(stem in key for key in module.state_dict())
