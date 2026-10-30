# SPDX-License-Identifier: Apache-2.0
"""Pose loss pairing steps aside when the point-density loss is altered."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import Trainer, seed_everything

from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl.module import DetectionLitModule


def _pose() -> DetectionLitModule:
    """Build an opt-in pose module whose loss replay is switched on as a CUDA run would.

    Examples:
        >>> _pose()._capture_loss_active
        True
    """
    model = DetectionLitModule(0.5, 0.25, 64, 2, task="keypoints", num_keypoints=3, capture_loss=True)
    model._capture_loss_active = True
    model._trainer = MagicMock(spec=Trainer, training=True)
    return model


def _batch() -> tuple[torch.Tensor, list[Targets]]:
    """Build one empty image beside one image with two fully labelled people.

    Examples:
        >>> [len(target.boxes) for target in _batch()[1]]
        [0, 2]
    """
    boxes = torch.tensor([[4.0, 6.0, 36.0, 42.0], [24.0, 18.0, 59.0, 62.0]])
    points = torch.tensor([[[10.0, 12.0], [18.0, 20.0], [28.0, 30.0]], [[30.0, 24.0], [40.0, 32.0], [50.0, 52.0]]])
    targets = [
        Targets(
            boxes=boxes[:n].clone(),
            labels=torch.arange(n),
            keypoints=points[:n].clone(),
            keypoint_vis=torch.full((n, 3), 2, dtype=torch.long),
        )
        for n in (0, 2)
    ]
    return torch.rand(2, 3, 64, 64), targets


def test_pose_step_pairs_density_calls_for_unaltered_loss() -> None:
    """Control: the opt-in pose step batches the two density calls when nothing observes the loss.

    Without this, the refusal cases below would pass on a step that never pairs at all.
    """
    seed_everything(5)
    model = _pose()
    with (
        patch.object(model, "log"),
        patch.object(model, "_paired_keypoint_terms", wraps=model._paired_keypoint_terms) as paired,
    ):
        model.training_step(_batch(), 0)
    assert paired.call_count == 1


@pytest.mark.parametrize(
    "register",
    [
        pytest.param(lambda loss: loss.register_forward_hook(lambda *args: None), id="forward_hook"),
        pytest.param(lambda loss: loss.register_forward_pre_hook(lambda *args: None), id="forward_pre_hook"),
        pytest.param(lambda loss: loss.register_full_backward_hook(lambda *args: None), id="backward_hook"),
        pytest.param(lambda loss: loss.register_full_backward_pre_hook(lambda *args: None), id="backward_pre_hook"),
        pytest.param(
            lambda loss: next(loss.parameters()).register_post_accumulate_grad_hook(lambda parameter: None),
            id="post_accumulate_grad_hook",
        ),
    ],
)
def test_pose_step_keeps_unpaired_scoring_when_density_loss_is_hooked(
    register: Callable[[torch.nn.Module], object],
) -> None:
    """A hook the user put on the pose module's `rle_loss` keeps its two original calls.

    The user observes each density call separately, so batching the two branches into one call would change what the
    hook sees. The step must fall back to the per-branch scoring path instead of the shared batch.
    """
    seed_everything(5)
    model = _pose()
    register(model.rle_loss)
    with (
        patch.object(model, "log"),
        patch.object(model, "_paired_keypoint_terms", wraps=model._paired_keypoint_terms) as paired,
    ):
        loss = model.training_step(_batch(), 0)
    assert paired.call_count == 0 and torch.isfinite(loss)
