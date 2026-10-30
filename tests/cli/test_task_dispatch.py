# SPDX-License-Identifier: Apache-2.0
"""Require an explicit CLI dispatch decision whenever a task is added."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from lucid_yolo.cli import eval as eval_cli
from lucid_yolo.cli import predict as predict_cli
from lucid_yolo.cli import train as train_cli
from lucid_yolo.tasks import Task

if TYPE_CHECKING:
    from types import ModuleType


@pytest.mark.parametrize(
    ("module", "table"),
    [
        pytest.param(eval_cli, "_EVAL_TASKS", id="eval-protocols"),
        pytest.param(eval_cli, "_EVAL_OPTIONS", id="eval-options"),
        pytest.param(predict_cli, "_PREDICT_TASKS", id="predict-handlers"),
        pytest.param(train_cli, "_TARGET_FIELD", id="train-target-fields"),
    ],
)
def test_dispatch_table_covers_every_task(module: ModuleType, table: str) -> None:
    """Every task has an explicit entry in each command's dispatch table.

    A new task cannot inherit the detection protocol or its accepted options from ``lucid-eval``, needs an explicit
    report-producing handler in ``lucid-predict``, and cannot silently disable all loader target types in
    ``lucid-yolo``.
    """
    assert set(getattr(module, table)) == set(Task)


@pytest.mark.parametrize(
    ("task", "field"),
    [
        pytest.param(Task.DETECT, None, id="detect"),
        pytest.param(Task.SEGMENT, "mask_targets", id="segment"),
        pytest.param(Task.OBB, "rotated_targets", id="obb"),
        pytest.param(Task.KEYPOINTS, "keypoint_targets", id="keypoints"),
    ],
)
def test_train_target_field_keeps_supervision(task: Task, field: str | None) -> None:
    """String task inputs select exactly the existing loader supervision field."""
    assert train_cli._TARGET_FIELD[task.value] == field
