# SPDX-License-Identifier: Apache-2.0
"""Require an explicit CLI dispatch decision whenever a task is added."""

from __future__ import annotations

import pytest

from lucid_yolo.cli import eval as eval_cli
from lucid_yolo.cli import predict as predict_cli
from lucid_yolo.cli import train as train_cli
from lucid_yolo.tasks import Task


@pytest.mark.parametrize("name", ["_EVAL_TASKS", "_EVAL_OPTIONS"])
def test_eval_tables_cover_tasks(name: str) -> None:
    """A new task cannot inherit the detection protocol or its accepted options."""
    assert set(getattr(eval_cli, name)) == set(Task)


def test_predict_table_covers_tasks() -> None:
    """Every prediction task needs an explicit report-producing handler."""
    assert set(predict_cli._PREDICT_TASKS) == set(Task)


def test_train_table_covers_tasks() -> None:
    """A new task cannot silently disable all loader target types."""
    assert set(train_cli._TARGET_FIELD) == set(Task)


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
