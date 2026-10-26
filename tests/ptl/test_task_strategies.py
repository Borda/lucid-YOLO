# SPDX-License-Identifier: Apache-2.0
"""Pin task strategy coverage and the module's one-time task resolution."""

from __future__ import annotations

import copy
import pickle

import pytest
import torch
from training import tiny_module

from lucid_yolo.ptl._task_strategies import (
    _STRATEGIES,
    _DetectionStrategy,
    _KeypointStrategy,
    _OrientedStrategy,
    _SegmentationStrategy,
)
from lucid_yolo.ptl.module import _oriented_replay_breach
from lucid_yolo.tasks import Task


def test_strategies_cover_tasks() -> None:
    """A new task requires an explicit strategy before a module can train it."""
    assert set(_STRATEGIES) == set(Task)
    assert all(isinstance(task, Task) for task in _STRATEGIES)


@pytest.mark.parametrize("task", list(Task))
def test_strategy_copies_keep_singleton(task: Task) -> None:
    """Copying or unpickling a strategy must resolve the process's registered instance."""
    strategy = _STRATEGIES[task]

    assert copy.copy(strategy) is strategy
    assert copy.deepcopy(strategy) is strategy
    assert pickle.loads(pickle.dumps(strategy)) is strategy


def test_deepcopied_oriented_module_keeps_native_strategy() -> None:
    """A copied module retains the strategy identity required by replay certification."""
    module = tiny_module(task="obb")

    copied = copy.deepcopy(module)

    assert copied._strategy is _STRATEGIES[Task.OBB]
    assert _oriented_replay_breach(copied) is None


@pytest.mark.parametrize(
    ("task", "strategy_type"),
    [
        pytest.param(Task.DETECT, _DetectionStrategy, id="detect"),
        pytest.param(Task.SEGMENT, _SegmentationStrategy, id="segment"),
        pytest.param(Task.OBB, _OrientedStrategy, id="obb"),
        pytest.param(Task.KEYPOINTS, _KeypointStrategy, id="keypoints"),
    ],
)
def test_module_resolves_task_strategy(task: Task, strategy_type: type) -> None:
    """The module selects its task's shared object without registering a new submodule."""
    module = tiny_module(task=task, num_keypoints=3 if task is Task.KEYPOINTS else None)

    assert module._strategy is _STRATEGIES[task]
    assert type(module._strategy) is strategy_type
    assert module.task == task.value
    assert type(module.hparams["task"]) is str
    assert "_strategy" not in dict(module.named_modules())
    assert all(not key.startswith("_strategy.") for key in module.state_dict())


def test_oriented_replay_refuses_replaced_strategy() -> None:
    """A replacement strategy must not bypass the native oriented consumer certificate."""
    module = tiny_module(task="obb")
    assert _oriented_replay_breach(module) is None

    module._strategy = _DetectionStrategy()

    assert _oriented_replay_breach(module) == "the native oriented objective is overridden"


def test_oriented_replay_refuses_changed_strategy_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping oriented terms can remove angle gradients, which full replay materializes."""
    module = tiny_module(task="obb")
    assert _oriented_replay_breach(module) is None

    monkeypatch.setattr(
        _OrientedStrategy, "extra_loss", lambda self, module, context, out, stage: torch.zeros_like(out.total)
    )

    assert _oriented_replay_breach(module) == "a native oriented objective helper changed"
