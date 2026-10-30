# SPDX-License-Identifier: Apache-2.0
"""Check how a copied or reloaded optimizer treats an absent or stale update replay owner, on CPU."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
from tensors import assert_bytes

from lucid_yolo.optim.musgd import MuSGD


def parameters() -> list[torch.nn.Parameter]:
    """Supply one matrix and one vector so both optimizer arms take part.

    Examples:
        >>> [tuple(param.shape) for param in parameters()]
        [(3, 4), (4,)]
    """
    matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
    vector = torch.linspace(-1, 1, 4)
    return [torch.nn.Parameter(matrix), torch.nn.Parameter(vector)]


def gradients(optimizer: MuSGD, step: int) -> None:
    """Assign new gradients without consuming an RNG stream.

    Examples:
        >>> optimizer = MuSGD(parameters(), lr=0.01)
        >>> gradients(optimizer, 0)
        >>> optimizer.param_groups[0]["params"][0].grad.shape
        torch.Size([3, 4])
    """
    for index, param in enumerate(optimizer.param_groups[0]["params"]):
        param.grad = torch.linspace(-1, 1, param.numel()).reshape_as(param) * (step + index + 1) / 10


def assert_state(left: MuSGD, right: MuSGD) -> None:
    """Compare parameters and momentum buffers independently of execution flags.

    Examples:
        >>> assert_state(MuSGD(parameters(), lr=0.01), MuSGD(parameters(), lr=0.01))
    """
    for first, second in zip(left.param_groups[0]["params"], right.param_groups[0]["params"], strict=True):
        assert_bytes(first, second)
        assert left.state.get(first, {}).keys() == right.state.get(second, {}).keys()
        for name, value in left.state.get(first, {}).items():
            assert_bytes(value, right.state[second][name])


class TestUpdateReplayOwner:
    @pytest.mark.parametrize(
        "clone",
        [
            pytest.param(copy.deepcopy, id="deepcopy"),
            pytest.param(lambda optimizer: pickle.loads(pickle.dumps(optimizer)), id="pickle"),
        ],
    )
    def test_copied_update_flag_steps_like_twin(self, clone: Callable[[MuSGD], MuSGD]) -> None:
        """A deep-copied or unpickled flag-on optimizer has no update replay owner yet and continues like its twin.

        ``Optimizer.__getstate__`` keeps only the defaults, state and groups, so the copy starts without the owner
        attribute; both the reload and the next step read it with a default and must give the twin's values. Reading
        it as a plain attribute fails here on CPU, where no CUDA run is needed to reach the copy.
        """
        original, twin = (MuSGD(parameters(), lr=0.01, capture_updates=True) for _ in range(2))
        for optimizer in (original, twin):
            gradients(optimizer, 0)
            optimizer.step()
        copied = clone(original)
        assert "_update_replay" in vars(original)
        assert "_update_replay" not in vars(copied)
        copied.load_state_dict(copy.deepcopy(copied.state_dict()))
        for optimizer in (copied, twin):
            gradients(optimizer, 1)
            optimizer.step()
        assert copied.param_groups[0]["capture_updates"] is True
        assert_state(copied, twin)
        assert copied._update_replay is not None
        assert not copied._update_replay.entries

    def test_reload_clears_planted_recordings(self) -> None:
        """Reloading state empties the owner, whatever recording it held from before the restore.

        A sentinel stands in for a recording because CUDA is the only route that makes a real one; the clear must not
        depend on what the entries are, only on the owner being present.
        """
        optimizer = MuSGD(parameters(), lr=0.01, capture_updates=True)
        gradients(optimizer, 0)
        optimizer.step()
        replay = optimizer._update_replay
        assert replay is not None
        replay.entries.update({1: cast("Any", object()), 2: cast("Any", object())})
        optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        assert not replay.entries
