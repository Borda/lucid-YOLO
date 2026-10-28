# SPDX-License-Identifier: Apache-2.0
"""Tensor helpers shared by test modules that compare bytes or draw from a seeded generator."""

from __future__ import annotations

import torch
from torch import Tensor


def seeded_generator(seed: int = 1234) -> torch.Generator:
    """Return a CPU generator seeded to ``seed`` for reproducible sampling.

    Examples:
        >>> seeded_generator(7).initial_seed()
        7
        >>> torch.rand(1, generator=seeded_generator()).item()
        0.028979241847991943
    """
    return torch.Generator().manual_seed(seed)


def assert_bytes(left: Tensor, right: Tensor) -> None:
    """Compare numerical storage without rounding or hiding signed zero.

    Examples:
        >>> assert_bytes(torch.tensor([-0.0]), torch.tensor([-0.0]))
    """
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert torch.equal(
        left.detach().contiguous().reshape(-1).view(torch.uint8),
        right.detach().contiguous().reshape(-1).view(torch.uint8),
    )
