# SPDX-License-Identifier: Apache-2.0
"""Deterministic-kernel policy for tests that compare eager runs with recorded or replayed ones."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def deterministic_algorithms(*, warn_only: bool = False, restore_cudnn_benchmark: bool = False) -> Iterator[None]:
    """Run under deterministic algorithms and give back the caller's numeric policy afterwards.

    Both flags are saved and restored, so a test that fails inside the block still
    leaves the next one with the policy it started with.

    Args:
        warn_only: Warn instead of raising when an operation has no deterministic form.
        restore_cudnn_benchmark: Also restore ``torch.backends.cudnn.benchmark``, which
            Lightning's trainer setup writes unconditionally and a fit test would leak.

    Examples:
        >>> with deterministic_algorithms(warn_only=True):
        ...     torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        (True, True)
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=previous_warn_only)
        if restore_cudnn_benchmark:
            torch.backends.cudnn.benchmark = benchmark
