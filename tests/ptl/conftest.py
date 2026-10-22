# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the detection and pose network-recording suites."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
from determinism import deterministic_algorithms
from markers import BF16_REPLAY_EXPECTED


@pytest.fixture
def graph_recordings() -> Iterator[MagicMock]:
    """Count CUDA graph recordings without judging the warnings a fit emits.

    The spy wraps the real recorder, so replay still runs. Use it directly when a
    retirement warning is the expected outcome; ``replay_spy`` is the variant for
    fits that must keep replaying.
    """
    with patch("torch.cuda.make_graphed_callables", wraps=torch.cuda.make_graphed_callables) as record:
        yield record


@pytest.fixture
def replay_spy(graph_recordings: MagicMock) -> Iterator[MagicMock]:
    """Count graph recordings and turn a retirement warning into a failure, so replay cannot silently lapse."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="capture_network=True stays eager")
        yield graph_recordings


@pytest.fixture
def bf16_replay_expected() -> bool:
    """Whether this runtime's torch is new enough to record BF16 replay (torch 2.9, the first verified release).

    Defined in ``markers`` from ``torch.__version__`` rather than read from the production gate,
    so a changed production threshold fails ``test_bf16_gate_tracks_torch_2_9`` instead of
    silently moving every expectation with it.
    """
    return BF16_REPLAY_EXPECTED


@pytest.fixture
def deterministic_kernels() -> Iterator[None]:
    """Run the test under deterministic kernels so eager and replayed results compare exactly.

    The caller's deterministic-algorithms policy is restored afterwards.
    """
    with deterministic_algorithms():
        yield


@pytest.fixture
def deterministic_kernels_pinning_cudnn_benchmark() -> Iterator[None]:
    """Run under deterministic kernels and also restore ``torch.backends.cudnn.benchmark`` afterwards.

    Lightning's trainer setup writes the flag unconditionally, so a test that fits would leak it.
    """
    with deterministic_algorithms(restore_cudnn_benchmark=True):
        yield
