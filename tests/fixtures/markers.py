# SPDX-License-Identifier: Apache-2.0
"""Shared pytest marks for tests that need an accelerator.

A test that needs a device carries two marks: ``gpu``, which keeps it out of the
offline ``-m "not gpu and not data"`` selection and into ``make test-gpu``, and a
``skipif`` that turns it into a skip rather than an error on a host without the
device. One written without the first runs in no CI job, one written without the
second errors on CPU, so the pair lives here as a single object instead of being
stacked by hand at every site.

Use the mark set as a decorator, or as the ``marks=`` of one ``pytest.param`` case
(``pytest.param("cuda", marks=requires_cuda)``). Marks that are wanted for a
narrower reason than the device -- a BF16 runtime, a second GPU -- stay with the test
that needs them and are stacked next to this one.
"""

from __future__ import annotations

from typing import TypeVar

import pytest
import torch
from packaging.version import Version

_T = TypeVar("_T")


class MarkSet(tuple[pytest.MarkDecorator, ...]):
    """A tuple of marks that also applies all of them when used as a decorator.

    A bare ``pytest.mark`` decorator carries one mark, and the two a device test needs
    cannot be composed into one. This tuple is accepted wherever pytest takes a
    collection of marks (``pytest.param(..., marks=...)``, ``pytest.fixture(params=...)``)
    and, called on a function or class, applies each of them in turn.

    Examples:
        >>> [mark.name for mark in MarkSet((pytest.mark.gpu, pytest.mark.data))]
        ['gpu', 'data']
        >>> @MarkSet((pytest.mark.gpu, pytest.mark.data))
        ... def test_example() -> None: ...
        >>> sorted(mark.name for mark in test_example.pytestmark)
        ['data', 'gpu']
    """

    __slots__ = ()

    def __call__(self, obj: _T) -> _T:
        """Apply every mark in the set to ``obj`` and return it."""
        for mark in reversed(self):
            obj = mark(obj)
        return obj


#: A test that needs a CUDA device: ``gpu`` plus a skip where CUDA is not available.
requires_cuda = MarkSet(
    (pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
)

#: A test that needs any accelerator the project supports (CUDA or MPS).
requires_accelerator = MarkSet(
    (
        pytest.mark.gpu,
        pytest.mark.skipif(
            not (torch.cuda.is_available() or torch.backends.mps.is_available()),
            reason="requires an accelerator (CUDA or MPS)",
        ),
    ),
)

#: Whether this runtime's torch is new enough to record BF16 replay (torch 2.9, the first verified release).
#: Derived from ``torch.__version__`` rather than read from the production gate, so a changed
#: production threshold fails ``test_bf16_gate_tracks_torch_2_9`` instead of moving every expectation with it.
BF16_REPLAY_EXPECTED = Version(torch.__version__) >= Version("2.9")

#: Skips a case that records BF16 replay on a runtime older than torch 2.9.
skip_unless_bf16_replay = pytest.mark.skipif(
    not BF16_REPLAY_EXPECTED, reason="BF16 replay requires the verified torch 2.9 runtime"
)

#: Independent expectation for the norm decomposition's torch 2.9 runtime gate.
#: ``test_norm_replay_gate_tracks_torch_2_9`` detects changes to the production threshold.
NORM_REPLAY_EXPECTED = Version(torch.__version__) >= Version("2.9")

#: Skips norm recordings on supported runtimes without the verified decomposition.
skip_unless_norm_replay = pytest.mark.skipif(
    not NORM_REPLAY_EXPECTED, reason="gradient-norm replay requires torch>=2.9"
)
