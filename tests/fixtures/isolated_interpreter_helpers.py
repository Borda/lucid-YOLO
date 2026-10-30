# SPDX-License-Identifier: Apache-2.0
"""Fresh-interpreter helper for the assignment-backend tests.

Import behaviour of the optional CUDA assignment backend can only be observed in a process that has not imported it yet.
``run_isolated`` runs a snippet in such a CPU-only process against this checkout's ``src`` tree.
Tests import this module by bare name because ``tests/conftest.py`` puts ``tests/fixtures`` on ``sys.path``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[2] / "src"


def run_isolated(code: str) -> str:
    """Run ``code`` in a fresh CPU-only interpreter importing this checkout, returning its stdout.

    Raises ``AssertionError`` carrying the child's output when it exits non-zero, so a failing snippet reads as a
    failing test.

    Examples:
        >>> run_isolated("print('ok')")
        'ok\\n'
    """
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(_SOURCE), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout
