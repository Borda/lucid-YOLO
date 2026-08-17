# SPDX-License-Identifier: Apache-2.0
"""Session setup for the figure-script suites (WP-067).

Three things every drawing test needs and neither the script nor pytest supplies.

The **backend** is forced to ``Agg`` before ``pyplot`` is imported anywhere in the
session, exactly as ``scripts/_tests/test_audit_figure_captions.py`` does it. A test run that opened
a windowing backend would need a display to draw a figure at all, which no gate has.

The **figures are closed** after every test. Matplotlib keeps every unclosed figure alive
in a module-level registry, so a suite that draws one per test leaks them all and starts
warning at twenty; worse, ``plt.gca()`` in a later test would find an earlier test's axes.
The fixture is ``autouse`` because forgetting it in one test is enough to reintroduce both.

Import mechanism: ``planted.py`` sits in ``tests/predict/`` — a non-package directory, as
``tests`` deliberately is (see ``tests/conftest.py``) — and is prepended to ``sys.path``
here so the end-to-end case can build its checkpoint with the same
``planted.write_checkpoint`` the inference suites use. A second copy of that saver here
would be free to drift from the ``load_eval_module`` contract it exists to satisfy. This
conftest lives under ``scripts/_tests/`` rather than ``tests/``, so the path to
``tests/predict/`` is repo-root-relative rather than a sibling lookup.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import pytest

matplotlib.use("Agg")

# Below the backend selection on purpose: pyplot binds whatever backend is current when it
# is first imported, and this conftest is the earliest import in the directory.
import matplotlib.pyplot as plt

_PREDICT_DIR = Path(__file__).resolve().parents[2] / "tests" / "predict"
if str(_PREDICT_DIR) not in sys.path:
    sys.path.insert(0, str(_PREDICT_DIR))

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close every figure a test opened, whether it passed or failed."""
    yield
    plt.close("all")
