# SPDX-License-Identifier: Apache-2.0
"""The shipped package never reaches for ``lucid_yolo._viz``, and only ``_viz`` reaches for matplotlib (WP-189).

``matplotlib`` is a declared runtime dependency — ``pytorch-lightning[extra]`` required it before WP-189 declared it
directly — so the promise here is not its absence but its confinement: nothing under ``lucid_yolo`` imports
``lucid_yolo._viz``, and nothing under ``lucid_yolo`` outside it imports matplotlib. ``_viz`` is private because the
notebooks are its only callers, and confinement is what keeps the shipped surface's import graph free of a figure
package it has no use for. Two checks pin that from two sides.

The **dynamic** check imports the shipped surface in a fresh interpreter and asserts ``lucid_yolo._viz`` never entered
``sys.modules``. A fresh interpreter, because this test session imported the package itself long before this file was
collected. What it does *not* assert is that matplotlib is absent from ``sys.modules`` afterwards: ``torchmetrics``
imports it whenever it is installed (``torchmetrics/utilities/plot.py``), which is always.

The **static** side is not a test: ruff's ``TID251`` banned-api rule (``pyproject.toml``) refuses an import of
matplotlib or ``_viz`` anywhere under ``src/`` outside ``_viz``. It catches the failure the dynamic check would miss: a
lazy, function-local ``from lucid_yolo._viz import ...`` that no import of the surface executes, and that the first
notebook call would.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

#: The shipped surface a consumer imports to predict with a checkpoint. An import of ``_viz`` anywhere beneath these
#: would show up in the child's ``sys.modules``.
SHIPPED_MODULES = ("lucid_yolo", "lucid_yolo.predict", "lucid_yolo.eval.checkpoint", "lucid_yolo.data")


def _probe(code: str, environment: dict[str, str] | None = None) -> str:
    """Run ``code`` in a fresh interpreter and return its stripped standard output.

    Examples:
        >>> _probe("print(1 + 1)")
        '2'
    """
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=environment)
    return result.stdout.strip()


@pytest.mark.parametrize("module", SHIPPED_MODULES)
def test_shipped_module_never_imports_viz_package(module: str) -> None:
    """A fresh interpreter importing ``module`` has no ``lucid_yolo._viz`` in ``sys.modules``.

    The failure this guards is a convenience import — ``from lucid_yolo._viz import ...`` landing in a shipped module
    because a helper was handy — which would put a figure package, and every matplotlib import it makes, on the path
    of a plain prediction.
    """
    probe = f"import sys, {module}; print('lucid_yolo._viz' in sys.modules)"

    assert _probe(probe) == "False"


def test_importing_viz_package_leaves_backend_alone() -> None:
    """A fresh interpreter's ``matplotlib.get_backend()`` reads the same before and after ``import lucid_yolo._viz``.

    ``matplotlib.use("Agg")`` is process-wide: a package that made that call at import would switch a notebook kernel
    away from its inline backend, and a figure returned as a cell's last expression would stop rendering. The scripts
    select ``Agg`` themselves; the package must not. ``MPLBACKEND`` is cleared for the child so the environment cannot
    pin the answer either way.
    """
    probe = (
        "import matplotlib; before = matplotlib.get_backend(); import lucid_yolo._viz; "
        "print(before == matplotlib.get_backend())"
    )
    environment = {key: value for key, value in os.environ.items() if key != "MPLBACKEND"}

    assert _probe(probe, environment) == "True"


def test_viz_package_import_matplotlib() -> None:
    """The control: importing ``lucid_yolo._viz`` in a fresh interpreter does load matplotlib.

    Without it the checks above could pass on a package that draws nothing — the dependency has to be shown to exist
    exactly where it is allowed to.
    """
    probe = "import sys, lucid_yolo._viz; print('matplotlib.pyplot' in sys.modules)"

    assert _probe(probe) == "True"
