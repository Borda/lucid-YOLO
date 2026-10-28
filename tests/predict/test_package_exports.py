# SPDX-License-Identifier: Apache-2.0
"""Gate on the top-level Python API: ``lucid_yolo.<name>`` for the post-training calls.

Two properties make the re-exports worth having and cheap to keep. Every name in
``__all__`` resolves to the very object its defining module holds, so the short path
and the long path cannot drift apart. And ``import lucid_yolo`` stays free of torch:
the names resolve on first access, because reading ``__version__`` (a packaging step,
a notebook's first cell) must not cost a torch and Lightning import.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import lucid_yolo


@pytest.mark.parametrize("name", sorted(set(lucid_yolo.__all__) - {"__version__"}))
def test_export_is_defining_module_object(name: str) -> None:
    """Each re-export is the defining module's own object, not a copy or a wrapper."""
    defining_module = importlib.import_module(lucid_yolo._LAZY_EXPORTS[name])

    assert getattr(lucid_yolo, name) is getattr(defining_module, name)


def test_lazy_table_matches_all() -> None:
    """``__all__`` and the lazy table name the same exports, so neither can gain one alone."""
    assert set(lucid_yolo._LAZY_EXPORTS) == set(lucid_yolo.__all__) - {"__version__"}


def test_package_import_loads_no_torch() -> None:
    """A fresh interpreter importing the package loads neither torch nor Lightning.

    Run in a subprocess, since this test session has long since imported both.
    """
    probe = "import sys, lucid_yolo; print(sorted({'torch', 'pytorch_lightning'} & set(sys.modules)))"

    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "[]"
