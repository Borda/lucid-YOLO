# SPDX-License-Identifier: Apache-2.0
"""Meta tests: the package version has exactly one source of truth.

``pyproject.toml`` carried its own ``version`` string beside
``lucid_yolo.__version__`` and the two had already drifted apart (``0.1.0``
against ``0.0.1.dev0``) -- a build published one number while every runtime
consumer reported the other. The packaging metadata is now declared dynamic and
read from the module attribute, and these tests pin that arrangement.

Deliberately *not* tested here: ``importlib.metadata.version("lucid-yolo")``
against ``__version__``. An editable install freezes its metadata at install
time, so that assertion would fail on every version bump until someone
reinstalled -- a test that fails for a reason unrelated to the property it
claims to guard.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
INIT = REPO_ROOT / "src" / "lucid_yolo" / "__init__.py"
VERSION_ATTR = "lucid_yolo.__version__"


def _pyproject() -> dict[str, object]:
    """Parse ``pyproject.toml`` into a dict."""
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_pyproject_declares_no_static_version() -> None:
    """``[project]`` states no ``version`` key -- the drift that made this file necessary."""
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    assert "version" not in project


def test_pyproject_reads_the_module_attribute() -> None:
    """``version`` is dynamic and resolved from ``lucid_yolo.__version__``."""
    config = _pyproject()
    project = config["project"]
    assert isinstance(project, dict)
    assert "version" in project["dynamic"]
    dynamic = config["tool"]["setuptools"]["dynamic"]  # type: ignore[index]
    assert dynamic["version"] == {"attr": VERSION_ATTR}


def test_version_is_a_plain_literal() -> None:
    """``__version__`` is a bare string literal, so setuptools resolves it without importing.

    A computed value (``importlib.metadata``, an f-string, a read from a file)
    would still be a valid attribute, but setuptools would have to *import* the
    package to evaluate it -- pulling torch into every build and every
    ``pip install`` of a source distribution.
    """
    module = ast.parse(INIT.read_text(encoding="utf-8"))
    literals = [
        node.value.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    ]
    assert len(literals) == 1
    assert isinstance(literals[0], str)
