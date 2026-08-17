# SPDX-License-Identifier: Apache-2.0
"""Doctest-coverage audit for ``test_*.py`` helpers (WP-129/130, split off WP-117).

Scans both ``tests/`` and ``scripts/_tests/`` by default -- the latter holds the
functional-core tests for ``scripts/`` modules (WP-130) and carries the same
helper-needs-an-Example expectation.

``pytest --doctest-modules`` (WP-085, ``make test``) turns a helper's ``Examples:``
block into an executable check the moment it exists, but nothing forces the block
to exist in the first place. This walks the same module-level-function population
by the same AST predicate ``--doctest-modules`` already exercises, and fails naming
every non-fixture, non-``test_`` helper still missing a ``>>>`` line, so a new
helper lands with an Example or fails the gate rather than joining the pile
silently.

Scope is deliberately narrow to the one invariant: a docstring's presence and its
``>>>`` content. Whether a group of helpers should be regrouped into a class is
WP-128's question, not this script's.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_test_doctests.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTS_DIR = REPO_ROOT / "tests"
DEFAULT_SCRIPTS_TESTS_DIR = REPO_ROOT / "scripts" / "_tests"

#: Decorator names that mark a function as a fixture rather than a helper, in
#: either bare (``@pytest.fixture``) or call (``@pytest.fixture(scope="module")``)
#: form, and under either the ``pytest.fixture`` or bare ``fixture`` import spelling.
_FIXTURE_DECORATOR_NAMES = frozenset({"fixture"})


def _decorator_name(node: ast.expr) -> str:
    """Return a decorator's trailing attribute or call name, e.g. ``"fixture"``.

    Handles the three shapes a decorator can take -- ``@fixture``, ``@fixture(...)``,
    and ``@pytest.fixture(...)`` -- by unwrapping a call to its callee first and then
    reading the attribute name off either a bare ``Name`` or an ``Attribute``.

    Examples:
        >>> import ast
        >>> _decorator_name(ast.parse("fixture", mode="eval").body)
        'fixture'
        >>> _decorator_name(ast.parse("pytest.fixture(scope='module')", mode="eval").body)
        'fixture'
    """
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return ""


def _is_fixture(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if any of ``func``'s decorators names a pytest fixture.

    Examples:
        >>> tree = ast.parse('''
        ... @pytest.fixture
        ... def thing(): ...
        ... ''')
        >>> _is_fixture(tree.body[0])
        True
    """
    return any(_decorator_name(dec) in _FIXTURE_DECORATOR_NAMES for dec in func.decorator_list)


def module_level_helpers(path: Path) -> list[str]:
    """Names of ``path``'s module-level, non-fixture, non-``test_`` functions.

    Deliberately module-level only: a closure nested inside a helper (a fake
    ``urlopen`` built by a factory, say) is private to that helper's own docstring
    and Example already, not a second surface a caller can reach, so it is not
    counted here -- the same boundary ``--doctest-modules`` draws when it collects
    a module's top-level functions.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "test_sample.py"
        ...     _ = sample.write_text(
        ...         "import pytest\\n"
        ...         "@pytest.fixture\\n"
        ...         "def a_fixture(): ...\\n"
        ...         "def _helper(): ...\\n"
        ...         "def test_thing(): ...\\n"
        ...     )
        ...     module_level_helpers(sample)
        ['_helper']
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("test_") or _is_fixture(node):
            continue
        names.append(node.name)
    return names


def has_doctest_example(path: Path, name: str) -> bool:
    """True if the module-level function ``name`` in ``path`` has a ``>>>`` docstring line.

    A missing docstring also fails this check (``ast.get_docstring`` returns
    ``None``, and containment on ``None`` would raise, so the ``None`` case is
    handled explicitly), which keeps the predicate a single check rather than
    two the caller has to combine.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "test_sample.py"
        ...     _ = sample.write_text(
        ...         "def _documented():\\n"
        ...         "    '''One line.\\n\\n"
        ...         "    >>> 1 + 1\\n"
        ...         "    2\\n"
        ...         "    '''\\n"
        ...         "def _bare():\\n"
        ...         "    'No example here.'\\n"
        ...     )
        ...     has_doctest_example(sample, "_documented"), has_doctest_example(sample, "_bare")
        (True, False)
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            doc = ast.get_docstring(node)
            return doc is not None and ">>>" in doc
    raise AssertionError(f"{name} not found at module level of {path}")  # pragma: no cover


def find_missing(tests_dir: Path) -> list[str]:
    """Every ``file::function`` under ``tests_dir`` still missing a doctest Example.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     sample = root / "test_sample.py"
        ...     _ = sample.write_text("def _bare():\\n    'No example.'\\n")
        ...     [item.split("/")[-1] for item in find_missing(root)]
        ['test_sample.py::_bare']
    """
    missing = []
    for path in sorted(tests_dir.rglob("test_*.py")):
        for name in module_level_helpers(path):
            if not has_doctest_example(path, name):
                missing.append(f"{path.relative_to(tests_dir.parent)}::{name}")
    return missing


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the audit, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` when any helper lacks an Example.
    """
    parser = argparse.ArgumentParser(description="Audit tests/**/test_*.py helpers for doctest Examples.")
    parser.add_argument(
        "--tests-dir",
        type=Path,
        action="append",
        dest="tests_dirs",
        default=None,
        help=f"directory to scan; repeatable (default: {DEFAULT_TESTS_DIR}, {DEFAULT_SCRIPTS_TESTS_DIR})",
    )
    args = parser.parse_args(argv)
    tests_dirs = args.tests_dirs if args.tests_dirs is not None else [DEFAULT_TESTS_DIR, DEFAULT_SCRIPTS_TESTS_DIR]

    missing = [item for tests_dir in tests_dirs for item in find_missing(tests_dir)]
    if missing:
        print(f"doctest-audit FAILED: {len(missing)} helper(s) missing a doctest Example")
        for item in missing:
            print(f"  - {item}")
        return 1
    print("doctest-audit clean: every non-fixture helper carries a doctest Example")
    return 0


if __name__ == "__main__":
    sys.exit(main())
