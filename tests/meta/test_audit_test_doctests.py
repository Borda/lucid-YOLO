# SPDX-License-Identifier: Apache-2.0
"""Meta tests: doctest-coverage audit script (WP-129, split off WP-117).

Covers ``scripts/lint/audit_test_doctests.py``'s public contract in isolation, on
synthetic files under ``tmp_path`` rather than against the live ``tests/`` tree --
the live tree is exercised instead by the pre-commit hook itself
(``.pre-commit-config.yaml``'s ``test-doctest-audit``) each time a test file
changes. This file is the functional-core check pytest owns; the hook is the
lint gate that runs it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "scripts" / "lint" / "audit_test_doctests.py"


def _load_audit() -> ModuleType:
    """Load ``scripts/lint/audit_test_doctests.py`` as an importable module.

    Examples:
        >>> module = _load_audit()
        >>> module.__name__
        'audit_test_doctests'
        >>> callable(module.find_missing)
        True
    """
    spec = importlib.util.spec_from_file_location("audit_test_doctests", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def test_module_level_helpers_excludes_fixtures_and_test_functions(tmp_path: Path) -> None:
    """Only bare module-level functions are counted, not fixtures or ``test_`` cases."""
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "import pytest\n@pytest.fixture\ndef a_fixture(): ...\ndef _helper(): ...\ndef test_thing(): ...\n",
        encoding="utf-8",
    )
    assert audit.module_level_helpers(sample) == ["_helper"]


def test_has_doctest_example_requires_a_prompt_line(tmp_path: Path) -> None:
    """A docstring without a ``>>>`` line fails the check even when non-empty."""
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "def _documented():\n    '''One line.\n\n    >>> 1 + 1\n    2\n    '''\ndef _bare():\n    'No example here.'\n",
        encoding="utf-8",
    )
    assert audit.has_doctest_example(sample, "_documented") is True
    assert audit.has_doctest_example(sample, "_bare") is False


def test_find_missing_names_the_offending_file_and_function(tmp_path: Path) -> None:
    """A helper lacking an Example is named as ``relative/path::function``."""
    (tmp_path / "test_sample.py").write_text("def _bare():\n    'No example.'\n", encoding="utf-8")

    missing = audit.find_missing(tmp_path)

    assert missing == [f"{tmp_path.name}/test_sample.py::_bare"]


def test_find_missing_is_empty_when_every_helper_has_an_example(tmp_path: Path) -> None:
    """A helper carrying a ``>>>`` Example does not appear in the missing list."""
    (tmp_path / "test_sample.py").write_text(
        "def _documented():\n    '''One line.\n\n    >>> 1 + 1\n    2\n    '''\n",
        encoding="utf-8",
    )
    assert audit.find_missing(tmp_path) == []


def test_main_returns_nonzero_when_helpers_are_missing_examples(tmp_path: Path) -> None:
    """The CLI exit code is 1 when the audit finds an undocumented helper."""
    (tmp_path / "test_sample.py").write_text("def _bare():\n    'No example.'\n", encoding="utf-8")

    assert audit.main(["--tests-dir", str(tmp_path)]) == 1


def test_main_returns_zero_on_a_clean_tree(tmp_path: Path) -> None:
    """The CLI exit code is 0 when every helper carries an Example."""
    assert audit.main(["--tests-dir", str(tmp_path)]) == 0


def test_the_live_tests_tree_is_currently_clean() -> None:
    """The real ``tests/`` tree the pre-commit hook scans has no missing Examples.

    This is the one place this file touches the live tree rather than a synthetic
    one: a regression here means the hook itself would fail on the next commit
    that touches any test file, not just a new one.
    """
    assert audit.find_missing(REPO_ROOT / "tests") == []
