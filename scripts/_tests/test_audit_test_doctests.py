# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: doctest-coverage audit script (WP-129, split off WP-117).

Covers ``scripts/lint/audit_test_doctests.py``'s public contract in isolation, on
synthetic files under ``tmp_path`` rather than against the live ``tests/`` tree --
the live tree is exercised instead by the pre-commit hook itself
(``.pre-commit-config.yaml``'s ``test-doctest-audit``) each time a test file
changes. This file is the functional-core check pytest owns; the hook is the
lint gate that runs it.
"""

from __future__ import annotations

import importlib.util
import sys
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
    sys.modules[spec.name] = module
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
    """A helper lacking an Example is named as ``relative/path::function -- reason``."""
    (tmp_path / "test_sample.py").write_text("def _bare():\n    'No example.'\n", encoding="utf-8")

    missing = audit.find_missing(tmp_path)

    assert missing == [f"{tmp_path.name}/test_sample.py::_bare -- no Example"]


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


class TestSkippedExamples:
    """A ``+SKIP`` Example is present and never executed, and the audit must say so."""

    def test_an_unreasoned_skip_does_not_count_as_coverage(self, tmp_path: Path) -> None:
        """A docstring whose only Example is `# doctest: +SKIP` fails, naming the skip.

        This is the whole defect: the directive satisfies collection, the presence
        scan and the reader, while running nothing -- so the number the audit
        reported was never the number it measured.
        """
        (tmp_path / "test_sample.py").write_text(
            "def _helper():\n    '''One line.\n\n    >>> _helper()  # doctest: +SKIP\n    '''\n",
            encoding="utf-8",
        )

        missing = audit.find_missing(tmp_path)

        assert missing == [f"{tmp_path.name}/test_sample.py::_helper -- every Example line is skipped, none says why"]

    def test_a_skip_with_a_stated_reason_counts(self, tmp_path: Path) -> None:
        """A reason before the directive passes -- the escape hatch is deliberate, not free.

        The reason must precede ``# doctest: +SKIP``: ``doctest`` reads a directive
        comment to end of line and rejects prose after the option list, so a reason
        written after it would break the very collection it annotates.
        """
        (tmp_path / "test_sample.py").write_text(
            "def _helper():\n    '''One line.\n\n    >>> _helper()  # needs a GPU  # doctest: +SKIP\n    '''\n",
            encoding="utf-8",
        )

        assert audit.find_missing(tmp_path) == []

    def test_one_runnable_line_beside_a_skip_is_enough(self, tmp_path: Path) -> None:
        """A docstring is covered when any Example line runs, skipped siblings included."""
        (tmp_path / "test_sample.py").write_text(
            "def _helper():\n    '''One line.\n\n    >>> _helper()  # doctest: +SKIP\n    >>> 1 + 1\n    2\n    '''\n",
            encoding="utf-8",
        )

        assert audit.find_missing(tmp_path) == []

    def test_a_missing_docstring_is_reported_apart_from_a_missing_example(self, tmp_path: Path) -> None:
        """No docstring and no Example are different repairs, so they are different messages."""
        (tmp_path / "test_sample.py").write_text("def _helper(): ...\n", encoding="utf-8")

        assert audit.find_missing(tmp_path) == [f"{tmp_path.name}/test_sample.py::_helper -- no docstring"]


class TestSourceTreeScan:
    """`src/` is scanned too, for its public surface only."""

    def test_a_public_source_function_without_an_example_fails(self, tmp_path: Path) -> None:
        """The requirement the Makefile states for `src/` now has a gate behind it."""
        (tmp_path / "module.py").write_text("def public():\n    'No example.'\n", encoding="utf-8")

        missing = audit.find_missing(tmp_path, pattern="*.py", include_private=False)

        assert missing == [f"{tmp_path.name}/module.py::public -- no Example"]

    def test_a_private_source_function_is_not_required_to_carry_one(self, tmp_path: Path) -> None:
        """In `src/` the requirement is about the public surface, not every function.

        A private helper's contract is stated by the public function that calls it;
        requiring an Example on each of the 82 private module-level functions in
        `src/` would be a different decision from the one the Makefile records.
        """
        (tmp_path / "module.py").write_text("def _private():\n    'No example.'\n", encoding="utf-8")

        assert audit.find_missing(tmp_path, pattern="*.py", include_private=False) == []

    def test_a_source_root_is_scanned_by_default(self, tmp_path: Path) -> None:
        """`--src-dir` is a default root, not an opt-in: naming neither option scans all three."""
        assert [root.pattern for root in audit._scan_roots(None, None)] == ["test_*.py", "test_*.py", "*.py"]

    def test_naming_one_option_replaces_the_default_set(self, tmp_path: Path) -> None:
        """`--tests-dir X` audits X alone, so a narrowing flag actually narrows."""
        roots = audit._scan_roots([tmp_path], None)

        assert [(root.path, root.pattern) for root in roots] == [(tmp_path, "test_*.py")]

    def test_the_live_source_tree_is_currently_clean(self) -> None:
        """The real `src/` tree has no public function without an executable Example."""
        assert audit.find_missing(REPO_ROOT / "src", pattern="*.py", include_private=False) == []


def test_the_live_tests_tree_is_currently_clean() -> None:
    """The real ``tests/`` tree the pre-commit hook scans has no missing Examples.

    This is the one place this file touches the live tree rather than a synthetic
    one: a regression here means the hook itself would fail on the next commit
    that touches any test file, not just a new one.
    """
    assert audit.find_missing(REPO_ROOT / "tests") == []


def test_the_live_scripts_tests_tree_is_currently_clean() -> None:
    """The real ``scripts/_tests/`` tree the pre-commit hook scans has no missing Examples.

    ``scripts/_tests/`` carries the functional-core tests for ``scripts/`` modules
    (WP-130) and is the hook's second default root, alongside ``tests/`` above.
    """
    assert audit.find_missing(REPO_ROOT / "scripts" / "_tests") == []
