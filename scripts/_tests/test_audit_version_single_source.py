# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the package version has exactly one source of truth (WP-001, split off WP-117).

Covers ``scripts/lint/audit_version_single_source.py``'s public contract in
isolation, on synthetic files under ``tmp_path`` rather than against the live
``pyproject.toml`` / ``src/lucid_yolo/__init__.py`` pair -- the live pair is
exercised instead by the pre-commit hook itself each time either file changes.
This file is the functional-core check pytest owns; the hook is the lint gate
that runs it.

Deliberately *not* tested here: ``importlib.metadata.version("lucid-yolo")``
against ``__version__``. An editable install freezes its metadata at install
time, so that assertion would fail on every version bump until someone
reinstalled -- a test that fails for a reason unrelated to the property it
claims to guard.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "scripts" / "lint" / "audit_version_single_source.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"
INIT = REPO_ROOT / "src" / "lucid_yolo" / "__init__.py"
README = REPO_ROOT / "README.md"


def _load_audit() -> ModuleType:
    """Load ``scripts/lint/audit_version_single_source.py`` as an importable module.

    Examples:
        >>> module = _load_audit()
        >>> module.__name__
        'audit_version_single_source'
        >>> callable(module.find_violations)
        True
    """
    spec = importlib.util.spec_from_file_location("audit_version_single_source", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def test_pyproject_declares_no_static_version_flags_a_version_key(tmp_path: Path) -> None:
    """A static ``version`` key in ``[project]`` is the drift this script guards against."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8")

    assert audit.check_pyproject_declares_no_static_version(pyproject) != []


def test_pyproject_declares_no_static_version_passes_without_one(tmp_path: Path) -> None:
    """A ``[project]`` table with no ``version`` key reports no violation."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "demo"\n', encoding="utf-8")

    assert audit.check_pyproject_declares_no_static_version(pyproject) == []


def test_pyproject_reads_the_module_attribute_flags_a_missing_dynamic_entry(tmp_path: Path) -> None:
    """``version`` absent from ``[project.dynamic]`` is rejected."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "demo"\n', encoding="utf-8")

    assert audit.check_pyproject_reads_the_module_attribute(pyproject) != []


def test_pyproject_reads_the_module_attribute_passes_when_wired_to_the_module(tmp_path: Path) -> None:
    """``dynamic`` lists ``version`` and ``[tool.setuptools.dynamic]`` points at the module attribute."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "demo"\ndynamic = ["version"]\n'
        '[tool.setuptools.dynamic]\nversion = {attr = "lucid_yolo.__version__"}\n',
        encoding="utf-8",
    )

    assert audit.check_pyproject_reads_the_module_attribute(pyproject) == []


def test_version_is_a_plain_literal_flags_a_computed_expression(tmp_path: Path) -> None:
    """A computed ``__version__`` (not a bare string literal) is rejected.

    A computed value (``importlib.metadata``, an f-string, a read from a file)
    would still be a valid attribute, but setuptools would have to *import* the
    package to evaluate it -- pulling torch into every build and every
    ``pip install`` of a source distribution.
    """
    init = tmp_path / "__init__.py"
    init.write_text("import sys\n__version__ = sys.version\n", encoding="utf-8")

    assert audit.check_version_is_a_plain_literal(init) != []


def test_version_is_a_plain_literal_passes_for_a_bare_string(tmp_path: Path) -> None:
    """A single bare string-literal assignment to ``__version__`` reports no violation."""
    init = tmp_path / "__init__.py"
    init.write_text("__version__ = '1.2.3'\n", encoding="utf-8")

    assert audit.check_version_is_a_plain_literal(init) == []


def test_the_live_pyproject_and_init_are_currently_clean() -> None:
    """The real ``pyproject.toml`` / ``__init__.py`` pair the pre-commit hook scans has no violations.

    This is the one place this file touches the live pair rather than a synthetic
    one: a regression here means the hook itself would fail on the next commit
    that touches either file, not just a new one.
    """
    assert audit.find_violations(PYPROJECT, INIT, README) == []


class TestReadmeNamesTheCurrentVersion:
    """``check_readme_names_the_current_version`` over a synthetic README / ``__init__.py`` pair.

    The drift this check was opened for is silent by construction: a release bumps
    ``__version__`` and the README's prose keeps naming the release before it, which
    every consumer of the repository reads first. WP-175 found three releases of it.
    """

    @staticmethod
    def _pair(root: Path, version: str, readme_body: str) -> tuple[Path, Path]:
        """Write an ``__init__.py`` at ``version`` and a README carrying ``readme_body``."""
        init = root / "__init__.py"
        init.write_text(f"__version__ = {version!r}\n", encoding="utf-8")
        readme = root / "README.md"
        readme.write_text(readme_body, encoding="utf-8")
        return readme, init

    def test_accepts_a_readme_in_step(self, tmp_path: Path) -> None:
        """A README naming the same version the module attribute carries reports no violation.

        The passing shape: the Versioning section's ``Current release`` line and
        ``lucid_yolo.__version__`` agreeing on one string.
        """
        readme, init = self._pair(tmp_path, "0.7.0", "Current release: **0.7.0** -- in step\n")

        assert audit.check_readme_names_the_current_version(readme, init) == []

    def test_names_the_stale_version_and_the_live_one(self, tmp_path: Path) -> None:
        """A README naming an older release is reported with both versions in the message.

        The exact state WP-175 audited -- README at 0.4.0 against ``__version__`` 0.7.0 --
        and the message carries both so the fix needs no second lookup.
        """
        readme, init = self._pair(tmp_path, "0.7.0", "Current release: **0.4.0** -- stale\n")

        assert audit.check_readme_names_the_current_version(readme, init) == [
            "README names 0.4.0 as the current release; lucid_yolo.__version__ is 0.7.0"
        ]

    def test_refuses_a_readme_that_states_no_current_release(self, tmp_path: Path) -> None:
        """A README with no ``Current release`` line at all is a violation, not a pass.

        Deleting the sentence would otherwise be the one edit that silences the check:
        an absent claim has to fail like a wrong one, or the guard is opt-out.
        """
        readme, init = self._pair(tmp_path, "0.7.0", "No versioning section here.\n")

        assert audit.check_readme_names_the_current_version(readme, init) == [
            'README states no "Current release: **<version>**" line to check against lucid_yolo.__version__'
        ]

    def test_stays_silent_when_the_version_literal_is_unreadable(self, tmp_path: Path) -> None:
        """A computed ``__version__`` yields no violation here -- the literal check owns that failure.

        Both checks run in one pass, so reporting an unreadable attribute twice would
        turn one defect into two messages naming different things.
        """
        readme = tmp_path / "README.md"
        readme.write_text("Current release: **0.7.0**\n", encoding="utf-8")
        init = tmp_path / "__init__.py"
        init.write_text("import sys\n__version__ = sys.version\n", encoding="utf-8")

        assert audit.check_readme_names_the_current_version(readme, init) == []
        assert audit.check_version_is_a_plain_literal(init) != []
