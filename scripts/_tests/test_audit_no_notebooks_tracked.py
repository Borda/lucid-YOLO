# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the no-tracked-notebook audit script (WP-188).

Covers ``scripts/lint/audit_no_notebooks_tracked.py`` on throwaway git repositories under
``tmp_path`` rather than against the live index -- the live index is what the
``no-ipynb-tracked`` pre-commit hook reads on every commit. Files are staged with
``git add -f`` throughout, so a contributor's global ``core.excludesFile`` cannot make a
case pass by never staging the notebook it meant to.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "scripts" / "lint" / "audit_no_notebooks_tracked.py"


def _load_audit() -> ModuleType:
    """Load ``scripts/lint/audit_no_notebooks_tracked.py`` as an importable module.

    Examples:
        >>> module = _load_audit()
        >>> module.__name__
        'audit_no_notebooks_tracked'
        >>> callable(module.find_violations)
        True
    """
    spec = importlib.util.spec_from_file_location("audit_no_notebooks_tracked", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def _stage(repo: Path, relative: str, text: str = "{}") -> None:
    """Write ``relative`` under ``repo`` and force-stage it, creating parent directories.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        ...     _stage(root, "deep/x.ipynb")
        ...     subprocess.run(["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True).stdout
        'deep/x.ipynb\\n'
    """
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "-f", relative], cwd=repo, check=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """An empty, freshly initialised git repository."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


class TestFindViolations:
    """`find_violations` reads the index, so staged and committed notebooks count alike."""

    def test_an_empty_index_is_clean(self, repo: Path) -> None:
        """A repository tracking nothing produces no violation."""
        assert audit.find_violations(repo) == []

    def test_a_staged_notebook_is_named(self, repo: Path) -> None:
        """A `.ipynb` staged with `git add -f` and not yet committed is already a violation.

        The hook runs at commit time, before the commit exists, so the index rather than
        `HEAD` is the surface it has to read -- a check over committed files alone would
        pass the very commit that adds the notebook.
        """
        _stage(repo, "notes.ipynb")

        assert audit.find_violations(repo) == [
            "tracked .ipynb files (notebooks are .py sources; the .ipynb is built): ['notes.ipynb']"
        ]

    def test_a_notebook_in_a_subdirectory_is_named_with_its_path(self, repo: Path) -> None:
        """The pathspec star crosses directory separators, so depth does not hide a notebook."""
        _stage(repo, "docs/notebooks/deep.ipynb")

        assert audit.find_violations(repo) == [
            "tracked .ipynb files (notebooks are .py sources; the .ipynb is built): ['docs/notebooks/deep.ipynb']"
        ]

    def test_a_percent_format_source_is_not_a_violation(self, repo: Path) -> None:
        """The `.py` source is what this repository commits; only the `.ipynb` is refused."""
        _stage(repo, "notebooks/wiring_gate.py", "# %%\nprint(1)\n")

        assert audit.find_violations(repo) == []


class TestMain:
    """The CLI's exit code is the hook's verdict."""

    def test_exits_zero_on_a_clean_index(self, repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A clean index reports so and exits `0`."""
        assert audit.main(["--repo-root", str(repo)]) == 0
        assert "clean" in capsys.readouterr().out

    def test_exits_one_and_names_the_file_when_a_notebook_is_staged(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A staged notebook exits `1` with the path in the report."""
        _stage(repo, "a.ipynb")

        code = audit.main(["--repo-root", str(repo)])

        assert code == 1
        assert "a.ipynb" in capsys.readouterr().out


def test_the_live_index_is_currently_clean() -> None:
    """The real repository tracks no `.ipynb` -- the state the hook asserts on every commit."""
    assert audit.find_violations(REPO_ROOT) == []
