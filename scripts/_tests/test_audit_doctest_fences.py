# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the doctest-fence audit script.

Covers ``scripts/lint/audit_doctest_fences.py`` on modules written under ``tmp_path``
rather than against the live tree -- the live tree is what the ``doctest-fences``
pre-commit hook reads on every commit that touches ``src/`` or ``scripts/``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "scripts" / "lint" / "audit_doctest_fences.py"

#: A module whose one doctest is fenced and closed after one blank line (source line 6 is the prompt).
FENCED = "\n".join(
    [
        "def f():",
        "    '''Say.",
        "",
        "    Examples:",
        "        ```pycon",
        "        >>> 1",
        "        1",
        "",
        "        ```",
        "    '''",
        "",
    ]
)
#: The same module with the doctest bare (source line 5 is the prompt).
BARE = "\n".join(
    [
        "def f():",
        "    '''Say.",
        "",
        "    Examples:",
        "        >>> 1",
        "        1",
        "    '''",
        "",
    ]
)


def _load_audit() -> ModuleType:
    """Load ``scripts/lint/audit_doctest_fences.py`` as an importable module.

    Examples:
        ```pycon
        >>> module = _load_audit()
        >>> module.__name__
        'audit_doctest_fences'
        >>> callable(module.find_violations)
        True

        ```
    """
    spec = importlib.util.spec_from_file_location("audit_doctest_fences", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def _module(tmp_path: Path, source: str) -> Path:
    """Write ``source`` to ``tmp_path / "module.py"`` and return the path.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _module(Path(tmp), "x = 1\\n").read_text(encoding="utf-8")
        'x = 1\\n'

        ```
    """
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")
    return path


class TestFindViolations:
    """The two rules, and what is never read as a doctest."""

    def test_fenced_block_clean(self, tmp_path: Path) -> None:
        """A ```pycon block closed after one blank line passes."""
        assert audit.find_violations([_module(tmp_path, FENCED)]) == []

    def test_bare_prompt_named_with_line(self, tmp_path: Path) -> None:
        """A bare prompt is refused at its own source line, once per run of prompts."""
        path = _module(tmp_path, BARE + BARE.replace("def f", "def g"))

        assert audit.find_violations([path]) == [
            f"{path}:5: doctest prompt outside a ```pycon block",
            f"{path}:12: doctest prompt outside a ```pycon block",
        ]

    def test_closing_fence_without_blank_line_refused(self, tmp_path: Path) -> None:
        """Without the blank line doctest reads the closing fence as expected output."""
        path = _module(tmp_path, FENCED.replace("        1\n\n", "        1\n"))

        assert audit.find_violations([path]) == [
            f"{path}:8: pycon block needs exactly one blank line before its closing fence"
        ]

    def test_two_blank_lines_before_closing_fence_refused(self, tmp_path: Path) -> None:
        """Exactly one: a second blank line is a stray line inside the block."""
        path = _module(tmp_path, FENCED.replace("        1\n\n", "        1\n\n\n"))

        assert len(audit.find_violations([path])) == 1

    def test_prompt_inside_other_language_fence_refused(self, tmp_path: Path) -> None:
        """Only ``pycon`` counts; a prompt under a ```python fence is still bare."""
        path = _module(tmp_path, FENCED.replace("```pycon", "```python"))

        assert audit.find_violations([path]) == [f"{path}:6: doctest prompt outside a ```pycon block"]

    def test_prompt_in_string_literal_not_doctest(self, tmp_path: Path) -> None:
        """A ``>>>`` in an ordinary string or comment is not a docstring example."""
        path = _module(tmp_path, 'PROMPT = """\n>>> 1\n"""\n# >>> 2\n')

        assert audit.find_violations([path]) == []


class TestMain:
    """The CLI's exit code is the hook's verdict."""

    def test_exits_zero_on_fenced_module(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A clean module reports so and exits `0`."""
        assert audit.main([str(_module(tmp_path, FENCED))]) == 0
        assert "clean" in capsys.readouterr().out

    def test_exits_one_and_names_file_on_bare_module(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A bare doctest exits `1` with the path in the report."""
        path = _module(tmp_path, BARE)

        assert audit.main([str(path)]) == 1
        assert str(path) in capsys.readouterr().out
