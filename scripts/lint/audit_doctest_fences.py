# SPDX-License-Identifier: Apache-2.0
r"""Doctest-fence audit: every ``src/`` docstring example sits in a ```` ```pycon ```` block.

``--doctest-modules`` collects a ``>>>`` prompt whether it is fenced or bare, so the
fence changes no gate result; what it buys is one form across the tree, which a reader
and the next edit can copy without first checking which convention the module chose.
The rule covers the package only; ``scripts/`` and the test trees are development
tooling with no reader beyond the next editor, and keep whichever form each module uses.
Two rules, read from docstrings only -- a ``>>>`` inside an ordinary string literal or a
comment is not a doctest and is never flagged:

* every ``>>>`` prompt line sits inside an open ```` ```pycon ```` fence;
* every ``pycon`` block ends with exactly one blank line before its closing
  ```` ``` ````. Doctest reads expected output up to the first blank line, so without
  it the fence itself is taken as output; with two, the block carries a stray line.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_doctest_fences.py src/lucid_yolo/data/coco.py
        $ python scripts/lint/audit_doctest_fences.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `scripts.lint` resolves under the hook
from scripts.lint._common import REPO_ROOT, print_failure, sorted_rglob

#: Roots scanned when no file is named on the command line.
DEFAULT_ROOTS = ("src",)
#: A fence line, opening (with its language) or closing (without one).
FENCE_RE = re.compile(r"^\s*```(?P<language>[A-Za-z0-9_-]*)\s*$")
#: A doctest prompt line.
PROMPT_RE = re.compile(r"^\s*>>>( |$)")

_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstrings(source: str) -> list[tuple[int, str]]:
    r"""Every docstring in ``source`` as ``(first source line, text)``, in line order.

    Examples:
        ```pycon
        >>> _docstrings("'''Top.'''\ndef f():\n    '''Inner.'''\n")
        [(1, 'Top.'), (3, 'Inner.')]

        ```
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, _DOCSTRING_OWNERS) or not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            found.append((first.lineno, first.value.value))
    return sorted(found)


def docstring_violations(text: str, first_line: int) -> list[tuple[int, str]]:
    r"""Fence violations in one docstring's text, as ``(source line, reason)`` pairs.

    Args:
        text: The docstring's value.
        first_line: Source line its opening quotes sit on; line ``n`` of ``text`` is
            reported as ``first_line + n``.

    Returns:
        One pair per bare prompt run and per badly closed ``pycon`` block.

    Examples:
        ```pycon
        >>> docstring_violations("Say.\n\n```pycon\n>>> 1\n1\n\n```\n", 10)
        []
        >>> docstring_violations("Say.\n\n>>> 1\n1\n>>> 2\n2\n", 10)
        [(12, 'doctest prompt outside a ```pycon block')]
        >>> docstring_violations("```pycon\n>>> 1\n1\n```\n", 10)
        [(13, 'pycon block needs exactly one blank line before its closing fence')]

        ```
    """
    violations = []
    lines = text.splitlines()
    language: str | None = None
    in_bare_run = False
    for offset, line in enumerate(lines):
        fence = FENCE_RE.match(line)
        if fence is not None:
            in_bare_run = False
            if language is None:
                language = fence.group("language")
                continue
            one_blank = offset >= 2 and not lines[offset - 1].strip() and lines[offset - 2].strip()
            if language == "pycon" and not one_blank:
                violations.append(
                    (first_line + offset, "pycon block needs exactly one blank line before its closing fence")
                )
            language = None
            continue
        if not line.strip():
            in_bare_run = False
        elif PROMPT_RE.match(line) and language != "pycon" and not in_bare_run:
            violations.append((first_line + offset, "doctest prompt outside a ```pycon block"))
            in_bare_run = True
    return violations


def find_violations(paths: list[Path]) -> list[str]:
    r"""One ``path:line: reason`` entry per violation across ``paths``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> bare = "def f():\n    '''Say.\n\n    >>> 1\n    1\n    '''\n"
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "bare.py"
        ...     _ = path.write_text(bare, encoding="utf-8")
        ...     [entry.split(": ", 1)[1] for entry in find_violations([path])]
        ['doctest prompt outside a ```pycon block']

        ```
    """
    entries: list[str] = []
    for path in paths:
        for first_line, text in _docstrings(path.read_text(encoding="utf-8")):
            entries.extend(f"{path}:{line}: {reason}" for line, reason in docstring_violations(text, first_line))
    return entries


def _default_paths(repo_root: Path) -> list[Path]:
    """Every ``.py`` file under :data:`DEFAULT_ROOTS` in ``repo_root``, sorted.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     (root / "src").mkdir()
        ...     _ = (root / "src" / "a.py").write_text("", encoding="utf-8")
        ...     [path.name for path in _default_paths(root)]
        ['a.py']

        ```
    """
    return [path for name in DEFAULT_ROOTS for path in sorted_rglob(repo_root / name, "*.py")]


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the audit, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` on any violation.
    """
    parser = argparse.ArgumentParser(description="Refuse docstring doctests outside a ```pycon fenced block.")
    parser.add_argument(
        "files", nargs="*", type=Path, help=f"modules to check (default: every .py under {DEFAULT_ROOTS})"
    )
    args = parser.parse_args(argv)

    paths = args.files or _default_paths(REPO_ROOT)
    violations = find_violations(paths)
    if violations:
        return print_failure("doctest-fence audit", violations)
    print(f"doctest-fence audit clean: {len(paths)} module(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
