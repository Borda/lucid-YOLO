# SPDX-License-Identifier: Apache-2.0
"""Plumbing shared by the repo-content checkers in this directory.

Every checker used to define ``REPO_ROOT``, hand-roll a ``--repo-root`` option, walk a tree with a sorted ``rglob``
and print the same ``<name> FAILED: N ...`` block; those four live here so the wording is written once, and a checker
that gains a fifth copy has somewhere to go instead.

Importing it: a checker is run as ``python scripts/lint/x.py``, which puts ``scripts/lint`` first on ``sys.path``
and would let a plain ``from _common import ...`` resolve -- but mypy cannot find a module that way, so each checker
puts the repository root on ``sys.path`` and imports ``scripts.lint._common``, which resolves for the hook, for a
test that loads the checker by path, and for the type checker alike.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

#: Repository root (``scripts/lint/`` is two levels below it), resolved from this file rather than from the process.
#: Every checker's paths and ``git`` calls are anchored here: read from the working directory instead, a checker
#: answers about wherever it happens to be standing — from a subdirectory it dies on a table it cannot find, and
#: from another checkout it would validate that checkout's history and report clean.
REPO_ROOT = Path(__file__).resolve().parents[2]


class FenceTracker:
    r"""Track fenced code using the caller's fence-line grammar.

    ``language`` is ``None`` outside a fence and an empty string for an unlabelled
    fence. A named ``language`` regex group retains the opening language when present.
    Every matching line toggles the fence, preserving the checkers' existing rules.

    Examples:
        >>> fences = FenceTracker(re.compile(r"^```(?P<language>\w*)$"))
        >>> fences.update("```pycon"), fences.language
        (True, 'pycon')
        >>> fences.update(">>> 1"), fences.language
        (False, 'pycon')
        >>> fences.update("```"), fences.language
        (True, None)
        >>> quoted = FenceTracker(re.compile(r"^\s*(?:>\s*)*(```|~~~)"))
        >>> quoted.update("> ~~~python"), quoted.language
        (True, '')
        >>> quoted.update("> ```"), quoted.language
        (True, None)
    """

    def __init__(self, pattern: re.Pattern[str]) -> None:
        """Start outside any fence, recognizing lines with ``pattern``."""
        self._pattern = pattern
        self.language: str | None = None

    def update(self, line: str) -> bool:
        """Update the open fence and return whether this line is a fence."""
        match = self._pattern.match(line)
        if match is None:
            return False
        self.language = match.groupdict().get("language", "") if self.language is None else None
        return True


def add_repo_root_argument(parser: argparse.ArgumentParser, purpose: str) -> None:
    """Add the ``--repo-root`` option every checker offers, defaulting to :data:`REPO_ROOT`.

    Args:
        parser: Parser to extend.
        purpose: What the checker reads under the root, as the start of the help text.

    Examples:
        ```pycon
        >>> parser = argparse.ArgumentParser()
        >>> add_repo_root_argument(parser, "repository to scan")
        >>> parser.parse_args(["--repo-root", "elsewhere"]).repo_root
        PosixPath('elsewhere')
        >>> parser.parse_args([]).repo_root == REPO_ROOT
        True

        ```
    """
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help=f"{purpose} (default: {REPO_ROOT})")


def sorted_rglob(root: Path, pattern: str) -> list[Path]:
    """Every path under ``root`` matching ``pattern``, in sorted order.

    A missing ``root`` yields nothing rather than failing, as ``Path.rglob`` does.

    Args:
        root: Directory to walk.
        pattern: Glob pattern, matched at every depth.

    Returns:
        The matching paths, sorted so a report reads the same on every filesystem.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     for name in ("b.py", "a.py", "sub/c.py", "d.txt"):
        ...         _ = (Path(tmp) / name).parent.mkdir(parents=True, exist_ok=True)
        ...         _ = (Path(tmp) / name).write_text("", encoding="utf-8")
        ...     [path.relative_to(tmp).as_posix() for path in sorted_rglob(Path(tmp), "*.py")]
        ...     sorted_rglob(Path(tmp) / "absent", "*.py")
        ['a.py', 'b.py', 'sub/c.py']
        []

        ```
    """
    return sorted(root.rglob(pattern))


def print_failure(name: str, items: Sequence[object], what: str = "violation(s)") -> int:
    """Print a checker's ``<name> FAILED`` block and return the failing exit code.

    Args:
        name: The checker's label, as it opens the headline.
        items: What was found; one ``  - item`` line each, after a headline counting them.
        what: The noun phrase that follows the count in the headline.

    Returns:
        ``1``, so a ``main`` can ``return print_failure(...)``.

    Examples:
        ```pycon
        >>> print_failure("demo-audit", ["a.py: bad", "b.py: worse"])
        demo-audit FAILED: 2 violation(s)
          - a.py: bad
          - b.py: worse
        1

        ```
    """
    print(f"{name} FAILED: {len(items)} {what}")
    for item in items:
        print(f"  - {item}")
    return 1
