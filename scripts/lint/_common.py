# SPDX-License-Identifier: Apache-2.0
"""Plumbing shared by the repo-content checkers in this directory.

Every checker used to define ``REPO_ROOT``, hand-roll a ``--repo-root`` option, walk a
tree with a sorted ``rglob`` and print the same ``<name> FAILED: N ...`` block; those four
live here so the wording is written once, and a checker that gains a fifth copy has
somewhere to go instead.

Importing it: a checker is run as ``python scripts/lint/x.py``, which puts ``scripts/lint``
first on ``sys.path`` and would let a plain ``from _common import ...`` resolve -- but mypy
cannot find a module that way, so each checker puts the repository root on ``sys.path``
and imports ``scripts.lint._common``, which resolves for the hook, for a test that loads
the checker by path, and for the type checker alike.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

#: Repository root (``scripts/lint/`` is two levels below it).
REPO_ROOT = Path(__file__).resolve().parents[2]


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
