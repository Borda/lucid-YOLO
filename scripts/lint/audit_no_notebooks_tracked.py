# SPDX-License-Identifier: Apache-2.0
"""No-tracked-notebook audit: a ``.ipynb`` is a build product, never a committed file (WP-188).

Notebooks live in this repository as jupytext percent-format ``notebooks/*.py`` sources;
the ``.ipynb`` the site renders is written by ``make notebooks`` into a gitignored
directory. A committed ``.ipynb`` carries outputs and kernel metadata that rot on every
re-run and diff as opaque JSON, which is the whole reason the source is a ``.py`` -- and
``.gitignore`` is not a gate, because ``git add -f`` walks straight past it. This reads
the index instead: any ``.ipynb`` git tracks, or has staged, is a violation, whichever
directory it sits in.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_no_notebooks_tracked.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def tracked_notebooks(repo_root: Path) -> list[str]:
    """Every ``.ipynb`` path in the git index under ``repo_root``, relative to the root.

    ``git ls-files`` reads the index, so a file staged with ``git add -f`` and not yet
    committed is already listed -- which is the moment a pre-commit hook has to catch it.
    The pathspec is ``*.ipynb`` without ``:(glob)`` magic, so the star crosses directory
    separators and a notebook three levels down is matched the same as one at the root.
    Run with the root as the working directory because ``ls-files`` scopes its listing to
    the caller's directory, not the repository's.

    Examples:
        >>> import subprocess
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        ...     _ = (root / "notes.ipynb").write_text("{}", encoding="utf-8")
        ...     tracked_notebooks(root)
        ...     _ = subprocess.run(["git", "add", "-f", "notes.ipynb"], cwd=root, check=True)
        ...     tracked_notebooks(root)
        []
        ['notes.ipynb']
    """
    listing = subprocess.run(
        ["git", "ls-files", "--", "*.ipynb"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in listing.stdout.splitlines() if line]


def find_violations(repo_root: Path) -> list[str]:
    """Every tracked-notebook violation under ``repo_root`` -- one line naming all of them.

    Examples:
        >>> import subprocess
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        ...     find_violations(root)
        ...     _ = (root / "a.ipynb").write_text("{}", encoding="utf-8")
        ...     _ = subprocess.run(["git", "add", "-f", "a.ipynb"], cwd=root, check=True)
        ...     find_violations(root)
        []
        ["tracked .ipynb files (notebooks are .py sources; the .ipynb is built): ['a.ipynb']"]
    """
    tracked = tracked_notebooks(repo_root)
    if not tracked:
        return []
    return [f"tracked .ipynb files (notebooks are .py sources; the .ipynb is built): {tracked}"]


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the audit, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` when any ``.ipynb`` is tracked or staged.
    """
    parser = argparse.ArgumentParser(description="Refuse any .ipynb in the git index; notebooks are .py sources.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=f"repository whose index to read (default: {REPO_ROOT})",
    )
    args = parser.parse_args(argv)

    violations = find_violations(args.repo_root)
    if violations:
        print(f"no-notebooks-tracked-audit FAILED: {len(violations)} violation(s)")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("no-notebooks-tracked-audit clean: no .ipynb in the index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
