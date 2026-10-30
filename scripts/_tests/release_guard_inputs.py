# SPDX-License-Identifier: Apache-2.0
"""Synthetic inputs for the release-guard suite: changelog, release flags, package, manifest and gate receipt.

``test_release_guard.py`` states each release it describes instead of reading this tree's current one, so every
helper here writes a small file under a caller-supplied directory and hands back the path or flags naming it.
``conftest.py`` puts this directory on ``sys.path``, which is how a test file reaches the module by bare name.
"""

from __future__ import annotations

from pathlib import Path


def write_changelog(tmp_path: Path, version: str) -> Path:
    """Write a minimal changelog carrying a ``## [<version>]`` section.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = write_changelog(Path(tmp), "0.1.0")
        ...     "## [0.1.0]" in path.read_text()
        True
    """
    path = tmp_path / "CHANGELOG.md"
    path.write_text(f"# Changelog\n\n## [{version}]\n\n### Added\n\n- thing\n", encoding="utf-8")
    return path


def release_flags(tmp_path: Path, version: str) -> list[str]:
    """Point the version and frozen-golden checks at a synthetic release of ``version``.

    Writes a package declaring that ``__version__`` and a frozen-golden snapshot for its minor, and returns the
    ``--init``/``--frozen-root`` flags naming them. Both checks otherwise read the live repository, which would make
    every test below an assertion about this tree's current release rather than about the guard.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     flags = release_flags(Path(tmp), "0.1.0")
        ...     flags[0], Path(flags[1]).read_text()
        ('--init', '__version__ = "0.1.0"\\n')
    """
    init = tmp_path / "pkg" / "__init__.py"
    init.parent.mkdir(exist_ok=True)
    init.write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    frozen = tmp_path / "frozen" / ".".join(version.split(".")[:2])
    frozen.mkdir(parents=True, exist_ok=True)
    (frozen / "optim_toy.json").write_text("{}", encoding="utf-8")
    return ["--init", str(init), "--frozen-root", str(frozen.parent)]


def write_package(tmp_path: Path, source: str) -> Path:
    """Write a one-module package whose imports the tier check will walk.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = write_package(Path(tmp), "import os\\n")
        ...     root.name
        'shipped'
    """
    root = tmp_path / "shipped"
    root.mkdir()
    (root / "module.py").write_text(source, encoding="utf-8")
    return root


def write_manifest(tmp_path: Path, runtime: str, dev: str = "") -> Path:
    """Write a manifest declaring the two dependency tiers the check reads.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = write_manifest(Path(tmp), '"torch"')
        ...     "[project]" in path.read_text()
        True
    """
    path = tmp_path / "pyproject.toml"
    path.write_text(
        f"[project]\ndependencies = [{runtime}]\n\n[dependency-groups]\ndev = [{dev}]\n",
        encoding="utf-8",
    )
    return path


def gate_receipt_cmd(receipt: Path) -> str:
    """Build a gate command appending one line to ``receipt`` per invocation.

    ``check_gate`` launches the command through a shell from the repository root, so the counter is a file rather
    than a Python stub: what these tests must observe is how many times that process boundary was crossed, and a stub
    inside this process cannot see a crossing that a re-entrant hook would make from another one.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     "ran" in gate_receipt_cmd(Path(tmp) / "receipt.txt")
        True
    """
    return f"printf 'ran\\n' >> '{receipt}'"
