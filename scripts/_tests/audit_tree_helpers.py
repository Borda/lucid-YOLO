# SPDX-License-Identifier: Apache-2.0
"""Synthetic-tree builders for the ``audit_*`` script suites.

Each audit script under ``scripts/lint/`` reads a small tree of files, and its tests build that tree under ``tmp_path``.
These helpers hold the file-writing steps those suites share, so a test body names the files it needs and nothing else.
``conftest.py`` puts this directory on ``sys.path``, which is how a test file reaches it.
"""

from __future__ import annotations

from pathlib import Path


def write_file(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` as UTF-8, creating the parent directories and overwriting any earlier file.

    Args:
        path: File to write.
        text: Its full contents.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     target = Path(tmp) / "a" / "b.txt"
        ...     write_file(target, "x")
        ...     target.read_text(encoding="utf-8")
        'x'
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_tree(root: Path, files: dict[str, str]) -> None:
    """Write every ``relative path -> text`` entry of ``files`` below ``root``.

    Args:
        root: Directory the relative paths hang from.
        files: Contents keyed by path relative to ``root``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     write_tree(Path(tmp), {"docs/index.md": "x", "mkdocs.yml": "nav: []\\n"})
        ...     sorted(path.name for path in Path(tmp).rglob("*") if path.is_file())
        ['index.md', 'mkdocs.yml']
    """
    for relative, text in files.items():
        write_file(root / relative, text)


def replace_in_file(path: Path, old: str, new: str) -> None:
    """Rewrite ``path`` with every ``old`` replaced by ``new``, to drop or alter one row of a valid fixture.

    Args:
        path: UTF-8 file to rewrite in place.
        old: Text to find.
        new: Text to put in its place.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     target = Path(tmp) / "f.md"
        ...     write_file(target, "| A1 |\\n| A2 |\\n")
        ...     replace_in_file(target, "| A2 |\\n", "")
        ...     target.read_text(encoding="utf-8")
        '| A1 |\\n'
    """
    write_file(path, path.read_text(encoding="utf-8").replace(old, new))


def make_docs_site(root: Path, mkdocs_text: str, pages: tuple[str, ...] = ()) -> tuple[Path, Path]:
    """Build a ``docs/`` directory holding ``pages`` plus a ``mkdocs.yml`` beside it.

    Args:
        root: Directory that receives ``docs/`` and ``mkdocs.yml``.
        mkdocs_text: Contents of ``mkdocs.yml``.
        pages: Markdown pages to create under ``docs/``, each holding a single character.

    Returns:
        The ``docs/`` directory and the ``mkdocs.yml`` path.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs, mkdocs_yml = make_docs_site(Path(tmp), "nav:\\n  - a.md\\n", pages=("a.md", "sub/b.md"))
        ...     sorted(path.name for path in docs.rglob("*.md")), mkdocs_yml.name
        (['a.md', 'b.md'], 'mkdocs.yml')
    """
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for page in pages:
        write_file(docs / page, "x")
    mkdocs_yml = root / "mkdocs.yml"
    write_file(mkdocs_yml, mkdocs_text)
    return docs, mkdocs_yml
