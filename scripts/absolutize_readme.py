# SPDX-License-Identifier: Apache-2.0
"""Rewrite the README's repository-relative links to absolute ones, for a PyPI build (WP-113b).

``pyproject.toml`` hands ``README.md`` to setuptools as the long description, and PyPI
renders that description with no notion of the repository it came from: every relative
target — ``docs/ASSUMPTIONS.md``, ``docs/figures/det_smoke_training.svg`` — resolves
against ``pypi.org`` and renders as a dead link or a broken image. The same file has to
stay relative in the git tree, where relative is what works, so the rewrite belongs to
the moment of packaging rather than to the file.

Nothing here runs implicitly. A build is only a *release* build when the operator says
so, and says which ref the links are pinned to: ``--ref v0.4.0`` or the environment
variable ``LUCID_YOLO_RELEASE_REF``. Without one, the script refuses and the ordinary
``make build`` ships the file exactly as committed.

The ref must be a tag or a full commit sha, never a branch. A wheel's README is a
snapshot of one tree, and a link into ``main`` describes whatever that branch holds when
a reader clicks it — which is how a released page starts documenting code that shipped
years later.

Two link classes, two hosts. Images resolve through ``raw.githubusercontent.com``, which
serves file bytes; document links resolve through ``github.com/<owner>/<repo>/blob``,
which serves the rendered page a reader wants. In-page anchors, ``mailto:`` and already
absolute URLs are left alone.

Usage::

    python scripts/absolutize_readme.py --ref v0.4.0   # rewrite, keeping a backup
    python scripts/absolutize_readme.py --revert       # restore the committed file

``make dist-pypi TAG=v0.4.0`` does both around a build, reverting whether or not the
build succeeded.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: ``--help`` one-liner. Spelled out rather than sliced off ``__doc__``, which is optional
#: at runtime under ``python -OO`` and typed as ``str | None``.
_DESCRIPTION = "Rewrite the README's repository-relative links to absolute ones, for a PyPI build."

#: Markdown inline link or image: ``[text](target)``, with a leading ``!`` for images.
#: Reference-style links and angle-bracketed targets are deliberately unmatched — the
#: README uses neither, and a regex that guessed at them would rewrite less predictably
#: than one that visibly does not handle them.
MARKDOWN_LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")

#: ``src="..."`` inside a raw HTML tag. The README is markdown, but an HTML ``<img>`` is
#: the usual way a figure gets a width attribute, and one arriving later must not slip
#: past this rewrite silently.
HTML_SRC = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")', flags=re.IGNORECASE)

#: Targets that are already resolvable from anywhere: absolute URLs, protocol-relative
#: URLs, in-page anchors, and mail links.
ABSOLUTE_PREFIXES = ("http://", "https://", "//", "#", "mailto:")

#: A release ref is a version tag or a full commit sha. Anything else — a branch, a short
#: sha, ``HEAD`` — names a moving or ambiguous tree; see the module docstring.
RELEASE_REF = re.compile(r"^(v\d+\.\d+\.\d+|[0-9a-f]{40})$")

#: Suffixes served as bytes rather than as a rendered page.
IMAGE_SUFFIXES = (".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp")

#: Where the untouched file is parked while the rewritten one is packaged.
DEFAULT_BACKUP = REPO_ROOT / "README.md.orig"


def repository_slug(pyproject: Path) -> str:
    """Read ``owner/name`` from the ``Homepage`` URL declared in ``pyproject.toml``.

    The slug is taken from packaging metadata rather than from ``git remote``, so that a
    checkout under any remote or directory name builds the same URLs, and so that there
    is exactly one place to correct if the repository is ever renamed.

    Args:
        pyproject: Path to the project's ``pyproject.toml``.

    Returns:
        The ``owner/name`` pair, without a trailing ``.git`` or slash.

    Raises:
        ValueError: If no ``project.urls.Homepage`` GitHub URL is declared.

    Examples:
        >>> import tempfile, pathlib
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = pathlib.Path(tmp) / "pyproject.toml"
        ...     _ = path.write_text(
        ...         '[project.urls]\\nHomepage = "https://github.com/Borda/lucid-YOLO"\\n'
        ...     )
        ...     repository_slug(path)
        'Borda/lucid-YOLO'
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    homepage = str(data.get("project", {}).get("urls", {}).get("Homepage", ""))
    match = re.match(r"https://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$", homepage)
    if match is None:
        raise ValueError(f"project.urls.Homepage is not a GitHub repository URL: {homepage!r}")
    return match.group(1)


def is_relative(target: str) -> bool:
    """Report whether ``target`` needs a repository to resolve against.

    Args:
        target: A link or image target as written in the markdown.

    Returns:
        ``True`` when the target resolves only inside a checkout.

    Examples:
        >>> is_relative("docs/ASSUMPTIONS.md")
        True
        >>> [is_relative(t) for t in ("https://pypi.org", "#quickstart", "mailto:a@b.c")]
        [False, False, False]
    """
    return not target.startswith(ABSOLUTE_PREFIXES)


def absolute_url(target: str, slug: str, ref: str, *, as_image: bool) -> str:
    """Build the absolute URL a relative ``target`` becomes at ``ref``.

    Args:
        target: Repository-relative path, optionally carrying a ``#fragment``.
        slug: ``owner/name`` of the repository.
        ref: Tag or full commit sha the link is pinned to.
        as_image: Serve the raw bytes rather than the rendered page.

    Returns:
        An absolute ``raw.githubusercontent.com`` or ``github.com/.../blob`` URL.

    Examples:
        >>> absolute_url("docs/figures/det.svg", "Borda/lucid-YOLO", "v0.4.0", as_image=True)
        'https://raw.githubusercontent.com/Borda/lucid-YOLO/v0.4.0/docs/figures/det.svg'
        >>> absolute_url("docs/ROADMAP.md#phase-9", "Borda/lucid-YOLO", "v0.4.0", as_image=False)
        'https://github.com/Borda/lucid-YOLO/blob/v0.4.0/docs/ROADMAP.md#phase-9'
    """
    path = target.lstrip("./")
    host = f"https://raw.githubusercontent.com/{slug}" if as_image else f"https://github.com/{slug}/blob"
    return f"{host}/{ref}/{path}"


def absolutize(text: str, slug: str, ref: str) -> str:
    """Rewrite every repository-relative link and image in ``text`` to an absolute URL.

    Idempotent: a second pass finds only absolute targets and changes nothing, so a build
    that reruns the rewrite cannot double-prefix a URL.

    Args:
        text: The README source.
        slug: ``owner/name`` of the repository.
        ref: Tag or full commit sha the links are pinned to.

    Returns:
        The rewritten source.

    Examples:
        >>> src = "See [the register](docs/ASSUMPTIONS.md).\\n![curves](docs/figures/det.svg)\\n"
        >>> print(absolutize(src, "Borda/lucid-YOLO", "v0.4.0"))
        See [the register](https://github.com/Borda/lucid-YOLO/blob/v0.4.0/docs/ASSUMPTIONS.md).
        ![curves](https://raw.githubusercontent.com/Borda/lucid-YOLO/v0.4.0/docs/figures/det.svg)
        <BLANKLINE>
        >>> absolutize(absolutize(src, "o/r", "v0.1.0"), "o/r", "v0.1.0") == absolutize(src, "o/r", "v0.1.0")
        True
    """

    def _link(match: re.Match[str]) -> str:
        bang, label, target = match.groups()
        if not is_relative(target):
            return match.group(0)
        as_image = bool(bang) or target.lower().endswith(IMAGE_SUFFIXES)
        return f"{bang}[{label}]({absolute_url(target, slug, ref, as_image=as_image)})"

    def _src(match: re.Match[str]) -> str:
        head, target, tail = match.groups()
        if not is_relative(target):
            return match.group(0)
        return f"{head}{absolute_url(target, slug, ref, as_image=True)}{tail}"

    return HTML_SRC.sub(_src, MARKDOWN_LINK.sub(_link, text))


def count_relative(text: str) -> int:
    """Count the link and image targets in ``text`` that only resolve inside a checkout.

    Args:
        text: The README source.

    Returns:
        How many targets need a repository to resolve against.

    Examples:
        >>> count_relative("[a](docs/A.md) [b](https://pypi.org) ![c](docs/figures/det.svg)")
        2
    """
    targets = [match.group(3) for match in MARKDOWN_LINK.finditer(text)]
    targets += [match.group(2) for match in HTML_SRC.finditer(text)]
    return sum(1 for target in targets if is_relative(target))


def rewrite(readme: Path, backup: Path, slug: str, ref: str) -> int:
    """Rewrite ``readme`` in place, parking the committed bytes at ``backup``.

    Args:
        readme: The README to rewrite.
        backup: Where the untouched file is stored for :func:`revert`.
        slug: ``owner/name`` of the repository.
        ref: Tag or full commit sha the links are pinned to.

    Returns:
        How many targets were made absolute.

    Raises:
        FileExistsError: If a backup is already present — an earlier rewrite was never
            reverted, and overwriting it would destroy the only copy of the committed file.

    Examples:
        >>> import tempfile, pathlib
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     readme = pathlib.Path(tmp) / "README.md"
        ...     _ = readme.write_text("![c](docs/figures/det.svg)\\n")
        ...     rewrite(readme, pathlib.Path(tmp) / "README.md.orig", "o/r", "v0.4.0")
        1
    """
    if backup.exists():
        raise FileExistsError(f"{backup} exists: an earlier rewrite was never reverted")
    original = readme.read_text(encoding="utf-8")
    backup.write_text(original, encoding="utf-8")
    readme.write_text(absolutize(original, slug, ref), encoding="utf-8")
    return count_relative(original)


def revert(readme: Path, backup: Path) -> None:
    """Restore ``readme`` from ``backup`` and remove the backup.

    Args:
        readme: The rewritten README.
        backup: The parked committed bytes.

    Raises:
        FileNotFoundError: If no backup is present — there is nothing to restore, and
            silently succeeding would leave a rewritten README looking committed.

    Examples:
        >>> import tempfile, pathlib
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     readme, backup = pathlib.Path(tmp) / "README.md", pathlib.Path(tmp) / "b.orig"
        ...     _ = readme.write_text("rewritten")
        ...     _ = backup.write_text("committed")
        ...     revert(readme, backup)
        ...     (readme.read_text(), backup.exists())
        ('committed', False)
    """
    if not backup.exists():
        raise FileNotFoundError(f"{backup} is missing: nothing to restore")
    readme.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    backup.unlink()


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        The configured parser.

    Examples:
        >>> build_parser().parse_args(["--ref", "v0.4.0"]).ref
        'v0.4.0'
    """
    parser = argparse.ArgumentParser(description=_DESCRIPTION, allow_abbrev=False)
    parser.add_argument(
        "--ref",
        default=os.environ.get("LUCID_YOLO_RELEASE_REF", ""),
        help="tag (v0.MINOR.PATCH) or full commit sha the links are pinned to",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("LUCID_YOLO_REPO", ""),
        help="owner/name override; defaults to project.urls.Homepage",
    )
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md", help="README to rewrite")
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP, help="where the committed bytes are parked")
    parser.add_argument("--revert", action="store_true", help="restore the README from its backup and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the rewrite or the revert named on the command line.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: ``0`` on success, ``1`` on a refusal.

    Examples:
        >>> main(["--ref", "main"])
        release ref must be a v0.MINOR.PATCH tag or a full commit sha, got 'main'
        1
    """
    args = build_parser().parse_args(argv)

    if args.revert:
        revert(args.readme, args.backup)
        print(f"restored {args.readme} from {args.backup}")
        return 0

    if not RELEASE_REF.match(args.ref):
        got = args.ref or ""
        print(f"release ref must be a v0.MINOR.PATCH tag or a full commit sha, got {got!r}")
        return 1

    slug = args.repo or repository_slug(REPO_ROOT / "pyproject.toml")
    count = rewrite(args.readme, args.backup, slug, args.ref)
    print(f"{args.readme}: {count} relative targets pinned to {slug}@{args.ref}; backup at {args.backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
