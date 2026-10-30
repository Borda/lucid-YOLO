# SPDX-License-Identifier: Apache-2.0
"""Cross-links between work-package pages and every docs page that names one (WP-196, D28).

Each work package with findings has one page, ``docs/work_packages/wp-<id>.md``. Two things keep those pages reachable
from each other and from the rest of the docs:

* **Mentions are links.** Every ``WP-<id>`` written in a ``docs/`` page, whose page exists, is a markdown link to it.
  Headings, code, HTML comments and text already inside a link are left alone, and so is a range (``WP-142…144``),
  which names no single page. A WP page never links to itself.
* **Each WP page opens with a generated header**, between its H1 and its first section marker, so the phase pages that
  include the sections never show it: the log pages that include it, the WPs it depends on and the ones that depend on
  it (both read off the roadmap's Dep column), and the previous and next WP that has a page.

Both are derived, so both are checked rather than trusted, and ``--fix`` writes them. The header in particular is a
copy of roadmap data, which is exactly what rots (D28): adding a row that depends on WP-125 changes WP-125's own header.

Links are written relative to the page that holds them, and every page sits at most one directory below ``docs/``.
That matters for text the snippets extension inlines: a WP section lands in ``research/`` or ``engineering/``,
a report table in ``model_cards/`` or ``method/``, all one level down, so ``../work_packages/…`` written in the source
still resolves on the page that shows it.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_wp_links.py          # report
        $ python scripts/lint/audit_wp_links.py --fix    # rewrite links and headers
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `scripts.lint` resolves under the hook
from scripts.lint._common import REPO_ROOT, FenceTracker, print_failure, sorted_rglob

DEFAULT_DOCS_DIR = REPO_ROOT / "docs"

#: Directory, under ``docs/``, holding one page per work package.
_WP_DIR = "work_packages"

#: The log directories whose phase pages include work-package sections, with the label a WP header gives each.
_LOGS = {"research": "Research log", "engineering": "Engineering log"}

#: A work-package mention: ``WP-125``, ``WP-091b``. The look-arounds reject a range (``WP-142…144``, or the same
#: with an en dash) and an id glued into a longer token.
_MENTION = re.compile(r"(?<![\w/-])WP-(\d{3}[a-z]?)(?![\w\u2026\u2013-])")

#: Inline spans a mention inside of is not prose: a code span, an HTML comment or tag, and a whole markdown link
#: or image (its text included, so an existing link is kept).
_SKIP_SPAN = re.compile(r"`[^`\n]*`|<!--.*?-->|!?\[[^\]\n]*\]\([^)\n]*\)|<[^>\n]+>")

#: Lines whose mentions are never linked: headings and snippet include lines.
_SKIP_LINE = re.compile(r"^\s*(?:#{1,6}\s|--8<--)")

#: A fenced code block's opening or closing line.
_FENCE = re.compile(r"^\s*(?:>\s*)*(```|~~~)")

#: First line of a WP page's first included section; the header ends just above it.
_SECTION_START = "<!-- --8<-- [start:"

#: A roadmap row: its id and its cells.
_ROW = re.compile(r"^\| (\d{3}[a-z]?) \|")

#: Each WP id mapped to ``(log label, phase page)`` for every phase page including it.
_Logs = dict[str, list[tuple[str, Path]]]


def _table_cells(line: str) -> list[str]:
    """Split one markdown table row into its cells, honoring backslash-escaped pipes.

    Examples:
        >>> _table_cells(r"| a | b\\|c | d |")
        ['a', 'b|c', 'd']
    """
    cells, current, escaped = [], "", False
    for char in line.strip().strip("|"):
        if escaped:
            current, escaped = current + char, False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append(current.strip())
            current = ""
        else:
            current += char
    return [*cells, current.strip()]


def roadmap_deps(roadmap: str) -> dict[str, list[str]]:
    """Each roadmap row's id mapped to the ids its Dep column names, in roadmap order.

    Examples:
        >>> roadmap_deps("| 001 | a | s | d | — | ✅ |\\n| 002 | b | s | d | 001, 003 | ⬜ |\\n")
        {'001': [], '002': ['001', '003']}
    """
    deps = {}
    for line in roadmap.splitlines():
        match = _ROW.match(line)
        if match:
            cell = _table_cells(line)[4]
            deps[match.group(1)] = [dep.strip() for dep in cell.split(",") if re.fullmatch(r"\d{3}[a-z]?", dep.strip())]
    return deps


def _href(page: Path, docs_dir: Path, wp_id: str) -> str:
    """Link target from ``page`` to WP ``wp_id``'s page, relative to the page's directory.

    Examples:
        >>> docs = Path("docs")
        >>> _href(docs / "ROADMAP.md", docs, "125"), _href(docs / "report" / "a.md", docs, "125")
        ('work_packages/wp-125.md', '../work_packages/wp-125.md')
    """
    depth = len(page.relative_to(docs_dir).parts) - 1
    return f"{'../' * depth}{_WP_DIR}/wp-{wp_id}.md"


def _own_id(page: Path, docs_dir: Path) -> str | None:
    """The WP id a page belongs to, when it is a work-package page.

    Examples:
        >>> docs = Path("docs")
        >>> _own_id(docs / "work_packages" / "wp-091b.md", docs), _own_id(docs / "ROADMAP.md", docs)
        ('091b', None)
    """
    if page.parent == docs_dir / _WP_DIR:
        return page.stem.removeprefix("wp-")
    return None


def mentions(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield ``(start, end, id)`` of each prose ``WP-<id>`` mention in ``text``.

    Skipped: fenced code, headings, snippet lines, code spans, HTML comments and tags,
    and anything inside a markdown link.

    Examples:
        >>> text = "See WP-125 and [WP-126](x.md), `WP-127`, WP-142…144.\\n## WP-128\\n"
        >>> [wp_id for _, _, wp_id in mentions(text)]
        ['125']
    """
    offset = 0
    fences = FenceTracker(_FENCE)
    for line in text.splitlines(keepends=True):
        if not fences.update(line) and fences.language is None and not _SKIP_LINE.match(line):
            masked = _SKIP_SPAN.sub(lambda span: " " * len(span.group()), line)
            for match in _MENTION.finditer(masked):
                yield offset + match.start(), offset + match.end(), match.group(1)
        offset += len(line)


def link_mentions(text: str, page: Path, docs_dir: Path, pages: set[str]) -> str:
    """``text`` with every prose mention of a WP in ``pages`` turned into a link.

    Examples:
        >>> docs = Path("docs")
        >>> link_mentions("WP-125 and WP-999.\\n", docs / "report" / "a.md", docs, {"125"})
        '[WP-125](../work_packages/wp-125.md) and WP-999.\\n'
    """
    own = _own_id(page, docs_dir)
    out, last = [], 0
    for start, end, wp_id in mentions(text):
        if wp_id in pages and wp_id != own:
            out += [text[last:start], f"[{text[start:end]}]({_href(page, docs_dir, wp_id)})"]
            last = end
    return "".join([*out, text[last:]])


def _wp_list(ids: list[str], page: Path, docs_dir: Path, pages: set[str]) -> str:
    """Comma-separated WP ids, linked where a page exists; ``none`` when empty.

    Examples:
        >>> docs = Path("docs")
        >>> _wp_list(["001", "002"], docs / "work_packages" / "wp-003.md", docs, {"001"})
        '[WP-001](../work_packages/wp-001.md), WP-002'
        >>> _wp_list([], docs / "work_packages" / "wp-003.md", docs, set())
        'none'
    """
    shown = [f"[WP-{i}]({_href(page, docs_dir, i)})" if i in pages else f"WP-{i}" for i in ids]
    return ", ".join(shown) or "none"


def _log_pages(docs_dir: Path) -> _Logs:
    """Each WP id mapped to ``(log label, phase page)`` for every phase page including it.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     (docs / "research").mkdir()
        ...     _ = (docs / "research" / "p1.md").write_text('--8<-- "work_packages/wp-001.md:research"\\n')
        ...     _log_pages(docs)["001"][0][0]
        'Research log'
    """
    found: _Logs = {}
    for log, label in _LOGS.items():
        for phase in sorted((docs_dir / log).glob("*.md")):
            text = phase.read_text(encoding="utf-8")
            for wp_id in dict.fromkeys(re.findall(rf'--8<-- "{_WP_DIR}/wp-(\w+)\.md:', text)):
                found.setdefault(wp_id, []).append((label, phase))
    return found


def _title(page: Path) -> str:
    """A page's H1 without the ``#`` marker and its leading emoji.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     page = Path(tmp) / "p.md"
        ...     _ = page.write_text("# 📦 Phase 10 — Consolidation\\n", encoding="utf-8")
        ...     _title(page)
        'Phase 10 — Consolidation'
    """
    first = page.read_text(encoding="utf-8").splitlines()[0].removeprefix("# ")
    lead, _, rest = first.partition(" ")
    return rest if rest and not any(c.isascii() and c.isalnum() for c in lead) else first


def header(wp_id: str, docs_dir: Path, deps: dict[str, list[str]], pages: set[str], logs: _Logs) -> str:
    """The generated header block of WP ``wp_id``'s page: logs, dependencies, neighbours.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     deps = {"001": [], "002": ["001"], "003": ["002"]}
        ...     print(header("002", docs, deps, {"001", "002"}, {}), end="")
        > 🔗 Depends on [WP-001](../work_packages/wp-001.md) · Needed by WP-003
        >
        > ⬅️ [WP-001](../work_packages/wp-001.md) · [Roadmap](../ROADMAP.md)
    """
    page = docs_dir / _WP_DIR / f"wp-{wp_id}.md"
    lines = [
        f"> {label} · [{_title(phase)}](../{phase.relative_to(docs_dir).as_posix()})"
        for label, phase in logs.get(wp_id, [])
    ]
    needed_by = [other for other, needs in deps.items() if wp_id in needs]
    lines.append(
        f"> 🔗 Depends on {_wp_list(deps.get(wp_id, []), page, docs_dir, pages)}"
        f" · Needed by {_wp_list(needed_by, page, docs_dir, pages)}"
    )
    order = [i for i in deps if i in pages]
    at = order.index(wp_id) if wp_id in order else -1
    nav = ["[Roadmap](../ROADMAP.md)"]
    if at > 0:
        nav.insert(0, f"⬅️ [WP-{order[at - 1]}]({_href(page, docs_dir, order[at - 1])})")
    if 0 <= at < len(order) - 1:
        nav.append(f"[WP-{order[at + 1]}]({_href(page, docs_dir, order[at + 1])}) ➡️")
    lines.append("> " + " · ".join(nav))
    return "\n>\n".join(lines) + "\n"


def with_header(text: str, block: str) -> str:
    """A WP page's text with everything between its H1 and first section replaced by ``block``.

    Examples:
        >>> with_header("# T\\n\\n> old\\n\\n<!-- --8<-- [start:research] -->\\nx\\n", "> new\\n")
        '# T\\n\\n> new\\n\\n<!-- --8<-- [start:research] -->\\nx\\n'
    """
    title, _, rest = text.partition("\n")
    at = rest.find(_SECTION_START)
    body = rest[at:] if at >= 0 else rest.lstrip("\n")
    return f"{title}\n\n{block}\n{body}"


def expected(page: Path, docs_dir: Path, deps: dict[str, list[str]], pages: set[str], logs: _Logs) -> str:
    """What ``page`` should read: mentions linked and, on a WP page, its header regenerated."""
    text = page.read_text(encoding="utf-8")
    own = _own_id(page, docs_dir)
    if own is not None:
        text = with_header(text, header(own, docs_dir, deps, pages, logs))
    return link_mentions(text, page, docs_dir, pages)


def find_stale(docs_dir: Path, fix: bool = False) -> list[str]:
    """Docs pages whose WP links or WP header differ from what they derive to; rewritten when ``fix``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     _ = (docs / "ROADMAP.md").write_text("| 001 | a | s | d | — | ✅ |\\n", encoding="utf-8")
        ...     (docs / "work_packages").mkdir()
        ...     _ = (docs / "work_packages" / "wp-001.md").write_text("# T\\n", encoding="utf-8")
        ...     _ = (docs / "index.md").write_text("See WP-001.\\n", encoding="utf-8")
        ...     find_stale(docs)
        ['index.md', 'work_packages/wp-001.md']
    """
    deps = roadmap_deps((docs_dir / "ROADMAP.md").read_text(encoding="utf-8"))
    pages = {path.stem.removeprefix("wp-") for path in (docs_dir / _WP_DIR).glob("wp-*.md")}
    logs = _log_pages(docs_dir)
    stale = []
    for page in sorted_rglob(docs_dir, "*.md"):
        want = expected(page, docs_dir, deps, pages, logs)
        if want != page.read_text(encoding="utf-8"):
            stale.append(page.relative_to(docs_dir).as_posix())
            if fix:
                page.write_text(want, encoding="utf-8")
    return stale


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, check (or fix) every docs page, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean or fixed, ``1`` when a page is stale and not fixed.
    """
    parser = argparse.ArgumentParser(description="Check that docs pages link every work-package mention.")
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR, help="docs directory to scan")
    parser.add_argument("--fix", action="store_true", help="rewrite stale links and WP headers in place")
    args = parser.parse_args(argv)

    stale = find_stale(args.docs_dir, fix=args.fix)
    if stale and not args.fix:
        return print_failure("wp-links audit", stale, "page(s) out of date; run with --fix")
    print(f"wp-links audit {'fixed ' + str(len(stale)) + ' page(s)' if stale else 'clean'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
