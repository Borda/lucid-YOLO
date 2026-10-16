# SPDX-License-Identifier: Apache-2.0
"""MkDocs build hooks: page titles, descriptions, and the copies agents read (WP-196).

Everything here is derived at build time from text the repository already holds, so
nothing it produces can drift from its source -- the same reason D28 includes findings
rather than copying them. Three things are added to the built site:

* **A title and description per page.** The sidebar keeps its short nav labels, but a
  browser tab, a search result or a link preview needs a page to name itself: thirteen
  pages were titled "Engineering" and every page carried the site's one description.
  ``page.meta`` takes a ``title`` and a ``description`` (front matter wins when a page
  sets one): work-package pages from their roadmap row, phase pages from their phase and
  log, every other page from its H1 and its first paragraph. ``overrides/main.html``
  renders them, with Open Graph tags and, on the home page, JSON-LD.
* **A markdown copy of every page**, at ``<page>/index.md``, with snippet includes
  resolved and relative links made absolute: an agent reading the page gets the text
  without the theme, and gets the assembled page rather than a bare ``--8<--`` line.
* **``llms.txt`` and ``llms-full.txt`` at the site root**: the first an index of every
  page by nav section with its description and markdown URL, the second the guide,
  method, results and model-card pages in full.

Registered under ``hooks:`` in ``mkdocs.yml``; ``make docs`` exercises it on every build.
"""

from __future__ import annotations

import json
import posixpath
import re
import tomllib
from pathlib import Path
from typing import Any

#: Length a description is cut to, at a word boundary: what a search result shows.
_DESCRIPTION_CHARS = 160

#: Nav sections whose pages ``llms-full.txt`` carries whole.
_FULL_SECTIONS = ("Home", "Results", "Models")

#: Title of the home page, which no H1 states: what the project is, for a search result.
_HOME_TITLE = "lucid-yolo: clean-room NMS-free YOLO in PyTorch Lightning"

_INCLUDE = re.compile(r'^--8<-- "([^"]+)"\s*$', re.M)
_SECTION_MARKER = re.compile(r"^.*--8<-- \[(?:start|end):[\w-]+\].*\n?", re.M)
_LINK = re.compile(r"(\]\()(?!https?:|mailto:|#)([^)\s#]+)(#[^)\s]*)?\)")
_COMMIT_PREFIX = re.compile(r"^\w+(?:\([^)]*\))?!?:\s*")
_ROW = re.compile(r"^\| (\d{3}[a-z]?) \|(.*)$", re.M)

#: State carried between build events: one record per page, and the nav.
_STATE: dict[str, Any] = {"pages": {}, "nav": None, "roadmap": {}}


def plain(text: str) -> str:
    """Markdown reduced to plain text: images dropped, links to their text, markup removed.

    Examples:
        >>> plain("See **the** [report](a.md), `n` and ![x](y.svg) <br> here.")
        'See the report, n and here.'
    """
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def clip(text: str, limit: int = _DESCRIPTION_CHARS) -> str:
    """``text`` cut at a word boundary to at most ``limit`` characters, with an ellipsis.

    Examples:
        >>> clip("one two three", 9)
        'one two…'
        >>> clip("short", 9)
        'short'
    """
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:—-") + "…"


def undecorated_h1(markdown: str) -> str:
    """A page's H1 without its leading emoji or code-span backticks; empty when it has none.

    Examples:
        >>> undecorated_h1("# 🎯 Detection — lucid-yolo `n`\\n\\ntext")
        'Detection — lucid-yolo n'
    """
    match = re.search(r"^# (.+)$", markdown, re.M)
    if not match:
        return ""
    lead, _, rest = match.group(1).partition(" ")
    title = rest if rest and not any(c.isascii() and c.isalnum() for c in lead) else match.group(1)
    return title.replace("`", "").strip()


def first_paragraph(markdown: str) -> str:
    """The first prose paragraph after the H1, as plain text; blockquote markers dropped.

    Headings, tables, images, anchors, comments and include lines are not prose.

    Examples:
        >>> first_paragraph("# T\\n\\n![a](b.svg)\\n\\n> The answer\\n> here.\\n\\nMore.")
        'The answer here.'
    """
    lines: list[str] = []
    for line in markdown.split("\n")[1:]:
        text = line.lstrip("> ").strip()
        skip = not text or re.match(r"^(#|\||<|--8<--|!\[|```)", text)
        if skip and lines:
            break
        if not skip:
            lines.append(text)
    return plain(" ".join(lines))


def page_title(src: str, markdown: str) -> str:
    """The title a page names itself by, distinct across the site.

    Examples:
        >>> page_title("work_packages/wp-125.md", "# 📦 WP-125 — `exp(ptl): Pose-smoke tier`")
        'WP-125 — Pose-smoke tier'
        >>> page_title("research/phase-04-musgd.md", "# 📉 Phase 4 — MuSGD")
        'Phase 4 — MuSGD — research log'
        >>> page_title("model_cards/obb.md", "# 🔄 Oriented detection — lucid-yolo `n`")
        'Oriented detection model card — lucid-yolo n'
        >>> page_title("report/detection.md", "# 🎯 0.1.0 — Detection")
        'Reproduction report v0.1.0: Detection'
    """
    h1 = undecorated_h1(markdown)
    top = src.split("/", maxsplit=1)[0]
    if src == "index.md":
        return _HOME_TITLE
    if top == "work_packages":
        wp, _, subject = h1.partition(" — ")
        return f"{wp} — {_COMMIT_PREFIX.sub('', subject)}"
    if top in ("research", "engineering"):
        return f"{h1} — {top} log"
    if top == "model_cards":
        task, _, model = h1.partition(" — ")
        return f"{task} model card — {model}"
    if top == "report":
        # A release section's H1 is "0.1.0 — Detection" (pinned by D10); its title reads "v0.1.0: Detection".
        release = re.match(r"(\d+\.\d+\.\d+) — (.+)", h1)
        return f"Reproduction report v{release[1]}: {release[2]}" if release else f"Reproduction report: {h1}"
    return h1


def roadmap_scopes(roadmap: str) -> dict[str, str]:
    """Each roadmap row's id mapped to its Scope cell, without the trailing log link.

    Examples:
        >>> roadmap_scopes("| 001 | `x` | Does a thing · [log](a.md) | d | — | ✅ |\\n")
        {'001': 'Does a thing'}
    """
    scopes = {}
    for wp_id, rest in _ROW.findall(roadmap):
        cells = re.split(r"(?<!\\) \| ", rest.strip(" |"))
        if len(cells) > 1:
            scopes[wp_id] = re.sub(r"\s*·\s*\[[^\]]*\]\([^)]*\)\s*$", "", cells[1])
    return scopes


def page_description(src: str, markdown: str, scopes: dict[str, str]) -> str:
    """A one-line description of a page, for a search result or an agent's index.

    Examples:
        >>> page_description("work_packages/wp-001.md", "# x", {"001": "Package **skeleton**"})
        'Package skeleton'
        >>> page_description("research/p.md", '# 🚰 Phase 1 — Data\\n--8<-- "work_packages/wp-004.md:research"', {})
        'Research log for Phase 1 — Data: the findings of WP-004, one section per work package.'
    """
    top = src.split("/", maxsplit=1)[0]
    if top == "work_packages":
        return clip(plain(scopes.get(Path(src).stem.removeprefix("wp-"), "")))
    ids = list(dict.fromkeys(re.findall(r'--8<-- "work_packages/wp-(\w+)\.md', markdown)))
    if top in ("research", "engineering") and ids:
        listed = ", ".join(f"WP-{i}" for i in ids)
        return clip(
            f"{top.capitalize()} log for {undecorated_h1(markdown)}: the findings of {listed}, "
            "one section per work package."
        )
    return clip(first_paragraph(markdown))


def resolve_includes(markdown: str, docs_dir: Path) -> str:
    """``markdown`` with each snippet include line replaced by the text it names.

    Section markers are dropped from the result: they are build instructions, not text.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     _ = (docs / "a.md").write_text("<!-- --8<-- [start:s] -->\\nbody\\n<!-- --8<-- [end:s] -->\\n")
        ...     resolve_includes('top\\n--8<-- "a.md:s"\\n', docs)
        'top\\nbody\\n'
    """

    def included(match: re.Match[str]) -> str:
        path, _, section = match.group(1).partition(":")
        text = (docs_dir / path).read_text(encoding="utf-8")
        if section:
            body = re.search(rf"\[start:{section}\][^\n]*\n(.*?)^[^\n]*\[end:{section}\]", text, re.S | re.M)
            text = body.group(1) if body else ""
        return text

    return _SECTION_MARKER.sub("", _INCLUDE.sub(included, markdown))


def absolute_links(markdown: str, src: str, urls: dict[str, str], site_url: str) -> str:
    """``markdown`` with links relative to page ``src`` rewritten to absolute site URLs.

    ``urls`` maps a docs-relative source path to its site path; a target it does not
    know is left as written.

    Examples:
        >>> absolute_links("[r](../ROADMAP.md#x)", "report/a.md", {"ROADMAP.md": "ROADMAP/"}, "https://s/")
        '[r](https://s/ROADMAP/#x)'
    """

    def rewrite(match: re.Match[str]) -> str:
        target = posixpath.normpath(posixpath.join(posixpath.dirname(src), match.group(2)))
        if target not in urls:
            return match.group(0)
        return f"{match.group(1)}{site_url}{urls[target]}{match.group(3) or ''})"

    return _LINK.sub(rewrite, markdown)


def _json_ld(config: Any) -> str:
    """The home page's ``SoftwareSourceCode`` description, read off ``mkdocs.yml`` and ``pyproject.toml``."""
    project = tomllib.loads((Path(config.config_file_path).parent / "pyproject.toml").read_text(encoding="utf-8"))
    return json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "SoftwareSourceCode",
            "name": config.site_name,
            "description": config.site_description,
            "url": config.site_url,
            "codeRepository": config.repo_url,
            "programmingLanguage": "Python",
            "license": project["project"]["license"],
            "keywords": project["project"].get("keywords", []),
            "author": {"@type": "Person", "name": config.site_author},
            "citation": "https://arxiv.org/abs/2606.03748",
        },
        ensure_ascii=False,
    ).replace("<", "\\u003c")  # inside <script>, a "</" in any value would end the element


def on_config(config: Any) -> Any:
    """Read the roadmap's scope cells once per build, for the work-package descriptions."""
    roadmap = Path(config.docs_dir) / "ROADMAP.md"
    _STATE.update(pages={}, nav=None, roadmap=roadmap_scopes(roadmap.read_text(encoding="utf-8")))
    return config


def on_nav(nav: Any, config: Any, files: Any) -> Any:
    """Keep the nav, which orders ``llms.txt`` by the same sections the tabs show."""
    _STATE["nav"] = nav
    return nav


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    """Set the page's title, description and markdown-copy URL, and record its assembled text."""
    src = page.file.src_uri
    if not src.endswith(".md"):
        return markdown
    page.meta.setdefault("title", page_title(src, markdown))
    page.meta.setdefault("description", page_description(src, markdown, _STATE["roadmap"]))
    page.meta["markdown_url"] = "index.md"
    if src == "index.md":
        # The home page sits under the Home nav section, so MkDocs' is_homepage is false.
        page.meta.update(home=True, json_ld=_json_ld(config))
    urls = {f.src_uri: f.url for f in files}
    text = resolve_includes(markdown, Path(config.docs_dir))
    _STATE["pages"][src] = {
        "title": page.meta["title"],
        "description": page.meta["description"],
        "url": f"{config.site_url}{page.url}",
        "dest": page.file.dest_uri.removesuffix(".html") + ".md",
        "markdown": absolute_links(text, src, urls, config.site_url),
    }
    return markdown


def toc_entries(items: list[Any]) -> int:
    """How many headings a page's table of contents lists below its H1.

    An H1 is the page title, not a place to jump to, so it is not counted; every heading
    nested under it, at any depth, is.

    Examples:
        >>> from types import SimpleNamespace as Node
        >>> toc_entries([Node(level=1, children=[Node(level=2, children=[]), Node(level=2, children=[])])])
        2
    """
    return sum((item.level > 1) + toc_entries(item.children) for item in items)


def on_page_context(context: Any, page: Any, config: Any, nav: Any) -> Any:
    """Hide a near-empty table of contents, and describe a notebook page.

    A right sidebar with one entry or none holds no navigation and takes the width a
    table-heavy register such as ``ASSUMPTIONS.md`` needs, so it is hidden there. The
    count is read off the rendered TOC, so included sections count. A notebook has no
    markdown source for the hooks above to read, so its description is set here.
    """
    if toc_entries(list(page.toc)) < 2:
        hide = page.meta.get("hide") or []
        page.meta["hide"] = [*hide, "toc"] if "toc" not in hide else hide
    if page.file.src_uri.endswith(".ipynb"):
        page.meta.setdefault(
            "description", f"Notebook: {page.title}. Source cells and a Colab badge; outputs are not stored."
        )
    return context


def _nav_entries(items: list[Any], trail: tuple[str, ...] = ()) -> list[tuple[str, str, str]]:
    """``(top section, label, source path)`` for every page in the nav, depth first."""
    entries = []
    for item in items:
        if getattr(item, "children", None):
            entries += _nav_entries(item.children, (*trail, item.title))
        elif getattr(item, "file", None):
            top = trail[0] if trail else item.title
            entries.append((top, " / ".join((*trail[1:], item.title)), item.file.src_uri))
    return entries


def _llms_line(label: str, record: dict[str, str]) -> str:
    """One ``llms.txt`` list entry: label linked to the markdown copy, then the description."""
    return f"- [{label}]({record['url']}index.md): {record['description']}"


def on_post_build(config: Any) -> None:
    """Write each page's markdown copy, then ``llms.txt`` and ``llms-full.txt``."""
    site = Path(config.site_dir)
    pages: dict[str, dict[str, str]] = _STATE["pages"]
    for copy in pages.values():
        (site / copy["dest"]).write_text(copy["markdown"], encoding="utf-8")

    sections: dict[str, list[str]] = {}
    full: list[str] = []
    for top, label, src in _nav_entries(_STATE["nav"].items):
        if src not in pages:
            sections.setdefault(top, []).append(f"- [{label}]({config.site_url}{src.removesuffix('.ipynb')}/)")
            continue
        sections.setdefault(top, []).append(_llms_line(label, pages[src]))
        if top in _FULL_SECTIONS:
            full.append(f"<!-- source: {pages[src]['url']} -->\n\n{pages[src]['markdown'].strip()}\n")
    sections["Optional"] = [
        _llms_line(record["title"], record) for src, record in sorted(pages.items()) if src.startswith("work_packages/")
    ]

    head = [
        f"# {config.site_name}",
        "",
        f"> {config.site_description}",
        "",
        "Every page below links to its markdown copy (`<page>/index.md`), with includes resolved and links absolute. "
        "`llms-full.txt` holds the guide, method, results and model-card pages in full. "
        f"Source: {config.repo_url}",
    ]
    body = [f"\n\n## {name}\n\n" + "\n".join(lines) for name, lines in sections.items()]
    (site / "llms.txt").write_text("\n".join(head) + "\n" + "".join(body) + "\n", encoding="utf-8")
    (site / "llms-full.txt").write_text("\n".join(full), encoding="utf-8")
