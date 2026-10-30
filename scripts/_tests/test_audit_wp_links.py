# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the work-package cross-link audit script (WP-196).

Covers ``scripts/lint/audit_wp_links.py``'s public contract in isolation, on synthetic files under ``tmp_path`` rather
than against the live docs tree -- the live tree is exercised instead by the ``wp-links`` pre-commit hook each time a
``docs/*.md`` changes. This file is the functional-core check pytest owns; the hook is the lint gate that runs it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]


audit = load_script("audit_wp_links")

_ROADMAP = "| 001 | a | s | d | — | ✅ |\n| 002 | b | s | d | 001 | ✅ |\n| 003 | c | s | d | 002 | ⬜ |\n"


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    """A docs tree with a three-row roadmap, pages for WP-001 and WP-002, and a phase page."""
    (tmp_path / "ROADMAP.md").write_text(_ROADMAP, encoding="utf-8")
    (tmp_path / "work_packages").mkdir()
    for wp_id in ("001", "002"):
        (tmp_path / "work_packages" / f"wp-{wp_id}.md").write_text(
            f"# WP-{wp_id}\n\n<!-- --8<-- [start:research] -->\nbody\n<!-- --8<-- [end:research] -->\n",
            encoding="utf-8",
        )
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "phase-01.md").write_text(
        '# 📦 Phase 1 — Data\n\n--8<-- "work_packages/wp-001.md:research"\n\n'
        '--8<-- "work_packages/wp-002.md:research"\n',
        encoding="utf-8",
    )
    audit.find_stale(tmp_path, fix=True)
    return tmp_path


class TestMentions:
    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("## WP-125 heading\n", id="heading"),
            pytest.param("`WP-125`\n", id="code-span"),
            pytest.param("```\nWP-125\n```\n", id="fence"),
            pytest.param("[the WP-125 run](x.md)\n", id="inside-link"),
            pytest.param("<!-- WP-125 -->\n", id="comment"),
            pytest.param("WP-125…127\n", id="ellipsis-range"),
            pytest.param("WP-125\N{EN DASH}127\n", id="en-dash-range"),
        ],
    )
    def test_skips_non_prose(self, text: str) -> None:
        """A mention in code, a heading, a link, a comment or a range is not linked.

        Each of these either already points somewhere or names no single page; linking it would nest links, break code,
        or point a range at its first member only.
        """
        assert list(audit.mentions(text)) == []

    def test_finds_suffixed_id(self) -> None:
        """A lettered id such as ``WP-091b`` is matched whole, suffix included.

        Twenty-nine WP pages carry a letter suffix; matching only the digits would link ``WP-091b`` to WP-091's page.
        """
        assert [wp_id for _, _, wp_id in audit.mentions("see WP-091b.\n")] == ["091b"]


class TestFindStale:
    def test_fix_links_mention_and_is_idempotent(self, docs: Path) -> None:
        """After ``--fix`` a prose mention is a link and a second check finds nothing stale.

        The hook's clean state has to be a fixed point of the fixer, or every commit after a fix would report the same
        pages again.
        """
        (docs / "index.md").write_text("See WP-001 and WP-003.\n", encoding="utf-8")
        audit.find_stale(docs, fix=True)
        assert (docs / "index.md").read_text(encoding="utf-8") == "See [WP-001](work_packages/wp-001.md) and WP-003.\n"
        assert audit.find_stale(docs) == []

    def test_reports_bare_mention(self, docs: Path) -> None:
        """A page naming a WP that has a page, without a link, is reported stale.

        This is the regression the hook guards: a new paragraph naming a WP as plain text.
        """
        (docs / "report").mkdir()
        (docs / "report" / "a.md").write_text("WP-002 measured it.\n", encoding="utf-8")
        assert audit.find_stale(docs) == ["report/a.md"]

    def test_reports_header_after_roadmap_change(self, docs: Path) -> None:
        """A new roadmap row depending on WP-002 makes WP-002's header stale.

        The header copies roadmap data, so the copy has to be re-derived and checked, not trusted: WP-002 gains
        a dependant it does not list.
        """
        roadmap = docs / "ROADMAP.md"
        roadmap.write_text(_ROADMAP + "| 004 | d | s | d | 002 | ⬜ |\n", encoding="utf-8")
        assert audit.find_stale(docs) == ["work_packages/wp-002.md"]


class TestHeader:
    def test_lists_log_dependencies_and_neighbours(self, docs: Path) -> None:
        """WP-002's header names its phase page, both roadmap directions and its predecessor.

        WP-003 has no page, so it is named as text; WP-001 has one, so it is a link.
        """
        text = (docs / "work_packages" / "wp-002.md").read_text(encoding="utf-8")
        assert text.startswith(
            "# WP-002\n\n"
            "> Research log · [Phase 1 — Data](../research/phase-01.md)\n>\n"
            "> 🔗 Depends on [WP-001](../work_packages/wp-001.md) · Needed by WP-003\n>\n"
            "> ⬅️ [WP-001](../work_packages/wp-001.md) · [Roadmap](../ROADMAP.md)\n\n"
            "<!-- --8<-- [start:research] -->\n"
        )

    def test_never_links_own_page(self, docs: Path) -> None:
        """A WP page's own sections keep their mention of that WP as text.

        A self-link would send a reader of the phase page to the page they are reading.
        """
        page = docs / "work_packages" / "wp-001.md"
        page.write_text(page.read_text(encoding="utf-8").replace("body", "WP-001 did it"), encoding="utf-8")
        assert audit.find_stale(docs) == []
