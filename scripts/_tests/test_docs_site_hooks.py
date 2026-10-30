# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the MkDocs build hooks behind page metadata and llms.txt (WP-196).

Covers ``scripts/docs_site_hooks.py``'s pure functions on synthetic text. The hook events themselves run on every
``make docs``, which builds the live tree strictly; this file pins the derivations those events rely on.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from script_loader import load_script

hooks = load_script("docs_site_hooks")


class TestPageTitle:
    @pytest.mark.parametrize(
        ("src", "h1", "title"),
        [
            pytest.param(
                "research/p.md", "# 🎯 Phase 6 — Evaluation", "Phase 6 — Evaluation — research log", id="research"
            ),
            pytest.param(
                "engineering/p.md",
                "# 🎯 Phase 6 — Evaluation",
                "Phase 6 — Evaluation — engineering log",
                id="engineering",
            ),
            pytest.param("TRAINING.md", "# 🏋️ Launching a training run", "Launching a training run", id="plain"),
        ],
    )
    def test_distinguishes_pages_sharing_heading(self, src: str, h1: str, title: str) -> None:
        """A phase's research and engineering pages get different titles from one H1.

        Both logs title their phase page identically; without the log suffix every phase with both pages shows two
        identical search results.
        """
        assert hooks.page_title(src, h1) == title

    def test_home_page_names_project(self) -> None:
        """The home page title says what the project is, not the nav label "Overview".

        It is the one title a search result shows for the bare site URL.
        """
        assert hooks.page_title("index.md", "# 🔦 lucid-yolo") == hooks._HOME_TITLE


class TestPageDescription:
    def test_work_package_reads_roadmap_scope(self) -> None:
        """A WP page is described by its roadmap Scope cell, stripped of markup and the log link.

        The roadmap is where a WP's one-paragraph summary already lives.
        """
        scopes = hooks.roadmap_scopes("| 007 | `x` | Adds **a** [thing](a.md) · [log](b.md) | d | — | ✅ |\n")
        assert hooks.page_description("work_packages/wp-007.md", "# x", scopes) == "Adds a thing"

    def test_long_paragraph_clipped_at_word(self) -> None:
        """A first paragraph longer than a search snippet ends at a word, with an ellipsis.

        A cut mid-word reads as an error in a result list.
        """
        description = hooks.page_description("TRAINING.md", "# T\n\n" + "word " * 60, {})
        assert len(description) <= hooks._DESCRIPTION_CHARS and description.endswith("word…")


class TestMarkdownCopy:
    def test_resolves_include_and_absolutises_link(self, tmp_path: Path) -> None:
        """A page's markdown copy carries the included text with its links made absolute.

        An agent reading the copy otherwise gets a bare include line, or a relative link that resolves against the
        copy's own URL rather than the page's.
        """
        (tmp_path / "report").mkdir()
        (tmp_path / "report" / "a.md").write_text(
            "<!-- --8<-- [start:eval] -->\nsee [WP-1](../work_packages/wp-001.md)\n<!-- --8<-- [end:eval] -->\n",
            encoding="utf-8",
        )
        text = hooks.resolve_includes('--8<-- "report/a.md:eval"\n', tmp_path)
        urls = {"work_packages/wp-001.md": "work_packages/wp-001/"}
        copy = hooks.absolute_links(text, "model_cards/card.md", urls, "https://site/")
        assert copy == "see [WP-1](https://site/work_packages/wp-001/)\n"


class TestTocEntries:
    @pytest.mark.parametrize(
        ("h2_count", "entries"),
        [
            pytest.param(0, 0, id="title-only"),
            pytest.param(1, 1, id="one-section"),
            pytest.param(3, 3, id="three-sections"),
        ],
    )
    def test_counts_headings_below_title(self, h2_count: int, entries: int) -> None:
        """The H1 is not counted, so a register with one section counts one entry.

        One entry is the threshold below which the right sidebar is hidden; counting the H1 would keep it on
        ASSUMPTIONS.md, whose only content is a wide table.
        """
        h2s = [SimpleNamespace(level=2, children=[]) for _ in range(h2_count)]
        assert hooks.toc_entries([SimpleNamespace(level=1, children=h2s)]) == entries


class TestRoadmapTagFilter:
    @pytest.mark.parametrize(
        ("target", "dropped"),
        [
            ("PRINCIPAL", True),
            ("GPU", True),
            ("DATA", True),
            ("lucid_yolo.data.Missing", False),
            ("PRINCIPAL-ish", False),
        ],
    )
    def test_drops_only_roadmap_tags(self, target: str, dropped: bool) -> None:
        """Only an unresolved roadmap tag is quieted; any other missing target still warns.

        `[GPU][PRINCIPAL]` is a reference link to nothing and renders as its own text, so its warning is noise; a
        misspelt API identifier is a broken link and must fail `--strict`.
        """
        message = f"ROADMAP.md: Could not find cross-reference target '{target}'"

        assert hooks.is_roadmap_tag_miss(message) is dropped
