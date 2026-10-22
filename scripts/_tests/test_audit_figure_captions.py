# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: figure-caption audit script (WP-080, split off for the lint-hook move).

Covers ``scripts/lint/audit_figure_captions.py``'s public contract in isolation, on
synthetic files under ``tmp_path`` rather than against the live ``docs/`` tree --
the live tree is exercised instead by the pre-commit hook itself each time a
figure or the reproduction report changes. This file is the functional-core
check pytest owns; the hook is the lint gate that runs it.

The one exception is ``test_drawn_captions_match_source_strings``, kept
parametrized over the real committed figures: a regression here means the hook
itself would fail on the next commit that touches a figure, not just a new one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES = sorted((REPO_ROOT / "docs" / "figures").glob("*.svg"))


audit = load_script("audit_figure_captions")


class TestCheckFiguresPresent:
    """Tests for ``audit.check_figures_present``."""

    def test_flags_missing_figure(self, tmp_path: Path) -> None:
        """A figure the report links but that is absent from disk is named as missing."""
        figures = tmp_path / "docs" / "figures"
        figures.mkdir(parents=True)
        (figures / "present.svg").write_text("<svg/>", encoding="utf-8")
        (tmp_path / "docs" / "REPRODUCTION_REPORT.md").write_text(
            "![a](figures/present.svg)\n![b](figures/missing.svg)\n", encoding="utf-8"
        )

        violations = audit.check_figures_present(tmp_path)

        assert violations == ["report links missing figures: figures/missing.svg"]

    def test_empty_when_every_linked_figure_exists(self, tmp_path: Path) -> None:
        """A report linking only figures that exist on disk reports no violation."""
        figures = tmp_path / "docs" / "figures"
        figures.mkdir(parents=True)
        (figures / "present.svg").write_text("<svg/>", encoding="utf-8")
        (tmp_path / "docs" / "REPRODUCTION_REPORT.md").write_text("![a](figures/present.svg)\n", encoding="utf-8")

        assert audit.check_figures_present(tmp_path) == []

    def test_reads_split_report_pages(self, tmp_path: Path) -> None:
        """A figure linked as ``../figures/`` from a ``docs/report/`` page is checked like a root link.

        The report is one page per release section, one level below ``docs/``, so its figure links
        climb a directory; a check reading only the landing page would see no figure and go vacuous.
        """
        figures = tmp_path / "docs" / "figures"
        figures.mkdir(parents=True)
        (figures / "present.svg").write_text("<svg/>", encoding="utf-8")
        (tmp_path / "docs" / "REPRODUCTION_REPORT.md").write_text("landing\n", encoding="utf-8")
        (tmp_path / "docs" / "report").mkdir()
        (tmp_path / "docs" / "report" / "detection.md").write_text(
            "![a](../figures/present.svg)\n![b](../figures/missing.svg)\n", encoding="utf-8"
        )

        violations = audit.check_figures_present(tmp_path)

        assert violations == ["report links missing figures: figures/missing.svg"]


def test_check_captions_match_flags_caption_edited_without_regenerating(tmp_path: Path) -> None:
    """A comment claiming text the drawn glyphs don't back up is reported as mismatched."""
    figures = tmp_path / "docs" / "figures"
    figures.mkdir(parents=True)
    elements = "".join(
        f'<!-- label {i} -->\n<g transform="x"><use xlink:href="#DejaVuSans-4b"/></g>\n'
        for i in range(audit._MIN_TEXT_ELEMENTS)
    )
    (figures / "bad.svg").write_text(elements, encoding="utf-8")

    violations = audit.check_captions_match(tmp_path)

    assert len(violations) == 1
    assert violations[0].startswith("bad.svg: markup and drawing disagree for")


def test_check_captions_match_empty_when_svg_matches_own_render(tmp_path: Path) -> None:
    """A freshly rendered SVG, saved as-is, agrees with itself on every label."""
    figures = tmp_path / "docs" / "figures"
    figures.mkdir(parents=True)
    labels = [f"label {i}" for i in range(audit._MIN_TEXT_ELEMENTS)]
    figure = plt.figure()
    for position, label in enumerate(labels):
        figure.text(0.0, position / len(labels), label)
    figure.savefig(figures / "clean.svg", format="svg")
    plt.close(figure)

    assert audit.check_captions_match(tmp_path) == []


@pytest.mark.parametrize("figure", FIGURES, ids=lambda path: path.name)
def test_drawn_captions_match_source_strings(figure: Path) -> None:
    """Every label in a committed figure draws the string its own markup claims.

    Fails on a figure that was edited as text instead of regenerated, whatever
    the edit was for: a rename, a typo fix, a units correction. Parametrized
    over the real committed figures (rather than a single synthetic fixture)
    because a regression here means the pre-commit hook would fail on the next
    commit that touches any figure, not just a new one.
    """
    elements = audit._text_elements(figure.read_text(encoding="utf-8"))
    assert len(elements) >= audit._MIN_TEXT_ELEMENTS, f"{figure.name}: parsed only {len(elements)} text elements"

    reference = audit._render([label for label, _ in elements])
    mismatched = [label for label, drawn in elements if reference.get(label) != drawn]
    assert not mismatched, f"{figure.name}: markup and drawing disagree for {mismatched}"
