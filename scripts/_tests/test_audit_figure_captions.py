# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: figure-caption audit script (WP-080, split off for the lint-hook move).

Covers ``scripts/lint/audit_figure_captions.py``'s public contract in isolation, on
synthetic files under ``tmp_path`` rather than against the live ``docs/`` tree --
the live tree is exercised instead by the pre-commit hook itself each time a
figure or the reproduction report changes. This file is the functional-core
check pytest owns; the hook is the lint gate that runs it.

The one exception is ``test_drawn_captions_match_their_source_strings``, kept
parametrized over the real committed figures: a regression here means the hook
itself would fail on the next commit that touches a figure, not just a new one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "scripts" / "lint" / "audit_figure_captions.py"
FIGURES = sorted((REPO_ROOT / "docs" / "figures").glob("*.svg"))


def _load_audit() -> ModuleType:
    """Load ``scripts/lint/audit_figure_captions.py`` as an importable module.

    Examples:
        >>> module = _load_audit()
        >>> module.__name__
        'audit_figure_captions'
        >>> callable(module.check_captions_match)
        True
    """
    spec = importlib.util.spec_from_file_location("audit_figure_captions", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def test_check_figures_present_flags_a_missing_figure(tmp_path: Path) -> None:
    """A figure the report links but that is absent from disk is named as missing."""
    figures = tmp_path / "docs" / "figures"
    figures.mkdir(parents=True)
    (figures / "present.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "docs" / "REPRODUCTION_REPORT.md").write_text(
        "![a](figures/present.svg)\n![b](figures/missing.svg)\n", encoding="utf-8"
    )

    violations = audit.check_figures_present(tmp_path)

    assert violations == ["report links missing figures: figures/missing.svg"]


def test_check_figures_present_is_empty_when_every_linked_figure_exists(tmp_path: Path) -> None:
    """A report linking only figures that exist on disk reports no violation."""
    figures = tmp_path / "docs" / "figures"
    figures.mkdir(parents=True)
    (figures / "present.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "docs" / "REPRODUCTION_REPORT.md").write_text("![a](figures/present.svg)\n", encoding="utf-8")

    assert audit.check_figures_present(tmp_path) == []


def test_check_captions_match_flags_a_caption_edited_without_regenerating(tmp_path: Path) -> None:
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


def test_check_captions_match_is_empty_when_the_svg_matches_its_own_render(tmp_path: Path) -> None:
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
def test_drawn_captions_match_their_source_strings(figure: Path) -> None:
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


def test_the_live_report_links_no_missing_figure() -> None:
    """The real reproduction report the pre-commit hook scans links no missing figure.

    This is the one place ``check_figures_present`` touches the live tree rather
    than a synthetic one: a regression here means the hook itself would fail on
    the next commit that touches the report or a figure.
    """
    assert audit.check_figures_present(REPO_ROOT) == []
