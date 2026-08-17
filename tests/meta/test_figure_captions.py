# SPDX-License-Identifier: Apache-2.0
"""Meta tests: committed figures draw what their markup says they draw (WP-080).

Matplotlib writes a text element twice. The characters become glyph outlines,
and the source string is repeated beside them as an XML comment that nothing
renders. A search-and-replace over a committed SVG therefore edits the comment
and leaves the drawing untouched, and the file then reports one caption and
displays another -- which is exactly what a repository-wide tier rename did to
``det_smoke_training.svg``, undetected, with every gate green: no checker here
reads a figure.

These tests read one. Each text group's glyph sequence is compared against a
fresh render of that group's own comment, so the two halves of every label must
still agree. Both sides come from the figure, so no caption is pinned in this
file and a legitimately retitled figure passes as soon as it is regenerated.
"""

import io
import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES = sorted((REPO_ROOT / "docs" / "figures").glob("*.svg"))

#: A text element: its source string as a comment, then the group that draws it.
_TEXT_GROUP = re.compile(r"<!-- (.*?) -->\s*<g [^>]*>(.*?)</g>", re.DOTALL)
#: One drawn glyph: the font it comes from and its index within that font.
_GLYPH = re.compile(r'xlink:href="#([A-Za-z]+)-([0-9a-f]+)"')

#: Below this many text elements, a figure is assumed to have been parsed wrongly
#: rather than to be sparse. Both committed figures carry 40+ (axis ticks alone
#: account for most of them), and a parser that silently matched nothing would
#: otherwise make every assertion below vacuously true.
_MIN_TEXT_ELEMENTS = 10

Glyphs = tuple[tuple[str, int], ...]


def _text_elements(svg: str) -> list[tuple[str, Glyphs]]:
    """Return each text element's source string paired with the glyphs drawn for it.

    Glyph indices are per-font and hex-encoded in the id, which is why the font
    name travels with each one: two fonts number their glyphs independently.

    Examples:
        ```pycon
        >>> svg = '<!-- hi -->\\n<g transform="x"><use xlink:href="#DejaVuSans-4b"/></g>'
        >>> _text_elements(svg)
        [('hi', (('DejaVuSans', 75),))]

        ```
    """
    return [
        (match.group(1), tuple((font, int(index, 16)) for font, index in _GLYPH.findall(match.group(2))))
        for match in _TEXT_GROUP.finditer(svg)
    ]


def _render(labels: list[str]) -> dict[str, Glyphs]:
    """Draw each label with matplotlib and return the glyphs it produced.

    One figure carries every label, so a whole SVG is checked against a single
    render. Glyph indices do not depend on position or size, and the reference
    goes to a string buffer, so nothing here touches the committed files.

    Examples:
        >>> glyphs = _render(["hi"])
        >>> list(glyphs)
        ['hi']
    """
    figure = plt.figure()
    for position, label in enumerate(labels):
        figure.text(0.0, position / max(len(labels), 1), label)
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg")
    plt.close(figure)
    return dict(_text_elements(buffer.getvalue()))


def test_figures_are_present() -> None:
    """The figures directory holds the SVGs the reproduction report links to."""
    report = (REPO_ROOT / "docs" / "REPRODUCTION_REPORT.md").read_text(encoding="utf-8")
    linked = {name for name in re.findall(r"!\[[^\]]*\]\(figures/([^)]+)\)", report)}
    assert linked, "the report links no figure; this gate would be vacuous"
    missing = sorted(name for name in linked if not (REPO_ROOT / "docs" / "figures" / name).is_file())
    assert not missing, f"report links missing figures: {missing}"


@pytest.mark.parametrize("figure", FIGURES, ids=lambda path: path.name)
def test_drawn_captions_match_their_source_strings(figure: Path) -> None:
    """Every label in a committed figure draws the string its own markup claims.

    Fails on a figure that was edited as text instead of regenerated, whatever
    the edit was for: a rename, a typo fix, a units correction.
    """
    elements = _text_elements(figure.read_text(encoding="utf-8"))
    assert len(elements) >= _MIN_TEXT_ELEMENTS, f"{figure.name}: parsed only {len(elements)} text elements"

    reference = _render([label for label, _ in elements])
    mismatched = [label for label, drawn in elements if reference.get(label) != drawn]
    assert not mismatched, f"{figure.name}: markup and drawing disagree for {mismatched}"
