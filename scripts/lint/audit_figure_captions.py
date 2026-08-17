# SPDX-License-Identifier: Apache-2.0
"""Figure-caption audit: committed SVGs draw what their own markup claims (WP-080).

Matplotlib writes a text element twice. The characters become glyph outlines,
and the source string is repeated beside them as an XML comment that nothing
renders. A search-and-replace over a committed SVG therefore edits the comment
and leaves the drawing untouched, and the file then reports one caption and
displays another -- which is exactly what a repository-wide tier rename did to
``det_smoke_training.svg``, undetected, with every gate green: no checker read
a figure.

This re-renders one text element's own comment and compares its glyphs against
the glyphs already drawn in the file, so both sides come from the figure and
no caption is pinned in this script -- a legitimately retitled figure passes
as soon as it is regenerated. It also confirms every figure the reproduction
report links actually exists on disk.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_figure_captions.py
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURES_DIR = REPO_ROOT / "docs" / "figures"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "REPRODUCTION_REPORT.md"

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


def check_figures_present(repo_root: Path) -> list[str]:
    """Every figure ``docs/REPRODUCTION_REPORT.md`` links must exist under ``docs/figures/``.

    Args:
        repo_root: Repository root; the report and figures directory are found relative to it.

    Returns:
        One message per missing figure; empty when every linked figure is present.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     (root / "docs" / "figures").mkdir(parents=True)
        ...     _ = (root / "docs" / "figures" / "present.svg").write_text("<svg/>")
        ...     _ = (root / "docs" / "REPRODUCTION_REPORT.md").write_text(
        ...         "![a](figures/present.svg)\\n![b](figures/missing.svg)\\n"
        ...     )
        ...     check_figures_present(root)
        ['report links missing figures: figures/missing.svg']
    """
    report = (repo_root / "docs" / "REPRODUCTION_REPORT.md").read_text(encoding="utf-8")
    linked = {name for name in re.findall(r"!\[[^\]]*\]\(figures/([^)]+)\)", report)}
    if not linked:
        return ["the report links no figure; this gate would be vacuous"]
    missing = sorted(name for name in linked if not (repo_root / "docs" / "figures" / name).is_file())
    if not missing:
        return []
    return [f"report links missing figures: {', '.join(f'figures/{name}' for name in missing)}"]


def check_captions_match(repo_root: Path) -> list[str]:
    """Every label in a committed figure must draw the string its own markup claims.

    Fails on a figure that was edited as text instead of regenerated, whatever
    the edit was for: a rename, a typo fix, a units correction.

    Args:
        repo_root: Repository root; figures are found under ``docs/figures/*.svg`` within it.

    Returns:
        One message per mismatched figure; empty when every committed figure is clean.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     figures = root / "docs" / "figures"
        ...     figures.mkdir(parents=True)
        ...     glyphs = "".join(
        ...         f'<!-- label {i} -->\\n<g transform="x"><use xlink:href="#DejaVuSans-4b"/></g>\\n'
        ...         for i in range(_MIN_TEXT_ELEMENTS)
        ...     )
        ...     _ = (figures / "bad.svg").write_text(glyphs)
        ...     violations = check_captions_match(root)
        ...     len(violations)
        1
    """
    violations = []
    for figure in sorted((repo_root / "docs" / "figures").glob("*.svg")):
        elements = _text_elements(figure.read_text(encoding="utf-8"))
        if len(elements) < _MIN_TEXT_ELEMENTS:
            violations.append(f"{figure.name}: parsed only {len(elements)} text elements")
            continue
        reference = _render([label for label, _ in elements])
        mismatched = [label for label, drawn in elements if reference.get(label) != drawn]
        if mismatched:
            violations.append(f"{figure.name}: markup and drawing disagree for {mismatched}")
    return violations


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run both checks, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` when any violation is found.
    """
    parser = argparse.ArgumentParser(description="Audit docs/figures/*.svg captions against the reproduction report.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=f"repository root to scan (default: {REPO_ROOT})",
    )
    args = parser.parse_args(argv)

    violations = check_figures_present(args.repo_root) + check_captions_match(args.repo_root)
    if violations:
        print(f"figure-caption-audit FAILED: {len(violations)} violation(s)")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("figure-caption-audit clean: every figure is present and every caption matches its drawing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
