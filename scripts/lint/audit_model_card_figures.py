# SPDX-License-Identifier: Apache-2.0
"""Model-card audit: the wiring-gate figure a card prints is the one in the golden it cites.

Each ``docs/model_cards/*.md`` states, on one line, what ``scripts/overfit_micro.py`` measures for its task: a
bold ``**value ± tolerance**`` pair and the golden that records it, written ``goldens/gpu/overfit_micro_<task>.json``.
The figures were copied by hand, and the goldens were resealed under them (WP-186) without the cards following,
so three cards disagreed with their goldens while every gate stayed green.

The anchor is that one line -- the one naming a ``goldens/gpu/overfit_micro_*.json`` file -- and only its bold
pair is compared. A historical figure ("first stood at 0.3357 in WP-132") is plain text beside a work-package
link, so it is exempt by construction: it describes the past and is allowed to differ from the present golden. A card
prints its figure to the precision it chooses, so the golden is rounded to the card's own decimals before comparing.

Run ``python scripts/lint/audit_model_card_figures.py`` (exit ``0`` clean, ``1`` on any disagreement). ``--repo-root``
points it at another tree; the tests use throwaway ones.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `scripts.lint` resolves under the hook
from scripts.lint._common import add_repo_root_argument, print_failure, sorted_rglob

#: Where the cards live, relative to the repository root.
CARDS_DIR = Path("docs") / "model_cards"
#: The golden a card cites, as a backticked repo-relative path.
_GOLDEN = re.compile(r"`(?P<path>goldens/gpu/overfit_micro_\w+\.json)`")
#: The bold ``value ± tolerance`` pair a card prints for the wiring gate.
_FIGURE = re.compile(r"\*\*(?P<value>\d+\.\d+) ± (?P<tolerance>\d+\.\d+)\*\*")


def _decimals(text: str) -> int:
    """How many places a printed number carries.

    Examples:
        >>> _decimals("0.4919"), _decimals("0.05")
        (4, 2)
    """
    return len(text.split(".")[1])


def _disagreement(label: str, printed: str, recorded: float) -> str | None:
    """``None`` when ``printed`` is ``recorded`` rounded to the card's own precision, else why not.

    Examples:
        >>> _disagreement("figure", "0.9966", 0.996622) is None
        True
        >>> _disagreement("figure", "0.3357", 0.491935)
        "figure 0.3357 but the golden records 0.491935 (0.4919 at the card's 4 places)"
    """
    places = _decimals(printed)
    expected = f"{recorded:.{places}f}"
    if printed == expected:
        return None
    return f"{label} {printed} but the golden records {recorded} ({expected} at the card's {places} places)"


def check_line(line: str, repo_root: Path) -> list[str]:
    """Disagreements between the bold figure on one card line and the golden that line cites.

    Args:
        line: One line of a model card that names a ``goldens/gpu/overfit_micro_*.json``.
        repo_root: Repository root the cited path is resolved against.

    Returns:
        One message per disagreement, or per structural problem (no single bold pair, an unreadable golden, a
        golden without exactly one tolerance).

    Examples:
        >>> import json, tempfile
        >>> line = "It measures recall **0.9966 ± 0.04** (`goldens/gpu/overfit_micro_det.json`)."
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     golden = Path(tmp) / "goldens" / "gpu" / "overfit_micro_det.json"
        ...     golden.parent.mkdir(parents=True)
        ...     _ = golden.write_text(
        ...         json.dumps({"values": {"recall": 0.996622}, "tolerances": {"recall": 0.04}}), encoding="utf-8"
        ...     )
        ...     check_line(line, Path(tmp))
        []
    """
    cited = _GOLDEN.search(line)
    assert cited is not None, "caller passes only lines that cite a golden"
    figures = list(_FIGURE.finditer(line))
    if len(figures) != 1:
        return [f"{cited['path']}: expected one bold '**value ± tolerance**' on the citing line, found {len(figures)}"]
    path = repo_root / cited["path"]
    try:
        golden = json.loads(path.read_text(encoding="utf-8"))
        ((metric, tolerance),) = golden["tolerances"].items()
        value = golden["values"][metric]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"{cited['path']}: cannot read one metric and its tolerance from the golden ({exc!r})"]
    figure = figures[0]
    problems = (
        _disagreement(f"{metric} value", figure["value"], value),
        _disagreement(f"{metric} tolerance", figure["tolerance"], tolerance),
    )
    return [f"{cited['path']}: {problem}" for problem in problems if problem]


def check_card(card: Path, repo_root: Path) -> list[str]:
    """Every wiring-gate disagreement on one card, each prefixed with ``<card>:<line>``.

    A card that cites no ``goldens/gpu/overfit_micro_*.json`` at all is itself a finding: the figure it should carry
    has nothing to be checked against.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     card = Path(tmp) / "bare.md"
        ...     _ = card.write_text("No gate here.\\n", encoding="utf-8")
        ...     [message.split(": ", 1)[1] for message in check_card(card, Path(tmp))]
        ['cites no goldens/gpu/overfit_micro_*.json, so its wiring-gate figure is unchecked']
    """
    shown = card.relative_to(repo_root).as_posix()
    cited_lines = [
        (number, line)
        for number, line in enumerate(card.read_text(encoding="utf-8").splitlines(), start=1)
        if _GOLDEN.search(line)
    ]
    if not cited_lines:
        return [f"{shown}: cites no goldens/gpu/overfit_micro_*.json, so its wiring-gate figure is unchecked"]
    return [f"{shown}:{number}: {problem}" for number, line in cited_lines for problem in check_line(line, repo_root)]


def find_violations(repo_root: Path) -> list[str]:
    """Every card disagreement under ``repo_root``, cards in sorted order.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     find_violations(Path(tmp))
        []
    """
    return [problem for card in sorted_rglob(repo_root / CARDS_DIR, "*.md") for problem in check_card(card, repo_root)]


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the audit, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` when any card disagrees with its golden.
    """
    parser = argparse.ArgumentParser(description="Audit model-card wiring-gate figures against the goldens they cite.")
    add_repo_root_argument(parser, "repository whose cards and goldens to read")
    args = parser.parse_args(argv)

    violations = find_violations(args.repo_root)
    if violations:
        return print_failure("model-card-figures-audit", violations)
    print("model-card-figures-audit clean: every card's wiring-gate figure matches its golden")
    return 0


if __name__ == "__main__":
    sys.exit(main())
