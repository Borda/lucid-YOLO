# SPDX-License-Identifier: Apache-2.0
"""Meta tests: policy documents exist and parse (WP-003).

Guards the clean-room paper trail: PROVENANCE, ASSUMPTIONS, DECISIONS,
ESCALATION, AGENTS, and ROADMAP must exist; assumption ids must parse as a
contiguous register; roadmap WP ids must be unique and complete.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"

#: Lowest work-package count ROADMAP.md is allowed to hold. A ratchet, not a target:
#: contiguity alone would not notice the last row being deleted. Raise it when adding
#: a work package; never lower it.
_WP_FLOOR = 93

#: Lowest decision count DECISIONS.md is allowed to hold. A ratchet, not a target:
#: contiguity alone would not notice the last row being deleted, since what remains
#: stays contiguous. Raise it when adding a decision; never lower it.
_DECISION_FLOOR = 15

REQUIRED_FILES = (
    DOCS / "PROVENANCE.md",
    DOCS / "ASSUMPTIONS.md",
    DOCS / "DECISIONS.md",
    DOCS / "ESCALATION.md",
    DOCS / "ROADMAP.md",
    DOCS / "REPRODUCTION_REPORT.md",
    DOCS / "MODEL_CARD_DETECTION.md",
    DOCS / "MODEL_CARD_SEGMENTATION.md",
    REPO_ROOT / "AGENTS.md",
)


def test_policy_docs_exist() -> None:
    """Every policy document required by the execution contract is present."""
    missing = [str(path.relative_to(REPO_ROOT)) for path in REQUIRED_FILES if not path.is_file()]
    assert not missing, f"missing policy docs: {missing}"


def test_assumption_ids_parse_contiguous() -> None:
    """ASSUMPTIONS.md register rows carry ids A1..AN with no gap or duplicate."""
    text = (DOCS / "ASSUMPTIONS.md").read_text(encoding="utf-8")
    ids = [int(m) for m in re.findall(r"^\| A(\d+) \|", text, flags=re.MULTILINE)]
    assert ids, "no assumption rows found"
    assert len(ids) == len(set(ids)), "duplicate assumption ids"
    assert sorted(ids) == list(range(1, max(ids) + 1)), f"non-contiguous assumption ids: {sorted(ids)}"
    assert max(ids) >= 26, "register must carry at least A1-A26"


def _table_cells(line: str) -> list[str]:
    """Split one markdown table row into its cells, honoring backslash-escaped pipes."""
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


def test_assumption_rows_fill_every_column() -> None:
    """Every register row carries all six columns, with a recognized status in the last.

    A row written one cell short does not look broken: the formatter pads it back to six,
    and the id and contiguity gates above keep passing because neither reads past the
    first column. What actually happens is that every value shifts left -- the validation
    plan lands under "Public source", the status under "Validation" -- so the register
    reads as though a sourced assumption were unsourced. Two rows shipped that way before
    this gate existed (A25, A42).
    """
    rows = [
        _table_cells(line)
        for line in (DOCS / "ASSUMPTIONS.md").read_text(encoding="utf-8").splitlines()
        if re.match(r"^\| A\d+ \|", line)
    ]
    assert rows, "no assumption rows found"

    malformed = [(row[0], len(row)) for row in rows if len(row) != 6]
    assert not malformed, f"rows without exactly six columns: {malformed}"
    unstatused = [(row[0], row[5]) for row in rows if row[5] not in {"open", "active", "validated", "revised"}]
    assert not unstatused, f"rows whose last column is not a status: {unstatused}"
    unsourced = [row[0] for row in rows if not row[3]]
    assert not unsourced, f"rows with an empty public-source column: {unsourced}"


def test_roadmap_wp_ids_unique_and_complete() -> None:
    """ROADMAP.md numbers its work packages contiguously from 001, each exactly once, and never shrinks.

    Contiguity and uniqueness are the real invariants and they hold at any size;
    the count is pinned separately by a floor, so adding a work package does not
    fail a test that is not reporting a defect. Dropping the last row would leave
    the remainder contiguous, which is what the floor is for. Raise it when adding
    a package — see :data:`_WP_FLOOR`.
    """
    text = (DOCS / "ROADMAP.md").read_text(encoding="utf-8")
    ids = [int(m) for m in re.findall(r"^\| (\d{3}) \|", text, flags=re.MULTILINE)]
    assert len(ids) == len(set(ids)), "duplicate WP ids in roadmap"
    assert sorted(ids) == list(range(1, max(ids) + 1)), f"roadmap ids are not contiguous from 1: {sorted(ids)}"
    assert len(ids) >= _WP_FLOOR, f"roadmap shrank below {_WP_FLOOR} work packages: {len(ids)}"


def test_roadmap_rows_fill_every_column() -> None:
    """Every numbered roadmap row carries all six columns.

    An unescaped pipe inside a code span opens a table cell, so a row that reads fine in
    the source renders its tail into the wrong columns on GitHub and drops the overflow.
    This has now happened twice: once to a WP-058 row, and once to WP-093's own row, where
    ``|d_theta|`` in a code span was expanded by the formatter into cell delimiters and
    left the row with four cells instead of six.

    Neither existing gate could see it. :func:`test_roadmap_wp_ids_unique_and_complete`
    reads only the first column, and :func:`test_roadmap_statuses_valid` matches the last
    one by regex, so a row can lose its middle entirely with both of them green.
    """
    rows = [
        (line[:9], _table_cells(line))
        for line in (DOCS / "ROADMAP.md").read_text(encoding="utf-8").splitlines()
        if re.match(r"^\| \d{3}[a-z]? \|", line)
    ]
    assert rows, "no roadmap rows found"
    malformed = [(head, len(cells)) for head, cells in rows if len(cells) != 6]
    assert not malformed, f"roadmap rows without exactly six columns: {malformed}"


def test_roadmap_statuses_valid() -> None:
    """Every roadmap row ends in a recognized status icon."""
    text = (DOCS / "ROADMAP.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\d{3}[a-z]?) \|.*\| (\S+) \|$", text, flags=re.MULTILINE)
    assert len(rows) >= 91
    bad = [(wp, status) for wp, status in rows if status not in {"⬜", "🔄", "✅", "⛔", "⊘"}]
    assert not bad, f"invalid status values: {bad}"


def test_roadmap_header_states_the_actual_package_count() -> None:
    """The opening paragraph's package count matches the numbered rows it describes.

    That count is prose, so nothing forced it to move when rows were added: it
    read "69 work packages" while the table carried 91, and the two ids-and-status
    gates above both passed the whole time because neither of them reads the
    sentence. A documented number with no gate is a number that decays.
    """
    text = (DOCS / "ROADMAP.md").read_text(encoding="utf-8")
    numbered = len(re.findall(r"^\| \d{3} \|", text, flags=re.MULTILINE))

    stated = re.search(r"(\d+) numbered work packages", text)

    assert stated is not None, "the header must state the numbered work-package count"
    assert int(stated.group(1)) == numbered


def test_provenance_carries_allowlist_ids() -> None:
    """PROVENANCE.md defines every source id R1..R21 used in commit trailers."""
    text = (DOCS / "PROVENANCE.md").read_text(encoding="utf-8")
    ids = {int(m) for m in re.findall(r"^\| R(\d+) \|", text, flags=re.MULTILINE)}
    assert ids >= set(range(1, 22)), f"provenance missing source ids: {sorted(set(range(1, 22)) - ids)}"


def test_decisions_carry_all_ids() -> None:
    """DECISIONS.md numbers its decisions contiguously from D1, never shrinks, and keeps the four ADRs.

    Two properties, deliberately kept separate. Contiguity catches a duplicated or
    skipped id, and it holds however many decisions the register grows to — a
    hardcoded upper bound would fail every time one is added, which is a test
    demanding maintenance rather than reporting a defect. The floor is what a bare
    contiguity check would miss: dropping the *last* row leaves the remainder
    perfectly contiguous, so the count is asserted never to fall below what the
    register has already reached. Raise the floor when adding a decision; that edit
    is the deliberate act, not a chore.
    """
    text = (DOCS / "DECISIONS.md").read_text(encoding="utf-8")
    d_ids = {int(m) for m in re.findall(r"^\| D(\d+) \|", text, flags=re.MULTILINE)}
    assert d_ids == set(range(1, max(d_ids) + 1)), f"decision ids are not contiguous from 1: {sorted(d_ids)}"
    assert max(d_ids) >= _DECISION_FLOOR, f"decisions shrank below D{_DECISION_FLOOR}: {sorted(d_ids)}"
    for adr in ("ADR-001", "ADR-002", "ADR-003", "ADR-004"):
        assert f"## {adr}" in text, f"missing {adr} section"
