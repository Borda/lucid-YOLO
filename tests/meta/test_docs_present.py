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

REQUIRED_FILES = (
    DOCS / "PROVENANCE.md",
    DOCS / "ASSUMPTIONS.md",
    DOCS / "DECISIONS.md",
    DOCS / "ESCALATION.md",
    DOCS / "ROADMAP.md",
    DOCS / "REPRODUCTION_REPORT.md",
    DOCS / "MODEL_CARD_DETECTION.md",
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


def test_roadmap_wp_ids_unique_and_complete() -> None:
    """ROADMAP.md rows carry WP ids 001..083, each exactly once (068-083 added 2026-08-02..06)."""
    text = (DOCS / "ROADMAP.md").read_text(encoding="utf-8")
    ids = [int(m) for m in re.findall(r"^\| (\d{3}) \|", text, flags=re.MULTILINE)]
    assert len(ids) == len(set(ids)), "duplicate WP ids in roadmap"
    assert sorted(ids) == list(range(1, 84)), f"roadmap must list WP 001-083, got {len(ids)} rows"


def test_roadmap_statuses_valid() -> None:
    """Every roadmap row ends in a recognized status icon."""
    text = (DOCS / "ROADMAP.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\d{3}) \|.*\| (\S+) \|$", text, flags=re.MULTILINE)
    assert len(rows) == 83
    bad = [(wp, status) for wp, status in rows if status not in {"⬜", "🔄", "✅", "⛔"}]
    assert not bad, f"invalid status values: {bad}"


def test_provenance_carries_allowlist_ids() -> None:
    """PROVENANCE.md defines every source id R1..R21 used in commit trailers."""
    text = (DOCS / "PROVENANCE.md").read_text(encoding="utf-8")
    ids = {int(m) for m in re.findall(r"^\| R(\d+) \|", text, flags=re.MULTILINE)}
    assert ids >= set(range(1, 22)), f"provenance missing source ids: {sorted(set(range(1, 22)) - ids)}"


def test_decisions_carry_all_ids() -> None:
    """DECISIONS.md lists D1-D13 and the four ADRs."""
    text = (DOCS / "DECISIONS.md").read_text(encoding="utf-8")
    d_ids = {int(m) for m in re.findall(r"^\| D(\d+) \|", text, flags=re.MULTILINE)}
    assert d_ids == set(range(1, 14)), f"decision ids: {sorted(d_ids)}"
    for adr in ("ADR-001", "ADR-002", "ADR-003", "ADR-004"):
        assert f"## {adr}" in text, f"missing {adr} section"
