# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: docs-governance-register audit script (WP-003, split off tests/meta).

Covers ``scripts/lint/audit_docs_present.py``'s public contract in isolation, on
synthetic files under ``tmp_path`` rather than against the live ``docs/`` tree -- the
live tree is exercised instead by the pre-commit hook itself each time a docs file
changes. This file is the functional-core check pytest owns; the hook is the lint
gate that runs it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "scripts" / "lint" / "audit_docs_present.py"


def _load_audit() -> ModuleType:
    """Load ``scripts/lint/audit_docs_present.py`` as an importable module.

    Examples:
        >>> module = _load_audit()
        >>> module.__name__
        'audit_docs_present'
        >>> callable(module.find_violations)
        True
    """
    spec = importlib.util.spec_from_file_location("audit_docs_present", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def _write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` as UTF-8, creating or overwriting it.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "f.md"
        ...     _write(sample, "hi\\n")
        ...     sample.read_text(encoding="utf-8")
        'hi\\n'
    """
    path.write_text(text, encoding="utf-8")


def _write_required_docs(docs_dir: Path, repo_root: Path) -> None:
    """Write a minimal, fully-valid set of every required governance document.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _write_required_docs(root / "docs", root)
        ...     (root / "docs" / "PROVENANCE.md").is_file(), (root / "AGENTS.md").is_file()
        (True, True)
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "model_cards").mkdir(exist_ok=True)
    _write(docs_dir / "PROVENANCE.md", "".join(f"| R{i} | source {i} |\n" for i in range(1, 22)))
    _write(docs_dir / "ASSUMPTIONS.md", "".join(f"| A{i} | t | v | src | p | open |\n" for i in range(1, 27)))
    _write(
        docs_dir / "DECISIONS.md",
        "".join(f"| D{i} | t |\n" for i in range(1, 20))
        + "".join(
            f"## Decision D{i} — {adr}\n"
            for i, adr in enumerate(("ADR-001", "ADR-002", "ADR-003", "ADR-004", "ADR-005"), 1)
        ),
    )
    _write(docs_dir / "ESCALATION.md", "escalation\n")
    _write(
        docs_dir / "ROADMAP.md",
        "160 numbered work packages.\n" + "".join(f"| {i:03d} | a | b | c | d | ✅ |\n" for i in range(1, 161)),
    )
    _write(docs_dir / "DATASETS.md", "datasets\n")
    _write(docs_dir / "TRAINING.md", "training\n")
    _write(
        docs_dir / "REPRODUCTION_REPORT.md",
        "## 0.1.0 — Detection\n## 0.2.0 — Instance segmentation\n"
        "## 0.3.0 — Oriented detection\n## Consolidated note — detection, segmentation, oriented detection\n"
        "## 0.5.0 — Keypoint detection\n",
    )
    _write(docs_dir / "RESEARCH_LOG.md", "no anchors here\n")
    _write(docs_dir / "ENGINEERING_LOG.md", "no anchors here\n")
    _write(docs_dir / "model_cards" / "detection.md", "detection\n")
    _write(docs_dir / "model_cards" / "segmentation.md", "segmentation\n")
    _write(docs_dir / "model_cards" / "obb.md", "obb\n")
    _write(docs_dir / "model_cards" / "keypoints.md", "keypoints\n")
    _write(repo_root / "AGENTS.md", "agents\n")


class TestCheckPolicyDocsExist:
    """Tests for ``audit.check_policy_docs_exist``."""

    def test_flags_a_missing_file(self, tmp_path: Path) -> None:
        """A required document absent from disk is reported by name."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        (tmp_path / "docs" / "DATASETS.md").unlink()

        violations = audit.check_policy_docs_exist(tmp_path / "docs", tmp_path)

        assert any("docs/DATASETS.md" in violation for violation in violations)

    def test_is_clean_when_every_file_present(self, tmp_path: Path) -> None:
        """Every required document present on disk reports no violation."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        assert audit.check_policy_docs_exist(tmp_path / "docs", tmp_path) == []


class TestCheckModelCardsAreRequired:
    """Tests for ``audit.check_model_cards_are_required``."""

    def test_flags_an_unlisted_card(self, tmp_path: Path) -> None:
        """A card in ``model_cards/`` that isn't a required file is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        _write(tmp_path / "docs" / "model_cards" / "extra.md", "extra\n")

        violations = audit.check_model_cards_are_required(tmp_path / "docs", tmp_path)

        assert violations == ["model cards not in REQUIRED_FILES: ['extra.md']"]

    def test_is_clean_for_the_exact_set(self, tmp_path: Path) -> None:
        """The exact required set of model cards, with nothing extra, reports no violation."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        assert audit.check_model_cards_are_required(tmp_path / "docs", tmp_path) == []


class TestCheckLogLinksResolve:
    """Tests for ``audit.check_log_links_resolve``."""

    def test_flags_an_undefined_anchor(self, tmp_path: Path) -> None:
        """A roadmap link to a log anchor the log file never defines is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        _write(tmp_path / "docs" / "ROADMAP.md", "See RESEARCH_LOG.md#missing-anchor for detail.\n")

        violations = audit.check_log_links_resolve(tmp_path / "docs", tmp_path)

        assert any("missing-anchor" in violation for violation in violations)

    def test_is_clean_when_every_anchor_is_defined(self, tmp_path: Path) -> None:
        """A roadmap link to a defined anchor reports no violation."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        _write(
            tmp_path / "docs" / "ROADMAP.md",
            "160 numbered work packages.\nSee RESEARCH_LOG.md#the-anchor.\n"
            + "".join(f"| {i:03d} | a | b | c | d | ✅ |\n" for i in range(1, 160)),
        )
        _write(tmp_path / "docs" / "RESEARCH_LOG.md", '<a id="the-anchor">note</a>\n')

        assert audit.check_log_links_resolve(tmp_path / "docs", tmp_path) == []


class TestCheckAssumptionIdsContiguous:
    """Tests for ``audit.check_assumption_ids_contiguous``."""

    def test_flags_a_gap(self, tmp_path: Path) -> None:
        """A gap in the A1..AN sequence is reported with the actual ids found."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        _write(tmp_path / "docs" / "ASSUMPTIONS.md", "| A1 | ... |\n| A3 | ... |\n")

        violations = audit.check_assumption_ids_contiguous(tmp_path / "docs", tmp_path)

        assert "non-contiguous assumption ids: [1, 3]" in violations

    def test_is_clean_for_a_full_register(self, tmp_path: Path) -> None:
        """A contiguous register at or above the floor reports no violation."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        assert audit.check_assumption_ids_contiguous(tmp_path / "docs", tmp_path) == []


class TestCheckAssumptionRowsComplete:
    """Tests for ``audit.check_assumption_rows_complete``."""

    def test_flags_a_short_row(self, tmp_path: Path) -> None:
        """A row with fewer than six columns is reported without crashing on the other checks."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        _write(tmp_path / "docs" / "ASSUMPTIONS.md", "| A1 | short |\n")

        violations = audit.check_assumption_rows_complete(tmp_path / "docs", tmp_path)

        assert violations == ["rows without exactly six columns: [('A1', 2)]"]

    def test_is_clean_for_full_rows(self, tmp_path: Path) -> None:
        """Six-column rows with a valid status and a non-empty source report no violation."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        assert audit.check_assumption_rows_complete(tmp_path / "docs", tmp_path) == []


class TestCheckRoadmapWpIdsUniqueAndComplete:
    """Tests for ``audit.check_roadmap_wp_ids_unique_and_complete``."""

    def test_flags_the_floor(self, tmp_path: Path) -> None:
        """A roadmap below the work-package floor is reported even when contiguous."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        _write(tmp_path / "docs" / "ROADMAP.md", "1 numbered work packages.\n| 001 | ... |\n")

        violations = audit.check_roadmap_wp_ids_unique_and_complete(tmp_path / "docs", tmp_path)

        assert any("shrank below 160" in violation for violation in violations)

    def test_is_clean_at_the_floor(self, tmp_path: Path) -> None:
        """A contiguous roadmap at the floor count reports no violation."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        assert audit.check_roadmap_wp_ids_unique_and_complete(tmp_path / "docs", tmp_path) == []


class TestCheckRoadmapRowsComplete:
    """Tests for ``audit.check_roadmap_rows_complete``."""

    def test_flags_a_short_row(self, tmp_path: Path) -> None:
        """A numbered roadmap row with fewer than six columns is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        _write(tmp_path / "docs" / "ROADMAP.md", "| 001 | x |\n")

        violations = audit.check_roadmap_rows_complete(tmp_path / "docs", tmp_path)

        assert violations == ["roadmap rows without exactly six columns: [('| 001 | x', 2)]"]

    def test_is_clean_for_full_rows(self, tmp_path: Path) -> None:
        """Six-column numbered roadmap rows report no violation."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        assert audit.check_roadmap_rows_complete(tmp_path / "docs", tmp_path) == []


class TestCheckRoadmapStatusesValid:
    """Tests for ``audit.check_roadmap_statuses_valid``."""

    def test_flags_an_unrecognized_icon(self, tmp_path: Path) -> None:
        """A status icon outside the recognized set is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        text = (tmp_path / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
        _write(
            tmp_path / "docs" / "ROADMAP.md",
            text.replace("| 001 | a | b | c | d | ✅ |", "| 001 | a | b | c | d | ??? |"),
        )

        violations = audit.check_roadmap_statuses_valid(tmp_path / "docs", tmp_path)

        assert any("invalid status values" in violation for violation in violations)

    def test_is_clean_for_recognized_icons(self, tmp_path: Path) -> None:
        """Every row ending in a recognized status icon, at or above the row floor, is clean."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        assert audit.check_roadmap_statuses_valid(tmp_path / "docs", tmp_path) == []


class TestCheckRoadmapHeaderCount:
    """Tests for ``audit.check_roadmap_header_count``."""

    def test_flags_a_mismatched_count(self, tmp_path: Path) -> None:
        """A stated package count that disagrees with the numbered rows is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        text = (tmp_path / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
        _write(
            tmp_path / "docs" / "ROADMAP.md", text.replace("160 numbered work packages.", "5 numbered work packages.")
        )

        violations = audit.check_roadmap_header_count(tmp_path / "docs", tmp_path)

        assert violations == ["header states 5 numbered work packages but the table carries 160"]

    def test_is_clean_when_counts_agree(self, tmp_path: Path) -> None:
        """A stated package count matching the numbered rows reports no violation."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        assert audit.check_roadmap_header_count(tmp_path / "docs", tmp_path) == []


class TestCheckProvenanceIds:
    """Tests for ``audit.check_provenance_ids``."""

    def test_flags_a_missing_source_id(self, tmp_path: Path) -> None:
        """A missing source id in the R1..R21 allowlist is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        _write(tmp_path / "docs" / "PROVENANCE.md", "| R1 | source |\n")

        violations = audit.check_provenance_ids(tmp_path / "docs", tmp_path)

        assert any("provenance missing source ids" in violation for violation in violations)

    def test_is_clean_for_the_full_allowlist(self, tmp_path: Path) -> None:
        """Every source id R1..R21 present reports no violation."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        assert audit.check_provenance_ids(tmp_path / "docs", tmp_path) == []


class TestCheckReportSections:
    """Tests for ``audit.check_report_sections``."""

    def test_flags_a_missing_heading(self, tmp_path: Path) -> None:
        """A required report section heading missing from the file is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        _write(tmp_path / "docs" / "REPRODUCTION_REPORT.md", "## 0.1.0 — Detection\n")

        violations = audit.check_report_sections(tmp_path / "docs", tmp_path)

        assert any("missing section headings" in violation for violation in violations)

    def test_is_clean_when_every_section_present(self, tmp_path: Path) -> None:
        """Every required section heading present reports no violation."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        assert audit.check_report_sections(tmp_path / "docs", tmp_path) == []


class TestCheckDecisionsIds:
    """Tests for ``audit.check_decisions_ids``."""

    def test_flags_a_missing_adr(self, tmp_path: Path) -> None:
        """A missing ADR section heading is reported even when ids are contiguous."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        _write(tmp_path / "docs" / "DECISIONS.md", "".join(f"| D{i} | t |\n" for i in range(1, 20)))

        violations = audit.check_decisions_ids(tmp_path / "docs", tmp_path)

        assert any("missing ADR-001 section" in violation for violation in violations)

    def test_is_clean_for_a_full_register(self, tmp_path: Path) -> None:
        """A contiguous register at the floor with every ADR section reports no violation."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        assert audit.check_decisions_ids(tmp_path / "docs", tmp_path) == []


def test_find_violations_survives_a_missing_file(tmp_path: Path) -> None:
    """A check whose source file is entirely absent reports a violation instead of crashing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    violations = audit.find_violations(docs_dir, tmp_path)

    assert any(violation.startswith("missing policy docs") for violation in violations)


def test_the_live_docs_tree_is_currently_clean() -> None:
    """The real ``docs/`` tree the pre-commit hook scans has no violations.

    This is the one place this file touches the live tree rather than a synthetic
    one: a regression here means the hook itself would fail on the next commit
    that touches any docs file, not just a new one.
    """
    assert audit.find_violations(REPO_ROOT / "docs", REPO_ROOT) == []
