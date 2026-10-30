# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: docs-governance-register audit script (WP-003, split off tests/meta).

Covers ``scripts/lint/audit_docs_present.py``'s public contract in isolation, on synthetic files under ``tmp_path``
rather than against the live ``docs/`` tree -- the live tree is exercised instead by the pre-commit hook itself each
time a docs file changes. This file is the functional-core check pytest owns; the hook is the lint gate that runs it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from audit_tree_helpers import replace_in_file, write_file
from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]


audit = load_script("audit_docs_present")


def _roadmap_rows(count: int) -> str:
    """Return numbered roadmap table rows ``001`` through ``count``, each in a recognised status.

    Examples:
        >>> _roadmap_rows(2).splitlines()
        ['| 001 | a | b | c | d | ✅ |', '| 002 | a | b | c | d | ✅ |']
    """
    return "".join(f"| {i:03d} | a | b | c | d | ✅ |\n" for i in range(1, count + 1))


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
    write_file(docs_dir / "PROVENANCE.md", "".join(f"| R{i} | source {i} |\n" for i in range(1, 22)))
    write_file(
        docs_dir / "ASSUMPTIONS.md",
        "".join(f"| A{i} | t | v | src | p | open |\n" for i in range(1, audit._ASSUMPTION_FLOOR + 1)),
    )
    write_file(
        docs_dir / "DECISIONS.md",
        "".join(f"| D{i} | t |\n" for i in range(1, 30))
        + "".join(
            f"## Decision D{i} — {adr}\n"
            for i, adr in enumerate(("ADR-001", "ADR-002", "ADR-003", "ADR-004", "ADR-005"), 1)
        ),
    )
    write_file(docs_dir / "ESCALATION.md", "escalation\n")
    write_file(
        docs_dir / "ROADMAP.md",
        f"{audit._WP_FLOOR} numbered work packages.\n" + _roadmap_rows(audit._WP_FLOOR),
    )
    write_file(docs_dir / "DATASETS.md", "datasets\n")
    write_file(docs_dir / "TRAINING.md", "training\n")
    write_file(
        docs_dir / "REPRODUCTION_REPORT.md",
        "## 0.1.0 — Detection\n## 0.2.0 — Instance segmentation\n"
        "## 0.3.0 — Oriented detection\n## Consolidated note — detection, segmentation, oriented detection\n"
        "## 0.5.0 — Keypoint detection\n",
    )
    write_file(docs_dir / "RESEARCH_LOG.md", "no anchors here\n")
    write_file(docs_dir / "ENGINEERING_LOG.md", "no anchors here\n")
    write_file(docs_dir / "model_cards" / "detection.md", "detection\n")
    write_file(docs_dir / "model_cards" / "segmentation.md", "segmentation\n")
    write_file(docs_dir / "model_cards" / "obb.md", "obb\n")
    write_file(docs_dir / "model_cards" / "keypoints.md", "keypoints\n")
    write_file(repo_root / "AGENTS.md", "agents\n")


@pytest.mark.parametrize(
    "check_name",
    [
        "check_policy_docs_exist",
        "check_model_cards_are_required",
        "check_assumption_ids_contiguous",
        "check_assumption_rows_complete",
        "check_roadmap_wp_ids_unique_and_complete",
        "check_roadmap_rows_complete",
        "check_roadmap_statuses_valid",
        "check_roadmap_header_count",
        "check_provenance_ids",
        "check_report_sections",
        "check_decisions_ids",
    ],
)
def test_every_check_clean_on_valid_docs_set(tmp_path: Path, check_name: str) -> None:
    """A minimal, fully valid set of governance documents reports no violation from any single check.

    The valid set holds every required document and exactly the model cards required, a contiguous assumption register
    and roadmap at their floors with six-column rows and recognised statuses, a header count matching the table, every
    provenance id R1..R21, every report section heading, and a contiguous decision register with its ADR sections.
    """
    _write_required_docs(tmp_path / "docs", tmp_path)

    assert getattr(audit, check_name)(tmp_path / "docs", tmp_path) == []


class TestCheckPolicyDocsExist:
    """Tests for ``audit.check_policy_docs_exist``."""

    def test_flags_missing_file(self, tmp_path: Path) -> None:
        """A required document absent from disk is reported by name."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        (tmp_path / "docs" / "DATASETS.md").unlink()

        violations = audit.check_policy_docs_exist(tmp_path / "docs", tmp_path)

        assert any("docs/DATASETS.md" in violation for violation in violations)


class TestCheckModelCardsAreRequired:
    """Tests for ``audit.check_model_cards_are_required``."""

    def test_flags_unlisted_card(self, tmp_path: Path) -> None:
        """A card in ``model_cards/`` that isn't a required file is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        write_file(tmp_path / "docs" / "model_cards" / "extra.md", "extra\n")

        violations = audit.check_model_cards_are_required(tmp_path / "docs", tmp_path)

        assert violations == ["model cards not in REQUIRED_FILES: ['extra.md']"]


class TestCheckLogLinksResolve:
    """Tests for ``audit.check_log_links_resolve``."""

    @pytest.mark.parametrize(
        ("roadmap", "anchor"),
        [
            pytest.param("See RESEARCH_LOG.md#missing-anchor for detail.\n", "missing-anchor", id="landing-file"),
            pytest.param("See engineering/phase-01.md#ghost.\n", "ghost", id="split-page"),
        ],
    )
    def test_flags_undefined_anchor(self, tmp_path: Path, roadmap: str, anchor: str) -> None:
        """A roadmap link to an anchor the log never defines is reported, on the landing file and on a split page.

        A link naming a split page but an anchor no page defines is still reported.
        """
        _write_required_docs(tmp_path / "docs", tmp_path)
        write_file(tmp_path / "docs" / "ROADMAP.md", roadmap)

        violations = audit.check_log_links_resolve(tmp_path / "docs", tmp_path)

        assert any(anchor in violation for violation in violations)

    @pytest.mark.parametrize("page", ["RESEARCH_LOG.md", "research/phase-01.md"])
    def test_clean_when_anchor_defined(self, tmp_path: Path, page: str) -> None:
        """A roadmap link to a defined anchor reports no violation, in the landing file or in ``research/<page>.md``.

        The research log is one page per phase, so its anchors are no longer in the landing file; a check reading only
        the landing file would flag every row as undefined.
        """
        _write_required_docs(tmp_path / "docs", tmp_path)
        write_file(
            tmp_path / "docs" / "ROADMAP.md",
            f"160 numbered work packages.\nSee {page}#the-anchor.\n" + _roadmap_rows(159),
        )
        write_file(tmp_path / "docs" / page, '<a id="the-anchor">note</a>\n')

        assert audit.check_log_links_resolve(tmp_path / "docs", tmp_path) == []


class TestCheckWorkPackageFiles:
    """Tests for ``audit.check_work_package_files``."""

    def test_clean_when_every_file_numbered_and_included(self, tmp_path: Path) -> None:
        """A WP file whose id is a roadmap row and which a phase page includes reports nothing."""
        docs = tmp_path / "docs"
        _write_required_docs(docs, tmp_path)
        (docs / "work_packages").mkdir()
        write_file(docs / "work_packages" / "wp-001.md", "x\n")
        (docs / "engineering").mkdir()
        write_file(docs / "engineering" / "phase-00.md", '--8<-- "work_packages/wp-001.md:engineering"\n')

        assert audit.check_work_package_files(docs, tmp_path) == []

    def test_flags_file_with_no_roadmap_row(self, tmp_path: Path) -> None:
        """A WP file the roadmap never numbered is reported, even when a phase page includes it.

        A typo in the file name or a renumbered row leaves findings filed under an id no row carries, and the roadmap's
        own checks cannot see a file they never read.
        """
        docs = tmp_path / "docs"
        _write_required_docs(docs, tmp_path)
        (docs / "work_packages").mkdir()
        write_file(docs / "work_packages" / "wp-999z.md", "x\n")
        (docs / "research").mkdir()
        write_file(docs / "research" / "phase-01.md", '--8<-- "work_packages/wp-999z.md:research"\n')

        assert audit.check_work_package_files(docs, tmp_path) == [
            "work-package files with no roadmap row: ['wp-999z.md']"
        ]

    def test_flags_file_no_phase_page_includes(self, tmp_path: Path) -> None:
        """A WP file no research or engineering phase page includes is reported as an orphan.

        The site would build and serve it, while the log a reader browses lacks the entry.
        """
        docs = tmp_path / "docs"
        _write_required_docs(docs, tmp_path)
        (docs / "work_packages").mkdir()
        write_file(docs / "work_packages" / "wp-001.md", "x\n")

        assert audit.check_work_package_files(docs, tmp_path) == [
            "work-package files no phase page includes: ['wp-001.md']"
        ]


class TestCheckAssumptionIdsContiguous:
    """Tests for ``audit.check_assumption_ids_contiguous``."""

    def test_flags_gap(self, tmp_path: Path) -> None:
        """A gap in the A1..AN sequence is reported with the actual ids found."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        write_file(tmp_path / "docs" / "ASSUMPTIONS.md", "| A1 | ... |\n| A3 | ... |\n")

        violations = audit.check_assumption_ids_contiguous(tmp_path / "docs", tmp_path)

        assert "non-contiguous assumption ids: [1, 3]" in violations

    def test_flags_row_deleted_from_tail(self, tmp_path: Path) -> None:
        """Deleting the last row leaves the remainder contiguous, and the row-count floor is what sees it."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        floor = audit._ASSUMPTION_FLOOR
        replace_in_file(tmp_path / "docs" / "ASSUMPTIONS.md", f"| A{floor} | t | v | src | p | open |\n", "")

        violations = audit.check_assumption_ids_contiguous(tmp_path / "docs", tmp_path)

        assert violations == [f"assumptions shrank below {floor} rows: {floor - 1}"]

    def test_flags_row_deleted_from_middle(self, tmp_path: Path) -> None:
        """Deleting A40 is reported as a shrink, not only as a gap.

        The ``max(ids)`` predicate this replaced could not see it: the last row still stood, so the floor stayed silent
        and the only finding was an id list a reader had to scan for the hole. The count says what actually happened.
        """
        _write_required_docs(tmp_path / "docs", tmp_path)
        replace_in_file(tmp_path / "docs" / "ASSUMPTIONS.md", "| A40 | t | v | src | p | open |\n", "")

        violations = audit.check_assumption_ids_contiguous(tmp_path / "docs", tmp_path)

        floor = audit._ASSUMPTION_FLOOR
        assert f"assumptions shrank below {floor} rows: {floor - 1}" in violations


class TestCheckAssumptionRowsComplete:
    """Tests for ``audit.check_assumption_rows_complete``."""

    def test_flags_short_row(self, tmp_path: Path) -> None:
        """A row with fewer than six columns is reported without crashing on the other checks."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        write_file(tmp_path / "docs" / "ASSUMPTIONS.md", "| A1 | short |\n")

        violations = audit.check_assumption_rows_complete(tmp_path / "docs", tmp_path)

        assert violations == ["rows without exactly six columns: [('A1', 2)]"]


class TestCheckRoadmapWpIdsUniqueAndComplete:
    """Tests for ``audit.check_roadmap_wp_ids_unique_and_complete``."""

    def test_flags_floor(self, tmp_path: Path) -> None:
        """A roadmap below the work-package floor is reported even when contiguous."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        floor = audit._WP_FLOOR
        write_file(
            tmp_path / "docs" / "ROADMAP.md", f"{floor - 1} numbered work packages.\n" + _roadmap_rows(floor - 1)
        )

        violations = audit.check_roadmap_wp_ids_unique_and_complete(tmp_path / "docs", tmp_path)

        assert violations == [f"roadmap shrank below {floor} work packages: {floor - 1}"]


class TestCheckRoadmapRowsComplete:
    """Tests for ``audit.check_roadmap_rows_complete``."""

    def test_flags_short_row(self, tmp_path: Path) -> None:
        """A numbered roadmap row with fewer than six columns is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        write_file(tmp_path / "docs" / "ROADMAP.md", "| 001 | x |\n")

        violations = audit.check_roadmap_rows_complete(tmp_path / "docs", tmp_path)

        assert violations == ["roadmap rows without exactly six columns: [('| 001 | x', 2)]"]


class TestCheckRoadmapStatusesValid:
    """Tests for ``audit.check_roadmap_statuses_valid``."""

    def test_flags_unrecognized_icon(self, tmp_path: Path) -> None:
        """A status icon outside the recognized set is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        replace_in_file(
            tmp_path / "docs" / "ROADMAP.md", "| 001 | a | b | c | d | ✅ |", "| 001 | a | b | c | d | ??? |"
        )

        violations = audit.check_roadmap_statuses_valid(tmp_path / "docs", tmp_path)

        assert any("invalid status values" in violation for violation in violations)


class TestCheckRoadmapHeaderCount:
    """Tests for ``audit.check_roadmap_header_count``."""

    def test_flags_mismatched_count(self, tmp_path: Path) -> None:
        """A stated package count that disagrees with the numbered rows is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        replace_in_file(
            tmp_path / "docs" / "ROADMAP.md", f"{audit._WP_FLOOR} numbered work packages.", "5 numbered work packages."
        )

        violations = audit.check_roadmap_header_count(tmp_path / "docs", tmp_path)

        assert violations == [f"header states 5 numbered work packages but the table carries {audit._WP_FLOOR}"]


class TestCheckProvenanceIds:
    """Tests for ``audit.check_provenance_ids``."""

    def test_flags_missing_source_id(self, tmp_path: Path) -> None:
        """A missing source id in the R1..R21 allowlist is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        write_file(tmp_path / "docs" / "PROVENANCE.md", "| R1 | source |\n")

        violations = audit.check_provenance_ids(tmp_path / "docs", tmp_path)

        assert any("provenance missing source ids" in violation for violation in violations)


class TestCheckReportSections:
    """Tests for ``audit.check_report_sections``."""

    def test_flags_missing_heading(self, tmp_path: Path) -> None:
        """A required report section heading missing from the file is reported."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        write_file(tmp_path / "docs" / "REPRODUCTION_REPORT.md", "## 0.1.0 — Detection\n")

        violations = audit.check_report_sections(tmp_path / "docs", tmp_path)

        assert any("missing section headings" in violation for violation in violations)

    def test_finds_section_heading_on_split_page(self, tmp_path: Path) -> None:
        """A required section whose heading moved to a ``report/`` page as an H1 still counts.

        The report is one page per release section, each promoted from H2 to H1; the check reads the landing file and
        every page so the split does not look like a dropped section.
        """
        _write_required_docs(tmp_path / "docs", tmp_path)
        write_file(tmp_path / "docs" / "REPRODUCTION_REPORT.md", "## 0.1.0 — Detection\n")
        (tmp_path / "docs" / "report").mkdir()
        write_file(tmp_path / "docs" / "report" / "keypoints.md", "# 0.5.0 — Keypoint detection\n")

        violations = audit.check_report_sections(tmp_path / "docs", tmp_path)

        assert "0.5.0 — Keypoint detection" not in violations[0]


class TestCheckDecisionsIds:
    """Tests for ``audit.check_decisions_ids``."""

    def test_flags_missing_adr(self, tmp_path: Path) -> None:
        """A missing ADR section heading is reported even when ids are contiguous."""
        _write_required_docs(tmp_path / "docs", tmp_path)
        write_file(tmp_path / "docs" / "DECISIONS.md", "".join(f"| D{i} | t |\n" for i in range(1, 30)))

        violations = audit.check_decisions_ids(tmp_path / "docs", tmp_path)

        assert any("missing ADR-001 section" in violation for violation in violations)

    def test_flags_row_deleted_from_middle(self, tmp_path: Path) -> None:
        """Deleting D5 from a full register drops the count below the floor and is reported.

        The predicate this pins: a ``max(ids)`` floor grades the register by its last row, so D28 surviving would hide
        D5's deletion behind a contiguity violation alone -- and a contiguity violation is what an id renumbering also
        produces. Counting rows makes the shrink its own finding.
        """
        _write_required_docs(tmp_path / "docs", tmp_path)
        replace_in_file(tmp_path / "docs" / "DECISIONS.md", "| D5 | t |\n", "")

        violations = audit.check_decisions_ids(tmp_path / "docs", tmp_path)

        assert any("decisions shrank below 29 rows: 28" in violation for violation in violations)


def test_find_violations_survives_missing_file(tmp_path: Path) -> None:
    """A check whose source file is entirely absent reports a violation instead of crashing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    violations = audit.find_violations(docs_dir, tmp_path)

    assert any(violation.startswith("missing policy docs") for violation in violations)


class TestCheckPullRequestAttestationCoversTheAllowlist:
    """Tests for ``audit.check_pull_request_attestation_covers_the_allowlist``."""

    @staticmethod
    def _write_template(repo_root: Path, body: str) -> None:
        """Write a pull-request template into a synthetic repository root."""
        (repo_root / ".github").mkdir(parents=True, exist_ok=True)
        (repo_root / ".github" / "PULL_REQUEST_TEMPLATE.md").write_text(body, encoding="utf-8")

    def test_flags_template_naming_only_denylist(self, tmp_path: Path) -> None:
        """Naming Ultralytics alone states one instance of the rule as though it were the rule.

        This is the pre-WP-143 template, and the violation it produces is the whole reason the check exists: a
        contributor reading that line would reasonably conclude a GPL detector was fair game.
        """
        self._write_template(tmp_path, "- [ ] Clean-room acknowledged: no Ultralytics source was consulted\n")

        violations = audit.check_pull_request_attestation_covers_the_allowlist(tmp_path / "docs", tmp_path)

        assert len(violations) == 1
        assert "AGPL" in violations[0]
        assert "proprietary" in violations[0]

    def test_clean_when_every_family_and_clause_named(self, tmp_path: Path) -> None:
        """A template naming each excluded family and both clauses reports nothing."""
        self._write_template(
            tmp_path,
            "AGPL, GPL, LGPL, SSPL, BSL, Elastic, PolyForm, proprietary, and any licence that cannot be read\n",
        )

        assert audit.check_pull_request_attestation_covers_the_allowlist(tmp_path / "docs", tmp_path) == []

    def test_flags_single_dropped_family(self, tmp_path: Path) -> None:
        """One family falling out of a rewrite is caught, which is the realistic failure.

        The template is prose a maintainer edits; a wholesale revert is obvious and a single dropped line is not.
        """
        self._write_template(
            tmp_path,
            "AGPL, GPL, LGPL, BSL, Elastic, PolyForm, proprietary, and any licence that cannot be read\n",
        )

        violations = audit.check_pull_request_attestation_covers_the_allowlist(tmp_path / "docs", tmp_path)

        assert violations == ["pull-request attestation does not name: SSPL"]

    def test_flags_missing_template(self, tmp_path: Path) -> None:
        """A deleted template is a missing attestation, not an empty one."""
        violations = audit.check_pull_request_attestation_covers_the_allowlist(tmp_path / "docs", tmp_path)

        assert violations == ["missing pull-request template: .github/PULL_REQUEST_TEMPLATE.md"]
