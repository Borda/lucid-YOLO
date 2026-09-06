# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: license and attribution hygiene audit script (WP-002, split off WP-117).

Covers ``scripts/lint/audit_license_headers.py``'s public contract in isolation, on
synthetic files under ``tmp_path`` rather than against the live repo tree -- the live
tree is exercised instead by the pre-commit hook itself each time a guarded file
changes. This file is the functional-core check pytest owns; the hook is the lint
gate that runs it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "scripts" / "lint" / "audit_license_headers.py"


def _load_audit() -> ModuleType:
    """Load ``scripts/lint/audit_license_headers.py`` as an importable module.

    Examples:
        >>> module = _load_audit()
        >>> module.__name__
        'audit_license_headers'
        >>> callable(module.find_violations)
        True
    """
    spec = importlib.util.spec_from_file_location("audit_license_headers", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def test_license_is_apache2_accepts_the_real_header(tmp_path: Path) -> None:
    """An Apache License, Version 2.0 header passes without a violation."""
    (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n", encoding="utf-8")
    assert audit.check_license_is_apache2(tmp_path) == []


def test_license_is_apache2_rejects_other_licenses(tmp_path: Path) -> None:
    """A non-Apache LICENSE file is flagged by name."""
    (tmp_path / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    assert audit.check_license_is_apache2(tmp_path) == ["LICENSE is not the Apache License, Version 2.0"]


class TestLicenseAppendixPlaceholder:
    """The appendix's copyright line says whether the licence was applied or only copied."""

    def test_an_unfilled_placeholder_is_rejected(self, tmp_path: Path) -> None:
        """`Copyright [yyyy] [name of copyright owner]` fails, wherever in the file it sits.

        It sits about 5 kB into the Apache template, past any head-read window, so a
        check reading the first 200 characters cannot see it however correct that
        check is about the title above it. The whole file is read for this one.
        """
        (tmp_path / "LICENSE").write_text(
            "Apache License\nVersion 2.0, January 2004\n" + "filler\n" * 200 + audit.COPYRIGHT_PLACEHOLDER + "\n",
            encoding="utf-8",
        )

        violations = audit.check_license_is_apache2(tmp_path)

        assert violations == [f"LICENSE still carries the Apache appendix placeholder: {audit.COPYRIGHT_PLACEHOLDER}"]

    def test_a_filled_appendix_passes(self, tmp_path: Path) -> None:
        """A named holder and year in the appendix is what the placeholder is replaced by."""
        (tmp_path / "LICENSE").write_text(
            "Apache License\nVersion 2.0, January 2004\n" + "filler\n" * 200 + "Copyright 2026 Jirka Borovec\n",
            encoding="utf-8",
        )

        assert audit.check_license_is_apache2(tmp_path) == []

    def test_the_real_license_is_filled(self) -> None:
        """The repository's own LICENSE carries a holder, not the template's brackets."""
        assert audit.COPYRIGHT_PLACEHOLDER not in (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")


class TestNoticeAttribution:
    """``check_notice_attribution`` over a synthetic NOTICE.

    Grouped once WP-175 added the paper-phrase case: the three cases share one subject
    and the flat-group audit refuses a third sibling at module level.
    """

    def test_accepts_a_complete_notice(self, tmp_path: Path) -> None:
        """A NOTICE carrying every required fragment passes without a violation.

        The fixture is the shape the live file has after WP-175 corrected it: the four
        legal fragments plus the nominative paper phrase.
        """
        (tmp_path / "NOTICE").write_text(
            "Redmon, arXiv:1506.02640, not affiliated with, endorsed by, or derived from Ultralytics, "
            f"Apache License, {audit.PAPER_PHRASE}\n",
            encoding="utf-8",
        )
        assert audit.check_notice_attribution(tmp_path) == []

    def test_names_each_missing_fragment(self, tmp_path: Path) -> None:
        """Every missing NOTICE fragment is reported, one violation per fragment.

        A NOTICE stripped of everything relevant is the worst case, and the assertion is
        on the whole list rather than its length so a fragment cannot be dropped from the
        required set without a test noticing.
        """
        (tmp_path / "NOTICE").write_text("Nothing relevant here.\n", encoding="utf-8")
        violations = audit.check_notice_attribution(tmp_path)
        assert violations == [
            "NOTICE missing: Redmon",
            "NOTICE missing: arXiv:1506.02640",
            "NOTICE missing: not affiliated with, endorsed by, or derived from Ultralytics",
            "NOTICE missing: Apache License",
            f"NOTICE missing: {audit.PAPER_PHRASE}",
        ]

    def test_rejects_the_vendor_bound_paper_phrase(self, tmp_path: Path) -> None:
        """A NOTICE naming *the Ultralytics YOLO26 paper* is refused for lacking the nominative form.

        This is the drift WP-161 corrected in the README and explicitly left in ``NOTICE``
        for a later row. The required fragment is what makes it fail rather than sit
        unread: ``the YOLO26 paper`` is not a substring of ``the Ultralytics YOLO26
        paper``, so the vendor-bound sentence satisfies every other fragment and still
        cannot pass.
        """
        (tmp_path / "NOTICE").write_text(
            "Redmon, arXiv:1506.02640, not affiliated with, endorsed by, or derived from Ultralytics, "
            "Apache License, methods described in the Ultralytics YOLO26 paper\n",
            encoding="utf-8",
        )
        assert audit.check_notice_attribution(tmp_path) == [f"NOTICE missing: {audit.PAPER_PHRASE}"]


class TestReadmeDisclaimer:
    """``check_readme_disclaimer`` over a synthetic README."""

    def test_accepts_all_fragments_present(self, tmp_path: Path) -> None:
        """A README carrying every disclaimer fragment passes without a violation.

        Built from ``DISCLAIMER_FRAGMENTS`` itself, so a fragment added to the audit is
        covered by this case the moment it is declared.
        """
        (tmp_path / "README.md").write_text(
            "\n".join(audit.DISCLAIMER_FRAGMENTS) + "\n",
            encoding="utf-8",
        )
        assert audit.check_readme_disclaimer(tmp_path) == []

    def test_names_each_missing_fragment(self, tmp_path: Path) -> None:
        """Every missing README disclaimer fragment is reported, one violation per fragment.

        The empty-README case: one violation per declared fragment and no collapsing of
        several missing fragments into a single message.
        """
        (tmp_path / "README.md").write_text("No disclaimer at all.\n", encoding="utf-8")
        violations = audit.check_readme_disclaimer(tmp_path)
        assert len(violations) == len(audit.DISCLAIMER_FRAGMENTS)

    def test_rejects_the_vendor_bound_paper_phrase(self, tmp_path: Path) -> None:
        """A README with every disclaimer fragment but the vendor-bound paper phrase is refused.

        The same guard on the other file the audit pins: a README whose banner regressed
        to *the Ultralytics YOLO26 paper* keeps all three legal fragments intact, so
        before this fragment existed nothing in the gate could tell that banner from the
        corrected one.
        """
        (tmp_path / "README.md").write_text(
            "\n".join(f for f in audit.DISCLAIMER_FRAGMENTS if f != audit.PAPER_PHRASE)
            + "\nmethods described in the Ultralytics YOLO26 paper\n",
            encoding="utf-8",
        )
        assert audit.check_readme_disclaimer(tmp_path) == [
            f"README missing disclaimer fragment: {audit.PAPER_PHRASE!r}"
        ]


def test_spdx_header_accepts_a_headered_file(tmp_path: Path) -> None:
    """A src/**/*.py file starting with the SPDX line passes without a violation."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "good.py").write_text(f"{audit.SPDX_LINE}\nimport os\n", encoding="utf-8")
    assert audit.check_source_files_carry_spdx_header(tmp_path) == []


def test_spdx_header_names_the_offending_file(tmp_path: Path) -> None:
    """A src/**/*.py file missing the SPDX line is named, relative to the repo root."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "bad.py").write_text("import os\n", encoding="utf-8")
    violations = audit.check_source_files_carry_spdx_header(tmp_path)
    assert violations == ["files missing SPDX header: ['src/bad.py']"]


class TestSpdxHeaderScope:
    """The header scope is the three trees this repository writes Python into."""

    def test_a_shebang_may_precede_the_header(self, tmp_path: Path) -> None:
        """An executable script carries the header on line two and is not unlicensed.

        A shebang has to be first to work at all, so a first-line-only read calls
        every runnable script headerless -- which is what the two `scripts/` files
        reported as missing headers actually were.
        """
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "runnable.py").write_text(f"#!/usr/bin/env python\n{audit.SPDX_LINE}\nimport os\n", encoding="utf-8")

        assert audit.check_source_files_carry_spdx_header(tmp_path) == []

    def test_a_scripts_file_without_the_header_is_named(self, tmp_path: Path) -> None:
        """`scripts/` is in scope, so a missing header there fails rather than passing unseen."""
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "bare.py").write_text("import os\n", encoding="utf-8")

        assert audit.check_source_files_carry_spdx_header(tmp_path) == [
            "files missing SPDX header: ['scripts/bare.py']"
        ]

    def test_a_tests_file_without_the_header_is_named(self, tmp_path: Path) -> None:
        """`tests/` is in scope too; a file that travels carries its marker or does not."""
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_bare.py").write_text("import os\n", encoding="utf-8")

        violations = audit.check_source_files_carry_spdx_header(tmp_path)

        assert violations == ["files missing SPDX header: ['tests/test_bare.py']"]

    def test_the_scope_is_stated_rather_than_implied_by_a_glob(self) -> None:
        """The trees are a named constant, so widening or narrowing is a visible decision."""
        assert audit.HEADER_DIRS == ("src", "scripts", "tests")


def test_the_live_repo_tree_is_currently_clean() -> None:
    """The real repo tree the pre-commit hook scans has no license/attribution violations.

    This is the one place this file touches the live tree rather than a synthetic
    one: a regression here means the hook itself would fail on the next commit
    that touches a guarded file, not just a new one.
    """
    assert audit.find_violations(REPO_ROOT) == []
