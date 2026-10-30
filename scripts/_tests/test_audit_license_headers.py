# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: license and attribution hygiene audit script (WP-002, split off WP-117).

Covers ``scripts/lint/audit_license_headers.py``'s public contract in isolation, on synthetic files under ``tmp_path``
rather than against the live repo tree -- the live tree is exercised instead by the pre-commit hook itself each time a
guarded file changes. This file is the functional-core check pytest owns; the hook is the lint gate that runs it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]


audit = load_script("audit_license_headers")


class TestLicenseApache2:
    """``check_license_is_apache2`` over a synthetic LICENSE."""

    def test_accepts_real_header(self, tmp_path: Path) -> None:
        """An Apache License, Version 2.0 header passes without a violation."""
        (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n", encoding="utf-8")
        assert audit.check_license_is_apache2(tmp_path) == []

    def test_rejects_other_licenses(self, tmp_path: Path) -> None:
        """A non-Apache LICENSE file is flagged by name."""
        (tmp_path / "LICENSE").write_text("MIT License\n", encoding="utf-8")
        assert audit.check_license_is_apache2(tmp_path) == ["LICENSE is not the Apache License, Version 2.0"]


class TestLicenseAppendixPlaceholder:
    """The appendix's copyright line says whether the licence was applied or only copied."""

    def test_unfilled_placeholder_rejected(self, tmp_path: Path) -> None:
        """`Copyright [yyyy] [name of copyright owner]` fails, wherever in the file it sits.

        It sits about 5 kB into the Apache template, past any head-read window, so a check reading the first 200
        characters cannot see it however correct that check is about the title above it. The whole file is read for
        this one.
        """
        (tmp_path / "LICENSE").write_text(
            "Apache License\nVersion 2.0, January 2004\n" + "filler\n" * 200 + audit.COPYRIGHT_PLACEHOLDER + "\n",
            encoding="utf-8",
        )

        violations = audit.check_license_is_apache2(tmp_path)

        assert violations == [f"LICENSE still carries the Apache appendix placeholder: {audit.COPYRIGHT_PLACEHOLDER}"]

    def test_filled_appendix_passes(self, tmp_path: Path) -> None:
        """A named holder and year in the appendix is what the placeholder is replaced by."""
        (tmp_path / "LICENSE").write_text(
            "Apache License\nVersion 2.0, January 2004\n" + "filler\n" * 200 + "Copyright 2026 Jirka Borovec\n",
            encoding="utf-8",
        )

        assert audit.check_license_is_apache2(tmp_path) == []


class TestNoticeAttribution:
    """``check_notice_attribution`` over a synthetic NOTICE.

    Grouped once WP-175 added the paper-phrase case: the three cases share one subject and the flat-group audit refuses
    a third sibling at module level.
    """

    def test_accepts_complete_notice(self, tmp_path: Path) -> None:
        """A NOTICE carrying every required fragment passes without a violation.

        The fixture is the shape the live file has after WP-175 corrected it: the four legal fragments plus the
        nominative paper phrase.
        """
        (tmp_path / "NOTICE").write_text(
            "Redmon, arXiv:1506.02640, not affiliated with, endorsed by, or derived from Ultralytics, "
            f"Apache License, {audit.PAPER_PHRASE}\n",
            encoding="utf-8",
        )
        assert audit.check_notice_attribution(tmp_path) == []

    def test_names_each_missing_fragment(self, tmp_path: Path) -> None:
        """Every missing NOTICE fragment is reported, one violation per fragment.

        A NOTICE stripped of everything relevant is the worst case, and the assertion is on the whole list rather than
        its length so a fragment cannot be dropped from the required set without a test noticing.
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

    def test_rejects_vendor_bound_paper_phrase(self, tmp_path: Path) -> None:
        """A NOTICE naming *the Ultralytics YOLO26 paper* is refused for lacking the nominative form.

        This is the drift WP-161 corrected in the README and explicitly left in ``NOTICE`` for a later row. The required
        fragment is what makes it fail rather than sit unread: ``the YOLO26 paper`` is not a substring of ``the
        Ultralytics YOLO26 paper``, so the vendor-bound sentence satisfies every other fragment and still cannot pass.
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

        Built from ``DISCLAIMER_FRAGMENTS`` itself, so a fragment added to the audit is covered by this case the moment
        it is declared.
        """
        (tmp_path / "README.md").write_text("\n".join(audit.DISCLAIMER_FRAGMENTS) + "\n", encoding="utf-8")
        assert audit.check_readme_disclaimer(tmp_path) == []

    def test_names_each_missing_fragment(self, tmp_path: Path) -> None:
        """Every missing README disclaimer fragment is reported, one violation per fragment.

        The empty-README case: one violation per declared fragment and no collapsing of several missing fragments into
        a single message.
        """
        (tmp_path / "README.md").write_text("No disclaimer at all.\n", encoding="utf-8")
        violations = audit.check_readme_disclaimer(tmp_path)
        assert len(violations) == len(audit.DISCLAIMER_FRAGMENTS)

    def test_rejects_vendor_bound_paper_phrase(self, tmp_path: Path) -> None:
        """A README with every disclaimer fragment but the vendor-bound paper phrase is refused.

        The same guard on the other file the audit pins: a README whose banner regressed to *the Ultralytics YOLO26
        paper* keeps all three legal fragments intact, so before this fragment existed nothing in the gate could tell
        that banner from the corrected one.
        """
        (tmp_path / "README.md").write_text(
            "\n".join(f for f in audit.DISCLAIMER_FRAGMENTS if f != audit.PAPER_PHRASE)
            + "\nmethods described in the Ultralytics YOLO26 paper\n",
            encoding="utf-8",
        )
        assert audit.check_readme_disclaimer(tmp_path) == [
            f"README missing disclaimer fragment: {audit.PAPER_PHRASE!r}"
        ]


class TestSpdxHeader:
    """The header scope is the four trees this repository writes Python into."""

    @pytest.mark.parametrize(
        ("relative", "source"),
        [
            # `src/` is the original scope: a library file missing the SPDX line is named, relative to the repo root.
            pytest.param("src/bad.py", "import os\n", id="src"),
            # `scripts/` is in scope, so a missing header there fails rather than passing unseen.
            pytest.param("scripts/bare.py", "import os\n", id="scripts"),
            # `tests/` is in scope too; a file that travels carries its marker or does not.
            pytest.param("tests/test_bare.py", "import os\n", id="tests"),
            # `notebooks/` is in scope (WP-188): the jupytext source is the file Colab hands a reader.
            pytest.param("notebooks/bare.py", "# %%\nimport os\n", id="notebooks"),
        ],
    )
    def test_file_without_header_named(self, tmp_path: Path, relative: str, source: str) -> None:
        """A Python file missing the SPDX line is named, relative to the repo root, in every scoped tree."""
        target = tmp_path / relative
        target.parent.mkdir()
        target.write_text(source, encoding="utf-8")

        violations = audit.check_source_files_carry_spdx_header(tmp_path)

        assert violations == [f"files missing SPDX header: ['{relative}']"]

    @pytest.mark.parametrize(
        ("relative", "source"),
        [
            # The plain case: the SPDX line is the first line.
            pytest.param("src/good.py", f"{audit.SPDX_LINE}\nimport os\n", id="first-line"),
            # A shebang has to be first to work at all, so a first-line-only read calls every runnable script
            # headerless -- which is what the two `scripts/` files reported as missing headers actually were.
            pytest.param("scripts/runnable.py", f"#!/usr/bin/env python\n{audit.SPDX_LINE}\nimport os\n", id="shebang"),
        ],
    )
    def test_headered_file_accepted(self, tmp_path: Path, relative: str, source: str) -> None:
        """A file starting with the SPDX line passes, and an executable script may carry it on line two."""
        target = tmp_path / relative
        target.parent.mkdir()
        target.write_text(source, encoding="utf-8")

        assert audit.check_source_files_carry_spdx_header(tmp_path) == []

    def test_absent_tree_not_failure(self, tmp_path: Path) -> None:
        """A scoped tree that does not exist yet yields no files rather than an error.

        `notebooks/` joined the scope before any source landed in it, so the walk has to tolerate a checkout without the
        directory -- `rglob` on a missing path is empty.
        """
        assert audit.check_source_files_carry_spdx_header(tmp_path) == []

    def test_scope_stated_rather_than_implied_by_glob(self) -> None:
        """The trees are a named constant, so widening or narrowing is a visible decision."""
        assert audit.HEADER_DIRS == ("src", "scripts", "tests", "notebooks")
