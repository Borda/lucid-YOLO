# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the docs-site audit script (WP-114).

Covers ``scripts/lint/audit_docs_site.py``'s public contract in isolation, on synthetic
files under ``tmp_path`` rather than against the live docs tree -- the live tree is
exercised instead by the pre-commit hook itself each time ``mkdocs.yml``, ``docs/**``,
``pyproject.toml``, or ``.github/workflows/docs.yml`` changes. This file is the
functional-core check pytest owns; the hook is the lint gate that runs it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "scripts" / "lint" / "audit_docs_site.py"


def _load_audit() -> ModuleType:
    """Load ``scripts/lint/audit_docs_site.py`` as an importable module.

    Examples:
        >>> module = _load_audit()
        >>> module.__name__
        'audit_docs_site'
        >>> callable(module.find_violations)
        True
    """
    spec = importlib.util.spec_from_file_location("audit_docs_site", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def test_every_docs_page_is_listed_in_the_nav_flags_an_orphan_page(tmp_path: Path) -> None:
    """A markdown file under ``docs/`` absent from the nav is reported by name."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "orphan.md").write_text("x", encoding="utf-8")
    mkdocs_yml = tmp_path / "mkdocs.yml"
    mkdocs_yml.write_text("nav:\n  - index.md\n", encoding="utf-8")

    assert audit.check_every_docs_page_is_listed_in_the_nav(docs, mkdocs_yml) == [
        "docs pages absent from the nav: ['orphan.md']"
    ]


def test_every_docs_page_is_listed_in_the_nav_passes_when_covered(tmp_path: Path) -> None:
    """A page named in the nav produces no violation."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.md").write_text("x", encoding="utf-8")
    mkdocs_yml = tmp_path / "mkdocs.yml"
    mkdocs_yml.write_text("nav:\n  - index.md\n", encoding="utf-8")

    assert audit.check_every_docs_page_is_listed_in_the_nav(docs, mkdocs_yml) == []


def test_every_nav_entry_points_at_a_file_that_exists_flags_a_dead_entry(tmp_path: Path) -> None:
    """A nav entry naming a file absent from disk is reported by name."""
    docs = tmp_path / "docs"
    docs.mkdir()
    mkdocs_yml = tmp_path / "mkdocs.yml"
    mkdocs_yml.write_text("nav:\n  - ghost.md\n", encoding="utf-8")

    assert audit.check_every_nav_entry_points_at_a_file_that_exists(docs, mkdocs_yml) == [
        "nav entries with no file on disk: ['ghost.md']"
    ]


def test_every_nav_entry_points_at_a_file_that_exists_passes_when_covered(tmp_path: Path) -> None:
    """A nav entry naming a file on disk produces no violation."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.md").write_text("x", encoding="utf-8")
    mkdocs_yml = tmp_path / "mkdocs.yml"
    mkdocs_yml.write_text("nav:\n  - index.md\n", encoding="utf-8")

    assert audit.check_every_nav_entry_points_at_a_file_that_exists(docs, mkdocs_yml) == []


def test_the_gfm_table_extension_is_declared_flags_a_missing_extension(tmp_path: Path) -> None:
    """A ``markdown_extensions`` list without ``tables`` is reported."""
    mkdocs_yml = tmp_path / "mkdocs.yml"
    mkdocs_yml.write_text("markdown_extensions:\n  - admonition\n", encoding="utf-8")

    assert audit.check_the_gfm_table_extension_is_declared(mkdocs_yml) == [
        "the 'tables' markdown extension is not declared"
    ]


def test_the_gfm_table_extension_is_declared_passes_when_present(tmp_path: Path) -> None:
    """A ``markdown_extensions`` list carrying ``tables`` produces no violation."""
    mkdocs_yml = tmp_path / "mkdocs.yml"
    mkdocs_yml.write_text("markdown_extensions:\n  - tables\n", encoding="utf-8")

    assert audit.check_the_gfm_table_extension_is_declared(mkdocs_yml) == []


def test_the_repo_url_matches_the_declared_homepage_flags_a_mismatch(tmp_path: Path) -> None:
    """A ``repo_url`` that disagrees with ``pyproject.toml``'s Homepage is reported."""
    mkdocs_yml = tmp_path / "mkdocs.yml"
    mkdocs_yml.write_text("repo_url: https://example.com/a\n", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('Homepage = "https://example.com/b"\n', encoding="utf-8")

    assert audit.check_the_repo_url_matches_the_declared_homepage(mkdocs_yml, pyproject) == [
        "repo_url 'https://example.com/a' does not match pyproject.toml Homepage 'https://example.com/b'"
    ]


class TestRepoName:
    """`repo_name` holds the same fact as `repo_url` in a second spelling, so it drifts."""

    def test_a_stale_repo_name_is_flagged(self, tmp_path: Path) -> None:
        """A `repo_name` naming a different slug than `repo_url` fails.

        This is the surface a rename leaves behind: Material renders `repo_name` as
        the header link's label on every page, so the old slug stays visible site-wide
        while every link still resolves and no build step reports anything.
        """
        mkdocs_yml = tmp_path / "mkdocs.yml"
        mkdocs_yml.write_text(
            "repo_url: https://github.com/Borda/lit-YOLOs\nrepo_name: Borda/lucid-YOLO\n", encoding="utf-8"
        )
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('Homepage = "https://github.com/Borda/lit-YOLOs"\n', encoding="utf-8")

        violations = audit.check_the_repo_url_matches_the_declared_homepage(mkdocs_yml, pyproject)

        assert violations == ["repo_name 'Borda/lucid-YOLO' does not name the repo_url slug 'Borda/lit-YOLOs'"]

    def test_a_matching_repo_name_passes(self, tmp_path: Path) -> None:
        """The slug taken from the URL's last two segments is what `repo_name` must equal."""
        mkdocs_yml = tmp_path / "mkdocs.yml"
        mkdocs_yml.write_text(
            "repo_url: https://github.com/Borda/lucid-YOLO\nrepo_name: Borda/lucid-YOLO\n", encoding="utf-8"
        )
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('Homepage = "https://github.com/Borda/lucid-YOLO"\n', encoding="utf-8")

        assert audit.check_the_repo_url_matches_the_declared_homepage(mkdocs_yml, pyproject) == []

    def test_an_absent_repo_name_is_not_invented(self, tmp_path: Path) -> None:
        """A config declaring no `repo_name` is not failed for it -- Material derives its own."""
        mkdocs_yml = tmp_path / "mkdocs.yml"
        mkdocs_yml.write_text("repo_url: https://example.com/a\n", encoding="utf-8")
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('Homepage = "https://example.com/a"\n', encoding="utf-8")

        assert audit.check_the_repo_url_matches_the_declared_homepage(mkdocs_yml, pyproject) == []

    def test_the_live_config_declares_a_matching_repo_name(self) -> None:
        """The real `mkdocs.yml` names the same slug its `repo_url` does."""
        assert (
            audit.check_the_repo_url_matches_the_declared_homepage(audit.DEFAULT_MKDOCS_YML, audit.DEFAULT_PYPROJECT)
            == []
        )


def test_the_repo_url_matches_the_declared_homepage_passes_when_equal(tmp_path: Path) -> None:
    """A matching ``repo_url`` and Homepage produce no violation."""
    mkdocs_yml = tmp_path / "mkdocs.yml"
    mkdocs_yml.write_text("repo_url: https://example.com/a\n", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('Homepage = "https://example.com/a"\n', encoding="utf-8")

    assert audit.check_the_repo_url_matches_the_declared_homepage(mkdocs_yml, pyproject) == []


def test_the_site_description_is_the_distribution_description_flags_a_mismatch(tmp_path: Path) -> None:
    """A ``site_description`` that diverges from ``pyproject.toml``'s description is reported."""
    mkdocs_yml = tmp_path / "mkdocs.yml"
    mkdocs_yml.write_text("site_description: A tool.\n", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('description = "A different tool."\n', encoding="utf-8")

    assert audit.check_the_site_description_is_the_distribution_description(mkdocs_yml, pyproject) == [
        "site_description 'A tool.' does not match pyproject.toml description 'A different tool.'"
    ]


def test_the_site_description_is_the_distribution_description_passes_when_equal(tmp_path: Path) -> None:
    """A matching ``site_description`` and description produce no violation."""
    mkdocs_yml = tmp_path / "mkdocs.yml"
    mkdocs_yml.write_text("site_description: A tool.\n", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('description = "A tool."\n', encoding="utf-8")

    assert audit.check_the_site_description_is_the_distribution_description(mkdocs_yml, pyproject) == []


def test_mkdocs_is_capped_below_the_unlicensed_major_flags_an_uncapped_pin(tmp_path: Path) -> None:
    """A ``mkdocs`` pin without an upper bound below 2.0 is reported."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('    "mkdocs>=1.6",\n', encoding="utf-8")

    assert audit.check_mkdocs_is_capped_below_the_unlicensed_major(pyproject) == [
        'mkdocs is not capped below 2.0: "mkdocs>=1.6"'
    ]


def test_mkdocs_is_capped_below_the_unlicensed_major_passes_when_capped(tmp_path: Path) -> None:
    """A ``mkdocs`` pin capped below 2.0 produces no violation."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('    "mkdocs>=1.6,<2",\n', encoding="utf-8")

    assert audit.check_mkdocs_is_capped_below_the_unlicensed_major(pyproject) == []


class TestCheckTheDocsWorkflowAuditsLicencesWhereTheDocsTreeIsInstalled:
    """Tests for ``audit.check_the_docs_workflow_audits_licences_where_the_docs_tree_is_installed``."""

    def test_flags_a_missing_audit(self, tmp_path: Path) -> None:
        """A ``docs.yml`` that never runs the licence audit is reported."""
        workflow = tmp_path / "docs.yml"
        workflow.write_text("run: python -m mkdocs build --strict\n", encoding="utf-8")

        assert audit.check_the_docs_workflow_audits_licences_where_the_docs_tree_is_installed(workflow) == [
            "the docs environment is never audited"
        ]

    def test_flags_a_late_audit(self, tmp_path: Path) -> None:
        """An audit that runs after the build, rather than before it, is reported."""
        workflow = tmp_path / "docs.yml"
        workflow.write_text(
            "run: python -m mkdocs build --strict\nrun: python scripts/lint/audit_licenses.py\n",
            encoding="utf-8",
        )

        assert audit.check_the_docs_workflow_audits_licences_where_the_docs_tree_is_installed(workflow) == [
            "the licence audit does not run before the docs build"
        ]

    def test_passes_when_ordered(self, tmp_path: Path) -> None:
        """An audit that runs before the build produces no violation."""
        workflow = tmp_path / "docs.yml"
        workflow.write_text(
            "run: python scripts/lint/audit_licenses.py\nrun: python -m mkdocs build --strict\n",
            encoding="utf-8",
        )

        assert audit.check_the_docs_workflow_audits_licences_where_the_docs_tree_is_installed(workflow) == []


def test_the_live_docs_site_is_currently_clean() -> None:
    """The real ``mkdocs.yml``, ``docs/``, ``pyproject.toml``, and CI workflow agree.

    This is the one place this file touches the live tree rather than a synthetic one: a
    regression here means the pre-commit hook itself would fail on the next commit that
    touches any of those paths, not just a new one.
    """
    assert audit.find_violations() == []
