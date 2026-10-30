# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the docs-site audit script (WP-114).

Covers ``scripts/lint/audit_docs_site.py``'s public contract in isolation, on synthetic files under ``tmp_path``
rather than against the live docs tree -- the live tree is exercised instead by the pre-commit hook itself each time
``mkdocs.yml``, ``docs/**``, ``pyproject.toml``, ``notebooks/*.py``, or ``.github/workflows/docs.yml`` changes. This
file is the functional-core check pytest owns; the hook is the lint gate that runs it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from audit_tree_helpers import make_docs_site, write_file, write_tree
from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]


audit = load_script("audit_docs_site")


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        pytest.param(("orphan.md",), ["docs pages absent from the nav: ['orphan.md']"], id="orphan-page"),
        pytest.param(("index.md",), [], id="page-in-nav"),
    ],
)
def test_docs_pages_listed_in_nav(tmp_path: Path, pages: tuple[str, ...], expected: list[str]) -> None:
    """A markdown file under ``docs/`` absent from the nav is reported by name; a page the nav names is not."""
    docs, mkdocs_yml = make_docs_site(tmp_path, "nav:\n  - index.md\n", pages)

    assert audit.check_every_docs_page_is_listed_in_the_nav(docs, mkdocs_yml) == expected


class TestPagesDeclaredNotInNav:
    """``not_in_nav`` patterns exempt pages from the nav-coverage check, and only those."""

    @pytest.mark.parametrize(
        ("page", "expected"),
        [
            pytest.param("index.md", [], id="matched-page-exempt"),
            pytest.param("orphan.md", ["docs pages absent from the nav: ['orphan.md']"], id="page-outside-pattern"),
        ],
    )
    def test_exempts_only_pages_matching_a_pattern(self, tmp_path: Path, page: str, expected: list[str]) -> None:
        """A work-package page matched by ``not_in_nav`` is not an orphan; a page the pattern misses still is.

        The ~160 per-WP pages are reached from roadmap rows and from the phase pages that include them; listing each in
        the sidebar is what the pattern exists to avoid. A page the pattern does not match is still reported when the
        nav omits it.
        """
        docs, mkdocs_yml = make_docs_site(
            tmp_path, "not_in_nav: |\n  /work_packages/*.md\nnav:\n  - index.md\n", ("work_packages/wp-001.md", page)
        )

        assert audit.check_every_docs_page_is_listed_in_the_nav(docs, mkdocs_yml) == expected


@pytest.mark.parametrize(
    ("entry", "pages", "expected"),
    [
        pytest.param("ghost.md", (), ["nav entries with no file on disk: ['ghost.md']"], id="dead-entry"),
        pytest.param("index.md", ("index.md",), [], id="entry-with-file"),
    ],
)
def test_nav_entries_point_at_files_that_exist(
    tmp_path: Path, entry: str, pages: tuple[str, ...], expected: list[str]
) -> None:
    """A nav entry naming a file absent from disk is reported by name; one naming a file on disk is not."""
    docs, mkdocs_yml = make_docs_site(tmp_path, f"nav:\n  - {entry}\n", pages)

    assert audit.check_every_nav_entry_points_at_a_file_that_exists(docs, mkdocs_yml) == expected


class TestNotebookPagesInTheNav:
    """A `notebooks/<name>.ipynb` nav entry is a build product resolved against its `.py` source (WP-188)."""

    def test_entry_whose_source_exists_passes_without_ipynb(self, tmp_path: Path) -> None:
        """The `.ipynb` is gitignored and may not have been written yet; the source is what has to exist.

        A cold checkout -- CI, or a contributor who has not run `make notebooks` -- has the `.py` and not the `.ipynb`,
        and the audit has to answer the same there as on a warm one.
        """
        docs, mkdocs_yml = make_docs_site(
            tmp_path, "nav:\n  - Notebooks:\n      - Wiring gate: notebooks/wiring_gate.ipynb\n"
        )
        write_file(tmp_path / "notebooks" / "wiring_gate.py", "# %%\n")

        violations = audit.check_every_nav_entry_points_at_a_file_that_exists(docs, mkdocs_yml, tmp_path / "notebooks")

        assert violations == []

    def test_entry_with_no_source_flagged_even_if_stale_ipynb_exists(self, tmp_path: Path) -> None:
        """A generated `.ipynb` left behind by a deleted source does not make the nav entry valid.

        Resolving against the product would pass on the warm checkout that deleted the source and fail on every cold
        one; the source is the only file the repository holds.
        """
        docs, mkdocs_yml = make_docs_site(tmp_path, "nav:\n  - notebooks/gone.ipynb\n", ("notebooks/gone.ipynb",))
        (tmp_path / "notebooks").mkdir()

        violations = audit.check_every_nav_entry_points_at_a_file_that_exists(docs, mkdocs_yml, tmp_path / "notebooks")

        assert violations == ["nav entries with no file on disk: ['notebooks/gone.ipynb']"]

    def test_ipynb_outside_notebooks_prefix_ordinary_page(self, tmp_path: Path) -> None:
        """Only the `notebooks/` prefix is derived; any other `.ipynb` must exist under `docs/` itself."""
        docs, mkdocs_yml = make_docs_site(tmp_path, "nav:\n  - elsewhere/x.ipynb\n")
        write_file(tmp_path / "notebooks" / "x.py", "# %%\n")

        violations = audit.check_every_nav_entry_points_at_a_file_that_exists(docs, mkdocs_yml, tmp_path / "notebooks")

        assert violations == ["nav entries with no file on disk: ['elsewhere/x.ipynb']"]


class TestEveryNotebookSourceHasANavEntry:
    """Every `notebooks/*.py` owes the nav a `notebooks/<name>.ipynb` entry (WP-188)."""

    def test_source_absent_from_nav_named(self, tmp_path: Path) -> None:
        """A notebook nobody can navigate to is one nobody reads -- the page rule, applied to sources."""
        write_tree(tmp_path, {"notebooks/orphan.py": "# %%\n", "mkdocs.yml": "nav:\n  - index.md\n"})

        violations = audit.check_every_notebook_source_has_a_nav_entry(tmp_path / "mkdocs.yml", tmp_path / "notebooks")

        assert violations == ["notebook sources absent from the nav: ['orphan.py']"]

    @pytest.mark.parametrize(
        "files",
        [
            pytest.param(
                {"notebooks/a.py": "# %%\n", "mkdocs.yml": "nav:\n  - Notebooks:\n      - A: notebooks/a.ipynb\n"},
                id="source-named-in-nav",
            ),
            pytest.param({"mkdocs.yml": "nav:\n  - index.md\n"}, id="missing-directory"),
            pytest.param(
                {"notebooks/README.md": "# how to\n", "mkdocs.yml": "nav:\n  - index.md\n"}, id="non-python-file"
            ),
        ],
    )
    def test_nothing_owed_to_nav_clean(self, tmp_path: Path, files: dict[str, str]) -> None:
        """A source whose `.ipynb` the nav lists, a missing `notebooks/` directory, and a non-source file are clean.

        No directory at all is the same as an empty one, not a failure to read; `notebooks/README.md` lives beside the
        sources and owes the nav nothing.
        """
        write_tree(tmp_path, files)

        assert audit.check_every_notebook_source_has_a_nav_entry(tmp_path / "mkdocs.yml", tmp_path / "notebooks") == []

    def test_no_sources_and_no_section_clean(self, tmp_path: Path) -> None:
        """An empty `notebooks/` with no `Notebooks` nav section is the state WP-188 ships in.

        MkDocs rejects a nav section with no children, so the section cannot exist before the first notebook does; the
        check must not demand it.
        """
        _, mkdocs_yml = make_docs_site(tmp_path, "nav:\n  - index.md\n")
        (tmp_path / "notebooks").mkdir()

        assert audit.check_every_notebook_source_has_a_nav_entry(mkdocs_yml, tmp_path / "notebooks") == []


@pytest.mark.parametrize(
    ("extensions", "expected"),
    [
        pytest.param("admonition", ["the 'tables' markdown extension is not declared"], id="missing-extension"),
        pytest.param("tables", [], id="declared-extension"),
    ],
)
def test_gfm_table_extension_declared(tmp_path: Path, extensions: str, expected: list[str]) -> None:
    """A ``markdown_extensions`` list without ``tables`` is reported; one carrying ``tables`` is not."""
    mkdocs_yml = tmp_path / "mkdocs.yml"
    write_file(mkdocs_yml, f"markdown_extensions:\n  - {extensions}\n")

    assert audit.check_the_gfm_table_extension_is_declared(mkdocs_yml) == expected


@pytest.mark.parametrize(
    ("repo_url", "homepage", "expected"),
    [
        pytest.param(
            "https://example.com/a",
            "https://example.com/b",
            ["repo_url 'https://example.com/a' does not match pyproject.toml Homepage 'https://example.com/b'"],
            id="mismatch",
        ),
        pytest.param("https://example.com/a", "https://example.com/a", [], id="equal"),
    ],
)
def test_repo_url_matches_declared_homepage(tmp_path: Path, repo_url: str, homepage: str, expected: list[str]) -> None:
    """A ``repo_url`` that disagrees with ``pyproject.toml``'s Homepage is reported; a matching one is not."""
    write_tree(tmp_path, {"mkdocs.yml": f"repo_url: {repo_url}\n", "pyproject.toml": f'Homepage = "{homepage}"\n'})

    assert (
        audit.check_the_repo_url_matches_the_declared_homepage(tmp_path / "mkdocs.yml", tmp_path / "pyproject.toml")
        == expected
    )


class TestRepoName:
    """`repo_name` holds the same fact as `repo_url` in a second spelling, so it drifts."""

    def test_stale_repo_name_flagged(self, tmp_path: Path) -> None:
        """A `repo_name` naming a different slug than `repo_url` fails.

        This is the surface a rename leaves behind: Material renders `repo_name` as the header link's label on every
        page, so the old slug stays visible site-wide while every link still resolves and no build step reports
        anything.
        """
        write_tree(
            tmp_path,
            {
                "mkdocs.yml": "repo_url: https://github.com/Borda/lit-YOLOs\nrepo_name: Borda/lucid-YOLO\n",
                "pyproject.toml": 'Homepage = "https://github.com/Borda/lit-YOLOs"\n',
            },
        )

        violations = audit.check_the_repo_url_matches_the_declared_homepage(
            tmp_path / "mkdocs.yml", tmp_path / "pyproject.toml"
        )

        assert violations == ["repo_name 'Borda/lucid-YOLO' does not name the repo_url slug 'Borda/lit-YOLOs'"]

    def test_matching_repo_name_passes(self, tmp_path: Path) -> None:
        """The slug taken from the URL's last two segments is what `repo_name` must equal."""
        write_tree(
            tmp_path,
            {
                "mkdocs.yml": "repo_url: https://github.com/Borda/lucid-YOLO\nrepo_name: Borda/lucid-YOLO\n",
                "pyproject.toml": 'Homepage = "https://github.com/Borda/lucid-YOLO"\n',
            },
        )

        assert (
            audit.check_the_repo_url_matches_the_declared_homepage(tmp_path / "mkdocs.yml", tmp_path / "pyproject.toml")
            == []
        )

    def test_absent_repo_name_not_invented(self, tmp_path: Path) -> None:
        """A config declaring no `repo_name` is not failed for it -- Material derives its own."""
        write_tree(
            tmp_path,
            {
                "mkdocs.yml": "repo_url: https://example.com/a\n",
                "pyproject.toml": 'Homepage = "https://example.com/a"\n',
            },
        )

        assert (
            audit.check_the_repo_url_matches_the_declared_homepage(tmp_path / "mkdocs.yml", tmp_path / "pyproject.toml")
            == []
        )


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        pytest.param(
            "A different tool.",
            ["site_description 'A tool.' does not match pyproject.toml description 'A different tool.'"],
            id="mismatch",
        ),
        pytest.param("A tool.", [], id="equal"),
    ],
)
def test_site_description_matches_distribution_description(
    tmp_path: Path, description: str, expected: list[str]
) -> None:
    """A ``site_description`` that diverges from ``pyproject.toml``'s description is reported; a matching one is not."""
    write_tree(
        tmp_path,
        {"mkdocs.yml": "site_description: A tool.\n", "pyproject.toml": f'description = "{description}"\n'},
    )

    assert (
        audit.check_the_site_description_is_the_distribution_description(
            tmp_path / "mkdocs.yml", tmp_path / "pyproject.toml"
        )
        == expected
    )


@pytest.mark.parametrize(
    ("pin", "expected"),
    [
        pytest.param('    "mkdocs>=1.6",\n', ['mkdocs is not capped below 2.0: "mkdocs>=1.6"'], id="uncapped-pin"),
        pytest.param('    "mkdocs>=1.6,<2",\n', [], id="capped-pin"),
    ],
)
def test_mkdocs_capped_below_unlicensed_major(tmp_path: Path, pin: str, expected: list[str]) -> None:
    """A ``mkdocs`` pin without an upper bound below 2.0 is reported; one capped below 2.0 is not."""
    pyproject = tmp_path / "pyproject.toml"
    write_file(pyproject, pin)

    assert audit.check_mkdocs_is_capped_below_the_unlicensed_major(pyproject) == expected


class TestCheckTheDocsWorkflowAuditsLicencesWhereTheDocsTreeIsInstalled:
    """Tests for ``audit.check_the_docs_workflow_audits_licences_where_the_docs_tree_is_installed``."""

    @pytest.mark.parametrize(
        ("workflow_text", "expected"),
        [
            pytest.param(
                "run: python -m mkdocs build --strict\n", ["the docs environment is never audited"], id="missing-audit"
            ),
            pytest.param(
                "run: python -m mkdocs build --strict\nrun: python scripts/lint/audit_licenses.py\n",
                ["the licence audit does not run before the docs build"],
                id="late-audit",
            ),
            pytest.param(
                "run: python scripts/lint/audit_licenses.py\nrun: python -m mkdocs build --strict\n",
                [],
                id="audit-before-build",
            ),
        ],
    )
    def test_audit_must_run_before_build(self, tmp_path: Path, workflow_text: str, expected: list[str]) -> None:
        """A ``docs.yml`` that never audits or audits after the build is reported; an audit before the build is not."""
        workflow = tmp_path / "docs.yml"
        write_file(workflow, workflow_text)

        assert audit.check_the_docs_workflow_audits_licences_where_the_docs_tree_is_installed(workflow) == expected
