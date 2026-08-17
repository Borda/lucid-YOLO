# SPDX-License-Identifier: Apache-2.0
"""Docs-site audit: the MkDocs Material site publishes the whole ``docs/`` tree (WP-114).

``mkdocs build --strict`` already fails on a page the nav omits -- but it only runs where
the docs dependency group is installed, which is neither ``make setup`` nor the offline
gate. This reads ``mkdocs.yml`` as data instead, so a register that drops out of the nav,
or an identity field that drifts from ``pyproject.toml``, fails in the suite every
contributor runs rather than in a workflow most of them never trigger.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_docs_site.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, cast

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEFAULT_DOCS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docs.yml"


class _NavLoader(yaml.SafeLoader):
    """``SafeLoader`` that tolerates the ``!!python/name:`` tag Material's mermaid fence needs.

    MkDocs itself parses this file with ``yaml.Loader``, which constructs that tag by
    importing the named object. Doing the same here would make this audit depend on
    ``pymdownx`` being installed; keeping the tag as its own suffix string reads the file
    without importing anything from it.
    """


_NavLoader.add_multi_constructor(  # type: ignore[no-untyped-call]  # yaml-stubs leaves this untyped
    "tag:yaml.org,2002:python/name:",
    lambda loader, suffix, node: suffix,
)


def _config(mkdocs_yml: Path) -> dict[str, Any]:
    """Parse ``mkdocs_yml`` into a plain dict.

    Examples:
        >>> "nav" in _config(DEFAULT_MKDOCS_YML)
        True
    """
    return cast("dict[str, Any]", yaml.load(mkdocs_yml.read_text(encoding="utf-8"), Loader=_NavLoader))


def _nav_targets(node: Any) -> list[str]:
    """Collect every page path a nav tree points at, at any depth.

    Examples:
        >>> _nav_targets(["a.md", {"Section": ["b.md", "c.md"]}])
        ['a.md', 'b.md', 'c.md']
    """
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [target for item in node for target in _nav_targets(item)]
    if isinstance(node, dict):
        return [target for value in node.values() for target in _nav_targets(value)]
    return []


def check_every_docs_page_is_listed_in_the_nav(docs_dir: Path, mkdocs_yml: Path) -> list[str]:
    """Violation for every markdown file under ``docs_dir`` absent from the nav.

    A page missing from the nav is still built and still served, reachable only by search
    or by guessing its URL -- which for a register nobody knows exists is indistinguishable
    from not publishing it at all. The failure mode is silent by construction: the site
    builds green and one document simply stops being findable.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     docs = root / "docs"
        ...     docs.mkdir()
        ...     _ = (docs / "orphan.md").write_text("x", encoding="utf-8")
        ...     mkdocs_yml = root / "mkdocs.yml"
        ...     _ = mkdocs_yml.write_text("nav:\\n  - index.md\\n", encoding="utf-8")
        ...     check_every_docs_page_is_listed_in_the_nav(docs, mkdocs_yml)
        ["docs pages absent from the nav: ['orphan.md']"]
    """
    on_disk = {str(path.relative_to(docs_dir)) for path in docs_dir.rglob("*.md")}
    in_nav = set(_nav_targets(_config(mkdocs_yml)["nav"]))
    missing = on_disk - in_nav
    if missing:
        return [f"docs pages absent from the nav: {sorted(missing)}"]
    return []


def check_every_nav_entry_points_at_a_file_that_exists(docs_dir: Path, mkdocs_yml: Path) -> list[str]:
    """Violation for every nav entry naming a file that does not exist under ``docs_dir``.

    Checked separately from the coverage above because the two fail for opposite reasons
    and a single set comparison would report either as "the nav and the tree disagree",
    which does not say which file to go look at.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     docs = root / "docs"
        ...     docs.mkdir()
        ...     mkdocs_yml = root / "mkdocs.yml"
        ...     _ = mkdocs_yml.write_text("nav:\\n  - ghost.md\\n", encoding="utf-8")
        ...     check_every_nav_entry_points_at_a_file_that_exists(docs, mkdocs_yml)
        ["nav entries with no file on disk: ['ghost.md']"]
    """
    missing = [target for target in _nav_targets(_config(mkdocs_yml)["nav"]) if not (docs_dir / target).is_file()]
    if missing:
        return [f"nav entries with no file on disk: {missing}"]
    return []


def check_the_gfm_table_extension_is_declared(mkdocs_yml: Path) -> list[str]:
    """Violation if ``tables`` is absent from ``markdown_extensions``.

    Supplying any extension list drops Python-Markdown's defaults, and this repository is
    wall-to-wall pipe tables -- the assumption register, the roadmap, every acceptance
    table. Without the extension they render as literal pipe characters on a build that
    reports success, so nothing but this check notices.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     mkdocs_yml = Path(tmp) / "mkdocs.yml"
        ...     _ = mkdocs_yml.write_text("markdown_extensions:\\n  - admonition\\n", encoding="utf-8")
        ...     check_the_gfm_table_extension_is_declared(mkdocs_yml)
        ["the 'tables' markdown extension is not declared"]
    """
    extensions = _config(mkdocs_yml)["markdown_extensions"]
    names = {item if isinstance(item, str) else next(iter(item)) for item in extensions}
    if "tables" not in names:
        return ["the 'tables' markdown extension is not declared"]
    return []


def check_the_repo_url_matches_the_declared_homepage(mkdocs_yml: Path, pyproject: Path) -> list[str]:
    """Violation if ``repo_url`` and ``pyproject.toml``'s Homepage name different repositories.

    Two places holding the same slug is exactly the shape that already went stale once:
    the project was renamed, GitHub kept redirecting the HTML URLs, and nothing failed
    until a raw URL -- which does not redirect -- was generated from the stale value.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     mkdocs_yml = root / "mkdocs.yml"
        ...     _ = mkdocs_yml.write_text("repo_url: https://example.com/a\\n", encoding="utf-8")
        ...     pyproject = root / "pyproject.toml"
        ...     _ = pyproject.write_text('Homepage = "https://example.com/b"\\n', encoding="utf-8")
        ...     check_the_repo_url_matches_the_declared_homepage(mkdocs_yml, pyproject)
        ["repo_url 'https://example.com/a' does not match pyproject.toml Homepage 'https://example.com/b'"]
    """
    text = pyproject.read_text(encoding="utf-8")
    homepage = re.search(r'^Homepage\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not homepage:
        return ["pyproject.toml declares no Homepage"]
    repo_url = _config(mkdocs_yml)["repo_url"]
    if repo_url.rstrip("/") != homepage.group(1).rstrip("/"):
        return [f"repo_url {repo_url!r} does not match pyproject.toml Homepage {homepage.group(1)!r}"]
    return []


def check_the_site_description_is_the_distribution_description(mkdocs_yml: Path, pyproject: Path) -> list[str]:
    """Violation if ``site_description`` does not restate ``pyproject.toml``'s description verbatim.

    It is the sentence a search engine shows under the site's title and the sentence PyPI
    shows under the package's, and the two describing the same project differently is the
    kind of drift no build step and no reader ever reports.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     mkdocs_yml = root / "mkdocs.yml"
        ...     _ = mkdocs_yml.write_text("site_description: A tool.\\n", encoding="utf-8")
        ...     pyproject = root / "pyproject.toml"
        ...     _ = pyproject.write_text('description = "A different tool."\\n', encoding="utf-8")
        ...     check_the_site_description_is_the_distribution_description(mkdocs_yml, pyproject)
        ["site_description 'A tool.' does not match pyproject.toml description 'A different tool.'"]
    """
    text = pyproject.read_text(encoding="utf-8")
    described = re.search(r'^description\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not described:
        return ["pyproject.toml declares no description"]
    site_description = _config(mkdocs_yml)["site_description"]
    if site_description != described.group(1):
        return [
            f"site_description {site_description!r} does not match pyproject.toml description {described.group(1)!r}"
        ]
    return []


def check_mkdocs_is_capped_below_the_unlicensed_major(pyproject: Path) -> list[str]:
    """Violation if the ``docs`` group's ``mkdocs`` pin is not capped below ``2``.

    MkDocs 2.0 is described by the Material team's own build-time notice as "Currently
    unlicensed", and unlicensed is stricter than the AGPL this project bans -- the default
    is no grant at all. Checked here because the licence audit cannot see it: it matches a
    GPL-family pattern against a declared licence, so a distribution declaring nothing
    matches nothing and passes. Lifting the cap is a decision, not a dependency bump.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     pyproject = Path(tmp) / "pyproject.toml"
        ...     _ = pyproject.write_text('    "mkdocs>=1.6",\\n', encoding="utf-8")
        ...     check_mkdocs_is_capped_below_the_unlicensed_major(pyproject)
        ['mkdocs is not capped below 2.0: "mkdocs>=1.6"']
    """
    text = pyproject.read_text(encoding="utf-8")
    # The operator class is what keeps this off `mkdocs-material`, which shares the prefix
    # and sits two lines away; without it the assertion would silently move to that pin.
    pin = re.search(r'^\s*"mkdocs([<>=!~][^"]*)"', text, flags=re.MULTILINE)
    if not pin:
        return ["the docs group declares no mkdocs pin"]
    if "<2" not in pin.group(1):
        return [f"mkdocs is not capped below 2.0: {pin.group(0).strip()}"]
    return []


def check_the_docs_workflow_audits_licences_where_the_docs_tree_is_installed(docs_workflow: Path) -> list[str]:
    """Violation if ``docs.yml`` does not run the licence audit before the site build.

    The docs dependency group is in no other workflow's environment, so this job is the
    only place CI ever sees the tree ``mkdocs-material`` pulls. An audit that scans the
    installed environment rather than the diff is only a control where the environment
    exists -- and after the build it would report on a job that already produced its
    artifact.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     workflow = Path(tmp) / "docs.yml"
        ...     _ = workflow.write_text("run: python -m mkdocs build --strict\\n", encoding="utf-8")
        ...     check_the_docs_workflow_audits_licences_where_the_docs_tree_is_installed(workflow)
        ['the docs environment is never audited']
    """
    workflow = docs_workflow.read_text(encoding="utf-8")
    audit = "scripts/lint/audit_licenses.py"
    build = "python -m mkdocs build --strict"
    if audit not in workflow:
        return ["the docs environment is never audited"]
    if workflow.index(audit) >= workflow.index(build):
        return ["the licence audit does not run before the docs build"]
    return []


def find_violations(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    mkdocs_yml: Path = DEFAULT_MKDOCS_YML,
    pyproject: Path = DEFAULT_PYPROJECT,
    docs_workflow: Path = DEFAULT_DOCS_WORKFLOW,
) -> list[str]:
    """Every docs-site violation across the nav, identity fields, and CI wiring.

    Examples:
        >>> find_violations(DEFAULT_DOCS_DIR, DEFAULT_MKDOCS_YML, DEFAULT_PYPROJECT, DEFAULT_DOCS_WORKFLOW)
        []
    """
    violations: list[str] = []
    violations += check_every_docs_page_is_listed_in_the_nav(docs_dir, mkdocs_yml)
    violations += check_every_nav_entry_points_at_a_file_that_exists(docs_dir, mkdocs_yml)
    violations += check_the_gfm_table_extension_is_declared(mkdocs_yml)
    violations += check_the_repo_url_matches_the_declared_homepage(mkdocs_yml, pyproject)
    violations += check_the_site_description_is_the_distribution_description(mkdocs_yml, pyproject)
    violations += check_mkdocs_is_capped_below_the_unlicensed_major(pyproject)
    violations += check_the_docs_workflow_audits_licences_where_the_docs_tree_is_installed(docs_workflow)
    return violations


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the audit, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` when any check fails.
    """
    parser = argparse.ArgumentParser(description="Audit the MkDocs site config against docs/ and pyproject.toml.")
    parser.add_argument(
        "--docs-dir", type=Path, default=DEFAULT_DOCS_DIR, help=f"docs tree (default: {DEFAULT_DOCS_DIR})"
    )
    parser.add_argument(
        "--mkdocs-yml", type=Path, default=DEFAULT_MKDOCS_YML, help=f"mkdocs config (default: {DEFAULT_MKDOCS_YML})"
    )
    parser.add_argument(
        "--pyproject", type=Path, default=DEFAULT_PYPROJECT, help=f"pyproject.toml (default: {DEFAULT_PYPROJECT})"
    )
    parser.add_argument(
        "--docs-workflow",
        type=Path,
        default=DEFAULT_DOCS_WORKFLOW,
        help=f"docs CI workflow (default: {DEFAULT_DOCS_WORKFLOW})",
    )
    args = parser.parse_args(argv)

    violations = find_violations(args.docs_dir, args.mkdocs_yml, args.pyproject, args.docs_workflow)
    if violations:
        print(f"docs-site-audit FAILED: {len(violations)} violation(s)")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("docs-site-audit clean: nav, identity fields, and CI wiring all match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
