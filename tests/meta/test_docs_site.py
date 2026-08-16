# SPDX-License-Identifier: Apache-2.0
"""Meta tests: the MkDocs site publishes the whole docs/ tree (WP-114).

``mkdocs build --strict`` already fails on a page the nav omits — but it only runs where
the docs dependency group is installed, which is neither ``make setup`` nor the offline
gate. These checks read ``mkdocs.yml`` as data instead, so a register that drops out of
the nav, or an identity field that drifts from ``pyproject.toml``, fails in the suite
every contributor runs rather than in a workflow most of them never trigger.
"""

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"


class _NavLoader(yaml.SafeLoader):
    """``SafeLoader`` that tolerates the ``!!python/name:`` tag Material's mermaid fence needs.

    MkDocs itself parses this file with ``yaml.Loader``, which constructs that tag by
    importing the named object. Doing the same here would make a test of the nav depend on
    ``pymdownx`` being installed; keeping the tag as its own suffix string reads the file
    without importing anything from it.
    """


_NavLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:",
    lambda loader, suffix, node: suffix,
)


def _config() -> dict[str, Any]:
    """Parse ``mkdocs.yml`` into a plain dict."""
    return yaml.load(MKDOCS_YML.read_text(encoding="utf-8"), Loader=_NavLoader)


def _nav_targets(node: Any) -> list[str]:
    """Collect every page path a nav tree points at, at any depth."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [target for item in node for target in _nav_targets(item)]
    if isinstance(node, dict):
        return [target for value in node.values() for target in _nav_targets(value)]
    return []


def test_every_docs_page_is_listed_in_the_nav() -> None:
    """No markdown file under ``docs/`` is published without a way to navigate to it.

    A page missing from the nav is still built and still served, reachable only by search
    or by guessing its URL — which for a register nobody knows exists is indistinguishable
    from not publishing it at all. The failure mode is silent by construction: the site
    builds green and one document simply stops being findable.
    """
    on_disk = {str(path.relative_to(DOCS)) for path in DOCS.rglob("*.md")}
    in_nav = set(_nav_targets(_config()["nav"]))

    assert on_disk <= in_nav, f"docs pages absent from the nav: {sorted(on_disk - in_nav)}"


def test_every_nav_entry_points_at_a_file_that_exists() -> None:
    """The other direction: a renamed or deleted page leaves a dead nav entry behind.

    Checked separately from the coverage above because the two fail for opposite reasons
    and a single set comparison would report either as "the nav and the tree disagree",
    which does not say which file to go look at.
    """
    missing = [target for target in _nav_targets(_config()["nav"]) if not (DOCS / target).is_file()]

    assert not missing, f"nav entries with no file on disk: {missing}"


def test_the_gfm_table_extension_is_declared() -> None:
    """``tables`` is listed explicitly, because supplying any list drops the defaults.

    This repository is wall-to-wall pipe tables — the assumption register, the roadmap,
    every acceptance table. Without the extension they render as literal pipe characters
    on a build that reports success, so nothing but this assertion notices.
    """
    extensions = _config()["markdown_extensions"]
    names = {item if isinstance(item, str) else next(iter(item)) for item in extensions}

    assert "tables" in names


def test_the_repo_url_matches_the_declared_homepage() -> None:
    """``repo_url`` and ``pyproject.toml``'s Homepage name the same repository.

    Two places holding the same slug is exactly the shape that already went stale once:
    the project was renamed, GitHub kept redirecting the HTML URLs, and nothing failed
    until a raw URL — which does not redirect — was generated from the stale value.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    homepage = re.search(r'^Homepage\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE)

    assert homepage, "pyproject.toml declares no Homepage"
    assert _config()["repo_url"].rstrip("/") == homepage.group(1).rstrip("/")


def test_the_site_description_is_the_distribution_description() -> None:
    """``site_description`` restates ``pyproject.toml``'s description verbatim.

    It is the sentence a search engine shows under the site's title and the sentence PyPI
    shows under the package's, and the two describing the same project differently is the
    kind of drift no build step and no reader ever reports.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    described = re.search(r'^description\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE)

    assert described, "pyproject.toml declares no description"
    assert _config()["site_description"] == described.group(1)


def test_mkdocs_is_capped_below_the_unlicensed_major() -> None:
    """The ``docs`` group pins ``mkdocs<2``, and the cap is a licence bound.

    MkDocs 2.0 is described by the Material team's own build-time notice as "Currently
    unlicensed", and unlicensed is stricter than the AGPL this project bans — the default
    is no grant at all. Asserted here because the licence audit cannot see it: it matches
    a GPL-family pattern against a declared licence, so a distribution declaring nothing
    matches nothing and passes. Lifting the cap is a decision, not a dependency bump.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # The operator class is what keeps this off `mkdocs-material`, which shares the prefix
    # and sits two lines away; without it the assertion would silently move to that pin.
    pin = re.search(r'^\s*"mkdocs([<>=!~][^"]*)"', pyproject, flags=re.MULTILINE)

    assert pin, "the docs group declares no mkdocs pin"
    assert "<2" in pin.group(1), f"mkdocs is not capped below 2.0: {pin.group(0)}"


def test_the_docs_workflow_audits_licences_where_the_docs_tree_is_installed() -> None:
    """``docs.yml`` runs the licence audit, and runs it before the build.

    The docs dependency group is in no other workflow's environment, so this job is the
    only place CI ever sees the tree ``mkdocs-material`` pulls. An audit that scans the
    installed environment rather than the diff is only a control where the environment
    exists — and after the build it would report on a job that already produced its
    artifact.
    """
    workflow = (REPO_ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")

    build = "python -m mkdocs build --strict"

    assert "scripts/audit_licenses.py" in workflow, "the docs environment is never audited"
    assert workflow.index("scripts/audit_licenses.py") < workflow.index(build)
