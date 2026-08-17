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


def test_notice_attribution_accepts_a_complete_notice(tmp_path: Path) -> None:
    """A NOTICE carrying every required fragment passes without a violation."""
    (tmp_path / "NOTICE").write_text(
        "Redmon, arXiv:1506.02640, not affiliated with, endorsed by, or derived from Ultralytics, Apache License\n",
        encoding="utf-8",
    )
    assert audit.check_notice_attribution(tmp_path) == []


def test_notice_attribution_names_each_missing_fragment(tmp_path: Path) -> None:
    """Every missing NOTICE fragment is reported, one violation per fragment."""
    (tmp_path / "NOTICE").write_text("Nothing relevant here.\n", encoding="utf-8")
    violations = audit.check_notice_attribution(tmp_path)
    assert violations == [
        "NOTICE missing: Redmon",
        "NOTICE missing: arXiv:1506.02640",
        "NOTICE missing: not affiliated with, endorsed by, or derived from Ultralytics",
        "NOTICE missing: Apache License",
    ]


def test_readme_disclaimer_accepts_all_fragments_present(tmp_path: Path) -> None:
    """A README carrying every disclaimer fragment passes without a violation."""
    (tmp_path / "README.md").write_text(
        "\n".join(audit.DISCLAIMER_FRAGMENTS) + "\n",
        encoding="utf-8",
    )
    assert audit.check_readme_disclaimer(tmp_path) == []


def test_readme_disclaimer_names_each_missing_fragment(tmp_path: Path) -> None:
    """Every missing README disclaimer fragment is reported, one violation per fragment."""
    (tmp_path / "README.md").write_text("No disclaimer at all.\n", encoding="utf-8")
    violations = audit.check_readme_disclaimer(tmp_path)
    assert len(violations) == len(audit.DISCLAIMER_FRAGMENTS)


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


def test_the_live_repo_tree_is_currently_clean() -> None:
    """The real repo tree the pre-commit hook scans has no license/attribution violations.

    This is the one place this file touches the live tree rather than a synthetic
    one: a regression here means the hook itself would fail on the next commit
    that touches a guarded file, not just a new one.
    """
    assert audit.find_violations(REPO_ROOT) == []
