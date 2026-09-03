# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the release guard refuses unshippable tags (WP-006).

Covers the DoD negative cases — a tag on a red gate, a tag missing its changelog
section, and a 1.x tag (ADR-002) are each refused — plus the happy path and a
malformed tag string. The guard is loaded via the spec-loading pattern used in
``scripts/_tests/test_audit_licenses.py`` so ``scripts/`` need not be importable.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "scripts" / "release_guard.py"


def _load_guard() -> ModuleType:
    """Load ``scripts/release_guard.py`` as an importable module.

    The module is registered in ``sys.modules`` before execution so its frozen
    dataclass resolves its own module under ``from __future__ import annotations``.

    Examples:
        >>> callable(_load_guard().main)
        True
    """
    spec = importlib.util.spec_from_file_location("release_guard", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def _write_changelog(tmp_path: Path, version: str) -> Path:
    """Write a minimal changelog carrying a ``## [<version>]`` section.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = _write_changelog(Path(tmp), "0.1.0")
        ...     "## [0.1.0]" in path.read_text()
        True
    """
    path = tmp_path / "CHANGELOG.md"
    path.write_text(f"# Changelog\n\n## [{version}]\n\n### Added\n\n- thing\n", encoding="utf-8")
    return path


def test_red_gate_refuses_tag(tmp_path: Path) -> None:
    """A valid tag with its changelog section is refused when the gate is red."""
    changelog = _write_changelog(tmp_path, "0.1.0")
    assert guard.main(["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", "false"]) == 1


def test_missing_changelog_section_refuses_tag(tmp_path: Path) -> None:
    """A valid tag on a green gate is refused when its changelog section is absent."""
    changelog = _write_changelog(tmp_path, "0.2.0")  # section for a different version
    assert guard.main(["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", "true"]) == 1


def test_one_x_tag_refused_by_adr_002(tmp_path: Path) -> None:
    """A 1.x tag is refused even with a green gate and a matching changelog section."""
    changelog = _write_changelog(tmp_path, "1.0.0")
    assert guard.main(["--tag", "v1.0.0", "--changelog", str(changelog), "--gate-cmd", "true"]) == 1


def test_happy_path_passes(tmp_path: Path) -> None:
    """A v0.MINOR.PATCH tag with its changelog section and a green gate ships."""
    changelog = _write_changelog(tmp_path, "0.1.0")
    assert guard.main(["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", "true"]) == 0


def test_malformed_tag_refused(tmp_path: Path) -> None:
    """A tag that is not a v<major>.<minor>.<patch> string is refused."""
    changelog = _write_changelog(tmp_path, "0.1")
    assert guard.main(["--tag", "v0.1", "--changelog", str(changelog), "--gate-cmd", "true"]) == 1


def test_adr_002_refusal_names_the_policy() -> None:
    """The 1.x refusal detail cites ADR-002's perpetual 0.x train, hard-coded."""
    result = guard.check_tag("v1.4.2")
    assert not result.passed
    assert "ADR-002" in result.detail


def test_zero_major_tag_check_passes() -> None:
    """The tag check accepts a plain zero-major tag on its own."""
    assert guard.check_tag("v0.3.0").passed


def test_main_without_tag_falls_back_to_head(monkeypatch, tmp_path: Path) -> None:
    """Omitting --tag resolves it from the current HEAD, exactly like passing it."""
    changelog = _write_changelog(tmp_path, "0.1.0")
    monkeypatch.setattr(guard, "_current_tag", lambda: "v0.1.0")
    assert guard.main(["--changelog", str(changelog), "--gate-cmd", "true"]) == 0


def test_main_without_tag_or_release_passes_with_nothing_to_check(monkeypatch) -> None:
    """HEAD not being exactly a tag is not a refusal -- most commits are not releases."""
    monkeypatch.setattr(guard, "_current_tag", lambda: None)
    assert guard.main([]) == 0


def _write_package(tmp_path: Path, source: str) -> Path:
    """Write a one-module package whose imports the tier check will walk.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = _write_package(Path(tmp), "import os\\n")
        ...     root.name
        'shipped'
    """
    root = tmp_path / "shipped"
    root.mkdir()
    (root / "module.py").write_text(source, encoding="utf-8")
    return root


def _write_manifest(tmp_path: Path, runtime: str, dev: str = "") -> Path:
    """Write a manifest declaring the two dependency tiers the check reads.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = _write_manifest(Path(tmp), '"torch"')
        ...     "[project]" in path.read_text()
        True
    """
    path = tmp_path / "pyproject.toml"
    path.write_text(
        f"[project]\ndependencies = [{runtime}]\n\n[dependency-groups]\ndev = [{dev}]\n",
        encoding="utf-8",
    )
    return path


def test_group_only_dependency_is_refused(tmp_path: Path) -> None:
    """A module whose distribution is declared only in a group refuses the tag.

    This is WP-159's own defect, stated as a test: `fuse-augmentations` sat in the
    `dev` group while five shipped modules imported it, so an install omitting that
    group raised `ImportError` from `lucid_yolo.data`. The development environment
    installs every group, which is exactly why the suite could not see it.
    """
    manifest = _write_manifest(tmp_path, '"torch"', '"helper-lib"')
    root = _write_package(tmp_path, "import helper\n")

    result = guard.check_dependency_tiers(manifest, root, {"helper": ["helper-lib"]})

    assert not result.passed
    assert "helper" in result.detail
    assert "'dev'" in result.detail, "the refusal must name the group, so it diagnoses rather than complains"


def test_runtime_declared_dependency_passes(tmp_path: Path) -> None:
    """The same import passes once its distribution is declared at runtime."""
    manifest = _write_manifest(tmp_path, '"helper-lib"', '"pytest"')
    root = _write_package(tmp_path, "import helper\n")

    assert guard.check_dependency_tiers(manifest, root, {"helper": ["helper-lib"]}).passed


def test_import_with_no_installed_distribution_is_refused(tmp_path: Path) -> None:
    """An import no distribution provides is reported, never silently accepted.

    A guard that reads "unknown" as "fine" is the failure mode this check closes.
    """
    manifest = _write_manifest(tmp_path, '"torch"')
    root = _write_package(tmp_path, "import mystery\n")

    result = guard.check_dependency_tiers(manifest, root, {})

    assert not result.passed
    assert "mystery" in result.detail


def test_stdlib_and_relative_imports_need_no_declaration(tmp_path: Path) -> None:
    """Neither the standard library nor an intra-package import is a dependency."""
    manifest = _write_manifest(tmp_path, "")
    root = _write_package(tmp_path, "import json\nfrom pathlib import Path\nfrom . import sibling\n")

    assert guard.check_dependency_tiers(manifest, root, {}).passed


def test_conditional_import_is_still_an_import(tmp_path: Path) -> None:
    """An import inside a function or a branch is one the installed package can run.

    Walking the module preamble alone would miss it, and a lazily imported
    dependency is no less required than an eagerly imported one -- it merely fails
    later, on the call rather than on the import.
    """
    manifest = _write_manifest(tmp_path, '"torch"', '"helper-lib"')
    root = _write_package(tmp_path, "def draw():\n    import helper\n    return helper\n")

    assert not guard.check_dependency_tiers(manifest, root, {"helper": ["helper-lib"]}).passed


def test_cli_refuses_a_tag_whose_package_imports_an_undeclared_module(tmp_path: Path) -> None:
    """The check is wired into the CLI, not merely importable from it."""
    changelog = _write_changelog(tmp_path, "0.1.0")
    manifest = _write_manifest(tmp_path, "")
    root = _write_package(tmp_path, "import torch\n")

    status = guard.main(
        [
            "--tag",
            "v0.1.0",
            "--changelog",
            str(changelog),
            "--gate-cmd",
            "true",
            "--pyproject",
            str(manifest),
            "--package-root",
            str(root),
        ]
    )

    assert status == 1
