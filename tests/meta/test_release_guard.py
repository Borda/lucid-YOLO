# SPDX-License-Identifier: Apache-2.0
"""Meta tests: the release guard refuses unshippable tags (WP-006).

Covers the DoD negative cases — a tag on a red gate, a tag missing its changelog
section, and a 1.x tag (ADR-002) are each refused — plus the happy path and a
malformed tag string. The guard is loaded via the spec-loading pattern used in
``tests/meta/test_license_audit.py`` so ``scripts/`` need not be importable.
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
    """
    spec = importlib.util.spec_from_file_location("release_guard", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def _write_changelog(tmp_path: Path, version: str) -> Path:
    """Write a minimal changelog carrying a ``## [<version>]`` section."""
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
