# SPDX-License-Identifier: Apache-2.0
"""Meta tests: license and attribution hygiene (WP-002).

Guards the legal shielding surface: Apache LICENSE, NOTICE attribution to the
Redmon-originated YOLO family, the README non-affiliation disclaimer, and a
per-file SPDX license header across the source tree.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPDX_LINE = "# SPDX-License-Identifier: Apache-2.0"

DISCLAIMER_FRAGMENTS = (
    "independent, from-scratch PyTorch Lightning implementation",
    "not affiliated with, endorsed by, or derived from Ultralytics",
    "No Ultralytics source code, configurations, or model weights were consulted or used",
)


def test_license_is_apache2() -> None:
    """LICENSE file is the Apache License, Version 2.0."""
    head = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")[:200]
    assert "Apache License" in head
    assert "Version 2.0" in head


def test_notice_attribution() -> None:
    """NOTICE attributes the YOLO family to Redmon and coauthors, and states independence."""
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Redmon" in notice
    assert "arXiv:1506.02640" in notice
    assert "not affiliated with, endorsed by, or derived from Ultralytics" in notice
    assert "Apache License" in notice


def test_readme_disclaimer_verbatim() -> None:
    """README carries every fragment of the blueprint non-affiliation header."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for fragment in DISCLAIMER_FRAGMENTS:
        assert fragment in readme, f"README missing disclaimer fragment: {fragment!r}"


def test_all_source_files_carry_spdx_header() -> None:
    """Every Python file under src/ starts with the SPDX Apache-2.0 line."""
    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "src").rglob("*.py"))
        if not path.read_text(encoding="utf-8").startswith(SPDX_LINE)
    ]
    assert not missing, f"files missing SPDX header: {missing}"
