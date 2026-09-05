# SPDX-License-Identifier: Apache-2.0
"""License and attribution hygiene audit (WP-002, split off WP-117).

Guards the legal shielding surface: Apache ``LICENSE``, ``NOTICE`` attribution
to the Redmon-originated YOLO family, the ``README`` non-affiliation
disclaimer, and a per-file SPDX license header across the source tree. Each
check was previously a ``tests/meta/test_license_headers.py`` assertion; this
walks the same four checks as a standalone script so the pre-commit hook can
run them without paying pytest's collection cost, while the slimmed test file
keeps the functional-core coverage on synthetic fixtures.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_license_headers.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPDX_LINE = "# SPDX-License-Identifier: Apache-2.0"

#: The only form ``docs/PROVENANCE.md`` sec. 3.5 admits for naming the method's source: a
#: nominative reference to the paper. WP-161 corrected the README's ``the Ultralytics YOLO26
#: paper`` to this and left the identical drift in ``NOTICE`` for the next row; pinning the
#: fragment in both required sets is what stops either from drifting back. The leading article
#: is load-bearing -- ``the YOLO26 paper`` is not a substring of ``the Ultralytics YOLO26
#: paper``, so requiring it rejects the vendor-bound form without a second, negative check
#: that would also have to exempt R1's own literal title in ``PROVENANCE.md``.
PAPER_PHRASE = "the YOLO26 paper"

DISCLAIMER_FRAGMENTS = (
    "independent, from-scratch PyTorch Lightning implementation",
    "not affiliated with, endorsed by, or derived from Ultralytics",
    "No Ultralytics source code, configurations, or model weights were consulted or used",
    PAPER_PHRASE,
)


def check_license_is_apache2(repo_root: Path) -> list[str]:
    """Violations if ``LICENSE`` is not the Apache License, Version 2.0.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = (root / "LICENSE").write_text("MIT License\\n", encoding="utf-8")
        ...     check_license_is_apache2(root)
        ['LICENSE is not the Apache License, Version 2.0']
    """
    head = (repo_root / "LICENSE").read_text(encoding="utf-8")[:200]
    if "Apache License" in head and "Version 2.0" in head:
        return []
    return ["LICENSE is not the Apache License, Version 2.0"]


def check_notice_attribution(repo_root: Path) -> list[str]:
    """Violations if ``NOTICE`` misses the Redmon/independence attribution.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = (root / "NOTICE").write_text("Nothing relevant here.\\n", encoding="utf-8")
        ...     violations = check_notice_attribution(root)
        ...     len(violations), violations[0]
        (5, 'NOTICE missing: Redmon')
    """
    notice = (repo_root / "NOTICE").read_text(encoding="utf-8")
    required = (
        "Redmon",
        "arXiv:1506.02640",
        "not affiliated with, endorsed by, or derived from Ultralytics",
        "Apache License",
        PAPER_PHRASE,
    )
    return [f"NOTICE missing: {fragment}" for fragment in required if fragment not in notice]


def check_readme_disclaimer(repo_root: Path) -> list[str]:
    """Violations if ``README.md`` drops a fragment of the non-affiliation disclaimer.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = (root / "README.md").write_text("No disclaimer at all.\\n", encoding="utf-8")
        ...     len(check_readme_disclaimer(root))
        4
    """
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    return [
        f"README missing disclaimer fragment: {fragment!r}"
        for fragment in DISCLAIMER_FRAGMENTS
        if fragment not in readme
    ]


def check_source_files_carry_spdx_header(repo_root: Path) -> list[str]:
    """Violations for every ``src/**/*.py`` file missing the leading SPDX line.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     src = root / "src"
        ...     src.mkdir()
        ...     _ = (src / "bad.py").write_text("import os\\n", encoding="utf-8")
        ...     check_source_files_carry_spdx_header(root)
        ["files missing SPDX header: ['src/bad.py']"]
    """
    missing = [
        str(path.relative_to(repo_root))
        for path in sorted((repo_root / "src").rglob("*.py"))
        if not path.read_text(encoding="utf-8").startswith(SPDX_LINE)
    ]
    if not missing:
        return []
    return [f"files missing SPDX header: {missing}"]


def find_violations(repo_root: Path) -> list[str]:
    """Every license/attribution violation found under ``repo_root``.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = (root / "LICENSE").write_text("Apache License\\nVersion 2.0\\n", encoding="utf-8")
        ...     _ = (root / "NOTICE").write_text(
        ...         "Redmon, arXiv:1506.02640, not affiliated with, endorsed by, or derived from "
        ...         "Ultralytics, Apache License, the YOLO26 paper\\n",
        ...         encoding="utf-8",
        ...     )
        ...     _ = (root / "README.md").write_text(
        ...         "independent, from-scratch PyTorch Lightning implementation\\n"
        ...         "not affiliated with, endorsed by, or derived from Ultralytics\\n"
        ...         "No Ultralytics source code, configurations, or model weights were consulted or used\\n"
        ...         "the YOLO26 paper\\n",
        ...         encoding="utf-8",
        ...     )
        ...     find_violations(root)
        []
    """
    violations: list[str] = []
    violations.extend(check_license_is_apache2(repo_root))
    violations.extend(check_notice_attribution(repo_root))
    violations.extend(check_readme_disclaimer(repo_root))
    violations.extend(check_source_files_carry_spdx_header(repo_root))
    return violations


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the audit, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` when any check finds a violation.
    """
    parser = argparse.ArgumentParser(description="Audit LICENSE/NOTICE/README/src for license hygiene.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=f"repository root to scan (default: {REPO_ROOT})",
    )
    args = parser.parse_args(argv)

    violations = find_violations(args.repo_root)
    if violations:
        print(f"license-headers-audit FAILED: {len(violations)} violation(s)")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("license-headers-audit clean: LICENSE, NOTICE, README, and src/ headers all check out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
