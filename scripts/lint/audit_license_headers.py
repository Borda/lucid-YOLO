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

#: The Apache appendix's own placeholder for the copyright line, left in the file by
#: anyone who copies the template and stops at the terms. It sits ~5 kB in, past the
#: window a "is this Apache 2.0" head-read looks at, so the check that reads the head
#: structurally cannot see it -- which is the whole reason it survived here.
COPYRIGHT_PLACEHOLDER = "Copyright [yyyy] [name of copyright owner]"

#: Trees whose ``*.py`` files must open with the SPDX line. ``src/`` alone was never a
#: stated scope, only the one the glob happened to name; ``scripts/`` and ``tests/``
#: already carry the header on every file, so widening the glob pins what is already
#: true rather than asking for new work.
HEADER_DIRS = ("src", "scripts", "tests")

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
    """Violations if ``LICENSE`` is not Apache 2.0, or still carries the template placeholder.

    Two checks over one file because they read the same document for opposite
    reasons: the first asks whether this is the licence claimed everywhere else, and
    reads the head, where the title is. The second asks whether the licence was
    *applied* or merely copied, and has to read the whole file -- the appendix's
    ``Copyright [yyyy] [name of copyright owner]`` sits far outside any head window,
    which is how it survived every run of the first check.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = (root / "LICENSE").write_text("MIT License\\n", encoding="utf-8")
        ...     check_license_is_apache2(root)
        ['LICENSE is not the Apache License, Version 2.0']
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = (root / "LICENSE").write_text(
        ...         "Apache License\\nVersion 2.0\\n" + " " * 400 + COPYRIGHT_PLACEHOLDER + "\\n",
        ...         encoding="utf-8",
        ...     )
        ...     check_license_is_apache2(root)
        ['LICENSE still carries the Apache appendix placeholder: Copyright [yyyy] [name of copyright owner]']
    """
    text = (repo_root / "LICENSE").read_text(encoding="utf-8")
    violations = []
    head = text[:200]
    if "Apache License" not in head or "Version 2.0" not in head:
        violations.append("LICENSE is not the Apache License, Version 2.0")
    if COPYRIGHT_PLACEHOLDER in text:
        violations.append(f"LICENSE still carries the Apache appendix placeholder: {COPYRIGHT_PLACEHOLDER}")
    return violations


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


def _carries_spdx_header(text: str) -> bool:
    """True if ``text`` opens with the SPDX line, a shebang permitted ahead of it.

    A shebang has to be the first line to work at all, so a runnable script cannot
    put the SPDX line first and remain runnable. Reading only line one calls the
    repository's two executable scripts unlicensed while they carry the header on
    line two, which is what the header-scope finding read as two missing headers.

    Examples:
        >>> _carries_spdx_header(SPDX_LINE + "\\nimport os\\n")
        True
        >>> _carries_spdx_header("#!/usr/bin/env python\\n" + SPDX_LINE + "\\n")
        True
        >>> _carries_spdx_header("import os\\n")
        False
    """
    lines = text.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    return bool(lines) and lines[0].startswith(SPDX_LINE)


def check_source_files_carry_spdx_header(repo_root: Path) -> list[str]:
    """Violations for every ``*.py`` file under :data:`HEADER_DIRS` missing the SPDX line.

    Scope is the three trees this repository actually writes Python into, not ``src/``
    alone: the header is a per-file legal marker and a file that travels -- a script
    pasted into a notebook, a test vendored into a bug report -- carries it or does
    not, whether or not it ships in the wheel.

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
        for name in HEADER_DIRS
        for path in sorted((repo_root / name).rglob("*.py"))
        if not _carries_spdx_header(path.read_text(encoding="utf-8"))
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
    parser = argparse.ArgumentParser(description="Audit LICENSE/NOTICE/README and per-file SPDX headers.")
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
    print(f"license-headers-audit clean: LICENSE, NOTICE, README, and {'/, '.join(HEADER_DIRS)}/ headers all check out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
