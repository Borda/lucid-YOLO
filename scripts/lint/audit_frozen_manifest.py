# SPDX-License-Identifier: Apache-2.0
"""Structural audit of the frozen-golden digest manifest (WP-168).

``scripts/check_goldens.py`` recomputes each golden's producer and compares the result
against **that same file's own** stored ``values`` within its own stored ``tolerances``.
Frozen files travel that identical path, which is the design: a release's snapshot
staying green is the statement that current code still satisfies every value the release
pinned. What no check ever asserted is that the snapshot is still the file the release
wrote. Edit a frozen golden's ``values`` and its ``tolerances`` together to match new
code and the whole gate is green, because the file is being compared with itself —
and "never modify a frozen golden" (``AGENTS.md`` §7, escalation trigger 4 in §4) was
prose only. Two principal overrides have been granted against that rule, so the decision
path existed while the mechanism did not.

This is the mechanism: ``goldens/frozen/MANIFEST.sha256`` records a SHA-256 digest per
frozen file, and this audit reports three ways the tree and the manifest can disagree —
a file whose bytes changed, a frozen file no row covers, and a row whose file is gone.
:data:`~freeze_goldens.MANIFEST_NAME` is excluded from its own digest set.

**What this narrows rather than closes.** The manifest is committed beside the files it
covers, so a commit that edits a frozen golden *and* re-runs ``freeze_goldens.py
--reseal`` still passes — the audit catches an inconsistent edit, not a consistent one.
That is the intended bound: an unsanctioned edit is an edit nobody meant to declare, and
this makes declaring it a separate, named, reviewable act (`--reseal`) instead of
something a values-and-tolerances edit does silently. Signing the manifest is what would
close it, and nothing in this repository has a key to sign with.

``scripts/freeze_goldens.py`` is the manifest's only writer: ``make freeze-goldens
MINOR=0.N`` seals the files it copies, and ``freeze_goldens.py --reseal`` recomputes
every row for a principal-approved move.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_frozen_manifest.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: Repository root (``scripts/lint/`` is two levels below it).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default frozen-golden root audited when no ``--frozen-root`` is given.
DEFAULT_FROZEN_ROOT = REPO_ROOT / "goldens" / "frozen"


def _load_writer() -> ModuleType:
    """Load ``scripts/freeze_goldens.py``, the manifest's only writer, as a module.

    Loaded by path rather than imported: ``scripts/`` is not a package. The reader and
    the format live with the writer on purpose — a checker carrying its own copy of the
    manifest format is a checker that can drift out of agreement with what writes it.

    Examples:
        >>> _load_writer().MANIFEST_NAME
        'MANIFEST.sha256'
    """
    path = REPO_ROOT / "scripts" / "freeze_goldens.py"
    spec = importlib.util.spec_from_file_location("freeze_goldens", path)
    if spec is None or spec.loader is None:  # pragma: no cover - a missing writer is an unrunnable repo
        raise RuntimeError(f"cannot load the manifest writer at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_violations(frozen_root: Path = DEFAULT_FROZEN_ROOT) -> list[str]:
    """Report every disagreement between the frozen tree and its manifest.

    Args:
        frozen_root: Directory holding the per-minor snapshots and the manifest
            (default: ``<repo>/goldens/frozen``).

    Returns:
        One string per violation: a missing manifest, a file whose digest changed, a
        frozen file no row covers, or a row whose file is gone. Empty when the manifest
        describes the tree exactly.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     (root / "0.1").mkdir()
        ...     _ = (root / "0.1" / "optim_toy.json").write_bytes(b'{"values": {}}')
        ...     violations = find_violations(root)
        ...     len(violations), violations[0].endswith("nothing pins the frozen goldens")
        (1, True)
    """
    writer = _load_writer()
    if not (frozen_root / writer.MANIFEST_NAME).is_file():
        return [f"no {writer.MANIFEST_NAME} at {frozen_root}: nothing pins the frozen goldens"]
    recorded = writer.read_manifest(frozen_root)
    present = {
        path.relative_to(frozen_root).as_posix(): writer.digest(path) for path in writer.frozen_files(frozen_root)
    }

    violations = [
        f"{name}: digest {present[name]} does not match the manifest's {recorded[name]}"
        for name in sorted(present.keys() & recorded.keys())
        if present[name] != recorded[name]
    ]
    violations += [
        f"{name}: frozen file covered by no manifest row" for name in sorted(present.keys() - recorded.keys())
    ]
    violations += [f"{name}: manifest row whose file is gone" for name in sorted(recorded.keys() - present.keys())]
    return violations


def main(argv: list[str] | None = None) -> int:
    """Audit the frozen-golden manifest; print a verdict and return the exit code.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when the manifest describes the frozen tree exactly, ``1`` otherwise.

    Examples:
        The verdict line names an absolute path, so the example captures stdout and
        asserts the exit code alone.

        >>> import contextlib, io
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     status = main([])
        >>> status
        0
    """
    parser = argparse.ArgumentParser(description="Check every frozen golden against its recorded digest.")
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=DEFAULT_FROZEN_ROOT,
        help="directory holding the per-minor snapshots and the manifest (default: <repo>/goldens/frozen)",
    )
    args = parser.parse_args(argv)

    violations = find_violations(args.frozen_root)
    if violations:
        print("FROZEN GOLDEN MANIFEST FAILED — the frozen tree is not what the manifest records:")
        for violation in violations:
            print(f"  {violation}")
        print(
            "  a frozen golden may not be modified (AGENTS.md section 7); a principal-approved move is "
            "recorded with 'python scripts/freeze_goldens.py --reseal'"
        )
        return 1
    covered = len(_load_writer().read_manifest(args.frozen_root))
    print(f"frozen golden manifest clean: {covered} file(s) match their recorded digest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
