# SPDX-License-Identifier: Apache-2.0
"""Release guard: decide whether a version tag may ship (WP-006).

The guard runs three independent checks and lets a tag ship only when all three
pass:

1. **Tag** — the tag must be a ``v0.MINOR.PATCH`` string. A ``1.x`` (or higher)
   tag is refused with an ADR-002 message: the project is a perpetual 0.x
   release train and no 1.0 is ever planned, promised, or tagged. Any other
   shape is refused as malformed.
2. **Changelog** — the changelog must carry a ``## [MINOR.PATCH]`` section for
   the tag, so every release ships with its notes written.
3. **Gate** — the gate command (``make gate`` by default) must exit ``0``. A
   non-zero exit is a red gate and refuses the tag.

The check functions are importable and pure over their inputs; :func:`main` is a
thin CLI that prints one verdict line per check and exits non-zero on refusal.

Examples:
    Guard a well-formed tag against a changelog that names it, with a trivially
    green gate::

        python scripts/release_guard.py --tag v0.1.0 --gate-cmd true
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Repository root (``scripts/`` is one level below it).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Default changelog consulted for the tag's version section.
DEFAULT_CHANGELOG = REPO_ROOT / "CHANGELOG.md"

#: Default gate command re-run before a tag may ship.
DEFAULT_GATE_CMD = "make gate"

#: A shippable tag: zero major, per ADR-002's perpetual 0.x train.
ZERO_MAJOR_TAG = re.compile(r"^v0\.\d+\.\d+$")

#: Any ``v<major>.<minor>.<patch>`` tag, used to detect a refused 1.x+ tag.
SEMVER_TAG = re.compile(r"^v(\d+)\.\d+\.\d+$")

#: Refusal message for a major-version->=1 tag (ADR-002, hard-coded).
ADR_002_REFUSAL = (
    "tag {tag} declares major version >= 1; ADR-002 mandates a perpetual 0.x "
    "release train — no 1.0 is ever planned, promised, or tagged"
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one release-guard check.

    Attributes:
        name: Short check name (``tag``, ``changelog``, or ``gate``).
        passed: Whether the check permits the tag to ship.
        detail: Human-readable verdict explaining the outcome.
    """

    name: str
    passed: bool
    detail: str


def check_tag(tag: str) -> CheckResult:
    """Validate that ``tag`` is a shippable ``v0.MINOR.PATCH`` string.

    A zero-major tag passes. A syntactically valid tag whose major version is
    ``>= 1`` is refused with the ADR-002 message. Anything else is refused as
    malformed.

    Args:
        tag: The candidate tag, e.g. ``"v0.1.0"``.

    Returns:
        A :class:`CheckResult` named ``"tag"``.

    Examples:
        ```pycon
        >>> check_tag("v0.1.0").passed
        True
        >>> check_tag("v1.0.0").passed
        False
        >>> check_tag("0.1").passed
        False

        ```
    """
    if ZERO_MAJOR_TAG.match(tag):
        return CheckResult("tag", True, f"tag {tag} matches the perpetual 0.x scheme")
    semver = SEMVER_TAG.match(tag)
    if semver is not None and int(semver.group(1)) >= 1:
        return CheckResult("tag", False, ADR_002_REFUSAL.format(tag=tag))
    return CheckResult("tag", False, f"tag {tag!r} is malformed; expected 'v0.MINOR.PATCH'")


def check_changelog(tag: str, changelog: Path) -> CheckResult:
    """Verify that ``changelog`` carries a ``## [MINOR.PATCH]`` section for ``tag``.

    Args:
        tag: The candidate tag; its leading ``v`` is stripped to form the version.
        changelog: Path to the changelog file to read.

    Returns:
        A :class:`CheckResult` named ``"changelog"``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "CHANGELOG.md"
        ...     _ = path.write_text("## [0.1.0]\\n- first release\\n")
        ...     check_changelog("v0.1.0", path).passed
        True

        ```
    """
    heading = f"## [{tag.removeprefix('v')}]"
    try:
        text = changelog.read_text(encoding="utf-8")
    except OSError as exc:
        return CheckResult("changelog", False, f"cannot read changelog {changelog}: {exc}")
    if heading not in text:
        return CheckResult("changelog", False, f"changelog {changelog} has no {heading!r} section")
    return CheckResult("changelog", True, f"changelog carries a {heading!r} section")


def check_gate(gate_cmd: str) -> CheckResult:
    """Run ``gate_cmd`` and pass only when it exits ``0``.

    Args:
        gate_cmd: A shell command string (e.g. ``"make gate"``), run from the
            repository root.

    Returns:
        A :class:`CheckResult` named ``"gate"``; a non-zero exit is a red gate.

    Examples:
        ```pycon
        >>> check_gate("true").passed
        True
        >>> check_gate("false").passed
        False

        ```
    """
    try:
        # gate_cmd is an operator-supplied command string (e.g. "make gate"), run as a shell command by design.
        completed = subprocess.run(gate_cmd, shell=True, cwd=REPO_ROOT, check=False)
    except OSError as exc:
        return CheckResult("gate", False, f"gate command {gate_cmd!r} failed to launch: {exc}")
    if completed.returncode != 0:
        return CheckResult("gate", False, f"gate command {gate_cmd!r} exited {completed.returncode} (red gate)")
    return CheckResult("gate", True, f"gate command {gate_cmd!r} passed")


def _current_tag() -> str | None:
    """Return the tag exactly naming ``HEAD``, or ``None`` when there is none.

    Backs :func:`main`'s ``--tag`` default so the guard is runnable as a
    pre-commit hook with no per-invocation argument: most commits are not on a
    tag, and the hook has nothing to check for them.

    Examples:
        >>> _current_tag() is None or isinstance(_current_tag(), str)
        True
    """
    completed = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def evaluate(tag: str, changelog: Path, gate_cmd: str) -> list[CheckResult]:
    """Run every release-guard check and return their results in order.

    Args:
        tag: The candidate tag.
        changelog: Path to the changelog file.
        gate_cmd: Shell command whose zero exit means a green gate.

    Returns:
        The ``[tag, changelog, gate]`` results.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "CHANGELOG.md"
        ...     _ = path.write_text("## [0.1.0]\\n")
        ...     [r.passed for r in evaluate("v0.1.0", path, "true")]
        [True, True, True]

        ```
    """
    return [check_tag(tag), check_changelog(tag, changelog), check_gate(gate_cmd)]


def main(argv: list[str] | None = None) -> int:
    """Guard a release tag and print each check's verdict.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` only when the tag, changelog, and gate checks all pass; ``1``
        otherwise.

    Examples:
        The verdict lines go to stdout and name the changelog by absolute path,
        so the example captures them and asserts the exit code alone.

        ```pycon
        >>> import contextlib, io
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     status = main(["--tag", "v1.0.0", "--gate-cmd", "true"])
        >>> status
        1

        ```
    """
    parser = argparse.ArgumentParser(description="Decide whether a release tag may ship.")
    parser.add_argument(
        "--tag",
        default=None,
        help="candidate tag, e.g. v0.1.0 (default: the tag exactly naming HEAD, if any)",
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        default=DEFAULT_CHANGELOG,
        help="changelog that must carry the tag's version section (default: <repo>/CHANGELOG.md)",
    )
    parser.add_argument(
        "--gate-cmd",
        default=DEFAULT_GATE_CMD,
        help="gate command re-run before shipping; non-zero exit refuses the tag (default: 'make gate')",
    )
    args = parser.parse_args(argv)

    tag = args.tag if args.tag is not None else _current_tag()
    if tag is None:
        print("release guard: HEAD is not exactly a tag — nothing to check")
        return 0

    results = evaluate(tag, args.changelog, args.gate_cmd)
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        print(f"{marker} [{result.name}] {result.detail}")
    passed = all(result.passed for result in results)
    print("release guard: " + ("all checks passed — tag may ship" if passed else "checks failed — tag refused"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
