# SPDX-License-Identifier: Apache-2.0
"""Commit-message validator for the provenance-carrying commit contract.

Enforces the commit format defined in ``AGENTS.md`` sec. 5: a Conventional
Commits subject plus mandatory ``WP:``/``Provenance:``/``Assumptions:``/``Gate:``
trailers, with every provenance source id resolvable against the allowlist in
``docs/PROVENANCE.md``. Hash-sign and at-sign characters may only appear in the
co-author trailers that follow the ``---`` separator.

Examples:
    Validate a single message file (exit 1 on any violation)::

        python scripts/check_commit_trailers.py --file .git/COMMIT_EDITMSG

    Validate every non-merge commit in a range::

        python scripts/check_commit_trailers.py --range origin/main..HEAD
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: Conventional-commit subject: allowed type, optional lowercase scope, ": " then text.
#: ``refine`` is this project's own addition to the conventional set, for a change
#: that improves something already correct — a rename, a clearer boundary, a
#: sharpened comment. ``refactor`` claims behaviour-preserving restructuring and
#: ``docs`` claims prose, and a rename that crosses code, configs and prose at once
#: is neither.
SUBJECT_RE = re.compile(r"^(feat|fix|test|ci|docs|chore|perf|refactor|refine|exp|release)(\([a-z0-9_-]+\))?: .+")
#: Maximum subject length (characters).
SUBJECT_MAX_LEN = 72

#: Trailer matchers (multiline, one capture group each).
#: ``WP:`` takes a roadmap row id, or the literal ``none`` for the D18 case: once the
#: reproduction is complete and the repository public, a change altering no shipped
#: behaviour, no public symbol, no golden and no documented assumption lands without a
#: tracked row. ``none`` rather than an omitted trailer, matching ``Assumptions:``, so
#: the message states that no row applies instead of leaving a reader to decide whether
#: one was forgotten. The other three trailers stay mandatory: the derivation question
#: does not go away because the tracking did (WP-144).
WP_RE = re.compile(r"^WP: (\d+|none)\s*$", re.MULTILINE)
PROVENANCE_RE = re.compile(r"^Provenance:\s*(.+)$", re.MULTILINE)
ASSUMPTIONS_RE = re.compile(r"^Assumptions:\s*(.+)$", re.MULTILINE)
GATE_RE = re.compile(r"^Gate:\s*(.+)$", re.MULTILINE)

#: Source id token inside a ``Provenance:`` line (e.g. ``R1`` in ``R1 3.2.1``).
SOURCE_ID_RE = re.compile(r"\bR(\d+)\b")
#: A single assumption id (e.g. ``A9``).
ASSUMPTION_ID_RE = re.compile(r"^A\d+$")
#: Provenance table row in ``docs/PROVENANCE.md`` (e.g. ``| R1 | ...``).
PROVENANCE_ROW_RE = re.compile(r"^\| R(\d+) \|", re.MULTILINE)
#: Co-author separator: a line containing only dashes.
SEPARATOR_RE = re.compile(r"^---\s*$", re.MULTILINE)

#: The repository this checker belongs to, resolved from the file rather than from the
#: process. Every path and every ``git`` call below is anchored here. Read from the
#: working directory instead, the checker answers about wherever it happens to be
#: standing: from a subdirectory it dies on the provenance table it cannot find, and
#: from another checkout it would validate that checkout's history and report clean.
#: ``release_guard.py`` already anchors this way; these two lint entry points did not.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default location of the provenance allowlist.
DEFAULT_PROVENANCE = REPO_ROOT / "docs" / "PROVENANCE.md"


def load_provenance_ids(path: Path) -> set[int]:
    """Parse the numeric source ids declared in the provenance allowlist.

    Args:
        path: Path to ``docs/PROVENANCE.md``.

    Returns:
        The set of integers ``N`` for every ``| RN |`` table row.

    Examples:
        >>> ids = load_provenance_ids(DEFAULT_PROVENANCE)
        >>> 1 in ids
        True
    """
    text = path.read_text(encoding="utf-8")
    return {int(match) for match in PROVENANCE_ROW_RE.findall(text)}


def _text_before_separator(message: str) -> str:
    """Return the message text preceding the first ``---`` co-author separator.

    Args:
        message: The full commit message.

    Returns:
        Everything before the separator; the whole message when absent.

    Examples:
        >>> _text_before_separator("body\\n---\\nCo-authored-by: x <a@b>")
        'body\\n'
    """
    return SEPARATOR_RE.split(message, maxsplit=1)[0]


def _check_subject(subject: str) -> list[str]:
    """Validate the subject line format and length.

    Args:
        subject: The first line of the commit message.

    Returns:
        A list of violation strings; empty when the subject is valid.

    Examples:
        >>> _check_subject("feat(models): add head")
        []
    """
    violations: list[str] = []
    if not SUBJECT_RE.match(subject):
        violations.append(f"subject does not match '<type>(<scope>): <text>': {subject!r}")
    if len(subject) > SUBJECT_MAX_LEN:
        violations.append(f"subject exceeds {SUBJECT_MAX_LEN} chars ({len(subject)}): {subject!r}")
    return violations


def _check_assumptions(value: str) -> list[str]:
    """Validate the value of an ``Assumptions:`` trailer.

    Args:
        value: The text following ``Assumptions:``.

    Returns:
        A list with one violation when malformed; empty when valid.

    Examples:
        >>> _check_assumptions("A9, A14")
        []
        >>> _check_assumptions("none")
        []
    """
    text = value.strip()
    if text == "none":
        return []
    tokens = [token.strip() for token in text.split(",")]
    bad = [token for token in tokens if not ASSUMPTION_ID_RE.match(token)]
    if bad:
        return [f"invalid assumption ids (expect A<digits> or literal 'none'): {bad}"]
    return []


def _check_provenance(body: str, valid_ids: set[int]) -> list[str]:
    """Validate the ``Provenance:`` trailer against the allowlist.

    Args:
        body: The message text preceding the co-author separator.
        valid_ids: Source ids declared in ``docs/PROVENANCE.md``.

    Returns:
        A list of violation strings; empty when the trailer is valid.

    Examples:
        >>> _check_provenance("Provenance: R1 3.2.1, R6", {1, 6})
        []
    """
    match = PROVENANCE_RE.search(body)
    if not match:
        return ["missing 'Provenance:' trailer"]
    ids = [int(found) for found in SOURCE_ID_RE.findall(match.group(1))]
    if not ids:
        return ["'Provenance:' trailer lists no source id"]
    unknown = sorted({found for found in ids if found not in valid_ids})
    if unknown:
        return [f"unknown provenance ids (not in docs/PROVENANCE.md): {[f'R{found}' for found in unknown]}"]
    return []


def _check_trailers(body: str, valid_ids: set[int]) -> list[str]:
    """Validate the mandatory ``WP``/``Provenance``/``Assumptions``/``Gate`` trailers.

    Args:
        body: The message text preceding the co-author separator.
        valid_ids: Source ids declared in ``docs/PROVENANCE.md``.

    Returns:
        A list of violation strings; empty when every trailer is valid.

    Examples:
        >>> body = "WP: 22\\nProvenance: R1\\nAssumptions: none\\nGate: t.py::x"
        >>> _check_trailers(body, {1})
        []
    """
    violations: list[str] = []
    wp = WP_RE.findall(body)
    if len(wp) != 1:
        violations.append(f"expected exactly one 'WP: <digits>' or 'WP: none' trailer, found {len(wp)}")
    violations += _check_provenance(body, valid_ids)
    assumptions = ASSUMPTIONS_RE.search(body)
    if not assumptions:
        violations.append("missing 'Assumptions:' trailer")
    else:
        violations += _check_assumptions(assumptions.group(1))
    gate = GATE_RE.search(body)
    if not gate or not gate.group(1).strip():
        violations.append("missing or empty 'Gate:' trailer")
    return violations


def _check_forbidden_chars(body: str) -> list[str]:
    """Reject ``#`` and ``@`` before the co-author separator.

    Args:
        body: The message text preceding the co-author separator.

    Returns:
        A list of violation strings; empty when no forbidden character appears.

    Examples:
        >>> _check_forbidden_chars("clean body\\n")
        []
    """
    return [f"forbidden character {char!r} appears before the '---' separator" for char in ("#", "@") if char in body]


def validate_message(message: str, valid_ids: set[int]) -> list[str]:
    """Validate one commit message against the full contract.

    Args:
        message: The full commit message text.
        valid_ids: Source ids declared in ``docs/PROVENANCE.md``.

    Returns:
        A list of violation strings; empty when the message is valid.

    Examples:
        >>> msg = "feat(x): y\\n\\nWP: 1\\nProvenance: R1\\nAssumptions: none\\nGate: t::x"
        >>> validate_message(msg, {1})
        []
    """
    before = _text_before_separator(message)
    subject = message.splitlines()[0] if message.strip() else ""
    violations = _check_subject(subject)
    violations += _check_trailers(before, valid_ids)
    violations += _check_forbidden_chars(before)
    return violations


def _commit_shas(commit_range: str) -> list[str]:
    """Return the non-merge commit hashes in a git range, newest first.

    Args:
        commit_range: A git range expression such as ``origin/main..HEAD``.

    Returns:
        Full commit hashes; empty when the range selects nothing.
    """
    result = subprocess.run(
        ["git", "log", "--no-merges", "--format=%H", commit_range],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _commit_message(sha: str) -> str:
    """Return the full commit message body for a commit hash.

    Args:
        sha: The commit hash to read.

    Returns:
        The raw commit message (subject and body).
    """
    result = subprocess.run(
        ["git", "log", "-1", "--format=%B", sha],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _report(reports: list[tuple[str, list[str]]]) -> int:
    """Print per-commit results and return the process exit code.

    Args:
        reports: ``(label, violations)`` pairs, one per validated message.

    Returns:
        ``1`` when any message has violations, otherwise ``0``.
    """
    failed = [(label, violations) for label, violations in reports if violations]
    for label, violations in failed:
        print(f"COMMIT {label}: FAIL")
        for item in violations:
            print(f"  - {item}")
    if failed:
        print(f"commit-trailer check FAILED: {len(failed)} of {len(reports)} message(s) invalid")
        return 1
    print(f"commit-trailer check clean: {len(reports)} message(s) validated")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, validate the selected message(s), and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` on any violation.

    Examples:
        Command-line usage (exit status is the process return code)::

            $ python scripts/check_commit_trailers.py --file .git/COMMIT_EDITMSG
            $ python scripts/check_commit_trailers.py --range origin/main..HEAD
    """
    parser = argparse.ArgumentParser(description="Validate provenance-carrying commit messages.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--file", type=Path, help="validate a single commit-message file")
    target.add_argument("--range", dest="commit_range", help="validate every non-merge commit in a git range")
    parser.add_argument(
        "--provenance",
        type=Path,
        default=DEFAULT_PROVENANCE,
        help=f"path to the provenance allowlist (default: {DEFAULT_PROVENANCE})",
    )
    args = parser.parse_args(argv)

    valid_ids = load_provenance_ids(args.provenance)
    reports: list[tuple[str, list[str]]] = []
    if args.file is not None:
        message = args.file.read_text(encoding="utf-8")
        reports.append((str(args.file), validate_message(message, valid_ids)))
    else:
        for sha in _commit_shas(args.commit_range):
            reports.append((sha[:12], validate_message(_commit_message(sha), valid_ids)))
    return _report(reports)


if __name__ == "__main__":
    sys.exit(main())
