# SPDX-License-Identifier: Apache-2.0
"""Commit-message validator for the provenance-carrying commit contract.

Enforces the commit format defined in ``AGENTS.md`` sec. 5: a Conventional
Commits subject plus mandatory ``WP:``/``Provenance:``/``Assumptions:``/``Gate:``
trailers, with every provenance source id resolvable against the allowlist in
``docs/PROVENANCE.md`` and every assumption id against the register in
``docs/ASSUMPTIONS.md``. Hash-sign and at-sign characters may only appear in the
co-author trailers that follow the ``---`` separator.

Three rules are worth stating here because each is a decision rather than a shape:

* **Denial is a value.** ``Provenance: none`` is accepted the way ``WP: none`` and
  ``Assumptions: none`` already are, and it may carry its explanation --
  ``none (implementation efficiency; R1 silent on target rasterisation)``. A value
  opening with ``none`` is read as *denying* provenance, so an id inside its
  explanation is a reference and never a citation: it must still resolve, but it
  cannot satisfy the requirement to name a source. Reading such a line as "cites R1"
  is the opposite of what it says.
* **Only the allowlist admits.** Source ids are read from the ``Source allowlist``
  section of ``docs/PROVENANCE.md`` and never from its ``Placeholders`` table, whose
  rows share the same shape. A withdrawn or citation-only row is refused by name
  rather than reported as unknown.
* **Assumption ids are resolved, not shape-checked.** ``A999`` matching ``^A\\d+$``
  is not evidence that the register holds a row for it.

Examples:
    Validate a single message file (exit 1 on any violation)::

        python scripts/lint/check_commit_trailers.py --file .git/COMMIT_EDITMSG

    Validate every non-merge commit in a range::

        python scripts/lint/check_commit_trailers.py --range origin/main..HEAD
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
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
PROVENANCE_ROW_RE = re.compile(r"^\| R(\d+) \|")
#: Assumption table row in ``docs/ASSUMPTIONS.md`` (e.g. ``| A9 | ...``).
ASSUMPTION_ROW_RE = re.compile(r"^\| A(\d+) \|")
#: Any markdown heading, with its level and its text.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
#: A trailer value that opens by denying the thing it names, e.g. ``none (…)``.
DENIAL_RE = re.compile(r"^none\b", re.IGNORECASE)
#: Co-author separator: a line containing only dashes.
SEPARATOR_RE = re.compile(r"^---\s*$", re.MULTILINE)

#: Heading text (lowercased, emoji-tolerant) marking the section whose rows admit a
#: source, and the subsection inside it whose rows do not.
_ALLOWLIST_HEADING = "source allowlist"
_PLACEHOLDER_HEADING = "placeholders"

#: The repository this checker belongs to, resolved from the file rather than from the
#: process. Every path and every ``git`` call below is anchored here. Read from the
#: working directory instead, the checker answers about wherever it happens to be
#: standing: from a subdirectory it dies on the provenance table it cannot find, and
#: from another checkout it would validate that checkout's history and report clean.
#: ``release_guard.py`` already anchors this way; these two lint entry points did not.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default location of the provenance allowlist.
DEFAULT_PROVENANCE = REPO_ROOT / "docs" / "PROVENANCE.md"
#: Default location of the assumption register.
DEFAULT_ASSUMPTIONS = REPO_ROOT / "docs" / "ASSUMPTIONS.md"


@dataclass(frozen=True, slots=True)
class Registers:
    """The three id populations a commit message is validated against.

    Attributes:
        sources: Source ids a message may cite -- the ``Source allowlist`` section's
            rows, minus its ``Placeholders`` table.
        placeholders: Source ids the allowlist declares and refuses to admit: a
            withdrawn row, or one registered as legal evidence rather than as an
            implementation source. Carried separately from ``sources`` so a message
            naming one is refused as *inadmissible* rather than as *unknown* -- the
            two are different mistakes and only one of them is a typo.
        assumptions: Assumption ids the register holds a row for.
    """

    sources: frozenset[int]
    placeholders: frozenset[int]
    assumptions: frozenset[int]


def _rows_by_section(text: str, row_re: re.Pattern[str]) -> list[tuple[str, str, int]]:
    """Every ``row_re`` match in ``text`` as ``(section, subsection, id)``, headings lowercased.

    Args:
        text: The markdown document to walk.
        row_re: A table-row matcher whose first group is the numeric id.

    Returns:
        One triple per matching row, in file order; a row above any heading carries
        empty strings for both heading levels.

    Examples:
        >>> _rows_by_section("## Allow\\n| R1 |\\n### Placeholders\\n| R2 |\\n", PROVENANCE_ROW_RE)
        [('allow', '', 1), ('allow', 'placeholders', 2)]
    """
    rows: list[tuple[str, str, int]] = []
    section = subsection = ""
    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if heading is not None:
            level, title = len(heading.group(1)), heading.group(2).strip().lower()
            if level <= 2:
                section, subsection = title, ""
            else:
                subsection = title
            continue
        row = row_re.match(line)
        if row is not None:
            rows.append((section, subsection, int(row.group(1))))
    return rows


def load_registers(provenance: Path = DEFAULT_PROVENANCE, assumptions: Path = DEFAULT_ASSUMPTIONS) -> Registers:
    """Parse the source allowlist and the assumption register into resolvable id sets.

    The allowlist is read by section rather than file-wide: ``docs/PROVENANCE.md``'s
    ``Placeholders`` table uses the same ``| RN |`` row shape as the admitting
    sections, so a file-wide scan hands back the withdrawn rows as valid ids.

    Args:
        provenance: Path to ``docs/PROVENANCE.md``.
        assumptions: Path to ``docs/ASSUMPTIONS.md``.

    Returns:
        The admissible source ids, the declared-but-inadmissible ones, and the
        registered assumption ids.

    Examples:
        >>> registers = load_registers()
        >>> 1 in registers.sources, 15 in registers.sources, 15 in registers.placeholders
        (True, False, True)
        >>> 9 in registers.assumptions, 999 in registers.assumptions
        (True, False)
    """
    rows = _rows_by_section(provenance.read_text(encoding="utf-8"), PROVENANCE_ROW_RE)
    admitted = {row for section, subsection, row in rows if _ALLOWLIST_HEADING in section and not subsection}
    admitted |= {
        row
        for section, subsection, row in rows
        if _ALLOWLIST_HEADING in section and subsection and _PLACEHOLDER_HEADING not in subsection
    }
    refused = {
        row for section, subsection, row in rows if _ALLOWLIST_HEADING in section and _PLACEHOLDER_HEADING in subsection
    }
    registered = {row for _, _, row in _rows_by_section(assumptions.read_text(encoding="utf-8"), ASSUMPTION_ROW_RE)}
    return Registers(frozenset(admitted - refused), frozenset(refused), frozenset(registered))


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


def _check_assumptions(value: str, registered: frozenset[int]) -> list[str]:
    """Validate the value of an ``Assumptions:`` trailer against the register.

    Shape and membership are separate failures: ``A9x`` is a malformed id, ``A999``
    is a well-formed id naming no row. Only the second needed adding -- an id that
    matches ``^A\\d+$`` and resolves to nothing cites a row the register does not
    hold, which is the same defect ``Provenance:`` has always refused.

    Args:
        value: The text following ``Assumptions:``.
        registered: Assumption ids declared in ``docs/ASSUMPTIONS.md``.

    Returns:
        A list of violation strings; empty when every id is well-formed and registered.

    Examples:
        >>> _check_assumptions("A9, A14", frozenset({9, 14}))
        []
        >>> _check_assumptions("none", frozenset())
        []
        >>> _check_assumptions("A999", frozenset({9}))
        ["unregistered assumption ids (not in docs/ASSUMPTIONS.md): ['A999']"]
    """
    text = value.strip()
    if text == "none":
        return []
    tokens = [token.strip() for token in text.split(",")]
    bad = [token for token in tokens if not ASSUMPTION_ID_RE.match(token)]
    if bad:
        return [f"invalid assumption ids (expect A<digits> or literal 'none'): {bad}"]
    unknown = [token for token in tokens if int(token[1:]) not in registered]
    if unknown:
        return [f"unregistered assumption ids (not in docs/ASSUMPTIONS.md): {unknown}"]
    return []


def _check_source_ids(ids: list[int], registers: Registers) -> list[str]:
    """Violations for source ids the allowlist does not admit, by why it does not.

    Args:
        ids: Numeric source ids read off a ``Provenance:`` value.
        registers: The parsed allowlist and register.

    Returns:
        At most two violations -- one naming the inadmissible ids, one the unknown.

    Examples:
        >>> registers = Registers(frozenset({1}), frozenset({15}), frozenset())
        >>> for violation in _check_source_ids([15, 99], registers):
        ...     print(violation)
        inadmissible provenance ids (declared, not admitted, in docs/PROVENANCE.md): ['R15']
        unknown provenance ids (not in docs/PROVENANCE.md): ['R99']
    """
    violations = []
    refused = sorted({found for found in ids if found in registers.placeholders})
    if refused:
        violations.append(
            "inadmissible provenance ids (declared, not admitted, in docs/PROVENANCE.md): "
            f"{[f'R{found}' for found in refused]}"
        )
    unknown = sorted({found for found in ids if found not in registers.sources and found not in registers.placeholders})
    if unknown:
        violations.append(f"unknown provenance ids (not in docs/PROVENANCE.md): {[f'R{found}' for found in unknown]}")
    return violations


def _check_provenance(body: str, registers: Registers) -> list[str]:
    """Validate the ``Provenance:`` trailer against the allowlist.

    A value opening with ``none`` denies provenance and is accepted as such, the way
    ``WP: none`` is: it may carry an explanation, and an id inside that explanation
    is a reference rather than a citation -- resolved, but never counted as the
    source the trailer was asked to name.

    Args:
        body: The message text preceding the co-author separator.
        registers: The parsed allowlist and register.

    Returns:
        A list of violation strings; empty when the trailer is valid.

    Examples:
        >>> registers = Registers(frozenset({1, 6}), frozenset({15}), frozenset())
        >>> _check_provenance("Provenance: R1 3.2.1, R6", registers)
        []
        >>> _check_provenance("Provenance: none (R1 is silent here)", registers)
        []
        >>> _check_provenance("Provenance: none (R15 is silent here)", registers)
        ["inadmissible provenance ids (declared, not admitted, in docs/PROVENANCE.md): ['R15']"]
    """
    match = PROVENANCE_RE.search(body)
    if not match:
        return ["missing 'Provenance:' trailer"]
    value = match.group(1).strip()
    ids = [int(found) for found in SOURCE_ID_RE.findall(value)]
    if DENIAL_RE.match(value):
        return _check_source_ids(ids, registers)
    if not ids:
        return ["'Provenance:' trailer lists no source id (expect R<digits>, or 'none' to deny one)"]
    return _check_source_ids(ids, registers)


def _check_trailers(body: str, registers: Registers) -> list[str]:
    """Validate the mandatory ``WP``/``Provenance``/``Assumptions``/``Gate`` trailers.

    Args:
        body: The message text preceding the co-author separator.
        registers: The parsed allowlist and register.

    Returns:
        A list of violation strings; empty when every trailer is valid.

    Examples:
        >>> body = "WP: 22\\nProvenance: R1\\nAssumptions: none\\nGate: t.py::x"
        >>> _check_trailers(body, Registers(frozenset({1}), frozenset(), frozenset()))
        []
    """
    violations: list[str] = []
    wp = WP_RE.findall(body)
    if len(wp) != 1:
        violations.append(f"expected exactly one 'WP: <digits>' or 'WP: none' trailer, found {len(wp)}")
    violations += _check_provenance(body, registers)
    assumptions = ASSUMPTIONS_RE.search(body)
    if not assumptions:
        violations.append("missing 'Assumptions:' trailer")
    else:
        violations += _check_assumptions(assumptions.group(1), registers.assumptions)
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


def validate_message(message: str, registers: Registers) -> list[str]:
    """Validate one commit message against the full contract.

    Args:
        message: The full commit message text.
        registers: The parsed allowlist and register.

    Returns:
        A list of violation strings; empty when the message is valid.

    Examples:
        >>> msg = "feat(x): y\\n\\nWP: 1\\nProvenance: R1\\nAssumptions: none\\nGate: t::x"
        >>> validate_message(msg, Registers(frozenset({1}), frozenset(), frozenset()))
        []
    """
    before = _text_before_separator(message)
    subject = message.splitlines()[0] if message.strip() else ""
    violations = _check_subject(subject)
    violations += _check_trailers(before, registers)
    violations += _check_forbidden_chars(before)
    return violations


class UnresolvableRange(RuntimeError):
    """A git range this repository cannot resolve, such as one naming an absent base ref."""


def _commit_shas(commit_range: str) -> list[str]:
    """Return the non-merge commit hashes in a git range, newest first.

    Args:
        commit_range: A git range expression such as ``origin/main..HEAD``.

    Returns:
        Full commit hashes; empty when the range selects nothing.

    Raises:
        UnresolvableRange: When git cannot resolve the range. A shallow checkout has no
            ``refs/remotes/origin/main``, so the default range exits 128 -- which used to
            reach the caller as a traceback, a checker crashing rather than reporting.
            CI fetches full history for the jobs that run this; a contributor cloning
            shallowly, or working in a fork whose remote is named otherwise, still lands
            here and is owed a sentence naming the fix rather than a stack trace.
    """
    result = subprocess.run(
        ["git", "log", "--no-merges", "--format=%H", commit_range],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        raise UnresolvableRange(detail[0] if detail else f"git could not resolve {commit_range!r}")
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

            $ python scripts/lint/check_commit_trailers.py --file .git/COMMIT_EDITMSG
            $ python scripts/lint/check_commit_trailers.py --range origin/main..HEAD
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
    parser.add_argument(
        "--assumptions",
        type=Path,
        default=DEFAULT_ASSUMPTIONS,
        help=f"path to the assumption register (default: {DEFAULT_ASSUMPTIONS})",
    )
    args = parser.parse_args(argv)

    registers = load_registers(args.provenance, args.assumptions)
    reports: list[tuple[str, list[str]]] = []
    if args.file is not None:
        message = args.file.read_text(encoding="utf-8")
        reports.append((str(args.file), validate_message(message, registers)))
    else:
        try:
            shas = _commit_shas(args.commit_range)
        except UnresolvableRange as unresolvable:
            print(f"commit-trailer check FAILED: cannot resolve range {args.commit_range!r}: {unresolvable}")
            print("  fetch the base ref first -- a shallow clone has no origin/main to measure against")
            return 1
        for sha in shas:
            reports.append((sha[:12], validate_message(_commit_message(sha), registers)))
    return _report(reports)


if __name__ == "__main__":
    sys.exit(main())
