# SPDX-License-Identifier: Apache-2.0
"""Structural audit of the ``docs/`` governance registers (WP-003, split off tests/meta).

``tests/meta/test_docs_present.py`` used to carry twelve independent test functions,
each checking one structural property of the policy documents that back the
clean-room paper trail: that every required file exists, that the ``ASSUMPTIONS.md``,
``ROADMAP.md``, ``DECISIONS.md`` and ``PROVENANCE.md`` registers parse as contiguous,
fully-populated id sequences, and that cross-file links between ``ROADMAP.md`` and the
two log files resolve. Those checks depend only on repo content, not on the installed
Python environment, so like ``audit_test_doctests.py`` before it they moved here: one
``check_<name>`` function per original test, each returning the violations it finds
instead of asserting, so a caller can run every check and report all of them at once
rather than stopping at the first failure.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_docs_present.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"

#: Lowest work-package count ROADMAP.md is allowed to hold. A ratchet, not a target:
#: contiguity alone would not notice the last row being deleted. Raise it when adding
#: a work package; never lower it.
_WP_FLOOR = 176

#: Lowest decision count DECISIONS.md is allowed to hold. A ratchet, not a target:
#: contiguity alone would not notice the last row being deleted, since what remains
#: stays contiguous. Raise it when adding a decision; never lower it.
_DECISION_FLOOR = 20

#: Lowest assumption count ASSUMPTIONS.md is allowed to hold, for the same ratchet
#: reason as the two floors above.
_ASSUMPTION_FLOOR = 73

#: Paths, relative to ``docs_dir``/``repo_root``, every required policy document must
#: resolve to. Mirrors the former ``REQUIRED_FILES`` tuple, split into its two roots so
#: each check can be exercised against a synthetic tree rather than the live repo.
_REQUIRED_DOCS_RELATIVE = (
    "PROVENANCE.md",
    "ASSUMPTIONS.md",
    "DECISIONS.md",
    "ESCALATION.md",
    "ROADMAP.md",
    "DATASETS.md",
    "TRAINING.md",
    "REPRODUCTION_REPORT.md",
    "RESEARCH_LOG.md",
    "ENGINEERING_LOG.md",
    "model_cards/detection.md",
    "model_cards/segmentation.md",
    "model_cards/obb.md",
    "model_cards/keypoints.md",
)

#: Section titles ``REPRODUCTION_REPORT.md`` must carry, in the order D10 appends them: the
#: four per-release tiers plus the WP-065 consolidation, which stays third in append order
#: since D10 forbids reordering a heading once appended. Copied from the file itself, not
#: retyped -- each uses an em dash, and a hyphen-typed copy would silently never match.
#: Stored without the ``##`` marker and without the leading emoji (WP-113): the heading's
#: decoration is presentation, and pinning it here would make a purely visual edit fail a
#: check about which sections exist.
_REPORT_SECTIONS = (
    "0.1.0 — Detection",
    "0.2.0 — Instance segmentation",
    "0.3.0 — Oriented detection",
    "Consolidated note — detection, segmentation, oriented detection",
    "0.5.0 — Keypoint detection",
)


def _undecorated(title: str) -> str:
    """Strip a heading's leading emoji, leaving the title the registers are asserted on.

    Every H1 and H2 across the docs carries a topical emoji (WP-113), which is decoration:
    a gate that pinned it would fail on a re-picked icon while reporting a missing section.
    A lead token holding no ASCII alphanumeric is that decoration; anything else is title.

    Examples:
        >>> _undecorated("🚀 Getting Started")
        'Getting Started'
        >>> _undecorated("Getting Started")
        'Getting Started'
    """
    lead, _, rest = title.partition(" ")
    if rest and not any(char.isalnum() and char.isascii() for char in lead):
        return rest.strip()
    return title


def _table_cells(line: str) -> list[str]:
    """Split one markdown table row into its cells, honoring backslash-escaped pipes.

    Examples:
        >>> _table_cells(r"| a | b\\|c | d |")
        ['a', 'b|c', 'd']
    """
    cells, current, escaped = [], "", False
    for char in line.strip().strip("|"):
        if escaped:
            current, escaped = current + char, False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append(current.strip())
            current = ""
        else:
            current += char
    return [*cells, current.strip()]


def _required_files(docs_dir: Path, repo_root: Path) -> list[Path]:
    """Every governance document required to exist, at its expected path.

    Docs-tree entries resolve under ``docs_dir``; ``AGENTS.md`` resolves under
    ``repo_root`` directly, since it lives at the repo root rather than in ``docs/``.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     paths = _required_files(root / "docs", root)
        ...     paths[0].name, paths[-1].name
        ('PROVENANCE.md', 'AGENTS.md')
    """
    return [docs_dir / rel for rel in _REQUIRED_DOCS_RELATIVE] + [repo_root / "AGENTS.md"]


def check_policy_docs_exist(docs_dir: Path, repo_root: Path) -> list[str]:
    """Every policy document required by the execution contract is present.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     (root / "docs").mkdir()
        ...     violations = check_policy_docs_exist(root / "docs", root)
        ...     violations[0].startswith("missing policy docs")
        True
    """
    missing = [str(path.relative_to(repo_root)) for path in _required_files(docs_dir, repo_root) if not path.is_file()]
    if not missing:
        return []
    return [f"missing policy docs: {missing}"]


def check_model_cards_are_required(docs_dir: Path, repo_root: Path) -> list[str]:
    """No card sits in ``docs_dir/model_cards`` unlisted among the required files (WP-106).

    The directory is the natural place to drop a fourth card, and a card nothing gates is
    a card that can be deleted or renamed silently. The listing is what makes each one
    required, so the two are checked against each other rather than kept in step by hand.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     cards = root / "docs" / "model_cards"
        ...     cards.mkdir(parents=True)
        ...     _ = (cards / "extra.md").write_text("x", encoding="utf-8")
        ...     check_model_cards_are_required(root / "docs", root)
        ["model cards not in REQUIRED_FILES: ['extra.md']"]
    """
    model_cards_dir = docs_dir / "model_cards"
    on_disk = {path.name for path in model_cards_dir.glob("*.md")}
    required = {path.name for path in _required_files(docs_dir, repo_root) if path.parent == model_cards_dir}
    if on_disk == required:
        return []
    return [f"model cards not in REQUIRED_FILES: {sorted(on_disk - required)}"]


def check_log_links_resolve(docs_dir: Path, repo_root: Path) -> list[str]:
    """Each ``<LOG>.md#anchor`` the roadmap cites is an anchor that log file defines.

    WP-118 split what was one research log into a fidelity log and an engineering log,
    by claim rather than by work package: a WP whose finding straddles both gets one
    entry in each, cross-linked, so a roadmap row may cite either file or both. That
    split is only safe while the pointers hold: a renamed section leaves a row citing
    evidence a reader cannot reach, and nothing about the roadmap itself would look
    wrong. Anchors are explicit ``<a id=>`` tags rather than heading slugs, so they are
    greppable and survive a retitle.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     _ = (docs / "ROADMAP.md").write_text(
        ...         "See RESEARCH_LOG.md#missing-anchor for detail.\\n", encoding="utf-8"
        ...     )
        ...     _ = (docs / "RESEARCH_LOG.md").write_text("no anchors here\\n", encoding="utf-8")
        ...     _ = (docs / "ENGINEERING_LOG.md").write_text("no anchors here\\n", encoding="utf-8")
        ...     check_log_links_resolve(docs, docs)
        ["roadmap cites undefined RESEARCH_LOG.md anchors: ['missing-anchor']"]
    """
    roadmap = (docs_dir / "ROADMAP.md").read_text(encoding="utf-8")
    logs = {
        "RESEARCH_LOG.md": (docs_dir / "RESEARCH_LOG.md").read_text(encoding="utf-8"),
        "ENGINEERING_LOG.md": (docs_dir / "ENGINEERING_LOG.md").read_text(encoding="utf-8"),
    }

    violations = []
    cited_total = 0
    for name, log in logs.items():
        cited = set(re.findall(rf"{re.escape(name)}#([\w-]+)", roadmap))
        defined = set(re.findall(r'<a id="([\w-]+)">', log))
        cited_total += len(cited)
        if not cited <= defined:
            violations.append(f"roadmap cites undefined {name} anchors: {sorted(cited - defined)}")
    if not cited_total:
        violations.append("no roadmap row links either log")
    return violations


def check_assumption_ids_contiguous(docs_dir: Path, repo_root: Path) -> list[str]:
    """ASSUMPTIONS.md register rows carry ids A1..AN with no gap or duplicate, and never shrink.

    The floor counts rows (``len``) rather than reading the highest id (``max``), for
    the reason :data:`_WP_FLOOR` already counts them: ``max`` grades the register by
    its last row alone, so deleting a row from the *middle* leaves the maximum
    untouched and the floor silent -- and the contiguity check above is what would
    have caught that, except a deletion from the tail defeats it instead. One
    predicate cannot see the tail and the other cannot see the middle, which is why
    the two are separate checks and why this one counts.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     _ = (docs / "ASSUMPTIONS.md").write_text("| A1 | ... |\\n| A3 | ... |\\n", encoding="utf-8")
        ...     check_assumption_ids_contiguous(docs, docs)
        ['non-contiguous assumption ids: [1, 3]', 'assumptions shrank below 73 rows: 2']
    """
    text = (docs_dir / "ASSUMPTIONS.md").read_text(encoding="utf-8")
    ids = [int(m) for m in re.findall(r"^\| A(\d+) \|", text, flags=re.MULTILINE)]
    if not ids:
        return ["no assumption rows found"]

    violations = []
    if len(ids) != len(set(ids)):
        violations.append("duplicate assumption ids")
    if sorted(ids) != list(range(1, max(ids) + 1)):
        violations.append(f"non-contiguous assumption ids: {sorted(ids)}")
    if len(ids) < _ASSUMPTION_FLOOR:
        violations.append(f"assumptions shrank below {_ASSUMPTION_FLOOR} rows: {len(ids)}")
    return violations


def check_assumption_rows_complete(docs_dir: Path, repo_root: Path) -> list[str]:
    """Every register row carries all six columns, with a recognized status in the last.

    A row written one cell short does not look broken: the formatter pads it back to six,
    and the id and contiguity gates above keep passing because neither reads past the
    first column. What actually happens is that every value shifts left -- the validation
    plan lands under "Public source", the status under "Validation" -- so the register
    reads as though a sourced assumption were unsourced. Two rows shipped that way before
    this check existed (A25, A42).

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     _ = (docs / "ASSUMPTIONS.md").write_text("| A1 | short |\\n", encoding="utf-8")
        ...     check_assumption_rows_complete(docs, docs)
        ["rows without exactly six columns: [('A1', 2)]"]
    """
    rows = [
        _table_cells(line)
        for line in (docs_dir / "ASSUMPTIONS.md").read_text(encoding="utf-8").splitlines()
        if re.match(r"^\| A\d+ \|", line)
    ]
    if not rows:
        return ["no assumption rows found"]

    violations = []
    malformed = [(row[0], len(row)) for row in rows if len(row) != 6]
    if malformed:
        violations.append(f"rows without exactly six columns: {malformed}")
    unstatused = [
        (row[0], row[5]) for row in rows if len(row) > 5 and row[5] not in {"open", "active", "validated", "revised"}
    ]
    if unstatused:
        violations.append(f"rows whose last column is not a status: {unstatused}")
    unsourced = [row[0] for row in rows if len(row) > 3 and not row[3]]
    if unsourced:
        violations.append(f"rows with an empty public-source column: {unsourced}")
    return violations


def check_roadmap_wp_ids_unique_and_complete(docs_dir: Path, repo_root: Path) -> list[str]:
    """ROADMAP.md numbers its work packages contiguously from 001, each exactly once, and never shrinks.

    Contiguity and uniqueness are the real invariants and they hold at any size;
    the count is pinned separately by a floor, so adding a work package does not
    fail a check that is not reporting a defect. Dropping the last row would leave
    the remainder contiguous, which is what the floor is for. Raise it when adding
    a package -- see :data:`_WP_FLOOR`.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     _ = (docs / "ROADMAP.md").write_text("| 001 | ... |\\n| 002 | ... |\\n", encoding="utf-8")
        ...     check_roadmap_wp_ids_unique_and_complete(docs, docs)
        ['roadmap shrank below 176 work packages: 2']
    """
    text = (docs_dir / "ROADMAP.md").read_text(encoding="utf-8")
    ids = [int(m) for m in re.findall(r"^\| (\d{3}) \|", text, flags=re.MULTILINE)]

    violations = []
    if len(ids) != len(set(ids)):
        violations.append("duplicate WP ids in roadmap")
    if not ids:
        violations.append("no roadmap WP ids found")
        return violations
    if sorted(ids) != list(range(1, max(ids) + 1)):
        violations.append(f"roadmap ids are not contiguous from 1: {sorted(ids)}")
    if len(ids) < _WP_FLOOR:
        violations.append(f"roadmap shrank below {_WP_FLOOR} work packages: {len(ids)}")
    return violations


def check_roadmap_rows_complete(docs_dir: Path, repo_root: Path) -> list[str]:
    """Every numbered roadmap row carries all six columns.

    An unescaped pipe inside a code span opens a table cell, so a row that reads fine in
    the source renders its tail into the wrong columns on GitHub and drops the overflow.
    This has now happened twice: once to a WP-058 row, and once to WP-093's own row, where
    ``|d_theta|`` in a code span was expanded by the formatter into cell delimiters and
    left the row with four cells instead of six.

    Neither id nor status check alone could see it: the id check reads only the first
    column, and the status check matches only the last one by regex, so a row can lose
    its middle entirely with both of them green.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     _ = (docs / "ROADMAP.md").write_text("| 001 | x |\\n", encoding="utf-8")
        ...     check_roadmap_rows_complete(docs, docs)
        ["roadmap rows without exactly six columns: [('| 001 | x', 2)]"]
    """
    rows = [
        (line[:9], _table_cells(line))
        for line in (docs_dir / "ROADMAP.md").read_text(encoding="utf-8").splitlines()
        if re.match(r"^\| \d{3}[a-z]? \|", line)
    ]
    if not rows:
        return ["no roadmap rows found"]
    malformed = [(head, len(cells)) for head, cells in rows if len(cells) != 6]
    if not malformed:
        return []
    return [f"roadmap rows without exactly six columns: {malformed}"]


def check_roadmap_statuses_valid(docs_dir: Path, repo_root: Path) -> list[str]:
    """Every roadmap row ends in a recognized status icon, and enough rows exist.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     _ = (docs / "ROADMAP.md").write_text("| 001 | a | b | c | d | ✅ |\\n", encoding="utf-8")
        ...     violations = check_roadmap_statuses_valid(docs, docs)
        ...     violations[0].startswith("expected at least 91")
        True
    """
    text = (docs_dir / "ROADMAP.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\d{3}[a-z]?) \|.*\| (\S+) \|$", text, flags=re.MULTILINE)

    violations = []
    if len(rows) < 91:
        violations.append(f"expected at least 91 roadmap rows, found {len(rows)}")
    bad = [(wp, status) for wp, status in rows if status not in {"⬜", "🔄", "✅", "⛔", "⏸", "⊘"}]
    if bad:
        violations.append(f"invalid status values: {bad}")
    return violations


def check_roadmap_header_count(docs_dir: Path, repo_root: Path) -> list[str]:
    """The opening paragraph's package count matches the numbered rows it describes.

    That count is prose, so nothing forces it to move when rows were added: it
    read "69 work packages" while the table carried 91, and the id and status
    checks both passed the whole time because neither of them reads the
    sentence. A documented number with no gate is a number that decays.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     _ = (docs / "ROADMAP.md").write_text(
        ...         "| 001 | a |\\n| 002 | a |\\nStates 5 numbered work packages.\\n", encoding="utf-8"
        ...     )
        ...     check_roadmap_header_count(docs, docs)
        ['header states 5 numbered work packages but the table carries 2']
    """
    text = (docs_dir / "ROADMAP.md").read_text(encoding="utf-8")
    numbered = len(re.findall(r"^\| \d{3} \|", text, flags=re.MULTILINE))
    stated = re.search(r"(\d+) numbered work packages", text)

    if stated is None:
        return ["the header must state the numbered work-package count"]
    if int(stated.group(1)) == numbered:
        return []
    return [f"header states {stated.group(1)} numbered work packages but the table carries {numbered}"]


def check_provenance_ids(docs_dir: Path, repo_root: Path) -> list[str]:
    """PROVENANCE.md defines every source id R1..R21 used in commit trailers.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     _ = (docs / "PROVENANCE.md").write_text("| R1 | ... |\\n| R2 | ... |\\n", encoding="utf-8")
        ...     violations = check_provenance_ids(docs, docs)
        ...     violations[0].startswith("provenance missing source ids")
        True
    """
    text = (docs_dir / "PROVENANCE.md").read_text(encoding="utf-8")
    ids = {int(m) for m in re.findall(r"^\| R(\d+) \|", text, flags=re.MULTILINE)}
    missing = sorted(set(range(1, 22)) - ids)
    if not missing:
        return []
    return [f"provenance missing source ids: {missing}"]


def check_report_sections(docs_dir: Path, repo_root: Path) -> list[str]:
    """REPRODUCTION_REPORT.md carries every tier section plus the WP-065 consolidation.

    The report is append-only (D10): a later release corrects an earlier claim by adding to
    it, never by editing the record away, so no existing heading may ever be renamed out of
    the file or dropped. This check reads only the headings, not their content, which is what
    keeps it compatible with that discipline -- a correction landing inside a section leaves
    every heading exactly as it was, and this check unaffected.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     _ = (docs / "REPRODUCTION_REPORT.md").write_text("## 0.1.0 — Detection\\n", encoding="utf-8")
        ...     violations = check_report_sections(docs, docs)
        ...     violations[0].startswith("REPRODUCTION_REPORT.md missing section headings")
        True
    """
    text = (docs_dir / "REPRODUCTION_REPORT.md").read_text(encoding="utf-8")
    headings = {line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("## ")}
    titles = {_undecorated(heading) for heading in headings}
    missing = [title for title in _REPORT_SECTIONS if title not in titles]
    if not missing:
        return []
    return [f"REPRODUCTION_REPORT.md missing section headings: {missing}"]


def check_decisions_ids(docs_dir: Path, repo_root: Path) -> list[str]:
    """DECISIONS.md numbers its decisions contiguously from D1, never shrinks, and keeps every ADR.

    Two properties, deliberately kept separate. Contiguity catches a duplicated or
    skipped id, and it holds however many decisions the register grows to -- a
    hardcoded upper bound would fail every time one is added, which is a check
    demanding maintenance rather than reporting a defect. The floor is what a bare
    contiguity check would miss: dropping the *last* row leaves the remainder
    perfectly contiguous, so the count is asserted never to fall below what the
    register has already reached. Raise the floor when adding a decision; that edit
    is the deliberate act, not a chore.

    The floor counts rows, matching :data:`_WP_FLOOR` and :data:`_ASSUMPTION_FLOOR`.
    It read the highest id until WP-168, which graded the register by its last row
    and so stayed silent on a row deleted from the middle -- the one case the
    contiguity check beside it cannot cover either, since deleting from the tail is
    what leaves the remainder contiguous.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     docs = Path(tmp)
        ...     _ = (docs / "DECISIONS.md").write_text("| D1 | ... |\\n", encoding="utf-8")
        ...     violations = check_decisions_ids(docs, docs)
        ...     violations[0].startswith("decisions shrank below 20 rows")
        True
    """
    text = (docs_dir / "DECISIONS.md").read_text(encoding="utf-8")
    d_ids = {int(m) for m in re.findall(r"^\| D(\d+) \|", text, flags=re.MULTILINE)}
    if not d_ids:
        return ["no decision rows found"]

    violations = []
    if d_ids != set(range(1, max(d_ids) + 1)):
        violations.append(f"decision ids are not contiguous from 1: {sorted(d_ids)}")
    if len(d_ids) < _DECISION_FLOOR:
        violations.append(f"decisions shrank below {_DECISION_FLOOR} rows: {len(d_ids)}")
    for adr in ("ADR-001", "ADR-002", "ADR-003", "ADR-004", "ADR-005"):
        if not re.search(rf"^## .*\b{adr}\b", text, flags=re.MULTILINE):
            violations.append(f"missing {adr} section")
    return violations


#: Every check, in the order the original test file declared them.
#: Licence families the pull-request attestation must name, keyed by the label the
#: template groups them under. Every family here is one D13 excludes from the permissive
#: allowlist, and the point of pinning them is that the attestation narrows silently: the
#: line named Ultralytics alone for eight months while meaning all of this, and a reader
#: could only have learned the real rule by finding D13 themselves.
_ATTESTATION_FAMILIES = ("AGPL", "GPL", "LGPL", "SSPL", "BSL", "Elastic", "PolyForm")

#: The two cases the families miss, checked as substrings for the same reason: neither is
#: a licence name, and both are the kind of clause that falls out of a rewrite unnoticed.
_ATTESTATION_CLAUSES = ("proprietary", "cannot be read")


def check_pull_request_attestation_covers_the_allowlist(docs_dir: Path, repo_root: Path) -> list[str]:
    """The pull-request template's clean-room line names the whole excluded set (WP-143).

    The admissible set is D13's permissive allowlist, so what a contributor attests to
    not having copied from is everything outside it. The template named the Ultralytics
    denylist alone, which is one instance of the rule stated as though it were the rule:
    a contributor reading it would conclude a GPL detector was fair game.

    ``docs_dir`` is unused and present so the check matches the signature every other
    check in this module carries, which is what lets :data:`_CHECKS` stay a plain tuple.

    Args:
        docs_dir: Unused; part of the shared check signature.
        repo_root: Repository root holding ``.github/PULL_REQUEST_TEMPLATE.md``.

    Returns:
        One violation naming every family or clause the template fails to mention.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     (root / ".github").mkdir()
        ...     _ = (root / ".github" / "PULL_REQUEST_TEMPLATE.md").write_text("- [ ] no Ultralytics")
        ...     check_pull_request_attestation_covers_the_allowlist(root / "docs", root)[0][:46]
        'pull-request attestation does not name: AGPL, '
    """
    del docs_dir
    template = repo_root / ".github" / "PULL_REQUEST_TEMPLATE.md"
    if not template.is_file():
        return [f"missing pull-request template: {template.relative_to(repo_root)}"]
    text = template.read_text(encoding="utf-8")
    missing = [family for family in _ATTESTATION_FAMILIES if family not in text]
    missing += [clause for clause in _ATTESTATION_CLAUSES if clause not in text.lower()]
    if missing:
        return [f"pull-request attestation does not name: {', '.join(missing)}"]
    return []


_CHECKS = (
    check_policy_docs_exist,
    check_pull_request_attestation_covers_the_allowlist,
    check_model_cards_are_required,
    check_log_links_resolve,
    check_assumption_ids_contiguous,
    check_assumption_rows_complete,
    check_roadmap_wp_ids_unique_and_complete,
    check_roadmap_rows_complete,
    check_roadmap_statuses_valid,
    check_roadmap_header_count,
    check_provenance_ids,
    check_report_sections,
    check_decisions_ids,
)


def find_violations(docs_dir: Path, repo_root: Path) -> list[str]:
    """Run every governance-register check and collect their violation strings.

    A check whose source file is entirely absent (rather than merely malformed) raises
    ``FileNotFoundError`` instead of returning a violation list; that is caught here and
    turned into one so a single missing file doesn't hide every other check's findings.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     (root / "docs").mkdir()
        ...     find_violations(root / "docs", root)[0].startswith("missing policy docs")
        True
    """
    violations = []
    for check in _CHECKS:
        try:
            violations.extend(check(docs_dir, repo_root))
        except FileNotFoundError as exc:
            violations.append(f"{check.__name__}: {exc}")
    return violations


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run every check, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` when any check reports a violation.
    """
    parser = argparse.ArgumentParser(description="Audit docs/ governance registers for structural violations.")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help=f"docs directory to scan (default: {DEFAULT_DOCS_DIR})",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=f"repo root, for AGENTS.md and relative-path reporting (default: {REPO_ROOT})",
    )
    args = parser.parse_args(argv)

    violations = find_violations(args.docs_dir, args.repo_root)
    if violations:
        print(f"docs-present audit FAILED: {len(violations)} violation(s)")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("docs-present audit clean: every governance register check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
