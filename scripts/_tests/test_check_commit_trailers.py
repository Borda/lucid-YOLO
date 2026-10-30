# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: commit-trailer validator (WP-004).

Guards the provenance-carrying commit contract (AGENTS.md sec. 5): a valid message passes, a missing WP trailer fails,
an unknown provenance id fails, a hash before the separator fails, an at-sign confined to the co-author trailers after
the separator passes, and the subject's type comes from the AGENTS.md sec. 5 allowed set.

Three classes of *resolution* failure are covered beside the shape ones, each one a gate that read a well-formed id as
evidence that a row exists for it: a denial of provenance read as a citation of the source it denies, a withdrawn or
citation-only placeholder row admitted as an allowlist row, and an assumption id matched against a regex instead of the
register.
"""

import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from commit_trailer_messages import make_message
from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_PATH = REPO_ROOT / "docs" / "PROVENANCE.md"
ASSUMPTIONS_PATH = REPO_ROOT / "docs" / "ASSUMPTIONS.md"

#: Builds a throwaway repository from messages (or ``(message, author)`` pairs), oldest first, returning the hashes.
RepositoryBuilder = Callable[[Sequence[str | tuple[str, str]]], list[str]]

#: The author a GitHub App installation commits under, as the bot branch behind pull request 4 carried it. Range mode
#: skips this author; the tests below build such a commit beside human ones and check that only the human ones are
#: validated.
BOT_AUTHOR = "pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>"
#: What pre-commit.ci writes: no conventional type, no trailer of any kind.
BOT_MESSAGE = "[pre-commit.ci] pre-commit autoupdate\n\nupdates:\n- ruff-pre-commit: v0.16.1 -> v0.16.7\n"

#: A contract-satisfying message with a substitutable subject, for the range-mode repositories below. ``R1`` is read
#: from the real ``docs/PROVENANCE.md``, which is what the validator loads its allowlist from whatever repository it is
#: walking.
RANGE_MESSAGE_TEMPLATE = (
    "{subject}\n\nOne paragraph of body.\n\nWP: 144\nProvenance: R1\nAssumptions: none\nGate: t.py::x\n"
)
#: A message whose subject opens with no conventional type, so range and unpushed modes must refuse it.
UNTYPED_MESSAGE = make_message("no conventional type on this subject", wp="144", gate="t.py::x")

validator = load_script("check_commit_trailers")
VALID_IDS = validator.load_registers(PROVENANCE_PATH, ASSUMPTIONS_PATH)


def _registers(sources: set[int], placeholders: set[int] = frozenset(), assumptions: set[int] = frozenset()) -> object:
    """Build a synthetic register triple, so a case states the ids it depends on.

    Args:
        sources: Admissible source ids.
        placeholders: Declared-but-inadmissible source ids.
        assumptions: Registered assumption ids.

    Returns:
        The validator's own ``Registers`` instance.

    Examples:
        >>> _registers({1}).sources
        frozenset({1})
    """
    return validator.Registers(frozenset(sources), frozenset(placeholders), frozenset(assumptions))


# The canonical example from AGENTS.md sec. 5.
VALID_MESSAGE = """feat(models): add dual detection head with reg_max=1

Implements the one-to-one (topk=7 -> topk2=1, 300 outputs) and
one-to-many (topk=10, dense) branches sharing neck features, with
direct 4-scalar ltrb regression and no DFL module.

WP: 022
Provenance: R1 3.2.1, R1 3.2.2, R1 Fig. S2, R6
Assumptions: A9
Gate: tests/models/test_head.py::test_dual_head_shapes
"""

MISSING_WP_MESSAGE = """feat(models): add dual detection head with reg_max=1

Provenance: R1 3.2.1, R6
Assumptions: A9
Gate: tests/models/test_head.py::test_dual_head_shapes
"""

UNKNOWN_PROVENANCE_MESSAGE = """feat(models): add dual detection head with reg_max=1

WP: 022
Provenance: R99 4.1
Assumptions: A9
Gate: tests/models/test_head.py::test_dual_head_shapes
"""

HASH_IN_BODY_MESSAGE = """feat(models): add dual detection head

Closes issue #42 from the tracker.

WP: 022
Provenance: R1
Assumptions: none
Gate: tests/models/test_head.py::test_dual_head_shapes
"""

AT_SIGN_AFTER_SEPARATOR_MESSAGE = """feat(models): add dual detection head

WP: 022
Provenance: R1
Assumptions: none
Gate: tests/models/test_head.py::test_dual_head_shapes

---
Co-authored-by: Claude <209825114+claude[bot]@users.noreply.github.com>
"""

HEADING_IN_BODY_MESSAGE = """feat(models): add dual detection head

## What changed

The head, as described. See section 3 of the paper, not issue tracker entries.

WP: 022
Provenance: R1
Assumptions: none
Gate: tests/models/test_head.py::test_dual_head_shapes
"""

SIGNOFF_BEFORE_SEPARATOR_MESSAGE = """feat(models): add dual detection head

WP: 022
Provenance: R1
Assumptions: none
Gate: tests/models/test_head.py::test_dual_head_shapes

Signed-off-by: Ada Lovelace <ada@example.com>

---
Co-authored-by: Claude <209825114+claude[bot]@users.noreply.github.com>
"""

MENTION_BESIDE_SIGNOFF_MESSAGE = """feat(models): add dual detection head

Thanks @ada for the review.

WP: 022
Provenance: R1
Assumptions: none
Gate: tests/models/test_head.py::test_dual_head_shapes

Signed-off-by: Ada Lovelace <ada@example.com>
"""


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(VALID_MESSAGE, id="canonical-agents-md-example"),
        pytest.param(HEADING_IN_BODY_MESSAGE, id="markdown-heading-in-body"),
        pytest.param(AT_SIGN_AFTER_SEPARATOR_MESSAGE, id="at-sign-after-separator"),
        pytest.param(SIGNOFF_BEFORE_SEPARATOR_MESSAGE, id="signoff-before-separator"),
    ],
)
def test_well_formed_message_reports_no_violation(message: str) -> None:
    """A message that follows the AGENTS.md section 5 contract reports no violation.

    Four shapes pass: the canonical example; a markdown heading in the body, which links nothing, so its hash-signs
    are not an issue reference; an at-sign confined to the co-author trailers after the separator; and the sign-off
    ``git commit -s`` appends before the separator.
    """
    assert validator.validate_message(message, VALID_IDS) == []


@pytest.mark.parametrize(
    ("message", "marker"),
    [
        pytest.param(MISSING_WP_MESSAGE, "WP", id="missing-wp-trailer"),
        pytest.param(UNKNOWN_PROVENANCE_MESSAGE, "R99", id="provenance-id-absent-from-allowlist"),
        pytest.param(HASH_IN_BODY_MESSAGE, "#", id="issue-reference-before-separator"),
        pytest.param(MENTION_BESIDE_SIGNOFF_MESSAGE, "'@'", id="mention-beside-signoff"),
    ],
)
def test_malformed_message_names_violation(message: str, marker: str) -> None:
    """A message breaking the contract is rejected and the violation names the offending part.

    A missing WP trailer, a provenance id absent from the allowlist, an issue reference in the body before the
    separator, and a mention on a body line next to a sign-off: the sign-off line exempts itself only.
    """
    violations = validator.validate_message(message, VALID_IDS)

    assert any(marker in violation for violation in violations)


@pytest.mark.parametrize(
    ("subject", "accepted"),
    [
        pytest.param("feat(models): add dual detection head with reg_max=1", True, id="feat"),
        pytest.param("refine(configs): name tiers by what they are", True, id="refine"),
        pytest.param("chore: bump the dev version", True, id="chore"),
        pytest.param("improve(models): make the head nicer", False, id="unlisted-type"),
        pytest.param("add dual detection head", False, id="no-type"),
    ],
)
def test_subject_type_checked_against_allowed_set(subject: str, accepted: bool) -> None:
    """Only the listed conventional types open a subject line.

    ``refine`` is this project's own addition and is the reason this test exists: it was rejected by the validator
    while already being used in the history, so the allowed set and the set actually in use had drifted apart with
    nothing watching. An unlisted type and a type-less subject pin the other side.
    """
    message = VALID_MESSAGE.replace(VALID_MESSAGE.splitlines()[0], subject, 1)

    violations = validator.validate_message(message, VALID_IDS)

    assert any("<type>(<scope>)" in violation for violation in violations) is not accepted


class TestProvenanceDenial:
    """`Provenance: none` is a value, and an id inside its explanation is not a citation."""

    @pytest.mark.parametrize(
        "provenance",
        [
            "none",
            pytest.param("none (implementation efficiency; R1 silent on target rasterisation)", id="with-explanation"),
        ],
    )
    def test_denial_accepted(self, provenance: str) -> None:
        """`Provenance: none` and `none (…)` pass, as `WP: none` and `Assumptions: none` already do.

        The trailer stays mandatory; what changes is that stating "nothing derives from a registered source here" is a
        way of satisfying it rather than a way of failing it. Before this, the only accepted way to say `none` was to
        also name a source, which is the opposite of what the word means. The explanation may name a source it is
        denying.
        """
        message = make_message("fix(data): correct the pad value", body="body", provenance=provenance)

        assert validator.validate_message(message, _registers({1})) == []

    def test_id_inside_denial_resolved(self) -> None:
        """A denial explaining itself against an unknown id is refused, not waved through.

        The id is a reference rather than a citation, but a reference to a row that does not exist is still a broken
        reference.
        """
        message = make_message(provenance="none (R99 is silent here)")

        violations = validator.validate_message(message, _registers({1}))

        assert any("R99" in violation for violation in violations)

    def test_trailer_naming_no_source_and_not_denying_refused(self) -> None:
        """Free text that neither cites nor denies remains a violation."""
        message = make_message(provenance="the usual place")

        violations = validator.validate_message(message, _registers({1}))

        assert any("lists no source id" in violation for violation in violations)


class TestPlaceholderRows:
    """The Placeholders table shares the allowlist's row shape and admits nothing."""

    def test_real_register_admits_no_placeholder_row(self) -> None:
        """R15 (withdrawn) and R19 (legal evidence only) parse as declared, not admitted.

        Read file-wide, `^| R(\\d+) |` returns both tables as one allowlist, which is how `Provenance: R15` -- naming
        the denylisted documentation site -- validated clean. The two ids are asserted by name because they are the
        two rows the register actually holds, not synthetic stand-ins.
        """
        assert {15, 19} <= VALID_IDS.placeholders
        assert not {15, 19} & VALID_IDS.sources
        assert 1 in VALID_IDS.sources

    def test_citing_placeholder_row_refused_as_inadmissible(self) -> None:
        """A message citing R15 fails, and the message says declared rather than unknown.

        Naming an id the register deliberately refuses is a different mistake from naming one it has never heard of,
        and a checker that reports "unknown" for the first sends the author to add a row that is already there.
        """
        message = make_message(provenance="R15")

        violations = validator.validate_message(message, _registers({1}, placeholders={15}))

        assert any("inadmissible" in violation and "R15" in violation for violation in violations)

    def test_real_history_carries_two_citations_of_placeholder(self) -> None:
        """Two pre-enforcement commits cite R19, so the hole this closes was load-bearing.

        Asserted as a property of the parse rather than by walking git: R19 is registered "legal-evidence citation
        only, never an implementation source", so any provenance citation of it must be refused whatever the history
        holds.
        """
        message = make_message("test(eval): x", provenance="R12, R19 (pycocotools)")

        violations = validator.validate_message(message, VALID_IDS)

        assert any("R19" in violation for violation in violations)


class TestAssumptionRegister:
    """An assumption id is resolved against `docs/ASSUMPTIONS.md`, not against a regex."""

    def test_unregistered_assumption_id_refused(self) -> None:
        """`A999` is well-formed and names no row, which is the whole of the defect."""
        message = make_message(assumptions="A999")

        violations = validator.validate_message(message, _registers({1}, assumptions={9}))

        assert any("A999" in violation and "ASSUMPTIONS" in violation for violation in violations)

    def test_registered_assumption_id_passes(self) -> None:
        """The register's own ids still validate, against the real register."""
        message = make_message(assumptions="A9, A14")

        assert validator.validate_message(message, VALID_IDS) == []

    def test_malformed_id_shape_violation(self) -> None:
        """Shape and membership stay separate messages; `A9x` never reaches the register."""
        message = make_message(assumptions="A9x")

        violations = validator.validate_message(message, _registers({1}, assumptions={9}))

        assert any("invalid assumption ids" in violation for violation in violations)


@pytest.fixture
def throwaway_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RepositoryBuilder:
    """Build a git repository holding the given messages and point the validator at it.

    Each entry is a message, or a ``(message, author)`` pair for a commit that should carry an author other than the
    fixture's own -- the bot case.

    ``REPO_ROOT`` is redirected to the new repository, so ``--range`` walks that rather than this checkout;
    ``DEFAULT_PROVENANCE`` is bound at import and keeps reading the real allowlist, which is what the messages are
    validated against. Git's global and system configuration is routed to ``os.devnull`` and the repository is created
    from an empty template, so a developer's ``core.hooksPath``, commit template or signing settings cannot reach a
    repository built to be predictable.

    Args:
        tmp_path: Per-test temporary directory.
        monkeypatch: Redirects ``REPO_ROOT`` and isolates git's configuration.

    Returns:
        A callable taking the commit messages (or message-author pairs), oldest first, and returning their hashes in the
        same order.
    """
    repo = tmp_path / "repository"
    template = tmp_path / "empty-git-template"
    template.mkdir()
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)

    def _git(*argv: str, stdin: str | None = None) -> str:
        result = subprocess.run(["git", *argv], cwd=repo, input=stdin, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def _build(messages: Sequence[str | tuple[str, str]]) -> list[str]:
        subprocess.run(
            ["git", "init", "--quiet", f"--template={template}", "--initial-branch=main", str(repo)],
            capture_output=True,
            check=True,
        )
        _git("config", "user.name", "Trailer Fixture")
        _git("config", "user.email", "fixture@example.invalid")
        _git("config", "commit.gpgsign", "false")
        shas = []
        for index, entry in enumerate(messages):
            message, author = entry if isinstance(entry, tuple) else (entry, None)
            (repo / f"file_{index}.txt").write_text(f"{index}\n", encoding="utf-8")
            _git("add", f"file_{index}.txt")
            author_flag = [f"--author={author}"] if author else []
            _git("commit", "--quiet", *author_flag, "--file", "-", stdin=message)
            shas.append(_git("rev-parse", "HEAD"))
        monkeypatch.setattr(validator, "REPO_ROOT", repo)
        return shas

    return _build


class TestContributorRange:
    """Range mode and the D18 trailer form, the two things CI relies on (WP-144)."""

    def test_wp_none_accepted(self) -> None:
        """`WP: none` is the D18 form: a change that needs no roadmap row still cites.

        Once the reproduction is complete and the repository public, a change altering no shipped behaviour, no public
        symbol, no golden and no documented assumption lands without a tracked row. The other three trailers stay
        mandatory, so the derivation question survives the tracking relaxation.
        """
        message = make_message("fix(data): correct the pad value", body="body", gate="t.py::x")

        assert validator.validate_message(message, _registers({1})) == []

    def test_missing_wp_trailer_refused(self) -> None:
        """Omission is not the relaxation -- `none` states it, absence leaves it unsaid."""
        message = make_message("fix(data): correct the pad value", body="body", wp=None, gate="t.py::x")

        violations = validator.validate_message(message, _registers({1}))

        assert any("WP" in violation for violation in violations)

    @pytest.mark.parametrize("row", ["197c", "22", "192b"])
    def test_lettered_wp_row_accepted(self, row: str) -> None:
        """A lettered roadmap row such as 197c is citable as itself, not as 197."""
        message = make_message(wp=row, gate="t.py::x")

        assert validator.validate_message(message, _registers({1})) == []

    @pytest.mark.parametrize("row", ["197C", "197cd", "c197", "197-c", "maybe"])
    def test_malformed_wp_row_refused(self, row: str) -> None:
        """Only digits with at most one lowercase letter after them, or the literal `none`, are row ids.

        Anything else is not a row id: a capital, two letters, a leading letter, a hyphenated suffix, or free text.
        """
        message = make_message(wp=row, gate="t.py::x")

        assert any("WP" in violation for violation in validator.validate_message(message, _registers({1})))

    def test_empty_range_clean(
        self, capsys: pytest.CaptureFixture[str], throwaway_repository: RepositoryBuilder
    ) -> None:
        """A range selecting no commit passes, exercising the real git path.

        The range runs from one commit to itself, so it is empty by construction.
        """
        shas = throwaway_repository([RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range")])

        status = validator.main(["--range", f"{shas[0]}..{shas[0]}"])

        assert status == 0
        assert "0 message(s) validated" in capsys.readouterr().out

    def test_range_mode_accepts_history_of_valid_messages(self, throwaway_repository: RepositoryBuilder) -> None:
        """Range mode walks git and reports clean when every message satisfies the contract.

        This ran against this repository's own last ten commits until the coupling was removed. Two mechanisms broke
        it, one proven and one suspected. The proven one: ``ci-tests.yml`` checks out at the default depth of one, so
        ``HEAD~10`` names an object the clone does not contain and ``git log`` exits 128 — deterministically, on every
        run, and equally in any contributor's shallow checkout. The suspected one: a local run overlapping a history
        rewrite, which is what the earlier ``HEAD``-resolving form was changed to avoid. Deepening the CI clone would
        have answered the first and left the second, and would have left the test asserting against a history that
        changes under it either way. A repository built for the test answers both, exercises the same ``git log``
        path, and additionally covers the refusal case below, which the live-history assertion never could.

        Real history keeps two guards that do not depend on this test: the pre-commit hook validates unpushed commits
        on every commit, and ``lint.yml``'s ``trailers`` job validates a pull request's own commits.
        """
        shas = throwaway_repository(
            [
                RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range"),
                RANGE_MESSAGE_TEMPLATE.format(subject="feat(models): add a dual detection head"),
                RANGE_MESSAGE_TEMPLATE.format(subject="fix(data): correct the pad value"),
            ]
        )

        assert validator.main(["--range", f"{shas[0]}..{shas[-1]}"]) == 0

    def test_unresolvable_range_reports_verdict_rather_than_traceback(
        self, capsys: pytest.CaptureFixture[str], throwaway_repository: RepositoryBuilder
    ) -> None:
        """A range naming a ref this checkout does not hold fails with a sentence, not a stack trace.

        The shallow-checkout case, which is how CI met it: a default-depth checkout has no ``refs/remotes/origin/main``,
        so the hook's own ``origin/main..HEAD`` exits 128. Fetching full history fixes the CI half and leaves a
        contributor's shallow clone, where the checker crashing is the difference between a fix they can act on and a
        traceback they file an issue about.
        """
        throwaway_repository([RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range")])

        status = validator.main(["--range", "origin/main..HEAD"])

        out = capsys.readouterr().out
        assert status == 1
        assert "cannot resolve range" in out
        assert "shallow clone" in out

    @staticmethod
    def _git_config(*pairs: tuple[str, str]) -> None:
        """Write branch and remote configuration into the throwaway repository."""
        for key, value in pairs:
            subprocess.run(["git", "config", key, value], cwd=validator.REPO_ROOT, capture_output=True, check=True)

    def test_unpushed_fork_checkout_falls_back_to_remotes(
        self, capsys: pytest.CaptureFixture[str], throwaway_repository: RepositoryBuilder
    ) -> None:
        """A URL upstream, as ``gh pr checkout`` writes for a fork, is checked rather than refused.

        ``branch.<name>.remote`` set to a URL stores no remote-tracking branch, so ``@{upstream}`` cannot resolve; the
        hook used to exit 1 there on every commit.
        """
        throwaway_repository(
            [
                RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range"),
                RANGE_MESSAGE_TEMPLATE.format(subject="fix(data): correct the pad value"),
            ]
        )
        self._git_config(
            ("branch.main.remote", "https://example.invalid/fork/repository.git"),
            ("branch.main.merge", "refs/heads/main"),
        )

        status = validator.main(["--unpushed"])

        out = capsys.readouterr().out
        assert status == 0
        assert "'HEAD --not --remotes'" in out
        assert "2 message(s) validated" in out

    def test_unpushed_never_pushed_branch_refuses_invalid_message(
        self, capsys: pytest.CaptureFixture[str], throwaway_repository: RepositoryBuilder
    ) -> None:
        """A branch with no upstream at all is the unpushed work the mode exists for, so it is validated."""
        shas = throwaway_repository([RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range"), UNTYPED_MESSAGE])

        status = validator.main(["--unpushed"])

        out = capsys.readouterr().out
        assert status == 1
        assert "1 of 2 message(s) invalid" in out
        assert shas[-1][:12] in out

    def test_unpushed_tracking_branch_checks_only_commits_ahead(
        self, capsys: pytest.CaptureFixture[str], throwaway_repository: RepositoryBuilder
    ) -> None:
        """With a resolvable upstream, commits the tracked branch already holds are not rechecked.

        The pushed commit carries an invalid message on purpose: a squash merge replaces it, so reaching it would
        refuse a published branch for a message that never lands.
        """
        shas = throwaway_repository(
            [UNTYPED_MESSAGE, RANGE_MESSAGE_TEMPLATE.format(subject="fix(data): correct the pad value")]
        )
        self._git_config(
            ("remote.origin.url", "https://example.invalid/upstream/repository.git"),
            ("remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*"),
            ("branch.main.remote", "origin"),
            ("branch.main.merge", "refs/heads/main"),
        )
        subprocess.run(
            ["git", "update-ref", "refs/remotes/origin/main", shas[0]],
            cwd=validator.REPO_ROOT,
            capture_output=True,
            check=True,
        )

        status = validator.main(["--unpushed"])

        out = capsys.readouterr().out
        assert status == 0
        assert "'@{upstream}..HEAD'" in out
        assert "1 message(s) validated" in out

    def test_range_mode_refuses_invalid_message_in_range(
        self, capsys: pytest.CaptureFixture[str], throwaway_repository: RepositoryBuilder
    ) -> None:
        """One bad message anywhere in the range fails the whole range, naming that commit.

        The walk had no failing case at all before: an empty range and a clean range both return zero, so a validator
        that reported clean unconditionally passed both.
        """
        shas = throwaway_repository(
            [
                RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range"),
                RANGE_MESSAGE_TEMPLATE.format(subject="feat(models): add a dual detection head"),
                UNTYPED_MESSAGE,
            ]
        )

        status = validator.main(["--range", f"{shas[0]}..{shas[-1]}"])

        out = capsys.readouterr().out
        assert status == 1
        assert "1 of 2 message(s) invalid" in out
        assert shas[-1][:12] in out

    def test_range_mode_skips_commit_automation_authored(
        self, capsys: pytest.CaptureFixture[str], throwaway_repository: RepositoryBuilder
    ) -> None:
        """A bot-authored commit is skipped by name; the human commits beside it are still checked.

        Pull request 4 is the case: pre-commit.ci's autoupdate commit carries no conventional subject and no trailer,
        and nobody wrote it. Refusing it would block every dependency-pin bump the service opens; validating a message
        no person wrote checks nothing. The skip is printed so the verdict says what was left out, and a human commit
        in the same range still fails on its own merits.
        """
        shas = throwaway_repository(
            [
                RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range"),
                (BOT_MESSAGE, BOT_AUTHOR),
                RANGE_MESSAGE_TEMPLATE.format(subject="fix(data): correct the pad value"),
            ]
        )

        status = validator.main(["--range", f"{shas[0]}..{shas[-1]}"])

        out = capsys.readouterr().out
        assert status == 0
        assert f"COMMIT {shas[1][:12]}: SKIP (automation author" in out
        assert "1 message(s) validated, 1 automation commit(s) skipped" in out

    def test_human_commit_beside_skipped_bot_commit_refused(
        self, capsys: pytest.CaptureFixture[str], throwaway_repository: RepositoryBuilder
    ) -> None:
        """The skip is per author, not per range: a bad human message next to a bot one still fails."""
        shas = throwaway_repository(
            [
                RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range"),
                (BOT_MESSAGE, BOT_AUTHOR),
                UNTYPED_MESSAGE,
            ]
        )

        status = validator.main(["--range", f"{shas[0]}..{shas[-1]}"])

        out = capsys.readouterr().out
        assert status == 1
        assert "1 of 1 message(s) invalid, 1 automation commit(s) skipped" in out
        assert shas[-1][:12] in out

    def test_bot_address_outside_github_noreply_not_skipped(
        self, capsys: pytest.CaptureFixture[str], throwaway_repository: RepositoryBuilder
    ) -> None:
        """Only GitHub's own App address is an automation; ``[bot]`` in a name alone is not.

        The skip keys on the address suffix GitHub assigns, so a display name ending in ``[bot]`` under some other
        domain is a contributor and is validated as one.
        """
        shas = throwaway_repository(
            [
                RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range"),
                (BOT_MESSAGE, "someone[bot] <bot@example.invalid>"),
            ]
        )

        status = validator.main(["--range", f"{shas[0]}..{shas[-1]}"])

        out = capsys.readouterr().out
        assert status == 1
        assert "SKIP" not in out
        assert "1 of 1 message(s) invalid" in out
