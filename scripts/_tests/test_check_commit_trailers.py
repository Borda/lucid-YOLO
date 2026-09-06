# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: commit-trailer validator (WP-004).

Guards the provenance-carrying commit contract (AGENTS.md sec. 5): a valid
message passes, a missing WP trailer fails, an unknown provenance id fails, a
hash before the separator fails, an at-sign confined to the co-author trailers
after the separator passes, and the subject's type comes from the AGENTS.md
sec. 5 allowed set.

Three classes of *resolution* failure are covered beside the shape ones, each
one a gate that read a well-formed id as evidence that a row exists for it: a
denial of provenance read as a citation of the source it denies, a withdrawn or
citation-only placeholder row admitted as an allowlist row, and an assumption id
matched against a regex instead of the register.
"""

import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "lint" / "check_commit_trailers.py"
PROVENANCE_PATH = REPO_ROOT / "docs" / "PROVENANCE.md"
ASSUMPTIONS_PATH = REPO_ROOT / "docs" / "ASSUMPTIONS.md"

#: A contract-satisfying message with a substitutable subject, for the range-mode
#: repositories below. ``R1`` is read from the real ``docs/PROVENANCE.md``, which is
#: what the validator loads its allowlist from whatever repository it is walking.
RANGE_MESSAGE_TEMPLATE = (
    "{subject}\n\nOne paragraph of body.\n\nWP: 144\nProvenance: R1\nAssumptions: none\nGate: t.py::x\n"
)


def _load_validator() -> ModuleType:
    """Load ``scripts/lint/check_commit_trailers.py`` as an importable module.

    Registered in ``sys.modules`` before execution rather than after: under
    ``from __future__ import annotations`` a dataclass body's annotations are
    strings, and ``dataclasses`` resolves them through ``sys.modules[__module__]``
    while the decorator runs. A module absent from that table at decoration time
    fails there with an ``AttributeError`` on ``None``.

    Examples:
        >>> module = _load_validator()
        >>> module.__name__
        'check_commit_trailers'
        >>> callable(module.validate_message)
        True
    """
    spec = importlib.util.spec_from_file_location("check_commit_trailers", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()
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


def test_valid_message_passes() -> None:
    """The canonical AGENTS.md section 5 message reports no violation."""
    assert validator.validate_message(VALID_MESSAGE, VALID_IDS) == []


def test_missing_wp_trailer_fails() -> None:
    """A message without a WP trailer is rejected, naming the WP trailer."""
    violations = validator.validate_message(MISSING_WP_MESSAGE, VALID_IDS)
    assert any("WP" in violation for violation in violations)


def test_unknown_provenance_id_fails() -> None:
    """A provenance id absent from the allowlist is rejected."""
    violations = validator.validate_message(UNKNOWN_PROVENANCE_MESSAGE, VALID_IDS)
    assert any("R99" in violation for violation in violations)


def test_hash_before_separator_fails() -> None:
    """A hash-sign in the body before the separator is rejected."""
    violations = validator.validate_message(HASH_IN_BODY_MESSAGE, VALID_IDS)
    assert any("#" in violation for violation in violations)


def test_at_sign_after_separator_passes() -> None:
    """An at-sign confined to the co-author trailers is permitted."""
    assert validator.validate_message(AT_SIGN_AFTER_SEPARATOR_MESSAGE, VALID_IDS) == []


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
def test_subject_type_is_checked_against_the_allowed_set(subject: str, accepted: bool) -> None:
    """Only the listed conventional types open a subject line.

    ``refine`` is this project's own addition and is the reason this test exists:
    it was rejected by the validator while already being used in the history, so
    the allowed set and the set actually in use had drifted apart with nothing
    watching. An unlisted type and a type-less subject pin the other side.
    """
    message = VALID_MESSAGE.replace(VALID_MESSAGE.splitlines()[0], subject, 1)

    violations = validator.validate_message(message, VALID_IDS)

    assert any("<type>(<scope>)" in violation for violation in violations) is not accepted


def test_the_docstrings_name_the_script_where_it_lives() -> None:
    """The usage blocks cite `scripts/lint/`, the directory WP-129 moved this script into.

    Four printed occurrences named the pre-move path, in the two `::` literal blocks
    of the module docstring and the two of ``main``. Literal blocks are not `>>>`
    examples, so ``--doctest-modules`` never ran them and nothing else read them --
    a copy-pasteable command that has not resolved since the move.
    """
    assert "scripts/lint/check_commit_trailers.py" in validator.__doc__
    assert "scripts/check_commit_trailers.py" not in validator.__doc__
    assert "scripts/lint/check_commit_trailers.py" in validator.main.__doc__
    assert "scripts/check_commit_trailers.py" not in validator.main.__doc__


class TestProvenanceDenial:
    """`Provenance: none` is a value, and an id inside its explanation is not a citation."""

    def test_a_bare_denial_is_accepted(self) -> None:
        """`Provenance: none` passes, as `WP: none` and `Assumptions: none` already do.

        The trailer stays mandatory; what changes is that stating "nothing derives
        from a registered source here" is a way of satisfying it rather than a way of
        failing it. Before this, the only accepted way to say `none` was to also name
        a source, which is the opposite of what the word means.
        """
        message = "fix(data): correct the pad value\n\nbody\n\nWP: none\nProvenance: none\nAssumptions: none\nGate: t\n"

        assert validator.validate_message(message, _registers({1})) == []

    def test_a_denial_may_carry_its_explanation(self) -> None:
        """`none (…)` passes, and the explanation may name a source it is denying."""
        message = (
            "fix(data): correct the pad value\n\nbody\n\nWP: none\n"
            "Provenance: none (implementation efficiency; R1 silent on target rasterisation)\n"
            "Assumptions: none\nGate: t\n"
        )

        assert validator.validate_message(message, _registers({1})) == []

    def test_an_id_inside_a_denial_is_still_resolved(self) -> None:
        """A denial explaining itself against an unknown id is refused, not waved through.

        The id is a reference rather than a citation, but a reference to a row that
        does not exist is still a broken reference.
        """
        message = "fix(data): x\n\nWP: none\nProvenance: none (R99 is silent here)\nAssumptions: none\nGate: t\n"

        violations = validator.validate_message(message, _registers({1}))

        assert any("R99" in violation for violation in violations)

    def test_a_trailer_naming_no_source_and_not_denying_one_is_refused(self) -> None:
        """Free text that neither cites nor denies remains a violation."""
        message = "fix(data): x\n\nWP: none\nProvenance: the usual place\nAssumptions: none\nGate: t\n"

        violations = validator.validate_message(message, _registers({1}))

        assert any("lists no source id" in violation for violation in violations)


class TestPlaceholderRows:
    """The Placeholders table shares the allowlist's row shape and admits nothing."""

    def test_the_real_register_admits_no_placeholder_row(self) -> None:
        """R15 (withdrawn) and R19 (legal evidence only) parse as declared, not admitted.

        Read file-wide, `^| R(\\d+) |` returns both tables as one allowlist, which is
        how `Provenance: R15` -- naming the denylisted documentation site -- validated
        clean. The two ids are asserted by name because they are the two rows the
        register actually holds, not synthetic stand-ins.
        """
        assert {15, 19} <= VALID_IDS.placeholders
        assert not {15, 19} & VALID_IDS.sources
        assert 1 in VALID_IDS.sources

    def test_citing_a_placeholder_row_is_refused_as_inadmissible(self) -> None:
        """A message citing R15 fails, and the message says declared rather than unknown.

        Naming an id the register deliberately refuses is a different mistake from
        naming one it has never heard of, and a checker that reports "unknown" for the
        first sends the author to add a row that is already there.
        """
        message = "fix(data): x\n\nWP: none\nProvenance: R15\nAssumptions: none\nGate: t\n"

        violations = validator.validate_message(message, _registers({1}, placeholders={15}))

        assert any("inadmissible" in violation and "R15" in violation for violation in violations)

    def test_the_real_history_carries_two_citations_of_a_placeholder(self) -> None:
        """Two pre-enforcement commits cite R19, so the hole this closes was load-bearing.

        Asserted as a property of the parse rather than by walking git: R19 is
        registered "legal-evidence citation only, never an implementation source", so
        any provenance citation of it must be refused whatever the history holds.
        """
        message = "test(eval): x\n\nWP: none\nProvenance: R12, R19 (pycocotools)\nAssumptions: none\nGate: t\n"

        violations = validator.validate_message(message, VALID_IDS)

        assert any("R19" in violation for violation in violations)


class TestAssumptionRegister:
    """An assumption id is resolved against `docs/ASSUMPTIONS.md`, not against a regex."""

    def test_an_unregistered_assumption_id_is_refused(self) -> None:
        """`A999` is well-formed and names no row, which is the whole of the defect."""
        message = "fix(data): x\n\nWP: none\nProvenance: R1\nAssumptions: A999\nGate: t\n"

        violations = validator.validate_message(message, _registers({1}, assumptions={9}))

        assert any("A999" in violation and "ASSUMPTIONS" in violation for violation in violations)

    def test_a_registered_assumption_id_passes(self) -> None:
        """The register's own ids still validate, against the real register."""
        message = "fix(data): x\n\nWP: none\nProvenance: R1\nAssumptions: A9, A14\nGate: t\n"

        assert validator.validate_message(message, VALID_IDS) == []

    def test_a_malformed_id_is_still_a_shape_violation(self) -> None:
        """Shape and membership stay separate messages; `A9x` never reaches the register."""
        message = "fix(data): x\n\nWP: none\nProvenance: R1\nAssumptions: A9x\nGate: t\n"

        violations = validator.validate_message(message, _registers({1}, assumptions={9}))

        assert any("invalid assumption ids" in violation for violation in violations)

    def test_the_real_register_is_contiguous_from_one(self) -> None:
        """The parse returns the whole register rather than a section of it."""
        ids = sorted(VALID_IDS.assumptions)

        assert ids == list(range(1, len(ids) + 1))
        assert len(ids) >= 73


@pytest.fixture
def throwaway_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[Sequence[str]], list[str]]:
    """Build a git repository holding the given messages and point the validator at it.

    ``REPO_ROOT`` is redirected to the new repository, so ``--range`` walks that rather
    than this checkout; ``DEFAULT_PROVENANCE`` is bound at import and keeps reading the
    real allowlist, which is what the messages are validated against. Git's global and
    system configuration is routed to ``os.devnull`` and the repository is created from
    an empty template, so a developer's ``core.hooksPath``, commit template or signing
    settings cannot reach a repository built to be predictable.

    Args:
        tmp_path: Per-test temporary directory.
        monkeypatch: Redirects ``REPO_ROOT`` and isolates git's configuration.

    Returns:
        A callable taking the commit messages, oldest first, and returning their hashes
        in the same order.
    """
    repo = tmp_path / "repository"
    template = tmp_path / "empty-git-template"
    template.mkdir()
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)

    def _git(*argv: str, stdin: str | None = None) -> str:
        result = subprocess.run(["git", *argv], cwd=repo, input=stdin, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def _build(messages: Sequence[str]) -> list[str]:
        subprocess.run(
            ["git", "init", "--quiet", f"--template={template}", "--initial-branch=main", str(repo)],
            capture_output=True,
            check=True,
        )
        _git("config", "user.name", "Trailer Fixture")
        _git("config", "user.email", "fixture@example.invalid")
        _git("config", "commit.gpgsign", "false")
        shas = []
        for index, message in enumerate(messages):
            (repo / f"file_{index}.txt").write_text(f"{index}\n", encoding="utf-8")
            _git("add", f"file_{index}.txt")
            _git("commit", "--quiet", "--file", "-", stdin=message)
            shas.append(_git("rev-parse", "HEAD"))
        monkeypatch.setattr(validator, "REPO_ROOT", repo)
        return shas

    return _build


class TestContributorRange:
    """Range mode and the D18 trailer form, the two things CI relies on (WP-144)."""

    def test_wp_none_is_accepted(self) -> None:
        """`WP: none` is the D18 form: a change that needs no roadmap row still cites.

        Once the reproduction is complete and the repository public, a change altering
        no shipped behaviour, no public symbol, no golden and no documented assumption
        lands without a tracked row. The other three trailers stay mandatory, so the
        derivation question survives the tracking relaxation.
        """
        message = (
            "fix(data): correct the pad value\n\nbody\n\nWP: none\nProvenance: R1\nAssumptions: none\nGate: t.py::x\n"
        )

        assert validator.validate_message(message, _registers({1})) == []

    def test_a_missing_wp_trailer_is_still_refused(self) -> None:
        """Omission is not the relaxation -- `none` states it, absence leaves it unsaid."""
        message = "fix(data): correct the pad value\n\nbody\n\nProvenance: R1\nAssumptions: none\nGate: t.py::x\n"

        violations = validator.validate_message(message, _registers({1}))

        assert any("WP" in violation for violation in violations)

    def test_a_non_numeric_wp_is_still_refused(self) -> None:
        """Only digits or the literal `none`; a free-text row id is not a row id."""
        message = "fix(data): x\n\nWP: maybe\nProvenance: R1\nAssumptions: none\nGate: t.py::x\n"

        assert any("WP" in violation for violation in validator.validate_message(message, _registers({1})))

    def test_an_empty_range_is_clean(
        self, capsys: pytest.CaptureFixture[str], throwaway_repository: Callable[[Sequence[str]], list[str]]
    ) -> None:
        """A range selecting no commit passes, exercising the real git path.

        The range runs from one commit to itself, so it is empty by construction.
        """
        shas = throwaway_repository([RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range")])

        status = validator.main(["--range", f"{shas[0]}..{shas[0]}"])

        assert status == 0
        assert "0 message(s) validated" in capsys.readouterr().out

    def test_range_mode_accepts_a_history_of_valid_messages(
        self, throwaway_repository: Callable[[Sequence[str]], list[str]]
    ) -> None:
        """Range mode walks git and reports clean when every message satisfies the contract.

        This ran against this repository's own last ten commits until the coupling was
        removed. Two mechanisms broke it, one proven and one suspected. The proven one:
        ``ci-tests.yml`` checks out at the default depth of one, so ``HEAD~10`` names an
        object the clone does not contain and ``git log`` exits 128 — deterministically,
        on every run, and equally in any contributor's shallow checkout. The suspected
        one: a local run overlapping a history rewrite, which is what the earlier
        ``HEAD``-resolving form was changed to avoid. Deepening the CI clone would have
        answered the first and left the second, and would have left the test asserting
        against a history that changes under it either way. A repository built for the
        test answers both, exercises the same ``git log`` path, and additionally covers
        the refusal case below, which the live-history assertion never could.

        Real history keeps two guards that do not depend on this test: the pre-commit
        hook validates unpushed commits on every commit, and ``lint.yml``'s ``trailers``
        job validates a pull request's own commits.
        """
        shas = throwaway_repository(
            [
                RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range"),
                RANGE_MESSAGE_TEMPLATE.format(subject="feat(models): add a dual detection head"),
                RANGE_MESSAGE_TEMPLATE.format(subject="fix(data): correct the pad value"),
            ]
        )

        assert validator.main(["--range", f"{shas[0]}..{shas[-1]}"]) == 0

    def test_an_unresolvable_range_reports_a_verdict_rather_than_a_traceback(
        self, capsys: pytest.CaptureFixture[str], throwaway_repository: Callable[[Sequence[str]], list[str]]
    ) -> None:
        """A range naming a ref this checkout does not hold fails with a sentence, not a stack trace.

        The shallow-checkout case, which is how CI met it: a default-depth checkout has no
        ``refs/remotes/origin/main``, so the hook's own ``origin/main..HEAD`` exits 128.
        Fetching full history fixes the CI half and leaves a contributor's shallow clone,
        where the checker crashing is the difference between a fix they can act on and a
        traceback they file an issue about.
        """
        throwaway_repository([RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range")])

        status = validator.main(["--range", "origin/main..HEAD"])

        out = capsys.readouterr().out
        assert status == 1
        assert "cannot resolve range" in out
        assert "shallow clone" in out

    def test_range_mode_refuses_an_invalid_message_in_the_range(
        self, capsys: pytest.CaptureFixture[str], throwaway_repository: Callable[[Sequence[str]], list[str]]
    ) -> None:
        """One bad message anywhere in the range fails the whole range, naming that commit.

        The walk had no failing case at all before: an empty range and a clean range both
        return zero, so a validator that reported clean unconditionally passed both.
        """
        shas = throwaway_repository(
            [
                RANGE_MESSAGE_TEMPLATE.format(subject="chore: seed the range"),
                RANGE_MESSAGE_TEMPLATE.format(subject="feat(models): add a dual detection head"),
                "no conventional type on this subject\n\nWP: 144\nProvenance: R1\nAssumptions: none\nGate: t.py::x\n",
            ]
        )

        status = validator.main(["--range", f"{shas[0]}..{shas[-1]}"])

        out = capsys.readouterr().out
        assert status == 1
        assert "1 of 2 message(s) invalid" in out
        assert shas[-1][:12] in out
