# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: commit-trailer validator (WP-004).

Guards the provenance-carrying commit contract (AGENTS.md sec. 5): a valid
message passes, a missing WP trailer fails, an unknown provenance id fails, a
hash before the separator fails, an at-sign confined to the co-author trailers
after the separator passes, and the subject's type comes from the AGENTS.md
sec. 5 allowed set.
"""

import importlib.util
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "lint" / "check_commit_trailers.py"
PROVENANCE_PATH = REPO_ROOT / "docs" / "PROVENANCE.md"

#: A contract-satisfying message with a substitutable subject, for the range-mode
#: repositories below. ``R1`` is read from the real ``docs/PROVENANCE.md``, which is
#: what the validator loads its allowlist from whatever repository it is walking.
RANGE_MESSAGE_TEMPLATE = (
    "{subject}\n\nOne paragraph of body.\n\nWP: 144\nProvenance: R1\nAssumptions: none\nGate: t.py::x\n"
)


def _load_validator() -> ModuleType:
    """Load ``scripts/lint/check_commit_trailers.py`` as an importable module.

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
    spec.loader.exec_module(module)
    return module


validator = _load_validator()
VALID_IDS = validator.load_provenance_ids(PROVENANCE_PATH)

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

        assert validator.validate_message(message, {1}) == []

    def test_a_missing_wp_trailer_is_still_refused(self) -> None:
        """Omission is not the relaxation -- `none` states it, absence leaves it unsaid."""
        message = "fix(data): correct the pad value\n\nbody\n\nProvenance: R1\nAssumptions: none\nGate: t.py::x\n"

        violations = validator.validate_message(message, {1})

        assert any("WP" in violation for violation in violations)

    def test_a_non_numeric_wp_is_still_refused(self) -> None:
        """Only digits or the literal `none`; a free-text row id is not a row id."""
        message = "fix(data): x\n\nWP: maybe\nProvenance: R1\nAssumptions: none\nGate: t.py::x\n"

        assert any("WP" in violation for violation in validator.validate_message(message, {1}))

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
