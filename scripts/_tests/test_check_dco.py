# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: Developer Certificate of Origin sign-off validator (WP-142).

Guards admission layer one (``docs/CONTRIBUTING.md``): a signed commit passes, an
unsigned one fails, a sign-off missing its name or its address fails as *malformed*
rather than as missing, several sign-offs on one co-authored commit pass, and an
address that does not match the author is accepted on purpose -- a sign-off is a
representation by the signer, not an identity proof.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "lint" / "check_dco.py"


def _load_validator() -> ModuleType:
    """Load ``scripts/lint/check_dco.py`` as an importable module.

    Examples:
        >>> module = _load_validator()
        >>> callable(module.validate_message)
        True
    """
    spec = importlib.util.spec_from_file_location("check_dco", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dco = _load_validator()

SIGNED = "fix(data): correct the pad value\n\nbody\n\nSigned-off-by: Ada Lovelace <ada@example.com>\n"


class TestValidateMessage:
    """Tests for ``check_dco.validate_message``."""

    def test_a_signed_commit_passes(self) -> None:
        """A well-formed trailer with a name and an address is accepted."""
        assert dco.validate_message(SIGNED) == []

    def test_an_unsigned_commit_fails(self) -> None:
        """A message carrying no sign-off at all is refused, and the fix is named."""
        violations = dco.validate_message("fix(data): correct the pad value\n\nbody\n")

        assert len(violations) == 1
        assert "missing" in violations[0]
        assert "-s" in violations[0], "the refusal should name the flag that fixes it"

    @pytest.mark.parametrize(
        "trailer",
        [
            "Signed-off-by: Ada Lovelace",
            "Signed-off-by: <ada@example.com>",
            "Signed-off-by: Ada Lovelace <ada@example>",
            "Signed off by: Ada Lovelace <ada@example.com>",
            "signed-off-by Ada Lovelace ada@example.com",
        ],
        ids=["no-address", "no-name", "domain-without-dot", "spaced-key", "no-brackets"],
    )
    def test_a_near_miss_fails_as_malformed_not_missing(self, trailer: str) -> None:
        """A line meaning to be a sign-off is reported as malformed, never as absent.

        The distinction is the whole value of the near-miss branch: "missing" sends a
        contributor looking for a trailer they believe they already wrote, while
        "malformed" is a one-character fix they can see.
        """
        violations = dco.validate_message(f"fix(data): x\n\n{trailer}\n")

        assert len(violations) == 1
        assert violations[0].startswith("malformed")

    def test_several_sign_offs_pass(self) -> None:
        """Co-authored work legitimately carries more than one sign-off."""
        message = SIGNED + "Signed-off-by: Grace Hopper <grace@example.com>\n"

        assert dco.validate_message(message) == []

    def test_a_sign_off_not_matching_the_author_is_accepted(self) -> None:
        """Author-matching is deliberately not required -- see the module docstring.

        A maintainer applying a contributor's patch signs their own line beside the
        original, and an author committing under an employer address may sign under a
        personal one. Requiring a match rejects both while catching nobody willing to
        type a false name.
        """
        assert dco.validate_message("fix: x\n\nSigned-off-by: Someone Else <other@example.org>\n") == []

    def test_a_sign_off_anywhere_in_the_body_counts(self) -> None:
        """The trailer is matched by shape, not by position in the message."""
        assert dco.validate_message("fix: x\n\nSigned-off-by: Ada Lovelace <ada@example.com>\n\nmore prose\n") == []


class TestMain:
    """Tests for ``check_dco.main``."""

    def test_file_mode_accepts_a_signed_message(self, tmp_path: Path) -> None:
        """The CLI validates a single message file and exits zero when it is signed."""
        path = tmp_path / "COMMIT_EDITMSG"
        path.write_text(SIGNED, encoding="utf-8")

        assert dco.main(["--file", str(path)]) == 0

    def test_file_mode_refuses_an_unsigned_message(self, tmp_path: Path) -> None:
        """The CLI exits non-zero on an unsigned message."""
        path = tmp_path / "COMMIT_EDITMSG"
        path.write_text("fix(data): x\n", encoding="utf-8")

        assert dco.main(["--file", str(path)]) == 1

    def test_a_target_is_required(self) -> None:
        """Neither --file nor --range is a usage error, not a silent pass."""
        with pytest.raises(SystemExit):
            dco.main([])

    def test_an_empty_range_is_clean(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A range selecting no commit passes: nothing was submitted unsigned.

        ``HEAD..HEAD`` is empty by construction, so this exercises the real git path
        without depending on what this repository's own history happens to contain.
        """
        status = dco.main(["--range", "HEAD..HEAD"])

        assert status == 0
        assert "0 commit(s) signed off" in capsys.readouterr().out
