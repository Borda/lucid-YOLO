# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: commit-trailer validator (WP-004).

Guards the provenance-carrying commit contract (AGENTS.md sec. 5): a valid
message passes, a missing WP trailer fails, an unknown provenance id fails, a
hash before the separator fails, an at-sign confined to the co-author trailers
after the separator passes, and the subject's type comes from the AGENTS.md
sec. 5 allowed set.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "lint" / "check_commit_trailers.py"
PROVENANCE_PATH = REPO_ROOT / "docs" / "PROVENANCE.md"


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
