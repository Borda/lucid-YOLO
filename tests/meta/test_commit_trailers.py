# SPDX-License-Identifier: Apache-2.0
"""Meta tests: commit-trailer validator (WP-004).

Guards the provenance-carrying commit contract (AGENTS.md sec. 5): a valid
message passes, a missing WP trailer fails, an unknown provenance id fails, a
hash before the separator fails, and an at-sign confined to the co-author
trailers after the separator passes.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "check_commit_trailers.py"
PROVENANCE_PATH = REPO_ROOT / "docs" / "PROVENANCE.md"


def _load_validator() -> ModuleType:
    """Load ``scripts/check_commit_trailers.py`` as an importable module."""
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
    """The canonical AGENTS.md sec. 5 message reports no violation."""
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
