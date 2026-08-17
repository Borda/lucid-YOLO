# SPDX-License-Identifier: Apache-2.0
"""Tests for the WP-029 synthetic assignment goldens.

The producer :func:`scripts.golden_producers.assignment_cases` freezes the
Phase-3 label-assignment exit-gate behavior on hand-placed synthetic scenes.
These tests assert the semantic claims directly on the produced metrics (a tiny
ground truth gets zero vanilla-TAL candidates and at least one STAL positive; the
surrogate clamp acts per dimension; the one-to-one branch keeps exactly one
positive per ground truth while one-to-many keeps more), confirm the frozen
``goldens/assignment_cases.json`` passes the golden harness, and confirm the
producer is deterministic across repeated calls.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from scripts.golden_producers import assignment_cases

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "scripts" / "check_goldens.py"
ASSIGNMENT_GOLDEN = REPO_ROOT / "goldens" / "assignment_cases.json"


def _load_harness() -> ModuleType:
    """Load ``scripts/check_goldens.py`` as an importable module.

    Examples:
        >>> module = _load_harness()
        >>> callable(module.check_golden)
        True
    """
    spec = importlib.util.spec_from_file_location("check_goldens", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered before exec
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


def test_tiny_target_tal_zero_stal_positive() -> None:
    """Scenario (i): a 6x6 GT gets zero vanilla-TAL candidates and >= 1 STAL positive."""
    metrics = assignment_cases()

    assert metrics["tal_candidates_tiny"] == 0.0
    assert metrics["tal_positives_tiny"] == 0.0
    assert metrics["stal_positives_tiny"] >= 1.0


def test_surrogate_clamp_is_per_dimension() -> None:
    """Scenario (ii): the surrogate inflates only the sub-``s_min`` side, leaving the other."""
    metrics = assignment_cases()

    assert metrics["stal_surrogate_6x20_width"] == 16.0  # 6 < s_min -> inflated to s_ref
    assert metrics["stal_surrogate_6x20_height"] == 20.0  # 20 >= s_min -> untouched
    assert metrics["tal_candidates_6x20"] == 0.0
    assert metrics["stal_candidates_6x20"] > metrics["tal_candidates_6x20"]


def test_one_to_one_unique_per_gt() -> None:
    """Scenario (iii): one-to-one keeps exactly one positive per GT; one-to-many keeps more."""
    metrics = assignment_cases()

    assert metrics["unique_positives_per_gt_max"] == 1.0
    assert metrics["unique_total_positives"] == 3.0
    assert metrics["o2m_total_positives"] > metrics["unique_total_positives"]


def test_assignment_golden_passes_harness() -> None:
    """The frozen ``assignment_cases.json`` recomputes and matches every value exactly."""
    result = harness.check_golden(ASSIGNMENT_GOLDEN)

    assert result.passed, harness.format_result(result, harness.DEFAULT_GOLDENS_DIR)


def test_producer_is_deterministic() -> None:
    """Two producer calls return byte-identical metric mappings (no RNG)."""
    assert assignment_cases() == assignment_cases()
