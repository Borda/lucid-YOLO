# SPDX-License-Identifier: Apache-2.0
"""Toy convergence golden: MuSGD reaches a loss threshold in fewer steps than SGD (WP-033).

Exercises the ``optim_toy`` producer end to end -- the directional convergence
claim mirroring R1 Table 4 at toy scale -- and pins it through the WP-005 golden
harness. The producer trains two byte-identical micro-CNNs on one fixed,
fully-seeded regression task, one with MuSGD and one with momentum-SGD at the
same learning rate, and reports the steps each needs to hit the threshold.

The harness and producer modules live under ``scripts/`` (not an importable
package), so they are loaded by file path via ``importlib.util`` -- the same
pattern ``scripts/_tests/test_check_goldens.py`` uses.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "scripts" / "check_goldens.py"
PRODUCERS_PATH = REPO_ROOT / "scripts" / "golden_producers.py"
GOLDEN = REPO_ROOT / "goldens" / "optim_toy.json"


def _load_module(name: str, path: Path) -> ModuleType:
    """Load a ``scripts/`` module by file path and register it for dataclass/exec support.

    Examples:
        >>> module = _load_module("check_goldens_doctest", HARNESS_PATH)
        >>> hasattr(module, "check_golden")
        True
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses/self-references need the module registered before exec
    spec.loader.exec_module(module)
    return module


harness = _load_module("check_goldens", HARNESS_PATH)
producers = _load_module("golden_producers", PRODUCERS_PATH)


def test_musgd_beats_sgd() -> None:
    """MuSGD hits the loss threshold in fewer steps than SGD, and both reach it before the cap."""
    metrics = producers.optim_toy()

    assert metrics["steps_to_threshold_musgd"] < metrics["steps_to_threshold_sgd"]
    assert metrics["steps_to_threshold_musgd"] < producers._TOY_MAX_STEPS  # reached threshold, not the cap
    assert metrics["steps_to_threshold_sgd"] < producers._TOY_MAX_STEPS
    assert metrics["final_loss_musgd"] <= producers._TOY_LOSS_THRESHOLD
    assert metrics["final_loss_sgd"] <= producers._TOY_LOSS_THRESHOLD


def test_producer_deterministic() -> None:
    """Two consecutive in-process runs return identical dicts (no ambient-RNG leakage)."""
    first = producers.optim_toy()
    second = producers.optim_toy()

    assert first == second


def test_golden_file_current() -> None:
    """The frozen ``optim_toy.json`` golden matches a fresh producer run within tolerance."""
    result = harness.check_golden(GOLDEN)

    assert result.passed, harness.format_result(result, harness.DEFAULT_GOLDENS_DIR)
