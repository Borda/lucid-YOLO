# SPDX-License-Identifier: Apache-2.0
"""Meta tests: golden harness and frozen-golden regression (WP-005).

Exercises the harness end to end against the repository's real goldens, then
drives its failure paths with throwaway golden directories: a tampered value, a
malformed JSON file, and an unresolvable producer must each fail. A copied
``frozen/`` file proves frozen snapshots are discovered and compared by the same
path as live goldens.
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "scripts" / "check_goldens.py"
REAL_GOLDEN = REPO_ROOT / "goldens" / "fixture_checksums.json"


def _load_harness() -> ModuleType:
    """Load ``scripts/check_goldens.py`` as an importable module.

    Examples:
        >>> module = _load_harness()
        >>> module.__name__
        'check_goldens'
        >>> callable(module.check_all)
        True
    """
    spec = importlib.util.spec_from_file_location("check_goldens", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered before exec
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


def test_real_goldens_pass() -> None:
    """The full harness run over the repository's real goldens passes (DoD a)."""
    results = harness.check_all()
    assert results, "expected at least one real golden to be discovered"
    assert all(r.passed for r in results), [harness.format_result(r, harness.DEFAULT_GOLDENS_DIR) for r in results]


def test_tampered_golden_fails(tmp_path: Path) -> None:
    """A copied golden with one perturbed value fails (DoD b, negative test)."""
    golden = tmp_path / "fixture_checksums.json"
    data = json.loads(REAL_GOLDEN.read_text())
    data["values"]["detseg_num_images"] += 1.0
    golden.write_text(json.dumps(data))
    result = harness.check_golden(golden)
    assert not result.passed
    assert any(c.metric == "detseg_num_images" and not c.passed for c in result.comparisons)


def test_malformed_json_fails(tmp_path: Path) -> None:
    """A golden file that is not valid JSON fails with an error (DoD c)."""
    golden = tmp_path / "broken.json"
    golden.write_text("{ this is not json ]")
    result = harness.check_golden(golden)
    assert not result.passed
    assert result.error is not None


def test_unknown_producer_fails(tmp_path: Path) -> None:
    """A golden naming a non-existent producer fails with an error (DoD d)."""
    golden = tmp_path / "unknown.json"
    golden.write_text(json.dumps({"producer": "scripts.golden_producers:no_such_function", "values": {"x": 1.0}}))
    result = harness.check_golden(golden)
    assert not result.passed
    assert result.error is not None


def test_frozen_golden_is_discovered_and_compared(tmp_path: Path) -> None:
    """A file under ``frozen/`` is discovered and passes the same comparison (DoD e)."""
    frozen_dir = tmp_path / "frozen" / "0.1"
    frozen_dir.mkdir(parents=True)
    frozen_copy = frozen_dir / "fixture_checksums.json"
    shutil.copy(REAL_GOLDEN, frozen_copy)

    discovered = harness.discover_goldens(tmp_path)
    results = harness.check_all(tmp_path)

    assert frozen_copy in discovered
    assert results and all(r.passed for r in results)


@pytest.mark.parametrize(
    "spec_valid",
    [
        pytest.param(True, id="resolvable"),
        pytest.param(False, id="malformed-spec"),
    ],
)
def test_resolve_producer_contract(spec_valid: bool) -> None:
    """A valid ``module:function`` spec resolves; a malformed spec raises ``GoldenError``."""
    if spec_valid:
        producer = harness.resolve_producer("scripts.golden_producers:fixture_checksums")
        assert callable(producer)
    else:
        with pytest.raises(harness.GoldenError):
            harness.resolve_producer("not-a-valid-spec")


def test_main_exit_code_zero_on_real_goldens() -> None:
    """The CLI returns 0 when every real golden passes."""
    assert harness.main([]) == 0


def test_main_exit_code_one_on_empty_dir(tmp_path: Path) -> None:
    """The CLI returns 1 when no goldens are found."""
    assert harness.main(["--goldens-dir", str(tmp_path)]) == 1
