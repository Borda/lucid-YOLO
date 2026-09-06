# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: golden harness and frozen-golden regression (WP-005).

Exercises the harness end to end against the repository's real goldens, then
drives its failure paths with throwaway golden directories: a tampered value, a
malformed JSON file, and an unresolvable producer must each fail. A copied
``frozen/`` file proves frozen snapshots are discovered and compared by the same
path as live goldens, and a copy marked ``"freezable": false`` proves the harness
rejects it there rather than silently comparing a snapshot nothing can keep
green (WP-154c).
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
#: A real, freezable golden — ``REAL_GOLDEN`` is deliberately not (WP-154c).
REAL_FREEZABLE_GOLDEN = REPO_ROOT / "goldens" / "assignment_cases.json"


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
    frozen_copy = frozen_dir / "assignment_cases.json"
    shutil.copy(REAL_FREEZABLE_GOLDEN, frozen_copy)

    discovered = harness.discover_goldens(tmp_path)
    results = harness.check_all(tmp_path)

    assert frozen_copy in discovered
    assert results and all(r.passed for r in results)


def test_non_freezable_golden_under_frozen_fails(tmp_path: Path) -> None:
    """A golden marked ``"freezable": false`` found under ``frozen/`` is rejected (WP-154c)."""
    frozen_dir = tmp_path / "frozen" / "0.1"
    frozen_dir.mkdir(parents=True)
    data = json.loads(REAL_GOLDEN.read_text())
    data["freezable"] = False
    frozen_copy = frozen_dir / "fixture_checksums.json"
    frozen_copy.write_text(json.dumps(data))

    result = harness.check_golden(frozen_copy)

    assert not result.passed
    assert result.error is not None
    assert "freezable" in result.error


def test_non_freezable_golden_outside_frozen_still_passes(tmp_path: Path) -> None:
    """A golden marked ``"freezable": false`` is checked normally when it is a live golden."""
    golden = tmp_path / "fixture_checksums.json"
    data = json.loads(REAL_GOLDEN.read_text())
    data["freezable"] = False
    golden.write_text(json.dumps(data))

    result = harness.check_golden(golden)

    assert result.passed


def _golden_minus_one_metric(source: Path, dest: Path) -> str:
    """Copy ``source`` to ``dest`` with its first pinned metric removed, and name that metric.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     out = Path(tmp) / "thinned.json"
        ...     dropped = _golden_minus_one_metric(REAL_FREEZABLE_GOLDEN, out)
        ...     dropped in json.loads(out.read_text())["values"]
        False

        ```
    """
    data = json.loads(source.read_text())
    dropped = next(iter(data["values"]))
    del data["values"][dropped]
    data["tolerances"].pop(dropped, None)
    dest.write_text(json.dumps(data))
    return dropped


def test_live_golden_missing_a_produced_metric_fails(tmp_path: Path) -> None:
    """A metric the producer emits but the live golden does not pin fails the check (M-44).

    The gate's purpose is to pin a producer's output. Comparing only the stored keys
    means adding a metric to a producer leaves it unpinned indefinitely while the gate
    keeps reporting ``PASS`` with the old metric count — the new number is never
    guarded, and nothing says so.
    """
    golden = tmp_path / "assignment_cases.json"
    dropped = _golden_minus_one_metric(REAL_FREEZABLE_GOLDEN, golden)

    result = harness.check_golden(golden)

    assert not result.passed
    assert any(c.metric == dropped and not c.passed for c in result.comparisons)


def test_frozen_golden_may_pin_a_subset_of_produced_metrics(tmp_path: Path) -> None:
    """A frozen snapshot pinning fewer metrics than the producer now emits still passes (M-44).

    A past release legitimately froze the metrics that existed then. Holding frozen
    files to the live files' "every produced metric is pinned" rule would turn every
    metric addition into a retroactive failure of every prior release's snapshot.
    """
    frozen_dir = tmp_path / "frozen" / "0.1"
    frozen_dir.mkdir(parents=True)
    _golden_minus_one_metric(REAL_FREEZABLE_GOLDEN, frozen_dir / "assignment_cases.json")

    result = harness.check_golden(frozen_dir / "assignment_cases.json")

    assert result.passed


def _count_producer_runs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap ``resolve_producer`` with a counter, returning a one-element mutable tally.

    Examples:
        ```pycon
        >>> import pytest
        >>> mp = pytest.MonkeyPatch()
        >>> runs = _count_producer_runs(mp)
        >>> _ = harness.resolve_producer("scripts.golden_producers:assignment_cases")
        >>> mp.undo()
        >>> runs
        [1]

        ```
    """
    tally = [0]
    original = harness.resolve_producer

    def counting(spec: str) -> object:
        tally[0] += 1
        return original(spec)

    monkeypatch.setattr(harness, "resolve_producer", counting)
    return tally


def _copy_tree_with_frozen(tmp_path: Path, frozen_bytes: bytes | None) -> Path:
    """Build a goldens dir holding one live golden plus a ``frozen/0.1`` copy of it.

    ``frozen_bytes`` overrides the frozen copy's content when given, so a caller can
    make the pair differ; ``None`` copies the live golden verbatim.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     d = _copy_tree_with_frozen(Path(tmp), None)
        ...     len(harness.discover_goldens(d))
        2

        ```
    """
    live = tmp_path / "assignment_cases.json"
    shutil.copy(REAL_FREEZABLE_GOLDEN, live)
    frozen_dir = tmp_path / "frozen" / "0.1"
    frozen_dir.mkdir(parents=True)
    frozen = frozen_dir / "assignment_cases.json"
    frozen.write_bytes(live.read_bytes() if frozen_bytes is None else frozen_bytes)
    return tmp_path


def test_byte_identical_frozen_golden_reuses_the_live_producer_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen file byte-identical to its live sibling is not recomputed (M-41).

    41 of the repository's 50 offline goldens are byte-identical copies of a live
    sibling, so the sweep runs the same seven producers up to six times each. The
    frozen file is still parsed and compared on its own — only the producer run is
    shared.
    """
    goldens_dir = _copy_tree_with_frozen(tmp_path, None)
    runs = _count_producer_runs(monkeypatch)

    results = harness.check_all(goldens_dir)

    assert all(r.passed for r in results)
    assert runs == [1]
    assert [r.reused for r in results] == [False, True]


@pytest.mark.parametrize(
    "differing",
    [
        pytest.param(True, id="frozen-differs-from-live"),
        pytest.param(False, id="frozen-has-no-live-sibling"),
    ],
)
def test_frozen_golden_without_an_identical_live_sibling_is_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, differing: bool
) -> None:
    """A frozen file that is not byte-identical to a live sibling gets the full recompute (M-41).

    The short-circuit's whole safety argument is byte-equality, so the two ways it can
    fail to hold — a frozen file whose bytes moved, and one whose live sibling was
    renamed or removed — must both fall back to running the producer.
    """
    if differing:
        data = json.loads(REAL_FREEZABLE_GOLDEN.read_text())
        data["tolerances"] = dict(data.get("tolerances", {}))
        goldens_dir = _copy_tree_with_frozen(tmp_path, json.dumps(data, indent=1).encode())
    else:
        goldens_dir = _copy_tree_with_frozen(tmp_path, None)
        (tmp_path / "assignment_cases.json").unlink()
    runs = _count_producer_runs(monkeypatch)

    results = harness.check_all(goldens_dir)

    assert all(r.passed for r in results)
    assert not any(r.reused for r in results)
    assert runs == [len(results)]


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
