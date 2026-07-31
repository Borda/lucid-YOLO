# SPDX-License-Identifier: Apache-2.0
"""Golden harness: recompute and verify frozen metric goldens (WP-005).

A *golden* is a JSON file pinning the output of a deterministic *producer* — a
zero-argument importable function returning ``dict[str, float]``. The harness
discovers every ``goldens/*.json`` and every ``goldens/frozen/**/*.json``,
resolves each file's producer, recomputes it, and compares each stored value
against the freshly computed one within a per-metric absolute tolerance. A
metric absent from ``tolerances`` (or given tolerance ``0.0``) must match
exactly.

Frozen files travel the identical comparison path, so a release's frozen goldens
staying green *is* the frozen-golden regression: current code must still satisfy
every value snapshotted at every past release.

Golden file schema::

    {
      "producer": "<module>:<function>",
      "tolerances": {"<metric>": <abs_tol_float>},
      "values": {"<metric>": <number>}
    }

The discovery/compare core is importable (``discover_goldens``, ``check_golden``,
``check_all``); :func:`main` is a thin CLI over it.

Examples:
    Verify the repository's goldens (exit 1 on any failure)::

        python scripts/check_goldens.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

#: Repository root (``scripts/`` is one level below it).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Default directory holding the top-level goldens and the ``frozen/`` tree.
DEFAULT_GOLDENS_DIR = REPO_ROOT / "goldens"

#: Producer return type: a flat mapping of metric name to numeric value.
Producer = Callable[[], "dict[str, float]"]


class GoldenError(Exception):
    """A golden file is malformed, or its producer cannot be resolved or run."""


@dataclass(frozen=True)
class MetricComparison:
    """Outcome of comparing one stored metric against its recomputed value.

    Attributes:
        metric: The metric name.
        expected: The value stored in the golden file.
        actual: The freshly recomputed value.
        tolerance: The absolute tolerance applied (``0.0`` means exact).
        passed: Whether ``expected`` and ``actual`` agree within ``tolerance``.
    """

    metric: str
    expected: float
    actual: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class GoldenResult:
    """Aggregate result for a single golden file.

    Attributes:
        path: The golden file's path.
        passed: ``True`` only when the file loaded, ran, and every metric matched.
        comparisons: Per-metric comparisons (empty when ``error`` is set).
        error: A failure message when the file is malformed or its producer
            failed, otherwise ``None``.
    """

    path: Path
    passed: bool
    comparisons: list[MetricComparison]
    error: str | None = None


def discover_goldens(goldens_dir: Path) -> list[Path]:
    """Discover every golden JSON file under ``goldens_dir``.

    Collects top-level ``*.json`` files and every ``frozen/**/*.json`` file, so
    frozen snapshots are checked by the same path as live goldens.

    Args:
        goldens_dir: The directory to scan (typically ``<repo>/goldens``).

    Returns:
        Sorted, de-duplicated list of golden file paths; empty if none exist.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     d = Path(tmp)
        ...     _ = (d / "a.json").write_text("{}")
        ...     _ = (d / "frozen" / "0.1").mkdir(parents=True)
        ...     _ = (d / "frozen" / "0.1" / "a.json").write_text("{}")
        ...     [p.name for p in discover_goldens(d)]
        ['a.json', 'a.json']

        ```
    """
    if not goldens_dir.is_dir():
        return []
    top = goldens_dir.glob("*.json")
    frozen = (goldens_dir / "frozen").rglob("*.json")
    return sorted(set(top) | set(frozen))


def resolve_producer(spec: str) -> Producer:
    """Resolve a ``"module:function"`` producer spec to a callable.

    The repository root is prepended to ``sys.path`` so ``scripts.*`` and
    ``tests.*`` module paths resolve as namespace packages regardless of the
    current working directory.

    Args:
        spec: A ``"module:function"`` string, e.g. ``"scripts.golden_producers:fixture_checksums"``.

    Returns:
        The resolved zero-argument callable.

    Raises:
        GoldenError: If ``spec`` is malformed, the module cannot be imported, or
            the attribute is missing or not callable.

    Examples:
        ```pycon
        >>> fn = resolve_producer("scripts.golden_producers:fixture_checksums")
        >>> callable(fn)
        True

        ```
    """
    if spec.count(":") != 1 or not all(part.strip() for part in spec.split(":")):
        raise GoldenError(f"invalid producer spec {spec!r}; expected 'module:function'")
    module_name, func_name = spec.split(":")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise GoldenError(f"cannot import producer module {module_name!r}: {exc}") from exc
    producer = getattr(module, func_name, None)
    if not callable(producer):
        raise GoldenError(f"producer {spec!r} is missing or not callable")
    return cast(Producer, producer)


def _parse_golden(path: Path) -> tuple[str, dict[str, float], dict[str, float]]:
    """Read and validate a golden file's schema.

    Args:
        path: The golden file path.

    Returns:
        A ``(producer_spec, tolerances, values)`` triple.

    Raises:
        GoldenError: If the file is not valid JSON, is not an object, or is
            missing a string ``producer`` or a ``values`` mapping.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenError(f"cannot read golden JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise GoldenError("golden root must be a JSON object")
    producer = data.get("producer")
    if not isinstance(producer, str) or not producer:
        raise GoldenError("golden 'producer' must be a non-empty string")
    values = data.get("values")
    if not isinstance(values, dict) or not values:
        raise GoldenError("golden 'values' must be a non-empty object")
    tolerances = data.get("tolerances", {})
    if not isinstance(tolerances, dict):
        raise GoldenError("golden 'tolerances' must be an object when present")
    return producer, tolerances, values


def _compare_values(
    expected: dict[str, float],
    actual: dict[str, float],
    tolerances: dict[str, float],
) -> list[MetricComparison]:
    """Compare stored ``expected`` metrics against recomputed ``actual`` values.

    A metric absent from ``tolerances``, or given tolerance ``0.0``, must match
    exactly; otherwise the absolute difference must not exceed the tolerance. A
    metric missing from ``actual`` is recorded as a failed comparison.

    Args:
        expected: Metric values stored in the golden file.
        actual: Metric values freshly produced.
        tolerances: Per-metric absolute tolerances.

    Returns:
        One :class:`MetricComparison` per stored metric.
    """
    comparisons: list[MetricComparison] = []
    for metric, exp in expected.items():
        tol = float(tolerances.get(metric, 0.0))
        if metric not in actual:
            comparisons.append(MetricComparison(metric, float(exp), float("nan"), tol, passed=False))
            continue
        act = float(actual[metric])
        passed = act == float(exp) if tol == 0.0 else abs(act - float(exp)) <= tol
        comparisons.append(MetricComparison(metric, float(exp), act, tol, passed))
    return comparisons


def check_golden(path: Path) -> GoldenResult:
    """Load, run, and compare a single golden file.

    Args:
        path: The golden file path.

    Returns:
        A :class:`GoldenResult`; ``error`` is set (and ``passed`` is ``False``)
        when the file is malformed or its producer cannot be resolved or run.

    Examples:
        ```pycon
        >>> from scripts.check_goldens import check_golden, DEFAULT_GOLDENS_DIR
        >>> check_golden(DEFAULT_GOLDENS_DIR / "fixture_checksums.json").passed
        True

        ```
    """
    try:
        spec, tolerances, values = _parse_golden(path)
        producer = resolve_producer(spec)
        actual = producer()
    except GoldenError as exc:
        return GoldenResult(path, passed=False, comparisons=[], error=str(exc))
    except Exception as exc:
        return GoldenResult(path, passed=False, comparisons=[], error=f"producer raised {type(exc).__name__}: {exc}")
    comparisons = _compare_values(values, actual, tolerances)
    return GoldenResult(path, passed=all(c.passed for c in comparisons), comparisons=comparisons)


def check_all(goldens_dir: Path = DEFAULT_GOLDENS_DIR) -> list[GoldenResult]:
    """Discover and check every golden under ``goldens_dir``.

    Args:
        goldens_dir: The directory to scan (defaults to ``<repo>/goldens``).

    Returns:
        One :class:`GoldenResult` per discovered golden, in discovery order.

    Examples:
        ```pycon
        >>> results = check_all()
        >>> all(r.passed for r in results)
        True

        ```
    """
    return [check_golden(path) for path in discover_goldens(goldens_dir)]


def format_result(result: GoldenResult, goldens_dir: Path) -> str:
    """Render a one-line ``PASS``/``FAIL`` summary for a golden result.

    Args:
        result: The result to render.
        goldens_dir: Base directory used to shorten the reported path.

    Returns:
        A human-readable status line; failures append the first offending detail.
    """
    try:
        shown = result.path.relative_to(goldens_dir)
    except ValueError:
        shown = result.path
    if result.error is not None:
        return f"FAIL {shown} — {result.error}"
    if result.passed:
        return f"PASS {shown} — {len(result.comparisons)} metric(s)"
    failed = next(c for c in result.comparisons if not c.passed)
    detail = f"{failed.metric}: expected {failed.expected}, got {failed.actual} (tol {failed.tolerance})"
    return f"FAIL {shown} — {detail}"


def main(argv: list[str] | None = None) -> int:
    """Check every golden and print per-file verdicts.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when every golden passes; ``1`` on any failure or when no goldens
        are found.
    """
    parser = argparse.ArgumentParser(description="Recompute and verify metric goldens.")
    parser.add_argument(
        "--goldens-dir",
        type=Path,
        default=DEFAULT_GOLDENS_DIR,
        help="directory holding goldens and the frozen/ subtree (default: <repo>/goldens)",
    )
    args = parser.parse_args(argv)

    results = check_all(args.goldens_dir)
    if not results:
        print(f"no goldens found under {args.goldens_dir}")
        return 1
    for result in results:
        print(format_result(result, args.goldens_dir))
    failures = [r for r in results if not r.passed]
    print(f"{len(results) - len(failures)}/{len(results)} goldens passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
