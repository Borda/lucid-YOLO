# SPDX-License-Identifier: Apache-2.0
"""Golden harness: recompute and verify frozen metric goldens (WP-005).

A *golden* is a JSON file pinning the output of a deterministic *producer* — a
zero-argument importable function returning ``dict[str, float]``. The harness
discovers every ``goldens/*.json`` and every ``goldens/frozen/**/*.json``,
resolves each file's producer, recomputes it, and compares each stored value
against the freshly computed one within a per-metric absolute tolerance. A
metric absent from ``tolerances`` (or given tolerance ``0.0``) must match
exactly.

The comparison runs in both directions on a live golden: a stored metric the producer
no longer emits fails, and so does a metric the producer emits that the golden does not
store. A golden pins its producer's *output*, so a metric nobody hand-added to the file
is an unguarded number rather than an absent one, and the gate reported ``PASS`` with
the old metric count while it stayed that way (M-44).

Frozen files travel the identical comparison path, so a release's frozen goldens
staying green *is* the frozen-golden regression: current code must still satisfy
every value snapshotted at every past release. They are exempt from the unpinned-metric
half alone — a past release pinned the metrics that existed then — and from the producer
run itself when byte-identical to a live sibling that has already run it, which is what
keeps a sweep from recomputing the same seven producers up to six times (:func:`check_all`).

The ``goldens/gpu/`` subtree is **not** part of the default discovery: those
goldens' producers retrain a model on an accelerator over a generated dataset, so
they would break an offline run. They are recomputed only with ``--include-gpu``
(see :func:`discover_gpu_goldens`), which the default ``make gate`` never passes.

Golden file schema::

    {
      "producer": "<module>:<function>",
      "tolerances": {"<metric>": <abs_tol_float>},
      "values": {"<metric>": <number>},
      "freezable": <bool, optional, default true>
    }

``"freezable": false`` marks a golden whose producer's output is pinned to an
external package rather than to this project's own code (WP-132: two producers
render synthetic images through ``fuse-augmentations``, so once that package's
generator moves, no future code change can satisfy "the frozen copy still holds"
— the frozen snapshot was never a real regression guard). Such a golden is
excluded from ``make freeze-goldens`` and must never appear under
``goldens/frozen/``; :func:`check_golden` fails loudly if one ever does, rather
than silently comparing a snapshot nothing can keep green (WP-154c).

The discovery/compare core is importable (``discover_goldens``, ``check_golden``,
``check_all``); :func:`main` is a thin CLI over it.

Examples:
    Verify the repository's goldens (exit 1 on any failure)::

        python scripts/check_goldens.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
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

    Two directions of absence are recorded with a ``nan`` on the missing side, and
    both are failures: ``actual`` is ``nan`` when the golden pins a metric the producer
    no longer emits, and ``expected`` is ``nan`` when the producer emits a metric the
    golden does not pin. :func:`format_result` reads the two apart.

    Attributes:
        metric: The metric name.
        expected: The value stored in the golden file, or ``nan`` when unpinned.
        actual: The freshly recomputed value, or ``nan`` when the producer omitted it.
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
        reused: ``True`` when the producer was not run for this file because a
            byte-identical live golden had already run it (see :func:`check_all`).
    """

    path: Path
    passed: bool
    comparisons: list[MetricComparison]
    error: str | None = None
    reused: bool = False


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


def discover_gpu_goldens(goldens_dir: Path) -> list[Path]:
    """Discover the accelerator-gated goldens under ``goldens_dir/gpu``.

    These live in a ``gpu/`` subdirectory that :func:`discover_goldens` deliberately
    does **not** glob, so the default offline harness never recomputes them (their
    producers need an accelerator and a generated dataset). They are checked only
    when the caller explicitly opts in via ``--include-gpu``.

    Args:
        goldens_dir: The directory holding the ``gpu/`` subtree (typically
            ``<repo>/goldens``).

    Returns:
        Sorted list of ``gpu/*.json`` golden paths; empty if the subdirectory is
        absent.

    Examples:
        ```pycon
        >>> from scripts.check_goldens import discover_gpu_goldens, DEFAULT_GOLDENS_DIR
        >>> all(p.parent.name == "gpu" for p in discover_gpu_goldens(DEFAULT_GOLDENS_DIR))
        True

        ```
    """
    gpu_dir = goldens_dir / "gpu"
    if not gpu_dir.is_dir():
        return []
    return sorted(gpu_dir.glob("*.json"))


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


def _parse_golden(path: Path) -> tuple[str, dict[str, float], dict[str, float], bool]:
    """Read and validate a golden file's schema.

    Args:
        path: The golden file path.

    Returns:
        A ``(producer_spec, tolerances, values, freezable)`` tuple. ``freezable``
        defaults to ``True`` when the field is absent.

    Raises:
        GoldenError: If the file is not valid JSON, is not an object, is missing
            a string ``producer`` or a ``values`` mapping, or gives ``freezable``
            a non-boolean value.
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
    freezable = data.get("freezable", True)
    if not isinstance(freezable, bool):
        raise GoldenError("golden 'freezable' must be a boolean when present")
    return producer, tolerances, values, freezable


def _compare_values(
    expected: dict[str, float],
    actual: dict[str, float],
    tolerances: dict[str, float],
    *,
    require_all_pinned: bool,
) -> list[MetricComparison]:
    """Compare stored ``expected`` metrics against recomputed ``actual`` values.

    A metric absent from ``tolerances``, or given tolerance ``0.0``, must match
    exactly; otherwise the absolute difference must not exceed the tolerance. A
    metric missing from ``actual`` is recorded as a failed comparison.

    With ``require_all_pinned``, a metric the producer emits that the golden does
    *not* store is also a failed comparison (WP-005 pins a producer's output, so an
    unpinned metric is an unguarded number, not an absent one — M-44). Frozen
    snapshots pass ``False``: a past release pinned the metrics that existed then,
    and holding it to today's set would fail every prior snapshot on every metric
    addition.

    Args:
        expected: Metric values stored in the golden file.
        actual: Metric values freshly produced.
        tolerances: Per-metric absolute tolerances.
        require_all_pinned: Report metrics present in ``actual`` but absent from
            ``expected`` as failures.

    Returns:
        One :class:`MetricComparison` per stored metric, then one per unpinned
        produced metric when ``require_all_pinned``.
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
    if require_all_pinned:
        for metric in sorted(set(actual) - set(expected)):
            comparisons.append(MetricComparison(metric, float("nan"), float(actual[metric]), 0.0, passed=False))
    return comparisons


def _digest(path: Path) -> str:
    """Return the SHA-256 hex digest of a golden file's bytes.

    The short-circuit key in :func:`check_all`: byte-equality is what makes reusing
    one file's producer run for another sound, since identical bytes carry an
    identical producer spec.

    Args:
        path: The golden file to hash.

    Returns:
        The 64-character lowercase hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_golden(path: Path, *, cache: dict[str, dict[str, float]] | None = None) -> GoldenResult:
    """Load, run, and compare a single golden file.

    Args:
        path: The golden file path.
        cache: Optional producer-output cache keyed by the golden file's SHA-256, as
            maintained by :func:`check_all`. A file whose digest is already present
            reuses that output instead of re-running the producer; a *live* file
            whose digest is absent records its output there. Frozen files never
            write to the cache, so a frozen snapshot is only ever short-circuited by
            a live golden it is byte-identical to.

    Returns:
        A :class:`GoldenResult`; ``error`` is set (and ``passed`` is ``False``)
        when the file is malformed or its producer cannot be resolved or run.
        ``reused`` reports whether the producer run was taken from ``cache``.

    Examples:
        ```pycon
        >>> from scripts.check_goldens import check_golden, DEFAULT_GOLDENS_DIR
        >>> check_golden(DEFAULT_GOLDENS_DIR / "fixture_checksums.json").passed
        True

        ```
    """
    frozen = "frozen" in path.parts
    reused = False
    try:
        spec, tolerances, values, freezable = _parse_golden(path)
        if not freezable and frozen:
            raise GoldenError(
                "golden is 'freezable': false but found under goldens/frozen/ — a non-freezable "
                "golden can never satisfy 'current code still satisfies every frozen value' once "
                "its producer's external dependency moves, so it must never be committed there"
            )
        key = _digest(path)
        if cache is not None and key in cache:
            actual, reused = cache[key], True
        else:
            actual = resolve_producer(spec)()
            if cache is not None and not frozen:
                cache[key] = actual
    except GoldenError as exc:
        return GoldenResult(path, passed=False, comparisons=[], error=str(exc))
    except Exception as exc:
        return GoldenResult(path, passed=False, comparisons=[], error=f"producer raised {type(exc).__name__}: {exc}")
    comparisons = _compare_values(values, actual, tolerances, require_all_pinned=not frozen)
    return GoldenResult(path, passed=all(c.passed for c in comparisons), comparisons=comparisons, reused=reused)


def check_all(goldens_dir: Path = DEFAULT_GOLDENS_DIR, include_gpu: bool = False) -> list[GoldenResult]:
    """Discover and check every golden under ``goldens_dir``.

    By default only the offline goldens (top-level and ``frozen/``) are checked, so
    a run on a machine with no accelerator and no generated dataset stays green.
    With ``include_gpu`` the ``gpu/`` subtree is appended — those producers retrain a
    model on the local accelerator (see :func:`discover_gpu_goldens`).

    A release freeze copies live goldens verbatim, so most frozen files are byte-identical
    to a live sibling — 41 of this repository's 50 offline goldens are, which had the seven
    freezable producers recomputed up to six times per sweep. Live goldens are therefore
    checked first and their producer output recorded against the file's SHA-256; a frozen
    file whose bytes match one of them reuses that output rather than re-running an
    identical computation, and is reported with ``reused``. Byte-equality is the whole
    safety argument, so a frozen file that differs from its live sibling, or has none,
    takes the full path — as does every parse, ``freezable`` and tolerance check, which
    still run per file (M-41).

    Args:
        goldens_dir: The directory to scan (defaults to ``<repo>/goldens``).
        include_gpu: Also check the accelerator-gated ``gpu/*.json`` goldens.
            Defaults to ``False``.

    Returns:
        One :class:`GoldenResult` per discovered golden, in discovery order.

    Examples:
        ```pycon
        >>> results = check_all()
        >>> all(r.passed for r in results)
        True

        ```
    """
    paths = discover_goldens(goldens_dir)
    if include_gpu:
        paths = paths + discover_gpu_goldens(goldens_dir)
    cache: dict[str, dict[str, float]] = {}
    live = [path for path in paths if "frozen" not in path.parts]
    results = {path: check_golden(path, cache=cache) for path in live}
    results.update({path: check_golden(path, cache=cache) for path in paths if path not in results})
    return [results[path] for path in paths]


def _failure_detail(failed: MetricComparison) -> str:
    """Describe one failed comparison, reading the two ``nan`` directions apart.

    Args:
        failed: The comparison to describe.

    Returns:
        A one-line description naming the metric and what went wrong with it.
    """
    if math.isnan(failed.expected):
        return f"{failed.metric}: produced {failed.actual} but the golden pins no such metric"
    if math.isnan(failed.actual):
        return f"{failed.metric}: pinned at {failed.expected} but the producer no longer emits it"
    return f"{failed.metric}: expected {failed.expected}, got {failed.actual} (tol {failed.tolerance})"


def format_result(result: GoldenResult, goldens_dir: Path) -> str:
    """Render a one-line ``PASS``/``FAIL`` summary for a golden result.

    Args:
        result: The result to render.
        goldens_dir: Base directory used to shorten the reported path.

    Returns:
        A human-readable status line; a pass whose producer run was reused is marked
        ``SAME as live``, and failures append the first offending detail.
    """
    try:
        shown = result.path.relative_to(goldens_dir)
    except ValueError:
        shown = result.path
    if result.error is not None:
        return f"FAIL {shown} — {result.error}"
    if result.passed:
        same = "SAME as live, " if result.reused else ""
        return f"PASS {shown} — {same}{len(result.comparisons)} metric(s)"
    return f"FAIL {shown} — {_failure_detail(next(c for c in result.comparisons if not c.passed))}"


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
    parser.add_argument(
        "--include-gpu",
        action="store_true",
        help="also recompute the accelerator-gated goldens/gpu/*.json (retrains models; needs an accelerator)",
    )
    args = parser.parse_args(argv)

    results = check_all(args.goldens_dir, include_gpu=args.include_gpu)
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
