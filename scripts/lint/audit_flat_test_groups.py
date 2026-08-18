# SPDX-License-Identifier: Apache-2.0
"""Flat-test-group audit: a mechanical enumeration outside a class (WP-128, split off 117).

WP-117's "8 groups across 6 files" was a one-off human read; a first-shared-word ``ast``
scan over every ``test_`` name returned ~90 candidate groups, because this suite's house
style already writes full descriptive sentences (``test_the_boundary_sits_at_the_midpoint``)
that share a topic noun -- indistinguishable from genuinely flat, mechanical enumeration
(``test_<subject>_<case>``) by name alone.

The discriminator this module commits to: a candidate group of ``test_`` functions is
flat rather than merely topic-sharing only when its shared name prefix's tokens also
appear, contiguously and in order, inside the tokens of something the file actually
*calls* -- a bare function, a ``module.function``/``obj.method`` attribute call, with any
single leading underscore stripped. ``test_check_data_passes_on_matching_counts`` and
``test_check_data_fails_on_count_mismatch`` share the prefix ``check_data``, which is
exactly the imported ``check_data`` alias both bodies call through; ``test_a_named_worker_
count_reaches_the_val_loader`` and its four siblings share only ``a_named``/``an_auto``/
``the_val``, prose fragments no callable in the file is named after, so they are left
alone. A confirmed group of three or more still sitting at module level -- not yet grouped
under a ``class Test<Subject>:``, :func:`tests/eval/test_tile_merge.py`'s own established
pattern -- fails the audit.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_flat_test_groups.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTS_DIR = REPO_ROOT / "tests"
DEFAULT_SCRIPTS_TESTS_DIR = REPO_ROOT / "scripts" / "_tests"

#: Minimum group size a mechanical enumeration must reach before it is enforced -- a
#: two-member group is common even among genuinely descriptive, unrelated test names.
_MIN_GROUP_SIZE = 3
#: Token-prefix length used as the grouping key, e.g. ``("check", "data")``.
_PREFIX_TOKENS = 2


def _name_tokens(identifier: str) -> tuple[str, ...]:
    """Split a snake_case identifier into its non-empty underscore-delimited tokens.

    Examples:
        >>> _name_tokens("_plan_archives")
        ('plan', 'archives')
        >>> _name_tokens("o2o_rotated_topk")
        ('o2o', 'rotated', 'topk')
    """
    return tuple(token for token in identifier.split("_") if token)


def _call_target_tokens(tree: ast.Module) -> list[tuple[str, ...]]:
    """Every ``Call`` node's function-name tokens, one tuple per call site.

    Examples:
        >>> tree = ast.parse("check_data.check_dota_root(root)")
        >>> _call_target_tokens(tree)
        [('check', 'dota', 'root')]
    """
    targets = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            targets.append(_name_tokens(func.attr))
        elif isinstance(func, ast.Name):
            targets.append(_name_tokens(func.id))
    return targets


def _is_contiguous_subsequence(candidate: tuple[str, ...], target: tuple[str, ...]) -> bool:
    """True if ``candidate``'s tokens appear in order, contiguously, anywhere in ``target``.

    Examples:
        >>> _is_contiguous_subsequence(("rotated", "topk"), ("o2o", "rotated", "topk"))
        True
        >>> _is_contiguous_subsequence(("deployed", "view"), ("deploy",))
        False
    """
    span = len(candidate)
    return any(target[i : i + span] == candidate for i in range(len(target) - span + 1))


def module_level_test_names(tree: ast.Module) -> list[str]:
    """Names of ``tree``'s module-level (not-yet-classed) ``test_`` functions, in order.

    Examples:
        >>> tree = ast.parse("def test_a(): ...\\nclass C:\\n    def test_b(self): ...\\n")
        >>> module_level_test_names(tree)
        ['test_a']
    """
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    ]


def find_flat_groups(path: Path) -> dict[str, list[str]]:
    """Confirmed flat groups in ``path``: ``{"prefix_tokens": [test names]}``, size >= 3.

    A group is confirmed only when its shared prefix also names something the file calls
    (see the module docstring); an unconfirmed shared prefix -- house-style prose sharing a
    topic noun -- is not reported, however many names share it.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "test_sample.py"
        ...     _ = sample.write_text(
        ...         "from mod import check_data\\n"
        ...         "def test_check_data_a(): check_data(1)\\n"
        ...         "def test_check_data_b(): check_data(2)\\n"
        ...         "def test_check_data_c(): check_data(3)\\n"
        ...     )
        ...     find_flat_groups(sample)
        {'check_data': ['test_check_data_a', 'test_check_data_b', 'test_check_data_c']}
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    call_targets = _call_target_tokens(tree)

    candidates: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for name in module_level_test_names(tree):
        tokens = _name_tokens(name)[1:]  # drop the leading "test" token
        if len(tokens) < _PREFIX_TOKENS:
            continue
        candidates[tokens[:_PREFIX_TOKENS]].append(name)

    confirmed = {}
    for prefix, members in candidates.items():
        if len(members) < _MIN_GROUP_SIZE:
            continue
        if any(_is_contiguous_subsequence(prefix, target) for target in call_targets):
            confirmed["_".join(prefix)] = members
    return confirmed


def find_all_flat_groups(tests_dir: Path) -> list[str]:
    """Every ``file::prefix (n tests)`` flat group still outside a class under ``tests_dir``.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     sample = root / "test_sample.py"
        ...     _ = sample.write_text("def test_only_one(): ...\\n")
        ...     find_all_flat_groups(root)
        []
    """
    findings = []
    for path in sorted(tests_dir.rglob("test_*.py")):
        for prefix, members in sorted(find_flat_groups(path).items()):
            relative = path.relative_to(tests_dir.parent)
            findings.append(f"{relative}::{prefix} ({len(members)} tests)")
    return findings


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the audit, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` when any flat group survives outside a class.
    """
    parser = argparse.ArgumentParser(description="Audit tests/**/test_*.py for flat, unclassed mechanical groups.")
    parser.add_argument(
        "--tests-dir",
        type=Path,
        action="append",
        dest="tests_dirs",
        default=None,
        help=f"directory to scan; repeatable (default: {DEFAULT_TESTS_DIR}, {DEFAULT_SCRIPTS_TESTS_DIR})",
    )
    args = parser.parse_args(argv)
    tests_dirs = args.tests_dirs if args.tests_dirs is not None else [DEFAULT_TESTS_DIR, DEFAULT_SCRIPTS_TESTS_DIR]

    findings = [item for tests_dir in tests_dirs for item in find_all_flat_groups(tests_dir)]
    if findings:
        print(f"flat-test-group audit FAILED: {len(findings)} group(s) still outside a class")
        for item in findings:
            print(f"  - {item}")
        return 1
    print("flat-test-group audit clean: every confirmed mechanical group is classed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
