# SPDX-License-Identifier: Apache-2.0
"""Doctest-coverage audit for test helpers and the shipped surface (WP-129/130, split off WP-117).

Scans ``tests/``, ``scripts/_tests/`` and ``src/`` by default. The first two hold
test helpers -- ``scripts/_tests/`` the functional-core tests for ``scripts/``
modules (WP-130) -- and carry the same helper-needs-an-Example expectation; the
third is the shipped surface, where ``Makefile``'s "every public function is
required to carry one" had no gate behind it at all.

``pytest --doctest-modules`` (WP-085, ``make test``) turns a helper's ``Examples:``
block into an executable check the moment it exists, but nothing forces the block
to exist in the first place. This walks the same module-level-function population
by the same AST predicate ``--doctest-modules`` already exercises, and fails naming
every helper still missing a ``>>>`` line, so a new one lands with an Example or
fails the gate rather than joining the pile silently.

**A skipped Example is not an executed one.** A ``>>>`` line carrying
``# doctest: +SKIP`` satisfies collection, satisfies a presence scan and satisfies
the human reading the diff, while executing nothing -- so a docstring whose every
Example line is skipped counts here only when each skip states why. The reason goes
*before* the directive (``>>> f()  # needs a checkpoint  # doctest: +SKIP``) and not
after it: CPython's ``doctest`` reads a directive comment to the end of its line and
rejects trailing prose there as an unknown option, so a reason written after
``+SKIP`` breaks the collection it is annotating.

Scope is deliberately narrow to the one invariant: a docstring's presence and its
``>>>`` content. Whether a group of helpers should be regrouped into a class is
WP-128's question, not this script's.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_test_doctests.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTS_DIR = REPO_ROOT / "tests"
DEFAULT_SCRIPTS_TESTS_DIR = REPO_ROOT / "scripts" / "_tests"
DEFAULT_SRC_DIR = REPO_ROOT / "src"

#: Decorator names that mark a function as a fixture rather than a helper, in
#: either bare (``@pytest.fixture``) or call (``@pytest.fixture(scope="module")``)
#: form, and under either the ``pytest.fixture`` or bare ``fixture`` import spelling.
_FIXTURE_DECORATOR_NAMES = frozenset({"fixture"})

#: A doctest example line inside a docstring.
_EXAMPLE_LINE_RE = re.compile(r"^\s*>>>")
#: A ``+SKIP`` directive on such a line. ``doctest`` reads the directive comment to
#: end of line, so nothing may follow the option list.
_SKIP_RE = re.compile(r"#\s*doctest:[^#\n]*\+SKIP")
#: A ``+SKIP`` preceded on the same line by a comment that is not itself a directive --
#: the only place a reason can sit without breaking ``doctest``'s own parse.
_REASONED_SKIP_RE = re.compile(r"#\s*(?!doctest:)\S[^#\n]*#\s*doctest:[^#\n]*\+SKIP")


@dataclass(frozen=True, slots=True)
class ScanRoot:
    """One tree to audit, and the population inside it that owes an Example.

    Attributes:
        path: Directory walked recursively.
        pattern: Glob selecting the files to parse under it.
        include_private: Whether a leading-underscore function owes an Example. True
            for test trees, where a module-level ``_helper`` is the whole population
            this audit was built for; False for ``src/``, where the requirement
            ``Makefile`` states is about the *public* surface and a private helper's
            contract is its caller's docstring.
    """

    path: Path
    pattern: str = "test_*.py"
    include_private: bool = True


def _decorator_name(node: ast.expr) -> str:
    """Return a decorator's trailing attribute or call name, e.g. ``"fixture"``.

    Handles the three shapes a decorator can take -- ``@fixture``, ``@fixture(...)``,
    and ``@pytest.fixture(...)`` -- by unwrapping a call to its callee first and then
    reading the attribute name off either a bare ``Name`` or an ``Attribute``.

    Examples:
        >>> import ast
        >>> _decorator_name(ast.parse("fixture", mode="eval").body)
        'fixture'
        >>> _decorator_name(ast.parse("pytest.fixture(scope='module')", mode="eval").body)
        'fixture'
    """
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return ""


def _is_fixture(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if any of ``func``'s decorators names a pytest fixture.

    Examples:
        >>> tree = ast.parse('''
        ... @pytest.fixture
        ... def thing(): ...
        ... ''')
        >>> _is_fixture(tree.body[0])
        True
    """
    return any(_decorator_name(dec) in _FIXTURE_DECORATOR_NAMES for dec in func.decorator_list)


def module_level_helpers(path: Path, include_private: bool = True) -> list[str]:
    """Names of ``path``'s module-level, non-fixture, non-``test_`` functions.

    Deliberately module-level only: a closure nested inside a helper (a fake
    ``urlopen`` built by a factory, say) is private to that helper's own docstring
    and Example already, not a second surface a caller can reach, so it is not
    counted here -- the same boundary ``--doctest-modules`` draws when it collects
    a module's top-level functions.

    Args:
        path: Python file to parse.
        include_private: Whether leading-underscore functions are counted.

    Returns:
        The function names owing an Example, in file order.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "test_sample.py"
        ...     _ = sample.write_text(
        ...         "import pytest\\n"
        ...         "@pytest.fixture\\n"
        ...         "def a_fixture(): ...\\n"
        ...         "def _helper(): ...\\n"
        ...         "def public(): ...\\n"
        ...         "def test_thing(): ...\\n"
        ...     )
        ...     module_level_helpers(sample), module_level_helpers(sample, include_private=False)
        (['_helper', 'public'], ['public'])
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("test_") or _is_fixture(node):
            continue
        if not include_private and node.name.startswith("_"):
            continue
        names.append(node.name)
    return names


def example_gap(doc: str | None) -> str | None:
    """Why ``doc`` fails to carry an executable Example, or ``None`` when it does not fail.

    Two distinct failures, reported apart because they call for different repairs:
    a docstring with no ``>>>`` line at all needs an Example written, while one whose
    every ``>>>`` line is skipped already has an Example and needs either the skip
    removed or a reason stated for it.

    Args:
        doc: A function's docstring, or ``None`` when it has none.

    Returns:
        A short reason, or ``None`` when at least one Example line will actually run
        (or is skipped for a stated reason).

    Examples:
        >>> example_gap(">>> 1 + 1\\n2\\n") is None
        True
        >>> example_gap("No example here.")
        'no Example'
        >>> example_gap(">>> f()  # doctest: +SKIP")
        'every Example line is skipped, none says why'
        >>> example_gap(">>> f()  # needs a checkpoint  # doctest: +SKIP") is None
        True
    """
    if doc is None:
        return "no docstring"
    lines = [line for line in doc.splitlines() if _EXAMPLE_LINE_RE.match(line)]
    if not lines:
        return "no Example"
    countable = [line for line in lines if not _SKIP_RE.search(line) or _REASONED_SKIP_RE.search(line) is not None]
    if not countable:
        return "every Example line is skipped, none says why"
    return None


def has_doctest_example(path: Path, name: str) -> bool:
    """True if the module-level function ``name`` in ``path`` carries an executable Example.

    A missing docstring also fails this check (``ast.get_docstring`` returns
    ``None``, and containment on ``None`` would raise, so the ``None`` case is
    handled explicitly), which keeps the predicate a single check rather than
    two the caller has to combine. So does a docstring whose Example lines are all
    skipped without a stated reason -- see :func:`example_gap`.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "test_sample.py"
        ...     _ = sample.write_text(
        ...         "def _documented():\\n"
        ...         "    '''One line.\\n\\n"
        ...         "    >>> 1 + 1\\n"
        ...         "    2\\n"
        ...         "    '''\\n"
        ...         "def _bare():\\n"
        ...         "    'No example here.'\\n"
        ...     )
        ...     has_doctest_example(sample, "_documented"), has_doctest_example(sample, "_bare")
        (True, False)
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return example_gap(ast.get_docstring(node)) is None
    raise AssertionError(f"{name} not found at module level of {path}")  # pragma: no cover


def find_missing(tests_dir: Path, pattern: str = "test_*.py", include_private: bool = True) -> list[str]:
    """Every ``file::function -- reason`` under ``tests_dir`` without an executable Example.

    Args:
        tests_dir: Directory walked recursively.
        pattern: Glob selecting the files to parse under it.
        include_private: Whether leading-underscore functions owe an Example.

    Returns:
        One finding per function, each naming why it failed.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     sample = root / "test_sample.py"
        ...     _ = sample.write_text("def _bare():\\n    'No example.'\\n")
        ...     [item.split("/")[-1] for item in find_missing(root)]
        ['test_sample.py::_bare -- no Example']
    """
    missing = []
    for path in sorted(tests_dir.rglob(pattern)):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        documented = {
            node.name: ast.get_docstring(node)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in module_level_helpers(path, include_private=include_private):
            gap = example_gap(documented[name])
            if gap is not None:
                missing.append(f"{path.relative_to(tests_dir.parent)}::{name} -- {gap}")
    return missing


def _scan_roots(tests_dirs: list[Path] | None, src_dirs: list[Path] | None) -> list[ScanRoot]:
    """Resolve the CLI's two repeatable directory options into scan roots.

    Naming either option replaces the whole default set rather than adding to it, so
    ``--tests-dir tests`` audits exactly that and nothing else -- the alternative,
    where a narrowing flag silently keeps scanning ``src/``, makes the flag useless
    for the one job it has.

    Args:
        tests_dirs: Directories given as ``--tests-dir``, or ``None``.
        src_dirs: Directories given as ``--src-dir``, or ``None``.

    Returns:
        The roots to walk, in the order given.

    Examples:
        >>> [root.pattern for root in _scan_roots(None, None)]
        ['test_*.py', 'test_*.py', '*.py']
        >>> [root.include_private for root in _scan_roots(None, [Path("src")])]
        [False]
    """
    if tests_dirs is None and src_dirs is None:
        tests_dirs, src_dirs = [DEFAULT_TESTS_DIR, DEFAULT_SCRIPTS_TESTS_DIR], [DEFAULT_SRC_DIR]
    roots = [ScanRoot(path) for path in tests_dirs or []]
    roots += [ScanRoot(path, pattern="*.py", include_private=False) for path in src_dirs or []]
    return roots


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the audit, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` when any function lacks an executable Example.

    Examples:
        Command-line usage (exit status is the process return code)::

            $ python scripts/lint/audit_test_doctests.py --tests-dir tests
    """
    parser = argparse.ArgumentParser(description="Audit test helpers and src/ public functions for Examples.")
    parser.add_argument(
        "--tests-dir",
        type=Path,
        action="append",
        dest="tests_dirs",
        default=None,
        help=f"test tree to scan for test_*.py helpers; repeatable "
        f"(default: {DEFAULT_TESTS_DIR}, {DEFAULT_SCRIPTS_TESTS_DIR})",
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        action="append",
        dest="src_dirs",
        default=None,
        help=f"source tree to scan for public functions in *.py; repeatable (default: {DEFAULT_SRC_DIR})",
    )
    args = parser.parse_args(argv)
    roots = _scan_roots(args.tests_dirs, args.src_dirs)

    missing = [item for root in roots for item in find_missing(root.path, root.pattern, root.include_private)]
    if missing:
        print(f"doctest-audit FAILED: {len(missing)} function(s) without an executable Example")
        for item in missing:
            print(f"  - {item}")
        return 1
    print("doctest-audit clean: every scanned function carries an Example that runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
