# SPDX-License-Identifier: Apache-2.0
"""Single-source-of-version audit for ``pyproject.toml`` / ``lucid_yolo.__init__`` (WP-001, split off WP-117).

``pyproject.toml`` once carried its own ``version`` string beside
``lucid_yolo.__version__`` and the two had already drifted apart (``0.1.0``
against ``0.0.1.dev0``) -- a build published one number while every runtime
consumer reported the other. The packaging metadata is now declared dynamic and
read from the module attribute, and this script pins that arrangement: no
static ``version`` key in ``[project]``, ``dynamic`` and
``[tool.setuptools.dynamic]`` wired to the module attribute, and the attribute
itself a bare string literal setuptools can resolve without importing torch.

Deliberately *not* checked here: ``importlib.metadata.version("lucid-yolo")``
against ``__version__``. An editable install freezes its metadata at install
time, so that assertion would fail on every version bump until someone
reinstalled -- a check that fails for a reason unrelated to the property it
claims to guard.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_version_single_source.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEFAULT_INIT = REPO_ROOT / "src" / "lucid_yolo" / "__init__.py"
DEFAULT_README = REPO_ROOT / "README.md"
VERSION_ATTR = "lucid_yolo.__version__"

#: The README's own statement of what the current release is, as the Versioning section
#: writes it. Packaging metadata was the drift this script was opened for; prose is the
#: drift it missed -- WP-175 found the README naming ``0.4.0`` while ``__version__`` had
#: reached ``0.7.0``, three releases with nothing to notice them. The most-read file in the
#: repository is a version consumer too, so it is held to the same single source.
README_VERSION_RE = re.compile(r"^Current release: \*\*(?P<version>[^*]+)\*\*", re.MULTILINE)


def _pyproject(pyproject_path: Path) -> dict[str, object]:
    """Parse a ``pyproject.toml`` file into a dict.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "pyproject.toml"
        ...     _ = sample.write_text('[project]\\nname = "demo"\\n')
        ...     "project" in _pyproject(sample)
        True
    """
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


def _table(container: dict[str, object], key: str) -> dict[str, object]:
    """Return the sub-table at ``key``, or an empty dict if absent or not a table.

    Examples:
        >>> _table({"tool": {"setuptools": {}}}, "tool")
        {'setuptools': {}}
        >>> _table({}, "tool")
        {}
    """
    value = container.get(key, {})
    return value if isinstance(value, dict) else {}


def check_pyproject_declares_no_static_version(pyproject_path: Path) -> list[str]:
    """``[project]`` states no ``version`` key -- the drift that made this script necessary.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "pyproject.toml"
        ...     _ = sample.write_text('[project]\\nname = "demo"\\nversion = "0.1.0"\\n')
        ...     check_pyproject_declares_no_static_version(sample)
        ['[project] declares a static "version" key; version must come from lucid_yolo.__version__ alone']
    """
    project = _pyproject(pyproject_path)["project"]
    assert isinstance(project, dict)
    if "version" in project:
        return [f'[project] declares a static "version" key; version must come from {VERSION_ATTR} alone']
    return []


def check_pyproject_reads_the_module_attribute(pyproject_path: Path) -> list[str]:
    """``version`` is dynamic and resolved from ``lucid_yolo.__version__``.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "pyproject.toml"
        ...     _ = sample.write_text('[project]\\nname = "demo"\\ndynamic = ["version"]\\n')
        ...     check_pyproject_reads_the_module_attribute(sample)
        ['[tool.setuptools.dynamic] does not point version at lucid_yolo.__version__']
    """
    config = _pyproject(pyproject_path)
    project = config["project"]
    assert isinstance(project, dict)
    violations = []
    if "version" not in project.get("dynamic", []):
        violations.append('[project] "dynamic" does not list "version"')
    dynamic = _table(_table(_table(config, "tool"), "setuptools"), "dynamic")
    if dynamic.get("version") != {"attr": VERSION_ATTR}:
        violations.append(f"[tool.setuptools.dynamic] does not point version at {VERSION_ATTR}")
    return violations


def _version_literals(init_path: Path) -> list[object]:
    """Every module-level ``__version__ = <constant>`` value assigned in ``init_path``.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "__init__.py"
        ...     _ = sample.write_text("__version__ = '1.2.3'\\n")
        ...     _version_literals(sample)
        ['1.2.3']
    """
    module = ast.parse(init_path.read_text(encoding="utf-8"))
    return [
        node.value.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    ]


def check_readme_names_the_current_version(readme_path: Path, init_path: Path) -> list[str]:
    """``README.md``'s "Current release" states the version the module attribute carries.

    The README is read far more than ``pyproject.toml`` and drifts more quietly: nothing
    fails when a release bumps ``__version__`` and leaves the prose behind, so the file
    that introduces the project keeps naming a release two or three minors old. Matching
    on the literal ``Current release: **X**`` line rather than on any version-shaped
    string keeps the check pinned to the one sentence that makes the claim -- every other
    version in the file (frozen golden sets, changelog references) is history and must not
    move.

    Args:
        readme_path: ``README.md`` to read.
        init_path: The ``__init__.py`` carrying ``__version__``.

    Returns:
        One violation if the README states no current release, one if it states a
        different one, and an empty list when the two agree.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     init = root / "__init__.py"
        ...     _ = init.write_text("__version__ = '0.7.0'\\n")
        ...     readme = root / "README.md"
        ...     _ = readme.write_text("Current release: **0.4.0** -- stale\\n")
        ...     check_readme_names_the_current_version(readme, init)
        ['README names 0.4.0 as the current release; lucid_yolo.__version__ is 0.7.0']
    """
    literals = _version_literals(init_path)
    if len(literals) != 1 or not isinstance(literals[0], str):
        return []  # check_version_is_a_plain_literal owns this failure and reports it once.
    version = literals[0]
    match = README_VERSION_RE.search(readme_path.read_text(encoding="utf-8"))
    if match is None:
        return [f'README states no "Current release: **<version>**" line to check against {VERSION_ATTR}']
    named = match.group("version").strip()
    if named != version:
        return [f"README names {named} as the current release; {VERSION_ATTR} is {version}"]
    return []


def check_version_is_a_plain_literal(init_path: Path) -> list[str]:
    """``__version__`` is a bare string literal, so setuptools resolves it without importing.

    A computed value (``importlib.metadata``, an f-string, a read from a file)
    would still be a valid attribute, but setuptools would have to *import* the
    package to evaluate it -- pulling torch into every build and every
    ``pip install`` of a source distribution.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "__init__.py"
        ...     _ = sample.write_text("__version__ = '1.2.3'\\n")
        ...     check_version_is_a_plain_literal(sample)
        []
    """
    literals = _version_literals(init_path)
    if len(literals) != 1 or not isinstance(literals[0], str):
        return ["__init__.py's __version__ is not a single bare string-literal assignment"]
    return []


def find_violations(pyproject_path: Path, init_path: Path, readme_path: Path) -> list[str]:
    """Every violation across the four single-source-of-version checks.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     pyproject = root / "pyproject.toml"
        ...     _ = pyproject.write_text(
        ...         '[project]\\nname = "demo"\\ndynamic = ["version"]\\n'
        ...         '[tool.setuptools.dynamic]\\nversion = {attr = "lucid_yolo.__version__"}\\n'
        ...     )
        ...     init = root / "__init__.py"
        ...     _ = init.write_text("__version__ = '1.2.3'\\n")
        ...     readme = root / "README.md"
        ...     _ = readme.write_text("Current release: **1.2.3** -- in step\\n")
        ...     find_violations(pyproject, init, readme)
        []
    """
    return [
        *check_pyproject_declares_no_static_version(pyproject_path),
        *check_pyproject_reads_the_module_attribute(pyproject_path),
        *check_version_is_a_plain_literal(init_path),
        *check_readme_names_the_current_version(readme_path, init_path),
    ]


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the audit, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` when any check fails.
    """
    parser = argparse.ArgumentParser(description="Audit pyproject.toml / __init__.py for a single version source.")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT,
        help=f"pyproject.toml to check (default: {DEFAULT_PYPROJECT})",
    )
    parser.add_argument(
        "--init",
        type=Path,
        default=DEFAULT_INIT,
        help=f"__init__.py to check (default: {DEFAULT_INIT})",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_README,
        help=f"README.md to check (default: {DEFAULT_README})",
    )
    args = parser.parse_args(argv)

    violations = find_violations(args.pyproject, args.init, args.readme)
    if violations:
        print(f"version-single-source-audit FAILED: {len(violations)} violation(s)")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("version-single-source-audit clean: pyproject.toml and __init__.py agree on one version source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
