# SPDX-License-Identifier: Apache-2.0
"""Release guard: decide whether a version tag may ship (WP-006).

The guard runs six independent checks and lets a tag ship only when all six
pass:

1. **Tag** — the tag must be a ``v0.MINOR.PATCH`` string. A ``1.x`` (or higher)
   tag is refused with an ADR-002 message: the project is a perpetual 0.x
   release train and no 1.0 is ever planned, promised, or tagged. Any other
   shape is refused as malformed.
2. **Changelog** — the changelog must carry a ``## [MINOR.PATCH]`` section for
   the tag, so every release ships with its notes written.
3. **Version** — ``lucid_yolo.__version__`` must be the version the tag names.
   Added by WP-168: the changelog check above is a substring test and never reads
   the package version, and ``audit_version_single_source.py`` checks only that
   ``pyproject.toml`` defers to ``__version__`` — neither of them ever sees a tag.
   Between them, ``v0.8.0`` over a tree still declaring ``0.7.0`` passed every
   check and shipped a wheel whose version contradicted the tag that built it.
   The version is parsed statically rather than imported: importing the package
   executes it, and a guard deciding whether a distribution is shippable must not
   need that distribution to be importable first.
4. **Frozen goldens** — ``goldens/frozen/<MAJOR.MINOR>/`` must exist and hold at
   least one file. A release's whole regression claim is that current code still
   satisfies every value the release pinned (``scripts/check_goldens.py``), and a
   minor tagged without its snapshot makes that claim about nothing. Also WP-168.
5. **Dependency tiers** — every distribution the shipped package imports must be
   declared in ``[project].dependencies``. Added by WP-160 after WP-159 found the
   guard blind to the one thing a release is: what a consumer installing the
   distribution actually receives. Phase 14 moved five modules under ``data/`` onto
   ``fuse-augmentations`` while its requirement sat in the ``dev`` dependency group,
   so an install omitting that group produced a package that raised ``ImportError``
   from ``lucid_yolo.data`` — no test caught it, because the development
   environment installs every group.
6. **Gate** — the gate command (``make gate`` by default) must exit ``0``. A
   non-zero exit is a red gate and refuses the tag.

Deliberately **not** checked: whether the runtime requirements are uploadable to
PyPI. ``[project].dependencies`` currently carries a direct reference — a git URL,
legal to build and install and illegal to upload — accepted as D20 with the
consequence recorded, so a check refusing it would re-litigate a decision rather
than protect one. What the tier check protects is different and unconditional: a
distribution that cannot import is broken however it was obtained.

The check functions are importable and pure over their inputs; :func:`main` is a
thin CLI that prints one verdict line per check and exits non-zero on refusal.

Examples:
    Guard a well-formed tag against a changelog that names it, with a trivially
    green gate::

        python scripts/release_guard.py --tag v0.1.0 --gate-cmd true
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from importlib.metadata import packages_distributions
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

#: Repository root (``scripts/`` is one level below it).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Default changelog consulted for the tag's version section.
DEFAULT_CHANGELOG = REPO_ROOT / "CHANGELOG.md"

#: Default manifest read for the declared dependency tiers.
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Default import root: the package a built distribution actually ships.
DEFAULT_PACKAGE_ROOT = REPO_ROOT / "src" / "lucid_yolo"

#: Default module read for the single-source ``__version__`` the tag must match.
DEFAULT_INIT = DEFAULT_PACKAGE_ROOT / "__init__.py"

#: Default root holding one ``<MAJOR.MINOR>/`` snapshot directory per release.
DEFAULT_FROZEN_ROOT = REPO_ROOT / "goldens" / "frozen"

#: Default gate command re-run before a tag may ship.
DEFAULT_GATE_CMD = "make gate"

#: A shippable tag: zero major, per ADR-002's perpetual 0.x train.
ZERO_MAJOR_TAG = re.compile(r"^v0\.\d+\.\d+$")

#: Any ``v<major>.<minor>.<patch>`` tag, used to detect a refused 1.x+ tag.
SEMVER_TAG = re.compile(r"^v(\d+)\.\d+\.\d+$")

#: Refusal message for a major-version->=1 tag (ADR-002, hard-coded).
ADR_002_REFUSAL = (
    "tag {tag} declares major version >= 1; ADR-002 mandates a perpetual 0.x "
    "release train — no 1.0 is ever planned, promised, or tagged"
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one release-guard check.

    Attributes:
        name: Short check name (``tag``, ``changelog``, ``version``, ``goldens``,
            ``dependencies``, or ``gate``).
        passed: Whether the check permits the tag to ship.
        detail: Human-readable verdict explaining the outcome.
    """

    name: str
    passed: bool
    detail: str


def check_tag(tag: str) -> CheckResult:
    """Validate that ``tag`` is a shippable ``v0.MINOR.PATCH`` string.

    A zero-major tag passes. A syntactically valid tag whose major version is
    ``>= 1`` is refused with the ADR-002 message. Anything else is refused as
    malformed.

    Args:
        tag: The candidate tag, e.g. ``"v0.1.0"``.

    Returns:
        A :class:`CheckResult` named ``"tag"``.

    Examples:
        ```pycon
        >>> check_tag("v0.1.0").passed
        True
        >>> check_tag("v1.0.0").passed
        False
        >>> check_tag("0.1").passed
        False

        ```
    """
    if ZERO_MAJOR_TAG.match(tag):
        return CheckResult("tag", True, f"tag {tag} matches the perpetual 0.x scheme")
    semver = SEMVER_TAG.match(tag)
    if semver is not None and int(semver.group(1)) >= 1:
        return CheckResult("tag", False, ADR_002_REFUSAL.format(tag=tag))
    return CheckResult("tag", False, f"tag {tag!r} is malformed; expected 'v0.MINOR.PATCH'")


def check_changelog(tag: str, changelog: Path) -> CheckResult:
    """Verify that ``changelog`` carries a ``## [MINOR.PATCH]`` section for ``tag``.

    Args:
        tag: The candidate tag; its leading ``v`` is stripped to form the version.
        changelog: Path to the changelog file to read.

    Returns:
        A :class:`CheckResult` named ``"changelog"``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "CHANGELOG.md"
        ...     _ = path.write_text("## [0.1.0]\\n- first release\\n")
        ...     check_changelog("v0.1.0", path).passed
        True

        ```
    """
    heading = f"## [{tag.removeprefix('v')}]"
    try:
        text = changelog.read_text(encoding="utf-8")
    except OSError as exc:
        return CheckResult("changelog", False, f"cannot read changelog {changelog}: {exc}")
    if heading not in text:
        return CheckResult("changelog", False, f"changelog {changelog} has no {heading!r} section")
    return CheckResult("changelog", True, f"changelog carries a {heading!r} section")


def _declared_version(init_path: Path) -> str | None:
    """Read ``__version__`` out of a module's source without executing it.

    Args:
        init_path: The module assigning ``__version__`` as a string literal.

    Returns:
        The assigned version, or ``None`` when the module makes no such literal
        assignment — a computed or absent ``__version__`` is not a version this can
        read, and saying so beats returning a guess.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "__init__.py"
        ...     _ = path.write_text('__version__ = "0.4.2"\\n')
        ...     _declared_version(path)
        '0.4.2'

        ```
    """
    for node in ast.walk(ast.parse(init_path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "__version__" in names and isinstance(node.value.value, str):
            return node.value.value
    return None


def check_version(tag: str, init_path: Path = DEFAULT_INIT) -> CheckResult:
    """Verify that the package's own ``__version__`` is the version ``tag`` names.

    Args:
        tag: The candidate tag; its leading ``v`` is stripped to form the version.
        init_path: Module declaring ``__version__`` (default: the shipped package's).

    Returns:
        A :class:`CheckResult` named ``"version"``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "__init__.py"
        ...     _ = path.write_text('__version__ = "0.7.0"\\n')
        ...     (check_version("v0.7.0", path).passed, check_version("v0.8.0", path).passed)
        (True, False)

        ```
    """
    version = tag.removeprefix("v")
    try:
        declared = _declared_version(init_path)
    except (OSError, SyntaxError) as exc:
        return CheckResult("version", False, f"cannot read {init_path}: {exc}")
    if declared is None:
        return CheckResult("version", False, f"{init_path} assigns no literal __version__ this guard can read")
    if declared != version:
        return CheckResult(
            "version",
            False,
            f"tag {tag} names version {version} but {init_path.parent.name}.__version__ is {declared!r}; "
            "the wheel a tag builds carries the package's version, not the tag's",
        )
    return CheckResult("version", True, f"tag {tag} matches __version__ {declared!r}")


def check_frozen_goldens(tag: str, frozen_root: Path = DEFAULT_FROZEN_ROOT) -> CheckResult:
    """Verify that the tag's minor carries a non-empty frozen-golden snapshot.

    Args:
        tag: The candidate tag; its ``MAJOR.MINOR`` prefix names the snapshot directory.
        frozen_root: Root holding one directory per released minor (default:
            ``<repo>/goldens/frozen``).

    Returns:
        A :class:`CheckResult` named ``"goldens"``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     (root / "0.4").mkdir()
        ...     _ = (root / "0.4" / "optim_toy.json").write_text("{}")
        ...     (check_frozen_goldens("v0.4.1", root).passed, check_frozen_goldens("v0.5.0", root).passed)
        (True, False)

        ```
    """
    minor = ".".join(tag.removeprefix("v").split(".")[:2])
    snapshot = frozen_root / minor
    frozen = sorted(path for path in snapshot.glob("*.json") if path.is_file()) if snapshot.is_dir() else []
    if not frozen:
        return CheckResult(
            "goldens",
            False,
            f"tag {tag} has no frozen goldens at {snapshot}; a release pins the values current code must keep "
            "satisfying, and an unfrozen minor claims a regression guard it never wrote — run "
            f"'make freeze-goldens MINOR={minor}'",
        )
    return CheckResult("goldens", True, f"{len(frozen)} frozen golden(s) snapshotted at {snapshot}")


def _imported_top_levels(package_root: Path) -> set[str]:
    """Collect the top-level modules ``package_root`` imports, minus stdlib and itself.

    Absolute imports only: a relative ``from .x import y`` names nothing outside the
    package, and a conditional or function-local import is still an import the
    installed package can execute, so the walk is over every ``Import`` node rather
    than the module preamble alone.

    Args:
        package_root: Directory of the package a built distribution ships.

    Returns:
        Top-level module names, e.g. ``{"torch", "yaml"}``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp) / "pkg"
        ...     root.mkdir()
        ...     _ = (root / "m.py").write_text("import os\\nfrom torch import nn\\n")
        ...     sorted(_imported_top_levels(root))
        ['torch']

        ```
    """
    tops: set[str] = set()
    for path in sorted(package_root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                tops.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                tops.add(node.module.split(".")[0])
    return {top for top in tops if top not in sys.stdlib_module_names} - {package_root.name}


def _declared_tiers(pyproject: Path) -> tuple[set[str], dict[str, str]]:
    """Read which distributions each dependency tier declares.

    Args:
        pyproject: Manifest to read.

    Returns:
        The canonicalized names in ``[project].dependencies``, and a mapping from
        every other declared name to the dependency group that declares it — the
        second being what turns a refusal into a diagnosis rather than a complaint.
    """
    manifest = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    runtime = {str(canonicalize_name(Requirement(req).name)) for req in manifest["project"].get("dependencies", [])}
    grouped: dict[str, str] = {}
    for group, requirements in manifest.get("dependency-groups", {}).items():
        for req in requirements:
            if not isinstance(req, str):
                continue
            name = canonicalize_name(Requirement(req).name)
            grouped.setdefault(name, group)
    return runtime, grouped


def check_dependency_tiers(
    pyproject: Path,
    package_root: Path,
    distributions: dict[str, list[str]] | None = None,
) -> CheckResult:
    """Refuse a tag whose package imports something ``[project].dependencies`` omits.

    A distribution declared only in a dependency group is absent from the built
    wheel's own ``Requires-Dist``, so an install that does not ask for that group
    receives a package raising ``ImportError`` on the module that needs it. The
    development environment installs every group, which is why no test sees this and
    why it belongs to the release guard rather than to the suite.

    A module that resolves to no installed distribution is reported rather than
    ignored: it means the environment cannot answer the question, and a guard that
    treats "unknown" as "fine" is the failure mode this check exists to close.

    Args:
        pyproject: Manifest declaring the tiers.
        package_root: Directory of the package the distribution ships.
        distributions: Top-level module to distribution names, defaulting to the
            installed environment's own mapping. Injectable so the check is testable
            against a stated environment rather than whatever happens to be present.

    Returns:
        A :class:`CheckResult` named ``"dependencies"``.

    Examples:
        A module whose distribution sits in a group, not in the runtime tier:

        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> manifest = '[project]\\ndependencies = ["torch"]\\n'
        >>> manifest += '[dependency-groups]\\ndev = ["helper-lib"]\\n'
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     toml = Path(tmp) / "pyproject.toml"
        ...     _ = toml.write_text(manifest)
        ...     root = Path(tmp) / "pkg"
        ...     root.mkdir()
        ...     _ = (root / "m.py").write_text("import helper\\n")
        ...     result = check_dependency_tiers(toml, root, {"helper": ["helper-lib"]})
        >>> result.passed
        False
        >>> "dev" in result.detail
        True

        ```
    """
    runtime, grouped = _declared_tiers(pyproject)
    mapping = packages_distributions() if distributions is None else distributions
    offenders: list[str] = []
    for top in sorted(_imported_top_levels(package_root)):
        dists = {canonicalize_name(dist) for dist in mapping.get(top, [])}
        if not dists:
            offenders.append(f"{top} (no installed distribution provides it)")
        elif not dists & runtime:
            declared = sorted({grouped[dist] for dist in dists if dist in grouped})
            where = f"dependency-group {declared[0]!r}" if declared else "nothing"
            offenders.append(f"{top} (provided by {sorted(dists)[0]}, declared in {where})")
    if offenders:
        return CheckResult(
            "dependencies",
            False,
            f"{package_root.name} imports {len(offenders)} module(s) absent from [project].dependencies, "
            f"so an install without every dependency group cannot import them: {'; '.join(offenders)}",
        )
    return CheckResult(
        "dependencies",
        True,
        f"every third-party module {package_root.name} imports is declared in [project].dependencies",
    )


def check_gate(gate_cmd: str) -> CheckResult:
    """Run ``gate_cmd`` and pass only when it exits ``0``.

    Args:
        gate_cmd: A shell command string (e.g. ``"make gate"``), run from the
            repository root.

    Returns:
        A :class:`CheckResult` named ``"gate"``; a non-zero exit is a red gate.

    Examples:
        ```pycon
        >>> check_gate("true").passed
        True
        >>> check_gate("false").passed
        False

        ```
    """
    try:
        # gate_cmd is an operator-supplied command string (e.g. "make gate"), run as a shell command by design.
        completed = subprocess.run(gate_cmd, shell=True, cwd=REPO_ROOT, check=False)
    except OSError as exc:
        return CheckResult("gate", False, f"gate command {gate_cmd!r} failed to launch: {exc}")
    if completed.returncode != 0:
        return CheckResult("gate", False, f"gate command {gate_cmd!r} exited {completed.returncode} (red gate)")
    return CheckResult("gate", True, f"gate command {gate_cmd!r} passed")


def _current_tag() -> str | None:
    """Return the tag exactly naming ``HEAD``, or ``None`` when there is none.

    Backs :func:`main`'s ``--tag`` default so the guard is runnable as a
    pre-commit hook with no per-invocation argument: most commits are not on a
    tag, and the hook has nothing to check for them.

    Examples:
        >>> _current_tag() is None or isinstance(_current_tag(), str)
        True
    """
    completed = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def evaluate(
    tag: str,
    changelog: Path,
    gate_cmd: str,
    pyproject: Path = DEFAULT_PYPROJECT,
    package_root: Path = DEFAULT_PACKAGE_ROOT,
    init_path: Path = DEFAULT_INIT,
    frozen_root: Path = DEFAULT_FROZEN_ROOT,
) -> list[CheckResult]:
    """Run every release-guard check and return their results in order.

    The gate runs last because it is the only expensive check: a malformed tag, an
    unwritten changelog section, a version the package contradicts, a missing frozen
    snapshot or a mis-tiered dependency is answerable in milliseconds, and there is
    no reason to spend three minutes on the suite first.

    Args:
        tag: The candidate tag.
        changelog: Path to the changelog file.
        gate_cmd: Shell command whose zero exit means a green gate.
        pyproject: Manifest declaring the dependency tiers.
        package_root: Directory of the package the distribution ships.
        init_path: Module declaring the ``__version__`` the tag must match.
        frozen_root: Root holding one frozen-golden directory per released minor.

    Returns:
        The ``[tag, changelog, version, goldens, dependencies, gate]`` results.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     changelog = Path(tmp) / "CHANGELOG.md"
        ...     _ = changelog.write_text("## [0.1.0]\\n")
        ...     toml = Path(tmp) / "pyproject.toml"
        ...     _ = toml.write_text('[project]\\ndependencies = []\\n')
        ...     root = Path(tmp) / "pkg"
        ...     root.mkdir()
        ...     _ = (root / "__init__.py").write_text('__version__ = "0.1.0"\\n')
        ...     frozen = Path(tmp) / "frozen" / "0.1"
        ...     frozen.mkdir(parents=True)
        ...     _ = (frozen / "optim_toy.json").write_text("{}")
        ...     results = evaluate(
        ...         "v0.1.0", changelog, "true", toml, root, root / "__init__.py", frozen.parent
        ...     )
        ...     [r.name for r in results]
        ['tag', 'changelog', 'version', 'goldens', 'dependencies', 'gate']

        ```
    """
    return [
        check_tag(tag),
        check_changelog(tag, changelog),
        check_version(tag, init_path),
        check_frozen_goldens(tag, frozen_root),
        check_dependency_tiers(pyproject, package_root),
        check_gate(gate_cmd),
    ]


def main(argv: list[str] | None = None) -> int:
    """Guard a release tag and print each check's verdict.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` only when the tag, changelog, version, frozen-golden, dependency-tier
        and gate checks all pass; ``1`` otherwise.

    Examples:
        The verdict lines go to stdout and name the changelog by absolute path,
        so the example captures them and asserts the exit code alone.

        ```pycon
        >>> import contextlib, io
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     status = main(["--tag", "v1.0.0", "--gate-cmd", "true"])
        >>> status
        1

        ```
    """
    parser = argparse.ArgumentParser(description="Decide whether a release tag may ship.")
    parser.add_argument(
        "--tag",
        default=None,
        help="candidate tag, e.g. v0.1.0 (default: the tag exactly naming HEAD, if any)",
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        default=DEFAULT_CHANGELOG,
        help="changelog that must carry the tag's version section (default: <repo>/CHANGELOG.md)",
    )
    parser.add_argument(
        "--gate-cmd",
        default=DEFAULT_GATE_CMD,
        help="gate command re-run before shipping; non-zero exit refuses the tag (default: 'make gate')",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT,
        help="manifest whose [project].dependencies must cover every shipped import (default: <repo>/pyproject.toml)",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=DEFAULT_PACKAGE_ROOT,
        help="package the distribution ships, walked for imports (default: <repo>/src/lucid_yolo)",
    )
    parser.add_argument(
        "--init",
        type=Path,
        default=DEFAULT_INIT,
        dest="init_path",
        help="module whose __version__ the tag must match (default: <repo>/src/lucid_yolo/__init__.py)",
    )
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=DEFAULT_FROZEN_ROOT,
        help="root holding one frozen-golden directory per released minor (default: <repo>/goldens/frozen)",
    )
    args = parser.parse_args(argv)

    tag = args.tag if args.tag is not None else _current_tag()
    if tag is None:
        print("release guard: HEAD is not exactly a tag — nothing to check")
        return 0

    results = evaluate(
        tag,
        args.changelog,
        args.gate_cmd,
        args.pyproject,
        args.package_root,
        args.init_path,
        args.frozen_root,
    )
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        print(f"{marker} [{result.name}] {result.detail}")
    passed = all(result.passed for result in results)
    print("release guard: " + ("all checks passed — tag may ship" if passed else "checks failed — tag refused"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
