# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the release guard refuses unshippable tags (WP-006).

Covers the DoD negative cases — a tag on a red gate, a tag missing its changelog
section, and a 1.x tag (ADR-002) are each refused — plus the happy path and a
malformed tag string. The guard is loaded via the spec-loading pattern used in
``scripts/_tests/test_audit_licenses.py`` so ``scripts/`` need not be importable.

WP-168 added the version and frozen-golden checks, and with them the reason every
``main`` call below now passes ``--init`` and ``--frozen-root``: those two checks
default to the live repository, so a test tagging ``v0.1.0`` would otherwise be
asserting against whatever version this tree happens to declare today and whichever
minors happen to be frozen. The fixtures state the release they describe instead.

The ``--metadata-only`` tests are the first in this file whose subject is the
*wiring* rather than the functions, and they exist because the guard's default
gate command is ``make gate``, which runs pre-commit, which runs the guard, so on a
tagged commit the fused form asked for a gate from inside the gate it was running.
Every function it calls was covered and the loop was still there, because what was
wrong was which caller ran which checks. Those two are asserted against the
committed ``.pre-commit-config.yaml`` and ``release.yml``, which is where the answer
lives; the gate-invocation tests count executions through a receipt file, so "runs
the gate exactly once" is measured rather than read.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "scripts" / "release_guard.py"
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"


def _load_guard() -> ModuleType:
    """Load ``scripts/release_guard.py`` as an importable module.

    The module is registered in ``sys.modules`` before execution so its frozen
    dataclass resolves its own module under ``from __future__ import annotations``.

    Examples:
        >>> callable(_load_guard().main)
        True
    """
    spec = importlib.util.spec_from_file_location("release_guard", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def _write_changelog(tmp_path: Path, version: str) -> Path:
    """Write a minimal changelog carrying a ``## [<version>]`` section.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = _write_changelog(Path(tmp), "0.1.0")
        ...     "## [0.1.0]" in path.read_text()
        True
    """
    path = tmp_path / "CHANGELOG.md"
    path.write_text(f"# Changelog\n\n## [{version}]\n\n### Added\n\n- thing\n", encoding="utf-8")
    return path


def _release_flags(tmp_path: Path, version: str) -> list[str]:
    """Point the version and frozen-golden checks at a synthetic release of ``version``.

    Writes a package declaring that ``__version__`` and a frozen-golden snapshot for
    its minor, and returns the ``--init``/``--frozen-root`` flags naming them. Both
    checks otherwise read the live repository, which would make every test below an
    assertion about this tree's current release rather than about the guard.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     flags = _release_flags(Path(tmp), "0.1.0")
        ...     flags[0], Path(flags[1]).read_text()
        ('--init', '__version__ = "0.1.0"\\n')
    """
    init = tmp_path / "pkg" / "__init__.py"
    init.parent.mkdir(exist_ok=True)
    init.write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    frozen = tmp_path / "frozen" / ".".join(version.split(".")[:2])
    frozen.mkdir(parents=True, exist_ok=True)
    (frozen / "optim_toy.json").write_text("{}", encoding="utf-8")
    return ["--init", str(init), "--frozen-root", str(frozen.parent)]


def test_red_gate_refuses_tag(tmp_path: Path) -> None:
    """A valid tag with its changelog section is refused when the gate is red."""
    changelog = _write_changelog(tmp_path, "0.1.0")
    args = ["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", "false"]
    assert guard.main([*args, *_release_flags(tmp_path, "0.1.0")]) == 1


def test_missing_changelog_section_refuses_tag(tmp_path: Path) -> None:
    """A valid tag on a green gate is refused when its changelog section is absent."""
    changelog = _write_changelog(tmp_path, "0.2.0")  # section for a different version
    args = ["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", "true"]
    assert guard.main([*args, *_release_flags(tmp_path, "0.1.0")]) == 1


def test_one_x_tag_refused_by_adr_002(tmp_path: Path) -> None:
    """A 1.x tag is refused even with a green gate and a matching changelog section."""
    changelog = _write_changelog(tmp_path, "1.0.0")
    args = ["--tag", "v1.0.0", "--changelog", str(changelog), "--gate-cmd", "true"]
    assert guard.main([*args, *_release_flags(tmp_path, "1.0.0")]) == 1


def test_happy_path_passes(tmp_path: Path) -> None:
    """A v0.MINOR.PATCH tag with its changelog section and a green gate ships."""
    changelog = _write_changelog(tmp_path, "0.1.0")
    args = ["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", "true"]
    assert guard.main([*args, *_release_flags(tmp_path, "0.1.0")]) == 0


def test_malformed_tag_refused(tmp_path: Path) -> None:
    """A tag that is not a v<major>.<minor>.<patch> string is refused."""
    changelog = _write_changelog(tmp_path, "0.1")
    args = ["--tag", "v0.1", "--changelog", str(changelog), "--gate-cmd", "true"]
    assert guard.main([*args, *_release_flags(tmp_path, "0.1")]) == 1


def test_version_mismatch_refuses_tag(tmp_path: Path) -> None:
    """A tag naming a version the package does not declare is refused.

    WP-168's own defect, stated as a test: tagging ``v0.8.0`` over a tree still
    declaring ``0.7.0`` passed the tag, changelog, dependency-tier and gate checks
    and shipped a wheel whose version contradicted the tag that built it. Neither
    the changelog substring test nor ``audit_version_single_source.py`` ever reads
    a tag, so nothing in the repository compared the two.
    """
    changelog = _write_changelog(tmp_path, "0.8.0")
    args = ["--tag", "v0.8.0", "--changelog", str(changelog), "--gate-cmd", "true"]
    flags = _release_flags(tmp_path, "0.7.0")
    (tmp_path / "frozen" / "0.8").mkdir()
    (tmp_path / "frozen" / "0.8" / "optim_toy.json").write_text("{}", encoding="utf-8")

    assert guard.main([*args, *flags]) == 1


def test_missing_frozen_goldens_refuses_tag(tmp_path: Path) -> None:
    """A tag whose minor has no frozen-golden snapshot is refused.

    A release's regression claim is that current code still satisfies every value the
    release pinned. A minor tagged with no ``goldens/frozen/<minor>/`` makes that
    claim about an empty set, and every later run of the harness agrees with it.
    """
    changelog = _write_changelog(tmp_path, "0.8.0")
    args = ["--tag", "v0.8.0", "--changelog", str(changelog), "--gate-cmd", "true"]
    flags = _release_flags(tmp_path, "0.8.0")
    (tmp_path / "frozen" / "0.8" / "optim_toy.json").unlink()

    assert guard.main([*args, *flags]) == 1


def test_version_check_reads_the_source_without_importing_it(tmp_path: Path) -> None:
    """``__version__`` is parsed statically, so an unimportable package still answers.

    A guard deciding whether a distribution is shippable must not require that
    distribution to import first — the import would run the package it is judging.
    """
    init = tmp_path / "__init__.py"
    init.write_text('raise SystemExit("importing me is fatal")\n__version__ = "0.7.0"\n', encoding="utf-8")

    assert guard.check_version("v0.7.0", init).passed


def test_version_check_reports_an_unreadable_version(tmp_path: Path) -> None:
    """A module assigning no literal ``__version__`` is refused rather than assumed to match."""
    init = tmp_path / "__init__.py"
    init.write_text("__version__ = compute_it()\n", encoding="utf-8")

    result = guard.check_version("v0.7.0", init)

    assert not result.passed
    assert "no literal __version__" in result.detail


def test_frozen_goldens_check_names_the_freeze_command(tmp_path: Path) -> None:
    """The refusal says which command writes the missing snapshot, so it diagnoses."""
    result = guard.check_frozen_goldens("v0.9.0", tmp_path)

    assert not result.passed
    assert "make freeze-goldens MINOR=0.9" in result.detail


def test_frozen_goldens_check_accepts_a_patch_release_of_a_frozen_minor(tmp_path: Path) -> None:
    """A patch tag reuses its minor's snapshot: 0.7.1 ships against goldens/frozen/0.7."""
    (tmp_path / "0.7").mkdir()
    (tmp_path / "0.7" / "optim_toy.json").write_text("{}", encoding="utf-8")

    assert guard.check_frozen_goldens("v0.7.1", tmp_path).passed


def test_adr_002_refusal_names_the_policy() -> None:
    """The 1.x refusal detail cites ADR-002's perpetual 0.x train, hard-coded."""
    result = guard.check_tag("v1.4.2")
    assert not result.passed
    assert "ADR-002" in result.detail


def test_zero_major_tag_check_passes() -> None:
    """The tag check accepts a plain zero-major tag on its own."""
    assert guard.check_tag("v0.3.0").passed


def test_main_without_tag_falls_back_to_head(monkeypatch, tmp_path: Path) -> None:
    """Omitting --tag resolves it from the current HEAD, exactly like passing it."""
    changelog = _write_changelog(tmp_path, "0.1.0")
    monkeypatch.setattr(guard, "_current_tag", lambda: "v0.1.0")
    args = ["--changelog", str(changelog), "--gate-cmd", "true"]
    assert guard.main([*args, *_release_flags(tmp_path, "0.1.0")]) == 0


def test_main_without_tag_or_release_passes_with_nothing_to_check(monkeypatch) -> None:
    """HEAD not being exactly a tag is not a refusal -- most commits are not releases."""
    monkeypatch.setattr(guard, "_current_tag", lambda: None)
    assert guard.main([]) == 0


def _write_package(tmp_path: Path, source: str) -> Path:
    """Write a one-module package whose imports the tier check will walk.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = _write_package(Path(tmp), "import os\\n")
        ...     root.name
        'shipped'
    """
    root = tmp_path / "shipped"
    root.mkdir()
    (root / "module.py").write_text(source, encoding="utf-8")
    return root


def _write_manifest(tmp_path: Path, runtime: str, dev: str = "") -> Path:
    """Write a manifest declaring the two dependency tiers the check reads.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = _write_manifest(Path(tmp), '"torch"')
        ...     "[project]" in path.read_text()
        True
    """
    path = tmp_path / "pyproject.toml"
    path.write_text(
        f"[project]\ndependencies = [{runtime}]\n\n[dependency-groups]\ndev = [{dev}]\n",
        encoding="utf-8",
    )
    return path


def test_group_only_dependency_is_refused(tmp_path: Path) -> None:
    """A module whose distribution is declared only in a group refuses the tag.

    This is WP-159's own defect, stated as a test: `fuse-augmentations` sat in the
    `dev` group while five shipped modules imported it, so an install omitting that
    group raised `ImportError` from `lucid_yolo.data`. The development environment
    installs every group, which is exactly why the suite could not see it.
    """
    manifest = _write_manifest(tmp_path, '"torch"', '"helper-lib"')
    root = _write_package(tmp_path, "import helper\n")

    result = guard.check_dependency_tiers(manifest, root, {"helper": ["helper-lib"]})

    assert not result.passed
    assert "helper" in result.detail
    assert "'dev'" in result.detail, "the refusal must name the group, so it diagnoses rather than complains"


def test_runtime_declared_dependency_passes(tmp_path: Path) -> None:
    """The same import passes once its distribution is declared at runtime."""
    manifest = _write_manifest(tmp_path, '"helper-lib"', '"pytest"')
    root = _write_package(tmp_path, "import helper\n")

    assert guard.check_dependency_tiers(manifest, root, {"helper": ["helper-lib"]}).passed


def test_import_with_no_installed_distribution_is_refused(tmp_path: Path) -> None:
    """An import no distribution provides is reported, never silently accepted.

    A guard that reads "unknown" as "fine" is the failure mode this check closes.
    """
    manifest = _write_manifest(tmp_path, '"torch"')
    root = _write_package(tmp_path, "import mystery\n")

    result = guard.check_dependency_tiers(manifest, root, {})

    assert not result.passed
    assert "mystery" in result.detail


def test_stdlib_and_relative_imports_need_no_declaration(tmp_path: Path) -> None:
    """Neither the standard library nor an intra-package import is a dependency."""
    manifest = _write_manifest(tmp_path, "")
    root = _write_package(tmp_path, "import json\nfrom pathlib import Path\nfrom . import sibling\n")

    assert guard.check_dependency_tiers(manifest, root, {}).passed


def test_conditional_import_is_still_an_import(tmp_path: Path) -> None:
    """An import inside a function or a branch is one the installed package can run.

    Walking the module preamble alone would miss it, and a lazily imported
    dependency is no less required than an eagerly imported one -- it merely fails
    later, on the call rather than on the import.
    """
    manifest = _write_manifest(tmp_path, '"torch"', '"helper-lib"')
    root = _write_package(tmp_path, "def draw():\n    import helper\n    return helper\n")

    assert not guard.check_dependency_tiers(manifest, root, {"helper": ["helper-lib"]}).passed


def test_cli_refuses_a_tag_whose_package_imports_an_undeclared_module(tmp_path: Path) -> None:
    """The check is wired into the CLI, not merely importable from it."""
    changelog = _write_changelog(tmp_path, "0.1.0")
    manifest = _write_manifest(tmp_path, "")
    root = _write_package(tmp_path, "import torch\n")

    status = guard.main(
        [
            "--tag",
            "v0.1.0",
            "--changelog",
            str(changelog),
            "--gate-cmd",
            "true",
            "--pyproject",
            str(manifest),
            "--package-root",
            str(root),
            *_release_flags(tmp_path, "0.1.0"),
        ]
    )

    assert status == 1


def _gate_receipt_cmd(receipt: Path) -> str:
    """Build a gate command appending one line to ``receipt`` per invocation.

    ``check_gate`` launches the command through a shell from the repository root, so
    the counter is a file rather than a Python stub: what these tests must observe is
    how many times that process boundary was crossed, and a stub inside this process
    cannot see a crossing that a re-entrant hook would make from another one.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     "ran" in _gate_receipt_cmd(Path(tmp) / "receipt.txt")
        True
    """
    return f"printf 'ran\\n' >> '{receipt}'"


def test_metadata_only_leaves_the_gate_command_unrun(tmp_path: Path) -> None:
    """``--metadata-only`` never launches the gate command at all.

    This is the release-recursion defect stated as a test. The guard's default gate
    command is ``make gate``, which runs pre-commit, which runs this guard: on a
    commit whose HEAD is a tag, the one-mode form asked for a second gate from inside
    the first, so release validation looped instead of reaching the test and build
    stages. An empty receipt is the form of "no re-entry" that holds without reading
    the Makefile — the command was never launched, so nothing it would have launched
    matters.
    """
    receipt = tmp_path / "gate-invocations.txt"
    changelog = _write_changelog(tmp_path, "0.1.0")
    gate = _gate_receipt_cmd(receipt)
    args = ["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", gate, "--metadata-only"]

    assert guard.main([*args, *_release_flags(tmp_path, "0.1.0")]) == 0
    assert not receipt.exists()


def test_a_certifying_run_invokes_the_gate_exactly_once(tmp_path: Path) -> None:
    """Without the flag the gate runs, and runs a single time.

    The other half of the acceptance check, and the reason the receipt counts lines
    rather than merely existing: one line means the split kept the certification that
    a tag push depends on, and zero would mean it went too far and nothing re-runs
    the merge bar before a release ships.
    """
    receipt = tmp_path / "gate-invocations.txt"
    changelog = _write_changelog(tmp_path, "0.1.0")
    args = ["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", _gate_receipt_cmd(receipt)]

    assert guard.main([*args, *_release_flags(tmp_path, "0.1.0")]) == 0
    assert receipt.read_text(encoding="utf-8").splitlines() == ["ran"]


def test_metadata_only_still_refuses_a_changelog_that_does_not_name_the_tag(tmp_path: Path) -> None:
    """The reduced mode is not a reduced standard: malformed metadata is refused as before.

    An escape hatch would be a mode that accepts what the full run refuses. This one
    drops the gate and keeps all five metadata checks, so the fixture that fails
    ``test_missing_changelog_section_refuses_tag`` fails here identically.
    """
    changelog = _write_changelog(tmp_path, "0.2.0")  # section for a different version
    args = ["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", "true", "--metadata-only"]

    assert guard.main([*args, *_release_flags(tmp_path, "0.1.0")]) == 1


def test_metadata_only_verdict_refuses_to_call_the_tag_shippable(tmp_path, capsys) -> None:
    """A passing metadata run reports that the gate did not run, not that the tag may ship.

    The verdict line is the whole of what a reader of hook output sees, and the gate
    command here is ``false`` — a red gate the mode deliberately never asks. A mode
    that skipped it and still printed "all checks passed — tag may ship" would be the
    silent skip this split exists to avoid, whatever the flag enabling it was named.
    """
    changelog = _write_changelog(tmp_path, "0.1.0")
    args = ["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", "false", "--metadata-only"]

    assert guard.main([*args, *_release_flags(tmp_path, "0.1.0")]) == 0

    verdict = capsys.readouterr().out
    assert "tag may ship" not in verdict
    assert "--metadata-only" in verdict
    assert "did NOT run" in verdict


def _local_hook(hook_id: str) -> dict[str, str]:
    """Return the committed local pre-commit hook whose id is ``hook_id``.

    Examples:
        >>> _local_hook("release-guard")["id"]
        'release-guard'
    """
    config = yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"] if hook.get("id") == hook_id]
    assert len(hooks) == 1, f"expected exactly one {hook_id!r} hook, found {len(hooks)}"
    return hooks[0]


def _release_workflow() -> dict:
    """Parse ``.github/workflows/release.yml``.

    The trigger block is read back as the key ``True``, not ``"on"``: YAML 1.1 spells
    booleans ``on``/``off`` as well as ``true``/``false``, and PyYAML implements 1.1.
    The workflow file is correct as written; the loader is what needs the note.

    Examples:
        >>> _release_workflow()["jobs"]["release"]["runs-on"]
        'ubuntu-latest'
    """
    return yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))


def _release_steps() -> list[dict]:
    """Return the steps of release.yml's single job.

    Examples:
        >>> len(_release_steps()) > 1
        True
    """
    return _release_workflow()["jobs"]["release"]["steps"]


def test_the_pre_commit_hook_asks_the_guard_for_metadata_only(tmp_path: Path) -> None:
    """The committed hook entry carries ``--metadata-only``, which is what breaks the cycle.

    Asserted against the file rather than described in a comment because the defect
    was never in a function: every check the guard runs was covered and correct, and
    the loop existed anyway, in which caller ran which checks. That answer lives in
    ``.pre-commit-config.yaml`` and nowhere else, so a revert there is a revert of the
    fix and this is the only test that would notice.
    """
    assert "--metadata-only" in _local_hook("release-guard")["entry"]


def test_exactly_one_release_workflow_step_certifies_a_tag_push() -> None:
    """release.yml runs the full guard once, on a tag push, and the rehearsal never certifies.

    The counterpart to the hook assertion above: the mode split is only safe while
    something still runs the gate. Exactly one guard step omits ``--metadata-only``,
    and it is the one gated on the push event — a dispatch rehearsal builds and
    uploads, and decides nothing about whether a release may ship.
    """
    certifying = [step for step in _release_steps() if "--metadata-only" not in step.get("run", "")]
    certifying = [step for step in certifying if "scripts/release_guard.py" in step.get("run", "")]

    assert len(certifying) == 1
    assert certifying[0]["if"] == "github.event_name == 'push'"


def test_the_release_workflow_is_runnable_by_hand_before_any_tag_exists() -> None:
    """A ``workflow_dispatch`` trigger takes the tag to rehearse against.

    Six releases shipped with zero tags, so the guard, the README rewrite and the
    build were code no event had ever executed: the first real tag would have run all
    of it at once. The dispatch path is how that stops being true before it matters.
    """
    dispatch = _release_workflow()[True]["workflow_dispatch"]

    assert dispatch["inputs"]["tag"]["required"] is True


def test_no_release_step_interpolates_a_workflow_context_into_shell_source() -> None:
    """Every context value reaches a ``run:`` script through the environment instead.

    ``git check-ref-format`` permits ``;``, ``$``, backticks and parentheses in a tag
    name, and ``v0.1.0;<command>`` matches this workflow's ``v0.*`` trigger. Pasted
    into a shell string that is a command; passed as ``"${TAG}"`` it is one literal
    argument that fails the guard's version check. The assertion is over every step
    rather than the two known ones, so a third added later is caught on arrival.
    """
    interpolating = [step.get("name", "<unnamed>") for step in _release_steps() if "${{" in step.get("run", "")]

    assert interpolating == []
