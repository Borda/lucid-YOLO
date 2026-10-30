# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the release guard refuses unshippable tags (WP-006).

Covers the DoD negative cases — a tag on a red gate, a tag missing its changelog section, and a 1.x tag (ADR-002) are
each refused — plus the happy path and a malformed tag string. The guard is loaded with the shared ``script_loader`` so
``scripts/`` need not be importable.

WP-168 added the version and frozen-golden checks, and with them the reason every ``main`` call below now passes
``--init`` and ``--frozen-root``: those two checks default to the live repository, so a test tagging ``v0.1.0`` would
otherwise be asserting against whatever version this tree happens to declare today and whichever minors happen to be
frozen. The fixtures state the release they describe instead.

The ``--metadata-only`` tests are the first in this file whose subject is the *wiring* rather than the functions, and
they exist because the guard's default gate command is ``make gate``, which runs pre-commit, which runs the guard, so on
a tagged commit the fused form asked for a gate from inside the gate it was running. Every function it calls was covered
and the loop was still there, because what was wrong was which caller ran which checks. Those two are asserted against
the committed ``.pre-commit-config.yaml`` and ``release.yml``, which is where the answer lives; the gate-invocation
tests count executions through a receipt file, so "runs the gate exactly once" is measured rather than read.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from release_guard_inputs import gate_receipt_cmd, release_flags, write_changelog, write_manifest, write_package
from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"

guard = load_script("release_guard")


@pytest.mark.parametrize(
    ("changelog_version", "tag", "gate_cmd", "release_version"),
    [
        # A valid tag with its changelog section, on a red gate.
        pytest.param("0.1.0", "v0.1.0", "false", "0.1.0", id="red-gate"),
        # A valid tag on a green gate whose changelog section is for a different version.
        pytest.param("0.2.0", "v0.1.0", "true", "0.1.0", id="missing-changelog-section"),
        # A 1.x tag with a green gate and a matching changelog section (ADR-002).
        pytest.param("1.0.0", "v1.0.0", "true", "1.0.0", id="one-x-tag-adr-002"),
        # A tag that is not a v<major>.<minor>.<patch> string.
        pytest.param("0.1", "v0.1", "true", "0.1", id="malformed-tag"),
    ],
)
def test_main_refuses_tag(
    tmp_path: Path, changelog_version: str, tag: str, gate_cmd: str, release_version: str
) -> None:
    """The guard exits 1 for a tag on a red gate, without its changelog section, in the 1.x line, or malformed.

    Each case is the DoD negative scenario named by its id; the other inputs are valid, so the one defect is the
    only reason for the refusal.
    """
    changelog = write_changelog(tmp_path, changelog_version)
    args = ["--tag", tag, "--changelog", str(changelog), "--gate-cmd", gate_cmd]
    assert guard.main([*args, *release_flags(tmp_path, release_version)]) == 1


def test_happy_path_passes(tmp_path: Path) -> None:
    """A v0.MINOR.PATCH tag with its changelog section and a green gate ships."""
    changelog = write_changelog(tmp_path, "0.1.0")
    args = ["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", "true"]
    assert guard.main([*args, *release_flags(tmp_path, "0.1.0")]) == 0


def test_version_mismatch_refuses_tag(tmp_path: Path) -> None:
    """A tag naming a version the package does not declare is refused.

    WP-168's own defect, stated as a test: tagging ``v0.8.0`` over a tree still declaring ``0.7.0`` passed the tag,
    changelog, dependency-tier and gate checks and shipped a wheel whose version contradicted the tag that built it.
    Neither the changelog substring test nor ``audit_version_single_source.py`` ever reads a tag, so nothing in the
    repository compared the two.
    """
    changelog = write_changelog(tmp_path, "0.8.0")
    args = ["--tag", "v0.8.0", "--changelog", str(changelog), "--gate-cmd", "true"]
    flags = release_flags(tmp_path, "0.7.0")
    (tmp_path / "frozen" / "0.8").mkdir()
    (tmp_path / "frozen" / "0.8" / "optim_toy.json").write_text("{}", encoding="utf-8")

    assert guard.main([*args, *flags]) == 1


def test_missing_frozen_goldens_refuses_tag(tmp_path: Path) -> None:
    """A tag whose minor has no frozen-golden snapshot is refused.

    A release's regression claim is that current code still satisfies every value the release pinned. A minor tagged
    with no ``goldens/frozen/<minor>/`` makes that claim about an empty set, and every later run of the harness agrees
    with it.
    """
    changelog = write_changelog(tmp_path, "0.8.0")
    args = ["--tag", "v0.8.0", "--changelog", str(changelog), "--gate-cmd", "true"]
    flags = release_flags(tmp_path, "0.8.0")
    (tmp_path / "frozen" / "0.8" / "optim_toy.json").unlink()

    assert guard.main([*args, *flags]) == 1


def test_version_check_reads_source_without_import(tmp_path: Path) -> None:
    """``__version__`` is parsed statically, so an unimportable package still answers.

    A guard deciding whether a distribution is shippable must not require that distribution to import first — the
    import would run the package it is judging.
    """
    init = tmp_path / "__init__.py"
    init.write_text('raise SystemExit("importing me is fatal")\n__version__ = "0.7.0"\n', encoding="utf-8")

    assert guard.check_version("v0.7.0", init).passed


def test_version_check_reports_unreadable_version(tmp_path: Path) -> None:
    """A module assigning no literal ``__version__`` is refused rather than assumed to match."""
    init = tmp_path / "__init__.py"
    init.write_text("__version__ = compute_it()\n", encoding="utf-8")

    result = guard.check_version("v0.7.0", init)

    assert not result.passed
    assert "no literal __version__" in result.detail


def test_frozen_goldens_check_names_freeze_command(tmp_path: Path) -> None:
    """The refusal says which command writes the missing snapshot, so it diagnoses."""
    result = guard.check_frozen_goldens("v0.9.0", tmp_path)

    assert not result.passed
    assert "make freeze-goldens MINOR=0.9" in result.detail


def test_frozen_goldens_check_accepts_patch_release_of_frozen_minor(tmp_path: Path) -> None:
    """A patch tag reuses its minor's snapshot: 0.7.1 ships against goldens/frozen/0.7."""
    (tmp_path / "0.7").mkdir()
    (tmp_path / "0.7" / "optim_toy.json").write_text("{}", encoding="utf-8")

    assert guard.check_frozen_goldens("v0.7.1", tmp_path).passed


def test_adr_002_refusal_names_policy() -> None:
    """The 1.x refusal detail cites ADR-002's perpetual 0.x train, hard-coded."""
    result = guard.check_tag("v1.4.2")
    assert not result.passed
    assert "ADR-002" in result.detail


def test_zero_major_tag_check_passes() -> None:
    """The tag check accepts a plain zero-major tag on its own."""
    assert guard.check_tag("v0.3.0").passed


@patch.object(guard, "_current_tag", new=lambda: "v0.1.0")
def test_main_without_tag_falls_back_to_head(tmp_path: Path) -> None:
    """Omitting --tag resolves it from the current HEAD, exactly like passing it."""
    changelog = write_changelog(tmp_path, "0.1.0")
    args = ["--changelog", str(changelog), "--gate-cmd", "true"]
    assert guard.main([*args, *release_flags(tmp_path, "0.1.0")]) == 0


@patch.object(guard, "_current_tag", new=lambda: None)
def test_main_without_tag_or_release_passes_with_nothing_to_check() -> None:
    """HEAD not being exactly a tag is not a refusal -- most commits are not releases."""
    assert guard.main([]) == 0


def test_group_only_dependency_refused(tmp_path: Path) -> None:
    """A module whose distribution is declared only in a group refuses the tag.

    This is WP-159's own defect, stated as a test: `fuse-augmentations` sat in the `dev` group while five shipped
    modules imported it, so an install omitting that group raised `ImportError` from `lucid_yolo.data`. The
    development environment installs every group, which is exactly why the suite could not see it.
    """
    manifest = write_manifest(tmp_path, '"torch"', '"helper-lib"')
    root = write_package(tmp_path, "import helper\n")

    result = guard.check_dependency_tiers(manifest, root, {"helper": ["helper-lib"]})

    assert not result.passed
    assert "helper" in result.detail
    assert "'dev'" in result.detail, "the refusal must name the group, so it diagnoses rather than complains"


def test_runtime_declared_dependency_passes(tmp_path: Path) -> None:
    """The same import passes once its distribution is declared at runtime."""
    manifest = write_manifest(tmp_path, '"helper-lib"', '"pytest"')
    root = write_package(tmp_path, "import helper\n")

    assert guard.check_dependency_tiers(manifest, root, {"helper": ["helper-lib"]}).passed


def test_import_with_no_installed_distribution_refused(tmp_path: Path) -> None:
    """An import no distribution provides is reported, never silently accepted.

    A guard that reads "unknown" as "fine" is the failure mode this check closes.
    """
    manifest = write_manifest(tmp_path, '"torch"')
    root = write_package(tmp_path, "import mystery\n")

    result = guard.check_dependency_tiers(manifest, root, {})

    assert not result.passed
    assert "mystery" in result.detail


def test_stdlib_and_relative_imports_need_no_declaration(tmp_path: Path) -> None:
    """Neither the standard library nor an intra-package import is a dependency."""
    manifest = write_manifest(tmp_path, "")
    root = write_package(tmp_path, "import json\nfrom pathlib import Path\nfrom . import sibling\n")

    assert guard.check_dependency_tiers(manifest, root, {}).passed


def test_conditional_import_counts_as_import(tmp_path: Path) -> None:
    """An import inside a function or a branch is one the installed package can run.

    Walking the module preamble alone would miss it, and a lazily imported dependency is no less required than an
    eagerly imported one -- it merely fails later, on the call rather than on the import.
    """
    manifest = write_manifest(tmp_path, '"torch"', '"helper-lib"')
    root = write_package(tmp_path, "def draw():\n    import helper\n    return helper\n")

    assert not guard.check_dependency_tiers(manifest, root, {"helper": ["helper-lib"]}).passed


def test_cli_refuses_tag_whose_package_imports_undeclared_module(tmp_path: Path) -> None:
    """The check is wired into the CLI, not merely importable from it."""
    changelog = write_changelog(tmp_path, "0.1.0")
    manifest = write_manifest(tmp_path, "")
    root = write_package(tmp_path, "import torch\n")

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
            *release_flags(tmp_path, "0.1.0"),
        ]
    )

    assert status == 1


def test_metadata_only_leaves_gate_command_unrun(tmp_path: Path) -> None:
    """``--metadata-only`` never launches the gate command at all.

    This is the release-recursion defect stated as a test. The guard's default gate command is ``make gate``, which
    runs pre-commit, which runs this guard: on a commit whose HEAD is a tag, the one-mode form asked for a second gate
    from inside the first, so release validation looped instead of reaching the test and build stages. An empty receipt
    is the form of "no re-entry" that holds without reading the Makefile — the command was never launched, so nothing
    it would have launched matters.
    """
    receipt = tmp_path / "gate-invocations.txt"
    changelog = write_changelog(tmp_path, "0.1.0")
    gate = gate_receipt_cmd(receipt)
    args = ["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", gate, "--metadata-only"]

    assert guard.main([*args, *release_flags(tmp_path, "0.1.0")]) == 0
    assert not receipt.exists()


def test_certifying_run_invokes_gate_exactly_once(tmp_path: Path) -> None:
    """Without the flag the gate runs, and runs a single time.

    The other half of the acceptance check, and the reason the receipt counts lines rather than merely existing: one
    line means the split kept the certification that a tag push depends on, and zero would mean it went too far and
    nothing re-runs the merge bar before a release ships.
    """
    receipt = tmp_path / "gate-invocations.txt"
    changelog = write_changelog(tmp_path, "0.1.0")
    args = ["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", gate_receipt_cmd(receipt)]

    assert guard.main([*args, *release_flags(tmp_path, "0.1.0")]) == 0
    assert receipt.read_text(encoding="utf-8").splitlines() == ["ran"]


def test_metadata_only_refuses_changelog_not_name_tag(tmp_path: Path) -> None:
    """The reduced mode is not a reduced standard: malformed metadata is refused as before.

    An escape hatch would be a mode that accepts what the full run refuses. This one drops the gate and keeps all five
    metadata checks, so the fixture that fails ``test_main_refuses_tag[missing-changelog-section]`` fails here
    identically.
    """
    changelog = write_changelog(tmp_path, "0.2.0")  # section for a different version
    args = ["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", "true", "--metadata-only"]

    assert guard.main([*args, *release_flags(tmp_path, "0.1.0")]) == 1


def test_metadata_only_verdict_refuses_to_call_tag_shippable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A passing metadata run reports that the gate did not run, not that the tag may ship.

    The verdict line is the whole of what a reader of hook output sees, and the gate command here is ``false`` — a red
    gate the mode deliberately never asks. A mode that skipped it and still printed "all checks passed — tag may ship"
    would be the silent skip this split exists to avoid, whatever the flag enabling it was named.
    """
    changelog = write_changelog(tmp_path, "0.1.0")
    args = ["--tag", "v0.1.0", "--changelog", str(changelog), "--gate-cmd", "false", "--metadata-only"]

    assert guard.main([*args, *release_flags(tmp_path, "0.1.0")]) == 0

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

    The trigger block is read back as the key ``True``, not ``"on"``: YAML 1.1 spells booleans ``on``/``off`` as well
    as ``true``/``false``, and PyYAML implements 1.1. The workflow file is correct as written; the loader is what needs
    the note.

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


def test_pre_commit_hook_asks_guard_for_metadata_only() -> None:
    """The committed hook entry carries ``--metadata-only``, which is what breaks the cycle.

    Asserted against the file rather than described in a comment because the defect was never in a function: every
    check the guard runs was covered and correct, and the loop existed anyway, in which caller ran which checks. That
    answer lives in ``.pre-commit-config.yaml`` and nowhere else, so a revert there is a revert of the fix and this is
    the only test that would notice.
    """
    assert "--metadata-only" in _local_hook("release-guard")["entry"]


def test_exactly_one_release_workflow_step_certifies_tag_push() -> None:
    """release.yml runs the full guard once, on a tag push, and the rehearsal never certifies.

    The counterpart to the hook assertion above: the mode split is only safe while something still runs the gate.
    Exactly one guard step omits ``--metadata-only``, and it is the one gated on the push event — a dispatch rehearsal
    builds and uploads, and decides nothing about whether a release may ship.
    """
    certifying = [step for step in _release_steps() if "--metadata-only" not in step.get("run", "")]
    certifying = [step for step in certifying if "scripts/release_guard.py" in step.get("run", "")]

    assert len(certifying) == 1
    assert certifying[0]["if"] == "github.event_name == 'push'"


def test_release_workflow_runnable_by_hand_before_any_tag_exists() -> None:
    """A ``workflow_dispatch`` trigger takes the tag to rehearse against.

    Six releases shipped with zero tags, so the guard, the README rewrite and the build were code no event had ever
    executed: the first real tag would have run all of it at once. The dispatch path is how that stops being true
    before it matters.
    """
    dispatch = _release_workflow()[True]["workflow_dispatch"]

    assert dispatch["inputs"]["tag"]["required"] is True


def test_no_release_step_interpolates_workflow_context_into_shell_source() -> None:
    """Every context value reaches a ``run:`` script through the environment instead.

    ``git check-ref-format`` permits ``;``, ``$``, backticks and parentheses in a tag name, and ``v0.1.0;<command>``
    matches this workflow's ``v0.*`` trigger. Pasted into a shell string that is a command; passed as ``"${TAG}"`` it
    is one literal argument that fails the guard's version check. The assertion is over every step rather than the two
    known ones, so a third added later is caught on arrival.
    """
    interpolating = [step.get("name", "<unnamed>") for step in _release_steps() if "${{" in step.get("run", "")]

    assert interpolating == []


@pytest.mark.parametrize(
    ("source", "passes"),
    [
        ("try:\n    import helper\nexcept ModuleNotFoundError:\n    pass\n", True),
        ("try:\n    import helper\nexcept (OSError, ImportError):\n    pass\n", True),
        ("try:\n    import helper\nexcept ImportError:\n    raise\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    raise RuntimeError('missing helper')\n", False),
        ("try:\n    import helper\nexcept (OSError, ImportError):\n    raise\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    if unavailable:\n        raise RuntimeError()\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    raise\nexcept ModuleNotFoundError:\n    pass\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    helper = None\n", True),
        ("try:\n    import helper\nexcept ValueError:\n    raise\nexcept ImportError:\n    pass\n", True),
        (
            "try:\n    try:\n        import helper\n    except ImportError:\n"
            "        raise RuntimeError()\nexcept ImportError:\n    pass\n",
            False,
        ),
        ("import helper\n", False),
        ("def use():\n    import helper\n", False),
        ("try:\n    def use():\n        import helper\nexcept ImportError:\n    pass\n", False),
        ("try:\n    import helper\nexcept ValueError:\n    pass\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    pass\nimport helper\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    sys.exit(1)\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    os._exit(1)\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    exit(1)\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    quit()\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    os.abort()\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    assert False\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    print(exit)\n", True),
        ("try:\n    import helper\nexcept Exception:\n    raise\nexcept ImportError:\n    pass\n", False),
        ("try:\n    import helper\nexcept BaseException:\n    raise\nexcept ImportError:\n    pass\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    pass\nexcept Exception:\n    raise\n", False),
        ("try:\n    import helper\nexcept ImportError:\n    pass\nexcept:\n    raise\n", False),
        ("try:\n    import helper\nexcept:\n    raise\n", False),
        ("try:\n    import helper\nexcept Exception:\n    sys.exit(1)\n", False),
        ("try:\n    import helper\nexcept Exception:\n    pass\n", True),
        ("try:\n    import helper\nexcept BaseException:\n    pass\n", True),
        ("try:\n    import helper\nexcept:\n    pass\n", True),
        ("try:\n    import helper\nexcept ValueError:\n    raise\nexcept:\n    pass\n", True),
    ],
)
def test_consumer_extra_requires_guarded_import(tmp_path: Path, source: str, passes: bool) -> None:
    """An unavailable extra stays optional only when every import handles its absence."""
    manifest = write_manifest(tmp_path, "")
    with manifest.open("a") as stream:
        stream.write('[project.optional-dependencies]\ncuda = ["helper"]\n')
    root = write_package(tmp_path, source)
    assert guard.check_dependency_tiers(manifest, root, {}).passed is passes


def test_guarded_group_only_dependency_refused(tmp_path: Path) -> None:
    """A development group never substitutes for an installable consumer extra."""
    manifest = write_manifest(tmp_path, "", '"helper"')
    root = write_package(tmp_path, "try:\n    import helper\nexcept ImportError:\n    pass\n")
    assert not guard.check_dependency_tiers(manifest, root, {"helper": ["helper"]}).passed


def test_guarded_extra_with_distribution_alias_passes(tmp_path: Path) -> None:
    """Installed metadata can resolve a guarded module to its differently named extra."""
    manifest = write_manifest(tmp_path, "")
    with manifest.open("a") as stream:
        stream.write('[project.optional-dependencies]\ncuda = ["helper-lib"]\n')
    root = write_package(tmp_path, "try:\n    import helper\nexcept ImportError:\n    pass\n")
    assert guard.check_dependency_tiers(manifest, root, {"helper": ["helper-lib"]}).passed
