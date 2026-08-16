# SPDX-License-Identifier: Apache-2.0
"""Tests for the packaging-time README link rewrite (WP-114).

What is actually at stake is a file that renders correctly in two places at once and
cannot: relative targets are right in the git tree and dead on PyPI. So the tests below
check both directions — that a release build makes every relative target absolute, and
that an ordinary build cannot trigger the rewrite by accident.
"""

from pathlib import Path

import pytest
from scripts.absolutize_readme import absolutize, count_relative, is_relative, main, repository_slug, revert, rewrite

REPO_ROOT = Path(__file__).resolve().parents[2]

SAMPLE = """# Title

See [the register](docs/ASSUMPTIONS.md) and [the roadmap](docs/ROADMAP.md#phase-9).

![Detection curves](docs/figures/det_smoke_training.svg)

<img src="docs/figures/seg.svg" width="600">

Jump to [Quickstart](#quickstart), read [the paper](https://arxiv.org/abs/2606.03748),
or mail [someone](mailto:nobody@example.com).
"""


def test_every_relative_target_becomes_absolute() -> None:
    """No target survives the rewrite still needing a checkout to resolve."""
    rewritten = absolutize(SAMPLE, "Borda/lucid-YOLO", "v0.4.0")

    assert count_relative(SAMPLE) == 4
    assert count_relative(rewritten) == 0


def test_images_and_documents_resolve_through_different_hosts() -> None:
    """Figures go to ``raw``, documents to ``blob``.

    Both hosts serve the same file and only one of them serves it as bytes: a figure
    linked through ``blob`` renders as a GitHub page inside an ``<img>``, which is a
    broken image, and a document linked through ``raw`` hands the reader unrendered
    markdown. The distinction is the whole reason the rewrite classifies targets at all.
    """
    rewritten = absolutize(SAMPLE, "Borda/lucid-YOLO", "v0.4.0")

    assert "https://raw.githubusercontent.com/Borda/lucid-YOLO/v0.4.0/docs/figures/det_smoke_training.svg" in rewritten
    assert "https://raw.githubusercontent.com/Borda/lucid-YOLO/v0.4.0/docs/figures/seg.svg" in rewritten
    assert "https://github.com/Borda/lucid-YOLO/blob/v0.4.0/docs/ASSUMPTIONS.md" in rewritten


def test_a_fragment_survives_the_rewrite() -> None:
    """``docs/ROADMAP.md#phase-9`` keeps its fragment, which is the half a reader wanted."""
    rewritten = absolutize(SAMPLE, "Borda/lucid-YOLO", "v0.4.0")

    assert "https://github.com/Borda/lucid-YOLO/blob/v0.4.0/docs/ROADMAP.md#phase-9" in rewritten


def test_absolute_anchor_and_mail_targets_are_left_alone() -> None:
    """Anything already resolvable from anywhere is not touched.

    An in-page anchor is the interesting case: it looks relative and is not. Prefixing it
    would turn every internal jump on the PyPI page into a round trip to GitHub.
    """
    rewritten = absolutize(SAMPLE, "Borda/lucid-YOLO", "v0.4.0")

    assert "[Quickstart](#quickstart)" in rewritten
    assert "(https://arxiv.org/abs/2606.03748)" in rewritten
    assert "(mailto:nobody@example.com)" in rewritten
    assert not is_relative("#quickstart")


def test_the_rewrite_is_idempotent() -> None:
    """A second pass changes nothing, so a rerun cannot double-prefix a URL."""
    once = absolutize(SAMPLE, "Borda/lucid-YOLO", "v0.4.0")

    assert absolutize(once, "Borda/lucid-YOLO", "v0.4.0") == once


@pytest.mark.parametrize("ref", ["main", "HEAD", "0.4.0", "abc1234", "v0.4", ""])
def test_a_ref_that_is_not_a_tag_or_sha_is_refused(
    ref: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only a tag or a full sha may be pinned; a branch names a tree that keeps moving.

    A released wheel's README is a snapshot. Pinned to ``main``, its links describe
    whatever that branch holds when a reader clicks them, which is how a page for 0.4.0
    ends up documenting code that shipped years later.
    """
    readme = tmp_path / "README.md"
    readme.write_text(SAMPLE, encoding="utf-8")

    status = main(["--ref", ref, "--readme", str(readme), "--backup", str(tmp_path / "b.orig")])

    assert status == 1
    assert "must be a v0.MINOR.PATCH tag or a full commit sha" in capsys.readouterr().out
    assert readme.read_text(encoding="utf-8") == SAMPLE, "a refused run must not touch the file"


def test_no_ref_means_no_rewrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An ordinary build leaves the README exactly as committed.

    This is the property that lets the file stay relative in the tree: the rewrite is
    opt-in per invocation, so nothing about ``make build`` or a plain ``pip install .``
    can reach it.
    """
    monkeypatch.delenv("LUCID_YOLO_RELEASE_REF", raising=False)
    readme = tmp_path / "README.md"
    readme.write_text(SAMPLE, encoding="utf-8")

    assert main(["--readme", str(readme), "--backup", str(tmp_path / "b.orig")]) == 1
    assert readme.read_text(encoding="utf-8") == SAMPLE


def test_the_environment_variable_supplies_the_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``LUCID_YOLO_RELEASE_REF`` is the flagless way a release runner opts in."""
    monkeypatch.setenv("LUCID_YOLO_RELEASE_REF", "v0.4.0")
    readme = tmp_path / "README.md"
    readme.write_text(SAMPLE, encoding="utf-8")

    status = main(["--repo", "Borda/lucid-YOLO", "--readme", str(readme), "--backup", str(tmp_path / "b.orig")])

    assert status == 0
    assert count_relative(readme.read_text(encoding="utf-8")) == 0


def test_revert_restores_the_committed_bytes(tmp_path: Path) -> None:
    """The tree a build ran in is left byte-identical to the one it started from."""
    readme, backup = tmp_path / "README.md", tmp_path / "b.orig"
    readme.write_text(SAMPLE, encoding="utf-8")

    assert rewrite(readme, backup, "Borda/lucid-YOLO", "v0.4.0") == 4
    revert(readme, backup)

    assert readme.read_text(encoding="utf-8") == SAMPLE
    assert not backup.exists()


def test_a_second_rewrite_refuses_rather_than_overwriting_the_backup(tmp_path: Path) -> None:
    """The backup is the only copy of the committed file; overwriting it destroys it.

    A backup already on disk means an earlier build never reverted. Rewriting over it
    would park the *rewritten* file as the thing to restore, and the revert afterwards
    would look like it worked.
    """
    readme, backup = tmp_path / "README.md", tmp_path / "b.orig"
    readme.write_text(SAMPLE, encoding="utf-8")
    rewrite(readme, backup, "Borda/lucid-YOLO", "v0.4.0")

    with pytest.raises(FileExistsError, match="never reverted"):
        rewrite(readme, backup, "Borda/lucid-YOLO", "v0.4.0")


def test_revert_without_a_backup_refuses(tmp_path: Path) -> None:
    """Nothing to restore is an error, not a silent success on a rewritten file."""
    readme = tmp_path / "README.md"
    readme.write_text(SAMPLE, encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="nothing to restore"):
        revert(readme, tmp_path / "missing.orig")


def test_the_slug_comes_from_the_declared_homepage() -> None:
    """The repository's own metadata is what the URLs are built from.

    Read live rather than pinned to a literal: a renamed repository redirects on
    ``github.com`` but **not** on ``raw.githubusercontent.com``, so a stale slug here
    would ship a page whose every figure is a 404 and whose every document link works.
    """
    slug = repository_slug(REPO_ROOT / "pyproject.toml")

    assert slug.count("/") == 1, f"not an owner/name pair: {slug!r}"
    assert not slug.endswith(".git")


def test_a_homepage_that_is_not_a_github_url_is_refused(tmp_path: Path) -> None:
    """No silent fallback: an undeclared homepage has no correct URL to guess."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project.urls]\nHomepage = "https://example.com/thing"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="not a GitHub repository URL"):
        repository_slug(pyproject)


def test_the_release_workflow_rewrites_before_it_builds() -> None:
    """``release.yml`` runs the rewrite ahead of ``uv build``, and reverts nothing after.

    The workflow's checkout is disposable, so the rewrite is never undone there — but the
    ordering is load-bearing and invisible in a diff of either file alone: a rewrite step
    that drifts below the build step still passes CI and ships the relative README.
    """
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "scripts/absolutize_readme.py" in workflow, "release build does not rewrite the README"
    assert workflow.index("scripts/absolutize_readme.py") < workflow.index("uv build")
