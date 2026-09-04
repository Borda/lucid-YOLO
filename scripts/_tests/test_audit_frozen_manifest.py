# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: the frozen-golden digest manifest (WP-168).

Covers ``scripts/lint/audit_frozen_manifest.py`` and the manifest half of
``scripts/freeze_goldens.py`` on synthetic trees under ``tmp_path``, never against the
live ``goldens/frozen/`` -- the pre-commit hook is what runs against that, and a test
asserting over it would fail for anyone mid-freeze rather than reporting a defect in
this code.

Four cases decide whether the manifest is a control or decoration: the clean tree, a
tampered digest, a frozen file no row covers, and a row whose file is gone. The tamper
case is the one the whole row exists for -- editing a frozen golden's values and its
tolerances together is green under ``check_goldens.py`` by construction, because that
harness compares each file with itself.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str) -> ModuleType:
    """Load a ``scripts/`` module by path, since ``scripts/`` is not a package.

    Examples:
        >>> _load("freeze_goldens", "scripts/freeze_goldens.py").MANIFEST_NAME
        'MANIFEST.sha256'
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


writer = _load("freeze_goldens", "scripts/freeze_goldens.py")
audit = _load("audit_frozen_manifest", "scripts/lint/audit_frozen_manifest.py")


@pytest.fixture
def sealed(tmp_path: Path) -> Path:
    """A two-minor frozen tree whose manifest matches it exactly.

    The starting point every case below perturbs in one way, so that the assertion is
    about the perturbation rather than about how the tree was built.
    """
    for minor, name in (("0.6", "optim_toy.json"), ("0.7", "optim_toy.json"), ("0.7", "params_flops_det.json")):
        (tmp_path / minor).mkdir(exist_ok=True)
        (tmp_path / minor / name).write_text(f'{{"values": {{"loss": 1.0}}, "minor": "{minor}"}}\n', encoding="utf-8")
    writer.reseal(tmp_path)
    return tmp_path


class TestFindViolations:
    """Tests for ``audit_frozen_manifest.find_violations``."""

    def test_is_clean_for_a_sealed_tree(self, sealed: Path) -> None:
        """A manifest describing the tree exactly reports no violation."""
        assert audit.find_violations(sealed) == []

    def test_flags_a_tampered_golden(self, sealed: Path) -> None:
        """Editing a frozen golden's values and tolerances together is reported.

        The case the row exists for. ``check_goldens.py`` recomputes the producer and
        compares it against that same file's stored values within its stored
        tolerances, so a coordinated edit of both is green there -- the file is being
        compared with itself. Only the digest notices the file changed at all.
        """
        target = sealed / "0.7" / "optim_toy.json"
        target.write_text('{"values": {"loss": 2.0}, "tolerances": {"loss": 1.0}}\n', encoding="utf-8")

        violations = audit.find_violations(sealed)

        assert len(violations) == 1
        assert violations[0].startswith("0.7/optim_toy.json: digest ")

    def test_flags_a_frozen_file_no_row_covers(self, sealed: Path) -> None:
        """A frozen file added without sealing it is reported rather than admitted.

        Without this the manifest would be satisfiable by deleting rows: a new file
        covered by nothing would be as good as a file whose digest matched.
        """
        (sealed / "0.7" / "smuggled.json").write_text('{"values": {}}\n', encoding="utf-8")

        assert audit.find_violations(sealed) == ["0.7/smuggled.json: frozen file covered by no manifest row"]

    def test_flags_a_row_whose_file_is_gone(self, sealed: Path) -> None:
        """Deleting a frozen golden is reported: the release's snapshot is now incomplete."""
        (sealed / "0.6" / "optim_toy.json").unlink()

        assert audit.find_violations(sealed) == ["0.6/optim_toy.json: manifest row whose file is gone"]

    def test_flags_a_missing_manifest(self, sealed: Path) -> None:
        """A frozen tree with no manifest at all is reported, not treated as trivially sealed."""
        (sealed / writer.MANIFEST_NAME).unlink()

        violations = audit.find_violations(sealed)

        assert len(violations) == 1
        assert violations[0].endswith("nothing pins the frozen goldens")

    def test_the_manifest_is_excluded_from_its_own_digest_set(self, sealed: Path) -> None:
        """The manifest does not appear as a row in itself, which would be unsatisfiable."""
        assert writer.MANIFEST_NAME not in writer.read_manifest(sealed)


class TestMain:
    """Tests for ``audit_frozen_manifest.main``."""

    def test_exits_zero_on_a_sealed_tree(self, sealed: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A clean run exits 0 and reports how many files it covered."""
        assert audit.main(["--frozen-root", str(sealed)]) == 0
        assert "3 file(s) match their recorded digest" in capsys.readouterr().out

    def test_exits_one_and_names_the_sanctioned_command(self, sealed: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A tampered tree exits 1 and the refusal says which command records a sanctioned move.

        A gate that only says no teaches the next reader to work around it; naming
        ``--reseal`` is what makes the approved path the obvious one.
        """
        (sealed / "0.7" / "optim_toy.json").write_text('{"values": {"loss": 9.0}}\n', encoding="utf-8")

        status = audit.main(["--frozen-root", str(sealed)])

        assert status == 1
        assert "freeze_goldens.py --reseal" in capsys.readouterr().out


class TestSeal:
    """Tests for ``freeze_goldens.seal`` and ``freeze_goldens.reseal``."""

    def test_seal_preserves_rows_it_did_not_write(self, sealed: Path) -> None:
        """Freezing a new minor leaves every earlier row exactly as it stood.

        Re-hashing the whole tree on every release freeze would let a snapshot tampered
        with under an earlier minor be re-blessed by the next release, which is the
        failure the manifest exists to catch -- so the release path seals only what it
        copied, and ``--reseal`` is the separate, named act.
        """
        before = writer.read_manifest(sealed)
        (sealed / "0.8").mkdir()
        added = sealed / "0.8" / "optim_toy.json"
        added.write_text('{"values": {"loss": 3.0}}\n', encoding="utf-8")

        writer.seal(sealed, [added])

        after = writer.read_manifest(sealed)
        assert after["0.8/optim_toy.json"] == writer.digest(added)
        assert {k: v for k, v in after.items() if k != "0.8/optim_toy.json"} == before

    def test_seal_does_not_launder_a_tampered_row(self, sealed: Path) -> None:
        """Freezing a new minor over a tampered tree still leaves the tamper reported."""
        (sealed / "0.6" / "optim_toy.json").write_text('{"values": {"loss": 9.0}}\n', encoding="utf-8")
        (sealed / "0.8").mkdir()
        added = sealed / "0.8" / "optim_toy.json"
        added.write_text('{"values": {"loss": 3.0}}\n', encoding="utf-8")

        writer.seal(sealed, [added])

        assert audit.find_violations(sealed) == [
            "0.6/optim_toy.json: digest "
            f"{writer.digest(sealed / '0.6' / 'optim_toy.json')} does not match the manifest's "
            f"{writer.read_manifest(sealed)['0.6/optim_toy.json']}"
        ]

    def test_reseal_accepts_a_sanctioned_move(self, sealed: Path) -> None:
        """``--reseal`` is what records a principal-approved golden move, and it clears the finding."""
        (sealed / "0.7" / "optim_toy.json").write_text('{"values": {"loss": 4.0}}\n', encoding="utf-8")
        assert audit.find_violations(sealed) != []

        writer.reseal(sealed)

        assert audit.find_violations(sealed) == []

    def test_reseal_drops_rows_for_deleted_files(self, sealed: Path) -> None:
        """A re-seal describes the tree as it stands, so a deleted file leaves no orphan row."""
        (sealed / "0.6" / "optim_toy.json").unlink()

        writer.reseal(sealed)

        assert "0.6/optim_toy.json" not in writer.read_manifest(sealed)


class TestFreeze:
    """Tests for ``freeze_goldens.freeze``'s manifest half."""

    def test_freezing_a_minor_seals_what_it_copied(self, tmp_path: Path) -> None:
        """``make freeze-goldens MINOR=0.N`` stays one command: the copy and the seal are one act."""
        (tmp_path / "optim_toy.json").write_text('{"values": {"loss": 1.0}}\n', encoding="utf-8")
        (tmp_path / "generated.json").write_text('{"values": {"x": 1.0}, "freezable": false}\n', encoding="utf-8")

        frozen, skipped = writer.freeze(tmp_path, "0.8")

        assert [p.name for p in frozen] == ["optim_toy.json"]
        assert [p.name for p in skipped] == ["generated.json"]
        assert audit.find_violations(writer.frozen_root(tmp_path)) == []
