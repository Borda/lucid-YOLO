# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: freezing goldens skips non-freezable ones (WP-154c).

WP-132 removed two generator-derived frozen goldens by hand, and WP-140 and WP-153 each silently reintroduced them
because ``make freeze-goldens`` was a blind copy with no notion of the distinction. These tests exercise the replacement
(``scripts/freeze_goldens.py``) against throwaway golden directories, so a golden marked ``"freezable": false`` is
provably excluded from every future release snapshot rather than trusted to a human noticing.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from script_loader import load_script

REPO_ROOT = Path(__file__).resolve().parents[2]

freeze_goldens = load_script("freeze_goldens")


def _write_golden(path: Path, freezable: bool | None) -> None:
    """Write a minimal golden file, omitting ``freezable`` when ``None``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     p = Path(tmp) / "g.json"
        ...     _write_golden(p, freezable=False)
        ...     json.loads(p.read_text())["freezable"]
        False

        ```
    """
    data: dict[str, object] = {"producer": "x:y", "values": {"m": 1.0}}
    if freezable is not None:
        data["freezable"] = freezable
    path.write_text(json.dumps(data))


def test_freezable_goldens_excludes_non_freezable(tmp_path: Path) -> None:
    """A golden marked ``"freezable": false`` is excluded from the eligible list."""
    _write_golden(tmp_path / "pure.json", freezable=None)
    _write_golden(tmp_path / "generator.json", freezable=False)

    eligible = freeze_goldens.freezable_goldens(tmp_path)

    assert [p.name for p in eligible] == ["pure.json"]


def test_freeze_copies_only_freezable_goldens(tmp_path: Path) -> None:
    """``freeze`` copies eligible goldens into ``frozen/<minor>/`` and reports the skip."""
    _write_golden(tmp_path / "pure.json", freezable=True)
    _write_golden(tmp_path / "generator.json", freezable=False)

    frozen, skipped = freeze_goldens.freeze(tmp_path, "0.9")

    dest = tmp_path / "frozen" / "0.9"
    assert [p.name for p in frozen] == ["pure.json"]
    assert [p.name for p in skipped] == ["generator.json"]
    assert (dest / "pure.json").exists()
    assert not (dest / "generator.json").exists()


def test_main_reports_usage_without_minor_argument() -> None:
    """The CLI refuses to run without a minor-version argument."""
    assert freeze_goldens.main([]) == 1


@patch.object(freeze_goldens, "_derives_from_generator", new=lambda spec: True)
def test_freeze_refuses_generator_derived_golden_omits_flag(tmp_path: Path) -> None:
    """A golden whose producer reaches the dataset generator cannot be frozen by omitting the flag (M-42).

    ``freezable`` defaulted to ``True``, so the WP-132 defect only stayed fixed while every future golden author
    remembered to write the field — the same trust that let WP-140 and WP-153 each reintroduce a generator-derived
    snapshot. The generator-derivation probe is stubbed to report the producer as generator-derived.
    """
    _write_golden(tmp_path / "generator.json", freezable=None)

    with pytest.raises(freeze_goldens.FreezeError, match=r"generator\.json"):
        freeze_goldens.freeze(tmp_path, "0.9")

    assert not (tmp_path / "frozen" / "0.9" / "generator.json").exists()


@patch.object(freeze_goldens, "_derives_from_generator", new=lambda spec: False)
def test_freeze_refuses_stale_non_freezable_flag(tmp_path: Path) -> None:
    """A golden declaring ``"freezable": false`` whose producer no longer reaches the generator is refused.

    The flag is hand-written in both directions. A producer that stopped rendering through the generator leaves a golden
    permanently excluded from every release snapshot for a reason that no longer holds, and nothing would say so. The
    generator-derivation probe is stubbed to report the producer as pure code.
    """
    _write_golden(tmp_path / "pure.json", freezable=False)

    with pytest.raises(freeze_goldens.FreezeError, match=r"pure\.json"):
        freeze_goldens.freeze(tmp_path, "0.9")


@patch.object(freeze_goldens, "_derives_from_generator", new=lambda spec: False)
def test_freeze_proceeds_when_flag_matches_producer(tmp_path: Path) -> None:
    """A declared flag agreeing with the derived one freezes exactly as before.

    The generator-derivation probe is stubbed to report pure code, matching the golden's ``freezable: true``.
    """
    _write_golden(tmp_path / "pure.json", freezable=True)

    frozen, skipped = freeze_goldens.freeze(tmp_path, "0.9")

    assert [p.name for p in frozen] == ["pure.json"]
    assert skipped == []


def test_freeze_falls_back_to_declared_flag_when_derivation_impossible(tmp_path: Path) -> None:
    """A golden with no resolvable producer keeps its declared flag rather than blocking the freeze.

    ``freeze_goldens`` reads raw JSON and does not validate the golden schema; ``check_goldens.py`` is what rejects a
    golden with no runnable producer, and it would fail such a file in the gate long before a release freeze.
    """
    (tmp_path / "no_producer.json").write_text(json.dumps({"values": {"m": 1.0}}))

    frozen, skipped = freeze_goldens.freeze(tmp_path, "0.9")

    assert [p.name for p in frozen] == ["no_producer.json"]
    assert skipped == []


@pytest.mark.parametrize(
    ("golden_name", "expected"),
    [
        pytest.param("fixture_checksums.json", True, id="generator-derived"),
        pytest.param("params_flops_det.json", False, id="pure-code"),
    ],
)
def test_derivation_separates_generator_derived_producers_from_pure_ones(golden_name: str, expected: bool) -> None:
    """The probe runs a real producer and reports whether it loaded the dataset generator.

    The one test that exercises the subprocess rather than a stub, against the two real producers that sit on opposite
    sides of the distinction: ``fixture_checksums`` renders its dataset through the generator, while
    ``params_flops_det`` counts parameters in this project's own code.
    """
    spec = json.loads((REPO_ROOT / "goldens" / golden_name).read_text())["producer"]

    assert freeze_goldens._derives_from_generator(spec) is expected
