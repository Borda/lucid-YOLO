# SPDX-License-Identifier: Apache-2.0
"""Functional-core tests: freezing goldens skips non-freezable ones (WP-154c).

WP-132 removed two generator-derived frozen goldens by hand, and WP-140 and
WP-153 each silently reintroduced them because ``make freeze-goldens`` was a
blind copy with no notion of the distinction. These tests exercise the
replacement (``scripts/freeze_goldens.py``) against throwaway golden
directories, so a golden marked ``"freezable": false`` is provably excluded
from every future release snapshot rather than trusted to a human noticing.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "freeze_goldens.py"


def _load_module() -> ModuleType:
    """Load ``scripts/freeze_goldens.py`` as an importable module.

    Examples:
        >>> callable(_load_module().freeze)
        True
    """
    spec = importlib.util.spec_from_file_location("freeze_goldens", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


freeze_goldens = _load_module()


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
