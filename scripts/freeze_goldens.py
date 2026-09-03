# SPDX-License-Identifier: Apache-2.0
"""Freeze the live goldens into a release snapshot, skipping non-freezable ones (WP-154c).

``make freeze-goldens`` used to be a blind ``cp goldens/*.json goldens/frozen/$(MINOR)/``,
with no notion that two goldens (``fixture_checksums.json``, ``data_checksums.json``) are
generator-derived: their producers render synthetic images through the external
``fuse-augmentations`` package, so once that package's generator moves, the frozen copy can
never again satisfy "current code still satisfies every value a past release pinned" — the
snapshot was never a real regression guard (WP-132). WP-132 removed two such snapshots by
hand; WP-140 and WP-153 each silently reintroduced them, because the blind copy had no way to
know. This script reads each live golden's ``"freezable"`` field (``scripts/check_goldens.py``
schema; defaults to ``True`` when absent) and copies only the freezable ones.

Examples:
    Freeze the current goldens into ``goldens/frozen/0.7``::

        python scripts/freeze_goldens.py 0.7
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

#: Repository root (``scripts/`` is one level below it).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directory holding the live top-level goldens.
GOLDENS_DIR = REPO_ROOT / "goldens"


def freezable_goldens(goldens_dir: Path) -> list[Path]:
    """List the live top-level goldens eligible to be frozen.

    A golden is eligible unless its schema sets ``"freezable": false``
    (``scripts/check_goldens.py``'s ``_parse_golden``; absent defaults to eligible).

    Args:
        goldens_dir: Directory holding the live ``*.json`` goldens.

    Returns:
        Sorted list of eligible golden paths.

    Examples:
        ```pycon
        >>> paths = freezable_goldens(GOLDENS_DIR)
        >>> "fixture_checksums.json" in {p.name for p in paths}
        False

        ```
    """
    eligible = []
    for path in sorted(goldens_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("freezable", True):
            eligible.append(path)
    return eligible


def freeze(goldens_dir: Path, minor: str) -> tuple[list[Path], list[Path]]:
    """Copy every freezable live golden into ``goldens_dir/frozen/<minor>/``.

    Args:
        goldens_dir: Directory holding the live ``*.json`` goldens.
        minor: Release minor version, e.g. ``"0.7"``.

    Returns:
        A ``(frozen, skipped)`` pair: the golden paths copied, and the live golden
        paths skipped because ``"freezable"`` is ``false``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     d = Path(tmp)
        ...     _ = (d / "a.json").write_text('{"values": {"x": 1.0}}')
        ...     _ = (d / "b.json").write_text('{"values": {"x": 1.0}, "freezable": false}')
        ...     frozen, skipped = freeze(d, "0.1")
        ...     ([p.name for p in frozen], [p.name for p in skipped])
        (['a.json'], ['b.json'])

        ```
    """
    all_live = sorted(goldens_dir.glob("*.json"))
    frozen = freezable_goldens(goldens_dir)
    skipped = [p for p in all_live if p not in frozen]
    dest = goldens_dir / "frozen" / minor
    dest.mkdir(parents=True, exist_ok=True)
    for path in frozen:
        shutil.copy(path, dest / path.name)
    return frozen, skipped


def main(argv: list[str] | None = None) -> int:
    """Freeze the current goldens into ``goldens/frozen/<minor>`` and report the result.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``); expects exactly
            one positional argument, the release minor version (e.g. ``"0.7"``).

    Returns:
        ``0`` on success, ``1`` when the minor version argument is missing.
    """
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or not argv[0]:
        print("usage: freeze_goldens.py <minor>", file=sys.stderr)
        return 1
    minor = argv[0]
    frozen, skipped = freeze(GOLDENS_DIR, minor)
    print(f"froze {len(frozen)} golden(s) into goldens/frozen/{minor}/")
    if skipped:
        print(f"skipped (freezable: false): {', '.join(p.name for p in skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
