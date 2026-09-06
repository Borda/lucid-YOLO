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

The field alone is a hand-written flag, and defaulting it to ``True`` left the WP-132 defect one
forgotten line away from a fourth appearance: a new generator-derived golden that simply omits
it is freezable by default, which is how WP-140 and WP-153 each got theirs back. So the flag is
no longer trusted on its own. Before copying anything, :func:`freeze` runs each live golden's
producer in its own interpreter and asks whether the run loaded :data:`GENERATOR_PACKAGE`
(:func:`_derives_from_generator`); a declared flag contradicting that answer, in either
direction, aborts the freeze with nothing written (M-42).

It is also the **only writer of** :data:`MANIFEST_NAME` (WP-168). ``check_goldens.py``
recomputes each frozen file's producer and compares it against that same file's own stored
values, so a commit editing a frozen golden's ``values`` and its ``tolerances`` together
passes the whole gate green — the file is compared with itself. The manifest pins the files
instead of the numbers, and ``scripts/lint/audit_frozen_manifest.py`` asserts every digest on
every commit touching ``goldens/frozen/``.

Two writing modes, because a release and a sanctioned move are different acts:

* ``freeze_goldens.py <minor>`` seals the files it just copied and leaves every other row
  untouched. A release must not re-bless snapshots it did not write.
* ``freeze_goldens.py --reseal`` recomputes every row from disk. This is the **only**
  sanctioned way to move a frozen golden, and moving one requires a recorded principal
  override first — ``AGENTS.md`` §4 escalation trigger 4, §7 standing prohibitions. The
  flag makes that move an explicit, reviewable act rather than a hand-edited digest.

Examples:
    Freeze the current goldens into ``goldens/frozen/0.7``::

        python scripts/freeze_goldens.py 0.7

    Re-seal every frozen file after a principal-approved golden move::

        python scripts/freeze_goldens.py --reseal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

#: Repository root (``scripts/`` is one level below it).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The ``fuse-augmentations`` subpackage whose presence marks a golden generator-derived.
#: Not the top-level package: since WP-157 ``lucid_yolo.data`` imports ``fuse_augmentations``
#: geometry helpers at module scope, so *every* producer loads the top-level package and it
#: separates nothing. ``fuse_augmentations.data`` is the dataset generator — the thing whose
#: movement can retire a snapshot no code change can satisfy (WP-132) — and is loaded only by
#: the producers that render a synthetic dataset.
GENERATOR_PACKAGE = "fuse_augmentations.data"

#: Probe run in a fresh interpreter: resolve the producer, run it, and report whether the
#: run loaded :data:`GENERATOR_PACKAGE`. A fresh process per producer is what makes the
#: ``sys.modules`` read meaningful — in one process the first generator-derived producer
#: would mark every later one.
_DERIVATION_PROBE = """
import sys
sys.path.insert(0, {repo!r})
from scripts.check_goldens import resolve_producer
resolve_producer({spec!r})()
print(any(m == {pkg!r} or m.startswith({pkg!r} + ".") for m in sys.modules))
"""

#: Directory holding the live top-level goldens.
GOLDENS_DIR = REPO_ROOT / "goldens"

#: Name of the digest manifest, sitting at the root of the frozen tree. The ``.sha256``
#: extension is load-bearing: ``check_goldens.py`` discovers ``goldens/frozen/**/*.json``,
#: and a manifest ending in ``.json`` would be handed to the golden parser as a malformed
#: golden.
MANIFEST_NAME = "MANIFEST.sha256"


def frozen_root(goldens_dir: Path) -> Path:
    """Return the directory holding the per-minor frozen snapshots.

    Args:
        goldens_dir: Directory holding the live ``*.json`` goldens.

    Returns:
        The ``frozen/`` subdirectory, whether or not it exists yet.

    Examples:
        ```pycon
        >>> frozen_root(GOLDENS_DIR).name
        'frozen'

        ```
    """
    return goldens_dir / "frozen"


def digest(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes.

    Args:
        path: File to hash.

    Returns:
        The 64-character lowercase hex digest.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     sample = Path(tmp) / "f.json"
        ...     _ = sample.write_bytes(b"{}")
        ...     digest(sample)[:16]
        '44136fa355b3678a'

        ```
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_files(root: Path) -> list[Path]:
    """List every frozen file the manifest must cover, manifest excluded.

    Args:
        root: The ``goldens/frozen`` directory.

    Returns:
        Sorted paths of every file under ``root`` except :data:`MANIFEST_NAME`, which
        is excluded from its own digest set — hashing it would change it.

    Examples:
        ```pycon
        >>> names = {p.name for p in frozen_files(frozen_root(GOLDENS_DIR))}
        >>> MANIFEST_NAME in names
        False

        ```
    """
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name != MANIFEST_NAME)


def read_manifest(root: Path) -> dict[str, str]:
    """Read the digest manifest into a ``{relative path: digest}`` mapping.

    Args:
        root: The ``goldens/frozen`` directory.

    Returns:
        One entry per manifest row, keyed by POSIX-style path relative to ``root``.
        Empty when the manifest does not exist, which the checker reports rather than
        this treating as an empty-and-therefore-satisfied set.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = (root / MANIFEST_NAME).write_text("abc  0.1/optim_toy.json\\n")
        ...     read_manifest(root)
        {'0.1/optim_toy.json': 'abc'}

        ```
    """
    path = root / MANIFEST_NAME
    if not path.is_file():
        return {}
    rows = (line.split("  ", 1) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {name: value for value, name in rows}


def write_manifest(root: Path, entries: dict[str, str]) -> Path:
    """Write ``entries`` to the manifest in ``sha256sum`` format, sorted by path.

    Args:
        root: The ``goldens/frozen`` directory.
        entries: ``{relative path: digest}`` to record.

    Returns:
        The manifest path written.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = write_manifest(Path(tmp), {"0.1/b.json": "bb", "0.1/a.json": "aa"})
        ...     path.read_text()
        'aa  0.1/a.json\\nbb  0.1/b.json\\n'

        ```
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_NAME
    path.write_text("".join(f"{entries[name]}  {name}\n" for name in sorted(entries)), encoding="utf-8")
    return path


def seal(root: Path, paths: list[Path]) -> Path:
    """Record ``paths`` in the manifest, leaving every other row as it stands.

    What a release freeze runs. A row for a file this call did not write is preserved
    verbatim: re-hashing the whole tree on every freeze would let a tampered snapshot
    from an earlier minor be re-blessed by the next release, which is the failure the
    manifest exists to catch.

    Args:
        root: The ``goldens/frozen`` directory.
        paths: Files to seal, each under ``root``.

    Returns:
        The manifest path written.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = write_manifest(root, {"0.1/old.json": "stale-row-kept"})
        ...     (root / "0.2").mkdir()
        ...     new = root / "0.2" / "optim_toy.json"
        ...     _ = new.write_bytes(b"{}")
        ...     _ = seal(root, [new])
        ...     sorted(read_manifest(root))
        ['0.1/old.json', '0.2/optim_toy.json']

        ```
    """
    entries = read_manifest(root)
    entries.update({path.relative_to(root).as_posix(): digest(path) for path in paths})
    return write_manifest(root, entries)


def reseal(root: Path) -> Path:
    """Recompute every manifest row from what is on disk right now.

    The sanctioned-move command, and the only one: moving a frozen golden's values is
    forbidden by ``AGENTS.md`` §7 and requires a recorded principal override under §4
    escalation trigger 4. This flag does not grant that permission — it makes acting on
    a granted one a single reviewable command instead of a hand-edited digest.

    Args:
        root: The ``goldens/frozen`` directory.

    Returns:
        The manifest path written.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = write_manifest(root, {"0.1/gone.json": "row-for-a-deleted-file"})
        ...     (root / "0.1").mkdir()
        ...     _ = (root / "0.1" / "here.json").write_bytes(b"{}")
        ...     _ = reseal(root)
        ...     sorted(read_manifest(root))
        ['0.1/here.json']

        ```
    """
    return write_manifest(root, {path.relative_to(root).as_posix(): digest(path) for path in frozen_files(root)})


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


def _derives_from_generator(spec: str) -> bool | None:
    """Run a producer in a fresh interpreter and report whether it loads the dataset generator.

    Args:
        spec: The golden's ``"module:function"`` producer spec.

    Returns:
        ``True`` when the run loaded :data:`GENERATOR_PACKAGE` (so the golden is
        generator-derived and must not be frozen), ``False`` when it did not, and
        ``None`` when the producer could not be resolved or raised — a verdict this
        cannot derive rather than a verdict of ``False``.
    """
    probe = _DERIVATION_PROBE.format(repo=str(REPO_ROOT), spec=spec, pkg=GENERATOR_PACKAGE)
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO_ROOT, check=False
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip().splitlines()[-1] == "True"


def _freezable_disagreements(paths: list[Path]) -> list[str]:
    """Report every live golden whose declared ``freezable`` contradicts its producer.

    The flag is hand-written, and ``check_goldens.py`` only catches the case where a
    ``freezable: false`` file has *already* reached ``goldens/frozen/``. WP-132 removed two
    generator-derived snapshots; WP-140 and WP-153 each put one back. A golden that simply
    omits the field used to default to freezable, which is the same silent path a third time
    (M-42). Both directions are reported: a generator-derived golden claiming to be freezable,
    and a golden claiming otherwise whose producer no longer reaches the generator, which
    would keep it out of every release snapshot for a reason that has expired.

    Args:
        paths: Live top-level golden paths to cross-check.

    Returns:
        One message per disagreement, empty when every derivable flag agrees. A golden
        with no producer spec, or one that cannot be run, is skipped: it is
        ``check_goldens.py``'s job to fail a golden with no runnable producer.
    """
    messages = []
    for path in paths:
        data = json.loads(path.read_text())
        spec = data.get("producer")
        if not isinstance(spec, str) or not spec:
            continue
        derived = _derives_from_generator(spec)
        declared_freezable = bool(data.get("freezable", True))
        if derived is True and declared_freezable:
            messages.append(
                f"{path.name}: producer {spec!r} loads {GENERATOR_PACKAGE}, so its values are pinned to an "
                "external generator and no future code change can keep a snapshot of them green — set "
                '"freezable": false'
            )
        elif derived is False and not declared_freezable:
            messages.append(
                f'{path.name}: declares "freezable": false, but producer {spec!r} does not load '
                f"{GENERATOR_PACKAGE} — it is excluded from every release snapshot for a reason that no "
                "longer holds; drop the field"
            )
    return messages


class FreezeError(Exception):
    """A live golden's declared ``freezable`` flag contradicts what its producer does."""


def freeze(goldens_dir: Path, minor: str) -> tuple[list[Path], list[Path]]:
    """Copy every freezable live golden into ``goldens_dir/frozen/<minor>/`` and seal it.

    Every declared ``freezable`` flag is cross-checked against what its producer actually
    does before anything is copied (:func:`_freezable_disagreements`), so a disagreement
    aborts the release freeze with nothing written rather than sealing a snapshot that can
    never be kept green. This runs each live producer once in its own interpreter, which on
    this repository's nine goldens costs roughly twenty seconds — paid by a release, not by
    ``make gate``.

    Args:
        goldens_dir: Directory holding the live ``*.json`` goldens.
        minor: Release minor version, e.g. ``"0.7"``.

    Returns:
        A ``(frozen, skipped)`` pair: the golden paths copied, and the live golden
        paths skipped because ``"freezable"`` is ``false``.

    Raises:
        FreezeError: If any live golden's declared ``freezable`` flag contradicts whether
            its producer loads :data:`GENERATOR_PACKAGE`.

    Examples:
        ```pycon
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     d = Path(tmp)
        ...     _ = (d / "a.json").write_text('{"values": {"x": 1.0}}')
        ...     _ = (d / "b.json").write_text('{"values": {"x": 1.0}, "freezable": false}')
        ...     frozen, skipped = freeze(d, "0.1")
        ...     ([p.name for p in frozen], [p.name for p in skipped], sorted(read_manifest(frozen_root(d))))
        (['a.json'], ['b.json'], ['0.1/a.json'])

        ```
    """
    all_live = sorted(goldens_dir.glob("*.json"))
    disagreements = _freezable_disagreements(all_live)
    if disagreements:
        raise FreezeError("declared 'freezable' contradicts the producer:\n  " + "\n  ".join(disagreements))
    frozen = freezable_goldens(goldens_dir)
    skipped = [p for p in all_live if p not in frozen]
    root = frozen_root(goldens_dir)
    dest = root / minor
    dest.mkdir(parents=True, exist_ok=True)
    for path in frozen:
        shutil.copy(path, dest / path.name)
    seal(root, [dest / path.name for path in frozen])
    return frozen, skipped


def main(argv: list[str] | None = None) -> int:
    """Freeze the current goldens into ``goldens/frozen/<minor>``, or re-seal the manifest.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` when neither a minor version nor ``--reseal`` is given.

    Examples:
        ```pycon
        >>> import contextlib, io
        >>> with contextlib.redirect_stderr(io.StringIO()):
        ...     main([])
        1

        ```
    """
    parser = argparse.ArgumentParser(description="Freeze the live goldens into a release snapshot.")
    parser.add_argument("minor", nargs="?", default=None, help="release minor version to freeze into, e.g. 0.7")
    parser.add_argument(
        "--reseal",
        action="store_true",
        help="recompute every manifest digest from disk; the sanctioned way to record a principal-approved "
        "frozen-golden move, and the only one that is not a hand-edited digest",
    )
    args = parser.parse_args(argv)

    root = frozen_root(GOLDENS_DIR)
    if args.reseal:
        reseal(root)
        print(f"re-sealed {len(read_manifest(root))} frozen file(s) in {MANIFEST_NAME}")
        return 0
    if not args.minor:
        print("usage: freeze_goldens.py <minor> | freeze_goldens.py --reseal", file=sys.stderr)
        return 1
    try:
        frozen, skipped = freeze(GOLDENS_DIR, args.minor)
    except FreezeError as exc:
        print(f"refusing to freeze — {exc}", file=sys.stderr)
        return 1
    print(f"froze {len(frozen)} golden(s) into goldens/frozen/{args.minor}/, sealed in {MANIFEST_NAME}")
    if skipped:
        print(f"skipped (freezable: false): {', '.join(p.name for p in skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
