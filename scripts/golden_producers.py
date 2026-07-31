# SPDX-License-Identifier: Apache-2.0
"""Golden-metric producers for the WP-005 golden harness.

A *producer* is a zero-argument function returning ``dict[str, float]`` whose
result the golden harness (``scripts/check_goldens.py``) recomputes and compares
against a frozen ``goldens/*.json`` file. Every metric here is deterministic so
that an unchanged codebase reproduces byte-identical values on every run.

The single real producer, :func:`fixture_checksums`, derives its metrics from the
seeded WP-007 synthetic fixtures. Because those fixtures live under
``tests/fixtures/`` (not an importable package), they are loaded by file path via
``importlib.util`` — the same trick ``tests/meta/test_license_audit.py`` uses.

Examples:
    ```pycon
    >>> metrics = fixture_checksums()
    >>> metrics["detseg_num_images"]
    16.0

    ```
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from types import ModuleType

#: Repository root (``scripts/`` is one level below it).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Path to the WP-007 synthetic-fixture helpers, loaded by file path.
_SYNTHETIC_PATH = REPO_ROOT / "tests" / "fixtures" / "synthetic.py"

#: Per-split COCO annotation filename emitted by the fixture generator.
_COCO_ANNOTATION = "_annotations.coco.json"

#: The single split the fixtures materialize into.
_SPLIT = "train"

#: Number of leading hex digits of a SHA-256 digest folded into an exact float.
_SHA_PREFIX_LEN = 8


def _load_synthetic() -> ModuleType:
    """Load ``tests/fixtures/synthetic.py`` as an importable module.

    Returns:
        The loaded module exposing ``generate_detseg_fixtures`` and
        ``generate_obb_fixtures``.

    Examples:
        ```pycon
        >>> mod = _load_synthetic()
        >>> callable(mod.generate_detseg_fixtures)
        True

        ```
    """
    spec = importlib.util.spec_from_file_location("wp007_synthetic", _SYNTHETIC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _annotation_sha_float(annotation_path: Path) -> float:
    """Fold a fixture's COCO-annotation SHA-256 into an exactly representable float.

    The first :data:`_SHA_PREFIX_LEN` hex digits of the digest form a 32-bit
    integer (``<= 0xffffffff``), which every IEEE-754 double represents exactly,
    so the value compares cleanly under a zero tolerance.

    Args:
        annotation_path: Path to a ``_annotations.coco.json`` file.

    Returns:
        ``float(int(sha256(bytes).hexdigest()[:8], 16))``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     p = Path(tmp) / "a.json"
        ...     _ = p.write_bytes(b"{}")
        ...     0.0 <= _annotation_sha_float(p) <= 0xFFFFFFFF
        True

        ```
    """
    digest = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    return float(int(digest[:_SHA_PREFIX_LEN], 16))


def _dataset_metrics(prefix: str, dataset_dir: Path) -> dict[str, float]:
    """Compute ``{prefix}_num_images``, ``{prefix}_num_annotations``, ``{prefix}_annotation_sha``.

    Args:
        prefix: Metric-name prefix identifying the fixture set (``detseg``/``obb``).
        dataset_dir: The generated dataset directory holding ``train/``.

    Returns:
        A three-entry metric mapping derived from the split's COCO annotation file.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> mod = _load_synthetic()
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     ds = mod.generate_detseg_fixtures(Path(tmp))
        ...     sorted(_dataset_metrics("detseg", ds))
        ['detseg_annotation_sha', 'detseg_num_annotations', 'detseg_num_images']

        ```
    """
    annotation_path = dataset_dir / _SPLIT / _COCO_ANNOTATION
    coco = json.loads(annotation_path.read_text())
    return {
        f"{prefix}_num_images": float(len(coco["images"])),
        f"{prefix}_num_annotations": float(len(coco["annotations"])),
        f"{prefix}_annotation_sha": _annotation_sha_float(annotation_path),
    }


def fixture_checksums() -> dict[str, float]:
    """Deterministic checksum metrics over the WP-007 synthetic fixtures.

    Generates both seeded fixture sets (detection/segmentation and oriented-box)
    into a throwaway temporary directory, then reports each set's image count,
    annotation count, and annotation-file SHA-256 (folded to a float). The seeds
    are fixed (A26), so every field is byte-stable across runs and machines,
    which lets the golden file pin them with a zero tolerance.

    Returns:
        A mapping of six exact metrics: ``{detseg,obb}_num_images``,
        ``{detseg,obb}_num_annotations``, ``{detseg,obb}_annotation_sha``.

    Examples:
        ```pycon
        >>> metrics = fixture_checksums()
        >>> metrics["obb_num_images"]
        8.0
        >>> metrics["detseg_annotation_sha"] == fixture_checksums()["detseg_annotation_sha"]
        True

        ```
    """
    synthetic = _load_synthetic()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        detseg_dir = synthetic.generate_detseg_fixtures(root)
        obb_dir = synthetic.generate_obb_fixtures(root)
        return {
            **_dataset_metrics("detseg", detseg_dir),
            **_dataset_metrics("obb", obb_dir),
        }
