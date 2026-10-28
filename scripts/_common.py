# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the top-level scripts: the repository root and a file digest.

Import as ``scripts._common``, with the repository root on ``sys.path`` first -- the
scripts are loaded both as ``python scripts/x.py`` (where their own directory leads
``sys.path``) and as the ``scripts`` namespace package (``check_goldens`` imports
``scripts.golden_producers`` that way), and only the second form resolves a plain
``_common``. The lint checkers have their own, ``scripts/lint/_common.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Repository root (``scripts/`` is one level below it).
REPO_ROOT = Path(__file__).resolve().parents[1]


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
