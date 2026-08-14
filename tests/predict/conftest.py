# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the single-image inference suites (WP-089, WP-090).

Import mechanism: ``planted.py`` sits beside this file in a non-package directory (no
``__init__.py``, so ``tests`` stays un-importable as a package — see
``tests/conftest.py``). This directory is prepended to ``sys.path`` here so both this
conftest and the two test modules can ``import planted`` by bare module name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torchvision.io import write_png

_PREDICT_DIR = Path(__file__).resolve().parent
if str(_PREDICT_DIR) not in sys.path:
    sys.path.insert(0, str(_PREDICT_DIR))

from planted import ORIGINAL_SIZE  # noqa: E402  (path is arranged just above)


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    """Write a 64x128 RGB image and return its path.

    Content is irrelevant — the stub modules ignore it — but the file has to decode, and
    its shape is what fixes the letterbox ratio and pad the assertions are written
    against.
    """
    path = tmp_path / "scene.png"
    write_png(torch.full((3, *ORIGINAL_SIZE), 128, dtype=torch.uint8), str(path))
    return path
