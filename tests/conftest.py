# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for the offline test suite.

Exposes the WP-007 synthetic micro-datasets (A26) as session-scoped fixtures.
Both sets are generated once into a gitignored on-disk cache
(``tests/fixtures/_generated/``) and reused across the session; the generator
helpers are idempotent, so a warm cache is left untouched.

Import mechanism: ``synthetic.py`` lives in ``tests/fixtures/`` (a non-package
directory, no ``__init__.py``). We prepend that directory to ``sys.path`` here so
both this conftest and ``tests/fixtures/test_fixtures_load.py`` can ``import
synthetic`` by bare module name, without turning ``tests`` into an importable
package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
if str(_FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(_FIXTURES_DIR))

import synthetic  # noqa: E402  (path is arranged just above)

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Persistent, gitignored generation cache shared across the session.
_GENERATED_CACHE = _FIXTURES_DIR / "_generated"


@pytest.fixture(scope="session")
def detseg_fixture_dir() -> Iterator[Path]:
    """Yield the detection/segmentation micro-dataset directory (16 images).

    Generated once into ``tests/fixtures/_generated/detseg`` and reused.
    """
    _GENERATED_CACHE.mkdir(parents=True, exist_ok=True)
    yield synthetic.generate_detseg_fixtures(_GENERATED_CACHE)


@pytest.fixture(scope="session")
def obb_fixture_dir() -> Iterator[Path]:
    """Yield the oriented-bounding-box micro-dataset directory (8 images).

    Generated once into ``tests/fixtures/_generated/obb`` and reused.
    """
    _GENERATED_CACHE.mkdir(parents=True, exist_ok=True)
    yield synthetic.generate_obb_fixtures(_GENERATED_CACHE)


@pytest.fixture(scope="session")
def keypoints_fixture_dir() -> Iterator[Path]:
    """Yield the keypoints micro-dataset directory (12 images, WP-121b).

    Generated once into ``tests/fixtures/_generated/keypoints`` and reused.
    """
    _GENERATED_CACHE.mkdir(parents=True, exist_ok=True)
    yield synthetic.generate_keypoints_fixtures(_GENERATED_CACHE)
