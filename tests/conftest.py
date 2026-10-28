# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for the offline test suite.

Seeds the torch RNG before every test (``_seed_torch_rng``) so no module can draw
from leftover state, and exposes the WP-007 synthetic micro-datasets (A26) as
session-scoped fixtures.
Both sets are generated once into a gitignored on-disk cache
(``tests/fixtures/_generated/``) and reused across the session; the generator
helpers are idempotent, so a warm cache is left untouched, and each set is published
by atomic rename, so pytest-xdist workers racing on a cold cache never read a
half-written one.

Import mechanism: ``synthetic.py`` lives in ``tests/fixtures/`` (a non-package
directory, no ``__init__.py``). We prepend that directory to ``sys.path`` here so
both this conftest and ``tests/fixtures/test_fixtures_load.py`` can ``import
synthetic`` by bare module name, without turning ``tests`` into an importable
package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# cuBLAS reads this before creating CUDA handles; retain an explicit caller setting.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
if str(_FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(_FIXTURES_DIR))

import synthetic  # noqa: E402  (path is arranged just above)

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Persistent, gitignored generation cache shared across the session.
_GENERATED_CACHE = _FIXTURES_DIR / "_generated"

#: Global seed applied before every test.
_SEED = 0


@pytest.fixture(autouse=True)
def _seed_torch_rng() -> None:
    """Reset the torch RNG before every test so no module draws from leftover state.

    Suite-wide default: modules that build random tensors would otherwise draw
    inputs that depend on whichever test ran before them. Seeding centrally here
    is the repo convention; a test module needs no seeding fixture of its own
    unless it seeds differently or does more than seed.

    Only torch is seeded: the suite draws no randomness from ``numpy.random`` or
    the stdlib ``random`` module (verified by grep over ``tests/``), so seeding
    those would be dead code. ``torch.manual_seed`` already covers all CUDA
    devices, so no separate ``manual_seed_all`` call is needed.

    A module fixture of this exact name would *override* this one rather than
    compose with it, silently disabling the seeding for that module.
    """
    torch.manual_seed(_SEED)


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
