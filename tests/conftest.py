# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for the offline test suite.

Seeds the torch RNG before every test (``_seed_torch_rng``) so no module can draw
from leftover state, and exposes the WP-007 synthetic micro-datasets (A26) as
session-scoped fixtures.
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
import torch

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
if str(_FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(_FIXTURES_DIR))

import synthetic  # noqa: E402  (path is arranged just above)

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Persistent, gitignored generation cache shared across the session.
_GENERATED_CACHE = _FIXTURES_DIR / "_generated"

#: Global seed applied before every test; matches the value the per-module
#: fixtures already use, so files carrying their own seeding see no change.
_SEED = 0


@pytest.fixture(autouse=True)
def _seed_torch_rng() -> None:
    """Reset the torch RNG before every test so no module draws from leftover state.

    Suite-wide default: a handful of modules (model, decode and rasterisation
    tests) build random tensors with no seeding of their own, which makes their
    inputs depend on whichever test ran before them. Seeding centrally here is
    the repo convention over per-test seeding, and is deliberately additive --
    ``torch.manual_seed`` is idempotent, so the ~30 modules that already declare
    their own ``autouse`` seeding fixture at the same value are unaffected
    whichever order the two fixtures run in.

    Only torch is seeded: the suite draws no randomness from ``numpy.random`` or
    the stdlib ``random`` module (verified by grep over ``tests/``), so seeding
    those would be dead code. ``torch.manual_seed`` already covers all CUDA
    devices, so no separate ``manual_seed_all`` call is needed.

    The name is deliberately distinct from the three per-module fixture names in
    use (``reset_random_seeds``, ``_seed_rng``, ``_seed``): a same-named fixture
    in a test module *overrides* the conftest one rather than composing with it,
    which would silently disable this for every module that already has one.
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
