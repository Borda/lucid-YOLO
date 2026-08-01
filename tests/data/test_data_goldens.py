# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-015 data goldens and the debug-grid visualizer.

Covers three contracts: the frozen ``goldens/data_checksums.json`` passes the
WP-005 harness against a live pipeline draw; the producer is deterministic on this
platform (two calls return identical metrics); and the ``dump_debug_grid`` CLI
renders a decodable annotated PNG from the WP-007 synthetic fixture directory
(offline, no dataset download). Modules under ``scripts/`` are loaded by file path
because ``scripts`` is not an importable package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "goldens" / "data_checksums.json"


def _load_module(name: str, path: Path) -> ModuleType:
    """Load a ``scripts/`` module by file path (``scripts`` is not a package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered before exec
    spec.loader.exec_module(module)
    return module


harness = _load_module("check_goldens", REPO_ROOT / "scripts" / "check_goldens.py")
producers = _load_module("golden_producers", REPO_ROOT / "scripts" / "golden_producers.py")
visualizer = _load_module("dump_debug_grid", REPO_ROOT / "scripts" / "dump_debug_grid.py")


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed the global RNG before each test (the pipeline uses explicit generators)."""
    torch.manual_seed(0)


def test_data_golden_passes_check() -> None:
    """The frozen data golden recomputes and matches within tolerance (DoD a)."""
    result = harness.check_golden(GOLDEN)
    assert result.passed, harness.format_result(result, GOLDEN.parent)


def test_producer_is_deterministic() -> None:
    """Two producer calls return byte-identical metrics on this platform (DoD b)."""
    first = producers.data_pipeline_metrics()
    second = producers.data_pipeline_metrics()
    assert first == second


def test_visualizer_writes_decodable_image(detseg_fixture_dir: Path, tmp_path: Path) -> None:
    """The CLI renders a decodable annotated PNG from the fixture dir (DoD c)."""
    out = tmp_path / "grid.png"
    exit_code = visualizer.main(
        ["--data-root", str(detseg_fixture_dir), "--out", str(out), "--samples", "4", "--seed", "0"]
    )
    assert exit_code == 0
    assert out.is_file()
    with Image.open(out) as image:
        image.verify()
