# SPDX-License-Identifier: Apache-2.0
"""Overfit-100 integration test: the full train loop memorizes a fixed slice (WP-040).

Exercises :mod:`scripts.overfit_micro` end to end. One accelerator-gated fixture
runs the whole pipeline once — generate the ~100-image synthetic slice, train the
``n``-scale detector, and score train-set recall at IoU 0.5 — and the ``gpu``/``data``
tests assert the WP-040 DoD against it (recall clears the ``>= 0.95`` floor, the
integer counts are exact, and the achieved recall matches the frozen golden within
tolerance). A separate mark-free test pins the offline-harness contract: the frozen
golden lives under ``goldens/gpu/`` and is therefore *excluded* from the default
``scripts/check_goldens.py`` discovery, so it runs in the offline unit gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import check_goldens, overfit_micro  # noqa: E402  (path arranged just above)

#: The exact integer-valued metrics the overfit slice must always produce.
_EXPECTED_COUNTS = (100.0, 592.0, 4.0, 100.0, 320.0)


def test_overfit_golden_excluded_from_offline_harness() -> None:
    """The frozen golden is discovered only under gpu/, never by the default harness."""
    goldens_dir = check_goldens.DEFAULT_GOLDENS_DIR
    assert overfit_micro._GOLDEN_PATH.is_file()
    assert overfit_micro._GOLDEN_PATH in check_goldens.discover_gpu_goldens(goldens_dir)
    assert overfit_micro._GOLDEN_PATH not in check_goldens.discover_goldens(goldens_dir)


@pytest.fixture(scope="module")
def overfit_metrics() -> dict[str, float]:
    """Run the full overfit pipeline once and share its metrics across the gpu tests."""
    return overfit_micro.run_overfit()


@pytest.mark.gpu
@pytest.mark.data
def test_overfit_recall_clears_floor(overfit_metrics: dict[str, float]) -> None:
    """Train-set recall at IoU 0.5 reaches the WP-040 DoD floor (>= 0.95)."""
    assert overfit_metrics["train_recall_at_050"] >= overfit_micro._RECALL_FLOOR


@pytest.mark.gpu
@pytest.mark.data
def test_overfit_integer_counts_are_exact(overfit_metrics: dict[str, float]) -> None:
    """The slice's integer metrics (images, instances, classes, epochs, size) are fixed."""
    counts = (
        overfit_metrics["num_images"],
        overfit_metrics["num_instances"],
        overfit_metrics["num_classes"],
        overfit_metrics["epochs"],
        overfit_metrics["img_size"],
    )
    assert counts == _EXPECTED_COUNTS


@pytest.mark.gpu
@pytest.mark.data
def test_overfit_recall_matches_frozen_golden(overfit_metrics: dict[str, float]) -> None:
    """The recomputed recall matches the frozen golden within its tolerance."""
    golden = json.loads(overfit_micro._GOLDEN_PATH.read_text())
    expected = golden["values"]["train_recall_at_050"]
    tolerance = golden["tolerances"]["train_recall_at_050"]
    assert abs(overfit_metrics["train_recall_at_050"] - expected) <= tolerance
