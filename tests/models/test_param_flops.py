# SPDX-License-Identifier: Apache-2.0
"""Phase-2 parameter/FLOP fidelity gate for the detector family (WP-023).

Holds the five built scale variants to the published R1 Table 7 numbers: the full
model's parameter count within +/-2% and the deployed NMS-free inference model's
conventional GFLOPs within +/-5%, for every scale. The gate validates the
clean-room block/head assumptions (A3/A4/A9) that the papers leave underspecified
— it is the blueprint's Phase-2 escalation site. A second test asserts the frozen
``goldens/params_flops_det.json`` regression lock still reproduces.

Params count the full checkpoint (both dual-head branches); GFLOPs exclude the
training-only one-to-many branch — the R6/YOLOv10 reporting convention (see
:mod:`open_yolos.models.build`).
"""

from __future__ import annotations

import pytest
import torch
from scripts.check_goldens import DEFAULT_GOLDENS_DIR, check_golden

from open_yolos.models import build_detector, count_flops, count_params

#: Published detection numbers at a 640-pixel input, R1 Table 7 (verified verbatim
#: against the arXiv HTML): (params in millions, conventional GFLOPs). Rounded to
#: 0.1 M / 0.1 G in the paper, so tolerances are taken relative to the printed
#: value.
_TABLE7: dict[str, tuple[float, float]] = {
    "n": (2.4, 5.4),
    "s": (9.5, 20.7),
    "m": (20.4, 68.2),
    "l": (24.8, 86.4),
    "x": (55.7, 193.9),
}

#: Gate tolerances (relative to the published value): params +/-2%, FLOPs +/-5%.
_PARAM_TOL = 0.02
_FLOP_TOL = 0.05

#: COCO class count the published table is measured at.
_NUM_CLASSES = 80


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed torch RNG so weight init is deterministic across the gate."""
    torch.manual_seed(0)


@pytest.mark.parametrize(
    ("variant", "published_params_m", "published_gflops"),
    [pytest.param(v, p, f, id=v) for v, (p, f) in _TABLE7.items()],
)
def test_det_vs_table7(variant: str, published_params_m: float, published_gflops: float) -> None:
    """Each variant's params (+/-2%) and inference GFLOPs (+/-5%) match R1 Table 7."""
    model = build_detector(variant, _NUM_CLASSES)

    params_m = count_params(model) / 1e6
    gflops = count_flops(model.deploy())

    param_delta = abs(params_m - published_params_m) / published_params_m
    flop_delta = abs(gflops - published_gflops) / published_gflops
    assert param_delta <= _PARAM_TOL, (
        f"{variant}: {params_m:.3f} M params is {param_delta:.1%} from Table 7's {published_params_m} M (>2%)"
    )
    assert flop_delta <= _FLOP_TOL, (
        f"{variant}: {gflops:.3f} GFLOPs is {flop_delta:.1%} from Table 7's {published_gflops} G (>5%)"
    )


def test_det_params_flops_golden() -> None:
    """The frozen goldens/params_flops_det.json regression lock still reproduces."""
    result = check_golden(DEFAULT_GOLDENS_DIR / "params_flops_det.json")

    assert result.passed, result.error or next(
        f"{c.metric}: expected {c.expected}, got {c.actual}" for c in result.comparisons if not c.passed
    )
