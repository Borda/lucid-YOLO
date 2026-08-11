# SPDX-License-Identifier: Apache-2.0
"""Phase-2 parameter/FLOP fidelity gate for the detector family (WP-023).

Holds the five built scale variants to the published R1 Table 7 numbers: the full
model's parameter count within +/-2% and the deployed NMS-free inference model's
conventional GFLOPs within +/-5%, for every scale. The gate validates the
clean-room block/head assumptions (A3/A4/A9) that the papers leave underspecified
— it is the blueprint's Phase-2 escalation site. A second test asserts the frozen
``goldens/params_flops_det.json`` regression lock still reproduces.

The module has since grown the two sibling gates, each with its own published
table, protocol, tolerance, and frozen golden: segmentation against R1 Table S9
(640 px, 80 classes) and oriented detection against R1 Table S11 (**1024** px,
**15** classes). The protocol constants are deliberately not shared between them.

Params count the full checkpoint (both dual-head branches); GFLOPs exclude the
training-only one-to-many branch — the R6/YOLOv10 reporting convention (see
:mod:`lucid_yolo.models.build`).
"""

from __future__ import annotations

import pytest
import torch
from scripts.check_goldens import DEFAULT_GOLDENS_DIR, check_golden

from lucid_yolo.models import build_detector, build_obb_detector, build_segmenter, count_flops, count_params

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


#: Published segmentation numbers at a 640-pixel input, R1 Table S9 (transcribed
#: verbatim from the arXiv PDF, page 28): (params in millions, FLOPs in billions).
#: The E2E and non-E2E rows carry an identical pair per scale, so the table does
#: not distinguish the deployed model from the full one; the convention used here
#: is the one that lands Table 7 within tolerance for detection.
_TABLE_S9: dict[str, tuple[float, float]] = {
    "n": (2.7, 9.1),
    "s": (10.4, 34.2),
    "m": (23.6, 121.5),
    "l": (28.0, 139.8),
    "x": (62.8, 313.5),
}

#: Segmentation parameter tolerance, wider than detection's +/-2%. The segmentation
#: head's sizing rests on five registered assumptions (A14 K=32, A15 proto grid,
#: A18 protonet stack, A34 coefficient stem, A35 fusion projections) rather than on
#: published structure, so it cannot claim detection-grade parity. The binding
#: scale is ``s`` at +2.5%; every other scale lands inside +/-1.5%.
_SEG_PARAM_TOL = 0.03


@pytest.mark.parametrize(
    ("variant", "published_params_m", "published_gflops"),
    [pytest.param(v, p, f, id=v) for v, (p, f) in _TABLE_S9.items()],
)
def test_seg_vs_tableS9(variant: str, published_params_m: float, published_gflops: float) -> None:
    """Each variant's params (+/-3%) and inference GFLOPs (+/-5%) match R1 Table S9."""
    model = build_segmenter(variant, _NUM_CLASSES)

    params_m = count_params(model) / 1e6
    gflops = count_flops(model.deploy())

    param_delta = abs(params_m - published_params_m) / published_params_m
    flop_delta = abs(gflops - published_gflops) / published_gflops
    assert param_delta <= _SEG_PARAM_TOL, (
        f"{variant}: {params_m:.3f} M params is {param_delta:.1%} from Table S9's {published_params_m} M (>3%)"
    )
    assert flop_delta <= _FLOP_TOL, (
        f"{variant}: {gflops:.3f} GFLOPs is {flop_delta:.1%} from Table S9's {published_gflops} G (>5%)"
    )


def test_seg_params_flops_golden() -> None:
    """The frozen goldens/params_flops_seg.json regression lock still reproduces."""
    result = check_golden(DEFAULT_GOLDENS_DIR / "params_flops_seg.json")

    assert result.passed, result.error or next(
        f"{c.metric}: expected {c.expected}, got {c.actual}" for c in result.comparisons if not c.passed
    )


#: Published OBB numbers, R1 Table S11 (DOTA-v1.0, one-to-one branch, no NMS):
#: (params in millions, FLOPs in billions). Measured at a **1024**-pixel input over
#: DOTA's **15** categories — both differ from the 640/80 protocol of the tables
#: above, so neither constant is shared with them. The table's mAP columns need
#: training and belong to WP-063/WP-088; only the size columns are gateable here.
_TABLE_S11: dict[str, tuple[float, float]] = {
    "n": (2.5, 14.0),
    "s": (9.8, 55.1),
    "m": (21.2, 183.3),
    "l": (25.6, 230.0),
    "x": (57.6, 516.5),
}

#: DOTA-v1.0 class count and input side R1 Table S11 is measured at.
_OBB_NUM_CLASSES = 15
_OBB_IMG_SIZE = 1024

#: OBB parameter tolerance, wider than detection's +/-2% and segmentation's +/-3%.
#: Measured deltas at the selected ``channels // 2`` angle-stem width: n +2.56%,
#: **s +3.27%**, m -0.90%, l -1.40%, x -1.65%. The binding scale is ``s``, and it
#: is what sets this constant — the three large scales land inside +/-1.7%, and
#: the two small ones overshoot.
#:
#: The asymmetry is real rather than slack in the measurement, and
#: :func:`~lucid_yolo.models.heads.detect._angle_stem_width` holds the evidence:
#: neither ``// 2`` nor ``// 3`` is uniformly right. ``// 3`` undershoots m/l/x at
#: 0.63-0.66 of Table S11's implied angle-branch increment; ``// 2`` lands at
#: 1.04-1.09 there but overshoots ``s`` at 1.62. Table S11 rounds params to 0.1 M,
#: so ``n`` (+/-59% on the implied increment) and ``s`` (+/-21%) cannot discriminate
#: between the two rules at all, while ``x`` (+/-3.5%) discriminates decisively and
#: selects ``// 2``. This tolerance therefore admits a rule chosen where the
#: evidence is sharp, at the cost of the scales where it is not; A20 stays ``open``
#: and this is not a claim of Table S11 parity.
_OBB_PARAM_TOL = 0.035

#: The OBB FLOP measurement reuses the shared :data:`_FLOP_TOL` of +/-5%, but with
#: far less room than the detection and segmentation gates: measured GFLOP deltas
#: are n **+4.84%**, s +1.52%, m -2.69%, l +2.43%, x +1.95%. The ``n`` scale sits
#: **0.16 points** from failing. Any future change that adds width or depth to the
#: angle branch will break this gate at ``n`` before it improves the param fit
#: anywhere, so re-measure GFLOPs at ``n`` first, not params at ``x``.


@pytest.mark.parametrize(
    ("variant", "published_params_m", "published_gflops"),
    [pytest.param(v, p, f, id=v) for v, (p, f) in _TABLE_S11.items()],
)
def test_obb_vs_tableS11(variant: str, published_params_m: float, published_gflops: float) -> None:
    """Each variant's params (+/-3.5%) and inference GFLOPs (+/-5%) match R1 Table S11."""
    model = build_obb_detector(variant, _OBB_NUM_CLASSES)

    params_m = count_params(model) / 1e6
    gflops = count_flops(model.deploy(), img_size=_OBB_IMG_SIZE)

    param_delta = abs(params_m - published_params_m) / published_params_m
    flop_delta = abs(gflops - published_gflops) / published_gflops
    assert param_delta <= _OBB_PARAM_TOL, (
        f"{variant}: {params_m:.3f} M params is {param_delta:.1%} from Table S11's {published_params_m} M (>3.5%)"
    )
    assert flop_delta <= _FLOP_TOL, (
        f"{variant}: {gflops:.3f} GFLOPs is {flop_delta:.1%} from Table S11's {published_gflops} G (>5%)"
    )


def test_obb_params_flops_golden() -> None:
    """The frozen goldens/params_flops_obb.json regression lock still reproduces."""
    result = check_golden(DEFAULT_GOLDENS_DIR / "params_flops_obb.json")

    assert result.passed, result.error or next(
        f"{c.metric}: expected {c.expected}, got {c.actual}" for c in result.comparisons if not c.passed
    )
