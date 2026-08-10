# SPDX-License-Identifier: Apache-2.0
"""Invariant gate for the detection objective (WP-082).

These tests exist because of what the Det-smoke tier runs cost. WP-078 (the L1 term
measured in pixels while its gain is calibrated for stride units) survived 474
tests, five green goldens, 96% line coverage, and three full COCO training runs.
It never crashed, never produced a NaN, never changed a tensor shape. It was a
*semantic* defect, and a suite that checks shapes and finiteness is structurally
blind to those.

Two invariants pin it:

* **frame** — passing strides must divide the L1 term by exactly the positive
  anchor's stride. This pins the coordinate frame itself, independently of what
  the predictions happen to be, and is the precise guard.
* **scale** — on a state whose boxes are essentially correct, the L1 term must be
  a fraction of a stride. In the pixel frame the same state reads 8x larger.

A note on what is deliberately *not* tested here: an absolute bound on any one
term's share of the objective. Measured on a converged state the L1 share is
10.7% in the broken frame against 1.5% in the correct one — both comfortably
under any threshold a reader would accept, so such a test would look like
protection while catching nothing. The 97.1% share that WP-078 actually produced
came from *unconverged* boxes with large pixel errors, a state no cheap fixture
reproduces faithfully. The ratio invariant above catches the defect at its
source instead.
"""

from __future__ import annotations

import pytest
import torch

from lucid_yolo.losses.dual_loss import DualBranchLoss
from lucid_yolo.models.heads.detect import decode_ltrb

_NUM_CLASSES = 6
_IMG_SIZE = 128

#: Ceiling on the raw (pre-gain) L1 term for a state whose boxes are essentially
#: correct. Stride-unit L1 measures 0.75 there; the pixel frame reads 5.99 on the
#: identical predictions, so 2.0 separates them with room on both sides.
_MAX_CONVERGED_L1 = 2.0

#: Offset applied to the encoded distances, in stride units, so the L1 term is
#: non-zero without making the boxes wrong enough to collapse the IoU weighting.
_LTRB_OFFSET = 0.2


def _converged_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a prediction state resembling a partially trained model.

    Balance cannot be measured at random initialization: alignment weights carry
    ``IoU ** beta`` with beta 6, so an untrained head's near-zero overlap drives
    every weight to ~1e-13 and the box and L1 terms read exactly zero no matter
    how the loss is scaled. The defect this file guards against only surfaces once
    boxes are roughly right and L1 is the term still carrying weight.

    Every anchor is therefore made to predict one ground-truth box, offset by a
    fifth of a stride, with class logits confident at the true class.

    Returns:
        ``(logits, ltrb, anchor_points, strides, gt_boxes, gt_labels)``.
    """
    box = torch.tensor([16.0, 16.0, 112.0, 112.0])
    grids = []
    for stride, side in ((8.0, _IMG_SIZE // 8), (16.0, _IMG_SIZE // 16), (32.0, _IMG_SIZE // 32)):
        centers = (torch.arange(side, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(centers, centers, indexing="ij")
        points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        grids.append((points, torch.full((points.shape[0],), stride)))
    anchor_points = torch.cat([points for points, _ in grids])
    strides = torch.cat([level for _, level in grids])

    left_top = (anchor_points - box[:2]) / strides.unsqueeze(-1)
    right_bottom = (box[2:] - anchor_points) / strides.unsqueeze(-1)
    ltrb = torch.cat([left_top, right_bottom], dim=1).clamp(min=0.0).unsqueeze(0) + _LTRB_OFFSET

    logits = torch.full((1, anchor_points.shape[0], _NUM_CLASSES), -4.0)
    logits[..., 0] = 1.5
    return logits, ltrb, anchor_points, strides, box.reshape(1, 1, 4), torch.zeros(1, 1, dtype=torch.long)


def _l1_term(strides: torch.Tensor | None) -> float:
    """Return the raw one-to-one L1 term for the converged state in the given frame."""
    logits, ltrb, anchor_points, anchor_strides, gt_boxes, gt_labels = _converged_state()
    boxes = decode_ltrb(ltrb, anchor_points, anchor_strides)
    gt_mask = torch.ones(1, 1, dtype=torch.bool)
    out = DualBranchLoss()(logits, boxes, logits, boxes, anchor_points, gt_boxes, gt_labels, gt_mask, strides=strides)
    assert float(out.o2o.box) > 0.0, "no positives assigned — the fixture, not the loss, is at fault"
    return float(out.o2o.l1)


def test_converged_boxes_keep_the_l1_term_small() -> None:
    """With boxes essentially correct the stride-unit L1 term stays well under one stride (WP-078 guard)."""
    _, _, _, strides, _, _ = _converged_state()
    assert _l1_term(strides) < _MAX_CONVERGED_L1


def test_pixel_frame_l1_would_trip_the_scale_guard() -> None:
    """The identical predictions read far larger in the pixel frame — the defect this guard exists for."""
    _, _, _, strides, _, _ = _converged_state()
    assert _l1_term(None) > _l1_term(strides)
    assert _l1_term(None) > _MAX_CONVERGED_L1


@pytest.mark.parametrize(
    "stride",
    [pytest.param(8.0, id="p3"), pytest.param(16.0, id="p4"), pytest.param(32.0, id="p5")],
)
def test_l1_term_is_measured_in_stride_units(stride: float) -> None:
    """The stride-normalized L1 term scales as 1/stride against the pixel frame (WP-078 guard).

    Pins the coordinate frame directly: whatever the predictions happen to be,
    passing strides must divide the term by exactly that stride.
    """
    generator = torch.Generator().manual_seed(0)
    anchor_points = torch.tensor([[4.0, 4.0]])
    strides = torch.tensor([stride])
    logits = torch.randn(1, 1, _NUM_CLASSES, generator=generator)
    ltrb = torch.rand(1, 1, 4, generator=generator) * 2.0 + 0.5
    boxes = decode_ltrb(ltrb, anchor_points, strides)
    gt_boxes = torch.tensor([[[0.0, 0.0, 24.0, 24.0]]])
    gt_labels = torch.zeros(1, 1, dtype=torch.long)
    gt_mask = torch.ones(1, 1, dtype=torch.bool)
    loss = DualBranchLoss()

    args = (logits, boxes, logits, boxes, anchor_points, gt_boxes, gt_labels, gt_mask)
    pixel = loss(*args)
    strided = loss(*args, strides=strides)

    assert float(strided.o2o.l1) == pytest.approx(float(pixel.o2o.l1) / stride, rel=1e-5)
    assert float(strided.o2o.box) == pytest.approx(float(pixel.o2o.box), rel=1e-6)
    assert float(strided.o2o.cls) == pytest.approx(float(pixel.o2o.cls), rel=1e-6)
