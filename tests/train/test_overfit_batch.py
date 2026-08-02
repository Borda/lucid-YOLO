# SPDX-License-Identifier: Apache-2.0
"""Phase 3 exit gate: the assignment + loss stack is optimizable end to end (WP-030).

This is the Phase 3 sign-off that the STAL / one-to-one assigners
(:mod:`open_yolos.assign`) and the dual-branch loss
(:class:`~open_yolos.losses.dual_loss.DualBranchLoss`) compose into a stack that a
plain optimizer can drive downhill. The detection head (WP-022) lands
concurrently and is deliberately *not* used here: instead of a network we
optimize two learnable prediction tensors directly, so a failure localizes to
the assignment / loss stack and nothing else.

The learnable predictions are:

- ``logits`` — a ``(B, A, C)`` :class:`~torch.nn.Parameter` of raw class logits,
  fed to :class:`DualBranchLoss` as both branches' logits; and
- ``box_raw`` — a ``(B, A, 4)`` :class:`~torch.nn.Parameter` mapped to valid
  ``xyxy`` boxes by :func:`_decode_boxes`.

Box parametrization (anchor-centred ltrb). ``box_raw`` holds four raw scalars per
anchor that :func:`torch.nn.functional.softplus` turns into non-negative
``(left, top, right, bottom)`` distances (scaled by the anchor's stride so the
initial extents match the level's footprint). The decoded box is
``(ax - left, ay - top, ax + right, ay + bottom)`` around the anchor centre
``(ax, ay)``. Two properties make this the right stand-in for a head: the box is
always valid (``x2 >= x1`` and ``y2 >= y1`` because every distance is
non-negative), and because a positive anchor's centre lies inside its assigned
ground truth (the STAL candidate filter guarantees it), that ground-truth box is
*exactly* representable — so a perfect overfit is reachable, not merely
approachable.

The 200-step run is shared: a module-scoped fixture runs it once and records the
per-step loss trajectory plus a single finiteness flag (every loss component and
every gradient finite at every step), which the monotonic-decrease and
no-NaN/Inf tests then read without re-running.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from open_yolos.assign import make_anchor_points
from open_yolos.data.coco import CocoDetectionDataset
from open_yolos.data.letterbox import Letterbox
from open_yolos.losses.dual_loss import DualBranchLoss, DualLossOutput

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from open_yolos.data.targets import Targets

#: Images per overfit batch (kept small so the 200-step run stays a few seconds).
_BATCH_SIZE = 2
#: Square letterbox side; divisible by every stride for an integer anchor grid.
_IMG_SIZE = 160
#: Feature-level strides; at 160 px this yields a 20x20 + 10x10 + 5x5 = 525 grid.
_STRIDES = (8, 16, 32)
#: Adam steps in the shared overfit run.
_STEPS = 200
#: Adam learning rate; large because the free per-anchor predictions overfit fast.
_LR = 0.1
#: Window length for the smoothed-trajectory checkpoints.
_WINDOW = 20
#: Checkpoint window start indices (early -> late); each window's mean must fall.
_CHECKPOINTS = (0, 60, 120, 180)
#: Required final/initial loss ratio: the overfit must at least halve the loss.
_MAX_FINAL_RATIO = 0.5


@dataclass(frozen=True)
class _Batch:
    """A ready-to-optimize fixture batch: anchors plus padded ground truth.

    Attributes:
        anchor_points: ``(A, 2)`` anchor-centre ``(x, y)`` pixels.
        strides: ``(A,)`` owning-level stride per anchor.
        gt_boxes: ``(B, N, 4)`` padded ``xyxy`` ground-truth boxes.
        gt_labels: ``(B, N)`` padded int64 class ids.
        gt_mask: ``(B, N)`` bool marking real ground truths.
        num_classes: Class count ``C`` of the source dataset.
    """

    anchor_points: Tensor
    strides: Tensor
    gt_boxes: Tensor
    gt_labels: Tensor
    gt_mask: Tensor
    num_classes: int


@dataclass(frozen=True)
class _Trajectory:
    """Result of the shared overfit run.

    Attributes:
        totals: Per-step total loss (length ``_STEPS``).
        all_finite: ``True`` iff every loss component and gradient stayed finite
            at every step.
    """

    totals: list[float]
    all_finite: bool


@dataclass(frozen=True)
class _StepResult:
    """One-backward diagnostics for the leaf-gradient and alpha-extreme checks.

    Attributes:
        components_finite: Every :class:`DualLossOutput` component is finite.
        grads_finite: Both leaf gradients are finite.
        logits_grad_norm: L2 norm of the ``logits`` gradient.
        box_grad_norm: L2 norm of the ``box_raw`` gradient.
    """

    components_finite: bool
    grads_finite: bool
    logits_grad_norm: float
    box_grad_norm: float


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed the global RNG before each test (the run itself is RNG-free)."""
    torch.manual_seed(0)
    yield


def _first_nonempty_targets(dataset: CocoDetectionDataset, count: int) -> list[Targets]:
    """Return the first ``count`` samples' targets that carry at least one box."""
    collected: list[Targets] = []
    for index in range(len(dataset)):
        _, targets = dataset[index]
        if targets.boxes.shape[0] > 0:
            collected.append(targets)
        if len(collected) == count:
            return collected
    raise AssertionError(f"fixture has fewer than {count} annotated images")


def _pad_targets(targets_list: list[Targets]) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a ragged list of targets into dense ``(B, N, *)`` boxes/labels/mask."""
    batch = len(targets_list)
    max_n = max(targets.boxes.shape[0] for targets in targets_list)
    gt_boxes = torch.zeros(batch, max_n, 4)
    gt_labels = torch.zeros(batch, max_n, dtype=torch.int64)
    gt_mask = torch.zeros(batch, max_n, dtype=torch.bool)
    for i, targets in enumerate(targets_list):
        n = targets.boxes.shape[0]
        gt_boxes[i, :n] = targets.boxes
        gt_labels[i, :n] = targets.labels
        gt_mask[i, :n] = True
    return gt_boxes, gt_labels, gt_mask


def _load_batch(fixture_dir: Path) -> _Batch:
    """Build the letterboxed overfit batch and its anchor grid from the fixture."""
    split = fixture_dir / "train"
    dataset = CocoDetectionDataset(split, split / "_annotations.coco.json", transforms=Letterbox(_IMG_SIZE))
    gt_boxes, gt_labels, gt_mask = _pad_targets(_first_nonempty_targets(dataset, _BATCH_SIZE))
    feature_sizes = [(_IMG_SIZE // stride, _IMG_SIZE // stride) for stride in _STRIDES]
    anchor_points, strides = make_anchor_points(feature_sizes, list(_STRIDES))
    return _Batch(
        anchor_points=anchor_points,
        strides=strides,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        gt_mask=gt_mask,
        num_classes=len(dataset.category_id_to_label),
    )


def _new_predictions(batch: _Batch) -> tuple[nn.Parameter, nn.Parameter]:
    """Create the two zero-initialized learnable prediction tensors for ``batch``."""
    anchors = batch.anchor_points.shape[0]
    logits = nn.Parameter(torch.zeros(_BATCH_SIZE, anchors, batch.num_classes))
    box_raw = nn.Parameter(torch.zeros(_BATCH_SIZE, anchors, 4))
    return logits, box_raw


def _decode_boxes(box_raw: Tensor, anchor_points: Tensor, strides: Tensor) -> Tensor:
    """Map raw box params to valid ``xyxy`` via anchor-centred, stride-scaled ltrb."""
    distances = F.softplus(box_raw) * strides.view(1, -1, 1)  # (B, A, 4), non-negative
    left, top, right, bottom = distances.unbind(-1)
    center_x = anchor_points[:, 0]
    center_y = anchor_points[:, 1]
    return torch.stack(
        (center_x - left, center_y - top, center_x + right, center_y + bottom),
        dim=-1,
    )


def _components(out: DualLossOutput) -> tuple[Tensor, ...]:
    """Return every scalar component of a dual-loss output for finiteness checks."""
    return (
        out.total,
        out.o2m.total,
        out.o2m.box,
        out.o2m.cls,
        out.o2m.l1,
        out.o2o.total,
        out.o2o.box,
        out.o2o.cls,
        out.o2o.l1,
    )


def _grad(param: Tensor) -> Tensor:
    """Return a leaf's populated gradient, asserting ``backward`` filled it in."""
    assert param.grad is not None, "expected a gradient after backward"
    return param.grad


def _step_is_finite(out: DualLossOutput, logits: Tensor, box_raw: Tensor) -> bool:
    """Whether every loss component and both post-backward gradients are finite."""
    if not all(bool(torch.isfinite(component).all()) for component in _components(out)):
        return False
    return bool(torch.isfinite(_grad(logits)).all()) and bool(torch.isfinite(_grad(box_raw)).all())


def _run_overfit(batch: _Batch) -> _Trajectory:
    """Overfit the learnable predictions on ``batch`` for ``_STEPS`` Adam steps."""
    logits, box_raw = _new_predictions(batch)
    loss_fn = DualBranchLoss()
    optimizer = torch.optim.Adam([logits, box_raw], lr=_LR)
    totals: list[float] = []
    all_finite = True
    for _ in range(_STEPS):
        optimizer.zero_grad()
        boxes = _decode_boxes(box_raw, batch.anchor_points, batch.strides)
        out = loss_fn(logits, boxes, logits, boxes, batch.anchor_points, batch.gt_boxes, batch.gt_labels, batch.gt_mask)
        out.total.backward()
        all_finite = all_finite and _step_is_finite(out, logits, box_raw)
        optimizer.step()
        totals.append(float(out.total.detach()))
    return _Trajectory(totals=totals, all_finite=all_finite)


def _one_step(batch: _Batch, alpha: float) -> _StepResult:
    """Run a single forward/backward at branch weight ``alpha`` and report diagnostics."""
    logits, box_raw = _new_predictions(batch)
    loss_fn = DualBranchLoss()
    loss_fn.alpha = alpha
    boxes = _decode_boxes(box_raw, batch.anchor_points, batch.strides)
    out = loss_fn(logits, boxes, logits, boxes, batch.anchor_points, batch.gt_boxes, batch.gt_labels, batch.gt_mask)
    out.total.backward()
    logits_grad, box_grad = _grad(logits), _grad(box_raw)
    components_finite = all(bool(torch.isfinite(component).all()) for component in _components(out))
    grads_finite = bool(torch.isfinite(logits_grad).all()) and bool(torch.isfinite(box_grad).all())
    return _StepResult(
        components_finite=components_finite,
        grads_finite=grads_finite,
        logits_grad_norm=float(logits_grad.norm()),
        box_grad_norm=float(box_grad.norm()),
    )


def _window_mean(totals: list[float], start: int) -> float:
    """Mean of the ``_WINDOW`` loss values beginning at ``start``."""
    window = totals[start : start + _WINDOW]
    return sum(window) / len(window)


@pytest.fixture(scope="module")
def overfit_batch(detseg_fixture_dir: Path) -> _Batch:
    """The shared letterboxed fixture batch (built once per module)."""
    return _load_batch(detseg_fixture_dir)


@pytest.fixture(scope="module")
def overfit_run(overfit_batch: _Batch) -> _Trajectory:
    """The shared 200-step overfit trajectory (run once per module)."""
    return _run_overfit(overfit_batch)


def test_monotonic_overfit(overfit_run: _Trajectory) -> None:
    """200 Adam steps at least halve the loss with a smoothed, monotone-down trend."""
    checkpoints = [_window_mean(overfit_run.totals, start) for start in _CHECKPOINTS]
    assert all(later < earlier for earlier, later in pairwise(checkpoints)), checkpoints
    assert checkpoints[-1] < _MAX_FINAL_RATIO * checkpoints[0], checkpoints


def test_no_nan_inf(overfit_run: _Trajectory) -> None:
    """Every loss value and every gradient across all 200 steps is finite."""
    assert overfit_run.all_finite
    assert all(torch.isfinite(torch.tensor(total)) for total in overfit_run.totals)


def test_all_leaf_grads_populated(overfit_batch: _Batch) -> None:
    """One backward populates non-zero gradients on both leaves; components are finite."""
    result = _one_step(overfit_batch, alpha=DualBranchLoss().alpha)
    assert result.components_finite
    assert result.grads_finite
    assert result.logits_grad_norm > 0.0
    assert result.box_grad_norm > 0.0


@pytest.mark.parametrize("alpha", [pytest.param(0.0, id="alpha-o2o-only"), pytest.param(1.0, id="alpha-o2m-only")])
def test_alpha_extremes_still_train(overfit_batch: _Batch, alpha: float) -> None:
    """Both branch-weight extremes yield finite loss components and finite gradients."""
    result = _one_step(overfit_batch, alpha=alpha)
    assert result.components_finite
    assert result.grads_finite
    assert result.logits_grad_norm > 0.0
    assert result.box_grad_norm > 0.0
