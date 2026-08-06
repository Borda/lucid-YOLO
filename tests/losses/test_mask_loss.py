# SPDX-License-Identifier: Apache-2.0
"""Unit gates for the WP-051 instance mask loss (A16, A36).

Pins the three properties the term is defined by — the box crop, the box-area
normalization, and the degenerate-box clamp — plus the no-positives and gradient
contracts. Expected values are hand-derived in the test docstrings: with all
logits at zero the per-pixel BCE is exactly ``ln 2`` regardless of the target, so
every scenario below reduces to counting cropped pixels against a box area.
"""

from __future__ import annotations

import math

import torch

from lucid_yolo.losses import instance_mask_loss

_LN2 = math.log(2.0)


def test_full_grid_box_gives_per_pixel_bce() -> None:
    """A whole-grid box with zero logits yields exactly ln 2, pinning the normalizer.

    Catches a normalizer that divides by pixel count, by the cropped-pixel count
    computed some other way, or not at all: with area == H*W == the number of
    cropped pixels, only the box-area convention lands on ln 2 for a grid whose
    height and width differ.
    """
    logits = torch.zeros(1, 4, 6)
    targets = torch.ones(1, 4, 6)
    boxes = torch.tensor([[0.0, 0.0, 6.0, 4.0]])

    loss = instance_mask_loss(logits, targets, boxes)

    assert math.isclose(loss.item(), _LN2, abs_tol=1e-6)


def test_pixels_outside_the_box_are_ignored() -> None:
    """Wildly wrong logits outside the box leave the loss numerically unchanged.

    This is the test that fails if the crop is dropped: the out-of-box logits are
    set to -50 against a target of 1, which would add ~50 nats per pixel to an
    uncropped sum.
    """
    targets = torch.ones(1, 8, 8)
    boxes = torch.tensor([[2.0, 2.0, 6.0, 6.0]])
    clean = torch.zeros(1, 8, 8)
    polluted = torch.full((1, 8, 8), -50.0)
    polluted[0, 2:6, 2:6] = 0.0

    clean_loss = instance_mask_loss(clean, targets, boxes)
    polluted_loss = instance_mask_loss(polluted, targets, boxes)

    assert torch.equal(clean_loss, polluted_loss), "pixels outside the box must not enter the loss"


def test_area_normalization_equalizes_box_sizes() -> None:
    """Two instances with equal per-pixel error contribute equally despite a 16x area gap.

    Catches a missing or wrong normalizer: instance 0's box covers 4 pixels and
    instance 1's covers 64, so an unnormalized sum would weight them 1:16, and
    the mean would be 34 ln 2 instead of ln 2.
    """
    logits = torch.zeros(2, 8, 8)
    targets = torch.ones(2, 8, 8)
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 8.0, 8.0]])

    both = instance_mask_loss(logits, targets, boxes)
    small = instance_mask_loss(logits[:1], targets[:1], boxes[:1])
    large = instance_mask_loss(logits[1:], targets[1:], boxes[1:])

    assert math.isclose(small.item(), _LN2, abs_tol=1e-6)
    assert math.isclose(large.item(), _LN2, abs_tol=1e-6)
    assert math.isclose(both.item(), _LN2, abs_tol=1e-6)


def test_degenerate_box_stays_finite() -> None:
    """A zero-width and a zero-height box give a finite loss, no NaN and no inf.

    Without the A36 clamp the box area is 0 and the per-instance division would
    produce NaN (0/0) or inf, poisoning the whole batch mean through the sum.
    """
    logits = torch.zeros(2, 6, 6)
    targets = torch.ones(2, 6, 6)
    boxes = torch.tensor([[1.0, 1.0, 1.0, 4.0], [1.0, 2.0, 5.0, 2.0]])

    loss = instance_mask_loss(logits, targets, boxes)

    assert torch.isfinite(loss).all(), "a degenerate box must not divide by zero area"


def test_no_instances_gives_finite_zero() -> None:
    """An empty positive set returns a finite zero scalar, not a NaN mean.

    Catches a plain ``.mean()`` over an empty batch, which is 0/0: on an image
    with no positives that NaN would propagate into every parameter gradient.
    """
    logits = torch.zeros(0, 5, 5)
    targets = torch.zeros(0, 5, 5)
    boxes = torch.zeros(0, 4)

    loss = instance_mask_loss(logits, targets, boxes)

    assert loss.shape == ()
    assert torch.isfinite(loss).all()
    assert loss.item() == 0.0


def test_gradients_reach_only_the_cropped_region() -> None:
    """backward() leaves non-zero grad inside the box and exactly zero outside it.

    Catches a crop applied to the forward value alone (e.g. via indexing that
    detaches, or a crop that leaks gradient through a soft weighting): the
    out-of-box logits must be entirely free of the objective.
    """
    logits = torch.zeros(1, 8, 8, requires_grad=True)
    targets = torch.ones(1, 8, 8)
    boxes = torch.tensor([[2.0, 3.0, 6.0, 7.0]])

    instance_mask_loss(logits, targets, boxes).backward()

    assert logits.grad is not None
    inside = logits.grad[0, 3:7, 2:6]
    outside_mask = torch.ones(8, 8, dtype=torch.bool)
    outside_mask[3:7, 2:6] = False
    assert (inside != 0.0).all(), "every cropped pixel must receive gradient"
    assert torch.equal(logits.grad[0][outside_mask], torch.zeros(int(outside_mask.sum())))


def test_sub_pixel_box_selects_nothing() -> None:
    """A box falling strictly between pixel centres selects no pixel at all.

    Pins the half-open membership test: a crop that rounded the box outwards, or
    that used ceil/floor on the edges, would claim pixel 0 and report a non-zero
    loss where the sampled-point convention gives exactly zero.
    """
    logits = torch.zeros(1, 4, 4)
    targets = torch.ones(1, 4, 4)
    boxes = torch.tensor([[0.6, 0.6, 0.9, 0.9]])  # inside pixel 0, misses its centre at 0.5

    loss = instance_mask_loss(logits, targets, boxes)

    assert loss.item() == 0.0


def test_upper_edge_is_exclusive() -> None:
    """A centre lying exactly on the box's far edge belongs to neither of two abutting boxes.

    The crop is documented as half-open, ``[x1, x2) x [y1, y2)``, so that two
    boxes sharing an edge cannot both claim the same pixel. With centres at half
    integers that claim is only observable when an edge lands exactly on one, so
    the boxes here abut at 2.5: the lower box must stop short of that centre and
    the upper box must own it.
    """
    logits = torch.zeros(2, 4, 4, requires_grad=True)
    targets = torch.ones(2, 4, 4)
    boxes = torch.tensor([[0.0, 0.0, 2.5, 4.0], [2.5, 0.0, 4.0, 4.0]])

    instance_mask_loss(logits, targets, boxes).backward()

    assert logits.grad is not None
    lower_columns = (logits.grad[0] != 0.0).any(dim=0)
    upper_columns = (logits.grad[1] != 0.0).any(dim=0)
    assert torch.equal(lower_columns, torch.tensor([True, True, False, False]))
    assert torch.equal(upper_columns, torch.tensor([False, False, True, True]))


def test_pixel_centres_decide_box_membership() -> None:
    """Fractional box edges select the centre-sampled window, not the corner-sampled one.

    Pins the A11 ``+0.5`` centring where it is actually observable. On ``[0.4,
    2.4)`` the centres 0.5 and 1.5 fall inside while 2.5 does not, so the window
    is index 0..1; sampling pixel corners instead would put 0.0 outside and 1.0
    and 2.0 inside, giving window 1..2. Both windows are 2x2, so a shape or
    pixel-count assertion cannot tell them apart — only their position can, which
    is read off the gradient support.
    """
    logits = torch.zeros(1, 4, 4, requires_grad=True)
    targets = torch.ones(1, 4, 4)
    boxes = torch.tensor([[0.4, 0.4, 2.4, 2.4]])

    instance_mask_loss(logits, targets, boxes).backward()

    assert logits.grad is not None
    selected = logits.grad[0] != 0.0
    expected = torch.zeros(4, 4, dtype=torch.bool)
    expected[0:2, 0:2] = True
    assert torch.equal(selected, expected), "crop must sample pixel centres, not pixel corners"
