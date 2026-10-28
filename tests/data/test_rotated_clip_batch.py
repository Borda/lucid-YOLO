# SPDX-License-Identifier: Apache-2.0
"""Batched clipping keeps the scalar clipper's arithmetic and ring order."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch
from markers import requires_cuda
from torch import Tensor

import lucid_yolo.data.rotated_aug as aug
from lucid_yolo.data.rotated_geom import canonicalize, rboxes_to_polygons
from lucid_yolo.data.tiling import _clip_to_window, _clip_to_window_batch


def _bits_equal(actual: Tensor, expected: Tensor) -> None:
    """Compare values at storage-bit precision, including signed zero.

    Examples:
        >>> _bits_equal(torch.tensor([0.0]), torch.tensor([0.0]))
    """
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _scalar_reference(rboxes: Tensor, height: float, width: float) -> tuple[Tensor, Tensor]:
    """The existing per-box path is independent of the batched clipper.

    Examples:
        >>> fitted, envelope = _scalar_reference(torch.empty(0, 5), 64.0, 64.0)
        >>> fitted.shape, envelope.shape
        (torch.Size([0, 5]), torch.Size([0, 4]))
    """
    corners = rboxes_to_polygons(rboxes)
    lower, upper = corners.amin(dim=1), corners.amax(dim=1)
    inside = (lower[:, 0] >= 0) & (lower[:, 1] >= 0) & (upper[:, 0] <= width) & (upper[:, 1] <= height)
    clipped, envelopes = canonicalize(rboxes), torch.cat([lower, upper], dim=1)
    for index in (~inside).nonzero().flatten().tolist():
        fitted, envelope = aug._clip_one(corners[index].double(), clipped[index].double(), height, width)
        clipped[index], envelopes[index] = fitted.to(clipped), envelope.to(envelopes)
    return clipped, envelopes


@pytest.mark.parametrize("count", [0, 1, 2, 7, 40])
@pytest.mark.parametrize("size", [-5.0, 0.0, 1.0, 64.0])
def test_mixed_rotated_boxes_match_scalar_path_bits(count: int, size: float) -> None:
    """Mixed empty, partial and interior rings retain scalar values and input ownership."""
    generator = torch.Generator().manual_seed(907)
    boxes = torch.rand(count, 5, generator=generator)
    boxes[:, :2] = boxes[:, :2] * 200 - 50
    boxes[:, 2:4] *= 150
    boxes[:, 4] = boxes[:, 4] * 12 - 6
    boxes = boxes.T.contiguous().T
    original = boxes.clone()
    expected = _scalar_reference(boxes, size, size + 5)
    actual = aug.clip_rboxes_to_canvas(boxes, size, size + 5)
    for result, reference in zip(actual, expected, strict=True):
        _bits_equal(result, reference)
    _bits_equal(boxes, original)


def test_clipping_regroups_changed_vertex_counts_without_reordering_rings() -> None:
    """A diamond becomes an octagon while a disjoint ring empties and an interior ring stays whole."""
    boxes = torch.tensor(
        [
            [32.0, 32.0, 80.0, 80.0, math.pi / 4],
            [-20.0, 20.0, 8.0, 4.0, 0.0],
            [20.0, 20.0, 8.0, 4.0, 0.0],
            [65.0, 32.0, 10.0, 20.0, 0.0],
        ]
    )
    rings = rboxes_to_polygons(boxes).double()
    actual = _clip_to_window_batch(rings, 0.0, 0.0, 64.0, 64.0)
    assert [len(part) for part in actual] == [8, 0, 4, 4]
    for part, ring in zip(actual, rings, strict=True):
        _bits_equal(part, _clip_to_window(ring, 0.0, 0.0, 64.0, 64.0))


def test_multiple_cpu_clips_share_clipping_and_refits() -> None:
    """The real entry point batches crossing boxes and leaves interior coordinates untouched."""
    boxes = torch.tensor([[0.0, 0.0, 8.0, 4.0, 0.2], [64.0, 64.0, 8.0, 4.0, 0.3], [20.0, 20.0, 8.0, 4.0, 0.0]])
    with patch.object(aug, "_clip_one", wraps=aug._clip_one) as scalar:
        actual = aug.clip_rboxes_to_canvas(boxes, 64.0, 64.0)
    assert scalar.call_count == 0
    for result, reference in zip(actual, _scalar_reference(boxes, 64.0, 64.0), strict=True):
        _bits_equal(result, reference)
    _bits_equal(actual[0][2], canonicalize(boxes)[2])


@pytest.mark.parametrize("kind", ["single", "double", "gradient", "nonfinite", "infinite-bound"])
def test_unsupported_or_single_clips_keep_scalar_path(kind: str) -> None:
    """Batching is gated to finite float32 CPU collections without gradients."""
    boxes = torch.tensor([[0.0, 0.0, 8.0, 4.0, 0.2], [64.0, 64.0, 8.0, 4.0, 0.3]])
    height = 64.0
    if kind == "single":
        boxes = boxes[:1]
    elif kind == "double":
        boxes = boxes.double()
    elif kind == "gradient":
        boxes.requires_grad_()
    elif kind == "nonfinite":
        boxes[0, 0] = float("nan")
    elif kind == "infinite-bound":
        height = float("inf")
    with patch.object(aug, "_clip_to_window_batch", side_effect=AssertionError("ineligible batch")):
        actual = aug.clip_rboxes_to_canvas(boxes, height, 64.0)
    expected = _scalar_reference(boxes, height, 64.0)
    for result, reference in zip(actual, expected, strict=True):
        _bits_equal(result, reference)


def test_gradient_fallback_keeps_scalar_derivatives() -> None:
    """Gradient-bearing inputs keep the original graph as well as its values."""
    boxes = torch.tensor([[0.0, 0.0, 8.0, 4.0, 0.2], [64.0, 64.0, 8.0, 4.0, 0.3]])
    actual_input, reference_input = boxes.clone().requires_grad_(), boxes.clone().requires_grad_()
    actual = aug.clip_rboxes_to_canvas(actual_input, 64.0, 64.0)
    expected = _scalar_reference(reference_input, 64.0, 64.0)
    actual_gradient = torch.autograd.grad(sum(value.sum() for value in actual), actual_input)[0]
    expected_gradient = torch.autograd.grad(sum(value.sum() for value in expected), reference_input)[0]
    _bits_equal(actual_gradient, expected_gradient)


@requires_cuda
def test_cuda_boxes_keep_scalar_clipping() -> None:
    """Accelerator tensors retain the pre-existing scalar route and output bits."""
    boxes = torch.tensor([[0.0, 0.0, 8.0, 4.0, 0.2], [64.0, 64.0, 8.0, 4.0, 0.3]], device="cuda")
    with patch.object(aug, "_clip_to_window_batch", side_effect=AssertionError("CPU-only batch")):
        actual = aug.clip_rboxes_to_canvas(boxes, 64.0, 64.0)
    for result, reference in zip(actual, _scalar_reference(boxes, 64.0, 64.0), strict=True):
        _bits_equal(result, reference)
