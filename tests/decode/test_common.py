# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the shared decode padders in :mod:`lucid_yolo.decode.common`.

:func:`~lucid_yolo.decode.common.pad_detections` and its index-side twin
:func:`~lucid_yolo.decode.common.pad_anchor_indices` exist to give both decode paths
one fixed output length, which is what lets a traced graph carry a static output
shape and what lets one evaluator consume either path. These tests pin that length
from both sides: the short input the padders were written for, and the over-length
input they used to return unchanged — a pass-through that quietly turned the fixed
shape into a variable one at whichever caller stopped capping. The two are exercised
against the same row counts, since their whole contract is that row ``n`` of the
detections and entry ``n`` of the indices survive or vanish together.

The same module holds the one per-anchor gather and the point-set letterbox inverse,
both pinned here against the forms they replaced, and the detection cap and image side
the ``models`` package cannot import, pinned equal to its literals.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.decode.common import (
    COCO_IMG_SIZE,
    DET_WIDTH,
    MAX_DETECTIONS,
    PAD_ANCHOR_INDEX,
    gather_by_anchor,
    pad_anchor_indices,
    pad_detections,
    points_to_letterboxed_original,
)
from lucid_yolo.models.build import count_flops
from lucid_yolo.models.heads.detect import o2o_topk_with_indices
from lucid_yolo.models.heads.obb import o2o_rotated_topk_with_indices


@pytest.mark.parametrize(
    ("kept", "max_det"),
    [
        pytest.param(2, 4, id="short-input-is-padded"),
        pytest.param(4, 4, id="exact-input-is-unchanged"),
        pytest.param(9, 4, id="over-length-input-is-truncated"),
    ],
)
def test_detections_come_out_at_exactly_max_det(kept: int, max_det: int) -> None:
    """The detection axis is ``max_det`` long whatever length went in.

    The fixed length is the export-friendly contract the module docstring states, so
    it has to hold on all three sides of ``max_det``, not just below it. The
    over-length case is the one that regressed: nine ranked rows capped at four used
    to come back as nine, and a caller that had stopped capping would have exported a
    graph whose output shape depended on how many detections the image happened to
    produce.
    """
    detections = torch.ones(1, kept, DET_WIDTH)

    padded = pad_detections(detections, max_det)

    assert padded.shape == (1, max_det, DET_WIDTH)


@pytest.mark.parametrize(
    ("kept", "max_det"),
    [
        pytest.param(2, 4, id="short-input-is-padded"),
        pytest.param(4, 4, id="exact-input-is-unchanged"),
        pytest.param(9, 4, id="over-length-input-is-truncated"),
    ],
)
def test_anchor_indices_come_out_at_exactly_max_det(kept: int, max_det: int) -> None:
    """The anchor-index axis is ``max_det`` long whatever length went in.

    The indices must track the detections exactly — the segmentation decode gathers
    each detection's mask coefficients by them — so this twin is held to the same
    length rule on the same three cases. An index axis that kept its surplus entries
    while the detections beside it were cut is precisely the mismatch that pairs a
    box with another anchor's coefficients.
    """
    indices = torch.arange(kept).unsqueeze(0)

    padded = pad_anchor_indices(indices, max_det)

    assert padded.shape == (1, max_det)


def test_truncation_keeps_highest_ranked_rows() -> None:
    """Truncation drops the tail, so the surviving rows are the top-ranked ones.

    Detections reach the padders score-descending, so cutting to ``max_det`` has to
    cut from the bottom for the result to mean "the ``max_det`` best". Cutting from
    the front, or reordering, would keep the shape right and the content wrong — the
    failure a shape-only assertion cannot see.
    """
    detections = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1).expand(1, 6, DET_WIDTH)

    truncated = pad_detections(detections, max_det=2)

    assert truncated[0, :, 0].tolist() == [0.0, 1.0]


def test_padders_agree_on_length_for_one_over_length_batch() -> None:
    """A detection and its anchor index are dropped together, never one without the other.

    The two functions are called back to back by every decoder with the same
    ``max_det``, and their alignment is a per-row property, not just a per-shape one.
    This drives both from the same over-length count and asserts the surviving
    lengths match, which is the invariant the segmentation decode depends on.
    """
    detections = torch.ones(1, 9, DET_WIDTH)
    indices = torch.arange(9).unsqueeze(0)

    padded_detections = pad_detections(detections, max_det=4)
    padded_indices = pad_anchor_indices(indices, max_det=4)

    assert padded_detections.shape[-2] == padded_indices.shape[-1]


def test_short_indices_filled_with_padding_sentinel() -> None:
    """A shortfall is filled with :data:`PAD_ANCHOR_INDEX`, not with a usable anchor row.

    The sentinel is negative precisely so a padding row cannot be mistaken for a real
    one, and so it cannot silently index the last anchor if it ever reached a gather.
    Filling with zeros instead would produce a plausible mask for every padding row.
    """
    padded = pad_anchor_indices(torch.tensor([[3, 7]]), max_det=4)

    assert padded[0, 2:].tolist() == [PAD_ANCHOR_INDEX, PAD_ANCHOR_INDEX]


class TestGatherByAnchor:
    """Tests for :func:`~lucid_yolo.decode.common.gather_by_anchor`, the one per-anchor gather."""

    @pytest.mark.parametrize("trailing", [pytest.param((3,), id="coefficients"), pytest.param((4, 2), id="points")])
    def test_real_rows_bit_equal_to_bare_gather(self, trailing: tuple[int, ...]) -> None:
        """With no padding row, the clamp and the product by one leave a bare gather's values untouched.

        This is what lets the export graphs, which used to gather mask coefficients with
        no padding rule at all, call the shared gather without moving a single output bit
        on any canvas that reaches the detection cap.
        """
        values = torch.randn(2, 5, *trailing, generator=torch.Generator().manual_seed(1))
        anchors = torch.tensor([[4, 0, 2], [1, 1, 3]])
        index = anchors.reshape(2, 3, *([1] * len(trailing))).expand(-1, -1, *trailing)

        gathered = gather_by_anchor(values, anchors)

        assert torch.equal(gathered, values.gather(1, index))

    @pytest.mark.parametrize("trailing", [pytest.param((3,), id="coefficients"), pytest.param((4, 2), id="points")])
    def test_padding_rows_zeroed_not_last_anchor(self, trailing: tuple[int, ...]) -> None:
        """A padding index yields all zeros rather than the last anchor's values.

        ``-1`` used directly as a gather index would read the last anchor: a plausible mask
        or pose for a row that is not a detection.
        """
        values = torch.ones(1, 5, *trailing)

        gathered = gather_by_anchor(values, torch.tensor([[2, PAD_ANCHOR_INDEX]]))

        assert torch.equal(gathered[0, 0], torch.ones(trailing))
        assert torch.equal(gathered[0, 1], torch.zeros(trailing))


class TestPointsToLetterboxedOriginal:
    """Tests for :func:`~lucid_yolo.decode.common.points_to_letterboxed_original`."""

    def test_matches_letterbox_inverse_point_for_point(self) -> None:
        """Every point takes the exact inverse the letterbox itself defines, whatever the leading shape.

        The function exists so the single-image and the batch keypoint paths stop each
        writing their own fold-map-unfold around this call; folding must not reorder points.
        """
        points = torch.rand(3, 5, 2, generator=torch.Generator().manual_seed(2)) * 64.0
        letterbox = Letterbox(64, allow_upscale=False)

        mapped = points_to_letterboxed_original(points, (40, 100), (64, 64), allow_upscale=False)

        expected = letterbox.inverse_map(points.reshape(-1, 2), (40, 100), (64, 64)).reshape(3, 5, 2)
        assert torch.equal(mapped, expected)

    def test_empty_input_keeps_point_axis(self) -> None:
        """An image with no detection comes back as an empty stack that still states ``K``."""
        mapped = points_to_letterboxed_original(torch.zeros(0, 17, 2), (40, 100), (64, 64))

        assert mapped.shape == (0, 17, 2)


def test_detection_cap_matches_model_ranking_defaults() -> None:
    """The model-side rankings default to the same cap the decoders import.

    ``models`` sits below ``decode`` in the import graph, so the two ranking helpers there
    cannot import :data:`MAX_DETECTIONS` without a cycle and carry the number as a literal;
    this pin is what keeps the three equal.
    """
    defaults = {
        parameters["k"].default
        for parameters in (
            inspect.signature(o2o_topk_with_indices).parameters,
            inspect.signature(o2o_rotated_topk_with_indices).parameters,
        )
    }

    assert defaults == {MAX_DETECTIONS}


def test_flop_measurement_side_matches_coco_img_size() -> None:
    """The FLOP counter measures at the COCO tiers' letterbox side.

    Same layering reason as the detection cap: :mod:`lucid_yolo.models.build` cannot import
    :data:`COCO_IMG_SIZE`, so its default is pinned equal here.
    """
    assert inspect.signature(count_flops).parameters["img_size"].default == COCO_IMG_SIZE
