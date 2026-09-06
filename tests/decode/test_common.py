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
"""

from __future__ import annotations

import pytest
import torch

from lucid_yolo.decode.common import DET_WIDTH, PAD_ANCHOR_INDEX, pad_anchor_indices, pad_detections


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


def test_truncation_keeps_the_highest_ranked_rows() -> None:
    """Truncation drops the tail, so the surviving rows are the top-ranked ones.

    Detections reach the padders score-descending, so cutting to ``max_det`` has to
    cut from the bottom for the result to mean "the ``max_det`` best". Cutting from
    the front, or reordering, would keep the shape right and the content wrong — the
    failure a shape-only assertion cannot see.
    """
    detections = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1).expand(1, 6, DET_WIDTH)

    truncated = pad_detections(detections, max_det=2)

    assert truncated[0, :, 0].tolist() == [0.0, 1.0]


def test_the_padders_agree_on_length_for_one_over_length_batch() -> None:
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


def test_short_indices_are_filled_with_the_padding_sentinel() -> None:
    """A shortfall is filled with :data:`PAD_ANCHOR_INDEX`, not with a usable anchor row.

    The sentinel is negative precisely so a padding row cannot be mistaken for a real
    one, and so it cannot silently index the last anchor if it ever reached a gather.
    Filling with zeros instead would produce a plausible mask for every padding row.
    """
    padded = pad_anchor_indices(torch.tensor([[3, 7]]), max_det=4)

    assert padded[0, 2:].tolist() == [PAD_ANCHOR_INDEX, PAD_ANCHOR_INDEX]
