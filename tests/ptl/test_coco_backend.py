# SPDX-License-Identifier: Apache-2.0
"""Tests for the hoisted-score COCO backend (WP-165).

The subclass overrides a private ``torchmetrics`` method, so the load-bearing test
is not that it is fast but that it is *indistinguishable*: every case here compares
its document against the stock backend's on the installed version of the library,
so a change upstream that this override cannot absorb fails the gate rather than
silently moving a validation metric.

Three shapes matter. Boxes are the detection path. Masks are the segmentation path,
and they carry upstream's one skip -- an image with no masks contributes no
annotations when the document has no boxes either -- which the override has to
reproduce to keep the scores aligned with the annotations. Ground truth passes
``scores=None`` and must come back with no ``score`` key at all.

Two later concerns share this file because they are about how the epoch metrics are
*built* rather than about what the training loop does with them: that the private
upstream surface the override depends on is checked before it is used (audit M-07),
and that both metrics the module constructs integrate on the report path's exact
recall grid rather than torchmetrics' float32 default (audit M-03).
"""

from __future__ import annotations

import pytest
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.helpers import CocoBackend

from lucid_yolo.eval.coco_eval import COCO_RECALL_GRID
from lucid_yolo.ptl.coco_backend import _HoistedScoreBackend, build_mean_average_precision
from lucid_yolo.ptl.module import DetectionLitModule

#: The backend both classes are constructed under, matching what the module builds.
_BACKEND = "faster_coco_eval"


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed torch before each test so the random detections are reproducible."""
    torch.manual_seed(0)


def _boxes(count: int) -> torch.Tensor:
    """Build ``count`` valid ``xywh`` boxes.

    Examples:
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> _boxes(3).shape
        torch.Size([3, 4])
    """
    top_left = torch.rand(count, 2) * 100.0
    size = torch.rand(count, 2) * 40.0 + 5.0
    return torch.cat([top_left, size], dim=1)


def _rle_masks(count: int) -> list[tuple[list[int], bytes]]:
    """Build ``count`` RLE masks on an 8x8 canvas, in the pair form the backend consumes.

    Examples:
        >>> size, counts = _rle_masks(1)[0]
        >>> size
        [8, 8]
    """
    masks = []
    for index in range(count):
        dense = torch.zeros(8, 8, dtype=torch.uint8)
        dense[index % 8, :] = 1
        encoded = CocoBackend(_BACKEND).mask_utils.encode(dense.numpy().astype("uint8", order="F"))
        masks.append([encoded["size"], encoded["counts"]])
    return masks


class TestHoistedScoreBackend:
    """The override must produce the stock backend's document, field for field."""

    def test_a_box_document_matches_the_stock_backend(self) -> None:
        """Detections converted with hoisted scores equal the per-annotation conversion.

        The detection path, at the shape validation actually feeds it: several
        images, several detections each, every annotation carrying a score.
        """
        labels = [torch.tensor([0, 1, 1]), torch.tensor([2, 0])]
        boxes = [_boxes(3), _boxes(2)]
        scores = [torch.tensor([0.9, 0.5, 0.1]), torch.tensor([0.75, 0.25])]

        stock = CocoBackend(_BACKEND)._get_coco_format(labels=labels, all_labels=[0, 1, 2], boxes=boxes, scores=scores)
        hoisted = _HoistedScoreBackend(_BACKEND)._get_coco_format(
            labels=labels, all_labels=[0, 1, 2], boxes=boxes, scores=scores
        )

        assert hoisted == stock

    def test_a_mask_document_with_an_empty_image_matches_the_stock_backend(self) -> None:
        """Upstream's skip of a maskless image is reproduced, so scores stay aligned.

        A segmentation document carries no boxes, and upstream then contributes no
        annotations for an image whose mask list is empty. An override that
        flattened every image's scores regardless would hand image three's scores
        to image two's annotations -- plausible masks with the wrong confidences,
        which no metric value reveals -- so the empty image sits in the middle
        rather than at the end, where an off-by-one would still line up.
        """
        labels = [torch.tensor([0, 1]), torch.tensor([], dtype=torch.long), torch.tensor([1])]
        masks = [_rle_masks(2), [], _rle_masks(1)]
        scores = [torch.tensor([0.8, 0.4]), torch.tensor([]), torch.tensor([0.6])]

        stock = CocoBackend(_BACKEND)._get_coco_format(
            labels=labels, all_labels=[0, 1], masks=masks, scores=scores, iou_type=("segm",)
        )
        hoisted = _HoistedScoreBackend(_BACKEND)._get_coco_format(
            labels=labels, all_labels=[0, 1], masks=masks, scores=scores, iou_type=("segm",)
        )

        assert hoisted == stock
        # The skipped image contributes nothing, so image three's score follows image one's.
        assert [annotation["score"] for annotation in hoisted["annotations"]] == pytest.approx([0.8, 0.4, 0.6])

    def test_a_ground_truth_document_carries_no_scores(self) -> None:
        """With ``scores=None`` the override returns upstream's document untouched.

        Ground truth is converted by the same method with no scores at all, so the
        override must not invent a ``score`` key that the stock backend omits.
        """
        labels = [torch.tensor([0, 1])]
        boxes = [_boxes(2)]

        stock = CocoBackend(_BACKEND)._get_coco_format(labels=labels, all_labels=[0, 1], boxes=boxes)
        hoisted = _HoistedScoreBackend(_BACKEND)._get_coco_format(labels=labels, all_labels=[0, 1], boxes=boxes)

        assert hoisted == stock
        assert all("score" not in annotation for annotation in hoisted["annotations"])


def test_the_metric_reports_the_value_the_stock_metric_reports() -> None:
    """End to end, the swapped backend leaves every mAP figure unchanged.

    The document comparisons above are the precise guard; this one closes the gap
    between "the document is equal" and "the number the training run logs is
    equal", which is what the swap is actually promising.
    """
    preds = [{"boxes": _boxes(5), "scores": torch.rand(5), "labels": torch.randint(0, 3, (5,))} for _ in range(4)]
    target = [{"boxes": _boxes(2), "labels": torch.randint(0, 3, (2,))} for _ in range(4)]
    stock = MeanAveragePrecision(backend=_BACKEND)
    hoisted = build_mean_average_precision(backend=_BACKEND)
    stock.update(preds, target)
    hoisted.update(preds, target)

    stock_result, hoisted_result = stock.compute(), hoisted.compute()

    assert stock_result.keys() == hoisted_result.keys()
    assert all(torch.equal(stock_result[key], hoisted_result[key]) for key in stock_result)


class TestScoreTypeValidation:
    """The hoist must refuse exactly the scores upstream refuses (audit L-38)."""

    def test_an_integral_score_raises_upstream_s_message(self) -> None:
        """A non-float score tensor is rejected with the stock backend's own wording.

        Upstream reads each score individually and raises when the converted value
        is not a ``float``; the hoisted conversion reads a whole image at once, and
        an ``int`` element there once passed through a ``float()`` coercion. That
        made this override's document *accept* an input the stock one rejects --
        the one difference it promises never to make -- so the check travels with
        the hoist rather than being spent on it.
        """
        labels = [torch.tensor([0, 1])]
        boxes = [_boxes(2)]
        integral_scores = [torch.tensor([1, 0])]

        with pytest.raises(ValueError, match="expected value of type float") as hoisted_error:
            _HoistedScoreBackend(_BACKEND)._get_coco_format(
                labels=labels, all_labels=[0, 1], boxes=boxes, scores=integral_scores
            )

        with pytest.raises(ValueError) as stock_error:
            CocoBackend(_BACKEND)._get_coco_format(
                labels=labels, all_labels=[0, 1], boxes=boxes, scores=integral_scores
            )
        assert str(hoisted_error.value) == str(stock_error.value)

    def test_float_scores_still_reach_every_annotation(self) -> None:
        """The normal path is unchanged: float scores land on the annotations in order.

        The guard above sits inside the flattening loop, so this pins that it
        rejects only the type it is aimed at rather than narrowing what the
        conversion accepts.
        """
        labels = [torch.tensor([0, 1]), torch.tensor([1])]
        boxes = [_boxes(2), _boxes(1)]
        scores = [torch.tensor([0.9, 0.5]), torch.tensor([0.25])]

        document = _HoistedScoreBackend(_BACKEND)._get_coco_format(
            labels=labels, all_labels=[0, 1], boxes=boxes, scores=scores
        )

        assert [annotation["score"] for annotation in document["annotations"]] == pytest.approx([0.9, 0.5, 0.25])


class TestPrivateSurfaceCanary:
    """The factory refuses a ``torchmetrics`` whose private surface has moved (audit M-07)."""

    def test_a_missing_get_coco_format_fails_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Losing upstream's ``_get_coco_format`` raises here instead of silently unhooking.

        The override defines the method itself, so a rename upstream leaves the
        subclass defining something nobody calls: every conversion would run
        upstream's own path and the metric would keep reporting success. The
        ``pyproject.toml`` ceiling stops that version arriving by accident; this is
        what happens if the ceiling is lifted without reading the override.
        """
        monkeypatch.delattr(CocoBackend, "_get_coco_format")

        with pytest.raises(RuntimeError, match="no longer defines _get_coco_format"):
            build_mean_average_precision(backend=_BACKEND)


class TestEpochMetricRecallGrid:
    """The training loop's epoch metrics integrate on the report path's grid (audit M-03)."""

    def test_the_factory_forwards_rec_thresholds(self) -> None:
        """``rec_thresholds`` passes through the factory to the metric unchanged.

        The factory takes ``**kwargs``, so nothing structurally guarantees a
        constructor argument survives the backend swap; this pins that it does,
        which is what the two module assertions below rest on.
        """
        metric = build_mean_average_precision(backend=_BACKEND, rec_thresholds=list(COCO_RECALL_GRID))

        assert tuple(metric.rec_thresholds) == COCO_RECALL_GRID

    def test_the_box_metric_uses_the_exact_hundredths_grid(self) -> None:
        """``val/mAP`` integrates over :data:`COCO_RECALL_GRID`, not torchmetrics' float32 default.

        The acceptance path pins that grid (``coco_eval._new_metric``) precisely to
        remove a downward-only bias at 36 of the 101 recall points. Left unpinned
        here, the epoch figure and the acceptance figure would be summed over
        different recall points while both were logged as ``mAP``.
        """
        module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=2)

        assert tuple(module._val_map.rec_thresholds) == COCO_RECALL_GRID

    def test_the_mask_metric_uses_the_exact_hundredths_grid(self) -> None:
        """``val/segm_mAP`` is built on the same grid as the box metric.

        The mask metric is constructed separately -- different ``iou_type``, its own
        call -- so it is its own chance to drift from the grid the box metric and
        the report path share.
        """
        module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=2, task="segment")

        assert module._val_segm is not None
        assert tuple(module._val_segm.rec_thresholds) == COCO_RECALL_GRID
