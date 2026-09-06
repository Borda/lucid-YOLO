# SPDX-License-Identifier: Apache-2.0
"""Unit gates for the WP-053b segm evaluation (bbox+segm, both decode paths).

The instrument this file pins is a *pair* of numbers, and almost every way it can
break leaves one of them looking perfectly healthy:

- :func:`test_bbox_and_segm` (the DoD) — perfect predictions score 1.0 on both
  metrics, and then perturbing **only** the masks moves the segm number while the
  bbox number does not. A segm column silently recomputed from box geometry, or
  aliased onto the bbox column, passes the first half and fails the second;
- a prediction whose mask is empty scores no segm credit and raises nothing —
  the model's ordinary output for a detection whose mask probabilities never
  clear the A37 threshold;
- COCO's polygon and RLE encodings of one shape must decode to the *same* target
  mask, or every ``iscrowd`` region (always RLE) would be scored on a different
  grid from every ordinary instance (always polygons);
- masks and boxes are dropped by the **same** score filter, tested against a
  batch whose padding rows sit where an off-by-one or a "keep the first N" filter
  would survive undetected;
- a detection-only checkpoint still evaluates exactly as before — same 12 keys,
  no segm anywhere — and a segmentation checkpoint reports both metrics on both
  paths through the real :class:`~lucid_yolo.eval.coco_eval.DualPathEvaluator`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from faster_coco_eval import mask as coco_mask

from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.decode import NMSDecoder, TopKDecoder
from lucid_yolo.eval import (
    DualPathEvaluator,
    annotations_to_target,
    coco_eval,
    detections_to_predictions,
    evaluate_bbox_and_segm,
    evaluate_segm,
    letterboxed_batches,
    load_eval_annotations,
)
from lucid_yolo.eval.coco_eval import _METRIC_KEYS, _SEGM_PREFIX, _BatchGeometry, _StreamingScorer
from lucid_yolo.models.build import Segmenter
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

_SPLIT = "train"
_ANNOTATION = "_annotations.coco.json"
_CANVAS = 160  # 128px fixtures upscaled onto a 160px canvas: the inverse letterbox does real work.
_NUM_IMAGES = 3  # a few fixture images exercise the loop without being slow.
_NUM_CLASSES = 4  # the detseg fixture carries category ids 1..4.
_MAX_DET = 10  # a small per-image cap keeps the mask decode and RLE encode fast.
_CATEGORY = 5  # an arbitrary COCO category id shared by the hand-built cases.
_OTHER_CATEGORY = 6  # a second id: mask/box pairing is only observable across distinct labels.

#: The one hand-built instance: an ``xyxy`` box and the mask that exactly fills it.
_BOX = [10.0, 12.0, 30.0, 42.0]
_IMAGE_SIZE = (64, 64)


def _filled_mask(box: list[float], image_size: tuple[int, int] = _IMAGE_SIZE) -> Tensor:
    """Return a mask that is ``True`` exactly on ``box``'s pixels.

    Examples:
        >>> mask = _filled_mask([1.0, 1.0, 3.0, 2.0], image_size=(4, 4))
        >>> mask.shape
        torch.Size([1, 4, 4])
        >>> int(mask.sum())
        2
    """
    mask = torch.zeros((1, *image_size), dtype=torch.bool)
    x1, y1, x2, y2 = (int(value) for value in box)
    mask[0, y1:y2, x1:x2] = True
    return mask


def _prediction(box: list[float], mask: Tensor) -> dict[str, Tensor]:
    """Build a single-detection prediction dict at score 1.0.

    Examples:
        >>> pred = _prediction([1.0, 2.0, 3.0, 4.0], torch.zeros((1, 4, 4), dtype=torch.bool))
        >>> pred["boxes"].tolist()
        [[1.0, 2.0, 3.0, 4.0]]
        >>> pred["labels"].tolist()
        [5]
    """
    return {
        "boxes": torch.tensor([box], dtype=torch.float32),
        "scores": torch.tensor([1.0]),
        "labels": torch.tensor([_CATEGORY]),
        "masks": mask,
    }


def _target(box: list[float], mask: Tensor) -> dict[str, Tensor]:
    """Build a single-instance ground-truth dict.

    Examples:
        >>> tgt = _target([1.0, 2.0, 3.0, 4.0], torch.zeros((1, 4, 4), dtype=torch.bool))
        >>> sorted(tgt.keys())
        ['boxes', 'labels', 'masks']
    """
    return {"boxes": torch.tensor([box], dtype=torch.float32), "labels": torch.tensor([_CATEGORY]), "masks": mask}


def test_bbox_and_segm() -> None:
    """Perfect predictions score 1.0 on both metrics, and perturbing only the masks moves only segm (DoD).

    The pairing is the point. Half of it alone proves little: a segm column
    aliased onto the bbox column, or recomputed from the box corners rather than
    from the decoded masks, reports 1.0 for the perfect case exactly as a correct
    implementation does. It is the second half — identical boxes, ruined masks —
    that separates them, and it also catches the mirror defect of a bbox column
    contaminated by mask overlap.
    """
    target = _target(_BOX, _filled_mask(_BOX))
    perfect = _prediction(_BOX, _filled_mask(_BOX))
    ruined_mask = torch.zeros((1, *_IMAGE_SIZE), dtype=torch.bool)
    ruined_mask[0, 12:14, 10:12] = True  # a 2x2 speck inside the same, unchanged box
    ruined = _prediction(_BOX, ruined_mask)

    exact = evaluate_bbox_and_segm([perfect], [target])
    perturbed = evaluate_bbox_and_segm([ruined], [target])

    assert exact["map"] == pytest.approx(1.0)
    assert exact["segm_map"] == pytest.approx(1.0)
    assert perturbed["map"] == pytest.approx(exact["map"])  # the boxes never moved
    assert perturbed["segm_map"] < exact["segm_map"]


def test_empty_predicted_mask_scores_zero_segm() -> None:
    """A detection whose mask is empty earns no segm credit and raises nothing.

    The everyday output of a model whose mask probabilities never clear the A37
    threshold inside the predicted box. An implementation that encodes an
    all-``False`` mask by way of its bounding box (there is none) or that treats
    an empty RLE as an error would crash here rather than score 0.
    """
    target = _target(_BOX, _filled_mask(_BOX))
    empty = _prediction(_BOX, torch.zeros((1, *_IMAGE_SIZE), dtype=torch.bool))

    stats = evaluate_segm([empty], [target])

    assert stats["map"] == pytest.approx(0.0)


def test_polygon_and_rle_targets_decode_to_the_same_mask() -> None:
    """One rectangle expressed as a polygon and as an RLE yields the identical target mask.

    COCO stores ordinary instances as polygons and ``iscrowd`` regions as RLE, so
    a decoder that handles only one encoding — or that transposes the ``(width,
    height)`` argument order the RLE helper takes, which a square test image would
    hide — puts the two kinds of ground truth on different grids while every
    shape still looks plausible. The image here is deliberately non-square.
    """
    height, width = 12, 20
    polygon = {"bbox": [2.0, 3.0, 6.0, 4.0], "category_id": _CATEGORY, "segmentation": [[2, 3, 8, 3, 8, 7, 2, 7]]}
    reference = np.zeros((height, width), dtype=np.uint8)
    reference[3:7, 2:8] = 1
    encoded = coco_mask.encode(np.asfortranarray(reference))
    rle = {"bbox": [2.0, 3.0, 6.0, 4.0], "category_id": _CATEGORY, "segmentation": encoded}

    from_polygon = annotations_to_target([polygon], image_size=(height, width))
    from_rle = annotations_to_target([rle], image_size=(height, width))

    assert torch.equal(from_polygon["masks"], from_rle["masks"])
    assert int(from_polygon["masks"].sum()) == 24  # the 6x4 rectangle, not its transpose


def test_padding_rows_drop_masks_and_boxes_by_the_same_filter() -> None:
    """The surviving mask is the surviving detection's own, not the first row of the stack.

    The batch is built so a "keep the first N rows" filter — the shape an
    off-by-one in the mask path naturally takes — survives with the wrong answer:
    the single scored detection is row 1, and rows 0 and 2 are score-zero padding.
    Boxes and masks are made distinguishable per row, so the assert below fails
    for any mask filter that is not the *same* boolean mask the boxes take.
    """
    detections = torch.tensor(
        [
            [
                [0.0, 0.0, 4.0, 4.0, 0.0, 0.0],  # padding, and it sits first
                [8.0, 8.0, 16.0, 16.0, 0.9, 0.0],  # the one real detection
                [20.0, 20.0, 24.0, 24.0, 0.0, 0.0],  # padding
            ]
        ]
    )
    masks = torch.zeros(3, 32, 32, dtype=torch.bool)
    masks[0, 0:4, 0:4] = True
    masks[1, 8:16, 8:16] = True
    masks[2, 20:24, 20:24] = True

    preds = detections_to_predictions(detections, label_to_category={0: _CATEGORY}, masks=[masks])

    assert preds[0]["boxes"].tolist() == [[8.0, 8.0, 16.0, 16.0]]
    assert torch.equal(preds[0]["masks"], masks[1:2])
    assert int(preds[0]["masks"].sum()) == 64  # the 8x8 mask of row 1, not the 4x4 of row 0


def _loader(fixture_dir: Path, letterbox: Letterbox) -> list[tuple[Tensor, list[int], list[tuple[int, int]]]]:
    """Materialize one letterboxed batch of the first few fixture images.

    Examples:
        >>> callable(_loader)  # needs the live detseg_fixture_dir fixture on disk
        True
    """
    split_dir = fixture_dir / _SPLIT
    images, _, _ = load_eval_annotations(split_dir / _ANNOTATION)
    return list(letterboxed_batches(images[:_NUM_IMAGES], split_dir, letterbox, _NUM_IMAGES))


def test_detection_only_model_reports_bbox_metrics_and_no_segm_key(detseg_fixture_dir: Path) -> None:
    """A checkpoint without a mask branch evaluates exactly as it did before WP-053b.

    The regression gate on the whole change: the segmentation support must be
    reachable only through a model that actually carries prototypes. A decoder
    call rerouted through the index-returning path for everyone, or a report
    padded with zeroed ``segm_`` entries "for consistency", would change what
    every existing detection run reports.
    """
    torch.manual_seed(0)
    letterbox = Letterbox(_CANVAS)
    split_dir = detseg_fixture_dir / _SPLIT
    _, targets, label_map = load_eval_annotations(split_dir / _ANNOTATION)
    model = DetectionLitModule(depth=0.34, width=0.25, max_channels=1024, num_classes=_NUM_CLASSES)
    evaluator = DualPathEvaluator(model, TopKDecoder(k=_MAX_DET), NMSDecoder(max_det=_MAX_DET), label_map, letterbox)

    report = evaluator.evaluate(_loader(detseg_fixture_dir, letterbox), targets, torch.device("cpu"))

    assert set(report) == {"e2e", "nms"}
    for stats in report.values():
        assert set(stats) == set(_METRIC_KEYS)
        assert not any(key.startswith(_SEGM_PREFIX) for key in stats)


def _saturate_mask_branch(model: Segmenter) -> None:
    """Bias an untrained segmenter into emitting non-degenerate boxes and non-empty masks.

    At initialization the mask logits are ~0, i.e. probability 0.5, which the
    strictly-greater A37 threshold rounds to **empty everywhere**, and the raw
    ltrb distances are ~0, i.e. zero-extent boxes whose crop is empty too. An
    end-to-end test on that model would pass while gathering coefficients from
    entirely the wrong anchors, because every mask is empty either way. Positive
    output biases on the box stems, the coefficient stems and the prototype
    convolution make the whole chain produce real geometry to compare.

    Examples:
        >>> model = Segmenter("n", num_classes=4).eval()
        >>> _saturate_mask_branch(model)
        >>> float(model.protonet.layers[-1].bias[0].detach())
        1.0
    """
    with torch.no_grad():
        torch.nn.init.constant_(model.protonet.layers[-1].bias, 1.0)
        for branch in (model.head.o2o, model.head.o2m):
            for stem in branch.coeff_stems:
                torch.nn.init.constant_(stem[-1].bias, 1.0)
            for stem in branch.box_stems:
                torch.nn.init.constant_(stem[-1].bias, 2.0)


def _segmentation_evaluator(label_map: dict[int, int], letterbox: Letterbox) -> DualPathEvaluator:
    """Build the evaluator around a saturated untrained segmenter.

    Examples:
        >>> evaluator = _segmentation_evaluator({0: 5}, Letterbox(160))
        >>> type(evaluator).__name__
        'DualPathEvaluator'
    """
    model = Segmenter("n", num_classes=_NUM_CLASSES).eval()
    _saturate_mask_branch(model)
    return DualPathEvaluator(model, TopKDecoder(k=_MAX_DET), NMSDecoder(max_det=_MAX_DET), label_map, letterbox)


def test_segmentation_model_reports_both_metrics_on_both_paths(detseg_fixture_dir: Path) -> None:
    """A real segmenter reports finite bbox *and* segm statistics for the E2E and dense paths.

    The end-to-end wiring gate, run on an untrained model where the numbers
    themselves mean nothing: prototypes reach the decode, each path gathers its
    own branch's coefficients by its own anchor indices, masks are assembled on
    the letterboxed canvas and landed on each image's original grid, and the
    ground-truth polygons decode onto that same grid. Any mismatch in that chain
    surfaces here as a shape error or a non-finite statistic rather than as a
    plausible number nobody can check.
    """
    torch.manual_seed(0)
    letterbox = Letterbox(_CANVAS)
    split_dir = detseg_fixture_dir / _SPLIT
    _, targets, label_map = load_eval_annotations(split_dir / _ANNOTATION, with_masks=True)
    evaluator = _segmentation_evaluator(label_map, letterbox)

    report = evaluator.evaluate(_loader(detseg_fixture_dir, letterbox), targets, torch.device("cpu"))

    expected = {*_METRIC_KEYS, *(_SEGM_PREFIX + key for key in _METRIC_KEYS)}
    assert set(report) == {"e2e", "nms"}
    for stats in report.values():
        assert set(stats) == expected
        assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())


def _mixed_batches() -> list[tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]]:
    """Return three single-image ``(preds, targets)`` batches that do not score alike.

    Batch quality has to *differ* for a streaming test to mean anything. An
    untrained model is useless here: it scores 0.0 on every statistic, so any
    subset of the batches gives the same answer as all of them and every
    accumulation bug is invisible. These three -- a perfect instance, a
    mask-missed instance, another perfect one -- land the aggregate strictly
    between 0 and 1, and each batch alone gives something else.

    Examples:
        >>> batches = _mixed_batches()
        >>> len(batches)
        3
        >>> [len(preds) for preds, _ in batches]
        [1, 1, 1]
    """
    good = (_prediction(_BOX, _filled_mask(_BOX)), _target(_BOX, _filled_mask(_BOX)))
    missed_mask = _prediction(_BOX, torch.zeros((1, *_IMAGE_SIZE), dtype=torch.bool))
    return [([good[0]], [good[1]]), ([missed_mask], [good[1]]), ([good[0]], [good[1]])]


def test_streaming_scoring_equals_scoring_everything_at_once() -> None:
    """Folding batches in one at a time reports what scoring them together reports.

    The evaluator no longer holds every image's predictions until the end -- it
    could not, for masks: a val2017 segmentation run would keep both paths'
    predicted masks and the ground truth dense and simultaneously. Per-batch
    updates bound that, and this pins the equivalence on all 24 statistics against
    the functional one-shot entry point.

    The non-vacuity assertion is the load-bearing part. With a metric reset on
    every update -- scoring the last batch alone -- an equality against
    zero-scoring predictions still passes, which is exactly how the first version
    of this test failed to catch that mutation. Requiring the aggregate to sit
    strictly between 0 and 1 makes "all the numbers are 0" unable to satisfy it.
    """
    batches = _mixed_batches()
    scorer = _StreamingScorer()
    for preds, targets in batches:
        scorer.update(preds, targets)
    streamed = scorer.compute()

    all_preds = [pred for preds, _ in batches for pred in preds]
    all_targets = [target for _, targets in batches for target in targets]
    expected = evaluate_bbox_and_segm(all_preds, all_targets)

    assert 0.0 < expected["segm_map"] < 1.0  # the batches genuinely disagree
    assert set(streamed) == set(expected)
    for key, value in expected.items():
        assert streamed[key] == pytest.approx(value)


def test_evaluate_feeds_every_batch_to_both_paths(detseg_fixture_dir: Path) -> None:
    """The evaluator's loop hands each batch to each path's scorer exactly once.

    The equivalence above pins the scorer; this pins the loop that drives it. A
    dropped final batch or a path updated from the other path's predictions leaves
    a complete, finite report behind, and on a fixture where every prediction
    scores zero it leaves an *identical* one.
    """
    torch.manual_seed(0)
    letterbox = Letterbox(_CANVAS)
    split_dir = detseg_fixture_dir / _SPLIT
    images, targets, label_map = load_eval_annotations(split_dir / _ANNOTATION, with_masks=True)
    batches = list(letterboxed_batches(images[:_NUM_IMAGES], split_dir, letterbox, 1))
    assert len(batches) == _NUM_IMAGES

    seen: list[int] = []
    real_update = coco_eval._StreamingScorer.update

    def _counting_update(
        self: coco_eval._StreamingScorer,
        preds: list[dict[str, Tensor]],
        target: list[dict[str, Tensor]],
    ) -> None:
        seen.append(len(preds))
        real_update(self, preds, target)

    evaluator = _segmentation_evaluator(label_map, letterbox)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(coco_eval._StreamingScorer, "update", _counting_update)
        evaluator.evaluate(batches, targets, torch.device("cpu"))

    assert seen == [1] * (2 * _NUM_IMAGES)  # both paths, every batch, one image each


def test_predicted_masks_land_on_the_original_grid_with_content(detseg_fixture_dir: Path) -> None:
    """Every path's masks come back at the original image size, row-matched to its boxes, non-empty.

    The gate on the geometry the report itself cannot show. A mask left on the
    160px letterboxed canvas, or scored before the inverse letterbox, still
    produces a full set of finite statistics — the segm numbers are simply wrong.
    The non-emptiness assert is what stops this test from passing vacuously: an
    all-``False`` stack has the right shape no matter which anchors its
    coefficients came from.
    """
    torch.manual_seed(0)
    letterbox = Letterbox(_CANVAS)
    split_dir = detseg_fixture_dir / _SPLIT
    _, _, label_map = load_eval_annotations(split_dir / _ANNOTATION, with_masks=True)
    evaluator = _segmentation_evaluator(label_map, letterbox)
    batch, _, orig_sizes = _loader(detseg_fixture_dir, letterbox)[0]

    with torch.no_grad():
        paths = evaluator._predict_batch(batch, torch.device("cpu"), orig_sizes)

    for predictions in paths:
        for prediction, orig_size in zip(predictions, orig_sizes, strict=True):
            masks = prediction["masks"]
            assert masks.dtype == torch.bool
            assert masks.shape == (len(prediction["boxes"]), *orig_size)
            assert bool(masks.any())


def test_masks_stay_paired_with_their_own_detections() -> None:
    """Each surviving mask stays attached to the box it was decoded for.

    The score filter keeps masks and boxes in step, but keeping the right *count*
    is not the same as keeping the right *order*. A stack permuted relative to the
    detections still has one mask per box and still passes every shape and count
    assertion, because each mask is a real mask of a real object — merely attached
    to the wrong one.

    The two instances must carry **different labels** for this to be observable at
    all. Segm mAP never looks at boxes: it matches predicted masks to ground-truth
    masks by IoU within a class. Give two same-class detections each other's mask
    and the predicted set is unchanged, so each mask simply matches the other's
    ground truth and the score stays a perfect 1.0 — the permutation is invisible.
    Distinct labels break that symmetry: the mask offered under label A is then
    scored against label A's ground truth, misses it entirely, and segm collapses
    while bbox, which never depended on the ordering, holds at 1.0.
    """
    box_a, box_b = [4.0, 4.0, 20.0, 20.0], [40.0, 40.0, 60.0, 60.0]
    mask_a, mask_b = _filled_mask(box_a), _filled_mask(box_b)
    padding = torch.zeros((1, *_IMAGE_SIZE), dtype=torch.bool)
    detections = torch.tensor(
        [[[*box_a, 0.9, 0.0], [*box_b, 0.8, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    stack = torch.cat((mask_a, mask_b, padding), dim=0)
    targets = [
        {
            "boxes": torch.tensor([box_a, box_b], dtype=torch.float32),
            "labels": torch.tensor([_CATEGORY, _OTHER_CATEGORY]),
            "masks": torch.cat((mask_a, mask_b), dim=0),
        }
    ]

    predictions = detections_to_predictions(detections, {0: _CATEGORY, 1: _OTHER_CATEGORY}, masks=[stack])
    stats = evaluate_bbox_and_segm(predictions, targets)

    assert stats["map"] == pytest.approx(1.0)
    assert stats["segm_map"] == pytest.approx(1.0), "each mask must stay paired with its own detection"


class TestPaddedRowMaskDecode:
    """``_path_masks`` decodes the real detections only, and pads the rest back."""

    @staticmethod
    def _scene(real: int, padded: int) -> tuple[DualPathEvaluator, Tensor, Tensor, Tensor, _BatchGeometry]:
        """One image of ``real`` scored detections inside a ``padded``-row block, with prototypes.

        Examples:
            >>> callable(TestPaddedRowMaskDecode._scene)
            True
        """
        generator = torch.Generator().manual_seed(0)
        rows = [[4.0 + 8 * n, 4.0 + 8 * n, 28.0 + 8 * n, 28.0 + 8 * n, 0.9 - 0.01 * n, 0.0] for n in range(real)]
        rows += [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * (padded - real)
        detections = torch.tensor([rows], dtype=torch.float32)
        anchor_index = torch.arange(padded).unsqueeze(0)
        coefficients = torch.rand(1, padded, 32, generator=generator) * 2.0 - 1.0
        prototypes = torch.rand(1, 32, 16, 16, generator=generator) * 2.0 - 1.0
        letterbox = Letterbox(_IMAGE_SIZE[0])
        geometry = _BatchGeometry(
            anchor_points=torch.zeros(padded, 2),
            strides=torch.ones(padded),
            canvas=_IMAGE_SIZE,
            orig_sizes=[_IMAGE_SIZE],
            prototypes=prototypes,
        )
        evaluator = DualPathEvaluator(
            model=torch.nn.Identity(),  # _path_masks never consults the model
            e2e_decoder=TopKDecoder(),
            nms_decoder=NMSDecoder(),
            label_to_category={0: _CATEGORY},
            letterbox=letterbox,
        )
        return evaluator, detections, anchor_index, coefficients, geometry

    def test_the_prefix_decode_matches_decoding_every_padded_row(self) -> None:
        """Slicing to the real detections returns exactly what decoding all 300 rows returned.

        The padded block is the fixed ``max_det`` shape ``pad_detections`` emits, so an
        image with a handful of objects still arrives with most of its rows carrying a
        score-zero, all-zero box. Those rows are skipped now instead of decoded, which is
        only sound if the full decode produced nothing for them: an all-zero box crops
        every pixel away, so it should. Asserted against the full decode rather than
        argued — the real rows bit-identical, the padding rows all ``False`` on both
        sides, so a future change that makes an empty box decode to *something* fails
        here rather than silently changing a score.
        """
        evaluator, detections, anchor_index, coefficients, geometry = self._scene(real=3, padded=16)
        every_row = torch.tensor([[[1.0, 1.0, float(_IMAGE_SIZE[1]), float(_IMAGE_SIZE[0]), 0.5, 0.0]] * 16])
        every_row[0, :3] = detections[0, :3]

        sliced = evaluator._path_masks(detections, anchor_index, coefficients, geometry)[0]
        unsliced = evaluator._path_masks(every_row, anchor_index, coefficients, geometry)[0]

        assert sliced.shape == (16, *_IMAGE_SIZE)
        assert torch.equal(sliced[:3], unsliced[:3])
        assert not bool(sliced[3:].any()), "a padding row decoded to a non-empty mask"

    def test_an_image_with_no_real_detections_yields_the_padded_shape(self) -> None:
        """An all-padding image returns ``max_det`` empty masks rather than an empty stack.

        The stack is indexed downstream by a boolean mask over every padded row, so a
        short stack mis-indexes rather than failing loudly. This is the degenerate case
        of that contract: nothing scored, so nothing is decoded at all, and the shape has
        to come from the block rather than from the decoder.
        """
        evaluator, detections, anchor_index, coefficients, geometry = self._scene(real=0, padded=16)

        masks = evaluator._path_masks(detections, anchor_index, coefficients, geometry)[0]

        assert masks.shape == (16, *_IMAGE_SIZE)
        assert masks.dtype == torch.bool
        assert not bool(masks.any())
