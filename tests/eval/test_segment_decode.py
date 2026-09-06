# SPDX-License-Identifier: Apache-2.0
"""Unit gates for the WP-053a segmentation mask decode (A10, A11, A16, A37).

Four properties define the decode, and every test below pins exactly one of
them:

- the top-k reduction still ranks as it always did, and now says **which**
  anchors it kept, so mask coefficients can be gathered by the same indices the
  boxes were (a second, independently written selection would pair a box with
  another anchor's coefficients and produce a plausible mask of the wrong
  object);
- the probability field is upsampled **before** it is binarized, so a mask
  boundary can land between prototype cells;
- the crop uses the A11 half-open pixel-centre rule the training loss uses;
- masks and boxes leave the letterboxed canvas through the *same* geometry, so a
  segm score cannot be quietly wrong while the bbox score is right.

Expected mask windows are hand-derived from the bilinear mapping
``input = (out + 0.5) / scale - 0.5`` (``align_corners=False``), so the column
indices asserted below are exact, not tolerances.
"""

from __future__ import annotations

import pytest
import torch

from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.targets import Targets
from lucid_yolo.decode.common import to_letterboxed_original
from lucid_yolo.eval import decode_instance_masks, masks_to_original
from lucid_yolo.models.heads.detect import o2o_topk, o2o_topk_with_indices

_BOX_CORNERS = 4
_SATURATED = 10.0  # sigmoid(+-10) is within 5e-5 of 1/0: a hard prototype edge.


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Seed the global RNG so the random top-k comparison is reproducible."""
    torch.manual_seed(0)


def _legacy_o2o_topk(scores: torch.Tensor, boxes: torch.Tensor, k: int) -> torch.Tensor:
    """Reimplement the pre-WP-053a ``o2o_topk`` body verbatim, as the reference ranking.

    Examples:
        >>> scores = torch.tensor([[[0.1, 0.9], [0.9, 0.1]]])
        >>> boxes = torch.zeros(1, 2, 4)
        >>> _legacy_o2o_topk(scores, boxes, k=1).shape
        torch.Size([1, 1, 6])
    """
    _, num_anchors, _ = scores.shape
    confidence = scores.sigmoid()
    max_conf, class_index = confidence.max(dim=-1)
    keep = min(k, num_anchors)
    top_conf, top_anchor = max_conf.topk(keep, dim=1)
    gather_box = top_anchor.unsqueeze(-1).expand(-1, -1, _BOX_CORNERS)
    top_boxes = boxes.gather(1, gather_box)
    top_class = class_index.gather(1, top_anchor).to(scores.dtype)
    return torch.cat((top_boxes, top_conf.unsqueeze(-1), top_class.unsqueeze(-1)), dim=-1)


def _saturated_prototypes(grid: int, region: slice) -> torch.Tensor:
    """Return a ``(1, 1, grid, grid)`` prototype that is ``+10`` inside ``region`` and ``-10`` outside.

    Examples:
        >>> proto = _saturated_prototypes(4, slice(1, 3))
        >>> proto.shape
        torch.Size([1, 1, 4, 4])
        >>> bool(proto[0, 0, 1:3, 1:3].eq(10.0).all()), float(proto[0, 0, 0, 0])
        (True, -10.0)
    """
    prototypes = torch.full((1, 1, grid, grid), -_SATURATED)
    prototypes[0, 0, region, region] = _SATURATED
    return prototypes


def test_o2o_topk_ranking_is_bit_identical_to_the_pre_split_body() -> None:
    """The wrapper reproduces the old selection exactly, not merely closely.

    Catches a rewrite of the reduction smuggled in with the index return — a
    reordered ``topk``, a ``max`` over probabilities taken elsewhere, or a
    different tie handling would move detections that the decode path has been
    accepted against.
    """
    scores = torch.randn(2, 50, 5)
    boxes = torch.randn(2, 50, 4)

    detections = o2o_topk(scores, boxes, k=7)

    assert torch.equal(detections, _legacy_o2o_topk(scores, boxes, 7))


def test_anchor_index_reproduces_the_detection_corners() -> None:
    """Gathering ``boxes`` by the returned indices yields the tuple's own corners.

    This is the contract every per-anchor quantity relies on: if the reported
    indices were stale, off by a permutation, or taken from a second ranking,
    the corners they select would differ from the corners the tuple carries.
    """
    scores = torch.randn(2, 50, 5)
    boxes = torch.randn(2, 50, 4)

    detections, anchor_index = o2o_topk_with_indices(scores, boxes, k=7)

    gathered = boxes.gather(1, anchor_index.unsqueeze(-1).expand(-1, -1, _BOX_CORNERS))
    assert torch.equal(gathered, detections[..., :_BOX_CORNERS])


def test_coefficients_gathered_by_anchor_index_belong_to_the_kept_anchors() -> None:
    """Coefficient rows selected by the returned indices are the kept anchors' own rows.

    The defect this catches is a second selection path: with a known score order
    the kept anchors are 1, 3, 4, so a coefficient map gathered by any other
    ranking (e.g. one recomputed from the same scores by a separate helper that
    breaks ties or sorts differently) yields different rows here while every box
    stays correct — a mask of the wrong object with a plausible score.
    """
    scores = torch.tensor([[[0.1], [0.9], [0.3], [0.7], [0.5], [0.2]]])  # (B=1, A=6, C=1)
    boxes = torch.zeros(1, 6, 4)
    coefficients = torch.tensor([[[float(row), -float(row)] for row in range(6)]])  # (B=1, A=6, K=2)

    _, anchor_index = o2o_topk_with_indices(scores, boxes, k=3)

    gathered = coefficients[0][anchor_index[0]]
    assert torch.equal(anchor_index, torch.tensor([[1, 3, 4]]))
    assert torch.equal(gathered, torch.tensor([[1.0, -1.0], [3.0, -3.0], [4.0, -4.0]]))


def test_one_hot_coefficient_reproduces_its_prototype_rectangle() -> None:
    """A one-hot coefficient recovers exactly the rectangle its prototype encodes.

    Catches an assembly that mixes the wrong prototypes (a transposed einsum
    would blend in the constant distractor channel), a missing sigmoid (raw
    logits of -10 would still threshold above 0.5 nowhere but +10 everywhere the
    signs differ), and a wrong upsample factor: at scale 4 the +-10 edge between
    prototype cells 1 and 2 lands at canvas coordinate 8, so the recovered window
    is exactly ``[8, 24)`` on both axes — bar its four corner pixels, where the
    2-D bilinear weight of the one interior prototype cell is only
    ``0.625 ** 2 = 0.39``. That rounded corner is itself a fingerprint of
    interpolating probabilities: a binary mask upsampled after thresholding has
    square corners.
    """
    prototypes = torch.cat((_saturated_prototypes(8, slice(2, 6)), torch.full((1, 1, 8, 8), -_SATURATED)), dim=1)
    coefficients = torch.tensor([[[1.0, 0.0]]])  # (B=1, N=1, K=2)
    boxes = torch.tensor([[[0.0, 0.0, 32.0, 32.0]]])  # the whole canvas: no clipping

    masks = decode_instance_masks(prototypes, coefficients, boxes, image_size=(32, 32))

    expected = torch.zeros(32, 32, dtype=torch.bool)
    expected[8:24, 8:24] = True
    expected[[8, 8, 23, 23], [8, 23, 8, 23]] = False
    assert torch.equal(masks[0, 0], expected)


def test_mask_is_zero_outside_the_predicted_box() -> None:
    """A box covering part of the prototype rectangle keeps only the boxed part.

    This is the test that fails if the crop is dropped or applied with the
    ground-truth-box geometry of the loss rather than the predicted box: without
    it the whole ``[8, 24)`` rectangle survives, since the prototype claims it.
    The single excluded pixel at ``(8, 8)`` is the prototype rectangle's own
    rounded corner (see
    :func:`test_one_hot_coefficient_reproduces_its_prototype_rectangle`), not a
    property of the crop.
    """
    prototypes = torch.cat((_saturated_prototypes(8, slice(2, 6)), torch.full((1, 1, 8, 8), -_SATURATED)), dim=1)
    coefficients = torch.tensor([[[1.0, 0.0]]])
    boxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0]]])  # top-left quarter of the canvas

    masks = decode_instance_masks(prototypes, coefficients, boxes, image_size=(32, 32))

    expected = torch.zeros(32, 32, dtype=torch.bool)
    expected[8:16, 8:16] = True
    expected[8, 8] = False
    assert torch.equal(masks[0, 0], expected)


@pytest.mark.parametrize(
    ("probability", "foreground"),
    [
        pytest.param(0.49, False, id="just-below-half"),
        pytest.param(0.51, True, id="just-above-half"),
    ],
)
def test_threshold_cuts_at_one_half(probability: float, foreground: bool) -> None:
    """A constant field just under 0.5 is empty and just over it is full inside the box.

    Pins the A37 binarization threshold itself: a cut placed at 0 (thresholding
    the logits instead of the probabilities) or at any value outside
    ``(0.49, 0.51)`` would make both parametrizations agree, and a decode that
    thresholds before the sigmoid would report foreground for the 0.49 case too,
    whose logit is negative.
    """
    prototypes = torch.full((1, 1, 4, 4), float(torch.logit(torch.tensor(probability))))
    coefficients = torch.ones(1, 1, 1)
    boxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0]]])

    masks = decode_instance_masks(prototypes, coefficients, boxes, image_size=(16, 16))

    assert bool(masks.all()) is foreground
    assert bool(masks.any()) is foreground


def test_output_is_bool_over_batch_and_instance_axes() -> None:
    """The decode returns ``(B, N, H, W)`` booleans, not floats or a flattened stack.

    Catches an output that keeps the probability dtype (an evaluator would then
    score a soft mask), and a batch/instance axis collapsed by a loop that
    concatenated instead of stacking.
    """
    prototypes = torch.randn(2, 4, 8, 8)
    coefficients = torch.randn(2, 3, 4).tanh()
    boxes = torch.tensor([[0.0, 0.0, 32.0, 32.0]]).expand(2, 3, 4).contiguous()

    masks = decode_instance_masks(prototypes, coefficients, boxes, image_size=(32, 32))

    assert masks.shape == (2, 3, 32, 32)
    assert masks.dtype == torch.bool


def test_no_kept_detections_returns_the_empty_stack() -> None:
    """An image with nothing to decode returns ``(B, 0, H, W)`` rather than raising.

    ``F.interpolate`` treats the instance axis as channels and rejects a zero-length
    one, so this raised ``RuntimeError`` for as long as the function has existed. No
    test caught it because the only caller was the evaluator, whose decoders always
    hand over a fixed 300 rows with their padding included — the axis was never empty.
    The predict path decodes only the survivors of a confidence cut, and an image can
    have none, which is how the case finally arrived.

    The shape contract is the point of returning rather than raising: a caller
    stacking per-image results needs an empty tensor of the right rank and dtype, and
    an exception forces every such caller to special-case the count itself.
    """
    prototypes = torch.randn(2, 4, 8, 8)
    coefficients = torch.zeros(2, 0, 4)
    boxes = torch.zeros(2, 0, 4)

    masks = decode_instance_masks(prototypes, coefficients, boxes, image_size=(32, 32))

    assert masks.shape == (2, 0, 32, 32)
    assert masks.dtype == torch.bool


def test_boundary_lands_between_prototype_cells() -> None:
    """A sloped probability edge binarizes at a position no upsampled binary mask could reach.

    The gate against thresholding first and upsampling afterwards. The prototype
    columns carry logits ``[3, 1, -3, -3]``, so the *probability* field crosses
    0.5 at canvas x ~= 7.35: output column 6 (centre-mapped probability 0.646)
    is foreground and column 7 (0.475) is not. Thresholding at the prototype grid
    gives the binary columns ``[1, 1, 0, 0]``, and nearest-upsampling those puts
    the boundary at 8 — column 7 foreground. Only the assert on column 7 tells
    the two implementations apart.
    """
    logits = torch.tensor([3.0, 1.0, -3.0, -3.0])
    prototypes = logits.reshape(1, 1, 1, 4).expand(1, 1, 4, 4).contiguous()
    coefficients = torch.ones(1, 1, 1)
    boxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0]]])

    masks = decode_instance_masks(prototypes, coefficients, boxes, image_size=(16, 16))

    expected_row = torch.zeros(16, dtype=torch.bool)
    expected_row[:7] = True
    assert torch.equal(masks[0, 0, 0], expected_row)


def test_masks_to_original_round_trips_a_real_letterbox() -> None:
    """A mask letterboxed by the validation transform comes back on the original grid.

    Catches padding removed from the wrong side or the wrong axis, and a resize
    that inverts the canvas ratio instead of the letterbox ratio: a 48x96 image
    into a 64x64 canvas scales by 2/3 and pads 16 rows top and bottom, so a
    ratio- or pad-blind inverse leaves the rectangle stretched or displaced,
    which the intersection-over-union assert below rejects even though the pixel
    count alone might survive it.
    """
    letterbox = Letterbox(64, pad_value=0.0)
    original = torch.zeros(48, 96, dtype=torch.bool)
    original[8:40, 12:84] = True
    empty = Targets(boxes=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long))
    canvas, _ = letterbox(original.float().unsqueeze(0), empty)

    recovered = masks_to_original(canvas > 0.5, letterbox, orig_size=(48, 96))[0]

    intersection = int((recovered & original).sum())
    union = int((recovered | original).sum())
    assert recovered.shape == original.shape
    assert intersection / union > 0.95, "the recovered mask must cover the original, not a shifted copy"
    assert abs(int(recovered.sum()) - int(original.sum())) <= 0.05 * int(original.sum())


def test_recovered_mask_and_box_agree_on_the_original_grid() -> None:
    """A mask filling its box still fills it after both inverse transforms.

    The gate against the silent failure mode: if ``masks_to_original`` derived
    its scale or padding independently of the box inverse, every bbox score would
    stay right while every segm score drifted by the difference. Both are pushed
    through their own inverse here and the mask's bounding box must land on the
    mapped box exactly, not merely near it.
    """
    letterbox = Letterbox(64)
    box = torch.tensor([12.0, 20.0, 44.0, 46.0])
    prototypes = torch.full((1, 1, 16, 16), _SATURATED)
    masks = decode_instance_masks(prototypes, torch.ones(1, 1, 1), box.reshape(1, 1, 4), image_size=(64, 64))

    detection = torch.cat((box, torch.tensor([0.9, 0.0]))).reshape(1, 1, 6)
    mapped_box = to_letterboxed_original(detection, orig_size=(48, 96), letterboxed_size=(64, 64))[0, 0, :4]
    recovered = masks_to_original(masks[0], letterbox, orig_size=(48, 96))[0]

    rows = recovered.any(dim=1).nonzero().flatten()
    cols = recovered.any(dim=0).nonzero().flatten()
    mask_box = torch.tensor([cols[0], rows[0], cols[-1] + 1, rows[-1] + 1], dtype=torch.float32)
    # atol measured, not guessed: this fixture is fully deterministic (no RNG anywhere in
    # it) and the two boxes agree to 0.0 on all four sides -- 48x96 into a 64-canvas is an
    # exact 2/3 with an exact 16-row pad, so the inverse lands on integers rather than
    # between pixels. The former atol=1.0 therefore bought nothing but a blind spot: it
    # accepted a whole-pixel shift per side, which is precisely the silent segm-only drift
    # the docstring above names. 1e-4 absorbs float noise on the division and nothing else.
    assert torch.allclose(mask_box, mapped_box, atol=1e-4), f"mask box {mask_box} vs detection box {mapped_box}"
