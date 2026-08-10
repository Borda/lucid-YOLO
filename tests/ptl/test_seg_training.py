# SPDX-License-Identifier: Apache-2.0
"""Tests for the segmentation supervision wired into the training step (WP-087).

Covers the DoD of the segmentation training path: a ``task="segment"`` module's
``training_step`` carries live mask and auxiliary terms on top of the detection
objective, the gradient of those terms reaches every branch they are supposed to
train, the degenerate no-instance batch stays finite, the mask targets are
gathered by the **assignment** rather than by positional order, and the
``task="detect"`` path is untouched.

The module is built at n-scale multipliers with a low channel cap and a 160-px
input so the real stack runs in a couple of seconds on CPU; batches are synthetic
tensors plus hand-built :class:`~lucid_yolo.data.targets.Targets` carrying one
rectangular polygon ring per box. Direct ``training_step`` calls replace
``module.log`` because Lightning's ``self.log`` requires trainer attachment — the
recording stub doubles as the observation point for the two pre-gain terms.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from lucid_yolo.assign import make_anchor_points
from lucid_yolo.assign.tal import AssignResult
from lucid_yolo.data.targets import Targets
from lucid_yolo.losses.dual_loss import DualBranchLoss
from lucid_yolo.losses.mask_loss import instance_mask_loss
from lucid_yolo.models.build import SegmentOutput
from lucid_yolo.models.heads.detect import DualHeadOutput, decode_ltrb
from lucid_yolo.models.heads.proto import assemble_masks
from lucid_yolo.ptl import DetectionLitModule, pad_targets
from lucid_yolo.ptl.datamodule import _PROTO_STRIDE, collate_detection, unpack_masks
from lucid_yolo.ptl.module import _VAL_SEGM_MAX_DET
from lucid_yolo.ptl.seg_targets import instance_mask_targets

#: Class count of the tiny test head.
_NUM_CLASSES = 4
#: Square input side; divisible by every stride for an integer anchor grid.
_IMG_SIZE = 160
#: Images per synthetic batch.
_BATCH_SIZE = 2
#: Feature-level input-pixel strides of the P3/P4/P5 head.
_STRIDES: tuple[int, int, int] = (8, 16, 32)
#: Loss gains the tiny module is built with (also used to rebuild the reference loss).
_BOX_GAIN, _CLS_GAIN, _L1_GAIN, _ALPHA = 7.5, 0.5, 6.0, 0.5


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed torch before each test so model init and synthetic batches are reproducible."""
    torch.manual_seed(0)


class _LogRecorder:
    """Stand-in for ``LightningModule.log`` that records every logged scalar."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def __call__(self, name: str, value: Tensor, **kwargs: object) -> None:
        """Record ``value`` under ``name``, ignoring Lightning's keyword arguments."""
        del kwargs
        self.values[name] = float(value.detach())


def _tiny_module(task: str = "segment") -> DetectionLitModule:
    """Build an n-scale module with a low channel cap for fast CPU tests."""
    return DetectionLitModule(
        depth=0.34,
        width=0.25,
        max_channels=256,
        num_classes=_NUM_CLASSES,
        task=task,
        box_gain=_BOX_GAIN,
        cls_gain=_CLS_GAIN,
        l1_gain=_L1_GAIN,
        alpha=_ALPHA,
    )


def _rectangle_ring(box: Tensor) -> Tensor:
    """Return the four-point polygon ring tracing an ``xyxy`` box."""
    x1, y1, x2, y2 = box.tolist()
    return torch.tensor([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=torch.float32)


def _synthetic_targets(num_boxes: int) -> Targets:
    """Build ``num_boxes`` valid ``xyxy`` targets, each with its own polygon ring."""
    top_left = torch.rand(num_boxes, 2) * 80.0
    size = torch.rand(num_boxes, 2) * 40.0 + 10.0
    boxes = torch.cat([top_left, top_left + size], dim=1)
    labels = torch.randint(0, _NUM_CLASSES, (num_boxes,))
    return Targets(boxes=boxes, labels=labels, polygons=[_rectangle_ring(box) for box in boxes])


def _synthetic_batch() -> tuple[Tensor, list[Targets]]:
    """Build a two-image batch with ragged (2 and 1) instance counts."""
    images = torch.randn(_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)
    return images, [_synthetic_targets(2), _synthetic_targets(1)]


def _has_gradient(module: nn.Module) -> bool:
    """Return whether any parameter of ``module`` carries a non-zero gradient."""
    return any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in module.parameters())


def _reference_detection_loss(module: DetectionLitModule, batch: tuple[Tensor, list[Targets]]) -> Tensor:
    """Score the dual detection loss over ``batch`` through the public API alone."""
    images, targets = batch
    head_out = module(images)
    feature_sizes = [(_IMG_SIZE // stride, _IMG_SIZE // stride) for stride in _STRIDES]
    anchor_points, strides = make_anchor_points(feature_sizes, list(_STRIDES))
    gt_boxes, gt_labels, gt_mask = pad_targets(targets)
    loss = DualBranchLoss(box_gain=_BOX_GAIN, cls_gain=_CLS_GAIN, l1_gain=_L1_GAIN, alpha=_ALPHA)
    return loss(
        head_out.o2m_cls,
        decode_ltrb(head_out.o2m_box, anchor_points, strides),
        head_out.o2o_cls,
        decode_ltrb(head_out.o2o_box, anchor_points, strides),
        anchor_points,
        gt_boxes,
        gt_labels,
        gt_mask,
        strides=strides,
    ).total


def test_segment_loss_exceeds_the_detection_only_loss() -> None:
    """The mask and auxiliary terms are live: sharing every detection weight, segment > detect.

    Catches the WP-034 stub surviving — an inert ``_task_extra_loss`` leaves the
    two totals equal, which no shape or finiteness check would notice.
    """
    detect = _tiny_module(task="detect")
    segment = _tiny_module(task="segment")
    segment.load_state_dict(detect.state_dict(), strict=False)
    batch = _synthetic_batch()
    detect.log = _LogRecorder()  # type: ignore[method-assign]
    segment.log = _LogRecorder()  # type: ignore[method-assign]

    detect_loss = detect.training_step(batch, 0)
    segment_loss = segment.training_step(batch, 0)

    assert bool(torch.isfinite(segment_loss)) and segment_loss.ndim == 0
    assert bool(segment_loss.detach() > detect_loss.detach())


def test_gradients_reach_every_mask_branch() -> None:
    """Backward through the segment loss reaches the prototype, auxiliary and coefficient stems.

    Catches an inert ``_task_extra_loss``, a detached mask target path, and a
    branch left unsupervised — in particular the one-to-one coefficient stem,
    whose absence from the loss would leave the deployed decode reading
    never-trained coefficients while every other test still passed.
    """
    module = _tiny_module()
    module.log = _LogRecorder()  # type: ignore[method-assign]

    module.training_step(_synthetic_batch(), 0).backward()

    assert _has_gradient(module.protonet)
    assert _has_gradient(module.proto_fusion)
    assert _has_gradient(module.semantic)
    assert _has_gradient(module.head.o2o.coeff_stems)
    assert _has_gradient(module.head.o2m.coeff_stems)


def test_image_without_instances_gives_a_finite_zero_mask_term() -> None:
    """An empty image yields a finite loss and no positives, not a ``0 / 0`` NaN.

    Catches a mean over an empty instance axis: the mask term would be NaN and
    poison the whole batch's total on the first background-only image.
    """
    module = _tiny_module()
    recorder = _LogRecorder()
    module.log = recorder  # type: ignore[method-assign]
    batch = (torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE), [Targets.empty()])

    loss = module.training_step(batch, 0)

    assert bool(torch.isfinite(loss))
    assert recorder.values["train/mask"] == 0.0


def test_boxes_without_polygons_are_rejected() -> None:
    """A detection-only annotation under ``task="segment"`` raises instead of training on nothing.

    Catches the silent-subset failure: without the guard, an image whose rings were
    dropped upstream would contribute no mask supervision and no error, and the
    only symptom would be a segm mAP nobody could explain.
    """
    module = _tiny_module()
    module.log = _LogRecorder()  # type: ignore[method-assign]
    images, targets = _synthetic_batch()
    boxes_only = [Targets(boxes=target.boxes, labels=target.labels) for target in targets]

    with pytest.raises(ValueError, match="one polygon ring per instance"):
        module.training_step((images, boxes_only), 0)


def test_validation_step_runs_without_the_training_only_semantic_branch() -> None:
    """A segment module validates in eval mode, where the auxiliary branch returns ``None`` (A17).

    Catches an unguarded ``semantic_aux_loss`` call: the branch is training-only,
    so the first validation batch of every tier run would raise on a ``None``.
    """
    module = _tiny_module().eval()
    module.log = _LogRecorder()  # type: ignore[method-assign]

    loss = module.validation_step(_synthetic_batch(), 0)

    assert bool(torch.isfinite(loss))


def test_mask_targets_follow_the_assignment_not_the_positive_order() -> None:
    """Positive ``k``'s mask is scored against ``gt_index[k]``, not against instance ``k``.

    The decisive test of the WP-087 pairing: with two positives whose assigned
    instances are swapped relative to their positional order, the naive pairing
    produces an equally finite, equally plausible loss. Only the value
    distinguishes them, so it is asserted against a hand-written swap.
    """
    prototypes = torch.randn(1, 2, 8, 8)
    coefficients = torch.randn(1, 4, 2).tanh()
    masks = torch.zeros(2, 8, 8)
    masks[0, :4, :] = 1.0  # instance 0: top half
    masks[1, :, :4] = 1.0  # instance 1: left half
    grid_boxes = torch.tensor([[[0.0, 0.0, 8.0, 4.0], [0.0, 0.0, 4.0, 8.0]]])
    assign = AssignResult(
        fg_mask=torch.tensor([[True, False, True, False]]),
        gt_index=torch.tensor([[1, -1, 0, -1]]),  # positive 0 -> instance 1, positive 2 -> instance 0
        target_labels=torch.tensor([[0, -1, 0, -1]]),
        target_boxes=torch.zeros(1, 4, 4),
        align_weights=torch.zeros(1, 4),
    )
    positive_coefficients = coefficients[:, [0, 2], :]
    mask_logits = assemble_masks(prototypes, positive_coefficients)[0]
    assigned = instance_mask_loss(mask_logits, masks[[1, 0]], grid_boxes[0][[1, 0]])
    naive = instance_mask_loss(mask_logits, masks[[0, 1]], grid_boxes[0][[0, 1]])

    scored = DetectionLitModule._branch_mask_loss(prototypes, coefficients, assign, masks.unsqueeze(0), grid_boxes)

    assert torch.equal(scored, assigned)
    assert not torch.isclose(assigned, naive)


def _looped_branch_mask_loss(
    prototypes: Tensor,
    coefficients: Tensor,
    assign: AssignResult,
    masks: list[Tensor],
    grid_boxes: Tensor,
) -> Tensor:
    """The per-image reference the batched implementation replaced.

    Kept verbatim as an oracle rather than deleted with the code: it is the form
    whose result the Det/Seg goldens were frozen against, and it is the only thing
    that can show the padded batched gather changed the *speed* and nothing else.
    """
    mask_logits: list[Tensor] = []
    mask_targets: list[Tensor] = []
    boxes: list[Tensor] = []
    for index, image_masks in enumerate(masks):
        gt_index = assign.gt_index[index][assign.fg_mask[index]]
        coefficient_rows = coefficients[index][assign.fg_mask[index]].unsqueeze(0)
        mask_logits.append(assemble_masks(prototypes[index : index + 1], coefficient_rows)[0])
        mask_targets.append(image_masks[gt_index])
        boxes.append(grid_boxes[index][gt_index])
    return instance_mask_loss(torch.cat(mask_logits), torch.cat(mask_targets), torch.cat(boxes))


@pytest.mark.parametrize(
    ("positives", "instances"),
    [
        pytest.param([3, 1], [2, 1], id="uneven-positive-counts"),
        pytest.param([2, 0], [1, 2], id="one-image-with-no-positives"),
        pytest.param([0, 0], [1, 1], id="no-positives-at-all"),
        pytest.param([4, 4], [2, 2], id="equal-counts-no-padding"),
    ],
)
def test_batched_mask_loss_equals_the_per_image_loop(positives: list[int], instances: list[int]) -> None:
    """The batched gather reproduces the per-image loop's value bit for bit.

    The looped form selected each image's positives with a boolean mask, whose
    output shape depends on the data — which forces a device sync per image per
    branch, measured at 1.2 s of stall per step at batch 32. The batched form pads
    to the largest positive count and selects once. That is a speed change only if
    the value is untouched, and the padding is where it could go wrong: padded rows
    carry the ``-1`` unassigned sentinel and must contribute nothing.

    Uneven counts are the point of the parametrization — with equal counts there is
    no padding, so a padding leak would pass unnoticed.
    """
    channels, grid = 2, 8
    anchors = max(positives) + 3
    prototypes = torch.randn(len(positives), channels, grid, grid)
    coefficients = torch.randn(len(positives), anchors, channels).tanh()
    fg_mask = torch.zeros(len(positives), anchors, dtype=torch.bool)
    gt_index = torch.full((len(positives), anchors), -1, dtype=torch.long)
    for image, (count, available) in enumerate(zip(positives, instances, strict=True)):
        fg_mask[image, :count] = True
        gt_index[image, :count] = torch.arange(count) % available
    masks = [torch.randint(0, 2, (count, grid, grid)).float() for count in instances]
    grid_boxes = torch.tensor([[[0.0, 0.0, float(grid), float(grid)]] * max(instances)] * len(positives))
    assign = AssignResult(
        fg_mask=fg_mask,
        gt_index=gt_index,
        target_labels=torch.zeros(len(positives), anchors, dtype=torch.long),
        target_boxes=torch.zeros(len(positives), anchors, 4),
        align_weights=torch.zeros(len(positives), anchors),
    )

    # The batched form takes the densified stack its caller now builds once for both
    # branches; the oracle keeps the ragged list the looped form consumed.
    padded = torch.zeros(len(masks), max(int(image_masks.shape[0]) for image_masks in masks), grid, grid)
    for index, image_masks in enumerate(masks):
        padded[index, : image_masks.shape[0]] = image_masks

    batched = DetectionLitModule._branch_mask_loss(prototypes, coefficients, assign, padded, grid_boxes)
    looped = _looped_branch_mask_loss(prototypes, coefficients, assign, masks, grid_boxes)

    assert torch.equal(batched, looped)
    assert bool(torch.isfinite(batched))


def test_detect_total_is_exactly_the_dual_detection_loss() -> None:
    """``task="detect"`` is bit-for-bit the dual detection loss — the extra term adds exactly zero.

    Catches any segmentation quantity leaking into the detection objective, which
    would silently invalidate the frozen Det-A trajectory and its goldens.
    """
    module = _tiny_module(task="detect")
    module.log = _LogRecorder()  # type: ignore[method-assign]
    batch = _synthetic_batch()

    total = module.training_step(batch, 0)

    assert torch.equal(total, _reference_detection_loss(module, batch))


def _collated_masks(images: Tensor, targets: list[Targets]) -> list[Tensor]:
    """Rasterise a batch's masks the way a segmentation loader's workers do."""
    _, packed = collate_detection(list(zip(images, targets, strict=True)), mask_targets=True)
    masks = unpack_masks(packed)
    assert masks is not None  # the collate was asked to rasterise
    return masks


@pytest.mark.parametrize(
    "counts",
    [
        pytest.param((2, 1), id="ragged"),
        pytest.param((3, 0), id="one-image-empty"),
        pytest.param((0, 0), id="no-instances-at-all"),
    ],
)
def test_loader_rasterises_the_masks_the_step_would_have(counts: tuple[int, ...]) -> None:
    """The collate's transported masks equal what the step rasterises from the same targets.

    The whole point of moving rasterisation into the loader workers is that it is
    the *same* rasterisation — the training process just stops doing it. The
    transport ships bool and the grid comes from the image size rather than from
    the prototypes, so this pins both: a wrong stride would produce a plausible
    mask stack at the wrong resolution, and a bool round-trip that lost a value
    would train against a hole no loss curve would explain.

    The empty cases are here because they are where a per-image split goes wrong:
    an image with no instances must consume no rows of the concatenated stack.
    """
    targets = [_synthetic_targets(count) if count else Targets.empty() for count in counts]
    images = torch.rand(len(counts), 3, _IMG_SIZE, _IMG_SIZE)
    grid = (_IMG_SIZE // _PROTO_STRIDE, _IMG_SIZE // _PROTO_STRIDE)

    masks = _collated_masks(images, targets)

    expected = [instance_mask_targets(target, (_IMG_SIZE, _IMG_SIZE), grid) for target in targets]
    assert [mask.shape for mask in masks] == [reference.shape for reference in expected]
    assert all(torch.equal(mask, reference) for mask, reference in zip(masks, expected, strict=True))


def test_loader_rasterised_masks_give_the_identical_loss() -> None:
    """A batch carrying the loader's masks scores exactly what the in-step fallback scores.

    This is the bit-exactness claim of the move, end to end through the real
    training step rather than through the rasteriser alone: same weights, same
    inputs, one batch with the third element and one without.
    """
    module = _tiny_module()
    module.log = _LogRecorder()  # type: ignore[method-assign]
    images, targets = _synthetic_batch()

    from_loader = module.training_step((images, targets, _collated_masks(images, targets)), 0)

    assert torch.equal(from_loader, module.training_step((images, targets), 0))


def test_masks_rasterised_for_another_grid_are_rejected() -> None:
    """Masks whose grid disagrees with the prototypes raise instead of supervising at the wrong scale.

    The collate derives the grid from the input canvas (A15) while the step reads
    it off the prototypes, so the two agree only as long as the prototype stride
    is what A15 says. Nothing about a half-resolution mask stack is malformed —
    it would broadcast, train, and quietly supervise the wrong pixels — so the
    disagreement has to be an error rather than a shape coincidence.
    """
    module = _tiny_module()
    module.log = _LogRecorder()  # type: ignore[method-assign]
    images, targets = _synthetic_batch()
    coarse = [mask[:, ::2, ::2] for mask in _collated_masks(images, targets)]

    with pytest.raises(ValueError, match="prototypes are"):
        module.training_step((images, targets, coarse), 0)


def _validated_module(batch: tuple[Tensor, list[Targets]] | tuple[Tensor, list[Targets], list[Tensor]]) -> _LogRecorder:
    """Run one validation batch plus the epoch end, and return the log recorder."""
    module = _tiny_module("segment" if len(batch) == 3 else "detect").eval()
    recorder = _LogRecorder()
    module.log = recorder  # type: ignore[method-assign]
    module.validation_step(batch, 0)
    module.on_validation_epoch_end()
    return recorder


def test_validation_logs_a_mask_map_beside_the_box_map() -> None:
    """A segmentation run reports the quality of the branch it exists for.

    Without this the only epoch metric of a ``task="segment"`` run is ``val/mAP``,
    which the mask branch cannot move: a run whose masks were degenerate and a run
    whose masks were perfect would log the identical curve, and the first sign of
    either would be a post-hoc ``scripts/eval_det.py`` pass hours later.
    """
    images, targets = _synthetic_batch()

    recorder = _validated_module((images, targets, _collated_masks(images, targets)))

    assert "val/segm_mAP" in recorder.values
    assert 0.0 <= recorder.values["val/segm_mAP"] <= 1.0


def test_detection_validation_logs_no_mask_metric() -> None:
    """A detection module has no mask branch, so it must not pay for or report one."""
    recorder = _validated_module(_synthetic_batch())

    assert "val/mAP" in recorder.values
    assert "val/segm_mAP" not in recorder.values


def test_mask_map_is_skipped_when_the_batch_carries_no_ground_truth_masks() -> None:
    """A segment module validating a two-element batch scores boxes only, rather than raising.

    ``compute`` on a metric no batch ever updated raises, so the epoch end has to
    know the difference between "no masks this epoch" and "masks scored zero".
    """
    recorder = _validated_module(_synthetic_batch())  # detect module, two-element batch
    assert "val/segm_mAP" not in recorder.values

    module = _tiny_module().eval()
    module.log = _LogRecorder()  # type: ignore[method-assign]
    module.validation_step(_synthetic_batch(), 0)
    epoch_recorder = _LogRecorder()
    module.log = epoch_recorder  # type: ignore[method-assign]
    module.on_validation_epoch_end()

    assert "val/mAP" in epoch_recorder.values
    assert "val/segm_mAP" not in epoch_recorder.values


#: Side of the toy prototype grid the pairing test scores on.
_TOY_GRID = 8
#: Prototype count of the toy segmentation output.
_TOY_COEFFS = 4
#: Anchor count of the toy segmentation output.
_TOY_ANCHORS = 8
#: Prototype logit magnitude — saturated, so every decoded pixel is an unambiguous
#: ``True``/``False`` rather than a value near the 0.5 threshold.
_TOY_LOGIT = 8.0


def _toy_segment_output() -> SegmentOutput:
    """Build a segmentation output whose decoded mask names the anchor it came from.

    Prototype ``k`` is positive on row ``k`` alone, and anchor ``a`` carries the
    one-hot coefficient selecting prototype ``a % _TOY_COEFFS``. A decoded mask is
    therefore a single lit row, and *which* row identifies the anchor whose
    coefficients were used — the one thing a real model's near-uniform masks cannot
    show.
    """
    prototypes = torch.full((1, _TOY_COEFFS, _TOY_GRID, _TOY_GRID), -_TOY_LOGIT)
    for k in range(_TOY_COEFFS):
        prototypes[0, k, k] = _TOY_LOGIT
    coefficients = torch.zeros(1, _TOY_ANCHORS, _TOY_COEFFS)
    for anchor in range(_TOY_ANCHORS):
        coefficients[0, anchor, anchor % _TOY_COEFFS] = 1.0
    zeros = torch.zeros(1, _TOY_ANCHORS, _NUM_CLASSES)
    detect = DualHeadOutput(
        o2m_cls=zeros,
        o2m_box=torch.zeros(1, _TOY_ANCHORS, 4),
        o2o_cls=zeros,
        o2o_box=torch.zeros(1, _TOY_ANCHORS, 4),
        o2o_coeff=coefficients,
    )
    return SegmentOutput(detect=detect, prototypes=prototypes, semantic=None)


def test_scored_masks_are_paired_with_the_boxes_own_anchors() -> None:
    """Detection ``j``'s mask is assembled from the coefficients of the anchor ``j`` came from.

    The decisive test of the metric. A coefficient row gathered by anything other
    than the indices its box was ranked by yields a perfectly plausible mask of the
    wrong object, and every aggregate — the mAP included — stays finite and
    unremarkable, so only a construction where each anchor decodes to a *visibly
    different* mask can catch it. A real module's masks cannot: at initialisation
    every prototype logit sits at the sigmoid's midpoint, and the whole batch
    decodes to empty masks that compare equal however they were paired.
    """
    module = _tiny_module().eval()
    captured: list[tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]] = []
    module._val_segm.update = lambda preds, targets: captured.append((preds, targets))  # type: ignore[union-attr]
    anchor_indices = torch.tensor([[5, 2, 7]])
    canvas = _TOY_GRID * _PROTO_STRIDE
    whole_canvas = torch.tensor([0.0, 0.0, float(canvas), float(canvas)])
    detections = torch.cat([whole_canvas.expand(1, 3, 4), torch.full((1, 3, 2), 0.9)], dim=-1)

    module._update_val_segm(
        _toy_segment_output(),
        detections,
        anchor_indices,
        [_synthetic_targets(1)],
        [torch.zeros(1, _TOY_GRID, _TOY_GRID)],
        (canvas, canvas),
    )

    lit_rows = [sorted(set(mask.nonzero()[:, 0].tolist())) for mask in captured[0][0][0]["masks"]]
    assert lit_rows == [[index % _TOY_COEFFS] for index in anchor_indices[0].tolist()]


def test_scored_masks_and_ground_truth_share_the_prototype_grid() -> None:
    """Predicted and ground-truth masks reach the metric on one grid, at the cap.

    Two frames would make every mask IoU wrong while leaving ``val/segm_mAP``
    finite and plausible, and decoding all 300 decoder rows would cost two thirds
    of the mask work for rows COCO's ``maxDets`` then drops.
    """
    module = _tiny_module().eval()
    module.log = _LogRecorder()  # type: ignore[method-assign]
    images, targets = _synthetic_batch()
    ground_truth_masks = _collated_masks(images, targets)
    captured: list[tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]] = []
    module._val_segm.update = lambda preds, targets: captured.append((preds, targets))  # type: ignore[union-attr]

    module.validation_step((images, targets, ground_truth_masks), 0)

    preds, scored_truth = captured[0]
    grid = (_IMG_SIZE // _PROTO_STRIDE, _IMG_SIZE // _PROTO_STRIDE)
    assert [pred["masks"].shape for pred in preds] == [(_VAL_SEGM_MAX_DET, *grid)] * _BATCH_SIZE
    assert all(
        torch.equal(entry["masks"], truth.bool()) for entry, truth in zip(scored_truth, ground_truth_masks, strict=True)
    )
