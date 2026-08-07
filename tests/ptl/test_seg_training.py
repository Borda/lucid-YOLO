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
from lucid_yolo.models.heads.detect import decode_ltrb
from lucid_yolo.models.heads.proto import assemble_masks
from lucid_yolo.ptl import DetectionLitModule, pad_targets

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

    scored = DetectionLitModule._branch_mask_loss(prototypes, coefficients, assign, [masks], grid_boxes)

    assert torch.equal(scored, assigned)
    assert not torch.isclose(assigned, naive)


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
