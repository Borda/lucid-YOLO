# SPDX-License-Identifier: Apache-2.0
"""Tests for the oriented supervision wired into the training step (WP-088).

Phase 8 built every oriented component standalone and wired none of them into
training: ``task="obb"`` constructed the angle stems, then trained the plain
detection objective while they never received a gradient. Every component gate
still passed, because each component was correct in isolation. These tests
therefore aim at the **wiring** rather than at the components again:

* the angle stems of **both** head branches carry gradient after one step — the
  test that would have caught the original defect;
* each of the three oriented gains has leverage on the total, so a term that is
  connected but weightless (the same defect wearing a disguise) fails here;
* the rotated and angle terms gather their targets by the *assignment*, not by
  positional order, so an anchor's orientation cannot be scored against another
  instance's box;
* ``task="detect"`` reproduces a snapshot captured **before** this work package
  touched the shared modules — not merely "the detection tests pass" — to the
  tolerance an architecture's choice of summation order forces on it;
* the oriented module's state-dict keys are the detection module's plus the angle
  stems, so the stages stayed flat and the accepted checkpoints keep loading;
* rotated ground truth survives the loader transport into
  :func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map`, difficult flags included.

The module is built at n-scale multipliers with a low channel cap and a 160-px
input so the real stack runs in a couple of seconds on CPU. Ground truths are
**elongated and rotated** on purpose: at initialisation the head emits
``theta ~ 0``, and both ``sin^2(2 d_theta)`` and ProbIoU have a vanishing angular
gradient at ``d_theta = 0``, so a square or axis-aligned fixture would report zero
gradient on perfectly wired code.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from lucid_yolo.assign.tal import AssignResult
from lucid_yolo.data.coco import build_scale_policy
from lucid_yolo.data.rotated_geom import canonicalize, rboxes_to_polygons
from lucid_yolo.data.targets import Targets
from lucid_yolo.decode.common import rboxes_to_letterboxed_original
from lucid_yolo.eval.dota_eval import evaluate_rotated_map
from lucid_yolo.losses.oriented_loss import oriented_branch_terms
from lucid_yolo.models.heads.detect import decode_ltrb
from lucid_yolo.ptl import DetectionLitModule, pad_rboxes, pad_targets
from lucid_yolo.ptl.datamodule import _TrainPipeline, collate_detection, unpack_targets

#: Class count of the tiny test head.
_NUM_CLASSES = 4
#: Square input side; divisible by every stride for an integer anchor grid.
_IMG_SIZE = 160
#: Images per synthetic batch.
_BATCH_SIZE = 2
#: Loss gains the tiny module is built with.
_BOX_GAIN, _CLS_GAIN, _L1_GAIN, _ANGLE_GAIN, _ALPHA = 7.5, 0.5, 6.0, 1.0, 0.5
#: Column count of a long-edge rotated box.
_RBOX_DIM = 5

#: The pre-WP-088 ``task="detect"`` training-step snapshot and the run that produced it.
_SNAPSHOT_FILE = Path(__file__).parent / "prechange_detect_step.json"
#: Relative tolerance on a term's leverage. The difference of two float32 totals near
#: 22 resolves to ~2e-6, so a term of order 1e-4 is recovered to about a percent; this
#: bounds that cancellation rather than the correctness of the arithmetic.
_LEVERAGE_RTOL = 0.05

#: Relative tolerance on the snapshot's scalars. Set from two measured cross-architecture
#: datapoints: the snapshot was captured on arm64, and x86-64 CI reproduces its total as
#: ``22.80112076`` against the recorded ``22.80110550`` — 1.5e-5 apart — while the same
#: run reproduces the ``o2o_l1`` component as ``0.024623577`` against ``0.024620384``,
#: 1.3e-4 apart. Both come from a reduction order the two builds choose differently; the
#: component drifts relatively further because it is a small term summed out of much
#: larger ones, so the same absolute wobble lands on a smaller magnitude. 1e-3 leaves
#: eight times the worse of the two and still fails any change to the objective, which
#: moves these numbers by percent rather than by ulps.
_SNAPSHOT_RTOL = 1e-3
#: Absolute floor beneath the relative tolerance. Two components are ``o2m`` terms of a
#: batch that assigns almost nothing, of order 1e-9 and 1e-8; a relative bound on those
#: would compare float32 noise against float32 noise and mean nothing.
_SNAPSHOT_ATOL = 1e-6

#: Relative tolerance on the snapshot's gradient norm — looser than the scalars above,
#: and by argument rather than by measurement: no cross-architecture value for it has
#: been observed, and it is a float64 reduction over 700+ tensors that each carry the
#: same class of drift. A real change to the objective moves it by percent.
_GRAD_NORM_RTOL = 1e-4


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed torch before each test so model init and synthetic batches are reproducible."""
    torch.manual_seed(0)


@pytest.fixture
def single_threaded() -> Iterator[None]:
    """Pin torch's intra-op thread count to one, restoring it afterwards.

    Thread count changes how reductions are split and so the summation order of the
    dense terms; a bit-identity golden that did not pin it would be a golden about the
    machine's core count. Restored on the way out so no other test inherits the pin.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class _LogRecorder:
    """Stand-in for ``LightningModule.log`` that records every logged scalar."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def __call__(self, name: str, value: Tensor, **kwargs: object) -> None:
        """Record ``value`` under ``name``, ignoring Lightning's keyword arguments."""
        del kwargs
        self.values[name] = float(value.detach())


def _tiny_module(task: str = "obb", **overrides: float) -> DetectionLitModule:
    """Build an n-scale module with a low channel cap for fast CPU tests.

    Examples:
        >>> module = _tiny_module()
        >>> module.task
        'obb'
    """
    gains: dict[str, float] = {
        "box_gain": _BOX_GAIN,
        "cls_gain": _CLS_GAIN,
        "l1_gain": _L1_GAIN,
        "angle_gain": _ANGLE_GAIN,
        "alpha": _ALPHA,
    }
    gains.update(overrides)
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=_NUM_CLASSES, task=task, **gains)


def _oriented_targets(num_boxes: int, difficult: Tensor | None = None) -> Targets:
    """Build elongated, rotated ground truths with envelopes fitted to the same corners.

    Elongation and rotation are load-bearing, not incidental: an axis-aligned or square
    target sits at the stationary point of both angular terms, where a correctly wired
    angle stem legitimately receives zero gradient.

    Examples:
        >>> targets = _oriented_targets(2)
        >>> targets.boxes.shape, targets.rboxes.shape
        (torch.Size([2, 4]), torch.Size([2, 5]))
    """
    centre = torch.rand(num_boxes, 2) * 60.0 + 50.0
    long_edge = torch.rand(num_boxes) * 30.0 + 30.0
    short_edge = torch.rand(num_boxes) * 8.0 + 8.0
    theta = torch.rand(num_boxes) * 0.9 + 0.2
    rboxes = canonicalize(torch.stack([centre[:, 0], centre[:, 1], long_edge, short_edge, theta], dim=1))
    corners = rboxes_to_polygons(rboxes)
    boxes = torch.cat([corners.amin(dim=1), corners.amax(dim=1)], dim=1)
    return Targets(
        boxes=boxes,
        labels=torch.randint(0, _NUM_CLASSES, (num_boxes,)),
        rboxes=rboxes,
        difficult=torch.zeros(num_boxes, dtype=torch.bool) if difficult is None else difficult,
    )


def _oriented_batch() -> tuple[Tensor, list[Targets]]:
    """Build a two-image batch with ragged (2 and 1) instance counts.

    Examples:
        >>> images, targets = _oriented_batch()
        >>> images.shape, [t.boxes.shape[0] for t in targets]
        (torch.Size([2, 3, 160, 160]), [2, 1])
    """
    images = torch.randn(_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)
    return images, [_oriented_targets(2), _oriented_targets(1)]


def _has_gradient(module: nn.Module) -> bool:
    """Return whether any parameter of ``module`` carries a non-zero gradient.

    Examples:
        >>> layer = nn.Linear(2, 1)
        >>> _has_gradient(layer)
        False
        >>> layer(torch.ones(1, 2)).backward()
        >>> _has_gradient(layer)
        True
    """
    return any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in module.parameters())


def test_gradients_reach_both_branches_angle_stems() -> None:
    """One ``task="obb"`` step leaves non-zero gradients on the o2o *and* o2m angle stems.

    This is the test that would have caught the Phase 8 defect: the stems existed,
    were constructed, appeared in the state dict and in the parameter count, and
    received no gradient because nothing in the objective read them.
    """
    module = _tiny_module()
    module.log = _LogRecorder()  # type: ignore[method-assign]

    module.training_step(_oriented_batch(), 0).backward()

    assert _has_gradient(module.head.o2o.angle_stems)
    assert _has_gradient(module.head.o2m.angle_stems)


@pytest.mark.parametrize(
    ("gain", "term", "value"),
    [
        pytest.param("box_gain", "train/rbox", _BOX_GAIN, id="rotated-iou-term"),
        pytest.param("l1_gain", "train/rl1", _L1_GAIN, id="rotated-l1-term"),
        pytest.param("angle_gain", "train/angle", _ANGLE_GAIN, id="angle-term"),
    ],
)
def test_each_oriented_gain_enters_the_total_at_its_weight(gain: str, term: str, value: float) -> None:
    """Zeroing any one oriented gain drops exactly ``gain * term`` from the total.

    A term that is wired but weightless is the Phase 8 defect wearing a disguise: the
    gradient test above would still pass through the other two terms while this one
    contributed nothing to what the optimizer minimises. The assertion is the *size*
    of the change rather than merely its presence, because at initialisation the total
    is dominated by the classification term — a bare inequality at any default
    tolerance would call the angle term absent while it was working.
    """
    weighted = _tiny_module()
    zeroed = _tiny_module(**{gain: 0.0})
    zeroed.load_state_dict(weighted.state_dict())
    recorder = _LogRecorder()
    weighted.log = recorder  # type: ignore[method-assign]
    zeroed.log = _LogRecorder()  # type: ignore[method-assign]
    batch = _oriented_batch()

    with_gain = weighted.training_step(batch, 0)

    without_gain = zeroed.training_step(batch, 0)
    dropped = float(with_gain.detach()) - float(without_gain.detach())
    assert dropped != 0.0
    assert dropped == pytest.approx(value * recorder.values[term], rel=_LEVERAGE_RTOL)


def test_the_angle_gain_default_is_the_registered_a22_value() -> None:
    """The angle term's default weight is A22's measured ``0.25``, not the assumed ``1.0``.

    Every other test here passes its gains explicitly, so until this one nothing read the
    default at all — it could have drifted back with the whole suite green. That matters
    more than for a hyperparameter nobody argued over: A22 is a registered gap R1 never
    fills, and ``0.25`` is the value WP-093 *measured* rather than assumed, after the
    inherited ``1.0`` was shown to destabilise angle regression on elongated targets (the
    oriented overfit cleared its floor on 1 of 5 seeds at ``1.0`` against 4 of 5 at
    ``0.25``, with the run-to-run spread falling from 0.667 to 0.096).
    """
    assert DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=1, task="obb")._angle_gain == 0.25


def test_obb_state_dict_is_the_detection_state_dict_plus_the_angle_stems() -> None:
    """Dropping the angle-stem keys from an ``obb`` module leaves exactly a ``detect`` module's.

    Pins the composition rule WP-087 established: the oriented module builds through
    :func:`~lucid_yolo.models.build.build_detection_stages` and keeps its stages flat.
    Nesting an :class:`~lucid_yolo.models.build.OrientedDetector` inside it would
    prefix every key with that attribute name and invalidate the accepted Det-smoke
    checkpoint — a change no shape, count or loss value would reveal.
    """
    oriented = _tiny_module()
    detection = _tiny_module(task="detect")

    keys = [key for key in oriented.state_dict() if ".angle_stems." not in key]

    assert keys == list(detection.state_dict())
    assert any(".angle_stems." in key for key in oriented.state_dict())


def test_detect_step_reproduces_the_pre_change_snapshot(single_threaded: None) -> None:
    """A ``task="detect"`` step reproduces the total and every component captured before this WP.

    WP-088 edited modules that also sit on the detection path — the dual loss, the
    target container, four geometric transforms, the COCO reader and the decode
    helpers. "The detection tests still pass" would not distinguish an objective that
    moved from one that did not; the frozen numbers do. The snapshot was produced by
    running this exact step against the ``fcf3040`` source tree, exported with
    ``git archive`` rather than remembered.

    **The comparison is to a tolerance, and it did not start that way.** Three values
    of the same total have now been observed from source trees that compute the same
    arithmetic: ``22.80110550`` on arm64 with one intra-op thread, ``22.80111694`` on
    arm64 with twelve, and ``22.80112076`` on x86-64 CI. Summation order is a property
    of the build and the core count, not of the objective, so bit-identity holds within
    an architecture and cannot hold across one. This test was written asserting equality
    and passed for the whole of Phase 8 — because until the 0.3.0 push it had only ever
    run on the machine its snapshot came from. What it can still catch is any change to
    the objective, which moves these numbers by percent; what it can no longer catch is
    a change of a single rounding step, and no amount of tolerance choosing recovers
    that across architectures.

    The ``single_threaded`` fixture therefore no longer decides pass from fail — the
    thread-count spread is well inside ``_SNAPSHOT_RTOL``. It is kept because it removes
    the one source of drift that *is* under this suite's control, leaving the tolerance
    to absorb only the architecture; without it the margin would be spent on whichever
    test last touched ``torch.set_num_threads`` (the WP-079 loader worker init does).
    """
    snapshot = json.loads(_SNAPSHOT_FILE.read_text())
    torch.manual_seed(snapshot["model_seed"])
    module = DetectionLitModule(
        depth=snapshot["depth"],
        width=snapshot["width"],
        max_channels=snapshot["max_channels"],
        num_classes=snapshot["num_classes"],
        box_gain=snapshot["gains"]["box"],
        cls_gain=snapshot["gains"]["cls"],
        l1_gain=snapshot["gains"]["l1"],
        alpha=snapshot["gains"]["alpha"],
    )
    recorder = _LogRecorder()
    module.log = recorder  # type: ignore[method-assign]
    torch.manual_seed(snapshot["batch_seed"])
    images = torch.randn(_BATCH_SIZE, 3, snapshot["img_size"], snapshot["img_size"])
    targets = [_snapshot_targets(2), _snapshot_targets(1)]

    total = module.training_step((images, targets), 0)
    total.backward()

    assert float(total.detach()) == pytest.approx(snapshot["total"], rel=_SNAPSHOT_RTOL, abs=_SNAPSHOT_ATOL)
    assert recorder.values == pytest.approx(snapshot["components"], rel=_SNAPSHOT_RTOL, abs=_SNAPSHOT_ATOL)
    assert len(module.state_dict()) == snapshot["state_dict_keys"]
    assert _gradient_norm(module) == pytest.approx(snapshot["grad_l2"], rel=_GRAD_NORM_RTOL)


def _snapshot_targets(num_boxes: int) -> Targets:
    """Rebuild the axis-aligned targets the pre-change snapshot was captured over.

    Examples:
        >>> targets = _snapshot_targets(3)
        >>> targets.boxes.shape, targets.labels.shape
        (torch.Size([3, 4]), torch.Size([3]))
    """
    top_left = torch.rand(num_boxes, 2) * 80.0
    size = torch.rand(num_boxes, 2) * 40.0 + 10.0
    return Targets(
        boxes=torch.cat([top_left, top_left + size], dim=1),
        labels=torch.randint(0, _NUM_CLASSES, (num_boxes,)),
    )


def _gradient_norm(module: nn.Module) -> float:
    """Return the float64 L2 norm over every parameter gradient of ``module``.

    Examples:
        >>> layer = nn.Linear(1, 1, bias=False)
        >>> with torch.no_grad():
        ...     _ = layer.weight.fill_(1.0)
        >>> layer(torch.tensor([[3.0]])).backward()
        >>> _gradient_norm(layer)
        3.0
    """
    squares = torch.stack(
        [parameter.grad.detach().flatten().double().pow(2).sum() for parameter in module.parameters()]
    )
    return float(squares.sum().sqrt())


def test_obb_step_needs_one_rotated_box_per_instance() -> None:
    """Axis-aligned targets under ``task="obb"`` raise instead of training on nothing.

    The silent-subset failure of this path: a loader left in axis-aligned mode ships
    empty ``rboxes``, and without the guard the rotated terms would score zero
    positives forever while the loss curve looked entirely ordinary.
    """
    module = _tiny_module()
    module.log = _LogRecorder()  # type: ignore[method-assign]
    images, targets = _oriented_batch()
    boxes_only = [Targets(boxes=target.boxes, labels=target.labels) for target in targets]

    with pytest.raises(ValueError, match="rotated boxes"):
        module.training_step((images, boxes_only), 0)


def test_image_without_instances_gives_a_finite_loss() -> None:
    """An empty batch yields finite oriented terms rather than a ``0 / 0`` NaN.

    With no ground truth the padded rotated axis has length zero, which no ``gather``
    can index; the term has to answer with an exact zero that is still connected to
    the predictions.
    """
    module = _tiny_module()
    recorder = _LogRecorder()
    module.log = recorder  # type: ignore[method-assign]

    loss = module.training_step((torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE), [Targets.empty()]), 0)

    assert bool(torch.isfinite(loss))
    assert recorder.values["train/rbox"] == 0.0
    assert recorder.values["train/angle"] == 0.0


def test_rotated_candidacy_reaches_the_assigner() -> None:
    """``gt_rboxes`` threads through the dual loss into both assigners' candidate filter.

    A25 restricts the rotated ground truth to candidacy, so the only observable effect
    is *which* anchors become positive. A thin diagonal box whose axis-aligned envelope
    is large is where the two tests disagree most, and where a dropped argument shows.
    """
    module = _tiny_module()
    thin = Targets(
        boxes=torch.tensor([[20.0, 20.0, 140.0, 140.0]]),
        labels=torch.tensor([1]),
        rboxes=torch.tensor([[80.0, 80.0, 160.0, 10.0, 0.7854]]),
    )
    images = torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE)
    arguments = _dense_inputs(module, images, thin)

    rotated = module.loss(*arguments, gt_rboxes=pad_rboxes([thin]))

    axis_aligned = module.loss(*arguments)
    assert int(rotated.o2m_assign.fg_mask.sum()) < int(axis_aligned.o2m_assign.fg_mask.sum())
    assert int(rotated.o2m_assign.fg_mask.sum()) > 0


def _dense_inputs(module: DetectionLitModule, images: Tensor, target: Targets) -> tuple[Tensor, ...]:
    """Return the positional arguments ``DualBranchLoss`` takes for a one-image batch.

    Examples:
        >>> module = _tiny_module()
        >>> images = torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE)
        >>> inputs = _dense_inputs(module, images, _oriented_targets(1))
        >>> len(inputs)
        9
    """
    head_out = module(images)
    anchor_points, strides = module._anchor_grid(images.shape[-2], images.shape[-1], images.device)
    gt_boxes, gt_labels, gt_mask = pad_targets([target])
    return (
        head_out.o2m_cls,
        decode_ltrb(head_out.o2m_box, anchor_points, strides),
        head_out.o2o_cls,
        decode_ltrb(head_out.o2o_box, anchor_points, strides),
        anchor_points,
        gt_boxes,
        gt_labels,
        gt_mask,
        strides,
    )


def test_oriented_terms_follow_the_assignment_not_the_positive_order() -> None:
    """Positive ``k``'s rotated box is scored against ``gt_index[k]``, not against instance ``k``.

    The decisive pairing test, and the one no loss value can stand in for: pairing an
    anchor with the wrong instance produces a perfectly finite, perfectly plausible
    number. Each positive is given the prediction that *matches* the instance its
    ``gt_index`` names, so the assignment pairing scores an exact zero and the naive
    positional pairing — the same two terms, merely swapped — does not. Predictions
    identical across anchors would make the two orders permutations of one sum and the
    test vacuous.
    """
    strides = torch.full((4,), 8.0)
    first = [20.0, 20.0, 40.0, 10.0, 0.2]
    second = [60.0, 60.0, 30.0, 12.0, 1.1]
    filler = [0.0, 0.0, 1.0, 1.0, 0.0]
    pred = torch.tensor([[second, filler, first, filler]])
    theta = torch.tensor([[second[4], 0.0, first[4], 0.0]])
    gt_rboxes = torch.tensor([[first, second]])
    assign = AssignResult(
        fg_mask=torch.tensor([[True, False, True, False]]),
        gt_index=torch.tensor([[1, -1, 0, -1]]),  # positive 0 -> instance 1, positive 2 -> instance 0
        target_labels=torch.tensor([[0, -1, 0, -1]]),
        target_boxes=torch.zeros(1, 4, 4),
        align_weights=torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
    )
    swapped = AssignResult(
        fg_mask=assign.fg_mask,
        gt_index=torch.tensor([[0, -1, 1, -1]]),  # the naive positional pairing
        target_labels=assign.target_labels,
        target_boxes=assign.target_boxes,
        align_weights=assign.align_weights,
    )

    scored = oriented_branch_terms(pred, theta, gt_rboxes, assign, strides)

    naive = oriented_branch_terms(pred, theta, gt_rboxes, swapped, strides)
    assert float(scored.rbox) == 0.0
    assert float(scored.rl1) == 0.0
    assert float(scored.angle) == 0.0
    assert float(naive.rbox) > 0.0
    assert float(naive.rl1) > 0.0
    assert float(naive.angle) > 0.0


def _validated_module(batch: tuple[Tensor, list[Targets]]) -> tuple[DetectionLitModule, _LogRecorder]:
    """Run one validation batch plus the epoch end on an ``obb`` module.

    Examples:
        >>> callable(_validated_module)  # needs a live _LogRecorder-backed module
        True
    """
    module = _tiny_module().eval()
    module.log = _LogRecorder()  # type: ignore[method-assign]
    module.validation_step(batch, 0)
    recorder = _LogRecorder()
    module.log = recorder  # type: ignore[method-assign]
    module.on_validation_epoch_end()
    return module, recorder


def test_validation_logs_the_rotated_map_and_not_the_box_map() -> None:
    """An ``obb`` run reports the metric of the branch it exists for, and only that one.

    ``val/mAP`` reads the A44 composition's pre-rotation rectangle: a run whose
    orientations were random and one whose orientations were right log the identical
    curve. Carrying it beside the rotated figure is not a free second opinion — its
    accumulator walks every detection of every batch on the CPU to produce a number
    that cannot answer the question the run is asking (WP-102).
    """
    _, recorder = _validated_module(_oriented_batch())

    assert "val/rotated_mAP50" in recorder.values
    assert "val/rotated_mAP" in recorder.values
    assert 0.0 <= recorder.values["val/rotated_mAP50"] <= 1.0
    assert "val/mAP" not in recorder.values


def test_detection_validation_logs_no_rotated_metric() -> None:
    """A detection module has no angle branch, so it must not report a rotated metric."""
    module = _tiny_module(task="detect").eval()
    module.log = _LogRecorder()  # type: ignore[method-assign]
    module.validation_step(_oriented_batch(), 0)
    recorder = _LogRecorder()
    module.log = recorder  # type: ignore[method-assign]

    module.on_validation_epoch_end()

    assert "val/mAP" in recorder.values
    assert "val/rotated_mAP50" not in recorder.values


def test_the_epoch_metric_clears_its_buffers() -> None:
    """The accumulators are emptied at the epoch end, so no epoch scores another's detections."""
    module, _ = _validated_module(_oriented_batch())

    assert module._val_rotated_preds == []
    assert module._val_rotated_targets == []


def test_rotated_targets_survive_the_loader_transport_into_the_metric() -> None:
    """Rotated ground truth reaches ``evaluate_rotated_map`` through the loader's packed batch.

    The whole target-side path in one assertion: pack for the worker boundary, unpack
    on arrival, run the validation step that builds the metric's target dicts, then
    score a prediction built from the *original* rotated boxes. Anything that dropped,
    reordered or re-fitted a rotated box on the way makes this less than a perfect
    match, while every intermediate shape would still look right.
    """
    targets = [_oriented_targets(2), _oriented_targets(1)]
    images = torch.rand(_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)
    restored = unpack_targets(collate_detection(list(zip(images, targets, strict=True)))[1])
    module = _tiny_module().eval()
    module.log = _LogRecorder()  # type: ignore[method-assign]

    module.validation_step((images, restored), 0)

    collected = module._val_rotated_targets
    perfect = [
        {"rboxes": target.rboxes, "scores": torch.ones(target.rboxes.shape[0]), "labels": target.labels}
        for target in targets
    ]
    assert [tuple(entry["rboxes"].shape) for entry in collected] == [(2, _RBOX_DIM), (1, _RBOX_DIM)]
    assert evaluate_rotated_map(perfect, collected)["map_50"] == pytest.approx(1.0)


def test_difficult_flags_reach_the_metric_through_the_loader() -> None:
    """An R18 difficult flag survives packing, unpacking and the letterbox (A48).

    Filtering difficult instances at load — or dropping the flag anywhere between the
    reader and the metric — turns every ignorable detection into a false positive and
    silently depresses the score. The flag has no other consumer, so nothing else in
    the suite would notice it going missing.
    """
    flagged = _oriented_targets(3, difficult=torch.tensor([True, False, True]))
    images = torch.rand(1, 3, _IMG_SIZE, _IMG_SIZE)

    restored = unpack_targets(collate_detection([(images[0], flagged)])[1])

    assert torch.equal(restored[0].difficult, flagged.difficult)
    module = _tiny_module().eval()
    module.log = _LogRecorder()  # type: ignore[method-assign]
    module.validation_step((images, restored), 0)
    assert torch.equal(module._val_rotated_targets[0]["difficult"], flagged.difficult)


def test_an_oriented_train_pipeline_suppresses_copy_paste() -> None:
    """An oriented loader never attempts copy-paste, which cannot act on rotated targets.

    Found by the acceptance run rather than by any component test, and it could only
    be found there: ``CopyPaste`` refuses rotated boxes correctly (WP-056 — it moves
    rasterised polygon masks and the oriented path deliberately carries no polygons),
    the datamodule composes it correctly for detection, and neither knew about the
    other. At the ``n`` policy's ``copy_paste = 0.1`` the first oriented epoch dies
    partway through, which is a crash rather than a silent defect — but a crash the
    wiring is supposed to prevent, since the composition is what chooses the
    transforms.

    The draw is still consumed (``_draw() >= 0.0`` is always true), so suppressing the
    transform does not shift the augmentation RNG stream that WP-079 pinned.
    """
    policy = build_scale_policy("n")

    oriented = _TrainPipeline(_FakeBase(), _IMG_SIZE, policy, seed=0, oriented=True)

    assert policy["copy_paste"] > 0.0  # otherwise this test asserts nothing
    assert oriented._copy_paste_prob == 0.0
    assert _TrainPipeline(_FakeBase(), _IMG_SIZE, policy, seed=0)._copy_paste_prob == policy["copy_paste"]
    assert oriented._mixup_prob == policy["mixup"]  # mixup carries rboxes (WP-058) and stays


class _FakeBase:
    """Minimal stand-in for the base dataset: a length and the mirror pairing it declares.

    Both readers publish ``keypoint_flip_pairs`` because A64 makes the left/right swap the
    dataset's statement to make, and the pipeline reads it when it builds the flip. An
    oriented base declares ``None``, which is what a rotated-box dataset has to say: it
    carries no landmark table for a swap to act on.
    """

    keypoint_flip_pairs: list[tuple[int, int]] | None = None

    def __len__(self) -> int:
        """Return a nominal image count."""
        return 4


def test_the_letterbox_inverse_returns_rotated_boxes_unturned() -> None:
    """Un-letterboxing an oriented detection scales it and leaves ``theta`` alone.

    A letterbox is an isotropic scale plus a translation, so the inverse is exact
    rather than a re-fit. Asserting ``theta`` explicitly is the point: an inverse that
    warped the angle would still return plausible boxes at plausible sizes.
    """
    detections = torch.tensor([[[40.0, 60.0, 32.0, 8.0, 0.7, 0.9, 3.0]]])

    mapped = rboxes_to_letterboxed_original(detections, orig_size=(80, 160), letterboxed_size=(160, 160))

    assert mapped[0, 0, 4] == detections[0, 0, 4]
    assert mapped[0, 0, 5:].tolist() == detections[0, 0, 5:].tolist()
    assert mapped[0, 0, :4].tolist() == pytest.approx([40.0, 20.0, 32.0, 8.0])
