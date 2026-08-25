# SPDX-License-Identifier: Apache-2.0
"""Tests for the keypoint supervision wired into the training step (WP-132).

Phase 12 built every keypoint component standalone and wired none of them into
training: the head could build point stems, the RLE loss existed with zero call
sites, and no ``task`` string reached either. That is the same shape of gap
WP-088 closed for the oriented path, so these tests aim where its own do — at
the **wiring** rather than at the components again:

* the point stems of **both** head branches, and the RLE flow's own parameters,
  carry gradient after one step — the test that would have caught a term that is
  computed and then discarded;
* the keypoint gain has leverage on the total at exactly its weight, so a term
  that is connected but weightless fails here;
* at ``keypoint_gain=0`` the step reproduces a ``task="detect"`` step **bit for
  bit** — the task adds and never disturbs (unlike ``"obb"``, which replaces),
  and "the detection tests still pass" would not distinguish the two;
* the RLE term gathers its targets by the *assignment*, not by positional order,
  so an anchor cannot be supervised towards another instance's pose;
* A66's visibility rule survives the padding path end to end: a point marked
  ``v == 0`` moves the term by nothing at all, however far away it is put;
* the keypoints module's state-dict keys are the detection module's plus the
  point stems and the flow, so the stages stayed flat and accepted checkpoints
  keep loading;
* keypoints survive the loader's packed transport, including a batch mixing
  annotated and instance-free images;
* the datamodule hands ``HorizontalFlip`` the pairing the *dataset* derived from
  its own annotation schema (A64) — the assertion whose absence let a mirrored
  sample train with left and right identities swapped the wrong way, silently,
  since supplying no pairing is not an error but a wrong answer.

The module is built at n-scale multipliers with a low channel cap and a 160-px
input so the real stack runs in a couple of seconds on CPU. ``K`` is 3 rather
than COCO's 17 for the same reason: the point axis is the one thing every shape
here is generic over, and three points exercise it as well as seventeen.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor, nn

from lucid_yolo.assign.tal import AssignResult
from lucid_yolo.data.targets import Targets
from lucid_yolo.losses.keypoint_nll_loss import LaplaceNLLLoss
from lucid_yolo.losses.rle_loss import RLELoss
from lucid_yolo.ptl import DetectionLitModule, normalize_keypoints_to_box, pad_keypoints
from lucid_yolo.ptl.datamodule import DetectionDataModule, collate_detection, unpack_targets
from lucid_yolo.ptl.module import _StepContext

if TYPE_CHECKING:
    from pathlib import Path

#: Class count of the tiny test head.
_NUM_CLASSES = 4
#: Point count the tiny test head predicts.
_NUM_KEYPOINTS = 3
#: Square input side; divisible by every stride for an integer anchor grid.
_IMG_SIZE = 160
#: Images per synthetic batch.
_BATCH_SIZE = 2
#: Loss gains the tiny module is built with.
_BOX_GAIN, _CLS_GAIN, _L1_GAIN, _KEYPOINT_GAIN, _ALPHA = 7.5, 0.5, 6.0, 1.0, 0.5

#: Relative tolerance on the keypoint term's leverage. The difference of two float32
#: totals of order 1e3 resolves to ~1e-4, so a term recovered from that difference is
#: good to a few parts in 1e6; this bounds the cancellation, not the arithmetic.
_LEVERAGE_RTOL = 1e-3


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


def _tiny_module(task: str = "keypoints", keypoint_loss: str = "rle", **overrides: float) -> DetectionLitModule:
    """Build an n-scale module with a low channel cap for fast CPU tests.

    ``keypoint_loss`` is spelled out rather than left to ``overrides`` because it is
    the one non-numeric knob here, and its default repeats the module's own so every
    caller predating WP-135 builds exactly what it did before.

    Examples:
        >>> module = _tiny_module()
        >>> module.task
        'keypoints'
        >>> type(_tiny_module(keypoint_loss="laplace_nll").rle_loss).__name__
        'LaplaceNLLLoss'
    """
    gains: dict[str, float] = {
        "box_gain": _BOX_GAIN,
        "cls_gain": _CLS_GAIN,
        "l1_gain": _L1_GAIN,
        "keypoint_gain": _KEYPOINT_GAIN,
        "alpha": _ALPHA,
    }
    gains.update(overrides)
    module = DetectionLitModule(
        depth=0.34,
        width=0.25,
        max_channels=256,
        num_classes=_NUM_CLASSES,
        task=task,
        num_keypoints=_NUM_KEYPOINTS if task == "keypoints" else None,
        keypoint_loss=keypoint_loss,
        **gains,
    )
    module.log = _LogRecorder()  # type: ignore[method-assign]
    return module


def _posed_targets(num_boxes: int, visibility: Tensor | None = None) -> Targets:
    """Build boxes with one point set per instance, inside the canvas.

    The points are placed independently of the boxes on purpose: nothing in the
    objective ties a point to its own box's interior, and a fixture that put them
    at box centres would let a decode that ignored the anchor grid look correct.

    Examples:
        >>> targets = _posed_targets(2)
        >>> targets.boxes.shape, targets.keypoints.shape, targets.keypoint_vis.shape
        (torch.Size([2, 4]), torch.Size([2, 3, 2]), torch.Size([2, 3]))
    """
    top_left = torch.rand(num_boxes, 2) * 80.0
    size = torch.rand(num_boxes, 2) * 40.0 + 10.0
    points = torch.rand(num_boxes, _NUM_KEYPOINTS, 2) * float(_IMG_SIZE)
    return Targets(
        boxes=torch.cat([top_left, top_left + size], dim=1),
        labels=torch.randint(0, _NUM_CLASSES, (num_boxes,)),
        keypoints=points,
        keypoint_vis=torch.full((num_boxes, _NUM_KEYPOINTS), 2, dtype=torch.int64)
        if visibility is None
        else visibility,
    )


def _posed_batch() -> tuple[Tensor, list[Targets]]:
    """Build a two-image batch with ragged (2 and 1) instance counts.

    Examples:
        >>> images, targets = _posed_batch()
        >>> images.shape, [t.boxes.shape[0] for t in targets]
        (torch.Size([2, 3, 160, 160]), [2, 1])
    """
    images = torch.randn(_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)
    return images, [_posed_targets(2), _posed_targets(1)]


def _boxes_only(targets: list[Targets]) -> list[Targets]:
    """Strip the point channels, leaving what a plain detection loader would hand over.

    Examples:
        >>> [t.keypoints.shape[0] for t in _boxes_only([_posed_targets(2)])]
        [0]
    """
    return [Targets(boxes=target.boxes, labels=target.labels) for target in targets]


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


class TestHeadConstruction:
    """The stems and the flow a ``task="keypoints"`` module is supposed to build."""

    def test_builds_point_stems_on_both_branches(self) -> None:
        """A keypoints module builds keypoint stems on the one-to-one *and* one-to-many branches.

        The one-to-one branch is what a NMS-free pose decode reads and the one-to-many
        branch is what carries the dense training signal; a module that built only one
        of them would train and then deploy different things.
        """
        module = _tiny_module()

        assert module.head.o2o.keypoint_stems is not None
        assert module.head.o2m.keypoint_stems is not None
        assert module.head.num_keypoints == _NUM_KEYPOINTS

    def test_carries_the_rle_flow_as_a_submodule(self) -> None:
        """The RLE flow is registered on the module, so it is optimized and checkpointed.

        R14's flow is the one loss in the project holding parameters. Held anywhere but
        the module tree it would never reach ``self.parameters()``, and the density the
        loss is written around would stay at its initialisation for the whole run while
        every loss value still looked ordinary.
        """
        module = _tiny_module()

        assert module.rle_loss is not None
        flow_parameters = {name for name, _ in module.named_parameters() if name.startswith("rle_loss.")}
        assert flow_parameters
        assert flow_parameters <= set(module.state_dict())

    def test_other_tasks_stay_keypoint_free(self) -> None:
        """A detection module builds no point stems and no flow, even given a ``num_keypoints``.

        ``num_keypoints`` names a property of the *dataset*, and a config that carried it
        over from a pose run must not quietly grow stems on a detection model — that would
        move the state-dict keys and stop the accepted checkpoints loading.
        """
        module = DetectionLitModule(
            depth=0.34, width=0.25, max_channels=256, num_classes=_NUM_CLASSES, num_keypoints=_NUM_KEYPOINTS
        )

        assert module.head.o2o.keypoint_stems is None
        assert module.rle_loss is None

    def test_the_task_requires_an_explicit_point_count(self) -> None:
        """``task="keypoints"`` without ``num_keypoints`` is refused at construction.

        There is no defensible default: COCO person is 17 points and another pose schema
        is not, so a fallback would be a silent claim about data the module never saw. The
        failure it buys is a head built for the wrong ``K``, which surfaces far from its
        cause.
        """
        with pytest.raises(ValueError, match="num_keypoints"):
            DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=1, task="keypoints")

    def test_state_dict_is_the_detection_state_dict_plus_the_point_stems_and_flow(self) -> None:
        """Dropping the point-stem and flow keys from a keypoints module leaves a detect module's.

        Pins the composition rule WP-087 established and WP-088 re-pinned: the pose module
        builds through :func:`~lucid_yolo.models.build.build_detection_stages` and keeps its
        stages flat. Nesting a pose-specific container inside it would prefix every key with
        that attribute name and invalidate the accepted checkpoints — a change no shape,
        count or loss value would reveal.
        """
        posed = _tiny_module()
        detection = _tiny_module(task="detect")

        keys = [key for key in posed.state_dict() if ".keypoint_stems." not in key and not key.startswith("rle_loss.")]

        assert keys == list(detection.state_dict())
        assert any(".keypoint_stems." in key for key in posed.state_dict())


class TestKeypointLossSelection:
    """WP-135's ``keypoint_loss`` switch: which objective fills the slot, and what it costs.

    R14 Table 7's "Laplace, learnable variance" row, wired as the second arm WP-125's
    mechanism claim needs. What these pin is that the switch is genuinely *additive* --
    the default reproduces the objective every existing config and checkpoint was built
    with -- and that the ablation arm actually trains, since an arm that quietly failed
    to would read as evidence for the flow rather than as a broken control.
    """

    def test_defaults_to_the_rle_objective(self) -> None:
        """A keypoints module built without ``keypoint_loss`` gets ``RLELoss``, as before WP-135.

        The non-breaking half of an additive change, asserted rather than assumed: every
        shipped config, every accepted figure and every checkpoint on disk was produced
        under R14's full loss, and a default that had shifted would silently retrain a
        different objective under the same recipe name.
        """
        assert isinstance(_tiny_module().rle_loss, RLELoss)

    def test_selects_the_flow_free_ablation_when_asked(self) -> None:
        """``keypoint_loss="laplace_nll"`` puts a ``LaplaceNLLLoss`` in the slot instead.

        The attribute keeps its ``rle_loss`` spelling under either objective because it
        prefixes the loss's own state-dict keys and renaming it would stop every
        pre-WP-135 keypoints checkpoint loading; the *type* is what changed.
        """
        module = _tiny_module(keypoint_loss="laplace_nll")

        assert isinstance(module.rle_loss, LaplaceNLLLoss)
        assert not isinstance(module.rle_loss, RLELoss)

    def test_an_unknown_objective_is_refused_at_construction(self) -> None:
        """A misspelled ``keypoint_loss`` raises, naming both arms, rather than falling back.

        The failure a silent fallback buys is the worst one this WP could ship: an
        ablation run that reports itself as the ablation while training the very
        objective it was supposed to control for, producing a comparison whose two arms
        are the same arm.
        """
        with pytest.raises(ValueError, match="keypoint_loss must be one of"):
            DetectionLitModule(
                depth=0.34,
                width=0.25,
                max_channels=256,
                num_classes=_NUM_CLASSES,
                task="keypoints",
                num_keypoints=_NUM_KEYPOINTS,
                keypoint_loss="laplace",
            )

    def test_the_ablation_adds_no_parameters_to_the_model(self) -> None:
        """An ablation module's state dict is a detection module's plus the point stems only.

        The structural difference between the two arms, stated where the model can see
        it: RLE contributes flow tensors an optimizer must pick up and a checkpoint must
        carry, the ablation contributes none at all. It also means an ablation module
        draws no RNG for its loss, so its weights are bit-for-bit a same-seed detection
        module's -- the construction-order caveat the module docstring states for the
        flow does not apply to this arm.
        """
        ablation = _tiny_module(keypoint_loss="laplace_nll")
        detection = _tiny_module(task="detect")

        assert not any(key.startswith("rle_loss.") for key in ablation.state_dict())
        assert [key for key in ablation.state_dict() if ".keypoint_stems." not in key] == list(detection.state_dict())

    def test_one_step_of_the_ablation_arm_trains_the_point_stems(self) -> None:
        """A ``laplace_nll`` step yields a finite scalar loss and gradient on both point stems.

        The RLE path's own wiring test asserts gradient on the stems *and* on the flow;
        this arm has no flow, so the stems are the whole of what the term has to move. A
        term that computed and discarded them would leave the ablation run flat while
        still logging an entirely ordinary loss curve -- and a flat arm reads as a large
        effect for the flow.
        """
        module = _tiny_module(keypoint_loss="laplace_nll")
        recorder = _LogRecorder()
        module.log = recorder  # type: ignore[method-assign]

        loss = module.training_step(_posed_batch(), 0)
        loss.backward()

        assert bool(torch.isfinite(loss))
        assert loss.ndim == 0
        assert _has_gradient(module.head.o2o.keypoint_stems)
        assert _has_gradient(module.head.o2m.keypoint_stems)
        assert math.isfinite(recorder.values["train/keypoint"])


class TestTrainingStep:
    """What one ``task="keypoints"`` step does to the total and to the gradients."""

    def test_gradients_reach_both_branches_point_stems_and_the_flow(self) -> None:
        """One step leaves non-zero gradients on both branches' point stems and on the flow.

        This is the test that would have caught the Phase 12 gap: the stems and the loss
        both existed, were constructed, appeared in the parameter count, and received no
        gradient because nothing in the objective read them. The flow is asserted beside
        them because it is the half of R14 that has no other way to be trained.
        """
        module = _tiny_module()

        module.training_step(_posed_batch(), 0).backward()

        assert _has_gradient(module.head.o2o.keypoint_stems)
        assert _has_gradient(module.head.o2m.keypoint_stems)
        assert module.rle_loss is not None
        assert _has_gradient(module.rle_loss)

    def test_the_keypoint_gain_enters_the_total_at_its_weight(self) -> None:
        """Zeroing ``keypoint_gain`` drops exactly ``gain * term`` from the total.

        A term that is wired but weightless is the Phase 12 gap wearing a disguise: the
        gradient test above would still pass while the term contributed nothing to what the
        optimizer minimises. The assertion is the *size* of the change rather than merely
        its presence, so a term entering at some other weight fails here too.
        """
        weighted = _tiny_module()
        zeroed = _tiny_module(keypoint_gain=0.0)
        zeroed.load_state_dict(weighted.state_dict())
        recorder = _LogRecorder()
        weighted.log = recorder  # type: ignore[method-assign]
        batch = _posed_batch()

        with_gain = weighted.training_step(batch, 0)

        without_gain = zeroed.training_step(batch, 0)
        dropped = float(with_gain.detach()) - float(without_gain.detach())
        assert dropped != 0.0
        assert dropped == pytest.approx(_KEYPOINT_GAIN * recorder.values["train/keypoint"], rel=_LEVERAGE_RTOL)

    def test_zero_gain_reproduces_the_detection_step_bit_for_bit(self) -> None:
        """At ``keypoint_gain=0`` the total equals a ``task="detect"`` module's, exactly.

        The decisive statement that this task **adds** rather than rearranges: ``"obb"``
        moves two gains out of the dual loss and would fail this by construction, while
        ``"keypoints"`` must leave every detection term precisely where it was. Equality is
        asserted rather than approximated because the two steps run the same arithmetic on
        the same weights in the same order — a tolerance here would hide exactly the small
        perturbation the test exists to forbid.

        The detection module's weights are copied *from* the pose module by name rather
        than both being seeded alike: the pose head draws its point stems from the same RNG
        stream, so a same-seed detection module's later parameters would legitimately
        differ, and the test would then measure the seed rather than the objective.
        """
        posed = _tiny_module(keypoint_gain=0.0)
        detection = _tiny_module(task="detect")
        shared = set(detection.state_dict())
        detection.load_state_dict({key: value for key, value in posed.state_dict().items() if key in shared})
        images, targets = _posed_batch()

        posed_total = posed.training_step((images, targets), 0)

        detection_total = detection.training_step((images, _boxes_only(targets)), 0)
        assert float(posed_total.detach()) == float(detection_total.detach())

    def test_an_image_without_instances_gives_a_finite_loss(self) -> None:
        """An empty batch yields a finite, exactly-zero keypoint term rather than a ``0 / 0`` NaN.

        With no ground truth there are no positives to gather and the padded point axis has
        length zero, which no gather can index. The term has to answer with a real zero.
        """
        module = _tiny_module()
        recorder = _LogRecorder()
        module.log = recorder  # type: ignore[method-assign]

        loss = module.training_step((torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE), [Targets.empty()]), 0)

        assert bool(torch.isfinite(loss))
        assert recorder.values["train/keypoint"] == 0.0

    def test_a_detection_batch_is_refused_rather_than_trained_on(self) -> None:
        """Point-free targets under ``task="keypoints"`` raise instead of training on nothing.

        The silent-subset failure of this path, and the one it is most exposed to today: the
        mosaic, the fused affine and mixup do not yet carry point channels, so an augmented
        training run arrives here with boxes alone (the letterbox does carry them, so the
        val loader is unaffected). Without the guard the RLE term would score zero points
        forever while the loss curve looked entirely ordinary.
        """
        module = _tiny_module()
        images, targets = _posed_batch()

        with pytest.raises(ValueError, match="keypoint sets"):
            module.training_step((images, _boxes_only(targets)), 0)

    def test_an_annotation_schema_of_the_wrong_width_is_refused(self) -> None:
        """A batch annotating a different ``K`` than the head predicts raises, naming both.

        A head built for 17 points fed a 5-point annotation set is a misconfiguration, not a
        broadcast: reading the first five points and inventing twelve is never the intended
        behaviour, and every intermediate shape on the way there looks reasonable.
        """
        module = _tiny_module()
        images = torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE)
        narrow = Targets(
            boxes=torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
            labels=torch.tensor([1]),
            keypoints=torch.rand(1, _NUM_KEYPOINTS + 2, 2) * 100.0,
            keypoint_vis=torch.full((1, _NUM_KEYPOINTS + 2), 2, dtype=torch.int64),
        )

        with pytest.raises(ValueError, match="the head predicts"):
            module.training_step((images, [narrow]), 0)


class TestVisibilityMasking:
    """A66: ``v == 0`` means no annotation exists, so it must not pull the model anywhere."""

    def test_an_unlabeled_point_moves_the_term_by_nothing(self) -> None:
        """Relocating a ``v == 0`` point across the canvas leaves the keypoint term identical.

        The end-to-end statement of A66, from ``Targets.keypoint_vis`` through
        :func:`~lucid_yolo.ptl.module.pad_keypoints` and the positive gather into the loss.
        A coordinate marked unlabeled is not a coordinate a human ever placed, so training
        against it optimizes towards an arbitrary number — and the failure is invisible in
        the loss curve, because the term stays perfectly finite either way. Moving the point
        by 500 px makes any leak enormous rather than marginal.
        """
        visibility = torch.tensor([[2, 0, 2]])
        target = _posed_targets(1, visibility=visibility)
        relocated = Targets(
            boxes=target.boxes,
            labels=target.labels,
            keypoints=target.keypoints + torch.tensor([[[0.0, 0.0], [500.0, 500.0], [0.0, 0.0]]]),
            keypoint_vis=visibility,
        )
        images = torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE)
        module = _tiny_module()
        recorder = _LogRecorder()
        module.log = recorder  # type: ignore[method-assign]

        module.training_step((images, [target]), 0)

        as_annotated = recorder.values["train/keypoint"]
        module.training_step((images, [relocated]), 0)
        assert recorder.values["train/keypoint"] == as_annotated

    def test_padding_rows_arrive_unlabeled(self) -> None:
        """Padding an instance-ragged batch marks every padded point ``v == 0``.

        The padding is inert in the loss by the *same* rule that excludes a genuinely
        unlabeled point, rather than by a second mechanism that could later disagree with
        it. A padding row filled with a visible zero coordinate would supervise anchors
        towards the canvas corner.
        """
        targets = [_posed_targets(2), _posed_targets(1)]

        coords, visibility = pad_keypoints(targets)

        assert coords.shape == (2, 2, _NUM_KEYPOINTS, 2)
        assert visibility.shape == (2, 2, _NUM_KEYPOINTS)
        assert torch.equal(visibility[1, 1], torch.zeros(_NUM_KEYPOINTS, dtype=torch.int64))
        assert torch.equal(coords[1, 1], torch.zeros(_NUM_KEYPOINTS, 2))


class TestPadKeypoints:
    """The instance-axis padding and the errors it refuses to paper over."""

    def test_pads_to_the_batch_maximum_on_the_instance_axis(self) -> None:
        """Every image is padded to the batch's largest instance count, keeping its own rows.

        The instance axis is shared with ``boxes`` and ``labels`` (WP-120), which is the only
        reason the assignment computed on the boxes may be reused to gather point targets.
        """
        first, second = _posed_targets(2), _posed_targets(1)

        coords, visibility = pad_keypoints([first, second])

        assert torch.equal(coords[0], first.keypoints)
        assert torch.equal(coords[1, :1], second.keypoints)
        assert torch.equal(visibility[0], first.keypoint_vis)

    def test_a_count_mismatch_is_a_hard_error(self) -> None:
        """An image whose point-set count differs from its instance count raises, naming the image.

        Whichever way such a batch were reconciled, some anchor would end up supervised
        towards another instance's pose — so there is no reading of it to fall back on.
        """
        with pytest.raises(ValueError, match="image 0 carries 1 instances but 0 keypoint sets"):
            pad_keypoints([Targets(boxes=torch.zeros(1, 4), labels=torch.zeros(1, dtype=torch.int64))])

    def test_images_disagreeing_on_the_point_count_are_refused(self) -> None:
        """Two images annotating different ``K`` raise rather than padding to the wider one.

        No single head predicts both counts, so padding the narrower image would invent
        points and marking them ``v == 0`` would quietly train a partial schema.
        """
        wide = _posed_targets(1)
        narrow = Targets(
            boxes=wide.boxes,
            labels=wide.labels,
            keypoints=torch.rand(1, _NUM_KEYPOINTS - 1, 2),
            keypoint_vis=torch.ones((1, _NUM_KEYPOINTS - 1), dtype=torch.int64),
        )

        with pytest.raises(ValueError, match="disagree on the keypoint count"):
            pad_keypoints([wide, narrow])

    def test_an_all_empty_batch_pads_to_a_zero_point_axis(self) -> None:
        """A batch holding no instances yields zero-width tensors rather than raising.

        The finite no-ground-truth case :func:`~lucid_yolo.ptl.module.pad_targets` already
        has; the step's own zero-positive branch is what keeps this shape away from the loss.
        """
        coords, visibility = pad_keypoints([Targets.empty(), Targets.empty()])

        assert coords.shape == (2, 0, 0, 2)
        assert visibility.shape == (2, 0, 0)


def _keypoint_context(anchor_points: Tensor, strides: Tensor, gt_keypoints: Tensor) -> _StepContext:
    """Build the minimal step context ``_branch_keypoint_loss`` reads.

    Examples:
        >>> context = _keypoint_context(torch.zeros(1, 2), torch.ones(1), torch.zeros(1, 1, 1, 2))
        >>> context.gt_keypoints.shape
        torch.Size([1, 1, 1, 2])
    """
    return _StepContext(
        head_out=None,  # type: ignore[arg-type]  # unread by the branch helper under test
        seg_out=None,
        targets=[],
        gt_boxes=torch.zeros(1, gt_keypoints.shape[1], 4),
        gt_rboxes=None,
        gt_keypoints=gt_keypoints,
        gt_keypoint_vis=torch.ones(gt_keypoints.shape[:3], dtype=torch.int64) * 2,
        anchor_points=anchor_points,
        strides=strides,
        image_size=(_IMG_SIZE, _IMG_SIZE),
        masks=None,
    )


def test_the_keypoint_term_follows_the_assignment_not_the_positive_order() -> None:
    """Positive ``k``'s points are scored against ``gt_index[k]``, not against instance ``k``.

    The decisive pairing test, and the one no loss value can stand in for: pairing an anchor
    with the wrong instance produces a perfectly finite, perfectly plausible number. Each
    positive is given the raw offsets that decode *exactly* onto the instance its ``gt_index``
    names, so the assignment pairing must reproduce the loss of a flawless prediction while
    the naive positional pairing — the same two point sets, merely swapped — must not.

    The reference is the module's own loss evaluated at zero residual rather than a
    hand-computed constant, because the flow's contribution at initialisation is a property
    of the seeded weights and asserting a number for it would be asserting the seed.
    """
    module = _tiny_module()
    anchor_points = torch.tensor([[8.0, 8.0], [24.0, 8.0], [8.0, 24.0], [24.0, 24.0]])
    strides = torch.full((4,), 8.0)
    first = torch.tensor([[40.0, 16.0], [56.0, 24.0], [16.0, 48.0]])
    second = torch.tensor([[96.0, 88.0], [72.0, 32.0], [120.0, 64.0]])
    gt_keypoints = torch.stack([first, second]).unsqueeze(0)  # (1, 2, K, 2)
    # Anchor 0 is told to predict instance 1's points and anchor 2 instance 0's, matching the
    # gt_index below; anchors 1 and 3 are negatives whose values are never gathered.
    wanted = torch.stack([second, torch.zeros_like(first), first, torch.zeros_like(first)]).unsqueeze(0)
    raw_points = (wanted - anchor_points.view(1, 4, 1, 2)) / strides.view(1, 4, 1, 1)
    raw_sigma = torch.zeros_like(raw_points)
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
    context = _keypoint_context(anchor_points, strides, gt_keypoints)
    assert module.rle_loss is not None
    flawless = torch.stack([second, first])
    perfect = module.rle_loss(
        flawless, torch.zeros_like(flawless), flawless, torch.full((2, _NUM_KEYPOINTS), 2, dtype=torch.int64)
    )

    scored = module._branch_keypoint_loss(raw_points, raw_sigma, assign, context)

    naive = module._branch_keypoint_loss(raw_points, raw_sigma, swapped, context)
    flawless_value = float(perfect.detach())
    assert float(scored.detach()) == pytest.approx(flawless_value, abs=1e-5)
    assert float(naive.detach()) != pytest.approx(flawless_value, abs=1e-5)


class TestLoaderTransport:
    """Keypoints crossing the DataLoader worker boundary and back."""

    def test_keypoints_survive_the_packed_transport(self) -> None:
        """Packing and unpacking a posed batch returns the point channels tensor-for-tensor.

        Without this the ``keypoint_targets`` flag would be the worst kind of dead code: the
        reader would parse the points and the transport would drop them, leaving a run that
        trains on nothing while every shape downstream still looks right.
        """
        targets = [_posed_targets(2), _posed_targets(1)]
        images = torch.rand(_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)

        restored = unpack_targets(collate_detection(list(zip(images, targets, strict=True)))[1])

        assert torch.equal(restored[0].keypoints, targets[0].keypoints)
        assert torch.equal(restored[1].keypoints, targets[1].keypoints)
        assert torch.equal(restored[0].keypoint_vis, targets[0].keypoint_vis)

    def test_an_instance_free_image_packs_beside_an_annotated_one(self) -> None:
        """A batch mixing posed and empty images packs, and each side restores to its own form.

        The empty image's canonical ``(0, 0, 2)`` point tensor disagrees with the batch's
        ``K`` on an axis ``torch.cat`` compares, so this is the case a naive concatenation
        fails on — and it is the ordinary case, since an image with no annotated object is
        common in every real split.
        """
        targets = [_posed_targets(2), Targets.empty()]
        images = torch.rand(_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)

        restored = unpack_targets(collate_detection(list(zip(images, targets, strict=True)))[1])

        assert torch.equal(restored[0].keypoints, targets[0].keypoints)
        assert restored[1].keypoints.shape == (0, 0, 2)
        assert restored[1].keypoint_vis.shape == (0, 0)

    def test_a_detection_batch_carries_no_point_axis(self) -> None:
        """A batch with no point annotations packs a zero-width keypoint axis, costing nothing.

        The transport is shared by every task, so the pose channels must not make a detection
        batch bigger than it was.
        """
        targets = _boxes_only([_posed_targets(2), _posed_targets(1)])
        images = torch.rand(_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)

        _, packed = collate_detection(list(zip(images, targets, strict=True)))

        assert packed.keypoints_cat.shape == (0, 0, 2)
        assert packed.keypoints_per_image.tolist() == [0, 0]


#: The animal fixture's own left/right swap, as named by its 16-point category schema
#: (``front_elbow_left``/``front_elbow_right`` and the three limb pairs after it). Written
#: out here rather than recomputed, so the test states the answer it expects.
_ANIMAL_FLIP_PAIRS = [(8, 9), (10, 11), (12, 13), (14, 15)]
#: Small letterbox side keeping the datamodule construction cheap.
_DM_IMG_SIZE = 64


def _keypoint_datamodule(fixture_dir: Path, keypoint_targets: bool = True) -> DetectionDataModule:
    """Build a datamodule over the keypoint fixture's single split, used for both loaders.

    Examples:
        >>> callable(_keypoint_datamodule)  # needs a live keypoints_fixture_dir fixture
        True
    """
    split = fixture_dir / "train"
    annotation = split / "_annotations.coco.json"
    return DetectionDataModule(
        data_root=fixture_dir,
        batch_size=2,
        num_workers=0,
        variant="n",
        img_size=_DM_IMG_SIZE,
        train_images_dir=split,
        train_ann_file=annotation,
        val_images_dir=split,
        val_ann_file=annotation,
        keypoint_targets=keypoint_targets,
    )


class TestFlipPairWiring:
    """The dataset's own pairing must reach the flip, not stop at the reader (A64)."""

    def test_the_dataset_schema_reaches_the_flip(self, keypoints_fixture_dir: Path) -> None:
        """The train pipeline's ``HorizontalFlip`` is constructed with the reader's derived pairs.

        This is the assertion whose absence let the bug sit unnoticed. The flip fires at
        ``fliplr=0.5`` on every training sample, and constructing it with no pairs is not a
        crash or a shape error — it mirrors coordinates while leaving identities in place, so
        every ``front_elbow_left`` in a mirrored sample is supervised toward the point
        ``front_elbow_right`` now occupies. That trains, converges to a left/right-confused
        head, and no gate in the repo reports it. Only checking the value arrives can.
        """
        datamodule = _keypoint_datamodule(keypoints_fixture_dir)

        datamodule.setup("fit")

        assert datamodule._train._flip.keypoint_flip_pairs == _ANIMAL_FLIP_PAIRS

    def test_a_detection_run_passes_no_pairing(self, keypoints_fixture_dir: Path) -> None:
        """Without ``keypoint_targets`` the flip receives ``None`` and only mirrors coordinates.

        The same fixture read as plain detection carries no point channel for a swap to act
        on. ``None`` has to keep meaning "mirror the coordinates, swap nothing" rather than
        becoming an error, since that is also the correct answer for any genuinely
        symmetry-free keypoint schema.
        """
        datamodule = _keypoint_datamodule(keypoints_fixture_dir, keypoint_targets=False)

        datamodule.setup("fit")

        assert datamodule._train._flip.keypoint_flip_pairs is None

    def test_points_survive_the_full_train_pipeline(self, keypoints_fixture_dir: Path) -> None:
        """A fully augmented training sample still carries one point set per surviving box.

        Every geometric stage on this path — mosaic, the fused affine, mixup, copy-paste,
        the flip — rebuilds its ``Targets``, and each rebuild is a place the point channel
        can be silently dropped. The affine's omission is what broke the keypoint gate, and
        the symptom appeared only later, at ``pad_keypoints``. Asserting on the assembled
        sample catches the next such omission at its own stage.
        """
        datamodule = _keypoint_datamodule(keypoints_fixture_dir)
        datamodule.setup("fit")

        _, targets = datamodule._train[0]

        assert targets.keypoints.shape[0] == targets.boxes.shape[0]
        assert targets.keypoint_vis.shape == targets.keypoints.shape[:2]


class TestBoxFrameNormalization:
    """A71: the frame the RLE residual is formed in.

    ``decode_keypoints`` emits absolute input pixels and R14 bounds ``sigma_hat`` into
    ``(0, 1)`` with a sigmoid, so scoring the two together unmodified makes the smallest
    expressible residual for a 40 px error equal to 40. These pin the mapping itself and
    the property that motivated it — that what the loss actually receives on a real batch
    is ``O(1)`` and not ``O(10^3)``.
    """

    def test_the_box_corners_map_to_the_unit_square(self) -> None:
        """The frame is the box: its top-left is the origin, its bottom-right is ``(1, 1)``."""
        points = torch.tensor([[[10.0, 20.0], [30.0, 40.0], [20.0, 30.0]]])
        boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

        normalized = normalize_keypoints_to_box(points, boxes)

        assert torch.allclose(normalized, torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]]))

    def test_each_axis_uses_its_own_extent(self) -> None:
        """Width normalizes ``x`` and height normalizes ``y``, not one shared object scale.

        The distinction is invisible on a square box and is the whole content of the
        choice on an elongated one, so the box here is deliberately 4:1.
        """
        points = torch.tensor([[[45.0, 10.0]]])
        boxes = torch.tensor([[5.0, 0.0, 85.0, 20.0]])  # 80 wide, 20 tall

        normalized = normalize_keypoints_to_box(points, boxes)

        assert torch.allclose(normalized, torch.tensor([[[0.5, 0.5]]]))

    def test_a_point_outside_its_box_leaves_the_unit_square(self) -> None:
        """A70's off-canvas points must stay representable rather than being clipped here.

        This function is a change of frame and nothing else: a point the affine pushed
        outside its box normalizes to a value outside ``[0, 1]``, which is the honest
        answer and the one the loss is expected to cope with.
        """
        points = torch.tensor([[[-30.0, 60.0]]])
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

        normalized = normalize_keypoints_to_box(points, boxes)

        assert torch.allclose(normalized, torch.tensor([[[-3.0, 6.0]]]))

    def test_a_degenerate_box_floors_at_one_pixel_instead_of_dividing_by_zero(self) -> None:
        """A zero-extent box returns a finite number, not an infinity.

        A sub-pixel box carries no pose to normalize by. The floor is not a correctness
        claim about such a box, only a refusal to answer a finite annotation with ``inf``.
        """
        points = torch.tensor([[[7.0, 3.0]]])
        boxes = torch.tensor([[5.0, 3.0, 5.0, 3.0]])

        normalized = normalize_keypoints_to_box(points, boxes)

        assert torch.isfinite(normalized).all()
        assert torch.allclose(normalized, torch.tensor([[[2.0, 0.0]]]))

    def test_the_residual_the_loss_receives_is_order_one(self) -> None:
        """The property the frame exists for, measured where it broke: on a real step.

        Un-normalized, this batch's median residual is in the hundreds and the flow
        overflows within two steps. The assertion is deliberately loose — the claim is an
        order of magnitude, not a value — because a tight bound here would pin the head's
        initialization rather than the frame.
        """
        residuals: list[Tensor] = []
        module = _tiny_module()
        original = RLELoss.forward

        def record(self: RLELoss, mu_hat: Tensor, sigma_raw: Tensor, mu_gt: Tensor, visibility: Tensor) -> Tensor:
            if mu_hat.numel():
                residuals.append(((mu_gt - mu_hat) / torch.sigmoid(sigma_raw)).detach().abs())
            return original(self, mu_hat, sigma_raw, mu_gt, visibility)

        RLELoss.forward = record  # type: ignore[method-assign]
        try:
            module.training_step(_posed_batch(), 0)
        finally:
            RLELoss.forward = original  # type: ignore[method-assign]

        assert residuals, "the keypoint term never reached the loss"
        assert float(torch.cat([r.reshape(-1) for r in residuals]).median()) < 10.0
