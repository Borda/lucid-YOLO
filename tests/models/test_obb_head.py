# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-062 OBB head, its direct angle branch, and the rotated decode.

Three properties carry the work package and are what this module pins.

*The angle branch is opt-in and additive* (A20). A head built without it must be
the accepted detection head down to the bit — same state-dict keys, same
parameter values at a fixed seed — because the segmentation precedent (WP-047)
established that a new branch may extend the head but never perturb it. The
strongest in-repo form of that claim is asserted here: every parameter an
angle-enabled head shares with an angle-free one is bit-equal, so enabling the
branch cannot have moved the initialization of anything already accepted.

*The angle is predicted raw* (R1 Eq. 13, ``theta_hat = z``). The previous
versions' Eq. 12, ``theta_hat = (sigmoid(z) - 0.25) * pi``, bounded the output to
one half turn; YOLO26 deletes that squashing. A head that quietly kept it would
pass every shape assertion, so the property is pinned by driving the stem to an
output far outside the Eq. 12 range and observing it arrive unchanged.

*The decode normalizes what the head does not* (A23). Because the emitted angle
is unbounded, :func:`~lucid_yolo.models.heads.obb.decode_rboxes` is the only place
the long-edge range is established, and it must hold for outputs an untrained
head really produces — huge magnitudes, exact range boundaries, negatives — not
merely for plausible ones. The boundary-continuity check A23's register row names
is here too: raw angles a half turn apart describe the same rectangle, so they
must decode to the same corners.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from lucid_yolo.assign.grid import make_anchor_points
from lucid_yolo.data.rotated_geom import rboxes_to_polygons
from lucid_yolo.models import (
    DualDetectionHead,
    OrientedDetector,
    build_obb_detector,
    count_params,
    decode_ltrb,
    decode_rboxes,
    o2o_rotated_topk,
    o2o_topk,
)
from lucid_yolo.models.heads.detect import _angle_stem_width, _build_angle_stem, _stem_width

#: The head's level strides, in the (8, 16, 32) order the neck emits.
_STRIDES = [8, 16, 32]

#: Neck output widths of the ``n`` scale — the wiring is scale-independent.
_N_SCALE_CHANNELS = (64, 128, 256)

#: Square input side; divisible by 32, so every level's grid is exact.
_IMG_SIZE = 128

#: DOTA-v1.0 class count, the setting R1 Table S11 measures the OBB models at.
_NUM_CLASSES = 15

#: Inclusive lower and exclusive upper bounds of the canonical angle range (A23).
_THETA_LOW = -math.pi / 4
_THETA_HIGH = 3 * math.pi / 4


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed torch RNG so weight init and random inputs are deterministic."""
    torch.manual_seed(0)


def _feature_sizes(input_size: int) -> list[tuple[int, int]]:
    """Return the per-level ``(H, W)`` cell counts for a square input.

    Examples:
        >>> _feature_sizes(128)
        [(16, 16), (8, 8), (4, 4)]
    """
    return [(input_size // stride, input_size // stride) for stride in _STRIDES]


def _make_features(input_size: int, batch: int = 2) -> tuple[torch.Tensor, ...]:
    """Build random neck-style feature maps at strides 8/16/32 for the ``n`` widths.

    Examples:
        >>> features = _make_features(128)
        >>> [f.shape for f in features]
        [torch.Size([2, 64, 16, 16]), torch.Size([2, 128, 8, 8]), torch.Size([2, 256, 4, 4])]
    """
    return tuple(
        torch.randn(batch, channels, height, width)
        for channels, (height, width) in zip(_N_SCALE_CHANNELS, _feature_sizes(input_size), strict=True)
    )


def _anchor_grid(input_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the shared anchor points and per-anchor strides for a square input.

    Examples:
        >>> points, strides = _anchor_grid(128)
        >>> points.shape, strides.shape
        (torch.Size([336, 2]), torch.Size([336]))
    """
    return make_anchor_points(_feature_sizes(input_size), _STRIDES)


def _oriented_head() -> DualDetectionHead:
    """Return an evaluated head with the angle branch enabled.

    Examples:
        >>> head = _oriented_head()
        >>> head.training
        False
    """
    return DualDetectionHead(_N_SCALE_CHANNELS, num_classes=_NUM_CLASSES, predict_angle=True).eval()


def test_angle_branch_is_absent_by_default() -> None:
    """A head built without the flag owns no angle stems and emits no angle.

    The opt-in half of A20: the accepted detection head must be reachable by
    default, so an unrelated caller cannot acquire orientation outputs — or their
    parameters — without asking for them.
    """
    head = DualDetectionHead(_N_SCALE_CHANNELS, num_classes=_NUM_CLASSES).eval()

    with torch.no_grad():
        out = head(_make_features(_IMG_SIZE))

    assert head.o2o.angle_stems is None
    assert head.o2m.angle_stems is None
    assert out.o2o_angle is None
    assert out.o2m_angle is None


def test_enabling_the_angle_branch_only_adds_angle_keys() -> None:
    """Enabling the branch adds angle-stem keys and renames or drops nothing.

    The additive half of A20 at the level of the checkpoint: a flag that shipped
    orientation by *reshaping* an existing tensor — widening the box stem's output
    to five, say — would satisfy every shape assertion downstream while silently
    invalidating every detection checkpoint. The exact parameter arithmetic pins
    that the added keys are the whole of the difference.
    """
    torch.manual_seed(0)
    plain = DualDetectionHead(_N_SCALE_CHANNELS, num_classes=_NUM_CLASSES)
    torch.manual_seed(0)
    oriented = DualDetectionHead(_N_SCALE_CHANNELS, num_classes=_NUM_CLASSES, predict_angle=True)

    added = set(oriented.state_dict()) - set(plain.state_dict())

    assert set(plain.state_dict()) <= set(oriented.state_dict()), "enabling the branch must not drop a key"
    assert added and all("angle_stems" in key for key in added), "only angle stems may be added"
    angle_params = count_params(oriented.o2o.angle_stems) + count_params(oriented.o2m.angle_stems)
    assert count_params(oriented) - count_params(plain) == angle_params


def test_enabling_the_angle_branch_leaves_the_shipped_branch_bit_identical() -> None:
    """No one-to-one parameter moves when the angle branch is enabled.

    The stems are constructed *after* the box and class stems, so the branch draws
    exactly the random numbers it drew before: had they been spliced in earlier,
    every parameter after them would shift, the module tree and every shape
    assertion would still look right, and the frozen detection goldens would move
    for no visible reason.

    The property is asserted on the one-to-one branch — the one that ships — and
    not on the head as a whole, because the head builds ``o2o`` before ``o2m``:
    any optional stem set consumes draws in the first branch and so shifts the
    second. That is not new here. The accepted ``num_coeffs`` flag of WP-047
    shifts the very same 33 one-to-many keys and zero one-to-one keys, measured
    the same way, so the angle flag reproduces the established behaviour rather
    than introducing a deviation.
    """
    torch.manual_seed(0)
    plain = DualDetectionHead(_N_SCALE_CHANNELS, num_classes=_NUM_CLASSES)
    torch.manual_seed(0)
    oriented = DualDetectionHead(_N_SCALE_CHANNELS, num_classes=_NUM_CLASSES, predict_angle=True)

    plain_state = plain.o2o.state_dict()
    oriented_state = oriented.o2o.state_dict()

    shared = [key for key in plain_state if "angle_stems" not in key]
    assert shared, "the one-to-one branch must own parameters for this gate to mean anything"
    for key in shared:
        assert torch.equal(plain_state[key], oriented_state[key]), f"{key} moved when the angle branch was enabled"


@pytest.mark.parametrize(
    ("level_channels", "expected_params"),
    [
        pytest.param(64, 4289, id="p3-64ch"),
        pytest.param(128, 14721, id="p4-128ch"),
        pytest.param(256, 54017, id="p5-256ch"),
    ],
)
def test_angle_stem_parameter_count_is_exact(level_channels: int, expected_params: int) -> None:
    """Each level's angle stem holds exactly the hand-derived parameter count.

    The gate that keeps the A20 structure honest independently of the fidelity
    comparison. Table S11 agreement is a whole-model number and many wrong stems
    reproduce it to within a tolerance; this pins the stem itself, arithmetic
    term by arithmetic term, so a silently changed width, an added third unit, or
    a lost BatchNorm is caught here and named rather than surfacing as a drifted
    golden.

    With ``h = max(16, c // 2)`` (:func:`_angle_stem_width`) the count is
    ``[11c + c*h + 2h] + [11h + h*h + 2h] + [h + 1]``, the two
    depthwise-separable units and the bias-carrying output 1x1. Each unit is a
    bias-free depthwise 3x3 with its BatchNorm (``9c + 2c = 11c``) followed by a
    bias-free pointwise 1x1 with its BatchNorm (``c*h + 2h``). At ``c = 64``,
    ``h = 32``: ``(704 + 2048 + 64) + (352 + 1024 + 64) + 33 = 4289``.
    """
    stem = _build_angle_stem(level_channels)

    assert sum(parameter.numel() for parameter in stem.parameters()) == expected_params


def test_angle_branch_uses_a_wider_stem_than_the_box_and_class_stems() -> None:
    """The angle stem is half the level width while the others stay at a third.

    The split is load-bearing and easy to "tidy" away: A28's ``channels // 3`` is
    shared by the box, class and coefficient stems and looks like it should cover
    the angle stem too. It does not — the Table S11 gate selected the wider stem —
    so the divergence is asserted rather than left as a convention someone
    reunifies on sight.
    """
    assert _angle_stem_width(64) == 32
    assert _stem_width(64) == 21
    assert _angle_stem_width(256) == 128
    assert _stem_width(256) == 85


def test_angle_outputs_are_dense_per_anchor_scalars() -> None:
    """Both branches emit one raw angle per anchor, shaped (B, A, 1).

    A20 specifies a single scalar per location; an angle map shaped per level, or
    carrying more than one channel, would still concatenate into something the
    decode accepts and would silently mean something else.
    """
    batch = 2
    head = _oriented_head()
    anchor_points, _ = _anchor_grid(_IMG_SIZE)

    with torch.no_grad():
        out = head(_make_features(_IMG_SIZE, batch=batch))

    assert out.o2o_angle is not None
    assert out.o2m_angle is not None
    for angle in (out.o2o_angle, out.o2m_angle):
        assert angle.shape == (batch, anchor_points.shape[0], 1)


def test_branch_angle_stems_are_disjoint() -> None:
    """The one-to-one and one-to-many angle stems share no parameter object.

    The dual head's defining invariant extended to the new stems: a shared
    orientation stem would let one-to-many supervision write directly into the
    branch that ships.
    """
    head = _oriented_head()

    o2o_ids = {id(parameter) for parameter in head.o2o.angle_stems.parameters()}
    o2m_ids = {id(parameter) for parameter in head.o2m.angle_stems.parameters()}

    assert o2o_ids and o2m_ids
    assert o2o_ids.isdisjoint(o2m_ids)


def test_angle_output_is_unsquashed_beyond_the_legacy_range() -> None:
    """A stem driven to 100 radians emits 100 radians, not a bounded value.

    The Eq. 13 gate. Under the removed Eq. 12 squashing,
    ``theta_hat = (sigmoid(z) - 0.25) * pi``, no output could exceed ``3*pi/4``
    however large the pre-activation grew, so an output two orders of magnitude
    past that bound is positive evidence the nonlinearity is gone rather than
    merely unobserved.
    """
    head = _oriented_head()
    for stem in head.o2o.angle_stems:
        output_conv = stem[-1]
        assert isinstance(output_conv, nn.Conv2d), "the angle stem must end at a raw convolution"
        nn.init.zeros_(output_conv.weight)
        nn.init.constant_(output_conv.bias, 100.0)

    with torch.no_grad():
        angle = head(_make_features(_IMG_SIZE)).o2o_angle

    assert torch.allclose(angle, torch.full_like(angle, 100.0))


def test_zero_angle_decode_reproduces_the_axis_aligned_box() -> None:
    """With theta = 0 the rotated decode equals the axis-aligned decode exactly.

    The composition rule's anchor: the angle branch adds orientation on top of the
    shared ltrb regression rather than reinterpreting it, so the oriented and
    axis-aligned paths must agree bit for bit wherever the predicted angle
    vanishes. Any divergence here means the two paths have started decoding
    different geometry from the same numbers.
    """
    anchor_points, strides = _anchor_grid(_IMG_SIZE)
    num_anchors = anchor_points.shape[0]
    # Positive right/bottom margins keep the decoded width above the height, so
    # `canonicalize` has no edge to swap and theta = 0 survives untouched.
    distances = torch.rand(1, num_anchors, 4) + torch.tensor([2.0, 0.5, 2.0, 0.5])
    angles = torch.zeros(1, num_anchors, 1)

    rboxes = decode_rboxes(distances, angles, anchor_points, strides)
    boxes = decode_ltrb(distances, anchor_points, strides)

    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    assert torch.equal(rboxes[..., 0], (x1 + x2) / 2)
    assert torch.equal(rboxes[..., 1], (y1 + y2) / 2)
    assert torch.equal(rboxes[..., 2], x2 - x1)
    assert torch.equal(rboxes[..., 3], y2 - y1)
    assert torch.equal(rboxes[..., 4], torch.zeros_like(rboxes[..., 4]))


@pytest.mark.parametrize(
    "raw_angle",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(_THETA_LOW, id="exact-lower-bound"),
        pytest.param(_THETA_HIGH, id="exact-upper-bound"),
        pytest.param(math.nextafter(_THETA_LOW, -math.inf), id="one-ulp-below-lower-bound"),
        pytest.param(math.nextafter(_THETA_HIGH, -math.inf), id="one-ulp-below-upper-bound"),
        pytest.param(-math.pi, id="negative-half-turn"),
        pytest.param(-1.0e6, id="huge-negative"),
        pytest.param(1.0e6, id="huge-positive"),
        pytest.param(-123.456, id="arbitrary-negative"),
    ],
)
def test_decode_canonicalizes_adversarial_raw_angles(raw_angle: float) -> None:
    """Every decoded angle lands in [-pi/4, 3*pi/4) with the long edge first.

    A23's guarantee has to survive what an *untrained* head emits, and Eq. 13
    leaves that unbounded: there is no squashing to keep the raw scalar anywhere
    near a plausible range, so the exact boundaries, the values one ulp outside
    them, and magnitudes six orders of magnitude past them are all reachable
    outputs rather than hypotheticals.
    """
    anchor_points, strides = _anchor_grid(_IMG_SIZE)
    num_anchors = anchor_points.shape[0]
    distances = torch.rand(1, num_anchors, 4) * 4.0
    angles = torch.full((1, num_anchors, 1), raw_angle)

    rboxes = decode_rboxes(distances, angles, anchor_points, strides)

    theta = rboxes[..., 4]
    assert torch.isfinite(theta).all()
    assert bool((theta >= _THETA_LOW).all()), f"theta below the canonical range: {theta.min()}"
    assert bool((theta < _THETA_HIGH).all()), f"theta at or above the canonical range: {theta.max()}"
    assert bool((rboxes[..., 2] >= rboxes[..., 3]).all()), "long-edge convention violated"


def test_decode_is_continuous_across_the_half_turn_identification() -> None:
    """Raw angles a half turn apart decode to the same rectangle.

    The boundary-continuity check A23's register row names. A rectangle is
    invariant under a 180-degree rotation, so ``z`` and ``z + pi`` are two
    encodings of one box; a decode that mapped them to different corners would
    put a discontinuity in the middle of the angle range, exactly where the
    long-edge convention was adopted to remove one.
    """
    anchor_points, strides = _anchor_grid(_IMG_SIZE)
    num_anchors = anchor_points.shape[0]
    distances = torch.rand(1, num_anchors, 4) * 4.0
    angles = torch.empty(1, num_anchors, 1).uniform_(-2.0, 2.0)

    base = decode_rboxes(distances, angles, anchor_points, strides)
    shifted = decode_rboxes(distances, angles + math.pi, anchor_points, strides)

    base_corners = rboxes_to_polygons(base.reshape(-1, 5))
    shifted_corners = rboxes_to_polygons(shifted.reshape(-1, 5))
    assert torch.allclose(base_corners, shifted_corners, atol=1e-4)


def test_rotated_topk_emits_the_oriented_detection_tuple() -> None:
    """The rotated top-k returns (B, k, 7) rows of [cx, cy, w, h, theta, score, class].

    The oriented analogue of the A9 tuple: the same trailing score and class
    columns, preceded by the five-column long-edge box instead of four corners, so
    an evaluator reads both layouts the same way apart from the geometry.
    """
    batch, num_anchors, num_classes, keep = 2, 40, _NUM_CLASSES, 5
    scores = torch.randn(batch, num_anchors, num_classes)
    rboxes = torch.rand(batch, num_anchors, 5)

    detections = o2o_rotated_topk(scores, rboxes, k=keep)

    assert detections.shape == (batch, keep, 7)
    assert bool(((detections[..., 5] >= 0.0) & (detections[..., 5] <= 1.0)).all()), "scores must be sigmoids"
    assert torch.equal(detections[..., 6], detections[..., 6].round()), "class column must be integral"
    assert bool((detections[..., 5].diff(dim=1) <= 0.0).all()), "rows must be score-descending"


def test_rotated_topk_pairs_each_angle_with_its_own_anchor() -> None:
    """The kept rows' full five-column boxes are the selected anchors' own rows.

    The failure this guards is silent and specific: a second ranking computed for
    the angle would produce a detection whose centre, extents, score, and class
    are all correct while its heading belongs to a different object. Comparing
    against the axis-aligned selection proves one ranking drives both, since
    :func:`o2o_topk` and the rotated helper must agree anchor for anchor.
    """
    batch, num_anchors, keep = 2, 40, 6
    scores = torch.randn(batch, num_anchors, _NUM_CLASSES)
    rboxes = torch.rand(batch, num_anchors, 5) * 10.0

    detections = o2o_rotated_topk(scores, rboxes, k=keep)
    axis_aligned = o2o_topk(scores, rboxes[..., :4], k=keep)

    assert torch.equal(detections[..., :4], axis_aligned[..., :4]), "selection diverged from the shared ranking"
    assert torch.equal(detections[..., 5:], axis_aligned[..., 4:]), "score/class diverged from the shared ranking"
    for image in range(batch):
        for row in range(keep):
            source = (rboxes[image, :, :4] == detections[image, row, :4]).all(dim=-1).nonzero()[0, 0]
            assert float(detections[image, row, 4]) == float(rboxes[image, source, 4])


def test_rotated_topk_returns_every_anchor_when_fewer_than_k() -> None:
    """With fewer anchors than ``k`` the helper returns them all rather than padding.

    Matches the axis-aligned helper's contract: fixed-size padding is the
    decoder's job, not the selection's.
    """
    num_anchors = 3
    scores = torch.randn(1, num_anchors, _NUM_CLASSES)
    rboxes = torch.rand(1, num_anchors, 5)

    detections = o2o_rotated_topk(scores, rboxes, k=300)

    assert detections.shape == (1, num_anchors, 7)


def test_oriented_detector_forward_emits_angles_on_both_branches() -> None:
    """The composite wires the angle branch through to both branches' outputs.

    Catches a model that builds the stems but never runs them — the parameters
    would be exported and counted while the orientation stayed unpredicted.
    """
    model = OrientedDetector("n", num_classes=_NUM_CLASSES).eval()
    anchor_points, _ = _anchor_grid(_IMG_SIZE)

    with torch.no_grad():
        out = model(torch.zeros(1, 3, _IMG_SIZE, _IMG_SIZE))

    assert out.o2o_angle is not None
    assert out.o2m_angle is not None
    assert out.o2o_angle.shape == (1, anchor_points.shape[0], 1)
    assert out.o2o_cls.shape == (1, anchor_points.shape[0], _NUM_CLASSES)


def test_deployed_view_drops_the_one_to_many_branch_entirely() -> None:
    """The deployed model holds no one-to-many parameter, angle stems included.

    The R6 convention the Table S11 FLOP column is read under. A deployed view
    that still referenced the training-only branch would export it, count it, and
    make the NMS-free claim false for the artifact that ships — and the new angle
    stems are the easiest thing to leave behind, since they hang off the branch
    rather than the head.
    """
    model = OrientedDetector("n", num_classes=_NUM_CLASSES).eval()

    deployed = model.deploy()

    o2m_ids = {id(parameter) for parameter in model.head.o2m.parameters()}
    deployed_ids = {id(parameter) for parameter in deployed.parameters()}
    assert o2m_ids.isdisjoint(deployed_ids)
    assert count_params(deployed) == count_params(model) - count_params(model.head.o2m)
    assert id(next(deployed.backbone.parameters())) == id(next(model.backbone.parameters()))


def test_deployed_view_decodes_end_to_end_into_oriented_detections() -> None:
    """The deployed triple flows through the rotated decode without adaptation.

    The end-to-end proof that the head's outputs and the decode's inputs agree on
    layout: a mismatched angle axis or anchor ordering would raise here rather
    than surfacing as quietly wrong headings during a later evaluation run.
    """
    model = build_obb_detector("n", num_classes=_NUM_CLASSES).eval()
    deployed = model.deploy()
    anchor_points, strides = _anchor_grid(_IMG_SIZE)
    keep = 300

    with torch.no_grad():
        cls, box, angle = deployed(torch.zeros(1, 3, _IMG_SIZE, _IMG_SIZE))
        detections = o2o_rotated_topk(cls, decode_rboxes(box, angle, anchor_points, strides), k=keep)

    assert detections.shape == (1, min(keep, anchor_points.shape[0]), 7)
    assert torch.isfinite(detections).all()
    theta = detections[..., 4]
    assert bool(((theta >= _THETA_LOW) & (theta < _THETA_HIGH)).all())
