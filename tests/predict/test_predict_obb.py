# SPDX-License-Identifier: Apache-2.0
"""Single-image oriented inference and its ``lucid-predict`` report (WP-091).

The detection suite planted a box and asked where it landed; this one plants a **rotated**
box and asks two questions of it. The first is the same geometry round trip on the same
non-square image — 64x128 into a 64 px canvas, ratio 0.5, a 16 px top pad and no left pad
— so a pad-blind or axis-swapped inverse cannot pass by symmetry. A rotated box travels
home through :func:`~lucid_yolo.decode.common.rboxes_to_letterboxed_original`, not the
function the two axis-aligned suites exercise, and that one has an extra thing to get
wrong: the two extents scale by the ratio while the angle must not change at all.

The second question is the one nothing before this work package could ask: is the
returned angle **canonical** (A23). A head emits R1 Eq. 13's raw pre-activation, unbounded
and unsquashed, so a test that plants an already-canonical angle proves only that nothing
corrupted it. The cases here therefore plant angles from outside ``[-pi/4, 3*pi/4)`` —
a half turn above, a value below the floor, a wild 500 radians — and a rectangle stated
**short edge first**, which canonical form must turn into the long-edge one with ``theta``
shifted by a quarter turn. All of them describe the same rectangle on the canvas as the
in-range case does, so all of them must come back at the same
``planted.EXPECTED_ORIGINAL_RBOX``; only ``theta`` differs, and only by the
representative canonicalization is obliged to pick.

Both assertions land after the letterbox inverse, which is deliberate: it is what pins
that canonical form survives the trip rather than needing a second normalization at the
end of it (the inverse is one isotropic scale plus a translation, so it turns no angle and
cannot make the short edge the long one).
"""

from __future__ import annotations

import inspect
import json
import math
from typing import TYPE_CHECKING

import pytest
import torch
from planted import (
    ABSENT_LOGIT,
    CANVAS_RBOX_CENTRE,
    CANVAS_RBOX_EXTENTS,
    CANVAS_RBOX_EXTENTS_SWAPPED,
    EXPECTED_ORIGINAL_RBOX,
    IMG_SIZE,
    NUM_CLASSES,
    PLANTED_LABEL,
    SUPERSEDED_SEAM,
    plant_detection,
    write_checkpoint,
)

from lucid_yolo.cli import predict as predict_cli
from lucid_yolo.cli.eval import DEFAULT_IMG_SIZE
from lucid_yolo.models.heads.detect import BranchName, BranchOutput, DualHeadOutput
from lucid_yolo.predict import DEFAULT_ORIENTED_IMG_SIZE, DecodePath, predict_oriented
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

#: The canonical angle range (A23), as the assertions read it rather than as
#: :mod:`lucid_yolo.data.rotated_geom` computes it: a test that imported the module's own
#: bounds would agree with it by construction even if both were wrong.
_THETA_LOW = -math.pi / 4
_THETA_HIGH = 3 * math.pi / 4

#: Absolute tolerance for an angle in radians. Not the ``pytest.approx`` relative default:
#: the out-of-range cases reach their answer by subtracting ``pi`` from a number a few
#: times larger, which spends float32 precision the relative default then measures against
#: the small result. 1e-5 rad is 0.0006 degrees — far below anything this suite is about.
_THETA_TOLERANCE = 1e-5


class _PlantedOrientedModule(DetectionLitModule):
    """An ``obb`` module whose forward reports exactly one object, at a chosen rotated box.

    Subclasses the real module for the reason its two siblings do — ``task`` is then
    genuinely the module's own property, read exactly as it would be read off a checkpoint
    — and replaces only
    :meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward_branch`, which is what
    :func:`~lucid_yolo.predict.predict_oriented` calls. The untrained backbone would
    answer with noise, and what is under test is the geometry between the file on disk and
    the returned tuple, not what a network saw. The superseded dual-branch :meth:`forward`
    raises, so a caller that regresses to it fails instead of quietly asserting against
    that noise -- see :data:`~planted.SUPERSEDED_SEAM`.

    The box is planted through its **axis-aligned envelope**, because that is the shape the
    head actually regresses: :func:`~lucid_yolo.models.heads.obb.decode_rboxes` decodes
    ltrb distances exactly as the axis-aligned path does and reads the centre and the two
    extents off the result (A44), then attaches the raw angle. Planting the envelope of
    ``(centre, extents)`` therefore plants exactly those five numbers, with ``raw_theta``
    reaching canonicalization unmodified.

    Args:
        centre: The rotated box's ``(cx, cy)`` in letterboxed-canvas pixels.
        extents: Its ``(w, h)``, in the order the head emits them — not necessarily long
            edge first, which is the point of the swap case.
        raw_theta: The angle in radians as R1 Eq. 13 emits it: any real number, in or out
            of the canonical range.

    Examples:
        >>> module = _PlantedOrientedModule((24.0, 32.0), (32.0, 16.0), raw_theta=0.3)
        >>> module.task
        'obb'
    """

    def __init__(self, centre: tuple[float, float], extents: tuple[float, float], raw_theta: float) -> None:
        super().__init__(depth=0.34, width=0.25, max_channels=64, num_classes=NUM_CLASSES, task="obb")
        self._centre = (float(centre[0]), float(centre[1]))
        self._extents = (float(extents[0]), float(extents[1]))
        self._raw_theta = float(raw_theta)

    def forward_branch(self, images: Tensor, branch: BranchName) -> BranchOutput:
        """Emit one confident anchor decoding to the planted rotated box, on the asked branch."""
        # Both branches carry the same rotated box and the same heading: this suite's
        # subject is the decode and the letterbox inverse, not which branch supplied them.
        del branch
        centre_x, centre_y = self._centre
        half_w, half_h = self._extents[0] / 2, self._extents[1] / 2
        envelope = (centre_x - half_w, centre_y - half_h, centre_x + half_w, centre_y + half_h)
        grid = plant_detection(images, envelope, PLANTED_LABEL)
        angles = torch.zeros_like(grid.box[..., :1])
        angles[:, grid.anchor, 0] = self._raw_theta
        return BranchOutput(cls=grid.cls, box=grid.box, angle=angles)

    def forward(self, images: Tensor) -> DualHeadOutput:
        """Refuse the superseded dual-branch seam, loudly."""
        del images
        raise AssertionError(SUPERSEDED_SEAM)


class _AngleFreeOrientedModule(DetectionLitModule):
    """An ``obb`` module whose branch output carries no angles at all.

    Unreachable from a checkpoint this project builds — an ``obb`` head always builds both
    orientation stems — and constructible here, which is the only way to prove the guard
    against it is a raise rather than a fall-through to an axis-aligned answer.

    Examples:
        >>> _AngleFreeOrientedModule().task
        'obb'
    """

    def __init__(self) -> None:
        super().__init__(depth=0.34, width=0.25, max_channels=64, num_classes=NUM_CLASSES, task="obb")

    def forward_branch(self, images: Tensor, branch: BranchName) -> BranchOutput:
        """Emit the detection outputs of an oriented head that has lost its angle stems."""
        del branch
        # Nothing is planted. This stub exists only to reach the missing-angle guard, which
        # fires before a single logit is read, so "absent everywhere" — the shape of a head
        # that detected nothing — is the honest content for it to carry.
        grid = plant_detection(images, CANVAS_RBOX_CENTRE + CANVAS_RBOX_EXTENTS, PLANTED_LABEL)
        return BranchOutput(cls=torch.full_like(grid.cls, ABSENT_LOGIT), box=torch.zeros_like(grid.box))

    def forward(self, images: Tensor) -> DualHeadOutput:
        """Refuse the superseded dual-branch seam, loudly."""
        del images
        raise AssertionError(SUPERSEDED_SEAM)


@pytest.mark.parametrize(
    ("extents", "raw_theta", "expected_theta"),
    [
        pytest.param(CANVAS_RBOX_EXTENTS, 0.3, 0.3, id="already-canonical"),
        pytest.param(CANVAS_RBOX_EXTENTS, 0.3 + math.pi, 0.3, id="half-turn-above-the-range"),
        pytest.param(CANVAS_RBOX_EXTENTS, -1.0, math.pi - 1.0, id="below-the-range"),
        pytest.param(CANVAS_RBOX_EXTENTS_SWAPPED, 0.0, math.pi / 2, id="short-edge-first"),
    ],
)
def test_a_planted_rotated_box_lands_canonical_in_original_coordinates(
    image_file: Path,
    extents: tuple[float, float],
    raw_theta: float,
    expected_theta: float,
) -> None:
    """A rotated box planted on the canvas comes back canonical at the coordinates it implies.

    Every case plants the *same* rectangle at canvas ``(24, 32)`` and must therefore
    return the same ``[48, 32, 64, 32]`` in the original image: ratio 0.5 undone on both
    extents, the 16 px pad subtracted from the centre's ``y`` and from nothing else. What
    varies is how the head stated the rectangle. Three cases hand canonicalization an
    angle outside ``[-pi/4, 3*pi/4)`` — R1 Eq. 13 emits an unbounded pre-activation, so
    that is the normal case, not the exotic one — and the fourth states the box short edge
    first, which canonical form has to turn into the long-edge one with ``theta`` a quarter
    turn on. A path that skipped canonicalization returns the raw angle here, and one that
    canonicalized before the letterbox inverse rather than accepting that the inverse
    preserves it would still be wrong the day a non-isotropic transform is introduced.
    """
    module = _PlantedOrientedModule(CANVAS_RBOX_CENTRE, extents, raw_theta=raw_theta).eval()

    detections = predict_oriented(module, image_file, img_size=IMG_SIZE)

    assert detections.shape == (1, 7)
    assert detections[0, :4].tolist() == pytest.approx(EXPECTED_ORIGINAL_RBOX)
    assert float(detections[0, 4]) == pytest.approx(expected_theta, abs=_THETA_TOLERANCE)
    assert _THETA_LOW <= float(detections[0, 4]) < _THETA_HIGH
    assert float(detections[0, 2]) >= float(detections[0, 3])  # long edge first
    assert int(detections[0, 6]) == PLANTED_LABEL
    assert float(detections[0, 5]) > 0.9


def test_a_wild_raw_angle_still_lands_inside_the_canonical_range(image_file: Path) -> None:
    """An angle of 500 radians is normalized rather than returned, and the box is untouched.

    The range assertion is the whole point and the expected value is deliberately not
    written out: computing which representative 500 radians folds to would restate
    :func:`~lucid_yolo.data.rotated_geom.canonicalize`'s own arithmetic in the test, which
    would then agree with a broken implementation. What is checked instead is the property
    A23 promises — the answer lies in the range, the long edge is first — and that the
    rectangle itself survived a normalization that is only allowed to change how the box is
    described, never which box it is.
    """
    module = _PlantedOrientedModule(CANVAS_RBOX_CENTRE, CANVAS_RBOX_EXTENTS, raw_theta=500.0).eval()

    detections = predict_oriented(module, image_file, img_size=IMG_SIZE)

    assert detections.shape == (1, 7)
    assert _THETA_LOW <= float(detections[0, 4]) < _THETA_HIGH
    assert float(detections[0, 2]) >= float(detections[0, 3])
    assert detections[0, :4].tolist() == pytest.approx(EXPECTED_ORIGINAL_RBOX)


@pytest.mark.parametrize("task", [pytest.param("detect", id="detect"), pytest.param("segment", id="segment")])
def test_a_non_oriented_checkpoint_fails_naming_its_task(image_file: Path, task: str) -> None:
    """Detection and segmentation checkpoints are refused by name, never run as oriented ones.

    The third leg of a seam WP-089 opened: every task's head returns the same
    :class:`~lucid_yolo.models.heads.detect.DualHeadOutput`, so an entry point that fell
    through would decode a box for a checkpoint that has no heading to give and attach
    whatever ``o2o_angle`` happens to be — ``None`` here, a plausible number elsewhere.
    """
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=NUM_CLASSES, task=task).eval()

    with pytest.raises(ValueError, match=task):
        predict_oriented(module, image_file, img_size=IMG_SIZE)


def test_the_suppression_path_decodes_the_dense_branchs_rotated_boxes(image_file: Path) -> None:
    """``--decoder nms`` runs the rotated suppression decoder and lands the same planted box.

    WP-091 refused this path rather than substituting the other, because suppressing
    rotated boxes by the overlap of their upright envelopes keeps or drops the wrong ones
    with nothing in the output to show it had. WP-091b supplies the missing decoder, so
    the refusal is gone; what has to stay true is that the flag now selects the *dense*
    branch and its own angle stem. The planted module emits both branches identically, so
    the geometry assertion is the same one the ``e2e`` case makes — the point here is that
    the second path reaches it at all, through
    :class:`~lucid_yolo.decode.rotated_nms.RotatedNMSDecoder` and its class-wise rotated
    suppression rather than through the top-k rank.
    """
    module = _PlantedOrientedModule(CANVAS_RBOX_CENTRE, CANVAS_RBOX_EXTENTS, raw_theta=0.3).eval()

    detections = predict_oriented(module, image_file, img_size=IMG_SIZE, decoder="nms")

    assert detections.shape == (1, 7)
    assert detections[0, :4].tolist() == pytest.approx(EXPECTED_ORIGINAL_RBOX)
    assert float(detections[0, 4]) == pytest.approx(0.3, abs=_THETA_TOLERANCE)
    assert int(detections[0, 6]) == PLANTED_LABEL


@pytest.mark.parametrize("decoder", [pytest.param("e2e", id="e2e"), pytest.param("nms", id="nms")])
def test_an_oriented_head_without_angle_stems_is_refused(image_file: Path, decoder: DecodePath) -> None:
    """A module claiming ``obb`` but emitting no angles raises rather than decoding boxes.

    The oriented twin of the missing-coefficients guard: the task says a heading exists,
    the head does not supply one, and the only silent answer available is an axis-aligned
    box wearing an oriented tuple's shape. Both decode paths are checked because each
    reads its **own** branch's angle stem, and a guard written against one of them would
    let the other fall through to exactly that answer.
    """
    module = _AngleFreeOrientedModule().eval()

    with pytest.raises(ValueError, match="angles"):
        predict_oriented(module, image_file, img_size=IMG_SIZE, decoder=decoder)


def test_nothing_above_the_threshold_returns_an_empty_oriented_tensor(image_file: Path) -> None:
    """An image with no surviving detection yields an empty ``(0, 7)`` tensor, not an error.

    The planted anchor scores ``sigmoid(8) = 0.9997``, so a threshold above that empties
    the survivor set. The shape has to stay seven columns wide: a caller slicing the angle
    column off an empty answer is doing nothing unusual, and a bare ``(0,)`` would fail
    there rather than here.
    """
    module = _PlantedOrientedModule(CANVAS_RBOX_CENTRE, CANVAS_RBOX_EXTENTS, raw_theta=0.3).eval()

    detections = predict_oriented(module, image_file, img_size=IMG_SIZE, conf_threshold=0.9999)

    assert detections.shape == (0, 7)


def test_the_default_letterbox_side_is_the_one_the_command_resolves_for_obb() -> None:
    """``predict_oriented``'s default ``img_size`` equals the command's table entry for ``obb``.

    Two places name the oriented tier's 1024 px (R18 sec. 4): the library default, so a
    caller in process gets the side the checkpoint was trained at without knowing the
    number, and ``cli.eval.DEFAULT_IMG_SIZE``, which the command reads for every task.
    Neither can import the other — a library that reached into ``cli`` for a constant would
    invert the dependency — so the agreement is pinned here instead of assumed.
    """
    default = inspect.signature(predict_oriented).parameters["img_size"].default

    assert default == DEFAULT_ORIENTED_IMG_SIZE
    assert default == DEFAULT_IMG_SIZE["obb"]


def test_the_report_carries_rotated_boxes_and_names_their_convention(
    tmp_path: Path,
    image_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--output`` writes each detection as a canonical ``rbox`` and its corner ring.

    Driven through a real untrained ``obb`` checkpoint rather than the planted stub,
    because a checkpoint is what the command reads and the stub cannot survive a
    ``torch.save``/``load_eval_module`` round trip. Nothing is asserted about *which*
    objects an untrained network finds — only that whatever it finds is serialised in a
    self-describing form, and that A23 holds on real head output rather than only on
    planted angles: every row's long edge comes first and every angle is in range, which
    is a claim about the shipping decode, not about the plant.
    """
    checkpoint = write_checkpoint("obb", tmp_path / "obb.ckpt")
    report = tmp_path / "rboxes.json"

    code = predict_cli.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--image",
            str(image_file),
            "--ema",
            "false",
            "--img_size",
            str(IMG_SIZE),
            "--conf_threshold",
            "0",
            "--output",
            str(report),
        ]
    )

    assert code == 0
    payload = json.loads(report.read_text())
    assert payload["info"]["task"] == "obb"
    assert payload["boxes"] == "obb-longedge-rad"
    assert payload["masks"] is None  # an oriented checkpoint has no mask branch either
    assert payload["detections"], "conf_threshold=0 keeps every decoded row"
    record = payload["detections"][0]
    assert len(record["rbox"]) == 5
    assert len(record["polygon"]) == 8  # four corners, as DOTA states an object
    rboxes = [row["rbox"] for row in payload["detections"]]
    assert all(rbox[2] >= rbox[3] for rbox in rboxes), "every row's long edge comes first"
    assert all(_THETA_LOW <= rbox[4] < _THETA_HIGH for rbox in rboxes), "every angle is canonical"
    assert "rbox=[" in capsys.readouterr().out
