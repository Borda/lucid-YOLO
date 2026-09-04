# SPDX-License-Identifier: Apache-2.0
"""Single-image keypoint inference and its letterbox inverse (WP-152, WP-174).

The detection suite planted a box and asked where it landed; the segmentation one added a
mask and asked whether the two agreed. This one plants a **point set** and asks the
question a pose prediction is only useful if it answers: does a joint come back on the
original pixel grid where the letterbox geometry says it must.

Points travel home by :meth:`~lucid_yolo.data.letterbox.Letterbox.inverse_map`, the same
analytic inverse the box corners take (A10), so the expected coordinates below are
computed by hand from the ratio and the pad rather than read off the implementation. The
geometry is ``planted.py``'s throughout — a 64x128 image at a 64 px canvas is ratio 0.5
with a 16 px top pad and no left pad, so ``x_orig = x / 0.5`` and
``y_orig = (y - 16) / 0.5``. A pad-blind inverse, a pad applied on the wrong axis, or a
ratio left out moves every joint and the literals below stop matching.

Two further things a pose prediction can get wrong while looking entirely reasonable, and
both are separated here by construction:

- **Which branch the points came from.** The point stem is dense over anchors and each
  decode path reads its *own* branch's stem, exactly as the segmentation path reads its
  own branch's mask coefficients. The two branches carry different point sets here, so a
  crossed branch returns a plausible pose of the wrong object beside a correct box.
- **Which anchor they were gathered at.** :func:`~lucid_yolo.predict.predict_keypoints`
  gathers by the anchor index its decoder reports, not by row order. A decoy point set is
  planted at anchor 0 — far from the planted box, so never the selected anchor — and a
  gather that ignored the reported index would return it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pytest
import torch
from planted import (
    ABSENT_LOGIT,
    CANVAS_BOX,
    EXPECTED_ORIGINAL_BOX,
    IMG_SIZE,
    NUM_CLASSES,
    PLANTED_LABEL,
    PRESENT_LOGIT,
    ltrb_from_anchor,
)

from lucid_yolo.assign.grid import HEAD_STRIDES, make_anchor_points
from lucid_yolo.models.heads.detect import DualHeadOutput
from lucid_yolo.predict import DECODE_PATHS, predict_keypoints
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

#: Point count ``K`` the stub head reports. Deliberately not COCO's 17: ``K`` is a
#: constructor argument (A64) and :func:`~lucid_yolo.predict.predict_keypoints` reads it
#: off the decoded tensor, so a function that assumed 17 must fail here.
_POINT_COUNT = 3

#: The planted point set each branch carries, in letterboxed-canvas pixels, and where the
#: letterbox inverse must put it. At ratio 0.5 with a 16 px top pad and no left pad the
#: inverse is ``x / 0.5`` on the horizontal and ``(y - 16) / 0.5`` on the vertical:
#:
#: - ``e2e`` (one-to-one branch): ``(8, 24) -> (16, 16)``; ``(24, 32) -> (48, 32)``;
#:   ``(40, 40) -> (80, 48)``.
#: - ``nms`` (dense branch): ``(12, 20) -> (24, 8)``; ``(28, 36) -> (56, 40)``;
#:   ``(44, 44) -> (88, 56)``.
#:
#: Every planted ``y`` lies inside the content band ``[16, 48]``, so no point sits in a
#: pad, and the two sets are disjoint, so a path reading the other branch's stem cannot
#: land on the coordinates asserted for it.
_CANVAS_KEYPOINTS: dict[str, tuple[tuple[float, float], ...]] = {
    "e2e": ((8.0, 24.0), (24.0, 32.0), (40.0, 40.0)),
    "nms": ((12.0, 20.0), (28.0, 36.0), (44.0, 44.0)),
}
_EXPECTED_ORIGINAL_KEYPOINTS: dict[str, list[list[float]]] = {
    "e2e": [[16.0, 16.0], [48.0, 32.0], [80.0, 48.0]],
    "nms": [[24.0, 8.0], [56.0, 40.0], [88.0, 56.0]],
}

#: A third point set, planted on **both** branches at :data:`_DECOY_ANCHOR`, and its own
#: hand-computed originals: ``(20, 28) -> (40, 24)``; ``(32, 36) -> (64, 40)``;
#: ``(48, 44) -> (96, 56)``. Nothing should ever return these — they are what a gather
#: that ignored the decoder's reported anchor index would produce.
_DECOY_CANVAS_KEYPOINTS: tuple[tuple[float, float], ...] = ((20.0, 28.0), (32.0, 36.0), (48.0, 44.0))
_DECOY_EXPECTED_ORIGINAL_KEYPOINTS: list[list[float]] = [[40.0, 24.0], [64.0, 40.0], [96.0, 56.0]]

#: Anchor the decoy set is planted at. Anchor 0 is the first level-8 cell, centred at
#: canvas ``(4, 4)``; the planted box is centred at ``(24, 36)``, so anchor 0 is never the
#: one either decoder selects and the decoy can only surface through a wrong gather.
_DECOY_ANCHOR = 0

#: Confidence cut above the planted anchor's own ``sigmoid(8) = 0.9997``, so the survivor
#: set is empty on either path.
_ABOVE_EVERY_SCORE = 0.9999


def raw_points_from_anchor(points: tuple[tuple[float, float], ...], centre: Tensor, stride: Tensor) -> Tensor:
    """Return the raw per-point offsets that decode to ``points`` from ``centre``.

    The inverse of :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints`, which is
    what a stub head has to do to plant a *chosen* point set: a raw point value is an
    ``(x, y)`` offset from the anchor centre in stride units. This is head arithmetic, the
    keypoint twin of ``planted.py``'s :func:`~planted.ltrb_from_anchor` — the transform
    this suite refuses to write twice is the letterbox inverse, and that one is called by
    the shipping code and asserted against literals, never reproduced here.

    Args:
        points: Target ``(x, y)`` positions in letterboxed-canvas pixels, one per point.
        centre: The anchor's ``(x, y)`` centre, canvas pixels.
        stride: That anchor's level stride.

    Returns:
        A ``(K, 2)`` tensor of raw offsets in stride units.

    Examples:
        >>> import torch
        >>> raw_points_from_anchor(((12.0, 20.0),), torch.tensor([4.0, 4.0]), torch.tensor(8.0))
        tensor([[1., 2.]])
    """
    return (torch.tensor(points, dtype=torch.float32) - centre.view(1, 2)) / stride


def _flat_xy(points: list[list[float]]) -> list[float]:
    """Flatten a point list into one ``[x, y, x, y, ...]`` sequence for comparison.

    ``pytest.approx`` refuses nested structures, and comparing the flat sequence loses
    nothing here: the ``(N, K, 2)`` shape is asserted before every use, so a point axis
    that had drifted would already have failed.

    Args:
        points: A list of ``[x, y]`` pairs.

    Returns:
        The coordinates in one flat list.

    Examples:
        >>> _flat_xy([[1.0, 2.0], [3.0, 4.0]])
        [1.0, 2.0, 3.0, 4.0]
    """
    return [value for point in points for value in point]


class _PlantedKeypointModule(DetectionLitModule):
    """A ``keypoints`` module whose forward reports one object with a chosen point set.

    Subclasses the real module for the reason the detection and segmentation stubs do —
    ``task`` is then genuinely the module's own property, read exactly as it would be read
    off a checkpoint — and replaces only :meth:`forward`, which is the entry point
    :func:`~lucid_yolo.predict.predict_keypoints` calls. The untrained backbone would
    answer with noise, and what is under test is the geometry between the file on disk and
    the returned joint coordinates, not what a network saw.

    Both branches carry the same box, so the two decode paths select the same detection;
    they carry **different** point sets, so each path's pose reveals which branch's stem it
    actually read. Every anchor other than the selected one and the decoy carries a zero
    offset, which decodes to that anchor's own centre.

    Args:
        canvas_box: The ``xyxy`` box, in letterboxed-canvas pixels, the single planted
            detection decodes to.
        label: Class index the planted detection carries.
        with_points: When ``False`` the head emits no point stems at all, the shape a
            checkpoint built without them has. Defaults to ``True``.

    Examples:
        >>> module = _PlantedKeypointModule((8.0, 24.0, 40.0, 48.0), label=1)
        >>> module.task
        'keypoints'
    """

    def __init__(self, canvas_box: tuple[float, ...], label: int, with_points: bool = True) -> None:
        super().__init__(
            depth=0.34,
            width=0.25,
            max_channels=64,
            num_classes=NUM_CLASSES,
            task="keypoints",
            num_keypoints=_POINT_COUNT,
        )
        self._canvas_box = tuple(float(value) for value in canvas_box)
        self._label = int(label)
        self._with_points = bool(with_points)

    def forward(self, images: Tensor) -> DualHeadOutput:
        """Emit one confident anchor on both branches, each carrying its own point set."""
        batch, _, height, width = images.shape
        points, strides = make_anchor_points([(height // s, width // s) for s in HEAD_STRIDES], list(HEAD_STRIDES))
        x1, y1, x2, y2 = self._canvas_box
        centre = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2])
        anchor = int((points - centre).pow(2).sum(dim=-1).argmin())

        cls_logits = torch.full((batch, points.shape[0], NUM_CLASSES), ABSENT_LOGIT)
        cls_logits[:, anchor, self._label] = PRESENT_LOGIT
        raw_ltrb = torch.zeros(batch, points.shape[0], 4)
        raw_ltrb[:, anchor] = ltrb_from_anchor(self._canvas_box, points[anchor], strides[anchor])
        branch_points = {
            path: self._raw_branch_points(batch, points, strides, anchor, path) for path in _CANVAS_KEYPOINTS
        }
        return DualHeadOutput(
            o2m_cls=cls_logits,
            o2m_box=raw_ltrb,
            o2o_cls=cls_logits,
            o2o_box=raw_ltrb,
            o2m_keypoints=branch_points["nms"] if self._with_points else None,
            o2o_keypoints=branch_points["e2e"] if self._with_points else None,
        )

    @staticmethod
    def _raw_branch_points(batch: int, points: Tensor, strides: Tensor, anchor: int, path: str) -> Tensor:
        """Build one branch's dense raw offsets: its own set at ``anchor``, the decoy at 0."""
        raw = torch.zeros(batch, points.shape[0], _POINT_COUNT, 2)
        raw[:, anchor] = raw_points_from_anchor(_CANVAS_KEYPOINTS[path], points[anchor], strides[anchor])
        raw[:, _DECOY_ANCHOR] = raw_points_from_anchor(
            _DECOY_CANVAS_KEYPOINTS, points[_DECOY_ANCHOR], strides[_DECOY_ANCHOR]
        )
        return raw


class TestPredictKeypoints:
    """WP-152's shipped pose entry point, over ``planted.py``'s 64x128 geometry."""

    _FOREIGN_TASKS: ClassVar[list[str]] = ["detect", "segment", "obb"]

    @pytest.mark.parametrize("decoder", [pytest.param(path, id=path) for path in DECODE_PATHS])
    def test_planted_points_land_at_hand_computed_original_coordinates(self, image_file: Path, decoder: str) -> None:
        """A point set planted on the canvas comes back at the coordinates the letterbox implies.

        The image is 64x128 letterboxed into 64x64: ratio 0.5, 16 px of padding on top,
        none on the left. The ``e2e`` set ``[(8, 24), (24, 32), (40, 40)]`` therefore has
        to land at ``[(16, 16), (48, 32), (80, 48)]`` in the original image, and does so
        only if the pad is subtracted before the ratio is undone and only on the axis that
        carries it. The expected numbers are computed by hand from the ratio and the pad —
        not read back from the implementation — so an inverse that changed would fail here
        rather than silently redefine what "original coordinates" means.

        The box is asserted alongside, at ``planted.py``'s own
        :data:`~planted.EXPECTED_ORIGINAL_BOX`: a pose and the box around it must not land
        on two different grids. The decoy comparison is what makes this also a statement
        about the *anchor* the points were gathered at.
        """
        module = _PlantedKeypointModule(CANVAS_BOX, label=PLANTED_LABEL).eval()

        prediction = predict_keypoints(module, image_file, img_size=IMG_SIZE, decoder=decoder)

        assert prediction.detections.shape == (1, 6)
        assert prediction.detections[0, :4].tolist() == pytest.approx(EXPECTED_ORIGINAL_BOX)
        assert int(prediction.detections[0, 5]) == PLANTED_LABEL
        assert prediction.keypoints.shape == (1, _POINT_COUNT, 2)
        assert prediction.num_keypoints == _POINT_COUNT
        returned = _flat_xy(prediction.keypoints[0].tolist())
        assert returned == pytest.approx(_flat_xy(_EXPECTED_ORIGINAL_KEYPOINTS[decoder]))
        assert returned != pytest.approx(_flat_xy(_DECOY_EXPECTED_ORIGINAL_KEYPOINTS))

    @pytest.mark.parametrize("decoder", [pytest.param(path, id=path) for path in DECODE_PATHS])
    def test_nothing_above_the_threshold_keeps_the_point_schema(self, image_file: Path, decoder: str) -> None:
        """An image with no surviving detection yields empty boxes and empty poses, with ``K`` intact.

        The planted anchor scores ``sigmoid(8) = 0.9997``, so a threshold above that
        empties the survivor set on either path. ``K`` still has to survive: a caller
        reading :attr:`~lucid_yolo.predict.KeypointPrediction.num_keypoints` on an image
        with nothing in it should still learn the checkpoint's schema, and the empty case
        skips the inverse entirely rather than calling it on a zero-length point list.
        """
        module = _PlantedKeypointModule(CANVAS_BOX, label=PLANTED_LABEL).eval()

        prediction = predict_keypoints(
            module, image_file, img_size=IMG_SIZE, decoder=decoder, conf_threshold=_ABOVE_EVERY_SCORE
        )

        assert prediction.detections.shape == (0, 6)
        assert prediction.keypoints.shape == (0, _POINT_COUNT, 2)
        assert prediction.num_keypoints == _POINT_COUNT

    @pytest.mark.parametrize("task", [pytest.param(task, id=task) for task in _FOREIGN_TASKS])
    def test_a_non_keypoint_checkpoint_is_refused_naming_its_task(self, image_file: Path, task: str) -> None:
        """Detection, segmentation and oriented checkpoints are refused by name, never posed.

        All three build a head whose plain forward returns the same
        :class:`~lucid_yolo.models.heads.detect.DualHeadOutput` a keypoint one does, so
        falling through would gather a point tensor that is ``None`` — or, worse, produce
        boxes with the pose branch silently unread. The refusal names the task the
        checkpoint actually carries and the entry point that handles it, so the message is
        a redirection rather than a dead end.
        """
        module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=NUM_CLASSES, task=task).eval()

        with pytest.raises(ValueError, match=task):
            predict_keypoints(module, image_file, img_size=IMG_SIZE)

    @pytest.mark.parametrize("decoder", [pytest.param(path, id=path) for path in DECODE_PATHS])
    def test_a_checkpoint_with_no_point_stems_is_refused_on_either_branch(self, image_file: Path, decoder: str) -> None:
        """A module claiming ``keypoints`` whose head emits no points raises rather than posing nothing.

        The task is the checkpoint's own declaration, and a checkpoint built without the
        point stems satisfies it while having no pose to give. Silently returning the boxes
        alone, or an all-zero point stack, would answer the caller's question with
        something that is not a pose; the refusal names the branch that came back empty,
        which is the half a reader needs to tell "no stems at all" from "wrong branch".
        """
        module = _PlantedKeypointModule(CANVAS_BOX, label=PLANTED_LABEL, with_points=False).eval()

        with pytest.raises(ValueError, match="emits no points"):
            predict_keypoints(module, image_file, img_size=IMG_SIZE, decoder=decoder)
