# SPDX-License-Identifier: Apache-2.0
"""Single-image segmentation inference and its ``lucid-predict`` report (WP-090).

The detection suite planted a box and asked where it landed. This one plants a box **and
a mask** and asks the harder question: do the two agree once they are both on the
original pixel grid. They travel by different routes —
:func:`~lucid_yolo.decode.common.to_letterboxed_original` maps the box corners through
the analytic letterbox inverse, while
:func:`~lucid_yolo.eval.segment_decode.masks_to_original` resamples the mask by mapping
each original pixel centre forward — and a disagreement between those two routes is the
defect this work package is most exposed to, because it leaves every box right and every
mask quietly wrong.

The plant, per ``planted.py``'s geometry (a 64x128 image at a 64 px canvas: ratio 0.5, a
16 px top pad, no left pad):

- the box is the same ``[8, 24, 40, 48]`` canvas box, landing at ``[16, 16, 80, 64]``;
- the prototypes are saturated step functions over whole prototype cells. The prototype
  grid is a quarter of the canvas per side (A15), so cell ``k`` covers canvas pixels
  ``[4k, 4k + 4)``, and a bilinear upsample of two symmetric probabilities crosses 0.5
  exactly midway between cell centres — the mask boundary therefore lands on the cell
  edge, with no pixel sitting on the threshold.

Prototype 0 covers canvas ``x`` in ``[0, 24)`` and prototype 1 covers ``x`` in
``[24, 48)``, both over ``y`` in ``[24, 48)``. Each is deliberately **half in and half
out** of the box, so the returned mask is bounded by the prototype on one side and by the
A11 box crop on the other: a mask that simply filled its box would fail, and so would one
the crop never touched.

The two prototypes are reached by different branches — the one-to-one coefficients select
prototype 0, the dense ones prototype 1 — which is what makes the decode-path
parametrisation say something. The mask path is *not* decoder-independent: each path
reads the mask coefficients of its own branch (``o2o_coeff`` / ``o2m_coeff``) and gathers
them by the anchor indices its own decoder reports, so a path that read the other
branch's coefficients would return a plausible mask of the wrong object beside a correct
box. Here that mistake puts the mask on the other half of the picture, and the case fails.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from faster_coco_eval import mask as mask_api
from planted import (
    ABSENT_LOGIT,
    CANVAS_BOX,
    EXPECTED_ORIGINAL_BOX,
    IMG_SIZE,
    NUM_CLASSES,
    ORIGINAL_SIZE,
    PLANTED_LABEL,
    PRESENT_LOGIT,
    SUPERSEDED_SEAM,
    plant_detection,
    write_checkpoint,
)

from lucid_yolo.cli import predict as predict_cli
from lucid_yolo.models.build import SegmentOutput
from lucid_yolo.models.heads.detect import DEFAULT_NUM_COEFFS, BranchName, BranchOutput
from lucid_yolo.predict import predict_image, predict_segmentation
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

#: Prototype grid cells the two planted rectangles occupy, ``(x_start, x_stop)`` in cell
#: units and shared ``(y_start, y_stop)``. A cell is 4 canvas pixels (A15), so these are
#: canvas ``x`` in ``[0, 24)`` and ``[24, 48)`` over ``y`` in ``[24, 48)``.
_PROTO_CELLS_X = ((0, 6), (6, 12))
_PROTO_CELLS_Y = (6, 12)

#: The tight ``xyxy`` extent, half-open and in **original** pixels, each branch's mask
#: must occupy: prototype 0 clipped on the right by the box, prototype 1 clipped on the
#: left by it. Canvas ``[8, 24)`` and ``[24, 40)`` at ratio 0.5 with no left pad; the
#: shared vertical span is canvas ``[24, 48)`` behind a 16 px top pad.
_EXPECTED_MASK_EXTENT = {"e2e": (16, 16, 48, 64), "nms": (48, 16, 80, 64)}


class _PlantedSegmentationModule(DetectionLitModule):
    """A ``segment`` module whose forward reports one object with a chosen mask.

    Subclasses the real module for the reason the detection stub does — ``task`` is then
    genuinely the module's own property, read exactly as it would be read off a
    checkpoint — and replaces only
    :meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward_segmentation_branch`, which
    is the entry point :func:`~lucid_yolo.predict.predict_segmentation` calls. The
    untrained backbone would answer with noise, and what is under test is the geometry
    between the file on disk and the returned mask, not what a network saw. The superseded
    :meth:`forward_segmentation` raises -- see :data:`~planted.SUPERSEDED_SEAM`.

    Both branches carry the same box, so the two decode paths select the same detection;
    they carry **different** mask coefficients, so each path's mask reveals which
    branch's coefficients it actually read. That asymmetry now also proves the branch
    argument reaches the plant: asking for the wrong branch returns the other path's mask.

    Args:
        canvas_box: The ``xyxy`` box, in letterboxed-canvas pixels, the single planted
            detection decodes to.
        label: Class index the planted detection carries.

    Examples:
        >>> module = _PlantedSegmentationModule((8.0, 24.0, 40.0, 48.0), label=1)
        >>> module.task
        'segment'
    """

    def __init__(self, canvas_box: tuple[float, ...], label: int) -> None:
        super().__init__(depth=0.34, width=0.25, max_channels=64, num_classes=NUM_CLASSES, task="segment")
        self._canvas_box = tuple(float(value) for value in canvas_box)
        self._label = int(label)

    def forward_segmentation_branch(self, images: Tensor, branch: BranchName) -> tuple[BranchOutput, Tensor]:
        """Emit one confident anchor pointing at the requested branch's own prototype."""
        batch, _, height, width = images.shape
        grid = plant_detection(images, self._canvas_box, self._label)
        # A one-hot coefficient vector makes the Eq. 7 combination select a single
        # prototype unchanged, so the mask that comes back is the rectangle planted below.
        # Prototype 0 belongs to the one-to-one branch and prototype 1 to the dense one,
        # which is what lets each decode path's mask name the branch it read.
        coefficients = torch.zeros(batch, grid.points.shape[0], DEFAULT_NUM_COEFFS)
        coefficients[:, grid.anchor, 0 if branch == "o2o" else 1] = 1.0
        return (
            BranchOutput(cls=grid.cls, box=grid.box, coeff=coefficients),
            _planted_prototypes(batch, height, width),
        )

    def forward_segmentation(self, images: Tensor) -> SegmentOutput:
        """Refuse the superseded dual-branch seam, loudly."""
        del images
        raise AssertionError(SUPERSEDED_SEAM)


def _planted_prototypes(batch: int, height: int, width: int) -> Tensor:
    """Build ``(B, K, H/4, W/4)`` prototypes holding one saturated rectangle each in 0 and 1.

    Examples:
        >>> prototypes = _planted_prototypes(1, 64, 128)
        >>> prototypes.shape
        torch.Size([1, 32, 16, 32])
        >>> bool((prototypes[0, 0, 6:12, 0:6] == PRESENT_LOGIT).all())
        True
        >>> float(prototypes[0, 0, 0, 0]) == ABSENT_LOGIT
        True
    """
    prototypes = torch.full((batch, DEFAULT_NUM_COEFFS, height // 4, width // 4), ABSENT_LOGIT)
    y_start, y_stop = _PROTO_CELLS_Y
    for channel, (x_start, x_stop) in enumerate(_PROTO_CELLS_X):
        prototypes[:, channel, y_start:y_stop, x_start:x_stop] = PRESENT_LOGIT
    return prototypes


def _tight_extent(mask: Tensor) -> tuple[int, int, int, int]:
    """Return a mask's occupied ``xyxy`` extent, half-open, in its own pixel indices.

    Examples:
        >>> mask = torch.zeros(8, 8, dtype=torch.bool)
        >>> mask[2:5, 3:6] = True
        >>> _tight_extent(mask)
        (3, 2, 6, 5)
    """
    rows = torch.nonzero(mask.any(dim=1)).flatten()
    cols = torch.nonzero(mask.any(dim=0)).flatten()
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


@pytest.mark.parametrize("decoder", [pytest.param(path, id=path) for path in _EXPECTED_MASK_EXTENT])
def test_mask_matches_box(image_file: Path, decoder: str) -> None:
    """The returned mask lies inside the box returned for the same detection, in the same frame.

    Box and mask reach original coordinates by two different transforms, so this is the
    assertion that says they agree: every foreground pixel of the mask falls inside the
    ``[16, 16, 80, 64]`` box, and the mask's own extent is exactly the planted rectangle
    clipped by that box — bounded by the prototype on one side and by the A11 crop on the
    other. A pad-blind mask inverse, or a crop applied after the inverse instead of
    before, moves one of the two and the containment fails.
    """
    module = _PlantedSegmentationModule(CANVAS_BOX, label=PLANTED_LABEL).eval()

    prediction = predict_segmentation(module, image_file, img_size=IMG_SIZE, decoder=decoder)

    assert prediction.detections.shape == (1, 6)
    assert prediction.detections[0, :4].tolist() == pytest.approx(EXPECTED_ORIGINAL_BOX)
    assert prediction.masks.shape == (1, *ORIGINAL_SIZE)
    assert prediction.masks.dtype == torch.bool
    mask_x1, mask_y1, mask_x2, mask_y2 = _tight_extent(prediction.masks[0])
    box_x1, box_y1, box_x2, box_y2 = EXPECTED_ORIGINAL_BOX
    assert (mask_x1, mask_y1) >= (box_x1, box_y1)
    assert (mask_x2, mask_y2) <= (box_x2, box_y2)
    assert (mask_x1, mask_y1, mask_x2, mask_y2) == _EXPECTED_MASK_EXTENT[decoder]


@pytest.mark.parametrize("decoder", [pytest.param(path, id=path) for path in _EXPECTED_MASK_EXTENT])
def test_nothing_above_the_threshold_returns_an_empty_row_aligned_pair(image_file: Path, decoder: str) -> None:
    """An image with no surviving detection yields empty boxes and empty masks, not an error.

    The planted anchor scores ``sigmoid(8) = 0.9997``, so a threshold above that empties
    the survivor set on either path. It is a live case for a real checkpoint and a
    reachable one for this code alone: the evaluator always hands
    :func:`~lucid_yolo.eval.segment_decode.decode_instance_masks` a fixed 300 rows,
    whereas this path decodes only the survivors — and ``F.interpolate`` rejects a
    zero-length instance axis outright, so an unguarded empty set raises from inside the
    mask decode rather than returning the empty answer it is.
    """
    module = _PlantedSegmentationModule(CANVAS_BOX, label=PLANTED_LABEL).eval()

    prediction = predict_segmentation(module, image_file, img_size=IMG_SIZE, decoder=decoder, conf_threshold=0.9999)

    assert prediction.detections.shape == (0, 6)
    assert prediction.masks.shape == (0, *ORIGINAL_SIZE)
    assert prediction.masks.dtype == torch.bool


def test_an_oriented_checkpoint_is_still_refused_by_name(image_file: Path) -> None:
    """An ``obb`` checkpoint is refused by both of these entry points, naming its task.

    The oriented head's plain forward returns the same detection output a detector's
    does, so falling through either way would produce boxes with the angle branch
    silently unread. WP-091 gave that checkpoint its own entry point
    (:func:`~lucid_yolo.predict.predict_oriented`) and this refusal is what keeps it the
    only way there: a third destination must not turn either of the first two into a path
    an oriented checkpoint can also take.
    """
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=NUM_CLASSES, task="obb").eval()

    with pytest.raises(ValueError, match="obb"):
        predict_segmentation(module, image_file, img_size=IMG_SIZE)
    with pytest.raises(ValueError, match="obb"):
        predict_image(module, image_file, img_size=IMG_SIZE)


def test_the_report_carries_each_detections_mask_as_coco_rle(
    tmp_path: Path,
    image_file: Path,
) -> None:
    """``--output`` writes decodable COCO RLE beside every box, and says so in the file.

    Driven through a real untrained ``segment`` checkpoint rather than the planted stub,
    because a checkpoint is what the command reads and the stub cannot survive a
    ``torch.save``/``load_eval_module`` round trip. Nothing is asserted about *which*
    objects an untrained network finds — only that whatever it finds is serialised in a
    form a reader can decode back to a mask on the original pixel grid, which is the
    claim the report's own ``masks`` key makes. ``--conf_threshold 0`` is what guarantees
    there is at least one record to check.
    """
    checkpoint = write_checkpoint("segment", tmp_path / "seg.ckpt")
    report = tmp_path / "masks.json"

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
    assert payload["info"]["task"] == "segment"
    assert payload["masks"] == "coco-rle"
    assert payload["detections"], "conf_threshold=0 keeps every decoded row"
    record = payload["detections"][0]
    assert record["segmentation"]["size"] == list(ORIGINAL_SIZE)
    decoded = mask_api.decode({"size": record["segmentation"]["size"], "counts": record["segmentation"]["counts"]})
    assert decoded.shape == ORIGINAL_SIZE
