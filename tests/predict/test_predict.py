# SPDX-License-Identifier: Apache-2.0
"""Single-image detection inference and the ``lucid-predict`` command (WP-089).

The substance of this suite is a **geometry round trip**, not detection quality. No
trained weights exist and none may be created (D14), so a detection is *planted*: a
module stub emits one high-confidence anchor whose ltrb distances decode to a box chosen
by hand on the letterboxed canvas, and the assertion is that
:func:`~lucid_yolo.predict.predict_image` lands it on the original pixel grid where the
letterbox geometry says it must. Everything between the plant and the assertion — the
read, the resize, the pad, the decode, the inverse — is the shipping code path.

The image is 64x128, deliberately not square: at a 64 px canvas that is ratio 0.5 with a
16 px top pad and no left pad, so a wrong inverse cannot pass by symmetry. An inverse
that dropped the pad, or applied it on the wrong axis, or forgot the ratio, moves the box
somewhere else and the arithmetic is small enough to check by eye.

The planted logits are written to **both** branches, so one parametrized case covers the
suppression-free path over the one-to-one branch and the suppression path over the dense
one, and neither can pass by reading the other's output.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from torchvision.io import write_png

from lucid_yolo.assign.grid import HEAD_STRIDES, make_anchor_points
from lucid_yolo.cli import predict as predict_cli
from lucid_yolo.models.heads.detect import DualHeadOutput
from lucid_yolo.predict import DECODE_PATHS, predict_image
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

#: Original image size ``(height, width)``: not square, so the letterbox pad is
#: one-sided and a pad-blind inverse cannot survive by accident.
_ORIGINAL_SIZE = (64, 128)

#: Letterbox side used throughout. 64 keeps the anchor grid at 84 points (8x8 + 4x4 +
#: 2x2) while staying divisible by 32, which the head requires.
_IMG_SIZE = 64

#: The planted box in **letterboxed canvas** pixels, and where the letterbox inverse
#: must put it in the original image. At ratio 0.5 with a 16 px top pad and no left pad:
#: ``x/0.5`` on the horizontal, ``(y - 16)/0.5`` on the vertical.
_CANVAS_BOX = (8.0, 24.0, 40.0, 48.0)
_EXPECTED_ORIGINAL_BOX = (16.0, 16.0, 80.0, 64.0)

#: Class index the planted detection carries, of the two the stub module is built for.
_PLANTED_LABEL = 1

#: Logit of the planted anchor's class, and of every other class on every other anchor.
#: ``sigmoid(8) = 0.9997`` clears any sane threshold; ``sigmoid(-8) = 0.0003`` clears
#: none, so the rest of the grid is noise that both decoders must discard.
_PRESENT_LOGIT = 8.0
_ABSENT_LOGIT = -8.0


def _ltrb_from_anchor(box: tuple[float, ...], centre: Tensor, stride: Tensor) -> Tensor:
    """Return the raw ltrb distances that decode to ``box`` from ``centre``.

    The inverse of :func:`~lucid_yolo.models.heads.detect.decode_ltrb`, which is what a
    stub head has to do to plant a *chosen* box: distances are offsets from the anchor
    centre in stride units. This is head arithmetic, not letterbox arithmetic — the
    transform this work package refuses to write twice is the letterbox inverse, and
    that one is called, never reproduced.

    Args:
        box: Target ``xyxy`` box in letterboxed-canvas pixels.
        centre: The anchor's ``(x, y)`` centre, canvas pixels.
        stride: That anchor's level stride.

    Returns:
        A ``(4,)`` tensor of ``(l, t, r, b)`` distances in stride units.
    """
    x1, y1, x2, y2 = box
    return torch.stack(
        (
            (centre[0] - x1) / stride,
            (centre[1] - y1) / stride,
            (x2 - centre[0]) / stride,
            (y2 - centre[1]) / stride,
        )
    )


class _PlantedDetectionModule(DetectionLitModule):
    """A ``detect`` module whose forward reports exactly one object, at a chosen box.

    Subclasses the real module rather than faking one, so ``task`` is genuinely the
    module's own property and :func:`~lucid_yolo.predict.predict_image` reads it the way
    it reads a checkpoint's. Only :meth:`forward` is replaced: the untrained backbone
    would answer with noise, and what is under test is the geometry between the file on
    disk and the returned coordinates, not what a network saw.

    Args:
        canvas_box: The ``xyxy`` box, in letterboxed-canvas pixels, the single planted
            detection decodes to.
        label: Class index the planted detection carries.
        num_classes: Class count the stub head reports over. Defaults to ``2``.

    Examples:
        >>> module = _PlantedDetectionModule((8.0, 24.0, 40.0, 48.0), label=1)
        >>> module.task
        'detect'
    """

    def __init__(self, canvas_box: tuple[float, ...], label: int, num_classes: int = 2) -> None:
        super().__init__(depth=0.34, width=0.25, max_channels=64, num_classes=num_classes, task="detect")
        self._canvas_box = tuple(float(value) for value in canvas_box)
        self._label = int(label)
        self._num_classes = int(num_classes)

    def forward(self, images: Tensor) -> DualHeadOutput:
        """Emit one confident anchor on both branches, decoding to the planted box."""
        batch, _, height, width = images.shape
        points, strides = make_anchor_points([(height // s, width // s) for s in HEAD_STRIDES], list(HEAD_STRIDES))
        x1, y1, x2, y2 = self._canvas_box
        centre = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2])
        anchor = int((points - centre).pow(2).sum(dim=-1).argmin())
        cls_logits = torch.full((batch, points.shape[0], self._num_classes), _ABSENT_LOGIT)
        cls_logits[:, anchor, self._label] = _PRESENT_LOGIT
        raw_ltrb = torch.zeros(batch, points.shape[0], 4)
        raw_ltrb[:, anchor] = _ltrb_from_anchor(self._canvas_box, points[anchor], strides[anchor])
        return DualHeadOutput(o2m_cls=cls_logits, o2m_box=raw_ltrb, o2o_cls=cls_logits, o2o_box=raw_ltrb)


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    """Write a 64x128 RGB image and return its path.

    Content is irrelevant — the stub module ignores it — but the file has to decode, and
    its shape is what fixes the letterbox ratio and pad the assertions are written
    against.
    """
    path = tmp_path / "scene.png"
    write_png(torch.full((3, *_ORIGINAL_SIZE), 128, dtype=torch.uint8), str(path))
    return path


def _write_checkpoint(task: str, path: Path) -> Path:
    """Save a tiny untrained checkpoint of ``task`` that :func:`load_eval_module` reads."""
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=2, task=task)
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "epoch": 0,
            "global_step": 0,
            "pytorch-lightning_version": "2.4.0",
            "loops": {},
            "callbacks": {},
            "optimizer_states": [],
            "lr_schedulers": [],
        },
        path,
    )
    return path


@pytest.mark.parametrize("decoder", [pytest.param(path, id=path) for path in DECODE_PATHS])
def test_known_object_lands_in_original_coordinates(image_file: Path, decoder: str) -> None:
    """A detection planted on the canvas comes back at the original coordinates it implies.

    The image is 64x128 letterboxed into 64x64: ratio 0.5, 16 px of padding on top, none
    on the left. A canvas box of ``[8, 24, 40, 48]`` therefore has to land at
    ``[16, 16, 80, 64]`` in the original image, and does so only if the pad is subtracted
    before the ratio is undone and only on the axis that carries it. Both decode paths
    are exercised, from the same planted logits.
    """
    module = _PlantedDetectionModule(_CANVAS_BOX, label=_PLANTED_LABEL).eval()

    detections = predict_image(module, image_file, img_size=_IMG_SIZE, decoder=decoder)

    assert detections.shape == (1, 6)
    assert detections[0, :4].tolist() == pytest.approx(_EXPECTED_ORIGINAL_BOX)
    assert int(detections[0, 5]) == _PLANTED_LABEL
    assert float(detections[0, 4]) > 0.9


@pytest.mark.parametrize("task", [pytest.param("segment", id="segment"), pytest.param("obb", id="obb")])
def test_a_non_detect_checkpoint_fails_naming_its_task(image_file: Path, task: str) -> None:
    """Segmentation and oriented checkpoints are refused by name, never run as detectors.

    Both build a head whose plain forward returns the same detection output a detector's
    does, so falling through would produce boxes with the mask or angle branch silently
    unread — a plausible answer to a question nobody asked. This is the seam WP-090 and
    WP-091 fill.
    """
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=2, task=task).eval()

    with pytest.raises(ValueError, match=task):
        predict_image(module, image_file, img_size=_IMG_SIZE)


def test_the_report_is_written_into_a_directory_that_does_not_exist_yet(
    tmp_path: Path,
    image_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--output`` creates its parent, and stdout summarises the same run.

    The forward pass is the expensive part of this command, so a missing parent
    directory discovered afterwards discards work that has already been done — the
    WP-105 defect, guarded here so it cannot come back through a second entry point.
    """
    checkpoint = _write_checkpoint("detect", tmp_path / "det.ckpt")
    report = tmp_path / "absent" / "nested" / "dets.json"

    code = predict_cli.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--image",
            str(image_file),
            "--ema",
            "false",
            "--img_size",
            str(_IMG_SIZE),
            "--output",
            str(report),
        ]
    )

    assert code == 0
    payload = json.loads(report.read_text())
    assert payload["decoder"] == "e2e"
    assert payload["info"]["task"] == "detect"
    assert isinstance(payload["detections"], list)
    assert str(image_file) in capsys.readouterr().out
