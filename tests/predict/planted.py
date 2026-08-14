# SPDX-License-Identifier: Apache-2.0
"""Shared geometry for the single-image inference suites (WP-089, WP-090, WP-091).

The detection suite and the segmentation one assert against the **same** planted box, and
that is why this module exists rather than a copy in each file: ``test_mask_matches_box``
means "the mask lands inside the box" only if the box it compares against is the very box
:mod:`tests.predict.test_predict` proved the letterbox inverse puts there. Two files each
carrying their own copy of the arithmetic would stay internally consistent while drifting
apart, and the second one's containment assertion would quietly stop saying anything.

The oriented suite lands on the same image and the same letterbox, with a rotated box of
its own: an ``xyxy`` box and a ``(cx, cy, w, h, theta)`` one are not the same statement,
and the second travels home through a different function
(:func:`~lucid_yolo.decode.common.rboxes_to_letterboxed_original`).

The image is 64x128, deliberately not square: at a 64 px canvas that is ratio 0.5 with a
16 px top pad and no left pad, so a wrong inverse cannot pass by symmetry. An inverse that
dropped the pad, or applied it on the wrong axis, or forgot the ratio, moves the box
somewhere else and the arithmetic is small enough to check by eye.

Imported by bare module name, the way ``tests/fixtures/synthetic.py`` is: ``tests`` is
deliberately not an importable package (see ``tests/conftest.py``), and the sibling
``conftest.py`` puts this directory on ``sys.path``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

#: Original image size ``(height, width)``: not square, so the letterbox pad is
#: one-sided and a pad-blind inverse cannot survive by accident.
ORIGINAL_SIZE = (64, 128)

#: Letterbox side used throughout. 64 keeps the anchor grid at 84 points (8x8 + 4x4 +
#: 2x2) while staying divisible by 32, which the head requires.
IMG_SIZE = 64

#: The planted box in **letterboxed canvas** pixels, and where the letterbox inverse
#: must put it in the original image. At ratio 0.5 with a 16 px top pad and no left pad:
#: ``x/0.5`` on the horizontal, ``(y - 16)/0.5`` on the vertical.
CANVAS_BOX = (8.0, 24.0, 40.0, 48.0)
EXPECTED_ORIGINAL_BOX = (16.0, 16.0, 80.0, 64.0)

#: The planted **rotated** box, in the two forms a head can emit it in, and where the
#: letterbox inverse must put either. Both describe the same rectangle centred at canvas
#: ``(24, 32)``: ``(32, 16)`` states the long edge first, ``(16, 32)`` the short edge
#: first — so the second is the first rotated a quarter turn, and canonicalization has to
#: turn it into the first with ``theta`` shifted by ``pi/2`` (A23). Their axis-aligned
#: envelopes, ``[8, 24, 40, 40]`` and ``[16, 16, 32, 48]``, both lie inside the 64 px
#: canvas's content band (``y`` in ``[16, 48]``), so neither planting reaches into a pad.
#: At ratio 0.5 with a 16 px top pad: ``cx/0.5``, ``(cy - 16)/0.5``, extents ``/0.5``.
CANVAS_RBOX_CENTRE = (24.0, 32.0)
CANVAS_RBOX_EXTENTS = (32.0, 16.0)
CANVAS_RBOX_EXTENTS_SWAPPED = (16.0, 32.0)
EXPECTED_ORIGINAL_RBOX = (48.0, 32.0, 64.0, 32.0)

#: Class index the planted detection carries, of the two the stub modules are built for.
PLANTED_LABEL = 1

#: Class count the stub heads report over.
NUM_CLASSES = 2

#: Logit of the planted anchor's class, and of every other class on every other anchor.
#: ``sigmoid(8) = 0.9997`` clears any sane threshold; ``sigmoid(-8) = 0.0003`` clears
#: none, so the rest of the grid is noise that both decoders must discard. The same pair
#: drives the planted prototypes, where their **symmetry** additionally matters: the two
#: probabilities sum to 1, so a bilinear interpolation between them crosses 0.5 exactly
#: midway and the mask boundary lands on a prototype-cell edge rather than near one.
PRESENT_LOGIT = 8.0
ABSENT_LOGIT = -8.0


def ltrb_from_anchor(box: tuple[float, ...], centre: Tensor, stride: Tensor) -> Tensor:
    """Return the raw ltrb distances that decode to ``box`` from ``centre``.

    The inverse of :func:`~lucid_yolo.models.heads.detect.decode_ltrb`, which is what a
    stub head has to do to plant a *chosen* box: distances are offsets from the anchor
    centre in stride units. This is head arithmetic, not letterbox arithmetic — the
    transform these work packages refuse to write twice is the letterbox inverse, and
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


def write_checkpoint(task: str, path: Path) -> Path:
    """Save a tiny untrained checkpoint of ``task`` that :func:`load_eval_module` reads.

    Args:
        task: The ``task`` the saved module — and therefore the checkpoint — carries.
        path: File to write the checkpoint to.

    Returns:
        ``path``, so a caller can name the file and use it in one expression.
    """
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=NUM_CLASSES, task=task)
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
