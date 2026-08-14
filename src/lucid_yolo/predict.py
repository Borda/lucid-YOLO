# SPDX-License-Identifier: Apache-2.0
"""Single-image detection inference from a trained checkpoint (WP-089).

One image file in, detections out, in the **original** image's coordinates. Everything
between is already written somewhere else and is called rather than restated: the
preprocessing is :func:`~lucid_yolo.eval.annotations.read_letterboxed_image`, the same
read-scale-letterbox chain the evaluator feeds val2017 through; the two decode paths are
:class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` and
:class:`~lucid_yolo.decode.nms_path.NMSDecoder`; and the way back to original pixels is
:func:`~lucid_yolo.decode.common.to_letterboxed_original`, which recovers the ratio and
the pads from :class:`~lucid_yolo.data.letterbox.Letterbox`'s own geometry.

That last one is the point of this module existing rather than a twenty-line script. A
letterbox inverse written a second time is the WP-053a defect class: two copies of one
transform, each with its own passing tests, free to disagree by a pad the day either
side's rounding changes. Nothing here computes a ratio, a pad or a corner.

Detection only. A ``segment`` or ``obb`` checkpoint is refused by name
(:func:`predict_image` raises), it does not fall through to this path — and the refusal
is here in the library rather than in the command, because
:meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward` returns a
:class:`~lucid_yolo.models.heads.detect.DualHeadOutput` for *every* task. A segmentation
checkpoint would therefore produce boxes, silently, with its mask branch never consulted:
plausible output, no error, and the masks the caller asked for absent. WP-090 and WP-091
fill that seam with the real thing.

Assumptions:
    Labels come out as **contiguous class indices**, not dataset category ids. The
    evaluator maps them back through the annotation file's category order, and a single
    image has no annotation file; the checkpoint carries ``num_classes`` and no names. A
    caller who knows which dataset trained the checkpoint owns that mapping.

Provenance: R1 sec. 3.2.1, R3 sec. 4. Assumptions: A9, A10.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from lucid_yolo.assign.grid import anchor_grid
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.decode.common import SCORE_COLUMN, to_letterboxed_original
from lucid_yolo.decode.nms_path import NMSDecoder
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.annotations import read_letterboxed_image

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from lucid_yolo.ptl.module import DetectionLitModule

__all__ = ["DECODE_PATHS", "DEFAULT_CONF_THRESHOLD", "DecodePath", "predict_image"]

#: The two selectable decode paths, spelled as
#: :func:`~lucid_yolo.eval.detect_eval.run` already prints and reports them, so one
#: vocabulary covers scoring and inference.
DecodePath = Literal["e2e", "nms"]

#: The same two strings as a runtime tuple, for a caller validating outside the type
#: system (a report writer, a test) without re-spelling them.
DECODE_PATHS: tuple[DecodePath, ...] = ("e2e", "nms")

#: Score below which a detection is dropped. Deliberately **not** the evaluator's
#: ``0.001``: that threshold exists so the mAP integration sees the low-confidence tail,
#: and a caller looking at one image wants the objects, not the tail — 300 rows of noise
#: per image is the wrong answer to "what is in this picture".
DEFAULT_CONF_THRESHOLD = 0.25

#: The task this module serves. WP-090 and WP-091 add the other two.
_SUPPORTED_TASK = "detect"


def predict_image(
    module: DetectionLitModule,
    image: Path,
    img_size: int = 640,
    decoder: DecodePath = "e2e",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    device: torch.device | None = None,
) -> Tensor:
    """Detect objects in one image file and return them in original-image coordinates.

    The image is letterboxed into an ``img_size`` square canvas, run through the module
    once, decoded by the selected path, and mapped back onto the original pixel grid by
    the exact analytic inverse of the letterbox that produced the canvas (A10). Rows
    below ``conf_threshold`` are dropped, which also removes the score-zero padding rows
    both decoders emit to keep their output shape fixed.

    Args:
        module: An eval-mode ``detect`` module, as
            :func:`~lucid_yolo.eval.checkpoint.load_eval_module` returns it.
        image: Path to the image file to read.
        img_size: Letterbox side the model sees. Defaults to ``640`` (R1 sec. 4.4).
        decoder: ``"e2e"`` for the suppression-free top-k path over the one-to-one
            branch, ``"nms"`` for the confidence-threshold plus class-wise suppression
            path over the dense branch.
        conf_threshold: Detections at or below this score are dropped. Defaults to
            :data:`DEFAULT_CONF_THRESHOLD`.
        device: Device to run on. Defaults to CPU; the command resolves ``auto`` through
            :func:`~lucid_yolo.eval.checkpoint.pick_device` and passes the result.

    Returns:
        A CPU tensor of shape ``(N, 6)`` whose rows are the A9 tuple
        ``[x1, y1, x2, y2, score, class]`` in original-image pixels, score-descending.
        ``class`` is a contiguous class index (see the module docstring).

    Raises:
        ValueError: If the module's task is not ``detect``, naming the task it is.

    Examples:
        >>> callable(predict_image)  # a real call needs a checkpoint and an image file
        True
    """
    if module.task != _SUPPORTED_TASK:
        raise ValueError(
            f"predict_image handles task={_SUPPORTED_TASK!r}; this checkpoint's task is {module.task!r}. "
            f"Oriented and segmentation inference are separate work packages (WP-090, WP-091)."
        )
    run_on = torch.device("cpu") if device is None else device
    letterbox = Letterbox(img_size)
    canvas_image, orig_size = read_letterboxed_image(image, letterbox)
    batch = canvas_image.unsqueeze(0).to(run_on)
    canvas = (int(batch.shape[-2]), int(batch.shape[-1]))

    module.to(run_on).eval()
    with torch.no_grad():
        head_out = module(batch)
    anchor_points, strides = anchor_grid(canvas, run_on)
    if decoder == "e2e":
        detections = TopKDecoder()(head_out.o2o_cls, head_out.o2o_box, anchor_points, strides)
    else:
        # Only this path is given the threshold: it decides what enters suppression, and
        # a box the threshold drops could only ever have been suppressed anyway, so the
        # survivor set is unchanged and the sort is cheaper. The top-k path's own
        # threshold merely zeroes scores that the filter below drops regardless, so
        # handing it the number too would state the same cut in two places.
        detections = NMSDecoder(conf_threshold=conf_threshold)(
            head_out.o2m_cls, head_out.o2m_box, anchor_points, strides
        )
    mapped = to_letterboxed_original(
        detections.cpu(), orig_size=orig_size, letterboxed_size=canvas, allow_upscale=letterbox.allow_upscale
    )
    image_detections = mapped[0]
    return image_detections[image_detections[:, SCORE_COLUMN] > conf_threshold]
