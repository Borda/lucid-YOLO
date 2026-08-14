# SPDX-License-Identifier: Apache-2.0
"""Single-image detection and segmentation inference from a trained checkpoint (WP-089, WP-090).

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

Two tasks, two entry points. :func:`predict_image` answers with boxes;
:func:`predict_segmentation` answers with boxes **and** their instance masks. Each
refuses the other's task by name rather than falling through, and the refusal is here in
the library rather than in the command, because
:meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward` returns a
:class:`~lucid_yolo.models.heads.detect.DualHeadOutput` for *every* task. A segmentation
checkpoint handed to :func:`predict_image` would otherwise produce boxes, silently, with
its mask branch never consulted: plausible output, no error, and the masks the caller
asked for absent. An ``obb`` checkpoint is refused by both; WP-091 fills that half of the
seam.

Two functions rather than one that switches on the task, because the two answers are
different shapes. A single function returning either a tensor or a tensor-and-masks pair
does not remove the switch, it moves it into every caller — and into every caller's type
checker, which can only see the union. The split is the one
:meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward` and
:meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward_segmentation` already make one
layer down, for the same reason.

Assumptions:
    Labels come out as **contiguous class indices**, not dataset category ids. The
    evaluator maps them back through the annotation file's category order, and a single
    image has no annotation file; the checkpoint carries ``num_classes`` and no names. A
    caller who knows which dataset trained the checkpoint owns that mapping.

    The mask threshold (A37) is **not** a parameter here, unlike ``conf_threshold``. The
    two look alike and are not: a confidence cut answers the caller's own question — how
    much of the low-scoring tail counts as "in this picture" — while the mask threshold
    is the neutral cut of a sigmoid trained with BCE, the calibration constant every
    reported segm mAP was measured at. Moving it drops no object and silently dilates or
    erodes every mask instead. It stays defined once, in
    :mod:`lucid_yolo.eval.segment_decode`, so a predicted mask and a scored mask are the
    same object; a caller who genuinely wants another cut calls
    :func:`~lucid_yolo.eval.segment_decode.decode_instance_masks` directly.

Provenance: R1 sec. 3.2.1, R1 Eq. 7, R3 sec. 4. Assumptions: A9, A10, A37.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from lucid_yolo.assign.grid import anchor_grid
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.decode.common import BOX_CORNERS, SCORE_COLUMN, to_letterboxed_original
from lucid_yolo.decode.nms_path import NMSDecoder
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.annotations import read_letterboxed_image
from lucid_yolo.eval.segment_decode import decode_instance_masks, masks_to_original

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from lucid_yolo.models.heads.detect import DualHeadOutput
    from lucid_yolo.ptl.module import DetectionLitModule

__all__ = [
    "DECODE_PATHS",
    "DEFAULT_CONF_THRESHOLD",
    "DecodePath",
    "SegmentedPrediction",
    "predict_image",
    "predict_segmentation",
]

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

#: The task :func:`predict_image` serves.
_DETECT_TASK = "detect"

#: The task :func:`predict_segmentation` serves. WP-091 adds the third.
_SEGMENT_TASK = "segment"


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
    if module.task != _DETECT_TASK:
        raise ValueError(
            f"predict_image handles task={_DETECT_TASK!r}; this checkpoint's task is {module.task!r}. "
            f"A segmentation checkpoint goes through predict_segmentation, which returns its masks "
            f"beside its boxes; oriented inference is a separate work package (WP-091)."
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


@dataclass(frozen=True)
class SegmentedPrediction:
    """One image's detections and their instance masks, both in original coordinates.

    A frozen pair rather than a bare tuple, because the two fields are only meaningful
    **row-aligned**: mask ``n`` belongs to detection ``n``, and a tuple invites a caller
    to pass one of them onward alone or to unpack the two the wrong way round. That
    pairing is the failure this whole path is most exposed to — a mask attached to the
    neighbouring object is a plausible answer no shape check catches — so it is stated in
    a type rather than in a comment.

    Attributes:
        detections: A CPU tensor of shape ``(N, 6)`` whose rows are the A9 tuple
            ``[x1, y1, x2, y2, score, class]`` in original-image pixels,
            score-descending, exactly as :func:`predict_image` returns them.
        masks: Boolean instance masks ``(N, orig_height, orig_width)`` on the original
            pixel grid (A10), row ``n`` being detection ``n``'s. Each is already cropped
            to its own detection's box (A11) and binarized at the A37 threshold.

    Examples:
        >>> import torch
        >>> prediction = SegmentedPrediction(torch.zeros(0, 6), torch.zeros(0, 4, 4, dtype=torch.bool))
        >>> prediction.detections.shape[0] == prediction.masks.shape[0]  # always row-aligned
        True
    """

    detections: Tensor
    masks: Tensor


def predict_segmentation(
    module: DetectionLitModule,
    image: Path,
    img_size: int = 640,
    decoder: DecodePath = "e2e",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    device: torch.device | None = None,
) -> SegmentedPrediction:
    """Segment objects in one image file and return boxes and masks in original coordinates.

    :func:`predict_image` plus Eq. 7, in the order
    :class:`~lucid_yolo.eval.coco_eval.DualPathEvaluator` established and for its reason:
    the model runs once, the selected path decodes boxes **and** reports the anchor each
    row came from, each surviving detection's mask coefficients are gathered by *that*
    index, the masks are assembled while the boxes are still on the letterboxed canvas
    — which is the frame
    :func:`~lucid_yolo.eval.segment_decode.decode_instance_masks` crops in — and only
    then are boxes and masks each landed on the original pixel grid. Cropping after the
    inverse letterbox would leave every box right and every mask quietly wrong.

    Rows below ``conf_threshold`` are dropped **before** the masks are assembled, which
    is where this differs from the evaluator: the evaluator needs a fixed-length output
    per image and pays for a mask per padding row, while a caller looking at one picture
    wants a mask per object. At 640 px the difference is 300 canvas-resolution masks
    instead of a handful, which is measured in hundreds of megabytes of transient float.

    Args:
        module: An eval-mode ``segment`` module, as
            :func:`~lucid_yolo.eval.checkpoint.load_eval_module` returns it.
        image: Path to the image file to read.
        img_size: Letterbox side the model sees. Defaults to ``640`` (R1 sec. 4.4).
        decoder: ``"e2e"`` for the suppression-free top-k path over the one-to-one
            branch, ``"nms"`` for the confidence-threshold plus class-wise suppression
            path over the dense branch. The choice reaches the masks too: each path reads
            the mask coefficients of its **own** branch.
        conf_threshold: Detections at or below this score are dropped, with their masks.
            Defaults to :data:`DEFAULT_CONF_THRESHOLD`.
        device: Device to run on. Defaults to CPU; the command resolves ``auto`` through
            :func:`~lucid_yolo.eval.checkpoint.pick_device` and passes the result.

    Returns:
        A :class:`SegmentedPrediction` holding the ``(N, 6)`` detections and their
        ``(N, orig_height, orig_width)`` boolean masks, row-aligned.

    Raises:
        ValueError: If the module's task is not ``segment``, naming the task it is; or if
            a module claiming that task has no mask coefficients to decode.

    Examples:
        >>> callable(predict_segmentation)  # a real call needs a checkpoint and an image file
        True
    """
    if module.task != _SEGMENT_TASK:
        raise ValueError(
            f"predict_segmentation handles task={_SEGMENT_TASK!r}; this checkpoint's task is {module.task!r}. "
            f"A detection checkpoint goes through predict_image, which has no masks to return; "
            f"oriented inference is a separate work package (WP-091)."
        )
    run_on = torch.device("cpu") if device is None else device
    letterbox = Letterbox(img_size)
    canvas_image, orig_size = read_letterboxed_image(image, letterbox)
    batch = canvas_image.unsqueeze(0).to(run_on)
    canvas = (int(batch.shape[-2]), int(batch.shape[-1]))

    module.to(run_on).eval()
    with torch.no_grad():
        # Called directly, with no forward-wrapper. The evaluator's `_SegmentationForward`
        # exists because the evaluator decides *structurally* — by what the forward returns
        # — whether to score masks, so it has to force the segmentation forward on a module
        # whose `forward` hides it. This function has already dispatched on the
        # checkpoint's own task, so a wrapper here would be a second spelling of a
        # composition that deliberately exists once.
        segmented = module.forward_segmentation(batch)
    anchor_points, strides = anchor_grid(canvas, run_on)
    detections, anchor_index, coefficients = _decode_with_coefficients(
        segmented.detect, decoder, conf_threshold, anchor_points, strides
    )

    # One selection, applied to the boxes and to the indices the coefficients are gathered
    # by, so the two cannot drift apart by a row. Padding rows are excluded explicitly and
    # not merely by their score: a row with no source anchor is not a detection at any
    # threshold, and letting one through would gather some real anchor's coefficients.
    keep = (detections[0, :, SCORE_COLUMN] > conf_threshold) & (anchor_index[0] >= 0)
    kept = detections[0][keep]
    canvas_masks = _canvas_masks(segmented.prototypes, coefficients[0], anchor_index[0][keep], kept, canvas)
    mapped = to_letterboxed_original(
        kept.unsqueeze(0).cpu(), orig_size=orig_size, letterboxed_size=canvas, allow_upscale=letterbox.allow_upscale
    )
    return SegmentedPrediction(
        detections=mapped[0],
        masks=masks_to_original(canvas_masks.cpu(), letterbox, orig_size),
    )


def _decode_with_coefficients(
    head_out: DualHeadOutput,
    decoder: DecodePath,
    conf_threshold: float,
    anchor_points: Tensor,
    strides: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode the selected path and return its detections, anchor indices and coefficients.

    The three come from one branch and must keep coming from one branch: the one-to-one
    coefficients are what the E2E decode reads and the dense ones are what the
    suppression path reads, so pairing either decoder's rows with the other branch's
    coefficients yields a plausible mask of a different object. Choosing all three in one
    place is what makes that pairing unrepresentable rather than merely unlikely.
    """
    if decoder == "e2e":
        detections, anchor_index = TopKDecoder().decode_with_indices(
            head_out.o2o_cls, head_out.o2o_box, anchor_points, strides
        )
        coefficients = head_out.o2o_coeff
    else:
        # Only this path is given the threshold, for the reason `predict_image` states:
        # it decides what enters suppression, and the top-k path's own threshold would
        # merely zero scores the survivor filter drops regardless.
        detections, anchor_index = NMSDecoder(conf_threshold=conf_threshold).decode_with_indices(
            head_out.o2m_cls, head_out.o2m_box, anchor_points, strides
        )
        coefficients = head_out.o2m_coeff
    if coefficients is None:
        raise ValueError(
            f"this checkpoint's task is {_SEGMENT_TASK!r} but its head emits no mask coefficients on the "
            f"{decoder!r} branch, so Eq. 7 has only one of its two halves. The checkpoint was built "
            f"without the coefficient stems and cannot produce a mask."
        )
    return detections, anchor_index, coefficients


def _canvas_masks(
    prototypes: Tensor,
    coefficients: Tensor,
    anchor_index: Tensor,
    detections: Tensor,
    canvas: tuple[int, int],
) -> Tensor:
    """Assemble the surviving detections' masks on the letterboxed canvas.

    ``anchor_index`` and ``detections`` are the *same* rows in the same order, so entry
    ``n`` of the gathered coefficients belongs to box ``n``. The gather is a plain
    :meth:`~torch.Tensor.index_select` rather than the evaluator's clamp-and-zero dance
    because the padding rows are already gone: the caller filtered them out with the
    boxes, and there is no fixed output length here to preserve.

    An image with nothing above the threshold returns an empty stack without decoding.
    That is not merely an optimisation: :func:`decode_instance_masks` upsamples with
    ``F.interpolate``, which rejects a zero-length instance axis outright. The evaluator
    never meets that because its decoders always hand it a full 300 rows.
    """
    if not detections.shape[0]:
        return prototypes.new_zeros((0, *canvas), dtype=torch.bool)
    gathered = coefficients.index_select(0, anchor_index).unsqueeze(0)
    boxes = detections[:, :BOX_CORNERS].unsqueeze(0)
    return decode_instance_masks(prototypes, gathered, boxes, image_size=canvas)[0]
