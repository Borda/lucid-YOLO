# SPDX-License-Identifier: Apache-2.0
"""Single-image inference from a trained checkpoint (WP-089, WP-090, WP-091).

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

Three tasks, three entry points. :func:`predict_image` answers with axis-aligned boxes;
:func:`predict_segmentation` answers with boxes **and** their instance masks;
:func:`predict_oriented` answers with rotated boxes. Each refuses the other tasks by name
rather than falling through, and the refusal is here in the library rather than in the
command, because :meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward` returns a
:class:`~lucid_yolo.models.heads.detect.DualHeadOutput` for *every* task. A segmentation
or oriented checkpoint handed to :func:`predict_image` would otherwise produce boxes,
silently, with its mask or angle branch never consulted: plausible output, no error, and
the masks or the headings the caller asked for absent.

Three functions rather than one that switches on the task, because the three answers are
different shapes: a ``(N, 6)`` A9 tuple, that tuple beside a mask stack, and a ``(N, 7)``
A45 tuple whose box columns are not corners at all. A single function returning a union
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

    Oriented output is **canonical** (A23) and stays canonical: the angle is normalized
    inside :func:`~lucid_yolo.models.heads.obb.decode_rboxes`, and the letterbox inverse
    that follows is one isotropic scale plus a translation, which preserves both ``w >= h``
    and ``theta``. :func:`predict_oriented` therefore re-normalizes nothing — see its
    docstring for why that seam is the decode rather than this module.

Provenance: R1 sec. 3.2.1, R1 Eq. 7, R1 Eq. 13, R3 sec. 4. Assumptions: A9, A10, A23,
A37, A45.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from lucid_yolo.assign.grid import anchor_grid
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.decode.common import (
    BOX_CORNERS,
    RBOX_COLUMNS,
    SCORE_COLUMN,
    rboxes_to_letterboxed_original,
    to_letterboxed_original,
)
from lucid_yolo.decode.nms_path import NMSDecoder
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.annotations import read_letterboxed_image
from lucid_yolo.eval.segment_decode import decode_instance_masks, masks_to_original
from lucid_yolo.models.heads.obb import decode_rboxes, o2o_rotated_topk

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from lucid_yolo.models.heads.detect import DualHeadOutput
    from lucid_yolo.ptl.module import DetectionLitModule

__all__ = [
    "DECODE_PATHS",
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_ORIENTED_IMG_SIZE",
    "ORIENTED_DECODE_PATH",
    "DecodePath",
    "SegmentedPrediction",
    "predict_image",
    "predict_oriented",
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

#: Letterbox side :func:`predict_oriented` defaults to. The oriented tier trains on
#: 1024 px crops (R18 sec. 4), so an oriented checkpoint has never seen 640. The number
#: is the one :data:`~lucid_yolo.cli.eval.DEFAULT_IMG_SIZE` gives ``obb``, and a test
#: pins the two equal rather than either importing the command's table into the library
#: or letting the pair drift.
DEFAULT_ORIENTED_IMG_SIZE = 1024

#: The only decode path an oriented checkpoint can be read through, and the reason
#: :func:`predict_oriented` refuses the other by name — see its ``Raises``.
ORIENTED_DECODE_PATH: DecodePath = "e2e"

#: The task :func:`predict_image` serves.
_DETECT_TASK = "detect"

#: The task :func:`predict_segmentation` serves.
_SEGMENT_TASK = "segment"

#: The task :func:`predict_oriented` serves.
_OBB_TASK = "obb"


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
            f"beside its boxes, and an oriented one through predict_oriented, which returns rotated "
            f"boxes rather than the axis-aligned corners this function's tuple carries."
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
            f"A detection checkpoint goes through predict_image, which has no masks to return, and an "
            f"oriented one through predict_oriented, which has an angle instead of them."
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


def predict_oriented(
    module: DetectionLitModule,
    image: Path,
    img_size: int = DEFAULT_ORIENTED_IMG_SIZE,
    decoder: DecodePath = ORIENTED_DECODE_PATH,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    device: torch.device | None = None,
) -> Tensor:
    """Detect oriented objects in one image file and return them in original coordinates.

    :func:`predict_image`'s shape with R1 Eq. 13 in it, and the same pieces the per-tile
    oriented evaluator (:mod:`lucid_yolo.eval.rotated_eval`) is built from rather than a
    second reading of them: :func:`~lucid_yolo.models.heads.obb.decode_rboxes` turns the
    one-to-one branch's ltrb distances and raw angles into canonical rotated boxes,
    :func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk` ranks them, and
    :func:`~lucid_yolo.decode.common.rboxes_to_letterboxed_original` lands them on the
    original pixel grid. Nothing here computes a pad, a ratio or a corner.

    **The column convention is A45**, the one the oriented evaluator and
    :func:`~lucid_yolo.eval.dota_eval.rotated_detections_to_predictions` already consume:
    ``[cx, cy, w, h, theta, score, class]``, which is the long-edge rotated box of
    :mod:`lucid_yolo.data.rotated_geom` followed by the two columns the A9 tuple ends
    with. It is **not** the A9 tuple with an angle appended — the four box columns are a
    centre and two extents, not corners — so a caller that slices ``[:, :4]`` expecting
    ``xyxy`` gets a plausible, wrong rectangle. :data:`~lucid_yolo.decode.common.RBOX_COLUMNS`
    names where the box columns end and the score begins.

    **Canonicalization is post-decode, and that is the right seam** (A23). Two places
    could hold it and both are worse. In the *head*, it would normalize an angle the loss
    is still training against a raw target, and R1 Eq. 13 explicitly emits the
    pre-activation. *Here*, after the letterbox inverse, it would run once per entry point
    — this one, the evaluator, an export graph — and each copy would be free to drift.
    Inside :func:`~lucid_yolo.models.heads.obb.decode_rboxes` it runs on the **dense**
    ``(B, A, 5)`` output, so every consumer of a decoded rotated box inherits ``w >= h``
    and ``theta`` in ``[-pi/4, 3*pi/4)`` unconditionally, including one that never ranks
    and never letterboxes. What this function then adds is the proof that the guarantee
    survives the trip home: a letterbox inverse is one isotropic scale plus a translation,
    which scales both extents by the same positive number and turns no angle, so the
    canonical form is invariant under it and is **not** recomputed here.

    Args:
        module: An eval-mode ``obb`` module, as
            :func:`~lucid_yolo.eval.checkpoint.load_eval_module` returns it.
        image: Path to the image file to read.
        img_size: Letterbox side the model sees. Defaults to
            :data:`DEFAULT_ORIENTED_IMG_SIZE`, the 1024 px the oriented tier trains at
            (R18 sec. 4), not the 640 the COCO tiers use.
        decoder: Must be :data:`ORIENTED_DECODE_PATH`; the parameter exists so this
            function's signature matches its two siblings and the command can pass its
            flag through unread, not because there is a choice to make.
        conf_threshold: Detections at or below this score are dropped. Defaults to
            :data:`DEFAULT_CONF_THRESHOLD`.
        device: Device to run on. Defaults to CPU; the command resolves ``auto`` through
            :func:`~lucid_yolo.eval.checkpoint.pick_device` and passes the result.

    Returns:
        A CPU tensor of shape ``(N, 7)`` whose rows are the A45 tuple
        ``[cx, cy, w, h, theta, score, class]`` in original-image pixels,
        score-descending, canonical (``w >= h``, ``theta`` in ``[-pi/4, 3*pi/4)``, radians,
        measured from ``+x`` towards ``+y`` on the y-down image grid). ``class`` is a
        contiguous class index (see the module docstring).

    Raises:
        ValueError: If the module's task is not ``obb``, naming the task it is; if
            ``decoder`` is anything but :data:`ORIENTED_DECODE_PATH`; or if a module
            claiming the task emits no angles to decode.

    Examples:
        >>> callable(predict_oriented)  # a real call needs a checkpoint and an image file
        True
    """
    if module.task != _OBB_TASK:
        raise ValueError(
            f"predict_oriented handles task={_OBB_TASK!r}; this checkpoint's task is {module.task!r}. "
            f"A detection checkpoint goes through predict_image and a segmentation one through "
            f"predict_segmentation; both answer with axis-aligned corners, which is what a head "
            f"without an angle stem can say."
        )
    if decoder != ORIENTED_DECODE_PATH:
        # A raise, not a fallback to the one path that exists: the report records the
        # decoder it was asked for, so quietly running the other one would put a claim in
        # the file that the run did not honour. The dense branch is not the missing half —
        # an `obb` head builds both angle stems — the missing half is a *rotated*
        # suppression decoder. `dota_eval.rotated_iou` is the ingredient for one and no
        # decode path uses it yet; running the axis-aligned `NMSDecoder` over these
        # extents would suppress by upright overlap and silently keep or drop the wrong
        # rotated boxes, which is exactly the plausible-wrong-answer this refuses.
        raise ValueError(
            f"predict_oriented decodes the {ORIENTED_DECODE_PATH!r} path only; got decoder={decoder!r}. "
            f"The suppression path needs rotated NMS, which no decoder implements: suppressing "
            f"rotated boxes by their axis-aligned overlap would answer plausibly and wrongly."
        )
    run_on = torch.device("cpu") if device is None else device
    letterbox = Letterbox(img_size)
    canvas_image, orig_size = read_letterboxed_image(image, letterbox)
    batch = canvas_image.unsqueeze(0).to(run_on)
    canvas = (int(batch.shape[-2]), int(batch.shape[-1]))

    module.to(run_on).eval()
    with torch.no_grad():
        head_out = module(batch)
    if head_out.o2o_angle is None:
        raise ValueError(
            f"this checkpoint's task is {_OBB_TASK!r} but its head emits no angles on the "
            f"{ORIENTED_DECODE_PATH!r} branch, so R1 Eq. 13 has nothing to read. The checkpoint was "
            f"built without the orientation stems and cannot produce a heading."
        )
    anchor_points, strides = anchor_grid(canvas, run_on)
    rboxes = decode_rboxes(head_out.o2o_box, head_out.o2o_angle, anchor_points, strides)
    detections = o2o_rotated_topk(head_out.o2o_cls, rboxes)
    mapped = rboxes_to_letterboxed_original(
        detections.cpu(), orig_size=orig_size, letterboxed_size=canvas, allow_upscale=letterbox.allow_upscale
    )
    image_detections = mapped[0]
    return image_detections[image_detections[:, RBOX_COLUMNS] > conf_threshold]
