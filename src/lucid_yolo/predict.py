# SPDX-License-Identifier: Apache-2.0
"""Single-image inference from a trained checkpoint (WP-089, WP-090, WP-091, WP-152).

One image file in, detections out, in the **original** image's coordinates. Everything
between is already written somewhere else and is called rather than restated: the
preprocessing is :func:`~lucid_yolo.eval.annotations.read_letterboxed_image`, the same
read-scale-letterbox chain the evaluator feeds val2017 through; the two decode paths are
:class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` and
:class:`~lucid_yolo.decode.nms_path.NMSDecoder`, with
:func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk` and
:class:`~lucid_yolo.decode.rotated_nms.RotatedNMSDecoder` as their oriented twins; and the
way back to original pixels is
:func:`~lucid_yolo.decode.common.to_letterboxed_original`, which recovers the ratio and
the pads from :class:`~lucid_yolo.data.letterbox.Letterbox`'s own geometry.

That last one is the point of this module existing rather than a twenty-line script. A
letterbox inverse written a second time is the WP-053a defect class: two copies of one
transform, each with its own passing tests, free to disagree by a pad the day either
side's rounding changes. Nothing here computes a ratio, a pad or a corner.

Four tasks, four entry points. :func:`predict_image` answers with axis-aligned boxes;
:func:`predict_segmentation` answers with boxes **and** their instance masks;
:func:`predict_oriented` answers with rotated boxes; :func:`predict_keypoints` answers
with boxes and their point sets. The fourth arrived at WP-152, a release after the task
it serves: 0.5.0 shipped keypoints with no shipped way to run a checkpoint on an image.
Each refuses the other tasks by name
rather than falling through, and the refusal is here in the library rather than in the
command, because :meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward` returns a
:class:`~lucid_yolo.models.heads.detect.DualHeadOutput` for *every* task. A segmentation
or oriented checkpoint handed to :func:`predict_image` would otherwise produce boxes,
silently, with its mask or angle branch never consulted: plausible output, no error, and
the masks or the headings the caller asked for absent.

Four functions rather than one that switches on the task, because the four answers are
different shapes: a ``(N, 6)`` A9 tuple, that tuple beside a mask stack, a ``(N, 7)``
A45 tuple whose box columns are not corners at all, and that A9 tuple beside a
``(N, K, 2)`` point stack. A single function returning a union
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

Provenance: R1 sec. 3.2.1, R1 Eq. 7, R1 Eq. 13, R3 sec. 4, R14. Assumptions: A9, A10,
A23, A37, A45, A61, A64.
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
from lucid_yolo.decode.rotated_nms import RotatedNMSDecoder
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.annotations import read_letterboxed_image
from lucid_yolo.eval.segment_decode import decode_instance_masks, masks_to_original
from lucid_yolo.models.heads.keypoint import decode_keypoints
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
    "KeypointPrediction",
    "SegmentedPrediction",
    "predict_image",
    "predict_keypoints",
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

#: The decode path :func:`predict_oriented` defaults to, and the one an oriented tier
#: report quotes as its headline number: the suppression-free branch is what this
#: architecture exists to demonstrate. Until WP-091b it was the *only* path an oriented
#: checkpoint could be read through and the other was refused by name; the refusal is
#: lifted now that :class:`~lucid_yolo.decode.rotated_nms.RotatedNMSDecoder` suppresses by
#: rotated overlap rather than by upright envelopes.
ORIENTED_DECODE_PATH: DecodePath = "e2e"

#: The task :func:`predict_image` serves.
_DETECT_TASK = "detect"

#: The task :func:`predict_segmentation` serves.
_SEGMENT_TASK = "segment"

#: The task :func:`predict_oriented` serves.
_OBB_TASK = "obb"

#: The task :func:`predict_keypoints` serves.
_KEYPOINTS_TASK = "keypoints"


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
    That is an optimisation and no longer a guard: :func:`decode_instance_masks` used to
    raise here, because ``F.interpolate`` rejects a zero-length instance axis outright and
    the evaluator never met that — its decoders always hand over a full 300 rows. WP-090b
    fixed it at the source, so both callers now get the empty stack; this one gets it
    without building the gather first.
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
        decoder: ``"e2e"`` for the suppression-free rotated top-k path over the one-to-one
            branch, ``"nms"`` for the rotated-suppression path over the dense branch
            (:class:`~lucid_yolo.decode.rotated_nms.RotatedNMSDecoder`, WP-091b). Defaults
            to :data:`ORIENTED_DECODE_PATH`. The second is the oriented comparison column
            rather than a better answer: it suppresses by exact rotated overlap, so it
            costs a Python-level greedy loop the first path does not run at all.
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
        ValueError: If the module's task is not ``obb``, naming the task it is; or if a
            module claiming the task emits no angles on the branch the selected decoder
            reads.

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
    run_on = torch.device("cpu") if device is None else device
    letterbox = Letterbox(img_size)
    canvas_image, orig_size = read_letterboxed_image(image, letterbox)
    batch = canvas_image.unsqueeze(0).to(run_on)
    canvas = (int(batch.shape[-2]), int(batch.shape[-1]))

    module.to(run_on).eval()
    with torch.no_grad():
        head_out = module(batch)
    anchor_points, strides = anchor_grid(canvas, run_on)
    detections = _decode_oriented(head_out, decoder, conf_threshold, anchor_points, strides)
    mapped = rboxes_to_letterboxed_original(
        detections.cpu(), orig_size=orig_size, letterboxed_size=canvas, allow_upscale=letterbox.allow_upscale
    )
    image_detections = mapped[0]
    return image_detections[image_detections[:, RBOX_COLUMNS] > conf_threshold]


def _decode_oriented(
    head_out: DualHeadOutput,
    decoder: DecodePath,
    conf_threshold: float,
    anchor_points: Tensor,
    strides: Tensor,
) -> Tensor:
    """Decode the selected oriented path into the fixed-size A45 batch it emits.

    The oriented analogue of :func:`_decode_with_coefficients`, and it exists for the same
    reason: the class logits, the ltrb distances and the **angles** have to come from one
    branch, and choosing all three in one place is what makes a mismatched trio
    unrepresentable rather than merely unlikely. Reading the one-to-one heading beside the
    dense branch's boxes would answer with a correct-looking rectangle at a heading no
    part of the model predicted for it.

    The two paths are not two spellings of one thing. ``e2e`` ranks and suppresses
    nothing, which is R1's claim for the one-to-one branch; ``nms`` is the dense branch's
    comparison baseline and suppresses by exact rotated overlap
    (:class:`~lucid_yolo.decode.rotated_nms.RotatedNMSDecoder`, A61). Only the second is
    given ``conf_threshold``, for the reason :func:`predict_image` states — it decides what
    enters suppression, while the top-k path's own threshold would merely zero scores the
    survivor filter drops regardless.
    """
    o2o = decoder == ORIENTED_DECODE_PATH
    angles = head_out.o2o_angle if o2o else head_out.o2m_angle
    if angles is None:
        branch = ORIENTED_DECODE_PATH if o2o else "one-to-many"
        raise ValueError(
            f"this checkpoint's task is {_OBB_TASK!r} but its head emits no angles on the "
            f"{branch!r} branch, so R1 Eq. 13 has nothing to read. The checkpoint was "
            f"built without the orientation stems and cannot produce a heading."
        )
    if o2o:
        rboxes = decode_rboxes(head_out.o2o_box, angles, anchor_points, strides)
        return o2o_rotated_topk(head_out.o2o_cls, rboxes)
    # Annotated rather than returned inline: `nn.Module.__call__` is typed `Any`, and
    # returning it straight would silently widen this function's contract to `Any` too.
    suppressed: Tensor = RotatedNMSDecoder(conf_threshold=conf_threshold)(
        head_out.o2m_cls, head_out.o2m_box, angles, anchor_points, strides
    )
    return suppressed


@dataclass(frozen=True)
class KeypointPrediction:
    """One image's detections and their point sets, both in original coordinates.

    A frozen pair for the same reason :class:`SegmentedPrediction` is one: the two
    fields mean nothing apart. Point set ``n`` belongs to detection ``n``, and a bare
    tuple invites a caller to carry one onward alone or unpack the two the wrong way
    round -- a pose attached to the neighbouring object being a plausible answer that
    no shape check catches.

    Attributes:
        detections: A CPU tensor of shape ``(N, 6)`` whose rows are the A9 tuple
            ``[x1, y1, x2, y2, score, class]`` in original-image pixels,
            score-descending, exactly as :func:`predict_image` returns them.
        keypoints: Point coordinates ``(N, K, 2)`` on the original pixel grid (A10),
            row ``n`` being detection ``n``'s. ``K`` is whatever the checkpoint's head
            predicts -- read from the decoded tensor, never assumed to be COCO's 17.

    Examples:
        ```pycon
        >>> import torch
        >>> prediction = KeypointPrediction(torch.zeros(0, 6), torch.zeros(0, 3, 2))
        >>> prediction.detections.shape[0] == prediction.keypoints.shape[0]
        True
        >>> KeypointPrediction(torch.zeros(2, 6), torch.zeros(2, 5, 2)).num_keypoints
        5

        ```
    """

    detections: Tensor
    keypoints: Tensor

    @property
    def num_keypoints(self) -> int:
        """Point count ``K`` this prediction carries, from the tensor rather than a constant.

        Returns:
            The size of the point axis.

        Examples:
            ```pycon
            >>> import torch
            >>> KeypointPrediction(torch.zeros(1, 6), torch.zeros(1, 17, 2)).num_keypoints
            17

            ```
        """
        return int(self.keypoints.shape[1])


def predict_keypoints(
    module: DetectionLitModule,
    image: Path,
    img_size: int = 640,
    decoder: DecodePath = "e2e",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    device: torch.device | None = None,
) -> KeypointPrediction:
    """Locate objects and their points in one image file, in original-image coordinates.

    :func:`predict_image` plus the point branch, composed the way
    :func:`predict_segmentation` composes the mask branch and for the same reason: the
    point stem is dense over anchors, so the selected path must report the *anchor*
    each surviving row came from and the points are gathered by that index. The
    decode-then-gather order is :class:`~lucid_yolo.export.KeypointExportGraph`'s and
    the training module's own ``val/oks_mAP``'s, so all three read one composition.

    Points are mapped back with :meth:`~lucid_yolo.data.letterbox.Letterbox.inverse_map`
    -- the same exact analytic inverse the boxes take (A10), applied to the point axis
    flattened into a plain point list, so a point and the box around it cannot land on
    two different grids.

    ``K`` is read from the decoded tensor, never assumed. The head's point count is a
    constructor argument (A64) and COCO's 17 is one instantiation of it; a function that
    assumed 17 would silently mis-shape every other schema.

    Args:
        module: An eval-mode ``keypoints`` module, as
            :func:`~lucid_yolo.eval.checkpoint.load_eval_module` returns it.
        image: Path to the image file to read.
        img_size: Letterbox side the model sees. Defaults to ``640`` (R1 sec. 4.4).
        decoder: ``"e2e"`` for the suppression-free top-k path over the one-to-one
            branch, ``"nms"`` for the confidence-threshold plus class-wise suppression
            path over the dense branch. The choice reaches the points too: each path
            reads its **own** branch's point stem.
        conf_threshold: Detections at or below this score are dropped, with their
            points. Defaults to :data:`DEFAULT_CONF_THRESHOLD`.
        device: Device to run on. Defaults to CPU; the command resolves ``auto`` through
            :func:`~lucid_yolo.eval.checkpoint.pick_device` and passes the result.

    Returns:
        A :class:`KeypointPrediction` holding the ``(N, 6)`` detections and their
        ``(N, K, 2)`` point coordinates, row-aligned.

    Raises:
        ValueError: If the module's task is not ``keypoints``, naming the task it is; or
            if a module claiming that task emits no points on the selected branch.

    Examples:
        ```pycon
        >>> callable(predict_keypoints)  # a real call needs a checkpoint and an image file
        True

        ```
    """
    if module.task != _KEYPOINTS_TASK:
        raise ValueError(
            f"predict_keypoints handles task={_KEYPOINTS_TASK!r}; this checkpoint's task is {module.task!r}. "
            f"A detection checkpoint goes through predict_image, which has no points to return, a "
            f"segmentation one through predict_segmentation, which has masks instead of them, and an "
            f"oriented one through predict_oriented, which has an angle."
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
    detections, anchor_index, raw_points = _decode_with_points(
        head_out, decoder, conf_threshold, anchor_points, strides
    )

    # One selection for the boxes and for the indices the points are gathered by, so the
    # two cannot drift by a row. Padding rows go by their missing anchor rather than by
    # their score: a row with no source anchor is not a detection at any threshold, and
    # letting one through would gather some real anchor's pose.
    keep = (detections[0, :, SCORE_COLUMN] > conf_threshold) & (anchor_index[0] >= 0)
    kept = detections[0][keep]
    dense_points = decode_keypoints(raw_points, anchor_points, strides)
    canvas_points = dense_points[0].index_select(0, anchor_index[0][keep].to(torch.int64))
    mapped = to_letterboxed_original(
        kept.unsqueeze(0).cpu(), orig_size=orig_size, letterboxed_size=canvas, allow_upscale=letterbox.allow_upscale
    )
    return KeypointPrediction(
        detections=mapped[0],
        keypoints=_points_to_original(canvas_points.cpu(), letterbox, orig_size, canvas),
    )


def _decode_with_points(
    head_out: DualHeadOutput,
    decoder: DecodePath,
    conf_threshold: float,
    anchor_points: Tensor,
    strides: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode the selected path and return its detections, anchor indices and raw points.

    :func:`_decode_with_coefficients` for the point stem, and the same rule holds: the
    three come from one branch and must keep coming from one branch, since pairing
    either decoder's rows with the other branch's points yields a plausible pose of a
    different object. Choosing all three in one place makes that pairing unrepresentable
    rather than merely unlikely.
    """
    if decoder == "e2e":
        detections, anchor_index = TopKDecoder().decode_with_indices(
            head_out.o2o_cls, head_out.o2o_box, anchor_points, strides
        )
        raw_points = head_out.o2o_keypoints
    else:
        detections, anchor_index = NMSDecoder(conf_threshold=conf_threshold).decode_with_indices(
            head_out.o2m_cls, head_out.o2m_box, anchor_points, strides
        )
        raw_points = head_out.o2m_keypoints
    if raw_points is None:
        raise ValueError(
            f"this checkpoint's task is {_KEYPOINTS_TASK!r} but its head emits no points on the "
            f"{decoder!r} branch. The checkpoint was built without the point stems and cannot "
            f"produce a pose."
        )
    return detections, anchor_index, raw_points


def _points_to_original(
    points: Tensor, letterbox: Letterbox, orig_size: tuple[int, int], canvas: tuple[int, int]
) -> Tensor:
    """Map ``(N, K, 2)`` canvas points onto the original pixel grid, keeping the axis shape.

    The inverse takes a flat point list, so the instance and point axes are folded
    together for the call and restored after. Folding rather than looping is not merely
    faster: one call means one mapping, so no row can take a different inverse from its
    neighbour.

    An empty prediction skips the call and returns an empty stack of the right shape --
    ``K`` survives, since a caller reading :attr:`KeypointPrediction.num_keypoints` on an
    image with nothing in it should still learn the schema.
    """
    if not points.shape[0]:
        return points
    instances, num_points, coords = points.shape
    flat = letterbox.inverse_map(points.reshape(instances * num_points, coords), orig_size, canvas)
    return flat.reshape(instances, num_points, coords)
