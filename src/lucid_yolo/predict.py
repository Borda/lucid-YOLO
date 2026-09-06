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

Module ownership:
    All four entry points **mutate the module they are given**. Each runs
    ``module.to(run_on).eval()`` before its forward pass, which moves the parameters and
    buffers of the caller's own object and switches its training flag — both in place,
    both still in effect after the function returns. A caller who hands over a CUDA
    module and passes ``device=torch.device("cpu")`` gets its detections and a module
    left on the CPU; a caller who was mid-training and reuses the module afterwards
    finds it in eval mode, with batch-norm reading its running statistics instead of the
    batch's.

    This is deliberate and not worth copying around: a module is the large object in the
    call and the normal caller — :mod:`lucid_yolo.cli.predict` — loads it, predicts, and
    drops it, so a defensive :func:`copy.deepcopy` would double peak memory for every
    such caller to protect one who has another use for the object. It is recorded here
    because it is invisible at the call site. A caller who does need the module back
    unchanged copies it, or restores it with ``module.to(previous_device).train()``.

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

from lucid_yolo.assign.grid import anchor_grid, require_grid_side
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
from lucid_yolo.validate import require_in_range, require_one_of

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from lucid_yolo.models.heads.detect import BranchName, BranchOutput
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

#: Each task's entry point and what that entry point answers with, in the wording its
#: siblings' refusals quote. One row per task, and every refusal below is composed from
#: the rows it does *not* own, so a message cannot name a subset of its siblings.
#:
#: This is the table because the hand-written form failed exactly once and silently.
#: Three of these functions predate the fourth; each was written naming the two siblings
#: that existed then, and WP-152 added ``predict_keypoints`` — whose own message names
#: all three — without revisiting the three that now had a sibling they never mentioned.
#: A reader holding a pose checkpoint and calling :func:`predict_image` was told about
#: segmentation and orientation and not about the function they wanted. A fifth task is
#: one row here and appears in all four messages at once.
_TASK_ENTRY_POINTS: dict[str, tuple[str, str]] = {
    _DETECT_TASK: ("predict_image", "answers with axis-aligned corners and nothing beside them"),
    _SEGMENT_TASK: ("predict_segmentation", "returns each object's instance mask beside its box"),
    _OBB_TASK: ("predict_oriented", "returns rotated boxes, a centre and two extents and an angle"),
    _KEYPOINTS_TASK: ("predict_keypoints", "returns a point set row-aligned with each box"),
}


def _wrong_task_message(handled: str, found: str) -> str:
    """Compose the refusal one entry point raises for a checkpoint of another task.

    The message names the function that was called, the task it serves, the task the
    checkpoint actually carries, and **every** other entry point with what it answers
    with — built from :data:`_TASK_ENTRY_POINTS` rather than written out, so no message
    can go stale when a task is added. The point of naming all of them is that the
    caller's next move is in the message: they are holding a checkpoint and want the
    function that reads it.

    Args:
        handled: The task the calling entry point serves.
        found: The task the checkpoint actually carries.

    Returns:
        The refusal text, ready to hand to :class:`ValueError`.

    Examples:
        >>> print(_wrong_task_message("detect", "keypoints"))  # doctest: +NORMALIZE_WHITESPACE
        predict_image handles task='detect'; this checkpoint's task is 'keypoints'.
        Each task has its own entry point: a 'segment' checkpoint goes through
        predict_segmentation, which returns each object's instance mask beside its box;
        a 'obb' checkpoint goes through predict_oriented, which returns rotated boxes,
        a centre and two extents and an angle; a 'keypoints' checkpoint goes through
        predict_keypoints, which returns a point set row-aligned with each box.
    """
    own_function = _TASK_ENTRY_POINTS[handled][0]
    siblings = "; ".join(
        f"a {task!r} checkpoint goes through {function}, which {answers}"
        for task, (function, answers) in _TASK_ENTRY_POINTS.items()
        if task != handled
    )
    return (
        f"{own_function} handles task={handled!r}; this checkpoint's task is {found!r}. "
        f"Each task has its own entry point: {siblings}."
    )


def _check_predict_arguments(decoder: DecodePath, conf_threshold: float, img_size: int) -> None:
    """Refuse the three argument values every entry point below would otherwise answer plausibly for.

    One helper rather than a guard per function: the same three arguments appear in all
    four signatures, and a check written four times is four chances to later fix three of
    them. It runs at the entry point rather than at the dispatch, because the dispatch is
    a forward pass later — a canvas the strides do not divide fails inside the neck's
    concatenation before the decode is reached at all.

    ``decoder`` is the one whose violation is silent. Every dispatch below reads
    ``decoder == "e2e"`` and takes the one-to-many branch for anything else, so ``"E2E"``,
    ``"nms "`` or a typo answers with the other branch's boxes — and, in
    :func:`predict_oriented`, the other branch's headings — with nothing in the output to
    read as wrong. :data:`DECODE_PATHS` is the runtime spelling of the ``Literal`` that
    protects statically-checked callers only, and this is the check it exists for.

    Args:
        decoder: The requested decode path.
        conf_threshold: The requested confidence cut.
        img_size: The requested letterbox side.

    Raises:
        ValueError: If ``decoder`` is not one of :data:`DECODE_PATHS`, if
            ``conf_threshold`` is outside ``[0, 1]``, or if ``img_size`` is not a positive
            multiple of every head stride.

    Examples:
        >>> _check_predict_arguments("e2e", 0.25, 640)  # a usable trio: returns nothing
        >>> _check_predict_arguments("E2E", 0.25, 640)
        Traceback (most recent call last):
            ...
        ValueError: decoder must be one of ('e2e', 'nms'); got 'E2E'
    """
    require_one_of("decoder", decoder, DECODE_PATHS)
    require_in_range("conf_threshold", conf_threshold, 0.0, 1.0)
    require_grid_side("img_size", img_size)


def _branch_for(decoder: DecodePath) -> BranchName:
    """Return the head branch the named decode path reads.

    The one place that states the pairing, so the four entry points cannot disagree about
    it: the top-k path reads the one-to-one branch and the suppression path the dense one,
    which is what every decode helper below already assumed field by field. Stating it once
    is what lets the module run a single branch — the other one used to be computed and
    dropped.

    The ``== "e2e"`` test and the one-to-many fallback are deliberately the same shape the
    decode helpers use, and :func:`_check_predict_arguments` has already rejected anything
    that is neither, so the fallback here is reached only by a genuine ``"nms"``.

    Args:
        decoder: The decode path, already validated.

    Returns:
        The name of the branch that path's decoder reads.

    Examples:
        >>> _branch_for("e2e"), _branch_for("nms")
        ('o2o', 'o2m')
    """
    return "o2o" if decoder == "e2e" else "o2m"


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
            **Mutated in place** — see the module docstring's *Module ownership*.
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
        ValueError: If the module's task is not ``detect``, naming the task it is; or if
            ``decoder``, ``conf_threshold`` or ``img_size`` is outside its domain, per
            :func:`_check_predict_arguments`.

    Examples:
        >>> callable(predict_image)  # a real call needs a checkpoint and an image file
        True
    """
    if module.task != _DETECT_TASK:
        raise ValueError(_wrong_task_message(_DETECT_TASK, module.task))
    _check_predict_arguments(decoder, conf_threshold, img_size)
    run_on = torch.device("cpu") if device is None else device
    letterbox = Letterbox(img_size)
    canvas_image, orig_size = read_letterboxed_image(image, letterbox)
    batch = canvas_image.unsqueeze(0).to(run_on)
    canvas = (int(batch.shape[-2]), int(batch.shape[-1]))

    module.to(run_on).eval()
    with torch.no_grad():
        # One branch, not both: the decode below reads a single branch's logits and boxes,
        # and running the other one's stems only to drop them is the cost this asks the
        # module not to pay.
        branch_out = module.forward_branch(batch, _branch_for(decoder))
    anchor_points, strides = anchor_grid(canvas, run_on)
    if decoder == "e2e":
        detections = TopKDecoder()(branch_out.cls, branch_out.box, anchor_points, strides)
    else:
        # Only this path is given the threshold: it decides what enters suppression, and
        # a box the threshold drops could only ever have been suppressed anyway, so the
        # survivor set is unchanged and the sort is cheaper. The top-k path's own
        # threshold merely zeroes scores that the filter below drops regardless, so
        # handing it the number too would state the same cut in two places.
        detections = NMSDecoder(conf_threshold=conf_threshold)(branch_out.cls, branch_out.box, anchor_points, strides)
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
            **Mutated in place** — see the module docstring's *Module ownership*.
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
        ValueError: If the module's task is not ``segment``, naming the task it is; if a
            module claiming that task has no mask coefficients to decode; or if
            ``decoder``, ``conf_threshold`` or ``img_size`` is outside its domain, per
            :func:`_check_predict_arguments`.

    Examples:
        >>> callable(predict_segmentation)  # a real call needs a checkpoint and an image file
        True
    """
    if module.task != _SEGMENT_TASK:
        raise ValueError(_wrong_task_message(_SEGMENT_TASK, module.task))
    _check_predict_arguments(decoder, conf_threshold, img_size)
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
        #
        # The branch-limited twin, because Eq. 7 reads one branch's coefficients: the other
        # branch's coefficient stem is the widest of the optional ones, and it used to be
        # computed here and dropped.
        branch_out, prototypes = module.forward_segmentation_branch(batch, _branch_for(decoder))
    anchor_points, strides = anchor_grid(canvas, run_on)
    detections, anchor_index, coefficients = _decode_with_coefficients(
        branch_out, decoder, conf_threshold, anchor_points, strides
    )

    # One selection, applied to the boxes and to the indices the coefficients are gathered
    # by, so the two cannot drift apart by a row. Padding rows are excluded explicitly and
    # not merely by their score: a row with no source anchor is not a detection at any
    # threshold, and letting one through would gather some real anchor's coefficients.
    keep = (detections[0, :, SCORE_COLUMN] > conf_threshold) & (anchor_index[0] >= 0)
    kept = detections[0][keep]
    canvas_masks = _canvas_masks(prototypes, coefficients[0], anchor_index[0][keep], kept, canvas)
    mapped = to_letterboxed_original(
        kept.unsqueeze(0).cpu(), orig_size=orig_size, letterboxed_size=canvas, allow_upscale=letterbox.allow_upscale
    )
    return SegmentedPrediction(
        detections=mapped[0],
        masks=masks_to_original(canvas_masks.cpu(), letterbox, orig_size),
    )


def _decode_with_coefficients(
    branch_out: BranchOutput,
    decoder: DecodePath,
    conf_threshold: float,
    anchor_points: Tensor,
    strides: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode the selected path and return its detections, anchor indices and coefficients.

    The three still come from one branch, but the branch is chosen upstream rather than
    here: :func:`_branch_for` names it once and the module runs only that half, so what
    arrives is a single :class:`~lucid_yolo.models.heads.detect.BranchOutput` with no
    second branch to pair against. That is a stronger guarantee than the field-by-field
    discipline this function used to keep — a plausible mask of a *different* object is
    now unrepresentable because the other branch's coefficients were never computed, not
    merely because three reads were written to agree.

    ``decoder`` still selects the *decoder*, which is a separate choice from the branch:
    the two paths rank and suppress differently, and only the second is given the
    threshold.
    """
    if decoder == "e2e":
        detections, anchor_index = TopKDecoder().decode_with_indices(
            branch_out.cls, branch_out.box, anchor_points, strides
        )
    else:
        # Only this path is given the threshold, for the reason `predict_image` states:
        # it decides what enters suppression, and the top-k path's own threshold would
        # merely zero scores the survivor filter drops regardless.
        detections, anchor_index = NMSDecoder(conf_threshold=conf_threshold).decode_with_indices(
            branch_out.cls, branch_out.box, anchor_points, strides
        )
    coefficients = branch_out.coeff
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
            **Mutated in place** — see the module docstring's *Module ownership*.
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
        ValueError: If the module's task is not ``obb``, naming the task it is; if a
            module claiming the task emits no angles on the branch the selected decoder
            reads; or if ``decoder``, ``conf_threshold`` or ``img_size`` is outside its
            domain, per :func:`_check_predict_arguments`.

    Examples:
        >>> callable(predict_oriented)  # a real call needs a checkpoint and an image file
        True
    """
    if module.task != _OBB_TASK:
        raise ValueError(_wrong_task_message(_OBB_TASK, module.task))
    _check_predict_arguments(decoder, conf_threshold, img_size)
    run_on = torch.device("cpu") if device is None else device
    letterbox = Letterbox(img_size)
    canvas_image, orig_size = read_letterboxed_image(image, letterbox)
    batch = canvas_image.unsqueeze(0).to(run_on)
    canvas = (int(batch.shape[-2]), int(batch.shape[-1]))

    module.to(run_on).eval()
    with torch.no_grad():
        # One branch, not both: R1 Eq. 13's heading is read from the branch the selected
        # decoder ranks, and the other branch's angle stem used to run and be dropped.
        branch_out = module.forward_branch(batch, _branch_for(decoder))
    anchor_points, strides = anchor_grid(canvas, run_on)
    detections = _decode_oriented(branch_out, decoder, conf_threshold, anchor_points, strides)
    mapped = rboxes_to_letterboxed_original(
        detections.cpu(), orig_size=orig_size, letterboxed_size=canvas, allow_upscale=letterbox.allow_upscale
    )
    image_detections = mapped[0]
    return image_detections[image_detections[:, RBOX_COLUMNS] > conf_threshold]


def _decode_oriented(
    branch_out: BranchOutput,
    decoder: DecodePath,
    conf_threshold: float,
    anchor_points: Tensor,
    strides: Tensor,
) -> Tensor:
    """Decode the selected oriented path into the fixed-size A45 batch it emits.

    The oriented analogue of :func:`_decode_with_coefficients`, and the branch discipline
    it used to enforce field by field is now structural: :func:`_branch_for` names the
    branch upstream and the module runs only that half, so the class logits, the ltrb
    distances and the **angles** arrive as one
    :class:`~lucid_yolo.models.heads.detect.BranchOutput` and cannot be mixed. Reading the
    one-to-one heading beside the dense branch's boxes — a correct-looking rectangle at a
    heading no part of the model predicted for it — is unrepresentable here rather than
    merely guarded against.

    The two paths are not two spellings of one thing. ``e2e`` ranks and suppresses
    nothing, which is R1's claim for the one-to-one branch; ``nms`` is the dense branch's
    comparison baseline and suppresses by exact rotated overlap
    (:class:`~lucid_yolo.decode.rotated_nms.RotatedNMSDecoder`, A61). Only the second is
    given ``conf_threshold``, for the reason :func:`predict_image` states — it decides what
    enters suppression, while the top-k path's own threshold would merely zero scores the
    survivor filter drops regardless.
    """
    o2o = decoder == ORIENTED_DECODE_PATH
    angles = branch_out.angle
    if angles is None:
        branch = ORIENTED_DECODE_PATH if o2o else "one-to-many"
        raise ValueError(
            f"this checkpoint's task is {_OBB_TASK!r} but its head emits no angles on the "
            f"{branch!r} branch, so R1 Eq. 13 has nothing to read. The checkpoint was "
            f"built without the orientation stems and cannot produce a heading."
        )
    if o2o:
        rboxes = decode_rboxes(branch_out.box, angles, anchor_points, strides)
        return o2o_rotated_topk(branch_out.cls, rboxes)
    # Annotated rather than returned inline: `nn.Module.__call__` is typed `Any`, and
    # returning it straight would silently widen this function's contract to `Any` too.
    suppressed: Tensor = RotatedNMSDecoder(conf_threshold=conf_threshold)(
        branch_out.cls, branch_out.box, angles, anchor_points, strides
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
            **Mutated in place** — see the module docstring's *Module ownership*.
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
        ValueError: If the module's task is not ``keypoints``, naming the task it is; if
            a module claiming that task emits no points on the selected branch; or if
            ``decoder``, ``conf_threshold`` or ``img_size`` is outside its domain, per
            :func:`_check_predict_arguments`.

    Examples:
        ```pycon
        >>> callable(predict_keypoints)  # a real call needs a checkpoint and an image file
        True

        ```
    """
    if module.task != _KEYPOINTS_TASK:
        raise ValueError(_wrong_task_message(_KEYPOINTS_TASK, module.task))
    _check_predict_arguments(decoder, conf_threshold, img_size)
    run_on = torch.device("cpu") if device is None else device
    letterbox = Letterbox(img_size)
    canvas_image, orig_size = read_letterboxed_image(image, letterbox)
    batch = canvas_image.unsqueeze(0).to(run_on)
    canvas = (int(batch.shape[-2]), int(batch.shape[-1]))

    module.to(run_on).eval()
    with torch.no_grad():
        # One branch, not both: the pose is gathered from the branch the selected decoder
        # ranked, and the other branch's point and sigma stems used to run and be dropped.
        branch_out = module.forward_branch(batch, _branch_for(decoder))
    anchor_points, strides = anchor_grid(canvas, run_on)
    detections, anchor_index, raw_points = _decode_with_points(
        branch_out, decoder, conf_threshold, anchor_points, strides
    )

    # One selection for the boxes and for the indices the points are gathered by, so the
    # two cannot drift by a row. Padding rows go by their missing anchor rather than by
    # their score: a row with no source anchor is not a detection at any threshold, and
    # letting one through would gather some real anchor's pose.
    keep = (detections[0, :, SCORE_COLUMN] > conf_threshold) & (anchor_index[0] >= 0)
    kept = detections[0][keep]
    dense_points = decode_keypoints(raw_points, anchor_points, strides)
    # No dtype cast on the index: both decoders' `decode_with_indices` document a *long*
    # tensor, so the cast this line used to carry was a no-op that only the keypoint path
    # had — its mask-path sibling in `_canvas_masks` gathers by the same indices with no
    # cast at all. Two spellings of one contract invite the reader to believe the paths
    # differ; they do not.
    canvas_points = dense_points[0].index_select(0, anchor_index[0][keep])
    mapped = to_letterboxed_original(
        kept.unsqueeze(0).cpu(), orig_size=orig_size, letterboxed_size=canvas, allow_upscale=letterbox.allow_upscale
    )
    return KeypointPrediction(
        detections=mapped[0],
        keypoints=_points_to_original(canvas_points.cpu(), letterbox, orig_size, canvas),
    )


def _decode_with_points(
    branch_out: BranchOutput,
    decoder: DecodePath,
    conf_threshold: float,
    anchor_points: Tensor,
    strides: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode the selected path and return its detections, anchor indices and raw points.

    :func:`_decode_with_coefficients` for the point stem, and the same rule holds in the
    same stronger form: the branch is chosen upstream by :func:`_branch_for`, so the three
    arrive as one :class:`~lucid_yolo.models.heads.detect.BranchOutput` and a plausible
    pose of a *different* object cannot be assembled here — the other branch's points were
    never computed.
    """
    if decoder == "e2e":
        detections, anchor_index = TopKDecoder().decode_with_indices(
            branch_out.cls, branch_out.box, anchor_points, strides
        )
    else:
        detections, anchor_index = NMSDecoder(conf_threshold=conf_threshold).decode_with_indices(
            branch_out.cls, branch_out.box, anchor_points, strides
        )
    raw_points = branch_out.keypoints
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
