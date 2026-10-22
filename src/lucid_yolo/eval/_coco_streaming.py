# SPDX-License-Identifier: Apache-2.0
"""The COCO box and mask metric engines behind :mod:`lucid_yolo.eval.coco_eval` (WP-205).

Holds the one place the torchmetrics :class:`~torchmetrics.detection.MeanAveragePrecision`
is configured (:func:`_new_metric`, on :data:`COCO_RECALL_GRID` and :data:`_MAX_DETS`), the
renaming of its output onto the reported statistic names (:func:`_named_stats`), and the two
batch-by-batch scorers :class:`~lucid_yolo.eval.coco_eval.DualPathEvaluator` streams into:
:class:`_StreamingScorer` over that metric, and its hotcoco-backed twin
:class:`_HotcocoStreamingScorer` (WP-138) with the availability probe and record builders it
needs. The one-shot entry points of :mod:`~lucid_yolo.eval.coco_eval` build their metric here
too, so both routes share one definition of average precision.

Moved out of :mod:`lucid_yolo.eval.coco_eval` unchanged, and re-exported from it under every
name a caller or test reaches it by. A patch of a name these functions look up, the metric
class among them, targets this module, not that one.
"""

from __future__ import annotations

import contextlib
import io
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from faster_coco_eval.core import mask as fce_mask
from torchmetrics.detection import MeanAveragePrecision

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from torch import Tensor

#: The 12 scalar :class:`MeanAveragePrecision` metrics, AP first then AR, in the
#: order torchmetrics reports them (``map`` = mAP50-95). The per-class and
#: ``classes`` entries of the raw compute dict are deliberately excluded.
_METRIC_KEYS: tuple[str, ...] = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
    "mar_small",
    "mar_medium",
    "mar_large",
)

#: Name prefix of the mask statistics in a combined bbox+segm report. The box
#: statistics deliberately keep their bare names, so a detection-only report and
#: the bbox half of a segmentation report are read the same way.
_SEGM_PREFIX = "segm_"

#: Number of interpolated recall points in the COCO protocol: 0.00 to 1.00 by 0.01 (R12).
_RECALL_POINTS = 101

#: The COCO recall grid as **correctly rounded** hundredths (WP-092, A46).
#:
#: Public, unlike the other protocol constants below it, because the training loop's
#: epoch metrics are built outside this module
#: (:mod:`~lucid_yolo.ptl.module` via :func:`~lucid_yolo.ptl.coco_backend.build_mean_average_precision`)
#: and must integrate on this same grid: two figures reported under one metric name and
#: summed over two different recall grids are two different statistics (audit M-03).
#:
#: Passed to the metric rather than left to its default, because torchmetrics builds this
#: grid with a **float32** ``torch.linspace`` and widens the result to Python floats: at 36
#: of the 101 indices the stored threshold is then strictly greater than the ``k/100`` it
#: stands for. The 65th is ``0.6500000357627869``, against the ``0.65`` that a recall of
#: ``13/20`` attains exactly. A class landing on such a boundary fails the comparison,
#: forfeits that point, and loses ``1/101`` of its average precision — a bias that is
#: small, one-directional and always downward.
#:
#: Evaluating ``k/100`` in float64 removes that bias outright rather than tightening it,
#: and the reason is arithmetic rather than empirical. IEEE division is correctly rounded,
#: so ``hits/positives`` and ``k/100`` land on the *same* double whenever they are equal as
#: rationals: the equality case — the only one floating point was getting wrong — is then
#: decided exactly, with no epsilon to choose. The unequal cases stay exact too at every
#: scale this instrument sees, since two distinct rationals with denominators at most ``Q``
#: differ by at least ``1/(100 Q)`` while doubles near 1 resolve ``2**-52``; the comparison
#: is settled by the values rather than by rounding for ``Q`` up to roughly ``4.5e13``,
#: against COCO classes whose ground-truth counts are measured in tens of thousands.
#:
#: This does not make the axis-aligned instrument bit-identical to
#: :func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map`, which decides the same question
#: in int64 and is exact for *any* ``Q``; it makes the two agree on every input either can
#: be given in practice. ``tests/eval/test_coco_eval.py::TestRecallGridBoundary`` pins the
#: agreement at the boundaries, and the 36-index defect itself, so a torchmetrics that
#: fixes its own grid is noticed rather than silently worked around forever.
COCO_RECALL_GRID: tuple[float, ...] = tuple(index / (_RECALL_POINTS - 1) for index in range(_RECALL_POINTS))

#: COCO's own canonical detection-count thresholds (R12) for the box and mask
#: protocols, stated explicitly rather than left to either engine's default — the
#: same reasoning as :data:`COCO_RECALL_GRID`, and given to *both* engines for the same
#: reason the recall grid is: a statistic whose detection cap depends on which
#: backend computed it is not the same statistic under one name. Equal to what
#: torchmetrics and hotcoco both default to today, but a citable,
#: version-independent number this project's report does not follow if either
#: default ever moves. Keypoints cap differently — see :data:`_KEYPOINT_MAX_DETS`.
_MAX_DETS: tuple[int, int, int] = (1, 10, 100)


def _new_metric(iou_type: str | tuple[str, ...]) -> MeanAveragePrecision:
    """Construct the metric for one ``iou_type``, on the exact recall grid.

    The single place the metric is configured, so the backend, the box format, the
    recall grid, the detection cap and the detection-cap warning setting cannot
    differ between the one-shot entry points and the streaming one
    :class:`DualPathEvaluator` drives. Passing :data:`COCO_RECALL_GRID` here is
    therefore what gives ``bbox``, ``segm``, the combined pass and the streaming
    path one definition of average precision rather than four; passing
    :data:`_MAX_DETS` is the same argument one level out, since that constant is
    what the hotcoco path already pins and a cap left to each library's default
    would let one engine's AR lines drift from the other's (WP-166).

    Why this route, and not the other two the work package weighed:
        ``rec_thresholds`` is a **documented constructor parameter** of
        :class:`~torchmetrics.detection.MeanAveragePrecision` (1.9), forwarded
        verbatim to the backend as ``params.recThrs`` in float64 — not a private
        attribute reached into after construction. The alternative of rebuilding
        average precision from the backend's raw matching would replace one
        supported keyword with a reimplementation of all twelve statistics against
        genuinely undocumented internals, to arrive at the same numbers; and
        carrying the bias in the report instead would be choosing to publish a
        known-low figure while the correction sat behind a public argument.

    What happens when this stops being true:
        Removed parameter — ``MeanAveragePrecision`` raises ``TypeError`` at
        construction, loudly, on its own. Accepted-but-ignored parameter — the
        quiet regression, which would silently restore the downward bias — is
        caught by the check below. Stored-but-unused is caught in the gate by
        ``tests/eval/test_coco_eval.py::TestRecallGridBoundary``, which asserts the
        sampled values rather than the configuration. None of the three can reach a
        report as a slightly-too-low mAP.
    """
    metric = MeanAveragePrecision(
        backend="faster_coco_eval",
        box_format="xyxy",
        iou_type=iou_type,  # type: ignore[arg-type]
        rec_thresholds=list(COCO_RECALL_GRID),
        max_detection_thresholds=list(_MAX_DETS),
    )
    if tuple(metric.rec_thresholds) != COCO_RECALL_GRID:
        raise RuntimeError(
            "MeanAveragePrecision did not honour the rec_thresholds it was given, so its "
            "recall grid is not the exact one this module requires (A46). Refusing to "
            f"report a silently biased mAP. Expected {_RECALL_POINTS} exact hundredths, "
            f"got {list(metric.rec_thresholds)[:3]}... — pin torchmetrics to a version "
            "whose rec_thresholds argument is honoured, or rework _new_metric."
        )
    if tuple(metric.max_detection_thresholds) != _MAX_DETS:
        raise RuntimeError(
            "MeanAveragePrecision did not honour the max_detection_thresholds it was given, "
            f"so its AR lines are not COCO's (WP-166). Expected {list(_MAX_DETS)}, got "
            f"{list(metric.max_detection_thresholds)}."
        )
    # The fixed 300-row decoder output routinely exceeds COCO's top-100 detection
    # cap; keeping only the 100 highest-scoring per image is the standard protocol
    # (COCOeval's maxDets), not a misconfiguration, so silence the per-call warning.
    metric.warn_on_many_detections = False
    return metric


def _named_stats(computed: Mapping[str, Tensor], combined: bool) -> dict[str, float]:
    """Rename one ``compute()`` dict onto this module's reported statistic names.

    torchmetrics names its outputs ``map``/``mar_*`` for a single ``iou_type`` but
    ``bbox_map``/``segm_map`` for a tuple of them. Both readings live here so the
    one-shot and streaming paths cannot report the same run under different names.
    """
    if not combined:
        return {key: float(computed[key]) for key in _METRIC_KEYS}
    stats = {key: float(computed[f"bbox_{key}"]) for key in _METRIC_KEYS}
    stats.update({_SEGM_PREFIX + key: float(computed[_SEGM_PREFIX + key]) for key in _METRIC_KEYS})
    return stats


class _StreamingScorer:
    """One decode path's metric, updated batch by batch rather than at the end.

    Accumulating every image's predictions and targets and scoring once is fine
    for boxes and impossible for masks. A segmentation eval of val2017 would hold
    both paths' predicted masks *and* the ground-truth masks dense and
    simultaneously — hundreds of gigabytes at original resolution. Updating per
    batch bounds that to one batch, because ``MeanAveragePrecision.update``
    RLE-encodes each mask into its state immediately (torchmetrics 1.9,
    ``_get_safe_item_values``); the dense tensors are free the moment the batch is
    scored.

    The ``iou_type`` is decided by the first batch, from whether its predictions
    carry masks, and the metric is constructed only then: a run that yields no
    batches at all reports the same all-zero dict the one-shot entry points do,
    rather than raising from a metric that was never updated.
    """

    def __init__(self) -> None:
        self._metric: MeanAveragePrecision | None = None
        self._combined = False

    def update(self, preds: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]) -> None:
        """Fold one batch into the metric state, constructing it on the first call."""
        if not preds:
            return
        if self._metric is None:
            self._combined = "masks" in preds[0]
            self._metric = _new_metric(("bbox", "segm") if self._combined else "bbox")
        with contextlib.redirect_stdout(io.StringIO()):
            self._metric.update(preds, targets)

    def compute(self) -> dict[str, float]:
        """Return the named statistics, or an all-zero dict if nothing was scored."""
        if self._metric is None:
            return dict.fromkeys(_METRIC_KEYS, 0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            computed: Mapping[str, Tensor] = self._metric.compute()
        return _named_stats(computed, self._combined)


def hotcoco_available() -> tuple[bool, str | None]:
    """Probe whether ``hotcoco`` is installed and its compiled extension actually works (WP-138).

    ``hotcoco`` ships ``cp39-abi3`` wheels for its supported platforms and falls
    back to a source build needing a Rust toolchain everywhere else (musllinux,
    most notably — no wheel is published for it at all). A caller with no Rust
    toolchain either never finished ``pip install`` (an import failure this
    catches) or, rarer, has a partially-broken compiled extension that imports
    but cannot construct anything (an import alone would not catch that, so
    this also builds an empty document — the cheapest real exercise of the
    extension). Either way the answer this project needs is "does it work
    *here*", not "is the package present", so both are checked.

    Returns:
        ``(True, None)`` when hotcoco is usable. ``(False, reason)`` otherwise,
        ``reason`` a short, printable explanation — never raised, since a
        missing Rust toolchain is an expected environment shape (R36), not a defect.

    Examples:
        ```pycon
        >>> available, reason = hotcoco_available()
        >>> isinstance(available, bool)
        True

        ```
    """
    try:
        import hotcoco  # noqa: PLC0415 - optional dependency, probed lazily rather than required at import

        hotcoco.COCO({"images": [], "annotations": [], "categories": []})
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    return True, None


def _encode_mask_rle(mask: Tensor) -> dict[str, object]:
    """RLE-encode one boolean mask for a hotcoco annotation/result record.

    Two of R36's five documented hotcoco traps live here, both silent-wrong-number
    failures rather than errors: ``mask.encode``
    accepts ``uint8`` only and raises a confusing ``TypeError`` on a boolean
    array, and its ``counts`` field comes back as ``bytes`` — hotcoco's own COCO
    constructor decodes only the ``str`` form, silently reading a bytes payload
    back as an **empty** mask (verified directly: a bytes-``counts`` round trip
    through this project's own installed ``hotcoco`` scored a 9-pixel mask as
    0 pixels, no exception). Centralized here so both traps are fixed in
    exactly one place rather than at every call site that builds a record.
    """
    dense = mask.detach().cpu().numpy().astype("uint8")
    rle = fce_mask.encode(np.asfortranarray(dense))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("utf-8")}


@contextlib.contextmanager
def _redirect_native_output() -> Iterator[None]:
    """Silence hotcoco's own stdout/stderr writes for the scope of ``summarize()``.

    ``contextlib.redirect_stdout`` only reroutes Python's ``sys.stdout`` object;
    hotcoco writes from Rust directly to file descriptor 1 (its summary table)
    and 2 (a warning naming any evaluator parameter that differs from COCO's
    defaults). Verified directly: wrapping a hotcoco ``summarize()`` call in
    ``contextlib.redirect_stdout`` left the summary table on the real terminal
    and an empty capture buffer — R36's finding, reproduced here. Descriptor-level
    ``os.dup2`` is the only redirect that reaches it.

    Scoped to ``summarize()`` alone, and measured rather than assumed: running
    each phase of a hotcoco pass with descriptors 1 and 2 captured separately
    (document construction, ``loadRes``, the evaluator constructor, the ``params``
    assignment, ``evaluate``, ``accumulate``, ``summarize``, ``get_results``)
    showed every phase but ``summarize`` writing zero bytes — on a ``bbox``
    document and on a ``segm`` one, and including a run whose ground truth carried
    a negative-width box, a negative ``area`` and an out-of-range ``iscrowd``.
    Blinding the other phases therefore bought no
    silence and cost the one channel a malformed record would be reported on —
    which matters here, since this scorer builds those records itself (WP-166).
    Both descriptors still stay closed over ``summarize`` because the parameter
    warning also lands there, not at the assignment that provokes it.
    """
    saved_fds = [os.dup(fd) for fd in (1, 2)]
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            yield
        finally:
            os.close(devnull)
    finally:
        for fd, saved in zip((1, 2), saved_fds, strict=True):
            os.dup2(saved, fd)
            os.close(saved)


def _hotcoco_stats(preds: list[dict[str, object]], gt: Mapping[str, object], iou_type: str) -> dict[str, float]:
    """Run one hotcoco ``COCOeval`` pass and return its ten canonical keys, named ``get_results()`` style.

    Reads :meth:`~hotcoco.COCOeval.get_results` rather than the positional
    ``stats`` array (both exist; the array is index-order-dependent and this
    project's own conventions refuse a silent-reorder risk when a named
    alternative is available — the same choice :func:`_named_stats` already
    makes for the torchmetrics path). ``recThrs``/``maxDets`` are get-mutate-
    reassign, not in-place: hotcoco's ``params`` is copy-on-read, so
    ``ev.params.recThrs = ...`` alone is a silent no-op (R36's first trap,
    reproduced against the installed hotcoco before trusting it).
    """
    # Aliased because faster_coco_eval exports a `COCO` of its own, which the sibling
    # `_coco_keypoints` imports at module level beside `COCOeval_faster`; this module
    # imports only faster_coco_eval's mask helpers, and the prefix keeps the two
    # packages' same-named classes apart for a reader of either module.
    from hotcoco import COCO as HotcocoCOCO  # noqa: PLC0415 - optional dependency, gated on availability
    from hotcoco import COCOeval as HotcocoEval  # noqa: PLC0415 - optional dependency, gated on availability

    ground_truth = HotcocoCOCO(dict(gt))
    predictions = ground_truth.loadRes(preds)
    evaluator = HotcocoEval(ground_truth, predictions, iou_type=iou_type)
    params = evaluator.params
    params.rec_thrs = list(COCO_RECALL_GRID)
    params.max_dets = list(_MAX_DETS)
    evaluator.params = params
    evaluator.evaluate()
    evaluator.accumulate()
    # Only summarize() writes to the raw descriptors, so only summarize() is
    # blinded -- the phases that would report a malformed record keep theirs.
    with _redirect_native_output():
        evaluator.summarize()
    results = evaluator.get_results()
    return {
        "map": results["AP"],
        "map_50": results["AP50"],
        "map_75": results["AP75"],
        "map_small": results["APs"],
        "map_medium": results["APm"],
        "map_large": results["APl"],
        "mar_1": results["AR1"],
        "mar_10": results["AR10"],
        "mar_100": results["AR100"],
        "mar_small": results["ARs"],
        "mar_medium": results["ARm"],
        "mar_large": results["ARl"],
    }


class _HotcocoStreamingScorer:
    """:class:`_StreamingScorer`'s hotcoco-backed twin (WP-138), same interface, same memory shape.

    hotcoco offers no incremental ``update()`` of its own — its ``COCOeval`` scores
    one whole document at once. That is not the same claim as "the whole document
    must be dense in memory": :meth:`update` RLE-encodes every mask the moment its
    batch arrives (:func:`_encode_mask_rle`), exactly where
    :class:`_StreamingScorer`'s own docstring says ``MeanAveragePrecision.update``
    already does it, so the dense tensors are equally free the moment each batch is
    scored either way. Only the small, already-encoded records accumulate across an
    epoch; the full document is assembled from them, and handed to hotcoco, only in
    :meth:`compute`.

    A combined bbox+segm model runs **two** independent hotcoco passes rather than
    one tuple-``iou_type`` call, because hotcoco's own ``dataset`` is copy-on-read
    (R36's second trap): an annotation's ``area`` field
    cannot be swapped in place between a box-area pass and a mask-area pass on one
    loaded document the way torchmetrics' internal combined path does, so this
    scorer builds two separate ground-truth/result documents instead — one whose
    fallback area is the box's, one whose fallback area is the RLE mask's —
    sharing the same underlying detections. Where the target carries the
    annotation's own ``area`` and ``iscrowd``, both documents take those instead,
    which is what torchmetrics' two halves do as well (WP-166).
    """

    def __init__(self) -> None:
        self._combined = False
        self._images: list[dict[str, object]] = []
        self._box_annotations: list[dict[str, object]] = []
        self._box_results: list[dict[str, object]] = []
        self._segm_annotations: list[dict[str, object]] = []
        self._segm_results: list[dict[str, object]] = []
        self._categories: set[int] = set()
        self._image_id = 0
        self._annotation_id = 1

    def update(self, preds: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]) -> None:
        """Encode one batch's boxes (and masks, if present) into small COCO-format records."""
        if not preds:
            return
        if self._image_id == 0:
            self._combined = "masks" in preds[0]
        for prediction, target in zip(preds, targets, strict=True):
            image_id = self._image_id
            self._image_id += 1
            height, width = _image_extent(prediction, target)
            self._images.append({"id": image_id, "height": height, "width": width})
            self._accumulate_boxes(image_id, prediction, target)
            if self._combined:
                self._accumulate_masks(image_id, prediction, target)

    def _accumulate_boxes(self, image_id: int, prediction: dict[str, Tensor], target: dict[str, Tensor]) -> None:
        """Add this image's box ground truth and detections to the box-pass records."""
        areas = target.get("area")
        crowds = target.get("iscrowd")
        for index, (box, label) in enumerate(zip(target["boxes"].tolist(), target["labels"].tolist(), strict=True)):
            x1, y1, x2, y2 = box
            self._categories.add(int(label))
            self._box_annotations.append(
                {
                    "id": self._annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": float(areas[index]) if areas is not None else max((x2 - x1) * (y2 - y1), 1.0),
                    "iscrowd": int(crowds[index]) if crowds is not None else 0,
                }
            )
            self._annotation_id += 1
        for box, score, label in zip(
            prediction["boxes"].tolist(), prediction["scores"].tolist(), prediction["labels"].tolist(), strict=True
        ):
            x1, y1, x2, y2 = box
            self._categories.add(int(label))
            self._box_results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )

    def _accumulate_masks(self, image_id: int, prediction: dict[str, Tensor], target: dict[str, Tensor]) -> None:
        """Add this image's RLE-encoded mask ground truth and detections to the segm-pass records."""
        areas = target.get("area")
        crowds = target.get("iscrowd")
        for index, (mask, label) in enumerate(zip(target["masks"], target["labels"].tolist(), strict=True)):
            rle = _encode_mask_rle(mask)
            # fce_mask.area accepts either RLE counts form (verified directly); no
            # second encode needed to get an area figure out of the same record
            # already built for the annotation itself. It is the fallback rather
            # than the value, because torchmetrics' segm half reads a supplied
            # ``area`` too -- measured on a fixture whose annotation area, mask
            # area and box area name three different buckets, and its segm
            # statistics landed in the annotation's (WP-166).
            self._segm_annotations.append(
                {
                    "id": self._annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "segmentation": rle,
                    "area": float(areas[index]) if areas is not None else float(fce_mask.area(rle)),
                    "iscrowd": int(crowds[index]) if crowds is not None else 0,
                }
            )
            self._annotation_id += 1
        for mask, score, label in zip(
            prediction["masks"], prediction["scores"].tolist(), prediction["labels"].tolist(), strict=True
        ):
            self._segm_results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "segmentation": _encode_mask_rle(mask),
                    "score": score,
                }
            )

    def compute(self) -> dict[str, float]:
        """Score the accumulated document(s) with hotcoco, or an all-zero dict if nothing was scored."""
        if not self._images:
            keys = [*_METRIC_KEYS] if not self._combined else [*_METRIC_KEYS, *(_SEGM_PREFIX + k for k in _METRIC_KEYS)]
            return dict.fromkeys(keys, 0.0)
        categories = [{"id": category, "name": str(category)} for category in sorted(self._categories)]
        box_gt = {"images": self._images, "annotations": self._box_annotations, "categories": categories}
        box_stats = _hotcoco_stats(self._box_results, box_gt, "bbox")
        if not self._combined:
            return box_stats
        segm_gt = {"images": self._images, "annotations": self._segm_annotations, "categories": categories}
        segm_stats = _hotcoco_stats(self._segm_results, segm_gt, "segm")
        return {**box_stats, **{_SEGM_PREFIX + key: value for key, value in segm_stats.items()}}


def _image_extent(prediction: dict[str, Tensor], target: dict[str, Tensor]) -> tuple[int, int]:
    """Return one image's ``(height, width)``, the field hotcoco's ``COCO`` requires and torchmetrics never tracks.

    Read from whichever side has a mask (its own dense shape is exact); boxes
    alone carry no canvas size, so the furthest box/mask corner across both
    prediction and target is the only bound this project has to build from —
    exact for a synthetic/test fixture sized to its own content, a slight
    underestimate only when every detection sits well inside the image margin,
    which never changes which detections match (IoU and OKS are corner-relative,
    not canvas-relative) and only ever affects the recorded ``height``/``width``
    metadata field itself.
    """
    if "masks" in target and target["masks"].numel():
        return int(target["masks"].shape[-2]), int(target["masks"].shape[-1])
    if "masks" in prediction and prediction["masks"].numel():
        return int(prediction["masks"].shape[-2]), int(prediction["masks"].shape[-1])
    corners = [
        tensor["boxes"].reshape(-1, 4)[:, 2:]
        for tensor in (prediction, target)
        if tensor.get("boxes") is not None and tensor["boxes"].numel()
    ]
    if not corners:
        return 1, 1
    furthest = torch.cat(corners).amax(dim=0)
    return max(1, int(furthest[1].ceil())), max(1, int(furthest[0].ceil()))
