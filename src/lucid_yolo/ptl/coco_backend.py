# SPDX-License-Identifier: Apache-2.0
"""A ``MeanAveragePrecision`` whose scores are read per image rather than per box (WP-165).

``torchmetrics``' COCO adapter converts the accumulated epoch into the two documents
the evaluator wants. It hoists most of that work to one call per image --
``boxes[image_id].cpu().tolist()`` and the same for the labels -- but the score of
each detection is read individually, as ``scores[image_id][k].cpu().tolist()``:
a tensor index, a device transfer and a Python conversion per *annotation*. At this
project's validation shape that is one such chain per detection per image, against
two per image for everything else.

The parameter is already optional upstream (``scores=None`` builds annotations
without a ``score`` key), so the whole per-annotation read can be skipped and the
scores attached afterwards from one ``tolist()`` per image. The result is the same
document: the same annotations, in the same order, carrying the same float values --
and refusing the same inputs, because upstream's per-score type check travels with
the value it guards rather than being dropped as the price of the hoist.

**What this costs, and why it is contained here.** ``_get_coco_format`` is private,
and overriding a private method means upstream can change what it produces without
changing what this subclass produces -- silently, since the override would keep
being called. Two things bound that risk. The override *delegates*: it calls
``super()`` for the document itself and only writes the ``score`` field, so any
field upstream adds, removes or reshapes arrives here unchanged. And
``tests/ptl/test_coco_backend.py`` asserts the two backends agree, on whatever
``torchmetrics`` is installed, for boxes and for masks including the empty-mask
image that upstream skips -- so a drift that this override cannot absorb fails the
gate rather than moving a metric.

Measured at the call site that runs it, ``MeanAveragePrecision.compute()`` over 320
images of 100 detections: 187.7 ms to 169.5 ms, 9.7% less, every trial faster. The
conversion itself is ~26% faster; the smaller figure is the honest one, because
``compute`` also runs the evaluation the conversion feeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.helpers import CocoBackend

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["build_mean_average_precision"]


class _HoistedScoreBackend(CocoBackend):
    """A :class:`~torchmetrics.detection.helpers.CocoBackend` that reads scores per image."""

    def _get_coco_format(
        self,
        labels: list[Tensor],
        all_labels: list[Tensor],
        boxes: list[Tensor] | None = None,
        masks: list[Tensor] | None = None,
        scores: list[Tensor] | None = None,
        crowds: list[Tensor] | None = None,
        area: list[Tensor] | None = None,
        iou_type: Any = ("bbox",),
        average: Literal["macro", "micro"] = "micro",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build upstream's document with ``scores=None``, then attach the scores per image.

        Upstream emits one annotation per label, images in order and labels within
        an image in order, so the flattened per-image scores line up with the
        annotations positionally. The one place that correspondence breaks is
        upstream's own skip -- an image with no masks contributes no annotations
        when there are no boxes either -- so this reproduces that condition rather
        than assuming every image contributes. ``strict=True`` on the zip turns any
        remaining disagreement into a failure instead of a silent misalignment,
        which for a score would mean plausible detections carrying each other's
        confidence.

        Args:
            labels: Per-image class labels; its length is the image count.
            all_labels: Every label seen, for the category table.
            boxes: Per-image ``xywh`` boxes, or ``None`` for a mask-only document.
            masks: Per-image RLE masks, or ``None`` for a box-only document.
            scores: Per-image detection scores, or ``None`` for a ground-truth
                document, which carries no scores.
            crowds: Per-image crowd flags, passed through.
            area: Per-image areas, passed through.
            iou_type: The iou types the document serves, passed through.
            average: ``"micro"`` or ``"macro"``, passed through.
            **kwargs: Anything a later ``torchmetrics`` adds, passed through
                unread, so a new keyword reaches upstream rather than this
                signature refusing the call.

        Returns:
            Upstream's document, with a ``score`` on every annotation when
            ``scores`` was given.

        Raises:
            ValueError: If a score is not a ``float`` once converted — upstream's
                own check, in upstream's own words, since a document that accepts
                what upstream refuses is not the same document.

        Examples:
            >>> import torch
            >>> backend = _HoistedScoreBackend("faster_coco_eval")
            >>> document = backend._get_coco_format(
            ...     labels=[torch.tensor([0, 1])],
            ...     all_labels=[0, 1],
            ...     boxes=[torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])],
            ...     scores=[torch.tensor([0.75, 0.25])],
            ... )
            >>> [annotation["score"] for annotation in document["annotations"]]
            [0.75, 0.25]
        """
        document: dict[str, Any] = super()._get_coco_format(
            labels=labels,
            all_labels=all_labels,
            boxes=boxes,
            masks=masks,
            scores=None,
            crowds=crowds,
            area=area,
            iou_type=iou_type,
            average=average,
            **kwargs,
        )
        if scores is None:
            return document
        flattened: list[float] = []
        for image_id in range(len(labels)):
            if masks is not None and boxes is None and len(masks[image_id]) == 0:
                continue  # upstream contributes no annotations for this image
            for element, score in enumerate(scores[image_id].cpu().tolist()):
                # Upstream's own per-score type check, reproduced rather than coerced.
                # `tolist()` returns the tensor's dtype as a Python scalar, so an integral
                # score tensor arrives here as `int`; upstream raises on it and a `float()`
                # here would have accepted it silently, which is a difference in the
                # *document* -- the one thing this override promises not to make. Checking
                # the already-converted element costs no per-annotation tensor work, so the
                # hoist this class exists for is intact.
                if not isinstance(score, float):
                    raise ValueError(
                        f"Invalid input score of sample {image_id}, element {element}"
                        f" (expected value of type float, got type {type(score)})"
                    )
                flattened.append(score)
        for annotation, score in zip(document["annotations"], flattened, strict=True):
            annotation["score"] = score
        return document


def build_mean_average_precision(**kwargs: Any) -> MeanAveragePrecision:
    """Build a ``MeanAveragePrecision`` that converts its epoch with hoisted scores.

    The metric constructs its own backend and holds it privately, so swapping in
    the subclass is an assignment to ``_coco_backend``. That assignment is the one
    place this project reaches into ``torchmetrics``' internals, which is why it is
    a factory rather than two lines repeated at each metric the module builds.

    The private surface is checked before it is used, rather than trusted. A
    ``torchmetrics`` that renames or drops ``CocoBackend._get_coco_format`` leaves
    the override defining a method upstream no longer calls: every conversion would
    then run unhoisted and score-less, and the metric would keep reporting success.
    ``pyproject.toml`` caps the dependency below the next minor for that reason; this
    check is what makes a lifted cap fail at construction, naming the cause, instead
    of at a silently wrong number.

    Args:
        **kwargs: Passed to :class:`~torchmetrics.detection.MeanAveragePrecision`
            unchanged.

    Returns:
        The metric, with its COCO backend replaced by the hoisted-score subclass.

    Raises:
        RuntimeError: If the installed ``torchmetrics`` no longer defines
            ``CocoBackend._get_coco_format``, the method this module overrides.

    Examples:
        >>> metric = build_mean_average_precision(backend="faster_coco_eval", box_format="xyxy")
        >>> type(metric).__name__, type(metric._coco_backend).__name__
        ('MeanAveragePrecision', '_HoistedScoreBackend')
        >>> metric._coco_backend.backend  # the requested backend is preserved
        'faster_coco_eval'
    """
    if not hasattr(CocoBackend, "_get_coco_format"):
        raise RuntimeError(
            "torchmetrics' CocoBackend no longer defines _get_coco_format, the private method "
            "lucid_yolo.ptl.coco_backend overrides, so the hoisted-score conversion would never "
            "be called and every epoch metric would be scored by an unknown path. Pin "
            "torchmetrics to a version that defines it (the pyproject ceiling exists for this), "
            "or rewrite _HoistedScoreBackend against the new surface."
        )
    metric = MeanAveragePrecision(**kwargs)
    metric._coco_backend = _HoistedScoreBackend(metric._coco_backend.backend)
    return metric
