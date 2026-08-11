# SPDX-License-Identifier: Apache-2.0
"""Four-image mosaic assembly for boxes and instance masks (WP-011, blueprint sec. 5.9).

Mosaic (R9, YOLOv4, arXiv:2004.10934) stitches four training images into one
``2S x 2S`` canvas so a single training sample carries four scenes' worth of
context and scale variety. A centre point ``(cx, cy)`` is sampled uniformly from
the central region ``[0.5S, 1.5S]`` on each axis; the four images are anchored to
that centre — the first with its bottom-right corner at ``(cx, cy)``, the second
its bottom-left, the third its top-right, the fourth its top-left — so each fills
the quadrant between the centre and one canvas corner, cropped where it overruns
its quadrant. Uncovered canvas is filled with the same ``114/255`` grey as
letterbox and affine.

Not a :class:`~lucid_yolo.data.transforms.GeometricTransform`:
    Every transform in that protocol maps *one* image and its targets to a
    transformed pair. Mosaic instead **consumes four** ``(image, targets)`` pairs
    and yields one, so it deliberately does not conform to the protocol; it is an
    *assembly* run before the per-image geometric transforms. The stitched
    ``2S x 2S`` result is typically letterboxed/affined down to ``S`` by the
    surrounding pipeline — that resize is **not** done here.

Targets:
    Each image's boxes (and polygon rings, when present) are shifted by that
    image's placement offset, clipped to the canvas, and filtered by a minimum
    clipped side length (``min_box_size``) and a minimum kept-area fraction
    (``min_visibility`` = clipped-extent area / pre-clip-extent area) — the same
    criteria :class:`~lucid_yolo.data.affine.RandomAffine` applies, kept local here
    to avoid coupling the two transforms. Because each image is anchored at the
    centre and extends only toward its canvas corner, clipping to the canvas is
    equivalent to clipping to the quadrant: no box can leak across the centre
    into a neighbour's territory. The four filtered target sets are merged with
    :meth:`~lucid_yolo.data.targets.Targets.concat`, which requires consistent
    polygon presence across all inputs.

Rotated boxes (WP-058):
    Placement is a pure translation, which preserves the long-edge form exactly, so
    a rotated box is shifted rather than re-fitted
    (:func:`~lucid_yolo.data.rotated_aug.shift_rboxes`). What the canvas edge does to
    it is the interesting part: the box is clipped by WP-057's clipper and re-fitted
    at its own orientation, ``boxes`` becomes the envelope of the clipped region, and
    the ``min_box_size`` / ``min_visibility`` rule above decides survival — the same
    rule, over both modalities. This is where augmentation **diverges from WP-057**:
    R18's rule *flags* a clipped part difficult and keeps it, but that is dataset
    preparation, and :class:`~lucid_yolo.data.targets.Targets` carries no ``difficult``
    field for a training-time transform to flag into, so an instance below the
    threshold is dropped instead (A40). Filtering means the rotated path needs
    WP-056's instance-axis invariant; an input breaking it raises
    :class:`ValueError`.

Testability:
    Centre sampling is driven by an optional :class:`torch.Generator`, so a seeded
    generator gives byte-identical results. The sampled centre is stashed on
    :attr:`MosaicAssembly.last_center` after each call for introspection (asserting
    quadrant placement, recovering the centre, etc.).
"""

from __future__ import annotations

import torch
from torch import Tensor

from lucid_yolo.data.rotated_aug import (
    check_rotated_pairing,
    clip_rboxes_to_canvas,
    rbox_envelopes,
    shift_rboxes,
)
from lucid_yolo.data.targets import Targets
from lucid_yolo.data.transforms import boxes_from_polygons

__all__ = ["MosaicAssembly"]

#: Default pad colour: mid-grey ``114/255`` per the YOLO-lineage convention (matches letterbox/affine).
_DEFAULT_PAD_VALUE = 114.0 / 255.0
#: Mosaic combines exactly this many images.
_MOSAIC_IMAGE_COUNT = 4


class MosaicAssembly:
    """Assemble four images and their targets into one ``2S x 2S`` mosaic (WP-011).

    Each call samples a centre ``(cx, cy)`` uniformly from ``[0.5S, 1.5S]`` on each
    axis, anchors the four images to that centre (image 0's bottom-right corner,
    image 1's bottom-left, image 2's top-right, image 3's top-left all meet at the
    centre), crops each to its quadrant, and grey-fills any uncovered canvas. Each
    image's boxes/polygons are shifted by its placement offset, clipped to the
    canvas, filtered by ``min_box_size`` and ``min_visibility``, then merged via
    :meth:`~lucid_yolo.data.targets.Targets.concat`.

    This is an **assembly, not** a :class:`~lucid_yolo.data.transforms.GeometricTransform`:
    it consumes four ``(image, targets)`` pairs rather than one, so its call
    signature deliberately differs from that protocol. The images may differ in
    size but must share a channel count, dtype and device.

    Rotated boxes are shifted with their image, clipped to the canvas and filtered
    by the same two thresholds (WP-058, module docstring), with ``boxes`` recomputed
    as the envelope of the rotated geometry. Because that path filters, an input
    whose ``rboxes`` do not share the instance axis with its ``boxes``, or that
    carries polygons alongside them, raises :class:`ValueError`.

    Args:
        target_size: The base size ``S``; the assembled canvas is ``2S x 2S``.
        generator: Optional :class:`torch.Generator` for seeded, reproducible
            centre sampling. Defaults to ``None`` (global RNG).
        pad_value: Fill value for uncovered canvas. Defaults to ``114 / 255``.
        min_box_size: Minimum clipped side length in pixels for an instance to be
            kept. Defaults to ``2.0``.
        min_visibility: Minimum kept-area fraction (clipped-extent area divided by
            pre-clip-extent area) for an instance to be kept. Defaults to ``0.1``.

    Attributes:
        last_center: The ``(cx, cy)`` integer centre sampled on the most recent
            call, or ``None`` before the first call.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> gen = torch.Generator().manual_seed(0)
        >>> mosaic = MosaicAssembly(target_size=16, generator=gen)
        >>> items = [(torch.zeros(3, 16, 16), Targets.empty()) for _ in range(4)]
        >>> out_image, out_targets = mosaic(items)
        >>> out_image.shape
        torch.Size([3, 32, 32])
        >>> out_targets.boxes.shape
        torch.Size([0, 4])

        ```
    """

    def __init__(
        self,
        target_size: int,
        generator: torch.Generator | None = None,
        pad_value: float = _DEFAULT_PAD_VALUE,
        min_box_size: float = 2.0,
        min_visibility: float = 0.1,
    ) -> None:
        self.target_size = int(target_size)
        self.generator = generator
        self.pad_value = float(pad_value)
        self.min_box_size = float(min_box_size)
        self.min_visibility = float(min_visibility)
        self.last_center: tuple[int, int] | None = None

    def __call__(self, items: list[tuple[Tensor, Targets]]) -> tuple[Tensor, Targets]:
        """Assemble ``items`` into one mosaic image and its merged targets.

        Args:
            items: Exactly four ``(image, targets)`` pairs. Each ``image`` is a CHW
                float tensor (in the same value range as the grey fill, i.e.
                ``[0, 1]``); the images may differ in spatial size but must share a
                channel count, dtype and device. Non-empty ``rboxes`` must share the
                instance axis with that input's ``boxes``, and polygon presence must
                be consistent across all four (a requirement of
                :meth:`~lucid_yolo.data.targets.Targets.concat`).

        Returns:
            The ``(C, 2S, 2S)`` stitched image and the shifted, clipped, filtered
            and merged targets.

        Raises:
            ValueError: If ``items`` does not hold exactly four pairs, or an input's
                non-empty ``rboxes`` break WP-056's instance-axis invariant.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> gen = torch.Generator().manual_seed(0)
            >>> mosaic = MosaicAssembly(target_size=8, generator=gen)
            >>> box = Targets(boxes=torch.tensor([[1.0, 1.0, 5.0, 5.0]]), labels=torch.tensor([0]))
            >>> items = [(torch.rand(3, 8, 8), box.clone()) for _ in range(4)]
            >>> out_image, out_targets = mosaic(items)
            >>> out_image.shape
            torch.Size([3, 16, 16])
            >>> mosaic.last_center is not None
            True

            ```
        """
        self._check_items(items)
        canvas_size = 2 * self.target_size
        reference = items[0][0]
        canvas = reference.new_full((reference.shape[0], canvas_size, canvas_size), self.pad_value)
        cx, cy = self._sample_center()
        self.last_center = (cx, cy)
        placed: list[Targets] = []
        for index, (image, targets) in enumerate(items):
            off_x, off_y, quadrant = self._placement(index, cx, cy, image.shape[1], image.shape[2], canvas_size)
            self._paste(canvas, image, off_x, off_y, quadrant)
            placed.append(self._place_targets(targets, off_x, off_y, canvas_size))
        return canvas, Targets.concat(placed)

    def _check_items(self, items: list[tuple[Tensor, Targets]]) -> None:
        """Validate the input count and every input's rotated instance-axis pairing."""
        if len(items) != _MOSAIC_IMAGE_COUNT:
            raise ValueError(f"MosaicAssembly requires exactly {_MOSAIC_IMAGE_COUNT} items; got {len(items)}")
        for _image, targets in items:
            check_rotated_pairing(targets)

    def _sample_center(self) -> tuple[int, int]:
        """Sample an integer centre ``(cx, cy)`` uniformly from ``[0.5S, 1.5S]`` per axis."""
        low = 0.5 * self.target_size
        high = 1.5 * self.target_size
        cx = int(torch.empty((), dtype=torch.float64).uniform_(low, high, generator=self.generator).item())
        cy = int(torch.empty((), dtype=torch.float64).uniform_(low, high, generator=self.generator).item())
        return cx, cy

    @staticmethod
    def _placement(
        index: int, cx: int, cy: int, height: int, width: int, canvas_size: int
    ) -> tuple[int, int, tuple[int, int, int, int]]:
        """Return the image's placement offset and its destination quadrant.

        The offset maps image-local pixel coordinates to canvas coordinates
        (``canvas = local + offset``). The quadrant ``(x1, y1, x2, y2)`` is the
        canvas region this image may occupy; the image is anchored so its inner
        corner sits at the centre and it extends toward the corresponding canvas
        corner. Indices ``0..3`` are top-left, top-right, bottom-left, bottom-right.
        """
        left = index in (0, 2)
        top = index in (0, 1)
        off_x = cx - width if left else cx
        off_y = cy - height if top else cy
        x1, x2 = (0, cx) if left else (cx, canvas_size)
        y1, y2 = (0, cy) if top else (cy, canvas_size)
        return off_x, off_y, (x1, y1, x2, y2)

    @staticmethod
    def _paste(canvas: Tensor, image: Tensor, off_x: int, off_y: int, quadrant: tuple[int, int, int, int]) -> None:
        """Copy the part of ``image`` that lands inside ``quadrant`` onto ``canvas`` in place."""
        _, height, width = image.shape
        qx1, qy1, qx2, qy2 = quadrant
        dst_x1 = max(off_x, qx1)
        dst_y1 = max(off_y, qy1)
        dst_x2 = min(off_x + width, qx2)
        dst_y2 = min(off_y + height, qy2)
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return
        canvas[:, dst_y1:dst_y2, dst_x1:dst_x2] = image[
            :, dst_y1 - off_y : dst_y2 - off_y, dst_x1 - off_x : dst_x2 - off_x
        ]

    def _place_targets(self, targets: Targets, off_x: int, off_y: int, canvas_size: int) -> Targets:
        """Shift, clip and filter one image's targets; dispatch on rotated/polygon presence."""
        if targets.rboxes.shape[0] > 0:
            return self._place_rotated(targets, off_x, off_y, canvas_size)
        if targets.polygons:
            return self._place_with_polygons(targets, off_x, off_y, canvas_size)
        return self._place_boxes_only(targets, off_x, off_y, canvas_size)

    def _place_rotated(self, targets: Targets, off_x: int, off_y: int, canvas_size: int) -> Targets:
        """Rotated path: shift exactly, clip to the canvas, filter both axes as one."""
        shifted = shift_rboxes(targets.rboxes, float(off_x), float(off_y))
        pre_boxes = rbox_envelopes(shifted)
        rboxes, post_boxes = clip_rboxes_to_canvas(shifted, float(canvas_size), float(canvas_size))
        keep = self._keep_mask(pre_boxes, post_boxes)
        full = Targets(boxes=post_boxes, labels=targets.labels.clone(), rboxes=rboxes)
        # WP-056's invariant, checked on the way in, is what makes one mask serve both axes.
        return full.filter(keep, rkeep=keep)

    def _place_with_polygons(self, targets: Targets, off_x: int, off_y: int, canvas_size: int) -> Targets:
        """Polygon path: shift rings, clamp to canvas, recompute boxes, filter."""
        offset = torch.tensor([off_x, off_y], dtype=torch.float32)
        shifted_rings = [ring + offset for ring in targets.polygons]
        pre_boxes = boxes_from_polygons(shifted_rings)
        clipped_rings = [ring.clamp(0.0, float(canvas_size)) for ring in shifted_rings]
        post_boxes = boxes_from_polygons(clipped_rings)
        keep = self._keep_mask(pre_boxes, post_boxes)
        full = Targets(boxes=post_boxes, labels=targets.labels.clone(), polygons=clipped_rings)
        return full.filter(keep)

    def _place_boxes_only(self, targets: Targets, off_x: int, off_y: int, canvas_size: int) -> Targets:
        """Box-only path: shift boxes, clip to canvas, filter."""
        shift = torch.tensor([off_x, off_y, off_x, off_y], dtype=torch.float32)
        pre_boxes = targets.boxes + shift
        post_boxes = pre_boxes.clamp(0.0, float(canvas_size))
        keep = self._keep_mask(pre_boxes, post_boxes)
        full = Targets(boxes=post_boxes, labels=targets.labels.clone())
        return full.filter(keep)

    def _keep_mask(self, pre_boxes: Tensor, post_boxes: Tensor) -> Tensor:
        """Boolean keep mask from clipped size and kept-area (visibility) thresholds."""
        widths = post_boxes[:, 2] - post_boxes[:, 0]
        heights = post_boxes[:, 3] - post_boxes[:, 1]
        pre_area = (pre_boxes[:, 2] - pre_boxes[:, 0]).clamp(min=0.0) * (pre_boxes[:, 3] - pre_boxes[:, 1]).clamp(
            min=0.0
        )
        post_area = widths.clamp(min=0.0) * heights.clamp(min=0.0)
        visibility = torch.where(pre_area > 0.0, post_area / pre_area.clamp(min=1e-12), torch.zeros_like(pre_area))
        return (widths >= self.min_box_size) & (heights >= self.min_box_size) & (visibility >= self.min_visibility)
