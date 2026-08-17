# SPDX-License-Identifier: Apache-2.0
"""Deployable graphs: a ``deploy()`` view composed with its E2E decode (WP-066, WP-112).

WP-066's ONNX export gate needed one importable ``nn.Module`` per task so
:func:`torch.onnx.export` had something to trace, and wrote three small wrappers
private to :mod:`tests.models.test_onnx_export` to get one. :mod:`lucid_yolo.predict`
composes the same pieces — a task's ``deploy()`` view plus its E2E decoder — for
single-image inference, independently, because the two call conventions genuinely
differ: :mod:`lucid_yolo.predict` calls a raw
:class:`~lucid_yolo.ptl.module.DetectionLitModule` (the object a checkpoint loads
into) and decodes on an unbatched result with confidence filtering, while an
exportable graph must be a plain ``nn.Module`` wrapping a ``deploy()`` view and must
answer with a **static** shape, because a traced or exported graph has none of the
former's dynamic control flow. Two independent compositions of the same decode is
the WP-053a defect class: nothing forced the export path to keep computing what the
checkpoint computes as either side changed. This module is their one shared home
(roadmap row 112); the three classes here reproduce
:func:`~lucid_yolo.predict.predict_image`, :func:`~lucid_yolo.predict.predict_segmentation`,
and :func:`~lucid_yolo.predict.predict_oriented`'s ``"e2e"`` compositions call for
call, so an exported graph and a single-image prediction can be shown to agree
rather than merely resemble each other.

**Anchor points and strides are baked into every graph as buffers**, not taken as
extra inputs. They are arguments to
:meth:`~lucid_yolo.decode.topk_e2e.TopKDecoder.forward` rather than module state,
so an exporter must either receive them or constant-fold them; baking makes a
graph's only input the image, which is what a deployment consumer expects, at the
cost of the graph being specific to the ``canvas`` it was built for.

**A segmentation graph cannot filter its padding rows the way
:func:`~lucid_yolo.predict.predict_segmentation` does.** That function drops rows
whose anchor index is :data:`~lucid_yolo.decode.common.PAD_ANCHOR_INDEX` before
gathering mask coefficients, because a caller looking at one picture wants a mask
per object and a boolean filter is free to shrink the output. :class:`SegmentExportGraph`
cannot do the same: a traced graph's output shape is fixed at trace time, so every
one of its :attr:`~lucid_yolo.decode.topk_e2e.TopKDecoder.k` rows is gathered,
padding rows included. Gathering a padding row's coefficients only works because
the padding index is a valid (if meaningless) position in the coefficient tensor —
which is true exactly when the anchor count exceeds ``k`` and no padding row is
ever produced. A caller building a graph for a canvas whose anchor count is at or
below ``k`` gets a padding row whose anchor index is
:data:`~lucid_yolo.decode.common.PAD_ANCHOR_INDEX` (``-1``), and gathering at ``-1``
raises rather than silently misassembling a mask. This is the same constraint
:mod:`tests.models.test_onnx_export`'s 128-pixel canvas is chosen to satisfy (336
anchors above the 300 cap); it is restated here because this module has callers
that fixture does not.

Provenance: R1 sec. 3.2.1, R3 sec. 4. Assumptions: A9, A23, A37, A45.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from lucid_yolo.assign.grid import anchor_grid
from lucid_yolo.decode.common import BOX_CORNERS
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.segment_decode import decode_instance_masks
from lucid_yolo.models.heads.obb import decode_rboxes, o2o_rotated_topk

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["DetectExportGraph", "E2EExportGraph", "OrientedExportGraph", "SegmentExportGraph"]


class E2EExportGraph(nn.Module):
    """A ``deploy()`` view plus its E2E decode, as one exportable module.

    The shared base of the three task graphs below: it bakes the anchor grid for
    one ``canvas`` into buffers and holds the
    :class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` every task decodes with, so a
    subclass need only compose its own task's dense outputs into a decoded tuple.
    ``forward`` is left to subclasses — that is where the three tasks differ, per
    the module docstring's compositions — so this class alone is not directly
    exportable.

    Args:
        deployed: The ``deploy()`` view whose dense outputs the decode consumes,
            as returned by :meth:`~lucid_yolo.models.build.Detector.deploy`,
            :meth:`~lucid_yolo.models.build.Segmenter.deploy`, or
            :meth:`~lucid_yolo.models.build.OrientedDetector.deploy`, matched to
            the subclass in use.
        canvas: Input canvas ``(height, width)`` in pixels, each divisible by
            every stride in :data:`~lucid_yolo.assign.grid.HEAD_STRIDES`. Baked
            into the anchor buffers, so the graph this constructs answers only
            for images of this exact size.
        device: Device the anchor buffers are built on. Defaults to CPU, the
            device :func:`torch.onnx.export` traces a graph from.

    Examples:
        >>> import torch
        >>> from lucid_yolo.models.build import Detector
        >>> deployed = Detector("n", num_classes=4).eval().deploy().eval()
        >>> graph = E2EExportGraph(deployed, canvas=(128, 128))
        >>> graph.anchor_points.shape, graph.strides.shape  # 16^2 + 8^2 + 4^2 = 336
        (torch.Size([336, 2]), torch.Size([336]))
    """

    anchor_points: Tensor
    strides: Tensor

    def __init__(self, deployed: nn.Module, canvas: tuple[int, int], device: torch.device | None = None) -> None:
        super().__init__()
        run_on = torch.device("cpu") if device is None else device
        self.deployed = deployed
        self.canvas = canvas
        points, strides = anchor_grid(canvas, run_on)
        self.register_buffer("anchor_points", points)
        self.register_buffer("strides", strides)
        self.decoder = TopKDecoder()


class DetectExportGraph(E2EExportGraph):
    """Detection E2E export graph: dense one-to-one outputs to the A9 tuple.

    Reproduces :func:`~lucid_yolo.predict.predict_image`'s ``"e2e"`` composition —
    the deployed model's one-to-one ``(cls, box)`` pair straight into
    :class:`~lucid_yolo.decode.topk_e2e.TopKDecoder`, same argument order — minus
    the letterbox inverse and the confidence filter, which are single-image
    concerns a batched, fixed-shape graph does not have.

    Examples:
        >>> import torch
        >>> from lucid_yolo.models.build import Detector
        >>> deployed = Detector("n", num_classes=4).eval().deploy().eval()
        >>> graph = DetectExportGraph(deployed, canvas=(128, 128)).eval()
        >>> with torch.no_grad():
        ...     detections = graph(torch.zeros(1, 3, 128, 128))
        >>> detections.shape  # (B, k, 6): the A9 tuple, k = 300 (A9)
        torch.Size([1, 300, 6])
    """

    def forward(self, image: Tensor) -> Tensor:
        """Decode an image batch into fixed-size A9 detections.

        Args:
            image: Input image batch of shape ``(B, 3, H, W)`` matching the
                ``canvas`` this graph was built for.

        Returns:
            Detections of shape ``(B, k, 6)`` whose last axis is the A9 tuple
            ``[x1, y1, x2, y2, score, class]`` (see
            :meth:`~lucid_yolo.decode.topk_e2e.TopKDecoder.forward`).
        """
        cls, box = self.deployed(image)
        detections: Tensor = self.decoder(cls, box, self.anchor_points, self.strides)
        return detections


class SegmentExportGraph(E2EExportGraph):
    """Segmentation E2E export graph: the A9 tuple beside its Eq. 7 instance masks.

    Reproduces :func:`~lucid_yolo.predict.predict_segmentation`'s ``"e2e"``
    composition — the same decoder call for the boxes, the same
    :meth:`~lucid_yolo.decode.topk_e2e.TopKDecoder.decode_with_indices` source of
    the anchor indices mask coefficients are gathered by, and the same
    :func:`~lucid_yolo.eval.segment_decode.decode_instance_masks` call with
    ``(prototypes, coefficients, boxes, image_size=...)`` in that order — with one
    necessary divergence the module docstring explains: every one of the ``k``
    rows is gathered, padding rows included, because the output shape is fixed.

    Examples:
        >>> import torch
        >>> from lucid_yolo.models.build import Segmenter
        >>> deployed = Segmenter("n", num_classes=4).eval().deploy().eval()
        >>> graph = SegmentExportGraph(deployed, canvas=(128, 128)).eval()
        >>> with torch.no_grad():
        ...     detections, masks = graph(torch.zeros(1, 3, 128, 128))
        >>> detections.shape, masks.shape  # (B, k, 6) beside (B, k, H, W)
        (torch.Size([1, 300, 6]), torch.Size([1, 300, 128, 128]))
    """

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Decode an image batch into fixed-size A9 detections and their masks.

        Args:
            image: Input image batch of shape ``(B, 3, H, W)`` matching the
                ``canvas`` this graph was built for.

        Returns:
            A pair of the ``(B, k, 6)`` A9 detections and their
            ``(B, k, H, W)`` boolean instance masks on the letterboxed canvas
            (see :func:`~lucid_yolo.eval.segment_decode.decode_instance_masks`),
            row-aligned.
        """
        cls, box, coeff, prototypes = self.deployed(image)
        detections, anchor_index = self.decoder.decode_with_indices(cls, box, self.anchor_points, self.strides)
        gathered = coeff.gather(1, anchor_index.unsqueeze(-1).expand(-1, -1, coeff.shape[-1]))
        masks: Tensor = decode_instance_masks(
            prototypes, gathered, detections[..., :BOX_CORNERS], image_size=self.canvas
        )
        return detections, masks


class OrientedExportGraph(E2EExportGraph):
    """Oriented E2E export graph: dense one-to-one outputs to the A45 tuple.

    Reproduces :func:`~lucid_yolo.predict.predict_oriented`'s ``"e2e"``
    composition — :func:`~lucid_yolo.models.heads.obb.decode_rboxes` over the
    deployed model's one-to-one ``(box, angle)`` pair, then
    :func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk` over the resulting
    canonical rotated boxes and the one-to-one class logits, same two calls in
    the same argument order.

    Examples:
        >>> import torch
        >>> from lucid_yolo.models.build import OrientedDetector
        >>> deployed = OrientedDetector("n", num_classes=4).eval().deploy().eval()
        >>> graph = OrientedExportGraph(deployed, canvas=(128, 128)).eval()
        >>> with torch.no_grad():
        ...     detections = graph(torch.zeros(1, 3, 128, 128))
        >>> detections.shape  # (B, k, 7): the A45 tuple, k = 300 (A9)
        torch.Size([1, 300, 7])
    """

    def forward(self, image: Tensor) -> Tensor:
        """Decode an image batch into fixed-size A45 rotated detections.

        Args:
            image: Input image batch of shape ``(B, 3, H, W)`` matching the
                ``canvas`` this graph was built for.

        Returns:
            Detections of shape ``(B, k, 7)`` whose last axis is the A45 tuple
            ``[cx, cy, w, h, theta, score, class]`` (see
            :func:`~lucid_yolo.models.heads.obb.o2o_rotated_topk`).
        """
        cls, box, angle = self.deployed(image)
        rboxes = decode_rboxes(box, angle, self.anchor_points, self.strides)
        return o2o_rotated_topk(cls, rboxes)
