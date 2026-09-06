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
(roadmap row 112); the task classes here reproduce
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

**One exported file answers for exactly one input shape, batch included.** The
canvas is fixed by the baked buffers, and the batch axis is fixed too:
:func:`export_graph` passes no ``dynamic_axes``, so the emitted graph declares the
static ``(B, 3, H, W)`` of the example it was traced on — ``B = 1`` unless a caller
supplies its own example. A consumer needing a second canvas or a second batch size
exports a second file rather than re-binding an axis of this one. Everything
downstream of the trace is built for that: the decoders answer with a fixed
``k``-row output precisely so no shape depends on how many objects an image holds.

**A segmentation graph cannot filter its padding rows the way
:func:`~lucid_yolo.predict.predict_segmentation` does, so it refuses a canvas that
would produce any.** That function drops rows whose anchor index is
:data:`~lucid_yolo.decode.common.PAD_ANCHOR_INDEX` before gathering mask
coefficients, because a caller looking at one picture wants a mask per object and a
boolean filter is free to shrink the output. :class:`SegmentExportGraph` cannot do
the same: a traced graph's output shape is fixed at trace time, so every one of its
:attr:`~lucid_yolo.decode.topk_e2e.TopKDecoder.k` rows is gathered, padding rows
included. Gathering a padding row's coefficients only works because the padding
index is a valid (if meaningless) position in the coefficient tensor — which is
true exactly when the anchor count reaches ``k`` and no padding row is ever
produced. Below that, the padding index is
:data:`~lucid_yolo.decode.common.PAD_ANCHOR_INDEX` (``-1``) and the gather raises
rather than silently misassembling a mask, so
:class:`SegmentExportGraph` checks the anchor count in ``__init__`` and rejects the
canvas there, naming the count and the cap. The alternative — clamping the padding
index and zeroing its gathered row, the way
:func:`~lucid_yolo.eval.coco_eval.gather_keypoints` does for
:class:`KeypointExportGraph` — would widen the supported canvases at the cost of a
second copy of that padding rule living here; nothing asks for a sub-``k``
segmentation canvas today, and a refusal that names the constraint is the cheaper
honest answer. This is the same constraint :mod:`tests.models.test_onnx_export`'s
128-pixel canvas is chosen to satisfy (336 anchors above the 300 cap); it is
restated here because this module has callers that fixture does not.

Provenance: R1 sec. 3.2.1, R3 sec. 4. Assumptions: A9, A23, A37, A45.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from lucid_yolo.assign.grid import anchor_grid
from lucid_yolo.decode.common import BOX_CORNERS
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.coco_eval import gather_keypoints
from lucid_yolo.eval.segment_decode import decode_instance_masks
from lucid_yolo.models.heads.keypoint import decode_keypoints
from lucid_yolo.models.heads.obb import decode_rboxes, o2o_rotated_topk

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "ONNX_OPSET",
    "DetectExportGraph",
    "E2EExportGraph",
    "KeypointExportGraph",
    "OrientedExportGraph",
    "SegmentExportGraph",
    "export_graph",
]

#: ONNX opset every graph in this module is exported at. Pinned rather than left to
#: :func:`torch.onnx.export`'s default (20 under torch 2.13) because the export gate
#: asserts op *types*, not an opset: a torch upgrade that moves the default would change
#: the emitted graph — different decompositions, a different IR version, a different
#: minimum runtime — while every assertion in :mod:`tests.models.test_onnx_export` kept
#: passing. Pinning makes that change a one-line edit with a visible diff instead. The
#: value is 17 because it is the last opset at IR version 8, the version the ONNX
#: runtimes and accelerator toolchains have accepted longest, and every op these four
#: graphs emit exists there.
ONNX_OPSET = 17


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
        device: Device the anchor buffers are built on *and* ``deployed`` is moved
            to, so the two halves of the graph cannot end up on different devices —
            buffers built on an accelerator while the wrapped model stayed wherever
            the caller left it is a failure at the first forward, not at
            construction. The move is in place, as
            :func:`~lucid_yolo.predict.predict_image` moves the module it is handed:
            a caller reusing ``deployed`` afterwards gets it on this device.
            Defaults to CPU, the device :func:`torch.onnx.export` traces a graph
            from.

    Raises:
        ValueError: If either ``canvas`` side is not positive and divisible by every
            stride, per :func:`~lucid_yolo.assign.grid.require_grid_canvas`, which
            :func:`~lucid_yolo.assign.grid.anchor_grid` applies as the buffers are
            built. The divisibility this class documents is therefore checked here
            rather than left to the caller: a canvas of 100 px would otherwise get
            96 px worth of anchors and pair every prediction with the wrong pixel.

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
        self.deployed = deployed.to(run_on)
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

    That divergence is what narrows this graph's supported canvases below its base
    class's, so the narrowing is checked where it can still be reported as a
    canvas choice: a canvas yielding fewer anchors than ``k`` is refused in
    ``__init__``. Left unchecked it surfaces at the first forward — and therefore
    mid-trace during :func:`export_graph`, since the exporter traces eagerly — as
    ``index -1 is out of bounds``, which names neither the canvas nor the cap.

    Args:
        deployed: The ``deploy()`` view of a
            :class:`~lucid_yolo.models.build.Segmenter`, as
            :class:`E2EExportGraph` describes.
        canvas: Input canvas ``(height, width)`` in pixels, additionally required to
            yield at least ``k`` anchors.
        device: Device the buffers are built on and ``deployed`` is moved to, as
            :class:`E2EExportGraph` describes.

    Raises:
        ValueError: If ``canvas`` fails :class:`E2EExportGraph`'s divisibility
            requirement, or yields fewer anchors than the decoder's ``k``.

    Examples:
        >>> import torch
        >>> from lucid_yolo.models.build import Segmenter
        >>> deployed = Segmenter("n", num_classes=4).eval().deploy().eval()
        >>> graph = SegmentExportGraph(deployed, canvas=(128, 128)).eval()
        >>> with torch.no_grad():
        ...     detections, masks = graph(torch.zeros(1, 3, 128, 128))
        >>> detections.shape, masks.shape  # (B, k, 6) beside (B, k, H, W)
        (torch.Size([1, 300, 6]), torch.Size([1, 300, 128, 128]))
        >>> SegmentExportGraph(deployed, canvas=(64, 64))  # 84 anchors, below k
        Traceback (most recent call last):
            ...
        ValueError: canvas (64, 64) yields 84 anchors, below the 300-detection cap; padding rows have no coefficients
    """

    def __init__(self, deployed: nn.Module, canvas: tuple[int, int], device: torch.device | None = None) -> None:
        super().__init__(deployed, canvas, device)
        anchors = int(self.anchor_points.shape[0])
        if anchors < self.decoder.k:
            raise ValueError(
                f"canvas {canvas} yields {anchors} anchors, below the {self.decoder.k}-detection cap; "
                f"padding rows have no coefficients"
            )

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


class KeypointExportGraph(E2EExportGraph):
    """Keypoint E2E export graph: the A9 tuple beside each detection's point set.

    The fourth task's graph, and the last of the four to exist -- 0.5.0 shipped a
    keypoint model that trains, validates and evaluates but had no exported view at
    all, so it was the only released task whose model could not leave PyTorch
    (WP-151).

    The composition follows :class:`SegmentExportGraph` rather than
    :class:`DetectExportGraph`, for the reason segmentation has it: the point stem
    is dense over anchors, so the decode needs the *source anchor* of each detection
    row, which is what
    :meth:`~lucid_yolo.decode.topk_e2e.TopKDecoder.decode_with_indices` returns and
    plain :meth:`~lucid_yolo.decode.topk_e2e.TopKDecoder.forward` discards. The
    ordering is the one WP-062's angle branch established and
    :class:`~lucid_yolo.ptl.module.DetectionLitModule` already uses for its own
    ``val/oks_mAP``: decode the whole dense tensor first, gather second.
    :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints` needs the entire
    anchor grid and offers no per-detection indexing, so gathering first would mean
    reconstructing each kept row's anchor and stride by hand.

    Padding rows are handled by
    :func:`~lucid_yolo.eval.coco_eval.gather_keypoints`, which clamps a padding
    index to a valid position and then zeroes that row, so a padding row yields the
    origin rather than the last anchor's pose. That is a weaker requirement than
    :class:`SegmentExportGraph`'s, which raises on one: a fixed-shape graph cannot
    drop rows, and this task's gather is defined on them.

    Examples:
        >>> import torch
        >>> from lucid_yolo.models.build import KeypointDetector
        >>> deployed = KeypointDetector("n", num_classes=4, num_keypoints=3).eval().deploy().eval()
        >>> graph = KeypointExportGraph(deployed, canvas=(128, 128)).eval()
        >>> with torch.no_grad():
        ...     detections, keypoints = graph(torch.zeros(1, 3, 128, 128))
        >>> detections.shape, keypoints.shape  # (B, k, 6) beside (B, k, K, 2)
        (torch.Size([1, 300, 6]), torch.Size([1, 300, 3, 2]))
    """

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Decode an image batch into fixed-size A9 detections and their point sets.

        Args:
            image: Input image batch of shape ``(B, 3, H, W)`` matching the
                ``canvas`` this graph was built for.

        Returns:
            A pair of the ``(B, k, 6)`` A9 detections and their ``(B, k, K, 2)``
            point coordinates in letterboxed-canvas pixels, row-aligned. Padding
            rows carry the origin and score zero.
        """
        cls, box, raw_points = self.deployed(image)
        detections, anchor_index = self.decoder.decode_with_indices(cls, box, self.anchor_points, self.strides)
        dense_points = decode_keypoints(raw_points, self.anchor_points, self.strides)
        return detections, gather_keypoints(dense_points, anchor_index)


def export_graph(
    graph: E2EExportGraph,
    path: str | Path,
    *,
    example: Tensor | None = None,
    opset_version: int = ONNX_OPSET,
) -> Path:
    """Write one export graph to an ONNX file at a pinned opset.

    The shipped counterpart to the four graph classes above: they compose the decode,
    this emits the file. It exists so the opset is pinned *somewhere a consumer runs*
    rather than left to whatever :func:`torch.onnx.export`'s default is on the torch a
    given machine installed — see :data:`ONNX_OPSET` for what that pin buys. The
    export gate calls this function rather than :func:`torch.onnx.export` directly, so
    the graph an operator gets is the graph the gate certified.

    The emitted file is static in every axis: no ``dynamic_axes`` is passed, so it
    answers for exactly the shape of ``example`` — the canvas ``graph`` baked into its
    buffers, and that example's batch. A consumer needing a second shape exports a
    second file.

    Args:
        graph: A built task graph — :class:`DetectExportGraph`,
            :class:`SegmentExportGraph`, :class:`OrientedExportGraph` or
            :class:`KeypointExportGraph` — in eval mode.
        path: Destination ``.onnx`` file. Its parent directory must exist.
        example: Input the graph is traced on, of shape ``(B, 3, H, W)`` matching
            ``graph.canvas``. Defaults to a single zero image on the graph's own
            device, which is enough: the graph has no data-dependent control flow, so
            the trace records the same ops whatever the values are.
        opset_version: ONNX opset the graph is emitted at. Defaults to
            :data:`ONNX_OPSET`; a caller pinned to a different runtime overrides it,
            and the gate's op assertions then no longer speak for the result.

    Returns:
        The written path.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> from pathlib import Path
        >>> from lucid_yolo.models.build import Detector
        >>> deployed = Detector("n", num_classes=4).eval().deploy().eval()
        >>> graph = DetectExportGraph(deployed, canvas=(64, 64)).eval()
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     written = export_graph(graph, Path(tmp) / "detect.onnx")
        ...     written.is_file()
        True
    """
    destination = Path(path)
    traced_on = torch.zeros(1, 3, *graph.canvas, device=graph.anchor_points.device) if example is None else example
    with torch.no_grad():
        # The legacy TorchScript exporter: `dynamo=True` (the torch 2.13 default) needs
        # `onnxscript`, and the legacy path emits a fully static graph here with no
        # extra dependency. It warns that it is deprecated; when it is removed, this is
        # the line that changes, and `onnxscript` joins the dev group.
        torch.onnx.export(graph, (traced_on,), str(destination), dynamo=False, opset_version=opset_version)
    return destination
