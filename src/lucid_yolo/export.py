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
:func:`~lucid_yolo.predict.predict_oriented` and
:func:`~lucid_yolo.predict.predict_keypoints`'s ``"e2e"`` compositions call for
call, so an exported graph and a single-image prediction can be shown to agree
rather than merely resemble each other.

:class:`DetectExportGraph`, :class:`SegmentExportGraph`, :class:`OrientedExportGraph`
and :class:`KeypointExportGraph` provide those four task compositions.

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

**What fills a row no anchor produced differs by task**, because a traced graph's
output shape is fixed at trace time and so cannot shrink to the anchors a small canvas
has. Every graph answers with :attr:`~lucid_yolo.decode.topk_e2e.TopKDecoder.k` rows; on
a canvas yielding fewer anchors than ``k`` the surplus are padding rows, whose anchor
index is :data:`~lucid_yolo.decode.common.PAD_ANCHOR_INDEX`:

- :class:`DetectExportGraph` and :class:`OrientedExportGraph` emit them as all-zero
  rows — score 0, class 0, zero box — the oriented one through
  :class:`~lucid_yolo.decode.topk_e2e.RotatedTopKDecoder`, which pads to ``k`` exactly
  as the axis-aligned decoder does.
- :class:`KeypointExportGraph` gathers each row's point set through
  :func:`~lucid_yolo.decode.common.gather_by_anchor`, which clamps a padding index to a
  valid position and zeroes the row, so a padding row carries the origin and score 0
  rather than the last anchor's pose.
- :class:`SegmentExportGraph` refuses such a canvas in ``__init__``, naming the anchor
  count and the cap. Its coefficient gather is that same clamp-and-zero gather now, so a
  padding row would no longer fail — it would carry a zero coefficient row and therefore
  an empty mask — but the refusal is kept rather than lifted: lifting it is a behaviour
  change of its own, and no deployment canvas comes near the bound (640 px yields 8400
  anchors; the bound bites at 96 px a side and below). :func:`~lucid_yolo.predict.predict_segmentation`
  has no such limit, because a single-image caller can drop padding rows with a boolean
  filter, which a fixed-shape graph cannot.

This is also why :mod:`tests.models.test_onnx_export` exports at 128 px: 336 anchors sit
above the 300 cap, so every compared row is a real detection, the regime every
production canvas is in.

**Known limits**, all deliberate and each a one-line change where it is a choice:

- ONNX opset :data:`ONNX_OPSET` (17) by default; :func:`export_graph` takes another.
- The legacy TorchScript exporter, ``torch.onnx.export(..., dynamo=False)``, which torch
  marks deprecated. It needs the ``onnx`` package installed to write the file, so a plain
  install cannot export; ``pip install "lucid-yolo[export]"`` adds it.
- Static ``(B, 3, H, W)`` input: the canvas is baked into the anchor buffers and no
  ``dynamic_axes`` is passed, so there is no dynamic batch either.
- The E2E (suppression-free) path only. No suppression graph is exported, and no other
  format, precision or quantization is produced.
- The segmentation graph refuses canvases under ``k`` anchors; the keypoint graph
  clamps; the oriented and detection graphs pad to ``k``.

**From a checkpoint.** A Lightning checkpoint loads into a
:class:`~lucid_yolo.ptl.module.DetectionLitModule`, which has no ``deploy()`` of its
own. :func:`graph_from_module` is the bridge: it rebuilds the composite model the
module's task names at the module's own multipliers, moves the module's weights into it
under a strict key check, and wraps that model's ``deploy()`` view in the task's graph.
``lucid-export`` (:mod:`lucid_yolo.cli.export`) is the same route as a command.

Provenance: R1 sec. 3.2.1, R3 sec. 4. Assumptions: A9, A23, A37, A45.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from lucid_yolo.assign.grid import anchor_grid
from lucid_yolo.decode.common import BOX_CORNERS
from lucid_yolo.decode.per_task import decode_with_coefficients, decode_with_keypoints
from lucid_yolo.decode.topk_e2e import RotatedTopKDecoder, TopKDecoder
from lucid_yolo.eval.segment_decode import _decode_instance_masks_unchunked
from lucid_yolo.models.build import Detector, KeypointDetector, OrientedDetector, Segmenter
from lucid_yolo.models.registry import ScaleSpec
from lucid_yolo.validate import require_one_of

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from torch import Tensor

    from lucid_yolo.ptl.module import DetectionLitModule

__all__ = [
    "ONNX_OPSET",
    "DetectExportGraph",
    "E2EExportGraph",
    "KeypointExportGraph",
    "OrientedExportGraph",
    "SegmentExportGraph",
    "export_graph",
    "graph_from_module",
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

    The shared base of :class:`DetectExportGraph`, :class:`SegmentExportGraph`,
    :class:`OrientedExportGraph` and :class:`KeypointExportGraph`: it bakes the anchor grid for
    one ``canvas`` into buffers and holds the
    :class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` whose ``k`` fixes every task's output
    length — the axis-aligned tasks decode with it directly, the oriented task with a
    :class:`~lucid_yolo.decode.topk_e2e.RotatedTopKDecoder` built at the same ``k`` — so a
    subclass need only compose its own task's dense outputs into a decoded tuple.
    ``forward`` is left to subclasses — that is where the four tasks differ, per
    the module docstring's compositions — so this class alone is not directly
    exportable.

    Args:
        deployed: The ``deploy()`` view whose dense outputs the decode consumes,
            as returned by :meth:`~lucid_yolo.models.build.Detector.deploy`,
            :meth:`~lucid_yolo.models.build.Segmenter.deploy`,
            :meth:`~lucid_yolo.models.build.OrientedDetector.deploy` or
            :meth:`~lucid_yolo.models.build.KeypointDetector.deploy`, matched to
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
        ```pycon
        >>> import torch
        >>> from lucid_yolo.models.build import Detector
        >>> deployed = Detector("n", num_classes=4).eval().deploy().eval()
        >>> graph = E2EExportGraph(deployed, canvas=(128, 128))
        >>> graph.anchor_points.shape, graph.strides.shape  # 16^2 + 8^2 + 4^2 = 336
        (torch.Size([336, 2]), torch.Size([336]))

        ```
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
        ```pycon
        >>> import torch
        >>> from lucid_yolo.models.build import Detector
        >>> deployed = Detector("n", num_classes=4).eval().deploy().eval()
        >>> graph = DetectExportGraph(deployed, canvas=(128, 128)).eval()
        >>> with torch.no_grad():
        ...     detections = graph(torch.zeros(1, 3, 128, 128))
        >>> detections.shape  # (B, k, 6): the A9 tuple, k = 300 (A9)
        torch.Size([1, 300, 6])

        ```
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
    mask assembly, resize, crop and threshold operations. Export retains the
    original single-resize graph while eager decode chunks its full-canvas
    temporaries. One shape divergence remains: every one of the ``k``
    rows is gathered, padding rows included, because the output shape is fixed.

    That divergence is why this graph's supported canvases are narrower than its
    base class's: a canvas yielding fewer anchors than ``k`` is refused in
    ``__init__``, where it can still be reported as a canvas choice. The coefficient
    gather is the shared clamp-and-zero one, so such a canvas would no longer fail at
    trace time; the module docstring states why the refusal is kept anyway.

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
        ```pycon
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

        ```
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
        detections, _, gathered = decode_with_coefficients(
            self.decoder, cls, box, coeff, self.anchor_points, self.strides
        )
        masks: Tensor = _decode_instance_masks_unchunked(
            prototypes, gathered, detections[..., :BOX_CORNERS], image_size=self.canvas
        )
        return detections, masks


class OrientedExportGraph(E2EExportGraph):
    """Oriented E2E export graph: dense one-to-one outputs to the A45 tuple.

    Reproduces :func:`~lucid_yolo.predict.predict_oriented`'s ``"e2e"``
    composition — one :class:`~lucid_yolo.decode.topk_e2e.RotatedTopKDecoder` call
    over the deployed model's one-to-one class logits, ``(box, angle)`` pair and the
    baked anchor grid, at the same ``k`` as the other graphs' decoder.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.models.build import OrientedDetector
        >>> deployed = OrientedDetector("n", num_classes=4).eval().deploy().eval()
        >>> graph = OrientedExportGraph(deployed, canvas=(128, 128)).eval()
        >>> with torch.no_grad():
        ...     detections = graph(torch.zeros(1, 3, 128, 128))
        >>> detections.shape  # (B, k, 7): the A45 tuple, k = 300 (A9)
        torch.Size([1, 300, 7])

        ```
    """

    def forward(self, image: Tensor) -> Tensor:
        """Decode an image batch into fixed-size A45 rotated detections.

        Args:
            image: Input image batch of shape ``(B, 3, H, W)`` matching the
                ``canvas`` this graph was built for.

        Returns:
            Detections of shape ``(B, k, 7)`` whose last axis is the A45 tuple
            ``[cx, cy, w, h, theta, score, class]`` (see
            :meth:`~lucid_yolo.decode.topk_e2e.RotatedTopKDecoder.forward`).
        """
        cls, box, angle = self.deployed(image)
        detections: Tensor = RotatedTopKDecoder(k=self.decoder.k)(cls, box, angle, self.anchor_points, self.strides)
        return detections


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

    Padding rows are handled by :func:`~lucid_yolo.decode.common.gather_by_anchor`,
    which clamps a padding index to a valid position and then zeroes that row, so a
    padding row yields the origin rather than the last anchor's pose. A fixed-shape
    graph cannot drop rows, and this task's gather is defined on them, so this graph
    accepts every canvas its base class does; :class:`SegmentExportGraph` uses the
    same gather but refuses canvases that would produce padding rows.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.models.build import KeypointDetector
        >>> deployed = KeypointDetector("n", num_classes=4, num_keypoints=3).eval().deploy().eval()
        >>> graph = KeypointExportGraph(deployed, canvas=(128, 128)).eval()
        >>> with torch.no_grad():
        ...     detections, keypoints = graph(torch.zeros(1, 3, 128, 128))
        >>> detections.shape, keypoints.shape  # (B, k, 6) beside (B, k, K, 2)
        (torch.Size([1, 300, 6]), torch.Size([1, 300, 3, 2]))

        ```
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
        detections, _, points = decode_with_keypoints(
            self.decoder, cls, box, raw_points, self.anchor_points, self.strides
        )
        return detections, points


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

    Raises:
        torch.onnx.errors.OnnxExporterError: If the ``onnx`` package is not installed.
            torch's TorchScript exporter imports it to finalise the file, so exporting
            needs it even though nothing here imports it; ``pip install
            "lucid-yolo[export]"`` adds it.

    Examples:
        ```pycon
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

        ```
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


#: The export graph class per task, for every task
#: :class:`~lucid_yolo.ptl.module.DetectionLitModule` trains, in the order a refusal
#: lists them.
_GRAPH_CLASSES: dict[str, type[E2EExportGraph]] = {
    "detect": DetectExportGraph,
    "segment": SegmentExportGraph,
    "obb": OrientedExportGraph,
    "keypoints": KeypointExportGraph,
}

#: State-dict prefixes a :class:`~lucid_yolo.ptl.module.DetectionLitModule` carries
#: that no composite model holds, dropped before the strict key comparison. The one
#: entry is ``rle_loss``: under ``keypoint_loss="rle"`` it holds R14's normalizing flow,
#: a *trained* objective, so its parameters are saved with the checkpoint — and read by
#: nothing at inference. Every other key must match exactly; an unlisted extra is a
#: mismatch, not something to skip, which is what makes the check strict.
_TRAINING_ONLY_PREFIXES: tuple[str, ...] = ("rle_loss.",)

#: How many offending keys a mismatch message names before summarising the rest.
_KEYS_NAMED = 8


def _key_preview(keys: list[str]) -> str:
    """Name up to :data:`_KEYS_NAMED` keys, then count the remainder.

    Args:
        keys: The keys to name, already sorted.

    Returns:
        A printable list, with a ``(+N more)`` suffix when it was cut.

    Examples:
        ```pycon
        >>> _key_preview(["a", "b"])
        "['a', 'b']"
        >>> _key_preview([f"k{i}" for i in range(10)])
        "['k0', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7'] (+2 more)"

        ```
    """
    shown = repr(keys[:_KEYS_NAMED])
    return shown if len(keys) <= _KEYS_NAMED else f"{shown} (+{len(keys) - _KEYS_NAMED} more)"


def _strict_state(
    expected: Collection[str],
    supplied: Mapping[str, Tensor],
    training_only: tuple[str, ...],
    target: str,
) -> dict[str, Tensor]:
    """Check a state dict against a model's keys exactly, after dropping training-only entries.

    The replacement for ``load_state_dict(..., strict=False)`` plus a missing-key
    assertion, which is what the export bridge used to be: that form reports a key the
    model lacks and silently ignores a key the model never asked for, so a checkpoint
    carrying a branch the composite does not build would export without it. Here both
    directions refuse, naming the keys.

    Args:
        expected: Every key the target model's state dict holds.
        supplied: The source state dict.
        training_only: Key prefixes dropped from ``supplied`` before comparing.
        target: What is being loaded into, for the message.

    Returns:
        ``supplied`` without its training-only entries, ready for
        ``load_state_dict(..., strict=True)``.

    Raises:
        ValueError: If any expected key is missing from ``supplied``, or ``supplied``
            holds any key outside ``expected`` and ``training_only``; the message names
            both lists.

    Examples:
        ```pycon
        >>> import torch
        >>> state = {"w": torch.zeros(1), "rle_loss.flow": torch.zeros(1)}
        >>> sorted(_strict_state({"w"}, state, ("rle_loss.",), "detect model"))
        ['w']
        >>> _strict_state({"w", "b"}, {"w": torch.zeros(1), "extra": torch.zeros(1)}, (), "detect model")
        Traceback (most recent call last):
            ...
        ValueError: the module's weights do not match the detect model: missing ['b']; unexpected ['extra']

        ```
    """
    kept = {key: value for key, value in supplied.items() if not key.startswith(training_only)}
    wanted = set(expected)
    missing = sorted(wanted - kept.keys())
    unexpected = sorted(kept.keys() - wanted)
    if missing or unexpected:
        raise ValueError(
            f"the module's weights do not match the {target}: "
            f"missing {_key_preview(missing)}; unexpected {_key_preview(unexpected)}"
        )
    return kept


def _composite_model(
    task: str, scaling: ScaleSpec, num_classes: int, num_keypoints: int | None
) -> Detector | Segmenter | OrientedDetector | KeypointDetector:
    """Build the composite model a task names, at raw multipliers, with fresh weights.

    Args:
        task: A key of :data:`_GRAPH_CLASSES`.
        scaling: The module's own ``depth``/``width``/``max_channels``.
        num_classes: The module's class count.
        num_keypoints: The module's point count; required for ``"keypoints"``.

    Returns:
        The untrained composite, whose state-dict keys are the ones the module's weights
        must supply.

    Raises:
        ValueError: If ``task`` is ``"keypoints"`` and ``num_keypoints`` is ``None``.

    Examples:
        ```pycon
        >>> model = _composite_model("obb", ScaleSpec(0.34, 0.25, 64), num_classes=2, num_keypoints=None)
        >>> type(model).__name__, model.head.predict_angle
        ('OrientedDetector', True)

        ```
    """
    if task == "keypoints":
        if num_keypoints is None:
            raise ValueError("a keypoints module must name num_keypoints; its hyperparameters carry none")
        return KeypointDetector(scaling, num_classes, num_keypoints=num_keypoints)
    if task == "segment":
        return Segmenter(scaling, num_classes)
    if task == "obb":
        return OrientedDetector(scaling, num_classes)
    return Detector(scaling, num_classes)


def graph_from_module(
    module: DetectionLitModule, canvas: tuple[int, int], *, device: torch.device | None = None
) -> E2EExportGraph:
    """Build a checkpoint-loaded module's export graph, its weights checked key for key.

    A Lightning checkpoint loads into a :class:`~lucid_yolo.ptl.module.DetectionLitModule`
    (:func:`~lucid_yolo.eval.checkpoint.load_eval_module`), and the export graphs wrap a
    composite model's ``deploy()`` view, which the Lightning module does not have. This
    is the bridge between the two. It reads the task from the module, rebuilds that
    task's composite — :class:`~lucid_yolo.models.build.Detector`,
    :class:`~lucid_yolo.models.build.Segmenter`,
    :class:`~lucid_yolo.models.build.OrientedDetector` or
    :class:`~lucid_yolo.models.build.KeypointDetector` — at the module's **own**
    multipliers, class count and point count, copies the module's weights into it, and
    wraps its ``deploy()`` view in the task's graph class.

    The rebuild is at raw multipliers rather than a variant name, so a module built
    outside the registry's five rows exports as readily as one the training CLI
    configured. The copy is checked strictly in both directions: the module must supply
    every key the composite holds and nothing else, apart from the training-only loss
    state in :data:`_TRAINING_ONLY_PREFIXES`. That works because the module keeps its
    stages as flat ``backbone``/``neck``/``head`` attributes (plus the three segmentation
    stages), the same names the composites use, so the two state dicts share their keys.

    The graph owns a copy of the weights: ``module`` is read and not modified, and stays
    on whatever device it was on. Whichever weights ``module`` holds are what is
    exported, so a module :func:`~lucid_yolo.eval.checkpoint.load_eval_module` returned
    with ``use_ema=True`` exports its EMA shadow.

    Args:
        module: The module to export, as
            :func:`~lucid_yolo.eval.checkpoint.load_eval_module` returns it.
        canvas: Input canvas ``(height, width)`` in pixels, each divisible by every head
            stride. Baked into the graph: the result answers only for this size.
        device: Device the graph is built on, as :class:`E2EExportGraph` describes.
            Defaults to CPU.

    Returns:
        The task's eval-mode graph, ready for :func:`export_graph`.

    Raises:
        ValueError: If the module's task is not one of ``detect``, ``segment``,
            ``obb`` and ``keypoints`` (the message lists them), if its weights miss a
            key the composite holds or carry one it does not (the message names them),
            if a weight's shape disagrees with the composite's, or if ``canvas`` is
            refused by the task's graph class.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.ptl.module import DetectionLitModule
        >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=2).eval()
        >>> graph = graph_from_module(module, canvas=(64, 64))
        >>> type(graph).__name__, graph.training
        ('DetectExportGraph', False)
        >>> with torch.no_grad():
        ...     detections = graph(torch.zeros(1, 3, 64, 64))
        >>> detections.shape
        torch.Size([1, 300, 6])

        ```
    """
    task = str(module.task)
    require_one_of("task", task, tuple(_GRAPH_CLASSES))
    hparams = module.hparams
    scaling = ScaleSpec(
        depth=float(hparams["depth"]), width=float(hparams["width"]), max_channels=int(hparams["max_channels"])
    )
    num_keypoints = hparams.get("num_keypoints")
    model = _composite_model(
        task, scaling, int(hparams["num_classes"]), None if num_keypoints is None else int(num_keypoints)
    )
    target = f"{task} model ({type(model).__name__})"
    state = _strict_state(model.state_dict().keys(), module.state_dict(), _TRAINING_ONLY_PREFIXES, target)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:  # every key matched, so what is left is a shape disagreement
        raise ValueError(f"the module's weights do not fit the {target}: {error}") from error
    graph = _GRAPH_CLASSES[task](model.eval().deploy().eval(), canvas, device)
    return graph.eval()
