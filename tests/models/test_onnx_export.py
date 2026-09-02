# SPDX-License-Identifier: Apache-2.0
"""ONNX export gate for the four end-to-end deploy paths (WP-066, WP-151).

Two claims are under test, and the second is the one that needs the machinery. The
keypoint path joined the other three at WP-151: 0.5.0 released a task whose model
could not leave PyTorch, so it was the one released task with no graph to gate.

**The graph carries no suppression.** R1's architectural claim for the one-to-one
branch is that it deploys *without* NMS, and the exported graph is where that claim
is falsifiable: a ``NonMaxSuppression`` node in the emitted model would mean the
NMS-free path is not what ships. ``test_e2e_graph_ops`` asserts that node's absence —
and, because an absence proves little on its own, asserts positively that the ops which
*must* be there are: ``TopK`` (the score ranking that replaces suppression),
``Conv``, ``Sigmoid``, and for segmentation the ``Einsum`` of Eq. 7. It also pins the
static output shape :class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` documents.

**The graph computes what the checkpoint computes.** An op-type assertion passes
just as happily on a silently wrong export — a mis-baked anchor constant, a decode
traced down the wrong branch, a transposed tensor.
``test_e2e_export_matches_checkpoint`` therefore runs the exported model under
onnxruntime and compares it against the checkpoint-loaded
:class:`~lucid_yolo.ptl.module.DetectionLitModule` decoded through the same E2E path
:mod:`lucid_yolo.predict` uses. The comparison is on the decoded A9/A45 detection
tuple, not on raw head logits: raw-logit parity would hold even if the decode were
exported wrong, which is the same vacuity the first test guards against, one layer down.

The route is the point. The fixture writes a Lightning checkpoint, loads it back
through :func:`~lucid_yolo.eval.checkpoint.load_eval_module` — the path an operator's
``lucid-eval`` and ``lucid-predict`` runs both take — and exports the deployed view of
*that*. Weights are random: this project ships no trained weights (D14) and the gate is
offline, so random weights are the honest substitute. Nothing here measures accuracy;
it measures that two backends agree along a real route.

**Why the fixture is doctored, and why it has to be.** A freshly built model is
useless for this comparison, and silently so. Its BatchNorms carry
``running_var=1``/``running_mean=0``, so in eval mode they rescale nothing and the
signal attenuates through depth until the classifier's contribution — order 1e-7 —
falls below the float32 resolution of its own ``-log((1-pi)/pi)`` prior bias at 4.6.
Every one of the 336 anchors then produces the *bit-identical* logit, ``topk`` becomes a
300-of-336 tie, and the two backends break that tie differently: the test would compare
two arbitrary orderings and report boxes 80 px apart. So :func:`_discriminative`
calibrates the BN running statistics with a few forward passes, rescales the
classification stems to a known logit spread, and biases the box stems positive. Without
the last of those, every decoded box is inverted or empty and every instance mask comes
out all-``False`` — a mask comparison that passes by comparing nothing.

Design call — **anchor points and strides are baked into the graph as buffers**, not
taken as extra inputs. They are arguments to
:meth:`~lucid_yolo.decode.topk_e2e.TopKDecoder.forward` rather than module state, so an
exporter must either receive them or constant-fold them; baking makes the graph's single
input the image, which is what a deployment consumer expects, at the cost of the graph
being **specific to one input size**. That is normal for ONNX and is stated here rather
than discovered later. ``test_e2e_graph_ops`` pins it by asserting the graph has exactly
one input.

The canvas is 128 px because it puts the anchor count at ``16^2 + 8^2 + 4^2 = 336``,
above the 300 detection cap, so the decoders emit exactly the documented ``(1, 300, 6)``
and ``(1, 300, 7)`` with no padding rows — the same regime as the 640 px production
input, where 8400 anchors sit above the same cap. Every compared row is a real
detection.

Provenance: R1 sec. 3.2.1, R3 sec. 4. Assumptions: A9, A23, A37, A45.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import Tensor, nn

from lucid_yolo.assign.grid import anchor_grid
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.checkpoint import load_eval_module
from lucid_yolo.eval.coco_eval import gather_keypoints
from lucid_yolo.eval.segment_decode import decode_instance_masks
from lucid_yolo.export import (
    DetectExportGraph,
    E2EExportGraph,
    KeypointExportGraph,
    OrientedExportGraph,
    SegmentExportGraph,
)
from lucid_yolo.models.build import Detector, KeypointDetector, OrientedDetector, Segmenter
from lucid_yolo.models.heads.keypoint import decode_keypoints
from lucid_yolo.models.heads.obb import decode_rboxes, o2o_rotated_topk
from lucid_yolo.models.registry import scale_spec
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

#: Square export canvas. Chosen so the anchor count (336) exceeds the detection cap.
_CANVAS = (128, 128)

#: Detection cap and fixed output length of both E2E decoders (A9).
_DET_CAP = 300

_NUM_CLASSES = 4
_VARIANT = "n"
_TASKS = ("detect", "segment", "obb", "keypoints")

#: Point count K the keypoint task is exported at. Deliberately not COCO's 17: K is a
#: constructor argument, and a gate that only ever ran at 17 would not catch a shape
#: baked to that number (A64).
_NUM_KEYPOINTS = 3

#: Deployable model class per task; each exposes the ``deploy()`` view under test.
_MODELS: dict[str, type[Detector] | type[Segmenter] | type[OrientedDetector] | type[KeypointDetector]] = {
    "detect": Detector,
    "segment": Segmenter,
    "obb": OrientedDetector,
    "keypoints": KeypointDetector,
}

#: Standard deviation the classification logits are rescaled to, so the top-k ranking is
#: decided by the scores rather than by the tie-break (see the module docstring).
_TARGET_LOGIT_STD = 2.0

#: Positive bias written into the box stems, so decoded boxes have interiors and the
#: instance masks are not uniformly empty.
_BOX_BIAS = 2.0

#: Decimals the detection rows are rounded to before they are ordered for comparison.
#: Two backends differ by float32 noise (measured at or below 5e-05), and rows tied on
#: score are ordered by whichever backend's noise came out larger; rounding well above
#: that noise makes the ordering — and therefore the pairing — identical on both sides.
_PAIR_QUANTUM = 3

#: Tolerances for the parity comparison, chosen after measuring: boxes came out within
#: 4.6e-05 and scores within 1.2e-07 across all three heads. Class ids are compared
#: exactly — they are integers, and a tolerance on them would hide a real disagreement.
_BOX_ATOL = 1e-3
_SCORE_ATOL = 1e-5


@dataclass(frozen=True)
class _Exported:
    """One task's exported graph beside the eager result it must reproduce.

    Attributes:
        task: One of ``"detect"``, ``"segment"``, ``"obb"``, ``"keypoints"``.
        path: The written ``.onnx`` file.
        image: The input the graph was exported and compared on.
        reference: Eager outputs of the checkpoint-loaded module, decoded through the
            E2E path — the detection tuple first, then instance masks for ``segment``
            or gathered point sets for ``keypoints``.
    """

    task: str
    path: Path
    image: Tensor
    reference: list[Tensor]


#: Deployable graph class per task. Promoted to :mod:`lucid_yolo.export` (WP-112) so
#: this fixture and :mod:`lucid_yolo.predict`'s single-image "e2e" path are provably
#: the same composition rather than two that merely resemble each other.
_GRAPHS: dict[str, type[E2EExportGraph]] = {
    "detect": DetectExportGraph,
    "segment": SegmentExportGraph,
    "obb": OrientedExportGraph,
    "keypoints": KeypointExportGraph,
}


def _discriminative(module: DetectionLitModule) -> DetectionLitModule:
    """Make an untrained module produce a decidable ranking and non-empty boxes.

    Three edits, each answering a specific degeneracy the module docstring describes:
    the BatchNorm running statistics are calibrated by forward passes in train mode (no
    gradients, no optimizer — this is not training); the classification stems are
    rescaled to :data:`_TARGET_LOGIT_STD` after their prior bias is cleared, so scores
    separate instead of collapsing onto one float32 value; and the box stems are biased
    positive so decoded boxes have interiors.

    Args:
        module: A freshly constructed module, modified in place.

    Returns:
        The same module, in eval mode.

    Examples:
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> spec = scale_spec(_VARIANT)
        >>> module = DetectionLitModule(
        ...     depth=spec.depth, width=spec.width, max_channels=spec.max_channels,
        ...     num_classes=_NUM_CLASSES, task="detect",
        ... )
        >>> tuned = _discriminative(module)
        >>> tuned.training
        False
        >>> with torch.no_grad():
        ...     probe = torch.rand(1, 3, *_CANVAS, generator=torch.Generator().manual_seed(5))
        ...     std = float(tuned(probe).o2o_cls.std())
        >>> round(std, 1)
        2.0
    """
    generator = torch.Generator().manual_seed(11)
    module.train()
    with torch.no_grad():
        for _ in range(3):
            module(torch.rand(2, 3, *_CANVAS, generator=generator))
    module.eval()
    with torch.no_grad():
        for stem in module.head.o2o.cls_stems:
            stem[-1].bias.zero_()
        probe = torch.rand(1, 3, *_CANVAS, generator=torch.Generator().manual_seed(5))
        factor = _TARGET_LOGIT_STD / float(module(probe).o2o_cls.std())
        for stem in module.head.o2o.cls_stems:
            stem[-1].weight.mul_(factor)
        for stem in module.head.o2o.box_stems:
            stem[-1].bias.fill_(_BOX_BIAS)
    return module


def _checkpoint_module(task: str, tmp: Path) -> DetectionLitModule:
    """Write a Lightning checkpoint for ``task`` and load it back the way an operator does.

    Args:
        task: One of ``"detect"``, ``"segment"``, ``"obb"``, ``"keypoints"``.
        tmp: Directory the ``.ckpt`` is written into.

    Returns:
        The eval-mode module :func:`~lucid_yolo.eval.checkpoint.load_eval_module` returns.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     tmp_path = Path(tmp)
        ...     module = _checkpoint_module("detect", tmp_path)
        ...     (tmp_path / "detect.ckpt").is_file()
        True
        >>> module.training
        False
    """
    spec = scale_spec(_VARIANT)
    built = _discriminative(
        DetectionLitModule(
            depth=spec.depth,
            width=spec.width,
            max_channels=spec.max_channels,
            num_classes=_NUM_CLASSES,
            task=task,
            num_keypoints=_NUM_KEYPOINTS if task == "keypoints" else None,
        )
    )
    path = tmp / f"{task}.ckpt"
    torch.save(
        {
            "state_dict": built.state_dict(),
            "hyper_parameters": dict(built.hparams),
            "pytorch-lightning_version": "2.4.0",
            "epoch": 0,
            "global_step": 0,
            "loops": {},
        },
        path,
    )
    loaded, _ = load_eval_module(path, use_ema=False)
    return loaded


def _deployed(module: DetectionLitModule, task: str) -> nn.Module:
    """Move the checkpoint's weights into the task's model and return its deploy view.

    The Lightning module keeps ``backbone``/``neck``/``head`` as flat attributes
    precisely so its state-dict keys match the composite models' (WP-087), which is what
    lets the shipped ``deploy()`` be exercised here rather than re-implemented. The
    missing-key assertion pins that: every deployed parameter came from the checkpoint.

    Args:
        module: The checkpoint-loaded module.
        task: One of ``"detect"``, ``"segment"``, ``"obb"``, ``"keypoints"``.

    Returns:
        The eval-mode ``deploy()`` view, sharing the checkpoint's weights.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     module = _checkpoint_module("detect", Path(tmp))
        >>> deployed = _deployed(module, "detect")
        >>> deployed.training
        False
    """
    model = (
        KeypointDetector(_VARIANT, num_classes=_NUM_CLASSES, num_keypoints=_NUM_KEYPOINTS)
        if task == "keypoints"
        else _MODELS[task](_VARIANT, num_classes=_NUM_CLASSES)
    )
    missing, _ = model.load_state_dict(module.state_dict(), strict=False)
    assert not missing, f"deployed {task} model has parameters the checkpoint did not supply: {missing}"
    return model.eval().deploy().eval()


def _eager_reference(module: DetectionLitModule, task: str, image: Tensor) -> list[Tensor]:
    """Decode the checkpoint-loaded module through the same E2E path :mod:`lucid_yolo.predict` uses.

    Args:
        module: The checkpoint-loaded module.
        task: One of ``"detect"``, ``"segment"``, ``"obb"``, ``"keypoints"``.
        image: Input batch of shape ``(1, 3, 128, 128)``.

    Returns:
        The decoded detection tuple, followed by instance masks for ``"segment"`` or
        gathered point sets for ``"keypoints"``.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     module = _checkpoint_module("detect", Path(tmp))
        >>> image = torch.rand(1, 3, *_CANVAS, generator=torch.Generator().manual_seed(7))
        >>> reference = _eager_reference(module, "detect", image)
        >>> len(reference)
        1
        >>> reference[0].shape
        torch.Size([1, 300, 6])
    """
    points, strides = anchor_grid(_CANVAS, torch.device("cpu"))
    with torch.no_grad():
        if task == "segment":
            segmented = module.forward_segmentation(image)
            head = segmented.detect
            detections, anchor_index = TopKDecoder().decode_with_indices(head.o2o_cls, head.o2o_box, points, strides)
            coeff = head.o2o_coeff
            assert coeff is not None  # a segmentation module always builds the coefficient stems
            gathered = coeff.gather(1, anchor_index.unsqueeze(-1).expand(-1, -1, coeff.shape[-1]))
            masks = decode_instance_masks(segmented.prototypes, gathered, detections[..., :4], image_size=_CANVAS)
            return [detections, masks]
        head_out = module(image)
        if task == "keypoints":
            raw_points = head_out.o2o_keypoints
            assert raw_points is not None  # a keypoints module always builds the point stems
            detections, anchor_index = TopKDecoder().decode_with_indices(
                head_out.o2o_cls, head_out.o2o_box, points, strides
            )
            dense_points = decode_keypoints(raw_points, points, strides)
            return [detections, gather_keypoints(dense_points, anchor_index)]
        if task == "obb":
            angle = head_out.o2o_angle
            assert angle is not None  # an oriented module always builds the angle stems
            rboxes = decode_rboxes(head_out.o2o_box, angle, points, strides)
            return [o2o_rotated_topk(head_out.o2o_cls, rboxes)]
        return [TopKDecoder()(head_out.o2o_cls, head_out.o2o_box, points, strides)]


def _op_types(model: onnx.ModelProto) -> set[str]:
    """Collect every op type in the graph, descending into subgraph attributes.

    Subgraphs are walked because a suppression node hidden inside an ``If`` or ``Loop``
    body would evade a top-level scan, and an absence claim that a nested node can evade
    is not an absence claim.

    Args:
        model: The loaded ONNX model.

    Returns:
        The set of op type names appearing anywhere in the model.

    Examples:
        >>> from onnx import helper, TensorProto
        >>> relu = helper.make_node("Relu", ["x"], ["y"])
        >>> sigmoid = helper.make_node("Sigmoid", ["y"], ["z"])
        >>> graph = helper.make_graph(
        ...     [relu, sigmoid], "g",
        ...     [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        ...     [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])],
        ... )
        >>> sorted(_op_types(helper.make_model(graph)))
        ['Relu', 'Sigmoid']
    """
    found: set[str] = set()

    def walk(graph: onnx.GraphProto) -> None:
        for node in graph.node:
            found.add(node.op_type)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    walk(attribute.g)
                for sub in attribute.graphs:
                    walk(sub)

    walk(model.graph)
    return found


def _static_shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    """Return a graph value's fully static shape.

    Args:
        value: A graph input or output value info.

    Returns:
        The dimension sizes.

    Raises:
        AssertionError: If any dimension is symbolic rather than a fixed size.

    Examples:
        >>> from onnx import helper, TensorProto
        >>> value = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 128, 128])
        >>> _static_shape(value)
        (1, 3, 128, 128)
        >>> symbolic = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])
        >>> try:
        ...     _static_shape(symbolic)
        ... except AssertionError as exc:
        ...     str(exc).startswith("y has a symbolic dimension: 'batch'")
        True
    """
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        assert dim.HasField("dim_value"), f"{value.name} has a symbolic dimension: {dim.dim_param!r}"
        dims.append(dim.dim_value)
    return tuple(dims)


def _paired(rows: np.ndarray) -> np.ndarray:
    """Order detection rows so two backends pair them identically.

    Rows tied on score may be emitted in either order by either backend without either
    being wrong, so they are put into a total order over all columns. The keys are
    rounded first: ordering on raw float32 keys re-introduces the same problem one level
    down, because clustered coordinates differing by less than the backends' own noise
    would sort differently on each side.

    Args:
        rows: One image's detections, shape ``(N, 6)`` or ``(N, 7)``.

    Returns:
        The permutation that puts ``rows`` in canonical order.

    Examples:
        >>> rows = np.array([[1.0, 2.0, 3.0, 4.0, 0.9, 1.0], [1.0, 2.0, 3.0, 4.0, 0.5, 0.0]])
        >>> _paired(rows).tolist()
        [1, 0]
    """
    keys = np.round(rows, _PAIR_QUANTUM)
    return np.lexsort(tuple(keys[:, column] for column in reversed(range(keys.shape[1]))))


@pytest.fixture(params=_TASKS, scope="module")
def exported(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> _Exported:
    """Export one task's E2E graph from a checkpoint and keep its eager reference."""
    task = str(request.param)
    tmp = tmp_path_factory.mktemp(f"onnx_{task}")
    torch.manual_seed(0)

    module = _checkpoint_module(task, tmp)
    graph = _GRAPHS[task](_deployed(module, task), canvas=_CANVAS).eval()
    image = torch.rand(1, 3, *_CANVAS, generator=torch.Generator().manual_seed(7))
    reference = _eager_reference(module, task, image)

    path = tmp / f"{task}.onnx"
    with torch.no_grad():
        # The legacy TorchScript exporter: `dynamo=True` (the torch 2.13 default) needs
        # `onnxscript`, and the legacy path emits a fully static graph here with no
        # extra dependency. It warns that it is deprecated; when it is removed, this is
        # the line that changes, and `onnxscript` joins the dev group.
        torch.onnx.export(graph, (image,), str(path), dynamo=False)
    return _Exported(task=task, path=path, image=image, reference=reference)


def test_e2e_graph_ops(exported: _Exported) -> None:
    """The exported E2E graph carries no suppression, does carry the ops it must, and is static.

    The absence of ``NonMaxSuppression`` is R1's deployment claim for the one-to-one
    branch. It is asserted beside the presence of ``TopK`` — the ranking that stands in
    for suppression — because an absence in a graph that never got as far as decoding
    would be free.
    """
    model = onnx.load(str(exported.path))
    onnx.checker.check_model(model, full_check=True)
    ops = _op_types(model)

    assert "NonMaxSuppression" not in ops, (
        f"the {exported.task} E2E graph contains a suppression node; the one-to-one "
        f"branch is supposed to deploy without one (R1 sec. 3.2.1, R3 sec. 4)"
    )
    assert {"Conv", "Sigmoid", "TopK"} <= ops, f"{exported.task} graph is missing core ops: {sorted(ops)}"
    if exported.task == "segment":
        assert "Einsum" in ops, "the segmentation graph does not carry the Eq. 7 mask assembly"

    # One input: the anchor grid and strides are baked in as constants, so the graph is
    # specific to a 128 px input and takes only the image (see the module docstring).
    assert len(model.graph.input) == 1, f"expected the image alone as input, got {len(model.graph.input)}"
    assert _static_shape(model.graph.input[0]) == (1, 3, *_CANVAS)

    columns = 7 if exported.task == "obb" else 6
    assert _static_shape(model.graph.output[0]) == (1, _DET_CAP, columns)
    if exported.task == "segment":
        assert _static_shape(model.graph.output[1]) == (1, _DET_CAP, *_CANVAS)
    if exported.task == "keypoints":
        assert _static_shape(model.graph.output[1]) == (1, _DET_CAP, _NUM_KEYPOINTS, 2)


def test_e2e_export_matches_checkpoint(exported: _Exported) -> None:
    """The exported graph answers what the checkpoint-loaded module answers.

    Run under onnxruntime and compared against the eager decode of the module
    :func:`~lucid_yolo.eval.checkpoint.load_eval_module` returned, on the decoded
    detection tuple rather than on raw logits. Rows are paired canonically first: score
    ties are genuine and the two backends may order them differently without either
    being wrong.
    """
    session = ort.InferenceSession(str(exported.path), providers=["CPUExecutionProvider"])
    produced = session.run(None, {session.get_inputs()[0].name: exported.image.numpy()})

    reference = exported.reference[0].numpy()[0]
    candidate = produced[0][0]
    box_columns = 5 if exported.task == "obb" else 4
    score_column = box_columns

    # The fixture must be discriminative, or the comparison below proves nothing: if
    # every score were tied, any pairing would match and the test would pass vacuously.
    scores = reference[:, score_column]
    assert np.ptp(scores) > 0.05, f"{exported.task} scores span only {np.ptp(scores):.3g}; ranking is degenerate"
    widths = reference[:, 2] - reference[:, 0] if box_columns == 4 else reference[:, 2]
    assert (widths > 0).all(), f"{exported.task} produced boxes without interiors; masks would be empty"

    order_reference, order_candidate = _paired(reference), _paired(candidate)
    ranked_reference, ranked_candidate = reference[order_reference], candidate[order_candidate]

    np.testing.assert_array_equal(
        ranked_reference[:, box_columns + 1],
        ranked_candidate[:, box_columns + 1],
        err_msg=f"{exported.task}: exported graph disagrees with the checkpoint on class ids",
    )
    np.testing.assert_allclose(
        ranked_candidate[:, :box_columns],
        ranked_reference[:, :box_columns],
        atol=_BOX_ATOL,
        err_msg=f"{exported.task}: exported box coordinates differ from the checkpoint's",
    )
    np.testing.assert_allclose(
        ranked_candidate[:, score_column],
        ranked_reference[:, score_column],
        atol=_SCORE_ATOL,
        err_msg=f"{exported.task}: exported scores differ from the checkpoint's",
    )

    if exported.task == "segment":
        masks_reference = exported.reference[1].numpy()[0][order_reference]
        masks_candidate = produced[1][0][order_candidate]
        assert masks_reference.any(), "every instance mask is empty; the comparison would be vacuous"
        np.testing.assert_array_equal(
            masks_candidate,
            masks_reference,
            err_msg="segment: exported instance masks differ from the checkpoint's",
        )

    if exported.task == "keypoints":
        points_reference = exported.reference[1].numpy()[0][order_reference]
        points_candidate = produced[1][0][order_candidate]
        # A gather at the wrong anchor still returns finite, in-canvas coordinates, so
        # the comparison needs the point sets to actually differ from one another --
        # otherwise every permutation of them matches and the parity claim is vacuous.
        assert np.ptp(points_reference) > 1.0, "every decoded point set is identical; the comparison proves nothing"
        np.testing.assert_allclose(
            points_candidate,
            points_reference,
            atol=_BOX_ATOL,
            err_msg="keypoints: exported point coordinates differ from the checkpoint's",
        )
