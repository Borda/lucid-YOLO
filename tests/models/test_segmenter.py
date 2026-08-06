# SPDX-License-Identifier: Apache-2.0
"""Unit gates for the WP-052 segmentation model wiring and its deploy-time view.

The composite itself is thin, so what is worth pinning is the wiring: that the
deployed view provably *lacks* the two training-only branches (the auxiliary
semantic classifier, A17, and the one-to-many detection branch, R6) rather than
merely skipping them, that it is a view sharing the trained parameters instead of
a copy, that the head's coefficient width and the prototype count agree well
enough for Eq. 7 to contract them, and that the Eq. 8 fused feature reaches both
consumers as one tensor. The published params/FLOPs fidelity comparison is a
separate gate and stays outside this module.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

from lucid_yolo.assign.grid import make_anchor_points
from lucid_yolo.models import Segmenter, assemble_masks, count_flops, count_params
from lucid_yolo.models.heads import SemanticAux

#: Smallest variant — the wiring is scale-independent, so the cheapest one proves it.
_VARIANT = "n"

#: Class count, coefficient width, and proto-grid side are kept mutually distinct so
#: no shape assertion below can pass by accidentally matching the wrong axis.
_NUM_CLASSES = 4
_NUM_COEFFS = 6

#: Square input side; divisible by 32, so every level's grid is exact.
_IMG_SIZE = 128

#: The head's level strides, in the (8, 16, 32) order the neck emits.
_STRIDES = [8, 16, 32]


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed torch RNG so weight init is deterministic across the gate."""
    torch.manual_seed(0)


def _anchor_count() -> int:
    """Return the anchor count for a square ``_IMG_SIZE`` input from the shared grid."""
    feature_sizes = [(_IMG_SIZE // stride, _IMG_SIZE // stride) for stride in _STRIDES]
    anchor_points, _ = make_anchor_points(feature_sizes, _STRIDES)
    return int(anchor_points.shape[0])


def _build() -> Segmenter:
    """Return the small evaluated segmenter shared by the gates below."""
    return Segmenter(_VARIANT, num_classes=_NUM_CLASSES, num_coeffs=_NUM_COEFFS).eval()


def _capture_input(store: dict[str, Tensor], key: str) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
    """Build a forward pre-hook recording the exact tensor object a module received."""

    def hook(module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        store[key] = inputs[0]

    return hook


def test_deployed_view_excludes_auxiliary_parameters() -> None:
    """No auxiliary-branch parameter object survives into the deployed view.

    Catches a deploy that keeps the training-only semantic classifier around in a
    disabled or unreachable form: those parameters would still be exported, still
    be counted, and the A17 "training-only" claim would be false for the artifact
    that actually ships.
    """
    segmenter = _build()

    deployed = segmenter.deploy()

    aux_ids = {id(parameter) for parameter in segmenter.semantic.parameters()}
    deployed_ids = {id(parameter) for parameter in deployed.parameters()}
    assert aux_ids, "the auxiliary branch must own parameters for this gate to mean anything"
    assert aux_ids.isdisjoint(deployed_ids), "deployed view still holds auxiliary-branch parameters"


def test_deployed_view_holds_no_semantic_submodule() -> None:
    """No submodule of the deployed view is a SemanticAux instance.

    Complements the parameter check: a branch held as an attribute but never
    called would pass a forward-output test and still ship in the module tree,
    the state dict, and any traced or scripted export.
    """
    segmenter = _build()

    deployed = segmenter.deploy()

    assert isinstance(segmenter.semantic, SemanticAux)
    assert not any(isinstance(module, SemanticAux) for module in deployed.modules())


def test_deployed_parameter_count_is_full_model_minus_training_only_branches() -> None:
    """Deployed params equal the full model minus the auxiliary and one-to-many branches.

    The blueprint's fused-parameter identity, extended by the one-to-many branch
    that :meth:`Segmenter.deploy` also drops. An inexact result would mean a
    parameter is shared between a kept and a dropped branch, or counted twice —
    either way the deployed model is not the clean subset it claims to be.
    """
    segmenter = _build()

    deployed = segmenter.deploy()

    expected = count_params(segmenter) - count_params(segmenter.semantic) - count_params(segmenter.head.o2m)
    assert count_params(deployed) == expected


def test_deployed_view_shares_backbone_and_neck_parameters() -> None:
    """Backbone and neck parameters are the same objects, so deploy copies nothing.

    Catches a deploy that deep-copies or rebuilds the shared trunk: the exported
    model would then silently diverge from the trained weights after any further
    training step, and would double peak memory while both live.
    """
    segmenter = _build()

    deployed = segmenter.deploy()

    assert id(next(deployed.backbone.parameters())) == id(next(segmenter.backbone.parameters()))
    assert id(next(deployed.neck.parameters())) == id(next(segmenter.neck.parameters()))


def test_deployed_output_shapes_are_mutually_consistent() -> None:
    """The four deployed tensors agree on batch, anchor, coefficient, and proto axes.

    Catches an axis swap or a stale width between the head and the prototype
    stack — for example coefficients emitted per level instead of per anchor, or
    prototypes left at P3 resolution instead of the A15 doubled grid.
    """
    segmenter = _build()
    deployed = segmenter.deploy()
    anchors = _anchor_count()

    with torch.no_grad():
        cls, box, coeff, prototypes = deployed(torch.zeros(1, 3, _IMG_SIZE, _IMG_SIZE))

    proto_side = _IMG_SIZE // 4
    assert cls.shape == (1, anchors, _NUM_CLASSES)
    assert box.shape == (1, anchors, 4)
    assert coeff.shape == (1, anchors, _NUM_COEFFS)
    assert prototypes.shape == (1, _NUM_COEFFS, proto_side, proto_side)


def test_deployed_outputs_assemble_into_instance_masks() -> None:
    """Deployed coefficients and prototypes contract through Eq. 7 without adaptation.

    The end-to-end proof that both sides really share one K: a coefficient width
    that drifted from the prototype count would raise inside the einsum rather
    than being caught only by a later training run.
    """
    segmenter = _build()
    deployed = segmenter.deploy()

    with torch.no_grad():
        _, _, coeff, prototypes = deployed(torch.zeros(1, 3, _IMG_SIZE, _IMG_SIZE))
        masks = assemble_masks(prototypes, coeff)

    proto_side = _IMG_SIZE // 4
    assert masks.shape == (1, _anchor_count(), proto_side, proto_side)
    assert torch.isfinite(masks).all()


def test_semantic_branch_is_none_at_eval_and_a_tensor_in_training() -> None:
    """The composite's semantic field follows the branch's own training-mode gate.

    Catches a wiring that materializes the auxiliary logits regardless of mode —
    or that drops them in training — either of which would break the WP-051
    auxiliary loss or leak the branch into evaluation.
    """
    segmenter = _build()
    image = torch.zeros(1, 3, _IMG_SIZE, _IMG_SIZE)

    with torch.no_grad():
        eval_output = segmenter(image)
        segmenter.train()
        train_output = segmenter(image)

    assert eval_output.semantic is None
    assert isinstance(train_output.semantic, Tensor)


def test_fused_feature_is_shared_by_prototype_and_auxiliary_branches() -> None:
    """Both mask-side consumers receive the identical fused-feature tensor object.

    Catches a forward that calls ProtoFusion twice: the results would be equal by
    value, so no numeric assertion would notice, while the fused subgraph's cost
    silently doubles in the FLOP measurement the fidelity gate reads.
    """
    segmenter = _build()
    received: dict[str, Tensor] = {}
    segmenter.protonet.register_forward_pre_hook(_capture_input(received, "protonet"))
    segmenter.semantic.register_forward_pre_hook(_capture_input(received, "semantic"))

    with torch.no_grad():
        segmenter(torch.zeros(1, 3, _IMG_SIZE, _IMG_SIZE))

    assert set(received) == {"protonet", "semantic"}
    assert received["protonet"] is received["semantic"]


def test_flop_counting_traces_both_the_full_and_deployed_models() -> None:
    """Both segmentation dataclass outputs survive the fvcore JIT trace.

    ``FlopCountAnalysis`` traces through ``torch.jit.trace``, which rejects
    dataclass outputs outright, so every new dataclass on a countable forward
    path has to be flattened by ``_TupleOutputAdapter``. Before this gate the
    adapter knew only ``DualHeadOutput``, and counting the full ``Segmenter``
    raised ``RuntimeError: ... received an input of unsupported type:
    SegmentOutput``. The full model must also cost strictly more than its
    deployed view, since deploy drops the one-to-many and auxiliary branches.
    """
    model = Segmenter(_VARIANT, num_classes=_NUM_CLASSES, num_coeffs=_NUM_COEFFS).eval()

    full = count_flops(model, img_size=_IMG_SIZE)
    deployed = count_flops(model.deploy(), img_size=_IMG_SIZE)

    assert full > 0.0
    assert deployed > 0.0
    assert full > deployed, "the training-only branches must cost something"
