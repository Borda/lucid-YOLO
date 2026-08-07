# SPDX-License-Identifier: Apache-2.0
"""Typed detector builder and param/FLOP counters (WP-023).

Assembles the WP-020…WP-022 components (:class:`~lucid_yolo.models.backbone.DetectionBackbone`,
:class:`~lucid_yolo.models.neck.DetectionNeck`, :class:`~lucid_yolo.models.heads.DualDetectionHead`)
into a single typed :class:`Detector` module for a named variant, and provides the
parameter- and FLOP-counting helpers the WP-023 fidelity gate holds to R1 Table 7.

There is no YAML and no config object (ADR-001): the topology is Python, the only
free parameters are the variant name and the class count, and the compound-scaling
multipliers come from :func:`~lucid_yolo.models.registry.scale_spec`.

**Two measured reporting conventions (WP-023), both matching the R1 Table 7 /
dual-assignment lineage (R6):**

1. *FLOPs are conventional GFLOPs* = ``2 x`` the fvcore multiply-accumulate (MAC)
   count. :class:`fvcore.nn.FlopCountAnalysis` tallies one MAC per count; the raw
   MAC total lands ~48% below Table 7 for every scale, and doubling it (the
   conventional "FLOP = one multiply + one add" definition) lands within
   tolerance. :func:`count_flops` therefore returns ``2 x`` the MAC total.

2. *Params count the full trained model, FLOPs count the deployed inference
   model.* The dual head owns two structurally identical branches (one-to-one and
   one-to-many); the one-to-many branch exists only to supervise training and is
   never executed at NMS-free (E2E) inference (R6). So the checkpoint's parameter
   count includes both branches (:func:`count_params` on the whole
   :class:`Detector`), while the inference GFLOPs count only one branch
   (:func:`count_flops` on :meth:`Detector.deploy`). This is the YOLOv10/R6
   reporting convention; with it, all five scales land within +/-2% params and
   +/-5% FLOPs of R1 Table 7.

Both were determined empirically by the fidelity gate — trying the alternatives
(raw MAC; both-branch FLOPs) is measurement, not an assumption iteration.

Provenance: R1 Table 7, R6. Assumptions: A3, A4, A28, A29.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from fvcore.nn import FlopCountAnalysis
from torch import Tensor, nn

from lucid_yolo.models.backbone import DetectionBackbone
from lucid_yolo.models.heads import DualDetectionHead, DualHeadOutput, ProtoFusion, ProtoNet, SemanticAux
from lucid_yolo.models.heads.detect import DEFAULT_NUM_COEFFS
from lucid_yolo.models.neck import DetectionNeck
from lucid_yolo.models.registry import scale_spec

__all__ = [
    "Detector",
    "SegmentOutput",
    "Segmenter",
    "build_detection_stages",
    "build_detector",
    "build_segmentation_stages",
    "build_segmenter",
    "count_flops",
    "count_params",
]

#: Default detection input side in pixels (square), R1 Table 7 protocol.
_DEFAULT_IMG_SIZE = 640

#: Default COCO detection class count.
_DEFAULT_NUM_CLASSES = 80


def build_detection_stages(
    depth: float,
    width: float,
    max_channels: int,
    num_classes: int,
    num_coeffs: int | None = None,
) -> tuple[DetectionBackbone, DetectionNeck, DualDetectionHead]:
    """Construct the three detection stages from raw compound-scaling numbers.

    The single construction site for ``backbone -> neck -> head``. It exists
    because the model is composed in two places — :class:`Detector` (what the R1
    Table 7 fidelity gate measures) and
    :class:`~lucid_yolo.ptl.module.DetectionLitModule` (what training actually
    runs) — and a duplicated composition lets the two drift apart while the gate
    keeps certifying the copy nobody trains.

    It takes the **raw** multipliers rather than a variant name because the
    Lightning module is configured with raw numbers from YAML (ADR-001, no config
    object); :class:`Detector` resolves its variant through
    :func:`~lucid_yolo.models.registry.scale_spec` and passes the numbers through.

    The stages are returned rather than wrapped in a container: each caller
    assigns them to the attribute names it already used, so no state-dict key
    moves and existing checkpoints keep loading.

    Args:
        depth: Compound-scaling depth multiplier.
        width: Compound-scaling width multiplier.
        max_channels: Channel-count ceiling applied after width scaling.
        num_classes: Number of object classes both head branches predict.
        num_coeffs: Optional mask-coefficient width ``K`` (A14) enabling the
            head's coefficient stems. ``None`` (the default) builds the
            detection-only head, whose parameters and state-dict keys are exactly
            those of the pre-segmentation head.

    Returns:
        The ``(backbone, neck, head)`` triple, already wired to each other's
        channel counts.

    Examples:
        >>> backbone, neck, head = build_detection_stages(0.34, 0.25, 1024, num_classes=4)
        >>> backbone.channels, neck.channels
        ((128, 128, 256), (64, 128, 256))
        >>> head.num_classes, head.o2o.coeff_stems is None  # detection-only by default
        (4, True)
    """
    backbone = DetectionBackbone(depth=depth, width=width, max_channels=max_channels)
    neck = DetectionNeck(backbone.channels, depth=depth, width=width, max_channels=max_channels)
    head = DualDetectionHead(neck.channels, num_classes=num_classes, num_coeffs=num_coeffs)
    return backbone, neck, head


def build_segmentation_stages(
    neck_channels: tuple[int, int, int],
    num_classes: int,
    num_coeffs: int,
) -> tuple[ProtoFusion, ProtoNet, SemanticAux]:
    """Construct the three segmentation branches that sit on the neck's features.

    The mask-side counterpart of :func:`build_detection_stages`, and the single
    construction site for the WP-047…WP-050 branches, so :class:`Segmenter` and
    :class:`~lucid_yolo.ptl.module.DetectionLitModule` cannot disagree about what
    the segmentation model is.

    ``num_coeffs`` is required, not defaulted: Eq. 7 contracts the head's
    coefficients against the prototypes one-for-one, so the count must come from
    the same place that built the head rather than being re-defaulted here.

    The fused-feature width is read off the constructed
    :class:`~lucid_yolo.models.heads.ProtoFusion` rather than recomputed, which is
    what keeps the prototype and auxiliary branches attached to the same feature.

    Args:
        neck_channels: The neck's per-level output channels, ``(P3, P4, P5)``.
        num_classes: Number of classes the auxiliary semantic branch predicts.
        num_coeffs: Mask-coefficient width ``K`` (A14), also the prototype count.

    Returns:
        The ``(proto_fusion, protonet, semantic)`` triple.

    Examples:
        >>> fusion, protonet, semantic = build_segmentation_stages((64, 128, 256), 4, 32)
        >>> protonet.num_prototypes
        32
        >>> semantic.classifier.in_channels == fusion.out_channels
        True
    """
    proto_fusion = ProtoFusion(neck_channels)
    protonet = ProtoNet(proto_fusion.out_channels, num_prototypes=num_coeffs)
    semantic = SemanticAux(proto_fusion.out_channels, num_classes)
    return proto_fusion, protonet, semantic


class Detector(nn.Module):
    """Composite YOLO26 detector: backbone -> neck -> dual detection head.

    A single typed module wiring the three WP-020…WP-022 stages for one scale
    variant. :meth:`forward` takes an image batch and returns the head's
    :class:`~lucid_yolo.models.heads.DualHeadOutput` (both the one-to-one and
    one-to-many dense predictions).

    Args:
        variant: Scale name (``"n"``/``"s"``/``"m"``/``"l"``/``"x"``) resolved
            through :func:`~lucid_yolo.models.registry.scale_spec`.
        num_classes: Number of object classes the head predicts.

    Examples:
        >>> import torch
        >>> model = Detector("n", num_classes=80).eval()
        >>> with torch.no_grad():
        ...     out = model(torch.zeros(1, 3, 640, 640))
        >>> out.o2o_cls.shape, out.o2o_box.shape
        (torch.Size([1, 8400, 80]), torch.Size([1, 8400, 4]))
    """

    def __init__(self, variant: str, num_classes: int = _DEFAULT_NUM_CLASSES) -> None:
        super().__init__()
        spec = scale_spec(variant)
        self.variant = variant
        self.num_classes = num_classes
        self.backbone, self.neck, self.head = build_detection_stages(
            spec.depth, spec.width, spec.max_channels, num_classes
        )

    def forward(self, image: Tensor) -> DualHeadOutput:
        """Run the backbone, neck, and dual head over an image batch.

        Args:
            image: Input image batch of shape ``(N, 3, H, W)`` with ``H`` and
                ``W`` divisible by 32.

        Returns:
            The head's :class:`~lucid_yolo.models.heads.DualHeadOutput` with dense
            class logits and raw ltrb distances for both branches.
        """
        output: DualHeadOutput = self.head(self.neck(self.backbone(image)))
        return output

    def deploy(self) -> nn.Module:
        """Return the NMS-free inference model: backbone -> neck -> one-to-one head.

        The one-to-many branch is training-only (R6) and never runs at E2E
        inference, so the deployed model executes a single detection branch. This
        is the module whose FLOPs match the R1 Table 7 GFLOP column (see the module
        docstring); its parameters are a strict subset of the full detector's, so
        model size is still reported from the whole :class:`Detector`. The returned
        module **shares** this detector's parameters (no copy).

        Returns:
            A :class:`torch.nn.Module` whose forward maps an image batch to the
            one-to-one branch's ``(cls_logits, ltrb)`` tensor pair.

        Examples:
            >>> import torch
            >>> deployed = Detector("n").deploy().eval()
            >>> with torch.no_grad():
            ...     cls, box = deployed(torch.zeros(1, 3, 640, 640))
            >>> cls.shape, box.shape
            (torch.Size([1, 8400, 80]), torch.Size([1, 8400, 4]))
        """
        return _DeployedDetector(self)


class _DeployedDetector(nn.Module):
    """Single-branch inference view of a :class:`Detector` (backbone/neck/o2o).

    Runs the backbone, neck, and only the one-to-one head branch, returning that
    branch's dense ``(cls, box)`` tensors. Holds references to the parent's
    submodules, so it shares parameters and adds none of its own.

    Args:
        detector: The full detector to expose an inference view of.
    """

    def __init__(self, detector: Detector) -> None:
        super().__init__()
        self.backbone = detector.backbone
        self.neck = detector.neck
        self.o2o = detector.head.o2o

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Run the backbone, neck, and one-to-one branch over an image batch.

        Args:
            image: Input image batch of shape ``(N, 3, H, W)`` with ``H`` and
                ``W`` divisible by 32.

        Returns:
            The one-to-one branch's ``(cls, box)`` pair: dense class logits of
            shape ``(N, A, num_classes)`` and raw ltrb distances ``(N, A, 4)``.
        """
        cls, box = self.o2o(self.neck(self.backbone(image)))
        return cls, box


def build_detector(variant: str, num_classes: int = _DEFAULT_NUM_CLASSES) -> Detector:
    """Build a :class:`Detector` for a named scale variant.

    Args:
        variant: Scale name (``"n"``/``"s"``/``"m"``/``"l"``/``"x"``).
        num_classes: Number of object classes. Defaults to 80 (COCO).

    Returns:
        The assembled :class:`Detector` module.

    Raises:
        KeyError: If ``variant`` is not one of the five published names.

    Examples:
        >>> model = build_detector("s")
        >>> model.variant, model.num_classes
        ('s', 80)
        >>> model.backbone.width
        0.5
    """
    return Detector(variant, num_classes)


@dataclass(frozen=True)
class SegmentOutput:
    """Dense predictions of the segmentation model's three branches.

    Groups what a segmentation forward pass produces without flattening it: the
    detection head's own dataclass is kept whole (so the dual-branch contract of
    :class:`~lucid_yolo.models.heads.DualHeadOutput` stays in one place), and the
    two mask-side tensors sit beside it.

    Attributes:
        detect: The full :class:`~lucid_yolo.models.heads.DualHeadOutput`,
            including both branches' tanh mask coefficients.
        prototypes: Raw Eq. 9 prototype maps of shape ``(B, K, 2*H3, 2*W3)``,
            i.e. twice the P3 grid (A15), unactivated.
        semantic: Auxiliary per-class logits ``(B, num_classes, H3, W3)`` in
            training mode, ``None`` at eval — the branch is training-only (A17)
            and :class:`~lucid_yolo.models.heads.SemanticAux` returns ``None``
            outside training rather than being skipped by the caller.
    """

    detect: DualHeadOutput
    prototypes: Tensor
    semantic: Tensor | None


class Segmenter(nn.Module):
    """Composite YOLO26 segmentation model: detector plus the mask and aux branches.

    Mirrors :class:`Detector` — same backbone and neck for a given variant — and
    adds the three WP-047…WP-050 segmentation stages on top: the detection head
    now also emits per-anchor mask coefficients,
    :class:`~lucid_yolo.models.heads.ProtoFusion` collapses the neck's three
    levels into the Eq. 8 fused feature, :class:`~lucid_yolo.models.heads.ProtoNet`
    turns that feature into ``K`` prototype maps, and
    :class:`~lucid_yolo.models.heads.SemanticAux` attaches the training-only
    auxiliary classifier to the same fused feature.

    ``num_coeffs`` is threaded into both the head's coefficient stems and the
    prototype count, one prototype per coefficient: Eq. 7 contracts the two
    against each other, so a single constructor argument is what stops the two
    sides from drifting apart.

    The fused feature is computed exactly once and shared by the prototype and
    auxiliary branches — recomputing it would silently double that subgraph's
    cost in the FLOP measurement.

    Args:
        variant: Scale name (``"n"``/``"s"``/``"m"``/``"l"``/``"x"``) resolved
            through :func:`~lucid_yolo.models.registry.scale_spec`.
        num_classes: Number of object classes the head and the auxiliary branch
            predict.
        num_coeffs: Mask-coefficient width ``K`` (A14), also the prototype count.

    Examples:
        >>> import torch
        >>> model = Segmenter("n", num_classes=4).eval()
        >>> with torch.no_grad():
        ...     out = model(torch.zeros(1, 3, 128, 128))
        >>> out.detect.o2o_cls.shape, out.detect.o2o_coeff.shape
        (torch.Size([1, 336, 4]), torch.Size([1, 336, 32]))
        >>> out.prototypes.shape
        torch.Size([1, 32, 32, 32])
        >>> out.semantic is None  # training-only branch (A17)
        True
    """

    def __init__(
        self,
        variant: str,
        num_classes: int = _DEFAULT_NUM_CLASSES,
        num_coeffs: int = DEFAULT_NUM_COEFFS,
    ) -> None:
        super().__init__()
        spec = scale_spec(variant)
        self.variant = variant
        self.num_classes = num_classes
        self.num_coeffs = num_coeffs
        self.backbone, self.neck, self.head = build_detection_stages(
            spec.depth, spec.width, spec.max_channels, num_classes, num_coeffs=num_coeffs
        )
        self.proto_fusion, self.protonet, self.semantic = build_segmentation_stages(
            self.neck.channels, num_classes, num_coeffs
        )

    def forward(self, image: Tensor) -> SegmentOutput:
        """Run the backbone, neck, dual head, prototype stack, and auxiliary branch.

        Args:
            image: Input image batch of shape ``(N, 3, H, W)`` with ``H`` and
                ``W`` divisible by 32.

        Returns:
            A :class:`SegmentOutput` holding the dual head's dense predictions,
            the raw prototype maps, and the auxiliary semantic logits (``None``
            outside training mode).
        """
        features: tuple[Tensor, Tensor, Tensor] = self.neck(self.backbone(image))
        detect: DualHeadOutput = self.head(features)
        fused: Tensor = self.proto_fusion(features)
        prototypes: Tensor = self.protonet(fused)
        semantic: Tensor | None = self.semantic(fused)
        return SegmentOutput(detect=detect, prototypes=prototypes, semantic=semantic)

    def deploy(self) -> nn.Module:
        """Return the inference model: backbone -> neck -> one-to-one head + prototypes.

        Two branches of the trained model are training-only and are therefore
        absent from the returned module: the one-to-many detection branch, which
        never runs on the NMS-free E2E path (R6), and the auxiliary semantic
        branch, which exists only to shape the shared prototype features (A17).
        Neither is disabled or skipped — neither is held at all. The returned
        module **shares** this segmenter's parameters (no copy), so it is a view
        rather than a second model.

        Returns:
            A :class:`torch.nn.Module` whose forward maps an image batch to the
            ``(cls, box, coeff, prototypes)`` tuple needed to assemble masks.

        Examples:
            >>> import torch
            >>> deployed = Segmenter("n", num_classes=4).deploy().eval()
            >>> with torch.no_grad():
            ...     cls, box, coeff, prototypes = deployed(torch.zeros(1, 3, 128, 128))
            >>> cls.shape, box.shape
            (torch.Size([1, 336, 4]), torch.Size([1, 336, 4]))
            >>> coeff.shape, prototypes.shape
            (torch.Size([1, 336, 32]), torch.Size([1, 32, 32, 32]))
        """
        return _DeployedSegmenter(self)


class _DeployedSegmenter(nn.Module):
    """Inference view of a :class:`Segmenter` (backbone/neck/o2o + prototypes).

    Runs the backbone, the neck, the one-to-one head branch, and the prototype
    path, returning the four tensors :func:`~lucid_yolo.models.heads.assemble_masks`
    and the E2E decode need. Holds references to the parent's submodules, so it
    shares parameters and adds none of its own.

    The one-to-many detection branch is training-only (R6) and the auxiliary
    semantic branch is training-only (A17), so this module carries neither: the
    auxiliary branch is not an attribute here at all, which is what makes its
    removal checkable by parameter identity rather than by trusting a flag. This
    is the module whose parameters and FLOPs the Phase 7 segmentation fidelity
    gate is measured on.

    Args:
        segmenter: The full segmentation model to expose an inference view of.
    """

    def __init__(self, segmenter: Segmenter) -> None:
        super().__init__()
        self.backbone = segmenter.backbone
        self.neck = segmenter.neck
        self.o2o = segmenter.head.o2o
        self.proto_fusion = segmenter.proto_fusion
        self.protonet = segmenter.protonet

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the backbone, neck, one-to-one branch, and prototype stack.

        Args:
            image: Input image batch of shape ``(N, 3, H, W)`` with ``H`` and
                ``W`` divisible by 32.

        Returns:
            The tuple ``(cls, box, coeff, prototypes)``: dense class logits
            ``(N, A, num_classes)``, raw ltrb distances ``(N, A, 4)``,
            tanh mask coefficients ``(N, A, K)``, and raw prototype maps
            ``(N, K, H/4, W/4)``.
        """
        features: tuple[Tensor, Tensor, Tensor] = self.neck(self.backbone(image))
        cls, box, coeff = self.o2o(features)
        prototypes: Tensor = self.protonet(self.proto_fusion(features))
        return cls, box, coeff, prototypes


def build_segmenter(variant: str, num_classes: int = _DEFAULT_NUM_CLASSES) -> Segmenter:
    """Build a :class:`Segmenter` for a named scale variant.

    Args:
        variant: Scale name (``"n"``/``"s"``/``"m"``/``"l"``/``"x"``).
        num_classes: Number of object classes. Defaults to 80 (COCO).

    Returns:
        The assembled :class:`Segmenter` module, with the default A14
        coefficient width ``K`` shared by the head and the prototype stack.

    Raises:
        KeyError: If ``variant`` is not one of the five published names.

    Examples:
        >>> model = build_segmenter("s")
        >>> model.variant, model.num_classes, model.num_coeffs
        ('s', 80, 32)
        >>> model.protonet.num_prototypes  # one prototype per coefficient
        32
    """
    return Segmenter(variant, num_classes)


def count_params(module: nn.Module) -> int:
    """Count the total number of parameters in a module.

    Args:
        module: Any :class:`torch.nn.Module`.

    Returns:
        The exact total element count summed over ``module.parameters()``.

    Examples:
        >>> count_params(build_detector("n")) > 2_000_000
        True
    """
    return sum(param.numel() for param in module.parameters())


def _tensor_fields(*fields: Tensor | None) -> tuple[Tensor, ...]:
    """Drop the ``None`` entries from a dataclass's flattened tensor fields.

    Args:
        fields: Dataclass field values in declaration order, any of which may be
            ``None`` for an optional branch that was not built or is inactive.

    Returns:
        The present tensors, in the given order.

    Examples:
        >>> import torch
        >>> _tensor_fields(torch.zeros(1), None, torch.ones(2))[1].numel()
        2
    """
    return tuple(field for field in fields if field is not None)


class _TupleOutputAdapter(nn.Module):
    """Wrap a module so its forward returns a tuple, for FLOP tracing.

    :class:`fvcore.nn.FlopCountAnalysis` traces through :func:`torch.jit.trace`,
    which rejects dataclass outputs like
    :class:`~lucid_yolo.models.heads.DualHeadOutput` and :class:`SegmentOutput`.
    This adapter unpacks such an output to a plain tuple of tensors (leaving
    already-tensor/tuple outputs untouched) so the trace succeeds; it adds no
    compute of its own.

    Args:
        module: The module whose forward FLOPs are to be counted.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, image: Tensor) -> tuple[Tensor, ...] | Tensor:
        """Run the wrapped module and flatten a dataclass output to a tuple.

        Args:
            image: Input image batch forwarded verbatim to the wrapped module.

        Returns:
            The wrapped module's output as a tensor tuple when it is a
            :class:`~lucid_yolo.models.heads.DualHeadOutput` or a
            :class:`SegmentOutput`, otherwise unchanged. Optional fields that are
            ``None`` are dropped: the coefficient tensors are absent unless the
            head was built with ``num_coeffs``, and the auxiliary semantic logits
            are always ``None`` here because FLOP counting runs in eval mode
            (A17). Dropping them is safe for a FLOP tally, which reads the traced
            graph rather than the returned values.
        """
        output = self.module(image)
        if isinstance(output, DualHeadOutput):
            return _tensor_fields(
                output.o2m_cls, output.o2m_box, output.o2m_coeff, output.o2o_cls, output.o2o_box, output.o2o_coeff
            )
        if isinstance(output, SegmentOutput):
            detect = output.detect
            return _tensor_fields(
                detect.o2m_cls,
                detect.o2m_box,
                detect.o2m_coeff,
                detect.o2o_cls,
                detect.o2o_box,
                detect.o2o_coeff,
                output.prototypes,
                output.semantic,
            )
        return cast("tuple[Tensor, ...] | Tensor", output)


def count_flops(module: nn.Module, img_size: int = _DEFAULT_IMG_SIZE) -> float:
    """Count a module's forward GFLOPs at a square input, in units of 1e9.

    Runs :class:`fvcore.nn.FlopCountAnalysis` on a single zero image of side
    ``img_size``. fvcore tallies multiply-accumulate operations (one MAC per
    count); this returns ``2 x`` that total so the result is *conventional*
    GFLOPs (one multiply + one add = 2 FLOPs), which is what R1 Table 7 reports —
    the raw MAC total lands ~48% low, doubling lands within tolerance (measured
    convention, WP-023; see the module docstring). fvcore's JIT trace rejects the
    :class:`~lucid_yolo.models.heads.DualHeadOutput` dataclass, so a dataclass-output
    module is wrapped to flatten it; tensor/tuple outputs (e.g.
    :meth:`Detector.deploy`) pass through untouched.

    For R1 Table 7 fidelity, pass the deployed inference model
    (``build_detector(v).deploy()``), whose GFLOPs exclude the training-only
    one-to-many branch (see the module docstring); passing the full
    :class:`Detector` counts both branches.

    Args:
        module: A module accepting a ``(1, 3, H, W)`` image (a
            :class:`Detector`, a :meth:`Detector.deploy` view, or any submodule).
        img_size: Square input side in pixels. Defaults to 640.

    Returns:
        The total forward GFLOPs (``2 x`` MACs) divided by 1e9.

    Examples:
        >>> round(count_flops(build_detector("n").deploy()), 1) > 0
        True
    """
    was_training = module.training
    module.eval()
    image = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        analysis = FlopCountAnalysis(_TupleOutputAdapter(module), image)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        total = analysis.total()
    if was_training:
        module.train()
    return 2.0 * float(total) / 1e9
