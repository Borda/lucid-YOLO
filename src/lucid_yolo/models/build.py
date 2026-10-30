# SPDX-License-Identifier: Apache-2.0
"""Typed detector builder and param/FLOP counters (WP-023).

Assembles the WP-020…WP-022 components (:class:`~lucid_yolo.models.backbone.DetectionBackbone`,
:class:`~lucid_yolo.models.neck.DetectionNeck`, :class:`~lucid_yolo.models.heads.DualDetectionHead`)
into a single typed :class:`Detector` module for a named variant, and provides the parameter- and FLOP-counting
helpers the WP-023 fidelity gate holds to R1 Table 7.

There is no YAML and no config object (ADR-001): the topology is Python, the only free parameters are the variant
name and the class count, and the compound-scaling multipliers come from :func:`~lucid_yolo.models.registry.scale_spec`.

**Two measured reporting conventions (WP-023), both matching the R1 Table 7 / dual-assignment lineage (R6):**

1. *FLOPs are conventional GFLOPs* = ``2 x`` the fvcore multiply-accumulate (MAC) count.
   :class:`fvcore.nn.FlopCountAnalysis` tallies one MAC per count; the raw MAC total lands ~48% below Table 7 for
   every scale, and doubling it (the conventional "FLOP = one multiply + one add" definition) lands within tolerance.
   :func:`count_flops` therefore returns ``2 x`` the MAC total.

2. *Params count the full trained model, FLOPs count the deployed inference model.* The dual head owns two structurally
   identical branches (one-to-one and one-to-many); the one-to-many branch exists only to supervise training and is
   never executed at NMS-free (E2E) inference (R6). So the checkpoint's parameter count includes both branches
   (:func:`count_params` on the whole :class:`Detector`), while the inference GFLOPs count only one branch
   (:func:`count_flops` on :meth:`Detector.deploy`). This is the YOLOv10/R6 reporting convention; with it, all five
   scales land within +/-2% params and +/-5% FLOPs of R1 Table 7.

Both were determined empirically by the fidelity gate — trying the alternatives (raw MAC; both-branch FLOPs) is
measurement, not an assumption iteration.

Provenance: R1 Table 7, R6. Assumptions: A3, A4, A28, A29.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import ClassVar, cast

import torch
from fvcore.nn import FlopCountAnalysis
from torch import Tensor, nn

from lucid_yolo.models.backbone import DetectionBackbone
from lucid_yolo.models.heads import (
    BranchOutput,
    DualDetectionHead,
    DualHeadOutput,
    ProtoFusion,
    ProtoNet,
    SemanticAux,
)
from lucid_yolo.models.heads.detect import DEFAULT_NUM_COEFFS
from lucid_yolo.models.neck import DetectionNeck
from lucid_yolo.models.registry import ScaleSpec, scale_spec
from lucid_yolo.tasks import Task

__all__ = [
    "Detector",
    "KeypointDetector",
    "OrientedDetector",
    "SegmentOutput",
    "Segmenter",
    "build_detection_stages",
    "build_detector",
    "build_keypoint_detector",
    "build_obb_detector",
    "build_segmentation_stages",
    "build_segmenter",
    "count_flops",
    "count_params",
]

#: Default detection input side in pixels (square), R1 Table 7 protocol.
_DEFAULT_IMG_SIZE = Task.DETECT.spec.img_size

#: Default COCO detection class count.
_DEFAULT_NUM_CLASSES = 80


def build_detection_stages(
    depth: float,
    width: float,
    max_channels: int,
    num_classes: int,
    num_coeffs: int | None = None,
    predict_angle: bool = False,
    num_keypoints: int | None = None,
) -> tuple[DetectionBackbone, DetectionNeck, DualDetectionHead]:
    """Construct the three detection stages from raw compound-scaling numbers.

    The single construction site for ``backbone -> neck -> head``. It exists because the model is composed in two
    places — :class:`Detector` (what the R1 Table 7 fidelity gate measures) and
    :class:`~lucid_yolo.ptl.module.DetectionLitModule` (what training actually runs) — and a duplicated composition
    lets the two drift apart while the gate keeps certifying the copy nobody trains.

    It takes the **raw** multipliers rather than a variant name because the Lightning module is configured with raw
    numbers from YAML (ADR-001, no config object); :class:`Detector` resolves its variant through
    :func:`~lucid_yolo.models.registry.scale_spec` and passes the numbers through.

    The stages are returned rather than wrapped in a container: each caller assigns them to the attribute names it
    already used, so no state-dict key moves and existing checkpoints keep loading.

    Args:
        depth: Compound-scaling depth multiplier.
        width: Compound-scaling width multiplier.
        max_channels: Channel-count ceiling applied after width scaling.
        num_classes: Number of object classes both head branches predict.
        num_coeffs: Optional mask-coefficient width ``K`` (A14) enabling the head's coefficient stems. ``None`` (the
            default) builds the detection-only head, whose parameters and state-dict keys are exactly those of the
            pre-segmentation head.
        predict_angle: Build the head's orientation stems (A20), the oriented path's opt-in. ``False`` (the default)
            leaves the head angle-free.
        num_keypoints: Optional point count ``K`` enabling the head's keypoint stems (WP-122), the pose path's opt-in.
            Unlike ``num_coeffs`` there is no project-wide default to fall back on: ``K`` is a property of the
            *dataset's* annotation schema (COCO person is 17, another pose set is not), so a default here would be a
            silent claim about data this factory has never seen. ``None`` (the default) leaves the head keypoint-free.

    Returns:
        The ``(backbone, neck, head)`` triple, already wired to each other's channel counts.

    Examples:
        ```pycon
        >>> backbone, neck, head = build_detection_stages(0.34, 0.25, 1024, num_classes=4)
        >>> backbone.channels, neck.channels
        ((128, 128, 256), (64, 128, 256))
        >>> head.num_classes, head.o2o.coeff_stems is None  # detection-only by default
        (4, True)
        >>> head.o2o.angle_stems is None  # and angle-free by default
        True
        >>> head.o2o.keypoint_stems is None  # and keypoint-free by default
        True

        ```
    """
    backbone = DetectionBackbone(depth=depth, width=width, max_channels=max_channels)
    neck = DetectionNeck(backbone.channels, depth=depth, width=width, max_channels=max_channels)
    head = DualDetectionHead(
        neck.channels,
        num_classes=num_classes,
        num_coeffs=num_coeffs,
        predict_angle=predict_angle,
        num_keypoints=num_keypoints,
    )
    return backbone, neck, head


def build_segmentation_stages(
    neck_channels: tuple[int, int, int],
    num_classes: int,
    num_coeffs: int,
) -> tuple[ProtoFusion, ProtoNet, SemanticAux]:
    """Construct the three segmentation branches that sit on the neck's features.

    The mask-side counterpart of :func:`build_detection_stages`, and the single construction site for the
    WP-047…WP-050 branches, so :class:`Segmenter` and :class:`~lucid_yolo.ptl.module.DetectionLitModule` cannot
    disagree about what the segmentation model is.

    ``num_coeffs`` is required, not defaulted: Eq. 7 contracts the head's coefficients against the prototypes
    one-for-one, so the count must come from the same place that built the head rather than being re-defaulted here.

    The fused-feature width is read off the constructed :class:`~lucid_yolo.models.heads.ProtoFusion` rather than
    recomputed, which is what keeps the prototype and auxiliary branches attached to the same feature.

    Args:
        neck_channels: The neck's per-level output channels, ``(P3, P4, P5)``.
        num_classes: Number of classes the auxiliary semantic branch predicts.
        num_coeffs: Mask-coefficient width ``K`` (A14), also the prototype count.

    Returns:
        The ``(proto_fusion, protonet, semantic)`` triple.

    Examples:
        ```pycon
        >>> fusion, protonet, semantic = build_segmentation_stages((64, 128, 256), 4, 32)
        >>> protonet.num_prototypes
        32
        >>> semantic.classifier.in_channels == fusion.out_channels
        True

        ```
    """
    proto_fusion = ProtoFusion(neck_channels)
    protonet = ProtoNet(proto_fusion.out_channels, num_prototypes=num_coeffs)
    semantic = SemanticAux(proto_fusion.out_channels, num_classes)
    return proto_fusion, protonet, semantic


def _resolve_scaling(variant: str | ScaleSpec) -> ScaleSpec:
    """Return the compound-scaling multipliers a composite model is built at.

    A name is looked up in the registry; a :class:`~lucid_yolo.models.registry.ScaleSpec` is taken as given. The
    second form exists for the one caller that holds multipliers rather than a name:
    a :class:`~lucid_yolo.ptl.module.DetectionLitModule` is configured with raw ``depth``/``width``/``max_channels``
    (ADR-001), and nothing requires those to be a registry row, so rebuilding its architecture as a composite — what
    :func:`~lucid_yolo.export.graph_from_module` does — has to accept them as they are rather than refuse every module
    the training CLI did not configure.

    Args:
        variant: Scale name (``"n"``/``"s"``/``"m"``/``"l"``/``"x"``) or raw multipliers.

    Returns:
        The multipliers to build at.

    Raises:
        KeyError: If ``variant`` is a name that is not one of the five published ones.

    Examples:
        ```pycon
        >>> _resolve_scaling("s")
        ScaleSpec(depth=0.5, width=0.5, max_channels=1024)
        >>> _resolve_scaling(ScaleSpec(depth=0.34, width=0.25, max_channels=64))
        ScaleSpec(depth=0.34, width=0.25, max_channels=64)

        ```
    """
    return variant if isinstance(variant, ScaleSpec) else scale_spec(variant)


class _SingleHeadModel(nn.Module):
    """Backbone -> neck -> dual detection head: the body every single-head composite shares.

    :class:`Detector`, :class:`OrientedDetector` and :class:`KeypointDetector` used to be three copies of this class
    that differed in one stage option each — none, ``predict_angle`` and ``num_keypoints`` — and in which one-to-one
    fields their :meth:`deploy` view returned. Both differences are data now: the stage option is an argument here,
    and the field list is :attr:`_DEPLOYED_FIELDS`, a class attribute each subclass sets (subclass hook).

    The three stages are assigned as ``backbone``, ``neck`` and ``head``, in that order, exactly as each copy assigned
    them, so every state-dict key and module name is the one the composite carried before it shared this body. The
    WP-023 parameter and FLOP goldens read those models, and a renamed key would move them.

    Args:
        variant: Scale name resolved through :func:`~lucid_yolo.models.registry.scale_spec`, or a
            :class:`~lucid_yolo.models.registry.ScaleSpec` of raw multipliers (see :func:`_resolve_scaling`). Stored
            as given on ``variant``.
        num_classes: Number of object classes the head predicts.
        predict_angle: Build the head's A20 orientation stems on both branches.
        num_keypoints: Build the head's point stems on both branches for this many points; ``None`` leaves the head
            keypoint-free.
    """

    #: One-to-one :class:`~lucid_yolo.models.heads.BranchOutput` fields :meth:`deploy` returns, in this order
    #: (subclass hook).
    _DEPLOYED_FIELDS: ClassVar[tuple[str, ...]] = ("cls", "box")

    def __init__(
        self,
        variant: str | ScaleSpec,
        num_classes: int,
        *,
        predict_angle: bool = False,
        num_keypoints: int | None = None,
    ) -> None:
        super().__init__()
        scaling = _resolve_scaling(variant)
        self.variant = variant
        self.num_classes = num_classes
        self.backbone, self.neck, self.head = build_detection_stages(
            scaling.depth,
            scaling.width,
            scaling.max_channels,
            num_classes,
            predict_angle=predict_angle,
            num_keypoints=num_keypoints,
        )

    def forward(self, image: Tensor) -> DualHeadOutput:
        """Run the backbone, neck, and dual head over an image batch.

        Args:
            image: Input image batch of shape ``(N, 3, H, W)`` with ``H`` and
                ``W`` divisible by 32.

        Returns:
            The head's :class:`~lucid_yolo.models.heads.DualHeadOutput` with dense class logits and raw ltrb distances
            for both branches, plus the optional fields the model's stage option enables (angle, or keypoints and
            keypoint sigma), populated for both branches.
        """
        output: DualHeadOutput = self.head(self.neck(self.backbone(image)))
        return output

    def deploy(self) -> nn.Module:
        """Return the NMS-free inference model: backbone -> neck -> one-to-one head.

        The one-to-many branch is training-only (R6) and never runs at E2E inference, so the deployed model executes a
        single detection branch, including only that branch's optional stems. The returned module **shares** this
        model's parameters (no copy); each subclass docstring states which tensors it returns and which golden reads
        its FLOPs.

        Returns:
            A :class:`torch.nn.Module` whose forward maps an image batch to the
            one-to-one branch's :attr:`_DEPLOYED_FIELDS` tensors, in that order.
        """
        return _DeployedO2O(self.backbone, self.neck, self.head.o2o, self._DEPLOYED_FIELDS)


class _DeployedO2O(nn.Module):
    """Single-branch inference view: backbone, neck and the one-to-one head branch.

    The one view every composite's ``deploy()`` returns, replacing four per-task copies that differed only in the
    fields they returned and, for segmentation, in also running the prototype path. It holds references to the
    parent's submodules under the names those copies used — ``backbone``, ``neck`` and ``o2o``, plus ``proto_fusion``
    and ``protonet`` when the prototype path is given — so it shares parameters, adds none of its own, and presents
    the same module tree to FLOP counting.

    What it does not hold matters as much. The one-to-many branch — its box, class, angle and point stems alike — is
    not an attribute here at all, and neither is segmentation's auxiliary semantic branch (A17), which is what makes
    their removal checkable by parameter identity rather than by trusting a flag.

    Args:
        backbone: The parent's backbone.
        neck: The parent's neck.
        o2o: The parent head's one-to-one branch.
        fields: :class:`~lucid_yolo.models.heads.BranchOutput` field names returned from that branch, in order.
        proto_fusion: The parent's prototype-fusion stage; given together with ``protonet`` or not at all.
        protonet: The parent's prototype stack. When given, the raw prototype maps are appended to the returned tuple.

    Raises:
        ValueError: If ``fields`` names something :class:`BranchOutput` has no field for, or if only one of
            ``proto_fusion`` and ``protonet`` is given.

    Examples:
        ```pycon
        >>> import torch
        >>> detector = Detector("n", num_classes=4).eval()
        >>> view = _DeployedO2O(detector.backbone, detector.neck, detector.head.o2o, ("cls", "box")).eval()
        >>> with torch.no_grad():
        ...     cls, box = view(torch.zeros(1, 3, 64, 64))
        >>> cls.shape, box.shape
        (torch.Size([1, 84, 4]), torch.Size([1, 84, 4]))
        >>> _DeployedO2O(detector.backbone, detector.neck, detector.head.o2o, ("cls", "angel"))
        Traceback (most recent call last):
            ...
        ValueError: unknown one-to-one fields ['angel']; BranchOutput has cls, box, coeff, ...

        ```
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        o2o: nn.Module,
        fields: tuple[str, ...],
        *,
        proto_fusion: nn.Module | None = None,
        protonet: nn.Module | None = None,
    ) -> None:
        super().__init__()
        known = [field.name for field in dataclass_fields(BranchOutput)]
        unknown = [name for name in fields if name not in known]
        if unknown:
            raise ValueError(f"unknown one-to-one fields {unknown}; BranchOutput has {', '.join(known)}")
        if (proto_fusion is None) != (protonet is None):
            raise ValueError("proto_fusion and protonet are given together or not at all")
        self.backbone = backbone
        self.neck = neck
        self.o2o = o2o
        self.fields = fields
        self.proto_fusion = proto_fusion
        self.protonet = protonet

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        """Run the backbone, neck, one-to-one branch and, when held, the prototype stack.

        Args:
            image: Input image batch of shape ``(N, 3, H, W)`` with ``H`` and
                ``W`` divisible by 32.

        Returns:
            The one-to-one branch's ``fields`` in order — dense class logits ``(N, A, num_classes)``, raw ltrb
            distances ``(N, A, 4)``, and whichever of tanh mask coefficients ``(N, A, K)``, raw angles ``(N, A, 1)`` or
            raw point offsets ``(N, A, K, 2)`` were asked for — followed by the raw prototype maps
            ``(N, K, H/4, W/4)`` when the prototype path is held.

        Raises:
            RuntimeError: If a requested field is ``None`` because the branch was built without its stems — a view
                wired to the wrong task, which has to fail loudly rather than hand a decoder a missing tensor.
        """
        features: tuple[Tensor, Tensor, Tensor] = self.neck(self.backbone(image))
        branch: BranchOutput = self.o2o(features)
        outputs: list[Tensor] = []
        for name in self.fields:
            value = getattr(branch, name)
            if value is None:
                raise RuntimeError(f"the one-to-one branch was built without {name!r} stems")
            outputs.append(value)
        if self.proto_fusion is not None and self.protonet is not None:
            outputs.append(self.protonet(self.proto_fusion(features)))
        return tuple(outputs)


class Detector(_SingleHeadModel):
    """Composite YOLO26 detector: backbone -> neck -> dual detection head.

    A single typed module wiring the three WP-020…WP-022 stages for one scale variant. :meth:`forward` takes an image
    batch and returns the head's :class:`~lucid_yolo.models.heads.DualHeadOutput` (both the one-to-one and one-to-many
    dense predictions).

    :meth:`deploy` returns the NMS-free inference model: backbone -> neck -> one-to-one head. The one-to-many branch
    is training-only (R6) and never runs at E2E inference, so the deployed model executes a single detection branch.
    This is the module whose FLOPs match the R1 Table 7 GFLOP column (see the module docstring); its parameters are a
    strict subset of the full detector's, so model size is still reported from the whole :class:`Detector`. The
    returned module **shares** this detector's parameters (no copy), and its forward maps an image batch to the
    one-to-one branch's ``(cls_logits, ltrb)`` tensor pair.

    Args:
        variant: Scale name (``"n"``/``"s"``/``"m"``/``"l"``/``"x"``) resolved through
            :func:`~lucid_yolo.models.registry.scale_spec`, or a :class:`~lucid_yolo.models.registry.ScaleSpec` of raw
            multipliers.
        num_classes: Number of object classes the head predicts.

    Examples:
        ```pycon
        >>> import torch
        >>> model = Detector("n", num_classes=80).eval()
        >>> with torch.no_grad():
        ...     out = model(torch.zeros(1, 3, 640, 640))
        >>> out.o2o_cls.shape, out.o2o_box.shape
        (torch.Size([1, 8400, 80]), torch.Size([1, 8400, 4]))

        ```

        The deployed view runs the one-to-one branch only:

        ```pycon
        >>> deployed = Detector("n").deploy().eval()
        >>> with torch.no_grad():
        ...     cls, box = deployed(torch.zeros(1, 3, 640, 640))
        >>> cls.shape, box.shape
        (torch.Size([1, 8400, 80]), torch.Size([1, 8400, 4]))

        ```
    """

    def __init__(self, variant: str | ScaleSpec, num_classes: int = _DEFAULT_NUM_CLASSES) -> None:
        super().__init__(variant, num_classes)


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
        ```pycon
        >>> model = build_detector("s")
        >>> model.variant, model.num_classes
        ('s', 80)
        >>> model.backbone.width
        0.5

        ```
    """
    return Detector(variant, num_classes)


class OrientedDetector(_SingleHeadModel):
    """Composite YOLO26 oriented detector: the detector plus the A20 angle branch.

    Structurally :class:`Detector` with ``predict_angle=True``: identical backbone, neck, and dual head for a given
    variant, with a third per-level stem on **each** head branch emitting the single orientation scalar of R1
    sec. 3.4.3. The angle is predicted raw (R1 Eq. 13, ``theta_hat = z``); the oriented box is assembled and
    normalized by :func:`~lucid_yolo.models.heads.obb.decode_rboxes` (A23), not by the model. :meth:`forward` returns a
    :class:`~lucid_yolo.models.heads.DualHeadOutput` whose angle fields are populated for both branches.

    The angle stems sit on the head's two branches rather than in a fourth top-level module, which is what makes the
    one-to-many angle stems training-only in the same sense the one-to-many box and class stems already are: they are
    dropped by :meth:`deploy` with the branch that owns them, rather than needing a rule of their own.

    :meth:`deploy` therefore executes a single detection branch — including only that branch's angle stems — and maps
    an image batch to the one-to-one branch's ``(cls_logits, ltrb, angle)`` tensor triple. This is the module whose
    FLOPs the R1 Table S11 gate reads, at the table's 1024-pixel input. The returned module **shares** this
    detector's parameters (no copy).

    Args:
        variant: Scale name (``"n"``/``"s"``/``"m"``/``"l"``/``"x"``) resolved through
            :func:`~lucid_yolo.models.registry.scale_spec`, or a :class:`~lucid_yolo.models.registry.ScaleSpec` of raw
            multipliers.
        num_classes: Number of object classes the head predicts. Defaults to 80 for signature consistency with
            :class:`Detector`; the R1 Table S11 fidelity gate and the DOTA path pass 15 explicitly.

    Examples:
        ```pycon
        >>> import torch
        >>> model = OrientedDetector("n", num_classes=15).eval()
        >>> with torch.no_grad():
        ...     out = model(torch.zeros(1, 3, 128, 128))
        >>> out.o2o_cls.shape, out.o2o_box.shape, out.o2o_angle.shape
        (torch.Size([1, 336, 15]), torch.Size([1, 336, 4]), torch.Size([1, 336, 1]))

        ```

        The deployed view returns the one-to-one angles beside the boxes:

        ```pycon
        >>> deployed = OrientedDetector("n", num_classes=15).deploy().eval()
        >>> with torch.no_grad():
        ...     cls, box, angle = deployed(torch.zeros(1, 3, 128, 128))
        >>> cls.shape, box.shape, angle.shape
        (torch.Size([1, 336, 15]), torch.Size([1, 336, 4]), torch.Size([1, 336, 1]))

        ```
    """

    _DEPLOYED_FIELDS: ClassVar[tuple[str, ...]] = ("cls", "box", "angle")

    def __init__(self, variant: str | ScaleSpec, num_classes: int = _DEFAULT_NUM_CLASSES) -> None:
        super().__init__(variant, num_classes, predict_angle=True)


def build_obb_detector(variant: str, num_classes: int = _DEFAULT_NUM_CLASSES) -> OrientedDetector:
    """Build an :class:`OrientedDetector` for a named scale variant.

    Args:
        variant: Scale name (``"n"``/``"s"``/``"m"``/``"l"``/``"x"``).
        num_classes: Number of object classes. Defaults to 80 to match :func:`build_detector`; DOTA-v1.0 work passes
            ``len(DOTA_CLASSES) == 15``.

    Returns:
        The assembled :class:`OrientedDetector` module.

    Raises:
        KeyError: If ``variant`` is not one of the five published names.

    Examples:
        ```pycon
        >>> model = build_obb_detector("s", num_classes=15)
        >>> model.variant, model.num_classes
        ('s', 15)
        >>> model.head.predict_angle
        True

        ```
    """
    return OrientedDetector(variant, num_classes)


class KeypointDetector(_SingleHeadModel):
    """Composite YOLO26 keypoint detector: the detector plus the R14 point branch.

    Structurally :class:`Detector` with ``num_keypoints=K``: identical backbone, neck, and dual head for a given
    variant, with a third per-level stem on **each** head branch emitting ``K`` raw ``(x, y)`` coordinate offsets and
    ``K`` raw per-axis sigma values. Coordinates are unbounded offsets composed against the anchor centre and stride
    by :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints` (A70), not by the model; sigma stays raw (A65)
    because the RLE flow (R14 Eq. 12) is what interprets it. :meth:`forward` returns a
    :class:`~lucid_yolo.models.heads.DualHeadOutput` whose keypoint and keypoint-sigma fields are populated for both
    branches.

    The point stems sit on the head's two branches rather than in a fourth top-level module, which is what makes the
    one-to-many point stems training-only in the same sense the one-to-many box and class stems already are: they are
    dropped by :meth:`deploy` with the branch that owns them, rather than needing a rule of their own. This mirrors
    :class:`OrientedDetector`'s angle stems.

    :meth:`deploy` therefore executes a single detection branch — including only that branch's point stems — and maps
    an image batch to the one-to-one branch's ``(cls_logits, ltrb, keypoints)`` tensor triple. This is the module the
    keypoint golden's GFLOPs are measured on, at detection's 640-pixel protocol. The returned module **shares** this
    detector's parameters (no copy).

    Sigma is **not** returned by the deployed view. It is consumed by the R14 loss during training and by nothing on
    the inference path: the decode (:func:`~lucid_yolo.models.heads.keypoint.decode_keypoints`, and
    :func:`~lucid_yolo.eval.coco_eval.gather_keypoints` after it) composes the coordinate offsets alone. Its cost is
    still counted, because one stem emits coordinates and sigma from the same convolution — omitting it from the
    return tuple drops a tensor, not a computation.

    The task is generic in ``K``: nothing here knows what a point *means*, so human pose is one instantiation rather
    than the subject.

    Args:
        variant: Scale name (``"n"``/``"s"``/``"m"``/``"l"``/``"x"``) resolved through
            :func:`~lucid_yolo.models.registry.scale_spec`, or a :class:`~lucid_yolo.models.registry.ScaleSpec` of raw
            multipliers.
        num_classes: Number of object classes the head predicts. Defaults to 80 for signature consistency with
            :class:`Detector`; COCO's ``person_keypoints`` path passes 1.
        num_keypoints: Point count ``K`` both head branches predict. Keyword-only and required, with no default:
            unlike ``num_classes`` there is nothing project-wide to fall back on, because ``K`` is a property of the
            *dataset's* annotation schema (COCO person is 17, another pose set is not). A default here would be a
            silent claim about data this model has never seen — the same refusal
            :class:`~lucid_yolo.ptl.module.DetectionLitModule` makes for ``task="keypoints"``.

    Examples:
        ```pycon
        >>> import torch
        >>> model = KeypointDetector("n", num_classes=4, num_keypoints=3).eval()
        >>> with torch.no_grad():
        ...     out = model(torch.zeros(1, 3, 128, 128))
        >>> out.o2o_cls.shape, out.o2o_keypoints.shape
        (torch.Size([1, 336, 4]), torch.Size([1, 336, 3, 2]))
        >>> out.o2o_keypoint_sigma.shape  # raw per-axis uncertainty (A65)
        torch.Size([1, 336, 3, 2])

        ```

        The deployed view returns the one-to-one point offsets, and no sigma:

        ```pycon
        >>> deployed = KeypointDetector("n", num_classes=4, num_keypoints=3).deploy().eval()
        >>> with torch.no_grad():
        ...     cls, box, keypoints = deployed(torch.zeros(1, 3, 128, 128))
        >>> cls.shape, box.shape, keypoints.shape
        (torch.Size([1, 336, 4]), torch.Size([1, 336, 4]), torch.Size([1, 336, 3, 2]))

        ```
    """

    _DEPLOYED_FIELDS: ClassVar[tuple[str, ...]] = ("cls", "box", "keypoints")

    def __init__(
        self, variant: str | ScaleSpec, num_classes: int = _DEFAULT_NUM_CLASSES, *, num_keypoints: int
    ) -> None:
        super().__init__(variant, num_classes, num_keypoints=num_keypoints)
        self.num_keypoints = num_keypoints


def build_keypoint_detector(
    variant: str, num_classes: int = _DEFAULT_NUM_CLASSES, *, num_keypoints: int
) -> KeypointDetector:
    """Build a :class:`KeypointDetector` for a named scale variant.

    Args:
        variant: Scale name (``"n"``/``"s"``/``"m"``/``"l"``/``"x"``).
        num_classes: Number of object classes. Defaults to 80 to match :func:`build_detector`; the COCO
            ``person_keypoints`` path passes 1.
        num_keypoints: Point count ``K``, keyword-only and required — the point count is a property of the annotation
            schema, not something a model may pick (see :class:`KeypointDetector`).

    Returns:
        The assembled :class:`KeypointDetector` module.

    Raises:
        KeyError: If ``variant`` is not one of the five published names.
        TypeError: If ``num_keypoints`` is omitted.

    Examples:
        ```pycon
        >>> model = build_keypoint_detector("s", num_classes=1, num_keypoints=17)
        >>> model.variant, model.num_classes, model.num_keypoints
        ('s', 1, 17)
        >>> model.head.o2o.keypoint_stems is None
        False

        ```
    """
    return KeypointDetector(variant, num_classes, num_keypoints=num_keypoints)


@dataclass(frozen=True)
class SegmentOutput:
    """Dense predictions of the segmentation model's three branches.

    Groups what a segmentation forward pass produces without flattening it: the detection head's own dataclass is
    kept whole (so the dual-branch contract of :class:`~lucid_yolo.models.heads.DualHeadOutput` stays in one place),
    and the two mask-side tensors sit beside it.

    Attributes:
        detect: The full :class:`~lucid_yolo.models.heads.DualHeadOutput`, including both branches' tanh mask
            coefficients.
        prototypes: Raw Eq. 9 prototype maps of shape ``(B, K, 2*H3, 2*W3)``, i.e. twice the P3 grid (A15),
            unactivated.
        semantic: Auxiliary per-class logits ``(B, num_classes, H3, W3)`` in training mode, ``None`` at eval — the
            branch is training-only (A17) and :class:`~lucid_yolo.models.heads.SemanticAux` returns ``None`` outside
            training rather than being skipped by the caller.
    """

    detect: DualHeadOutput
    prototypes: Tensor
    semantic: Tensor | None


class Segmenter(nn.Module):
    """Composite YOLO26 segmentation model: detector plus the mask and aux branches.

    Mirrors :class:`Detector` — same backbone and neck for a given variant — and adds the three WP-047…WP-050
    segmentation stages on top: the detection head now also emits per-anchor mask coefficients,
    :class:`~lucid_yolo.models.heads.ProtoFusion` collapses the neck's three levels into the Eq. 8 fused feature,
    :class:`~lucid_yolo.models.heads.ProtoNet` turns that feature into ``K`` prototype maps, and
    :class:`~lucid_yolo.models.heads.SemanticAux` attaches the training-only auxiliary classifier to the same fused
    feature.

    ``num_coeffs`` is threaded into both the head's coefficient stems and the prototype count, one prototype per
    coefficient: Eq. 7 contracts the two against each other, so a single constructor argument is what stops the two
    sides from drifting apart.

    The fused feature is computed exactly once and shared by the prototype and auxiliary branches — recomputing it
    would silently double that subgraph's cost in the FLOP measurement.

    Args:
        variant: Scale name (``"n"``/``"s"``/``"m"``/``"l"``/``"x"``) resolved through
            :func:`~lucid_yolo.models.registry.scale_spec`, or a :class:`~lucid_yolo.models.registry.ScaleSpec` of raw
            multipliers.
        num_classes: Number of object classes the head and the auxiliary branch predict.
        num_coeffs: Mask-coefficient width ``K`` (A14), also the prototype count.

    Examples:
        ```pycon
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

        ```
    """

    def __init__(
        self,
        variant: str | ScaleSpec,
        num_classes: int = _DEFAULT_NUM_CLASSES,
        num_coeffs: int = DEFAULT_NUM_COEFFS,
    ) -> None:
        super().__init__()
        spec = _resolve_scaling(variant)
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
            A :class:`SegmentOutput` holding the dual head's dense predictions, the raw prototype maps, and the
            auxiliary semantic logits (``None`` outside training mode).
        """
        features: tuple[Tensor, Tensor, Tensor] = self.neck(self.backbone(image))
        detect: DualHeadOutput = self.head(features)
        fused: Tensor = self.proto_fusion(features)
        prototypes: Tensor = self.protonet(fused)
        semantic: Tensor | None = self.semantic(fused)
        return SegmentOutput(detect=detect, prototypes=prototypes, semantic=semantic)

    def deploy(self) -> nn.Module:
        """Return the inference model: backbone -> neck -> one-to-one head + prototypes.

        Two branches of the trained model are training-only and are therefore absent from the returned module: the
        one-to-many detection branch, which never runs on the NMS-free E2E path (R6), and the auxiliary semantic
        branch, which exists only to shape the shared prototype features (A17). Neither is disabled or skipped —
        neither is held at all. The returned module **shares** this segmenter's parameters (no copy), so it is a view
        rather than a second model. This is the module whose parameters and FLOPs the Phase 7 segmentation fidelity
        gate is measured on.

        Returns:
            A :class:`torch.nn.Module` whose forward maps an image batch to the ``(cls, box, coeff, prototypes)``
            tuple needed to assemble masks: dense class logits ``(N, A, num_classes)``, raw ltrb distances
            ``(N, A, 4)``, tanh mask coefficients ``(N, A, K)``, and raw prototype maps ``(N, K, H/4, W/4)``.

        Examples:
            ```pycon
            >>> import torch
            >>> deployed = Segmenter("n", num_classes=4).deploy().eval()
            >>> with torch.no_grad():
            ...     cls, box, coeff, prototypes = deployed(torch.zeros(1, 3, 128, 128))
            >>> cls.shape, box.shape
            (torch.Size([1, 336, 4]), torch.Size([1, 336, 4]))
            >>> coeff.shape, prototypes.shape
            (torch.Size([1, 336, 32]), torch.Size([1, 32, 32, 32]))

            ```
        """
        return _DeployedO2O(
            self.backbone,
            self.neck,
            self.head.o2o,
            ("cls", "box", "coeff"),
            proto_fusion=self.proto_fusion,
            protonet=self.protonet,
        )


def build_segmenter(
    variant: str, num_classes: int = _DEFAULT_NUM_CLASSES, num_coeffs: int = DEFAULT_NUM_COEFFS
) -> Segmenter:
    """Build a :class:`Segmenter` for a named scale variant.

    ``num_coeffs`` is forwarded rather than fixed. :class:`Segmenter` has always accepted it, and a builder that
    dropped it made the class's own third argument unreachable through the function the package exports for building
    one — a caller wanting a narrower coefficient width had to bypass the builder and instantiate the class, which is
    precisely the seam builders exist to remove.

    The default is unchanged, so every model this function built before is the model it builds now, byte-identical
    state dict included; the WP-023 parameter and FLOP goldens read the default path and do not move.

    Args:
        variant: Scale name (``"n"``/``"s"``/``"m"``/``"l"``/``"x"``).
        num_classes: Number of object classes. Defaults to 80 (COCO).
        num_coeffs: Mask-coefficient width ``K`` (A14), shared by the head's coefficient stems and the prototype
            stack — one prototype per coefficient. Defaults to :data:`DEFAULT_NUM_COEFFS`.

    Returns:
        The assembled :class:`Segmenter` module, its head and prototype stack both built at ``num_coeffs``.

    Raises:
        KeyError: If ``variant`` is not one of the five published names.

    Examples:
        ```pycon
        >>> model = build_segmenter("s")
        >>> model.variant, model.num_classes, model.num_coeffs
        ('s', 80, 32)
        >>> model.protonet.num_prototypes  # one prototype per coefficient
        32

        ```

        A narrower coefficient width reaches both halves of Eq. 7 together:

        ```pycon
        >>> narrow = build_segmenter("n", num_classes=4, num_coeffs=16)
        >>> narrow.num_coeffs, narrow.protonet.num_prototypes
        (16, 16)

        ```
    """
    return Segmenter(variant, num_classes, num_coeffs)


def count_params(module: nn.Module) -> int:
    """Count the total number of parameters in a module.

    Args:
        module: Any :class:`torch.nn.Module`.

    Returns:
        The exact total element count summed over ``module.parameters()``.

    Examples:
        ```pycon
        >>> count_params(build_detector("n")) > 2_000_000
        True

        ```
    """
    return sum(param.numel() for param in module.parameters())


def _tensor_fields(output: DualHeadOutput | SegmentOutput) -> tuple[Tensor, ...]:
    """Flatten a dataclass output's tensor fields, in declaration order.

    The field list is **derived** from :func:`dataclasses.fields` rather than written out. Two hand-maintained
    enumerations stood here before, and they had already drifted: the segmentation one listed ten fields where the
    detection one listed twelve, silently dropping the four keypoint fields. Since :func:`build_detection_stages`
    accepts ``num_coeffs`` and ``num_keypoints`` together, a coefficient-and-keypoint model is constructible today,
    and its point stems would have been absent from the traced graph — a FLOP tally quietly missing a whole subgraph.
    Deriving the list makes that class of omission unrepresentable.

    Nested dataclass fields are flattened in place (this is how :class:`SegmentOutput` reaches the
    :class:`~lucid_yolo.models.heads.DualHeadOutput` it holds); the recursion covers the two output dataclasses named
    in the signature rather than any dataclass, so a *third* one nested here later needs adding to both. That is a
    type-level edit a type checker will point at, unlike the per-field omission this replaced. ``None`` fields — an
    optional branch that was not built, or the training-only semantic head at eval (A17) — are dropped. Dropping them
    is safe for a FLOP tally, which reads the traced graph rather than the returned values.

    Args:
        output: A :class:`~lucid_yolo.models.heads.DualHeadOutput` or :class:`SegmentOutput` instance.

    Returns:
        The present tensors, in dataclass declaration order, depth first.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.models.heads import DualHeadOutput
        >>> out = DualHeadOutput(torch.zeros(1), torch.zeros(2), torch.zeros(3), torch.zeros(4))
        >>> [tensor.numel() for tensor in _tensor_fields(out)]  # optional fields dropped
        [1, 2, 3, 4]

        ```
    """
    collected: list[Tensor] = []
    for field in dataclass_fields(output):
        value = getattr(output, field.name)
        if isinstance(value, Tensor):
            collected.append(value)
        elif isinstance(value, DualHeadOutput | SegmentOutput):
            collected.extend(_tensor_fields(value))
    return tuple(collected)


class _TupleOutputAdapter(nn.Module):
    """Wrap a module so its forward returns a tuple, for FLOP tracing.

    :class:`fvcore.nn.FlopCountAnalysis` traces through :func:`torch.jit.trace`, which rejects dataclass outputs like
    :class:`~lucid_yolo.models.heads.DualHeadOutput` and :class:`SegmentOutput`. This adapter unpacks such an output
    to a plain tuple of tensors (leaving already-tensor/tuple outputs untouched) so the trace succeeds; it adds no
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
            The wrapped module's output as a tensor tuple when it is a :class:`~lucid_yolo.models.heads.DualHeadOutput`
            or a :class:`SegmentOutput`, otherwise unchanged. Optional fields that are ``None`` are dropped: the
            coefficient tensors are absent unless the head was built with ``num_coeffs``, the angle tensors unless it
            was built with ``predict_angle``, the keypoint and keypoint-sigma tensors unless it was built with
            ``num_keypoints``, and the auxiliary semantic logits are always ``None`` here because FLOP counting runs
            in eval mode (A17). Dropping them is safe for a FLOP tally, which reads the traced graph rather than the
            returned values.
        """
        output = self.module(image)
        if isinstance(output, DualHeadOutput | SegmentOutput):
            return _tensor_fields(output)
        return cast("tuple[Tensor, ...] | Tensor", output)


def count_flops(module: nn.Module, img_size: int = _DEFAULT_IMG_SIZE) -> float:
    """Count a module's forward GFLOPs at a square input, in units of 1e9.

    Runs :class:`fvcore.nn.FlopCountAnalysis` on a single zero image of side ``img_size``. fvcore tallies
    multiply-accumulate operations (one MAC per count); this returns ``2 x`` that total so the result is
    *conventional* GFLOPs (one multiply + one add = 2 FLOPs), which is what R1 Table 7 reports — the raw MAC total
    lands ~48% low, doubling lands within tolerance (measured convention, WP-023; see the module docstring). fvcore's
    JIT trace rejects the :class:`~lucid_yolo.models.heads.DualHeadOutput` dataclass, so a dataclass-output module is
    wrapped to flatten it; tensor/tuple outputs (e.g. :meth:`Detector.deploy`) pass through untouched.

    For R1 Table 7 fidelity, pass the deployed inference model (``build_detector(v).deploy()``), whose GFLOPs exclude
    the training-only one-to-many branch (see the module docstring); passing the full :class:`Detector` counts both
    branches.

    Counting needs eval mode, and **every** submodule's mode is snapshotted and restored individually rather than the
    module's own flag being toggled back. A ``deploy()`` view is a freshly constructed module holding *shared*
    references to its parent's backbone, neck, and one-to-one branch, so its own ``training`` flag is always ``True``
    no matter what the parent's is. Reading that one flag and calling ``train()`` on the way out therefore pushed
    three quarters of an eval-mode detector into train mode through the shared references — leaving the parent
    reporting ``eval`` while a following forward would use batch statistics and mutate BN running stats, with nothing
    reporting it. Per-module restoration is what makes the counter observably free of side effects on a shared view.

    Args:
        module: A module accepting a ``(1, 3, H, W)`` image (a :class:`Detector`, a :meth:`Detector.deploy` view, or
            any submodule).
        img_size: Square input side in pixels. Defaults to 640.

    Returns:
        The total forward GFLOPs (``2 x`` MACs) divided by 1e9.

    Examples:
        ```pycon
        >>> round(count_flops(build_detector("n").deploy()), 1) > 0
        True

        ```
    """
    modes = {name: submodule.training for name, submodule in module.named_modules()}
    module.eval()
    image = torch.zeros(1, 3, img_size, img_size)
    try:
        with torch.no_grad():
            analysis = FlopCountAnalysis(_TupleOutputAdapter(module), image)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            total = analysis.total()
    finally:
        for name, submodule in module.named_modules():
            submodule.training = modes[name]
    return 2.0 * float(total) / 1e9
