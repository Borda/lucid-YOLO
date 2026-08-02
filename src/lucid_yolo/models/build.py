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

from typing import cast

import torch
from fvcore.nn import FlopCountAnalysis
from torch import Tensor, nn

from lucid_yolo.models.backbone import DetectionBackbone
from lucid_yolo.models.heads import DualDetectionHead, DualHeadOutput
from lucid_yolo.models.neck import DetectionNeck
from lucid_yolo.models.registry import scale_spec

__all__ = ["Detector", "build_detector", "count_flops", "count_params"]

#: Default detection input side in pixels (square), R1 Table 7 protocol.
_DEFAULT_IMG_SIZE = 640

#: Default COCO detection class count.
_DEFAULT_NUM_CLASSES = 80


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
        self.backbone = DetectionBackbone(spec.depth, spec.width, spec.max_channels)
        self.neck = DetectionNeck(self.backbone.channels, spec.depth, spec.width, spec.max_channels)
        self.head = DualDetectionHead(self.neck.channels, num_classes)

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


class _TupleOutputAdapter(nn.Module):
    """Wrap a module so its forward returns a tuple, for FLOP tracing.

    :class:`fvcore.nn.FlopCountAnalysis` traces through :func:`torch.jit.trace`,
    which rejects dataclass outputs like
    :class:`~lucid_yolo.models.heads.DualHeadOutput`. This adapter unpacks such an
    output to a plain tuple of tensors (leaving already-tensor/tuple outputs
    untouched) so the trace succeeds; it adds no compute of its own.

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
            :class:`~lucid_yolo.models.heads.DualHeadOutput`, otherwise unchanged.
        """
        output = self.module(image)
        if isinstance(output, DualHeadOutput):
            return (output.o2m_cls, output.o2m_box, output.o2o_cls, output.o2o_box)
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
