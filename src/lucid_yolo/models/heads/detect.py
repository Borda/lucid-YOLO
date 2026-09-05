# SPDX-License-Identifier: Apache-2.0
"""Dual detection head with direct ltrb regression (WP-022).

The anchor-free detection head of the YOLO26 detector, built by hand from the
method paper's formulation (R1 sec. 3.2.1-3.2.2, Fig. S2) and the dual-branch
lineage of R6. It consumes the neck's three refined feature maps ``(N3, N4,
N5)`` at strides 8/16/32 and predicts, on each of two parallel branches, a class
score map and a box-distance map per level:

- **one-to-one** — the branch supervised so exactly one prediction survives per
  ground-truth object, giving an NMS-free path at inference (R1 3.2.1, R6);
- **one-to-many** — the branch supervised densely (many positives per object) to
  provide the rich gradient signal that trains the shared neck features.

The two branches share the neck features but own **disjoint** prediction stems;
neither reads the other's parameters. Each branch, at each of the three levels,
carries the Fig. S2 stem pair — both **depthwise-separable** for a lightweight
head (the box/cls symmetry is the WP-023 A28 revision; see below):

- **box stem** — two depthwise-separable units
  (``DepthwiseConv(·, ·, 3)`` + ``ConvBNAct(·, ·, 1)``) followed by a 1x1
  convolution to 4 outputs;
- **class stem** — two depthwise-separable units followed by a 1x1 convolution
  to ``num_classes`` outputs.

The stem hidden width is ``max(16, channels // 3)`` for both stems (WP-023
fidelity gate — see :func:`_stem_width`). The box stem was originally two full
``3x3`` convolutions at ``channels // 4``, which made it ~3.7x heavier than the
depthwise-separable class stem despite emitting 4 vs ``num_classes`` channels;
the WP-023 parameter gate flagged the resulting head as ~7-16x the reference
budget, so the box stem was rebuilt depthwise-separable and the class stem's
``max(num_classes, channels // 2)`` floor (which bloated the ``n``/``s`` head)
was dropped. See docs/ASSUMPTIONS.md A28.

**No DFL** (R1 3.2.2): the box 1x1 emits 4 raw scalars per location — the
``ltrb`` distances (left, top, right, bottom) from the anchor centre in stride
units, with unconstrained range (no softmax bins, no range cap). This is the
``reg_max = 1`` degenerate of a distribution-focal box head.

Per branch the flattened per-level maps are concatenated level-by-level in
row-major order into dense predictions ``(B, A, num_classes)`` and ``(B, A, 4)``,
where ``A = 8400`` at a 640-pixel input. That ``A`` ordering matches
:func:`~lucid_yolo.assign.grid.make_anchor_points` position-for-position, so the
anchor point and stride of prediction ``a`` are simply row ``a`` of its outputs.

Three branch outputs are **opt-in** and absent by default, so the accepted
detection-only head keeps its exact module tree, state-dict keys, and parameter
count: the mask coefficients of the segmentation path (WP-047), the orientation
angle of the oriented path (WP-062), and generic ``K``-point coordinates with
per-axis uncertainty (WP-122). ``K`` is a constructor argument like
``num_classes`` and ``num_coeffs``; the head does not encode a human-pose task.
The keypoint stem emits raw coordinate offsets and raw, unbounded sigma values.
No positivity mapping happens anywhere in this module: WP-123's hand-written RLE
loss owns that equation-tied decision (A65).

The angle branch is A20's reading of R1 sec. 3.4.3 — "a separate branch is
adopted to predict the orientation angle" — built as a third per-level stem
sharing the class and coefficient stems' shape but **not** their width
(``channels // 2`` rather than A28's ``channels // 3``;
:func:`_angle_stem_width` records the Table S11 measurement behind the split),
emitting **one scalar per location**. That scalar is the angle itself: R1 Eq. 13
makes ``theta_hat = z`` with no squashing nonlinearity, so
:func:`_build_angle_stem` ends at a raw 1x1 and no activation follows it anywhere
in the head. Decoding and range normalization live in
:mod:`lucid_yolo.models.heads.obb`.

Two pure helpers accompany the module and are reused by the E2E decoder
(WP-041): :func:`decode_ltrb` turns raw distances into ``xyxy`` boxes against an
anchor grid, and :func:`o2o_topk` reduces the one-to-one branch to the
score-ranked ``(B, 300, 6)`` detection tuple ``[x1, y1, x2, y2, score, class]``
(A9). The full NMS-free E2E decode module lands in WP-041 and reuses
:func:`o2o_topk`.

Provenance: R1 sec. 3.2.1, R1 sec. 3.2.2, R1 sec. 3.4.3, R1 Eq. 13, R1 Fig. S2, R6, R14.
Assumptions: A3, A9, A20, A28, A65.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lucid_yolo.models.blocks import ConvBNAct, DepthwiseConv

__all__ = [
    "CLS_PRIOR_PROB",
    "DEFAULT_NUM_COEFFS",
    "BranchOutput",
    "DualDetectionHead",
    "DualHeadOutput",
    "decode_ltrb",
    "init_cls_prior_bias",
    "o2o_topk",
    "o2o_topk_with_indices",
]

#: Number of box regression outputs per anchor (ltrb distances; ``reg_max = 1``).
_BOX_OUTPUTS = 4

#: Number of orientation outputs per anchor: one scalar angle (A20, R1 sec. 3.4.3).
_ANGLE_OUTPUTS = 1

#: Number of raw coordinate outputs per keypoint: x and y offsets.
_KEYPOINT_COORD_OUTPUTS = 2

#: Number of raw uncertainty outputs per keypoint: sigma_x and sigma_y (R14).
_KEYPOINT_SIGMA_OUTPUTS = 2

#: Default mask-coefficient width ``K=32`` from assumption A14; callers opt in
#: explicitly so the accepted detection-only module tree remains unchanged.
DEFAULT_NUM_COEFFS = 32

#: Default per-image detection cap of the one-to-one branch (R3 sec. 4, A9).
_DEFAULT_TOPK = 300

#: Prior probability for the classification-output bias init (RetinaNet sec. 5.1,
#: R24 arXiv:1708.02002: "we set pi = 0.01"; A30). Every class sigmoid starts
#: near this value, so the dense background BCE begins at ~0.01 nats per element
#: instead of ~0.69 — without it the summed classification loss opens six orders
#: of magnitude too large and the first optimizer step destroys the network
#: (observed on the Det-smoke launch: loss 6.2e5 -> collapse to a dead all-zero
#: predictor within three steps).
CLS_PRIOR_PROB = 0.01


def init_cls_prior_bias(conv: nn.Conv2d) -> None:
    """Apply the RetinaNet prior-probability bias init to a dense-classifier 1x1.

    Sets every bias entry to ``-log((1 - pi) / pi)`` with ``pi =``
    :data:`CLS_PRIOR_PROB`, so each output sigmoid starts at ~``pi`` (A30, R24
    sec. 5.1). Shared by every dense sigmoid classifier over a mostly-background
    map — the detection class stems here and the auxiliary semantic branch — so
    the formula exists exactly once and cannot drift between copies.

    Args:
        conv: The output convolution whose bias is initialized. Must have a bias.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> conv = nn.Conv2d(4, 3, 1)
        >>> init_cls_prior_bias(conv)
        >>> round(float(conv.bias.sigmoid()[0]), 4)  # ~pi
        0.01
    """
    assert conv.bias is not None  # nn.Conv2d default; narrows the Optional for mypy
    nn.init.constant_(conv.bias, -math.log((1.0 - CLS_PRIOR_PROB) / CLS_PRIOR_PROB))


def _stem_width(channels: int) -> int:
    """Return the shared hidden width for a prediction stem of ``channels`` inputs.

    The width ``max(16, channels // 3)`` (assumptions A3/A9): a third of the level
    width, floored at 16 so the smallest scales keep a usable stem. Both the box
    and class stems use this one width — the WP-023 parameter/FLOP gate selected
    ``channels // 3`` (from the original box ``channels // 4`` / class
    ``max(num_classes, channels // 2)``) as the value that lands all five scales
    within the R1 Table 7 tolerance. The convention is unpublished; the gate
    validates it, and this is the first knob to turn if that gate misses.

    Args:
        channels: Input channel count of the level.

    Returns:
        The stem hidden channel width.

    Examples:
        >>> _stem_width(64)
        21
        >>> _stem_width(256)
        85
    """
    return max(16, channels // 3)


def _depthwise_separable(in_channels: int, out_channels: int) -> nn.Sequential:
    """Build a depthwise-separable unit: a 3x3 depthwise then a 1x1 pointwise.

    The Fig. S2 stem building block — spatial mixing by the depthwise
    convolution, channel projection by the pointwise :class:`ConvBNAct`. Shared by
    both the box and class stems (assumption A9).

    Args:
        in_channels: Input channel count (also the depthwise group count).
        out_channels: Output channel count of the pointwise projection.

    Returns:
        The depthwise-separable unit mapping ``in_channels`` -> ``out_channels``.

    Examples:
        >>> import torch
        >>> unit = _depthwise_separable(64, 80).eval()
        >>> unit(torch.zeros(1, 64, 8, 8)).shape
        torch.Size([1, 80, 8, 8])
    """
    return nn.Sequential(
        DepthwiseConv(in_channels, in_channels, 3),
        ConvBNAct(in_channels, out_channels, 1),
    )


def _build_box_stem(channels: int) -> nn.Sequential:
    """Build one level's box stem: two depthwise-separable units then a 1x1 to 4.

    The box stem mirrors the class stem's depthwise-separable structure
    (assumption A9, WP-023): two depthwise-separable units project the level width
    to the stem hidden width and keep it, then a 1x1 convolution maps to the 4
    ltrb outputs. This lightweight form replaced the original two full ``3x3``
    convolutions, which the WP-023 parameter gate flagged as far too heavy.

    Args:
        channels: Input channel count of the level.

    Returns:
        The box-regression stem for a single level, emitting ``(B, 4, H, W)``.

    Examples:
        >>> import torch
        >>> stem = _build_box_stem(64).eval()
        >>> stem(torch.zeros(1, 64, 8, 8)).shape
        torch.Size([1, 4, 8, 8])
    """
    hidden = _stem_width(channels)
    return nn.Sequential(
        _depthwise_separable(channels, hidden),
        _depthwise_separable(hidden, hidden),
        nn.Conv2d(hidden, _BOX_OUTPUTS, 1),
    )


def _build_cls_stem(channels: int, num_classes: int) -> nn.Sequential:
    """Build one level's class stem: two depthwise-separable units then a 1x1.

    Per Fig. S2 the stem stacks two depthwise-separable units; the first
    projects the level width to the stem hidden width, the second keeps it,
    then a 1x1 convolution maps to ``num_classes`` score logits.

    Args:
        channels: Input channel count of the level.
        num_classes: Number of object classes.

    Returns:
        The classification stem for a single level, emitting ``(B, num_classes,
        H, W)``.

    The final 1x1 convolution's bias is initialized by
    :func:`init_cls_prior_bias` to ``-log((1 - pi) / pi)`` with ``pi = 0.01``
    (:data:`CLS_PRIOR_PROB`), the
    RetinaNet prior-probability init (R24 sec. 5.1, A30): at the first step every
    class sigmoid evaluates to ~``pi``, keeping the dense background BCE — summed
    over anchors and classes, normalized by the alignment-weight sum (R4) — at a
    trainable magnitude instead of exploding on the first batch.

    Examples:
        >>> import torch
        >>> stem = _build_cls_stem(64, 80).eval()
        >>> stem(torch.zeros(1, 64, 8, 8)).shape
        torch.Size([1, 80, 8, 8])
        >>> round(float(stem(torch.zeros(1, 64, 8, 8)).sigmoid().mean().detach()), 4)  # ~pi
        0.01
    """
    hidden = _stem_width(channels)
    output = nn.Conv2d(hidden, num_classes, 1)
    init_cls_prior_bias(output)
    return nn.Sequential(
        _depthwise_separable(channels, hidden),
        _depthwise_separable(hidden, hidden),
        output,
    )


def _build_coeff_stem(channels: int, num_coeffs: int) -> nn.Sequential:
    """Build one level's mask-coefficient stem without classification bias init.

    The coefficient-stem internals are unspecified by the method paper, so this
    follows the symmetric Fig. S2 class stem selected by assumption A34: two
    depthwise-separable units project and retain the shared hidden width, then a
    1x1 convolution emits ``num_coeffs`` channels. Unlike class logits, these
    outputs are tanh regressands, so the prior-probability bias initialization
    for sigmoid classification has no meaning here.

    Args:
        channels: Input channel count of the level.
        num_coeffs: Number of mask coefficients emitted per anchor.

    Returns:
        The coefficient stem for one level, emitting ``(B, num_coeffs, H, W)``.

    Examples:
        >>> import torch
        >>> stem = _build_coeff_stem(64, 32).eval()
        >>> stem(torch.zeros(1, 64, 8, 8)).shape
        torch.Size([1, 32, 8, 8])
    """
    hidden = _stem_width(channels)
    return nn.Sequential(
        _depthwise_separable(channels, hidden),
        _depthwise_separable(hidden, hidden),
        nn.Conv2d(hidden, num_coeffs, 1),
    )


def _angle_stem_width(channels: int) -> int:
    """Return the hidden width of an orientation stem: ``max(16, channels // 2)``.

    Deliberately **not** :func:`_stem_width`. The box and class stems keep A28's
    ``channels // 3``; only the angle stem is half the level width, and the split
    is what the R1 Table S11 fidelity gate selected.

    Both divisors were measured across all five scales rather than reasoned about,
    because Table S11 rounds its parameter column to 0.1 M and that rounding is
    what decides which scales can discriminate at all. Expressed as the ratio of
    the measured angle-branch cost to the increment Table S11 implies over Table 7
    (after correcting for the 80 -> 15 class saving), with the rounding
    uncertainty that increment carries:

    ==========  ==========  ==========  ============
    variant     ``// 3``    ``// 2``    uncertainty
    ==========  ==========  ==========  ============
    n           0.76        1.22        +/-59%
    s           0.98        1.62        +/-21%
    m           0.66        1.09        +/-8%
    l           0.66        1.09        +/-8%
    x           0.63        1.04        +/-3.5%
    ==========  ==========  ==========  ============

    At ``n`` and ``s`` the published increment is so small that rounding swamps
    the comparison and neither divisor is discriminable. At ``m``, ``l`` and
    ``x`` it is not: ``// 3`` lands at 0.63-0.66 of the published increment, far
    outside the uncertainty, while ``// 2`` lands at 1.04-1.09, inside or beside
    it. ``x`` is the most precise point in the table at +/-3.5% and it selects
    ``// 2`` decisively.

    **Neither divisor is uniformly right**, and that is the honest reading rather
    than a caveat: ``// 3`` undershoots the large scales and ``// 2`` overshoots
    ``s`` (ratio 1.62, though at +/-21% that is much weaker evidence than ``x``).
    The published angle cost therefore does not follow any one fraction of the
    level width, and this is the rule that matches the evidence where the evidence
    discriminates — not a recovery of the paper's structure. Anyone tempted to
    "restore" symmetry with :func:`_stem_width` should re-measure first: doing so
    moves ``x`` from -1.65% to -3.07% against Table S11.

    Args:
        channels: Input channel count of the level.

    Returns:
        The orientation stem's hidden channel width.

    Examples:
        >>> _angle_stem_width(64), _stem_width(64)  # wider than the box/class stems
        (32, 21)
        >>> _angle_stem_width(16)  # floored, as the shared convention floors
        16
    """
    return max(16, channels // 2)


def _build_angle_stem(channels: int) -> nn.Sequential:
    """Build one level's orientation stem: two depthwise-separable units then a 1x1 to 1.

    A20's structure for R1 sec. 3.4.3's "separate branch ... to predict the
    orientation angle": a third stem beside the box and class stems, sharing their
    *shape* (Fig. S2 shows one stem shape, not one per output kind) and ending at
    a 1x1 that emits a **single** scalar per location.

    The shape is shared; the width is not. The hidden width comes from
    :func:`_angle_stem_width` (``channels // 2``), not from A28's
    :func:`_stem_width` (``channels // 3``) — the Table S11 gate selected the
    wider stem, and that function's docstring holds the per-scale measurement and
    the reason a uniform width does not exist.

    Nothing follows that 1x1. R1 Eq. 13 sets ``theta_hat = z`` — the previous
    versions' Eq. 12 squashing, ``theta_hat = (sigmoid(z) - 0.25) * pi``, is
    exactly what YOLO26 removes — so an activation here would reintroduce the
    bounded range the paper deletes. The consequence is that the emitted angle is
    unbounded; :func:`~lucid_yolo.models.heads.obb.decode_rboxes` is where it is
    brought into the canonical range (A23), not here.

    Args:
        channels: Input channel count of the level.

    Returns:
        The orientation stem for one level, emitting ``(B, 1, H, W)``.

    Examples:
        >>> import torch
        >>> stem = _build_angle_stem(64).eval()
        >>> stem(torch.zeros(1, 64, 8, 8)).shape
        torch.Size([1, 1, 8, 8])
        >>> sum(p.numel() for p in stem.parameters())  # 11c + c*h + 2h, twice, + h + 1
        4289
    """
    hidden = _angle_stem_width(channels)
    return nn.Sequential(
        _depthwise_separable(channels, hidden),
        _depthwise_separable(hidden, hidden),
        nn.Conv2d(hidden, _ANGLE_OUTPUTS, 1),
    )


def _build_keypoint_stem(channels: int, num_keypoints: int) -> nn.Sequential:
    """Build one level's raw coordinate-and-uncertainty keypoint stem.

    The stem follows the shared two-depthwise-separable-unit shape used by the
    coefficient stem, with :func:`_stem_width` providing A28's
    ``max(16, channels // 3)`` hidden width. Unlike the angle task, no R1 Table
    S11 or other allowlisted measurement covers a keypoint task — R1 does not
    cover one at all, and R14 is registered for the future pose milestone only —
    so inventing a keypoint-specific width would be unmeasured. Reusing the
    shared width matches :func:`_build_coeff_stem`'s precedent.

    The final 1x1 emits four raw channels per point in point-major order: for
    zero-based point ``i``, channels ``4*i`` through ``4*i+3`` are
    ``(x_offset, y_offset, sigma_x, sigma_y)``. Nothing follows the convolution.
    Coordinates are unbounded offsets like the angle, and sigma is equally raw
    and unbounded here; WP-123's RLE loss owns any positivity mapping (A65).

    Args:
        channels: Input channel count of the level.
        num_keypoints: Number of generic points emitted per anchor.

    Returns:
        The keypoint stem for one level, emitting ``(B, 4 * num_keypoints, H,
        W)``.

    Examples:
        >>> import torch
        >>> stem = _build_keypoint_stem(64, 3).eval()
        >>> stem(torch.zeros(1, 64, 8, 8)).shape
        torch.Size([1, 12, 8, 8])
    """
    hidden = _stem_width(channels)
    outputs = (_KEYPOINT_COORD_OUTPUTS + _KEYPOINT_SIGMA_OUTPUTS) * num_keypoints
    return nn.Sequential(
        _depthwise_separable(channels, hidden),
        _depthwise_separable(hidden, hidden),
        nn.Conv2d(hidden, outputs, 1),
    )


def _flatten_level(feature_map: Tensor) -> Tensor:
    """Flatten a ``(B, C, H, W)`` prediction map to ``(B, H * W, C)`` row-major.

    The spatial axes flatten in row-major order (``H`` then ``W``), matching
    :func:`~lucid_yolo.assign.grid.make_anchor_points` so anchor row ``a`` owns
    prediction row ``a``.

    Args:
        feature_map: A prediction map of shape ``(B, C, H, W)``.

    Returns:
        The tensor reshaped to ``(B, H * W, C)``.

    Examples:
        >>> import torch
        >>> _flatten_level(torch.zeros(2, 4, 8, 8)).shape
        torch.Size([2, 64, 4])
    """
    return feature_map.flatten(2).transpose(1, 2)


@dataclass(frozen=True)
class BranchOutput:
    """Dense predictions of a single detection branch.

    One branch's half of :class:`DualHeadOutput`, with the branch prefix dropped:
    :class:`DualDetectionHead` builds its twelve fields from two of these. The
    per-field contract is the same one :class:`DualHeadOutput` documents — raw
    class logits, raw ltrb distances in stride units, tanh-activated
    coefficients, unactivated angles, and raw unbounded keypoint coordinates and
    sigma.

    It is a dataclass rather than the tuple this used to be because the tuple
    fixed only its *width*. Six positional slots, four of them optional and all
    but two typed ``Tensor | None``, were destructured at five call sites, four
    of them with underscore placeholders that made the shape of what was skipped
    invisible. Inserting or reordering an output silently rebound every one of
    them, and nothing — not the annotations, not the type checker — would have
    said so. Naming the fields is what makes that class of change loud, and it is
    the same reasoning that made the enclosing :class:`DualHeadOutput` a
    dataclass one layer up.

    Attributes:
        cls: Class logits, shape ``(B, A, num_classes)``, raw.
        box: Raw ltrb distances, shape ``(B, A, 4)``, in stride units.
        coeff: Tanh mask coefficients, shape ``(B, A, num_coeffs)``, or ``None``
            when the branch was built without coefficient stems.
        angle: Raw orientation angles in radians, shape ``(B, A, 1)``, or
            ``None`` when the branch was built without orientation stems.
        keypoints: Raw point-coordinate offsets, shape ``(B, A, K, 2)``, or
            ``None`` when the branch was built without keypoint stems.
        keypoint_sigma: Raw, unbounded per-axis uncertainty, shape
            ``(B, A, K, 2)``, or ``None`` when the branch was built without
            keypoint stems.
    """

    cls: Tensor
    box: Tensor
    coeff: Tensor | None = None
    angle: Tensor | None = None
    keypoints: Tensor | None = None
    keypoint_sigma: Tensor | None = None


class _DetectionBranch(nn.Module):
    """One prediction branch: per-level box/class stems plus optional extra stems.

    Owns three box stems and three class stems (one pair per level, strides
    8/16/32). :class:`DualDetectionHead` holds two of these — the one-to-one and
    one-to-many branches — with disjoint parameters. Forward flattens and
    concatenates the per-level maps into dense ``(B, A, num_classes)`` scores and
    ``(B, A, 4)`` raw ltrb distances. Three further stem sets are built only when
    explicitly requested: mask-coefficient stems emitting tanh-bounded
    ``(B, A, num_coeffs)`` vectors (WP-047), orientation stems emitting the raw
    ``(B, A, 1)`` angle of A20 (WP-062), and keypoint stems emitting raw
    coordinates and uncertainty as ``(B, A, K, 2)`` tensors (WP-122).

    Optional stems are constructed **after** the box and class stems, and a new
    optional stem is constructed after any other already-enabled optional stem.
    A previously accepted configuration therefore draws exactly the random
    numbers it drew before: enabling a later stem set must not perturb the
    initialization of parameters that already existed.

    Args:
        in_channels: Per-level input channel counts ``(N3, N4, N5)`` in stride
            order (8, 16, 32).
        num_classes: Number of object classes.
        num_coeffs: Optional mask-coefficient count. ``None`` leaves the module
            tree and prediction computation detection-only.
        predict_angle: Build the orientation stems (A20). ``False`` leaves the
            module tree and prediction computation angle-free.
        num_keypoints: Optional generic point count. ``None`` leaves the module
            tree and prediction computation keypoint-free.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        num_classes: int,
        num_coeffs: int | None = None,
        predict_angle: bool = False,
        num_keypoints: int | None = None,
    ) -> None:
        super().__init__()
        self.coeff_stems: nn.ModuleList | None = None
        self.angle_stems: nn.ModuleList | None = None
        self.keypoint_stems: nn.ModuleList | None = None
        self.box_stems = nn.ModuleList(_build_box_stem(channels) for channels in in_channels)
        self.cls_stems = nn.ModuleList(_build_cls_stem(channels, num_classes) for channels in in_channels)
        if num_coeffs is not None:
            self.coeff_stems = nn.ModuleList(_build_coeff_stem(channels, num_coeffs) for channels in in_channels)
        if predict_angle:
            self.angle_stems = nn.ModuleList(_build_angle_stem(channels) for channels in in_channels)
        if num_keypoints is not None:
            self.keypoint_stems = nn.ModuleList(
                _build_keypoint_stem(channels, num_keypoints) for channels in in_channels
            )

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> BranchOutput:
        """Predict dense scores, ltrb distances, and whichever optional outputs exist.

        The result carries a **fixed** set of named fields whatever the branch
        was built with, the disabled outputs coming back as ``None``. A shape
        that varied with the enabled stem sets would make a three-value result
        ambiguous between coefficients and angles, which is precisely the kind
        of confusion that pairs one output with another's consumer — and the
        names close the half of that hole a fixed-width tuple left open, where
        the *order* was still positional at every call site.

        Args:
            features: The neck maps ``(n3, n4, n5)`` at strides 8, 16, and 32.

        Returns:
            A :class:`BranchOutput` whose ``cls`` is raw class logits ``(B, A,
            num_classes)``, ``box`` raw ltrb distances ``(B, A, 4)``, ``coeff``
            tanh-bounded ``(B, A, num_coeffs)`` or ``None``, ``angle`` the raw
            ``(B, A, 1)`` orientation of R1 Eq. 13 or ``None``, and whose last
            two fields are raw ``(B, A, K, 2)`` coordinate offsets and unbounded
            uncertainty or ``None``. ``A`` sums ``H * W`` over levels.
        """
        cls_levels: list[Tensor] = []
        box_levels: list[Tensor] = []
        for feature, box_stem, cls_stem in zip(features, self.box_stems, self.cls_stems, strict=True):
            box_levels.append(_flatten_level(box_stem(feature)))
            cls_levels.append(_flatten_level(cls_stem(feature)))
        cls = torch.cat(cls_levels, dim=1)
        box = torch.cat(box_levels, dim=1)
        keypoints, keypoint_sigma = self._keypoints(features)
        return BranchOutput(
            cls=cls,
            box=box,
            coeff=self._coefficients(features),
            angle=self._angles(features),
            keypoints=keypoints,
            keypoint_sigma=keypoint_sigma,
        )

    def _coefficients(self, features: tuple[Tensor, Tensor, Tensor]) -> Tensor | None:
        """Run the coefficient stems, tanh-activated, or return ``None`` if absent.

        Args:
            features: The neck maps ``(n3, n4, n5)`` at strides 8, 16, and 32.

        Returns:
            Dense tanh coefficients ``(B, A, num_coeffs)``, or ``None`` when the
            branch was built without coefficient stems.
        """
        if self.coeff_stems is None:
            return None
        levels = [
            _flatten_level(torch.tanh(coeff_stem(feature)))
            for feature, coeff_stem in zip(features, self.coeff_stems, strict=True)
        ]
        return torch.cat(levels, dim=1)

    def _angles(self, features: tuple[Tensor, Tensor, Tensor]) -> Tensor | None:
        """Run the orientation stems unactivated, or return ``None`` if absent.

        No activation is applied: R1 Eq. 13 predicts ``theta_hat = z`` directly,
        deleting the Eq. 12 sigmoid squashing of the previous versions. The
        unbounded scalar is normalized at decode time
        (:func:`~lucid_yolo.models.heads.obb.decode_rboxes`, A23), never here.

        Args:
            features: The neck maps ``(n3, n4, n5)`` at strides 8, 16, and 32.

        Returns:
            Dense raw angles ``(B, A, 1)`` in radians, or ``None`` when the branch
            was built without orientation stems.
        """
        if self.angle_stems is None:
            return None
        levels = [
            _flatten_level(angle_stem(feature)) for feature, angle_stem in zip(features, self.angle_stems, strict=True)
        ]
        return torch.cat(levels, dim=1)

    def _keypoints(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor | None, Tensor | None]:
        """Run the keypoint stems unactivated, or return ``(None, None)`` if absent.

        Each flattened level follows the stem's point-major channel convention:
        ``(x_offset, y_offset, sigma_x, sigma_y)`` per point. Reshaping makes the
        point axis explicit before the first two and last two values are split.
        Neither side receives an activation; in particular sigma remains raw and
        unbounded until WP-123's RLE loss decides its positivity mapping (A65).

        Args:
            features: The neck maps ``(n3, n4, n5)`` at strides 8, 16, and 32.

        Returns:
            Dense raw coordinate offsets and sigma, each ``(B, A, K, 2)``, or
            ``(None, None)`` when the branch was built without keypoint stems.
        """
        if self.keypoint_stems is None:
            return None, None
        levels: list[Tensor] = []
        point_outputs = _KEYPOINT_COORD_OUTPUTS + _KEYPOINT_SIGMA_OUTPUTS
        for feature, keypoint_stem in zip(features, self.keypoint_stems, strict=True):
            flat = _flatten_level(keypoint_stem(feature))
            levels.append(flat.reshape(flat.shape[0], flat.shape[1], -1, point_outputs))
        points = torch.cat(levels, dim=1)
        return points[..., :_KEYPOINT_COORD_OUTPUTS], points[..., _KEYPOINT_COORD_OUTPUTS:]


@dataclass(frozen=True)
class DualHeadOutput:
    """Dense predictions of both detection branches.

    Every field is a dense per-anchor tensor with the ``A`` axis ordered
    level-by-level in row-major order (matching
    :func:`~lucid_yolo.assign.grid.make_anchor_points`). Class fields are raw
    logits; box fields are raw ltrb distances in stride units (decode them with
    :func:`decode_ltrb`). Coefficient fields, when enabled, are already
    tanh-activated in the head and lie in ``[-1, 1]``. This deliberate asymmetry
    from the raw class and box outputs keeps the activation with its regression
    head; a later mask loss must not move it. Angle fields go the other way and
    are deliberately **unactivated and unbounded**: R1 Eq. 13 predicts
    ``theta_hat = z``, so any range normalization belongs to the decode
    (:func:`~lucid_yolo.models.heads.obb.decode_rboxes`, A23), not here.
    Keypoint-coordinate fields are likewise raw offsets. Keypoint-sigma fields
    are raw and unbounded: no range or positivity normalization happens anywhere
    in this head, and WP-123's RLE loss owns that mapping (A65).

    Attributes:
        o2m_cls: One-to-many class logits, shape ``(B, A, num_classes)``.
        o2m_box: One-to-many raw ltrb distances, shape ``(B, A, 4)``.
        o2o_cls: One-to-one class logits, shape ``(B, A, num_classes)``.
        o2o_box: One-to-one raw ltrb distances, shape ``(B, A, 4)``.
        o2m_coeff: One-to-many tanh mask coefficients, shape ``(B, A, K)``, or
            ``None`` when coefficients are disabled.
        o2o_coeff: One-to-one tanh mask coefficients, shape ``(B, A, K)``, or
            ``None`` when coefficients are disabled.
        o2m_angle: One-to-many raw orientation angles in radians, shape
            ``(B, A, 1)``, or ``None`` when the angle branch is disabled.
        o2o_angle: One-to-one raw orientation angles in radians, shape
            ``(B, A, 1)``, or ``None`` when the angle branch is disabled.
        o2m_keypoints: One-to-many raw point-coordinate offsets, shape ``(B, A,
            K, 2)``, or ``None`` when keypoints are disabled.
        o2o_keypoints: One-to-one raw point-coordinate offsets, shape ``(B, A,
            K, 2)``, or ``None`` when keypoints are disabled.
        o2m_keypoint_sigma: One-to-many raw, unbounded per-axis uncertainty,
            shape ``(B, A, K, 2)``, or ``None`` when keypoints are disabled.
            WP-123's RLE loss owns the positivity mapping.
        o2o_keypoint_sigma: One-to-one raw, unbounded per-axis uncertainty,
            shape ``(B, A, K, 2)``, or ``None`` when keypoints are disabled.
            WP-123's RLE loss owns the positivity mapping.
    """

    o2m_cls: Tensor
    o2m_box: Tensor
    o2o_cls: Tensor
    o2o_box: Tensor
    o2m_coeff: Tensor | None = None
    o2o_coeff: Tensor | None = None
    o2m_angle: Tensor | None = None
    o2o_angle: Tensor | None = None
    o2m_keypoints: Tensor | None = None
    o2o_keypoints: Tensor | None = None
    o2m_keypoint_sigma: Tensor | None = None
    o2o_keypoint_sigma: Tensor | None = None


class DualDetectionHead(nn.Module):
    """Anchor-free dual detection head with direct ltrb regression (reg_max=1).

    Builds the R1 sec. 3.2.1-3.2.2 / Fig. S2 head: two branches
    (:attr:`o2o` for the NMS-free one-to-one path, :attr:`o2m` for dense
    one-to-many training supervision) share the neck features ``(N3, N4, N5)``
    and each own per-level box and class stems over strides 8/16/32. Boxes are
    predicted as 4 raw ltrb scalars per anchor with no DFL (``reg_max = 1``).

    Args:
        in_channels: Neck output channel counts ``(N3, N4, N5)`` in stride order
            (8, 16, 32) — typically :attr:`DetectionNeck.channels`.
        num_classes: Number of object classes.
        num_coeffs: Optional mask-coefficient count. ``None`` preserves the
            detection-only head exactly; callers explicitly pass
            :data:`DEFAULT_NUM_COEFFS` to enable coefficients.
        predict_angle: Build both branches' orientation stems (A20). ``False``
            preserves the detection-only head exactly; the oriented model passes
            ``True``.
        num_keypoints: Optional generic point count. ``None`` preserves the
            existing head exactly; passing ``K`` builds coordinate-and-sigma
            stems on both branches.

    Examples:
        >>> import torch
        >>> head = DualDetectionHead(in_channels=(64, 128, 256), num_classes=80).eval()
        >>> feats = (torch.zeros(1, 64, 80, 80), torch.zeros(1, 128, 40, 40), torch.zeros(1, 256, 20, 20))
        >>> with torch.no_grad():
        ...     out = head(feats)
        >>> out.o2o_cls.shape, out.o2o_box.shape
        (torch.Size([1, 8400, 80]), torch.Size([1, 8400, 4]))
        >>> out.o2o_angle is None  # the angle branch is opt-in
        True
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        num_classes: int,
        num_coeffs: int | None = None,
        predict_angle: bool = False,
        num_keypoints: int | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_coeffs = num_coeffs
        self.predict_angle = predict_angle
        self.num_keypoints = num_keypoints
        self.o2o = _DetectionBranch(
            in_channels,
            num_classes,
            num_coeffs=num_coeffs,
            predict_angle=predict_angle,
            num_keypoints=num_keypoints,
        )
        self.o2m = _DetectionBranch(
            in_channels,
            num_classes,
            num_coeffs=num_coeffs,
            predict_angle=predict_angle,
            num_keypoints=num_keypoints,
        )

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> DualHeadOutput:
        """Run both branches over the neck features.

        Args:
            features: The neck maps ``(n3, n4, n5)`` at strides 8, 16, and 32,
                with channel counts matching the constructor's ``in_channels``.

        Returns:
            A :class:`DualHeadOutput` with dense class logits, raw ltrb
            distances, and — for whichever optional stems were built — tanh mask
            coefficients, raw orientation angles, and raw point-coordinate and
            uncertainty tensors, for both branches.
        """
        o2m: BranchOutput = self.o2m(features)
        o2o: BranchOutput = self.o2o(features)
        return DualHeadOutput(
            o2m_cls=o2m.cls,
            o2m_box=o2m.box,
            o2o_cls=o2o.cls,
            o2o_box=o2o.box,
            o2m_coeff=o2m.coeff,
            o2o_coeff=o2o.coeff,
            o2m_angle=o2m.angle,
            o2o_angle=o2o.angle,
            o2m_keypoints=o2m.keypoints,
            o2o_keypoints=o2o.keypoints,
            o2m_keypoint_sigma=o2m.keypoint_sigma,
            o2o_keypoint_sigma=o2o.keypoint_sigma,
        )


def decode_ltrb(distances: Tensor, anchor_points: Tensor, strides: Tensor) -> Tensor:
    """Decode raw ltrb distances into ``xyxy`` boxes against an anchor grid.

    Each distance quadruple ``(l, t, r, b)`` gives the left/top/right/bottom
    offsets from its anchor centre in stride units; the box corners are the
    anchor centre displaced by those offsets scaled to input pixels::

        x1 = ax - l * s    y1 = ay - t * s
        x2 = ax + r * s    y2 = ay + b * s

    Corner order is ``[x1, y1, x2, y2]`` (A9). Pure function — shared by both
    branches and by the WP-041 E2E decoder.

    Args:
        distances: Raw ltrb distances of shape ``(B, A, 4)`` (order l, t, r, b).
        anchor_points: Anchor-centre ``(x, y)`` coordinates of shape ``(A, 2)``
            in input pixels, as returned by
            :func:`~lucid_yolo.assign.grid.make_anchor_points`.
        strides: Per-anchor level stride of shape ``(A,)``.

    Returns:
        Boxes of shape ``(B, A, 4)`` in ``xyxy`` input-pixel coordinates.

    Examples:
        >>> import torch
        >>> distances = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (1, 1, 4)
        >>> anchor_points = torch.tensor([[10.0, 10.0]])  # (1, 2)
        >>> strides = torch.tensor([2.0])  # (1,)
        >>> decode_ltrb(distances, anchor_points, strides)
        tensor([[[ 8.,  6., 16., 18.]]])
    """
    centre_x = anchor_points[:, 0]
    centre_y = anchor_points[:, 1]
    left, top, right, bottom = distances.unbind(dim=-1)
    x1 = centre_x - left * strides
    y1 = centre_y - top * strides
    x2 = centre_x + right * strides
    y2 = centre_y + bottom * strides
    return torch.stack((x1, y1, x2, y2), dim=-1)


def o2o_topk_with_indices(scores: Tensor, boxes: Tensor, k: int = _DEFAULT_TOPK) -> tuple[Tensor, Tensor]:
    """Reduce the one-to-one branch to the top-k detections *and* the anchors kept.

    The selection itself is :func:`o2o_topk`'s — this is the one place it is
    written. The extra return is the anchor index each kept detection came from,
    which any per-anchor quantity that is **not** part of the A9 tuple must be
    gathered by: the segmentation decode reads its mask coefficients from the
    dense ``(B, A, K)`` coefficient map, and a coefficient row paired with the
    wrong anchor's box yields a perfectly plausible mask of the wrong object.
    Ranking the scores a second time in a separate helper is exactly how such a
    pairing goes silently wrong, so :func:`o2o_topk` is a wrapper over this
    function rather than a second copy.

    Args:
        scores: Raw class logits of shape ``(B, A, C)``.
        boxes: Decoded ``xyxy`` boxes of shape ``(B, A, 4)``, aligned with
            ``scores`` on the anchor axis.
        k: Maximum detections kept per image. Defaults to 300.

    Returns:
        A pair ``(detections, anchor_index)``. ``detections`` is the A9 tuple
        batch of shape ``(B, min(k, A), 6)``; ``anchor_index`` is the ``(B,
        min(k, A))`` long tensor of source anchor rows, score-descending like the
        detections themselves.

    Examples:
        >>> import torch
        >>> scores = torch.tensor([[[2.0, -1.0], [-3.0, 0.5]]])  # (1, 2, 2)
        >>> boxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 2.0, 2.0]]])
        >>> det, anchor_index = o2o_topk_with_indices(scores, boxes, k=1)
        >>> anchor_index  # anchor 0 scores sigmoid(2.0), anchor 1 only sigmoid(0.5)
        tensor([[0]])
        >>> torch.equal(det[..., :4], boxes[:, anchor_index[0]])
        True
    """
    _, num_anchors, _ = scores.shape
    confidence = scores.sigmoid()
    max_conf, class_index = confidence.max(dim=-1)  # both (B, A)
    keep = min(k, num_anchors)
    top_conf, top_anchor = max_conf.topk(keep, dim=1)  # both (B, keep)
    gather_box = top_anchor.unsqueeze(-1).expand(-1, -1, _BOX_OUTPUTS)
    top_boxes = boxes.gather(1, gather_box)  # (B, keep, 4)
    top_class = class_index.gather(1, top_anchor).to(scores.dtype)  # (B, keep)
    detections = torch.cat((top_boxes, top_conf.unsqueeze(-1), top_class.unsqueeze(-1)), dim=-1)
    return detections, top_anchor


def o2o_topk(scores: Tensor, boxes: Tensor, k: int = _DEFAULT_TOPK) -> Tensor:
    """Reduce the one-to-one branch to the top-k score-ranked detections.

    Convenience for the NMS-free path: score each anchor by its maximum class
    confidence (after a sigmoid), keep the ``k`` highest-scoring anchors per
    image, and emit the A9 detection tuple ``[x1, y1, x2, y2, score, class]``.
    The full E2E decode module lands in WP-041 and reuses this helper. When the
    anchor count ``A`` is below ``k`` every anchor is returned.

    Thin wrapper over :func:`o2o_topk_with_indices`, which owns the selection and
    additionally reports which anchors were kept; callers needing only the
    detection tuple use this name.

    Args:
        scores: Raw class logits of shape ``(B, A, C)``.
        boxes: Decoded ``xyxy`` boxes of shape ``(B, A, 4)``, aligned with
            ``scores`` on the anchor axis.
        k: Maximum detections kept per image. Defaults to 300.

    Returns:
        A tensor of shape ``(B, min(k, A), 6)`` whose last axis is
        ``[x1, y1, x2, y2, score, class]``; ``score`` lies in ``[0, 1]`` and
        ``class`` holds the integral class index as a float.

    Examples:
        >>> import torch
        >>> scores = torch.tensor([[[2.0, -1.0], [-3.0, 0.5]]])  # (1, 2, 2)
        >>> boxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 2.0, 2.0]]])
        >>> det = o2o_topk(scores, boxes, k=1)
        >>> det.shape
        torch.Size([1, 1, 6])
        >>> det[0, 0, 5]  # class index of the top detection
        tensor(0.)
    """
    return o2o_topk_with_indices(scores, boxes, k)[0]
