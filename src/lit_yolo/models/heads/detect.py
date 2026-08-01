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
carries the Fig. S2 stem pair:

- **box stem** — ``ConvBNAct(c, c_box, 3)`` -> ``ConvBNAct(c_box, c_box, 3)`` ->
  a 1x1 convolution to 4 outputs;
- **class stem** — two depthwise-separable units
  (``DepthwiseConv(·, ·, 3)`` + ``ConvBNAct(·, ·, 1)``) followed by a 1x1
  convolution to ``num_classes`` outputs.

**No DFL** (R1 3.2.2): the box 1x1 emits 4 raw scalars per location — the
``ltrb`` distances (left, top, right, bottom) from the anchor centre in stride
units, with unconstrained range (no softmax bins, no range cap). This is the
``reg_max = 1`` degenerate of a distribution-focal box head.

Per branch the flattened per-level maps are concatenated level-by-level in
row-major order into dense predictions ``(B, A, num_classes)`` and ``(B, A, 4)``,
where ``A = 8400`` at a 640-pixel input. That ``A`` ordering matches
:func:`~lit_yolo.assign.grid.make_anchor_points` position-for-position, so the
anchor point and stride of prediction ``a`` are simply row ``a`` of its outputs.

Two pure helpers accompany the module and are reused by the E2E decoder
(WP-041): :func:`decode_ltrb` turns raw distances into ``xyxy`` boxes against an
anchor grid, and :func:`o2o_topk` reduces the one-to-one branch to the
score-ranked ``(B, 300, 6)`` detection tuple ``[x1, y1, x2, y2, score, class]``
(A9). The full NMS-free E2E decode module lands in WP-041 and reuses
:func:`o2o_topk`.

Provenance: R1 sec. 3.2.1, R1 sec. 3.2.2, R1 Fig. S2, R6. Assumptions: A3, A9.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lit_yolo.models.blocks import ConvBNAct, DepthwiseConv

__all__ = ["DualDetectionHead", "DualHeadOutput", "decode_ltrb", "o2o_topk"]

#: Number of box regression outputs per anchor (ltrb distances; ``reg_max = 1``).
_BOX_OUTPUTS = 4

#: Default per-image detection cap of the one-to-one branch (R3 sec. 4, A9).
_DEFAULT_TOPK = 300


def _box_stem_width(channels: int) -> int:
    """Return the box-stem hidden width for a level of ``channels`` inputs.

    The width ``max(16, channels // 4)`` (assumptions A3/A9): a quarter of the
    level width, floored at 16 so the smallest scales keep a usable box stem. The
    convention is unpublished — the WP-023 parameter/FLOP gate validates it, and
    this is the first knob to turn if that gate misses.

    Args:
        channels: Input channel count of the level.

    Returns:
        The box-stem hidden channel width.

    Examples:
        >>> _box_stem_width(64)
        16
        >>> _box_stem_width(256)
        64
    """
    return max(16, channels // 4)


def _cls_stem_width(channels: int, num_classes: int) -> int:
    """Return the class-stem hidden width for a level of ``channels`` inputs.

    The width ``max(num_classes, channels // 2)`` (assumptions A3/A9): half the
    level width, floored at ``num_classes`` so the stem never narrows below the
    number of scores it must emit. Unpublished convention — validated by the
    WP-023 parameter/FLOP gate.

    Args:
        channels: Input channel count of the level.
        num_classes: Number of object classes.

    Returns:
        The class-stem hidden channel width.

    Examples:
        >>> _cls_stem_width(64, 80)
        80
        >>> _cls_stem_width(512, 80)
        256
    """
    return max(num_classes, channels // 2)


def _build_box_stem(channels: int) -> nn.Sequential:
    """Build one level's box stem: two 3x3 units then a 1x1 to 4 ltrb outputs.

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
    hidden = _box_stem_width(channels)
    return nn.Sequential(
        ConvBNAct(channels, hidden, 3),
        ConvBNAct(hidden, hidden, 3),
        nn.Conv2d(hidden, _BOX_OUTPUTS, 1),
    )


def _depthwise_separable(in_channels: int, out_channels: int) -> nn.Sequential:
    """Build a depthwise-separable unit: a 3x3 depthwise then a 1x1 pointwise.

    The Fig. S2 class-stem building block — spatial mixing by the depthwise
    convolution, channel projection by the pointwise :class:`ConvBNAct`.

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


def _build_cls_stem(channels: int, num_classes: int) -> nn.Sequential:
    """Build one level's class stem: two depthwise-separable units then a 1x1.

    Per Fig. S2 the stem stacks two depthwise-separable units; the first
    projects the level width to the class-stem hidden width, the second keeps it,
    then a 1x1 convolution maps to ``num_classes`` score logits.

    Args:
        channels: Input channel count of the level.
        num_classes: Number of object classes.

    Returns:
        The classification stem for a single level, emitting ``(B, num_classes,
        H, W)``.

    Examples:
        >>> import torch
        >>> stem = _build_cls_stem(64, 80).eval()
        >>> stem(torch.zeros(1, 64, 8, 8)).shape
        torch.Size([1, 80, 8, 8])
    """
    hidden = _cls_stem_width(channels, num_classes)
    return nn.Sequential(
        _depthwise_separable(channels, hidden),
        _depthwise_separable(hidden, hidden),
        nn.Conv2d(hidden, num_classes, 1),
    )


def _flatten_level(feature_map: Tensor) -> Tensor:
    """Flatten a ``(B, C, H, W)`` prediction map to ``(B, H * W, C)`` row-major.

    The spatial axes flatten in row-major order (``H`` then ``W``), matching
    :func:`~lit_yolo.assign.grid.make_anchor_points` so anchor row ``a`` owns
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


class _DetectionBranch(nn.Module):
    """One prediction branch: per-level box and class stems over three levels.

    Owns three box stems and three class stems (one pair per level, strides
    8/16/32). :class:`DualDetectionHead` holds two of these — the one-to-one and
    one-to-many branches — with disjoint parameters. Forward flattens and
    concatenates the per-level maps into dense ``(B, A, num_classes)`` scores and
    ``(B, A, 4)`` raw ltrb distances.

    Args:
        in_channels: Per-level input channel counts ``(N3, N4, N5)`` in stride
            order (8, 16, 32).
        num_classes: Number of object classes.
    """

    def __init__(self, in_channels: tuple[int, int, int], num_classes: int) -> None:
        super().__init__()
        self.box_stems = nn.ModuleList(_build_box_stem(channels) for channels in in_channels)
        self.cls_stems = nn.ModuleList(_build_cls_stem(channels, num_classes) for channels in in_channels)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Predict dense scores and raw ltrb distances across the three levels.

        Args:
            features: The neck maps ``(n3, n4, n5)`` at strides 8, 16, and 32.

        Returns:
            A pair ``(cls, box)`` with ``cls`` of shape ``(B, A, num_classes)``
            (raw class logits) and ``box`` of shape ``(B, A, 4)`` (raw ltrb
            distances), where ``A`` sums ``H * W`` over the three levels.
        """
        cls_levels: list[Tensor] = []
        box_levels: list[Tensor] = []
        for feature, box_stem, cls_stem in zip(features, self.box_stems, self.cls_stems, strict=True):
            box_levels.append(_flatten_level(box_stem(feature)))
            cls_levels.append(_flatten_level(cls_stem(feature)))
        return torch.cat(cls_levels, dim=1), torch.cat(box_levels, dim=1)


@dataclass(frozen=True)
class DualHeadOutput:
    """Dense predictions of both detection branches.

    Every field is a dense per-anchor tensor with the ``A`` axis ordered
    level-by-level in row-major order (matching
    :func:`~lit_yolo.assign.grid.make_anchor_points`). Class fields are raw
    logits; box fields are raw ltrb distances in stride units (decode them with
    :func:`decode_ltrb`).

    Attributes:
        o2m_cls: One-to-many class logits, shape ``(B, A, num_classes)``.
        o2m_box: One-to-many raw ltrb distances, shape ``(B, A, 4)``.
        o2o_cls: One-to-one class logits, shape ``(B, A, num_classes)``.
        o2o_box: One-to-one raw ltrb distances, shape ``(B, A, 4)``.
    """

    o2m_cls: Tensor
    o2m_box: Tensor
    o2o_cls: Tensor
    o2o_box: Tensor


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

    Examples:
        >>> import torch
        >>> head = DualDetectionHead(in_channels=(64, 128, 256), num_classes=80).eval()
        >>> feats = (torch.zeros(1, 64, 80, 80), torch.zeros(1, 128, 40, 40), torch.zeros(1, 256, 20, 20))
        >>> with torch.no_grad():
        ...     out = head(feats)
        >>> out.o2o_cls.shape, out.o2o_box.shape
        (torch.Size([1, 8400, 80]), torch.Size([1, 8400, 4]))
    """

    def __init__(self, in_channels: tuple[int, int, int], num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.o2o = _DetectionBranch(in_channels, num_classes)
        self.o2m = _DetectionBranch(in_channels, num_classes)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> DualHeadOutput:
        """Run both branches over the neck features.

        Args:
            features: The neck maps ``(n3, n4, n5)`` at strides 8, 16, and 32,
                with channel counts matching the constructor's ``in_channels``.

        Returns:
            A :class:`DualHeadOutput` with dense class logits and raw ltrb
            distances for both the one-to-many and one-to-one branches.
        """
        o2m_cls, o2m_box = self.o2m(features)
        o2o_cls, o2o_box = self.o2o(features)
        return DualHeadOutput(o2m_cls=o2m_cls, o2m_box=o2m_box, o2o_cls=o2o_cls, o2o_box=o2o_box)


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
            :func:`~lit_yolo.assign.grid.make_anchor_points`.
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


def o2o_topk(scores: Tensor, boxes: Tensor, k: int = _DEFAULT_TOPK) -> Tensor:
    """Reduce the one-to-one branch to the top-k score-ranked detections.

    Convenience for the NMS-free path: score each anchor by its maximum class
    confidence (after a sigmoid), keep the ``k`` highest-scoring anchors per
    image, and emit the A9 detection tuple ``[x1, y1, x2, y2, score, class]``.
    The full E2E decode module lands in WP-041 and reuses this helper. When the
    anchor count ``A`` is below ``k`` every anchor is returned.

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
    _, num_anchors, _ = scores.shape
    confidence = scores.sigmoid()
    max_conf, class_index = confidence.max(dim=-1)  # both (B, A)
    keep = min(k, num_anchors)
    top_conf, top_anchor = max_conf.topk(keep, dim=1)  # both (B, keep)
    gather_box = top_anchor.unsqueeze(-1).expand(-1, -1, _BOX_OUTPUTS)
    top_boxes = boxes.gather(1, gather_box)  # (B, keep, 4)
    top_class = class_index.gather(1, top_anchor).to(scores.dtype)  # (B, keep)
    return torch.cat((top_boxes, top_conf.unsqueeze(-1), top_class.unsqueeze(-1)), dim=-1)
