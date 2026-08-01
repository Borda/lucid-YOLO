# SPDX-License-Identifier: Apache-2.0
"""Detection backbone with P3/P4/P5 taps (WP-020).

The convolutional trunk of the YOLO26 detector, assembled from the WP-016…WP-019
primitives (:class:`~lit_yolo.models.blocks.ConvBNAct`,
:class:`~lit_yolo.models.blocks.C3k2`, :class:`~lit_yolo.models.blocks.SPPF`,
:class:`~lit_yolo.models.blocks.C2PSA`) into the stack of blueprint sec. 5.2. Five
strided convolutions take a ``3x640x640`` image down to stride 32; three feature
maps are tapped at strides 8, 16, and 32 (the neck inputs P3, P4, P5).

Compound scaling (blueprint sec. 5.1) is applied by the constructor's raw
multipliers rather than a named variant (the ``n``/``s``/``m``/``l``/``x``
registry lands in WP-023): channel widths are ``int(min(base, max_channels) *
width)`` and per-stage repeat counts are ``max(1, round(2 * depth))``. Truncating
``int()`` after the width multiply is the A3-consistent rounding, validated by the
WP-023 parameter-count gate.

The tap placement follows the YOLO-lineage convention (assumption A3): the
diagram's ``-> skip to neck`` arrows sit on the strided ``Conv`` lines, but the
feature handed to the neck is the output of the CSP stage that *follows* each
downsampling conv — P3 after the ``512`` (``e=0.25``) C3k2 stage, P4 after the
``512`` (``c3k=True``) C3k2 stage, and P5 after the terminal ``C2PSA`` — so the
taps carry the refined stride-8/16/32 features, not the raw downsampled ones.

Provenance: R1 Fig. S1, R3 Fig. 1. Assumptions: A3, A4.
"""

from __future__ import annotations

from torch import Tensor, nn

from lit_yolo.models.blocks import C2PSA, SPPF, C3k2, ConvBNAct

_BACKBONE_REPEATS = 2
"""Base per-stage repeat count of every C3k2/C2PSA stage before depth scaling."""


def _scale_channels(base_channels: int, width: float, max_channels: int) -> int:
    """Return the width-scaled, max-channel-clamped channel count for a stage.

    Args:
        base_channels: Unscaled channel count from the blueprint sec. 5.2 stack.
        width: Width multiplier ``w`` of the variant (blueprint sec. 5.1).
        max_channels: Channel cap ``mc`` applied before the width multiply.

    Returns:
        ``int(min(base_channels, max_channels) * width)`` — truncating ``int()``
        is the A3-consistent rounding validated by the WP-023 parameter gate.

    Examples:
        >>> _scale_channels(512, 0.5, 1024)  # s-scale, below the cap
        256
        >>> _scale_channels(1024, 1.0, 512)  # m-scale, cap bites
        512
    """
    return int(min(base_channels, max_channels) * width)


def _scale_repeats(base_repeats: int, depth: float) -> int:
    """Return the depth-scaled inner-unit repeat count, floored at one.

    Args:
        base_repeats: Unscaled inner-unit count of a CSP stage.
        depth: Depth multiplier ``d`` of the variant (blueprint sec. 5.1).

    Returns:
        ``max(1, round(base_repeats * depth))`` — at least one unit always
        survives so no stage degenerates to a bare fusion.

    Examples:
        >>> _scale_repeats(2, 0.5)  # n/s/m-scale
        1
        >>> _scale_repeats(2, 1.0)  # l/x-scale
        2
    """
    return max(1, round(base_repeats * depth))


class DetectionBackbone(nn.Module):
    """YOLO26 convolutional backbone emitting P3/P4/P5 neck features.

    Builds the blueprint sec. 5.2 stack from the shared primitives and taps three
    feature maps at strides 8, 16, and 32. The constructor takes the raw compound
    -scaling multipliers directly (the named-variant registry is WP-023): channel
    counts follow ``int(min(base, max_channels) * width)`` and every CSP stage
    repeats ``max(1, round(2 * depth))`` inner units.

    Args:
        depth: Depth multiplier ``d`` scaling per-stage repeat counts.
        width: Width multiplier ``w`` scaling channel counts.
        max_channels: Channel cap ``mc`` applied before the width multiply; bites
            at the P5 stage for the ``m``/``l``/``x`` variants (``mc=512``).

    Examples:
        >>> import torch
        >>> backbone = DetectionBackbone(depth=0.5, width=0.25, max_channels=1024).eval()
        >>> backbone.channels  # n-scale tap widths (P3, P4, P5)
        (128, 128, 256)
        >>> with torch.no_grad():
        ...     p3, p4, p5 = backbone(torch.zeros(1, 3, 640, 640))
        >>> p3.shape, p4.shape, p5.shape
        (torch.Size([1, 128, 80, 80]), torch.Size([1, 128, 40, 40]), torch.Size([1, 256, 20, 20]))
    """

    def __init__(self, depth: float, width: float, max_channels: int) -> None:
        super().__init__()
        self.depth = depth
        self.width = width
        self.max_channels = max_channels

        ch64 = _scale_channels(64, width, max_channels)
        ch128 = _scale_channels(128, width, max_channels)
        ch256 = _scale_channels(256, width, max_channels)
        ch512 = _scale_channels(512, width, max_channels)
        ch1024 = _scale_channels(1024, width, max_channels)
        repeats = _scale_repeats(_BACKBONE_REPEATS, depth)

        self.stem = ConvBNAct(3, ch64, 3, stride=2)  # P1/2
        self.down_p2 = ConvBNAct(ch64, ch128, 3, stride=2)  # P2/4
        self.stage_p2 = C3k2(ch128, ch256, n=repeats, e=0.25, c3k=False)
        self.down_p3 = ConvBNAct(ch256, ch256, 3, stride=2)  # P3/8
        self.stage_p3 = C3k2(ch256, ch512, n=repeats, e=0.25, c3k=False)  # -> P3 tap
        self.down_p4 = ConvBNAct(ch512, ch512, 3, stride=2)  # P4/16
        self.stage_p4 = C3k2(ch512, ch512, n=repeats, c3k=True)  # -> P4 tap
        self.down_p5 = ConvBNAct(ch512, ch1024, 3, stride=2)  # P5/32
        self.stage_p5 = C3k2(ch1024, ch1024, n=repeats, c3k=True)
        self.sppf = SPPF(ch1024, ch1024, pool_kernel=5)
        self.attn_p5 = C2PSA(ch1024, ch1024, n=repeats)  # -> P5 tap

        self._channels = (ch512, ch512, ch1024)

    @property
    def channels(self) -> tuple[int, int, int]:
        """Tap channel counts ``(P3, P4, P5)`` for the neck to consume.

        Returns:
            The width-scaled output channel counts of the three tapped stages, in
            stride order (8, 16, 32).

        Examples:
            >>> DetectionBackbone(depth=0.5, width=0.5, max_channels=1024).channels
            (256, 256, 512)
        """
        return self._channels

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run the trunk and return the three tapped neck features.

        Args:
            x: Input image batch of shape ``(N, 3, H, W)`` with ``H`` and ``W``
                divisible by 32.

        Returns:
            The ``(p3, p4, p5)`` feature maps at strides 8, 16, and 32 — spatial
            sizes ``H/8``, ``H/16``, ``H/32`` and channel counts :attr:`channels`.
        """
        x = self.stem(x)
        x = self.down_p2(x)
        x = self.stage_p2(x)
        x = self.down_p3(x)
        p3 = self.stage_p3(x)
        x = self.down_p4(p3)
        p4 = self.stage_p4(x)
        x = self.down_p5(p4)
        x = self.stage_p5(x)
        x = self.sppf(x)
        p5 = self.attn_p5(x)
        return p3, p4, p5
