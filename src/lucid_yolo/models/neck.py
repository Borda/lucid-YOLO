# SPDX-License-Identifier: Apache-2.0
"""Detection neck with a PSABlock attention tail (WP-021).

The bidirectional feature-pyramid neck of the YOLO26 detector, assembled from the
WP-016…WP-019 primitives (:class:`~lucid_yolo.models.blocks.C3k2`,
:class:`~lucid_yolo.models.blocks.C3k`, :class:`~lucid_yolo.models.blocks.PSABlock`,
:class:`~lucid_yolo.models.blocks.ConvBNAct`) into the stack of blueprint sec. 5.3.
It consumes the backbone's P3/P4/P5 taps (strides 8/16/32) and returns three
refined feature maps at the same strides for the detection head.

The topology is the PAN-style top-down then bottom-up path:

- **Top-down** — upsample P5, concatenate with P4, refine with a ``c3k=True``
  ``C3k2`` (call it ``T4``); upsample ``T4``, concatenate with P3, refine with a
  ``C3k2`` to the stride-8 output ``N3``.
- **Bottom-up** — downsample ``N3``, concatenate with ``T4``, refine with a
  ``C3k2`` to the stride-16 output ``N4``; downsample ``N4``, concatenate with
  P5, refine with the attention-augmented ``C3k2`` to the stride-32 output ``N5``.

The final ``C3k2`` carries a :class:`~lucid_yolo.models.blocks.PSABlock` attention
layer with a fixed ``n=1`` inner unit — the "one additional attention layer in the
detection neck" refinement worth +0.2 AP in the ablation (R1 sec. 4.2 Table 2;
R3 sec. 4). It is introduced through ``C3k2``'s ``inner_block_factory`` seam so the
composite block class is left untouched: each inner unit becomes a
``c3k=True`` :class:`~lucid_yolo.models.blocks.C3k` followed by a
:class:`~lucid_yolo.models.blocks.PSABlock` on its output.

Compound scaling reuses the WP-020 backbone helpers (``_scale_channels``,
``_scale_repeats``) so channel widths and repeat counts follow exactly the same
rules as the trunk (blueprint sec. 5.1): the ``C3k2`` output widths are the base
values 512/256/512/1024 scaled by ``width``, and every ``x2`` stage repeats
``max(1, round(2 * depth))`` inner units. The attention tail's ``n=1`` is fixed by
the spec and is **not** depth-scaled.

Provenance: R1 Fig. S1, R3 Fig. 1, R1 sec. 4.2 Table 2. Assumptions: A3.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lucid_yolo.models.backbone import _scale_channels, _scale_repeats
from lucid_yolo.models.blocks import C3k, C3k2, ConvBNAct, PSABlock

_NECK_REPEATS = 2
"""Base per-stage repeat count of every ``x2`` neck ``C3k2`` before depth scaling."""


def _attn_inner_factory(hidden_channels: int, shortcut: bool) -> nn.Module:
    """Build one attention-augmented inner unit for the final neck ``C3k2``.

    The composite is the ``c3k=True`` inner unit — a :class:`~lucid_yolo.models.blocks.C3k`
    with ``n=2`` and ``expansion=0.5`` — followed by a
    :class:`~lucid_yolo.models.blocks.PSABlock` on its output, matching the attention
    tail of blueprint sec. 5.3. It is passed to :class:`~lucid_yolo.models.blocks.C3k2`
    as its ``inner_block_factory`` so the block class needs no attention-aware branch.

    Args:
        hidden_channels: Channel width of the ``C3k2`` split half this unit refines;
            both the ``C3k`` and the ``PSABlock`` preserve it end to end.
        shortcut: Enable the residual identity inside the nested ``C3k``.

    Returns:
        An ``nn.Sequential`` of the ``C3k`` refinement followed by the ``PSABlock``.

    Examples:
        >>> import torch
        >>> unit = _attn_inner_factory(128, True).eval()
        >>> unit(torch.zeros(1, 128, 8, 8)).shape
        torch.Size([1, 128, 8, 8])
        >>> type(unit[1]).__name__  # attention layer on the C3k output
        'PSABlock'
    """
    return nn.Sequential(
        C3k(hidden_channels, hidden_channels, n=2, shortcut=shortcut, expansion=0.5),
        PSABlock(hidden_channels),
    )


class DetectionNeck(nn.Module):
    """YOLO26 bidirectional feature-pyramid neck with an attention tail.

    Builds the blueprint sec. 5.3 top-down/bottom-up stack from the shared
    primitives, consuming the backbone's P3/P4/P5 taps and emitting three refined
    maps ``(N3, N4, N5)`` at the same strides 8/16/32 for the detection head. The
    constructor takes the raw compound-scaling multipliers directly (the named
    -variant registry is WP-023) and reuses the backbone's scaling helpers so the
    channel and repeat rules match the trunk exactly.

    The ``C3k2`` output widths are the base values 512 (``T4``), 256 (``N3``), 512
    (``N4``), and 1024 (``N5``) scaled by ``width`` and clamped by ``max_channels``;
    the three ``x2`` stages repeat ``max(1, round(2 * depth))`` inner units while the
    final attention-augmented stage is fixed at ``n=1``.

    Args:
        in_channels: The backbone tap channel counts ``(P3, P4, P5)`` in stride
            order (8, 16, 32) — typically :attr:`DetectionBackbone.channels`.
        depth: Depth multiplier ``d`` scaling the ``x2`` stages' repeat counts.
        width: Width multiplier ``w`` scaling the ``C3k2`` output channel widths.
        max_channels: Channel cap ``mc`` applied before the width multiply; bites at
            the ``N5`` stage (base 1024) for the ``m``/``l``/``x`` variants (``mc=512``).

    Examples:
        >>> import torch
        >>> neck = DetectionNeck(in_channels=(128, 128, 256), depth=0.5, width=0.25, max_channels=1024).eval()
        >>> neck.channels  # n-scale output widths (N3, N4, N5)
        (64, 128, 256)
        >>> with torch.no_grad():
        ...     feats = (torch.zeros(1, 128, 80, 80), torch.zeros(1, 128, 40, 40), torch.zeros(1, 256, 20, 20))
        ...     n3, n4, n5 = neck(feats)
        >>> n3.shape, n4.shape, n5.shape
        (torch.Size([1, 64, 80, 80]), torch.Size([1, 128, 40, 40]), torch.Size([1, 256, 20, 20]))
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        depth: float,
        width: float,
        max_channels: int,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.width = width
        self.max_channels = max_channels

        in_p3, in_p4, in_p5 = in_channels
        t4_ch = _scale_channels(512, width, max_channels)
        n3_ch = _scale_channels(256, width, max_channels)
        n4_ch = _scale_channels(512, width, max_channels)
        n5_ch = _scale_channels(1024, width, max_channels)
        repeats = _scale_repeats(_NECK_REPEATS, depth)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down: fuse coarse features into finer levels.
        self.top_down_p4 = C3k2(in_p5 + in_p4, t4_ch, n=repeats, c3k=True)  # -> T4
        self.top_down_p3 = C3k2(t4_ch + in_p3, n3_ch, n=repeats)  # -> N3 (P3 out)

        # Bottom-up: fuse fine features back up, ending in the attention tail.
        self.down_n3 = ConvBNAct(n3_ch, n3_ch, 3, stride=2)
        self.bottom_up_p4 = C3k2(n3_ch + t4_ch, n4_ch, n=repeats)  # -> N4 (P4 out)
        self.down_n4 = ConvBNAct(n4_ch, n4_ch, 3, stride=2)
        self.bottom_up_p5 = C3k2(
            n4_ch + in_p5,
            n5_ch,
            n=1,  # fixed by spec — the single attention layer, not depth-scaled
            e=0.5,
            c3k=True,
            inner_block_factory=_attn_inner_factory,
        )  # -> N5 (P5 out)

        self._channels = (n3_ch, n4_ch, n5_ch)

    @property
    def channels(self) -> tuple[int, int, int]:
        """Output channel counts ``(N3, N4, N5)`` for the detection head to consume.

        Returns:
            The width-scaled output channel counts of the three neck outputs, in
            stride order (8, 16, 32).

        Examples:
            >>> DetectionNeck(in_channels=(256, 256, 512), depth=0.5, width=0.5, max_channels=1024).channels
            (128, 256, 512)
        """
        return self._channels

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Fuse the backbone taps top-down then bottom-up into head features.

        Args:
            features: The backbone taps ``(p3, p4, p5)`` at strides 8, 16, and 32,
                with channel counts matching the constructor's ``in_channels``.

        Returns:
            The ``(n3, n4, n5)`` refined feature maps at strides 8, 16, and 32 —
            spatial sizes preserved from the taps and channel counts :attr:`channels`.
        """
        p3, p4, p5 = features

        t4 = self.top_down_p4(torch.cat((self.upsample(p5), p4), dim=1))
        n3 = self.top_down_p3(torch.cat((self.upsample(t4), p3), dim=1))

        n4 = self.bottom_up_p4(torch.cat((self.down_n3(n3), t4), dim=1))
        n5 = self.bottom_up_p5(torch.cat((self.down_n4(n4), p5), dim=1))
        return n3, n4, n5
