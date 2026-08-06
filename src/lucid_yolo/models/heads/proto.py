# SPDX-License-Identifier: Apache-2.0
"""Multi-scale prototype-feature fusion from Eq. 8 (WP-048).

The fusion preserves the highest-resolution P3 feature ``X_1`` directly, then
adds each coarser P4/P5 feature after a learned 1x1 projection into P3's channel
space and nearest-neighbour upsampling to P3's exact spatial resolution. It
returns only the fused feature; prototype generation, semantic supervision, and
mask assembly intentionally land in later work packages.

Provenance: R1 Eq. 8. Assumptions: A35.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional

__all__ = ["ProtoFusion"]


class ProtoFusion(nn.Module):
    """Fuse stride-8/16/32 features into the Eq. 8 prototype input ``F_proto``.

    The finest feature ``X_1`` (P3) is deliberately not projected: its identity
    path is the first term of Eq. 8. P4 and P5 each receive their own bare 1x1
    convolution into P3's channel space before nearest-neighbour interpolation
    to P3's exact size, which keeps the sum well-formed for odd and non-square
    feature maps.

    Args:
        in_channels: Per-level neck channel counts ``(N3, N4, N5)`` in stride
            order (8, 16, 32).

    Attributes:
        out_channels: Channel count of the fused P3-resolution output.
        projections: Independent 1x1 P4 and P5 projections into ``N3`` channels.

    Examples:
        >>> import torch
        >>> fusion = ProtoFusion((64, 128, 256)).eval()
        >>> features = (
        ...     torch.zeros(1, 64, 80, 80),
        ...     torch.zeros(1, 128, 40, 40),
        ...     torch.zeros(1, 256, 20, 20),
        ... )
        >>> fusion(features).shape
        torch.Size([1, 64, 80, 80])
    """

    def __init__(self, in_channels: tuple[int, int, int]) -> None:
        """Initialize the two coarse-level projections required by Eq. 8.

        Args:
            in_channels: Per-level neck channel counts ``(N3, N4, N5)`` in
                stride order (8, 16, 32).
        """
        super().__init__()
        n3_channels, n4_channels, n5_channels = in_channels
        self.out_channels: int = n3_channels
        self.projections = nn.ModuleList(
            (
                nn.Conv2d(n4_channels, n3_channels, 1),
                nn.Conv2d(n5_channels, n3_channels, 1),
            )
        )

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Return ``X_1 + U(phi_2(X_2)) + U(phi_3(X_3))`` at P3 resolution.

        Args:
            features: Neck features ``(X_1, X_2, X_3)`` at strides 8, 16, and
                32, with channel counts matching ``in_channels``.

        Returns:
            The fused feature map with shape ``(B, N3, H1, W1)``.
        """
        x1, x2, x3 = features
        target_size = x1.shape[-2:]
        p4 = functional.interpolate(self.projections[0](x2), size=target_size, mode="nearest")
        p5 = functional.interpolate(self.projections[1](x3), size=target_size, mode="nearest")
        return x1 + p4 + p5
