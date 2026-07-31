# SPDX-License-Identifier: Apache-2.0
"""Backbone/neck primitive blocks (WP-016).

The three convolutional primitives shared by every composite backbone and neck
block in the YOLO26 family: a fused convolution-normalization-activation unit, a
depthwise variant expressed on top of it, and a residual bottleneck. Later work
packages (C3k2, PSABlock/C2PSA, SPPF) compose these — see blueprint sec. 5.2.

Internals follow the YOLO11 lineage described in the method paper and its
supplementary block diagram (assumption A3): a bias-free convolution folded into
BatchNorm, SiLU activation, and CSP-style bottlenecks with an optional identity
shortcut. Padding is auto-computed so odd kernels preserve spatial size at
stride 1 ("same" convolution).

Provenance: R11 (arXiv:2501.13400), R1 Fig. S2. Assumptions: A3.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    """Return the symmetric padding that keeps spatial size fixed at stride 1.

    Valid for odd kernels, where the effective (dilated) kernel is also odd and
    the "same" padding is integral. Even kernels have no symmetric same-padding
    and are not used by this family.

    Args:
        kernel_size: Convolution kernel size (expected odd).
        dilation: Dilation factor of the convolution.

    Returns:
        The per-side padding amount ``dilation * (kernel_size - 1) // 2``.

    Examples:
        >>> _same_padding(3)
        1
        >>> _same_padding(5)
        2
        >>> _same_padding(3, dilation=2)
        2
    """
    return dilation * (kernel_size - 1) // 2


class ConvBNAct(nn.Module):
    """Fused convolution + BatchNorm + SiLU activation.

    The canonical convolutional unit of the backbone and neck. The convolution
    is bias-free because the following BatchNorm supplies its own affine shift;
    padding is auto-computed so that odd kernels preserve spatial resolution at
    stride 1.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size (odd for "same" padding).
        stride: Convolution stride. Defaults to 1.
        groups: Number of blocked connections; ``in_channels`` yields a
            depthwise convolution. Defaults to 1.
        dilation: Convolution dilation factor. Defaults to 1.

    Examples:
        >>> import torch
        >>> block = ConvBNAct(3, 16, kernel_size=3, stride=2).eval()
        >>> block(torch.zeros(1, 3, 32, 32)).shape
        torch.Size([1, 16, 16, 16])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=_same_padding(kernel_size, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolution, normalization, then activation.

        Args:
            x: Input tensor of shape ``(N, in_channels, H, W)``.

        Returns:
            Activated tensor of shape ``(N, out_channels, H', W')``.
        """
        return cast(Tensor, self.act(self.bn(self.conv(x))))


class DepthwiseConv(nn.Module):
    """Depthwise ConvBNAct — one convolution group per input channel.

    A thin specialization of :class:`ConvBNAct` with ``groups == in_channels``,
    the spatial-mixing half of a depthwise-separable convolution. ``out_channels``
    must be divisible by ``in_channels`` (the PyTorch grouped-convolution
    constraint).

    Args:
        in_channels: Number of input channels (also the group count).
        out_channels: Number of output channels; must be a multiple of
            ``in_channels``.
        kernel_size: Convolution kernel size (odd for "same" padding).
        stride: Convolution stride. Defaults to 1.
        dilation: Convolution dilation factor. Defaults to 1.

    Examples:
        >>> import torch
        >>> block = DepthwiseConv(16, 16, kernel_size=5).eval()
        >>> block(torch.zeros(1, 16, 8, 8)).shape
        torch.Size([1, 16, 8, 8])
        >>> block.conv.conv.groups
        16
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=in_channels,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the depthwise convolution unit.

        Args:
            x: Input tensor of shape ``(N, in_channels, H, W)``.

        Returns:
            Tensor of shape ``(N, out_channels, H', W')``.
        """
        return cast(Tensor, self.conv(x))


class Bottleneck(nn.Module):
    """Residual bottleneck of two stacked ConvBNAct layers.

    The inner block of the CSP-style backbone stages. Channels are squeezed to
    ``int(out_channels * expansion)`` by the first convolution and restored by
    the second. An identity shortcut is added when requested and the input and
    output channel counts match; otherwise the block is purely feed-forward.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        shortcut: Add the residual identity when ``in_channels == out_channels``.
            Defaults to ``True``.
        kernel_sizes: Kernel sizes of the two convolutions. Defaults to ``(3, 3)``.
        expansion: Hidden-channel ratio ``int(out_channels * expansion)``.
            Defaults to ``0.5``.

    Examples:
        >>> import torch
        >>> block = Bottleneck(32, 32).eval()
        >>> block(torch.zeros(1, 32, 8, 8)).shape
        torch.Size([1, 32, 8, 8])
        >>> block.cv1.conv.out_channels  # hidden = int(32 * 0.5)
        16
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        kernel_sizes: tuple[int, int] = (3, 3),
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNAct(in_channels, hidden_channels, kernel_sizes[0])
        self.cv2 = ConvBNAct(hidden_channels, out_channels, kernel_sizes[1])
        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Apply the two convolutions with an optional residual add.

        Args:
            x: Input tensor of shape ``(N, in_channels, H, W)``.

        Returns:
            Tensor of shape ``(N, out_channels, H, W)``; the input is added when
            the shortcut is active.
        """
        y = cast(Tensor, self.cv2(self.cv1(x)))
        return x + y if self.add_shortcut else y
