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

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

#: Inner-bottleneck count of the nested :class:`C3k` used by ``C3k2`` when
#: ``c3k=True`` (assumption A3). Reduced from 2 to 1 by the WP-023 parameter/FLOP
#: fidelity gate: the depth-2 (``l``/``x``) variants carry two of these nested
#: units per stage, and ``n=2`` overshot the R1 Table 7 FLOP budget at those
#: scales — ``n=1`` lands all five within tolerance. See docs/ASSUMPTIONS.md A3.
_C3K_INNER_UNITS = 1


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


def _conv_bn(in_channels: int, out_channels: int, kernel_size: int, groups: int = 1) -> nn.Sequential:
    """Return a bias-free convolution folded into BatchNorm, with no activation.

    The convolutional unit used inside the attention block: unlike
    :class:`ConvBNAct` it omits the SiLU. The attention projections (qkv, the
    depthwise positional term, and the output projection) are left unactivated so
    the non-linearity does not distort the attention logits or the residual
    stream — a choice recorded under assumption A3.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size (odd for "same" padding).
        groups: Number of blocked connections; ``in_channels`` yields a depthwise
            convolution. Defaults to 1.

    Returns:
        An ``nn.Sequential`` of a bias-free :class:`~torch.nn.Conv2d` followed by
        :class:`~torch.nn.BatchNorm2d`.

    Examples:
        >>> import torch
        >>> unit = _conv_bn(16, 16, 3, groups=16).eval()
        >>> unit(torch.zeros(1, 16, 8, 8)).shape
        torch.Size([1, 16, 8, 8])
        >>> len(unit)  # conv + batchnorm, no activation
        2
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=_same_padding(kernel_size),
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    )


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


class C3k(nn.Module):
    """CSP block with three 1x1 fusion convolutions and ``n`` inner bottlenecks.

    The heavier CSP variant used for the deeper backbone/neck stages. Two 1x1
    convolutions (``cv1``, ``cv2``) split the input into two ``hidden``-channel
    streams; ``cv1``'s stream is refined by ``n`` sequential :class:`Bottleneck`
    blocks (kernel sizes ``(3, 3)``, internal expansion ``1.0`` so the bottleneck
    keeps full width), then the two streams are concatenated and fused back to
    ``out_channels`` by a third 1x1 convolution (``cv3``).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        n: Number of inner bottlenecks on the refined stream. Defaults to 1.
        shortcut: Enable the residual identity inside each inner bottleneck.
            Defaults to ``True``.
        expansion: Hidden-channel ratio ``int(out_channels * expansion)`` for the
            two split streams. Defaults to ``0.5``.

    Examples:
        >>> import torch
        >>> block = C3k(64, 64, n=2).eval()
        >>> block(torch.zeros(1, 64, 8, 8)).shape
        torch.Size([1, 64, 8, 8])
        >>> len(block.blocks)  # n inner bottlenecks
        2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNAct(in_channels, hidden_channels, 1)
        self.cv2 = ConvBNAct(in_channels, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *(
                Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut, kernel_sizes=(3, 3), expansion=1.0)
                for _ in range(n)
            )
        )
        self.cv3 = ConvBNAct(2 * hidden_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Split into two streams, refine one, then concatenate and fuse.

        Args:
            x: Input tensor of shape ``(N, in_channels, H, W)``.

        Returns:
            Tensor of shape ``(N, out_channels, H, W)``.
        """
        refined = cast(Tensor, self.blocks(self.cv1(x)))
        return cast(Tensor, self.cv3(torch.cat((refined, self.cv2(x)), dim=1)))


class C3k2(nn.Module):
    """Fast CSP block with densely chained inner units (YOLO11 lineage).

    The workhorse composite of the backbone and neck. A single 1x1 convolution
    (``cv1``) lifts the input to ``2 * hidden`` channels, split into two
    ``hidden``-channel halves. The second half is fed through ``n`` inner units
    with dense CSP chaining: each unit consumes the most recent tensor and its
    output is appended, so the pre-fusion concatenation accumulates
    ``(2 + n) * hidden`` channels (both original halves plus every unit output).
    A final 1x1 convolution (``cv2``) fuses that stack to ``out_channels``.

    The inner-unit type is selected by ``c3k``: a plain :class:`Bottleneck`
    (internal expansion ``1.0``) when ``False``, or a nested :class:`C3k`
    (``n=`` :data:`_C3K_INNER_UNITS`, ``expansion=0.5``) when ``True``.
    ``inner_block_factory`` overrides
    this selection entirely — the seam by which the attention-augmented neck
    variant (a bottleneck followed by a ``PSABlock``) is introduced without
    touching this class.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        n: Number of densely chained inner units. Defaults to 1.
        e: Hidden-channel ratio ``int(out_channels * e)``; early backbone stages
            use ``0.25``. Defaults to ``0.5``.
        c3k: Use nested :class:`C3k` inner units instead of :class:`Bottleneck`.
            Defaults to ``False``.
        shortcut: Enable the residual identity inside each inner unit. Defaults
            to ``True``.
        inner_block_factory: Optional ``(hidden_channels, shortcut) -> Module``
            builder that replaces the default ``c3k`` selection for every inner
            unit. Defaults to ``None`` (use the built-in selection).

    Examples:
        >>> import torch
        >>> block = C3k2(64, 128, n=2, c3k=False).eval()
        >>> block(torch.zeros(1, 64, 8, 8)).shape
        torch.Size([1, 128, 8, 8])
        >>> block.cv2.conv.in_channels  # (2 + n) * hidden, hidden = int(128 * 0.5)
        256
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        e: float = 0.5,
        c3k: bool = False,
        shortcut: bool = True,
        inner_block_factory: Callable[[int, bool], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * e)
        self.hidden_channels = hidden_channels
        self.cv1 = ConvBNAct(in_channels, 2 * hidden_channels, 1)
        make_inner = inner_block_factory or self._default_inner_factory(c3k)
        self.blocks = nn.ModuleList(make_inner(hidden_channels, shortcut) for _ in range(n))
        self.cv2 = ConvBNAct((2 + n) * hidden_channels, out_channels, 1)

    @staticmethod
    def _default_inner_factory(c3k: bool) -> Callable[[int, bool], nn.Module]:
        """Return the built-in inner-unit builder selected by ``c3k``.

        Args:
            c3k: Select nested :class:`C3k` units when ``True``, otherwise plain
                :class:`Bottleneck` units.

        Returns:
            A ``(hidden_channels, shortcut) -> Module`` factory.

        Examples:
            >>> factory = C3k2._default_inner_factory(c3k=False)
            >>> type(factory(32, True)).__name__
            'Bottleneck'
        """

        def factory(hidden_channels: int, shortcut: bool) -> nn.Module:
            if c3k:
                return C3k(hidden_channels, hidden_channels, n=_C3K_INNER_UNITS, shortcut=shortcut, expansion=0.5)
            return Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut, expansion=1.0)

        return factory

    def forward(self, x: Tensor) -> Tensor:
        """Split, densely chain the inner units, then concatenate and fuse.

        Args:
            x: Input tensor of shape ``(N, in_channels, H, W)``.

        Returns:
            Tensor of shape ``(N, out_channels, H, W)``.
        """
        y: list[Tensor] = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            y.append(cast(Tensor, block(y[-1])))
        return cast(Tensor, self.cv2(torch.cat(y, dim=1)))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast, with a YOLO26 input-to-output shortcut.

    The multi-scale pooling tail of the backbone. A 1x1 :class:`ConvBNAct`
    (``cv1``) squeezes the input to ``in_channels // 2`` hidden channels; a single
    ``pool_kernel`` max-pool is applied three times in sequence, so the chained
    receptive fields grow to the same coverage as one pool of size
    ``3 * (pool_kernel - 1) + 1`` (a 13x13 field for the default 5x5 kernel) while
    reusing one small kernel. The hidden stream and its three pooled copies are
    concatenated (``4 * hidden`` channels) and fused back to ``out_channels`` by a
    second 1x1 :class:`ConvBNAct` (``cv2``).

    The YOLO26 refinement (assumption A4) adds the block input to the ``cv2``
    output — an input-to-output shortcut around the whole pooling stack. As in
    :class:`Bottleneck`, the identity is added only when ``in_channels ==
    out_channels`` (the backbone always uses SPPF this way); a widening SPPF with
    mismatched channels is purely feed-forward. Max-pool uses stride 1 and
    symmetric ``pool_kernel // 2`` padding so spatial resolution is preserved.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        pool_kernel: Kernel size of the repeated max-pool. Defaults to 5.

    Examples:
        >>> import torch
        >>> block = SPPF(64, 64).eval()
        >>> block(torch.zeros(1, 64, 8, 8)).shape
        torch.Size([1, 64, 8, 8])
        >>> block.cv1.conv.out_channels  # hidden = in_channels // 2
        32
        >>> block.add_shortcut
        True
    """

    def __init__(self, in_channels: int, out_channels: int, pool_kernel: int = 5) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden_channels, 1)
        self.pool = nn.MaxPool2d(pool_kernel, stride=1, padding=pool_kernel // 2)
        self.cv2 = ConvBNAct(4 * hidden_channels, out_channels, 1)
        self.add_shortcut = in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Pool at three chained scales, fuse, then add the optional shortcut.

        Args:
            x: Input tensor of shape ``(N, in_channels, H, W)``.

        Returns:
            Tensor of shape ``(N, out_channels, H, W)``; the input is added when
            the shortcut is active.
        """
        x1 = cast(Tensor, self.cv1(x))
        y1 = self.pool(x1)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        y = cast(Tensor, self.cv2(torch.cat((x1, y1, y2, y3), dim=1)))
        return x + y if self.add_shortcut else y


class SpatialAttention(nn.Module):
    """Multi-head self-attention over the spatial positions of a feature map.

    The attention half of a :class:`PSABlock`. Each of the ``H * W`` positions of
    a ``(N, C, H, W)`` map attends to every other position. A single 1x1
    convolution produces the packed query/key/value projection; per head the
    channels split into ``key_dim`` for the query, ``key_dim`` for the key, and
    ``head_dim`` for the value. Scaled dot-product attention over the flattened
    spatial axis mixes the values, a depthwise 3x3 convolution on the values adds
    a local positional term, and a final 1x1 projection restores ``C`` channels.

    Head geometry follows the YOLO11 lineage (assumption A3): ``num_heads =
    max(1, C // 64)``, ``head_dim = C // num_heads``, and ``key_dim = head_dim //
    2``, so the attention logits are scaled by ``key_dim ** -0.5``. The qkv,
    positional, and output convolutions are :func:`_conv_bn` units (convolution +
    BatchNorm, **no** activation) so the non-linearity never distorts the
    attention logits or the residual stream.

    ``C`` must be divisible by ``num_heads`` (always true for the channel counts
    used by the family, which are multiples of 64).

    Args:
        channels: Number of input and output channels ``C``.

    Examples:
        >>> import torch
        >>> attn = SpatialAttention(128).eval()
        >>> attn.num_heads  # max(1, 128 // 64)
        2
        >>> attn(torch.zeros(1, 128, 8, 8)).shape
        torch.Size([1, 128, 8, 8])
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        num_heads = max(1, channels // 64)
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.key_dim = self.head_dim // 2
        self.scale = self.key_dim**-0.5
        qkv_channels = num_heads * (2 * self.key_dim + self.head_dim)
        self.qkv = _conv_bn(channels, qkv_channels, 1)
        self.pe = _conv_bn(channels, channels, 3, groups=channels)
        self.proj = _conv_bn(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Mix spatial positions by scaled dot-product attention.

        Args:
            x: Input tensor of shape ``(N, C, H, W)``.

        Returns:
            Tensor of shape ``(N, C, H, W)``, the projected attention output.
        """
        n, c, h, w = x.shape
        hw = h * w
        qkv = cast(Tensor, self.qkv(x)).view(n, self.num_heads, 2 * self.key_dim + self.head_dim, hw)
        q = qkv[:, :, : self.key_dim]
        k = qkv[:, :, self.key_dim : 2 * self.key_dim]
        v = qkv[:, :, 2 * self.key_dim :]
        attn = (q.transpose(-2, -1) @ k) * self.scale  # (N, num_heads, HW, HW)
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(n, c, h, w)
        out = out + cast(Tensor, self.pe(v.reshape(n, c, h, w)))
        return cast(Tensor, self.proj(out))


class PSABlock(nn.Module):
    """Position-sensitive attention block: attention followed by a feed-forward.

    The inner unit of :class:`C2PSA`. It applies a residual :class:`SpatialAttention`
    then a residual feed-forward network, mirroring a transformer encoder layer
    adapted to convolutional feature maps (assumption A3):

    - ``x = x + attention(x)``
    - ``x = x + ffn(x)``

    The feed-forward network expands to ``2 * C`` channels through a
    :class:`ConvBNAct` (convolution + BatchNorm + SiLU) and projects back to ``C``
    through a :func:`_conv_bn` unit (convolution + BatchNorm, no activation) so the
    residual branch stays linear at its output. Both residual adds require
    matching channel counts, which the block preserves throughout.

    Args:
        channels: Number of input and output channels ``C``.

    Examples:
        >>> import torch
        >>> block = PSABlock(128).eval()
        >>> block(torch.zeros(1, 128, 8, 8)).shape
        torch.Size([1, 128, 8, 8])
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = SpatialAttention(channels)
        self.ffn = nn.Sequential(
            ConvBNAct(channels, 2 * channels, 1),
            _conv_bn(2 * channels, channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual attention then residual feed-forward.

        Args:
            x: Input tensor of shape ``(N, C, H, W)``.

        Returns:
            Tensor of shape ``(N, C, H, W)``.
        """
        x = x + self.attn(x)
        return cast(Tensor, x + self.ffn(x))


class C2PSA(nn.Module):
    """CSP wrapper around ``n`` stacked :class:`PSABlock` attention units.

    The attention tail of the detection neck. A single 1x1 :class:`ConvBNAct`
    (``cv1``) lifts the input to ``2 * hidden`` channels (``hidden = int(out_channels
    * e)``) and splits it into two ``hidden``-channel halves. The first half is
    carried through unchanged; the second is refined by ``n`` sequential
    :class:`PSABlock` units. The two halves are concatenated and fused back to
    ``out_channels`` by a final 1x1 :class:`ConvBNAct` (``cv2``). In the neck it is
    used with ``in_channels == out_channels`` (assumption A3).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        n: Number of stacked :class:`PSABlock` units on the refined half. Defaults
            to 1.
        e: Hidden-channel ratio ``int(out_channels * e)`` for each split half.
            Defaults to ``0.5``.

    Examples:
        >>> import torch
        >>> block = C2PSA(256, 256, n=2).eval()
        >>> block(torch.zeros(1, 256, 8, 8)).shape
        torch.Size([1, 256, 8, 8])
        >>> len(block.blocks)  # n stacked PSABlocks
        2
    """

    def __init__(self, in_channels: int, out_channels: int, n: int = 1, e: float = 0.5) -> None:
        super().__init__()
        hidden_channels = int(out_channels * e)
        self.hidden_channels = hidden_channels
        self.cv1 = ConvBNAct(in_channels, 2 * hidden_channels, 1)
        self.blocks = nn.Sequential(*(PSABlock(hidden_channels) for _ in range(n)))
        self.cv2 = ConvBNAct(2 * hidden_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Split, refine one half with attention, then concatenate and fuse.

        Args:
            x: Input tensor of shape ``(N, in_channels, H, W)``.

        Returns:
            Tensor of shape ``(N, out_channels, H, W)``.
        """
        a, b = self.cv1(x).chunk(2, dim=1)
        refined = cast(Tensor, self.blocks(b))
        return cast(Tensor, self.cv2(torch.cat((a, refined), dim=1)))
