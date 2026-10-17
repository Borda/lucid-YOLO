# SPDX-License-Identifier: Apache-2.0
"""Restore deferred CUDA colour in one pixel pass from the existing expressions.

The training hook already owns a quantization boundary after HSV. The kernel
keeps each floating operation's order, uses precise division where the existing
operands are tensors, and disables multiply-add contraction. Only the rounded
uint8 codes survive that boundary; they are restored with CUDA's existing
host-scalar reciprocal and stored directly in the consumer's channels-last layout.
Triton is optional: importing this module without it leaves the native path usable.
"""

import os
import shutil
from functools import lru_cache
from typing import Any  # noqa: F401 — mypy reads the DSL type comments.

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as lib
except ImportError:
    _AVAILABLE = False
else:
    _AVAILABLE = True

#: The balanced launch from the real-shape L4 matrix, including small tail blocks.
_BLOCK_SIZE = 256

if _AVAILABLE:

    @triton.jit  # type: ignore[untyped-decorator]
    def _is_nan(x):  # type: (Any) -> Any
        """Keep the ordered NaN comparison explicit in the CUDA DSL."""
        return x != x  # noqa: PLR0124 — IEEE NaN predicate in the DSL.

    @triton.jit  # type: ignore[untyped-decorator]
    def _remainder_one(x):  # type: (Any) -> Any
        """Keep the existing positive-divisor remainder, including its signed zero."""
        r = lib.fmod(x, 1.0)
        return tl.where(r < 0.0, r + 1.0, r)

    @triton.jit  # type: ignore[untyped-decorator]
    def _clip_unit(x):  # type: (Any) -> Any
        """Clamp a channel while preserving NaN for the later quantization."""
        return tl.where(~_is_nan(x), tl.minimum(tl.maximum(x, 0.0), 1.0), x)

    @triton.jit  # type: ignore[untyped-decorator]
    def _restore_code(x):  # type: (Any) -> Any
        """Round the uint8 code and apply CUDA's existing scalar-divisor restore."""
        x = lib.nearbyint(x * 255.0)
        x = tl.where(~_is_nan(x), tl.minimum(tl.maximum(x, 0.0), 255.0), 0.0)
        return x.to(tl.uint8).to(tl.float32) * (1.0 / 255.0)

    @triton.jit  # type: ignore[untyped-decorator]
    def _restore_kernel(images, gains, out, PIXELS, HW: tl.constexpr, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
        """Write one consumer pixel from the unchanged HSV and quantization expressions."""
        linear = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = linear < PIXELS
        image = linear // HW
        px = linear % HW
        r = tl.load(images + image * 3 * HW + px, valid, 0.0)
        g = tl.load(images + image * 3 * HW + HW + px, valid, 0.0)
        b = tl.load(images + image * 3 * HW + 2 * HW + px, valid, 0.0)
        maxc = tl.maximum(tl.maximum(r, g), b)
        minc = tl.minimum(tl.minimum(r, g), b)
        delta = maxc - minc
        safe = tl.where(delta > 0, delta, 1.0)
        rc = tl.div_rn(maxc - r, safe)
        gc = tl.div_rn(maxc - g, safe)
        bc = tl.div_rn(maxc - b, safe)
        hue = 4.0 + gc - rc
        hue = tl.where(maxc == g, 2.0 + rc - bc, hue)
        hue = tl.where(maxc == r, bc - gc, hue)
        hue = tl.where(delta > 0, _remainder_one(tl.div_rn(hue, 6.0)), 0.0)
        sat = tl.where(maxc > 0, tl.div_rn(delta, tl.where(maxc > 0, maxc, 1.0)), 0.0)
        hg = tl.load(gains + image * 3, valid, 0.0).to(tl.float32)
        sg = (1.0 + tl.load(gains + image * 3 + 1, valid, 0.0)).to(tl.float32)
        vg = (1.0 + tl.load(gains + image * 3 + 2, valid, 0.0)).to(tl.float32)
        hue = _remainder_one(hue + hg)
        sat = _clip_unit(sat * sg)
        value = _clip_unit(maxc * vg)
        h6 = hue * 6.0
        sector = tl.floor(h6)
        frac = h6 - sector
        p = value * (1.0 - sat)
        q = value * (1.0 - sat * frac)
        t = value * (1.0 - sat * (1.0 - frac))
        index = sector.to(tl.int64) % 6
        index = tl.where(index < 0, index + 6, index)
        red = tl.where(
            index == 0,
            value,
            tl.where(index == 1, q, tl.where(index == 2, p, tl.where(index == 3, p, tl.where(index == 4, t, value)))),
        )
        green = tl.where(
            index == 0,
            t,
            tl.where(
                index == 1, value, tl.where(index == 2, value, tl.where(index == 3, q, tl.where(index == 4, p, p)))
            ),
        )
        blue = tl.where(
            index == 0,
            p,
            tl.where(
                index == 1, p, tl.where(index == 2, t, tl.where(index == 3, value, tl.where(index == 4, value, q)))
            ),
        )
        nan_input = _is_nan(r) | _is_nan(g) | _is_nan(b)
        red = tl.where(nan_input, 0.0, _restore_code(red))
        green = tl.where(nan_input, 0.0, _restore_code(green))
        blue = tl.where(nan_input, 0.0, _restore_code(blue))
        tl.store(out + linear * 3, red, valid)
        tl.store(out + linear * 3 + 1, green, valid)
        tl.store(out + linear * 3 + 2, blue, valid)


@lru_cache(maxsize=1)
def _host_compiler_available() -> bool:
    """Keep the native restore when Triton's host launcher cannot be built.

    A CUDA runtime install need not carry a C compiler. Respect an explicit
    ``CC`` first, then the GCC/Clang discovery the launcher needs; the decision
    is process-wide metadata, not a scan or device synchronization per batch.
    """
    compiler = os.environ.get("CC")
    return bool(shutil.which(compiler)) if compiler else bool(shutil.which("gcc") or shutil.which("clang"))


def _restore_hsv_cuda(images: Tensor, gains: Tensor) -> Tensor | None:
    """Return the fused consumer batch when this CUDA backend supports the inputs.

    The native restore remains the fallback for missing Triton, other devices,
    non-contiguous inputs, capture scopes and other precision modes. No tensor or output buffer is
    cached: the JIT owns compiled code, while every call returns fresh image storage.
    The device guard preserves a caller's current device on multi-GPU processes.

    Args:
        images: Contiguous ``(B, 3, H, W)`` float32 geometric images.
        gains: Contiguous ``(B, 3)`` float64 stated gains on the image device.

    Returns:
        A float32 channels-last batch, or ``None`` for the native fallback.

    Examples:
        ```pycon
        >>> images = torch.zeros(1, 3, 2, 2)
        >>> gains = torch.zeros(1, 3, dtype=torch.float64)
        >>> _restore_hsv_cuda(images, gains) is None
        True

        ```
    """
    if (
        not _AVAILABLE
        or type(images) is not Tensor
        or type(gains) is not Tensor
        or not images.is_cuda
        or not _host_compiler_available()
        or torch.version.hip is not None
        or torch.compiler.is_compiling()
        or images.dtype != torch.float32
        or gains.dtype != torch.float64
        or gains.device != images.device
        or images.ndim != 4
        or images.shape[1] != 3
        or gains.shape != (images.shape[0], 3)
        or images.numel() == 0
        or not images.is_contiguous()
        or not gains.is_contiguous()
    ):
        return None
    with torch.cuda.device(images.device):
        if torch.cuda.get_device_capability(images.device)[0] < 8 or torch.cuda.is_current_stream_capturing():
            return None
        batch, _, height, width = images.shape
        result = torch.empty(images.shape, device=images.device, dtype=torch.float32, memory_format=torch.channels_last)
        pixels = batch * height * width
        _restore_kernel[(triton.cdiv(pixels, _BLOCK_SIZE),)](
            images, gains, result, pixels, height * width, _BLOCK_SIZE, enable_fp_fusion=False, num_warps=4
        )
    return result
