# SPDX-License-Identifier: Apache-2.0
"""Hold the uint8 code grid shared by the image transport and the fused CUDA restore.

The grid lives in ``data/`` rather than beside the collate in
:mod:`lucid_yolo.ptl.datamodule` because the fused kernel in
:mod:`lucid_yolo.data._hsv_cuda` quantizes onto the same grid, and ``data/`` never
imports ``ptl/``. The two conversions onto and off that grid live here with it, so the
scale and every expression using it are read in one place. Importing the module pulls in
torch and nothing else — never the optional compiler.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: uint8 quantization scale for image transport. :func:`~lucid_yolo.ptl.datamodule.collate_detection`
#: quantizes the post-augmentation float pixels (``[0, 1]``) to ``round(x * 255)`` uint8 codes so
#: the image batch crosses the DataLoader worker boundary at a quarter of its float32
#: size, and the transfer hook (or :func:`~lucid_yolo.ptl.datamodule.unpack_batch`) restores
#: them as ``code / 255`` on the destination device; the fused CUDA restore rounds and
#: restores on the same grid. ``torch.round`` is round-half-to-even, so the worst-case
#: round-trip error is one half-step — ``1 / 510``.
_PIXEL_QUANT_MAX = 255


def _quantize_images(images: Tensor) -> Tensor:
    """Round augmented float pixels to the existing uint8 transport codes."""
    return images.mul(_PIXEL_QUANT_MAX).round().clamp_(0, _PIXEL_QUANT_MAX).to(torch.uint8)


def _dequantize_images(images: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """Restore uint8 transport images to floating-point ``[0, 1]``; pass floats through.

    The inverse of the :func:`~lucid_yolo.ptl.datamodule.collate_detection` quantization:
    uint8 codes cast to ``dtype`` and divide by :data:`_PIXEL_QUANT_MAX`. The ``0``…``255``
    codes and the divisor are exactly representable in float16/bfloat16, so a
    half-precision ``dtype`` keeps the same ``1/510`` quantization bound. An already-float
    tensor (a direct, non-quantized feed) is returned untouched — its own dtype preserved —
    so the restore is safe to run over either form.

    Args:
        images: A ``(B, C, H, W)`` uint8 transport batch or an already-float batch.
        dtype: Floating-point dtype for the restored images (default
            ``torch.float32``); used only on the uint8 dequantization path.

    Returns:
        The images in ``[0, 1]`` at ``dtype`` (dequantized when the input was uint8,
        otherwise the input unchanged).
    """
    if images.dtype == torch.uint8:
        return images.to(dtype).div(_PIXEL_QUANT_MAX)
    return images
