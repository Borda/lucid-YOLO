# SPDX-License-Identifier: Apache-2.0
"""Hold the uint8 code scale shared by the image transport and the fused CUDA restore.

The constant lives in ``data/`` rather than beside the collate in
:mod:`lucid_yolo.ptl.datamodule` because the fused kernel in
:mod:`lucid_yolo.data._hsv_cuda` quantizes onto the same grid, and ``data/`` never
imports ``ptl/``. Importing it costs nothing: it pulls in neither the optional
compiler nor torch.
"""

#: uint8 quantization scale for image transport. :func:`~lucid_yolo.ptl.datamodule.collate_detection`
#: quantizes the post-augmentation float pixels (``[0, 1]``) to ``round(x * 255)`` uint8 codes so
#: the image batch crosses the DataLoader worker boundary at a quarter of its float32
#: size, and the transfer hook (or :func:`~lucid_yolo.ptl.datamodule.unpack_batch`) restores
#: them as ``code / 255`` on the destination device; the fused CUDA restore rounds and
#: restores on the same grid. ``torch.round`` is round-half-to-even, so the worst-case
#: round-trip error is one half-step — ``1 / 510``.
_PIXEL_QUANT_MAX = 255
