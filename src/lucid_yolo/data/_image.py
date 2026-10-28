# SPDX-License-Identifier: Apache-2.0
"""Decode an image file to RGB and scale it into ``[0, 1]`` by its own bit depth.

Both dataset readers — :class:`~lucid_yolo.data.coco.CocoDetectionDataset` and
:class:`~lucid_yolo.data.yolo.YoloDetectionDataset` — decode through
:func:`torchvision.io.read_image`, which keeps a source's depth: ``uint8`` for an 8-bit file,
``uint16`` for a 16-bit PNG. Each used to divide whatever it decoded by the 8-bit 255, so a
16-bit code of 60000 read as 235.3 rather than 0.9155, and the uint8 image transport clamped
it to white after the photometric transforms had already run on the out-of-range value.

:func:`to_unit_float` divides by the decoded dtype's own maximum instead. For ``uint8`` that
maximum is 255, so every 8-bit pixel reads exactly as it always did.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torchvision.io import ImageReadMode, read_image

if TYPE_CHECKING:
    from pathlib import Path


def decode_rgb(path: Path) -> Tensor:
    """Decode ``path`` into a ``(3, H, W)`` RGB tensor at the source's own depth.

    Args:
        path: Image file to decode.

    Returns:
        The decoded tensor: ``uint8`` for an 8-bit source, ``uint16`` for a 16-bit PNG.

    Raises:
        OSError: If the file is missing or cannot be decoded; the path is named.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from torchvision.io import write_png
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "a.png"
        ...     _ = write_png(torch.zeros(3, 4, 6, dtype=torch.uint8), str(path))
        ...     decoded = decode_rgb(path)
        >>> tuple(decoded.shape), decoded.dtype
        ((3, 4, 6), torch.uint8)

        ```
    """
    try:
        raw = read_image(str(path), ImageReadMode.RGB)
    except (RuntimeError, OSError) as error:
        raise OSError(f"failed to read image {path}: {error}") from error
    return cast("Tensor", raw)


def to_unit_float(raw: Tensor) -> Tensor:
    """Scale unsigned integer image codes into float32 ``[0, 1]`` by their dtype's maximum.

    Args:
        raw: Decoded image codes of an unsigned integer dtype, such as ``uint8`` or ``uint16``.

    Returns:
        ``raw`` as float32 divided by ``torch.iinfo(raw.dtype).max``: 255 for ``uint8``, so an
        8-bit image reads exactly as ``code / 255.0``, and 65535 for ``uint16``.

    Raises:
        TypeError: If ``raw`` is a floating-point, complex, signed or boolean tensor, none of
            which carries an unsigned code range to scale from.

    Examples:
        ```pycon
        >>> to_unit_float(torch.tensor([0, 255], dtype=torch.uint8)).tolist()
        [0.0, 1.0]
        >>> round(to_unit_float(torch.tensor([60000], dtype=torch.uint16)).item(), 4)
        0.9155

        ```
    """
    if raw.dtype.is_signed or raw.dtype == torch.bool:
        # `is_signed` is also true of every floating-point and complex dtype.
        raise TypeError(f"expected unsigned integer image codes, got {raw.dtype}")
    return raw.to(torch.float32) / float(torch.iinfo(raw.dtype).max)
