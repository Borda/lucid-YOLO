# SPDX-License-Identifier: Apache-2.0
"""Hold the geometry widths, pad colour and scalar draw the ``data`` modules share.

Each of these used to be restated in every module that needed it. Most copies were
harmless repetition, but the pad colour is not: the letterbox pad, the mosaic canvas and
the affine border fill must be the *same* grey, or the canvas and the border it is warped
into fill with different values, and three literals can drift apart silently where one name
cannot. Every name here is private to the package, and each consumer imports it under its
old name, so ``affine._DEFAULT_PAD_VALUE`` and the rest still resolve where they always did.

The module imports nothing but torch, so any ``data`` module may import it without a cycle.
"""

from __future__ import annotations

import torch

#: Column count of a plane point ``(x, y)`` — the width every ring, quad and keypoint row
#: flattens to before a shared affine maps it.
_POINT_DIM = 2
#: Corner count of the quadrilateral form of a rotated box.
_QUAD_CORNERS = 4
#: Column count of an ``xyxy`` axis-aligned box.
_BOX_DIM = 4
#: Column count of a long-edge rotated box ``(cx, cy, w, h, theta)``.
_RBOX_DIM = 5
#: Default pad colour: mid-grey ``114/255`` per the YOLO-lineage convention (A10). The
#: letterbox pad, the mosaic canvas and the affine border fill all read this one value.
_DEFAULT_PAD_VALUE = 114.0 / 255.0


def _uniform(low: float, high: float, generator: torch.Generator | None) -> float:
    """Sample one float uniformly from ``[low, high]``, returning ``low`` when the range is empty.

    Shared by the affine, photometric and mixing transforms, which each draw their
    parameters one scalar at a time from the caller's generator.

    Args:
        low: Inclusive lower bound.
        high: Inclusive upper bound; when ``<= low`` the bound ``low`` is returned
            directly, so a zeroed range (``degrees=0``, ``hsv_h=0``, a zero probability)
            is exact and draws nothing from ``generator``.
        generator: Optional RNG for reproducible sampling.

    Returns:
        A single sampled value as a Python float.

    Examples:
        ```pycon
        >>> _uniform(0.25, 0.25, None)
        0.25
        >>> 1.0 <= _uniform(1.0, 2.0, torch.Generator().manual_seed(0)) <= 2.0
        True

        ```
    """
    if high <= low:
        return float(low)
    return float(torch.empty((), dtype=torch.float64).uniform_(low, high, generator=generator).item())
