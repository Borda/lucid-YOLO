# SPDX-License-Identifier: Apache-2.0
"""Compound-scaling registry for the YOLO26 model family (WP-023).

The five published variants (``n``/``s``/``m``/``l``/``x``) differ only by three
compound-scaling multipliers — depth ``d``, width ``w``, and the pre-width channel
cap ``mc`` (blueprint sec. 5.1, sourced from R3 Fig. 1). This module pins those
rows as a frozen :class:`ScaleSpec` dataclass registry and exposes a single
:func:`scale_spec` lookup, the one place the multipliers live (ADR-001: the
topology is typed Python, never a config DSL).

The multipliers feed the backbone/neck scaling helpers
(:func:`~open_yolos.models.backbone._scale_channels`,
:func:`~open_yolos.models.backbone._scale_repeats`) unchanged: channel widths are
``int(min(base, mc) * w)`` and per-stage repeats ``max(1, round(2 * d))``. The
resulting parameter and FLOP counts are held to R1 Table 7 by the WP-023 fidelity
gate (``tests/models/test_param_flops.py``).

Provenance: R3 Fig. 1, R1 Table 7. Assumptions: A3.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["VARIANTS", "ScaleSpec", "scale_spec"]


@dataclass(frozen=True)
class ScaleSpec:
    """Compound-scaling multipliers for one YOLO26 variant.

    Attributes:
        depth: Depth multiplier ``d`` scaling every CSP stage's inner-unit repeat
            count (``max(1, round(2 * depth))``).
        width: Width multiplier ``w`` scaling channel counts
            (``int(min(base, max_channels) * width)``).
        max_channels: Channel cap ``mc`` applied to the base channel count before
            the width multiply; bites at the deepest (base-1024) stages for the
            ``m``/``l``/``x`` variants (``mc=512``).
    """

    depth: float
    width: float
    max_channels: int


#: The five published variants keyed by name (blueprint sec. 5.1 / R3 Fig. 1).
VARIANTS: dict[str, ScaleSpec] = {
    "n": ScaleSpec(depth=0.50, width=0.25, max_channels=1024),
    "s": ScaleSpec(depth=0.50, width=0.50, max_channels=1024),
    "m": ScaleSpec(depth=0.50, width=1.00, max_channels=512),
    "l": ScaleSpec(depth=1.00, width=1.00, max_channels=512),
    "x": ScaleSpec(depth=1.00, width=1.50, max_channels=512),
}


def scale_spec(variant: str) -> ScaleSpec:
    """Look up the compound-scaling multipliers for a variant by name.

    Args:
        variant: One of ``"n"``, ``"s"``, ``"m"``, ``"l"``, ``"x"``.

    Returns:
        The frozen :class:`ScaleSpec` for ``variant``.

    Raises:
        KeyError: If ``variant`` is not one of the five published names; the
            message lists the valid names.

    Examples:
        >>> scale_spec("s")
        ScaleSpec(depth=0.5, width=0.5, max_channels=1024)
        >>> scale_spec("m").max_channels
        512
        >>> scale_spec("z")
        Traceback (most recent call last):
            ...
        KeyError: "unknown variant 'z'; expected one of n, s, m, l, x"
    """
    try:
        return VARIANTS[variant]
    except KeyError:
        valid = ", ".join(VARIANTS)
        raise KeyError(f"unknown variant {variant!r}; expected one of {valid}") from None
