# SPDX-License-Identifier: Apache-2.0
"""Argument-domain checks the public entry points share (WP-171).

An annotation states what a value *is*, never which of those values means anything.
``batch_size: int`` admits ``0``, ``limit: int`` admits ``-5``, ``conf_threshold: float``
admits ``-1.0`` and a ``Literal`` decoder admits ``"E2E"`` from every caller a type
checker never sees — a parsed command line, a notebook, a script. Each of those reaches
arithmetic or a dispatch and answers **plausibly**: ``images[:-5]`` scores every image
but the last five while the banner prints the truncated count as though it were the
request, ``math.ceil(n / 0)`` raises three frames below the flag that caused it, and an
unrecognised decoder spelling takes the ``else`` branch and reports the other path's
boxes. A wrong answer with no error is the failure this module exists to convert into a
refusal.

Three functions, because three shapes cover every scalar this project's entry points
take: a lower bound, a closed interval, and a fixed vocabulary. They live here rather
than at each call site because a refusal is only worth raising if it *diagnoses*, and
one voice — the argument's own name, the constraint, then the offending value — is what
makes several refusals read as one rule rather than as several opinions. Each names the
argument the caller spelled, which is what the roadmap row asks for: a ``ValueError``
reading "invalid value" tells a caller nothing they did not already know.

What is deliberately **not** here is the canvas rule. Divisibility by the head's strides
is a property of the architecture rather than of arithmetic, so it lives with the strides
themselves, in :func:`~lucid_yolo.assign.grid.require_grid_side`.

NaN is refused by both numeric checks rather than passed through. Every comparison
against NaN is false, so a bound written the obvious way (``value < minimum``) admits it,
and a NaN threshold or gain then propagates silently into every number downstream instead
of failing at the boundary it entered through.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["require_at_least", "require_in_range", "require_one_of"]


def require_at_least(name: str, value: float, minimum: float) -> None:
    """Refuse a number below ``minimum``, or NaN, naming the argument it arrived as.

    Args:
        name: The argument or flag spelling the caller used, quoted back in the message
            so the refusal names the thing they typed.
        value: The value to check.
        minimum: The smallest value ``name`` accepts.

    Raises:
        ValueError: If ``value`` is NaN or below ``minimum``.

    Examples:
        >>> require_at_least("batch_size", 32, 1)  # in range: returns nothing
        >>> require_at_least("batch_size", 0, 1)
        Traceback (most recent call last):
            ...
        ValueError: batch_size must be a number >= 1; got 0
    """
    if math.isnan(value) or value < minimum:
        raise ValueError(f"{name} must be a number >= {minimum}; got {value}")


def require_in_range(name: str, value: float, low: float, high: float) -> None:
    """Refuse a number outside the closed interval ``[low, high]``, or NaN.

    Args:
        name: The argument or flag spelling the caller used.
        value: The value to check.
        low: Smallest accepted value, itself accepted.
        high: Largest accepted value, itself accepted.

    Raises:
        ValueError: If ``value`` is NaN or outside ``[low, high]``.

    Examples:
        >>> require_in_range("conf_threshold", 0.25, 0.0, 1.0)  # in range: returns nothing
        >>> require_in_range("conf_threshold", -0.1, 0.0, 1.0)
        Traceback (most recent call last):
            ...
        ValueError: conf_threshold must be in [0.0, 1.0]; got -0.1
    """
    if math.isnan(value) or not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]; got {value}")


def require_one_of(name: str, value: object, allowed: Sequence[object]) -> None:
    """Refuse a value outside a fixed vocabulary, naming every spelling that is in it.

    The alternatives are listed rather than merely counted: the values this guards are
    dispatch keys, and the mistakes are near-misses — a capital, a trailing space, a
    plural — which a caller fixes from the list and cannot fix from "unknown".

    Args:
        name: The argument or flag spelling the caller used.
        value: The value to check, compared by equality against ``allowed``.
        allowed: Every accepted value, in the order the message should list them.

    Raises:
        ValueError: If ``value`` equals no member of ``allowed``.

    Examples:
        >>> require_one_of("decoder", "nms", ("e2e", "nms"))  # known: returns nothing
        >>> require_one_of("decoder", "E2E", ("e2e", "nms"))
        Traceback (most recent call last):
            ...
        ValueError: decoder must be one of ('e2e', 'nms'); got 'E2E'
    """
    if value not in allowed:
        raise ValueError(f"{name} must be one of {tuple(allowed)}; got {value!r}")
