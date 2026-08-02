# SPDX-License-Identifier: Apache-2.0
"""Two-image mixup blend and polygon copy-paste assembly (WP-012, blueprint sec. 5.9).

Both operations here are **assemblies**, not
:class:`~lucid_yolo.data.transforms.GeometricTransform` implementations: like
:class:`~lucid_yolo.data.mosaic.MosaicAssembly` consumes four ``(image, targets)``
pairs, these consume **two** and yield one. They deliberately do not conform to
the one-image-in / one-image-out transform protocol; they run before the
per-image geometric transforms.

:class:`Mixup` (R9-lineage, [R1] Table S3)
    With probability ``p`` it convex-blends two **same-size** images,
    ``lam * a + (1 - lam) * b``, with ``lam`` drawn from ``Beta(alpha, alpha)``,
    and concatenates both label sets **unchanged**. Detection-lineage mixup keeps
    every box from both images at full weight — there is no target-level label
    smoothing by ``lam`` (that is a classification-mixup convention); the blend
    lives in pixel space only. When not triggered, the **first** pair is returned
    untouched.

:class:`CopyPaste` (R9-lineage, [R1] Table S3)
    With probability ``p`` **per candidate instance**, polygon-carrying instances
    of image ``b`` are pasted onto image ``a``: the source polygon is rasterised
    to a pixel mask (even-odd point-in-polygon rule, :func:`_rasterize_polygon`),
    the masked pixels are copied, and the instance's box/label/polygon are appended
    to ``a``'s targets. Instances without a polygon are never pasted — the mask is
    mandatory — so the destination targets must themselves carry polygons (or be
    empty) for the merged set to stay polygon-consistent.

Scale-aware strengths ([R1] Table S3): the n-scale recipe is mildest (mixup ``0``,
copy-paste ``0.1``); larger scales grow stronger (mixup up to ``0.2``, copy-paste
up to ``0.6``). Those probabilities are supplied by the datamodule config; this
module only implements the operations.

Rotated boxes:
    Rotated-aware assembly is Phase 8 work (WP-058); a non-empty ``rboxes`` on
    either input raises :class:`NotImplementedError` rather than mangling angles.

Determinism:
    Every random quantity — the trigger draw, the ``Beta`` blend factor and the
    per-instance paste draws — comes from an optional :class:`torch.Generator`, so
    a seeded generator gives byte-identical output. The ``Beta`` factor is built
    from two ``Gamma`` draws, ``lam = X / (X + Y)`` with ``X, Y ~ Gamma(alpha, 1)``,
    via :func:`torch._standard_gamma`, the one gamma sampler that accepts a
    generator (``torch.distributions.Gamma.sample`` does not).
"""

from __future__ import annotations

import torch
from torch import Tensor

from lucid_yolo.data.targets import Targets

__all__ = ["CopyPaste", "Mixup"]

#: Mixup and copy-paste each combine exactly this many images.
_PAIR_IMAGE_COUNT = 2
#: Smallest positive float32 magnitude, used to keep the Beta divisor finite.
_TINY = torch.finfo(torch.float32).tiny


def _uniform(low: float, high: float, generator: torch.Generator | None) -> float:
    """Sample one float uniformly from ``[low, high]`` (returns ``low`` when degenerate).

    Args:
        low: Inclusive lower bound.
        high: Inclusive upper bound; when ``<= low`` the bound ``low`` is returned
            directly so a zeroed probability is exact.
        generator: Optional RNG for reproducible sampling.

    Returns:
        A single sampled value as a Python float.

    Examples:
        ```pycon
        >>> _uniform(0.25, 0.25, None)
        0.25

        ```
    """
    if high <= low:
        return float(low)
    return float(torch.empty((), dtype=torch.float64).uniform_(low, high, generator=generator).item())


def _rasterize_polygon(ring: Tensor, height: int, width: int) -> Tensor:
    """Rasterise a polygon ring to a boolean pixel mask by the even-odd rule.

    Each pixel is tested at its integer-coordinate centre ``(x, y)`` with the
    classic ray-casting (PNPOLY) even-odd crossing test, vectorised over the whole
    ``height x width`` grid: for every polygon edge, pixels whose horizontal ray to
    ``-inf`` crosses that edge flip their inside/outside parity. The test is
    ``O(P * H * W)`` in the ring's point count ``P`` — acceptable at training-time
    resolutions (e.g. 640 px) and kept deliberately simple over a scanline sweep.

    Boundary pixels (a centre lying exactly on an edge) follow PNPOLY's half-open
    convention: the left/top edges count as inside, the right/bottom as outside, so
    an axis-aligned rectangle rasterises to a clean half-open pixel block.

    Args:
        ring: ``(P, 2)`` float polygon points ``(x, y)``; ``P >= 3`` for any area.
        height: Mask height in pixels.
        width: Mask width in pixels.

    Returns:
        ``(height, width)`` boolean mask, ``True`` where a pixel centre is inside
        the polygon.

    Examples:
        ```pycon
        >>> import torch
        >>> square = torch.tensor([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
        >>> _rasterize_polygon(square, 6, 6).sum().item()
        9

        ```
    """
    ys = torch.arange(height, dtype=torch.float32).view(height, 1)
    xs = torch.arange(width, dtype=torch.float32).view(1, width)
    inside = torch.zeros((height, width), dtype=torch.bool)
    point_count = ring.shape[0]
    for i in range(point_count):
        yi = ring[i, 1]
        yj = ring[i - 1, 1]
        # A horizontal ray at row `ys` crosses edge (i-1 -> i) only where the edge
        # straddles that row; `straddles` is False for horizontal edges, so the
        # divide-by-zero below lands only on masked-out entries.
        straddles = (yi > ys) != (yj > ys)
        xi = ring[i, 0]
        xj = ring[i - 1, 0]
        x_cross = (xj - xi) * (ys - yi) / (yj - yi) + xi
        inside = inside ^ (straddles & (xs < x_cross))
    return inside


class Mixup:
    """Convex pixel blend of two same-size images with concatenated targets (WP-012).

    Each call draws a trigger uniform: with probability ``p`` the two images are
    blended ``lam * a + (1 - lam) * b`` with ``lam`` from ``Beta(alpha, alpha)`` and
    both target sets are concatenated **unchanged** (every box from both images is
    kept at full weight — detection-lineage mixup applies no ``lam`` label
    smoothing). With probability ``1 - p`` the **first** pair passes through
    untouched.

    This is an **assembly, not** a :class:`~lucid_yolo.data.transforms.GeometricTransform`:
    it consumes two ``(image, targets)`` pairs rather than one. The two images must
    share the same shape; concatenating the targets additionally requires
    consistent polygon presence (a :meth:`~lucid_yolo.data.targets.Targets.concat`
    rule).

    Rotated boxes are **not** supported: any input carrying a non-empty ``rboxes``
    raises :class:`NotImplementedError` (rotated-aware assembly is Phase 8, WP-058).

    Args:
        p: Probability of blending on a given call.
        alpha: Shared ``Beta`` concentration; both shape parameters equal ``alpha``,
            so the blend factor is symmetric about ``0.5`` and (for large ``alpha``)
            concentrated there. Defaults to ``32.0``.
        generator: Optional :class:`torch.Generator` for seeded, reproducible
            trigger and blend-factor draws. Defaults to ``None`` (global RNG).

    Attributes:
        last_lam: The blend factor used on the most recent call, or ``None`` if that
            call did not trigger (passed the first pair through) or has not run yet.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> gen = torch.Generator().manual_seed(0)
        >>> mixup = Mixup(p=1.0, generator=gen)
        >>> a = (torch.ones(3, 8, 8), Targets(boxes=torch.zeros((1, 4)), labels=torch.tensor([0])))
        >>> b = (torch.zeros(3, 8, 8), Targets(boxes=torch.ones((1, 4)), labels=torch.tensor([1])))
        >>> out_image, out_targets = mixup([a, b])
        >>> out_targets.labels.tolist()
        [0, 1]
        >>> bool(torch.allclose(out_image, torch.full_like(out_image, mixup.last_lam)))
        True

        ```
    """

    def __init__(self, p: float, alpha: float = 32.0, generator: torch.Generator | None = None) -> None:
        self.p = float(p)
        self.alpha = float(alpha)
        self.generator = generator
        self.last_lam: float | None = None

    def __call__(self, items: list[tuple[Tensor, Targets]]) -> tuple[Tensor, Targets]:
        """Blend ``items`` with probability ``p``; otherwise return the first pair.

        Args:
            items: Exactly two ``(image, targets)`` pairs. The two images must share
                shape, dtype and device; every ``targets.rboxes`` must be empty, and
                polygon presence must be consistent across both (a
                :meth:`~lucid_yolo.data.targets.Targets.concat` requirement).

        Returns:
            Either the blended ``(image, targets)`` — the convex pixel blend and the
            concatenated targets — or, when the call does not trigger, the first
            input pair untouched.

        Raises:
            ValueError: If ``items`` does not hold exactly two pairs, or the two
                images differ in shape.
            NotImplementedError: If either input carries a non-empty ``rboxes``.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> mixup = Mixup(p=0.0)
            >>> a = (torch.ones(3, 4, 4), Targets.empty())
            >>> b = (torch.zeros(3, 4, 4), Targets.empty())
            >>> out_image, _ = mixup([a, b])
            >>> torch.equal(out_image, a[0]) and mixup.last_lam is None
            True

            ```
        """
        _check_pair(items)
        (image_a, targets_a), (image_b, targets_b) = items
        if _uniform(0.0, 1.0, self.generator) >= self.p:
            self.last_lam = None
            return image_a, targets_a
        if image_a.shape != image_b.shape:
            raise ValueError(f"Mixup requires same-size images; got {tuple(image_a.shape)} and {tuple(image_b.shape)}")
        lam = self._sample_lam()
        self.last_lam = lam
        blended = lam * image_a + (1.0 - lam) * image_b
        return blended, Targets.concat([targets_a, targets_b])

    def _sample_lam(self) -> float:
        """Draw one ``Beta(alpha, alpha)`` factor as ``X / (X + Y)`` of two gamma draws."""
        concentration = torch.tensor(self.alpha, dtype=torch.float32)
        x = torch._standard_gamma(concentration, self.generator)
        y = torch._standard_gamma(concentration, self.generator)
        return float(x / (x + y).clamp(min=_TINY))


class CopyPaste:
    """Paste polygon instances from one image onto another (WP-012).

    The two ``(image, targets)`` inputs are the **destination** ``a`` and the
    **source** ``b``. For each polygon-carrying instance of ``b`` a paste draw is
    taken; with probability ``p`` (and up to ``max_paste`` pastes) the instance's
    polygon is rasterised to a pixel mask, those source pixels overwrite the
    destination, and the instance's box/label/polygon are appended to ``a``'s
    targets. Instances of ``b`` without a polygon are never pasted — the mask is
    mandatory.

    Because pasted instances always carry a polygon, the destination ``a`` must
    itself carry polygons (one ring per box) or be empty; otherwise the merged
    target set would mix polygon-present and polygon-absent instances, which
    :class:`~lucid_yolo.data.targets.Targets` rejects. This matches the segmentation
    lineage the augmentation comes from.

    This is an **assembly, not** a :class:`~lucid_yolo.data.transforms.GeometricTransform`:
    it consumes two pairs rather than one. Both images must share the same shape.

    Rotated boxes are **not** supported: any input carrying a non-empty ``rboxes``
    raises :class:`NotImplementedError` (rotated-aware assembly is Phase 8, WP-058).

    Args:
        p: Per-candidate-instance paste probability.
        max_paste: Optional cap on the number of instances pasted in one call.
            ``None`` (the default) means no cap — every candidate is drawn against
            ``p``.
        generator: Optional :class:`torch.Generator` for seeded, reproducible paste
            draws. Defaults to ``None`` (global RNG).

    Attributes:
        last_pasted: The number of instances pasted on the most recent call, or
            ``None`` before the first call.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> ring = torch.tensor([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
        >>> src = Targets(boxes=torch.tensor([[1.0, 1.0, 4.0, 4.0]]), labels=torch.tensor([5]), polygons=[ring])
        >>> a = (torch.zeros(3, 6, 6), Targets.empty())
        >>> b = (torch.ones(3, 6, 6), src)
        >>> copy_paste = CopyPaste(p=1.0)
        >>> out_image, out_targets = copy_paste([a, b])
        >>> out_targets.labels.tolist(), copy_paste.last_pasted
        ([5], 1)
        >>> bool(out_image[0, 2, 2] == 1.0)
        True

        ```
    """

    def __init__(self, p: float, max_paste: int | None = None, generator: torch.Generator | None = None) -> None:
        self.p = float(p)
        self.max_paste = None if max_paste is None else int(max_paste)
        self.generator = generator
        self.last_pasted: int | None = None

    def __call__(self, items: list[tuple[Tensor, Targets]]) -> tuple[Tensor, Targets]:
        """Paste sampled source instances onto the destination image and targets.

        Args:
            items: Exactly two ``(image, targets)`` pairs — destination first,
                source second. The two images must share shape, dtype and device;
                every ``rboxes`` must be empty. Only source instances that carry a
                polygon are paste candidates.

        Returns:
            The destination image with the pasted pixels and its targets with the
            pasted instances appended. When nothing is pasted, a copy of the
            destination image and a clone of its targets are returned.

        Raises:
            ValueError: If ``items`` does not hold exactly two pairs, or the two
                images differ in shape.
            NotImplementedError: If either input carries a non-empty ``rboxes``.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> ring = torch.tensor([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
            >>> src = Targets(boxes=torch.tensor([[1.0, 1.0, 3.0, 3.0]]), labels=torch.tensor([2]), polygons=[ring])
            >>> copy_paste = CopyPaste(p=0.0)
            >>> a = (torch.zeros(3, 5, 5), Targets.empty())
            >>> out_image, out_targets = copy_paste([a, (torch.ones(3, 5, 5), src)])
            >>> torch.equal(out_image, a[0]) and out_targets.boxes.shape[0] == 0
            True

            ```
        """
        _check_pair(items)
        (image_a, targets_a), (image_b, targets_b) = items
        if image_a.shape != image_b.shape:
            raise ValueError(
                f"CopyPaste requires same-size images; got {tuple(image_a.shape)} and {tuple(image_b.shape)}"
            )
        out_image = image_a.clone()
        selected = self._select_and_paste(out_image, image_b, targets_b)
        self.last_pasted = len(selected)
        if not selected:
            return out_image, targets_a.clone()
        return out_image, self._merge(targets_a, targets_b, selected)

    def _select_and_paste(self, out_image: Tensor, source: Tensor, targets_b: Targets) -> list[int]:
        """Draw each candidate against ``p`` (up to ``max_paste``) and paste its mask."""
        height, width = out_image.shape[1], out_image.shape[2]
        cap = len(targets_b.polygons) if self.max_paste is None else self.max_paste
        selected: list[int] = []
        for index, ring in enumerate(targets_b.polygons):
            if len(selected) >= cap:
                break
            if _uniform(0.0, 1.0, self.generator) < self.p:
                mask = _rasterize_polygon(ring, height, width)
                out_image[:, mask] = source[:, mask]
                selected.append(index)
        return selected

    @staticmethod
    def _merge(targets_a: Targets, targets_b: Targets, selected: list[int]) -> Targets:
        """Append the selected source instances' box/label/polygon to ``a``'s targets."""
        index = torch.tensor(selected, dtype=torch.int64)
        boxes = torch.cat([targets_a.boxes, targets_b.boxes[index]], dim=0)
        labels = torch.cat([targets_a.labels, targets_b.labels[index]], dim=0)
        polygons = [ring.clone() for ring in targets_a.polygons]
        polygons.extend(targets_b.polygons[i].clone() for i in selected)
        return Targets(boxes=boxes, labels=labels, polygons=polygons)


def _check_pair(items: list[tuple[Tensor, Targets]]) -> None:
    """Validate the two-item count and reject rotated boxes on either input.

    Args:
        items: The candidate list of ``(image, targets)`` pairs.

    Raises:
        ValueError: If ``items`` does not hold exactly two pairs.
        NotImplementedError: If either input carries a non-empty ``rboxes``.
    """
    if len(items) != _PAIR_IMAGE_COUNT:
        raise ValueError(f"expected exactly {_PAIR_IMAGE_COUNT} items; got {len(items)}")
    for _image, targets in items:
        if targets.rboxes.shape[0] > 0:
            raise NotImplementedError(
                "Mixup/CopyPaste do not support rotated boxes; rotated-aware assembly lands in Phase 8 (WP-058)."
            )
