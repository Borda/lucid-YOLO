# SPDX-License-Identifier: Apache-2.0
"""Two-image mixup blend and polygon copy-paste assembly (WP-012, blueprint section 5.9).

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
    to a pixel mask (even-odd point-in-polygon rule,
    :func:`~lucid_yolo.data.rasterize.rasterize_polygon`, imported here under its
    former private name so this module's call sites and tests are unchanged),
    the masked pixels are copied, and the instance's box/label/polygon are appended
    to ``a``'s targets. Instances without a polygon are never pasted — the mask is
    mandatory — so the destination targets must themselves carry polygons (or be
    empty) for the merged set to stay polygon-consistent.

Scale-aware strengths ([R1] Table S3): the n-scale recipe is mildest (mixup ``0``,
copy-paste ``0.1``); larger scales grow stronger (mixup up to ``0.2``, copy-paste
up to ``0.6``). Those probabilities are supplied by the datamodule config; this
module only implements the operations.

Rotated boxes (WP-058):
    The two operations part company here. :class:`Mixup` moves no geometry at all —
    it blends pixels and concatenates target sets — so rotated boxes ride through it
    untouched and canonical, on their own instance axis, exactly as boxes and labels
    do; nothing is clipped and nothing is dropped, so it needs no rotated machinery.
    :class:`CopyPaste` still rejects them, and not as deferred work: its unit of
    transfer is a rasterised **polygon mask**, and the oriented path carries no
    polygons (WP-056), so there is no rotated instance it could paste and no ring it
    could invent for one. Pasting into a rotated-carrying destination would silently
    drop that destination's ``rboxes``, which is the failure the guard prevents.

Determinism:
    Every random quantity — the trigger draw, the ``Beta`` blend factor and the
    per-instance paste draws — comes from an optional :class:`torch.Generator`, so
    a seeded generator gives byte-identical output. The ``Beta`` factor is built
    from two ``Gamma`` draws, ``lam = X / (X + Y)`` with ``X, Y ~ Gamma(alpha, 1)``,
    via :func:`torch._standard_gamma`, the one gamma sampler that accepts a
    generator (``torch.distributions.Gamma.sample`` does not).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lucid_yolo.data.rasterize import rasterize_polygon as _rasterize_polygon
from lucid_yolo.data.targets import Targets

__all__ = ["CopyPaste", "CopyPasteParams", "Mixup", "MixupParams"]

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


@dataclass(frozen=True)
class MixupParams:
    """What one mixup call decided: blend at this factor, or pass the first pair through.

    Attributes:
        lam: The convex blend factor, or ``None`` when the call does not trigger --
            the same two-valued meaning :attr:`Mixup.last_lam` has carried since
            WP-012, so a stated no-op is expressible rather than only a stated blend.

    Examples:
        ```pycon
        >>> MixupParams(lam=None).lam is None
        True

        ```
    """

    lam: float | None


@dataclass(frozen=True)
class CopyPasteParams:
    """Which source instances one copy-paste call pastes, by index.

    Attributes:
        selected: Indices into the source's ``polygons`` list, in paste order. Empty
            means the call pastes nothing.

    Examples:
        ```pycon
        >>> CopyPasteParams(selected=(0, 2)).selected
        (0, 2)

        ```
    """

    selected: tuple[int, ...]


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

    Rotated boxes are carried through unchanged (WP-058): the blend moves no
    geometry, so both inputs' ``rboxes`` are concatenated exactly as their boxes and
    labels are, still canonical, with nothing clipped or dropped.

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
                shape, dtype and device; polygon presence must be consistent across
                both (a :meth:`~lucid_yolo.data.targets.Targets.concat` requirement).
                Rotated boxes, if any, are concatenated untouched.

        Returns:
            Either the blended ``(image, targets)`` — the convex pixel blend and the
            concatenated targets — or, when the call does not trigger, the first
            input pair untouched.

        Raises:
            ValueError: If ``items`` does not hold exactly two pairs, or the two
                images differ in shape.

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
        return self.apply(items, self.sample())

    def sample(self) -> MixupParams:
        """Draw the trigger and, when it fires, the blend factor.

        The sampling half of the seam: this is the only method that touches
        :attr:`generator`, so a caller wanting a stated blend rather than a drawn one
        builds :class:`MixupParams` directly (WP-147). The trigger uniform is always
        consumed; the two gamma draws behind ``lam`` are consumed only when it fires,
        which is the draw order ``__call__`` has always had.

        Returns:
            The sampled decision: a blend factor, or ``None`` for a pass-through.

        Examples:
            ```pycon
            >>> Mixup(p=0.0).sample()
            MixupParams(lam=None)
            >>> 0.0 <= (Mixup(p=1.0).sample().lam or 0.0) <= 1.0
            True

            ```
        """
        if _uniform(0.0, 1.0, self.generator) >= self.p:
            return MixupParams(lam=None)
        return MixupParams(lam=self._sample_lam())

    def apply(self, items: list[tuple[Tensor, Targets]], params: MixupParams) -> tuple[Tensor, Targets]:
        """Blend ``items`` at ``params``' factor, or pass the first pair through.

        Args:
            items: Exactly two ``(image, targets)`` pairs, as :meth:`__call__`
                requires.
            params: The blend factor, or ``None`` for a pass-through.

        Returns:
            Either the blended ``(image, targets)`` or the first input pair
            untouched.

        Raises:
            ValueError: If the two images differ in shape, checked only when
                ``params`` calls for a blend -- a pass-through never reads the second
                image, which is the behaviour ``__call__`` has always had.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> a = (torch.ones(3, 4, 4), Targets.empty())
            >>> b = (torch.zeros(3, 4, 4), Targets.empty())
            >>> out_image, _ = Mixup(p=1.0).apply([a, b], MixupParams(lam=0.25))
            >>> float(out_image[0, 0, 0])
            0.25

            ```
        """
        (image_a, targets_a), (image_b, targets_b) = items
        if params.lam is None:
            self.last_lam = None
            return image_a, targets_a
        if image_a.shape != image_b.shape:
            raise ValueError(f"Mixup requires same-size images; got {tuple(image_a.shape)} and {tuple(image_b.shape)}")
        self.last_lam = params.lam
        blended = params.lam * image_a + (1.0 - params.lam) * image_b
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

    Rotated boxes are **not** supported, for the reason the module docstring gives:
    the paste unit is a polygon mask and the oriented path carries no polygons
    (WP-056), so any input with a non-empty ``rboxes`` raises
    :class:`NotImplementedError` rather than have its rotated modality dropped.

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
        _reject_rboxes(items)
        return self.apply(items, self.sample(len(items[1][1].polygons)))

    def sample(self, candidate_count: int) -> CopyPasteParams:
        """Draw each of ``candidate_count`` candidates against ``p``, up to ``max_paste``.

        The sampling half of the seam: this is the only method that touches
        :attr:`generator`, so a caller wanting a stated selection rather than a drawn
        one builds :class:`CopyPasteParams` directly (WP-147). The draw count depends
        on the candidate count and the cap alone -- never on pixels -- which is what
        lets the loop split from the paste it used to interleave with.

        Args:
            candidate_count: Number of polygon-carrying source instances, i.e.
                ``len(targets_b.polygons)``.

        Returns:
            The selected indices, in paste order. Nothing is stashed on the instance
            -- ``last_pasted`` is set by :meth:`apply`.

        Examples:
            ```pycon
            >>> CopyPaste(p=1.0).sample(3).selected
            (0, 1, 2)
            >>> CopyPaste(p=1.0, max_paste=1).sample(3).selected
            (0,)
            >>> CopyPaste(p=0.0).sample(3).selected
            ()

            ```
        """
        cap = candidate_count if self.max_paste is None else self.max_paste
        selected: list[int] = []
        for index in range(candidate_count):
            if len(selected) >= cap:
                break
            if _uniform(0.0, 1.0, self.generator) < self.p:
                selected.append(index)
        return CopyPasteParams(selected=tuple(selected))

    def apply(self, items: list[tuple[Tensor, Targets]], params: CopyPasteParams) -> tuple[Tensor, Targets]:
        """Paste the instances ``params`` names onto the destination, drawing nothing.

        Args:
            items: Exactly two ``(image, targets)`` pairs -- destination first,
                source second -- as :meth:`__call__` requires.
            params: The source instance indices to paste, in paste order.

        Returns:
            The destination image with the pasted pixels and its targets with the
            pasted instances appended. When ``params`` selects nothing, a copy of the
            destination image and a clone of its targets are returned.

        Raises:
            ValueError: If the two images differ in shape.
            NotImplementedError: If either input carries a non-empty ``rboxes``. The guard
                is repeated here rather than left to :meth:`__call__` because ``apply`` is
                its own entrance (WP-147): a caller building :class:`CopyPasteParams`
                directly never passes through the wrapper, and :meth:`_merge` would drop
                the modality instead of refusing it. :meth:`__call__` keeps its own copy so
                that a rotated input is rejected *before* :meth:`sample` consumes a draw.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> ring = torch.tensor([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
            >>> src = Targets(boxes=torch.tensor([[1.0, 1.0, 4.0, 4.0]]), labels=torch.tensor([5]), polygons=[ring])
            >>> items = [(torch.zeros(3, 6, 6), Targets.empty()), (torch.ones(3, 6, 6), src)]
            >>> _, out = CopyPaste(p=1.0).apply(items, CopyPasteParams(selected=(0,)))
            >>> out.labels.tolist()
            [5]

            ```
        """
        _reject_rboxes(items)
        (image_a, targets_a), (image_b, targets_b) = items
        if image_a.shape != image_b.shape:
            raise ValueError(
                f"CopyPaste requires same-size images; got {tuple(image_a.shape)} and {tuple(image_b.shape)}"
            )
        out_image = image_a.clone()
        selected = list(params.selected)
        self._paste_selected(out_image, image_b, targets_b, selected)
        self.last_pasted = len(selected)
        if not selected:
            return out_image, targets_a.clone()
        return out_image, self._merge(targets_a, targets_b, selected)

    @staticmethod
    def _paste_selected(out_image: Tensor, source: Tensor, targets_b: Targets, selected: list[int]) -> None:
        """Rasterise each selected ring and copy its source pixels onto ``out_image`` in place."""
        height, width = out_image.shape[1], out_image.shape[2]
        for index in selected:
            mask = _rasterize_polygon(targets_b.polygons[index], height, width)
            out_image[:, mask] = source[:, mask]

    @staticmethod
    def _merge(targets_a: Targets, targets_b: Targets, selected: list[int]) -> Targets:
        """Append the selected source instances to ``a``'s targets, carrying every channel.

        Every field on the shared instance axis is taken by the same ``index`` the boxes
        are, so the merged set is aligned by construction. ``difficult`` is one of them
        (A51): a rebuild that omitted it would not raise, because
        :class:`~lucid_yolo.data.targets.Targets` refills an omitted flag column with one
        ``False`` per instance — which is exactly what makes the loss silent, and what
        A48's discard rule then misreads as an ordinary false positive.

        ``rboxes`` is not carried, and that is deliberate rather than an omission: a paste
        transfers a rasterised polygon mask, the oriented path carries no polygons
        (WP-056), and :func:`_reject_rboxes` therefore refuses a rotated input at both
        entrances before it can reach here. Carrying the modality would be claiming to
        paste something the transform cannot paste.

        Args:
            targets_a: The destination targets, kept in full and in order.
            targets_b: The source targets the pastes were selected from.
            selected: The source instance indices to append, in paste order.

        Returns:
            The merged targets: ``a``'s instances followed by the selected ones.
        """
        index = torch.tensor(selected, dtype=torch.int64)
        boxes = torch.cat([targets_a.boxes, targets_b.boxes[index]], dim=0)
        labels = torch.cat([targets_a.labels, targets_b.labels[index]], dim=0)
        difficult = torch.cat([targets_a.difficult, targets_b.difficult[index]], dim=0)
        polygons = [ring.clone() for ring in targets_a.polygons]
        polygons.extend(targets_b.polygons[i].clone() for i in selected)
        keypoints, keypoint_vis = CopyPaste._merge_keypoints(targets_a, targets_b, index)
        return Targets(
            boxes=boxes,
            labels=labels,
            polygons=polygons,
            difficult=difficult,
            keypoints=keypoints,
            keypoint_vis=keypoint_vis,
        )

    @staticmethod
    def _merge_keypoints(targets_a: Targets, targets_b: Targets, index: Tensor) -> tuple[Tensor, Tensor]:
        """Carry the pasted instances' points across with their boxes (WP-132).

        A paste moves an instance between images without moving it *within* one — the
        masked pixels land at the same coordinates they occupied in the source — so the
        points need no warp, only selection by the same ``index`` the boxes take. That is
        also why A70 never arises here: nothing is displaced, so nothing can be pushed
        off-canvas.

        Presence must agree, exactly as it must for polygons: a merged set that mixed
        point-carrying and point-free instances is not representable, and
        :class:`~lucid_yolo.data.targets.Targets` rejects it on construction rather than
        letting the caller discover a half-annotated batch later. The rule is the one
        :meth:`~lucid_yolo.data.targets.Targets.concat` already states for this container.

        Args:
            targets_a: The destination targets.
            targets_b: The source targets the pastes were selected from.
            index: ``(P,)`` int64 indices of the pasted source instances.

        Returns:
            The merged ``(keypoints, keypoint_vis)`` pair, or the canonical empties when
            neither side carries points — the no-op every polygon-based segmentation run
            takes, since copy-paste needs rings and those tasks bring no landmarks.

        Raises:
            ValueError: If both sides carry points but disagree on ``K``.
        """
        source_points = targets_b.keypoints.shape[0] > 0
        if not source_points:
            return targets_a.keypoints.clone(), targets_a.keypoint_vis.clone()
        pasted, pasted_vis = targets_b.keypoints[index], targets_b.keypoint_vis[index]
        if targets_a.keypoints.shape[0] == 0:
            return pasted.clone(), pasted_vis.clone()
        if targets_a.keypoints.shape[1] != pasted.shape[1]:
            raise ValueError(
                f"cannot paste instances with differing keypoint counts: {targets_a.keypoints.shape[1]} "
                f"and {pasted.shape[1]}"
            )
        return (
            torch.cat([targets_a.keypoints, pasted], dim=0),
            torch.cat([targets_a.keypoint_vis, pasted_vis], dim=0),
        )


def _check_pair(items: list[tuple[Tensor, Targets]]) -> None:
    """Validate that exactly two ``(image, targets)`` pairs were supplied.

    Args:
        items: The candidate list of ``(image, targets)`` pairs.

    Raises:
        ValueError: If ``items`` does not hold exactly two pairs.

    Examples:
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> _check_pair([(torch.zeros(3, 2, 2), Targets.empty())] * 2)
    """
    if len(items) != _PAIR_IMAGE_COUNT:
        raise ValueError(f"expected exactly {_PAIR_IMAGE_COUNT} items; got {len(items)}")


def _reject_rboxes(items: list[tuple[Tensor, Targets]]) -> None:
    """Reject rotated boxes on either copy-paste input (module docstring, WP-058).

    Args:
        items: The two ``(image, targets)`` pairs, destination first.

    Raises:
        NotImplementedError: If either input carries a non-empty ``rboxes``.

    Examples:
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> _reject_rboxes([(torch.zeros(3, 2, 2), Targets.empty())] * 2)
    """
    for _image, targets in items:
        if targets.rboxes.shape[0] > 0:
            raise NotImplementedError(
                "CopyPaste does not support rotated boxes: it transfers rasterised polygon masks and the "
                "oriented path carries no polygons (WP-056), so a rotated instance has nothing to paste."
            )
