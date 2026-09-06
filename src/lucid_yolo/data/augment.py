# SPDX-License-Identifier: Apache-2.0
"""Photometric HSV jitter and horizontal flip augmentations (WP-013, blueprint section 5.9).

Two more :class:`~lucid_yolo.data.transforms.GeometricTransform` implementations from
the YOLO-lineage recipe ([R1] Table S3: ``fliplr=0.5`` plus per-channel HSV gains):

:class:`HSVJitter`
    A **photometric-only** transform. It perturbs the image in hue/saturation/value
    space and leaves every target modality untouched — geometry is invariant under a
    colour change, so the :class:`~lucid_yolo.data.targets.Targets` pass straight
    through. The HSV conversion is done in pure torch by the module-level
    :func:`rgb_to_hsv` / :func:`hsv_to_rgb` pair, written from the standard
    colorimetric (hexcone) definition so the project carries no torchvision
    dependency for this hot-path op. The two converters are exact round-trip inverses
    of each other up to float error, which the unit gate pins.

:class:`HorizontalFlip`
    A geometric mirror about the vertical axis. With probability ``p`` it reverses the
    image columns and mirrors every carried modality about ``x = (W - 1) / 2``:
    axis-aligned boxes swap and reflect their x-extent (``x1' = (W - 1) - x2``,
    ``x2' = (W - 1) - x1``), polygon points reflect (``x' = (W - 1) - x``), and
    rotated boxes go through
    :func:`~lucid_yolo.data.rotated_aug.mirror_rboxes` — centre reflected, long-edge
    direction reflected with it, result re-canonicalized. The mirror itself is *exact* for
    the carried long-edge representation, so no instance is ever dropped here and no guard
    is needed. The re-canonicalization is WP-058 closing a WP-013 defect: the bare negation
    this transform used to carry out left every box with ``theta > pi/4`` outside the
    ``[-pi/4, 3*pi/4)`` long-edge range.

    The keypoint identity swap is :func:`~fuse_augmentations.permute_keypoint_pairs` since
    WP-157, fed a full-length index this module builds from the dataset's pair list. It
    permutes *coordinates* and nothing else, so the matching permutation of
    ``keypoint_vis`` stays here — without it a mirrored sample would carry a visible point
    marked occluded and its partner marked visible, at identical shapes and with every
    coordinate assertion still passing.

Both transforms draw every random quantity from an optional :class:`torch.Generator`,
so a seeded generator gives byte-identical output (the determinism policy of blueprint
sec. 7).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from fuse_augmentations import permute_keypoint_pairs  # type: ignore[import-untyped]
from torch import Tensor

from lucid_yolo.data.rotated_aug import mirror_rboxes
from lucid_yolo.data.targets import Targets

__all__ = ["FlipParams", "HSVJitter", "HSVParams", "HorizontalFlip", "hsv_to_rgb", "rgb_to_hsv"]

#: Channel count of an RGB / HSV image (leading dimension of a CHW tensor).
_RGB_CHANNELS = 3


def _uniform(low: float, high: float, generator: torch.Generator | None) -> float:
    """Sample one float uniformly from ``[low, high]`` (returns ``low`` when degenerate).

    Args:
        low: Inclusive lower bound.
        high: Inclusive upper bound; when ``<= low`` the bound ``low`` is returned
            directly so a zeroed range (e.g. ``hsv_h=0``) is exact.
        generator: Optional RNG for reproducible sampling.

    Returns:
        A single sampled value as a Python float.

    Examples:
        ```pycon
        >>> import torch
        >>> _uniform(1.0, 1.0, None)
        1.0

        ```
    """
    if high <= low:
        return float(low)
    return float(torch.empty((), dtype=torch.float64).uniform_(low, high, generator=generator).item())


def _keypoint_flip_index(keypoint_flip_pairs: list[tuple[int, int]], keypoint_count: int) -> Tensor:
    """Expand dataset ``(left, right)`` pairs into upstream's full-length permutation.

    The two shapes disagree and the conversion is this project's, not the caller's: a
    dataset publishes the *pairs* it wants swapped (A64 keeps that the public surface, so
    ``K`` stays generic), while :func:`~fuse_augmentations.permute_keypoint_pairs` takes a
    ``(K,)`` index in which slot ``i`` takes its value from slot ``index[i]`` — identity
    slots included, not omitted, since upstream rejects an index that is not one entry per
    slot. Applying the swaps in list order to ``range(K)`` reproduces the sequential
    per-pair swap this transform used to perform, overlapping pairs included: ``[(0, 1),
    (1, 2)]`` composes to the same 3-cycle either way.

    Args:
        keypoint_flip_pairs: ``(left_index, right_index)`` pairs on the keypoint axis.
        keypoint_count: ``K``, the number of keypoint slots per instance.

    Returns:
        A ``(K,)`` ``int64`` permutation, usable as the index for both the coordinate
        permutation and the visibility one.

    Raises:
        ValueError: If either index of a pair falls outside ``[0, keypoint_count)``. A bad
            pair is a caller error whichever side does the permutation, so the check stays
            here rather than surfacing as an out-of-bounds gather from upstream.

    Examples:
        ```pycon
        >>> _keypoint_flip_index([(1, 2), (3, 4)], keypoint_count=5).tolist()
        [0, 2, 1, 4, 3]
        >>> _keypoint_flip_index([], keypoint_count=3).tolist()
        [0, 1, 2]

        ```
    """
    index = list(range(keypoint_count))
    for pair in keypoint_flip_pairs:
        left_index, right_index = pair
        if not (0 <= left_index < keypoint_count) or not (0 <= right_index < keypoint_count):
            raise ValueError(f"keypoint flip pair {pair} is out of range for K={keypoint_count}")
        index[left_index], index[right_index] = index[right_index], index[left_index]
    return torch.tensor(index, dtype=torch.int64)


def rgb_to_hsv(image: Tensor) -> Tensor:
    """Convert a CHW RGB float image to HSV, in pure torch (hexcone definition).

    Implements the standard colorimetric conversion (matching :func:`colorsys.rgb_to_hsv`
    channel-wise): value is the per-pixel channel maximum, saturation is the chroma
    relative to that maximum, and hue is the angular position on the colour hexcone
    normalised to ``[0, 1)``. Achromatic pixels (zero chroma) map to hue ``0`` and
    saturation ``0``. On red/green/blue ties the red sector wins, then green, matching
    the reference ``colorsys`` ordering.

    Args:
        image: ``(3, H, W)`` RGB image with values in ``[0, 1]``.

    Returns:
        ``(3, H, W)`` HSV image with all three channels in ``[0, 1]``, in ``image``'s
        dtype.

    Raises:
        ValueError: If ``image`` is not a ``(3, H, W)`` tensor.

    Examples:
        ```pycon
        >>> import torch
        >>> red = torch.tensor([1.0, 0.0, 0.0]).reshape(3, 1, 1)
        >>> rgb_to_hsv(red).flatten().tolist()
        [0.0, 1.0, 1.0]

        ```
    """
    if image.ndim != 3 or image.shape[0] != _RGB_CHANNELS:
        raise ValueError(f"image must be (3, H, W); got shape {tuple(image.shape)}")
    r, g, b = image[0], image[1], image[2]
    maxc = image.amax(dim=0)
    minc = image.amin(dim=0)
    delta = maxc - minc
    safe_delta = torch.where(delta > 0, delta, torch.ones_like(delta))
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta
    # Sectors applied lowest-priority first so red wins ties, then green, then blue.
    hue = torch.full_like(maxc, 4.0) + gc - rc
    hue = torch.where(maxc == g, 2.0 + rc - bc, hue)
    hue = torch.where(maxc == r, bc - gc, hue)
    hue = torch.where(delta > 0, (hue / 6.0) % 1.0, torch.zeros_like(hue))
    saturation = torch.where(
        maxc > 0, delta / torch.where(maxc > 0, maxc, torch.ones_like(maxc)), torch.zeros_like(maxc)
    )
    return torch.stack([hue, saturation, maxc], dim=0)


def hsv_to_rgb(image: Tensor) -> Tensor:
    """Convert a CHW HSV float image back to RGB, in pure torch (hexcone definition).

    Exact inverse of :func:`rgb_to_hsv` up to float rounding: hue selects one of six
    colour-hexcone sectors and the fractional position within the sector interpolates
    the RGB triple. Achromatic input (saturation ``0``) yields a neutral grey ``(v, v,
    v)`` for any hue, which the ``p == q == t == v`` degeneracy handles without a
    special case.

    Args:
        image: ``(3, H, W)`` HSV image with all channels in ``[0, 1]``.

    Returns:
        ``(3, H, W)`` RGB image with values in ``[0, 1]``, in ``image``'s dtype.

    Raises:
        ValueError: If ``image`` is not a ``(3, H, W)`` tensor.

    Examples:
        ```pycon
        >>> import torch
        >>> cyan = torch.tensor([0.5, 1.0, 1.0]).reshape(3, 1, 1)
        >>> [round(c, 4) for c in hsv_to_rgb(cyan).flatten().tolist()]
        [0.0, 1.0, 1.0]

        ```
    """
    if image.ndim != 3 or image.shape[0] != _RGB_CHANNELS:
        raise ValueError(f"image must be (3, H, W); got shape {tuple(image.shape)}")
    hue, saturation, value = image[0], image[1], image[2]
    sector = torch.floor(hue * 6.0)
    frac = hue * 6.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * frac)
    t = value * (1.0 - saturation * (1.0 - frac))
    index = sector.long() % 6
    red = _select_sector(index, [value, q, p, p, t, value])
    green = _select_sector(index, [t, value, value, q, p, p])
    blue = _select_sector(index, [p, p, t, value, value, q])
    return torch.stack([red, green, blue], dim=0)


def _select_sector(index: Tensor, choices: list[Tensor]) -> Tensor:
    """Pick, per pixel, the channel value for the hexcone sector named by ``index``.

    Args:
        index: Integer sector index in ``[0, 5]`` per pixel.
        choices: The six per-sector value maps, ordered by sector.

    Returns:
        The selected value map, same shape as ``index``.
    """
    result = choices[-1]
    for sector, value_map in enumerate(choices[:-1]):
        result = torch.where(index == sector, value_map, result)
    return result


@dataclass(frozen=True)
class HSVParams:
    """One sampled set of HSV gains, in the units the jitter applies them in.

    Attributes:
        hue: Additive hue gain, as a fraction of the hue circle.
        saturation: Multiplicative saturation gain, applied as ``1 + saturation``.
        value: Multiplicative value gain, applied as ``1 + value``.

    Examples:
        ```pycon
        >>> HSVParams(hue=0.0, saturation=0.0, value=0.0).as_tuple()
        (0.0, 0.0, 0.0)

        ```
    """

    hue: float
    saturation: float
    value: float

    def as_tuple(self) -> tuple[float, float, float]:
        """Return the three gains in ``(hue, saturation, value)`` order.

        Returns:
            The gains as a plain tuple, the shape :attr:`HSVJitter.last_gains` has
            carried since WP-013.

        Examples:
            ```pycon
            >>> HSVParams(hue=0.1, saturation=-0.2, value=0.3).as_tuple()
            (0.1, -0.2, 0.3)

            ```
        """
        return (self.hue, self.saturation, self.value)


@dataclass(frozen=True)
class FlipParams:
    """Whether one horizontal-flip call mirrors, decided by its draw.

    Attributes:
        flipped: ``True`` when the image and every modality are mirrored.

    Examples:
        ```pycon
        >>> FlipParams(flipped=True).flipped
        True

        ```
    """

    flipped: bool


class HSVJitter:
    """Photometric HSV jitter applied to the image alone (WP-013).

    Each call samples three independent gains — hue from ``[-hsv_h, hsv_h]``,
    saturation from ``[-hsv_s, hsv_s]`` and value from ``[-hsv_v, hsv_v]`` — converts
    the image to HSV, applies the hue gain **additively modulo 1**, the saturation and
    value gains **multiplicatively** as ``channel * (1 + gain)`` clamped to ``[0, 1]``,
    and converts back to RGB. Targets are geometry only and pass through unchanged.

    Args:
        hsv_h: Half-width of the additive hue-gain range, as a fraction of the hue
            circle. Defaults to ``0.015``.
        hsv_s: Half-width of the multiplicative saturation-gain range. Defaults to
            ``0.7``.
        hsv_v: Half-width of the multiplicative value-gain range. Defaults to ``0.4``.
        generator: Optional :class:`torch.Generator` for seeded, reproducible
            sampling. Defaults to ``None`` (global RNG).

    Attributes:
        last_gains: The ``(hue, saturation, value)`` gains sampled on the most recent
            call, or ``None`` before the first call.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> from lucid_yolo.data.transforms import GeometricTransform
        >>> isinstance(HSVJitter(), GeometricTransform)
        True
        >>> jitter = HSVJitter(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
        >>> image = torch.rand(3, 8, 8)
        >>> out_image, out_targets = jitter(image, Targets.empty())
        >>> torch.allclose(out_image, image, atol=1e-5)
        True

        ```
    """

    def __init__(
        self,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        generator: torch.Generator | None = None,
    ) -> None:
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.generator = generator
        self.last_gains: tuple[float, float, float] | None = None

    def __call__(self, image: Tensor, targets: Targets) -> tuple[Tensor, Targets]:
        """Jitter ``image`` in HSV space; return it with ``targets`` unchanged.

        Args:
            image: CHW RGB image tensor with values in ``[0, 1]``.
            targets: Geometry carried alongside the image; returned untouched.

        Returns:
            The colour-jittered ``(C, H, W)`` image and the original ``targets``.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> gen = torch.Generator().manual_seed(0)
            >>> jitter = HSVJitter(generator=gen)
            >>> out_image, _ = jitter(torch.rand(3, 8, 8), Targets.empty())
            >>> out_image.shape
            torch.Size([3, 8, 8])
            >>> len(jitter.last_gains)
            3

            ```
        """
        return self.apply(image, targets, self.sample())

    def sample(self) -> HSVParams:
        """Draw one :class:`HSVParams` from the configured ranges, consuming the RNG.

        The sampling half of the seam: this is the only method that touches
        :attr:`generator`, so a caller wanting stated gains rather than drawn ones
        builds :class:`HSVParams` directly (WP-147).

        Returns:
            The three sampled gains. Nothing is stashed on the instance --
            ``last_gains`` is set by :meth:`apply`.

        Examples:
            ```pycon
            >>> import torch
            >>> gen = torch.Generator().manual_seed(0)
            >>> jitter = HSVJitter(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, generator=gen)
            >>> tuple(abs(gain) for gain in jitter.sample().as_tuple())
            (0.0, 0.0, 0.0)

            ```
        """
        return HSVParams(
            hue=_uniform(-self.hsv_h, self.hsv_h, self.generator),
            saturation=_uniform(-self.hsv_s, self.hsv_s, self.generator),
            value=_uniform(-self.hsv_v, self.hsv_v, self.generator),
        )

    def apply(self, image: Tensor, targets: Targets, params: HSVParams) -> tuple[Tensor, Targets]:
        """Apply ``params`` to ``image`` in HSV space, drawing nothing.

        Args:
            image: CHW RGB image tensor with values in ``[0, 1]``.
            targets: Geometry carried alongside the image; returned untouched.
            params: The gains to apply.

        Returns:
            The colour-jittered ``(C, H, W)`` image and the original ``targets``.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> image = torch.rand(3, 8, 8)
            >>> out, _ = HSVJitter().apply(image, Targets.empty(), HSVParams(hue=0.0, saturation=0.0, value=0.0))
            >>> torch.allclose(out, image, atol=1e-5)
            True

            ```
        """
        self.last_gains = params.as_tuple()
        hsv = rgb_to_hsv(image)
        hue = (hsv[0] + params.hue) % 1.0
        saturation = (hsv[1] * (1.0 + params.saturation)).clamp(0.0, 1.0)
        value = (hsv[2] * (1.0 + params.value)).clamp(0.0, 1.0)
        out_image = hsv_to_rgb(torch.stack([hue, saturation, value], dim=0))
        return out_image, targets


class HorizontalFlip:
    """Left-right image flip with matching target mirroring (WP-013).

    With probability ``p`` (drawn per call) the image columns are reversed and every
    target modality is mirrored about the vertical axis at ``x = (W - 1) / 2`` — the
    axis the column reversal itself reflects about, since a coordinate names a sample
    point and the outermost samples sit at ``0`` and ``W - 1`` (WP-154b). Boxes swap and
    reflect their x-extent (``x1' = (W - 1) - x2``, ``x2' = (W - 1) - x1``), polygon and
    keypoint points reflect (``x' = (W - 1) - x``), and rotated boxes reflect their centre
    (``cx' = (W - 1) - cx``) with the long edge reflected and re-canonicalized (WP-058).
    When a dataset keypoint pair map is supplied, it then swaps left/right keypoint
    identities (WP-120) — coordinates and visibility flags together, so a mirrored "left
    elbow" arrives in the right slot carrying its own flag. With probability ``1 - p`` the
    image and targets pass through unchanged.

    The rotated-box mirror is exact — an isometry maps the rectangle to a rectangle, and
    ``-theta`` names the mirrored long edge up to the half turn a rectangle is invariant
    under — so nothing is dropped and, unlike
    :class:`~lucid_yolo.data.affine.RandomAffine`, this transform imposes no instance-axis
    requirement on the rotated modality.

    Args:
        p: Probability of flipping. Defaults to ``0.5``.
        generator: Optional :class:`torch.Generator` for a seeded, reproducible flip
            draw. Defaults to ``None`` (global RNG).
        keypoint_flip_pairs: Dataset-supplied ``(left_index, right_index)`` pairs on
            the keypoint axis. ``None`` mirrors every keypoint x-coordinate without
            an identity swap, the safe default when a task has no left/right symmetry.

    Attributes:
        last_flipped: Whether the most recent call flipped, or ``None`` before the
            first call.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> from lucid_yolo.data.transforms import GeometricTransform
        >>> isinstance(HorizontalFlip(), GeometricTransform)
        True
        >>> flip = HorizontalFlip(p=1.0)
        >>> image = torch.arange(4.0).reshape(1, 1, 4).expand(3, 1, 4).contiguous()
        >>> boxes = torch.tensor([[0.0, 0.0, 1.0, 2.0]])
        >>> _, out = flip(image, Targets(boxes=boxes, labels=torch.tensor([0])))
        >>> out.boxes.tolist()
        [[2.0, 0.0, 3.0, 2.0]]
        >>> pose = Targets(
        ...     boxes=torch.zeros((1, 4)), labels=torch.tensor([0]),
        ...     keypoints=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        ...     keypoint_vis=torch.tensor([[2, 1]]),
        ... )
        >>> _, mirrored = HorizontalFlip(p=1.0, keypoint_flip_pairs=[(0, 1)])(
        ...     torch.zeros((3, 2, 4)), pose
        ... )
        >>> mirrored.keypoints.tolist(), mirrored.keypoint_vis.tolist()
        ([[[0.0, 4.0], [2.0, 2.0]]], [[1, 2]])

        ```
    """

    def __init__(
        self,
        p: float = 0.5,
        generator: torch.Generator | None = None,
        keypoint_flip_pairs: list[tuple[int, int]] | None = None,
    ) -> None:
        self.p = float(p)
        self.generator = generator
        self.keypoint_flip_pairs = keypoint_flip_pairs
        self.last_flipped: bool | None = None

    def __call__(self, image: Tensor, targets: Targets) -> tuple[Tensor, Targets]:
        """Flip ``image`` and mirror ``targets`` with probability ``p``.

        Args:
            image: CHW image tensor.
            targets: Geometry to mirror when the image is flipped.

        Returns:
            Either the original ``(image, targets)`` (no flip) or the left-right
            flipped image with every target modality mirrored.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> flip = HorizontalFlip(p=0.0)
            >>> image = torch.rand(3, 4, 6)
            >>> out_image, _ = flip(image, Targets.empty())
            >>> torch.equal(out_image, image)
            True

            ```
        """
        return self.apply(image, targets, self.sample())

    def sample(self) -> FlipParams:
        """Draw whether this call flips, consuming one uniform against ``p``.

        The sampling half of the seam: this is the only method that touches
        :attr:`generator`, so a caller wanting a stated flip rather than a drawn one
        builds :class:`FlipParams` directly (WP-147).

        Returns:
            The drawn decision. Nothing is stashed on the instance --
            ``last_flipped`` is set by :meth:`apply`.

        Examples:
            ```pycon
            >>> HorizontalFlip(p=1.0).sample()
            FlipParams(flipped=True)
            >>> HorizontalFlip(p=0.0).sample()
            FlipParams(flipped=False)

            ```
        """
        return FlipParams(flipped=_uniform(0.0, 1.0, self.generator) < self.p)

    def apply(self, image: Tensor, targets: Targets, params: FlipParams) -> tuple[Tensor, Targets]:
        """Mirror ``image`` and ``targets`` when ``params`` says so, drawing nothing.

        Args:
            image: CHW image tensor.
            targets: Geometry to mirror when the image is flipped.
            params: Whether to mirror.

        Returns:
            Either the original ``(image, targets)`` or the left-right flipped image
            with every target modality mirrored.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> boxes = torch.tensor([[0.0, 0.0, 1.0, 2.0]])
            >>> t = Targets(boxes=boxes, labels=torch.tensor([0]))
            >>> _, out = HorizontalFlip().apply(torch.zeros(3, 2, 4), t, FlipParams(flipped=True))
            >>> out.boxes.tolist()
            [[2.0, 0.0, 3.0, 2.0]]

            ```
        """
        self.last_flipped = params.flipped
        if not params.flipped:
            return image, targets
        width = float(image.shape[-1])
        return image.flip(-1), self._mirror_targets(targets, width, self.keypoint_flip_pairs)

    @staticmethod
    def _mirror_targets(targets: Targets, width: float, keypoint_flip_pairs: list[tuple[int, int]] | None) -> Targets:
        """Mirror every modality about ``x = (width - 1) / 2``, then re-clip the boxes to the canvas.

        Two canvas conventions meet here, and the transform needs both:

        * The **mirror axis** is ``(width - 1) / 2``, not ``width / 2``. A coordinate names
          a sample point, the outermost samples sit at ``0`` and ``width - 1``, and that is
          the axis ``image.flip(-1)`` itself reflects about — so ``width - 1`` is the only
          choice under which a target keeps bounding the pixels it bounded before the flip
          (WP-154b). ``width`` would displace every mirrored coordinate by one pixel.
        * The **canvas extent** is ``[0, width]``, the area reading the readers and every
          other transform clamp an ``x`` coordinate to (``coco.py`` on load, ``affine.py``
          after a warp, and :func:`~fuse_augmentations.clip_bbox_xyxy` upstream).

        Their meeting point is the defect this closes: a box legitimately clamped to
        ``x2 = width`` on load mirrors to ``x1 = (width - 1) - width = -1``, one pixel off
        the canvas. The flip is the last stage of the train pipeline, so that coordinate
        goes straight to the assigner unless it is clipped here.

        The clamp is on ``x`` alone, and only for boxes. A mirror moves nothing else, so
        clamping ``y`` would edit geometry this transform never touched — a box that arrived
        off-canvas vertically is another stage's business, and silently trimming it here
        would make the flip's output depend on a height it does not otherwise read.
        Keypoints are left where the mirror put them, as
        :class:`~lucid_yolo.data.affine.RandomAffine` leaves its warped points unclipped and
        A70 treats off-canvas as a visibility question rather than a coordinate one. Rotated
        boxes are left alone because clamping a centre while keeping the extents describes a
        *different* rectangle rather than a clipped one, and the mirror is an isometry that
        cannot push a wholly on-canvas box off the canvas.

        Args:
            targets: The geometry to mirror.
            width: Canvas width in pixels; both the mirror axis and the ``x`` clamp bound.
            keypoint_flip_pairs: ``(left, right)`` keypoint identity swaps, or ``None``.

        Returns:
            The mirrored targets: box ``x`` extents on-canvas, every modality carried across.
        """
        axis = width - 1.0
        boxes = targets.boxes.clone()
        x1 = boxes[:, 0].clone()
        boxes[:, 0] = axis - boxes[:, 2]
        boxes[:, 2] = axis - x1
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, width)
        polygons = []
        for ring in targets.polygons:
            mirrored = ring.clone()
            mirrored[:, 0] = axis - ring[:, 0]
            polygons.append(mirrored)
        rboxes = mirror_rboxes(targets.rboxes, width)
        keypoints = targets.keypoints.clone()
        keypoint_vis = targets.keypoint_vis.clone()
        if keypoints.shape[0] > 0:
            keypoints[..., 0] = axis - targets.keypoints[..., 0]
            if keypoint_flip_pairs is not None:
                flip_index = _keypoint_flip_index(keypoint_flip_pairs, keypoints.shape[1])
                reversed_mask = torch.ones(keypoints.shape[0], dtype=torch.bool, device=keypoints.device)
                # Upstream moves coordinates only -- it says so, and it is the right split:
                # which slot is "left elbow" is dataset schema, and a visibility flag is not
                # geometry at all. So the same index is applied to `keypoint_vis` here, which
                # is what makes each flag follow its own point instead of staying behind on a
                # slot whose identity just changed (A70 keeps it a permutation, never a
                # demotion).
                keypoints = permute_keypoint_pairs(keypoints, flip_index, reversed_mask)
                # `_keypoint_flip_index` builds on the CPU because it is a Python-list
                # construction; upstream moves it to the keypoint device itself, and this
                # side does the same rather than inheriting a transform that only works on
                # the CPU the gate happens to run on.
                keypoint_vis = keypoint_vis.index_select(1, flip_index.to(device=keypoint_vis.device))
        # A mirror keeps the instance axis, so the per-instance R18 flags ride along.
        return Targets(
            boxes=boxes,
            labels=targets.labels.clone(),
            polygons=polygons,
            rboxes=rboxes,
            difficult=targets.difficult.clone(),
            keypoints=keypoints,
            keypoint_vis=keypoint_vis,
        )
