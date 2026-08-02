# SPDX-License-Identifier: Apache-2.0
"""Photometric HSV jitter and horizontal flip augmentations (WP-013, blueprint sec. 5.9).

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
    image columns and mirrors every carried modality: axis-aligned boxes swap and
    reflect their x-extent (``x1' = W - x2``, ``x2' = W - x1``), polygon points reflect
    (``x' = W - x``), and rotated boxes reflect their centre (``cx' = W - cx``) with the
    angle negated (``theta' = -theta``). The mirror is *exact* for the carried
    long-edge representation, so — unlike :class:`~lucid_yolo.data.affine.RandomAffine` —
    there is no rotated-box guard here. Re-canonicalising the mirrored angle back into
    the ``[-pi/4, 3*pi/4)`` long-edge range is a separate concern that lands with the
    rest of rotated-box canonicalization in Phase 8 (WP-055/WP-058); the value carried
    out of this transform is the raw negated angle.

Both transforms draw every random quantity from an optional :class:`torch.Generator`,
so a seeded generator gives byte-identical output (the determinism policy of blueprint
sec. 7).
"""

from __future__ import annotations

import torch
from torch import Tensor

from lucid_yolo.data.targets import Targets

__all__ = ["HSVJitter", "HorizontalFlip", "hsv_to_rgb", "rgb_to_hsv"]

#: Channel count of an RGB / HSV image (leading dimension of a CHW tensor).
_RGB_CHANNELS = 3
#: Small positive floor used when a hue/saturation divisor would otherwise be zero.
_EPS = 1e-12


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
        hue_gain = _uniform(-self.hsv_h, self.hsv_h, self.generator)
        sat_gain = _uniform(-self.hsv_s, self.hsv_s, self.generator)
        val_gain = _uniform(-self.hsv_v, self.hsv_v, self.generator)
        self.last_gains = (hue_gain, sat_gain, val_gain)
        hsv = rgb_to_hsv(image)
        hue = (hsv[0] + hue_gain) % 1.0
        saturation = (hsv[1] * (1.0 + sat_gain)).clamp(0.0, 1.0)
        value = (hsv[2] * (1.0 + val_gain)).clamp(0.0, 1.0)
        out_image = hsv_to_rgb(torch.stack([hue, saturation, value], dim=0))
        return out_image, targets


class HorizontalFlip:
    """Left-right image flip with matching target mirroring (WP-013).

    With probability ``p`` (drawn per call) the image columns are reversed and every
    target modality is mirrored about the vertical axis at ``x = W / 2``: boxes swap and
    reflect their x-extent (``x1' = W - x2``, ``x2' = W - x1``), polygon points reflect
    (``x' = W - x``), and rotated-box centres reflect (``cx' = W - cx``) with the angle
    negated (``theta' = -theta``). With probability ``1 - p`` the image and targets pass
    through unchanged.

    The rotated-box mirror is exact for the carried long-edge representation, so there
    is no rotated-box guard here (contrast :class:`~lucid_yolo.data.affine.RandomAffine`).
    Re-canonicalising the negated angle into ``[-pi/4, 3*pi/4)`` is deferred to Phase 8
    (WP-055/WP-058); the raw ``-theta`` is what this transform carries out.

    Args:
        p: Probability of flipping. Defaults to ``0.5``.
        generator: Optional :class:`torch.Generator` for a seeded, reproducible flip
            draw. Defaults to ``None`` (global RNG).

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
        [[3.0, 0.0, 4.0, 2.0]]

        ```
    """

    def __init__(self, p: float = 0.5, generator: torch.Generator | None = None) -> None:
        self.p = float(p)
        self.generator = generator
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
        self.last_flipped = self._draw()
        if not self.last_flipped:
            return image, targets
        width = float(image.shape[-1])
        return image.flip(-1), self._mirror_targets(targets, width)

    def _draw(self) -> bool:
        """Return whether this call flips, sampling one uniform draw against ``p``."""
        return _uniform(0.0, 1.0, self.generator) < self.p

    @staticmethod
    def _mirror_targets(targets: Targets, width: float) -> Targets:
        """Mirror every modality about ``x = width / 2``, keeping alignment intact."""
        boxes = targets.boxes.clone()
        x1 = boxes[:, 0].clone()
        boxes[:, 0] = width - boxes[:, 2]
        boxes[:, 2] = width - x1
        polygons = []
        for ring in targets.polygons:
            mirrored = ring.clone()
            mirrored[:, 0] = width - ring[:, 0]
            polygons.append(mirrored)
        rboxes = targets.rboxes.clone()
        rboxes[:, 0] = width - rboxes[:, 0]
        rboxes[:, 4] = -rboxes[:, 4]
        return Targets(boxes=boxes, labels=targets.labels.clone(), polygons=polygons, rboxes=rboxes)
