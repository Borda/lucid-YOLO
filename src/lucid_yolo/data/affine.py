# SPDX-License-Identifier: Apache-2.0
"""Random affine augmentation for boxes and instance masks (WP-010, blueprint section 5.9).

Random affine is the geometric workhorse of the YOLO-lineage recipe: it samples a
rotation, an anisotropic shear, a uniform scale and a translation, composes them
about the canvas centre into one ``3x3`` pixel-space matrix, and applies that
single matrix to *both* the image and every target modality. Keeping one matrix
as the source of geometric truth — routed through
:func:`~lucid_yolo.data.transforms.apply_affine_to_points` exactly as WP-009's
:class:`~lucid_yolo.data.letterbox.Letterbox` does — is what keeps boxes, polygons
and (eventually) rotated boxes mutually consistent rather than drifting across
three parallel reimplementations (blueprint sec. 5.9: "all geometric transforms
must operate consistently on boxes, polygon/instance masks, and rotated boxes").

Image warp (WP-156):
    The warp is ``fuse``'s and the geometry is this project's. What stays here is R1
    Table S3's ranges, the sampling of them, and the matrix they compose to;
    :func:`~fuse_augmentations.build_segments` owns the resampling — fusion, matrix
    inversion, the sampling grid, the constant fill — behind
    :class:`~fuse_augmentations.TransformAdapter`, the seam the package publishes for
    a caller with its own parameterisation.

    :class:`_StatedGeometryAdapter` is this project's implementation of that seam. Its
    ``build_matrix`` answers with :meth:`AffineParams.matrix`, so the matrix reaching
    the pixels is the one reaching the boxes and neither moved when the engine
    changed. Handing *parameters* to upstream instead would have re-composed them
    under its own decomposition — two sequential shears after the rotation, against
    one combined shear before it — which agrees with this one only at zero shear.
    Out-of-canvas samples take the same ``114/255`` grey as letterbox, spelled
    ``fill=`` with ``padding_mode="zeros"``.

Targets:
    When polygons are present each ring is transformed point-wise, clamped to the
    canvas, and its enclosing box is recomputed from the clamped ring via
    :func:`~lucid_yolo.data.transforms.boxes_from_polygons` — so mask/box agreement
    holds by construction. With no polygons each box's four corners are warped and
    the axis-aligned extent is taken, then clamped. Instances are filtered by a
    minimum clipped side length (``min_box_size``) and a minimum kept-area fraction
    (``min_visibility`` = clipped-extent area / pre-clip-extent area); the single
    keep mask is applied across boxes, labels and polygons through
    :meth:`~lucid_yolo.data.targets.Targets.filter`.

Keypoints (WP-132):
    Keypoints ride the same matrix as everything else and are the one modality that is
    **not** clipped afterwards. This transform manufactures the case A70 decides — it
    translates by up to 10% of the canvas and scales, so a point can land off-canvas while
    its box still clears ``min_visibility`` — and A70's answer is to carry the point
    through unchanged rather than clamp it to the edge or demote it to invisible. The
    argument is in :meth:`RandomAffine._warp_keypoints`. Rows still drop with their
    instance, since the keep mask runs on the shared axis.

Rotated boxes (WP-058):
    A general affine does not map a rectangle to a rectangle — the sampled shear
    sends one to a parallelogram — so a rotated box cannot be warped by
    transporting ``(w, h, theta)``. The rotated path instead expands each box to
    the four corners the image warp moves, pushes them through the same matrix,
    and re-fits a canonical long-edge box
    (:func:`~lucid_yolo.data.rotated_aug.warp_rboxes`); the fit is exact under a
    similarity and approximate under shear, as that module states. The warped box
    is then clipped to the canvas with WP-057's clipper, ``boxes`` is recomputed
    as the envelope of the rotated geometry, and the *same* ``min_box_size`` /
    ``min_visibility`` rule the axis-aligned path uses decides which instances
    survive — one policy over both modalities (A40). Because that rule drops
    instances, the rotated path requires WP-056's instance-axis invariant
    (``rboxes`` 1:1 with ``boxes``, no polygons) and rejects targets that break it.

Testability:
    Sampling is fully driven by an optional :class:`torch.Generator`, so a seeded
    generator gives byte-identical results. After each call the sampled
    :class:`AffineParams` and the resulting forward matrix are stashed on
    :attr:`RandomAffine.last_params` / :attr:`RandomAffine.last_matrix` for
    introspection (recovering the sampled translation, asserting determinism, etc.).

Fused letterbox (WP-070, delegated WP-156):
    Training warps a source canvas by this random affine and then letterboxes it down
    to ``img_size``, which as two transforms is two bilinear resamples. Passing
    ``letterbox=`` to :class:`RandomAffine` appends upstream's deterministic
    aspect-preserving fit to the same pipeline: it sits after the geometry and before
    any colour operation, so the two compose into one matrix and one resample. The
    local class that used to perform that composition is gone — this row and WP-155
    composed inside ``fuse`` *are* that composition.

    Targets are still clipped and filtered at the **source** canvas and only then
    mapped through the letterbox affine by
    :meth:`~lucid_yolo.data.letterbox.Letterbox.warp_targets`. That order is not
    incidental: the source canvas maps onto the letterboxed canvas's *content*
    region, so clipping after the letterbox would clip against the padded canvas
    instead and keep instances the two-transform path drops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn, cast

import torch
from fuse_augmentations import (  # type: ignore[import-untyped]
    FusedAffineSegment,
    TransformCategory,
    build_segments,
    instance_keep_mask,
    letterbox_matrix,
    transform_bbox_xyxy,
)
from torch import Tensor

from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.rotated_aug import check_rotated_pairing, clip_rboxes_to_canvas, rbox_envelopes, warp_rboxes
from lucid_yolo.data.targets import Targets
from lucid_yolo.data.transforms import apply_affine_to_points, boxes_from_polygons

__all__ = ["AffineParams", "RandomAffine"]

#: Default pad colour: mid-grey ``114/255`` per the YOLO-lineage convention (matches letterbox).
_DEFAULT_PAD_VALUE = 114.0 / 255.0
#: Lower bound on the sampled scale factor, keeping the matrix invertible (``> 0``).
_MIN_SCALE = 1e-3
#: Reused axis-size guards.
_POINT_DIM = 2
#: Fill in the validated form ``build_segments`` takes: a tuple, length one for a scalar
#: broadcast across channels. Requires ``padding_mode="zeros"`` (WP-154b).
_FILL = (_DEFAULT_PAD_VALUE,)
#: XOR salt separating the discarded-gate stream's seed from the caller's own, so the two
#: streams are decorrelated while both remain a function of the one seed the caller set.
#: An arbitrary constant -- nothing downstream reads a gate value (L-32).
_GATE_SEED_SALT = 0x9E3779B97F4A7C15


def _uniform(low: float, high: float, generator: torch.Generator | None) -> float:
    """Sample one float uniformly from ``[low, high]`` (returns ``low`` when degenerate).

    Args:
        low: Inclusive lower bound.
        high: Inclusive upper bound; when ``<= low`` the bound ``low`` is returned
            directly so a zeroed range (e.g. ``degrees=0``) is exact.
        generator: Optional RNG for reproducible sampling.

    Returns:
        A single sampled value as a Python float.
    """
    if high <= low:
        return float(low)
    return float(torch.empty((), dtype=torch.float64).uniform_(low, high, generator=generator).item())


@dataclass(frozen=True)
class AffineParams:
    """One sampled affine transform, in the units the matrix is built from.

    Attributes:
        angle: Rotation angle in radians.
        shear_x: Horizontal shear angle in radians.
        shear_y: Vertical shear angle in radians.
        scale: Uniform scale factor (``> 0``).
        translate_x: Horizontal translation in pixels.
        translate_y: Vertical translation in pixels.

    Examples:
        ```pycon
        >>> import torch
        >>> p = AffineParams(
        ...     angle=0.0, shear_x=0.0, shear_y=0.0, scale=1.0, translate_x=2.0, translate_y=3.0
        ... )
        >>> p.matrix(height=10, width=10)
        tensor([[1., 0., 2.],
                [0., 1., 3.],
                [0., 0., 1.]], dtype=torch.float64)

        ```
    """

    angle: float
    shear_x: float
    shear_y: float
    scale: float
    translate_x: float
    translate_y: float

    def matrix(self, height: int, width: int) -> Tensor:
        """Build the ``3x3`` forward (source-to-destination) pixel matrix.

        The transform is composed about the canvas centre in the order
        centre → (rotate+scale) → shear → translate → un-centre, so a source pixel
        ``p`` maps to ``A @ (p - c) + t + c`` where ``A`` is the ``2x2`` linear part,
        ``c`` the canvas centre and ``t`` the translation.

        This composition is **this project's**, and WP-156 kept it that way while
        delegating the warp: it reaches upstream's resampling engine through
        :class:`_StatedGeometryAdapter`, which hands this matrix back as the
        segment's ``build_matrix`` answer. Upstream's own direct-parameter adapter
        decomposes an affine differently — two sequential shears applied *after* the
        rotation, where this applies one combined shear *before* it — and the two
        agree only at zero shear. Routing the decomposition through the adapter seam
        rather than through a parameter dictionary is what lets the engine change
        without the geometry moving.

        The centre is the **pixel centre** ``((W - 1) / 2, (H - 1) / 2)``, not the
        canvas corner midpoint ``(W / 2, H / 2)``: a coordinate names a sample
        point, so the outermost samples sit at ``0`` and ``W - 1`` and their
        midpoint is what a rotation leaves fixed. This is the convention
        ``fuse-augmentations`` composes about (WP-154b), and one convention shared
        between image and coordinate transport is what keeps the two agreeing.

        Args:
            height: Canvas height in pixels (sets the centre and vertical axis).
            width: Canvas width in pixels (sets the centre and horizontal axis).

        Returns:
            The ``(3, 3)`` float64 forward affine matrix.

        Examples:
            ```pycon
            >>> import torch
            >>> p = AffineParams(
            ...     angle=0.0, shear_x=0.0, shear_y=0.0, scale=2.0, translate_x=0.0, translate_y=0.0
            ... )
            >>> torch.allclose(p.matrix(4, 4)[:2, :2], 2.0 * torch.eye(2, dtype=torch.float64))
            True

            ```
        """
        cos_a = math.cos(self.angle) * self.scale
        sin_a = math.sin(self.angle) * self.scale
        rotate_scale = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float64)
        shear = torch.tensor([[1.0, math.tan(self.shear_x)], [math.tan(self.shear_y), 1.0]], dtype=torch.float64)
        linear = rotate_scale @ shear
        center = torch.tensor([(width - 1) / 2.0, (height - 1) / 2.0], dtype=torch.float64)
        translate = torch.tensor([self.translate_x, self.translate_y], dtype=torch.float64)
        offset = center - linear @ center + translate
        matrix = torch.eye(3, dtype=torch.float64)
        matrix[:2, :2] = linear
        matrix[:2, 2] = offset
        return matrix


class _StatedAffine:
    """One already-drawn affine, in the shape upstream's segment machinery classifies.

    A ``fuse`` segment is built from *transform objects* it hands back to an adapter,
    so the object this project passes carries what this project has: the
    :class:`AffineParams` :meth:`RandomAffine.sample` already drew. Nothing here is
    random — the draw happened before the pipeline was built, which is what keeps the
    WP-147 seam intact with the warp delegated.

    Attributes:
        params: The affine this transform states.
    """

    #: Applied to every sample: a stated transform has no probability to gate on.
    prob = 1.0
    #: One parameter set covers the batch — there is one, and it is not drawn per item.
    same_on_batch = True

    def __init__(self, params: AffineParams) -> None:
        self.params = params


class _StatedLetterbox:
    """The deterministic aspect-preserving fit, as one ``CROP_RESIZE_FIXED`` transform.

    Classified as a shape-changing crop-resize so upstream fuses it into the geometric
    run ahead of it rather than resampling twice: the segment composes
    ``M_letterbox @ M_affine`` and applies one ``grid_sample`` at the output size.

    Attributes:
        out_h: Output canvas height in pixels.
        out_w: Output canvas width in pixels.
        allow_upscale: Whether the fit may enlarge content past its native size.
    """

    #: Deterministic and shape-changing: applied to every sample, identically.
    prob = 1.0
    same_on_batch = True

    def __init__(self, out_h: int, out_w: int, allow_upscale: bool) -> None:
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.allow_upscale = bool(allow_upscale)


class _StatedGeometryAdapter:
    """Bridge this project's own composition to ``fuse``'s resampling engine (WP-156).

    :class:`~fuse_augmentations.TransformAdapter` is the seam the package exposes for
    exactly this: an adapter answers *what category is this transform*, *what are its
    parameters* and *what matrix does it compose to*, and the engine owns everything
    after that — fusion, inversion, the sampling grid, the fill, the target routing.
    That protocol is structural, so this class conforms by shape rather than by
    inheritance — the package ships untyped, and subclassing an ``Any`` is a fiction
    a strict checker is right to reject. ``test_aug_contract.py`` asserts the
    conformance against the published ``@runtime_checkable`` protocol instead, which
    is the check that would actually catch upstream adding a required method.
    Upstream's own :class:`_DirectParamAdapter` is one implementation of that seam, and
    its ``rotation → scale → shear_x → shear_y → translate`` decomposition is that
    adapter's choice rather than the package's convention.

    Implementing the seam here is what makes the delegation move no geometry.
    :meth:`build_matrix` returns :meth:`AffineParams.matrix` — R1 Table S3's parameters
    composed the way this project has always composed them — so the matrix that reaches
    the pixels is the matrix that reaches the boxes, unchanged from before the swap.
    Handing the *parameters* to upstream instead would have re-composed them under a
    different decomposition, which agrees with this one only at zero shear.

    :meth:`sample_params` draws nothing: the draw is :meth:`RandomAffine.sample`'s, on
    the caller's generator, and by the time a segment exists the values are settled.
    The ``generator`` keyword is accepted (and ``supports_generator`` set) so that the
    engine's own per-transform activation gate — drawn even at ``prob = 1.0`` — has a
    generator to draw from instead of falling back to the global stream.
    """

    #: Lets the engine pass its generator down rather than rejecting it at the segment
    #: boundary; this adapter draws nothing with it (see the class docstring).
    supports_generator = True

    def category(self, transform: object) -> TransformCategory:
        """Classify one transform for the segment planner.

        Args:
            transform: A :class:`_StatedAffine` or :class:`_StatedLetterbox`.

        Returns:
            ``CROP_RESIZE_FIXED`` for the letterbox — the category that fuses into the
            geometric run before it — and ``GEOMETRIC_INTERP`` for the affine.
        """
        if isinstance(transform, _StatedLetterbox):
            return TransformCategory.CROP_RESIZE_FIXED
        return TransformCategory.GEOMETRIC_INTERP

    def sample_params(
        self,
        transform: object,
        input_shape: tuple[int, int, int, int],
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """Return the settled parameters for one transform, drawing nothing.

        Args:
            transform: A :class:`_StatedAffine` or :class:`_StatedLetterbox`.
            input_shape: ``(batch, channels, height, width)`` of the image being warped.
            device: Device the returned tensors must live on.
            generator: Accepted for the protocol's benefit and deliberately unused.

        Returns:
            The output size for a letterbox — the two keys the fused crop segment reads
            — and an empty mapping for the affine, whose parameters ride on the
            transform object itself.
        """
        if isinstance(transform, _StatedLetterbox):
            batch = input_shape[0]
            return {
                "target_h": torch.full((batch,), transform.out_h, device=device, dtype=torch.int64),
                "target_w": torch.full((batch,), transform.out_w, device=device, dtype=torch.int64),
            }
        return {}

    def build_matrix(
        self,
        transform: object,
        params: dict[str, Tensor],
        height: int,
        width: int,
    ) -> Tensor:
        """Return the ``(1, 3, 3)`` forward pixel matrix for one transform.

        The batch axis is length one and the engine expands it, since both transforms
        here are one stated thing rather than a per-item draw.

        Args:
            transform: A :class:`_StatedAffine` or :class:`_StatedLetterbox`.
            params: Unused — :meth:`sample_params` keeps nothing the matrix needs.
            height: Height of the canvas the transform reads, in pixels.
            width: Width of the canvas the transform reads, in pixels.

        Returns:
            The ``(1, 3, 3)`` float64 forward matrix.
        """
        if isinstance(transform, _StatedLetterbox):
            matrix: Tensor = letterbox_matrix(
                height, width, transform.out_h, transform.out_w, transform.allow_upscale, dtype=torch.float64
            )
            return matrix
        stated: _StatedAffine = transform  # type: ignore[assignment]
        return stated.params.matrix(height, width).unsqueeze(0)

    def exact_flip_dims(self, transform: object) -> list[int]:
        """Refuse the lossless-flip path: this adapter states no discrete transform."""
        return self._unsupported("exact_flip_dims", transform)

    def exact_apply(self, transform: object, image: Tensor) -> Tensor:
        """Refuse the lossless-flip path: this adapter states no discrete transform."""
        return self._unsupported("exact_apply", transform)

    def call_nonfused(self, transform: object, image: Tensor, **kwargs: object) -> Tensor:
        """Refuse the passthrough path: both transforms here fuse, so nothing falls back."""
        return self._unsupported("call_nonfused", transform)

    def build_color_matrix(self, transform: object, params: dict[str, Tensor], mean: Tensor | None = None) -> Tensor:
        """Refuse colour fusion: this adapter states geometry only."""
        return self._unsupported("build_color_matrix", transform)

    def build_lut(self, transform: object, params: dict[str, Tensor], values: Tensor) -> Tensor:
        """Refuse intensity-map fusion: this adapter states geometry only."""
        return self._unsupported("build_lut", transform)

    @staticmethod
    def _unsupported(method: str, transform: object) -> NoReturn:
        """Raise for a protocol method no transform this adapter states can reach.

        The protocol asks an adapter to decline what it does not support rather than
        omit it, so that upstream's segment planner gets an answer instead of an
        ``AttributeError`` from somewhere deeper. Declining loudly is also what keeps
        these five honest: reaching one means a transform was classified into a path
        this project never builds.

        Args:
            method: The protocol method that was called.
            transform: The transform it was called for.

        Raises:
            NotImplementedError: Always.
        """
        msg = (
            f"{type(_ADAPTER).__name__} does not implement {method}: it states one affine and one letterbox, "
            f"both of which fuse into a geometric segment. Reached for {type(transform).__name__}."
        )
        raise NotImplementedError(msg)


#: One adapter serves every instance: it holds no state, every answer coming from the
#: transform object it is handed.
_ADAPTER = _StatedGeometryAdapter()


class RandomAffine:
    """Random affine warp applied jointly to an image and its targets (WP-010).

    Each call samples a rotation in ``[-degrees, degrees]``, per-axis shears in
    ``[-shear, shear]``, a uniform scale in ``[1 - scale, 1 + scale]`` (floored just
    above zero) and a translation of ``[-translate, translate]`` times the canvas
    size on each axis — R1 Table S3's ranges, which is what this class still owns —
    and states them to one ``fuse-augmentations`` segment, which resamples the image
    once (bilinear, grey ``114/255`` fill). Every target modality rides the same
    composed matrix. Boxes/polygons are clipped to the canvas and filtered by
    ``min_box_size`` and ``min_visibility``.

    With ``letterbox`` set, upstream's aspect-preserving fit joins the same segment,
    so the source canvas reaches the letterboxed one in a single resample rather than
    two (WP-070's saving, WP-156's delegation). Targets are clipped and filtered at
    the source canvas and only then mapped through the letterbox affine.

    Rotated boxes take the module docstring's corner-warp-and-re-fit path (WP-058)
    and are clipped and filtered by the same two thresholds, with ``boxes``
    recomputed as the envelope of the rotated geometry. That path filters, so it
    requires ``rboxes`` 1:1 with ``boxes`` and no polygons (WP-056); anything else
    raises :class:`ValueError`.

    Args:
        degrees: Maximum absolute rotation in degrees. Defaults to ``0.0``.
        translate: Maximum absolute translation as a fraction of the canvas size
            (applied per axis). Defaults to ``0.1``.
        scale: Half-width of the uniform scale range ``[1 - scale, 1 + scale]``; the
            sampled factor is floored just above zero. Defaults to ``0.5``.
        shear: Maximum absolute shear in degrees, sampled independently per axis.
            Defaults to ``0.0``.
        generator: Optional :class:`torch.Generator` for seeded, reproducible
            sampling. Defaults to ``None`` (global RNG).
        min_box_size: Minimum clipped side length in pixels for an instance to be
            kept. Defaults to ``2.0``.
        min_visibility: Minimum kept-area fraction (clipped-extent area divided by
            pre-clip-extent area) for an instance to be kept. Defaults to ``0.1``.
        letterbox: Output canvas the warp lands on, as a single ``int`` (square) or
            an explicit ``(height, width)`` pair, composed into the same resample.
            Defaults to ``None`` (warp to the input canvas).
        allow_upscale: Whether ``letterbox`` may enlarge content past its native size
            when the output canvas is larger than the source. Defaults to ``True``;
            ignored without ``letterbox``.

    Attributes:
        last_params: The :class:`AffineParams` sampled on the most recent call, or
            ``None`` before the first call.
        last_matrix: The ``(3, 3)`` float64 forward matrix from the most recent
            call, or ``None`` before the first call. This is the **source-canvas**
            affine, before any letterbox.
        letterbox: The :class:`~lucid_yolo.data.letterbox.Letterbox` supplying the
            output-canvas fit, or ``None`` when the warp lands on the input canvas.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> from lucid_yolo.data.transforms import GeometricTransform
        >>> isinstance(RandomAffine(), GeometricTransform)
        True
        >>> aff = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0)
        >>> image = torch.zeros(3, 8, 8)
        >>> t = Targets(boxes=torch.tensor([[1.0, 1.0, 5.0, 5.0]]), labels=torch.tensor([0]))
        >>> out_image, out_targets = aff(image, t)
        >>> out_image.shape
        torch.Size([3, 8, 8])
        >>> out_targets.boxes
        tensor([[1., 1., 5., 5.]])

        ```

        The letterboxed form lands on its own canvas in one resample:

        ```pycon
        >>> fused = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, letterbox=16)
        >>> out_image, _ = fused(torch.rand(3, 20, 40), Targets.empty())
        >>> out_image.shape
        torch.Size([3, 16, 16])

        ```
    """

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        generator: torch.Generator | None = None,
        min_box_size: float = 2.0,
        min_visibility: float = 0.1,
        letterbox: int | tuple[int, int] | None = None,
        allow_upscale: bool = True,
    ) -> None:
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.shear = float(shear)
        self.generator = generator
        self.min_box_size = float(min_box_size)
        self.min_visibility = float(min_visibility)
        self.allow_upscale = bool(allow_upscale)
        self.letterbox = None if letterbox is None else Letterbox(letterbox, allow_upscale=allow_upscale)
        # A second generator, private to the discarded activation gates. Upstream's segment
        # draws a per-sample activation gate for every transform it holds, unconditionally --
        # at `prob = 1.0` the draw is made and then ignored. That draw must not come from
        # `generator`, which would shift the caller's sequence and break the property WP-079
        # was opened by losing, nor from the global stream, which every other transform
        # shares. It is seeded off the caller's generator so that a caller who seeds gets a
        # reproducible gate stream too, and two affines under different seeds get different
        # ones. Left unseeded it was reproducible only by the conjunction of two premises the
        # code never stated -- that every gate sits at `prob = 1.0` so its value is discarded,
        # *and* that `torch.Generator()`'s unseeded state is a fixed constant rather than
        # entropy. Both hold today; neither is this class's to rely on (L-32).
        self._gate_stream = torch.Generator()
        if generator is not None:
            self._gate_stream.manual_seed(generator.initial_seed() ^ _GATE_SEED_SALT)
        self.last_params: AffineParams | None = None
        self.last_matrix: Tensor | None = None

    def __call__(self, image: Tensor, targets: Targets) -> tuple[Tensor, Targets]:
        """Warp ``image`` and ``targets`` through one freshly-sampled affine.

        Args:
            image: CHW image tensor (float, in the same value range as the grey
                fill, i.e. ``[0, 1]``).
            targets: Geometry to warp alongside the image. Non-empty ``rboxes`` must
                share the instance axis with ``boxes`` and carry no polygons.

        Returns:
            The warped ``(C, H, W)`` image (same canvas size as the input) and the
            warped, clipped and filtered targets.

        Raises:
            ValueError: If ``rboxes`` is non-empty and breaks WP-056's instance-axis
                invariant (length mismatch, or polygons alongside).

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> gen = torch.Generator().manual_seed(0)
            >>> aff = RandomAffine(degrees=10.0, translate=0.1, scale=0.2, generator=gen)
            >>> out_image, _ = aff(torch.rand(3, 16, 16), Targets.empty())
            >>> out_image.shape
            torch.Size([3, 16, 16])
            >>> aff.last_matrix.shape
            torch.Size([3, 3])

            ```
        """
        _, height, width = image.shape
        return self.apply(image, targets, self.sample(height, width))

    def sample(self, height: int, width: int) -> AffineParams:
        """Draw one :class:`AffineParams` from the configured ranges, consuming the RNG.

        The sampling half of the seam: this is the only method that touches
        :attr:`generator`, so a caller that wants a *stated* transform rather than a
        drawn one skips it and builds :class:`AffineParams` directly (WP-147).

        Args:
            height: Canvas height in pixels; scales the vertical translation range.
            width: Canvas width in pixels; scales the horizontal translation range.

        Returns:
            The sampled parameters. Nothing is stashed on the instance -- the
            ``last_params`` / ``last_matrix`` attributes are set by :meth:`apply`.

        Examples:
            ```pycon
            >>> import torch
            >>> gen = torch.Generator().manual_seed(0)
            >>> params = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, generator=gen).sample(16, 16)
            >>> (abs(params.angle), abs(params.translate_x), params.scale)
            (0.0, 0.0, 1.0)

            ```
        """
        gen = self.generator
        angle = math.radians(_uniform(-self.degrees, self.degrees, gen))
        shear_x = math.radians(_uniform(-self.shear, self.shear, gen))
        shear_y = math.radians(_uniform(-self.shear, self.shear, gen))
        scale = max(_uniform(1.0 - self.scale, 1.0 + self.scale, gen), _MIN_SCALE)
        translate_x = _uniform(-self.translate, self.translate, gen) * width
        translate_y = _uniform(-self.translate, self.translate, gen) * height
        return AffineParams(
            angle=angle,
            shear_x=shear_x,
            shear_y=shear_y,
            scale=scale,
            translate_x=translate_x,
            translate_y=translate_y,
        )

    def apply(self, image: Tensor, targets: Targets, params: AffineParams) -> tuple[Tensor, Targets]:
        """Warp ``image`` and ``targets`` through ``params``, drawing nothing.

        The application half of the seam. Given the same ``params`` this is a pure
        function of its inputs, which is what lets an expectation be frozen against
        stated parameters rather than against a seed (WP-147).

        Args:
            image: CHW image tensor (float, in the same value range as the grey
                fill, i.e. ``[0, 1]``).
            targets: Geometry to warp alongside the image. Non-empty ``rboxes`` must
                share the instance axis with ``boxes`` and carry no polygons.
            params: The affine to apply.

        Returns:
            The warped image — the input canvas, or ``letterbox``'s when one is
            configured — and the warped, clipped and filtered targets.

        Raises:
            ValueError: If ``rboxes`` is non-empty and breaks WP-056's instance-axis
                invariant (length mismatch, or polygons alongside).
            RuntimeError: If the transforms did not fuse into one matrix-bearing
                geometric segment — see :meth:`_segment`.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> quarter = AffineParams(
            ...     angle=0.0, shear_x=0.0, shear_y=0.0, scale=1.0, translate_x=2.0, translate_y=0.0
            ... )
            >>> box = Targets(boxes=torch.tensor([[1.0, 1.0, 5.0, 5.0]]), labels=torch.tensor([0]))
            >>> _, out = RandomAffine().apply(torch.zeros(3, 8, 8), box, quarter)
            >>> out.boxes.tolist()
            [[3.0, 1.0, 7.0, 5.0]]

            ```
        """
        check_rotated_pairing(targets)
        _, height, width = image.shape
        matrix = self._record(params, height, width)
        out_image = self._resample(image, params)
        # Clipped and filtered at the *source* canvas, before any letterbox: the letterbox
        # maps this canvas onto the output canvas's content region, so clipping afterwards
        # would clip against the padding too and keep instances this path drops.
        out_targets = self._warp_targets(targets, matrix, height, width)
        if self.letterbox is not None:
            out_targets = self.letterbox.warp_targets(out_targets, height, width)
        return out_image, out_targets

    def _record(self, params: AffineParams, height: int, width: int) -> Tensor:
        """Build the matrix for ``params`` and stash both on ``last_params``/``last_matrix``."""
        matrix = params.matrix(height, width)
        self.last_params = params
        self.last_matrix = matrix
        return matrix

    def _segment(self, params: AffineParams) -> FusedAffineSegment:
        """Build the one upstream segment that warps ``params`` (and the letterbox).

        The transforms are stated rather than configured: :class:`_StatedAffine` carries
        the parameters :meth:`sample` already drew, so the segment is rebuilt per call
        and :meth:`apply` stays the pure function of its inputs WP-147 made it. A build
        costs a fraction of the warp it sets up, and the alternative — one segment
        holding a parameter slot rewritten before each call — is hidden mutable state on
        a class that a dataloader worker reuses thousands of times.

        Segmentation is upstream's, so this is also where the row's tier-C condition is
        enforced. Only *adjacent* geometric operations group into one run: a colour
        operation between two of them would split the run, and each half would then warp
        the image separately with nothing raising. Two conditions establish that the one
        matrix is the whole chain — a single segment, and that segment being the
        matrix-bearing fused kind rather than an exact or crop-resize one, which carry
        no composed affine (an exact-only or letterbox-only pipeline is exactly the case
        upstream reports no matrix for).

        Args:
            params: The affine to state, in this module's units (radians and pixels).

        Returns:
            The single fused segment: the affine, plus the letterbox when configured.

        Raises:
            RuntimeError: If the transforms did not fuse into one matrix-bearing
                geometric segment.
        """
        transforms: list[object] = [_StatedAffine(params)]
        if self.letterbox is not None:
            transforms.append(_StatedLetterbox(self.letterbox.out_h, self.letterbox.out_w, self.allow_upscale))
        segments = build_segments(
            transforms=transforms,
            adapter=_ADAPTER,
            interpolation="bilinear",
            padding_mode="zeros",
            fill=_FILL,
            # Upstream draws a per-transform activation gate even at `prob = 1.0`. It
            # changes nothing here (the gate always passes), but it draws from *some*
            # stream, and neither the caller's generator nor the global one may be it.
            generator=self._gate_stream,
        )
        if len(segments) != 1 or not isinstance(segments[0], FusedAffineSegment):
            kinds = [type(segment).__name__ for segment in segments]
            msg = (
                f"the affine did not fuse into one matrix-bearing geometric segment: got {kinds}. "
                "Only adjacent geometric operations group into one run, so anything else means the image "
                "is warped in more passes than the geometry describes."
            )
            raise RuntimeError(msg)
        return segments[0]

    def _resample(self, image: Tensor, params: AffineParams) -> Tensor:
        """Warp ``image`` through the upstream segment in a single pass.

        Args:
            image: CHW image tensor (float, in the grey-fill value range ``[0, 1]``).
            params: The affine to apply.

        Returns:
            The warped ``(C, out_h, out_w)`` image, on the letterbox canvas when one
            is configured and on the input canvas otherwise.

        Raises:
            RuntimeError: If the transforms did not fuse into one segment.
        """
        segment = self._segment(params)
        # A segment warps a batch; this transform is per-sample, so the batch axis is added
        # and dropped around the one call. `forward` is called rather than the module, as
        # upstream's own pipeline does, since there are no hooks to dispatch.
        warped = segment.forward(image.unsqueeze(0), None)
        return cast("Tensor", warped).squeeze(0)

    def _warp_targets(self, targets: Targets, matrix: Tensor, height: int, width: int) -> Targets:
        """Warp, clip and filter every modality; dispatch on rotated-box/polygon presence."""
        if targets.rboxes.shape[0] > 0:
            return self._warp_rotated(targets, matrix, height, width)
        if targets.polygons:
            return self._warp_with_polygons(targets, matrix, height, width)
        return self._warp_boxes_only(targets, matrix, height, width)

    def _warp_rotated(self, targets: Targets, matrix: Tensor, height: int, width: int) -> Targets:
        """Rotated path: re-fit the warped corners, clip to the canvas, filter both axes."""
        warped = warp_rboxes(targets.rboxes, matrix)
        pre_boxes = rbox_envelopes(warped)
        rboxes, post_boxes = clip_rboxes_to_canvas(warped, float(height), float(width))
        keep = instance_keep_mask(pre_boxes, post_boxes, min_size=self.min_box_size, min_visibility=self.min_visibility)
        full = Targets(
            boxes=post_boxes,
            labels=targets.labels.clone(),
            rboxes=rboxes,
            difficult=targets.difficult.clone(),
            keypoints=self._warp_keypoints(targets.keypoints, matrix),
            keypoint_vis=targets.keypoint_vis.clone(),
        )
        # One mask over both axes: WP-056's invariant is what makes `rkeep=keep` correct,
        # and `check_rotated_pairing` has already refused anything that breaks it.
        return full.filter(keep, rkeep=keep)

    def _warp_with_polygons(self, targets: Targets, matrix: Tensor, height: int, width: int) -> Targets:
        """Polygon path: warp rings, clamp to canvas, recompute boxes, filter."""
        warped_rings = [self._warp_points(ring, matrix) for ring in targets.polygons]
        pre_boxes = boxes_from_polygons(warped_rings)
        clipped_rings = [self._clip_points(ring, height, width) for ring in warped_rings]
        post_boxes = boxes_from_polygons(clipped_rings)
        keep = instance_keep_mask(pre_boxes, post_boxes, min_size=self.min_box_size, min_visibility=self.min_visibility)
        full = Targets(
            boxes=post_boxes,
            labels=targets.labels.clone(),
            polygons=clipped_rings,
            difficult=targets.difficult.clone(),
            # Warped from the raw affine output, never from `clipped_rings`: the ring clamp
            # exists to keep a mask inside the canvas it is rasterised on, and A70 wants the
            # opposite for a point. See `_warp_keypoints`.
            keypoints=self._warp_keypoints(targets.keypoints, matrix),
            keypoint_vis=targets.keypoint_vis.clone(),
        )
        return full.filter(keep)

    def _warp_boxes_only(self, targets: Targets, matrix: Tensor, height: int, width: int) -> Targets:
        """Box-only path: warp corners to an axis-aligned extent, clip, filter."""
        pre_boxes = self._warp_boxes(targets.boxes, matrix)
        post_boxes = self._clip_boxes(pre_boxes, height, width)
        keep = instance_keep_mask(pre_boxes, post_boxes, min_size=self.min_box_size, min_visibility=self.min_visibility)
        full = Targets(
            boxes=post_boxes,
            labels=targets.labels.clone(),
            difficult=targets.difficult.clone(),
            keypoints=self._warp_keypoints(targets.keypoints, matrix),
            keypoint_vis=targets.keypoint_vis.clone(),
        )
        return full.filter(keep)

    @staticmethod
    def _warp_boxes(boxes: Tensor, matrix: Tensor) -> Tensor:
        """Warp ``xyxy`` boxes through the affine, upstream's corner-warp-and-re-fit (WP-156).

        :func:`~fuse_augmentations.transform_bbox_xyxy` computes the four corners,
        maps them through the forward matrix and returns the axis-aligned box that
        wraps them — the same operation the local ``_transform_box_corners`` performed,
        and the same trade-off: a general affine sends a rectangle to a parallelogram,
        so an ``xyxy`` result is the enclosing extent rather than the shape itself.
        It is batched over images as well as boxes, so the single sample this class
        handles is wrapped and unwrapped around the one call.

        Args:
            boxes: ``(N, 4)`` ``xyxy`` boxes on the source canvas.
            matrix: The ``(3, 3)`` float64 forward affine.

        Returns:
            The warped ``(N, 4)`` extents, in ``boxes``'s dtype.
        """
        warped: Tensor = transform_bbox_xyxy(boxes.to(matrix.dtype).unsqueeze(0), matrix.unsqueeze(0))
        return warped.squeeze(0).to(boxes.dtype)

    @staticmethod
    def _warp_points(points: Tensor, matrix: Tensor) -> Tensor:
        """Apply the float64 ``matrix`` to float32 ``points``, restoring float32."""
        warped = apply_affine_to_points(points.to(matrix.dtype), matrix)
        return warped.to(points.dtype)

    @staticmethod
    def _warp_keypoints(keypoints: Tensor, matrix: Tensor) -> Tensor:
        """Warp ``(N, K, 2)`` points through the affine and stop there — no clip (A70).

        Every other modality on this path is clipped after the warp, so the omission
        here is the whole content of the method. A70 settles what becomes of a point the
        affine pushes off-canvas on an instance the affine *keeps*: it is carried through
        unchanged, true coordinate and visibility both. Clamping it to the boundary would
        invent a target — supervising the model toward a location the anatomy is
        demonstrably not at — and zeroing its visibility would overload A66's "no
        annotation exists" with a second, unrecoverable meaning. Neither is a safety
        measure; both are wrong answers, so this warps and returns.

        The rows themselves still filter with their boxes: the caller hands the result to
        :meth:`~lucid_yolo.data.targets.Targets.filter`, which selects keypoints on the
        shared instance axis, so a *dropped* instance takes its points with it. What A70
        governs is only the kept ones.

        A keypoint-free ``Targets`` carries the canonical ``(0, 0, 2)`` empty, which
        round-trips through the reshape and returns the same empty — an exact no-op for
        detect/segment/obb, which is what keeps the frozen goldens on this transform still.

        Args:
            keypoints: ``(N, K, 2)`` point coordinates on the source canvas.
            matrix: The ``(3, 3)`` float64 forward affine.

        Returns:
            The warped points, shaped and typed as given, unclipped.
        """
        return RandomAffine._warp_points(keypoints.reshape(-1, _POINT_DIM), matrix).reshape(keypoints.shape)

    @staticmethod
    def _clip_boxes(boxes: Tensor, height: int, width: int) -> Tensor:
        """Clamp ``xyxy`` boxes to the ``[0, width] x [0, height]`` canvas."""
        clipped = boxes.clone()
        clipped[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(width))
        clipped[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(height))
        return clipped

    @staticmethod
    def _clip_points(ring: Tensor, height: int, width: int) -> Tensor:
        """Clamp a polygon ring's points to the canvas bounds (point-wise)."""
        clipped = ring.clone()
        clipped[:, 0] = ring[:, 0].clamp(0.0, float(width))
        clipped[:, 1] = ring[:, 1].clamp(0.0, float(height))
        return clipped
