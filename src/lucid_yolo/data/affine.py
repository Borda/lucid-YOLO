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

Image warp:
    The image is resampled with :func:`torch.nn.functional.grid_sample` over a grid
    built from :func:`torch.nn.functional.affine_grid`. ``affine_grid``'s ``theta``
    maps *output* normalised coordinates to *input* normalised coordinates, i.e. it
    encodes the **inverse** of the forward pixel matrix expressed in the
    ``[-1, 1]`` normalised frame; :meth:`RandomAffine._theta_from_pixel_matrix`
    derives it from the pixel matrix and the ``align_corners=False`` pixel/normalised
    convention. Out-of-canvas samples are filled with the same ``114/255`` grey as
    letterbox by the subtract-pad / zero-pad-sample / add-pad trick.

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

Fused letterbox (WP-070):
    Training warps a source canvas by this random affine and then letterboxes it
    down to ``img_size`` — two bilinear resamples. :class:`FusedAffineLetterbox`
    composes the letterbox's pure scale-and-translation affine into the random one
    so the image is resampled **once** (via :meth:`RandomAffine.warp_to`, which
    warps to a differently-sized output canvas under a caller-supplied post-affine).
    Targets are still clipped and filtered at the source canvas and then mapped
    through the letterbox affine, so boxes and polygons are byte-identical to the
    two-transform path; only the image resampling differs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.rotated_aug import check_rotated_pairing, clip_rboxes_to_canvas, rbox_envelopes, warp_rboxes
from lucid_yolo.data.targets import Targets
from lucid_yolo.data.transforms import apply_affine_to_points, boxes_from_polygons

__all__ = ["AffineParams", "FusedAffineLetterbox", "RandomAffine"]

#: Default pad colour: mid-grey ``114/255`` per the YOLO-lineage convention (matches letterbox).
_DEFAULT_PAD_VALUE = 114.0 / 255.0
#: Lower bound on the sampled scale factor, keeping the matrix invertible (``> 0``).
_MIN_SCALE = 1e-3
#: Reused axis-size guards.
_POINT_DIM = 2


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
        center = torch.tensor([width / 2.0, height / 2.0], dtype=torch.float64)
        translate = torch.tensor([self.translate_x, self.translate_y], dtype=torch.float64)
        offset = center - linear @ center + translate
        matrix = torch.eye(3, dtype=torch.float64)
        matrix[:2, :2] = linear
        matrix[:2, 2] = offset
        return matrix


class RandomAffine:
    """Random affine warp applied jointly to an image and its targets (WP-010).

    Each call samples a rotation in ``[-degrees, degrees]``, per-axis shears in
    ``[-shear, shear]``, a uniform scale in ``[1 - scale, 1 + scale]`` (floored just
    above zero) and a translation of ``[-translate, translate]`` times the canvas
    size on each axis, composes them about the canvas centre and warps the image
    (bilinear ``grid_sample``, grey ``114/255`` fill) and every target modality
    through the one matrix. Boxes/polygons are clipped to the canvas and filtered by
    ``min_box_size`` and ``min_visibility``.

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

    Attributes:
        last_params: The :class:`AffineParams` sampled on the most recent call, or
            ``None`` before the first call.
        last_matrix: The ``(3, 3)`` float64 forward matrix from the most recent
            call, or ``None`` before the first call.

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
    ) -> None:
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.shear = float(shear)
        self.generator = generator
        self.min_box_size = float(min_box_size)
        self.min_visibility = float(min_visibility)
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
            The warped ``(C, H, W)`` image (same canvas size as the input) and the
            warped, clipped and filtered targets.

        Raises:
            ValueError: If ``rboxes`` is non-empty and breaks WP-056's instance-axis
                invariant (length mismatch, or polygons alongside).

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
        out_image = self._warp_image(image, matrix, height, width)
        out_targets = self._warp_targets(targets, matrix, height, width)
        return out_image, out_targets

    def warp_to(
        self, image: Tensor, targets: Targets, post_matrix: Tensor, out_h: int, out_w: int
    ) -> tuple[Tensor, Targets]:
        """Sample one affine and warp to ``(out_h, out_w)`` composed with ``post_matrix``.

        The image is resampled **once**: the freshly-sampled affine (source canvas
        to source canvas) is composed with ``post_matrix`` (source canvas to output
        canvas) into a single source-to-output matrix and applied with one
        ``grid_sample``. Targets are warped, clipped and filtered at the source
        canvas exactly as :meth:`__call__` does; ``post_matrix`` is **not** applied
        to them here, so the caller composes the output-canvas mapping onto the
        returned canvas-scale targets (see :class:`FusedAffineLetterbox`). The
        sampled :attr:`last_params` / :attr:`last_matrix` are the source-canvas
        affine, unchanged by ``post_matrix``.

        Args:
            image: CHW image tensor (float, in the grey-fill value range ``[0, 1]``).
            targets: Geometry to warp alongside the image. Non-empty ``rboxes`` must
                share the instance axis with ``boxes`` and carry no polygons.
            post_matrix: ``(3, 3)`` affine mapping source-canvas pixels to the
                output canvas, composed after the random affine for the image warp.
            out_h: Output canvas height in pixels.
            out_w: Output canvas width in pixels.

        Returns:
            The warped ``(C, out_h, out_w)`` image and the canvas-scale warped,
            clipped and filtered targets (before ``post_matrix``).

        Raises:
            ValueError: If ``rboxes`` is non-empty and breaks WP-056's instance-axis
                invariant (length mismatch, or polygons alongside).

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> aff = RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=0.0)
            >>> half = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
            >>> out_image, _ = aff.warp_to(torch.rand(3, 16, 16), Targets.empty(), half, 8, 8)
            >>> out_image.shape
            torch.Size([3, 8, 8])

            ```
        """
        _, in_h, in_w = image.shape
        return self.apply_to(image, targets, self.sample(in_h, in_w), post_matrix, out_h, out_w)

    def apply_to(
        self,
        image: Tensor,
        targets: Targets,
        params: AffineParams,
        post_matrix: Tensor,
        out_h: int,
        out_w: int,
    ) -> tuple[Tensor, Targets]:
        """Apply ``params`` composed with ``post_matrix``, drawing nothing.

        The stated-parameter sibling of :meth:`warp_to`, standing to it as
        :meth:`apply` stands to :meth:`__call__` (WP-147).

        Args:
            image: CHW image tensor (float, in the grey-fill value range ``[0, 1]``).
            targets: Geometry to warp alongside the image. Non-empty ``rboxes`` must
                share the instance axis with ``boxes`` and carry no polygons.
            params: The source-canvas affine to apply.
            post_matrix: ``(3, 3)`` affine mapping source-canvas pixels to the
                output canvas, composed after ``params`` for the image warp.
            out_h: Output canvas height in pixels.
            out_w: Output canvas width in pixels.

        Returns:
            The warped ``(C, out_h, out_w)`` image and the canvas-scale warped,
            clipped and filtered targets (before ``post_matrix``).

        Raises:
            ValueError: If ``rboxes`` is non-empty and breaks WP-056's instance-axis
                invariant (length mismatch, or polygons alongside).

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> identity = AffineParams(
            ...     angle=0.0, shear_x=0.0, shear_y=0.0, scale=1.0, translate_x=0.0, translate_y=0.0
            ... )
            >>> half = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
            >>> out_image, _ = RandomAffine().apply_to(
            ...     torch.rand(3, 16, 16), Targets.empty(), identity, half, 8, 8
            ... )
            >>> out_image.shape
            torch.Size([3, 8, 8])

            ```
        """
        check_rotated_pairing(targets)
        _, in_h, in_w = image.shape
        matrix = self._record(params, in_h, in_w)
        image_matrix = post_matrix.to(matrix.dtype) @ matrix
        out_image = self._warp_image(image, image_matrix, out_h, out_w)
        out_targets = self._warp_targets(targets, matrix, in_h, in_w)
        return out_image, out_targets

    def _record(self, params: AffineParams, height: int, width: int) -> Tensor:
        """Build the matrix for ``params`` and stash both on ``last_params``/``last_matrix``."""
        matrix = params.matrix(height, width)
        self.last_params = params
        self.last_matrix = matrix
        return matrix

    def _warp_image(self, image: Tensor, matrix: Tensor, out_h: int, out_w: int) -> Tensor:
        """Resample ``image`` under ``matrix`` into an ``(out_h, out_w)`` canvas.

        ``matrix`` is the forward source-to-output pixel map; when it composes a
        downscaling letterbox (see :meth:`warp_to`) the output canvas differs from
        the input, and the letterbox padding region — mapping outside the source —
        is grey-filled by the same subtract-pad / zero-pad-sample / add-pad trick
        that fills the affine's out-of-canvas borders.
        """
        _, in_h, in_w = image.shape
        theta = self._theta_from_pixel_matrix(matrix, in_h, in_w, out_h, out_w).to(image.dtype)
        grid = F.affine_grid(theta.unsqueeze(0), [1, image.shape[0], out_h, out_w], align_corners=False)
        shifted = image.unsqueeze(0) - _DEFAULT_PAD_VALUE
        sampled = F.grid_sample(shifted, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return sampled.squeeze(0) + _DEFAULT_PAD_VALUE

    @staticmethod
    def _theta_from_pixel_matrix(matrix: Tensor, in_h: int, in_w: int, out_h: int, out_w: int) -> Tensor:
        """Convert a forward pixel matrix to an ``affine_grid`` ``theta`` (2x3, normalised).

        ``affine_grid`` needs the output→input normalised map, i.e. the inverse of
        the forward pixel matrix conjugated by the pixel↔normalised change of basis
        for ``align_corners=False`` (pixel ``p`` on an axis of size ``L`` maps to
        ``g = (2p + 1) / L - 1``). Input and output sizes may differ: the output
        pixels are de-normalised with ``(out_h, out_w)`` and the resulting source
        pixels re-normalised with ``(in_h, in_w)``, so the same routine serves both
        the same-size affine warp and the fused affine+letterbox downscale.
        """
        inverse = torch.inverse(matrix)
        to_norm = torch.tensor(
            [[2.0 / in_w, 0.0, 1.0 / in_w - 1.0], [0.0, 2.0 / in_h, 1.0 / in_h - 1.0], [0.0, 0.0, 1.0]],
            dtype=matrix.dtype,
        )
        from_norm = torch.tensor(
            [[out_w / 2.0, 0.0, (out_w - 1.0) / 2.0], [0.0, out_h / 2.0, (out_h - 1.0) / 2.0], [0.0, 0.0, 1.0]],
            dtype=matrix.dtype,
        )
        return (to_norm @ inverse @ from_norm)[:2, :]

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
        keep = self._keep_mask(pre_boxes, post_boxes)
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
        keep = self._keep_mask(pre_boxes, post_boxes)
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
        pre_boxes = self._transform_box_corners(targets.boxes, matrix)
        post_boxes = self._clip_boxes(pre_boxes, height, width)
        keep = self._keep_mask(pre_boxes, post_boxes)
        full = Targets(
            boxes=post_boxes,
            labels=targets.labels.clone(),
            difficult=targets.difficult.clone(),
            keypoints=self._warp_keypoints(targets.keypoints, matrix),
            keypoint_vis=targets.keypoint_vis.clone(),
        )
        return full.filter(keep)

    def _keep_mask(self, pre_boxes: Tensor, post_boxes: Tensor) -> Tensor:
        """Boolean keep mask from clipped size and kept-area (visibility) thresholds."""
        widths = post_boxes[:, 2] - post_boxes[:, 0]
        heights = post_boxes[:, 3] - post_boxes[:, 1]
        pre_area = (pre_boxes[:, 2] - pre_boxes[:, 0]).clamp(min=0.0) * (pre_boxes[:, 3] - pre_boxes[:, 1]).clamp(
            min=0.0
        )
        post_area = widths.clamp(min=0.0) * heights.clamp(min=0.0)
        visibility = torch.where(pre_area > 0.0, post_area / pre_area.clamp(min=1e-12), torch.zeros_like(pre_area))
        return (widths >= self.min_box_size) & (heights >= self.min_box_size) & (visibility >= self.min_visibility)

    @staticmethod
    def _transform_box_corners(boxes: Tensor, matrix: Tensor) -> Tensor:
        """Warp each box's four corners and take the axis-aligned extent."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        corners = torch.stack(
            [
                torch.stack([x1, y1], dim=1),
                torch.stack([x2, y1], dim=1),
                torch.stack([x2, y2], dim=1),
                torch.stack([x1, y2], dim=1),
            ],
            dim=1,
        ).reshape(-1, _POINT_DIM)
        warped = RandomAffine._warp_points(corners, matrix).reshape(-1, 4, _POINT_DIM)
        lo = warped.amin(dim=1)
        hi = warped.amax(dim=1)
        return torch.cat([lo, hi], dim=1)

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


class FusedAffineLetterbox:
    """Random affine and letterbox composed into a single image resample (WP-070).

    The training geometric base warps a source canvas (a mosaic assembly or a
    single base image) by a random affine and then letterboxes it down to a fixed
    square. Done as two transforms that is two full bilinear resamples per sample;
    this composes the letterbox's pure scale-and-translate affine into the random
    affine so the image is resampled **once**, straight from the source canvas to
    the ``target_size`` output. The letterbox padding region is grey-filled by the
    same ``114/255`` border trick the affine already uses.

    Geometry stays split by responsibility: a :class:`RandomAffine` owns the
    random sampling and the canvas-scale box/polygon clip-and-filter, and a
    :class:`~lucid_yolo.data.letterbox.Letterbox` owns the aspect-preserving
    geometry (ratio and symmetric padding) resolved per sample from the source
    size. Targets are warped, clipped and filtered at the source canvas by the
    affine, then mapped through the letterbox affine — identical arithmetic to the
    two-transform path, so boxes and polygons are unchanged; only the image
    resampling differs (one bilinear pass instead of two).

    The single ``grid_sample`` pass is **not antialiased** (``grid_sample`` has no
    antialias mode), unlike the standalone :class:`~lucid_yolo.data.letterbox.Letterbox`,
    whose downscale uses ``antialias=True`` and which remains the validation/eval
    path. This is a recorded train-time deviation (A32): cv2-lineage training
    resizes are conventionally non-antialiased, and dropping the antialias pass is
    a large part of the measured speedup.

    Rotated boxes ride the wrapped affine's rotated path (WP-058) and are then
    mapped through the letterbox affine, whose uniform scale and translation
    preserve the long-edge form: centres warped, ``w``/``h`` scaled by ``r``,
    ``theta`` unchanged and still canonical.

    Args:
        target_size: Output square side (a single ``int``) or explicit
            ``(height, width)`` pair the source canvas is letterboxed into.
        degrees: Maximum absolute rotation in degrees. Defaults to ``0.0``.
        translate: Maximum absolute translation as a fraction of the source canvas
            size (per axis). Defaults to ``0.1``.
        scale: Half-width of the uniform scale range ``[1 - scale, 1 + scale]``.
            Defaults to ``0.5``.
        shear: Maximum absolute shear in degrees, sampled per axis. Defaults to
            ``0.0``.
        generator: Optional :class:`torch.Generator` for seeded sampling. Defaults
            to ``None`` (global RNG).
        min_box_size: Minimum clipped side length in pixels (source canvas) for an
            instance to be kept. Defaults to ``2.0``.
        min_visibility: Minimum kept-area fraction for an instance to be kept.
            Defaults to ``0.1``.
        allow_upscale: Whether the letterbox may enlarge content beyond native
            size when the target is larger than the source. Defaults to ``True``.

    Attributes:
        affine: The wrapped :class:`RandomAffine`; its ``last_params`` /
            ``last_matrix`` expose the most recently sampled source-canvas affine.
        letterbox: The wrapped :class:`~lucid_yolo.data.letterbox.Letterbox`
            supplying the per-sample aspect-preserving geometry.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> from lucid_yolo.data.transforms import GeometricTransform
        >>> isinstance(FusedAffineLetterbox(32), GeometricTransform)
        True
        >>> fused = FusedAffineLetterbox(32, degrees=0.0, translate=0.0, scale=0.0)
        >>> out_image, _ = fused(torch.rand(3, 20, 40), Targets.empty())
        >>> out_image.shape
        torch.Size([3, 32, 32])

        ```
    """

    def __init__(
        self,
        target_size: int | tuple[int, int],
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        generator: torch.Generator | None = None,
        min_box_size: float = 2.0,
        min_visibility: float = 0.1,
        allow_upscale: bool = True,
    ) -> None:
        self.affine = RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            generator=generator,
            min_box_size=min_box_size,
            min_visibility=min_visibility,
        )
        self.letterbox = Letterbox(target_size, allow_upscale=allow_upscale)

    def __call__(self, image: Tensor, targets: Targets) -> tuple[Tensor, Targets]:
        """Warp ``image`` and ``targets`` through the fused affine+letterbox.

        Args:
            image: CHW image tensor (float, in the grey-fill value range ``[0, 1]``);
                the source canvas (mosaic assembly or single base image).
            targets: Geometry to warp alongside the image. Non-empty ``rboxes`` must
                share the instance axis with ``boxes`` and carry no polygons.

        Returns:
            The ``(C, target_h, target_w)`` letterboxed image resampled once, and
            the warped, clipped and filtered targets in output-canvas coordinates.

        Raises:
            ValueError: If ``rboxes`` is non-empty and breaks WP-056's instance-axis
                invariant (length mismatch, or polygons alongside).

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> fused = FusedAffineLetterbox(16, degrees=0.0, translate=0.0, scale=0.0)
            >>> box = Targets(boxes=torch.tensor([[2.0, 2.0, 10.0, 10.0]]), labels=torch.tensor([0]))
            >>> out_image, out_targets = fused(torch.rand(3, 16, 16), box)
            >>> out_image.shape, out_targets.boxes.shape
            (torch.Size([3, 16, 16]), torch.Size([1, 4]))

            ```
        """
        _, in_h, in_w = image.shape
        return self.apply(image, targets, self.sample(in_h, in_w))

    def sample(self, height: int, width: int) -> AffineParams:
        """Draw the source-canvas affine, consuming the RNG; the letterbox is deterministic.

        Args:
            height: Source canvas height in pixels.
            width: Source canvas width in pixels.

        Returns:
            The sampled source-canvas :class:`AffineParams`. The letterbox half of
            this transform draws nothing -- its geometry is resolved from the source
            size -- so the affine is the whole of what a call samples (WP-147).

        Examples:
            ```pycon
            >>> import torch
            >>> fused = FusedAffineLetterbox(16, degrees=0.0, translate=0.0, scale=0.0)
            >>> fused.sample(20, 40).scale
            1.0

            ```
        """
        return self.affine.sample(height, width)

    def apply(self, image: Tensor, targets: Targets, params: AffineParams) -> tuple[Tensor, Targets]:
        """Warp through ``params`` and the per-sample letterbox, drawing nothing.

        Args:
            image: CHW image tensor (float, in the grey-fill value range ``[0, 1]``);
                the source canvas (mosaic assembly or single base image).
            targets: Geometry to warp alongside the image. Non-empty ``rboxes`` must
                share the instance axis with ``boxes`` and carry no polygons.
            params: The source-canvas affine to apply.

        Returns:
            The ``(C, target_h, target_w)`` letterboxed image resampled once, and
            the warped, clipped and filtered targets in output-canvas coordinates.

        Raises:
            ValueError: If ``rboxes`` is non-empty and breaks WP-056's instance-axis
                invariant (length mismatch, or polygons alongside).

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> identity = AffineParams(
            ...     angle=0.0, shear_x=0.0, shear_y=0.0, scale=1.0, translate_x=0.0, translate_y=0.0
            ... )
            >>> out_image, _ = FusedAffineLetterbox(16).apply(torch.rand(3, 32, 32), Targets.empty(), identity)
            >>> out_image.shape
            torch.Size([3, 16, 16])

            ```
        """
        _, in_h, in_w = image.shape
        post_matrix, out_h, out_w = self.letterbox.forward_affine(in_h, in_w)
        out_image, canvas_targets = self.affine.apply_to(image, targets, params, post_matrix, out_h, out_w)
        return out_image, self.letterbox.warp_targets(canvas_targets, in_h, in_w)
