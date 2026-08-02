# SPDX-License-Identifier: Apache-2.0
"""Aspect-preserving letterbox resize with an exact coordinate inverse (A10).

Letterboxing scales a CHW image by a single ratio ``r = min(target_h / H,
target_w / W)`` — preserving aspect — and pads the residual slack symmetrically
with a constant grey value, so the content is centred inside the target canvas
(A10: "letterbox, aspect-preserving", the YOLO-lineage resize convention noted in
third-party literature, docs/ASSUMPTIONS.md). Because the mapping from original
to letterboxed pixel coordinates is the pure scale-plus-translation affine
``[[r, 0, pad_left], [0, r, pad_top], [0, 0, 1]]``, it has an exact analytic
inverse — eval-time predictions in letterboxed space can be un-letterboxed back
to original-image coordinates without loss.

:class:`Letterbox` conforms to
:class:`~open_yolos.data.transforms.GeometricTransform`: one call warps the image
and every modality carried by :class:`~open_yolos.data.targets.Targets` (boxes,
polygons, rotated boxes) through the *same* affine, routed via
:func:`~open_yolos.data.transforms.apply_affine_to_points` so the three paths share
one source of geometric truth. Rotated boxes need only their centres warped as
points, their ``w``/``h`` scaled by ``r`` and ``theta`` left unchanged, since the
transform carries no rotation or shear.

Inverse API:
    :meth:`Letterbox.inverse_map` is stateless in the image content — given a
    batch of letterboxed points, the original image size and the letterboxed
    canvas size it recomputes the ratio and padding and applies the inverse
    affine, returning original-image coordinates. It depends only on those sizes
    and the instance's ``allow_upscale`` flag, never on which image produced the
    predictions, which is exactly what an evaluation loop has on hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from open_yolos.data.targets import Targets
from open_yolos.data.transforms import apply_affine_to_points

__all__ = ["Letterbox"]

#: Default pad colour: mid-grey ``114/255`` per the YOLO-lineage convention (A10).
_DEFAULT_PAD_VALUE = 114.0 / 255.0


def _as_hw(target_size: int | tuple[int, int]) -> tuple[int, int]:
    """Normalise a target size to an ``(height, width)`` pair.

    Args:
        target_size: A single ``int`` (square target) or an explicit
            ``(height, width)`` pair.

    Returns:
        The target ``(height, width)`` as a pair of ints.
    """
    if isinstance(target_size, int):
        return target_size, target_size
    height, width = target_size
    return int(height), int(width)


@dataclass(frozen=True)
class _LetterboxGeom:
    """Resolved letterbox geometry for one original/target size pairing.

    Attributes:
        r: Aspect-preserving scale ratio applied to both axes.
        new_h: Height of the resized content region, ``round(H * r)``.
        new_w: Width of the resized content region, ``round(W * r)``.
        pad_left: Left padding (floor of half the horizontal slack).
        pad_top: Top padding (floor of half the vertical slack).
        out_h: Target canvas height.
        out_w: Target canvas width.
    """

    r: float
    new_h: int
    new_w: int
    pad_left: int
    pad_top: int
    out_h: int
    out_w: int

    def forward_matrix(self, dtype: torch.dtype) -> Tensor:
        """Return the ``3x3`` original-to-letterboxed affine in ``dtype``."""
        return torch.tensor(
            [[self.r, 0.0, self.pad_left], [0.0, self.r, self.pad_top], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )

    def inverse_matrix(self, dtype: torch.dtype) -> Tensor:
        """Return the ``3x3`` letterboxed-to-original affine in ``dtype``."""
        inv_r = 1.0 / self.r
        return torch.tensor(
            [[inv_r, 0.0, -self.pad_left * inv_r], [0.0, inv_r, -self.pad_top * inv_r], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )


def _resolve_geometry(orig_h: int, orig_w: int, out_h: int, out_w: int, allow_upscale: bool) -> _LetterboxGeom:
    """Compute the letterbox geometry mapping ``(orig_h, orig_w)`` into ``(out_h, out_w)``.

    Args:
        orig_h: Original image height.
        orig_w: Original image width.
        out_h: Target canvas height.
        out_w: Target canvas width.
        allow_upscale: If ``False``, clamp the ratio at ``1.0`` so the content is
            never enlarged beyond its native size.

    Returns:
        The resolved :class:`_LetterboxGeom`.
    """
    r = min(out_h / orig_h, out_w / orig_w)
    if not allow_upscale:
        r = min(r, 1.0)
    new_w = max(1, round(orig_w * r))
    new_h = max(1, round(orig_h * r))
    pad_left = (out_w - new_w) // 2
    pad_top = (out_h - new_h) // 2
    return _LetterboxGeom(r=r, new_h=new_h, new_w=new_w, pad_left=pad_left, pad_top=pad_top, out_h=out_h, out_w=out_w)


class Letterbox:
    """Aspect-preserving resize-and-pad transform with an exact inverse (A10).

    The image is scaled by ``r = min(target_h / H, target_w / W)`` (a single
    ratio for both axes), resized with antialiased bilinear interpolation and
    padded symmetrically to the target canvas with ``pad_value``. Boxes, polygons
    and rotated-box centres are warped through the same scale-plus-translation
    affine; rotated-box ``w``/``h`` scale by ``r`` and ``theta`` is unchanged.

    Args:
        target_size: Target canvas as a single ``int`` (square) or an explicit
            ``(height, width)`` pair.
        pad_value: Constant fill for the padded border, in the image's value
            range. Defaults to ``114/255`` (mid-grey, the YOLO-lineage
            convention).
        allow_upscale: Whether the content may be enlarged beyond native size
            when the target is larger than the source. Defaults to ``True`` for
            training-time parity; set ``False`` to cap the ratio at ``1.0``.

    Examples:
        ```pycon
        >>> from open_yolos.data.transforms import GeometricTransform
        >>> isinstance(Letterbox(640), GeometricTransform)
        True

        ```
    """

    def __init__(
        self,
        target_size: int | tuple[int, int],
        pad_value: float = _DEFAULT_PAD_VALUE,
        allow_upscale: bool = True,
    ) -> None:
        self.out_h, self.out_w = _as_hw(target_size)
        self.pad_value = float(pad_value)
        self.allow_upscale = allow_upscale

    def __call__(self, image: Tensor, targets: Targets) -> tuple[Tensor, Targets]:
        """Letterbox ``image`` and warp ``targets`` through the matching affine.

        Args:
            image: CHW image tensor (float, in the same value range as
                ``pad_value``).
            targets: Geometry to warp alongside the image.

        Returns:
            The padded ``(C, target_h, target_w)`` image and the warped targets.

        Examples:
            ```pycon
            >>> import torch
            >>> from open_yolos.data.targets import Targets
            >>> lb = Letterbox(4)
            >>> image = torch.zeros(3, 2, 4)
            >>> t = Targets(boxes=torch.tensor([[0.0, 0.0, 4.0, 2.0]]), labels=torch.tensor([0]))
            >>> out_image, out_targets = lb(image, t)
            >>> out_image.shape
            torch.Size([3, 4, 4])
            >>> out_targets.boxes
            tensor([[0., 1., 4., 3.]])

            ```
        """
        _, height, width = image.shape
        geom = _resolve_geometry(height, width, self.out_h, self.out_w, self.allow_upscale)
        out_image = self._resize_pad(image, geom)
        out_targets = self._warp_targets(targets, geom)
        return out_image, out_targets

    def inverse_map(self, points: Tensor, orig_size: tuple[int, int], letterboxed_size: tuple[int, int]) -> Tensor:
        """Map letterboxed points back to original-image coordinates exactly.

        Args:
            points: ``(K, 2)`` points in the letterboxed canvas.
            orig_size: Original image ``(height, width)``.
            letterboxed_size: Letterboxed canvas ``(height, width)`` the points
                live in.

        Returns:
            ``(K, 2)`` points in original-image coordinates, in ``points``'s
            dtype.

        Examples:
            ```pycon
            >>> import torch
            >>> lb = Letterbox(4)
            >>> pts = torch.tensor([[0.0, 1.0], [4.0, 3.0]])
            >>> lb.inverse_map(pts, orig_size=(2, 4), letterboxed_size=(4, 4))
            tensor([[0., 0.],
                    [4., 2.]])

            ```
        """
        orig_h, orig_w = orig_size
        out_h, out_w = letterboxed_size
        geom = _resolve_geometry(orig_h, orig_w, out_h, out_w, self.allow_upscale)
        return self._warp(points, geom.inverse_matrix(torch.float64))

    def forward_affine(self, orig_h: int, orig_w: int) -> tuple[Tensor, int, int]:
        """Return the forward affine and output size mapping a source into this canvas.

        Exposes the pure scale-and-translation letterbox geometry for a source of
        ``(orig_h, orig_w)`` without resampling any image, so a caller can compose
        it into another warp (see
        :class:`~open_yolos.data.affine.FusedAffineLetterbox`).

        Args:
            orig_h: Source image height in pixels.
            orig_w: Source image width in pixels.

        Returns:
            A ``(matrix, out_h, out_w)`` triple: the ``(3, 3)`` float64 forward
            affine mapping source pixels into the letterboxed canvas, and the
            target canvas height and width.

        Examples:
            ```pycon
            >>> import torch
            >>> matrix, out_h, out_w = Letterbox(4).forward_affine(8, 8)
            >>> (out_h, out_w)
            (4, 4)
            >>> matrix[:2, :2]  # r = min(4/8, 4/8) = 0.5 on both axes
            tensor([[0.5000, 0.0000],
                    [0.0000, 0.5000]], dtype=torch.float64)

            ```
        """
        geom = _resolve_geometry(orig_h, orig_w, self.out_h, self.out_w, self.allow_upscale)
        return geom.forward_matrix(torch.float64), geom.out_h, geom.out_w

    def warp_targets(self, targets: Targets, orig_h: int, orig_w: int) -> Targets:
        """Warp ``targets`` through the letterbox affine for a source size, no image.

        Applies the same forward affine :meth:`__call__` applies to the geometry —
        boxes and polygons mapped point-wise, rotated-box centres warped with
        extents scaled and angle fixed — but touches no image. It lets a fused warp
        letterbox its targets after another transform has already produced them at
        the source canvas.

        Args:
            targets: Geometry at the source canvas to map into the letterboxed
                canvas.
            orig_h: Source image height in pixels.
            orig_w: Source image width in pixels.

        Returns:
            A new :class:`~open_yolos.data.targets.Targets` in letterboxed-canvas
            coordinates.

        Examples:
            ```pycon
            >>> import torch
            >>> from open_yolos.data.targets import Targets
            >>> box = Targets(boxes=torch.tensor([[0.0, 0.0, 4.0, 2.0]]), labels=torch.tensor([0]))
            >>> Letterbox(4).warp_targets(box, orig_h=2, orig_w=4).boxes
            tensor([[0., 1., 4., 3.]])

            ```
        """
        geom = _resolve_geometry(orig_h, orig_w, self.out_h, self.out_w, self.allow_upscale)
        return self._warp_targets(targets, geom)

    def _resize_pad(self, image: Tensor, geom: _LetterboxGeom) -> Tensor:
        """Resize ``image`` to the content region and pad it to the target canvas."""
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(geom.new_h, geom.new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        pad_right = geom.out_w - geom.new_w - geom.pad_left
        pad_bottom = geom.out_h - geom.new_h - geom.pad_top
        return F.pad(
            resized, (geom.pad_left, pad_right, geom.pad_top, pad_bottom), mode="constant", value=self.pad_value
        )

    def _warp_targets(self, targets: Targets, geom: _LetterboxGeom) -> Targets:
        """Warp every modality of ``targets`` through the forward affine."""
        matrix = geom.forward_matrix(torch.float64)
        boxes = self._warp(targets.boxes.reshape(-1, 2), matrix).reshape(-1, 4)
        polygons = [self._warp(ring, matrix) for ring in targets.polygons]
        rboxes = self._warp_rboxes(targets.rboxes, matrix, geom.r)
        return Targets(boxes=boxes, labels=targets.labels.clone(), polygons=polygons, rboxes=rboxes)

    def _warp_rboxes(self, rboxes: Tensor, matrix: Tensor, r: float) -> Tensor:
        """Warp rotated boxes: centres as points, ``w``/``h`` scaled by ``r``, ``theta`` fixed."""
        centers = self._warp(rboxes[:, :2], matrix)
        wh = rboxes[:, 2:4] * r
        theta = rboxes[:, 4:5]
        return torch.cat([centers, wh, theta], dim=1)

    @staticmethod
    def _warp(points: Tensor, matrix: Tensor) -> Tensor:
        """Apply ``matrix`` to ``points`` in the matrix dtype, restoring the input dtype.

        The affine is carried in ``float64`` for an exact inverse; the shared
        helper requires matching dtypes, so points are promoted for the matmul
        and the result is cast back to ``points``'s original dtype.
        """
        warped = apply_affine_to_points(points.to(matrix.dtype), matrix)
        return warped.to(points.dtype)
