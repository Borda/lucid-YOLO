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

The fit and the resample both come from ``fuse-augmentations`` (WP-155):
:func:`~fuse_augmentations.letterbox_geometry` resolves the ratio, the content
size and the pads, :func:`~fuse_augmentations.letterbox_matrix` builds the
forward affine above, and a letterbox-only
:class:`~fuse_augmentations.Compose` resamples the image in a single warp from
the source canvas straight to the letterboxed one. What stays local is the
inverse — built analytically from the same resolved fit rather than by inverting
a matrix numerically — and the routing of each target modality through the
forward affine.

:class:`Letterbox` conforms to
:class:`~lucid_yolo.data.transforms.GeometricTransform`: one call warps the image
and every modality carried by :class:`~lucid_yolo.data.targets.Targets` (boxes,
polygons, rotated boxes, keypoints) through the *same* affine, routed via
:func:`~lucid_yolo.data.transforms.apply_affine_to_points` so the four paths share
one source of geometric truth. Rotated boxes need only their centres warped as
points, their ``w``/``h`` scaled by ``r`` and ``theta`` left unchanged, since the
transform carries no rotation or shear. Keypoints are plain points and need less
than that: no extent to scale, no angle to fix (WP-132).

Inverse API:
    :meth:`Letterbox.inverse_map` is stateless in the image content — given a
    batch of letterboxed points, the original image size and the letterboxed
    canvas size it recomputes the ratio and padding and applies the inverse
    affine, returning original-image coordinates. It depends only on those sizes
    and the instance's ``allow_upscale`` flag, never on which image produced the
    predictions, which is exactly what an evaluation loop has on hand.
"""

from __future__ import annotations

import torch
from fuse_augmentations import (  # type: ignore[import-untyped]
    Compose,
    LetterboxGeometry,
    letterbox_geometry,
    letterbox_matrix,
)
from torch import Tensor

from lucid_yolo.data.targets import Targets
from lucid_yolo.data.transforms import apply_affine_to_points

__all__ = ["Letterbox"]

#: Default pad colour: mid-grey ``114/255`` per the YOLO-lineage convention (A10).
_DEFAULT_PAD_VALUE = 114.0 / 255.0

#: Coordinate count of a plane point ``(x, y)`` — the width every modality flattens to
#: before the shared affine maps it.
_POINT_DIM = 2


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


def _forward_matrix(orig_h: int, orig_w: int, out_h: int, out_w: int, allow_upscale: bool) -> Tensor:
    """Return the ``3x3`` float64 original-to-letterboxed affine for one size pairing.

    A thin adapter over :func:`~fuse_augmentations.letterbox_matrix`, which builds the
    matrix batched: it resolves the same fit :func:`~fuse_augmentations.letterbox_geometry`
    does and returns ``[[r, 0, pad_left], [0, r, pad_top], [0, 0, 1]]``. ``float64`` is
    what makes the inverse exact, and the leading batch axis is dropped because every
    caller here maps one size pairing at a time.

    Args:
        orig_h: Source image height in pixels.
        orig_w: Source image width in pixels.
        out_h: Target canvas height in pixels.
        out_w: Target canvas width in pixels.
        allow_upscale: Whether the fit may enlarge content past its native size.

    Returns:
        The ``(3, 3)`` float64 forward affine.
    """
    matrix: Tensor = letterbox_matrix(orig_h, orig_w, out_h, out_w, allow_upscale, dtype=torch.float64)[0]
    return matrix


def _inverse_matrix(geometry: LetterboxGeometry, dtype: torch.dtype) -> Tensor:
    """Return the ``3x3`` letterboxed-to-original affine for a resolved fit, in ``dtype``.

    Written out rather than obtained by inverting the forward matrix: a pure
    scale-plus-translation has a closed-form inverse, and spelling it keeps
    :meth:`Letterbox.inverse_map` free of the residual a numerical inversion would
    leave in the round trip A10 requires to be exact.

    Args:
        geometry: The resolved fit, from :func:`~fuse_augmentations.letterbox_geometry`.
        dtype: Dtype of the returned matrix.

    Returns:
        The ``(3, 3)`` inverse affine.
    """
    inv_r = 1.0 / geometry.r
    return torch.tensor(
        [[inv_r, 0.0, -geometry.pad_left * inv_r], [0.0, inv_r, -geometry.pad_top * inv_r], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )


class Letterbox:
    """Aspect-preserving resize-and-pad transform with an exact inverse (A10).

    The image is scaled by ``r = min(target_h / H, target_w / W)`` (a single
    ratio for both axes) and padded to the target canvas with ``pad_value``, in
    one upstream warp from the source canvas to the letterboxed one. Boxes,
    polygons, rotated-box centres and keypoints are warped through the same
    scale-plus-translation affine; rotated-box ``w``/``h`` scale by ``r`` and
    ``theta`` is unchanged, and keypoint visibilities carry over untouched (a
    letterbox never crops, so no annotated point leaves the canvas).

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
        >>> from lucid_yolo.data.transforms import GeometricTransform
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
        # A `fuse` pipeline holding nothing but the letterbox, built once: the canvas, the
        # fill and the upscale policy are all fixed here, and building it here is also what
        # makes an unusable `pad_value` fail at construction rather than at the first image.
        # A constant pad colour is spelled `fill=` and requires `padding_mode="zeros"` —
        # the value stays in the image's own range, so `114/255` transfers unchanged.
        self._resample = Compose.from_params(
            letterbox=(self.out_h, self.out_w),
            allow_upscale=self.allow_upscale,
            fill=self.pad_value,
            padding_mode="zeros",
        )

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
            >>> from lucid_yolo.data.targets import Targets
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
        # The pipeline warps a batch; this transform is per-sample, so the batch axis is
        # added and dropped around the one call rather than carried through the class.
        out_image: Tensor = self._resample(image.unsqueeze(0)).squeeze(0)
        return out_image, self.warp_targets(targets, height, width)

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
        geometry = letterbox_geometry(orig_h, orig_w, out_h, out_w, self.allow_upscale)
        return self._warp(points, _inverse_matrix(geometry, torch.float64))

    def forward_affine(self, orig_h: int, orig_w: int) -> tuple[Tensor, int, int]:
        """Return the forward affine and output size mapping a source into this canvas.

        Exposes the pure scale-and-translation letterbox geometry for a source of
        ``(orig_h, orig_w)`` without resampling any image, so a caller can compose
        it into another warp (see
        :class:`~lucid_yolo.data.affine.FusedAffineLetterbox`).

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
        return _forward_matrix(orig_h, orig_w, self.out_h, self.out_w, self.allow_upscale), self.out_h, self.out_w

    def warp_targets(self, targets: Targets, orig_h: int, orig_w: int) -> Targets:
        """Warp ``targets`` through the letterbox affine for a source size, no image.

        The single target-side implementation: :meth:`__call__` routes through it after
        resampling the image, and a fused warp calls it directly to letterbox targets
        another transform has already produced at the source canvas. Boxes, polygons and
        keypoints are mapped point-wise and rotated-box centres are warped with extents
        scaled and angle fixed — no image is touched either way.

        Args:
            targets: Geometry at the source canvas to map into the letterboxed
                canvas.
            orig_h: Source image height in pixels.
            orig_w: Source image width in pixels.

        Returns:
            A new :class:`~lucid_yolo.data.targets.Targets` in letterboxed-canvas
            coordinates.

        Examples:
            ```pycon
            >>> import torch
            >>> from lucid_yolo.data.targets import Targets
            >>> box = Targets(boxes=torch.tensor([[0.0, 0.0, 4.0, 2.0]]), labels=torch.tensor([0]))
            >>> Letterbox(4).warp_targets(box, orig_h=2, orig_w=4).boxes
            tensor([[0., 1., 4., 3.]])

            ```
        """
        matrix = _forward_matrix(orig_h, orig_w, self.out_h, self.out_w, self.allow_upscale)
        # The matrix carries the ratio in its diagonal, but a rotated box needs it as a
        # scalar; both come from the same upstream fit over the same arguments, so they
        # cannot disagree about what ``r`` is.
        ratio = float(letterbox_geometry(orig_h, orig_w, self.out_h, self.out_w, self.allow_upscale).r)
        boxes = self._warp(targets.boxes.reshape(-1, 2), matrix).reshape(-1, 4)
        polygons = [self._warp(ring, matrix) for ring in targets.polygons]
        rboxes = self._warp_rboxes(targets.rboxes, matrix, ratio)
        keypoints = self._warp_keypoints(targets.keypoints, matrix)
        # The instance axis is untouched by a letterbox, so the R18 difficult flags carry
        # over row for row, and so do the keypoint visibilities (WP-132). A letterbox
        # resizes and pads: it never crops and never pushes content off the canvas, so no
        # annotated point can stop being annotated here. Deciding what visibility a point
        # warped *out of frame* should take is a real question, but it belongs to the
        # transforms that crop — the mosaic, the fused affine, mixup — not to this one.
        # This is also the one geometric transform on the *evaluation* path, which is
        # exactly where A48 needs the difficult flag to survive (WP-088).
        return Targets(
            boxes=boxes,
            labels=targets.labels.clone(),
            polygons=polygons,
            rboxes=rboxes,
            difficult=targets.difficult.clone(),
            keypoints=keypoints,
            keypoint_vis=targets.keypoint_vis.clone(),
        )

    def _warp_keypoints(self, keypoints: Tensor, matrix: Tensor) -> Tensor:
        """Warp ``(N, K, 2)`` points through the affine, preserving the point axis.

        The points are flattened to a plain ``(N * K, 2)`` point list, mapped by the
        *same* matrix the box corners are mapped by, and folded back — so a keypoint
        and a box corner that coincide on the source canvas still coincide on the
        letterboxed one. Nothing keypoint-specific enters: there is no separate
        scaling rule, because a letterbox is one isotropic scale plus a translation
        and a point has no extent for that scale to act on differently.

        A keypoint-free ``Targets`` carries the canonical ``(0, 0, 2)`` empty, which
        round-trips through the reshape unchanged and yields the same empty back, so
        this is an exact no-op for every task that has no points.

        Args:
            keypoints: ``(N, K, 2)`` point coordinates on the source canvas.
            matrix: The ``(3, 3)`` forward affine, in ``float64``.

        Returns:
            The points in letterboxed-canvas coordinates, shaped and typed as given.
        """
        return self._warp(keypoints.reshape(-1, _POINT_DIM), matrix).reshape(keypoints.shape)

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
