# SPDX-License-Identifier: Apache-2.0
"""Per-image target container carried through the augmentation pipeline.

The :class:`Targets` dataclass is the single mutable-shape record every geometric
transform (letterbox, affine, mosaic, mixup, flip — WP-009…013) reads and
rewrites. It holds all target modalities the reproduction touches so one
transform call keeps axis-aligned boxes, instance polygons and rotated boxes in
lock-step (blueprint sec. 5.9: "all geometric transforms must operate
consistently on boxes, polygon/instance masks, and rotated boxes").

Conventions:
    * ``boxes`` — ``xyxy`` pixel coordinates, ``float32``.
    * ``labels`` — integer class ids, ``int64``, one per box.
    * ``polygons`` — per-instance point rings; either absent (empty list) or one
      ring per box.
    * ``rboxes`` — rotated boxes in **long-edge** form ``(cx, cy, w, h, theta)``
      with ``w >= h`` and ``theta`` in radians on ``[-pi/4, 3*pi/4)``. This module
      only *carries* rotated boxes; canonicalization (the ``w >= h`` / angle-range
      guarantee) is implemented and enforced in WP-055, so ``__post_init__`` here
      validates shape and dtype only, never the angle range.
    * ``difficult`` — R18's per-instance ``difficult`` flag, on the shared instance
      axis. Omitted means "no instance is difficult", which is what every
      non-oriented dataset means (WP-088).
    * ``keypoints`` / ``keypoint_vis`` — ``(N, K, 2)`` float32 xy coordinates and
      matching ``(N, K)`` int64 visibility values on the shared instance axis.

The ``difficult`` channel (A48, A51):
    R18 marks instances a detector is not penalised for missing *or* for finding,
    and :func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map` implements that rule
    — but only if the flag reaches it. Filtering difficult instances at load instead
    (``keep_difficult=False``) deletes the ground truth an ignorable detection would
    have matched, so every such detection is scored as a false positive and the
    metric is silently depressed. A48 therefore makes ``keep_difficult=True`` a
    requirement of the *evaluation* loader, and this field is the channel that
    obligation needs: without somewhere to put the flag, keeping the instance and
    dropping the instance are the same thing downstream.

    The flag survives :meth:`Targets.filter`, :meth:`Targets.concat`,
    :meth:`Targets.clone`, the letterbox transform (the only geometric transform on
    the evaluation path) and the loader's packed transport. Train-time
    multi-image assemblies rebuild their instance axis and are documented at each
    site; the flag is not consumed in training at all (A39 leaves that policy open),
    so this WP evaluates per tile and trains on whatever the loader kept.

Angles are radians throughout. Images are CHW ``float`` tensors elsewhere in the
pipeline; this container is image-agnostic and stores geometry only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

__all__ = ["Targets"]

#: Column count of an ``xyxy`` axis-aligned box.
_BOX_DIM = 4
#: Column count of a long-edge rotated box ``(cx, cy, w, h, theta)``.
_RBOX_DIM = 5
#: Column count of a polygon point ``(x, y)``.
_POINT_DIM = 2
#: Coordinate count of a keypoint ``(x, y)``.
_KEYPOINT_DIM = 2


def _empty_boxes() -> Tensor:
    """Return an empty ``(0, 4)`` float32 box tensor."""
    return torch.zeros((0, _BOX_DIM), dtype=torch.float32)


def _empty_labels() -> Tensor:
    """Return an empty ``(0,)`` int64 label tensor."""
    return torch.zeros((0,), dtype=torch.int64)


def _empty_rboxes() -> Tensor:
    """Return an empty ``(0, 5)`` float32 rotated-box tensor."""
    return torch.zeros((0, _RBOX_DIM), dtype=torch.float32)


def _empty_keypoints() -> Tensor:
    """Return an empty ``(0, 0, 2)`` float32 keypoint tensor."""
    return torch.zeros((0, 0, _KEYPOINT_DIM), dtype=torch.float32)


def _empty_keypoint_vis() -> Tensor:
    """Return an empty ``(0, 0)`` int64 keypoint-visibility tensor."""
    return torch.zeros((0, 0), dtype=torch.int64)


def _empty_flags() -> Tensor:
    """Return an empty ``(0,)`` bool flag tensor (the "no instance is difficult" default)."""
    return torch.zeros((0,), dtype=torch.bool)


def _validate_mask(mask: Tensor, expected_len: int, name: str) -> None:
    """Validate a boolean selection mask against an expected axis length.

    Args:
        mask: Candidate 1-D boolean tensor.
        expected_len: Length the mask must have.
        name: Human-readable mask name used in error messages.

    Raises:
        TypeError: If ``mask`` is not of dtype ``torch.bool``.
        ValueError: If ``mask`` is not 1-D of length ``expected_len``.
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} mask must be bool; got {mask.dtype}")
    if mask.ndim != 1 or mask.shape[0] != expected_len:
        raise ValueError(f"{name} mask must be 1-D of length {expected_len}; got shape {tuple(mask.shape)}")


@dataclass
class Targets:
    """Geometry for a single image, kept aligned across the transform pipeline.

    The ``boxes``/``labels``/``polygons`` triple shares one instance axis of
    length ``N``: ``labels`` has one entry per box and ``polygons``, when present,
    has one ring per box. ``rboxes`` lives on an independent axis of length ``M``
    and carries no labels in this container (rotated-target labelling lands with
    the OBB head in Phase 8).

    Attributes:
        boxes: ``(N, 4)`` float32 axis-aligned boxes in ``xyxy`` pixel coords.
        labels: ``(N,)`` int64 class ids, one per box.
        polygons: List of ``(P_i, 2)`` float32 instance rings; length ``0`` (no
            masks) or ``N`` (one ring per box).
        rboxes: ``(M, 5)`` float32 long-edge rotated boxes
            ``(cx, cy, w, h, theta)``; empty ``(0, 5)`` when there are none.
        difficult: ``(N,)`` bool R18 difficult flags on the shared instance axis.
            Defaults to empty, which ``__post_init__`` expands to an all-``False``
            row per instance — so "omitted" and "nothing is difficult" are the same
            statement and every consumer may read the field unconditionally.
        keypoints: ``(N, K, 2)`` float32 xy pixel coordinates. Its count is either
            zero (absent) or ``N`` (one K-point set per box).
        keypoint_vis: ``(N, K)`` int64 visibility values paired with ``keypoints``.
            This container preserves values without constraining their meaning.

    Examples:
        ```pycon
        >>> import torch
        >>> t = Targets(
        ...     boxes=torch.tensor([[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 2.0, 2.0]]),
        ...     labels=torch.tensor([3, 7]),
        ... )
        >>> len(t.labels)
        2
        >>> t.filter(torch.tensor([True, False])).labels.tolist()
        [3]

        ```
    """

    boxes: Tensor
    labels: Tensor
    polygons: list[Tensor] = field(default_factory=list)
    rboxes: Tensor = field(default_factory=_empty_rboxes)
    difficult: Tensor = field(default_factory=_empty_flags)
    keypoints: Tensor = field(default_factory=_empty_keypoints)
    keypoint_vis: Tensor = field(default_factory=_empty_keypoint_vis)

    def __post_init__(self) -> None:
        """Validate shapes and dtypes of every modality; raise on any mismatch."""
        self._validate_boxes_labels()
        self._validate_polygons()
        self._validate_rboxes()
        self._validate_keypoints()
        self._resolve_difficult()

    def _validate_boxes_labels(self) -> None:
        """Check box/label shapes, dtypes and their shared length."""
        if self.boxes.ndim != 2 or self.boxes.shape[1] != _BOX_DIM:
            raise ValueError(f"boxes must be (N, 4); got shape {tuple(self.boxes.shape)}")
        if self.boxes.dtype != torch.float32:
            raise TypeError(f"boxes must be float32; got {self.boxes.dtype}")
        if self.labels.ndim != 1:
            raise ValueError(f"labels must be 1-D (N,); got shape {tuple(self.labels.shape)}")
        if self.labels.dtype != torch.int64:
            raise TypeError(f"labels must be int64; got {self.labels.dtype}")
        if self.labels.shape[0] != self.boxes.shape[0]:
            raise ValueError(
                f"boxes/labels length mismatch: {self.boxes.shape[0]} boxes, {self.labels.shape[0]} labels"
            )

    def _validate_polygons(self) -> None:
        """Check polygon count is 0 or N and every ring is a float32 ``(P, 2)``."""
        n = self.boxes.shape[0]
        if len(self.polygons) not in (0, n):
            raise ValueError(f"polygons count must be 0 or N={n}; got {len(self.polygons)}")
        for i, poly in enumerate(self.polygons):
            if poly.ndim != 2 or poly.shape[1] != _POINT_DIM:
                raise ValueError(f"polygon[{i}] must be (P, 2); got shape {tuple(poly.shape)}")
            if poly.dtype != torch.float32:
                raise TypeError(f"polygon[{i}] must be float32; got {poly.dtype}")

    def _validate_rboxes(self) -> None:
        """Check rotated-box shape and dtype (angle range is a WP-055 concern)."""
        if self.rboxes.ndim != 2 or self.rboxes.shape[1] != _RBOX_DIM:
            raise ValueError(f"rboxes must be (M, 5); got shape {tuple(self.rboxes.shape)}")
        if self.rboxes.dtype != torch.float32:
            raise TypeError(f"rboxes must be float32; got {self.rboxes.dtype}")

    def _validate_keypoints(self) -> None:
        """Check optional K-point coordinates and visibility share the instance axis."""
        if self.keypoints.ndim < 1:
            raise ValueError(f"keypoints must be (N, K, 2); got shape {tuple(self.keypoints.shape)}")
        if self.keypoint_vis.ndim < 1:
            raise ValueError(f"keypoint_vis must be (N, K); got shape {tuple(self.keypoint_vis.shape)}")

        count = self.keypoints.shape[0]
        box_count = self.boxes.shape[0]
        if count not in (0, box_count):
            raise ValueError(f"keypoints count must be 0 or N={box_count}; got {count}")
        if count == 0:
            if self.keypoint_vis.shape[0] != 0:
                raise ValueError(
                    f"keypoint_vis count must be 0 when keypoints is empty; got {self.keypoint_vis.shape[0]}"
                )
            return
        if self.keypoints.ndim != 3 or self.keypoints.shape[-1] != _KEYPOINT_DIM:
            raise ValueError(f"keypoints must be (N, K, 2); got shape {tuple(self.keypoints.shape)}")
        if self.keypoints.dtype != torch.float32:
            raise TypeError(f"keypoints must be float32; got {self.keypoints.dtype}")
        if self.keypoint_vis.shape != self.keypoints.shape[:2]:
            raise ValueError(
                f"keypoint_vis must be (N, K) matching keypoints shape {tuple(self.keypoints.shape[:2])}; "
                f"got shape {tuple(self.keypoint_vis.shape)}"
            )
        if self.keypoint_vis.dtype != torch.int64:
            raise TypeError(f"keypoint_vis must be int64; got {self.keypoint_vis.dtype}")

    def _resolve_difficult(self) -> None:
        """Expand an omitted ``difficult`` to one ``False`` per instance, then validate it.

        Filling the default here rather than leaving it empty is what lets every
        consumer index the field alongside ``labels`` without first asking whether the
        producer supplied it — the alternative is an ``Optional`` that each of them
        re-defaults, differently.
        """
        count = self.boxes.shape[0]
        if self.difficult.numel() == 0 and count:
            self.difficult = torch.zeros((count,), dtype=torch.bool)
            return
        if self.difficult.dtype != torch.bool:
            raise TypeError(f"difficult must be bool; got {self.difficult.dtype}")
        if self.difficult.ndim != 1 or self.difficult.shape[0] != count:
            raise ValueError(f"difficult must be 1-D of length {count}; got shape {tuple(self.difficult.shape)}")

    def clone(self) -> Targets:
        """Return a deep copy whose every tensor is independent of ``self``.

        Returns:
            A new :class:`Targets` with cloned box/label/rbox/keypoint/visibility
            tensors and a fresh list of cloned polygon rings; mutating either leaves
            the other intact.

        Examples:
            ```pycon
            >>> import torch
            >>> a = Targets(boxes=torch.zeros((1, 4)), labels=torch.zeros(1, dtype=torch.int64))
            >>> b = a.clone()
            >>> b.boxes.add_(1.0)  # doctest: +ELLIPSIS
            tensor(...)
            >>> a.boxes.sum().item()
            0.0

            ```
        """
        return Targets(
            boxes=self.boxes.clone(),
            labels=self.labels.clone(),
            polygons=[poly.clone() for poly in self.polygons],
            rboxes=self.rboxes.clone(),
            difficult=self.difficult.clone(),
            keypoints=self.keypoints.clone(),
            keypoint_vis=self.keypoint_vis.clone(),
        )

    def filter(self, keep: Tensor, rkeep: Tensor | None = None) -> Targets:
        """Select a subset of targets, keeping every modality aligned.

        ``keep`` selects along the shared instance axis (boxes, labels, keypoints and,
        when present, polygons). Because rotated boxes live on an independent axis,
        they need their own mask: when ``rboxes`` is non-empty ``rkeep`` is
        **required**, and when ``rboxes`` is empty ``rkeep`` must be omitted. This
        keeps the two axes explicit rather than silently coupling their lengths.

        Args:
            keep: ``(N,)`` bool mask over the instance axis.
            rkeep: ``(M,)`` bool mask over the rotated-box axis. Required iff
                ``rboxes`` is non-empty; must be ``None`` otherwise.

        Returns:
            A new :class:`Targets` holding only the selected instances, keypoint rows,
            and rotated boxes; the returned tensors are independent copies.

        Raises:
            ValueError: If a mask has the wrong length, or ``rkeep`` is missing
                for non-empty ``rboxes`` / supplied for empty ``rboxes``.
            TypeError: If a mask is not boolean.

        Examples:
            ```pycon
            >>> import torch
            >>> t = Targets(
            ...     boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]),
            ...     labels=torch.tensor([1, 2]),
            ...     polygons=[torch.zeros((3, 2)), torch.ones((4, 2))],
            ... )
            >>> kept = t.filter(torch.tensor([False, True]))
            >>> kept.labels.tolist(), len(kept.polygons)
            ([2], 1)

            ```
        """
        _validate_mask(keep, self.boxes.shape[0], "keep")
        selected_polygons = (
            [poly.clone() for poly, flag in zip(self.polygons, keep.tolist(), strict=True) if flag]
            if self.polygons
            else []
        )
        keypoints, keypoint_vis = self._filter_keypoints(keep)
        return Targets(
            boxes=self.boxes[keep].clone(),
            labels=self.labels[keep].clone(),
            polygons=selected_polygons,
            rboxes=self._filter_rboxes(rkeep),
            difficult=self.difficult[keep].clone(),
            keypoints=keypoints,
            keypoint_vis=keypoint_vis,
        )

    def _filter_rboxes(self, rkeep: Tensor | None) -> Tensor:
        """Apply (or reject) the rotated-box mask per the :meth:`filter` contract."""
        if self.rboxes.shape[0] == 0:
            if rkeep is not None:
                raise ValueError("rkeep must be None when rboxes is empty")
            return _empty_rboxes()
        if rkeep is None:
            raise ValueError("rboxes is non-empty; a separate rkeep mask is required")
        _validate_mask(rkeep, self.rboxes.shape[0], "rkeep")
        return self.rboxes[rkeep].clone()

    def _filter_keypoints(self, keep: Tensor) -> tuple[Tensor, Tensor]:
        """Select shared-axis keypoint rows, preserving the canonical empty pair."""
        if self.keypoints.shape[0] == 0:
            return _empty_keypoints(), _empty_keypoint_vis()
        return self.keypoints[keep].clone(), self.keypoint_vis[keep].clone()

    @classmethod
    def concat(cls, items: list[Targets]) -> Targets:
        """Concatenate several images' targets into one (mosaic/mix assembly).

        Boxes, labels and rotated boxes are concatenated along their axes. Polygon and
        keypoint presence must be consistent: either every contributing instance carries
        that modality or none does — a mix is rejected rather than silently dropped.

        Args:
            items: Targets to merge, in order. An empty list yields
                :meth:`empty`.

        Returns:
            A single :class:`Targets` with the merged instances, keypoint rows, and
            rotated boxes.

        Raises:
            ValueError: If polygon/keypoint presence is mixed across ``items``, or
                present items carry different keypoint counts.

        Examples:
            ```pycon
            >>> import torch
            >>> a = Targets(boxes=torch.zeros((1, 4)), labels=torch.tensor([0]))
            >>> b = Targets(boxes=torch.ones((2, 4)), labels=torch.tensor([1, 2]))
            >>> Targets.concat([a, b]).labels.tolist()
            [0, 1, 2]

            ```
        """
        if not items:
            return cls.empty()
        keypoints, keypoint_vis = cls._concat_keypoints(items)
        return cls(
            boxes=torch.cat([t.boxes for t in items], dim=0),
            labels=torch.cat([t.labels for t in items], dim=0),
            polygons=cls._concat_polygons(items),
            rboxes=torch.cat([t.rboxes for t in items], dim=0),
            difficult=torch.cat([t.difficult for t in items], dim=0),
            keypoints=keypoints,
            keypoint_vis=keypoint_vis,
        )

    @staticmethod
    def _concat_polygons(items: list[Targets]) -> list[Tensor]:
        """Merge polygon rings across ``items``, rejecting mixed presence."""
        merged: list[Tensor] = []
        for t in items:
            merged.extend(poly.clone() for poly in t.polygons)
        total_boxes = sum(t.boxes.shape[0] for t in items)
        if merged and len(merged) != total_boxes:
            raise ValueError(
                f"cannot concat targets with mixed polygon presence: {len(merged)} rings for {total_boxes} boxes"
            )
        return merged

    @staticmethod
    def _concat_keypoints(items: list[Targets]) -> tuple[Tensor, Tensor]:
        """Merge shared-axis keypoints, rejecting mixed presence or K values."""
        present = [target for target in items if target.keypoints.shape[0] > 0]
        total_boxes = sum(target.boxes.shape[0] for target in items)
        total_keypoint_rows = sum(target.keypoints.shape[0] for target in present)
        if present and total_keypoint_rows != total_boxes:
            raise ValueError(
                "cannot concat targets with mixed keypoint presence: "
                f"{total_keypoint_rows} keypoint rows for {total_boxes} boxes"
            )
        if not present:
            return _empty_keypoints(), _empty_keypoint_vis()

        keypoint_count = present[0].keypoints.shape[1]
        for target in present[1:]:
            other_count = target.keypoints.shape[1]
            if other_count != keypoint_count:
                raise ValueError(
                    f"cannot concat targets with differing keypoint counts: {keypoint_count} and {other_count}"
                )
        return (
            torch.cat([target.keypoints for target in present], dim=0),
            torch.cat([target.keypoint_vis for target in present], dim=0),
        )

    @classmethod
    def empty(cls) -> Targets:
        """Return a valid, fully-empty target set (zero instances, no rotated boxes or keypoints).

        Returns:
            A :class:`Targets` with ``(0, 4)`` boxes, ``(0,)`` labels, no polygons
            and ``(0, 5)`` rboxes / ``(0, 0, 2)`` keypoints — a safe identity for
            :meth:`concat` and a valid input to :meth:`filter`.

        Examples:
            ```pycon
            >>> t = Targets.empty()
            >>> t.boxes.shape, t.labels.shape, t.rboxes.shape, t.keypoints.shape
            (torch.Size([0, 4]), torch.Size([0]), torch.Size([0, 5]), torch.Size([0, 0, 2]))

            ```
        """
        return cls(
            boxes=_empty_boxes(),
            labels=_empty_labels(),
            polygons=[],
            rboxes=_empty_rboxes(),
            keypoints=_empty_keypoints(),
            keypoint_vis=_empty_keypoint_vis(),
        )
