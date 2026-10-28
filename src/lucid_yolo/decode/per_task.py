# SPDX-License-Identifier: Apache-2.0
"""Per-task decode compositions shared by prediction, export and evaluation.

A segmentation or keypoint detection is a box **and** a per-anchor value beside it —
Eq. 7's mask coefficients, the point stem's pose. Both are produced the same way: one
branch is ranked by its decoder, the decoder reports the anchor each kept row came
from, and the per-anchor value is gathered by that index. The functions here are that
composition, written once, so :mod:`lucid_yolo.predict`, :mod:`lucid_yolo.export` and
:class:`~lucid_yolo.eval.coco_eval.DualPathEvaluator` read one decode rather than three
spellings of it. Before they existed the mask-coefficient gather was written three ways
and the keypoint gather two, held together only by a test replica.

What stays outside is each caller's own business: :mod:`lucid_yolo.predict` drops
low-scoring and padding rows **after** the shared decode, an export graph keeps all ``k``
rows because its output shape is fixed, and the evaluator scores every row. The masks
themselves are assembled by each caller through
:func:`~lucid_yolo.eval.segment_decode.decode_instance_masks`, on whichever rows it
keeps; this module returns the gathered coefficients, not the masks.

The oriented task needs no composition here: its per-anchor value, the angle, is part of
the A45 tuple, and :class:`~lucid_yolo.decode.topk_e2e.RotatedTopKDecoder` is its decode.

Provenance: R1 sec. 3.2.1, R1 Eq. 7, R3 sec. 4. Assumptions: A9, A37, A64.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lucid_yolo.decode.common import gather_by_anchor
from lucid_yolo.models.heads.keypoint import decode_keypoints

if TYPE_CHECKING:
    from torch import Tensor

    from lucid_yolo.decode.common import IndexingDecoder

__all__ = ["decode_with_coefficients", "decode_with_keypoints", "gather_keypoint_sets"]


def decode_with_coefficients(
    decoder: IndexingDecoder,
    cls_logits: Tensor,
    raw_ltrb: Tensor,
    coefficients: Tensor,
    anchor_points: Tensor,
    strides: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode one branch's boxes and gather each row's mask coefficients by its own anchor.

    The segmentation half of Eq. 7 before the prototypes: ``decoder`` ranks the branch
    and reports each row's source anchor, and
    :func:`~lucid_yolo.decode.common.gather_by_anchor` takes that anchor's coefficient
    vector. All four inputs must come from the **same** branch — the decoder's ranking
    of one branch paired with the other branch's coefficients is a plausible mask of a
    different object at a correct box.

    Args:
        decoder: The path's decoder, e.g.
            :class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` or
            :class:`~lucid_yolo.decode.nms_path.NMSDecoder`.
        cls_logits: Raw class logits of shape ``(B, A, C)``.
        raw_ltrb: Raw ltrb distances of shape ``(B, A, 4)``.
        coefficients: The same branch's mask coefficients, shape ``(B, A, M)``.
        anchor_points: Anchor-centre ``(x, y)`` coordinates of shape ``(A, 2)``.
        strides: Per-anchor level stride of shape ``(A,)``.

    Returns:
        A row-aligned triple: the decoder's ``(B, N, 6)`` detections, its ``(B, N)``
        anchor indices, and the ``(B, N, M)`` gathered coefficients, all-zero on
        padding rows.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.assign.grid import make_anchor_points
        >>> from lucid_yolo.decode.topk_e2e import TopKDecoder
        >>> points, strides = make_anchor_points([(1, 2)], [8])
        >>> cls_logits = torch.tensor([[[-5.0], [5.0]]])  # anchor 1 is the confident one
        >>> coefficients = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
        >>> _, anchors, gathered = decode_with_coefficients(
        ...     TopKDecoder(k=3), cls_logits, torch.ones(1, 2, 4), coefficients, points, strides
        ... )
        >>> anchors
        tensor([[ 1,  0, -1]])
        >>> gathered  # anchor 1's vector first, the padding row zeroed
        tensor([[[2., 2.],
                 [1., 1.],
                 [0., 0.]]])

        ```
    """
    detections, anchor_index = decoder.decode_with_indices(cls_logits, raw_ltrb, anchor_points, strides)
    return detections, anchor_index, gather_by_anchor(coefficients, anchor_index)


def gather_keypoint_sets(raw_points: Tensor, anchor_index: Tensor, anchor_points: Tensor, strides: Tensor) -> Tensor:
    """Decode one branch's dense point offsets and take each detection's point set.

    Decode first, gather second, and the order is forced:
    :func:`~lucid_yolo.models.heads.keypoint.decode_keypoints` composes the raw offsets
    with the whole anchor grid and has no per-detection indexing, so gathering first
    would mean reconstructing each kept row's anchor and stride by hand. The gather is
    :func:`~lucid_yolo.decode.common.gather_by_anchor`, so a padding row yields the
    origin rather than the last anchor's pose.

    Args:
        raw_points: Raw point offsets of shape ``(B, A, K, 2)``.
        anchor_index: Source anchor per detection, shape ``(B, N)``.
        anchor_points: Anchor-centre ``(x, y)`` coordinates of shape ``(A, 2)``.
        strides: Per-anchor level stride of shape ``(A,)``.

    Returns:
        Point coordinates in canvas pixels, shape ``(B, N, K, 2)``, padding rows zero.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.assign.grid import make_anchor_points
        >>> points, strides = make_anchor_points([(1, 2)], [8])
        >>> raw_points = torch.zeros(1, 2, 1, 2)  # one point per anchor, no offset
        >>> gather_keypoint_sets(raw_points, torch.tensor([[1, -1]]), points, strides)
        tensor([[[[12.,  4.]],
        <BLANKLINE>
                 [[ 0.,  0.]]]])

        ```
    """
    return gather_by_anchor(decode_keypoints(raw_points, anchor_points, strides), anchor_index)


def decode_with_keypoints(
    decoder: IndexingDecoder,
    cls_logits: Tensor,
    raw_ltrb: Tensor,
    raw_points: Tensor,
    anchor_points: Tensor,
    strides: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode one branch's boxes and take each row's decoded point set by its own anchor.

    :func:`decode_with_coefficients` for the point stem: ``decoder`` ranks the branch
    and :func:`gather_keypoint_sets` decodes that branch's dense offsets and takes each
    kept row's set. All inputs must come from the same branch, for the reason
    :func:`decode_with_coefficients` states.

    Args:
        decoder: The path's decoder, e.g.
            :class:`~lucid_yolo.decode.topk_e2e.TopKDecoder` or
            :class:`~lucid_yolo.decode.nms_path.NMSDecoder`.
        cls_logits: Raw class logits of shape ``(B, A, C)``.
        raw_ltrb: Raw ltrb distances of shape ``(B, A, 4)``.
        raw_points: The same branch's raw point offsets, shape ``(B, A, K, 2)``.
        anchor_points: Anchor-centre ``(x, y)`` coordinates of shape ``(A, 2)``.
        strides: Per-anchor level stride of shape ``(A,)``.

    Returns:
        A row-aligned triple: the decoder's ``(B, N, 6)`` detections, its ``(B, N)``
        anchor indices, and the ``(B, N, K, 2)`` point sets in canvas pixels, zero on
        padding rows.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.assign.grid import make_anchor_points
        >>> from lucid_yolo.decode.topk_e2e import TopKDecoder
        >>> points, strides = make_anchor_points([(1, 2)], [8])
        >>> cls_logits = torch.tensor([[[-5.0], [5.0]]])
        >>> _, _, sets = decode_with_keypoints(
        ...     TopKDecoder(k=3), cls_logits, torch.ones(1, 2, 4), torch.zeros(1, 2, 1, 2), points, strides
        ... )
        >>> sets[0, :, 0]  # anchor 1's centre, anchor 0's, then the padding row's origin
        tensor([[12.,  4.],
                [ 4.,  4.],
                [ 0.,  0.]])

        ```
    """
    detections, anchor_index = decoder.decode_with_indices(cls_logits, raw_ltrb, anchor_points, strides)
    return detections, anchor_index, gather_keypoint_sets(raw_points, anchor_index, anchor_points, strides)
