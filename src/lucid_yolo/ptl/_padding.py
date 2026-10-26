# SPDX-License-Identifier: Apache-2.0
"""Padding and batch helpers of :class:`~lucid_yolo.ptl.module.DetectionLitModule`.

The ragged-to-dense padding of a batch's per-image
:class:`~lucid_yolo.data.targets.Targets` onto the instance axis
(:func:`pad_targets`, :func:`pad_rboxes`, :func:`pad_keypoints`, with packed scatter
or the ragged :func:`_pad_instance_axis` fallback), the step-batch type and its split, and the one
host transfer the validation accumulators make per field. Every name here stays
importable from :mod:`lucid_yolo.ptl.module`, where it was defined until the module
was split by responsibility; the three public padding functions keep their place in
that module's ``__all__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lucid_yolo.ptl._transport import PackedTargets, _PackedTargetList

if TYPE_CHECKING:
    from lucid_yolo.data.targets import Targets

#: Column count of an ``xyxy`` axis-aligned box.
_BOX_DIM = 4

#: Column count of a long-edge rotated box ``(cx, cy, w, h, theta)``.
_RBOX_DIM = 5

#: Coordinate count of a keypoint ``(x, y)``.
_KEYPOINT_DIM = 2


def _pad_instance_axis(
    pieces: list[Tensor],
    counts: list[int],
    max_n: int,
    trailing: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Copy each image's per-instance rows into one zero-filled tensor padded on the instance axis.

    The one write every padding function shares: a ``(B, max_n, *trailing)`` tensor of
    zeros, with image ``i``'s first ``counts[i]`` rows overwritten by ``pieces[i]`` (cast
    to ``dtype`` by the assignment) and an image with no instances skipped, so its row
    stays padding. The caller owns everything that differs between the padded fields:
    which tensor each image contributes, the counts it validated them against, the
    padded width and the device it read off the batch.

    Args:
        pieces: One tensor per image, each with ``counts[i]`` rows of shape ``trailing``.
        counts: Each image's instance count.
        max_n: The padded instance-axis length, the batch maximum of ``counts``.
        trailing: The per-instance shape after the instance axis; ``()`` for a scalar
            per instance.
        dtype: The padded tensor's dtype.
        device: The padded tensor's device.

    Returns:
        The ``(len(pieces), max_n, *trailing)`` padded tensor.

    Examples:
        ```pycon
        >>> pieces = [torch.tensor([[1.0, 2.0]]), torch.zeros(0, 2)]
        >>> _pad_instance_axis(pieces, [1, 0], 1, (2,), torch.float32, torch.device("cpu"))
        tensor([[[1., 2.]],
        <BLANKLINE>
                [[0., 0.]]])

        ```
    """
    padded = torch.zeros((len(pieces), max_n, *trailing), dtype=dtype, device=device)
    for index, (piece, count) in enumerate(zip(pieces, counts, strict=True)):
        if count:
            padded[index, :count] = piece
    return padded


def _pad_packed_axis(values: Tensor, packed: PackedTargets, dtype: torch.dtype, *, minimum: int = 0) -> Tensor:
    """Scatter flat packed rows into their final padded positions in one write."""
    counts = packed._counts("boxes_per_image")
    width = max(max(counts, default=0), minimum)
    padded = torch.zeros((len(counts), width, *values.shape[1:]), dtype=dtype, device=values.device)
    indices = packed._padding_indices
    assert indices is not None
    padded.flatten(0, 1).index_copy_(0, indices, values.to(dtype))
    return padded


def _pad_mask_targets(masks: list[Tensor], targets: list[Targets], reference: Tensor) -> Tensor:
    """Pad current mask stacks once, using packed row positions when still valid."""
    counts = [int(mask.shape[0]) for mask in masks]
    if (
        isinstance(targets, _PackedTargetList)
        and targets._flat_field("boxes") is not None
        and tuple(counts) == targets._packed._counts("boxes_per_image")
    ):
        # Float mask views share one cast of the packed bool stack. Replacements
        # rebuild from current slices; ordinary per-pixel edits need no copy.
        return _pad_packed_axis(targets._flat_masks(masks), targets._packed, reference.dtype, minimum=1)
    return _pad_instance_axis(
        masks, counts, max(*counts, 1), tuple(reference.shape[-2:]), reference.dtype, reference.device
    )


def pad_targets(targets: list[Targets]) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a ragged batch of per-image targets into dense ground-truth tensors.

    The batch's :class:`~lucid_yolo.data.targets.Targets` each carry their own
    instance count, so they are padded to the batch maximum ``N_max`` and paired
    with a boolean mask marking the real (non-padding) rows — the dense form the
    :class:`~lucid_yolo.losses.dual_loss.DualBranchLoss` assigners consume. An image
    with no instances contributes an all-``False`` mask row (no positives), and a
    batch in which *every* image is empty yields ``N_max = 0`` tensors, which the
    loss handles as its finite no-ground-truth case.

    Args:
        targets: Length-``B`` list of per-image targets (the datamodule batch's
            second element). All tensors must share one device.

    Returns:
        A triple ``(gt_boxes, gt_labels, gt_mask)`` with shapes ``(B, N_max, 4)``
        float32 ``xyxy`` boxes, ``(B, N_max)`` int64 class ids, and ``(B, N_max)``
        bool mask. Padding rows are zero boxes / zero labels / ``False`` mask.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> a = Targets(boxes=torch.tensor([[0.0, 0.0, 4.0, 4.0]]), labels=torch.tensor([3]))
        >>> b = Targets.empty()
        >>> boxes, labels, mask = pad_targets([a, b])
        >>> boxes.shape, labels.shape, mask.shape
        (torch.Size([2, 1, 4]), torch.Size([2, 1]), torch.Size([2, 1]))
        >>> mask.tolist()
        [[True], [False]]

        ```
    """
    if isinstance(targets, _PackedTargetList):
        boxes_cat, labels_cat = targets._flat_field("boxes"), targets._flat_field("labels")
        if boxes_cat is not None and labels_cat is not None:
            padded_boxes = _pad_packed_axis(boxes_cat, targets._packed, torch.float32)
            padded_labels = _pad_packed_axis(labels_cat, targets._packed, torch.int64)
            counts_tensor = targets._packed.boxes_per_image
            mask = torch.arange(padded_boxes.shape[1], device=padded_boxes.device)[None, :] < counts_tensor[:, None]
            return padded_boxes, padded_labels, mask
    batch_size = len(targets)
    counts = [int(target.boxes.shape[0]) for target in targets]
    max_n = max(counts) if counts else 0
    device = targets[0].boxes.device if targets else torch.device("cpu")
    boxes = [target.boxes for target in targets]
    gt_boxes = _pad_instance_axis(boxes, counts, max_n, (_BOX_DIM,), torch.float32, device)
    labels = [target.labels for target in targets]
    gt_labels = _pad_instance_axis(labels, counts, max_n, (), torch.int64, device)
    gt_mask = torch.zeros((batch_size, max_n), dtype=torch.bool, device=device)
    for index, count in enumerate(counts):
        if count:
            gt_mask[index, :count] = True
    return gt_boxes, gt_labels, gt_mask


def pad_rboxes(targets: list[Targets]) -> Tensor:
    """Pad a ragged batch's rotated boxes into one dense ``(B, N_max, 5)`` tensor (WP-088).

    The rotated companion of :func:`pad_targets`, kept as its own function rather than
    a fourth element of that tuple: only ``task="obb"`` needs it, and widening a return
    every caller unpacks would put an oriented concern in the detection path.

    The padding is by the **instance** axis, not by a rotated-box axis of its own. WP-056
    pins ``rboxes[i]`` to the same object as ``boxes[i]`` and ``labels[i]``, and that
    pairing is the whole reason the assignment computed on the axis-aligned envelopes may
    be reused to gather rotated targets. An image whose two axes disagree is therefore a
    hard error here rather than something to reconcile: whichever way it were resolved,
    some anchor would be supervised towards another instance's orientation.

    Args:
        targets: The length-``B`` per-image target list, each carrying one rotated box
            per instance. All tensors must share one device.

    Returns:
        ``(B, N_max, 5)`` float32 long-edge rotated boxes on the targets' device, padded
        with zero rows exactly where :func:`pad_targets` pads with ``False``.

    Raises:
        ValueError: If any image's rotated-box count differs from its instance count —
            including the all-important case of *zero* rotated boxes beside real ones,
            which is what a detection loader hands an oriented run.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> one = Targets(
        ...     boxes=torch.tensor([[0.0, 0.0, 4.0, 2.0]]),
        ...     labels=torch.tensor([3]),
        ...     rboxes=torch.tensor([[2.0, 1.0, 4.0, 2.0, 0.0]]),
        ... )
        >>> pad_rboxes([one, Targets.empty()]).shape
        torch.Size([2, 1, 5])
        >>> pad_rboxes([Targets(boxes=torch.zeros(1, 4), labels=torch.zeros(1, dtype=torch.int64))])
        Traceback (most recent call last):
        ...
        ValueError: image 0 carries 1 instances but 0 rotated boxes; task='obb' needs one per instance

        ```
    """
    counts = [int(target.boxes.shape[0]) for target in targets]
    for index, (target, count) in enumerate(zip(targets, counts, strict=True)):
        if int(target.rboxes.shape[0]) != count:
            raise ValueError(
                f"image {index} carries {count} instances but {int(target.rboxes.shape[0])} rotated boxes; "
                f"task='obb' needs one per instance"
            )
    if isinstance(targets, _PackedTargetList) and targets._flat_field("boxes") is not None:
        flat = targets._flat_field("rboxes")
        if flat is not None:
            return _pad_packed_axis(flat, targets._packed, torch.float32)
    max_n = max(counts) if counts else 0
    device = targets[0].rboxes.device if targets else torch.device("cpu")
    rboxes = [target.rboxes for target in targets]
    return _pad_instance_axis(rboxes, counts, max_n, (_RBOX_DIM,), torch.float32, device)


def pad_keypoints(targets: list[Targets]) -> tuple[Tensor, Tensor]:
    """Pad a ragged batch's keypoints and visibilities onto the instance axis (WP-132).

    The pose companion of :func:`pad_targets`, and a sibling of :func:`pad_rboxes` in
    both shape and argument. Like the rotated boxes, keypoints are padded by the
    **instance** axis rather than by an axis of their own: WP-120 pins
    ``keypoints[i]`` to the same object as ``boxes[i]`` and ``labels[i]``, and that
    pairing is the only reason the assignment computed on the boxes may be reused to
    gather point targets. An image whose two axes disagree is a hard error for the
    same reason it is one there — whichever way it were reconciled, some anchor would
    be supervised towards another instance's pose.

    The visibility padding is zeros, which is not merely a filler value: A66 reads
    ``v == 0`` as "no annotation exists", so
    :class:`~lucid_yolo.losses.rle_loss.RLELoss` already excludes exactly those
    points. The padding rows are therefore inert in the loss by the same rule that
    excludes a genuinely unlabeled point, rather than by a second mechanism that
    could disagree with it.

    Args:
        targets: The length-``B`` per-image target list, each carrying one point set
            per instance. All tensors must share one device.

    Returns:
        A pair ``(keypoints, visibility)`` of shapes ``(B, N_max, K, 2)`` float32 and
        ``(B, N_max, K)`` int64, padded with zero coordinates and zero (unlabeled)
        visibility exactly where :func:`pad_targets` pads with ``False``. ``K`` is
        the batch's shared point count, and ``0`` for a batch holding no instances
        at all.

    Raises:
        ValueError: If any image's keypoint-set count differs from its instance
            count — including the case of *zero* keypoints beside real instances,
            which is what a plain detection loader hands a keypoints run — or if two
            images disagree on ``K``, which no single head could predict for both.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> one = Targets(
        ...     boxes=torch.tensor([[0.0, 0.0, 4.0, 2.0]]),
        ...     labels=torch.tensor([3]),
        ...     keypoints=torch.tensor([[[1.0, 1.0], [3.0, 1.0]]]),
        ...     keypoint_vis=torch.tensor([[2, 1]]),
        ... )
        >>> coords, visibility = pad_keypoints([one, Targets.empty()])
        >>> coords.shape, visibility.shape
        (torch.Size([2, 1, 2, 2]), torch.Size([2, 1, 2]))
        >>> visibility.tolist()  # the padded image is unlabeled everywhere (A66)
        [[[2, 1]], [[0, 0]]]
        >>> pad_keypoints([Targets(boxes=torch.zeros(1, 4), labels=torch.zeros(1, dtype=torch.int64))])
        Traceback (most recent call last):
        ...
        ValueError: image 0 carries 1 instances but 0 keypoint sets; task='keypoints' needs one per instance

        ```
    """
    counts = [int(target.boxes.shape[0]) for target in targets]
    for index, (target, count) in enumerate(zip(targets, counts, strict=True)):
        if int(target.keypoints.shape[0]) != count:
            raise ValueError(
                f"image {index} carries {count} instances but {int(target.keypoints.shape[0])} keypoint sets; "
                f"task='keypoints' needs one per instance"
            )
    point_counts = {int(target.keypoints.shape[1]) for target in targets if target.keypoints.shape[0]}
    if len(point_counts) > 1:
        raise ValueError(f"the batch's images disagree on the keypoint count K: {sorted(point_counts)}")
    if isinstance(targets, _PackedTargetList) and targets._flat_field("boxes") is not None:
        flat, visibility = targets._flat_field("keypoints"), targets._flat_field("keypoint_vis")
        if flat is not None and visibility is not None:
            return (
                _pad_packed_axis(flat, targets._packed, torch.float32),
                _pad_packed_axis(visibility, targets._packed, torch.int64),
            )
    num_points = point_counts.pop() if point_counts else 0
    max_n = max(counts) if counts else 0
    device = targets[0].keypoints.device if targets else torch.device("cpu")
    points = [target.keypoints for target in targets]
    coords = _pad_instance_axis(points, counts, max_n, (num_points, _KEYPOINT_DIM), torch.float32, device)
    vis = [target.keypoint_vis for target in targets]
    visibility = _pad_instance_axis(vis, counts, max_n, (num_points,), torch.int64, device)
    return coords, visibility


#: A step batch. The datamodule's transfer hook produces the three-element form; the
#: two-element form is the hand-built feed a direct caller passes to
#: :meth:`DetectionLitModule.training_step`.
StepBatch = tuple[Tensor, list["Targets"]] | tuple[Tensor, list["Targets"], list[Tensor] | None]


def _host_split(tensors: list[Tensor]) -> tuple[Tensor, ...]:
    """Move a batch's per-image tensors to the host in one transfer rather than one each.

    Every validation accumulator needs its ground truth on the host. Moving each
    image's tensor on its own issues one device-to-host transfer per image per
    field, and what each one carries is a few kilobytes against the accelerator's
    fixed per-transfer latency -- a batch of 16 with three fields is 48 transfers
    for a payload that fits in one. Concatenating along the batch axis pays that
    latency once and splits the result back into the same per-image pieces, with
    the same values in the same order (WP-163).

    Only the tensors holding rows are concatenated. An image without instances may
    carry a different trailing shape: :meth:`~lucid_yolo.data.targets.Targets.empty`
    stores ``(0, 0, 2)`` keypoints whatever the schema's point count, so a keypoint
    batch mixing it with 17-point images disagrees after the first axis. Such a
    tensor holds no values to transfer and comes back with its own shape (WP-209).

    Args:
        tensors: One tensor per image. The tensors with at least one row agree on
            every dimension after the first; a tensor with none may differ. An empty
            list returns an empty tuple.

    Returns:
        The same tensors, on the host, in the order given.

    Examples:
        ```pycon
        >>> _host_split([torch.tensor([[1.0, 2.0]]), torch.zeros(0, 2)])
        (tensor([[1., 2.]]), tensor([], size=(0, 2)))
        >>> [tuple(piece.shape) for piece in _host_split([torch.zeros(2, 17, 2), torch.zeros(0, 0, 2)])]
        [(2, 17, 2), (0, 0, 2)]
        >>> _host_split([])
        ()

        ```
    """
    filled = [tensor for tensor in tensors if tensor.shape[0]]
    host = torch.cat(filled).cpu() if filled else None
    pieces: list[Tensor] = []
    start = 0
    for tensor in tensors:
        if host is None or not tensor.shape[0]:
            pieces.append(tensor.cpu())
            continue
        stop = start + int(tensor.shape[0])
        pieces.append(host[start:stop])
        start = stop
    return tuple(pieces)


def _split_batch(batch: StepBatch) -> tuple[Tensor, list[Targets], list[Tensor] | None]:
    """Split a step batch into images, targets and the loader's mask stacks.

    :meth:`~lucid_yolo.ptl.datamodule.DetectionDataModule.on_after_batch_transfer`
    hands over a third element — the per-image instance masks a segmentation loader
    rasterised in its workers, or ``None``. A two-element batch is still accepted so
    that a hand-built ``(images, targets)`` feed (every direct-call test, every
    script that drives the module without the datamodule) keeps working and simply
    rasterises in the step, as it always did.

    Args:
        batch: The ``(images, targets)`` or ``(images, targets, masks)`` step batch.

    Returns:
        The ``(images, targets, masks)`` triple, with ``masks`` ``None`` when the
        batch carried none.

    Examples:
        ```pycon
        >>> import torch
        >>> images, targets, masks = _split_batch((torch.zeros(1, 3, 4, 4), []))
        >>> masks is None
        True

        ```
    """
    images, targets, *rest = batch
    return images, targets, rest[0] if rest else None
