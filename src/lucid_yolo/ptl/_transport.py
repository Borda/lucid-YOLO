# SPDX-License-Identifier: Apache-2.0
"""Worker-to-consumer batch transport behind :mod:`lucid_yolo.ptl.datamodule`.

Holds the transport half of the datamodule's batch contract (described in that module's
docstring): :class:`PackedTargets` and its pack/unpack helpers, which flatten a ragged
``list[Targets]`` into a fixed set of dense tensors for the DataLoader's shared-memory hop;
the collates that build the ``(images, PackedTargets)`` pair from ordinary or
colour-deferred samples; and the restore that turns a transported batch back into float
images and ``list[Targets]`` on the consumer's device. Every public name here is
re-exported by :mod:`lucid_yolo.ptl.datamodule`, which stays the import path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from pytorch_lightning.plugins.precision import Precision
from torch import Tensor

from lucid_yolo.data._common import _POINT_DIM
from lucid_yolo.data._quantization import _dequantize_images, _quantize_images
from lucid_yolo.data.augment import HSVParams, _jitter_hsv_batch
from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl.seg_targets import instance_mask_targets

#: Image dtype a deferred-HSV batch must still carry when its colour transform runs:
#: the worker path jitters float32 pixels, so any narrower dtype rounds them first.
_DEFERRED_IMAGE_DTYPE = torch.float32
#: Gain dtype a deferred-HSV batch must still carry: ``1 + gain`` is formed in double,
#: as the worker path's Python addition does, before it takes the image dtype.
_DEFERRED_GAIN_DTYPE = torch.float64

#: Input-pixel stride of the prototype grid the mask targets are rasterised onto.
#: :class:`~lucid_yolo.models.heads.proto.ProtoNet` upsamples the fused P3 feature
#: (stride 8) by two (A15), so the grid is a fixed quarter of the input canvas. The
#: collate needs the grid without the model, and
#: :meth:`~lucid_yolo.ptl.module.DetectionLitModule._segment_terms` asserts the
#: transported masks match the prototypes it actually got, so a topology that broke
#: this fails loudly rather than supervising at the wrong resolution.
_PROTO_STRIDE = 4


@dataclass
class PackedTargets:
    """Batch of per-image :class:`~lucid_yolo.data.targets.Targets` flattened for IPC.

    A DataLoader worker ships every distinct tensor as its own shared-memory
    segment, so the natural ``list[Targets]`` batch — a per-image list whose every
    image also carries a *list* of per-instance polygon rings — explodes into
    hundreds of tiny segments and exhausts the consumer's mmap budget
    (``vm.max_map_count``) at realistic worker counts. This container collapses the
    whole ragged batch into a **fixed set of dense tensors**, plus optional mask
    stacks and deferred HSV gains: the modality
    rows are concatenated along their instance/ring/point axes and paired with the
    per-image (and per-ring) counts needed to split them back apart. The tensor
    count is constant regardless of how many instances the batch holds, so the
    transport costs a handful of segments instead of hundreds.

    This is a *transport* form only: :func:`unpack_targets` reconstructs the exact
    ``list[Targets]`` consumers expect (see :meth:`DetectionDataModule.on_after_batch_transfer`).
    The dataclass is deliberately **not frozen** — PyTorch Lightning moves batches
    across devices with ``apply_to_collection``, which rejects frozen dataclasses.

    Attributes:
        boxes_cat: ``(sum_N, 4)`` float32 ``xyxy`` boxes of every image, concatenated.
        labels_cat: ``(sum_N,)`` int64 class ids aligned with ``boxes_cat``.
        difficult_cat: ``(sum_N,)`` bool R18 difficult flags aligned with
            ``boxes_cat`` (WP-088). Transported rather than defaulted on arrival
            because A48 makes the oriented evaluation depend on the flag: a
            difficult ground truth that reaches the metric as an ordinary one turns
            every detection of it into a false positive.
        boxes_per_image: ``(B,)`` int64 instance count per image; splits
            ``boxes_cat``/``labels_cat``/``difficult_cat`` back into ``B`` images.
        rboxes_cat: ``(sum_M, 5)`` float32 long-edge rotated boxes, concatenated.
        rboxes_per_image: ``(B,)`` int64 rotated-box count per image; splits
            ``rboxes_cat``.
        keypoints_cat: ``(sum_N, K, 2)`` float32 point coordinates, concatenated
            (WP-132). Transported rather than left behind because the alternative is
            the worst kind of dead flag: ``keypoint_targets=True`` would parse the
            points in the reader and drop them at this boundary, and the run would
            train on nothing while every shape downstream still looked right.
        keypoint_vis_cat: ``(sum_N, K)`` int64 COCO visibilities aligned with
            ``keypoints_cat``. Shipped rather than defaulted for the reason A48 gives
            for ``difficult_cat``: A66 makes ``v == 0`` mean "no annotation exists",
            so a visibility that arrives defaulted supervises the model towards
            coordinates nobody ever placed.
        keypoints_per_image: ``(B,)`` int64 point-set count per image; splits
            ``keypoints_cat``/``keypoint_vis_cat``. A separate count from
            ``boxes_per_image`` even though WP-120 pairs the two axes 1:1 when
            keypoints are present at all, because a detection batch has ``N`` boxes
            and *no* point sets — one count cannot say both.
        polygon_points_cat: ``(sum_P, 2)`` float32 polygon points of every ring of
            every image, concatenated.
        points_per_ring: ``(R,)`` int64 point count per ring; splits
            ``polygon_points_cat`` into ``R`` rings.
        rings_per_image: ``(B,)`` int64 ring count per image (``0`` when an image
            carries no polygons, else its instance count); groups the ``R`` rings
            back into per-image lists.
        masks_cat: ``(sum_N, Hp, Wp)`` bool instance masks of every image,
            concatenated, or ``None`` on a detection loader. Present only when the
            collate rasterises (``mask_targets=True``), which moves that CPU work
            out of the training process and into the workers; the split is
            ``boxes_per_image``, since a segmentation annotation carries one ring
            per box. Shipped as ``bool`` rather than the ``float32``
            :func:`~lucid_yolo.ptl.seg_targets.instance_mask_targets` returns: the
            values are ``{0, 1}``, so the cast back is exact and the transport is a
            quarter of the size.
        hsv_gains: Optional ``(B, 3)`` float64 colour gains in sample order. When
            present, the accompanying images remain float32 until the consumer
            applies these gains and the usual uint8 quantization. ``None`` retains
            the default transport and is always used by validation.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> t = Targets(boxes=torch.tensor([[0.0, 0.0, 4.0, 4.0]]), labels=torch.tensor([3]))
        >>> packed = pack_targets([t, Targets.empty()])
        >>> packed.boxes_per_image.tolist()
        [1, 0]

        ```
    """

    boxes_cat: Tensor
    labels_cat: Tensor
    difficult_cat: Tensor
    boxes_per_image: Tensor
    rboxes_cat: Tensor
    rboxes_per_image: Tensor
    keypoints_cat: Tensor
    keypoint_vis_cat: Tensor
    keypoints_per_image: Tensor
    polygon_points_cat: Tensor
    points_per_ring: Tensor
    rings_per_image: Tensor
    masks_cat: Tensor | None = None
    hsv_gains: Tensor | None = None


def _concat_keypoints(targets: list[Targets]) -> tuple[Tensor, Tensor]:
    """Concatenate a batch's point sets and visibilities along the instance axis.

    Split out of :func:`pack_targets` for one reason the other modalities do not
    have: an instance-free image carries the canonical ``(0, 0, 2)`` empty, whose
    point axis is ``0`` rather than the batch's ``K``, and ``torch.cat`` compares
    every axis but the one it joins. The empties are therefore reshaped to the
    batch's own ``K`` first, as :meth:`~lucid_yolo.data.targets.Targets.concat`
    reads it (``Targets._shared_keypoint_count``) — free at zero elements, and it
    keeps a mixed batch of annotated and empty images packing at all. Two images
    that genuinely disagree on ``K`` still raise, from that same read and naming
    both counts, which is right: no single head predicts both.

    Args:
        targets: The per-image target list being packed.

    Returns:
        The ``(keypoints_cat, keypoint_vis_cat)`` pair of shapes ``(sum_N, K, 2)``
        float32 and ``(sum_N, K)`` int64, with ``K = 0`` for a batch that annotates
        no points at all.

    Raises:
        ValueError: If two images carrying point sets disagree on ``K``.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> posed = Targets(
        ...     boxes=torch.zeros((1, 4)),
        ...     labels=torch.tensor([0]),
        ...     keypoints=torch.zeros((1, 3, 2)),
        ...     keypoint_vis=torch.ones((1, 3), dtype=torch.int64),
        ... )
        >>> coords, visibility = _concat_keypoints([posed, Targets.empty()])
        >>> coords.shape, visibility.shape
        (torch.Size([1, 3, 2]), torch.Size([1, 3]))

        ```
    """
    num_points = Targets._shared_keypoint_count(targets)
    coords = [
        target.keypoints if target.keypoints.shape[0] else target.keypoints.new_zeros((0, num_points, _POINT_DIM))
        for target in targets
    ]
    visibility = [
        target.keypoint_vis if target.keypoints.shape[0] else target.keypoint_vis.new_zeros((0, num_points))
        for target in targets
    ]
    return torch.cat(coords, dim=0), torch.cat(visibility, dim=0)


def pack_targets(targets: list[Targets]) -> PackedTargets:
    """Flatten a ragged ``list[Targets]`` into a :class:`PackedTargets` transport.

    Every modality is concatenated along its own axis and paired with the counts
    that split it back per image (and per ring). The inverse is
    :func:`unpack_targets`; ``unpack_targets(pack_targets(x))`` reproduces ``x``
    tensor-for-tensor (same dtypes, shapes and values).

    Args:
        targets: The non-empty length-``B`` per-image target list to pack. All
            tensors must share one device.

    Returns:
        A :class:`PackedTargets` holding the batch as twelve dense tensors on the
        inputs' device. Its two optional fields, ``masks_cat`` and ``hsv_gains``, stay
        ``None``: the collate sets them.

    Raises:
        ValueError: If two images carrying point sets disagree on ``K``.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> a = Targets(boxes=torch.zeros((2, 4)), labels=torch.tensor([1, 2]))
        >>> packed = pack_targets([a, Targets.empty()])
        >>> packed.boxes_cat.shape, packed.boxes_per_image.tolist()
        (torch.Size([2, 4]), [2, 0])

        ```
    """
    boxes_cat = torch.cat([target.boxes for target in targets], dim=0)
    labels_cat = torch.cat([target.labels for target in targets], dim=0)
    rboxes_cat = torch.cat([target.rboxes for target in targets], dim=0)
    rings = [ring for target in targets for ring in target.polygons]
    polygon_points_cat = torch.cat(rings, dim=0) if rings else boxes_cat.new_zeros((0, _POINT_DIM))
    keypoints_cat, keypoint_vis_cat = _concat_keypoints(targets)

    def counts(values: list[int]) -> Tensor:
        return torch.tensor(values, dtype=torch.int64, device=boxes_cat.device)

    return PackedTargets(
        boxes_cat=boxes_cat,
        labels_cat=labels_cat,
        difficult_cat=torch.cat([target.difficult for target in targets], dim=0),
        boxes_per_image=counts([int(target.boxes.shape[0]) for target in targets]),
        rboxes_cat=rboxes_cat,
        rboxes_per_image=counts([int(target.rboxes.shape[0]) for target in targets]),
        keypoints_cat=keypoints_cat,
        keypoint_vis_cat=keypoint_vis_cat,
        keypoints_per_image=counts([int(target.keypoints.shape[0]) for target in targets]),
        polygon_points_cat=polygon_points_cat,
        points_per_ring=counts([int(ring.shape[0]) for ring in rings]),
        rings_per_image=counts([len(target.polygons) for target in targets]),
    )


def unpack_targets(packed: PackedTargets) -> list[Targets]:
    """Reconstruct the ``list[Targets]`` a :class:`PackedTargets` transports.

    The exact inverse of :func:`pack_targets`: each modality is split back along
    its axis by the packed counts and the per-ring polygons are regrouped into the
    per-image lists. The reconstructed tensors live on the packed tensors' device
    (so calling this in :meth:`DetectionDataModule.on_after_batch_transfer` yields
    on-device targets), and equal the originals tensor-for-tensor.

    Args:
        packed: The transport container to expand.

    Returns:
        The length-``B`` per-image target list, identical to the one
        :func:`pack_targets` consumed.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> a = Targets(boxes=torch.zeros((2, 4)), labels=torch.tensor([1, 2]))
        >>> restored = unpack_targets(pack_targets([a, Targets.empty()]))
        >>> [len(t.labels) for t in restored]
        [2, 0]

        ```
    """
    boxes_per_image = [int(count) for count in packed.boxes_per_image.tolist()]
    rboxes_per_image = [int(count) for count in packed.rboxes_per_image.tolist()]
    keypoints_per_image = [int(count) for count in packed.keypoints_per_image.tolist()]
    rings_per_image = [int(count) for count in packed.rings_per_image.tolist()]
    points_per_ring = [int(count) for count in packed.points_per_ring.tolist()]

    boxes = torch.split(packed.boxes_cat, boxes_per_image)
    labels = torch.split(packed.labels_cat, boxes_per_image)
    difficult = torch.split(packed.difficult_cat, boxes_per_image)
    rboxes = torch.split(packed.rboxes_cat, rboxes_per_image)
    keypoints = torch.split(packed.keypoints_cat, keypoints_per_image)
    keypoint_vis = torch.split(packed.keypoint_vis_cat, keypoints_per_image)
    rings = list(torch.split(packed.polygon_points_cat, points_per_ring)) if points_per_ring else []

    targets: list[Targets] = []
    ring_cursor = 0
    for index, ring_count in enumerate(rings_per_image):
        image_rings = rings[ring_cursor : ring_cursor + ring_count]
        ring_cursor += ring_count
        # An image with no point sets is restored to the canonical (0, 0, 2) empty
        # rather than to the (0, K, 2) slice the split hands back, so the roundtrip
        # this function promises is tensor-for-tensor on a mixed batch too.
        empty_points = keypoints_per_image[index] == 0
        targets.append(
            Targets(
                boxes=boxes[index],
                labels=labels[index],
                polygons=list(image_rings),
                rboxes=rboxes[index],
                difficult=difficult[index],
                keypoints=packed.keypoints_cat.new_zeros((0, 0, _POINT_DIM)) if empty_points else keypoints[index],
                keypoint_vis=packed.keypoint_vis_cat.new_zeros((0, 0)) if empty_points else keypoint_vis[index],
            )
        )
    return targets


def unpack_masks(packed: PackedTargets) -> list[Tensor] | None:
    """Split a packed mask stack back into one ``(N_i, Hp, Wp)`` tensor per image.

    The mask counterpart of :func:`unpack_targets`, kept separate because the masks
    are an optional modality: a detection loader ships none and this returns
    ``None``, which is the signal the consumer falls back to rasterising itself.
    ``torch.split`` returns views, so the per-image tensors cost no copy and stay on
    whatever device the transport was moved to.

    Args:
        packed: The transport container, whose ``masks_cat`` may be ``None``.

    Returns:
        One float mask stack per image in image order, or ``None`` when the batch
        carries no masks. The bool transport is restored to float32 ``{0.0, 1.0}``,
        the dtype :func:`~lucid_yolo.ptl.seg_targets.instance_mask_targets` produces.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> unpack_masks(pack_targets([Targets.empty()])) is None
        True

        ```
    """
    if packed.masks_cat is None:
        return None
    counts = [int(count) for count in packed.boxes_per_image.tolist()]
    return [chunk.to(torch.float32) for chunk in torch.split(packed.masks_cat, counts)]


@dataclass(frozen=True)
class _DeferredHSVSample:
    """Keep a sample's stated colour gains beside its flipped image and targets.

    Only the worker-side pipeline and collate consume this record; the packed
    batch transports one gain tensor rather than one Python object per sample.

    Attributes:
        image: Floating-point image after the geometric transforms and flip.
        targets: Geometry transformed with that image.
        params: The HSV draw made before the flip draw, as in the host pipeline.
    """

    image: Tensor
    targets: Targets
    params: HSVParams


_TrainingSample = tuple[Tensor, Targets] | _DeferredHSVSample


def collate_detection(
    batch: Sequence[tuple[Tensor, Targets]], *, mask_targets: bool = False
) -> tuple[Tensor, PackedTargets]:
    """Collate ``(image, Targets)`` samples into ``(images, PackedTargets)``.

    All images are letterboxed to a common ``img_size`` upstream, so they stack
    into one dense tensor. The ragged per-image
    :class:`~lucid_yolo.data.targets.Targets` are flattened into a
    :class:`PackedTargets` (:func:`pack_targets`) so the whole batch crosses the
    DataLoader worker boundary as a handful of shared-memory segments rather than
    hundreds of tiny ones. The datamodule restores the ``list[Targets]`` consumer
    contract on the destination device in
    :meth:`DetectionDataModule.on_after_batch_transfer`; a direct caller converts
    with :func:`unpack_targets`.

    Under ``mask_targets`` the segmentation mask targets are rasterised **here**,
    which is what puts that work in the loader workers: the DataLoader hands a whole
    batch to one worker and runs its ``collate_fn`` in that same process, so the
    batches of a run are rasterised concurrently across workers and ahead of the step
    that needs them. Rasterising in the training process instead cost a measured
    1005 ms per step at batch 32 on COCO, serial with the GPU, which is most of why
    the accelerator idled. The grid is the input canvas at :data:`_PROTO_STRIDE`
    (A15) rather than a size read off the model, so the collate needs no model;
    :meth:`~lucid_yolo.ptl.module.DetectionLitModule._segment_terms` checks the
    transported grid against the prototypes it actually got.

    Args:
        batch: The per-sample ``(image, Targets)`` pairs from the dataset.
        mask_targets: Rasterise each image's instance masks onto the prototype grid
            and ship them in ``packed.masks_cat``. Requires one polygon ring per box
            (:func:`~lucid_yolo.ptl.seg_targets.instance_mask_targets` raises
            otherwise), so it is set only by a segmentation loader.

    Returns:
        A ``(images, packed)`` pair: ``images`` is the batch quantized to a
        ``(B, C, H, W)`` **uint8** tensor (float ``[0, 1]`` pixels rounded to
        ``1/255`` steps — :data:`~lucid_yolo.data._quantization._PIXEL_QUANT_MAX` — for a 4x-smaller IPC hop) and
        ``packed`` is the batch's targets as a :class:`PackedTargets`. The transfer
        hook (or :func:`unpack_batch`) restores float32 ``[0, 1]`` images on the
        destination device.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> batch = [(torch.zeros(3, 4, 4), Targets.empty()) for _ in range(2)]
        >>> images, packed = collate_detection(batch)
        >>> images.dtype, images.shape, packed.boxes_per_image.tolist()
        (torch.uint8, torch.Size([2, 3, 4, 4]), [0, 0])

        ```
    """
    return _collate_training(batch, mask_targets=mask_targets)


def _collate_training(batch: Sequence[_TrainingSample], *, mask_targets: bool = False) -> tuple[Tensor, PackedTargets]:
    """Collate the training pipeline's samples, ordinary or colour-deferred, into the transport.

    The body behind :func:`collate_detection`, widened to the pipeline's private
    sample union: the training loader routes through it because, under
    ``device_hsv``, :class:`~lucid_yolo.ptl._pipeline._TrainPipeline` can yield :class:`_DeferredHSVSample`
    records, which the public signature does not name. Ordinary ``(image, Targets)``
    pairs take the public path unchanged. A batch of deferred records keeps its
    float32 images unquantized and attaches their stated gains as
    ``packed.hsv_gains``, one float64 row per image; colour conversion and the uint8
    quantization then run where the batch is restored (:func:`unpack_batch` or the
    transfer hook). A batch mixing both kinds is refused, so a gain row is never
    paired with an already-jittered image.

    Args:
        batch: Ordinary ``(image, Targets)`` pairs or deferred colour samples, all of
            one kind.
        mask_targets: As in :func:`collate_detection`.

    Returns:
        The ``(images, packed)`` transport pair: uint8 images for ordinary samples,
        float32 images with ``packed.hsv_gains`` set for deferred ones.

    Raises:
        ValueError: If the batch mixes deferred and ordinary samples.

    Examples:
        ```pycon
        >>> sample = _DeferredHSVSample(torch.zeros(3, 4, 4), Targets.empty(), HSVParams(0.0, 0.0, 0.0))
        >>> images, packed = _collate_training([sample])
        >>> images.dtype, packed.hsv_gains.shape
        (torch.float32, torch.Size([1, 3]))

        ```
    """
    deferred = bool(batch) and isinstance(batch[0], _DeferredHSVSample)
    if any(isinstance(sample, _DeferredHSVSample) != deferred for sample in batch):
        raise ValueError("cannot collate deferred and already-jittered HSV samples together")
    images = torch.stack(
        [sample.image if isinstance(sample, _DeferredHSVSample) else sample[0] for sample in batch], dim=0
    )
    image_size = (int(images.shape[-2]), int(images.shape[-1]))
    if not deferred:
        images = _quantize_images(images)
    targets = [sample.targets if isinstance(sample, _DeferredHSVSample) else sample[1] for sample in batch]
    packed = pack_targets(targets)
    if deferred:
        packed.hsv_gains = torch.tensor(
            [sample.params.as_tuple() for sample in batch if isinstance(sample, _DeferredHSVSample)],
            dtype=torch.float64,
        )
    if mask_targets:
        grid = (image_size[0] // _PROTO_STRIDE, image_size[1] // _PROTO_STRIDE)
        rasterised = [instance_mask_targets(target, image_size, grid).to(torch.bool) for target in targets]
        packed.masks_cat = torch.cat(rasterised) if rasterised else torch.zeros((0, *grid), dtype=torch.bool)
    return images, packed


def _restore_images(
    images: Tensor, packed: PackedTargets, dtype: torch.dtype, *, channels_last: bool = False
) -> Tensor:
    """Finish a deferred colour transform before restoring the consumer's image dtype.

    Only the transfer hook requests ``channels_last`` and loads the optional CUDA
    compiler there, and only while ``fused_hsv`` allows the fused restore; the hook
    makes the native result channels-last itself otherwise. Direct unpacking keeps the
    native expression and its layout.

    Raises:
        TypeError: If a deferred batch no longer carries float32 images and float64
            gains. Something cast it between the collate and this restore, and running
            the colour transform on the cast values would change the training pixels.
    """
    if packed.hsv_gains is not None:
        if images.dtype != _DEFERRED_IMAGE_DTYPE or packed.hsv_gains.dtype != _DEFERRED_GAIN_DTYPE:
            raise TypeError(
                f"deferred HSV batch arrived with {images.dtype} images and {packed.hsv_gains.dtype} gains, "
                f"expected {_DEFERRED_IMAGE_DTYPE} and {_DEFERRED_GAIN_DTYPE}: the batch was cast before its "
                "colour transform ran. Restore it before casting, or leave device_hsv off for this precision setup."
            )
        if channels_last and dtype == torch.float32 and images.is_cuda:
            from lucid_yolo.data._hsv_cuda import _restore_hsv_cuda  # noqa: PLC0415

            restored = _restore_hsv_cuda(images, packed.hsv_gains)
            if restored is not None:
                return restored
        images = _quantize_images(_jitter_hsv_batch(images, packed.hsv_gains))
    return _dequantize_images(images, dtype)


def _transport_survives(plugin: Precision) -> bool:
    """Report whether a precision plugin's input conversion keeps the deferred-HSV dtypes.

    Lightning passes every batch through ``precision_plugin.convert_input`` before the
    transfer hooks, so a plugin that casts floating tensors rounds the float32 images
    or the float64 gains before :func:`_restore_images` sees them. The plugin's own
    conversion is probed with one tensor of each dtype rather than matched by name, so
    a plugin this module does not know about is judged by what it does.

    Args:
        plugin: The attached trainer's precision plugin.

    Returns:
        ``True`` when both probe tensors come back at their original dtypes.

    Examples:
        ```pycon
        >>> from pytorch_lightning.plugins.precision import DoublePrecision
        >>> _transport_survives(Precision()), _transport_survives(DoublePrecision())
        (True, False)

        ```
    """
    probe = (torch.zeros(1, dtype=_DEFERRED_IMAGE_DTYPE), torch.zeros(1, dtype=_DEFERRED_GAIN_DTYPE))
    images, gains = cast("tuple[Tensor, Tensor]", plugin.convert_input(probe))
    return images.dtype == _DEFERRED_IMAGE_DTYPE and gains.dtype == _DEFERRED_GAIN_DTYPE


def unpack_batch(
    batch: tuple[Tensor, PackedTargets], dtype: torch.dtype = torch.float32
) -> tuple[Tensor, list[Targets]]:
    """Restore the consumer batch from a collated transport, finishing any deferred colour first.

    Reverses the collate end to end. A :func:`collate_detection` batch carries uint8
    images, which are dequantized to ``dtype`` in ``[0, 1]`` (``code / 255``). A
    training batch collated with deferred colour (``device_hsv``) instead carries
    float32 images and ``packed.hsv_gains``: the batched HSV jitter and the same
    uint8 quantization run first, on the images' device, and the dequantization
    follows, so both forms come back on one ``1/255`` code grid. The
    :class:`PackedTargets` is expanded back to the ragged ``list[Targets]``
    (:func:`unpack_targets`). This is
    the single public restore a direct (non-Lightning) consumer uses when it drives
    the loader itself; :meth:`DetectionDataModule.on_after_batch_transfer` runs the
    equivalent on-device restore inside a Lightning run, matching the attached
    module's precision. Restoring only the targets — leaving the images in their
    transport dtype — stays available via :func:`unpack_targets`.

    Args:
        batch: The ``(images, PackedTargets)`` transport pair: uint8 images from
            :func:`collate_detection`, or float32 images with ``packed.hsv_gains`` from
            the training loader under ``device_hsv``. ``packed`` must be on the images'
            device, since the gains are applied to the images where they stand and are
            never moved.
        dtype: Floating-point dtype for the restored images (default
            ``torch.float32``); pass ``torch.float16``/``torch.bfloat16`` to match a
            half-precision consumer.

    Returns:
        The ``(images, list[Targets])`` consumer pair: ``images`` is ``(B, C, H, W)``
        ``dtype`` in ``[0, 1]`` and the targets are the ragged length-``B`` per-image list.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> batch = [(torch.zeros(3, 4, 4), Targets.empty()) for _ in range(2)]
        >>> images, targets = unpack_batch(collate_detection(batch))
        >>> images.dtype, len(targets)
        (torch.float32, 2)

        ```
    """
    images, packed = batch
    return _restore_images(images, packed, dtype), unpack_targets(packed)
