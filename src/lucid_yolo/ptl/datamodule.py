# SPDX-License-Identifier: Apache-2.0
"""Detection :class:`~pytorch_lightning.LightningDataModule` over either dataset layout (WP-014).

:class:`DetectionDataModule` wires the WP-014 :class:`~lucid_yolo.data.coco.CocoDetectionDataset`
into the Phase 1 augmentation pipeline and exposes train/val
:class:`~torch.utils.data.DataLoader` s. It builds both splits from the COCO 2017
layout of blueprint sec. 14.3 (``train2017/`` + ``annotations/instances_train2017.json``
and the matching ``val2017``) and, per the dataset contract (AGENTS.md sec. 3),
contains **no download logic**: a missing or mis-shaped root is validated by
``lucid-data check`` before any ``[DATA]`` run, never fetched here.

Layout dispatch (WP-099b):
    ``data_root`` may also be a YOLO root — a ``data.yaml`` beside per-split ``images``/
    ``labels`` trees — in which case both splits are built by
    :class:`~lucid_yolo.data.yolo.YoloDetectionDataset` instead, and everything downstream is
    unchanged because both readers emit the same
    :class:`~lucid_yolo.data.targets.Targets`. WP-099 delivered that reader and left the
    choice unmade, which meant a YOLO root was reachable from a library call and not from
    ``lucid-yolo fit``. Which reader a root gets is decided in three steps:

    1. an explicit ``layout`` wins outright — it is the only way to name a layout the
       filesystem cannot be asked about (see :attr:`DetectionDataModule.layout`);
    2. otherwise, naming any of the four ``*_images_dir``/``*_ann_file`` overrides *is* a
       statement of the COCO layout, since an annotation file is a concept only that
       convention has;
    3. otherwise the root is probed by :func:`~lucid_yolo.data.layout.detect_layout`, which
       raises rather than guess when the root satisfies neither convention or both.

    The probe runs at :meth:`DetectionDataModule.setup`, not at construction: constructing
    this class is a pure configuration act — ``LightningCLI(run=False)`` instantiates every
    shipped config, whose ``data_root`` is a placeholder path on the machine that parses it —
    so a filesystem verdict there would fail runs that have not yet been pointed at data.
    Steps 1 and 2 need no filesystem and are settled in ``__init__``, which is why a layout
    the YOLO path cannot serve is rejected there.

Multi-image augmentation composition:
    Mosaic, mixup and copy-paste each consume *several* images, so they cannot be
    ordinary single-image :class:`~lucid_yolo.data.transforms.GeometricTransform` s
    chained in a :class:`~lucid_yolo.data.transforms.Compose`. They are composed
    instead by :class:`_TrainPipeline`, a wrapper :class:`~torch.utils.data.Dataset`
    that, for each requested index, draws the extra source indices it needs from a
    single seeded :class:`torch.Generator` and assembles the sample:

        1. geometric base — with probability :attr:`_TrainPipeline.mosaic_p` a
           four-image :class:`~lucid_yolo.data.mosaic.MosaicAssembly` (else the single image),
           then :class:`~lucid_yolo.data.affine.RandomAffine` with ``letterbox=``, which
           composes the random affine and the letterbox down to ``img_size`` into one resample;
        2. with probability ``mixup`` a second full geometric sample is blended in
           via :class:`~lucid_yolo.data.mixup.Mixup`;
        3. with probability ``copy_paste`` polygon instances from a third geometric
           sample are pasted in via :class:`~lucid_yolo.data.mixup.CopyPaste`;
        4. photometric :class:`~lucid_yolo.data.augment.HSVJitter` and
           :class:`~lucid_yolo.data.augment.HorizontalFlip` finish the sample.

    The per-sample mixup/copy-paste probabilities come from
    :func:`~lucid_yolo.data.coco.build_scale_policy` (size-aware, [R1] Table S3);
    mosaic runs at ``p=1.0`` (blueprint sec. 5.9). Because every draw comes from
    one seeded generator, a fixed ``seed`` and a deterministic access order
    (``num_workers=0``) give byte-identical epochs. With workers the pipeline is
    seeded *per worker and per epoch* by :func:`_init_worker` (WP-079) — a fresh
    stream each time, still fully determined by ``seed``.

    **Reproducibility is a function of** ``(seed, num_workers)`` **jointly, not of**
    ``seed`` **alone.** Both regimes above are deterministic, and they are
    deterministically *different*: at ``num_workers=0`` every draw comes from the
    parent generator in index order, while at ``num_workers=N`` torch derives each
    worker's seed as ``base_seed + worker_id`` from the loader's own generator at
    every iterator creation, so the augmentation parameters a given sample receives
    depend on which worker happened to own its index. A run at 32 workers therefore
    does **not** reproduce the same seed at 0 workers, and does not reproduce it at
    16 either. Reproducing a published run means matching its worker count as well
    as its seed; the tier recipes in ``docs/TRAINING.md`` state both for that reason.

Batch contract:
    Two forms, split by the DataLoader worker boundary:

    * **Transport form** (what :func:`collate_detection` returns): the equal-sized
      (letterboxed) images stacked and quantized into a single
      ``(B, C, img_size, img_size)`` **uint8** tensor — each post-augmentation float
      pixel in ``[0, 1]`` rounded to a ``1/255`` step (:data:`_PIXEL_QUANT_MAX`) —
      paired with a :class:`PackedTargets`, the batch's ragged per-image
      :class:`~lucid_yolo.data.targets.Targets` flattened into a fixed set of
      dense tensors. A worker ships each distinct tensor as its own shared-memory
      segment: the natural ``list[Targets]`` (with its per-image *list* of polygon
      rings) would explode into hundreds of tiny segments and exhaust the consumer's
      mmap budget, so packing bounds the batch to a handful; the uint8 images are a
      quarter of the float32 bytes, keeping the in-flight worker queue inside the
      container's shared-memory budget. Both the augmented train path and the
      letterbox-only val path go through this collate, so both transport uint8 (val
      images already start from a JPEG-decoded uint8 source, so their letterbox
      interpolation loses no more than the same bounded half-step).
      With ``device_hsv=True`` on CUDA, training instead transports float32 pixels
      and one ``(B, 3)`` gain tensor; the consumer applies HSV before the same
      quantization. This spends four times the image transport bytes to remove
      colour conversion from the workers. The worker cap budgets float32 bytes, but
      only for a count the module chooses (``num_workers`` unset) in one process: a
      named ``num_workers`` is neither capped nor warned, and each DDP rank reads the
      free shm on its own.
    * **Consumer form** (what every downstream step sees): the pair
      ``(images, list[Targets])`` with float32 ``[0, 1]`` images unchanged.
      :meth:`DetectionDataModule.on_after_batch_transfer` restores the images
      (``uint8 -> float32 / 255``) and unpacks the transport targets on the
      destination device, so the module and everything downstream keep consuming a
      ragged length-``B`` target list exactly as before. A direct (non-Lightning)
      consumer converts the whole pair with :func:`unpack_batch`.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast
from weakref import ref

import torch
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm.auto import tqdm

from lucid_yolo.assign.grid import require_grid_side
from lucid_yolo.data._quantization import _PIXEL_QUANT_MAX
from lucid_yolo.data.affine import _DEFAULT_PAD_VALUE, RandomAffine
from lucid_yolo.data.augment import HorizontalFlip, HSVJitter, HSVParams, _jitter_hsv_batch
from lucid_yolo.data.coco import CocoDetectionDataset, build_scale_policy
from lucid_yolo.data.layout import DatasetLayout, detect_layout, resolve_split
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.mixup import CopyPaste, Mixup
from lucid_yolo.data.mosaic import MosaicAssembly
from lucid_yolo.data.targets import Targets
from lucid_yolo.data.yolo import YoloDetectionDataset
from lucid_yolo.ptl._epoch_loader import _EpochPrefetchLoader
from lucid_yolo.ptl.seg_targets import instance_mask_targets

__all__ = [
    "DetectionDataModule",
    "PackedTargets",
    "collate_detection",
    "pack_targets",
    "unpack_batch",
    "unpack_masks",
    "unpack_targets",
]

#: Column count of a polygon point ``(x, y)`` — the width of a packed ring row.
_POINT_DIM = 2

#: Either layout's reader. Both yield ``(image, Targets)`` samples and size the same way, which
#: is what lets the augmentation pipeline, the collate and every consumer stay layout-blind.
_DetectionSource = CocoDetectionDataset | YoloDetectionDataset

#: Three RGB channels, each retained as an IEEE float32 rather than an 8-bit code.
_RGB_FLOAT_PIXEL_BYTES = 3 * torch.finfo(torch.float32).bits // 8

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

#: Default mosaic probability: every training sample (blueprint sec. 5.9: ``mosaic
#: p=1.0``). The ``close_mosaic`` late-epoch disable flips the per-pipeline
#: :attr:`_TrainPipeline.mosaic_p` seam to ``0`` via
#: :meth:`DetectionDataModule.set_mosaic_prob` (driven by ``CloseMosaicCallback``, WP-036).
_MOSAIC_PROB = 1.0
#: Standard YOLO-lineage affine translate fraction ([R1] Table S3; scale is size-aware).
_AFFINE_TRANSLATE = 0.1
#: Standard YOLO-lineage horizontal-flip probability ([R1] Table S3: ``fliplr=0.5``).
_FLIP_PROB = 0.5

#: Default per-worker prefetched batches (the DataLoader default). The whole
#: queue — ``num_workers x prefetch_factor`` batches — lives in POSIX shared
#: memory, so a deeper default multiplies /dev/shm pressure by batch size and
#: worker count and ENOMEMs containerized runs (Colab) at large batches; raise
#: it per run via ``--data.prefetch_factor`` only when shm headroom allows.
_PREFETCH_FACTOR = 2


#: Workers that saturate a letterbox-only validation loader at :data:`_VAL_REFERENCE_SIDE`
#: (WP-073) — a throughput target, not a memory ceiling, and it is only ever *applied* as a
#: bound because the alternative it replaces (inheriting an auto-chosen train count) is
#: larger. Val samples are letterbox-only, so few workers saturate them, while inheriting a
#: large train count doubles the resident worker-process population every epoch boundary
#: (train workers are persistent) — the observed Colab host-OOM trigger on long runs. The
#: memory side of the question is :func:`_shm_capped_workers`, which is a ceiling.
_VAL_MAX_WORKERS = 4
#: Letterbox side the target above was measured at. Decode cost scales with the pixel count,
#: so the target scales with it too — *up*, not down: the constant encodes "how many workers
#: saturate a letterbox-only loader", and a 1024 px sample costs more decode per worker, so
#: it takes more of them, which is where "a fraction of the train pipeline's work" stopped
#: being true at 640 (WP-102). This is why :func:`_val_worker_target` grows with ``img_size``
#: while a ceiling would shrink.
_VAL_REFERENCE_SIDE = 640


#: Fraction of currently-free ``/dev/shm`` the worker queue may claim. Worker
#: batches travel to the main process as shared-memory segments, so the queue's
#: worst case — ``num_workers x prefetch_factor`` stacked image batches — must
#: fit the shm tmpfs with headroom for the ragged target tensors and everything
#: else sharing it, or workers die mid-epoch with tiny-mmap ENOMEM.
_SHM_BUDGET_FRACTION = 0.5


def _shm_capped_workers(workers: int, batch_size: int, img_size: int, prefetch: int) -> int:
    """Cap a worker count so the prefetch queue fits the free ``/dev/shm`` budget.

    On Linux every batch a worker hands over is backed by shared memory, so the
    in-flight queue claims about ``workers x prefetch x batch_bytes`` of the shm
    tmpfs; a many-core host with a large batch fills it before the first step
    (observed: 64-vCPU Colab, batch 64 -> 30+ GB queued -> ``unable to mmap ...
    Cannot allocate memory``). The cap bounds the queue to
    :data:`_SHM_BUDGET_FRACTION` of the *currently free* shm space. Hosts
    without ``/dev/shm`` (macOS, Windows) are returned unchanged.

    ``batch_bytes`` counts four bytes per image element, so the float32 batches a
    deferred-HSV run ships fit the same budget. The cap applies only where it is
    called: the train count the module chooses at construction (``num_workers``
    unset) and the val count :meth:`DetectionDataModule._val_loader_workers`
    inherits. A named train count bypasses it. Each process reads ``statvfs`` on
    its own, so N ranks sharing one host can jointly claim N times the fraction —
    a known limit of a per-process budget.

    Args:
        workers: The worker count before the cap.
        batch_size: Loader batch size (stacked-image bytes scale linearly).
        img_size: Square letterbox side of every emitted image.
        prefetch: Per-worker prefetched batch count.

    Returns:
        ``workers`` bounded below by 1 and above by the shm budget.
    """
    shm = Path("/dev/shm")
    if not shm.exists():
        return workers
    try:
        stat = os.statvfs(shm)
    except OSError:
        return workers
    free_bytes = stat.f_bavail * stat.f_frsize
    batch_bytes = 4 * 3 * batch_size * img_size * img_size
    budget = int(free_bytes * _SHM_BUDGET_FRACTION)
    return max(1, min(workers, budget // max(1, prefetch * batch_bytes)))


def _val_worker_target(img_size: int) -> int:
    """Return the worker count that saturates a letterbox-only val loader at ``img_size``.

    This is a *target*, not a ceiling: it grows with the pixel count, because decode cost
    per sample does, so a larger letterbox side takes more workers to keep the loader fed.
    A ceiling would move the other way. The caller
    (:meth:`DetectionDataModule._resolve_val_workers`) applies it as an upper bound only
    because the number it is bounding — an auto-chosen train worker count — is the larger
    of the two; the memory-driven ceiling is :func:`_shm_capped_workers`.

    Args:
        img_size: Square letterbox side of every emitted sample.

    Returns:
        :data:`_VAL_MAX_WORKERS` scaled by the pixel count relative to
        :data:`_VAL_REFERENCE_SIDE`, at least 1.

    Examples:
        ```pycon
        >>> _val_worker_target(640), _val_worker_target(1024)
        (4, 10)

        ```
    """
    scale = (img_size / _VAL_REFERENCE_SIDE) ** 2
    return max(1, round(_VAL_MAX_WORKERS * scale))


def _init_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: cap torch threads and re-seed the worker's RNG.

    **Thread cap.** Without it every worker process inherits torch's default
    intra-op thread pool (one thread per core), so ``num_workers`` workers
    oversubscribe the CPU by that factor and the tensor ops inside the
    augmentation pipeline (affine warps, HSV jitter, blends) thrash instead of
    running in parallel. One thread per worker makes worker throughput scale
    with ``num_workers``.

    **Re-seed (WP-079).** :class:`_TrainPipeline` seeds its single generator once,
    at construction, in the *parent* process. Every worker therefore inherits a
    copy of the **same** generator state, and the parent's copy never advances
    (the parent never calls ``__getitem__`` when workers are used). Two failures
    follow, and both were live in the Det-smoke tier runs:

    1. every worker replays one identical augmentation-parameter stream, cutting
       the per-epoch parameter diversity to ``1 / num_workers``;
    2. with ``persistent_workers=False`` (the WP-076 default) workers are rebuilt
       from that same parent state every epoch, so the whole run replays a single
       epoch's worth of augmentation parameters.

    Seeding from :attr:`torch.utils.data.WorkerInfo.seed` fixes both: torch
    derives it as ``base_seed + worker_id``, and draws ``base_seed`` from the
    loader's own generator at every iterator creation — unique per worker, fresh
    per epoch, and still fully determined by the datamodule's ``seed``.

    Datasets with no ``_generator`` attribute (the val pipeline is letterbox-only,
    hence deterministic) are left untouched.

    Args:
        worker_id: The worker index (unused; the seed comes from the worker info).
    """
    del worker_id
    torch.set_num_threads(1)
    info = get_worker_info()
    if info is None:
        return
    generator = getattr(info.dataset, "_generator", None)
    if isinstance(generator, torch.Generator):
        generator.manual_seed(info.seed)


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
    batch's own ``K`` first — free at zero elements, and it keeps a mixed batch of
    annotated and empty images packing at all. Two images that genuinely disagree
    on ``K`` still raise from ``torch.cat``, which is right: no single head predicts
    both.

    Args:
        targets: The per-image target list being packed.

    Returns:
        The ``(keypoints_cat, keypoint_vis_cat)`` pair of shapes ``(sum_N, K, 2)``
        float32 and ``(sum_N, K)`` int64, with ``K = 0`` for a batch that annotates
        no points at all.

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
    num_points = max((int(target.keypoints.shape[1]) for target in targets if target.keypoints.shape[0]), default=0)
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
        A :class:`PackedTargets` holding the batch as eleven dense tensors on the
        inputs' device.

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


def _host_sample(sample: _TrainingSample) -> tuple[Tensor, Targets]:
    """Complete a deferred sample for host-side diagnostics without quantizing it.

    The debug grid and augmentation golden read unquantized dataset samples rather
    than collated batches. Resolving their colour here retains that convention when
    the pipeline defers HSV; ordinary samples pass through unchanged.

    Args:
        sample: One ordinary or deferred training sample.

    Returns:
        The fully augmented image and its original targets.

    Examples:
        ```pycon
        >>> sample = (torch.zeros(3, 2, 2), Targets.empty())
        >>> _host_sample(sample) is sample
        True

        ```
    """
    if isinstance(sample, _DeferredHSVSample):
        return HSVJitter().apply(sample.image, sample.targets, sample.params)
    return sample


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
        ``1/255`` steps — :data:`_PIXEL_QUANT_MAX` — for a 4x-smaller IPC hop) and
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
    ``device_hsv``, :class:`_TrainPipeline` can yield :class:`_DeferredHSVSample`
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


def _quantize_images(images: Tensor) -> Tensor:
    """Round augmented float pixels to the existing uint8 transport codes."""
    return images.mul(_PIXEL_QUANT_MAX).round().clamp_(0, _PIXEL_QUANT_MAX).to(torch.uint8)


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


def _dequantize_images(images: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """Restore uint8 transport images to floating-point ``[0, 1]``; pass floats through.

    The inverse of the :func:`collate_detection` quantization: uint8 codes cast to
    ``dtype`` and divide by :data:`_PIXEL_QUANT_MAX`. The ``0``…``255`` codes and the
    divisor are exactly representable in float16/bfloat16, so a half-precision
    ``dtype`` keeps the same ``1/510`` quantization bound. An already-float tensor (a
    direct, non-quantized feed) is returned untouched — its own dtype preserved — so
    the restore is safe to run over either form.

    Args:
        images: A ``(B, C, H, W)`` uint8 transport batch or an already-float batch.
        dtype: Floating-point dtype for the restored images (default
            ``torch.float32``); used only on the uint8 dequantization path.

    Returns:
        The images in ``[0, 1]`` at ``dtype`` (dequantized when the input was uint8,
        otherwise the input unchanged).
    """
    if images.dtype == torch.uint8:
        return images.to(dtype).div(_PIXEL_QUANT_MAX)
    return images


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


class _PreparedTrainingSource(Dataset[tuple[Tensor, Targets]]):
    """Retain the immutable, fill-centred float image before random geometry.

    Converting a source and subtracting the affine fill are independent of every
    random draw. Mosaic can assemble those pixels on a zero-filled canvas; the
    affine then receives exactly the tensor its ordinary fill subtraction creates.
    The fill is restored after the same resample, before any colour or blending.
    This costs four bytes per source pixel channel instead of the byte cache's one.
    ``base`` must be a raw reader without ``transforms``: its pixels are retained as
    the source canvas the training geometry starts from.
    """

    def __init__(self, base: _DetectionSource) -> None:
        self.base = base
        self.keypoint_flip_pairs = base.keypoint_flip_pairs
        estimate = ""
        if isinstance(base, CocoDetectionDataset):
            expected = _RGB_FLOAT_PIXEL_BYTES * sum(record.height * record.width for record in base._images)
            estimate = f"about {expected / 1e9:.1f} GB ({expected:,} bytes) from the annotation sizes; "
        warnings.warn(
            f"prepared image cache: {estimate}retaining float32 source images costs four times "
            "an 8-bit decoded image cache in host RAM per training process",
            UserWarning,
            stacklevel=3,
        )
        self._items: list[tuple[Tensor, Targets]] = []
        for index in tqdm(range(len(base)), desc="Preparing images", unit="img"):
            image, targets = base[index]
            # Subtract out of place: the reader's tensor is never written, so nothing
            # here depends on whether a reader hands out fresh or shared storage. The
            # cost is one transient image per record, released on the next iteration.
            self._items.append((image - _DEFAULT_PAD_VALUE, targets))
        retained = sum(image.nbytes for image, _ in self._items)
        warnings.warn(
            f"prepared image cache: holding {retained:,} bytes of float32 pixels "
            f"for {len(self._items)} training records",
            UserWarning,
            stacklevel=3,
        )

    def __len__(self) -> int:
        """Return the number of retained training records."""
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[Tensor, Targets]:
        """Return one retained image and its original targets; the image is shared storage, never write it."""
        return self._items[index]


class _TrainPipeline(Dataset[_TrainingSample]):
    """Train-time wrapper composing mosaic/mixup/copy-paste over a base dataset.

    For each index the pipeline draws whatever extra source indices it needs from
    one seeded :class:`torch.Generator`, so a fixed seed and a deterministic access
    order reproduce an epoch byte-for-byte. Every produced sample is letterboxed to
    ``img_size`` so the collate can stack the batch.

    Attributes:
        mosaic_p: Probability that a sample is built from a four-image mosaic rather
            than its single base image (default :data:`_MOSAIC_PROB`). A mutable seam:
            the ``close_mosaic`` schedule sets it to ``0`` for the final epochs via
            :meth:`DetectionDataModule.set_mosaic_prob`. Consulted per draw, so a
            change takes effect on the next sample assembled.

    Args:
        base: The raw (untransformed) reader — either layout's.
        img_size: Target square side for the letterboxed output.
        policy: The size-aware strengths from
            :func:`~lucid_yolo.data.coco.build_scale_policy` (``scale``, ``mixup``,
            ``copy_paste``).
        seed: Seed for the single :class:`torch.Generator` driving every draw.
        oriented: Whether the base dataset yields rotated targets. It suppresses
            copy-paste, which cannot act on a modality that carries no polygons
            (WP-056); every other stage is unchanged, and in particular the draw
            sequence is not — the copy-paste probability is *consulted* as always,
            it simply never fires.
        polygon_free: The same suppression for the same reason, reached the other way:
            a base whose *format* carries no rings rather than whose modality does not.
            The YOLO reading is one (WP-099: ``polygons`` is always empty there), and
            unlike the oriented case copy-paste would not raise on it — it would find no
            paste candidates and quietly do nothing, having already decoded a whole extra
            source sample to discover that. Suppressing it says so and stops paying for it.
        defer_hsv: Sample the same colour gains but apply them after transfer. The
            pointwise colour transform commutes with the following horizontal flip.
    """

    def __init__(
        self,
        base: _DetectionSource | _PreparedTrainingSource,
        img_size: int,
        policy: dict[str, float],
        seed: int,
        oriented: bool = False,
        polygon_free: bool = False,
        defer_hsv: bool = False,
    ) -> None:
        self._base = base
        self._defer_hsv = defer_hsv
        self._img_size = int(img_size)
        self.mosaic_p = float(_MOSAIC_PROB)
        self._mixup_prob = policy["mixup"]
        # Copy-paste is structurally unavailable on the oriented path, not merely
        # unhelpful: it transfers *rasterised polygon masks*, and WP-056 keeps rotated
        # targets without polygons on purpose, so a rotated instance has nothing to
        # paste. `CopyPaste` says exactly that by raising, which means the composition —
        # not the transform — is what has to know. Mosaic and mixup are unaffected: both
        # carry `rboxes` through (WP-058). `polygon_free` is the same absence arrived at from
        # the format instead of the modality (WP-099b); there the transform stays silent
        # rather than raising, so the composition is the *only* thing that can know.
        self._copy_paste_prob = 0.0 if (oriented or polygon_free) else policy["copy_paste"]
        self._generator = torch.Generator().manual_seed(seed)
        # A prepared source has already subtracted the affine's fill from every pixel, so
        # the mosaic leaves uncovered canvas at zero and the affine restores the fill after
        # its resample; otherwise both fill with the same grey as usual.
        centred = isinstance(base, _PreparedTrainingSource)
        self._mosaic = MosaicAssembly(
            self._img_size, generator=self._generator, pad_value=0.0 if centred else _DEFAULT_PAD_VALUE
        )
        self._affine = RandomAffine(
            scale=policy["scale"],
            translate=_AFFINE_TRANSLATE,
            generator=self._generator,
            letterbox=self._img_size,
            centered_input=centred,
        )
        self._mixup = Mixup(p=1.0, generator=self._generator)
        self._copy_paste = CopyPaste(p=1.0, generator=self._generator)
        self._hsv = HSVJitter(generator=self._generator)
        # The pairing comes from `base`'s own annotation schema, never from a constant
        # here — A64's requirement, and the one this line previously broke by omission.
        # `HorizontalFlip` is K-generic by design and cannot know which index mirrors
        # which; supplying nothing does not mean "no swap needed", it means every sided
        # point in a mirrored sample supervises the model toward its opposite number's
        # location. That trains, converges to a left/right-confused head, and reports
        # nothing, which is why the value is read from the dataset rather than defaulted.
        # `None` still reaches the flip intact for a schema with no left/right symmetry,
        # where mirroring coordinates alone is the correct answer.
        self._flip = HorizontalFlip(
            p=_FLIP_PROB,
            generator=self._generator,
            keypoint_flip_pairs=base.keypoint_flip_pairs,
        )

    def __len__(self) -> int:
        """Return the number of base images."""
        return len(self._base)

    def __getitem__(self, index: int) -> _TrainingSample:
        """Assemble a sample, keeping deferred colour gains beside it when requested."""
        image, targets = self._geometric(index)
        image, targets = self._maybe_mixup(image, targets)
        image, targets = self._maybe_copy_paste(image, targets)
        if self._defer_hsv:
            params = self._hsv.sample()
            self._hsv.last_gains = params.as_tuple()
            image, targets = self._flip(image, targets)
            return _DeferredHSVSample(image, targets, params)
        image, targets = self._hsv(image, targets)
        return self._flip(image, targets)

    def _geometric(self, index: int) -> tuple[Tensor, Targets]:
        """Build the geometric base: optional mosaic, then the affine onto the letterbox canvas.

        The random affine and the letterbox down to ``img_size`` compose into a single
        image resample (:class:`~lucid_yolo.data.affine.RandomAffine` with
        ``letterbox=``), so the geometric path costs one bilinear pass instead of two.
        """
        if self._draw() < self.mosaic_p:
            items = [self._base[i] for i in self._mosaic_indices(index)]
            image, targets = self._mosaic(items)
        else:
            image, targets = self._base[index]
        return self._affine(image, targets)

    def _maybe_mixup(self, image: Tensor, targets: Targets) -> tuple[Tensor, Targets]:
        """With probability ``mixup`` blend a second full geometric sample in."""
        if self._draw() >= self._mixup_prob:
            return image, targets
        other = self._geometric(self._random_index())
        return self._mixup([(image, targets), other])

    def _maybe_copy_paste(self, image: Tensor, targets: Targets) -> tuple[Tensor, Targets]:
        """With probability ``copy_paste`` paste polygon instances from another sample."""
        if self._draw() >= self._copy_paste_prob:
            return image, targets
        source = self._geometric(self._random_index())
        return self._copy_paste([(image, targets), source])

    def _mosaic_indices(self, index: int) -> list[int]:
        """Return ``index`` plus three further sampled indices for a mosaic."""
        return [index, self._random_index(), self._random_index(), self._random_index()]

    def _random_index(self) -> int:
        """Draw a uniform base-dataset index from the seeded generator."""
        return int(torch.randint(0, len(self._base), (1,), generator=self._generator).item())

    def _draw(self) -> float:
        """Draw one uniform in ``[0, 1)`` from the seeded generator."""
        return float(torch.rand((), generator=self._generator).item())


def _check_shared_label_space(train_map: dict[int, int], val_map: dict[int, int]) -> None:
    """Refuse two COCO splits whose ``categories`` lists disagree (WP-172).

    Each split's contiguous labels are derived from its own file alone —
    ``_build_category_maps`` sorts that file's category ids and numbers them from zero — so
    the two mappings are independent derivations that nothing downstream ever compares. One
    category present in only one file renumbers every label above it, and the failure is
    silent by construction: the model trains against one numbering, is scored against
    another, and reports a perfectly plausible number for the wrong question. This is the
    only place both mappings are in hand, so it is the only place the disagreement is
    visible.

    Comparing the id *sets* is sufficient rather than merely necessary: the mapping is a
    function of the sorted id set, so equal sets imply equal mappings. That is why the error
    names the symmetric difference rather than diffing two dicts.

    Args:
        train_map: The train split's ``category_id_to_label``.
        val_map: The val split's ``category_id_to_label``.

    Raises:
        ValueError: If the splits declare different category ids, naming the ids each side
            holds alone.

    Examples:
        ```pycon
        >>> _check_shared_label_space({1: 0, 2: 1}, {1: 0, 2: 1})
        >>> try:
        ...     _check_shared_label_space({1: 0}, {1: 0, 7: 1})
        ... except ValueError as error:
        ...     print("only in val: [7]" in str(error))
        True

        ```
    """
    only_train = sorted(set(train_map) - set(val_map))
    only_val = sorted(set(val_map) - set(train_map))
    if only_train or only_val:
        raise ValueError(
            "train and val declare different COCO label spaces, so the same label names a "
            f"different class in each split; only in train: {only_train}, only in val: {only_val}. "
            "Both annotation files must declare the same categories."
        )


class DetectionDataModule(LightningDataModule):
    """Detection datamodule assembling the Phase 1 augmentation pipeline (WP-014).

    Train and val splits are built from the COCO 2017 layout under ``data_root``
    (``train2017/`` + ``annotations/instances_train2017.json`` and the ``val2017``
    equivalents); each split path may be overridden explicitly (used by the offline
    fixture tests, which have a single Roboflow-style split). A ``data_root`` written in
    the YOLO convention instead is read by
    :class:`~lucid_yolo.data.yolo.YoloDetectionDataset` — see the module docstring for the
    three-step dispatch (WP-099b). Training samples pass
    through :class:`_TrainPipeline` (mosaic/mixup/copy-paste/affine/hsv/flip +
    letterbox); validation is letterbox-only. No split is ever downloaded.

    Args:
        data_root: Dataset root: a COCO 2017 root (holding ``train2017``, ``val2017``,
            ``annotations``), any other layout :data:`~lucid_yolo.data.layout.CANDIDATES`
            names, or a YOLO root (a ``data.yaml`` beside ``images``/``labels`` trees).
        batch_size: Samples per batch for both loaders.
        num_workers: DataLoader worker processes. ``None`` (default) resolves
            to ``min(batch_size, cpu count)``, further capped so the in-flight
            worker queue fits the free ``/dev/shm`` budget on Linux
            (:func:`_shm_capped_workers`) — a many-core host with a large batch
            otherwise fills the shm tmpfs before the first step. Determinism of
            the seeded pipeline is guaranteed only at an explicit ``0`` (a
            single in-process generator).
        variant: Model size letter selecting the augmentation strength policy
            (``"n"``…``"x"``); validated at construction.
        img_size: Square letterbox side for every emitted sample. Defaults to 640, and
            must be a positive multiple of every head stride — validated at construction,
            for the reason ``variant`` is.
        train_images_dir: Override for the train images directory. Defaults to
            whichever of ``data_root/"train2017"`` and ``data_root/"train"``
            :func:`~lucid_yolo.data.layout.resolve_split` finds, so a tiled
            oriented root needs no override at all.
        train_ann_file: Override for the train annotation JSON. Resolved
            alongside the images directory, never independently of it.
        val_images_dir: Override for the val images directory. Resolved as the
            train one is.
        val_ann_file: Override for the val annotation JSON. Resolved alongside
            the val images directory.
        seed: Seed for the training pipeline's generator and loader shuffling.
        pin_memory: Whether loader batches land in page-locked host memory for
            async host-to-device copies. ``None`` (default) resolves to ``True``
            exactly when CUDA is available — MPS and CPU runs get ``False``
            (pinning buys nothing there and MPS warns on it).
        prefetch_factor: Batches each worker keeps prefetched (defaults to
            :data:`_PREFETCH_FACTOR`); ignored at ``num_workers=0`` where the
            DataLoader forbids it.
        val_num_workers: Worker count for the validation loader only, capped by
            nothing — the flag an operator states to spend the whole machine on
            validation. ``None`` (default) resolves to an auto-chosen
            ``num_workers`` bounded by :func:`_val_worker_target` (WP-073, WP-102), and
            to a *named* ``num_workers`` bounded by the in-flight memory budget of
            :meth:`_val_loader_workers` (WP-103) — letterbox-only val samples need
            few workers, and both worker populations are resident at the epoch
            boundary where validation runs.
        persistent_workers: Keep loader workers alive across epochs. Defaults
            to ``False`` (WP-076): respawning costs seconds per epoch while a
            persistent pool accumulates per-worker memory for the whole run —
            the observed slow-creep OOM on long containerized runs.
        cache_train_images: Decode the training split once during :meth:`setup`
            and retain read-only RGB tensors — ``uint8`` for 8-bit sources, ``uint16``
            (twice the bytes) for a 16-bit PNG. Defaults to ``False``;
            enabling it trades the uncompressed split size in host RAM for removing
            image decode from every augmented draw in every epoch. :meth:`setup`
            only pays this cost for a ``"fit"``/``None`` stage, so a
            ``"validate"``/``"test"``/``"predict"`` call never decodes the train
            split it will not read. The decode draws a progress bar and reports the
            cache's size as a ``UserWarning`` — before decoding on a COCO root, from
            the annotation file's image sizes, and after it on a YOLO root, which
            states none — so an undersized host sees the figure rather than a silent
            stall ending in an out-of-memory kill.
        cache_prepared_train_images: Retain float32, fill-centred source images
            instead of repeatedly converting bytes and centring each random mosaic.
            Off by default. Takes precedence over ``cache_train_images``, with a
            ``UserWarning`` at construction when both are set, and costs
            four bytes per pixel channel, per training record and training process;
            duplicate COCO records retain separate images, plus target metadata.
            YOLO labels are snapshotted with their images. The random geometry,
            fill-restored pixels, targets and RNG draws stay unchanged. Fit setup
            builds it once, drawing a progress bar; other stages release it. Host RAM
            must hold the split.
        device_hsv: Apply training HSV and quantization on CUDA, in batches, under
            float32 or mixed precision. Off by default; other backends and precisions
            retain worker-side colour augmentation. The workers
            keep their RNG draws and transport float32 images plus the stated gains,
            using four times the default image IPC and transfer bytes. The worker cap
            budgets float32 images only for a count chosen here (``num_workers``
            unset) in one process; a named ``num_workers`` is neither capped nor
            warned, and each DDP rank budgets the free shm on its own. Validation
            stays uint8.
        fused_hsv: Let the transfer hook restore a deferred-HSV batch with the fused
            Triton kernel where the host supports it (:mod:`lucid_yolo.data._hsv_cuda`).
            On by default, so a supported host takes it whenever Triton and a host C
            compiler are present; ``False`` keeps ``device_hsv`` on the native CUDA
            restore, to step around a toolchain or bisect a difference between the two.
            Pixels are identical either way. Has no effect without ``device_hsv``.
        rotated_targets: Read every annotation's four-point ring as a rotated box
            (:class:`~lucid_yolo.data.coco.CocoDetectionDataset`'s ``oriented`` mode,
            WP-088), so both loaders emit ``Targets`` whose ``rboxes`` pair 1:1 with
            their ``boxes``. Off by default because it requires a quadrilateral ring
            per instance; the CLI turns it on for ``model.task="obb"``. Costs no extra
            transport — the packed batch has carried a rotated-box axis since WP-055.
        mask_targets: Rasterise the segmentation mask targets in the loader workers
            (:func:`collate_detection`) instead of leaving that CPU work in the
            training process. Off by default because it requires a polygon ring per
            box; the CLI turns it on for ``model.task="segment"``. Adds the mask
            stack to the batch transport — roughly 27 MB per batch of 32 at COCO
            instance density — so a container with a small ``/dev/shm`` may need
            fewer workers or a lower ``prefetch_factor``. **Unavailable on a YOLO root**:
            that format carries no per-instance rings at all, so the request is refused
            as soon as the layout is known rather than left to produce a segmentation run
            supervised by nothing.
        keypoint_targets: Parse every annotation's COCO ``keypoints`` field
            (:class:`~lucid_yolo.data.coco.CocoDetectionDataset`'s ``keypoints`` mode,
            WP-121), so both loaders emit ``Targets`` whose ``keypoints`` and
            ``keypoint_vis`` pair 1:1 with their ``boxes``. Off by default because it
            requires a keypoints-annotated file; the CLI turns it on for
            ``model.task="keypoints"``. Adds a ``(sum_N, K, 2)`` coordinate axis and a
            ``(sum_N, K)`` visibility axis to the batch transport — small beside the
            images. **Unavailable on a YOLO root**, which carries no keypoint fields.

            Both loaders carry points the whole way (WP-132). The **validation** path is
            letterbox-only and needed nothing but
            :class:`~lucid_yolo.data.letterbox.Letterbox`; the **training** path also runs
            the mosaic, the fused affine, mixup and the horizontal flip, and each of those
            rebuilds its ``Targets``, so each had to be taught the point channels
            explicitly. Unlike the letterbox they crop and filter, which is why they need
            A70's rule for a point warped off the canvas — it is carried through with its
            true coordinate rather than clamped or demoted, a case A66's ``v == 0`` ("no
            annotation exists") does not answer. The flip needs more than a rule: it needs
            this dataset's own left/right pairing, which is why the reader publishes
            ``keypoint_flip_pairs`` (A64) rather than the transform assuming one.
        layout: Which reader ``data_root`` calls for — ``"coco"``, ``"yolo"``, or ``None``
            (the default) to infer it. Two things are unsayable without it, which is the
            whole of its justification: a root holding *both* conventions, which
            :func:`~lucid_yolo.data.layout.detect_layout` refuses to break a tie on because
            the two readings are different label spaces; and a YOLO root whose ``data.yaml``
            points its splits outside the convention table, which the reader resolves and
            the probe cannot see. Stating ``"coco"`` or ``"yolo"`` skips the probe entirely,
            so the reader — not this dispatch — names any path it then fails to open.

        epoch_prefetch: Queue up to ``num_workers x prefetch_factor`` batches of the next
            training epoch during the last batch and validation, using the existing
            persistent workers. Opt-in: queued training inputs can remain resident
            during validation. Only full,
            single-CUDA epochs with the module-owned sampler use the early reset;
            limits, step bounds, overfit, reloads and distributed sampling stay native.
            CUDA is the only accelerator it was measured on (an L4); others are
            unmeasured, not known to fail.
            A request the attached trainer cannot honour warns once, on rank zero,
            naming the first unmet condition.

    Examples:
        ```pycon
        >>> DetectionDataModule  # doctest: +SKIP
        >>> # dm = DetectionDataModule(root, batch_size=2, num_workers=0, variant="n")
        >>> # dm.setup("fit"); images, targets = next(iter(dm.train_dataloader()))

        ```
    """

    def __init__(  # noqa: PLR0913 — flat hyperparameter surface is deliberate: WP-038 LightningCLI configures each knob directly from YAML (only 4 are required positionals)
        self,
        data_root: Path,
        batch_size: int,
        num_workers: int | None,
        variant: str,
        img_size: int = 640,
        *,
        train_images_dir: Path | None = None,
        train_ann_file: Path | None = None,
        val_images_dir: Path | None = None,
        val_ann_file: Path | None = None,
        seed: int = 0,
        pin_memory: bool | None = None,
        prefetch_factor: int = _PREFETCH_FACTOR,
        val_num_workers: int | None = None,
        persistent_workers: bool = False,
        cache_train_images: bool = False,
        mask_targets: bool = False,
        rotated_targets: bool = False,
        keypoint_targets: bool = False,
        layout: str | None = None,
        device_hsv: bool = False,
        fused_hsv: bool = True,
        epoch_prefetch: bool = False,
        cache_prepared_train_images: bool = False,
    ) -> None:
        super().__init__()
        self._mask_targets = bool(mask_targets)
        self._device_hsv = bool(device_hsv)
        self._fused_hsv = bool(fused_hsv)
        self._epoch_prefetch = bool(epoch_prefetch)
        self._warned_device_hsv = False
        self._warned_mosaic_prefetch = False
        self._warned_epoch_prefetch = False
        self._rotated_targets = bool(rotated_targets)
        self._keypoint_targets = bool(keypoint_targets)
        self._batch_size = int(batch_size)
        # Checked here because this is where the side enters a run: every image the two
        # loaders emit is letterboxed to it, and the head divides that canvas by its own
        # strides. A side the strides do not divide floors to a feature map the neck
        # cannot concatenate, which surfaces as a shape error inside the first training
        # step -- an hour into a queued run, and about a tensor rather than about the YAML
        # key that set it (WP-171).
        require_grid_side("img_size", img_size)
        self._img_size = int(img_size)
        self._prefetch_factor = int(prefetch_factor)
        workers_were_chosen_here = num_workers is None
        if num_workers is None:
            num_workers = _shm_capped_workers(
                min(self._batch_size, os.cpu_count() or 1),
                self._batch_size,
                self._img_size,
                self._prefetch_factor,
            )
        self._num_workers = int(num_workers)
        self._val_workers_were_inherited = val_num_workers is None and not workers_were_chosen_here
        self._val_num_workers = self._resolve_val_workers(val_num_workers, workers_were_chosen_here)
        self._persistent_workers = bool(persistent_workers)
        self._cache_train_images = bool(cache_train_images)
        self._cache_prepared_train_images = bool(cache_prepared_train_images)
        if self._cache_train_images and self._cache_prepared_train_images:
            warnings.warn(
                "cache_train_images and cache_prepared_train_images are both set; the prepared "
                "float32 cache takes precedence and the decoded 8-bit cache is not built. "
                "Set only one of them.",
                UserWarning,
                stacklevel=2,
            )
        self._prepared_train: _PreparedTrainingSource | None = None
        self._seed = int(seed)
        self._pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        self._policy = build_scale_policy(variant)
        self._data_root = data_root
        # The COCO reading of the root is resolved here whether or not it is the one that ends
        # up used: it is a naming rule with a fallback, never an existence check, so it costs a
        # few stat calls and cannot fail. Which reading is *taken* is `_layout`'s decision.
        default_train_images, default_train_ann = resolve_split(data_root, "train")
        default_val_images, default_val_ann = resolve_split(data_root, "val")
        self._train_images_dir = train_images_dir or default_train_images
        self._train_ann_file = train_ann_file or default_train_ann
        self._val_images_dir = val_images_dir or default_val_images
        self._val_ann_file = val_ann_file or default_val_ann
        self._named_paths = any(
            path is not None for path in (train_images_dir, train_ann_file, val_images_dir, val_ann_file)
        )
        self._layout = self._layout_without_probing(layout)
        self._train: _TrainPipeline | None = None
        self._val: _DetectionSource | None = None
        # The train reader a fit-stage `setup` decoded under `cache_train_images`, kept so a
        # repeat fit-stage call reuses it instead of decoding a second copy (see `setup`).
        self._cached_train_base: _DetectionSource | None = None

    def _layout_without_probing(self, requested: str | None) -> DatasetLayout | None:
        """Settle the layout as far as the arguments alone can, before any filesystem is read.

        Steps 1 and 2 of the module docstring's dispatch. Both are decidable from the call
        itself, so a run misconfigured in either way fails at the point the configuration is
        written rather than at the epoch that needed the data. ``None`` means step 3 — the probe
        — still has to run, which :meth:`setup` does once the root is supposed to exist.

        Args:
            requested: The ``layout`` argument as given.

        Returns:
            The stated layout, :attr:`~lucid_yolo.data.layout.DatasetLayout.COCO` when the
            COCO-shaped path overrides state it, or ``None`` when the root must be probed.

        Raises:
            ValueError: If ``layout`` names neither convention, or if it names the YOLO one
                alongside a path override only the COCO one has a meaning for.
        """
        if requested is None:
            return DatasetLayout.COCO if self._named_paths else None
        try:
            layout = DatasetLayout(requested)
        except ValueError:
            known = ", ".join(repr(member.value) for member in DatasetLayout)
            raise ValueError(f"unknown layout {requested!r}; expected one of {known}") from None
        if layout is DatasetLayout.YOLO and self._named_paths:
            raise ValueError(
                f"layout={DatasetLayout.YOLO.value!r} cannot be combined with the train_images_dir/"
                "train_ann_file/val_images_dir/val_ann_file overrides: those name a COCO "
                "annotation file and the images directory beside it, while a YOLO root resolves "
                "its splits from its own data.yaml. Drop the overrides, or point data_root at "
                "the root holding that data.yaml"
            )
        self._check_layout_support(layout)
        return layout

    def _check_layout_support(self, layout: DatasetLayout) -> None:
        """Reject a request the resolved layout's reader cannot serve.

        Both such requests today are on a YOLO root: segmentation supervision, and keypoint
        supervision. Each is refused rather than downgraded because the failure it replaces is
        silent in the worst way — the rings the mask targets rasterise, and the point fields the
        pose targets read, are simply never there (WP-099), so the run would train a head on
        empty supervision and report a plausible detection number.

        Args:
            layout: The layout that will be read.

        Raises:
            ValueError: If ``mask_targets`` or ``keypoint_targets`` was asked for on a YOLO root.
        """
        if layout is not DatasetLayout.YOLO:
            return
        if self._mask_targets:
            raise ValueError(
                "mask_targets=True is unavailable on a YOLO root: the format carries no "
                "per-instance polygon rings, so there is nothing to rasterise masks from. "
                "Train segmentation from a COCO-format root, or drop mask_targets"
            )
        if self._keypoint_targets:
            raise ValueError(
                "keypoint_targets=True is unavailable on a YOLO root: the format's label rows "
                "carry a class and a box and no point fields at all, so there is nothing to "
                "read keypoints from. Train pose from a COCO-format root, or drop keypoint_targets"
            )

    @property
    def layout(self) -> DatasetLayout:
        """Return the layout this datamodule reads, probing the root once if need be.

        Returns:
            The resolved :class:`~lucid_yolo.data.layout.DatasetLayout`, cached after the
            first call so the probe's verdict cannot change mid-run.

        Raises:
            FileNotFoundError: If the root satisfies neither convention; the message names
                both and every path tried (:func:`~lucid_yolo.data.layout.detect_layout`).
            ValueError: If it satisfies both, or asks for something that layout cannot serve.

        Examples:
            ```pycon
            >>> DetectionDataModule.layout  # doctest: +SKIP
            >>> # DetectionDataModule(yolo_root, batch_size=2, num_workers=0, variant="n").layout
            >>> # <DatasetLayout.YOLO: 'yolo'>

            ```
        """
        if self._layout is None:
            layout = detect_layout(self._data_root)
            self._check_layout_support(layout)
            self._layout = layout
        return self._layout

    def _resolve_val_workers(self, requested: int | None, workers_were_chosen_here: bool) -> int:
        """Decide the validation loader's *intended* worker count.

        The letterbox cap applies to what this class chose, never to what a caller named.
        An operator who writes ``--data.num_workers 16`` has stated how much of the
        machine this run may use, and silently running validation on a quarter of it is
        the library second-guessing a decision that was not its to make — the surprise
        costs more than the OOM it was avoiding, because it shows up as an idle GPU with
        no message. When the count was auto-chosen the cap still applies, because then
        the library owns the consequences of its own guess.

        An *inherited* count — named for training, never stated for validation — is
        neither, and :meth:`_val_loader_workers` bounds it by the memory budget when the
        loader is built (WP-103).

        Args:
            requested: Explicit ``val_num_workers``, or ``None`` to derive one.
            workers_were_chosen_here: Whether the train worker count was auto-chosen.

        Returns:
            The validation loader's worker count before any build-time memory budget.
        """
        if requested is not None:
            return int(requested)
        if not workers_were_chosen_here:
            return self._num_workers
        return min(self._num_workers, _val_worker_target(self._img_size))

    def _val_loader_workers(self) -> int:
        """Return the worker count this validation loader may actually start (WP-103).

        A count *inherited* from a named ``num_workers`` is bounded here by
        :func:`_shm_capped_workers`, loudly. WP-102 was right that an operator's number
        should reach the val loader and wrong about what to do with the guard: it deleted
        the cap rather than the silence, and a named 32 at 1024 px queues about 51 GB for
        validation alone (``32 x 2 x 4 x 3 x 64 x 1024**2``) on top of a train pool that
        is still resident — the host dies. Parallelism is what an operator states with
        that flag; gigabytes are what the flag *costs*, and the second is this class's
        arithmetic to do. Naming ``val_num_workers`` outright is still sovereign, capped
        by nothing, which is where an operator who wants the whole machine says so.

        The budget is read **when the loader is built**, not at construction: by then the
        training queue is resident, so :func:`os.statvfs` reports the space the second
        pool will actually find rather than the space that existed before the first one
        spawned. The overlap of the two pools is the failure mode, so it is the state the
        decision has to be made against.

        Returns:
            The validation loader's worker count, bounded by the memory budget when the
            count was inherited rather than stated.
        """
        workers = self._val_num_workers
        if not self._val_workers_were_inherited:
            return workers
        capped = _shm_capped_workers(workers, self._batch_size, self._img_size, self._prefetch_factor)
        if capped < workers:
            warnings.warn(
                f"validation would inherit num_workers={workers}, queueing about "
                f"{workers * self._prefetch_factor * 4 * 3 * self._batch_size * self._img_size**2 / 1e9:.0f} GB "
                f"of {self._img_size} px batches beside the training loader's own queue; "
                f"using {capped} workers instead. Set --data.val_num_workers to override.",
                stacklevel=2,
            )
        return capped

    def prepare_data(self) -> None:
        """No-op: datasets are provisioned out of band, never downloaded (AGENTS.md section 3)."""

    def setup(self, stage: str | None = None) -> None:
        """Build the train pipeline and the letterbox-only val dataset.

        The layout is resolved here (:attr:`layout`) rather than at construction, because this
        is the first moment the root is required to exist: a root that satisfies neither
        convention raises before a single image is decoded, naming both conventions and every
        path tried.

        Args:
            stage: The Lightning stage (``"fit"``/``"validate"``/…). Both splits are
                always built, so a repeated call yields the same splits and a fresh train
                pipeline — mosaic probability and draw generator back at their initial
                state. ``cache_train_images`` only decodes the train split when ``stage``
                is ``"fit"`` or ``None``: a ``"validate"``/``"test"``/``"predict"`` call
                never pays to cache a split it will not read, and releases a cache an
                earlier fit-stage call built. A repeat ``"fit"``/``None`` call with the
                cache on keeps the already-decoded train reader rather than decoding the
                split again beside it, so the result is idempotent and the decode cost is
                paid once, not once per call.

                A call that raises installs nothing it built: the val dataset and both
                caches keep their previous values, and no train pipeline is left behind,
                so :meth:`train_dataloader` raises rather than serving the pipeline an
                earlier stage built.

        Raises:
            FileNotFoundError: If the root satisfies neither dataset convention.
            ValueError: If it satisfies both, or the resolved layout cannot serve a
                requested modality.
        """
        # Dropped before anything can raise: the previous pipeline may belong to another
        # stage (a validate-stage one never defers HSV and reads no prepared source), so a
        # caller that catches a failed fit setup must not go on training with it.
        self._train = None
        yolo = self.layout is DatasetLayout.YOLO
        prepare_train = self._cache_prepared_train_images and stage in {None, "fit"}
        cache_train_images = self._cache_train_images and stage in {None, "fit"} and not prepare_train
        # A second decoded reader would hold two copies of the split until the first one is
        # released, and decode every image again for an identical result. The reader is
        # immutable and its paths are fixed at construction, so reusing it changes nothing a
        # caller can see: val and the augmentation pipeline around the reader are rebuilt as
        # before. A stage that reads no train images keeps nothing, which lets the cache go.
        kept_train = self._cached_train_base if cache_train_images or prepare_train else None
        base, val = (
            self._build_yolo_splits(cache_train_images, kept_train)
            if yolo
            else self._build_coco_splits(cache_train_images, kept_train)
        )
        prepared = self._prepared_train if prepare_train else None
        if prepare_train and prepared is None:
            prepared = _PreparedTrainingSource(base)
        train = _TrainPipeline(
            prepared if prepared is not None else base,
            self._img_size,
            self._policy,
            self._seed,
            oriented=self._rotated_targets,
            polygon_free=yolo,
            defer_hsv=self._defer_hsv(stage),
        )
        # Every build above has succeeded, so the four fields change together.
        self._val = val
        self._train = train
        self._cached_train_base = base if cache_train_images or prepare_train else None
        self._prepared_train = prepared

    def _defer_hsv(self, stage: str | None) -> bool:
        """Select CUDA colour augmentation when input precision preserves the transport.

        The worker still samples HSV before sampling the flip. A pointwise colour
        transform commutes with that pixel permutation, so applying the colour
        operations after transfer keeps both the geometry and the RNG sequence.
        When the worker count is auto-chosen, its shared-memory cap already budgets
        four bytes per image element. A named ``num_workers`` is not capped, and each
        process reads the free space on its own, so DDP ranks each claim that budget.

        Lightning passes every batch through the precision plugin's ``convert_input``
        before this hook, so the precision label alone does not decide: deferral also
        needs that conversion to keep float32 images and float64 gains
        (:func:`_transport_survives`). True-half and double plugins cast both, and the
        FSDP and DeepSpeed plugins cast at least the gains under every accepted
        precision, so all of them keep worker-side HSV rather than rounding images or
        stated gains first. The plugin is probed only when the trainer exposes one;
        the precision allowlist applies either way.

        Without an attached trainer the answer is provisional: Lightning attaches its
        trainer before calling ``setup`` again, which re-evaluates it. That case returns
        ``False`` silently and leaves the once-per-module warning unspent, so a later
        genuine fallback still warns rather than an early call claiming "off CUDA".
        """
        if not self._device_hsv or stage not in {None, "fit"}:
            return False
        trainer = self.trainer
        if trainer is None:
            return False
        if isinstance(trainer.accelerator, CUDAAccelerator):
            plugin = getattr(trainer, "precision_plugin", None)
            if trainer.precision not in {"32-true", "16-mixed", "bf16-mixed"}:
                reason = f"for precision={trainer.precision!r}"
            elif plugin is not None and not _transport_survives(plugin):
                reason = (
                    f"for precision={trainer.precision!r} under {type(plugin).__name__}, "
                    "which casts floating batch inputs before the transfer hook"
                )
            else:
                return True
        else:
            reason = "off CUDA"
        if not self._warned_device_hsv:
            warnings.warn(
                f"device_hsv=True is ignored {reason}; HSV stays in the data workers",
                UserWarning,
                stacklevel=3,
            )
            self._warned_device_hsv = True
        return False

    def _build_coco_splits(
        self, cache_train_images: bool, kept_train: _DetectionSource | None
    ) -> tuple[_DetectionSource, _DetectionSource]:
        """Build both splits from the resolved (or overridden) COCO paths.

        Each reader derives its own contiguous label space from its own file, so the two are
        compared here before either is served: two disagreeing ``categories`` lists is a
        wrong-dataset error that otherwise trains to completion and reports a metric
        (:func:`_check_shared_label_space`). The comparison runs before the train images
        are cached, so a mismatched pair fails in the seconds two JSON reads take rather
        than after the whole train split has been decoded.

        Args:
            cache_train_images: Decode and retain the train split's images, once both
                label spaces agree. The caller already narrows this to the stages that read
                the train split.
            kept_train: The train reader an earlier fit-stage call already decoded, reused
                as it stands, or ``None`` to read the annotation file into a fresh one.

        Returns:
            The raw train reader and the letterbox-only val reader.

        Raises:
            ValueError: If the two annotation files declare different category ids.
        """
        train = (
            kept_train
            if isinstance(kept_train, CocoDetectionDataset)
            else CocoDetectionDataset(
                self._train_images_dir,
                self._train_ann_file,
                oriented=self._rotated_targets,
                keypoints=self._keypoint_targets,
            )
        )
        val = CocoDetectionDataset(
            self._val_images_dir,
            self._val_ann_file,
            transforms=Letterbox(self._img_size),
            oriented=self._rotated_targets,
            keypoints=self._keypoint_targets,
        )
        _check_shared_label_space(train.category_id_to_label, val.category_id_to_label)
        if cache_train_images:
            train.cache_images(progress=True)
        return train, val

    def _build_yolo_splits(
        self, cache_train_images: bool, kept_train: _DetectionSource | None
    ) -> tuple[_DetectionSource, _DetectionSource]:
        """Build both splits from the root's own ``data.yaml`` and label trees.

        Each split is resolved by the reader
        (:meth:`~lucid_yolo.data.yolo.YoloDetectionDataset.from_root`), so the dataset's own
        split entries outrank the naming convention — the published spelling writes
        ``val: ../valid/images``, which no convention table would find. ``rotated_targets``
        selects the nine-field oriented reading on *both* splits, since the variant is a
        property of the dataset and is declared, never sniffed (WP-099).

        Args:
            cache_train_images: Decode and retain the train split's images now. The
                caller already narrows this to the stages that read the train split.
            kept_train: The train reader an earlier fit-stage call already decoded, reused
                as it stands, or ``None`` to list the split into a fresh one.

        Returns:
            The raw train reader and the letterbox-only val reader.
        """
        train = (
            kept_train
            if isinstance(kept_train, YoloDetectionDataset)
            else YoloDetectionDataset.from_root(self._data_root, "train", oriented=self._rotated_targets)
        )
        if cache_train_images:
            train.cache_images(progress=True)
        return (
            train,
            YoloDetectionDataset.from_root(
                self._data_root,
                "val",
                Letterbox(self._img_size),
                oriented=self._rotated_targets,
            ),
        )

    def on_after_batch_transfer(
        self, batch: tuple[Tensor, PackedTargets] | tuple[Tensor, list[Targets]], dataloader_idx: int
    ) -> tuple[Tensor, list[Targets], list[Tensor] | None]:
        """Restore the float images and ``list[Targets]`` consumer contract on-device.

        Lightning calls this after moving the batch to the accelerator, so both the
        uint8 images and the :class:`PackedTargets` tensor fields are already
        on-device: the images are dequantized to the attached module's dtype
        (:meth:`_consumer_dtype` — ``code / 255`` at that precision,
        :func:`_dequantize_images`) and the packed targets are unpacked, all on the
        same device the module trains on, with no extra host-to-device hop.
        A deferred training batch first applies its stated HSV gains and quantizes
        the result, then reaches this same dequantization. Matching
        the module dtype keeps a ``"16-true"``/``"bf16-true"`` run's half weights and
        inputs aligned; ``"16-mixed"`` leaves the module float32 and lets autocast
        cast. Already-float images and already-list targets (a non-packed direct
        feed) pass through untouched, so the hook is safe to run over either form.
        Both the train and val loaders share this single restore. On CUDA the
        restored images are made ``channels_last``, the layout the module's
        parameters take in :meth:`DetectionLitModule.setup`; masks and every other
        device keep the default layout.

        Args:
            batch: The transferred ``(uint8 images, PackedTargets)`` transport batch,
                or an already-restored ``(float images, list[Targets])`` batch.
            dataloader_idx: The loader index (unused; the conversion is uniform).

        Returns:
            The ``(images, list[Targets], masks)`` consumer batch — ``[0, 1]`` images
            at the module's dtype, the ragged per-image targets, and the per-image
            mask stacks a segmentation loader rasterised in its workers (``None`` on
            a detection loader, or on any already-restored feed, which is the signal
            the module rasterises the masks itself).

        Raises:
            TypeError: If a deferred-HSV batch arrives cast away from float32 images
                and float64 gains (:func:`_restore_images`).

        Examples:
            ```pycon
            >>> DetectionDataModule.on_after_batch_transfer  # doctest: +SKIP
            >>> # images, targets = dm.on_after_batch_transfer(next(iter(dm.val_dataloader())), 0)

            ```
        """
        del dataloader_idx
        images, targets = batch
        dtype = self._consumer_dtype()
        images = (
            _restore_images(images, targets, dtype, channels_last=images.is_cuda and self._fused_hsv)
            if isinstance(targets, PackedTargets)
            else _dequantize_images(images, dtype)
        )
        if images.is_cuda:
            images = images.contiguous(memory_format=torch.channels_last)
        if isinstance(targets, PackedTargets):
            return images, unpack_targets(targets), unpack_masks(targets)
        return images, targets, None

    def _consumer_dtype(self) -> torch.dtype:
        """Resolve the floating-point dtype the dequantized images should take.

        Returns the attached :class:`~pytorch_lightning.LightningModule`'s parameter
        dtype so the restored images match the trainer's precision: a
        ``"16-true"``/``"bf16-true"`` run holds half weights the inputs must match,
        while ``"16-mixed"`` leaves the module float32 (autocast casts as needed).
        Falls back to ``torch.float32`` when no trainer or module is attached — the
        path a direct :func:`unpack_batch` consumer never reaches.

        Returns:
            The attached module's dtype, or ``torch.float32`` when unattached.
        """
        trainer = self.trainer
        if trainer is None:
            return torch.float32
        module_dtype = trainer.lightning_module.dtype  # str | torch.dtype in the PL stubs
        return module_dtype if isinstance(module_dtype, torch.dtype) else torch.float32

    def _loader_kwargs(self) -> dict[str, object]:
        """Return the streaming DataLoader kwargs shared by both loaders.

        Accelerator-aware: ``pin_memory`` is on only where it helps (resolved in
        the constructor — CUDA yes, MPS/CPU no), ``prefetch_factor`` and the
        the per-worker init (:func:`_init_worker` — thread cap plus RNG re-seed)
        apply only with workers, so the ``num_workers=0`` path is untouched.

        Workers are **recycled every epoch** unless ``persistent_workers=True``
        was requested (WP-076): a respawn costs seconds per epoch, while a
        persistent pool accumulates per-worker memory — allocator high-water
        growth, arena fragmentation, residual copy-on-write pages — for the
        whole run, the observed slow-creep OOM on long containerized runs.
        """
        workers = self._num_workers > 0
        return {
            "batch_size": self._batch_size,
            "num_workers": self._num_workers,
            # A partial, not a closure or a lambda: the worker start method pickles
            # `collate_fn`, and `functools.partial` over a module-level function is
            # picklable where a local function is not.
            "collate_fn": partial(collate_detection, mask_targets=self._mask_targets),
            "persistent_workers": self._persistent_workers and workers,
            "pin_memory": self._pin_memory,
            "prefetch_factor": self._prefetch_factor if workers else None,
            "worker_init_fn": _init_worker if workers else None,
        }

    def train_dataloader(self) -> DataLoader[tuple[Tensor, PackedTargets]]:
        """Return the shuffled training loader with the packed batch transport.

        DataLoader's type parameter describes dataset items, while this collate
        returns a different form. The cast records the collated boundary rather
        than exposing the pipeline's private deferred-sample union to consumers.
        The loader collates through :func:`_collate_training`, the private twin of
        :func:`collate_detection` that also accepts that union, so the public collate
        keeps its ``(image, Targets)`` signature.
        """
        if self._train is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        shuffle_generator = torch.Generator().manual_seed(self._seed)
        kwargs = self._loader_kwargs() | {
            # A partial over a module-level function, as in `_loader_kwargs`, so the
            # worker start method can still pickle it.
            "collate_fn": partial(_collate_training, mask_targets=self._mask_targets),
        }
        trainer = self.trainer
        loader: DataLoader[object]
        if self._epoch_prefetch and trainer is not None:
            gate = self._unmet_prefetch_gate(trainer)
            if gate is None:
                return self._prefetch_loader(trainer, shuffle_generator, kwargs)
            if not self._warned_epoch_prefetch:
                rank_zero_warn(f"epoch_prefetch=True is ignored with {gate}; training epochs use the ordinary loader")
                self._warned_epoch_prefetch = True
        loader = DataLoader(
            self._train,
            shuffle=True,
            generator=shuffle_generator,
            **kwargs,  # type: ignore[arg-type]
        )
        return cast("DataLoader[tuple[Tensor, PackedTargets]]", loader)

    def _unmet_prefetch_gate(self, trainer: Trainer) -> str | None:
        """Name the first condition that keeps the ordinary training loader, or ``None``.

        The early reset needs a persistent pool to reuse and complete single-CUDA epochs
        whose count it can trust: a limit, a step bound, overfitting, a reload or a
        distributed sampler would each make the prepared epoch the wrong one. The
        conditions are checked in that order and the first one unmet is named, so the
        caller's warning points at the setting to change.

        The CUDA condition marks the edge of the measurements (an L4), not a known
        failure of other accelerators; widening it needs its own measurement.
        """
        overfit = getattr(trainer, "overfit_batches", 1)
        reload_every = getattr(trainer, "reload_dataloaders_every_n_epochs", 1)
        limit = trainer.limit_train_batches
        gates = (
            (self._persistent_workers, "persistent_workers=False"),
            (self._num_workers > 0, "num_workers=0"),
            (trainer.world_size == 1, f"world_size={trainer.world_size}"),
            (isinstance(trainer.strategy, SingleDeviceStrategy), f"strategy={type(trainer.strategy).__name__}"),
            # CUDA only because only CUDA was measured (an L4); other accelerators are unmeasured, not known to fail.
            (isinstance(trainer.accelerator, CUDAAccelerator), f"accelerator={type(trainer.accelerator).__name__}"),
            (isinstance(limit, float) and limit == 1.0, f"limit_train_batches={limit!r}"),
            (trainer.max_steps == -1, f"max_steps={trainer.max_steps}"),
            (not overfit, f"overfit_batches={overfit!r}"),
            (not reload_every, f"reload_dataloaders_every_n_epochs={reload_every!r}"),
            (trainer.max_epochs is not None and trainer.max_epochs > 0, f"max_epochs={trainer.max_epochs!r}"),
        )
        return next((reason for met, reason in gates if not met), None)

    def _prefetch_loader(
        self, trainer: Trainer, generator: torch.Generator, kwargs: dict[str, object]
    ) -> DataLoader[tuple[Tensor, PackedTargets]]:
        """Build the training loader that prepares each next epoch during the current one's end.

        The predicate holds the trainer weakly, so a kept loader never keeps the model
        alive, and stops preparing at the configured final epoch or once training was
        asked to stop.
        """
        max_epochs = trainer.max_epochs or 0
        weak_trainer = ref(trainer)

        def prepare_next() -> bool:
            owner = weak_trainer()
            return owner is not None and owner.current_epoch + 1 < max_epochs and not owner.should_stop

        loader: DataLoader[object] = _EpochPrefetchLoader(
            self._train,
            shuffle=True,
            generator=generator,
            prepare_next=prepare_next,
            **kwargs,
        )
        return cast("DataLoader[tuple[Tensor, PackedTargets]]", loader)

    def val_dataloader(self) -> DataLoader[tuple[Tensor, PackedTargets]]:
        """Return the unshuffled validation loader (letterbox-only samples).

        The loader runs on its own worker count — ``val_num_workers``, defaulting to a
        pixel-scaled cap on an auto-chosen train count (WP-073, WP-102) and to the
        memory budget of :meth:`_val_loader_workers` on a named one (WP-103): letterbox
        only samples need a fraction of the train pipeline's CPU, and both worker
        populations are resident at the epoch boundary where validation runs.

        Annotated with the collated ``(images, PackedTargets)`` transport, as
        :meth:`train_dataloader` is: the dataset yields ``(image, Targets)`` pairs, and
        the cast records the form :func:`collate_detection` actually hands the trainer.
        """
        if self._val is None:
            raise RuntimeError("setup() must be called before val_dataloader()")
        num_workers = self._val_loader_workers()
        workers = num_workers > 0
        kwargs = self._loader_kwargs() | {
            "num_workers": num_workers,
            "persistent_workers": self._persistent_workers and workers,
            "prefetch_factor": self._prefetch_factor if workers else None,
            "worker_init_fn": _init_worker if workers else None,
        }
        loader = DataLoader(
            self._val,
            shuffle=False,
            **kwargs,  # type: ignore[arg-type]
        )
        return cast("DataLoader[tuple[Tensor, PackedTargets]]", loader)

    @property
    def mosaic_prob(self) -> float:
        """Return the training pipeline's current mosaic probability.

        Returns:
            The probability with which each training sample is built from a
            four-image mosaic (``1.0`` until the ``close_mosaic`` schedule flips it).

        Raises:
            RuntimeError: If :meth:`setup` has not built the training pipeline yet.

        Examples:
            ```pycon
            >>> DetectionDataModule.mosaic_prob  # doctest: +SKIP
            >>> # dm.setup("fit"); dm.mosaic_prob
            >>> # 1.0

            ```
        """
        if self._train is None:
            raise RuntimeError("setup() must be called before reading mosaic_prob")
        return self._train.mosaic_p

    def set_mosaic_prob(self, prob: float) -> None:
        """Set the training pipeline's mosaic probability.

        This is the seam ``CloseMosaicCallback`` (WP-036) drives to disable mosaic
        for the final training epochs: it sets ``prob=0``. The value lands on the
        main-process pipeline, which loader worker processes copy when they start:
        in-process loading (``num_workers=0``) uses it from the next sample, and a
        pool started after this call — every later epoch's pool under the default
        ``persistent_workers=False`` — samples with it. Persistent workers keep the
        copy they started with and never see the change; with ``epoch_prefetch``
        also requested, a change of value warns once, since the next epoch's first
        batches are also already queued by then. Repeat calls are harmless.

        Args:
            prob: The new mosaic probability, forwarded to
                :attr:`_TrainPipeline.mosaic_p`.

        Raises:
            RuntimeError: If :meth:`setup` has not built the training pipeline yet.

        Examples:
            ```pycon
            >>> DetectionDataModule.set_mosaic_prob  # doctest: +SKIP
            >>> # dm.setup("fit"); dm.set_mosaic_prob(0.0); dm.mosaic_prob
            >>> # 0.0

            ```
        """
        if self._train is None:
            raise RuntimeError("setup() must be called before set_mosaic_prob()")
        prob = float(prob)
        if (
            prob != self._train.mosaic_p
            and self._epoch_prefetch
            and self._persistent_workers
            and self._num_workers > 0
            and not self._warned_mosaic_prefetch
        ):
            warnings.warn(
                f"set_mosaic_prob({prob}) does not reach the training workers: under epoch_prefetch with "
                "persistent_workers they keep the pipeline copy they started with, so the remaining epochs, "
                "a CloseMosaicCallback window included, still sample with "
                f"mosaic_prob={self._train.mosaic_p}. Set persistent_workers=False for a run whose "
                "augmentation changes between epochs",
                UserWarning,
                stacklevel=2,
            )
            self._warned_mosaic_prefetch = True
        self._train.mosaic_p = prob
