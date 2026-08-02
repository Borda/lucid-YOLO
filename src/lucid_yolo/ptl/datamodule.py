# SPDX-License-Identifier: Apache-2.0
"""COCO detection :class:`~pytorch_lightning.LightningDataModule` (WP-014).

:class:`DetectionDataModule` wires the WP-014 :class:`~lucid_yolo.data.coco.CocoDetectionDataset`
into the Phase 1 augmentation pipeline and exposes train/val
:class:`~torch.utils.data.DataLoader` s. It builds both splits from the COCO 2017
layout of blueprint sec. 14.3 (``train2017/`` + ``annotations/instances_train2017.json``
and the matching ``val2017``) and, per the dataset contract (AGENTS.md sec. 3),
contains **no download logic**: a missing or mis-shaped root is validated by
``make check-data`` before any ``[DATA]`` run, never fetched here.

Multi-image augmentation composition:
    Mosaic, mixup and copy-paste each consume *several* images, so they cannot be
    ordinary single-image :class:`~lucid_yolo.data.transforms.GeometricTransform` s
    chained in a :class:`~lucid_yolo.data.transforms.Compose`. They are composed
    instead by :class:`_TrainPipeline`, a wrapper :class:`~torch.utils.data.Dataset`
    that, for each requested index, draws the extra source indices it needs from a
    single seeded :class:`torch.Generator` and assembles the sample:

        1. geometric base — with probability :attr:`_TrainPipeline.mosaic_p` a
           four-image :class:`~lucid_yolo.data.mosaic.MosaicAssembly` (else the single image),
           then :class:`~lucid_yolo.data.affine.FusedAffineLetterbox`, which composes
           the random affine and the letterbox down to ``img_size`` into one resample;
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
    (``num_workers=0``) give byte-identical epochs.

Batch contract:
    Two forms, split by the DataLoader worker boundary:

    * **Transport form** (what :func:`collate_detection` returns): the equal-sized
      (letterboxed) images stacked into a single ``(B, C, img_size, img_size)``
      float32 tensor, paired with a :class:`PackedTargets` — the batch's ragged
      per-image :class:`~lucid_yolo.data.targets.Targets` flattened into a fixed
      set of eight dense tensors. A worker ships each distinct tensor as its own
      shared-memory segment, so the natural ``list[Targets]`` (with its per-image
      *list* of polygon rings) would explode into hundreds of tiny segments and
      exhaust the consumer's mmap budget; packing bounds the batch to a handful.
    * **Consumer form** (what every downstream step sees): the pair
      ``(images, list[Targets])``. :meth:`DetectionDataModule.on_after_batch_transfer`
      unpacks the transport form on the destination device, so the module and
      everything downstream keep consuming a ragged length-``B`` target list
      exactly as before. A direct (non-Lightning) consumer converts with
      :func:`unpack_targets`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lucid_yolo.data.affine import FusedAffineLetterbox
from lucid_yolo.data.augment import HorizontalFlip, HSVJitter
from lucid_yolo.data.coco import CocoDetectionDataset, build_scale_policy
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.data.mixup import CopyPaste, Mixup
from lucid_yolo.data.mosaic import MosaicAssembly
from lucid_yolo.data.targets import Targets

__all__ = [
    "DetectionDataModule",
    "PackedTargets",
    "collate_detection",
    "pack_targets",
    "unpack_targets",
]

#: Column count of a polygon point ``(x, y)`` — the width of a packed ring row.
_POINT_DIM = 2

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


def _limit_worker_threads(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` capping each worker to one torch CPU thread.

    Without the cap every worker process inherits torch's default intra-op
    thread pool (one thread per core), so ``num_workers`` workers oversubscribe
    the CPU by that factor and the tensor ops inside the augmentation pipeline
    (affine warps, HSV jitter, blends) thrash instead of running in parallel.
    One thread per worker makes worker throughput scale with ``num_workers``.

    Args:
        worker_id: The worker index (unused; required by the DataLoader API).
    """
    del worker_id
    torch.set_num_threads(1)


@dataclass
class PackedTargets:
    """Batch of per-image :class:`~lucid_yolo.data.targets.Targets` flattened for IPC.

    A DataLoader worker ships every distinct tensor as its own shared-memory
    segment, so the natural ``list[Targets]`` batch — a per-image list whose every
    image also carries a *list* of per-instance polygon rings — explodes into
    hundreds of tiny segments and exhausts the consumer's mmap budget
    (``vm.max_map_count``) at realistic worker counts. This container collapses the
    whole ragged batch into a **fixed set of eight dense tensors**: the modality
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
        boxes_per_image: ``(B,)`` int64 instance count per image; splits
            ``boxes_cat``/``labels_cat`` back into ``B`` images.
        rboxes_cat: ``(sum_M, 5)`` float32 long-edge rotated boxes, concatenated.
        rboxes_per_image: ``(B,)`` int64 rotated-box count per image; splits
            ``rboxes_cat``.
        polygon_points_cat: ``(sum_P, 2)`` float32 polygon points of every ring of
            every image, concatenated.
        points_per_ring: ``(R,)`` int64 point count per ring; splits
            ``polygon_points_cat`` into ``R`` rings.
        rings_per_image: ``(B,)`` int64 ring count per image (``0`` when an image
            carries no polygons, else its instance count); groups the ``R`` rings
            back into per-image lists.

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
    boxes_per_image: Tensor
    rboxes_cat: Tensor
    rboxes_per_image: Tensor
    polygon_points_cat: Tensor
    points_per_ring: Tensor
    rings_per_image: Tensor


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
        A :class:`PackedTargets` holding the batch as eight dense tensors on the
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

    def counts(values: list[int]) -> Tensor:
        return torch.tensor(values, dtype=torch.int64, device=boxes_cat.device)

    return PackedTargets(
        boxes_cat=boxes_cat,
        labels_cat=labels_cat,
        boxes_per_image=counts([int(target.boxes.shape[0]) for target in targets]),
        rboxes_cat=rboxes_cat,
        rboxes_per_image=counts([int(target.rboxes.shape[0]) for target in targets]),
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
    rings_per_image = [int(count) for count in packed.rings_per_image.tolist()]
    points_per_ring = [int(count) for count in packed.points_per_ring.tolist()]

    boxes = torch.split(packed.boxes_cat, boxes_per_image)
    labels = torch.split(packed.labels_cat, boxes_per_image)
    rboxes = torch.split(packed.rboxes_cat, rboxes_per_image)
    rings = list(torch.split(packed.polygon_points_cat, points_per_ring)) if points_per_ring else []

    targets: list[Targets] = []
    ring_cursor = 0
    for index, ring_count in enumerate(rings_per_image):
        image_rings = rings[ring_cursor : ring_cursor + ring_count]
        ring_cursor += ring_count
        targets.append(
            Targets(boxes=boxes[index], labels=labels[index], polygons=list(image_rings), rboxes=rboxes[index])
        )
    return targets


def collate_detection(batch: list[tuple[Tensor, Targets]]) -> tuple[Tensor, PackedTargets]:
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

    Args:
        batch: The per-sample ``(image, Targets)`` pairs from the dataset.

    Returns:
        A ``(images, packed)`` pair: ``images`` is ``(B, C, H, W)`` float32 and
        ``packed`` is the batch's targets as a :class:`PackedTargets`.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> batch = [(torch.zeros(3, 4, 4), Targets.empty()) for _ in range(2)]
        >>> images, packed = collate_detection(batch)
        >>> images.shape, packed.boxes_per_image.tolist()
        (torch.Size([2, 3, 4, 4]), [0, 0])

        ```
    """
    images = torch.stack([image for image, _ in batch], dim=0)
    packed = pack_targets([target for _, target in batch])
    return images, packed


class _TrainPipeline(Dataset[tuple[Tensor, Targets]]):
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
        base: The raw (untransformed) :class:`~lucid_yolo.data.coco.CocoDetectionDataset`.
        img_size: Target square side for the letterboxed output.
        policy: The size-aware strengths from
            :func:`~lucid_yolo.data.coco.build_scale_policy` (``scale``, ``mixup``,
            ``copy_paste``).
        seed: Seed for the single :class:`torch.Generator` driving every draw.
    """

    def __init__(self, base: CocoDetectionDataset, img_size: int, policy: dict[str, float], seed: int) -> None:
        self._base = base
        self._img_size = int(img_size)
        self.mosaic_p = float(_MOSAIC_PROB)
        self._mixup_prob = policy["mixup"]
        self._copy_paste_prob = policy["copy_paste"]
        self._generator = torch.Generator().manual_seed(seed)
        self._mosaic = MosaicAssembly(self._img_size, generator=self._generator)
        self._fused = FusedAffineLetterbox(
            self._img_size,
            scale=policy["scale"],
            translate=_AFFINE_TRANSLATE,
            generator=self._generator,
        )
        self._mixup = Mixup(p=1.0, generator=self._generator)
        self._copy_paste = CopyPaste(p=1.0, generator=self._generator)
        self._hsv = HSVJitter(generator=self._generator)
        self._flip = HorizontalFlip(p=_FLIP_PROB, generator=self._generator)

    def __len__(self) -> int:
        """Return the number of base images."""
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[Tensor, Targets]:
        """Assemble the fully-augmented ``(image, Targets)`` sample for ``index``."""
        image, targets = self._geometric(index)
        image, targets = self._maybe_mixup(image, targets)
        image, targets = self._maybe_copy_paste(image, targets)
        image, targets = self._hsv(image, targets)
        return self._flip(image, targets)

    def _geometric(self, index: int) -> tuple[Tensor, Targets]:
        """Build the geometric base: optional mosaic, then the fused affine+letterbox.

        The random affine and the letterbox down to ``img_size`` are composed into
        a single image resample (:class:`~lucid_yolo.data.affine.FusedAffineLetterbox`),
        so the geometric path costs one bilinear pass instead of two.
        """
        if self._draw() < self.mosaic_p:
            items = [self._base[i] for i in self._mosaic_indices(index)]
            image, targets = self._mosaic(items)
        else:
            image, targets = self._base[index]
        return self._fused(image, targets)

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


class DetectionDataModule(LightningDataModule):
    """COCO detection datamodule assembling the Phase 1 augmentation pipeline (WP-014).

    Train and val splits are built from the COCO 2017 layout under ``data_root``
    (``train2017/`` + ``annotations/instances_train2017.json`` and the ``val2017``
    equivalents); each split path may be overridden explicitly (used by the offline
    fixture tests, which have a single Roboflow-style split). Training samples pass
    through :class:`_TrainPipeline` (mosaic/mixup/copy-paste/affine/hsv/flip +
    letterbox); validation is letterbox-only. No split is ever downloaded.

    Args:
        data_root: COCO 2017 root directory (holds ``train2017``, ``val2017``,
            ``annotations``).
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
        img_size: Square letterbox side for every emitted sample. Defaults to 640.
        train_images_dir: Override for the train images directory. Defaults to
            ``data_root/"train2017"``.
        train_ann_file: Override for the train annotation JSON. Defaults to
            ``data_root/"annotations"/"instances_train2017.json"``.
        val_images_dir: Override for the val images directory. Defaults to
            ``data_root/"val2017"``.
        val_ann_file: Override for the val annotation JSON. Defaults to
            ``data_root/"annotations"/"instances_val2017.json"``.
        seed: Seed for the training pipeline's generator and loader shuffling.
        pin_memory: Whether loader batches land in page-locked host memory for
            async host-to-device copies. ``None`` (default) resolves to ``True``
            exactly when CUDA is available — MPS and CPU runs get ``False``
            (pinning buys nothing there and MPS warns on it).
        prefetch_factor: Batches each worker keeps prefetched (defaults to
            :data:`_PREFETCH_FACTOR`); ignored at ``num_workers=0`` where the
            DataLoader forbids it.

    Examples:
        ```pycon
        >>> DetectionDataModule  # doctest: +SKIP
        >>> # dm = DetectionDataModule(root, batch_size=2, num_workers=0, variant="n")
        >>> # dm.setup("fit"); images, targets = next(iter(dm.train_dataloader()))

        ```
    """

    def __init__(
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
    ) -> None:
        super().__init__()
        self._batch_size = int(batch_size)
        self._img_size = int(img_size)
        self._prefetch_factor = int(prefetch_factor)
        if num_workers is None:
            num_workers = _shm_capped_workers(
                min(self._batch_size, os.cpu_count() or 1),
                self._batch_size,
                self._img_size,
                self._prefetch_factor,
            )
        self._num_workers = int(num_workers)
        self._seed = int(seed)
        self._pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        self._policy = build_scale_policy(variant)
        self._train_images_dir = train_images_dir or data_root / "train2017"
        self._train_ann_file = train_ann_file or data_root / "annotations" / "instances_train2017.json"
        self._val_images_dir = val_images_dir or data_root / "val2017"
        self._val_ann_file = val_ann_file or data_root / "annotations" / "instances_val2017.json"
        self._train: _TrainPipeline | None = None
        self._val: CocoDetectionDataset | None = None

    def prepare_data(self) -> None:
        """No-op: datasets are provisioned out of band, never downloaded (AGENTS.md sec. 3)."""

    def setup(self, stage: str | None = None) -> None:
        """Build the train pipeline and the letterbox-only val dataset.

        Args:
            stage: The Lightning stage (``"fit"``/``"validate"``/…); unused, both
                splits are always built so repeated calls are idempotent.
        """
        base = CocoDetectionDataset(self._train_images_dir, self._train_ann_file)
        self._train = _TrainPipeline(base, self._img_size, self._policy, self._seed)
        self._val = CocoDetectionDataset(self._val_images_dir, self._val_ann_file, transforms=Letterbox(self._img_size))

    def on_after_batch_transfer(
        self, batch: tuple[Tensor, PackedTargets] | tuple[Tensor, list[Targets]], dataloader_idx: int
    ) -> tuple[Tensor, list[Targets]]:
        """Restore the ``list[Targets]`` consumer contract on the destination device.

        Lightning calls this after moving the batch to the accelerator, so the
        :class:`PackedTargets` tensor fields are already on-device when they are
        unpacked — the reconstructed per-image targets land on the same device the
        module trains on, with no extra host-to-device hop. A batch whose targets
        are already a list (a non-packed direct feed) is passed through untouched,
        so the hook is safe to run over either form. Both the train and val loaders
        share this single conversion.

        Args:
            batch: The transferred ``(images, PackedTargets)`` transport batch, or
                an already-unpacked ``(images, list[Targets])`` batch.
            dataloader_idx: The loader index (unused; the conversion is uniform).

        Returns:
            The ``(images, list[Targets])`` consumer batch — the exact contract the
            module and every downstream step expect.

        Examples:
            ```pycon
            >>> DetectionDataModule.on_after_batch_transfer  # doctest: +SKIP
            >>> # images, targets = dm.on_after_batch_transfer(next(iter(dm.val_dataloader())), 0)

            ```
        """
        del dataloader_idx
        images, targets = batch
        if isinstance(targets, PackedTargets):
            return images, unpack_targets(targets)
        return images, targets

    def _loader_kwargs(self) -> dict[str, object]:
        """Return the streaming DataLoader kwargs shared by both loaders.

        Accelerator-aware: ``pin_memory`` is on only where it helps (resolved in
        the constructor — CUDA yes, MPS/CPU no), ``prefetch_factor`` and the
        one-thread-per-worker cap (:func:`_limit_worker_threads`) apply only
        with workers, so the deterministic ``num_workers=0`` path is untouched.
        """
        workers = self._num_workers > 0
        return {
            "batch_size": self._batch_size,
            "num_workers": self._num_workers,
            "collate_fn": collate_detection,
            "persistent_workers": workers,
            "pin_memory": self._pin_memory,
            "prefetch_factor": self._prefetch_factor if workers else None,
            "worker_init_fn": _limit_worker_threads if workers else None,
        }

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Targets]]:
        """Return the shuffled training loader over the augmentation pipeline."""
        if self._train is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        shuffle_generator = torch.Generator().manual_seed(self._seed)
        return DataLoader(
            self._train,
            shuffle=True,
            generator=shuffle_generator,
            **self._loader_kwargs(),  # type: ignore[arg-type]
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Targets]]:
        """Return the unshuffled validation loader (letterbox-only samples)."""
        if self._val is None:
            raise RuntimeError("setup() must be called before val_dataloader()")
        return DataLoader(
            self._val,
            shuffle=False,
            **self._loader_kwargs(),  # type: ignore[arg-type]
        )

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
        for the final training epochs: it sets ``prob=0``. The change is consulted
        on the next sample the pipeline assembles. Repeat calls are harmless.

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
        self._train.mosaic_p = float(prob)
