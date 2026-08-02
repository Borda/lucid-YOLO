# SPDX-License-Identifier: Apache-2.0
"""COCO detection :class:`~pytorch_lightning.LightningDataModule` (WP-014).

:class:`DetectionDataModule` wires the WP-014 :class:`~open_yolos.data.coco.CocoDetectionDataset`
into the Phase 1 augmentation pipeline and exposes train/val
:class:`~torch.utils.data.DataLoader` s. It builds both splits from the COCO 2017
layout of blueprint sec. 14.3 (``train2017/`` + ``annotations/instances_train2017.json``
and the matching ``val2017``) and, per the dataset contract (AGENTS.md sec. 3),
contains **no download logic**: a missing or mis-shaped root is validated by
``make check-data`` before any ``[DATA]`` run, never fetched here.

Multi-image augmentation composition:
    Mosaic, mixup and copy-paste each consume *several* images, so they cannot be
    ordinary single-image :class:`~open_yolos.data.transforms.GeometricTransform` s
    chained in a :class:`~open_yolos.data.transforms.Compose`. They are composed
    instead by :class:`_TrainPipeline`, a wrapper :class:`~torch.utils.data.Dataset`
    that, for each requested index, draws the extra source indices it needs from a
    single seeded :class:`torch.Generator` and assembles the sample:

        1. geometric base — with probability :attr:`_TrainPipeline.mosaic_p` a
           four-image :class:`~open_yolos.data.mosaic.MosaicAssembly` (else the single image),
           then :class:`~open_yolos.data.affine.RandomAffine`, then
           :class:`~open_yolos.data.letterbox.Letterbox` down to ``img_size``;
        2. with probability ``mixup`` a second full geometric sample is blended in
           via :class:`~open_yolos.data.mixup.Mixup`;
        3. with probability ``copy_paste`` polygon instances from a third geometric
           sample are pasted in via :class:`~open_yolos.data.mixup.CopyPaste`;
        4. photometric :class:`~open_yolos.data.augment.HSVJitter` and
           :class:`~open_yolos.data.augment.HorizontalFlip` finish the sample.

    The per-sample mixup/copy-paste probabilities come from
    :func:`~open_yolos.data.coco.build_scale_policy` (size-aware, [R1] Table S3);
    mosaic runs at ``p=1.0`` (blueprint sec. 5.9). Because every draw comes from
    one seeded generator, a fixed ``seed`` and a deterministic access order
    (``num_workers=0``) give byte-identical epochs.

Batch contract:
    :func:`collate_detection` stacks the equal-sized (letterboxed) images into a
    single ``(B, C, img_size, img_size)`` float32 tensor and returns the per-image
    :class:`~open_yolos.data.targets.Targets` **as a list** of length ``B`` — targets
    are ragged (each image has its own instance count), so they are deliberately
    not padded into a dense tensor here. A batch is the pair
    ``(images, list[Targets])``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from open_yolos.data.affine import RandomAffine
from open_yolos.data.augment import HorizontalFlip, HSVJitter
from open_yolos.data.coco import CocoDetectionDataset, build_scale_policy
from open_yolos.data.letterbox import Letterbox
from open_yolos.data.mixup import CopyPaste, Mixup
from open_yolos.data.mosaic import MosaicAssembly
from open_yolos.data.targets import Targets

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["DetectionDataModule", "collate_detection"]

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


def collate_detection(batch: list[tuple[Tensor, Targets]]) -> tuple[Tensor, list[Targets]]:
    """Collate ``(image, Targets)`` samples into ``(images, list[Targets])``.

    All images are letterboxed to a common ``img_size`` upstream, so they stack
    into one dense tensor; the per-image :class:`~open_yolos.data.targets.Targets`
    are ragged and kept as a list (never padded into a dense target tensor).

    Args:
        batch: The per-sample ``(image, Targets)`` pairs from the dataset.

    Returns:
        A ``(images, targets)`` pair: ``images`` is ``(B, C, H, W)`` float32 and
        ``targets`` is a length-``B`` list of :class:`~open_yolos.data.targets.Targets`.

    Examples:
        ```pycon
        >>> import torch
        >>> from open_yolos.data.targets import Targets
        >>> batch = [(torch.zeros(3, 4, 4), Targets.empty()) for _ in range(2)]
        >>> images, targets = collate_detection(batch)
        >>> images.shape, len(targets)
        (torch.Size([2, 3, 4, 4]), 2)

        ```
    """
    images = torch.stack([image for image, _ in batch], dim=0)
    targets = [target for _, target in batch]
    return images, targets


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
        base: The raw (untransformed) :class:`~open_yolos.data.coco.CocoDetectionDataset`.
        img_size: Target square side for the letterboxed output.
        policy: The size-aware strengths from
            :func:`~open_yolos.data.coco.build_scale_policy` (``scale``, ``mixup``,
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
        self._affine = RandomAffine(scale=policy["scale"], translate=_AFFINE_TRANSLATE, generator=self._generator)
        self._letterbox = Letterbox(self._img_size)
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
        """Build the geometric base: optional mosaic, then affine, then letterbox."""
        if self._draw() < self.mosaic_p:
            items = [self._base[i] for i in self._mosaic_indices(index)]
            image, targets = self._mosaic(items)
        else:
            image, targets = self._base[index]
        image, targets = self._affine(image, targets)
        return self._letterbox(image, targets)

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
            to ``min(batch_size, cpu count)`` — scale workers with the batch a
            step consumes, but never beyond the cores that can actually run
            them. Determinism of the seeded pipeline is guaranteed only at an
            explicit ``0`` (a single in-process generator).
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
        if num_workers is None:
            num_workers = min(self._batch_size, os.cpu_count() or 1)
        self._num_workers = int(num_workers)
        self._img_size = int(img_size)
        self._seed = int(seed)
        self._pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        self._prefetch_factor = int(prefetch_factor)
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
