# SPDX-License-Identifier: Apache-2.0
"""Worker-side training sample pipeline behind :mod:`lucid_yolo.ptl.datamodule`.

Holds what produces one training sample inside a DataLoader worker: :class:`_TrainPipeline`,
which composes mosaic, the fused affine, mixup, copy-paste, HSV and flip over a raw reader
(the "Multi-image augmentation composition" section of the datamodule's docstring);
:class:`_PreparedTrainingSource`, the optional fill-centred float32 source it can read from;
:func:`_init_worker`, the loaders' per-worker thread cap and re-seed; and
:func:`_host_sample`, which finishes a colour-deferred sample for host-side diagnostics.
:class:`~lucid_yolo.ptl.datamodule.DetectionDataModule`, which wires these into loaders,
stays in :mod:`lucid_yolo.ptl.datamodule`.
"""

from __future__ import annotations

import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor
from torch.utils.data import Dataset, get_worker_info
from tqdm.auto import tqdm

from lucid_yolo.data._common import _DEFAULT_PAD_VALUE
from lucid_yolo.data.affine import RandomAffine
from lucid_yolo.data.augment import HorizontalFlip, HSVJitter
from lucid_yolo.data.coco import CocoDetectionDataset
from lucid_yolo.data.mixup import CopyPaste, Mixup
from lucid_yolo.data.mosaic import MosaicAssembly
from lucid_yolo.data.targets import Targets
from lucid_yolo.data.yolo import YoloDetectionDataset
from lucid_yolo.ptl._transport import _DeferredHSVSample, _TrainingSample

#: Either layout's reader. Both yield ``(image, Targets)`` samples and size the same way, which
#: is what lets the augmentation pipeline, the collate and every consumer stay layout-blind.
_DetectionSource = CocoDetectionDataset | YoloDetectionDataset

#: Three RGB channels, each retained as an IEEE float32 rather than an 8-bit code.
_RGB_FLOAT_PIXEL_BYTES = 3 * torch.finfo(torch.float32).bits // 8

#: Default mosaic probability: every training sample (blueprint sec. 5.9: ``mosaic
#: p=1.0``). The ``close_mosaic`` late-epoch disable flips the per-pipeline
#: :attr:`_TrainPipeline.mosaic_p` seam to ``0`` via
#: :meth:`DetectionDataModule.set_mosaic_prob` (driven by ``CloseMosaicCallback``, WP-036).
_MOSAIC_PROB = 1.0
#: Standard YOLO-lineage affine translate fraction ([R1] Table S3; scale is size-aware).
_AFFINE_TRANSLATE = 0.1
#: Standard YOLO-lineage horizontal-flip probability ([R1] Table S3: ``fliplr=0.5``).
_FLIP_PROB = 0.5


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


class _PreparedTrainingSource(Dataset[tuple[Tensor, Targets]]):
    """Retain the immutable, fill-centred float image before random geometry.

    Converting a source and subtracting the affine fill are independent of every
    random draw. Mosaic can assemble those pixels on a zero-filled canvas; the
    affine then receives exactly the tensor its ordinary fill subtraction creates.
    The fill is restored after the same resample, before any colour or blending.
    This costs four bytes per source pixel channel instead of the byte cache's one.
    ``base`` must be a raw reader without ``transforms``: its pixels are retained as
    the source canvas the training geometry starts from. The memory reports come
    from rank zero only, so a multi-process run states them once.

    Raises:
        ValueError: If ``base`` carries ``transforms``; their output would be retained
            and then transformed again by the training geometry.
    """

    def __init__(self, base: _DetectionSource) -> None:
        if base._transforms is not None:
            raise ValueError(
                "prepared image cache needs a raw reader without transforms; "
                "the training geometry would apply them a second time"
            )
        self.base = base
        self.keypoint_flip_pairs = base.keypoint_flip_pairs
        estimate = ""
        if isinstance(base, CocoDetectionDataset):
            expected = _RGB_FLOAT_PIXEL_BYTES * sum(record.height * record.width for record in base._images)
            estimate = f"about {expected / 1e9:.1f} GB ({expected:,} bytes) from the annotation sizes; "
        rank_zero_warn(
            f"prepared image cache: {estimate}retaining float32 source images costs four times "
            "an 8-bit decoded image cache in host RAM per training process"
        )
        self._items: list[tuple[Tensor, Targets]] = []
        for index in tqdm(range(len(base)), desc="Preparing images", unit="img"):
            image, targets = base[index]
            # Subtract out of place: the reader's tensor is never written, so nothing
            # here depends on whether a reader hands out fresh or shared storage. The
            # cost is one transient image per record, released on the next iteration.
            self._items.append((image - _DEFAULT_PAD_VALUE, targets))
        retained = sum(image.nbytes for image, _ in self._items)
        rank_zero_warn(
            f"prepared image cache: holding {retained:,} bytes of float32 pixels "
            f"for {len(self._items)} training records"
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
            a base whose *reader* yields no rings rather than whose modality does not.
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
