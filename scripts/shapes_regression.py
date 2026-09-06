# SPDX-License-Identifier: Apache-2.0
"""Synthetic-shapes held-out validation generalization golden (WP-083).

This accelerator-gated regression producer trains an ``n``-scale detector on a
seeded 2,000-image synthetic shape dataset, then evaluates its separate validation
split with :class:`lucid_yolo.eval.coco_eval.DualPathEvaluator`. Unlike the
overfit-100 golden, training and scoring never reuse the same images: its three
held-out mAP metrics make coordinate-frame mistakes, classification starvation, and
collapsed augmentation randomness visible without running a COCO tier.

Mosaic remains enabled throughout the training portion of the recipe and closes
only for the final epoch. That intentionally exercises the multi-image
augmentation pipeline and its per-worker, per-epoch RNG reseeding; an overfit
golden instead disables mosaic from epoch zero to make memorization stationary.

The model and optimizer recipe is loaded from the packaged Det-smoke n-scale config so
the run-level hyperparameters remain single-source-of-truth. This script pins only
the small generalization-gate budget, image size, dataset split, and worker count.
The latter is deliberately fixed because a machine-dependent worker count changes
which worker consumes each augmentation draw, making the golden unreproducible
across machines.

Golden policy: ``goldens/gpu/shapes_regression_det.json`` is deliberately absent
until the lead runs this producer twice on real hardware and replaces the marked
tolerance placeholder with the measured tolerance. The default offline golden
harness excludes ``goldens/gpu/``; recomputing this producer is an explicit
accelerator run.

Provenance: R1 Eq. 2-3, R1 Table S3, R21. Assumptions: A8, A13, A26, A31.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import yaml
from fuse_augmentations.data import generate_dataset  # type: ignore[import-untyped]
from fuse_augmentations.data.config import SplitRatios  # type: ignore[import-untyped]
from pytorch_lightning import Trainer, seed_everything

import lucid_yolo
from lucid_yolo.data.coco import CocoDetectionDataset
from lucid_yolo.data.letterbox import Letterbox
from lucid_yolo.decode.nms_path import NMSDecoder
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.eval.annotations import letterboxed_batches, load_eval_annotations
from lucid_yolo.eval.coco_eval import DualPathEvaluator
from lucid_yolo.models.registry import scale_spec
from lucid_yolo.ptl.callbacks import CloseMosaicCallback
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.module import DetectionLitModule

#: Repository root (``scripts/`` is one level below it).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Packaged n-scale detector recipe; it remains the single source of truth for
#: optimizer, loss, and LR-schedule hyperparameters shared with the Det-smoke run.
_RECIPE_PATH = Path(lucid_yolo.__file__).resolve().parent / "configs" / "det_nano_smoke.yaml"

#: The frozen accelerator golden, isolated from the default offline golden harness.
_GOLDEN_PATH = REPO_ROOT / "goldens" / "gpu" / "shapes_regression_det.json"

#: Harness producer spec recorded in the golden JSON schema.
_PRODUCER_SPEC = "scripts.shapes_regression:shapes_regression_det"

#: Gitignored canonical dataset cache; probes use sibling cache directories so a
#: small CLI run cannot make the default 2,000-image dataset appear complete.
_DATASET_DIR = REPO_ROOT / ".cache" / "shapes_regression"

#: A 2,000-image corpus gives the 90/10 held-out split enough examples to expose
#: semantic training regressions while staying markedly cheaper than mini-COCO.
_NUM_IMAGES = 2000

#: Fixed held-out train/validation partition; test remains empty because this gate
#: measures only validation generalization.
_SPLIT_RATIOS = SplitRatios(train=0.9, val=0.1, test=0.0)

#: Fresh synthetic-scene seed, distinct from overfit_micro's 20260801 seed.
_SLICE_SEED = 20260803

#: Square detector canvas: small enough for a cheap regression run and divisible by
#: every P3/P4/P5 stride used by the detection head.
_IMG_SIZE = 320

#: Six epochs leave mosaic active for five epochs and reserve exactly the final epoch
#: for un-composited samples, exercising both sides of the close-mosaic schedule.
_EPOCHS = 6

#: Fixed nonzero workers exercise the per-worker augmentation RNG seam. A
#: machine-dependent auto count would change the augmentation draw sequence and make
#: the golden unreproducible across machines.
_NUM_WORKERS = 2

#: Final-epoch mosaic window; unlike the overfit gate, this does not disable mosaic
#: for the full training budget.
_CLOSE_MOSAIC_EPOCHS = 1

#: Absolute band on every held-out mAP metric. Three consecutive macOS-arm64 MPS runs
#: reproduced 0.8823 / 0.8583 / 0.9842 to four decimals, so the same-platform spread is
#: zero and this band exists only to absorb cross-accelerator kernel differences (A26
#: records the same libm-rounding divergence for the fixture checksums). It is set an
#: order of magnitude below the regressions it must catch: WP-078 moved Det-smoke mAP by
#: 36% relative, which on this gate would be a swing of ~0.3, six times the band.
#:
#: What it is *not* validated against, stated because the band's whole justification is
#: the cross-accelerator case: no training run of this gate has ever happened on a second
#: device, so the divergence this absorbs has never been measured. The one cross-accelerator
#: datum this project holds -- oriented evaluation agreeing between MPS and CUDA to 1e-4 --
#: does not bound it: that is a fixed checkpoint scored twice, while this trains for six
#: epochs, where a kernel difference moves the gradients and compounds through the whole
#: trajectory rather than appearing once at the end.
#:
#: The cost of the band being loose is on record. At WP-167's re-freeze the NMS mAP50-95
#: moved 0.882337 to 0.870817 and passed silently; the same run absorbed a 0.026 shift on
#: the oriented overfit gate. A band wide enough to hide the movement it was sized for is
#: the defect ``clamped_tolerance`` fixes at the floor end, and this is its other end.
#: Narrowing it needs the measurement that does not exist yet: this gate trained on two
#: accelerators, several times each, with the spread between them read off the results.
#: Until then the number stays where a measurement put it rather than where a guess would.
_MAP_TOLERANCE = 0.05

#: Held-out NMS mAP50-95 — the discriminating metric of the three, with headroom in
#: both directions at 0.88.
_VAL_MAP_50_95_NMS_TOLERANCE = _MAP_TOLERANCE

#: Held-out E2E mAP50-95. Pinned alongside the NMS path so the two decode paths cannot
#: silently diverge; the observed gap on this easy task is 0.024 AP.
_VAL_MAP_50_95_E2E_TOLERANCE = _MAP_TOLERANCE

#: Held-out NMS mAP50. Kept as a cheap sanity pin, not as the primary guard: at 0.9842
#: it sits near the ceiling and so is the least sensitive of the three to a regression.
_VAL_MAP_50_NMS_TOLERANCE = _MAP_TOLERANCE

#: Roboflow-style annotation name emitted by the configured COCO generator.
_ANNOTATION_NAME = "_annotations.coco.json"

#: Task accepted by this WP's detection-only CLI.
_SUPPORTED_TASK = "det"


@dataclass(frozen=True)
class Recipe:
    """Hyperparameters read from the packaged Det-smoke n-scale YAML recipe.

    Attributes:
        variant: Scale letter selecting the compound-scaling multipliers.
        max_epochs: Producer epoch budget, pinned by this regression gate.
        close_mosaic: Final-epoch mosaic-disable window.
        seed: Global seed for model initialization and loader order.
        batch_size: Samples per batch.
        gradient_clip: Training gradient-norm clip.
        lr: MuSGD base learning rate.
        lrf: Final fraction of the linear LR decay.
        warmup_epochs: Initial linear warmup duration.
        momentum: MuSGD momentum.
        weight_decay: MuSGD decoupled weight decay.
        box_gain: CIoU-term gain shared by both loss branches.
        cls_gain: Classification-term gain shared by both loss branches.
        l1_gain: Stride-normalized L1-box-term gain shared by both branches.
        alpha_init: First-epoch one-to-many loss weight.
        alpha_final: Last-epoch one-to-many loss weight.
    """

    variant: str
    max_epochs: int
    close_mosaic: int
    seed: int
    batch_size: int
    gradient_clip: float
    lr: float
    lrf: float
    warmup_epochs: float
    momentum: float
    weight_decay: float
    box_gain: float
    cls_gain: float
    l1_gain: float
    alpha_init: float
    alpha_final: float


@dataclass(frozen=True)
class _DatasetSummary:
    """Exact dataset counts carried alongside an evaluator report."""

    num_train_images: int
    num_val_images: int
    num_val_instances: int
    num_classes: int


def load_recipe(path: Path = _RECIPE_PATH) -> Recipe:
    """Load the regression recipe from the packaged Det-smoke n-scale YAML.

    The packaged config supplies model, optimizer, loss, schedule, seed, and batch
    settings. The producer owns only its six-epoch budget and one-epoch
    close-mosaic window, making the generalization gate reproducible without
    duplicating the stable detector recipe.

    Args:
        path: Recipe YAML path. Defaults to the packaged ``det_nano_smoke.yaml``.

    Returns:
        The parsed regression :class:`Recipe`.

    Examples:
        >>> recipe = load_recipe()
        >>> recipe.variant, recipe.max_epochs, recipe.close_mosaic
        ('n', 6, 1)
    """
    config = yaml.safe_load(path.read_text())
    model = config["model"]
    trainer = config["trainer"]
    return Recipe(
        variant=str(config["variant"]),
        max_epochs=_EPOCHS,
        close_mosaic=_CLOSE_MOSAIC_EPOCHS,
        seed=int(config["seed_everything"]),
        batch_size=int(config["data"]["batch_size"]),
        gradient_clip=float(trainer["gradient_clip_val"]),
        lr=float(model["lr"]),
        lrf=float(model["lrf"]),
        warmup_epochs=float(model["warmup_epochs"]),
        momentum=float(model["momentum"]),
        weight_decay=float(model["weight_decay"]),
        box_gain=float(model["box_gain"]),
        cls_gain=float(model["cls_gain"]),
        l1_gain=float(model["l1_gain"]),
        alpha_init=float(model["alpha_init"]),
        alpha_final=float(model["alpha_final"]),
    )


def generate_shapes_dataset(
    dataset_dir: Path = _DATASET_DIR,
    num_images: int = _NUM_IMAGES,
) -> Path:
    """Materialize the seeded train/validation synthetic-shapes dataset.

    The generator writes COCO-format segmentation annotations so each shape carries
    both the polygon required by the repository reader and its detection box.
    Existing datasets with both split annotations are reused, matching the overfit
    producer's idempotent cache behavior.

    Args:
        dataset_dir: Parent directory for the generated ``train`` and ``val`` splits.
        num_images: Total image count split according to :data:`_SPLIT_RATIOS`.

    Returns:
        The materialized dataset directory.

    Examples:
        >>> root = generate_shapes_dataset()  # doctest: +SKIP
        >>> (root / 'val' / '_annotations.coco.json').is_file()  # doctest: +SKIP
        True
    """
    if not _has_split_annotations(dataset_dir):
        generate_dataset(
            dataset_dir,
            num_images=num_images,
            fmt="coco",
            task="segmentation",
            class_mode="shape",
            split_ratios=_SPLIT_RATIOS,
            seed=_SLICE_SEED,
            img_size=_IMG_SIZE,
        )
    return dataset_dir


def build_datamodule(dataset_dir: Path, recipe: Recipe) -> DetectionDataModule:
    """Build the train/validation datamodule for the generated split directories.

    Args:
        dataset_dir: Dataset parent holding ``train`` and ``val`` COCO splits.
        recipe: Parsed hyperparameters supplying batch size, seed, and n-scale
            augmentation policy.

    Returns:
        A datamodule with a fixed nonzero worker count for reproducible augmentation
        stream coverage.

    Examples:
        >>> dm = build_datamodule(generate_shapes_dataset(), load_recipe())  # doctest: +SKIP
        >>> dm.setup('fit')  # doctest: +SKIP
    """
    train_split = dataset_dir / "train"
    val_split = dataset_dir / "val"
    return DetectionDataModule(
        data_root=dataset_dir,
        batch_size=recipe.batch_size,
        num_workers=_NUM_WORKERS,
        variant=recipe.variant,
        img_size=_IMG_SIZE,
        train_images_dir=train_split,
        train_ann_file=train_split / _ANNOTATION_NAME,
        val_images_dir=val_split,
        val_ann_file=val_split / _ANNOTATION_NAME,
        seed=recipe.seed,
    )


def build_module(recipe: Recipe, num_classes: int) -> DetectionLitModule:
    """Build the n-scale detection module with the packaged recipe's loss settings.

    Args:
        recipe: Parsed recipe providing the n-scale and optimizer/loss parameters.
        num_classes: Contiguous class count read from the generated annotations.

    Returns:
        A configured :class:`~lucid_yolo.ptl.module.DetectionLitModule`.

    Examples:
        >>> module = build_module(load_recipe(), num_classes=4)
        >>> module.head.num_classes
        4
    """
    spec = scale_spec(recipe.variant)
    return DetectionLitModule(
        depth=spec.depth,
        width=spec.width,
        max_channels=spec.max_channels,
        num_classes=num_classes,
        lr=recipe.lr,
        lrf=recipe.lrf,
        warmup_epochs=recipe.warmup_epochs,
        momentum=recipe.momentum,
        weight_decay=recipe.weight_decay,
        box_gain=recipe.box_gain,
        cls_gain=recipe.cls_gain,
        l1_gain=recipe.l1_gain,
        alpha=recipe.alpha_init,
        alpha_init=recipe.alpha_init,
        alpha_final=recipe.alpha_final,
    )


def run_regression(num_images: int = _NUM_IMAGES, epochs: int = _EPOCHS) -> dict[str, float]:
    """Train the held-out shapes recipe and return exact counts plus val mAP metrics.

    Probe image counts use a sibling cache directory so ``--images`` cannot make a
    partial dataset masquerade as the canonical 2,000-image cache. Deterministic
    execution is attempted first; the MPS deterministic-kernel fallback mirrors
    the overfit producer and restarts from the fixed seed before retrying.

    Args:
        num_images: Total generated images. Defaults to :data:`_NUM_IMAGES`.
        epochs: Training epochs. Defaults to :data:`_EPOCHS`.

    Returns:
        Exact dataset/configuration counts and val NMS/E2E mAP metrics.

    Examples:
        >>> metrics = run_regression(num_images=40, epochs=1)  # doctest: +SKIP
        >>> sorted(metrics)  # doctest: +SKIP
        ['epochs', 'img_size', ..., 'val_map_50_nms']
    """
    _validate_budget(num_images, epochs)
    recipe = replace(load_recipe(), max_epochs=epochs, close_mosaic=_CLOSE_MOSAIC_EPOCHS)
    dataset_dir = _probe_dataset_dir(num_images)
    generate_shapes_dataset(dataset_dir, num_images)
    try:
        report, summary = _train_and_evaluate(recipe, dataset_dir, deterministic=True)
    except RuntimeError:
        report, summary = _train_and_evaluate(recipe, dataset_dir, deterministic=False)
    return _metric_values(
        num_train_images=summary.num_train_images,
        num_val_images=summary.num_val_images,
        num_val_instances=summary.num_val_instances,
        num_classes=summary.num_classes,
        epochs=recipe.max_epochs,
        report=report,
    )


def shapes_regression_det() -> dict[str, float]:
    """Produce the canonical zero-argument synthetic-shapes golden metrics.

    The golden harness resolves this function from :data:`_PRODUCER_SPEC`. It is
    intentionally accelerator-gated because it generates data and retrains the
    detector before evaluating both decode paths on the held-out validation split.

    Returns:
        The canonical 2,000-image metric mapping from :func:`run_regression`.

    Examples:
        >>> metrics = shapes_regression_det()  # doctest: +SKIP
        >>> metrics['num_val_images']  # doctest: +SKIP
        200
    """
    return run_regression()


def write_golden(metrics: dict[str, float], path: Path = _GOLDEN_PATH) -> None:
    """Write real producer metrics in the accelerator-golden harness schema.

    The tolerance fields deliberately retain their marked placeholder until the
    lead measures two full runs. This function never invents metric values: callers
    pass the measured mapping returned by :func:`run_regression`.

    Args:
        metrics: Real metric mapping to store under the golden ``values`` key.
        path: Destination JSON path. Defaults to :data:`_GOLDEN_PATH`.

    Examples:
        >>> write_golden(shapes_regression_det())  # doctest: +SKIP
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    golden = {
        "producer": _PRODUCER_SPEC,
        "tolerances": {
            "val_map_50_95_nms": _VAL_MAP_50_95_NMS_TOLERANCE,
            "val_map_50_95_e2e": _VAL_MAP_50_95_E2E_TOLERANCE,
            "val_map_50_nms": _VAL_MAP_50_NMS_TOLERANCE,
        },
        "values": metrics,
    }
    path.write_text(json.dumps(golden, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the held-out shapes regression and optionally write its golden JSON.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when the run completed, or ``1`` when a task or positive-budget guard
        rejects the request.

    Examples:
        The rejection reason is printed rather than returned, so the example
        captures stdout and asserts the exit code.

        >>> import contextlib, io
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     status = main(['--task', 'seg'])
        >>> status
        1
    """
    parser = argparse.ArgumentParser(description="Train and evaluate held-out synthetic shapes.")
    parser.add_argument("--task", default=_SUPPORTED_TASK, help="detection task (only 'det' is wired for WP-083)")
    parser.add_argument("--freeze", action="store_true", help="write goldens/gpu/shapes_regression_det.json")
    parser.add_argument("--images", type=int, default=_NUM_IMAGES, help="total synthetic images (probe override)")
    parser.add_argument("--epochs", type=int, default=_EPOCHS, help="training epochs (probe override)")
    args = parser.parse_args(argv)

    if args.task != _SUPPORTED_TASK:
        print(f"unsupported task {args.task!r}; only {_SUPPORTED_TASK!r} is wired for WP-083")
        return 1
    try:
        metrics = run_regression(num_images=args.images, epochs=args.epochs)
    except ValueError as error:
        print(f"invalid regression budget: {error}")
        return 1
    print(
        f"shapes-{args.task}: nms mAP50-95={metrics['val_map_50_95_nms']:.4f} "
        f"e2e mAP50-95={metrics['val_map_50_95_e2e']:.4f} "
        f"nms mAP50={metrics['val_map_50_nms']:.4f}"
    )
    if args.freeze:
        write_golden(metrics)
        print(f"froze golden -> {_GOLDEN_PATH.relative_to(REPO_ROOT)}")
    return 0


def _has_split_annotations(dataset_dir: Path) -> bool:
    """Return whether both generated COCO split annotations are present."""
    return all((dataset_dir / split / _ANNOTATION_NAME).is_file() for split in ("train", "val"))


def _validate_budget(num_images: int, epochs: int) -> None:
    """Reject nonpositive generation or training budgets before touching the cache."""
    if num_images <= 0:
        raise ValueError(f"num_images must be positive, got {num_images}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")


def _probe_dataset_dir(num_images: int) -> Path:
    """Return the canonical cache for the default size, else an isolated probe cache."""
    if num_images == _NUM_IMAGES:
        return _DATASET_DIR
    return _DATASET_DIR.with_name(f"{_DATASET_DIR.name}_probe_{num_images}")


def _make_trainer(recipe: Recipe, deterministic: bool) -> Trainer:
    """Build the quiet trainer that leaves validation to the dual-path evaluator."""
    return Trainer(
        max_epochs=recipe.max_epochs,
        accelerator="auto",
        deterministic=deterministic,
        gradient_clip_val=recipe.gradient_clip,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[CloseMosaicCallback(close_mosaic=recipe.close_mosaic)],
    )


def _train_and_evaluate(
    recipe: Recipe,
    dataset_dir: Path,
    deterministic: bool,
) -> tuple[dict[str, dict[str, float]], _DatasetSummary]:
    """Train once and score the held-out split through the two-path evaluator."""
    seed_everything(recipe.seed, workers=True)
    summary = _dataset_summary(dataset_dir)
    module = build_module(recipe, summary.num_classes)
    datamodule = build_datamodule(dataset_dir, recipe)
    _make_trainer(recipe, deterministic).fit(module, datamodule=datamodule)
    return _evaluate_validation(module, dataset_dir, recipe.batch_size), summary


def _dataset_summary(dataset_dir: Path) -> _DatasetSummary:
    """Read exact training and held-out validation counts from generated annotations."""
    train_split = dataset_dir / "train"
    val_split = dataset_dir / "val"
    train_dataset = CocoDetectionDataset(train_split, train_split / _ANNOTATION_NAME)
    val_images, val_targets, label_to_category = load_eval_annotations(val_split / _ANNOTATION_NAME)
    return _DatasetSummary(
        num_train_images=len(train_dataset),
        num_val_images=len(val_images),
        num_val_instances=sum(int(target["boxes"].shape[0]) for target in val_targets.values()),
        num_classes=len(label_to_category),
    )


def _evaluate_validation(
    module: DetectionLitModule,
    dataset_dir: Path,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    """Score the held-out split in original image coordinates through both decode paths."""
    val_split = dataset_dir / "val"
    images, targets, label_to_category = load_eval_annotations(val_split / _ANNOTATION_NAME)
    letterbox = Letterbox(_IMG_SIZE)
    evaluator = DualPathEvaluator(module, TopKDecoder(), NMSDecoder(), label_to_category, letterbox)
    device = torch.device(str(module.device))
    return evaluator.evaluate(letterboxed_batches(images, val_split, letterbox, batch_size), targets, device)


def _metric_values(
    *,
    num_train_images: int,
    num_val_images: int,
    num_val_instances: int,
    num_classes: int,
    epochs: int,
    report: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Flatten exact counts and the three chosen held-out evaluator statistics."""
    return {
        "num_train_images": num_train_images,
        "num_val_images": num_val_images,
        "num_val_instances": num_val_instances,
        "num_classes": num_classes,
        "epochs": epochs,
        "img_size": _IMG_SIZE,
        "val_map_50_95_nms": report["nms"]["map"],
        "val_map_50_95_e2e": report["e2e"]["map"],
        "val_map_50_nms": report["nms"]["map_50"],
    }


if __name__ == "__main__":
    sys.exit(main())
