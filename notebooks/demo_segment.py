# SPDX-License-Identifier: Apache-2.0
# %% [markdown]
# # Instance segmentation on COCO 2017, end to end
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Borda/lucid-YOLO/blob/gh-pages/notebooks/demo_segment.ipynb)
#
# Provision COCO, fit the `n`-scale segmenter for two epochs, look at its curves, score
# boxes and masks with `trainer.validate`, and draw what it predicts beside what is
# annotated — through the library's own objects, built from the values
# `seg_nano_smoke.yaml` carries, so the module that trained is the one that answers at
# the end. This is a demonstration, not the Seg-smoke tier: that one trains on
# `train2017` for 50 epochs and is what `docs/model_cards/segmentation.md` reports.
# The run here uses `val2017` as both splits for two epochs — the layout of the A78
# validation run (`docs/ASSUMPTIONS.md`), a paired 4-epoch `det_nano_smoke` run on
# val2017 — so its numbers are a wiring check, not a generalisation figure.

# %%
import os
import sys
from pathlib import Path

if "google.colab" in sys.modules:
    # !git clone --depth=1 https://github.com/Borda/lucid-YOLO
    # %pip install -e lucid-YOLO
    os.chdir("lucid-YOLO")  # the cells below run from the checkout root, as the smoke test does
    sys.path.insert(0, "src")  # the editable install's .pth is read at interpreter start, not by a running kernel
FAST = os.environ.get("LUCID_NOTEBOOK_FAST") == "1"

# %% [markdown]
# ## The dataset
#
# `docs/DATASETS.md`, "COCO 2017 (R12)": "Automated, because the archives are served
# from a stable public host (`images.cocodataset.org`) at fixed URLs [...] an
# already-extracted split is skipped without touching the network." The full run
# downloads **val only** (about 1 GB) into the gitignored `data/coco2017`; `train2017`
# is 18 GB and two epochs do not need it. The same annotation file serves both loaders:
# `instances_val2017.json` carries the polygon rings the mask branch trains against.
#
# Under `LUCID_NOTEBOOK_FAST=1` nothing is downloaded: the dataset is the synthetic slice
# `scripts/overfit_micro.py` draws for the wiring gate, one hundred 128 px scenes whose
# every annotation carries a polygon, in a temporary directory the kernel removes when it exits.

# %%
import json
import tempfile

from lucid_yolo.data.download import download_coco

if FAST:
    sys.path.insert(0, "scripts")
    from overfit_micro import generate_slice

    # Held in a name so the kernel removes the directory when it exits.
    SCRATCH = tempfile.TemporaryDirectory(prefix="lucid_demo_segment_")
    WORK = Path(SCRATCH.name)
    DATA_ROOT = WORK / "slice"
    IMAGES = generate_slice(DATA_ROOT, generator_task="segmentation")
    ANN = IMAGES / "_annotations.coco.json"
else:
    WORK = Path("results/demo_segment")
    DATA_ROOT = Path("data/coco2017")
    download_coco(DATA_ROOT, splits=("val",))
    IMAGES = DATA_ROOT / "val2017"
    ANN = DATA_ROOT / "annotations" / "instances_val2017.json"
NUM_CLASSES = len(json.loads(ANN.read_text(encoding="utf-8"))["categories"])
print(f"{ANN}: {NUM_CLASSES} classes")

# %% [markdown]
# ## Model and data module
#
# `seg_nano_smoke.yaml` is the detection recipe with `task: segment`, and every value
# below is read off `src/lucid_yolo/configs/seg_nano_smoke.yaml`:
#
# - `variant: n` — `scale_spec("n")` expands the registry row into `depth`, `width`
#   and `max_channels` (ADR-001), and the same letter picks the augmentation policy.
# - `task: segment` — builds and supervises the coefficient stems, prototypes and aux
#   branch.
# - `lr: 0.01`, `lrf: 0.01`, `momentum: 0.95`, `weight_decay: 0.0005`.
# - `box_gain: 7.5`, `cls_gain: 0.5`, `l1_gain: 6.0`, `alpha_init: 0.8`,
#   `alpha_final: 0.1` — the detection objective, unchanged.
# - `mask_gain: 2.5`, `semantic_gain: 0.5` — the two WP-087 mask terms (A38) riding on
#   top of it.
# - `warmup_epochs: 3` becomes **1**: a two-epoch run would be all warmup, and the
#   learning rate would never decay.
#
# `mask_targets=True` on the data module is what `lucid-yolo fit` links from
# `model.task`: the loaders rasterise the polygon rings in their worker processes
# rather than on the training step. `batch_size 64` is the recipe's batch, so `lr` needs
# no scaling; the fast contract takes batch 4 at 128 px with no worker processes.

# %%
from pytorch_lightning import seed_everything

from lucid_yolo.models.registry import scale_spec
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.module import DetectionLitModule

seed_everything(0)
spec = scale_spec("n")
model = DetectionLitModule(
    spec.depth,
    spec.width,
    spec.max_channels,
    num_classes=NUM_CLASSES,
    task="segment",
    lr=0.01,
    lrf=0.01,
    warmup_epochs=1,
    momentum=0.95,
    weight_decay=0.0005,
    box_gain=7.5,
    cls_gain=0.5,
    l1_gain=6.0,
    mask_gain=2.5,
    semantic_gain=0.5,
    alpha_init=0.8,
    alpha_final=0.1,
)
dm = DetectionDataModule(
    DATA_ROOT,
    batch_size=4 if FAST else 64,
    num_workers=0 if FAST else 8,
    variant="n",
    img_size=128 if FAST else 640,
    train_images_dir=IMAGES,
    train_ann_file=ANN,
    val_images_dir=IMAGES,
    val_ann_file=ANN,
    mask_targets=True,
)

# %% [markdown]
# ## Train
#
# The same run from the shell, as `docs/TRAINING.md` "Instance segmentation — COCO 2017"
# gives it for the full tier:
#
# ```bash
# lucid-yolo fit --config seg_nano_smoke.yaml \
#   --data.data_root /content/coco2017 \
#   --data.batch_size 128 --data.num_workers 32 --data.prefetch_factor 2 \
#   --model.lr 0.02 --trainer.max_epochs 50 --trainer.precision bf16-mixed \
#   --trainer.default_root_dir /content/drive/MyDrive/lucid_runs
#
# lucid-eval --checkpoint <checkpoint> --data_root /content/coco2017 --output seg_report.json
# ```
#
# Two epochs is what fits a session. `accelerator="auto"` takes the CUDA device when
# there is one, else MPS or the CPU, where Lightning swaps `16-mixed` for `bf16-mixed`
# itself. The trainer carries the YAML's
# `gradient_clip_val: 10.0` (A31) and the logger pair `lucid-yolo fit` installs —
# TensorBoard plus CSV pinned to one `lightning_logs/version_N` directory, so
# `metrics.csv` lands beside the checkpoint; the YAML's `CloseMosaicCallback` (the
# final ten epochs of fifty) and EMA callback (a shadow for `lucid-eval` to load) are
# left out of a run that predicts with its live weights. On Colab, mount Drive and point
# `default_root_dir` there (`docs/TRAINING.md`, "Always: point `default_root_dir` at
# storage that outlives the runtime"); from a checkout the run lands in the gitignored
# `results/demo_segment`. The fast contract is one epoch of eight batches on the CPU.

# %%
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

tensorboard = TensorBoardLogger(save_dir=WORK)
loggers = [tensorboard, CSVLogger(save_dir=WORK, version=tensorboard.version)]
if FAST:
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        limit_train_batches=8,
        limit_val_batches=4,
        num_sanity_val_steps=0,
        gradient_clip_val=10.0,
        default_root_dir=WORK,
        logger=loggers,
    )
else:
    trainer = Trainer(
        max_epochs=2,
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=10.0,
        default_root_dir=WORK,
        logger=loggers,
    )
trainer.fit(model, datamodule=dm)

# %% [markdown]
# ## Training curves
#
# The run's `CSVLogger` wrote `metrics.csv` under `trainer.log_dir`: a step row carries
# the training terms, an epoch-end row the validation ones — `val/loss`, `val/mAP` and
# `val/segm_mAP` beside it. `plot_curves` draws the loss and the box and mask mAP from
# it. The fast case prints the file's last row instead.

# %%
import csv

from lucid_yolo._viz import plot_curves

METRICS = Path(trainer.log_dir) / "metrics.csv"
with METRICS.open(encoding="utf-8") as handle:
    last_row = list(csv.DictReader(handle))[-1]
print({key: value for key, value in last_row.items() if value})
# The cell's last expression, so the notebook renders the returned figure inline.
None if FAST else plot_curves(METRICS, title="seg_nano_smoke, val2017 as both splits")

# %% [markdown]
# ## Evaluate
#
# `trainer.validate` runs the validation epoch once more and reports what the run logs:
# `val/mAP` for the boxes and `val/segm_mAP` for the masks, the torchmetrics COCO
# mAP50-95 over the one-to-one branch's NMS-free decode, with `val/loss` and its terms
# (`val/mask`, `val/semantic` among them). From `docs/TRAINING.md`: "`lucid-eval` reads
# the task off the checkpoint and scores masks when the checkpoint has them, so the same
# command serves both COCO tiers." That dual-path acceptance report is
# `lucid_yolo.eval.detect_eval.run`, and it reads a checkpoint; it is named here, not
# called. Under the fast contract `limit_val_batches` already caps the call.

# %%
report = trainer.validate(model, datamodule=dm)[0]
for key in sorted(report):
    print(f"{key}: {report[key]:.4f}")

# %% [markdown]
# ## Predictions against ground truth
#
# `predict_and_draw` dispatches on the module's task: for `segment` it calls
# `predict_segmentation` and draws the predicted boxes and masks solid;
# `draw_ground_truth` lays the annotated polygons over them dashed, matched by file name
# from the annotation file the loaders read. The fast case draws nothing; it runs
# `predict_segmentation` on one slice image and checks the answer's shape — a
# `SegmentedPrediction` whose `(N, 6)` detections and `(N, H, W)` boolean masks are
# row-aligned.

# %%
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from lucid_yolo._viz.overlay import RenderOptions, draw_ground_truth, load_ground_truth, predict_and_draw
from lucid_yolo.predict import predict_segmentation

images = sorted(IMAGES.glob("*.jpg"))[: 2 if FAST else 4]
truth = load_ground_truth(ANN, "segment")


def prediction_grid(img_size: int) -> Figure:
    """One panel per image: predicted boxes and masks solid, the annotated polygons dashed."""
    figure, axes = plt.subplots(len(images) // 2, 2, figsize=(12, 4.5 * len(images) // 2))
    for ax, image in zip(axes.flat, images, strict=True):
        predict_and_draw(model, image, RenderOptions(img_size=img_size), axes=ax)
        draw_ground_truth(ax, truth[image.name], "segment")
    figure.tight_layout()
    return figure


if FAST:
    prediction = predict_segmentation(model, images[0], img_size=128, device=torch.device("cpu"))
    assert prediction.detections.shape[1:] == (6,), prediction.detections.shape
    assert prediction.masks.shape[0] == prediction.detections.shape[0], prediction.masks.shape
    print(f"{images[0].name}: {len(prediction.detections)} detections, masks {tuple(prediction.masks.shape)}")
# The cell's last expression, so the notebook renders the returned figure inline.
None if FAST else prediction_grid(640)

# %% [markdown]
# Two epochs on a nano model match few of the dashed outlines; the picture shows what the
# model answered and what it was asked for. The Seg-smoke tier's numbers — 26.1 box and
# 19.0 segm mAP50-95 on val2017 — are the model card's, from 50 epochs on `train2017`.
# The fast run writes only into a temporary directory the kernel removes when it exits.
