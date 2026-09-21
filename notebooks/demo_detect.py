# SPDX-License-Identifier: Apache-2.0
# %% [markdown]
# # Detection on COCO 2017: download, fit, eval, predict
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Borda/lucid-YOLO/blob/gh-pages/notebooks/demo_detect.ipynb)
#
# The detection tier of `docs/TRAINING.md` "Detection — COCO 2017" through the library
# rather than its console scripts: `lucid_yolo.data.download.download_coco` for the
# dataset, `DetectionLitModule` and `DetectionDataModule` built from the values
# `det_nano_smoke.yaml` carries, a Lightning `Trainer` for the fit and the validation,
# and `lucid_yolo._viz` for the two figures. Every object is a Python object in this
# kernel, so the model that trained is the one that predicts at the end.
#
# This is a smoke of the pipeline, not a reproduction of the model card. The card's
# 25.3 mAP took 50 epochs over the 18 GB `train2017` split on a datacentre GPU
# (`docs/model_cards/detection.md`); a Colab session has neither the disk nor the hours.
# So this notebook downloads `val2017` only (1 GB, 5000 images) and trains **on `val2017`
# as both splits, for two epochs**, the way the Colab gate of WP-184 did
# (`docs/ASSUMPTIONS.md`, A78). The number the validation reports is therefore a model
# scored on its own training images, and after two epochs from scratch it is near zero
# anyway — the gate measured `~1e-5` after four. What the run proves is that download,
# fit, validate and predict still compose into one working path.

# %%
import os
import sys
from pathlib import Path

if "google.colab" in sys.modules:
    # !git clone --depth=1 https://github.com/Borda/lucid-YOLO
    # %pip install -e lucid-YOLO
    os.chdir("lucid-YOLO")
    # An editable install's `.pth` is read at interpreter start, not by a running kernel.
    sys.path.insert(0, "src")
FAST = os.environ.get("LUCID_NOTEBOOK_FAST") == "1"

# %% [markdown]
# ## The dataset
#
# `docs/DATASETS.md`, "COCO 2017 (R12)":
#
# > Automated, because the archives are served from a stable public host
# > (`images.cocodataset.org`) at fixed URLs [...] Transfers stream to a `.part` file and
# > are renamed on completion, so a killed run resumes rather than leaving a truncated
# > archive, and an already-extracted split is skipped without touching the network.
#
# `download_coco` is that command as a function. `splits=("val",)` rather than
# TRAINING.md's `[train,val]` is the 1 GB / 18 GB choice above; the annotations archive
# comes with it. From a checkout the root is the gitignored `data/coco2017`. Under the
# fast contract there is no download: the dataset is the seeded synthetic slice
# `scripts/overfit_micro.py` trains its gates on, one hundred 128 px scenes of geometric
# shapes in COCO format, written into a temporary directory the kernel removes when it exits.
# The head's class count is a property of the annotation file either way — 80 for COCO,
# four shapes for the slice.

# %%
import json
import tempfile

from lucid_yolo.data.download import download_coco

if FAST:
    sys.path.insert(0, "scripts")
    from overfit_micro import generate_slice

    # Held in a name so the kernel removes the directory when it exits.
    SCRATCH = tempfile.TemporaryDirectory(prefix="lucid_demo_detect_")
    WORK = Path(SCRATCH.name)
    DATA_ROOT = WORK / "slice"
    IMAGES = generate_slice(DATA_ROOT)
    ANN = IMAGES / "_annotations.coco.json"
else:
    WORK = Path("results/demo_detect")
    DATA_ROOT = Path("data/coco2017")
    download_coco(DATA_ROOT, splits=("val",))
    IMAGES = DATA_ROOT / "val2017"
    ANN = DATA_ROOT / "annotations" / "instances_val2017.json"
NUM_CLASSES = len(json.loads(ANN.read_text(encoding="utf-8"))["categories"])
print(f"{ANN}: {NUM_CLASSES} classes")

# %% [markdown]
# ## Model and data module
#
# `lucid-yolo fit --config det_nano_smoke.yaml` builds these two objects from the YAML;
# here the same values are passed by hand, each one read off
# `src/lucid_yolo/configs/det_nano_smoke.yaml`:
#
# - `variant: n` — a registry row, not a topology (ADR-001). `scale_spec("n")` expands it
#   into the `depth`, `width` and `max_channels` the module takes, and the same letter
#   picks the data module's augmentation policy.
# - `lr: 0.01`, `lrf: 0.01`, `momentum: 0.95`, `weight_decay: 0.0005` — the optimizer
#   and the linear-decay schedule.
# - `box_gain: 7.5`, `cls_gain: 0.5`, `l1_gain: 6.0`, `alpha_init: 0.8`,
#   `alpha_final: 0.1` — the loss gains and the ProgLoss ramp.
# - `warmup_epochs: 3` in the file becomes **1** here: a two-epoch run would otherwise
#   be spent entirely inside the opening linear warmup.
# - `num_classes: 80` in the file is the annotation file's count above, so the slice's
#   four classes work too.
#
# `seed_everything: 0` at the top of the file is one call. The data module is pointed at
# `val2017` for both splits explicitly; `batch_size 64` is the recipe's own batch, which
# is why `lr` needs no linear scaling (the `lr 0.02` in TRAINING.md is for batch 128),
# and `img_size 640` is the COCO side. The fast contract takes batch 4 at 128 px with
# no worker processes.

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
    task="detect",
    lr=0.01,
    lrf=0.01,
    warmup_epochs=1,
    momentum=0.95,
    weight_decay=0.0005,
    box_gain=7.5,
    cls_gain=0.5,
    l1_gain=6.0,
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
)

# %% [markdown]
# ## Train
#
# The same run from the shell, as `docs/TRAINING.md` "Detection — COCO 2017" gives it
# for the full tier:
#
# ```bash
# lucid-data download --data_root /content/coco2017 --splits '[train,val]' --verify true
# lucid-data check    --data_root /content/coco2017
#
# lucid-yolo fit --config det_nano_smoke.yaml \
#   --data.data_root /content/coco2017 \
#   --data.batch_size 128 --data.num_workers 32 --data.prefetch_factor 1 \
#   --model.lr 0.02 --trainer.max_epochs 50 --trainer.precision 16-mixed \
#   --trainer.default_root_dir /content/drive/MyDrive/lucid_runs
#
# lucid-eval --checkpoint <checkpoint> --data_root /content/coco2017 --output det_report.json
# ```
#
# Two epochs is what fits a Colab session. `accelerator="auto"` takes the CUDA device when
# there is one, else MPS or the CPU, where Lightning swaps `16-mixed` for `bf16-mixed`
# itself. The trainer below carries the YAML's
# `gradient_clip_val: 10.0` (A31, bounding the violent opening steps of a from-scratch
# recipe) and the loggers `lucid-yolo fit` installs — TensorBoard plus CSV pinned to one
# `lightning_logs/version_N` directory, so `metrics.csv` lands beside the checkpoint.
# The YAML's two callbacks are left out: `CloseMosaicCallback` closes mosaic for the
# final ten epochs of fifty, and the EMA shadow is for a checkpoint that `lucid-eval`
# loads, while this notebook predicts with the live weights. `docs/TRAINING.md`,
# "Always: point `default_root_dir` at storage that outlives the runtime":
#
# > `--trainer.default_root_dir` is the `save_dir` of both default loggers (TensorBoard
# > and CSV, pinned to a single `lightning_logs/version_N` directory) and the directory
# > checkpoints land in. On a hosted runtime it is the whole of what survives a
# > disconnect.
#
# On Colab, mount Drive and point `default_root_dir` there; from a checkout the run
# lands in the gitignored `results/demo_detect`. The fast contract is one epoch of eight
# batches on the CPU, into the temporary directory.

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
# ## The curves
#
# The run's `CSVLogger` wrote `metrics.csv` under `trainer.log_dir` — a sparse table
# where a step row carries the training terms and an epoch-end row the validation ones,
# `val/loss` and `val/mAP` among them. `lucid_yolo._viz.plot_curves` draws it the way the
# reproduction report's figures were drawn. The fast contract draws nothing and prints the
# last row instead.

# %%
import csv

from lucid_yolo._viz import plot_curves

METRICS = Path(trainer.log_dir) / "metrics.csv"
with METRICS.open(encoding="utf-8") as handle:
    last_row = list(csv.DictReader(handle))[-1]
print({key: value for key, value in last_row.items() if value})
# The figure is the cell's last expression, which is what a notebook renders inline.
None if FAST else plot_curves(METRICS, title="det_nano_smoke, val2017 as both splits")

# %% [markdown]
# ## Evaluate
#
# `trainer.validate` runs the validation epoch once more and reports what the run logs:
# `val/loss` with its per-term components and `val/mAP`, the torchmetrics COCO mAP50-95
# over the one-to-one branch's NMS-free decode. The dual-path acceptance report that
# `lucid-eval` writes — one forward per batch decoded once through the end-to-end path
# and once through NMS, reported as `e2e` and `nms` — is `lucid_yolo.eval.detect_eval.run`,
# and it reads a checkpoint; it is named here, not called. Under the fast contract the
# same call is already capped by `limit_val_batches`.

# %%
report = trainer.validate(model, datamodule=dm)[0]
for key in sorted(report):
    print(f"{key}: {report[key]:.4f}")

# %% [markdown]
# ## Predictions against the ground truth
#
# `predict_and_draw` answers for one image with the module in this kernel and draws the
# predictions as solid boxes labelled with their score; `draw_ground_truth` lays the
# truth from the same annotation file the loaders read over them as dashed outlines, both
# in the class's own colour. A dashed box with no solid one on it is a miss; after two
# epochs expect mostly those. The fast contract draws nothing; it runs `predict_image`
# on one slice image instead, which is the same answer as one `(N, 6)` tensor of
# `[x1, y1, x2, y2, score, class]` rows in original-image pixels.

# %%
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from lucid_yolo._viz.overlay import RenderOptions, draw_ground_truth, load_ground_truth, predict_and_draw
from lucid_yolo.predict import predict_image

images = sorted(IMAGES.glob("*.jpg"))[: 2 if FAST else 4]
truth = load_ground_truth(ANN, "detect")


def prediction_grid(img_size: int) -> Figure:
    """One panel per image: the module's predictions solid, the annotation file's boxes dashed."""
    figure, axes = plt.subplots(len(images) // 2, 2, figsize=(12, 4.5 * len(images) // 2))
    for ax, image in zip(axes.flat, images, strict=True):
        predict_and_draw(model, image, RenderOptions(img_size=img_size), axes=ax)
        draw_ground_truth(ax, truth[image.name], "detect")
    figure.tight_layout()
    return figure


if FAST:
    detections = predict_image(model, images[0], img_size=128, device=torch.device("cpu"))
    assert detections.shape[1:] == (6,), detections.shape
    print(f"{images[0].name}: {len(detections)} detections")
# The figure is the cell's last expression, which is what a notebook renders inline.
None if FAST else prediction_grid(640)

# %% [markdown]
# ## What this did, and did not, show
#
# Four library calls composed into one path from an empty runtime: a dataset on local
# disk, a fitted module with its `metrics.csv` and checkpoint under `default_root_dir`,
# a validation report, and a picture. What it did not show is the model card's number,
# and it could not: the card's detector saw `train2017` fifty times, and this one saw
# `val2017` twice. The recipe that produces the card is the TRAINING.md block quoted
# above, unchanged; the only things this notebook altered are the split, the epoch count
# and the batch size. The fast run writes only into a temporary directory the kernel
# removes when it exits.
