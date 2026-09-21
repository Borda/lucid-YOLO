# SPDX-License-Identifier: Apache-2.0
# %% [markdown]
# # Oriented detection on DOTA-v1.0 tiles
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Borda/lucid-YOLO/blob/gh-pages/notebooks/demo_obb.ipynb)
#
# The oriented tier end to end, through the library: a DOTA-v1.0 root you provisioned by
# hand, tiled by `lucid_yolo.data.tiles.build_tiles`, a short fit of the
# `obb_nano_smoke.yaml` recipe with `DetectionLitModule` and `DetectionDataModule` built
# from its values, the training curves, `trainer.validate` on the val tiles, and a few
# tiles with the predicted rotated boxes drawn over the annotated ones. Every object is a
# Python object in this kernel, so the module that trained is the one that predicts.
#
# Two epochs is a wiring run, not a result: the tier is ~50 epochs, and the number the
# model card claims comes from that run.

# %%
import os
import sys
from pathlib import Path

if "google.colab" in sys.modules:
    # !git clone --depth=1 https://github.com/Borda/lucid-YOLO
    # %pip install -e lucid-YOLO
    os.chdir("lucid-YOLO")  # cwd must be the checkout root so `scripts/`, `data/`, `results/` resolve
    sys.path.insert(0, "src")  # the editable install's .pth is read at interpreter start, not now
FAST = os.environ.get("LUCID_NOTEBOOK_FAST") == "1"

# %% [markdown]
# ## The data: a manual step
#
# `docs/DATASETS.md` on DOTA-v1.0, quoted with elisions marked:
#
# > **Manual, and it stays manual.** The distribution has no equivalent of
# > `images.cocodataset.org`: the dataset page offers Google Drive and Baidu Drive
# > folders, which are interactive pages rather than archive URLs. [...] So `lucid-data
# > download` covers COCO only, and DOTA provisioning is an operator step.
# >
# > Start at the dataset page of the official DOTA site, maintained by the authors of R18:
# >
# > - <https://captain-whu.github.io/DOTA/dataset.html>
# >
# > Under **Data Download**, take the **DOTA-v1.0** section — the page also carries
# > DOTA-v1.5 and DOTA-v2.0 sections, and those are different label sets. [...] The
# > testing images are not needed and not downloaded [...]
# >
# > Unpack so that each split directory holds an `images/` and a `labelTxt/` directory,
# > with one label file per image sharing its stem:
# >
# > ```text
# > <dota_root>/
# >     train/
# >         images/P0000.png ...
# >         labelTxt/P0000.txt ...
# >     val/
# >         images/P0003.png ...
# >         labelTxt/P0003.txt ...
# > ```
#
# Do that under the gitignored `data/dota` of the checkout — or anywhere, and export
# `DOTA_ROOT` before starting the kernel; under Colab, mount Drive and point at the
# unpacked tree. The tier trains on 1024 px tiles, never on the original tree, and
# `build_tiles` writes them as a COCO container with quadrilateral rings — the layout
# `obb_nano_smoke.yaml` names — into `data/dota_tiles`; `overlap=512` is R18's own crop
# stride (`docs/DATASETS.md`, "Then build the tiles"). Tiles already on disk are reused.
#
# With `LUCID_NOTEBOOK_FAST=1` the variable is not read: the fast run uses the synthetic
# oriented slice `scripts/overfit_micro.py` trains its wiring gate on — the same 128 px
# scenes with each object's rotated box as a four-corner ring, one split serving as both
# train and val — and touches no DOTA.

# %%
import json
import tempfile

from lucid_yolo.data.layout import resolve_split
from lucid_yolo.data.tiles import build_tiles

if FAST:
    sys.path.insert(0, "scripts")
    from overfit_micro import generate_slice

    # Held in a name so the kernel removes the directory when it exits.
    SCRATCH = tempfile.TemporaryDirectory(prefix="lucid_demo_obb_")
    WORK = Path(SCRATCH.name)
    TILES = WORK / "slice"
    IMAGES = generate_slice(TILES, generator_task="obb")  # writes <slice>/train/ with its COCO file
    ANN = IMAGES / "_annotations.coco.json"
else:
    DOTA_ROOT = Path(os.environ.get("DOTA_ROOT", "data/dota"))  # <- the unpacked <dota_root> from the quote above
    if not DOTA_ROOT.is_dir():
        raise SystemExit(
            f"{DOTA_ROOT} is not a directory. DOTA-v1.0 is provisioned by hand (docs/DATASETS.md): download the "
            "DOTA-v1.0 training and validation sets from the dataset page, unpack them so each split "
            "holds images/ and labelTxt/, and put them there or set DOTA_ROOT to the directory before "
            "running this notebook. Set LUCID_NOTEBOOK_FAST=1 to run the synthetic fast case instead."
        )
    WORK = Path("results/demo_obb")
    TILES = Path("data/dota_tiles")
    # The same naming rule the data module reads the tiles root with (WP-098); tiles
    # already on disk are reused rather than rebuilt.
    IMAGES, ANN = resolve_split(TILES, "val")
    if not ANN.is_file():
        if build_tiles(DOTA_ROOT, TILES, splits="train,val", overlap=512):
            raise SystemExit(f"build_tiles could not tile {DOTA_ROOT}; see the FAIL line above")
        IMAGES, ANN = resolve_split(TILES, "val")
NUM_CLASSES = len(json.loads(ANN.read_text(encoding="utf-8"))["categories"])
print(f"{ANN}: {NUM_CLASSES} classes")

# %% [markdown]
# ## Model and data module
#
# `obb_nano_smoke.yaml` is the detection recipe with `task: obb`; every value below is
# read off `src/lucid_yolo/configs/obb_nano_smoke.yaml`:
#
# - `variant: n` — `scale_spec("n")` expands the registry row into `depth`, `width`
#   and `max_channels` (ADR-001), and the same letter picks the augmentation policy.
# - `task: obb` — builds both branches' angle stems and swaps in the rotated box terms.
# - `num_classes: 15` — DOTA-v1.0's classes; read off the annotation file here, so the
#   slice's four shapes work too.
# - `lr: 0.01`, `lrf: 0.01`, `momentum: 0.95`, `weight_decay: 0.0005`.
# - `box_gain: 7.5`, `l1_gain: 6.0` — the IoU and L1 *slots*, which under this task
#   hold ProbIoU (A49) and the rotated L1 (A50); `cls_gain: 0.5` unchanged by the task.
# - `angle_gain: 0.25` — R1 Eq. 15's weight (A22); `rotated_iou_form: hellinger` — R17's
#   bounded form (A49).
# - `alpha_init: 0.8`, `alpha_final: 0.1` — the ProgLoss ramp.
# - `warmup_epochs: 3` becomes **1**, so the second epoch trains at the decaying rate
#   rather than still ramping.
#
# `rotated_targets=True` on the data module is what `lucid-yolo fit` links from
# `model.task`: the loader fits a long-edge rotated box to each four-corner ring. The
# full run reads the tiles at the recipe's `img_size: 1024` — R18's crop side — at
# batch 16, the placeholder the recipe ships ("tune to accelerator memory"); the fast
# contract takes batch 4 at 128 px, one split as both, with no worker processes. The run
# is single-device by construction: `lucid-yolo fit` refuses `task: obb` on more than
# one process, because the rotated mAP accumulates per-rank lists nothing gathers
# (`docs/TRAINING.md`).

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
    task="obb",
    lr=0.01,
    lrf=0.01,
    warmup_epochs=1,
    momentum=0.95,
    weight_decay=0.0005,
    box_gain=7.5,
    cls_gain=0.5,
    l1_gain=6.0,
    angle_gain=0.25,
    rotated_iou_form="hellinger",
    alpha_init=0.8,
    alpha_final=0.1,
)
if FAST:
    dm = DetectionDataModule(
        TILES,
        batch_size=4,
        num_workers=0,
        variant="n",
        img_size=128,
        train_images_dir=IMAGES,
        train_ann_file=ANN,
        val_images_dir=IMAGES,
        val_ann_file=ANN,
        rotated_targets=True,
    )
else:
    # The tiles root alone is enough: `lucid_yolo.data.layout` resolves `train`/`val`
    # and their `annotations/instances_<split>.json` (WP-098).
    dm = DetectionDataModule(TILES, batch_size=16, num_workers=8, variant="n", img_size=1024, rotated_targets=True)

# %% [markdown]
# ## Train
#
# The same run from the shell, as `docs/TRAINING.md` "Oriented detection — DOTA-v1.0
# tiles" gives it for the full tier:
#
# ```bash
# lucid-data check       --data_root /content/dota --dataset dota
# lucid-data build-tiles --root /content/dota --out /content/dota_tiles --splits train,val --overlap 512 --workers 8
#
# lucid-yolo fit --config obb_nano_smoke.yaml \
#   --data.data_root /content/dota_tiles \
#   --data.batch_size 16 --data.num_workers 8 \
#   --trainer.max_epochs 50 --trainer.precision bf16-mixed \
#   --trainer.default_root_dir /content/drive/MyDrive/lucid_runs
#
# lucid-eval --checkpoint <checkpoint> --data_root /content/dota_tiles --split val --output obb_report.json
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
# `results/demo_obb`. The fast contract is one epoch of eight batches on the CPU.

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
# The run's `CSVLogger` wrote `metrics.csv` under `trainer.log_dir`: loss per step, and
# at each epoch end `val/loss`, `val/rotated_mAP50` and `val/rotated_mAP` (mAP50-95).
# `plot_curves` draws the oriented panel from it. The fast case prints the last row
# instead of drawing.

# %%
import csv

from lucid_yolo._viz import plot_curves

METRICS = Path(trainer.log_dir) / "metrics.csv"
with METRICS.open(encoding="utf-8") as handle:
    last_row = list(csv.DictReader(handle))[-1]
print({key: value for key, value in last_row.items() if value})
# The last expression of the cell, so the notebook renders the returned Figure inline.
None if FAST else plot_curves(METRICS, title="obb_nano_smoke, DOTA-v1.0 val tiles")

# %% [markdown]
# ## Evaluate
#
# `trainer.validate` runs the validation epoch once more and reports what an oriented
# run logs: `val/rotated_mAP50` and `val/rotated_mAP`, rotated mAP per tile over the
# NMS-free decode, with `val/loss` and its rotated terms — and no `val/mAP`, since the
# axis-aligned metric answers a different question. The per-whole-image figure, the
# tiles merged back on the window provenance `build_tiles` wrote, is what `lucid-eval`
# reports: that is `lucid_yolo.eval.rotated_eval.run`, and it reads a checkpoint; it is
# named here, not called. Under the fast contract `limit_val_batches` already caps the
# call.

# %%
report = trainer.validate(model, datamodule=dm)[0]
for key in sorted(report):
    print(f"{key}: {report[key]:.4f}")

# %% [markdown]
# ## Predictions against ground truth
#
# A few val tiles: `predict_and_draw` dispatches on the module's task, so for `obb` it
# calls `predict_oriented` and draws the predicted rotated boxes solid;
# `draw_ground_truth` lays the annotated quadrilaterals over them dashed, matched by
# file name from the tiles' own annotation file. After two epochs the picture is
# headings starting to line up with the dashed truth, not the tier's result. The fast
# case draws nothing; it runs `predict_oriented` on one slice image and checks the
# answer's shape — `(N, 7)` rows of `[cx, cy, w, h, theta, score, class]`, the A45
# convention, in original-image pixels.

# %%
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from lucid_yolo._viz.overlay import RenderOptions, draw_ground_truth, load_ground_truth, predict_and_draw
from lucid_yolo.predict import predict_oriented

images = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in {".png", ".jpg"})[: 2 if FAST else 4]
truth = load_ground_truth(ANN, "obb")


def prediction_grid(img_size: int) -> Figure:
    """One panel per tile: the predicted rotated boxes solid, the annotated quadrilaterals dashed."""
    figure, axes = plt.subplots(len(images) // 2, 2, figsize=(12, 6 * len(images) // 2))
    for ax, image in zip(axes.flat, images, strict=True):
        predict_and_draw(model, image, RenderOptions(img_size=img_size), axes=ax)
        draw_ground_truth(ax, truth[image.name], "obb")
    figure.tight_layout()
    return figure


if FAST:
    detections = predict_oriented(model, images[0], img_size=128, device=torch.device("cpu"))
    assert detections.shape[1:] == (7,), detections.shape
    print(f"{images[0].name}: {len(detections)} rotated detections")
# The last expression of the cell, so the notebook renders the returned Figure inline.
None if FAST else prediction_grid(1024)

# %% [markdown]
# ## What is left on disk
#
# The full run keeps `data/dota_tiles/` and `results/demo_obb/`: the tiles, the
# Lightning run with its checkpoint and `metrics.csv`. What it did not show is the model
# card's number: that comes from the ~50-epoch tier on the same tiles. The fast run
# writes only into a temporary directory the kernel removes when it exits.
