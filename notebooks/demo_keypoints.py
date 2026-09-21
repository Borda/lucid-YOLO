# SPDX-License-Identifier: Apache-2.0
# %% [markdown]
# # Keypoints on COCO `person_keypoints`
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Borda/lucid-YOLO/blob/gh-pages/notebooks/demo_keypoints.ipynb)
#
# The fourth task through the library: `download_coco` fetches COCO 2017 `val2017` and
# its annotations, `DetectionLitModule` and `DetectionDataModule` are built from the
# values `pose_nano_smoke.yaml` carries and a `Trainer` fits the `n`-scale keypoint
# model for two short epochs, the run's curves are plotted, `trainer.validate` scores
# the module with OKS beside box AP, and a few images are drawn with predictions beside
# ground truth. Nothing here is a recipe for a good pose model — the accepted tier is 50
# epochs on `train2017` (`docs/TRAINING.md`, `docs/model_cards/keypoints.md`); this is
# the wiring, on `val2017` as both splits, the layout the A78 validation run used
# (`docs/ASSUMPTIONS.md`), so every number below is a training-set number.
#
# From `docs/TRAINING.md`, the two things that make this task the odd one out:
#
# > The config states `train_ann_file`/`val_ann_file` explicitly: `lucid_yolo.data.layout`
# > resolves only the `instances_` spelling, and `person_keypoints_{split}2017.json` sits
# > beside it in the same annotations archive under a different name.
#
# > **The task is `K`-generic; the 17-point COCO schema is one instantiation.**
# > `num_keypoints: 17` selects R12's own sigma table for OKS. Any other `K` is
# > trainable, but `lucid-eval`'s pose protocol refuses it rather than scoring it
# > against a schema it does not share — the ground truth, point ordering and sigmas
# > are all the person schema's.

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
# ## Data: the second annotation file
#
# `docs/DATASETS.md`, "The keypoint tier needs a second annotation file, and must be
# told so":
#
# > Nothing extra has to be downloaded. `person_keypoints_train2017.json` and
# > `person_keypoints_val2017.json` ship inside `annotations_trainval2017.zip` — the same
# > archive `lucid-data download` already fetches — and land beside the `instances_`
# > files in `annotations/`.
# >
# > What *is* required is naming them. `lucid_yolo.data.layout` resolves only the
# > `instances_` spelling, in every one of its `CANDIDATES` rows [...]
#
# So `download_coco` with the default `splits=("val",)` fetches `val2017` plus the
# annotations archive (about 1 GB), and the data module below names the
# `person_keypoints` file for both splits. Under `FAST` the same paths point at the
# synthetic 7-point kite slice the wiring gate `scripts/overfit_micro.py --task kp`
# trains on — `generate_slice(..., "keypoints", _KP_SHAPES)` is the call
# `run_overfit("kp")` makes — in a temporary directory the kernel removes when it exits, and
# `K` is 7, the slice's own; nothing is downloaded.

# %%
import json
import tempfile

from lucid_yolo.data.download import download_coco

if FAST:
    sys.path.insert(0, "scripts")
    from overfit_micro import _KP_SHAPES, generate_slice

    # Held in a name so the kernel removes the directory when it exits.
    SCRATCH = tempfile.TemporaryDirectory(prefix="lucid_demo_keypoints_")
    WORK = Path(SCRATCH.name)
    DATA_ROOT = WORK / "slice"
    IMAGES = generate_slice(DATA_ROOT, "keypoints", _KP_SHAPES)
    ANN = IMAGES / "_annotations.coco.json"
    NUM_KEYPOINTS = 7
else:
    WORK = Path("results/demo_keypoints")
    DATA_ROOT = Path("data/coco2017")
    download_coco(DATA_ROOT, splits=("val",))
    IMAGES = DATA_ROOT / "val2017"
    ANN = DATA_ROOT / "annotations" / "person_keypoints_val2017.json"
    NUM_KEYPOINTS = 17
coco = json.loads(ANN.read_text(encoding="utf-8"))
NUM_CLASSES = len(coco["categories"])
# The skeleton is dataset metadata, as the flip pairs are (A64): the head predicts `K`
# points and knows nothing about what they mean. COCO lists its edges 1-based.
SKELETON = [(start - 1, end - 1) for start, end in coco["categories"][0].get("skeleton", [])]
print(f"{ANN}: {NUM_CLASSES} classes, K={NUM_KEYPOINTS}, {len(SKELETON)} skeleton edges")

# %% [markdown]
# ## Model and data module
#
# `pose_nano_smoke.yaml` is the detection recipe with `task: keypoints`; every value
# below is read off `src/lucid_yolo/configs/pose_nano_smoke.yaml`:
#
# - `variant: n` — `scale_spec("n")` expands the registry row into `depth`, `width`
#   and `max_channels` (ADR-001), and the same letter picks the augmentation policy.
# - `task: keypoints` — builds the point stem and adds the R14 residual term.
# - `num_classes: 1` — COCO `person_keypoints` has one category, read off the
#   annotation file here; `num_keypoints: 17` is required and has no default, and the
#   slice's is 7.
# - `lr: 0.01`, `lrf: 0.01`, `momentum: 0.95`, `weight_decay: 0.0005`.
# - `box_gain: 7.5`, `cls_gain: 0.5`, `l1_gain: 6.0`, `alpha_init: 0.8`,
#   `alpha_final: 0.1` — the detection objective, unchanged.
# - `keypoint_gain: 1.0` — the residual log-likelihood term riding on top (A68).
# - `warmup_epochs: 3` becomes **1**, since the recipe's three warmup epochs would
#   otherwise cover the whole run and the learning rate would never decay.
#
# `keypoint_targets=True` on the data module is what `lucid-yolo fit` links from
# `model.task`: the loader reads the point fields, and the annotation file names the
# left/right flip pairs the horizontal flip needs. Naming all four split paths states
# the layout outright, so the root is never probed for an `instances_` file. `batch_size
# 64` is the recipe's batch, so `lr` needs no scaling; the fast contract takes batch 4
# at 128 px with no worker processes.

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
    task="keypoints",
    num_keypoints=NUM_KEYPOINTS,
    lr=0.01,
    lrf=0.01,
    warmup_epochs=1,
    momentum=0.95,
    weight_decay=0.0005,
    box_gain=7.5,
    cls_gain=0.5,
    l1_gain=6.0,
    keypoint_gain=1.0,
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
    keypoint_targets=True,
)

# %% [markdown]
# ## Train
#
# The same run from the shell, as `docs/TRAINING.md` "Keypoints — COCO 2017
# `person_keypoints`" gives it for the full tier:
#
# ```bash
# lucid-yolo fit --config pose_nano_smoke.yaml \
#   --data.data_root /content/coco2017 \
#   --data.batch_size 128 --data.num_workers 32 --data.prefetch_factor 1 \
#   --model.lr 0.02 --trainer.max_epochs 50 --trainer.precision bf16-mixed \
#   --trainer.default_root_dir /content/drive/MyDrive/lucid_runs
#
# lucid-eval --checkpoint <checkpoint> --data_root /content/coco2017 --output pose_report.json
# ```
#
# Two epochs is what fits a session. `accelerator="auto"` takes the CUDA device when
# there is one, else MPS or the CPU, where Lightning swaps `16-mixed` for `bf16-mixed`
# itself. The trainer carries the YAML's
# `gradient_clip_val: 10.0` (A31) and the logger pair `lucid-yolo fit` installs —
# TensorBoard plus CSV pinned to one `lightning_logs/version_N` directory, so
# `metrics.csv` lands beside the checkpoint; the YAML's `CloseMosaicCallback` (the
# final ten epochs of fifty) and EMA callback (a shadow for `lucid-eval` to load) are
# left out of a run that predicts with its live weights. Keypoint runs are single-device
# by construction: `val/oks_mAP` accumulates per-rank lists nothing gathers
# (`docs/TRAINING.md`). On Colab, mount Drive and point `default_root_dir` there; from
# a checkout the run lands in the gitignored `results/demo_keypoints`. The fast contract
# is one epoch of eight batches on the CPU.

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
# ## Curves
#
# The run's `CSVLogger` wrote `metrics.csv` under `trainer.log_dir`. A keypoint run logs
# `val/mAP` for the box branch and `val/oks_mAP` for the point branch side by side —
# neither bounds the other (`docs/model_cards/keypoints.md`). `plot_curves` draws the
# keypoint panel from it; the fast case prints the last row instead.

# %%
import csv

from lucid_yolo._viz import plot_curves

METRICS = Path(trainer.log_dir) / "metrics.csv"
with METRICS.open(encoding="utf-8") as handle:
    last_row = list(csv.DictReader(handle))[-1]
print({key: value for key, value in last_row.items() if value})
# The cell's last expression, so the notebook renders the returned figure inline.
None if FAST else plot_curves(METRICS, title="pose_nano_smoke, val2017 as both splits")

# %% [markdown]
# ## Evaluate: OKS beside box AP
#
# `trainer.validate` runs the validation epoch once more and reports what the run logs:
# `val/mAP` against the person boxes and `val/oks_mAP`, the OKS-matched AP over the
# NMS-free decode, with `val/loss` and its terms (`val/keypoint`, `val/oks` among them).
# The sigma table is picked by `K`, which is why the fast run's 7-point slice is scored
# here where `lucid-eval` would refuse it: the dual-path acceptance report with the ten
# `oks_` statistics for both decode paths is `lucid_yolo.eval.pose_eval.run`, it reads a
# checkpoint and it scores COCO's 17-point person schema only — named here, not called.
# Under the fast contract `limit_val_batches` already caps the call.

# %%
report = trainer.validate(model, datamodule=dm)[0]
for key in sorted(report):
    print(f"{key}: {report[key]:.4f}")

# %% [markdown]
# ## Predictions beside ground truth
#
# `predict_and_draw` dispatches on the module's task: for `keypoints` it calls
# `predict_keypoints` — a `KeypointPrediction` whose `(N, 6)` detections and `(N, K, 2)`
# point sets are row-aligned, in original-image pixels — and draws the boxes and points
# solid, with the skeleton the annotation file names passed through `RenderOptions`;
# `draw_ground_truth` lays the truth over them dashed with hollow markers, matched by
# file name from `person_keypoints_val2017.json`. The first few images the annotation file gives at
# least one labelled point are the ones drawn. The fast case draws nothing; it runs
# `predict_keypoints` on one slice image and checks that `K` came back as 7.

# %%
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from lucid_yolo._viz.overlay import RenderOptions, draw_ground_truth, load_ground_truth, predict_and_draw
from lucid_yolo.predict import predict_keypoints

file_names = {image["id"]: image["file_name"] for image in coco["images"]}
with_points = dict.fromkeys(file_names[a["image_id"]] for a in coco["annotations"] if a.get("num_keypoints", 0) > 0)
images = [IMAGES / name for name in list(with_points)[: 2 if FAST else 4]]
truth = load_ground_truth(ANN, "keypoints")


def prediction_grid(img_size: int) -> Figure:
    """One panel per image: predicted boxes, points and skeleton solid, the labelled points dashed and hollow."""
    figure, axes = plt.subplots(len(images) // 2, 2, figsize=(12, 4.5 * len(images) // 2))
    for ax, image in zip(axes.flat, images, strict=True):
        predict_and_draw(model, image, RenderOptions(img_size=img_size, skeleton=SKELETON), axes=ax)
        draw_ground_truth(ax, truth[image.name], "keypoints")
    figure.tight_layout()
    return figure


if FAST:
    prediction = predict_keypoints(model, images[0], img_size=128, device=torch.device("cpu"))
    assert prediction.num_keypoints == NUM_KEYPOINTS, prediction.keypoints.shape
    assert prediction.keypoints.shape[0] == prediction.detections.shape[0], prediction.keypoints.shape
    print(f"{images[0].name}: {len(prediction.detections)} detections, K={prediction.num_keypoints}")
# The cell's last expression, so the notebook renders the returned figure inline.
None if FAST else prediction_grid(640)

# %% [markdown]
# ## What to read next
#
# - `docs/TRAINING.md` — the tier command and the `train2017` run this file stands in for.
# - `docs/model_cards/keypoints.md` — where `K`'s genericity stops: the model generalizes,
#   the OKS protocol does not.
# - `scripts/overfit_micro.py --task kp` — the wiring gate whose slice the fast run borrows.
#
# The fast run writes only into a temporary directory the kernel removes when it exits.
