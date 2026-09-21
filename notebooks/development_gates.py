# SPDX-License-Identifier: Apache-2.0
# %% [markdown]
# # Development gates
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Borda/lucid-YOLO/blob/gh-pages/notebooks/development_gates.ipynb)
#
# The checks a developer runs from a checkout before a training run that costs hours,
# each one the command `docs/TRAINING.md` and the `Makefile` name, run here as that
# command. Nothing is imported from the scripts: the verdict is the line each one prints.
#
# From `docs/TRAINING.md`, "Before any of them: the wiring gate":
#
# > Each tier has a minutes-long overfit gate that must pass before a launch that costs
# > hours. It is development tooling and ships in `scripts/`, not in the wheel, so it runs
# > from a checkout
#
# > A tier that cannot overfit a handful of images will not converge on the full set, and
# > finding that out after the first epoch of a fifty-epoch run costs the run.

# %%
import os
import sys

if "google.colab" in sys.modules:
    # !git clone --depth=1 https://github.com/Borda/lucid-YOLO
    # %pip install -e lucid-YOLO
    os.chdir("lucid-YOLO")
FAST = os.environ.get("LUCID_NOTEBOOK_FAST") == "1"
# The smoke test runs this file with `LUCID_NOTEBOOK_FAST=1`: every gate then trains a
# few batches of one epoch instead of its budget, prints its score and reports PROBE
# rather than PASS or FAIL -- the wiring is exercised, the floor is not. Without the
# variable the commands below are exactly the ones a developer runs.
PROBE = "--epochs 1 --batches 4" if FAST else ""
SHAPES_PROBE = "--images 40 --epochs 1" if FAST else ""

# %% [markdown]
# ## The wiring gate, one task at a time
#
# `scripts/overfit_micro.py` trains the `n`-scale model from scratch on a fixed synthetic
# slice of about a hundred images and scores it on the same images. Every piece it
# exercises -- loader, assignment, loss, optimizer, decode, metric -- has its own tests;
# what the gate asks is whether they compose. A loop that is wired correctly memorizes
# the slice and clears the floor; one piece on the wrong grid, one stem left
# unsupervised, one target channel silently dropped, and it does not, in minutes rather
# than after a COCO epoch. Four tasks, four floors, each on its own metric.

# %%
# !python scripts/overfit_micro.py --task detect $PROBE

# %% [markdown]
# Train recall at IoU 0.5 over the one-to-one branch, floor 0.95.

# %%
# !python scripts/overfit_micro.py --task segment $PROBE

# %% [markdown]
# Mean train mask IoU, floor 0.7; the slice's shapes carry polygons, so every box has a
# mask to learn.

# %%
# !python scripts/overfit_micro.py --task obb $PROBE

# %% [markdown]
# Train rotated mAP50, floor 0.9, on a slice of rotated rectangles.

# %%
# !python scripts/overfit_micro.py --task keypoints $PROBE

# %% [markdown]
# Train OKS AP, floor 0.30. The floor looks low beside the other three and is not
# measuring less: this is the one slice not drawn from the geometric shapes -- a 7-point
# synthetic symbol schema -- and OKS at a uniform sigma is a cliff on objects a few dozen
# pixels across (`docs/TRAINING.md` has the derivation).
#
# ## The generalization regression
#
# `scripts/shapes_regression.py` is the other development gate: rather than memorizing a
# slice it trains on 1800 generated scenes and scores 200 held-out ones through both
# decode paths, and the frozen `goldens/gpu/shapes_regression_det.json` holds the band
# the numbers must stay in. It is detection-only and about two minutes on an
# accelerator; `make shapes` is the same command.

# %%
# !python scripts/shapes_regression.py --task det $SHAPES_PROBE

# %% [markdown]
# ## What a green run means
#
# Four PASS lines and a shapes line inside its band say the training loop composes for
# every task at this commit. They say nothing about COCO or DOTA numbers; those are the
# tiers in `docs/TRAINING.md`, and the per-task demo notebooks walk one short run of
# each. `make gate-gpu` runs these same gates nightly against the frozen goldens.
