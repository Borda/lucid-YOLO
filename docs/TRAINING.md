# 🏋️ Launching a training run

Four tiers, one command. `lucid-yolo fit --config <name>.yaml` resolves the name against the configs packaged inside the wheel, so a run needs no checkout — `det_nano_smoke.yaml`, `seg_nano_smoke.yaml`, `obb_nano_smoke.yaml` and `pose_nano_smoke.yaml` are all installed with the package.

What the configs carry is run-level configuration only: schedule, optimizer and loss gains, and placeholder data paths. Topology is named by `variant` and lives in the registry, never in a config file (ADR-001). Every value below that overrides a config is an override *of a placeholder or of a batch-size-dependent value*, not a correction.

`--data.data_root` is the only path any tier needs. The split directories and their annotation files are resolved from it by convention — COCO 2017's `train2017` spelling, the plain `train` the tiler writes, an `images/<split>` tree, or a per-split `_annotations.coco.json` export (`lucid_yolo.data.layout`). A layout none of those names still takes the four explicit overrides.

Provisioning the datasets these recipes read: docs/DATASETS.md. What the accepted runs actually measured, with the exact commands as they were run at the time: docs/REPRODUCTION_REPORT.md.

## 💾 Always: point `default_root_dir` at storage that outlives the runtime

`--trainer.default_root_dir` is the `save_dir` of both default loggers (TensorBoard and CSV, pinned to a single `lightning_logs/version_N` directory) and the directory checkpoints land in. On a hosted runtime it is the whole of what survives a disconnect.

This is not a precaution in the abstract. 0.2.0's Seg-smoke ran with `default_root_dir` under `/content/`, and its model card records the hardware line as unrecorded because the logs went with the runtime.

```python
from google.colab import drive

drive.mount("/content/drive")
```

Then `--trainer.default_root_dir /content/drive/MyDrive/lucid_runs` on every command below. The **dataset** goes the other way: it stays on the runtime's local disk, because every epoch reads every image and Drive is a network filesystem mounted through FUSE.

## 🎯 Detection — COCO 2017

```bash
lucid-data download --data_root /content/coco2017 --splits '[train,val]' --verify true
lucid-data check    --data_root /content/coco2017

lucid-yolo fit --config det_nano_smoke.yaml \
  --data.data_root /content/coco2017 \
  --data.batch_size 128 --data.num_workers 32 --data.prefetch_factor 1 \
  --model.lr 0.02 --trainer.max_epochs 50 --trainer.precision 16-mixed \
  --trainer.default_root_dir /content/drive/MyDrive/lucid_runs

lucid-eval --checkpoint <checkpoint> --data_root /content/coco2017 --output det_report.json
```

`lr 0.02` against the file's 0.01 is the linear scaling for batch 128: the config carries the batch-64 recipe value, and the unscaled one has no clean measurement of its own on this codebase.

## 🖌️ Instance segmentation — COCO 2017

The detection recipe with `task: segment`; the same data root serves both, so a tree already provisioned above needs nothing further.

```bash
lucid-yolo fit --config seg_nano_smoke.yaml \
  --data.data_root /content/coco2017 \
  --data.batch_size 128 --data.num_workers 32 --data.prefetch_factor 2 \
  --model.lr 0.02 --trainer.max_epochs 50 --trainer.precision bf16-mixed \
  --trainer.default_root_dir /content/drive/MyDrive/lucid_runs

lucid-eval --checkpoint <checkpoint> --data_root /content/coco2017 --output seg_report.json
```

`lucid-eval` reads the task off the checkpoint and scores masks when the checkpoint has them, so the same command serves both COCO tiers. `--masks false` scores boxes only from a segmentation checkpoint.

## 🕺 Keypoints — COCO 2017 `person_keypoints`

The same COCO tree again, read through its other annotation file. `num_keypoints` is required and has no default — the point count is a property of the annotation schema, not something a model can pick.

```bash
lucid-yolo fit --config pose_nano_smoke.yaml \
  --data.data_root /content/coco2017 \
  --data.batch_size 128 --data.num_workers 32 --data.prefetch_factor 1 \
  --model.lr 0.02 --trainer.max_epochs 50 --trainer.precision bf16-mixed \
  --trainer.default_root_dir /content/drive/MyDrive/lucid_runs

lucid-eval --checkpoint <checkpoint> --data_root /content/coco2017 --output pose_report.json
```

The config states `train_ann_file`/`val_ann_file` explicitly: `lucid_yolo.data.layout` resolves only the `instances_` spelling, and `person_keypoints_{split}2017.json` sits beside it in the same annotations archive under a different name. `lucid-eval` dispatches on the checkpoint's own task here as it does for the other three, reading the single-category person file and reporting OKS beside box AP — a keypoints checkpoint scored through the detection protocol reports roughly one eightieth of its true figure, since that protocol averages over 80 categories of which 79 have no prediction.

**The task is `K`-generic; the 17-point COCO schema is one instantiation.** `num_keypoints: 17` selects R12's own sigma table for OKS. Any other `K` is trainable, but `lucid-eval`'s pose protocol refuses it rather than scoring it against a schema it does not share — the ground truth, point ordering and sigmas are all the person schema's.

Swap `keypoint_loss` to run the flow-free control instead: `pose_nano_smoke_laplace_nll.yaml` is the same file with that one line changed, and is what the accepted tier paired its RLE arm against.

## 🔄 Oriented detection — DOTA-v1.0 tiles

DOTA is provisioned by hand and then tiled; both steps are docs/DATASETS.md. Training reads the tiles, never the original tree.

```bash
lucid-data check       --data_root /content/dota --dataset dota
lucid-data build-tiles --root /content/dota --out /content/dota_tiles --splits train,val --overlap 512 --workers 8

lucid-yolo fit --config obb_nano_smoke.yaml \
  --data.data_root /content/dota_tiles \
  --data.batch_size 16 --data.num_workers 8 \
  --trainer.max_epochs 50 --trainer.precision bf16-mixed \
  --trainer.default_root_dir /content/drive/MyDrive/lucid_runs

lucid-eval --checkpoint <checkpoint> --data_root /content/dota_tiles --split val --output obb_report.json
```

Two differences from the COCO tiers, forced by the data rather than chosen:

- **Worker counts are memory, not just parallelism, at 1024 px.** A batch of 64 stacked 1024 px images is 805 MB, and every worker holds `prefetch_factor` of them in shared memory — so `--data.num_workers 32` queues about 51 GB per loader, and the validation pool spawns while the training one is still resident. A count named for training is bounded to what fits before validation inherits it, with a warning naming the number (WP-103); `--data.val_num_workers` overrides that bound outright when the machine has the room.
- **Batch 16 at `img_size: 1024`**, against 128 at 640 for COCO. The tile side is R18's own crop size, and 16 is the placeholder `obb_nano_smoke.yaml` ships — "tune to accelerator memory". No oriented tier run has measured a batch that fits, so it is a starting point rather than a figure: at 2.56x the pixels per image it is a third of the COCO batch's pixel budget, not a match for it. `--model.lr` stays at the config's 0.01 for the same reason — the 0.02 above is the batch-128 linear scaling, and it does not carry over to a batch nobody has settled yet.

`lucid-eval` picks the rotated protocol from the checkpoint's own task, and with it the 1024 px letterbox and batch 8; an explicit `--img_size` or `--batch_size` still wins.

## 🖥️ Oriented and keypoint runs are single-device

`lucid-yolo fit` refuses `task: obb` and `task: keypoints` on more than one process, and says so at setup rather than partway through the first validation. Their epoch metrics — `val/rotated_mAP` and `val/oks_mAP` — accumulate plain Python lists that nothing gathers, so each rank would score its own shard and log it as the split's number. Average precision ranks every detection of the split against every other by confidence, so per-rank values cannot be averaged back into the right one; a wrong number that looks plausible is worse than a refusal. Leave `trainer.devices` at `1` for these two tasks, or pick a bigger accelerator rather than more of them. Detection and segmentation are unaffected: their metrics are torchmetrics metrics and synchronise across ranks. Whatever the task, a CUDA run trains in `channels_last` memory format automatically — parameters and images both, no flag (D23) — while CPU and MPS keep the default layout.

## ⚡ torch.compile is opt-in (CUDA, detect)

`--model.compile_step true` (or `compile_step: true` under `model:`; `det_nano_smoke.yaml` carries it at `false`) compiles the detection objective with Inductor. What is compiled is one region, `DetectionLitModule._detect_objective`: the forward, both branch decodes, the task-aligned assignment and the dual loss, traced as a single graph with no breaks — `tests/ptl/test_compile.py` pins that on CPU for a batch with ground truths and for one with none. Compiling the model half alone was measured first and was not worth a flag: x1.09–1.16 on the step, because the assignment and the loss are where the small kernels and the launch overhead live. With WP-183's dense box terms (no data-dependent gather) and the label check skipped under tracing, the whole region compiles: on an RTX PRO 6000, variant `n`, COCO, batch 16, deterministic, the eager training step is 74.7 ms and the compiled one 60.9 ms, x1.23. A bf16 run was recorded at x1.46 on the same bench and is **not** enabled by this flag — precision is a separate decision the fidelity gate has not been asked about.

What it costs: 110–140 s of compilation per process before the first step, and a handful of recompiles over a run: the training graph, the first `no_grad` evaluation step, and — because Dynamo specialises sizes 0 and 1 whatever is marked — one each for the first batch with no ground truths (`tal.py`'s own branch), the first with exactly one padded instance, and the first with one image, if the loader ever produces it. Measured on CPU with the `eager` backend over five batch shapes: four graphs. Batch size and padded instance count are marked dynamic on the images and the ground truths before every compiled call, so the last, smaller batch of an epoch and a batch with more instances than the one before share one graph; without the marks Dynamo would specialise until `cache_size_limit` (8) and then drop to eager without saying so, and with them the assigner's `F.one_hot(best_gt, num_gt)` — a concrete class count — had to become an `arange` comparison, since the enforcing marks refuse a pinned dim rather than silently recompiling. `TORCH_LOGS=recompiles` shows the count.

Where it refuses, it says so once at `setup` and trains eager. Off CUDA: the MPS Inductor backend crashes in `convolution_backward`, and CPU has nothing to gain. Under `task: segment`, `obb` or `keypoints`: their extra terms read tensors on the host inside `_task_extra_loss`, outside the region, so compiling the detection part alone would buy a fraction of the x1.23 for the same compile cost; they stay eager until their terms are made traceable. The compile wraps the bound method, not the module, so parameter names, the state dict, the EMA shadow, checkpoints and export are exactly what they are without the flag — a compiled *module* would prefix every key with `_orig_mod.`.

The default is off on cost, not on doubt. A78 — whether the compiled objective trains as eager does — was validated on an A100 (paired 4-epoch run: 1 recompile, step 180.6 → 144.2 ms, loss parity `3.6e-5` with TF32 off), but compilation added 210 s of wall to a 4-epoch run of 313 steps, a break-even near 19 epochs at that size and under one epoch on COCO train; turn it on per recipe when the run is long enough. When comparing the two arms, compare mAP, or set `torch.set_float32_matmul_precision("highest")` first: under TF32 the per-batch `train/loss` differs by up to `1e-1`, because the task-aligned assignment is discrete and TF32 rounding moves its top-k set. D24 carries the numbers.

## 🎲 What `deterministic: true` and `seed` actually pin

Every tier config sets `deterministic: true` and seed 0. Two boundaries sit inside that, and neither fails loudly.

**On MPS it is not strict.** `default_determinism` (`lucid_yolo/cli/train.py`) resolves the trainer's `deterministic` flag to `"warn_only"` whenever MPS is the auto-picked accelerator, because MPS ships no deterministic implementation for some backward kernels this model reaches — `index_put_with_accumulate`, hit by the assignment and loss backward — and strict `True` aborts mid-step there. Every op with a deterministic kernel stays deterministic; the rest downgrade to a warning. CPU and CUDA keep strict `True`. So an Apple-silicon run under D12c is reproducible only up to those kernels, and says so in the log rather than by failing.

**The worker count is part of the seed.** A seeded run is byte-identical only against another run at the *same* `--data.num_workers`. At `0` the pipeline draws every augmentation parameter from one generator in the parent process, in index order. With workers each is re-seeded per worker and per epoch from the loader's own generator (WP-079), which is what keeps per-epoch diversity alive. Both streams are fully determined by `seed` and they are **different streams**: the recipes above run at 32 workers, and rerunning one at 0 workers, or at 16, reproduces the schedule and not the sample-by-sample augmentation. Reproducing a published run means matching its worker count as well as its seed, which is why every command above states one.

## 🔐 What `--checkpoint` will deserialize

Every command above reads its checkpoint through one loader, and that loader reads the file twice in a fixed order. The first read is restricted — `torch.load(..., weights_only=True)` — and refuses anything the file names outside torch's allowlist while the pickle stream is still being parsed, so a checkpoint carrying an arbitrary reducer is rejected before that reducer runs. Only a file that survives that read is handed to Lightning.

The order is what makes this a policy rather than a hope. `weights_only` reached Lightning's `load_from_checkpoint` in 2.6.0, and this project's floor is `pytorch-lightning>=2.4`; below 2.6 the argument is absorbed as a hyper-parameter override and changes nothing about how the file is read. So the restriction cannot be stated at that call on every supported install — but it can be stated at the first read, on all of them.

Two consequences before pointing `--checkpoint` at a file someone else produced. A checkpoint whose `hyper_parameters` or callback state carries a non-tensor object is refused rather than loaded, which is the same refusal a hostile file gets: the reader cannot tell the two apart, and does not guess. And a file swapped between the two reads is checked in its first form and loaded in its second — the gate is a check on the file, not a lock on it.

## 📁 A YOLO-format root instead of COCO's

Every tier above works unchanged against a YOLO-format root — no dedicated config, just three overrides on `det_nano_smoke.yaml`:

```bash
lucid-data check --data_root /path/to/yolo_dataset   # validates the tree, prints the class count

lucid-yolo fit --config det_nano_smoke.yaml \
  --data.data_root /path/to/yolo_dataset --data.layout yolo \
  --model.num_classes <len(names) in the root's own data.yaml>
```

`--data.layout yolo` is stated rather than left to be probed: `detect_layout` resolves a bare YOLO root on its own, but stating it also covers a root that happens to satisfy both conventions (COCO and YOLO — the probe refuses to break that tie, A63) and skips the probe entirely, so a missing path is named by the reader that actually needed it, not the probe. `--model.num_classes` must equal `len(names)` in the root's own `data.yaml` — a label row's class index is 0-based into that list and is never remapped (WP-099), so a mismatched count trains against the wrong label space silently. `lucid-data check --data_root ...` prints both numbers so they can be compared before a launch, not after one.

**Not available on this layout: `--model.task segment`.** A YOLO label row carries five or nine normalized numbers and never a polygon ring, so there is nothing to rasterize a mask from — `DetectionDataModule` refuses `mask_targets=True` on a YOLO root rather than training a segmentation head against empty masks (WP-099b). Full format/layout mechanics: docs/DATASETS.md.

## 🔌 Before any of them: the wiring gate

Each tier has a minutes-long overfit gate that must pass before a launch that costs hours. It is development tooling and ships in `scripts/`, not in the wheel, so it runs from a checkout:

```bash
python scripts/overfit_micro.py --task detect     # one-to-one train recall >= 0.95
python scripts/overfit_micro.py --task segment    # train mask IoU >= 0.7
python scripts/overfit_micro.py --task obb        # train rotated mAP50 >= 0.9
python scripts/overfit_micro.py --task keypoints  # train OKS AP >= 0.30
```

A tier that cannot overfit a handful of images will not converge on the full set, and finding that out after the first epoch of a fifty-epoch run costs the run.

The keypoint floor looks low beside the other three and is not measuring less. Its slice is the only one not drawn from the geometric shapes — `task: keypoints` needs a keypoint-bearing family, so it draws a 7-point synthetic symbol schema, and OKS at that schema's uniform sigma is a cliff on objects a few dozen pixels across: feeding the ground truth back as the prediction scores exactly 1.0, and displacing every point by 3 px scores 0.269. The floor was set from that measured slope rather than guessed.
