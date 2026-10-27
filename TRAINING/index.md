# 🏋️ Launching a training run

Four tiers, one command. `lucid-yolo fit --config <name>.yaml` resolves the name against the configs packaged inside the installed package, so a run needs no checkout once lucid-yolo is installed (the [Install section of the README](https://github.com/Borda/lucid-YOLO#install) has the one-line command) — `det_nano_smoke.yaml`, `seg_nano_smoke.yaml`, `obb_nano_smoke.yaml` and `pose_nano_smoke.yaml` are all installed with the package.

What the configs carry is run-level configuration only: schedule, optimizer and loss gains, and placeholder data paths. Topology is named by `variant` and lives in the registry, never in a config file (ADR-001). Every value below that overrides a config is an override *of a placeholder or of a batch-size-dependent value*, not a correction.

`--data.data_root` is the only path any tier needs. The split directories and their annotation files are resolved from it by convention — COCO 2017's `train2017` spelling, the plain `train` the tiler writes, an `images/<split>` tree, or a per-split `_annotations.coco.json` export (`lucid_yolo.data.layout`). A layout none of those names still takes the four explicit overrides.

Provisioning the datasets these recipes read: docs/DATASETS.md. What the accepted runs actually measured, with the exact commands as they were run at the time: docs/REPRODUCTION_REPORT.md.

## Opt-in CUDA performance flags

Opt-in performance flags — colour augmentation on the device, epoch prefetch, the image caches, CUDA graph replay, `torch.compile` — live on their own page: [Opt-in CUDA performance flags](https://borda.github.io/lucid-YOLO/advanced/cuda-performance/). Most need a CUDA device and belong to a roadmap row still open. Two exceptions: the host-side image cache `--data.cache_train_images` is built in the data module and needs no accelerator, and its row ([WP-192b](https://borda.github.io/lucid-YOLO/work_packages/wp-192b/)) and the `--model.compile_step` row (WP-184) are done. None of them is needed for the recipes below, and each defaults to off or to the ordinary path.

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

`lucid-eval` reads the task off the checkpoint and scores masks when the checkpoint has them, so the same command serves both COCO tiers. `--masks false` scores boxes only from a segmentation checkpoint. For the COCO tiers (detect, segment, keypoints) `--ann_file` and `--images_dir` override the annotation JSON and image directory independently; the defaults stay `annotations/instances_val2017.json` (`person_keypoints_val2017.json` for keypoints) and `val2017/` under `--data_root`. An option the checkpoint's task does not use (`--split`, `--variant` for a non-oriented checkpoint; `--masks` other than for segment; `--eval_backend`, `--ann_file`, `--images_dir` for an oriented one) is refused by name when passed explicitly.

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

- **Worker counts are memory, not just parallelism, at 1024 px.** A batch of 64 stacked 1024 px images is 805 MB, and every worker holds `prefetch_factor` of them in shared memory — so `--data.num_workers 32` queues about 51 GB per loader, and the validation pool spawns while the training one is still resident. A count named for training is bounded to what fits before validation inherits it, with a warning naming the number ([WP-103](https://borda.github.io/lucid-YOLO/work_packages/wp-103/)); `--data.val_num_workers` overrides that bound outright when the machine has the room.
- **Batch 16 at `img_size: 1024`**, against 128 at 640 for COCO. The tile side is R18's own crop size, and 16 is the placeholder `obb_nano_smoke.yaml` ships — "tune to accelerator memory". No oriented tier run has measured a batch that fits, so it is a starting point rather than a figure: at 2.56x the pixels per image it is a third of the COCO batch's pixel budget, not a match for it. `--model.lr` stays at the config's 0.01 for the same reason — the 0.02 above is the batch-128 linear scaling, and it does not carry over to a batch nobody has settled yet.

`lucid-eval` picks the rotated protocol from the checkpoint's own task, and with it the 1024 px letterbox and batch 8; an explicit `--img_size` or `--batch_size` still wins.

## 🖥️ Distributed validation

All four tasks accept DDP validation. Detection and segmentation use torchmetrics; oriented detection and keypoints gather whole-image records with private torchmetrics metrics before calling the existing rotated mAP and OKS AP scorers. The validation loader carries dataset-position IDs so repeated sampler-padding images are scored once, keeping the first occurrence and the loader's original dataset order. Custom validation loaders for oriented detection or keypoints must retain the datamodule's private indexed targets; a distributed batch without identity is refused rather than silently double-counted. Public sample and batch signatures are unchanged ([WP-213](https://borda.github.io/lucid-YOLO/work_packages/wp-213/)).

The multi-device proof is scoped to a CPU gloo two-process test; no two-GPU run was made, and a real multi-GPU check remains outstanding. Training recipes remain measured on one device. Whatever the task, CUDA training uses `channels_last` memory format automatically for parameters and images (D23); CPU and MPS retain the default layout.

## 🎲 What `deterministic: true` and `seed` actually pin

Every tier config sets `deterministic: true` and seed 0. Two boundaries sit inside that, and neither fails loudly.

**On MPS it is not strict.** `default_determinism` (`lucid_yolo/cli/train.py`) resolves the trainer's `deterministic` flag to `"warn_only"` whenever MPS is the auto-picked accelerator, because MPS ships no deterministic implementation for some backward kernels this model reaches — `index_put_with_accumulate`, hit by the assignment and loss backward — and strict `True` aborts mid-step there. Every op with a deterministic kernel stays deterministic; the rest downgrade to a warning. CPU and CUDA keep strict `True`. So an Apple-silicon run under D12c is reproducible only up to those kernels, and says so in the log rather than by failing.

**The worker count is part of the seed.** A seeded run is byte-identical only against another run at the *same* `--data.num_workers`. At `0` the pipeline draws every augmentation parameter from one generator in the parent process, in index order. With workers each is re-seeded per worker and per epoch from the loader's own generator ([WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/)), which is what keeps per-epoch diversity alive. Both streams are fully determined by `seed` and they are **different streams**: the recipes above run at 32 workers, and rerunning one at 0 workers, or at 16, reproduces the schedule and not the sample-by-sample augmentation. Reproducing a published run means matching its worker count as well as its seed, which is why every command above states one.

## ⏯️ Resuming an interrupted run

Add `--ckpt_path` to the command that started the run and point it at a checkpoint the run wrote. With the default loggers that is `<default_root_dir>/lightning_logs/version_N/checkpoints/` (`N` numbers the run; the first run is `version_0`). With `logger: false` there is no version directory and Lightning writes to `<default_root_dir>/checkpoints/`:

```bash
lucid-yolo fit --config det_nano_smoke.yaml --data.data_root /data/coco \
  --trainer.default_root_dir /content/drive/MyDrive/lucid_runs \
  --ckpt_path /content/drive/MyDrive/lucid_runs/lightning_logs/version_0/checkpoints/<file>.ckpt
```

The resumed run continues the step and epoch counters, the EMA shadow and its update count, the MuSGD momentum, and the progressive-loss weight. That weight is a function of the current and final epoch and holds no state of its own, so restoring the epoch counter is what restores it. Keep `--trainer.max_epochs` equal to the original run's: the weight and the learning-rate schedule are both functions of it. Augmentation random-number state is not checkpointed, so a resumed run is not bit-identical to an uninterrupted one. The command-line path is covered by `tests/cli/test_resume_cli.py`, which asserts each of the four on a CPU smoke run; `tests/ptl/test_resume.py` covers trajectory equality on a fixed stream without augmentation.

## 🔐 What `--checkpoint` will deserialize

`lucid-eval`, `lucid-predict`, `lucid-export` and `lucid_yolo.load_eval_module` read their checkpoint through one loader, and that loader reads the file once, restricted. `fit --ckpt_path` (resuming) does not go through it: it is Lightning's own load, outside the guarantee below. The read is `torch.load(..., weights_only=True)`, which refuses anything the file names outside torch's allowlist while the pickle stream is still being parsed, so a checkpoint carrying an arbitrary reducer is rejected before that reducer runs. The module is then built from that same payload: its saved `hyper_parameters`, filtered to the arguments `DetectionLitModule` takes, construct it, and Lightning's public `on_load_checkpoint` hook and a `load_state_dict` follow. Lightning's own loader, which would read the file a second time without the restriction, is never called.

Earlier releases read the file twice, restricted first as a gate and then through Lightning, because the restriction only reached Lightning's loader in 2.6.0 and this project's floor is `pytorch-lightning>=2.4`. The single read states the restriction on every supported install, and closes the gap in which a file swapped between the two reads was checked in its first form and loaded in its second. What it skips is Lightning's migration of legacy checkpoints, which only rewrites loop and callback state from Lightning 1.x layouts, none of which this project reads.

Two consequences before pointing `--checkpoint` at a file someone else produced. A checkpoint whose `hyper_parameters` or callback state carries a non-tensor object is refused rather than loaded, which is the same refusal a hostile file gets: the reader cannot tell the two apart, and does not guess. And a checkpoint with no `hyper_parameters` or `state_dict` mapping is refused with a `ValueError` naming the missing key and the file.

## 🐍 Using a checkpoint from Python

The loader, the four predict functions and the COCO scorer are importable from the top-level package without loading torch until one is called ([Inference API](https://borda.github.io/lucid-YOLO/api/inference/)):

```python
from pathlib import Path

from torchvision.io import decode_image

import lucid_yolo

module, info = lucid_yolo.load_eval_module(Path("runs/det.ckpt"), use_ema=True)  # EMA weights, eval mode
device = lucid_yolo.pick_device("auto")
image = Path("street.jpg")
boxes = lucid_yolo.predict_image(module, image, device=device)  # (N, 6): x1 y1 x2 y2 score class
same = lucid_yolo.predict_image(module, decode_image(str(image)), device=device)  # CHW uint8 tensor in
score = lucid_yolo.score_coco(module, Path("~/data/coco2017").expanduser(), limit=500)
print(score.report["e2e"]["map"], score.report["nms"]["map"], score.images, score.seconds)
```

- **Image input** is a file path or a decoded `(3, H, W)` RGB tensor holding `uint8` or `uint16` codes, or floats in `[0, 1]`. One image per call: a batch is not accepted. An `(H, W, 3)` array, a float image outside `[0, 1]` and signed, bool or complex codes are refused with a message saying what to change.
- **`predict_segmentation`, `predict_oriented` and `predict_keypoints`** take the same two forms. A checkpoint of another task is refused by the function whose task it is not.
- **`score_coco`** picks the protocol from `module.task` (`detect`, `segment` or `keypoints`), prints nothing and writes nothing; it returns a `CocoScore` with the report, the image count and the scoring time. It refuses `obb` checkpoints: those are scored by `lucid-eval` or `lucid_yolo.eval.rotated_eval`. The dataset layout it expects is the one in [Provisioning the datasets](https://borda.github.io/lucid-YOLO/DATASETS/).
- **The scoring engine differs from the command's.** `score_coco` accepts only `eval_backend="faster_coco_eval"` (the default) or `"hotcoco"`; any other value, `"auto"` included, is refused. `lucid-eval` defaults to `auto`, which uses `hotcoco` when it is usable and falls back to `faster_coco_eval` otherwise. To match the command, pass `eval_backend="hotcoco"` explicitly; `lucid-eval` reports a `RuntimeError` for an explicit `hotcoco` that is not usable, and installing the `hotcoco` package is a precondition.

## 📦 Exporting to ONNX

`lucid-export` writes one checkpoint as one ONNX file. The task is read from the checkpoint, as `lucid-eval` and `lucid-predict` read it, and the graph is the suppression-free one-to-one path that task deploys on: a top-k node and no `NonMaxSuppression` node. Writing the file needs the `onnx` package, which torch's exporter imports to finalise it, so install the extra first:

```bash
pip install "lucid-yolo[export]"
lucid-export --checkpoint runs/det.ckpt --output det.onnx
```

Without the extra the command stops before reading the checkpoint, with `lucid-export needs the onnx package to write the file: pip install "lucid-yolo[export]"`. A run on a nano detection checkpoint at `--img_size 128` printed:

```text
export: runs/det.ckpt -> det.onnx (task=detect, ema=True, epoch=0, opset=17)
input: image (1, 3, 128, 128) float32, static shape
output: (1, 300, 6) detections [x1, y1, x2, y2, score, class]
```

| Flag | Meaning |
| -- | -- |
| `--checkpoint`, `--output` | Required. The parent directory of `--output` is created if absent. |
| `--img_size N` | Square input side. Defaults per task: 640 for detection, segmentation and keypoints, 1024 for oriented. |
| `--canvas "[H, W]"` | Non-square input, height first, written as one JSON list: `--canvas 640 480` does not parse. Exclusive with `--img_size`. |
| `--ema true\|false` | Export the EMA shadow (default) or the raw weights. A checkpoint without an EMA callback entry needs `--ema false`. |
| `--opset N` | ONNX opset, 17 by default, the only one the export gate certifies. |
| `--config` | Every command takes one. |

**What the file answers with.** Every graph returns 300 rows per image, best first; rows no anchor produced have score 0. Detection returns `(1, 300, 6)` as `[x1, y1, x2, y2, score, class]`. Segmentation adds `(1, 300, H, W)` boolean masks on the letterboxed canvas. Oriented returns `(1, 300, 7)` as `[cx, cy, w, h, theta, score, class]`. Keypoints adds `(1, 300, K, 2)` points on the letterboxed canvas; the per-point sigma is not exported. Coordinates are on the letterboxed canvas: the caller letterboxes the image, and maps results back through the inverse letterbox itself.

**Limits.**

- The input shape is static, batch included. A second canvas or batch size is a second export; there is no dynamic batch.
- Opset 17 through torch's legacy TorchScript exporter (`dynamo=False`), which torch marks deprecated. Only the suppression-free path is exported: no NMS graph, no other format, no precision or quantization.
- A segmentation export refuses canvases that yield fewer than 300 anchors, which is 96 px a side and below. The oriented and detection graphs pad to 300 rows there, and the keypoint graph pads with origin points.
- The oriented graph on such a small canvas now returns 300 rows where it returned one row per anchor before; every row an anchor produced is unchanged, and canvases of 128 px and above are unchanged. `lucid-predict` output is unchanged on every canvas, since its confidence filter drops the padding rows.

From Python, `lucid_yolo.export.graph_from_module(module, (640, 640))` rebuilds the graph from a loaded checkpoint under a strict key check and `export_graph(graph, "m.onnx")` writes it ([Export API](https://borda.github.io/lucid-YOLO/api/export/)). It accepts any module built at raw multipliers, not only registry variants.

## 📁 A YOLO-format root instead of COCO's

Every tier above works unchanged against a YOLO-format root — no dedicated config, just three overrides on `det_nano_smoke.yaml`:

```bash
lucid-data check --data_root /path/to/yolo_dataset   # validates the tree, prints the class count

lucid-yolo fit --config det_nano_smoke.yaml \
  --data.data_root /path/to/yolo_dataset --data.layout yolo \
  --model.num_classes <len(names) in the root's own data.yaml>
```

`--data.layout yolo` is stated rather than left to be probed: `detect_layout` resolves a bare YOLO root on its own, but stating it also covers a root that happens to satisfy both conventions (COCO and YOLO — the probe refuses to break that tie, A63) and skips the probe entirely, so a missing path is named by the reader that actually needed it, not the probe. `--model.num_classes` must equal `len(names)` in the root's own `data.yaml` — a label row's class index is 0-based into that list and is never remapped ([WP-099](https://borda.github.io/lucid-YOLO/work_packages/wp-099/)), so a mismatched count trains against the wrong label space silently. `lucid-data check --data_root ...` prints both numbers so they can be compared before a launch, not after one.

**Not available on this layout: `--model.task segment`.** A YOLO label row carries five or nine normalized numbers and never a polygon ring, so there is nothing to rasterize a mask from — `DetectionDataModule` refuses `mask_targets=True` on a YOLO root rather than training a segmentation head against empty masks ([WP-099b](https://borda.github.io/lucid-YOLO/work_packages/wp-099b/)). Full format/layout mechanics: docs/DATASETS.md.

## 🔌 Before any of them: the wiring gate

Each tier has a minutes-long overfit gate that must pass before a launch that costs hours. It is development tooling and ships in `scripts/`, not in the wheel, so it runs from a checkout:

```bash
python scripts/overfit_micro.py --task detect     # one-to-one train recall >= 0.95
python scripts/overfit_micro.py --task segment    # train mask IoU >= 0.7
python scripts/overfit_micro.py --task obb        # train rotated mAP50 >= 0.9
python scripts/overfit_micro.py --task keypoints  # train OKS AP >= 0.30
```

A tier that cannot overfit a handful of images will not converge on the full set, and finding that out after the first epoch of a fifty-epoch run costs the run. The four commands, and the shapes regression beside them, run as they are in the [development gates notebook](https://borda.github.io/lucid-YOLO/notebooks/development_gates/).

The keypoint floor looks low beside the other three and is not measuring less. Its slice is the only one not drawn from the geometric shapes — `task: keypoints` needs a keypoint-bearing family, so it draws a 7-point synthetic symbol schema, and OKS at that schema's uniform sigma is a cliff on objects a few dozen pixels across: feeding the ground truth back as the prediction scores exactly 1.0, and displacing every point by 3 px scores 0.269. The floor was set from that measured slope rather than guessed.
