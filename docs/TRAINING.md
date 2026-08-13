# Launching a training run

Three tiers, one command. `lucid-yolo fit --config <name>.yaml` resolves the name against the configs packaged inside the wheel, so a run needs no checkout — `det_smoke.yaml`, `seg_smoke.yaml` and `obb_smoke.yaml` are all installed with the package.

What the configs carry is run-level configuration only: schedule, optimizer and loss gains, and placeholder data paths. Topology is named by `variant` and lives in the registry, never in a config file (ADR-001). Every value below that overrides a config is an override *of a placeholder or of a batch-size-dependent value*, not a correction.

`--data.data_root` is the only path any tier needs. The split directories and their annotation files are resolved from it by convention — COCO 2017's `train2017` spelling, the plain `train` the tiler writes, an `images/<split>` tree, or a per-split `_annotations.coco.json` export (`lucid_yolo.data.layout`). A layout none of those names still takes the four explicit overrides.

Provisioning the datasets these recipes read: docs/DATASETS.md. What the accepted runs actually measured, with the exact commands as they were run at the time: docs/REPRODUCTION_REPORT.md.

## Always: point `default_root_dir` at storage that outlives the runtime

`--trainer.default_root_dir` is the `save_dir` of both default loggers (TensorBoard and CSV, pinned to a single `lightning_logs/version_N` directory) and the directory checkpoints land in. On a hosted runtime it is the whole of what survives a disconnect.

This is not a precaution in the abstract. 0.2.0's Seg-smoke ran with `default_root_dir` under `/content/`, and its model card records the hardware line as unrecorded because the logs went with the runtime.

```python
from google.colab import drive

drive.mount("/content/drive")
```

Then `--trainer.default_root_dir /content/drive/MyDrive/lucid_runs` on every command below. The **dataset** goes the other way: it stays on the runtime's local disk, because every epoch reads every image and Drive is a network filesystem mounted through FUSE.

## Detection — COCO 2017

```bash
lucid-data download --data_root /content/coco2017 --splits '[train,val]' --verify true
lucid-data check    --data_root /content/coco2017

lucid-yolo fit --config det_smoke.yaml \
  --data.data_root /content/coco2017 \
  --data.batch_size 128 --data.num_workers 32 --data.prefetch_factor 1 \
  --model.lr 0.02 --trainer.max_epochs 50 --trainer.precision 16-mixed \
  --trainer.default_root_dir /content/drive/MyDrive/lucid_runs

lucid-eval --checkpoint <checkpoint> --data_root /content/coco2017 --output det_report.json
```

`lr 0.02` against the file's 0.01 is the linear scaling for batch 128: the config carries the batch-64 recipe value, and the unscaled one has no clean measurement of its own on this codebase.

## Instance segmentation — COCO 2017

The detection recipe with `task: segment`; the same data root serves both, so a tree already provisioned above needs nothing further.

```bash
lucid-yolo fit --config seg_smoke.yaml \
  --data.data_root /content/coco2017 \
  --data.batch_size 128 --data.num_workers 32 --data.prefetch_factor 2 \
  --model.lr 0.02 --trainer.max_epochs 50 --trainer.precision bf16-mixed \
  --trainer.default_root_dir /content/drive/MyDrive/lucid_runs

lucid-eval --checkpoint <checkpoint> --data_root /content/coco2017 --output seg_report.json
```

`lucid-eval` reads the task off the checkpoint and scores masks when the checkpoint has them, so the same command serves both COCO tiers. `--masks false` scores boxes only from a segmentation checkpoint.

## Oriented detection — DOTA-v1.0 tiles

DOTA is provisioned by hand and then tiled; both steps are docs/DATASETS.md. Training reads the tiles, never the original tree.

```bash
lucid-data check       --data_root /content/dota --dataset dota
lucid-data build-tiles --root /content/dota --out /content/dota_tiles --splits train,val --overlap 512 --workers 8

lucid-yolo fit --config obb_smoke.yaml \
  --data.data_root /content/dota_tiles \
  --data.batch_size 16 --data.num_workers 8 \
  --trainer.max_epochs 50 --trainer.precision bf16-mixed \
  --trainer.default_root_dir /content/drive/MyDrive/lucid_runs

lucid-eval --checkpoint <checkpoint> --data_root /content/dota_tiles --split val --output obb_report.json
```

One difference from the COCO tiers, forced by the data rather than chosen:

- **Batch 16 at `img_size: 1024`**, against 128 at 640 for COCO. The tile side is R18's own crop size, and 16 is the placeholder `obb_smoke.yaml` ships — "tune to accelerator memory". No oriented tier run has measured a batch that fits, so it is a starting point rather than a figure: at 2.56x the pixels per image it is a third of the COCO batch's pixel budget, not a match for it. `--model.lr` stays at the config's 0.01 for the same reason — the 0.02 above is the batch-128 linear scaling, and it does not carry over to a batch nobody has settled yet.

`lucid-eval` picks the rotated protocol from the checkpoint's own task, and with it the 1024 px letterbox and batch 8; an explicit `--img_size` or `--batch_size` still wins.

## Before any of them: the wiring gate

Each tier has a minutes-long overfit gate that must pass before a launch that costs hours. It is development tooling and ships in `scripts/`, not in the wheel, so it runs from a checkout:

```bash
python scripts/overfit_micro.py --task detect     # one-to-one train recall >= 0.95
python scripts/overfit_micro.py --task segment    # train mask IoU >= 0.7
python scripts/overfit_micro.py --task obb        # train rotated mAP50 >= 0.9
```

A tier that cannot overfit a handful of images will not converge on the full set, and finding that out after the first epoch of a fifty-epoch run costs the run.
