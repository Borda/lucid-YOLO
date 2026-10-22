# 🏋️ Launching a training run

Four tiers, one command. `lucid-yolo fit --config <name>.yaml` resolves the name against the configs packaged inside the installed package, so a run needs no checkout once lucid-yolo is installed (the [Install section of the README](https://github.com/Borda/lucid-YOLO#install) has the one-line command) — `det_nano_smoke.yaml`, `seg_nano_smoke.yaml`, `obb_nano_smoke.yaml` and `pose_nano_smoke.yaml` are all installed with the package.

What the configs carry is run-level configuration only: schedule, optimizer and loss gains, and placeholder data paths. Topology is named by `variant` and lives in the registry, never in a config file (ADR-001). Every value below that overrides a config is an override *of a placeholder or of a batch-size-dependent value*, not a correction.

`--data.data_root` is the only path any tier needs. The split directories and their annotation files are resolved from it by convention — COCO 2017's `train2017` spelling, the plain `train` the tiler writes, an `images/<split>` tree, or a per-split `_annotations.coco.json` export (`lucid_yolo.data.layout`). A layout none of those names still takes the four explicit overrides.

Provisioning the datasets these recipes read: docs/DATASETS.md. What the accepted runs actually measured, with the exact commands as they were run at the time: docs/REPRODUCTION_REPORT.md.

## CUDA colour augmentation

`--data.device_hsv true` moves training HSV and quantization to CUDA with `32-true`, `16-mixed` or `bf16-mixed` precision. The workers still sample the same gains and horizontal flips, and validation keeps its existing path. The default is `false`; other backends and precisions warn once and keep worker-side HSV. Any precision plugin whose input conversion casts the float32 images or float64 gains before the data hook (true-half and double, and the FSDP and DeepSpeed plugins under every accepted precision) keeps the original path as well, with the same one-time warning.

On Linux NVIDIA GPUs with compute capability 8.0 or newer, the float32 consumer restore can fuse the existing colour operations and quantization into one channels-last write. This is on by default wherever Triton and a host C compiler are present — torch's CUDA wheel already selects a Triton, so an existing `device_hsv` run there takes the fused path without installing anything (`uv pip install -e ".[cuda-hsv]"` adds Triton where it is missing). `--data.fused_hsv false` keeps `device_hsv` on the native CUDA restore, to step around a toolchain or bisect a difference between the two. The first call compiles the kernel and needs a host C compiler. The extra also installs the existing Setuptools build dependency needed by older Triton SDKs. A Triton `ImportError`, a missing compiler, other devices, noncontiguous buffers and other consumer precisions retain the native restore. So does a first compile or launch that fails: it warns once with its cause, and the native restore runs for the rest of the process. Gain sampling, rounding and the uint8 boundary are unchanged. Direct `unpack_batch` calls keep their existing path.

This can help a CPU-limited training run, including one with decoded images cached by `--data.cache_train_images true`. It transports float32 images until the colour transform finishes, so image IPC and host-to-device bytes are four times the default uint8 transport. The shared-memory worker cap budgets float32 images only when the module chooses the worker count (`--data.num_workers` unset) in one process; segmentation masks still need their own memory allowance. A named `--data.num_workers` is neither capped nor warned: batch 64 at 640 px with 8 workers and `prefetch_factor` 2 queues about 5.0 GB of `/dev/shm` in float32, against 1.26 GB in uint8. Each DDP rank budgets the free `/dev/shm` on its own, so ranks sharing one host can jointly overcommit it — a known limit. The cap cannot shrink a batch: with a small `/dev/shm`, enlarge it or use `--data.num_workers 0`. Measure the complete run on the intended host.

## Prepare the next epoch during validation

`--data.epoch_prefetch true` with `--data.persistent_workers true` lets the existing workers prepare up to `num_workers x prefetch_factor` batches of the next training epoch during the last step and validation. It is opt-in because the training queue can remain resident alongside validation inputs: `num_workers x prefetch_factor` batches of host memory, about 1.26 GB of uint8 images for the batch 64, 640 px, 8 worker, `prefetch_factor` 2 example above, or about 5.0 GB of float32 with `--data.device_hsv true`. With pinned host memory, which is on by default when CUDA is available, queued batches can sit in page-locked RAM that the `/dev/shm` worker cap does not budget. Use it when dataset and augmentation configuration remain stable between epochs. Every packaged config breaks that: its `CloseMosaicCallback` switches mosaic off for the final epochs through `set_mosaic_prob`, which persistent workers never see, because each keeps the pipeline copy it started with — those epochs would still train on mosaics. The module warns once when such a change arrives under `epoch_prefetch`; keep `--data.persistent_workers false` for these configs. It applies to full, single-CUDA epochs with the module's own shuffle generator; batch limits, a step bound, overfit, loader reloads and distributed training keep the ordinary loader, as do non-persistent workers, and the run warns once naming the first unmet condition. CUDA is the boundary because the overlap was measured only on one CUDA GPU (an L4); other accelerators are unmeasured, not known to fail. The configured final epoch does not prepare another one. Measure the complete fit: the benefit is greatest when first-batch preparation is a meaningful fraction of the epoch.

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

### Independent segmentation predictions on CUDA

`--model.capture_segment_heads true --trainer.precision bf16-mixed` records the existing segmentation prediction routines, including their first-order backward work. The default is `false`. Box/class stems remain independent; coefficients, prototypes and semantic logits have separate backward nodes so an empty mask objective keeps absent gradients. With this option alone, common features, prototype fusion, assignment, losses, state updates and schedules stay native. Enable `capture_network` separately to replay the common features too.

Byte-exact agreement with native training is verified only on NVIDIA L4 (sm_89) with torch 2.9.1+cu129; other GPUs and torch builds are unverified. The route requires PyTorch 2.9 or newer, one CUDA device, automatic optimisation and `accumulate_grad_batches=1`. Gradient accumulation stays native because replay was validated only at `accumulate_grad_batches=1`, not because of a known defect in the recorded backward. CPU, MPS, FP32, FP16 and older BF16 runtimes warn and use native calls; evaluation stays native too. Custom or changed prediction dispatch, hooks, frozen prediction parameters, aliased storage and retained feature gradients retire replay for the fit. Each prediction group records one input signature per fit (shape, stride, dtype, device, gradient flag and CUDA stream); a call with any other signature, a partial batch included, runs the native head without adding another graph, and the recorded graph pools stay reserved until the fit ends, so a run whose batch shapes keep changing pays that memory without replaying. Copying, checkpoints and fit teardown retain the original parameter names and no recording.

A recording failure raises after restoring running buffers and RNG state; disable the flag to keep that run native. The flag retains CUDA graph pools, so check reserved memory and throughput on the actual batch size. A replayed step's mask coefficients, prototypes and semantic logits live in CUDA graph memory that the next replay overwrites in place; box and class predictions are copies. Anything that keeps those three tensors, or a gradient adopted from their backward, beyond the training step that produced them has to clone it. Measurements and rejected alternatives belong to [WP-193e](work_packages/wp-193e.md#wp-193e); a small-batch result does not establish a large-batch speed-up.

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

- **Worker counts are memory, not just parallelism, at 1024 px.** A batch of 64 stacked 1024 px images is 805 MB, and every worker holds `prefetch_factor` of them in shared memory — so `--data.num_workers 32` queues about 51 GB per loader, and the validation pool spawns while the training one is still resident. A count named for training is bounded to what fits before validation inherits it, with a warning naming the number ([WP-103](work_packages/wp-103.md)); `--data.val_num_workers` overrides that bound outright when the machine has the room.
- **Batch 16 at `img_size: 1024`**, against 128 at 640 for COCO. The tile side is R18's own crop size, and 16 is the placeholder `obb_nano_smoke.yaml` ships — "tune to accelerator memory". No oriented tier run has measured a batch that fits, so it is a starting point rather than a figure: at 2.56x the pixels per image it is a third of the COCO batch's pixel budget, not a match for it. `--model.lr` stays at the config's 0.01 for the same reason — the 0.02 above is the batch-128 linear scaling, and it does not carry over to a batch nobody has settled yet.

`lucid-eval` picks the rotated protocol from the checkpoint's own task, and with it the 1024 px letterbox and batch 8; an explicit `--img_size` or `--batch_size` still wins.

## 🖥️ Oriented and keypoint runs are single-device

`lucid-yolo fit` refuses `task: obb` and `task: keypoints` on more than one process, and says so at setup rather than partway through the first validation. Their epoch metrics — `val/rotated_mAP` and `val/oks_mAP` — accumulate plain Python lists that nothing gathers, so each rank would score its own shard and log it as the split's number. Average precision ranks every detection of the split against every other by confidence, so per-rank values cannot be averaged back into the right one; a wrong number that looks plausible is worse than a refusal. Leave `trainer.devices` at `1` for these two tasks, or pick a bigger accelerator rather than more of them. Detection and segmentation are unaffected: their metrics are torchmetrics metrics and synchronise across ranks. Whatever the task, a CUDA run trains in `channels_last` memory format automatically — parameters and images both, no flag (D23) — while CPU and MPS keep the default layout.

## 🗃️ Decoded training-image cache

`--data.cache_train_images true` decodes each distinct training file name during datamodule setup and keeps the RGB `uint8` tensor in host RAM. Mosaic, affine, colour and flip augmentation still run on fresh float storage for every draw; only the repeated JPEG/PNG decode disappears. COCO and YOLO training readers support the flag, and validation is deliberately not cached.

Budget the option from the uncompressed images, not their files: approximately `sum(3 * height * width)` bytes for 8-bit sources, before Python/tensor metadata — a 16-bit PNG decodes to `uint16` and costs two bytes per channel, so twice that — and once per device: a multi-GPU run spawns one process per rank, each building its own `DetectionDataModule` and therefore its own cache, so the single-process budget above multiplies by `trainer.devices`. The cache is keyed by file name (by path for the YOLO reader), so two records naming one file share an entry, while the same picture stored under two names is decoded and held twice. It defaults off.

Whether DataLoader workers share the cache depends on the multiprocessing start method, not on the operating system, and the data module leaves the start method at the platform default. Under `fork` the workers read the parent's cache through copy-on-write pages, so it costs one copy of RAM however many workers there are. Under `spawn` and `forkserver` — the default on macOS and Windows, and on Linux from Python 3.14 — the cache is pickled to every worker, torch moves each cached tensor into shared memory to do it (one segment or file descriptor per image), and with `--data.persistent_workers false`, the default, the hand-over repeats every epoch. The values are identical either way; a non-fork start method does not promise the one-copy cost or the speedup, and a large split can meet file-descriptor or `/dev/shm` limits at a size nobody has measured. The pipeline never writes to the cache — every draw converts into fresh float storage. Leave it off when the split does not fit comfortably beside the loader queue, model, optimizer and operating-system cache.

### Prepared training-source cache

`--data.cache_prepared_train_images true` also reuses float conversion and affine centring before random augmentation. It takes precedence over `cache_train_images`, keeps one float32 source per training record and snapshots YOLO labels with their images. Mosaic, geometry, colour, targets and RNG draws keep their existing results. The option defaults off.

Budget about `12 * sum(height * width)` bytes for the retained RGB pixels, plus target and reader metadata, per training process. Extrapolated from the 1,024-image slice measured in [WP-192c](work_packages/wp-192c.md) (3.30 MB per image), the full COCO train2017 split would need about 390 GB per process; this is an extrapolation, not a measured run. Every DDP rank is a separate process and holds its own copy, so ranks on one host multiply that figure. Each float32 pixel channel uses four times the bytes of an 8-bit channel. The prepared cache retains separate images for duplicate COCO records, whereas the decoded cache deduplicates file names. COCO reports an estimate before decoding; YOLO has no annotation sizes and reports the retained pixel bytes afterwards. Under `spawn` or `forkserver` (see above), workers receive cached tensors through shared memory, so budget the cache alongside the queued batches. Forked workers share its read-only pages. Fit-stage setup builds the cache once, repeated fit setup reuses it, and other stages release the datamodule's references. Measure the complete fit on the intended host.

## Opt-in CUDA network replay

`--model.capture_network true` records the detection backbone, neck and dual head with PyTorch's `make_graphed_callables`, including their backward pass. Assignment, losses, optimizer, LR/alpha schedules and EMA keep the ordinary loop. It is separate from `compile_step` and cannot be combined with that flag. The default is `false`. The measured speed-up is limited to a small-batch workload (batch 4); batch-32 screens changed direction, so check throughput on your own batch size before enabling it.

For `task: keypoints`, the same flag records backbone and neck plus eighteen independent box/class/point stems through the private prediction runner. Each point stem has its own backward node: a batch with no assigned positives keeps its point-parameter gradients absent. The current RLE/OKS or Laplace objectives stay native. Gradient accumulation, a fully frozen backbone and neck, changed head topology or dispatch, and head hooks/compilation keep pose native; a partial shared-feature freeze still replays. A fully frozen start or a later change to the shared-feature trainable set keeps the rest of the fit native after one warning. Changing a stem's trainable set or module mode retires that stem's recording, with a warning naming the stem; replacing a head module retires the parent before feature replay. The measured benefit is for nano/640/BF16, batch 4 on L4: five complete pairs save 12.97%, including setup, validation and checkpoints, against the existing shared-feature replay. Reserved memory rises from 1384 to 1730 MiB. Check time and memory on another batch or device. [WP-193f](work_packages/wp-193f.md) records the measurements and unverified limits; [WP-193d](work_packages/wp-193d.md) retains the earlier feature-only result.

For `task: obb`, the flag records backbone, neck and the complete dual head, including both angle outputs. The native classification and oriented losses stay outside capture. Their empty-target expressions retain zero angle gradients. Overridden training consumers, classification losses or oriented helpers keep the complete native call; gradient accumulation also stays native. The oriented speed-up is likewise limited to a small-batch workload (Nano OBB at batch 4, BF16, one L4): a single batch-8 pair changed fit time by only 4.10%, and replay raised peak reserved memory from 3974 to 4300 MiB, so check throughput and memory on your own batch size. The measurements and remaining limits are in [WP-200](work_packages/wp-200.md).

This path requires single-device CUDA automatic optimisation and `task: detect`, `task: keypoints`, `task: obb` or `task: segment`. BF16 replay requires a final PyTorch 2.9 or newer release; 2.9 pre-release and NGC `a0` builds are excluded by design. 2.9 is the first verified runtime: older BF16 runs warn and stay eager because the 2.4 comparison did not meet the exact eager oracle. FP32 and FP16 replay of detection, oriented detection and pose shared features are checked on 2.4 as well; pose stem replay has run on 2.9.1 only ([WP-193f](work_packages/wp-193f.md)). CPU, MPS and other strategies warn and stay eager; other tasks or `compile_step` raise at construction. One image signature is recorded per fit. A partial batch uses eager execution without growing a graph cache. Changes to the recorded stages’ parameter/buffer storage, module modes, hooks or precision retire replay for the rest of that fit. The supplied network must keep its ordinary stage implementations and static Python configuration. Root forward overrides, stage instance-forward overrides and registered hooks use the eager path. Anomaly detection (`--trainer.detect_anomaly true`) also stays eager so backward NaN checks run normally. A failure while recording raises instead of falling back to eager (RNG state and buffers are restored first), so disable `--model.capture_network` if recording fails on your model. Higher-order differentiation is not supported.

Recording warmup restores RNG and BatchNorm buffers before the first real batch. Accumulation or custom optimiser/zero-grad hooks preserve live gradients outside the graph-owned backward buffers before they can be overwritten. AMP uses an uncached autocast scope as the PyTorch API requires. Validation, prediction and public `forward` calls stay eager. Checkpoints retain the ordinary parameter names and the flag, with no graph state. Each fit records afresh and drops its recording on completion or an exception. CUDA graph pools can increase reserved memory even when peak allocated memory falls; measure both on the intended workload. The measured scope and remaining validation gaps are in [WP-193c](work_packages/wp-193c.md).

For segmentation, the same `--model.capture_network true` option records only backbone and neck. Prediction groups, prototype fusion and both losses keep their existing arithmetic and gradient boundaries. It can combine with `--model.capture_segment_heads true`, which records the prediction groups separately. Gradient accumulation stays native, as do overridden or compiled segmentation forwards. The default remains `false`; a recording is transient and follows the same fit cleanup and input-signature rules above. See [WP-193g](work_packages/wp-193g.md) for the measured workload and limits.

## Optional CUDA detection and pose loss execution

`--model.capture_loss true` replays the existing detection assignment and loss forward/backward with fresh predictions, targets and the current alpha tensor. Default `false`: capture costs setup time and retains CUDA workspace, so short or compute-bound runs may not benefit. It accepts `task: detect` or `task: keypoints`, and requires single-device CUDA automatic optimisation and `compile_step: false`. It can combine with `capture_network`, `capture_orthogonalization` and `capture_updates`; `capture_loss` itself records no optimizer update, so the optimizer step is whatever those two optimizer options select (the existing eager path when both are off).

For pose, the unrotated detection objective uses that recorder while point gathers and OKS stay native. During training, the built-in RLE densities can score both branches in one batch, retaining separate visible-point means and branch weights. That batching is gated on the option being active for the fit, not on the recorder: it continues after the recorder retires and on steps whose detection loss takes its native call, such as a customized criterion or BF16 on torch older than 2.9. This reduces dispatch but can change matrix-kernel selection and parameter-gradient summation rounding: pose execution retains the mathematical objective, but training is not byte-identical. Hooked, overridden or compiled densities, altered pose scoring and validation use their original calls. Branches with different input dtypes/devices or tensor subclasses keep separate density calls. Laplace-only pose keeps native point scoring. See [WP-199b](work_packages/wp-199b.md) for the measured scope and unverified accuracy checks.

One masked target workspace grows from observed instance counts and replaces the previous recording. Returned scalars and assignments own their storage. Current label bounds are checked before replay. Global module hooks, unsafe custom epsilon/exponent settings, entirely empty targets, partial batches, non-finite inputs, unsupported layouts or autograd contexts, another CUDA stream and older BF16 runtimes keep the native loss. FP32 and FP16 have focused CUDA coverage on torch 2.4.0 and 2.9.1; BF16 replay starts at the verified final 2.9 runtime. Changed criterion implementations, gains or precision retire the recording. A capture failure raises after clearing the partial owner. Higher-order differentiation and multiple backward passes through one recorded result are outside this automatic-training option.

On the measured L4 nano/640/BF16 workload, five batch-4 complete-fit pairs saved 13.14%. A separate batch-16 exploratory pair was 7.767% slower. Treat this as a small-batch launch-overhead option and measure the complete fit before enabling it for another workload.

[WP-199](work_packages/wp-199.md) records the measured workload, discarded screens and remaining validation gaps. Check complete-fit throughput and memory on your own batch size before enabling it.

## 📼 CUDA graph replay of orthogonalization is opt-in

`--model.capture_orthogonalization true` records MuSGD's existing Newton--Schulz matrix calls and replays them with each new gradient. The default is `false`: private graph pools retain extra device workspace, and setup may outweigh the gain on short or compute-bound runs. Momentum, parameter updates and the learning-rate schedule remain eager. CPU and MPS keep the original functions. [WP-197b](work_packages/wp-197b.md) records the measured workload and compatibility envelope.

## 📼 CUDA update preparation replay is opt-in

`--model.capture_updates true` additionally records MuSGD's update preparation, including momentum, for eligible dense float32 CUDA groups. Learning-rate changes and parameter writes remain outside capture. This option can be used with or without `capture_orthogonalization`; the two recordings do not nest. CPU, MPS and unsupported CUDA layouts or execution contexts retain the native step. A failure while recording raises instead of falling back to the native step (gradients and momentum are restored first), so disable `--model.capture_updates` if recording fails on your model. Loading optimizer state discards old recordings. The default is `false` because capture adds setup and retains workspace. [WP-197c](work_packages/wp-197c.md) reports five complete public-data fits, numerical checks and measured memory cost; other workloads need their own measurement.

## ⚡ torch.compile is opt-in (CUDA, detect)

`--model.compile_step true` (or `compile_step: true` under `model:`; `det_nano_smoke.yaml` carries it at `false`) compiles the detection objective with Inductor. What is compiled is one region, `DetectionLitModule._detect_objective`: the forward, both branch decodes, the task-aligned assignment and the dual loss, traced as a single graph with no breaks — `tests/ptl/test_compile.py` pins that on CPU for a batch with ground truths and for one with none. Compiling the model half alone was measured first and was not worth a flag: x1.09–1.16 on the step, because the assignment and the loss are where the small kernels and the launch overhead live. With [WP-183](work_packages/wp-183.md)'s dense box terms (no data-dependent gather) and the label check skipped under tracing, the whole region compiles: on an RTX PRO 6000, variant `n`, COCO, batch 16, deterministic, the eager training step is 74.7 ms and the compiled one 60.9 ms, x1.23. A bf16 run was recorded at x1.46 on the same bench and is **not** enabled by this flag — precision is a separate decision the fidelity gate has not been asked about.

What it costs: 110–140 s of compilation per process before the first step, and a handful of recompiles over a run: the training graph, the first `no_grad` evaluation step, and — because Dynamo specialises sizes 0 and 1 whatever is marked — one each for the first batch with no ground truths (`tal.py`'s own branch), the first with exactly one padded instance, and the first with one image, if the loader ever produces it. Measured on CPU with the `eager` backend over five batch shapes: four graphs. Batch size and padded instance count are marked dynamic on the images and the ground truths before every compiled call, so the last, smaller batch of an epoch and a batch with more instances than the one before share one graph; without the marks Dynamo would specialise until `cache_size_limit` (8) and then drop to eager without saying so, and with them the assigner's `F.one_hot(best_gt, num_gt)` — a concrete class count — had to become an `arange` comparison, since the enforcing marks refuse a pinned dim rather than silently recompiling. `TORCH_LOGS=recompiles` shows the count.

Where it refuses, it says so once at `setup` and trains eager. Off CUDA: the MPS Inductor backend crashes in `convolution_backward`, and CPU has nothing to gain. Under `task: segment`, `obb` or `keypoints`: their extra terms read tensors on the host inside `_task_extra_loss`, outside the region, so compiling the detection part alone would buy a fraction of the x1.23 for the same compile cost; they stay eager until their terms are made traceable. The compile wraps the bound method, not the module, so parameter names, the state dict, the EMA shadow, checkpoints and export are exactly what they are without the flag — a compiled *module* would prefix every key with `_orig_mod.`.

The default is off on cost, not on doubt. A78 — whether the compiled objective trains as eager does — was validated on an A100 (paired 4-epoch run: 1 recompile, step 180.6 → 144.2 ms, loss parity `3.6e-5` with TF32 off), but compilation added 210 s of wall to a 4-epoch run of 313 steps, a break-even near 19 epochs at that size and under one epoch on COCO train; turn it on per recipe when the run is long enough. When comparing the two arms, compare mAP, or set `torch.set_float32_matmul_precision("highest")` first: under TF32 the per-batch `train/loss` differs by up to `1e-1`, because the task-aligned assignment is discrete and TF32 rounding moves its top-k set. D24 carries the numbers.

## 🎲 What `deterministic: true` and `seed` actually pin

Every tier config sets `deterministic: true` and seed 0. Two boundaries sit inside that, and neither fails loudly.

**On MPS it is not strict.** `default_determinism` (`lucid_yolo/cli/train.py`) resolves the trainer's `deterministic` flag to `"warn_only"` whenever MPS is the auto-picked accelerator, because MPS ships no deterministic implementation for some backward kernels this model reaches — `index_put_with_accumulate`, hit by the assignment and loss backward — and strict `True` aborts mid-step there. Every op with a deterministic kernel stays deterministic; the rest downgrade to a warning. CPU and CUDA keep strict `True`. So an Apple-silicon run under D12c is reproducible only up to those kernels, and says so in the log rather than by failing.

**The worker count is part of the seed.** A seeded run is byte-identical only against another run at the *same* `--data.num_workers`. At `0` the pipeline draws every augmentation parameter from one generator in the parent process, in index order. With workers each is re-seeded per worker and per epoch from the loader's own generator ([WP-079](work_packages/wp-079.md)), which is what keeps per-epoch diversity alive. Both streams are fully determined by `seed` and they are **different streams**: the recipes above run at 32 workers, and rerunning one at 0 workers, or at 16, reproduces the schedule and not the sample-by-sample augmentation. Reproducing a published run means matching its worker count as well as its seed, which is why every command above states one.

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

`--data.layout yolo` is stated rather than left to be probed: `detect_layout` resolves a bare YOLO root on its own, but stating it also covers a root that happens to satisfy both conventions (COCO and YOLO — the probe refuses to break that tie, A63) and skips the probe entirely, so a missing path is named by the reader that actually needed it, not the probe. `--model.num_classes` must equal `len(names)` in the root's own `data.yaml` — a label row's class index is 0-based into that list and is never remapped ([WP-099](work_packages/wp-099.md)), so a mismatched count trains against the wrong label space silently. `lucid-data check --data_root ...` prints both numbers so they can be compared before a launch, not after one.

**Not available on this layout: `--model.task segment`.** A YOLO label row carries five or nine normalized numbers and never a polygon ring, so there is nothing to rasterize a mask from — `DetectionDataModule` refuses `mask_targets=True` on a YOLO root rather than training a segmentation head against empty masks ([WP-099b](work_packages/wp-099b.md)). Full format/layout mechanics: docs/DATASETS.md.

## 🔌 Before any of them: the wiring gate

Each tier has a minutes-long overfit gate that must pass before a launch that costs hours. It is development tooling and ships in `scripts/`, not in the wheel, so it runs from a checkout:

```bash
python scripts/overfit_micro.py --task detect     # one-to-one train recall >= 0.95
python scripts/overfit_micro.py --task segment    # train mask IoU >= 0.7
python scripts/overfit_micro.py --task obb        # train rotated mAP50 >= 0.9
python scripts/overfit_micro.py --task keypoints  # train OKS AP >= 0.30
```

A tier that cannot overfit a handful of images will not converge on the full set, and finding that out after the first epoch of a fifty-epoch run costs the run. The four commands, and the shapes regression beside them, run as they are in the [development gates notebook](notebooks/development_gates.ipynb).

The keypoint floor looks low beside the other three and is not measuring less. Its slice is the only one not drawn from the geometric shapes — `task: keypoints` needs a keypoint-bearing family, so it draws a 7-point synthetic symbol schema, and OKS at that schema's uniform sigma is a cliff on objects a few dozen pixels across: feeding the ground truth back as the prediction scores exactly 1.0, and displacing every point by 3 px scores 0.269. The floor was set from that measured slope rather than guessed.
