# 📦 WP-193h — `perf(ptl): omit gradients cleared by the native closure`

> Engineering log · [Phase 18 — CUDA graph replay and pipeline throughput](https://borda.github.io/lucid-YOLO/engineering/phase-18-cuda-replay-throughput/)
>
> 🔗 Depends on [WP-208](https://borda.github.io/lucid-YOLO/work_packages/wp-208/) · Needed by none
>
> ⬅️ [WP-193g](https://borda.github.io/lucid-YOLO/work_packages/wp-193g/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-194](https://borda.github.io/lucid-YOLO/work_packages/wp-194/) ➡️


## WP-193h — omit only the snapshots discarded before backward

<a id="wp-193h"></a>

Network replay preserves live gradients at forward time because its recorded backward owns static gradient buffers ([WP-208](https://borda.github.io/lucid-YOLO/work_packages/wp-208/)). That copy remains necessary under accumulation. R41's native automatic-optimization closure instead runs forward, zero-grad and backward in that order. With accumulation one and no zero-gradient observer, its copies become tensors that the following zero-grad discards.

The module now scopes the parameters in that real optimizer's live groups during the native closure's forward. The shared snapshot helper leaves those gradients in place. Zeroing, backward, callbacks and updates retain their original order and arithmetic. There is no new flag or checkpoint state. A context token resets on success or failure, and the original closure is restored in `finally`.

Only detection with active network replay, the native single-device strategy, native precision plugin and pass-through profiler, MuSGD step and zeroing, and the inspected Lightning 2.4–2.6 closure qualify. Direct forwards, accumulation, other tasks, custom objectives, model/strategy/precision hooks, zero-gradient callbacks, non-default or customized Lightning profilers, or custom optimizer wrappers and unknown runtime contracts keep the original snapshots. Optimizer parameters removed from a live group still receive a snapshot. Native optimizer profiler and LambdaLR wrappers qualify without accepting arbitrary `functools.wraps` wrappers.

No new method constant, numerical convention or assumption is introduced. The basis is the repository's own gradient ownership and CUDA profiles, R41's native closure ordering and R43's graph-owned gradient buffers. No external detection implementation was consulted.

## Isolated L4 measurement

Five alternating fresh-process pairs per baseline, nano detection at batch 4, 640 px, BF16, 20 epochs, global/model seed 41, data-module seed 0 (its default), strict determinism (the CUDA CLI default) and `cudnn.benchmark=False`. The dataset is 384 public COCO val2017 training photographs and 32 disjoint validation photographs, with the original 80-class annotations (3,318/255 instances). Shared filesystem, driver and existing Triton caches are warm after the controls; each process recreates its image cache, model, Trainer and graph recordings. Each epoch consumes all training records and validates on all 32 validation records. EMA, gradient clipping at 10, mosaic, prepared-image cache, device HSV, epoch prefetch and the merged network/loss/orthogonalization/update replay options are enabled in both arms. Learning rate is 0.001, with three warmup epochs and four persistent data workers. The second baseline adds the norm replay/bookkeeping change at c5bc00ae6c23e364ac6348b505319e8a80f45a7d, an intermediate commit of the branch later squash-merged to main as ef020b2 (PR 28). The merged source differs from that commit, so re-measuring this baseline needs c5bc00ae itself.

Loggers, periodic Lightning checkpointing, progress bars and model summary are disabled; the final state checkpoint remains included.

Seconds below are median (minimum–maximum). The workflow clock includes CUDA-fenced model/data/Trainer construction, cache preparation, graph recording, every training and validation epoch, teardown and the final state checkpoint. The separate process clock also includes imports and CUDA context initialization.

| Baseline | Original workflow | With this change | Time saved |
| -- | -- | -- | -- |
| Main 303a7e0 | 182.970 (182.441–184.103) | 166.914 (166.530–167.857) | 8.78% |
| Main plus norm replay/bookkeeping | 164.251 (163.440–165.069) | 149.493 (149.299–150.553) | 8.99% |

Whole-process times are 188.001 (187.487–189.245) → 172.121 (171.612–173.042) seconds on main, an 8.45% reduction; with norm replay/bookkeeping, 169.435 (168.550–170.203) → 154.508 (154.279–155.628), an 8.81% reduction. The last complete epoch averages 80.750 (79.773–83.055) → 72.227 (70.916–74.850) ms/step on main, and 70.810 (69.611–74.207) → 64.488 (63.508–64.920) ms/step on the second baseline. Allocated peaks after resetting at the second epoch are 781,228,544 → 771,284,480 bytes and 792,121,344 → 782,177,280 bytes, respectively, constant across five runs. Those peaks include later training and validation; no capacity benefit is claimed.

Runtime: NVIDIA L4, driver 580.178.04, CUDA 12.9, Python 3.11.18, torch 2.9.1+cu129, Lightning 2.6.6, g2-standard-8 with four CPU threads. One external NVML observer spans the matrix, outside each workflow/process clock. Raw commands, logs, per-step losses, runtime receipts and state fingerprints were recorded during the run but are not committed to the repository. The 1,920 losses per run and model, optimiser, EMA, scheduler and validation states match byte for byte across all runs within each baseline. Native controls also match each other. Scalar comparisons include IEEE-754 signed-zero bytes.

Profiling before implementation attributed 156.432 ms of 1,055.133 ms across six instrumented detection steps to the snapshot owner. Its 2,424 clones become 210, removing 369 per step. These instrumented durations explain the mechanism; the unprofiled complete workflows above provide the performance result.

Focused CUDA coverage passes 108 cases on the measured L4 runtime, including raw pre-clipping gradient bytes in FP32/BF16, and 35 new cases pass with Lightning 2.4.0 over torch 2.9.1. CPU/doctest checks pass 46 cases with three explicit skips; pre-commit and strict typing pass. The closure-shadow, profiler-observer and precision-dispatch regressions fail before their respective fixes and pass afterward.

This establishes a small-batch, replay-enabled detection result, not a universal training gain. One exploratory batch-16, three-epoch pair saved 4.86%; it changes both batch size and run length and does not isolate either effect. Larger datasets, other hardware, multi-GPU, MPS, torch 2.4, full `make gate`, aggregate coverage, accelerator goldens and long-run accuracy remain unverified. Other tasks and unrecognised closure contracts retain the original snapshots.

