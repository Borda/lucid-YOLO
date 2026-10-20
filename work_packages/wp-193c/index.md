# 📦 WP-193c — `perf(ptl): replay detection network CUDA graphs`

> Engineering log · [Phase 17 — R1 pose-objective parity](https://borda.github.io/lucid-YOLO/engineering/phase-17-pose-objective-parity/)
>
> 🔗 Depends on [WP-193b](https://borda.github.io/lucid-YOLO/work_packages/wp-193b/) · Needed by [WP-193d](https://borda.github.io/lucid-YOLO/work_packages/wp-193d/)
>
> ⬅️ [WP-193b](https://borda.github.io/lucid-YOLO/work_packages/wp-193b/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-193d](https://borda.github.io/lucid-YOLO/work_packages/wp-193d/) ➡️


## WP-193c — replay the network, keep the dynamic objective eager

<a id="wp-193c"></a>

**The source is the current training loop and R43's general-purpose API.** This follow-up was profiled from `01b4d4c` after compatible MuSGD batching merged, then rebased and remeasured on `50c2092` after the augmentation contribution merged. It changes how existing network work is dispatched, without rewriting any model, assignment, loss or optimizer arithmetic. `capture_network=False` is the default. Unlike the Inductor objective option, the recording contains only the detection network's forward and backward work; targets, loss, schedules, optimizer and EMA remain eager. Unlike the orthogonalization replay ([WP-197b](https://borda.github.io/lucid-YOLO/work_packages/wp-197b/), already on main), it records no optimizer work.

**One fit owns one recording.** A private tensor-tuple view composes the same backbone, neck and head. PyTorch supplies the autograd bridge. Recording restores RNG and all buffers after warmup, and mixed precision disables the autocast cache. A partial batch stays eager. Changed storage, modes, hooks or precision retire the recording. Public forward, validation and prediction remain ordinary calls. The holder registers no parameters and serializes no CUDA state; fit completion and exceptions clear it. CUDA single-device automatic optimisation and detection are the measured scope; higher-order differentiation and `compile_step` are excluded.

**BF16 has a measured runtime boundary.** FP32 and FP16 replay pass the exact eager comparisons on torch 2.4.1. BF16 does not, so BF16 replay starts at the verified 2.9 runtime and older BF16 fits remain eager. This avoids choosing a tolerance to hide the failed exact oracle; it does not claim that every intervening release was tested.

**Accumulation retains its own gradients.** The CUDA recorder reuses backward gradient storage. If a later backward must add to live gradients, the holder first clones those values out of that storage. This also covers custom optimiser or zero-grad hooks. An analytic scalar probe and independent eager-network regressions expose the unprotected alias: two microbatches can otherwise produce 6 where the derivative is 4. Ordinary one-step optimisation keeps the uncopied route.

**The claim is the complete short fit.** Five alternating paired, fresh-process `Trainer.fit` runs after two discarded warm-up fits used detection n (depth 0.50, width 0.25, max channels 1024), batch 4, 640 px, BF16 mixed precision, four workers, persistent workers and decoded-image caching on 256 real COCO val2017 images. Each six-epoch / 384-step fit included cache setup, augmentation, network and loss, backward, MuSGD, LR warmup/decay, alpha scheduling, EMA, eight validation batches per epoch and checkpoints. The validation images are a 32-image subset of the training subset: this is throughput and trajectory evidence, never held-out accuracy. Persistent workers retain their augmentation state, so the mosaic-close callback is not evidence that worker-side mosaic actually closed.

On an isolated NVIDIA L4, driver 580.178.04, torch 2.9.1+cu129, Lightning 2.6.6, Python 3.12.3, synchronized fit time fell from median **92.589 s** (91.614-93.691) to **77.330 s** (76.234-78.265): **16.48%** by the ratio of medians. Recording setup is inside each fit; imports and model construction are outside both arms. Baseline-against-itself and every paired arm produced identical SHA256 digests over the raw tensor bytes of losses, model, optimizer, scheduler, EMA and metrics. This certifies those deterministic short trajectories, not all precisions, devices or long-run accuracy.

**Memory has two directions.** Median peak allocated memory fell from 828.5 MiB (822.0-831.4) to 711.3 MiB (711.3-711.3), while median peak reserved memory rose from 1022.0 MiB (1022.0-1038.0) to 1260.0 MiB (1260.0-1262.0). The graph pool is a cost of the opt-in path; no avoided-OOM or batch-capacity improvement is claimed. Fit callbacks drop the holder's recording, rather than promising that every outstanding autograd reference or allocator reservation vanishes immediately.

**The speed claim is limited to this small-batch workload.** Batch-32 screens changed direction relative to the batch-4 result, so the gain above is not a general claim; the shipped detection recipes train at batch 16, which was not measured here.

**Validation remains explicitly bounded.** Full `make gate`, accelerator goldens and final accuracy have not been run; this row stays open on those checks. Focused regression tests and timing logs are retained with the contribution's review evidence.

