# 📦 WP-197b — `perf(optim): replay CUDA orthogonalization graphs`

> Engineering log · [Phase 17 — R1 pose-objective parity](https://borda.github.io/lucid-YOLO/engineering/phase-17-pose-objective-parity/)
>
> 🔗 Depends on WP-196 · Needed by none
>
> ⬅️ [WP-197](https://borda.github.io/lucid-YOLO/work_packages/wp-197/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-198](https://borda.github.io/lucid-YOLO/work_packages/wp-198/) ➡️


## WP-197b — static matrix iterations keep their arithmetic and lose host dispatch

<a id="wp-197b"></a>

**The captured region is pure.** `capture_orthogonalization`, off by default, records the existing scalar and batched Newton--Schulz functions rather than the optimizer step. Each new matrix is copied in and each returned result is copied out. Momentum, parameter writes, learning-rate schedules, gradient scaling and EMA stay outside the graphs. Neither function's arithmetic changes. CPU, MPS, autograd, autocast, non-contiguous and offset inputs keep the eager functions; an enclosing capture or compiler also keeps the original operations.

**The optimizer owns the workspace.** The cache is bounded by its matrix-parameter count. Function, shape, stride, dtype, device, step count, epsilon, CUDA stream, GEMM precision, BLAS backend and determinism select a recording. New signatures fall back to eager when that bound is reached. Separate graph pools allow skipped gradients and changing bucket order. A failed recording publishes no entry; loading optimizer state drops old recordings. Legacy state without the new field honours the constructor's requested mode.

**The evidence is complete fits rather than a helper timer.** [MEASURED] Five paired six-epoch `Trainer.fit` runs per starting tree on an NVIDIA L4, driver 580.178.04, PyTorch 2.9.1+cu129 and Lightning 2.6.6. Nano detection, 640 px, batch 4, BF16 mixed, four persistent workers and decoded-image caching use the first 256 sorted public COCO val2017 images (1,834 annotations). The 32 validation images overlap the training slice: this is a throughput and numerical-trajectory workload, not held-out accuracy or a new acceptance tier. Timing includes graph setup, loading, augmentations, backward, optimizer, EMA, validation and checkpoints; process imports and the CPU module constructor precede the measured fit.

| Starting tree | Eager fit seconds, median (range) | Replay fit seconds, median (range) | Time saved |
| -- | -- | -- | -- |
| Main before the compatible-batching merge (01b4d4c), no device HSV | 99.092 (98.506–99.243) | 79.515 (79.141–80.062) | 19.76% |
| Compatible batching and device HSV applied, expected benefit on current main | 88.009 (87.149–88.255) | 76.960 (76.671–78.001) | 12.55% |

[MEASURED] Each of the ten pairs has identical byte hashes for model, optimizer momentum state (not the full optimizer state_dict: the `capture_orthogonalization` flag sits in `param_groups` and differs by design), scheduler, EMA, loss sequence and validation metrics; the baselines match themselves too. Separate source hashes bind the runs to the executable code. Neither original orthogonalization body changes. Replacing replay with eager execution fails the committed Python-work-count regression.

[MEASURED] Main peak allocated CUDA memory is 829.8 (821.0–829.8) to 880.8 (880.8–880.9) MiB; peak reserved memory is 1078.0 (1070.0–1086.0) to 1298.0 (1298.0–1306.0) MiB. [MEASURED] The composition peak allocated CUDA memory is 809.7 (806.1–817.8) to 867.1 (863.1–872.4) MiB; peak reserved memory is 1082.0 (1082.0–1090.0) to 1270.0 (1268.0–1278.0) MiB. These are allocator metrics, not an OOM or capacity benefit.

[MEASURED] Five separate two-update real-data probes time the 41 initial recordings at median 188.1 ms (188.0–192.7); this setup is also included in each full-fit result above. An earlier batch-8 prototype was screened only once and was not advanced without repeated complete-fit evidence. Larger batches, other models, DDP and other accelerators have no performance claim; retained workspace and startup make the flag opt-in.

**Scope and checks.** [MEASURED] The focused CPU/doctest runs pass 228 cases. The final replay test module passes 35 cases (30 CUDA and 5 CPU/doctest) on both PyTorch 2.4.0+cu124 and 2.9.1+cu129 on the L4; full precommit passes. Focused CPU and real-CUDA regressions cover independent returned storage, fresh inputs, bounded signatures, precision changes, skipped gradients, resume, eager fallbacks and failed capture. Reusing a cached integer signature for a float step count is specifically refused through the original eager error. `test_cli.py::test_capture_orthogonalization_flag_reaches_module` mirrors WP-184's `compile_step` precedent and proves `--model.capture_orthogonalization` reaches `DetectionLitModule.hparams` through the real parser rather than a constructor fixture. The full development and accelerator golden gates are outside this run's authorized validation scope; this row therefore remains open.

