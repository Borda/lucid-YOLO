# 📦 WP-014 — `feat(data): COCO dataset and LightningDataModule`

> Research log · [Phase 1 — Data pipeline](https://borda.github.io/lucid-YOLO/research/phase-01-data-pipeline/)
>
> 🔗 Depends on WP-012, [WP-013](https://borda.github.io/lucid-YOLO/work_packages/wp-013/) · Needed by WP-015, WP-068, WP-070, [WP-071](https://borda.github.io/lucid-YOLO/work_packages/wp-071/), [WP-073](https://borda.github.io/lucid-YOLO/work_packages/wp-073/), [WP-076](https://borda.github.io/lucid-YOLO/work_packages/wp-076/), [WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/), [WP-121](https://borda.github.io/lucid-YOLO/work_packages/wp-121/)
>
> ⬅️ [WP-013](https://borda.github.io/lucid-YOLO/work_packages/wp-013/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-022](https://borda.github.io/lucid-YOLO/work_packages/wp-022/) ➡️


## WP-014 — the DataLoader transport saga

<a id="wp-014"></a>

Five commits over one work package, each fixing the failure the previous one exposed. Worth reading as one story, because the later packages ([WP-071](https://borda.github.io/lucid-YOLO/work_packages/wp-071/#wp-071), [WP-073](https://borda.github.io/lucid-YOLO/work_packages/wp-073/#wp-073), [WP-076](https://borda.github.io/lucid-YOLO/work_packages/wp-076/#wp-076), [WP-103](https://borda.github.io/lucid-YOLO/work_packages/wp-103/#wp-103)) are all continuations of it.

1. **Starvation.** CUDA runs showed GPU utilization spikes: a mosaic sample alone decodes four images, and the loaders shipped with no `pin_memory`, prefetch depth 2, and every worker inheriting torch's full intra-op thread pool — `num_workers` processes each oversubscribing the CPU by the core count. Fixed with pin-memory-when-CUDA, prefetch 4, and a `worker_init_fn` capping each worker to one torch thread.
2. **Prefetch 4 was wrong.** It ENOMEMed containerized CUDA runs: the queue holds `num_workers × prefetch_factor` batches in POSIX shared memory, and at batch 64 (~315 MB of stacked 640 px images) × 12 workers × 4 that is **~15 GB of /dev/shm segments in flight**. The pin-memory thread dies first, on `mmap` with "Cannot allocate memory". Default returned to the DataLoader's own 2.
3. **Auto worker count was also wrong.** `min(batch_size, cores)` on a ~64-vCPU Colab at batch 64 launched ~64 workers and refilled shm — prefetch 2 alone had not closed the failure mode. `_shm_capped_workers` bounds the *auto* count so `workers × prefetch × batch bytes` fits half the currently free `/dev/shm` (via `statvfs`), floor one worker. Hosts without `/dev/shm` (macOS, Windows) and explicit counts are untouched. **This is the guard [WP-102](https://borda.github.io/lucid-YOLO/work_packages/wp-102/#wp-102) later removed for named counts and [WP-103](https://borda.github.io/lucid-YOLO/work_packages/wp-103/#wp-103) restored for inherited ones.**
4. **Segment count, not byte count.** Torch's `file_descriptor` sharing ships every tensor crossing the worker boundary as its own shm segment, each costing the consumer one `mmap`. The natural `(images, list[Targets])` batch is **~600 tiny segments at batch 64**, so in-flight batches blow the per-process `vm.max_map_count` (65530, read-only on Colab): observed "unable to mmap 96/144/336 bytes". Fixed by flattening the ragged batch into eight dense tensors and restoring it on the destination device.

The through-line: three distinct resources exhaust in this pipeline — shm **bytes**, shm **segments**, and CPU **threads** — and each has its own failure signature at a different layer. A fix aimed at one does not touch the others, which is why the same symptom ("the loader died at step 0") kept returning with a different cause.

