# 📦 WP-076 — `perf(data): epoch-recycled loader workers`

> Research log · [Phase 6 — Evaluation, v0.1.0](https://borda.github.io/lucid-YOLO/research/phase-06-evaluation/)
>
> 🔗 Depends on [WP-014](https://borda.github.io/lucid-YOLO/work_packages/wp-014/), [WP-073](https://borda.github.io/lucid-YOLO/work_packages/wp-073/) · Needed by [WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/)
>
> ⬅️ [WP-073](https://borda.github.io/lucid-YOLO/work_packages/wp-073/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/) ➡️


## WP-076 — epoch-recycled loader workers

<a id="wp-076"></a>

[WP-073](https://borda.github.io/lucid-YOLO/work_packages/wp-073/)'s compact store did not end the OOMs: Colab still killed the run at **epoch 24 and epoch 38**. The residue is not one leak but a class of them — allocator high-water growth, arena fragmentation, and whatever copy-on-write pages survive — all of which are per-worker and none of which a dataset change reaches.

Respawning the pool costs seconds against a 9-minute epoch and hard-resets every one of them, so `persistent_workers` defaults **off**. This is a case where the cheap blunt fix dominates: no diagnosis of *which* allocator behaviour dominated was needed, because recycling ends all of them and the measurement said the cost was noise.

