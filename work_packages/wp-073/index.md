# 📦 WP-073 — `perf(data): compact annotation store + val-loader worker cap`

> Research log · [Phase 6 — Evaluation, v0.1.0](https://borda.github.io/lucid-YOLO/research/phase-06-evaluation/)
>
> 🔗 Depends on [WP-014](https://borda.github.io/lucid-YOLO/work_packages/wp-014/) · Needed by [WP-076](https://borda.github.io/lucid-YOLO/work_packages/wp-076/), [WP-074](https://borda.github.io/lucid-YOLO/work_packages/wp-074/)
>
> ⬅️ [WP-071](https://borda.github.io/lucid-YOLO/work_packages/wp-071/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-076](https://borda.github.io/lucid-YOLO/work_packages/wp-076/) ➡️


## WP-073 — compact annotation store

<a id="wp-073"></a>

`CocoDetectionDataset` held the raw JSON annotation dicts after building its per-image `Targets`. Millions of small Python objects, each with a refcount touched on every access — and refcount traffic materializes copy-on-write pages in every persistent DataLoader worker, so a fork-shared dataset is only shared until Python reads it.

Observed on a Colab 176 GB host, killed at validation boundaries: **epoch 1 at 48 workers, epoch 24 at 32 workers**. The worker count sets how fast the leak accumulates, not whether it does.

Dropping the dicts fixed the growth; the val-loader worker cap (`min(num_workers, 4)`) addressed the second half, a persistent val pool doubling the worker population at exactly the moment the training pool is still resident. That cap is what [WP-102](https://borda.github.io/lucid-YOLO/work_packages/wp-102/#wp-102) later narrowed and [WP-103](https://borda.github.io/lucid-YOLO/work_packages/wp-103/#wp-103) restored for inherited counts.

