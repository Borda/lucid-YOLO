# 📦 WP-005 — `ci(gates): golden harness and frozen-golden regression`

> Engineering log · [Phase 0 — Foundation](https://borda.github.io/lucid-YOLO/engineering/phase-00-foundation/)
>
> 🔗 Depends on [WP-004](https://borda.github.io/lucid-YOLO/work_packages/wp-004/) · Needed by WP-006
>
> ⬅️ [WP-004](https://borda.github.io/lucid-YOLO/work_packages/wp-004/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-013](https://borda.github.io/lucid-YOLO/work_packages/wp-013/) ➡️


## WP-005 — goldens that survive a second machine

<a id="wp-005"></a>

The first fixture golden byte-hashed the generated annotation JSON. Those hashes are per-platform stable only — see [Float reproducibility](https://borda.github.io/lucid-YOLO/engineering/cross-cutting/#cross-float) — so they were replaced with structural metrics: exact integer counts (images, annotations, categories, polygon points) plus bbox area and coordinate sums pinned with a small absolute tolerance.

The split that has held ever since: **integers exact, aggregates to tolerance**. Counts are platform-stable by construction; sums of floats are not. Every golden this project has frozen follows it.

