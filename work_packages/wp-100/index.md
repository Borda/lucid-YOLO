# 📦 WP-100 — `perf(data): parallel tiling with a progress bar`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-094](https://borda.github.io/lucid-YOLO/work_packages/wp-094/) · Needed by [WP-101](https://borda.github.io/lucid-YOLO/work_packages/wp-101/)
>
> ⬅️ [WP-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-101](https://borda.github.io/lucid-YOLO/work_packages/wp-101/) ➡️


## WP-100 — parallel tiling

<a id="wp-100"></a>

A full DOTA build is **1,869 multi-megapixel PNG decodes**, each cropped into dozens of tiles and written back, and it ran silently on one core for about an hour — an operator had no way to tell a working build from a hung one.

Two constraints shaped the parallel form. Each worker pins itself to a **single torch thread**, since a pool of processes each fanning out over every core is oversubscription rather than parallelism. And ids are deliberately **not** assigned in the workers: a worker cannot know how many tiles preceded it, so numbering there would make the output JSON depend on which process finished first. The parent numbers every record in source order, and the test that protects this compares a four-image pooled build's JSON byte-for-byte against the serial one.

