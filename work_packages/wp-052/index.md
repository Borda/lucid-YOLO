# 📦 WP-052 — `feat(models): Segmenter wiring; aux branch absent from deploy()`

> Research log · [Phase 7 — Instance segmentation, v0.2.0](https://borda.github.io/lucid-YOLO/research/phase-07-instance-segmentation/)
>
> 🔗 Depends on WP-051 · Needed by [WP-052b](https://borda.github.io/lucid-YOLO/work_packages/wp-052b/), WP-053
>
> ⬅️ [WP-085](https://borda.github.io/lucid-YOLO/work_packages/wp-085/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-052b](https://borda.github.io/lucid-YOLO/work_packages/wp-052b/) ➡️


## WP-052 — Segmenter wiring

<a id="wp-052"></a>

Absence of the training-only branches from `deploy()` is proved twice rather than asserted: by parameter identity and a module walk, and independently by arithmetic — `deployed == full - aux - o2m`, exact at **2,717,284 − 5,200 − 152,690 = 2,559,394**. Two proofs because the failure mode is a branch that is disabled rather than absent, which a module walk can miss and arithmetic cannot.

