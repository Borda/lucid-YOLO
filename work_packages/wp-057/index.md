# 📦 WP-057 — `feat(data): 1024 px overlapping crop tiling`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-056](https://borda.github.io/lucid-YOLO/work_packages/wp-056/) · Needed by [WP-058](https://borda.github.io/lucid-YOLO/work_packages/wp-058/), [WP-094](https://borda.github.io/lucid-YOLO/work_packages/wp-094/)
>
> ⬅️ [WP-056](https://borda.github.io/lucid-YOLO/work_packages/wp-056/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-058](https://borda.github.io/lucid-YOLO/work_packages/wp-058/) ➡️


## WP-057 — tiling is what creates difficult instances

<a id="wp-057"></a>

DOTA ships a per-object `difficult` flag and [WP-056](https://borda.github.io/lucid-YOLO/work_packages/wp-056/#wp-056) declined to decide its fate at load, deferring to A39. Tiling is where that deferral turns load-bearing: R18's partial-object rule flags an instance difficult when its clipped area falls below 0.7 of its original, so **the tiler manufactures difficult instances the source annotations never carried**. A39 therefore governs a population that DOTA's own flag column does not describe, and the number the decision was made from has to survive the crop — which is why `U` is carried out as `visible_fraction` rather than consumed and discarded.

