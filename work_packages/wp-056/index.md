# 📦 WP-056 — `feat(data): DOTA parsing and long-edge conversion`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-055](https://borda.github.io/lucid-YOLO/work_packages/wp-055/) · Needed by [WP-057](https://borda.github.io/lucid-YOLO/work_packages/wp-057/), [WP-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/)
>
> ⬅️ [WP-055](https://borda.github.io/lucid-YOLO/work_packages/wp-055/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-057](https://borda.github.io/lucid-YOLO/work_packages/wp-057/) ➡️


## WP-056 — one mask filters every modality

<a id="wp-056"></a>

What the DOTA parser guarantees beyond its shapes is an **alignment**: `rboxes[i]` is the same instance as `boxes[i]` and `labels[i]`, for every `i`. That is why it is stated as an invariant rather than left as an implementation detail — one boolean mask filters every modality at once, so a caller dropping difficult instances, or [WP-057](https://borda.github.io/lucid-YOLO/work_packages/wp-057/#wp-057) dropping instances a window lost, writes the filter once instead of three that are free to drift apart. Every stage of the oriented path downstream of the loader relies on it.

