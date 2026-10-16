# 📦 WP-087 — `feat(ptl): segmentation training path`

> Research log · [Phase 7 — Instance segmentation, v0.2.0](https://borda.github.io/lucid-YOLO/research/phase-07-instance-segmentation/)
>
> 🔗 Depends on WP-053 · Needed by [WP-087b](https://borda.github.io/lucid-YOLO/work_packages/wp-087b/), [WP-087c](https://borda.github.io/lucid-YOLO/work_packages/wp-087c/), [WP-087e](https://borda.github.io/lucid-YOLO/work_packages/wp-087e/), [WP-087f](https://borda.github.io/lucid-YOLO/work_packages/wp-087f/), [WP-054](https://borda.github.io/lucid-YOLO/work_packages/wp-054/), [WP-137](https://borda.github.io/lucid-YOLO/work_packages/wp-137/)
>
> ⬅️ [WP-052b](https://borda.github.io/lucid-YOLO/work_packages/wp-052b/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-087b](https://borda.github.io/lucid-YOLO/work_packages/wp-087b/) ➡️


## WP-087 — factories, not a `Detector` object

<a id="wp-087"></a>

Composing the Phase 7 modules onto the shared backbone and neck went behind `build.py` factories rather than behind a `Detector` object holding them. The reason is the state dict: an object that owns the backbone and neck renames every parameter under it, and flat keys are what keep the accepted Det-smoke checkpoint loading **strict** after a second task lands beside it. That checkpoint is the artifact later phases load, so the DoD asserts the strict load directly rather than trusting the module tree to have stayed the same shape.

