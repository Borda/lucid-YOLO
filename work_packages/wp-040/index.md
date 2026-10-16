# 📦 WP-040 — `test(lit): overfit-100 integration golden`

> Research log · [Phase 5 — Lightning training loop](https://borda.github.io/lucid-YOLO/research/phase-05-training-loop/)
>
> 🔗 Depends on [WP-039](https://borda.github.io/lucid-YOLO/work_packages/wp-039/) · Needed by WP-045, [WP-127](https://borda.github.io/lucid-YOLO/work_packages/wp-127/), [WP-136](https://borda.github.io/lucid-YOLO/work_packages/wp-136/)
>
> ⬅️ [WP-039](https://borda.github.io/lucid-YOLO/work_packages/wp-039/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-071](https://borda.github.io/lucid-YOLO/work_packages/wp-071/) ➡️


## WP-040 — overfit-100 on an accelerator

<a id="wp-040"></a>

Train-set recall **0.9747** against the 0.95 floor, MPS, 4:47 wall clock, decoded NMS-free through the deployed path.

The structural decision that outlived it: `goldens/gpu/` is excluded from default harness discovery and recomputed only under `--include-gpu`, which is what keeps the offline gate accelerator-free while still holding accelerator numbers under version control. The cost of that split is that a `gpu/` golden can go stale unnoticed — `shapes_regression_det.json` currently has an outstanding re-freeze.

*Superseded by [WP-167](https://borda.github.io/lucid-YOLO/work_packages/wp-167/#wp-167).* The outstanding re-freeze is closed: all five files under `goldens/gpu/`, `shapes_regression_det.json` among them, were regenerated on an L4. The sentence above is left as written because the staleness it predicted is exactly what happened — `golden-gpu` returned 46/48 on the pre-WP-169 values — and a log that edits its own correct predictions into the past tense stops being evidence that anything was predicted.

