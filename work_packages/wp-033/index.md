# 📦 WP-033 — `test(optim): toy convergence golden vs SGD`

> Research log · [Phase 4 — MuSGD](https://borda.github.io/lucid-YOLO/research/phase-04-musgd/)
>
> 🔗 Depends on WP-032 · Needed by none
>
> ⬅️ [WP-030](https://borda.github.io/lucid-YOLO/work_packages/wp-030/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-038](https://borda.github.io/lucid-YOLO/work_packages/wp-038/) ➡️


## WP-033 — toy convergence golden

<a id="wp-033"></a>

MuSGD reaches the 0.1 MSE threshold in **47 steps against momentum-SGD's 90** on a fully seeded micro-CNN regression task.

The tolerance is the interesting part: step counts are pinned at **±15**, final losses at 0.005. Cross-platform torch float drift shifts trajectories slightly ([Float reproducibility](https://borda.github.io/lucid-YOLO/engineering/cross-cutting/#cross-float)), so a tight pin would be a golden about the machine. The 43-step separation is what keeps the *directional* claim decisive inside a tolerance that loose — the golden asserts the ordering it cares about rather than the number it happened to get.

