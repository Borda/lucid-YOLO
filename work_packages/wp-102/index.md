# 📦 WP-102 — `perf(ptl): validation that is not CPU-bound`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-088](https://borda.github.io/lucid-YOLO/work_packages/wp-088/) · Needed by [WP-103](https://borda.github.io/lucid-YOLO/work_packages/wp-103/), [WP-137](https://borda.github.io/lucid-YOLO/work_packages/wp-137/)
>
> ⬅️ [WP-101](https://borda.github.io/lucid-YOLO/work_packages/wp-101/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-103](https://borda.github.io/lucid-YOLO/work_packages/wp-103/) ➡️


## WP-102 — validation that is not CPU-bound

<a id="wp-102"></a>

Validation left the GPU idle for most of its wall time, from two defaults nobody had revisited at 1024 px.

The val worker cap of 4 ([WP-073](https://borda.github.io/lucid-YOLO/work_packages/wp-073/#wp-073)) applied even when the operator had named `--data.num_workers 16`. A number an operator states says how much of the machine the run may use, and quietly validating on a quarter of it surfaces as an idle accelerator with **no message at all** — worse than the host-OOM the cap was avoiding, because it is silent. The cap now applies only to a count this class chose itself, and scales with pixel count rather than sitting at a constant reasoned about 640 px samples, where val was "a fraction of the train pipeline's work". At 1024 px the decode *is* the work.

Second, an `obb` run was accumulating the axis-aligned `val/mAP` beside the rotated one — a CPU pass over every detection of every batch, to produce a figure that reads the A44 composition's **pre-rotation** rectangle. It cannot separate a run whose orientations were right from one whose orientations were random. An oriented run now logs `val/rotated_mAP` and no `val/mAP`.

