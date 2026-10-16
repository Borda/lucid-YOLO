# 📦 WP-103 — `fix(ptl): a val worker count that fits the host`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-102](https://borda.github.io/lucid-YOLO/work_packages/wp-102/) · Needed by [WP-104](https://borda.github.io/lucid-YOLO/work_packages/wp-104/)
>
> ⬅️ [WP-102](https://borda.github.io/lucid-YOLO/work_packages/wp-102/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-104](https://borda.github.io/lucid-YOLO/work_packages/wp-104/) ➡️


## WP-103 — a val worker count that fits the host

<a id="wp-103"></a>

**This package exists to fix a defect the previous one introduced.** [WP-102](https://borda.github.io/lucid-YOLO/work_packages/wp-102/) let a named `--data.num_workers` reach the validation loader and deleted the guard rather than the silence.

The arithmetic it deleted: at 1024 px a batch of 64 stacked float32 RGB images is `4 × 3 × 64 × 1024²` = **805 MB**. A named 32 workers at prefetch 2 queues about **51 GB for validation alone**, beside a training pool still resident at the epoch boundary. The result was the observed host-RAM crash on the first oriented tier run.

The division that was wrong the first time and is now explicit: an operator states parallelism with that flag; **the gigabytes it costs are this class's arithmetic**. A count inherited from a named train count is bounded by the memory budget and warns with the number and the override flag; a count stated as `--data.val_num_workers` is bounded by nothing, which is where an operator asks for the whole machine.

The general lesson, recorded because it was a real mistake and not a hypothetical: the principle behind [WP-102](https://borda.github.io/lucid-YOLO/work_packages/wp-102/) was right — a stated number should not be silently reduced — and the implementation deleted the wrong thing. Removing the *silence* was the fix; removing the *bound* was not.

