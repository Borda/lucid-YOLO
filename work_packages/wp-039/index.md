# 📦 WP-039 — `feat(ptl): deterministic checkpoint and resume`

> Research log · [Phase 5 — Lightning training loop](https://borda.github.io/lucid-YOLO/research/phase-05-training-loop/)
>
> 🔗 Depends on [WP-038](https://borda.github.io/lucid-YOLO/work_packages/wp-038/) · Needed by [WP-040](https://borda.github.io/lucid-YOLO/work_packages/wp-040/)
>
> ⬅️ [WP-038](https://borda.github.io/lucid-YOLO/work_packages/wp-038/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-040](https://borda.github.io/lucid-YOLO/work_packages/wp-040/) ➡️


## WP-039 — a package whose finding is that no code was needed

<a id="wp-039"></a>

Deterministic checkpoint and resume was proved rather than implemented: a four-epoch reference run against a run stopped after two epochs and resumed produces the same per-step loss trajectory over the post-resume epochs (rel 1e-4), with `global_step`, `current_epoch` and every MuSGD momentum buffer restored. **No source changes were needed** — Lightning's checkpointing plus existing module and optimizer state already round-trips.

Two scoping choices made that provable: all runs share `max_epochs`, so the progressive-loss alpha ramp (a pure function of `current_epoch / max_epochs`, recomputed each epoch) is identical either side of the resume; and the data stream is a fixed unshuffled in-memory dataset, keeping sampler-state restoration out of scope. Without the first, a correct resume would still have produced divergent losses.

