# 📦 WP-079 — `fix(data): per-worker, per-epoch augmentation RNG`

> Research log · [Phase 6 — Evaluation, v0.1.0](https://borda.github.io/lucid-YOLO/research/phase-06-evaluation/)
>
> Engineering log · [Phase 6 — Evaluation, v0.1.0](https://borda.github.io/lucid-YOLO/engineering/phase-06-evaluation/)
>
> 🔗 Depends on [WP-014](https://borda.github.io/lucid-YOLO/work_packages/wp-014/), [WP-076](https://borda.github.io/lucid-YOLO/work_packages/wp-076/) · Needed by [WP-082](https://borda.github.io/lucid-YOLO/work_packages/wp-082/), WP-045
>
> ⬅️ [WP-076](https://borda.github.io/lucid-YOLO/work_packages/wp-076/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-082](https://borda.github.io/lucid-YOLO/work_packages/wp-082/) ➡️


## WP-079 — per-worker, per-epoch augmentation RNG: what the collision cost

<a id="wp-079"></a>

Measured exposure: **1/3200 of the intended draws** at 32 workers over 100 epochs. Two DataLoader defects compose multiplicatively and neither is visible in a loss curve — the run trains, it simply trains on far less augmentation variety than the config describes. The mechanism is a `_TrainPipeline` bug fixed the same day; the full story is at [wp-079.md#wp-079-engineering](https://borda.github.io/lucid-YOLO/work_packages/wp-079/#wp-079-engineering).


> Engineering log · [Phase 6 — Evaluation, v0.1.0](https://borda.github.io/lucid-YOLO/engineering/phase-06-evaluation/)


## WP-079 — per-worker, per-epoch augmentation RNG

<a id="wp-079-engineering"></a>

`_TrainPipeline` seeded one generator in the **parent** process. A parent's generator never advances when workers are used, so every worker replayed one identical augmentation-parameter stream — diversity `1/num_workers`. [WP-076](https://borda.github.io/lucid-YOLO/work_packages/wp-076/) then made workers non-persistent, so the pool restarted from that same state every epoch, replaying a single epoch's parameters for the entire run.

What the two defects together cost the training-data variety is measured at [wp-079.md#wp-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/#wp-079).

