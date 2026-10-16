# 📦 WP-083 — `test(eval): synthetic-shapes generalization golden`

> Research log · [Phase 6 — Evaluation, v0.1.0](https://borda.github.io/lucid-YOLO/research/phase-06-evaluation/)
>
> Engineering log · [Phase 6 — Evaluation, v0.1.0](https://borda.github.io/lucid-YOLO/engineering/phase-06-evaluation/)
>
> 🔗 Depends on [WP-082](https://borda.github.io/lucid-YOLO/work_packages/wp-082/) · Needed by none
>
> ⬅️ [WP-082](https://borda.github.io/lucid-YOLO/work_packages/wp-082/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-084](https://borda.github.io/lucid-YOLO/work_packages/wp-084/) ➡️


## WP-083 — synthetic-shapes generalization golden

<a id="wp-083"></a>

Frozen at NMS mAP50-95 **0.8823**, E2E **0.8583**, mAP50 **0.9842** — three MPS runs identical to four decimals, 127–169 s wall clock. The spread in wall clock against the exactness of the metric is the point: it is a cheap gate whose *number* is stable even though its *timing* is not. The harness this measures — 2000 generated scenes, held-out scoring, mosaic held on for five of six epochs specifically so a collapsed augmentation RNG ([WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/#wp-079)) would be detectable — is described at [wp-083.md#wp-083-engineering](https://borda.github.io/lucid-YOLO/work_packages/wp-083/#wp-083-engineering).


> Engineering log · [Phase 6 — Evaluation, v0.1.0](https://borda.github.io/lucid-YOLO/engineering/phase-06-evaluation/)


## WP-083 — synthetic-shapes generalization golden

<a id="wp-083-engineering"></a>

The cheap stand-in for a tier run: 2000 generated scenes split 1800/200, n-scale, 6 epochs at 320 px, scored on held-out images through both decode paths. Unlike the overfit-100 gate it never scores what it trained on, so it measures generalization rather than loop composition, and mosaic stays on for five of six epochs so a collapsed augmentation RNG ([WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/#wp-079-engineering)) is detectable rather than merely absent.

The frozen numbers this golden pins are recorded at [wp-083.md#wp-083](https://borda.github.io/lucid-YOLO/work_packages/wp-083/#wp-083).

