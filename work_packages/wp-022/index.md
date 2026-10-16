# 📦 WP-022 — `feat(models): dual detection head, reg_max=1`

> Research log · [Phase 3 — Assignment and losses](https://borda.github.io/lucid-YOLO/research/phase-03-assignment-and-losses/)
>
> 🔗 Depends on WP-021 · Needed by [WP-023](https://borda.github.io/lucid-YOLO/work_packages/wp-023/), WP-041
>
> ⬅️ [WP-014](https://borda.github.io/lucid-YOLO/work_packages/wp-014/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-023](https://borda.github.io/lucid-YOLO/work_packages/wp-023/) ➡️


## WP-022 — the dead predictor

<a id="wp-022"></a>

The mechanism and its fix are A30. What belongs here is the failure signature, because it is the most misleading one this project has produced.

With zero-initialized class-conv biases every sigmoid opens at 0.5, the TAL-normalized summed BCE starts near **1.8M nats**, the first MuSGD step at lr 0.01 blows the weights out (**box predictions reached ±6000 px**), TAL then assigns no positives, and the network saturates to an **exact-zero loss with zero gradient**. The progress bar shows `train/loss 0.000` — which reads as a perfectly converged model rather than a dead one. Reproduced identically on CPU and MPS from real COCO batches.

Post-fix verification was a 30-step real-COCO check showing monotone descent **14k → 1.5k** with box predictions bounded within image scale — i.e. the fix was confirmed on the same signal that had lied, not on "the tests pass".

