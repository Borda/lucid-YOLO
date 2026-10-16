# ⚖️ Phase 3 — Assignment and losses


## WP-022 — the dead predictor

<a id="wp-022"></a>

The mechanism and its fix are A30. What belongs here is the failure signature, because it is the most misleading one this project has produced.

With zero-initialized class-conv biases every sigmoid opens at 0.5, the TAL-normalized summed BCE starts near **1.8M nats**, the first MuSGD step at lr 0.01 blows the weights out (**box predictions reached ±6000 px**), TAL then assigns no positives, and the network saturates to an **exact-zero loss with zero gradient**. The progress bar shows `train/loss 0.000` — which reads as a perfectly converged model rather than a dead one. Reproduced identically on CPU and MPS from real COCO batches.

Post-fix verification was a 30-step real-COCO check showing monotone descent **14k → 1.5k** with box predictions bounded within image scale — i.e. the fix was confirmed on the same signal that had lied, not on "the tests pass".



## WP-030 — the Phase 3 exit gate

<a id="wp-030"></a>

Proves the assignment-plus-loss stack is optimizable end to end **without the head**, which had not landed yet: learnable prediction tensors (logits, and anchor-centred softplus ltrb boxes so assigned ground truths are exactly representable) overfit a real letterboxed fixture batch for 200 Adam steps. Loss falls **182.3 → 6.9** with strictly decreasing window checkpoints, every value and gradient finite, and both alpha extremes (0 and 1) train.

Constructing the learnable tensors so the target is exactly representable is what makes a failure here attributable to the objective rather than to the parameterization.

