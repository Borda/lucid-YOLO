# 📦 WP-030 — `test(train): single-batch overfit and gradient flow`

> Research log · [Phase 3 — Assignment and losses](https://borda.github.io/lucid-YOLO/research/phase-03-assignment-and-losses/)
>
> 🔗 Depends on WP-029 · Needed by WP-034
>
> ⬅️ [WP-023](https://borda.github.io/lucid-YOLO/work_packages/wp-023/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-033](https://borda.github.io/lucid-YOLO/work_packages/wp-033/) ➡️


## WP-030 — the Phase 3 exit gate

<a id="wp-030"></a>

Proves the assignment-plus-loss stack is optimizable end to end **without the head**, which had not landed yet: learnable prediction tensors (logits, and anchor-centred softplus ltrb boxes so assigned ground truths are exactly representable) overfit a real letterboxed fixture batch for 200 Adam steps. Loss falls **182.3 → 6.9** with strictly decreasing window checkpoints, every value and gradient finite, and both alpha extremes (0 and 1) train.

Constructing the learnable tensors so the target is exactly representable is what makes a failure here attributable to the objective rather than to the parameterization.

