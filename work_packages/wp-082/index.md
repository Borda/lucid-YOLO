# 📦 WP-082 — `test(losses): objective invariant gate`

> Research log · [Phase 6 — Evaluation, v0.1.0](https://borda.github.io/lucid-YOLO/research/phase-06-evaluation/)
>
> 🔗 Depends on WP-078, [WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/) · Needed by [WP-083](https://borda.github.io/lucid-YOLO/work_packages/wp-083/)
>
> ⬅️ [WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-083](https://borda.github.io/lucid-YOLO/work_packages/wp-083/) ➡️


## WP-082 — objective invariant gate

<a id="wp-082"></a>

Written after WP-078 and [WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/) survived 474 tests and three tier runs, which is the fact that justifies the package: the suite was large and did not contain the shape of test that would have caught either.

The useful negative result is the test **not** written. An obvious guard for WP-078's defect is "the L1 term must not exceed some share of the objective". On a converged fixture the broken pixel frame reads **10.7%** against the correct frame's **1.5%** — both comfortably under any threshold anyone would have picked. Such a test would look like protection and catch nothing. The gate pins the coordinate frame directly instead (`1/stride` against the pixel frame, per level) and its scale on a near-converged state (0.75 stride-units against 5.99 broken).

