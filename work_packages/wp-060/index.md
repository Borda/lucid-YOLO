# 📦 WP-060 — `feat(losses): square-object angle loss`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-059](https://borda.github.io/lucid-YOLO/work_packages/wp-059/) · Needed by [WP-061](https://borda.github.io/lucid-YOLO/work_packages/wp-061/)
>
> ⬅️ [WP-059](https://borda.github.io/lucid-YOLO/work_packages/wp-059/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-061](https://borda.github.io/lucid-YOLO/work_packages/wp-061/) ➡️


## WP-060 — square-object angle loss

<a id="wp-060"></a>

R1 Table 11's own numbers set `lambda = 3`: **50.2 mAP against 49.0 with no angle loss, and 47.1 at `lambda = 5`** — i.e. the paper's own ablation shows the term can be worse than omitting it when mis-weighted. That is the context in which [WP-093](https://borda.github.io/lucid-YOLO/work_packages/wp-093/#wp-093) later found the gain A22 assumed was destabilising.

Two floating-point contracts rather than approximations: Eq. 14's `round` tie rule is unstated, so the wrap settles ties upwards into the half-open `[-pi/2, pi/2)` and is exactly idempotent; and the residual is folded once more into `[-pi/4, pi/4)`, a **Sterbenz-exact** shift under which the two representatives [WP-055](https://borda.github.io/lucid-YOLO/work_packages/wp-055/) leaves a square score bit-identically.

