# 📦 WP-059 — `feat(losses): ProbIoU rotated loss`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-055](https://borda.github.io/lucid-YOLO/work_packages/wp-055/) · Needed by [WP-060](https://borda.github.io/lucid-YOLO/work_packages/wp-060/)
>
> ⬅️ [WP-058](https://borda.github.io/lucid-YOLO/work_packages/wp-058/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-060](https://borda.github.io/lucid-YOLO/work_packages/wp-060/) ➡️


## WP-059 — ProbIoU

<a id="wp-059"></a>

R17's formulas are *evaluated* through cancellation-free identities (2×2 adjugate linearity for `B1`'s numerator, the `det(S1+S2)` expansion, `det S = w²h²/144`), and that is what makes float32 sufficient. The measurement that forced it: the **literal transcription loses 377% of `B_D` at a 1e-4 perturbation and returns `NaN` on parts of a 1000:1 aspect sweep**. Float64 internals are not an available fallback, because a regression loss runs on the training device and D12c puts training on MPS, which has no float64.

Both of R17's losses ship side by side — bounded Hellinger `L1` and unbounded `L2 = B_D` — because R17 suggests starting on the second and switching. Which one the oriented path uses, and the measured deficit of the alternative, is A49.

A DoD was **moved** here rather than met: the package originally owed a 1e-4 shapely agreement test. ProbIoU is a Hellinger-distance similarity between two Gaussians and R17 asserts no equality with polygon overlap, so that test could not pass and a correlation threshold would have been invented. The shapely oracle belongs to [WP-063](https://borda.github.io/lucid-YOLO/work_packages/wp-063/)'s exact rotated IoU (A24); ProbIoU got its own exact oracles instead (quadrature of the definition at 2e-14, literal float64 transcription at 1e-12). Recorded because "the DoD was wrong" is a legitimate outcome and the alternative — a tolerance chosen until it passes — is the failure this project is most exposed to.

