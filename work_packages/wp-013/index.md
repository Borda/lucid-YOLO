# 📦 WP-013 — `feat(data): HSV jitter and horizontal flip`

> Research log · [Phase 1 — Data pipeline](https://borda.github.io/lucid-YOLO/research/phase-01-data-pipeline/)
>
> 🔗 Depends on WP-010 · Needed by [WP-014](https://borda.github.io/lucid-YOLO/work_packages/wp-014/), WP-070, [WP-120](https://borda.github.io/lucid-YOLO/work_packages/wp-120/)
>
> ⬅️ [WP-005](https://borda.github.io/lucid-YOLO/work_packages/wp-005/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-014](https://borda.github.io/lucid-YOLO/work_packages/wp-014/) ➡️


## WP-013 — a defect deferred on purpose

<a id="wp-013"></a>

`HorizontalFlip` mirrors a rotated box by negating `theta`, and the commit documents re-canonicalization as a Phase 8 concern rather than doing it. That deferral was correct at the time and became a live defect the moment [WP-058](https://borda.github.io/lucid-YOLO/work_packages/wp-058/#wp-058) removed the guards — the note is what made it findable.

