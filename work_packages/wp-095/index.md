# 📦 WP-095 — `feat(eval): per-tile oriented acceptance instrument`

> Engineering log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/engineering/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-094](https://borda.github.io/lucid-YOLO/work_packages/wp-094/) · Needed by [WP-096](https://borda.github.io/lucid-YOLO/work_packages/wp-096/), WP-098
>
> ⬅️ [WP-094](https://borda.github.io/lucid-YOLO/work_packages/wp-094/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-096](https://borda.github.io/lucid-YOLO/work_packages/wp-096/) ➡️


## WP-095 — a run monitor is not an acceptance instrument

<a id="wp-095"></a>

The oriented tier reached its release row with no way to produce an acceptance number. What existed was the epoch metric `val/rotated_mAP` ([WP-102](https://borda.github.io/lucid-YOLO/work_packages/wp-102/#wp-102)), and that is a run monitor: it scores whatever the training loader hands it, under torchmetrics' own protocol rather than A46's recall grid, A47's cap and A48's difficult rule. Two figures computed under two protocols are not the same measurement, and the one a release quotes has to be the registered one.

One decision inside the instrument is worth recording. It scores in **letterbox coordinates, with the ground truth letterboxed alongside**, rather than inverting the predictions back to tile coordinates first. Both sides then pass through one geometry, which makes the comparison an equality rather than a proxy — the inverse is where a frame error would enter, and here it is not on the path at all. The same reasoning put checkpoint loading in `lucid_yolo.eval.checkpoint` instead of a second copy: one loader, one place for a task to be read wrong.

