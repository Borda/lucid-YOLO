# 📦 WP-087e — `refine(ptl): epoch mask mAP during validation`

> Research log · [Phase 7 — Instance segmentation, v0.2.0](https://borda.github.io/lucid-YOLO/research/phase-07-instance-segmentation/)
>
> 🔗 Depends on [WP-087](https://borda.github.io/lucid-YOLO/work_packages/wp-087/) · Needed by none
>
> ⬅️ [WP-087d](https://borda.github.io/lucid-YOLO/work_packages/wp-087d/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-087f](https://borda.github.io/lucid-YOLO/work_packages/wp-087f/) ➡️


## WP-087e — epoch mask mAP during validation

<a id="wp-087e"></a>

A `task="segment"` run logged `val/mAP` and nothing about its masks, so the branch the run exists for moved no epoch metric at all: a run whose masks were degenerate and one whose masks were right produced the same curve, and the first sign of either was an evaluation pass hours after the run ended.

Two measurement decisions are load-bearing and were argued from cost rather than taste:

**Scoring on the prototype grid**, where the loader's ground-truth masks already are. Upsampling both sides by four would cost sixteen times the memory to compare the same two fields at a finer sampling of one boundary. The metric is a proxy in exactly the sense `val/mAP` already is (WP-077); the acceptance figure stays the standalone evaluation.

**A second `MeanAveragePrecision` rather than `iou_type=("bbox", "segm")`** on the existing one. One metric holding both would take each instance's area from whichever frame torchmetrics picked, mis-bucketing the small/medium/large splits by a factor of sixteen. This is a case where the convenient API is wrong for a reason that would never surface as an error — only as quietly wrong area breakdowns.

