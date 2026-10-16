# 📦 WP-063 — `feat(eval): rotated mAP on DOTA val`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-062](https://borda.github.io/lucid-YOLO/work_packages/wp-062/) · Needed by [WP-088](https://borda.github.io/lucid-YOLO/work_packages/wp-088/), [WP-064](https://borda.github.io/lucid-YOLO/work_packages/wp-064/), WP-092
>
> ⬅️ [WP-062](https://borda.github.io/lucid-YOLO/work_packages/wp-062/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-088](https://borda.github.io/lucid-YOLO/work_packages/wp-088/) ➡️


## WP-063 — four constants a library would otherwise have chosen

<a id="wp-063"></a>

R1 reports rotated mAP50-95 on DOTA-v1.0 val and defines none of the machinery that produces the number — not the IoU, the matching rule, the recall interpolation, the detection cap nor the class averaging (A24). Four constants therefore had to come from somewhere, and what is recorded here is that they come from the register rather than from whichever library the accumulator was built on: the 101-point recall grid (A46), the 300-detection per-image cap matching what the head emits (A47), the exclusion of classes with no non-difficult ground truth, and R18's difficult rule with an explicit precedence against the argmax match (A48).

A default inherited silently from a dependency is an unregistered assumption wearing a library's name, and it moves a published figure exactly as far as a registered one does. The difference is only that nobody can find it later.

