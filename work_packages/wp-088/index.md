# 📦 WP-088 — `feat(ptl): oriented-detection training path`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-063](https://borda.github.io/lucid-YOLO/work_packages/wp-063/) · Needed by [WP-094](https://borda.github.io/lucid-YOLO/work_packages/wp-094/), [WP-102](https://borda.github.io/lucid-YOLO/work_packages/wp-102/), [WP-093](https://borda.github.io/lucid-YOLO/work_packages/wp-093/), [WP-136](https://borda.github.io/lucid-YOLO/work_packages/wp-136/)
>
> ⬅️ [WP-063](https://borda.github.io/lucid-YOLO/work_packages/wp-063/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-094](https://borda.github.io/lucid-YOLO/work_packages/wp-094/) ➡️


## WP-088 — oriented training path

<a id="wp-088"></a>

This row was **left open with its criterion unmet**, which is the part worth recording. The training path verified clean — gradient reaches both branches' angle stems, each oriented gain has leverage, the detection step reproduces a snapshot replayed from a clean `fcf3040` export — and A49/A50 were measured rather than argued (their alternatives cost 0.5046 and 0.7112 rotated mAP50). But the overfit floor cleared on **0 of 3 seeds, mean 0.840, spread 0.091**.

The package stayed open rather than being closed on "the code is right", because the code being right was not what the DoD asked. It closed only when [WP-093](https://borda.github.io/lucid-YOLO/work_packages/wp-093/#wp-093)'s revised gain took the gate to 0.9390 at the recipe's own seed.

The criterion is met as written and not more than that: the five-seed picture behind that number is 4 of 5, which lives in [WP-093](https://borda.github.io/lucid-YOLO/work_packages/wp-093/)'s record rather than being folded into this one.

