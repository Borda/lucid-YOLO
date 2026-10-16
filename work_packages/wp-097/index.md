# 📦 WP-097 — `fix(data): report DOTA totals, require only stated ones`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> Engineering log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/engineering/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-056](https://borda.github.io/lucid-YOLO/work_packages/wp-056/), [WP-096](https://borda.github.io/lucid-YOLO/work_packages/wp-096/) · Needed by none
>
> ⬅️ [WP-096](https://borda.github.io/lucid-YOLO/work_packages/wp-096/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-100](https://borda.github.io/lucid-YOLO/work_packages/wp-100/) ➡️


## WP-097 — what R18's published totals actually describe

<a id="wp-097"></a>

R18's headline figures — 2,806 images, 188,282 instances, 15 categories — describe DOTA-v1.0 **whole**, and R18 releases ground truth for two of its three splits. An annotated root therefore holds about two thirds of that instance count, not the published total — the fact `check_dota_root` got backwards, fixed at [wp-097.md#wp-097-engineering](https://borda.github.io/lucid-YOLO/work_packages/wp-097/#wp-097-engineering).


> Engineering log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/engineering/phase-08-oriented-detection/)


## WP-097 — a published total no correct download can reach

<a id="wp-097-engineering"></a>

`check_dota_root` compared its summed train+val totals against R18's whole-dataset figures ([wp-097.md#wp-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/#wp-097)), so it failed on every correct download, at the first command an operator is told to run.

The fix is not a looser tolerance but a different question. A count the caller did not state is **reported**, not compared: on an unstated expectation the check's job is to say what it found, and a number with no stated referent cannot fail. What still fails is a stated expectation that misses — and, new here, a DOTA expectation aimed at a COCO root, which had previously been accepted and silently ignored.

Recorded because the failure was in the gate rather than in the data, and a gate that fires on every correct input is the one shape of gate that teaches its operator to disbelieve gates.

