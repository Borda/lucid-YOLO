# 📦 WP-104 — `perf(eval): match every IoU threshold in one pass`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-103](https://borda.github.io/lucid-YOLO/work_packages/wp-103/) · Needed by [WP-105](https://borda.github.io/lucid-YOLO/work_packages/wp-105/)
>
> ⬅️ [WP-103](https://borda.github.io/lucid-YOLO/work_packages/wp-103/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-105](https://borda.github.io/lucid-YOLO/work_packages/wp-105/) ➡️


## WP-104 — one-pass IoU threshold matching

<a id="wp-104"></a>

The oriented epoch boundary stalled for about a minute after validation reached 100%. Diagnosed before anything was changed: `evaluate_rotated_map` scoring ~10k val tiles at 300 detections each, **178 s at that scale, 58% of it in `_match_image`**.

The ten IoU thresholds share the overlap matrix and the score order and differ only in what counts as a hit, so ten passes re-walked the same detections to consume different subsets of the same ground truths. The availability state becomes `(T, G)` and the walk happens once. A second win falls out: detections reaching no ground truth at the lowest threshold skip the walk entirely — they cannot score, cannot be discarded and consume nothing at any threshold — which on an early-epoch model is nearly the whole loop.

**178 s → 38.6 s**, every pinned value unchanged.

**Rejected: a circumradius prefilter on the IoU kernel.** Written, measured at **12%**, reverted. The residual cost is per-call overhead across ~150k small calls, not pair count, so a cheaper rejection test attacks the wrong term. Carrying an extra geometric shortcut in hand-written rotated geometry is not worth 12% — the code it would live in is the code whose correctness the whole oriented evaluation rests on.

