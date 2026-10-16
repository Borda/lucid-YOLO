# 📦 WP-164 — `perf(ptl): the mask stack is not copied to drop nothing`

> Engineering log · [Phase 15 — Audit remediation, v0.8.0](https://borda.github.io/lucid-YOLO/engineering/phase-15-audit-remediation/)
>
> 🔗 Depends on [WP-163](https://borda.github.io/lucid-YOLO/work_packages/wp-163/) · Needed by [WP-165](https://borda.github.io/lucid-YOLO/work_packages/wp-165/)
>
> ⬅️ [WP-163](https://borda.github.io/lucid-YOLO/work_packages/wp-163/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-165](https://borda.github.io/lucid-YOLO/work_packages/wp-165/) ➡️


## WP-164 — the copy that dropped nothing, and the three faults that hid it

<a id="wp-164"></a>

This row lands the candidate [WP-163](https://borda.github.io/lucid-YOLO/work_packages/wp-163/) rejected one entry above, and the more useful half of it is why that rejection was wrong. The rejection is left standing there per this log's convention of not editing a historical paragraph to match a later correction; what follows is the correction.

**What the change is.** `_update_val_segm` decoded every one of its 100 detection rows and then wrote `masks[keep]`. In the deployed configuration that filter drops **nothing**: the validation decoder is `TopKDecoder()`, whose `conf_threshold` defaults to `0.0`, and a 640-px canvas carries more anchors than the cap of 300, so no row is padding and no row is zeroed — every score is a sigmoid output, and a sigmoid is never zero. The boolean index was therefore an unconditional second copy of a 100x160x160 boolean stack, 2.56 MB per image, to select all of it. Counting the live rows on `cpu_detections` — a host copy the loop already has — and slicing the decode to that count removes the copy. Where a caller *does* set a confidence threshold the same slice shrinks the decode itself, but that is the secondary effect, not the reason this lands.

**Why the slice is sound.** It rests on the survivors being a leading run, which is a property of the decoder rather than of this call site, and three separate mechanisms could have broken it: `o2o_topk_with_indices` ranks score-descending, `pad_detections` appends its zero rows *after* the ranked ones, and `_zero_below_threshold` cuts a descending column from some point on. All three are suffix operations, so `score > 0` can only ever be a prefix. Nothing stated that anywhere, which would have made a future reordering of the decode break this caller silently, so `test_scored_rows_leading_run` now states it against `TopKDecoder` itself — with `k` above the anchor count and a threshold set, so padding and the cut are both live in the same case.

**Measured on the shape that actually runs.** 16 images, 100 rows each, all live — the deployed case, not a favourable one: CPU **27.68 ms to 24.44 ms (+11.7%)** and MPS **11.54 ms to 9.76 ms (+15.4%)**, medians of five interleaved trials of 20 timed runs each, every individual trial positive on both devices. With a threshold set so only a handful of rows survive, the same change is +62% to +93%. No golden moves and no metric value changes: the rows fed to `MeanAveragePrecision` are the same rows in the same order.

**The three faults that produced [WP-163](https://borda.github.io/lucid-YOLO/work_packages/wp-163/)'s opposite conclusion.** They compounded, and each is worth naming because each is a benchmark-design error rather than a measurement accident. First, the harness's patched arm reconstructed the full 100-row tensor with a scatter (`out[:, keep] = decoded`) that the real call site never performs — it measured a formulation nobody proposed, and that scatter is where MPS lost. Second, the proxy decoded on-device and stopped there, omitting the `.cpu()` transfer the real loop performs, so the thing most likely to dominate was outside the timed region. Third, MPS timings drift downward across a process's life well past five warm-ups; timing base first and patched second, once, hands the first arm the cold clock, and trial 0 of every re-run reproduces exactly that — +41% falling to −5% by trial 4 on unchanged code. Interleaving arm order across trials and taking medians of trials makes the drift visible instead of letting it sign the result. The general lesson, which is the one worth keeping: benchmark the call site's own work profile, not a sketch of it, and on a lazily-executing accelerator treat a single ordered A-then-B run as unmeasured.

**Unverified on CUDA.** The mechanism — one fewer copy of a boolean tensor, no data-dependent indexing — has no reason to invert there, and this project has no CUDA device to confirm it on. Stated rather than assumed.

