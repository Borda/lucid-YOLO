# 📦 WP-099c — `feat(data): YOLO layout in the check module and a worked config`

> Engineering log · [Phase 9 — Inference and generalization](https://borda.github.io/lucid-YOLO/engineering/phase-09-inference/)
>
> 🔗 Depends on [WP-099b](https://borda.github.io/lucid-YOLO/work_packages/wp-099b/) · Needed by [WP-099e](https://borda.github.io/lucid-YOLO/work_packages/wp-099e/), [WP-136](https://borda.github.io/lucid-YOLO/work_packages/wp-136/)
>
> ⬅️ [WP-099d](https://borda.github.io/lucid-YOLO/work_packages/wp-099d/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-099e](https://borda.github.io/lucid-YOLO/work_packages/wp-099e/) ➡️


## WP-099c — a pre-flight that asks the run's question

<a id="wp-099c"></a>

`check_dataset` defaulted to `coco`, which was correct for as long as COCO was the only layout a run could read. [WP-099b](https://borda.github.io/lucid-YOLO/work_packages/wp-099b/) ended that, and left the check answering a question its own run no longer asks: pointed at a YOLO root with no flag, it reported a missing `train2017` for a tree `fit` trains on without complaint. The fix is not a third branch but a deletion — an unstated `--dataset` now runs `detect_layout`, the same probe the datamodule dispatches on, so the pre-flight and the run resolve the root identically or not at all. `DATASETS` is built from `DatasetLayout` members rather than string literals, which is what stops the flag and the probe drifting apart later. DOTA stays outside it deliberately: no run reads a DOTA root — it is tiled to COCO first ([WP-094](https://borda.github.io/lucid-YOLO/work_packages/wp-094/)) — so a probe covering it would advertise a path that does not exist, and the "satisfies no convention" report ends by naming `--dataset dota` for the operator who has one.

Both of the probe's undecidable states are verdicts about a disk, so both come back as report lines and exit 1. [WP-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/) already paid for that lesson once, on the first command an operator runs against a fresh provisioning: a traceback where a report belongs reads as a broken tool rather than an unusable dataset.

The extraction that mattered is `resolve_split_dirs`. A validator that resolves splits its own way validates a tree the reader will not open — and the YOLO layout has exactly the case that makes this bite, a `data.yaml` naming `../valid/images` for a directory no convention row spells (A58). Shared code between a feature and its checker is usually a smell, since the checker then inherits the feature's bugs; here it is the point, because the alternative is a check that passes on a different dataset than the one that trains. `scan_yolo_label_file` is the other half: the same grammar `load_yolo_targets` reads, stopping before the denormalization, so checking a split does not decode every image to obtain a pixel scale it immediately discards. Validation that costs an epoch is validation that gets skipped.

`DotaSplitCheck` became `PairedSplitCheck`, and the naming records what the two layouts actually share. It is not that both are "not COCO" — it is that both put one text file per image beside it, paired by stem, so both need the same two-directional pairing report and the same totals arithmetic. COCO's single manifest cannot be checked that way at all. The shape of the annotation decides the container, and the rename says so where a `Yolo`-prefixed twin would have said nothing.

No register row. A64 is still unconsumed, which is worth stating explicitly because the run made three decisions that look like assumption material and are not: the inference default is grounded in A63 and [WP-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/), the YOLO branch reuses A54–A58 through the reader rather than restating them, and the flag guard extends [WP-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/)'s own rule to one more layout. A register row for a decision that already has a documented parent is noise in the register that matters.

One absence left standing: `splits` is still not a CLI flag, for YOLO as it never was for DOTA, so a root shipping only `train` fails on the missing `val`. That is right for a `fit` pre-flight, which needs both, and wrong for anyone checking a partial download — a small follow-up rather than a defect, and recorded here so it is a choice rather than an oversight.

