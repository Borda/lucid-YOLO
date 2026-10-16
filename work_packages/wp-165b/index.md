# 📦 WP-165b — `docs: Phase 15 opened and the audit grouped into rows`

> Engineering log · [Phase 15 — Audit remediation, v0.8.0](https://borda.github.io/lucid-YOLO/engineering/phase-15-audit-remediation/)
>
> 🔗 Depends on [WP-165](https://borda.github.io/lucid-YOLO/work_packages/wp-165/) · Needed by [WP-166](https://borda.github.io/lucid-YOLO/work_packages/wp-166/), [WP-168](https://borda.github.io/lucid-YOLO/work_packages/wp-168/), [WP-169](https://borda.github.io/lucid-YOLO/work_packages/wp-169/), [WP-170](https://borda.github.io/lucid-YOLO/work_packages/wp-170/), [WP-171](https://borda.github.io/lucid-YOLO/work_packages/wp-171/), [WP-172](https://borda.github.io/lucid-YOLO/work_packages/wp-172/), [WP-173](https://borda.github.io/lucid-YOLO/work_packages/wp-173/), [WP-174](https://borda.github.io/lucid-YOLO/work_packages/wp-174/)
>
> ⬅️ [WP-165](https://borda.github.io/lucid-YOLO/work_packages/wp-165/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-166](https://borda.github.io/lucid-YOLO/work_packages/wp-166/) ➡️


## WP-165b — a phase opened from an adversarial read, and what the grouping is for

<a id="wp-165b"></a>

Phase 14 ended at its own release and six rows kept landing after it, belonging to no phase. They open Phase 15 rather than trailing a closed one, and the header that read `WP-154…165` now reads `WP-154…159`.

**What the audit was.** Seven readers over separate dimensions — numerics, data, models, eval, gate, docs, provenance — plus one external reviewer over all of it, each working from the tree at `3ba7880` and writing independently. After cross-source de-duplication (roughly 30 findings arrived from two or three readers at once, which is the signal that a finding is about the code rather than about a reader's angle) 125 distinct gaps remain, alongside 26 boundaries that are correct but under-stated. The audit file itself is not committed: it lives under `.plans/`, which is gitignored, and the roadmap cells carry its per-finding ids so a row can be traced back to it.

**Why the rows group by file surface rather than by finding.** The obvious grouping is one row per reader, or one row per severity. Both produce rows that touch the whole tree, and a row that touches the whole tree cannot be reverted without reverting everything else that happened to ride along. Grouping by the surface each finding touches means a row is one revertable change to one part of the tree, and the price — a critical and a low-severity item landing in the same commit because they live in the same module — is worth paying. [WP-166](https://borda.github.io/lucid-YOLO/work_packages/wp-166/) is the clearest case: two critical findings and five consistency items, all in `eval/coco_eval.py`, one commit.

**What the clean-room read found.** Nothing. Three readers hunted contamination independently: no denylisted source is referenced anywhere, the licence audit is clean over 107 distributions and 344 shipped binaries, and the naming discipline holds. Every one of the 125 findings is about machinery certifying less than the prose beside it claims — which is a different problem, and the one this phase is about.

**Two rows do not open with a commit.** [WP-169](https://borda.github.io/lucid-YOLO/work_packages/wp-169/) and [WP-172](https://borda.github.io/lucid-YOLO/work_packages/wp-172/) each move a value pinned in `goldens/frozen/0.2` through `0.7`. Under sec. 7 that is a standing prohibition and under sec. 4 an escalation trigger, and the two overrides already on record ([WP-154b](https://borda.github.io/lucid-YOLO/work_packages/wp-154b/), [WP-155](https://borda.github.io/lucid-YOLO/work_packages/wp-155/)) were both written down as dated exceptions rather than precedent. Each of those rows opens with its own `docs/ESCALATION.md` entry and a ruling. Three further rows need an accelerator and cannot go green offline.

