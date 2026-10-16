# 📦 WP-118 — `refine(docs): split the log by claim, trim the table to match`

> Engineering log · [Phase 11 — Rolling, toward v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-11-rolling/)
>
> 🔗 Depends on WP-003 · Needed by none
>
> ⬅️ [WP-130b](https://borda.github.io/lucid-YOLO/work_packages/wp-130b/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-119](https://borda.github.io/lucid-YOLO/work_packages/wp-119/) ➡️


## WP-118 — one log grew two audiences

<a id="wp-118"></a>

`RESEARCH_LOG.md` had accumulated 85 entries answering two different questions -- what a reproduction claim cost to establish, and what the repo's own tooling cost to build -- under one charter that named only the first. The immediate trigger was smaller: rows 115, 115b and 116 had each grown to a paragraph of debugging narrative in the roadmap's own Scope column, because the log existed but the habit of writing to it first did not.

The split rule: does an entry record what the reproduction *claims* (model, loss, data protocol, eval numbers, paper ambiguity), or *how the repo builds, checks, and publishes itself* (CI, hooks, packaging, licence, formatting, CLI, docs tooling)? Most of the 85 sorted cleanly -- the whole of `Cross-cutting` and Phase 0 turned out to be tooling-kind, the whole of Phase 7 and most of Phase 8/9 stayed fidelity-kind. Five did not sort cleanly, because their finding genuinely was both: [WP-066](https://borda.github.io/lucid-YOLO/work_packages/wp-066/) (export tooling *and* an architecture-fidelity confirmation), [WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/) (a DataLoader bug *and* a measured cost to augmentation diversity), [WP-083](https://borda.github.io/lucid-YOLO/work_packages/wp-083/) (a golden-harness design *and* the frozen numbers it pins), [WP-091b](https://borda.github.io/lucid-YOLO/work_packages/wp-091b/) (a decoder implementation *and* an unsourced-threshold provenance investigation), [WP-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/) (a download-verify bug *and* what R18's published totals actually describe). Those five were not forced into one bucket; each is cut into two pieces, cross-linked both ways, sized to what each piece actually carries rather than split down the middle.

The mechanical risk in a retroactive split of this size is silent loss -- a paragraph dropped in the move, an anchor a roadmap row still cites that no longer resolves. The move itself was scripted rather than hand-edited per entry, asserting the parsed anchor set equalled the classification set before writing anything, so nothing could be silently skipped or double-counted; `scripts/_tests/test_audit_docs_present.py::test_every_log_link_resolves` (renamed from its single-file predecessor) now checks both files' anchors against the roadmap's citations of either.

