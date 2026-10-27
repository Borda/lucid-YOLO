# WP-201 — contract record and user-docs accuracy sweep

> Engineering log · [Phase 19 — Codebase audit remediation](https://borda.github.io/lucid-YOLO/engineering/phase-19-codebase-audit/)
>
> 🔗 Depends on [WP-200](https://borda.github.io/lucid-YOLO/work_packages/wp-200/) · Needed by [WP-202](https://borda.github.io/lucid-YOLO/work_packages/wp-202/), [WP-203](https://borda.github.io/lucid-YOLO/work_packages/wp-203/)
>
> ⬅️ [WP-200](https://borda.github.io/lucid-YOLO/work_packages/wp-200/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-202](https://borda.github.io/lucid-YOLO/work_packages/wp-202/) ➡️


<a id="wp-201"></a>

## WP-201 — the prose that was wrong, corrected in one commit

**Batch 1 of the codebase-audit roadmap.** The audit of 2026-10-21 found the contract text and the user-facing documentation disagreeing with the tree. Nothing here changes code behaviour, and the gate is unchanged: `make gate` is this package's gate, as for every package (AGENTS.md sec. 1, D30), with `make docs` (a strict build) alongside it. Per D30 the audit roadmap lands as one work package and one commit per batch, WP-201 to [WP-205](https://borda.github.io/lucid-YOLO/work_packages/wp-205/); this is the first.

**Contract record.** D30 records the principal's statement that O3's review is done, so O3 is cleared. With the repository public and [WP-125](https://borda.github.io/lucid-YOLO/work_packages/wp-125/) accepted, all three of D18's preconditions hold and D18 is in force. O3's entry under Open items is changed only in its status cell. `AGENTS.md` sec. 1 now says the relaxation is in force instead of "not in force", and `docs/CONTRIBUTING.md` conditions D18 on the same three facts. `AGENTS.md` sec. 2 no longer says the tree holds eight `gpu`/`data` marks: a recount at HEAD found 107 `pytest.mark.gpu` and 5 `pytest.mark.data` across 20 files, and `-m "not gpu and not data"` deselects 360 of 4210 collected tests, so the paragraph's conclusion about what an offline green run covers is reworded. Every other line of `AGENTS.md` is byte-identical to its pre-edit copy.

**Decided by the principal (2026-10-21).** D31 is approved: it supersedes D3's s-scale headline clause (every tier ran at n-scale; the n-scale smoke is the accepted tier), and D3 itself is not edited. The README and `docs/index.md` banner change is approved: the keypoint clause follows D22 and R1 sec. 3.4.2, and the evaluation wording is corrected too, so the loss is taken from RLE and the evaluation is COCO OKS. The banner is the text O3 reviewed, which is why it needed that approval. The install line uses `@main` as its ref, because no release tags exist.

**User-docs sweep.** The model cards state the current accelerator-golden value and tolerance for each task, read from `goldens/gpu/overfit_micro_*.json`. The README matplotlib sentence now says it is a runtime dependency, and `docs/PROVENANCE.md` R29 records the same fact (runtime since [WP-189](https://borda.github.io/lucid-YOLO/work_packages/wp-189/)). The README has one canonical Install section; its ref is `@main`, because no release tag exists. The `docs/index.md` tab count, the CHANGELOG Unreleased headings, the segmentation-report smoke-config note and the README bare doc paths are fixed; the README "Current release" paragraph names 0.8.0 and points at the CHANGELOG, and the index reading table is split into use-it and audit-it. `docs/DATASETS.md` and the README state that the YOLO text reader refuses segmentation and keypoints.

**Limits.** No hook yet asserts that a card figure equals its golden (batch 3). The CHANGELOG 0.9.0 cut stays a separate principal tag decision. `make docs` (`mkdocs build --strict`) and the full pre-commit run pass on the changed files; `make gate` is the gate for this package and no exemption is claimed.

