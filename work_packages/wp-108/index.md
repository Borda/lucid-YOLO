# 📦 WP-108 — `docs(roadmap): the Scope column says what a package does`

> Engineering log · [Phase 9 — Inference and generalization](https://borda.github.io/lucid-YOLO/engineering/phase-09-inference/)
>
> 🔗 Depends on [WP-107](https://borda.github.io/lucid-YOLO/work_packages/wp-107/) · Needed by none
>
> ⬅️ [WP-107](https://borda.github.io/lucid-YOLO/work_packages/wp-107/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-074](https://borda.github.io/lucid-YOLO/work_packages/wp-074/) ➡️


## WP-108 — a cell grows where its section is missing

<a id="wp-108"></a>

The Scope column had drifted to a 286-character median with a 1709-character worst case, and the drift is not random: of the 32 rows over 400, **12 had no RESEARCH_LOG section at all**. A row whose package never got a section is a row where the only place to write down why a thing was done is the row itself, so the reasoning lands there and stays. The length is the symptom; the missing section is the cause, which is why this package wrote the 12 sections before compressing anything.

The gate was considered and rejected, and the reasoning is worth keeping because it runs against this project's usual instinct. Every other convention here is enforced by a meta test — status icons, six columns, the header's package count, the register ids — and the argument for those is that a documented rule with no gate decays. A length gate is different in kind: the others fail on a fact about the file, while a length gate fails on a judgement about how much a row needs to say, and the cheapest way to make it pass is to delete a claim. A rule whose easiest satisfaction is losing substance should not be automatic. So the median is written into the header as guidance, and the enforcement is a reader noticing that a cell runs several times its column's median.

What the compression turned up is that most of the overflow was already recorded somewhere with an owner. Twenty rows needed no new writing at all: their long clauses restate what the log section, or the assumption register, already carries — A40's exact-under-similarity split, A59's core-ownership rule, A24's polygon measure. That is the duplication the header paragraph has always warned about, visible for the first time because something counted it. Three claims had no other home and were moved rather than cut, the sharpest being [WP-064](https://borda.github.io/lucid-YOLO/work_packages/wp-064/)'s: that the release discloses its per-tile figures in the report, the model card and the README. A disclosure claim living only in a roadmap cell is one edit from disappearing.

Two rows deliberately got no section: 091d and 091e were ⬜, and their content was a finding of 091c's, recorded there. A log section for unstarted work would be a stub asserting what the package will conclude, which is the opposite of what this file is for — so both rows pointed their `· log` tail at `#wp-091c` instead, and the sections got written when the packages produced something to record, which is the paragraph below this one.

