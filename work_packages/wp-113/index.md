# 📦 WP-113 — `docs(all): an emoji on every top-level heading`

> Engineering log · [Phase 11 — Rolling, toward v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-11-rolling/)
>
> 🔗 Depends on [WP-067](https://borda.github.io/lucid-YOLO/work_packages/wp-067/) · Needed by [WP-114](https://borda.github.io/lucid-YOLO/work_packages/wp-114/)
>
> ⬅️ [WP-112](https://borda.github.io/lucid-YOLO/work_packages/wp-112/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-113b](https://borda.github.io/lucid-YOLO/work_packages/wp-113b/) ➡️


## WP-113 — an emoji changes the anchor a renderer derives

<a id="wp-113"></a>

Decoration with one load-bearing consequence. GitHub and Python-Markdown both slugify a heading by stripping what is not a word character and joining the rest with hyphens, so `## 🧠 How a modern YOLO works` no longer answers to `#how-a-modern-yolo-works` — the stripped emoji leaves the leading separator behind. The README's own audience table links to four of its H2 sections, and every one of those links would have gone dead in the same commit that made the page prettier, with nothing failing anywhere.

The fix is not to spell the new slug. The two renderers this project publishes through do not have to agree on what a slug becomes, and one that today keeps a leading hyphen is free to trim it tomorrow — so a link written against a derived anchor is a link written against a renderer's current behaviour. Explicit `<a id=>` tags above the four headings are what the research log already uses for exactly this reason: they are greppable, they survive a retitle, and they mean the same thing to both renderers.

**Two gates read heading text and had to stop.** `test_report_sections` pinned four `## `-prefixed literals from the reproduction report and `test_decisions_carry_all_ids` pinned `## ADR-00N`; both would have failed on a purely visual edit while reporting a missing section, which is a gate that misdescribes its own finding. They now compare undecorated titles — a lead token holding no ASCII alphanumeric is decoration, anything else is title — so the assertion is about which sections exist, which is what it was always for.

The emoji themselves are chosen per section and reused across files only where the subject is the same: 🎯 for detection, 🖌️ for segmentation, 🔄 for oriented detection, in the roadmap phase, the log phase, the model card and the training recipe alike. A palette walked in file order would have been faster to produce and would carry no information.

