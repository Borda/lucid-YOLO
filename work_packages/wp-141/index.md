# 📦 WP-141 — `docs: contributor guide and the tracking relaxation`

> Engineering log · [Phase 13 — Contributor admission, augmentation guard and keypoint parity, v0.6.0](https://borda.github.io/lucid-YOLO/engineering/phase-13-contributor-admission/)
>
> 🔗 Depends on [WP-125](https://borda.github.io/lucid-YOLO/work_packages/wp-125/) · Needed by [WP-142](https://borda.github.io/lucid-YOLO/work_packages/wp-142/), [WP-143](https://borda.github.io/lucid-YOLO/work_packages/wp-143/), [WP-144](https://borda.github.io/lucid-YOLO/work_packages/wp-144/)
>
> ⬅️ [WP-140b](https://borda.github.io/lucid-YOLO/work_packages/wp-140b/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-142](https://borda.github.io/lucid-YOLO/work_packages/wp-142/) ➡️


## WP-141 — the guide is written; the condition it waits on is not this row's

<a id="wp-141"></a>

`docs/CONTRIBUTING.md`, three issue templates, and the sentence in `AGENTS.md` that D18 was always going to flip. The file sits under `docs/` rather than at the repository root deliberately: `docs/` is one of the three locations GitHub resolves `CONTRIBUTING.md` from, and it is the only one that is also inside the mkdocs tree, so a single file satisfies the platform and the docs-site audit at once instead of a root copy drifting from a nav copy.

**What the row could not do, and did not pretend to.** D18's relaxation is conditional on two things: the reproduction report carrying all four accepted tiers, and *the repository being public*. The first is satisfied — the fourth tier was accepted at [WP-125](https://borda.github.io/lucid-YOLO/work_packages/wp-125/). The second is not: O3 is still `blocked-on-principal` in the open-items register, and its own deadline is "before the repository or any tag is public", so it is precisely the thing standing in the way. `AGENTS.md` used to say "Until WP-141 lands, the contract above applies in full", which names the wrong gate; it now names the repository's visibility, because that is what actually decides. The relaxation is written down and gated rather than in force, and no row here changes that.

**Three layers, and the table saying what each one is not.** The guide's central table is the part worth defending: DCO establishes ownership and says nothing about derivation; the licence attestation covers derivation and produces nothing reviewable; the provenance trailers produce the reviewable artifact and say nothing about the right to submit. Each column of "what it does not" is what justifies the next layer existing, and a contributor reading only the middle row would reasonably conclude that a checkbox is the whole system.

**The clean-room rule is stated at its real width.** [WP-143](https://borda.github.io/lucid-YOLO/work_packages/wp-143/) widens the pull-request template; the guide states the same thing in prose, because the template is a checklist and a checklist is a poor place to learn a rule. Ultralytics is enumerated as the instance a YOLO contributor reaches for by reflex, not as the definition, and the guide says explicitly that a contributor who has already read one should say so rather than conceal it — an undisclosed derivation is the only version of this that cannot be repaired.

**No unverifiable link ships.** `config.yml` would naturally carry `contact_links` to the guide and the security advisory form, both absolute URLs into a repository that is not yet public, so neither could be fetched and checked before shipping. Rather than ship two links on the assumption they will resolve later, the file carries none and the pointers moved into the templates as prose. The one external link in the guide, the Developer Certificate of Origin, was fetched and its clause (a) read before being cited.

