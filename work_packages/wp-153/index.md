# 📦 WP-153 — `release: v0.6.0 augmentation guard and keypoint parity`

> Engineering log · [Phase 13 — Contributor admission, augmentation guard and keypoint parity, v0.6.0](https://borda.github.io/lucid-YOLO/engineering/phase-13-contributor-admission/)
>
> 🔗 Depends on [WP-150](https://borda.github.io/lucid-YOLO/work_packages/wp-150/), [WP-152](https://borda.github.io/lucid-YOLO/work_packages/wp-152/) · Needed by [WP-153b](https://borda.github.io/lucid-YOLO/work_packages/wp-153b/)
>
> ⬅️ [WP-152b](https://borda.github.io/lucid-YOLO/work_packages/wp-152b/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-153b](https://borda.github.io/lucid-YOLO/work_packages/wp-153b/) ➡️


## WP-153 — the guard is published before anything it guards moves

<a id="wp-153"></a>

0.6.0 releases a regression guard and a task's deployment parity. It adds no model, no loss and no evaluation protocol, which makes it the first release on this train whose subject is entirely the ability to detect a future defect.

**The release order is the whole argument for the phase split.** Every row in Phase 14 deletes a local implementation and replaces it with an upstream one, and what authorizes each of those deletions is the four-tier expectation suite frozen here. A guard that shipped in the same release as the swaps it authorizes would be a guard whose own correctness was only ever demonstrated against the code it was written to replace. Publishing it first means the version a swap is measured against is a released one, reachable by tag rather than by commit range.

**What this release does not claim.** The contributor-admission layers are not built: [WP-141](https://borda.github.io/lucid-YOLO/work_packages/wp-141/) through [WP-144](https://borda.github.io/lucid-YOLO/work_packages/wp-144/) stand at ⬜, [WP-141](https://borda.github.io/lucid-YOLO/work_packages/wp-141/) being a principal row with three preconditions outside the roadmap — O3 cleared, the repository made public, [WP-125](https://borda.github.io/lucid-YOLO/work_packages/wp-125/) accepted. So the phase ships its release with four rows trailing, which is not new; 0.5.0 did the same with [WP-125](https://borda.github.io/lucid-YOLO/work_packages/wp-125/). A68, A69 and A70 remain open and each still names an experiment nobody has run. No trained weights ship (D14), and the guard is offline arithmetic on synthetic geometry rather than any statement about accuracy.

**The freeze is larger than 0.5.0's by exactly one file.** `make freeze-goldens MINOR=0.6` copies the live metric sets into `goldens/frozen/0.6/`; the new one is `aug_invariants.json`, whose first freeze this is. It is worth naming which of the four tiers reaches the frozen set and which does not: the golden carries tier-B scalar aggregates, tiers A and C live in pytest literals and are frozen by being written down, and tier D deliberately reaches neither — its values pin whichever sampler is installed and are re-frozen at every swap, so freezing them into a release snapshot would state a cross-release regression contract over exactly the numbers that are expected to move.

**One stale claim corrected at the source.** `README.md` said the export gate asserts the NMS-free property "on all three tasks". It runs four as of [WP-151](https://borda.github.io/lucid-YOLO/work_packages/wp-151/), and the sentence is the project's own headline architectural claim, so it is the one place where an off-by-one count is a claim about the deliverable rather than about the documentation. The keypoint model card needed nothing: its "the other three tasks" phrasings are comparisons, and comparisons of one task against three others stay correct at four.

No tag is cut here and no distribution is published, as at 0.4.0 and 0.5.0. `release.yml` fires on a pushed `v0.*` tag and re-runs the guard, which refuses a tag whose changelog section is missing or whose gate is red; both are the operator's to trigger. `v0.4.0` and `v0.5.0` are also both still uncut, so the train has now shipped three release commits ahead of its own tags.

