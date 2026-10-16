# 📦 WP-143 — `docs: the licence attestation covers every non-permissive source`

> Engineering log · [Phase 13 — Contributor admission, augmentation guard and keypoint parity, v0.6.0](https://borda.github.io/lucid-YOLO/engineering/phase-13-contributor-admission/)
>
> 🔗 Depends on [WP-141](https://borda.github.io/lucid-YOLO/work_packages/wp-141/) · Needed by none
>
> ⬅️ [WP-142](https://borda.github.io/lucid-YOLO/work_packages/wp-142/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-144](https://borda.github.io/lucid-YOLO/work_packages/wp-144/) ➡️


## WP-143 — one instance of a rule, stated as though it were the rule

<a id="wp-143"></a>

The pull-request template's clean-room line read "no denylisted source (Ultralytics repo, mirror, package copy, or `docs.ultralytics.com`) was consulted". Every word true, and the rule it enforces is much wider: D13's admissible set is the permissive allowlist — MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC — so what a contributor attests to is not having copied from anything outside it. A contributor reading the old line would reasonably conclude a GPL detector was fair game, which is not a misreading; it is what the line said.

The checklist now enumerates copyleft (AGPL, GPL, LGPL, SSPL), source-available (BSL, Elastic, PolyForm), paid or proprietary source of any kind, and any source whose licence cannot be read — D17's case, where unreadable is not permissive. Ultralytics stays, moved to the end and labelled as the instance a YOLO contributor reaches for by reflex rather than as the definition. A second item asks for the sign-off [WP-142](https://borda.github.io/lucid-YOLO/work_packages/wp-142/) checks, so the template and the CI job agree about what a pull request owes.

**The row's real deliverable is that it cannot narrow again silently.** A checklist is prose a maintainer edits, and the failure mode is not a wholesale revert — which is obvious — but one family falling out of a rewrite. `check_pull_request_attestation_covers_the_allowlist` in `audit_docs_present.py` pins each family and both clauses, so a dropped `SSPL` fails the gate naming `SSPL` rather than passing as an attestation that still looks thorough. That the previous line survived eight months while meaning all of this is the argument for gating it rather than trusting the next edit.

Two notes on the check itself. It takes `docs_dir` and discards it, so it matches the signature every other check in that module carries and `_CHECKS` stays a plain tuple rather than growing a dispatch. And the two non-licence conditions — "proprietary", "cannot be read" — are matched as substrings against the lowercased text, because neither is a licence name and both are exactly the kind of clause that falls out of a rewrite unnoticed.

