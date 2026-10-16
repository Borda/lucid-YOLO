# 📦 WP-140b — `docs: the principal, the contract's end, and two phases`

> Engineering log · [Phase 12 — Keypoint detection, v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-12-keypoints/)
>
> 🔗 Depends on [WP-140](https://borda.github.io/lucid-YOLO/work_packages/wp-140/) · Needed by none
>
> ⬅️ [WP-140](https://borda.github.io/lucid-YOLO/work_packages/wp-140/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-141](https://borda.github.io/lucid-YOLO/work_packages/wp-141/) ➡️


## WP-140b — the contract names a role, and says when it ends

<a id="wp-140b"></a>

Lettered off 140: governance work that follows the release rather than any package. Three things land together because each is the reason for the next.

**`[HUMAN]` becomes `[PRINCIPAL]`.** The tag never meant *a person rather than a machine*; it meant *the party who drives the project rather than the party who executes it*, and every use of it is a call that comes from intent, ownership or risk appetite — a release, a push, a GPU budget, an acceptance. `[OPERATOR]` was tried first and rejected on reading: an operator is one who *operates the machine*, which names the executing half, exactly the half the tag does not mean. `principal` is the term of art for the party whose intent an agent serves, and it survives the human/bot question permanently, because a principal can itself be an agent one level up. The rename is governance-sense only. Anatomical uses stay untouched — "COCO human pose", "human anatomy", "human-readable", "human-drawn polygons" — as do the pre-existing uses of `operator` meaning *whoever runs the command*, which now read as the useful contrast rather than as a competing sense of one word: the principal decides, the operator executes.

**The WP contract is a reproduction instrument, and instruments end (D18).** One WP, one commit, one roadmap row exists so that a from-scratch reproduction of three papers is auditable row by row. Once the reproduction report carries all four accepted tiers that purpose is served, and continuing to demand a tracked row for a typo fix would be ceremony rather than evidence. What relaxes is only the tracking: a change altering no shipped behaviour, no public symbol, no golden and no documented assumption becomes an ordinary gated commit, and the gate itself never relaxes. What restarts it is a new **task** — a fifth head after detection, segmentation, oriented detection and keypoints — because a new task is a new reproduction claim and D2's standard applies to it exactly as it applied to the first four.

**Admission is three layers, because a signature is not a detector.** The exposure is not one vendor. It is any source this project cannot redistribute under Apache-2.0: copyleft, source-available, paid, proprietary, or with a licence that cannot be read. Ultralytics is the loudest instance of that class, not the class itself, and today's pull-request template names it where the rule is D13's whole allowlist. No attestation detects such code arriving, so the layers are picked for how differently they fail — a DCO sign-off records who submitted it and under what claim of right, a generalized attestation fixes what that claim covers, and a provenance trailer on every contributor commit is the only one of the three that yields something reviewable: an id resolvable against `docs/PROVENANCE.md`, whose absence on a change to a loss or an assigner is visible before release rather than after. Enforcement is a check in this repository rather than a third-party app, because a survey on 2026-09-02 found every free hosted CLA service either archived, years stale, or a two-star project, and an admission record should not depend on someone else's uptime.

**Phases 13 and 14 are written down.** The augmentation stack leaves for `fuse-augmentations` — the plan of record is kept outside the tree, and only its roadmap rows land here. The split into two phases is not cosmetic: the guard that authorizes a swap is *published* in 0.6.0 before anything it guards is allowed to move in 0.7.0, so a swap is measured against a released version rather than a working tree. The guard is deliberately not an old-versus-new comparison, which would enshrine legacy behaviour as the definition of correct and keep the code alive as its own reference, blocking the deletion the check exists to authorize. Keypoint ONNX export and `predict_keypoints` join Phase 13 because 0.5.0's own release entry names them as the parity a later phase owes.

