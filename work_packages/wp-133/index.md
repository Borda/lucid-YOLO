# 📦 WP-133 — `feat(configs): pose-smoke draft, changelog Unreleased entry`

> Engineering log · [Phase 12 — Keypoint detection, v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-12-keypoints/)
>
> 🔗 Depends on none · Needed by none
>
> ⬅️ [WP-131](https://borda.github.io/lucid-YOLO/work_packages/wp-131/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/) ➡️


## WP-133 — an Unreleased section, and the rule it follows

<a id="wp-133"></a>

`CHANGELOG.md` had no `## [Unreleased]` section at all. 27 commits had landed since the 0.4.0 release commit (`0a3ef8b`) -- the dev-version bump to `0.5.0.dev0` and [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/)'s own keypoint training path among them -- and every one of them carried zero changelog entries, so the file read as though nothing had shipped since 0.4.0. Surfaced by a direct question, not a check: nothing in `make gate` or the pre-commit hooks reads `CHANGELOG.md`'s content (`release_guard.py` checks a tag has *a* matching entry, not that intervening commits do), so the gap had no way to fail loud.

**Backfilled from `git log 0a3ef8b..HEAD`**, one bullet per user-visible change, grouped Added/Changed/Fixed to match the existing release sections' own convention. Internal-only commits -- test reorganization, roadmap bookkeeping, CI hook wiring, docstring formatting -- are omitted by the same judgment the existing sections already apply; none of the four shipped 0.x sections carries an entry for its own test suite. The keypoint task's five component WPs (120-124) plus its training-path composition (132) and the ringless-annotation fix (121b) collapse into one Added bullet rather than six, since a changelog reader wants "the keypoint task landed and what it cost to get right," not a WP-by-WP replay -- that granularity already exists in this table and in RESEARCH_LOG.md.

**The section states its own closing rule inline**, per the standing instruction this row also establishes: a commit that sets `__version__` to a plain, non-dev, non-rc value closes the `## [Unreleased]` section and opens the next one. Going forward, a WP's own commit should add its own `## [Unreleased]` bullet rather than the whole set being reconstructed after the fact the way this row had to -- the reconstruction is legible from git history exactly once; a habit of writing it live is not.

**Two pre-existing date mismatches surfaced and are recorded rather than silently corrected.** `[0.2.0] - 2026-08-10` against release commit `eb81caa`'s `2026-08-11`; `[0.4.0] - 2026-08-15` against `0a3ef8b`'s `2026-08-16`. Both one day earlier than the commit, both the same direction -- consistent with the date recorded being the day a run was accepted rather than the day the release commit landed, not a typo in either single case. `[0.3.0] - 2026-08-14` matches its commit exactly, so the convention is not applied consistently even under that reading. Left as a known discrepancy: correcting two dates nobody had complained about is a scope this row did not need, and the pattern is now written down for whoever next touches a release date to decide.

**`configs/pose_smoke.yaml` folded into this same commit, at the user's direction, rather than opening a separate WP.** [WP-125](https://borda.github.io/lucid-YOLO/work_packages/wp-125/) needed a launch config before a Colab session already underway could proceed; writing one is mechanical against the `det_smoke.yaml`/`seg_smoke.yaml` precedent (`task: keypoints`, `num_keypoints: 17` for COCO `person_keypoints`, `keypoint_gain: 1.0` per A68's un-tuned default, explicit `train_ann_file`/`val_ann_file` since `lucid_yolo.data.layout` only ever resolves the `instances_` spelling and `person_keypoints_{split}2017.json` sits beside it in the same annotations archive under a different name). Verified by dry-parse only -- `tests/ptl/test_cli.py::test_config_dry_parses[pose_smoke.yaml]`, the same DoD every other shipped config carries -- not by a training run. This is explicitly not [WP-125](https://borda.github.io/lucid-YOLO/work_packages/wp-125/)'s acceptance: that row is `[GPU][PRINCIPAL]` and its criterion is RLE's mechanism claim against an OKS-only ablation, which a config file cannot settle and this entry does not claim to.

