# 🛠️ Engineering Log

What executing the work packages taught about the repository itself, split out of `docs/RESEARCH_LOG.md`: CI and commit-time gates, packaging and licensing, formatting and doc-generation tooling, environment reproducibility, and the CLI/report code the reproduction runs on top of but that carries no fidelity claim of its own. The two logs are split by claim, not by work package -- a WP whose finding straddles both gets one entry in each, cross-linked; most don't straddle and land in exactly one.

**This file is not a register**, for the same reason `RESEARCH_LOG.md` is not: a choice the papers left open belongs in ASSUMPTIONS.md, cited by id rather than restated. Release-level results live in REPRODUCTION_REPORT.md.

Each finding is written once, in its work package's own file, `work_packages/wp-<id>.md`, between hidden `engineering` markers; the phase pages below only include it (`--8<-- "work_packages/wp-XXX.md:engineering"`), and the roadmap links to the WP file's anchor (`#wp-XXX`, or `#wp-XXX-engineering` where the research log already uses that id). To add one, write the section into the WP file and add its include line to the phase page, then run `scripts/lint/audit_wp_links.py --fix`: it links every `WP-<id>` you named and writes the WP page's header (log pages, depends on, needed by, neighbours), and the `wp-links` hook refuses a commit where either is stale. A package with nothing worth recording here has no section -- and a package whose whole finding is a fidelity claim rather than tooling has none here at all; see `RESEARCH_LOG.md` for it.

Dates are the day the finding was recorded.

## 🗂️ Pages

Pages, one per phase group:

- [Cross-cutting](https://borda.github.io/lucid-YOLO/engineering/cross-cutting/)
- [Phase 0 — Foundation](https://borda.github.io/lucid-YOLO/engineering/phase-00-foundation/)
- [Phase 5 — Lightning training loop](https://borda.github.io/lucid-YOLO/engineering/phase-05-training-loop/)
- [Phase 6 — Evaluation, v0.1.0](https://borda.github.io/lucid-YOLO/engineering/phase-06-evaluation/)
- [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/engineering/phase-08-oriented-detection/)
- [Phase 9 — Inference and generalization](https://borda.github.io/lucid-YOLO/engineering/phase-09-inference/)
- [Phase 10 — Consolidation](https://borda.github.io/lucid-YOLO/engineering/phase-10-consolidation/)
- [Phase 11 — Rolling, toward v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-11-rolling/)
- [Phase 12 — Keypoint detection, v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-12-keypoints/)
- [Phase 13 — Contributor admission, augmentation guard and keypoint parity, v0.6.0](https://borda.github.io/lucid-YOLO/engineering/phase-13-contributor-admission/)
- [Phase 14 — Augmentation upstreaming to `fuse`, v0.7.0](https://borda.github.io/lucid-YOLO/engineering/phase-14-augmentation-upstreaming/)
- [Phase 15 — Audit remediation, v0.8.0](https://borda.github.io/lucid-YOLO/engineering/phase-15-audit-remediation/)
- [Phase 16 — Training throughput](https://borda.github.io/lucid-YOLO/engineering/phase-16-training-throughput/)
- [Phase 17 — R1 pose-objective parity](https://borda.github.io/lucid-YOLO/engineering/phase-17-pose-objective-parity/)
- [Phase 18 — CUDA graph replay and pipeline throughput](https://borda.github.io/lucid-YOLO/engineering/phase-18-cuda-replay-throughput/)
- [Phase 19 — Codebase audit remediation](https://borda.github.io/lucid-YOLO/engineering/phase-19-codebase-audit/)
