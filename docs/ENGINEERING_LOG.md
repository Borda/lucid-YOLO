# 🛠️ Engineering Log

What executing the work packages taught about the repository itself, split out of `docs/RESEARCH_LOG.md`: CI and commit-time gates, packaging and licensing, formatting and doc-generation tooling, environment reproducibility, and the CLI/report code the reproduction runs on top of but that carries no fidelity claim of its own. The two logs are split by claim, not by work package -- a WP whose finding straddles both gets one entry in each, cross-linked; most don't straddle and land in exactly one.

**This file is not a register**, for the same reason `RESEARCH_LOG.md` is not: a choice the papers left open belongs in ASSUMPTIONS.md, cited by id rather than restated. Release-level results live in REPRODUCTION_REPORT.md.

Each finding is written once, in its work package's own file, `work_packages/wp-<id>.md`, between hidden `engineering` markers; the phase pages below only include it (`--8<-- "work_packages/wp-XXX.md:engineering"`), and the roadmap links to the WP file's anchor (`#wp-XXX`, or `#wp-XXX-engineering` where the research log already uses that id). To add one, write the section into the WP file and add its include line to the phase page, then run `scripts/lint/audit_wp_links.py --fix`: it links every `WP-<id>` you named and writes the WP page's header (log pages, depends on, needed by, neighbours), and the `wp-links` hook refuses a commit where either is stale. A package with nothing worth recording here has no section -- and a package whose whole finding is a fidelity claim rather than tooling has none here at all; see `RESEARCH_LOG.md` for it.

Dates are the day the finding was recorded.

## 🗂️ Pages

Pages, one per phase group:

- [Cross-cutting](engineering/cross-cutting.md)
- [Phase 0 — Foundation](engineering/phase-00-foundation.md)
- [Phase 5 — Lightning training loop](engineering/phase-05-training-loop.md)
- [Phase 6 — Evaluation, v0.1.0](engineering/phase-06-evaluation.md)
- [Phase 8 — Oriented detection, v0.3.0](engineering/phase-08-oriented-detection.md)
- [Phase 9 — Inference and generalization](engineering/phase-09-inference.md)
- [Phase 10 — Consolidation](engineering/phase-10-consolidation.md)
- [Phase 11 — Rolling, toward v0.5.0](engineering/phase-11-rolling.md)
- [Phase 12 — Keypoint detection, v0.5.0](engineering/phase-12-keypoints.md)
- [Phase 13 — Contributor admission, augmentation guard and keypoint parity, v0.6.0](engineering/phase-13-contributor-admission.md)
- [Phase 14 — Augmentation upstreaming to `fuse`, v0.7.0](engineering/phase-14-augmentation-upstreaming.md)
- [Phase 15 — Audit remediation, v0.8.0](engineering/phase-15-audit-remediation.md)
- [Phase 16 — Training throughput](engineering/phase-16-training-throughput.md)
- [Phase 17 — R1 pose-objective parity](engineering/phase-17-pose-objective-parity.md)
- [Phase 18 — CUDA graph replay and pipeline throughput](engineering/phase-18-cuda-replay-throughput.md)
- [Phase 19 — Codebase audit remediation](engineering/phase-19-codebase-audit.md)
