# 🔬 Research Log

What executing the work packages taught about the reproduction itself: the paper-fidelity measurements, the modeling and data-protocol choices tried and abandoned, and the results that do not flatter the project. Repo-building findings -- CI, packaging, licensing, formatting, docs tooling -- live in `docs/ENGINEERING_LOG.md` instead; the two are split by claim, not by work package, so a single WP whose finding straddles both gets one entry in each, cross-linked.

**This file is not a register.** Where a finding fixed an implementation choice the papers left open, the choice lives in ASSUMPTIONS.md and the number lives there with it; those are cited by id here, never restated. Release-level results live in REPRODUCTION_REPORT.md for the same reason. What lands here is what has no other home: fidelity measurements, negative results, and the reasoning behind a rejection.

Each finding is written once, in its work package's own file, `work_packages/wp-<id>.md`, between hidden `research` markers; the phase pages below only include it (`--8<-- "work_packages/wp-087d.md:research"`), and the roadmap links to the WP file's anchor (`#wp-087d`). To add one, write the section into the WP file and add its include line to the phase page, then run `scripts/lint/audit_wp_links.py --fix`: it links every `WP-<id>` you named and writes the WP page's header (log pages, depends on, needed by, neighbours), and the `wp-links` hook refuses a commit where either is stale. A package with nothing worth recording here has no section -- and a package whose whole finding is about tooling rather than fidelity has none here at all; see `ENGINEERING_LOG.md` for it.

**Phases 1-5 are backfilled** (2026-08-14) and read differently from the rest. Those roadmap rows were transcribed from the blueprint *before* the work and never rewritten afterwards, so nothing was carried out of them -- their entries here are reconstructed from the commit record, which is the durable artifact, and not from anyone's memory of the work. Phase 6 onward is the reverse: those rows had accumulated the retrospective detail in place, and this file is where it moved to. Phase 0's own findings were all tooling-kind and moved to `ENGINEERING_LOG.md` whole.

Dates are the day the finding was recorded.

## 🗂️ Pages

Pages, one per phase:

- [Phase 1 — Data pipeline](research/phase-01-data-pipeline.md)
- [Phase 2 — Architecture](research/phase-02-architecture.md)
- [Phase 3 — Assignment and losses](research/phase-03-assignment-and-losses.md)
- [Phase 4 — MuSGD](research/phase-04-musgd.md)
- [Phase 5 — Lightning training loop](research/phase-05-training-loop.md)
- [Phase 6 — Evaluation, v0.1.0](research/phase-06-evaluation.md)
- [Phase 7 — Instance segmentation, v0.2.0](research/phase-07-instance-segmentation.md)
- [Phase 8 — Oriented detection, v0.3.0](research/phase-08-oriented-detection.md)
- [Phase 9 — Inference and generalization](research/phase-09-inference.md)
- [Phase 10 — Consolidation](research/phase-10-consolidation.md)
- [Phase 12 — Keypoint detection, v0.5.0](research/phase-12-keypoints.md)
