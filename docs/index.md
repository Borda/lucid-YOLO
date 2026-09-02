# 🔦 lucid-yolo

> lucid-yolo is an independent, from-scratch PyTorch Lightning implementation of the real-time detection, instance segmentation, and oriented detection methods described in the Ultralytics YOLO26 paper (arXiv:2606.03748). "YOLO" refers to the family of real-time detectors originated by Redmon et al. (2016). This project is not affiliated with, endorsed by, or derived from Ultralytics or its codebase. No Ultralytics source code, configurations, or model weights were consulted or used. See docs/PROVENANCE.md.

## 🧭 What this is

A paper describes three real-time vision models. This repository implements them from that description — the equations, the tables, the figures — and from the primary literature the paper cites, and from nothing else. The reference implementation was never opened, and neither was any mirror, package copy, or documentation site generated from it. That constraint is the point of the exercise rather than an obstacle to it: an implementation that shares no lineage with the original is the only kind that can independently test whether the published claims follow from the published method.

Three habits keep that claim auditable rather than asserted.

- **Every design decision cites a public source.** Each commit carries a `Provenance:` trailer naming an entry in the [provenance log](PROVENANCE.md), and that log is a strict allowlist — the three papers, their cited primary literature, and neutral dataset or tooling documentation.
- **Every gap in the papers is a recorded assumption.** Where the papers underdetermine the implementation, the choice, its public basis, and the test that would validate it are written into the [assumption register](ASSUMPTIONS.md) *before* the code lands. 23 of them were exercised by the three accepted runs, and the record says which of those a passing run actually isolated and which merely rode along ([reproduction report, *Assumption outcomes, consolidated*](REPRODUCTION_REPORT.md)).
- **Guessing is escalated, not performed.** An implementer who cannot answer a question from an allowed source stops and writes an [escalation entry](ESCALATION.md) instead of resolving the ambiguity by looking at someone else's code.

What is claimed is a faithful reproduction of the *method*, not of the paper's exact numbers ([decision record, D2](DECISIONS.md)). Each of the three models has been trained once, at the smallest of five scales, on a deliberately short schedule — a smoke tier, in this project's vocabulary, sized to prove the mechanism works rather than to compete with a published leaderboard. No trained weights are published ([decision record, D14](DECISIONS.md)); the source, the frozen golden metrics, and the report below are the release artifacts.

## 🧠 The four tasks

Each task is the previous one plus a head, not a new model: all four share one backbone, one neck, one DFL-free dual detection head, one NMS-free deploy path, and one optimizer ([reproduction report, *One trunk, three heads*](REPRODUCTION_REPORT.md) — written when there were three, and the argument is unchanged by the fourth).

| Task | What it adds to the trunk | Data | Size at `n` scale | Card |
| -- | -- | -- | -- | -- |
| Detection | the trunk itself — dual head, direct `ltrb` boxes, no distribution bins | COCO 2017, 640 px | 2.4 M params, 5.4 GFLOPs | [detection](model_cards/detection.md) |
| Instance segmentation | prototype–coefficient masks and a training-only auxiliary semantic branch | COCO 2017, 640 px | 2.72 M params, 9.00 GFLOPs | [segmentation](model_cards/segmentation.md) |
| Oriented detection | per-branch angle stems, a rotated IoU term, a retargeted L1 | DOTA-v1.0 tiles, 1024 px | 2.56 M params, 14.68 GFLOPs | [oriented](model_cards/obb.md) |
| Keypoints | a `K`-generic point stem, and a normalizing-flow residual likelihood (RLE) as its loss | COCO 2017 `person_keypoints`, 640 px | 2.38 M params, 5.51 GFLOPs | [keypoints](model_cards/keypoints.md) |

**The fourth task is keypoints, not pose.** `K` is a constructor argument the way the class count is; nothing in the head, the loss or the decode path knows what a point means. Human pose is the instantiation the shipped checkpoint trained on (`K = 17`, COCO's `person` schema) and the one the OKS metric's sigma table is defined for — the wiring gate runs a 7-point synthetic symbol schema instead.

Parameter and FLOP counts are from the *What was reproduced* subsection of each release section of the [reproduction report](REPRODUCTION_REPORT.md). For detection and segmentation they are gated against the paper's own published tables — all five scales within ±2% params and ±5% FLOPs for detection, ±3% and ±5% for segmentation. For oriented detection the paper publishes no such table, so the gate holds the numbers against this project's own frozen goldens: it catches drift, and corroborates nothing. For keypoints there is no gate at all: the source paper (R14) states a loss and an evaluation protocol, never an architecture, so there is nothing to hold a number against.

## 📊 What was reproduced, in numbers

Every figure below is quoted from [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md), with the section it comes from named beside it. All four runs are `n` scale, roughly 50 epochs, seed 0, single seed throughout.

| Result | Value | Source section |
| -- | -- | -- |
| Detection, COCO val2017, EMA weights, NMS path | **25.30** mAP50-95 (38.12 mAP50) | `0.1.0 — Detection` → *Det-smoke acceptance* |
| Detection, same weights, NMS-free end-to-end path | 23.84 mAP50-95 | `0.1.0 — Detection` → *Det-smoke acceptance* |
| Cost of dropping NMS, detection | 1.11 AP on raw weights, 1.46 on EMA | `0.1.0 — Detection` → *Det-smoke acceptance* |
| Segmentation, COCO val2017, EMA weights, NMS path | **19.01** segm mAP50-95, 26.12 box mAP50-95 | `0.2.0 — Instance segmentation` → *The Seg-smoke run* |
| Mask accuracy as a fraction of box accuracy | **0.728** (NMS), 0.732 (E2E), against a 0.65 acceptance floor | `0.2.0 — Instance segmentation` → *Seg-smoke acceptance* |
| Cost of dropping NMS, segmentation | 1.27 box AP, 0.81 segm AP | `0.2.0 — Instance segmentation` → *The Seg-smoke run* |
| Oriented detection, DOTA-v1.0 val, EMA weights, **per tile** | **0.2914** rotated mAP50-95, 0.5242 rotated mAP50 | `0.3.0 — Oriented detection` → *The OBB-smoke run* |
| Cost of dropping NMS, oriented detection | not measured | `Consolidated note` → *One trunk, three heads* |
| Keypoints, COCO val2017, EMA weights, NMS-free end-to-end path | **0.2738** OKS AP, 0.5030 box mAP50-95 | `0.5.0 — Keypoint detection` → *The Pose-smoke run* |
| RLE's learned flow against its own flow-free control | 0.2738 vs **0.2527** OKS AP — same direction as the source paper's 70.5-vs-67.4 | `0.5.0 — Keypoint detection` → *Acceptance* |

Four readings the report insists on, and this page repeats rather than smooths over.

**The oriented numbers are per tile and are comparable to nothing published.** DOTA images are cut into overlapping 1024 px crops and scored crop by crop; detections are not merged back onto whole images, so no tile ever pays the duplicate cost a whole-image protocol charges. The report states this as a property of the measurement rather than a caveat about its precision, and makes no comparison to the paper's oriented tables anywhere (`0.3.0 — Oriented detection` → *The OBB-smoke run*).

**The NMS-free deficit is wider than the paper's.** The paper reports 0.6–0.8 AP between its NMS-free and NMS paths; this reproduction measures 1.11 AP on raw weights. Right sign, right order of magnitude, wider — recorded as a reproducible claim of its own rather than explained away (`0.1.0 — Detection` → *Det-smoke acceptance*).

**Three earlier detection runs failed before the one above passed**, at 3.96, 6.28 and 4.00 mAP50-95, and the report treats the failures as the more instructive half: two semantic defects — a loss term computed in the wrong coordinate frame, and an augmentation random stream that every worker replayed identically — that crashed nothing, failed no test, and cost roughly 35 GPU-hours (`0.1.0 — Detection` → *What went wrong first, and why it matters*).

**These are smoke tiers.** Roughly 50 epochs at the smallest scale against the paper's from-scratch 500- and 600-epoch schedules, no Objects365 pretraining, no evolutionary hyperparameter search, one seed (`Consolidated note` → *Deviations*).

## 🗂️ Reading the rest of this site

The documents fall into three registers, and which one you want depends on what you came for.

| If you want to | Read | It answers |
| -- | -- | -- |
| run a model yourself | [Launching a training run](TRAINING.md) | the launch command for each of the three tiers, and what to override |
| get the data first | [Provisioning the datasets](DATASETS.md) | where COCO 2017 and DOTA-v1.0 come from, what has to be on disk, how each layout is read |
| know what was measured | [Reproduction report](REPRODUCTION_REPORT.md) | one section per release: what was reproduced, what was assumed, what diverged |
| judge one model before using it | [Model cards](model_cards/detection.md) | intended and out-of-scope use, limitations, licensing, per task family |
| check the clean-room claim | [Provenance log](PROVENANCE.md) | the source allowlist and the audit trail behind every commit trailer |
| find where the papers ran out | [Assumption register](ASSUMPTIONS.md) | every gap, the choice made, its public basis, and how it was validated |
| understand why the project is shaped this way | [Decision record](DECISIONS.md) | architecture and policy decisions, including four ADRs |
| see what is done and what is next | [Work-package roadmap](ROADMAP.md) | the numbered work queue with live status |
| learn what execution actually cost | [Research log](RESEARCH_LOG.md) | fidelity measurements, rejected modeling approaches, and negative results, per work package |
| learn what the repo tooling cost | [Engineering log](ENGINEERING_LOG.md) | CI, packaging, licensing, and doc-tooling findings, split out of the research log by claim |
| see where work stopped for the principal | [Escalation log](ESCALATION.md) | the anti-guessing rule and every entry raised under it |

The roadmap and the two logs are deliberately split: the roadmap says what a work package does, the logs say what executing it taught. Between the two logs the split is by claim, not by work package -- a fidelity finding in one, a tooling finding in the other, and a WP whose finding is genuinely both gets one entry in each, cross-linked. A row that restates any of them is duplicating a record with an owner.

## 🚀 Running it yourself

Datasets are never committed and never downloaded by the test suite. [Provisioning the datasets](DATASETS.md) covers acquiring them; [launching a training run](TRAINING.md) carries the exact command for each tier, and the minutes-long overfit gate worth running before a launch that costs hours.

Four commands ship with the package — training and validation, dataset download and checking and tiling, acceptance scoring, and single-image prediction — so a remote run needs no checkout. Contributors and agents working on the repository itself start at `AGENTS.md` in the repository root, which is the execution contract the roadmap is run under.
