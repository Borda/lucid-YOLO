# Changelog

All notable changes to lucid-yolo are documented here, following the Keep a
Changelog convention; versioning is a perpetual 0.x release train — no 1.0 is
ever planned, promised, or tagged — per ADR-002 (docs/DECISIONS.md).

## [Unreleased]

### Added

- Long-edge rotated-box primitives: canonicalization to `w >= h` with the angle on
  `[-45, 135)` degrees, quadrilateral conversion in both directions, and a
  vectorized point-in-rotated-rect test. Three conventions the paper leaves open
  are fixed here and inherited by the whole oriented path — the angle turns
  `+x` towards `+y`, containment is edge-inclusive, and an exact square folds
  towards zero (WP-055).
- DOTA-v1.0 label parsing: the 15 categories in the published order, metadata
  headers skipped, category names normalized across their hyphenated and
  underscored spellings, and each annotated quadrilateral converted to a
  canonical long-edge box. Rotated boxes are kept on the same instance axis as
  the axis-aligned envelopes and labels, so one mask filters every modality
  (WP-056).
- `make check-data DATASET=dota` — DOTA root layout, image/label pairing and
  published-count validation beside the existing COCO path (WP-056).

## [0.2.0] - 2026-08-10

### Added

- Repository scaffold: `src/` package layout, pinned `pyproject.toml`, Makefile
  gate targets, and pre-commit configuration (WP-001).
- Legal and policy baseline: Apache-2.0 LICENSE with Redmon NOTICE attribution,
  a non-affiliation README, and the PROVENANCE / ASSUMPTIONS / DECISIONS /
  AGENTS / ROADMAP governance set (WP-002, WP-003).
- Continuous integration: a pull-request workflow running ruff, strict mypy, the
  offline pytest suite with coverage, a copyleft-dependency license audit, and a
  provenance-carrying commit-trailer validator (WP-004).
- Golden gate: a recompute-and-compare harness with per-metric tolerances and a
  `goldens/frozen/` regression path wired into `make gate` (WP-005).
- Tag-gated release workflow: a three-check guard (tag shape, CHANGELOG section
  present, `make gate` green) and this CHANGELOG's own scaffold (WP-006).
- Data foundation: seeded synthetic micro-dataset fixtures (boxes, polygons, and
  rotated scenes via fuse-augmentations) and the `Targets` container with its
  type-generic transform API (WP-007, WP-008).
- Augmentation pipeline: letterbox resize with an exact inverse, random affine
  for boxes and masks, mosaic assembly, mixup and copy-paste, and HSV jitter
  with horizontal flip (WP-009 through WP-013).
- COCO dataset and `LightningDataModule`, plus round-trip augmentation goldens
  and a debug visualizer (WP-014, WP-015).
- Model and optimizer primitives: Conv / DWConv / Bottleneck blocks, the CIoU
  regression loss, and Newton-Schulz orthogonalization for the MuSGD optimizer
  (WP-016, WP-024, WP-031).
- Architecture stack: C3k2, PSABlock/C2PSA, SPPF with a shortcut, a backbone
  with P3/P4/P5 taps, an attention-tail neck, a dual detection head with
  `reg_max=1`, and a scale registry with a param/FLOP fidelity gate against
  Table 7 (WP-017 through WP-023).
- Assignment and detection losses: the Task-Aligned Assigner, STAL surrogate
  candidate filtering for small targets, the detection branch loss, and the
  o2m/o2o dual-branch composition, plus synthetic assignment goldens and a
  single-batch overfit gradient-flow gate (WP-025 through WP-030).
- MuSGD optimizer with parameter-type split, and its toy convergence golden
  against plain SGD (WP-032, WP-033).
- Lightning training loop: task-conditional `LightningModule`, the
  ProgressiveLossSchedule hook, CloseMosaic and EMA callbacks, a LightningCLI
  entry with tiered experiment configs, deterministic checkpoint and resume,
  and the overfit-100 integration golden (WP-034 through WP-040).
- Detection decode and evaluation: score-based top-k end-to-end decoding, a
  class-wise NMS path, a dual-path pycocotools evaluator with an oracle
  round-trip gate, and a checkpoint eval script reporting both paths
  (WP-041 through WP-045).
- COCO 2017 downloader module and the `lucid-download` CLI (WP-068).
- CLI conveniences: packaged-config name resolution, the Det-smoke recipe as
  the default when no `--config` is given, and a default TensorBoard+CSV
  logger pair (WP-038).
- A8 learning-rate schedule: linear warmup followed by linear decay (WP-072).
- Epoch validation mAP surfaced in the training progress bar (WP-077).
- Regression coverage for the training objective: an objective-invariant gate
  pinning the loss coordinate frame and its scale, and a synthetic-shapes
  generalization golden scoring held-out data through both decode paths
  (WP-082, WP-083).
- Training-curve figures: a deterministic SVG script rendering a run's
  `metrics.csv` as a three-panel figure (a fourth panel for segmentation
  runs), gated against its own drawn output so a caption cannot silently
  diverge from what the figure draws (WP-080).

Instance segmentation:

- Mask-coefficient branch: K=32 tanh-activated coefficients added to both
  detection branches, opt-in and off by default so the accepted detector
  stays byte-for-byte unaffected (WP-047).
- Multi-scale prototype-fusion pathway (Eq. 8) and a four-layer prototype
  generation stack (Eq. 9), producing raw K-channel prototypes at twice the
  P3 resolution (WP-048, WP-049).
- Training-only auxiliary semantic branch, returning `None` whenever the
  module is in eval mode and sharing the detection head's prior-probability
  bias init (WP-050).
- Segmentation losses: box-cropped, area-normalized per-pixel instance mask
  BCE and a BCE+Dice auxiliary semantic loss, both built on one shared Eq. 7
  mask-assembly implementation that the decode path also uses (WP-051).
- `Segmenter` model wiring: backbone, neck, dual head, prototype pathway, and
  auxiliary branch composed into one module, whose `deploy()` holds neither
  the o2m branch nor the auxiliary head, verified by both module-walk
  identity and exact parameter arithmetic; a Table S9 param/FLOP fidelity
  gate across all five scales (WP-052).
- Segmentation mask decode: sigmoid, bilinear upsample, box-crop, threshold,
  with masks inverted through the same letterbox geometry used for boxes,
  plus both-path (bbox and segm) COCO evaluation and a task-aware checkpoint
  eval script (WP-053).
- Segmentation training path: prototype, coefficient, and semantic heads
  wired into the Lightning module through shared `build.py` factories, mask
  supervision reusing the box loss's own assignment, and both branches'
  coefficients supervised under the shared alpha schedule; frozen at an
  overfit-100 mask IoU of 0.8146 (WP-087).
- Epoch `val/segm_mAP` logged beside `val/mAP` during segmentation
  validation, decoded through the same path deployment uses (WP-087).

### Changed

- Detection evaluator rebuilt on `torchmetrics.detection.MeanAveragePrecision`
  with the faster-coco-eval backend, dropping the pycocotools dependency
  (WP-069).
- Data-loading throughput: a fused affine+letterbox single warp (4.4x on the
  geometric path), packed ragged targets and uint8 image transport across
  the DataLoader IPC boundary, accelerator-aware `pin_memory`/prefetch/
  thread-cap loader streaming, and an auto `num_workers` default capped by
  both core count and free `/dev/shm` (WP-014, WP-070, WP-071).
- Compact per-image annotation store and a separate, smaller validation-loader
  worker cap, closing a persistent-worker memory creep that killed long
  containerized runs (WP-073); loader workers now recycle every epoch instead
  of staying persistent, for the same reason (WP-076).
- Segmentation training throughput: batched mask-loss gather, replacing 64
  forced device syncs a step with two per branch, and polygon rasterization
  vectorised, batched, and moved into the loader workers with each ring
  rasterised inside its own bounding window (WP-087).
- CI reorganized: the copyleft license audit and commit-trailer checks move
  to commit-time hooks, lint and test workflows split into their own files,
  and the `dev` extra becomes a PEP 735 dependency group so build/test
  tooling no longer ships in the published wheel's metadata (WP-084).
- Doctests joined the offline gate, first for `src` — 558 to 699 collected
  cases, and four Examples that had rotted unnoticed — and then for `scripts`
  as well, 767 to 819; a separate `gate-gpu` target now recomputes the
  GPU-marked tests and goldens on its own schedule (WP-085).
- Experiment tier configs and their historical mentions renamed to describe
  what they are (`det_smoke`, `det_ablations`, `seg_smoke`) instead of
  letters that read backwards (WP-087).

### Fixed

- S3 path-style URLs for COCO downloads, avoiding a TLS certificate mismatch
  on the official host's CNAME (WP-068).
- Accelerator-aware determinism default in the CLI, avoiding an MPS abort on
  a kernel with no deterministic implementation (WP-038).
- Prior-probability class-bias initialization and gradient clipping in the
  shipped recipes, fixing a dead-gradient launch failure (WP-022).
- Notebook-safe `tqdm` progress-bar default, avoiding a Rich live-render
  newline flood in notebook output; a spurious args/argv overlap warning
  from console-script runs removed, and TF32 matmuls enabled on CUDA
  (WP-038).
- Default DataLoader prefetch factor and a free-`/dev/shm` budget cap on the
  auto worker count, both closing shared-memory exhaustion on containerized
  hosts (WP-014).
- Stride-normalized L1 box-regression term, correcting a term that had grown
  to 97% of the training objective and starved classification (WP-078).
- Per-worker, per-epoch augmentation RNG re-seeding, correcting a generator
  that replayed one draw stream across every worker and every epoch
  (WP-079).
- Package version declared in one place, the module's `__version__` literal,
  instead of drifting between it and `pyproject.toml` — a build had shipped
  0.1.0 while the runtime reported 0.0.1.dev0 (WP-001).

### Documentation

- ADR-004 (D13): the source allowlist widened to permit reading, never
  copying, permissively licensed detection implementations when diagnosing a
  structural defect.
- RF100-VL generalization work packages recorded on the roadmap (WP-074,
  WP-075).
- Policy that releases ship no trained weights, since the reproduction's
  claim rests on frozen goldens and fidelity gates rather than a retrainable
  binary (WP-086).
- Segmentation training-path and inference work packages added to the
  roadmap for Phases 7-9 (WP-087).
- Det-smoke report section and detector model card documenting the accepted
  run (val mAP50-95 25.30 EMA-NMS) and the two failed-attempt root causes
  across both decode paths (WP-045, WP-080, WP-081).
- Seg-smoke run write-up: the 0.2.0 `REPRODUCTION_REPORT.md` section
  (observed 26.12 box mAP, 0.728 segm mAP NMS / 0.732 E2E) and
  `MODEL_CARD_SEGMENTATION.md` (WP-054).
