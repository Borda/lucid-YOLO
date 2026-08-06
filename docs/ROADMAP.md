# Work-Package Roadmap

The agent's work queue: 69 work packages, one commit each, executed in
dependency order per the AGENTS.md loop. Transcribed from the governing
blueprint (sec. 15) with a live status column.

Legend: **Dep** = prerequisite WPs · **[DATA]** needs a real dataset (synthetic
stand-in per A26 for offline development only) · **[GPU]** needs an accelerator
(MPS first, D12c) · **[HUMAN]** requires human action — never started
autonomously. DoD = the test id(s) that must pass; every WP additionally
requires `make gate` green. Status icons: ⬜ todo · 🔄 in-progress · ✅ done ·
⛔ blocked (see docs/ESCALATION.md). The status column is flipped to ✅ in the
same commit that completes its WP.

## Phase 0 — Foundation (WP-001…007)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 001 | `chore(repo): scaffold src layout, pyproject, Makefile` | Package skeleton, pinned deps, Makefile targets, pre-commit config | `make lint` green on empty typed package | — | ✅ |
| 002 | `docs(legal): add LICENSE, NOTICE, README non-affiliation` | Apache-2.0, NOTICE with Redmon attribution (R20), README header verbatim | `tests/meta/test_license_headers.py` | 001 | ✅ |
| 003 | `docs(policy): seed PROVENANCE, ASSUMPTIONS, DECISIONS, AGENTS, ROADMAP` | Source allowlist, register A1–A26, D1–D12 + ADR-001/002/003, AGENTS.md, this file | `tests/meta/test_docs_present.py` | 001 | ✅ |
| 004 | `ci(pr): lint, types, tests, coverage, license audit` | PR workflow; dependency-license audit; commit-trailer validator | Workflow green; negative test: AGPL dev-dep rejected | 002,003 | ✅ |
| 005 | `ci(gates): golden harness and frozen-golden regression` | `goldens/` loader, tolerance comparison, `make gate`, `goldens/frozen/` semantics | `tests/meta/test_golden_harness.py`; tampered golden fails | 004 | ✅ |
| 006 | `ci(release): tag-gated release workflow and CHANGELOG` | `release.yml`; CHANGELOG scaffold; `make freeze-goldens` | Negative test: tag on red commit refused | 005 | ✅ |
| 007 | `test(fixtures): micro dataset with boxes, polygons, rotated scenes` | Seeded synthetic scenes via fuse-augmentations (A26): det/seg boxes+polygons, rotated scenes; loaders | `tests/fixtures/test_fixtures_load.py` | 003 | ✅ |

## Phase 1 — Data pipeline (WP-008…015)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 008 | `feat(data): target containers and type-generic transform API` | `Targets` dataclass (boxes, labels, masks, rboxes); transform protocol over every target type | `tests/data/test_targets.py` | 007 | ✅ |
| 009 | `feat(data): letterbox resize with exact inverse` | Aspect-preserving pad/resize + inverse map (A10) | `test_letterbox.py::test_roundtrip_subpixel` | 008 | ✅ |
| 010 | `feat(data): random affine for boxes and masks` | scale/translate/shear/degrees per Table S3; joint box+polygon transform, clipping | `test_affine.py::test_box_mask_consistency` | 009 | ✅ |
| 011 | `feat(data): mosaic assembly` | 4-image mosaic (R9), border handling, target remap | `test_mosaic.py::test_bounds_and_counts` | 010 | ✅ |
| 012 | `feat(data): mixup and copy-paste` | Table S3 probabilities, scale-aware policy | `test_mixup_copypaste.py` | 011 | ✅ |
| 013 | `feat(data): HSV jitter and horizontal flip` | hsv_h/s/v, fliplr=0.5 with target mirroring | `test_photometric.py` | 010 | ✅ |
| 014 | `feat(data): COCO dataset and LightningDataModule` [DATA] | Detection + polygon parsing, scale-aware augmentation policy, `make check-data` | `test_coco.py` (fixture-backed) + `check-data` on real COCO | 012,013 | ✅ |
| 015 | `test(data): round-trip goldens and debug visualizer` | Augmented-batch checksums; annotated grid dump script | `goldens/data_checksums.json` frozen | 014 | ✅ |

## Phase 2 — Architecture (WP-016…023)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 016 | `feat(models): Conv, DWConv, Bottleneck primitives` | Conv-BN-SiLU, depthwise variant, residual bottleneck | `test_blocks.py::test_primitives` | 003 | ✅ |
| 017 | `feat(models): C3k2 block` | CSP split, n inner blocks, e ratio, c3k switch (A3) | `test_blocks.py::test_c3k2_shapes` | 016 | ✅ |
| 018 | `feat(models): PSABlock and C2PSA` | Attention + FFN block; split/concat wrapper (A3) | `test_blocks.py::test_c2psa` | 016 | ✅ |
| 019 | `feat(models): SPPF with shortcut` | 1x1 -> 3x MaxPool(5) -> concat -> 1x1, plus input-output shortcut (A4) | `test_blocks.py::test_sppf_shortcut` | 016 | ✅ |
| 020 | `feat(models): backbone` | Backbone stack with P3/P4/P5 taps | `test_backbone.py::test_tap_shapes` | 017,018,019 | ✅ |
| 021 | `feat(models): neck with attention tail` | Top-down/bottom-up; final C3k2 n=1 e=0.5 attn=True | `test_neck.py::test_output_shapes` | 020 | ✅ |
| 022 | `feat(models): dual detection head, reg_max=1` | o2o (300x6) + o2m (nc+4, 8400) branches, DFL-free ltrb regression (A9) | `test_head.py::test_dual_head_shapes` | 021 | ✅ |
| 023 | `feat(models): scale registry, builder, param/FLOP fidelity gate` | 5-row dataclass registry, typed builders (ADR-001), fvcore counting | `test_param_flops.py::test_det_vs_table7` — plus/minus 2% params / 5% FLOPs, all 5 scales; **golden frozen** | 022 | ✅ |

## Phase 3 — Assignment and losses (WP-024…030)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 024 | `feat(losses): CIoU` | CIoU per R10 (A1), batched, autograd-safe | `test_ciou.py::test_against_closed_form` | 003 | ✅ |
| 025 | `feat(assign): anchor grid and Task-Aligned Assigner` | Centers at (i+0.5)*stride (A11); t = s^1 * u^6 (A2); topk selection | `test_tal.py::test_alignment_and_topk` | 024 | ✅ |
| 026 | `feat(assign): STAL surrogate candidate filtering` | Eq. 4–6; per-dimension clamp d<8 -> 16; original box preserved for scoring/regression | `test_stal.py::test_tiny_box_gains_candidates`, `::test_per_dim_clamp`, `::test_targets_unchanged` | 025 | ✅ |
| 027 | `feat(losses): detection branch loss` | CIoU + L1 (dfl-gain field, A13) + BCE, TAL-weighted | `test_detection_loss.py::test_components` | 026 | ✅ |
| 028 | `feat(losses): dual-branch composition` | o2m topk=10 / o2o topk=7->1 wiring; static alpha combination (schedule lands WP-035) | `test_dual_loss.py::test_one_positive_per_gt` | 027 | ✅ |
| 029 | `test(assign): synthetic assignment goldens` | 6x6 px GT: STAL >=1 candidate, vanilla TAL exactly 0 at stride 8 | `goldens/assignment_cases.json` frozen | 028 | ✅ |
| 030 | `test(train): single-batch overfit and gradient flow` | 200-step monotonic loss decrease; no NaN/Inf; all leaf grads populated | `test_overfit_batch.py` | 029 | ✅ |

## Phase 4 — MuSGD (WP-031…033)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 031 | `feat(optim): Newton-Schulz orthogonalization` | Pure function from R7/R8, 5 iterations (A5), fp32 under AMP | `test_newton_schulz.py::test_orthogonality` | 003 | ✅ |
| 032 | `feat(optim): MuSGD with parameter-type split` | >=2D: w_muon*Muon + w_sgd*SGD (A6, A7); 1D: pure SGD, no weight decay (A12) | `test_musgd.py::test_param_split`, `::test_step_shapes` | 031 | ✅ |
| 033 | `test(optim): toy convergence golden vs SGD` | Fixed synthetic regression + micro-CNN; MuSGD reaches threshold in fewer steps | `goldens/optim_toy.json` frozen | 032 | ✅ |

## Phase 5 — Lightning training loop (WP-034…040)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 034 | `feat(ptl): LightningModule with task-conditional losses` | Automatic optimization; det losses active, seg/obb hooks inert | `test_module.py::test_training_step` | 030,032 | ✅ |
| 035 | `feat(ptl): ProgressiveLossSchedule hook` | Eq. 3 in `on_train_epoch_start`, (0.8,0.2)->(0.1,0.9) | `test_proglos.py::test_alpha_at_t0_mid_end` | 034 | ✅ |
| 036 | `feat(ptl): CloseMosaic callback` | Disables mosaic for final `close_mosaic` epochs | `test_close_mosaic.py::test_flip_epoch` | 034 | ✅ |
| 037 | `feat(ptl): EMA callback` | Decay schedule, checkpointed, used for eval | `test_ema.py::test_shadow_updates` | 034 | ✅ |
| 038 | `feat(ptl): LightningCLI entry and experiment configs` | `configs/` tier matrix (ADR-001); resolved config logged per run | `test_cli.py::test_yaml_roundtrip`; all configs dry-parse | 035,036,037 | ✅ |
| 039 | `feat(ptl): deterministic checkpoint and resume` | Seeded resume reproduces the loss trajectory within tolerance | `test_resume.py::test_trajectory_match` | 038 | ✅ |
| 040 | `test(lit): overfit-100 integration golden` [GPU] | n-scale on a 100-image subset -> >=0.95 recall at IoU 0.5 on train | `goldens/overfit_micro_det.json` frozen | 039 | ✅ |

## Phase 6 — Evaluation, release 0.1.0 (WP-041…046)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 041 | `feat(decode): score-based top-k E2E decoding` | No IoU, no NMS, cap 300 (R3 sec. 4, A9) | `test_topk_e2e.py::test_no_nms_path` | 022 | ✅ |
| 042 | `feat(decode): NMS path for the dense branch` | Conf threshold + class-wise NMS (torchvision) | `test_nms_path.py` | 041 | ✅ |
| 043 | `feat(eval): pycocotools bbox evaluator, both paths` | One command evaluates E2E and non-E2E from one checkpoint | `test_coco_eval.py::test_dual_path_report` | 042 | ✅ |
| 044 | `test(eval): oracle round-trip` | Perfect predictions -> mAP 1.0; shuffled classes -> approx 0 | `test_coco_eval.py::test_oracle` | 043 | ✅ |
| 068 | `feat(data): COCO 2017 downloader module and CLI` | Official-host download into `check_data.py` layout; val-only default; `lucid-download` + `python -m lucid_yolo.data.download` (added 2026-08-02, user request) | `tests/data/test_download.py` offline suite | 014 | ✅ |
| 069 | `refactor(eval): torchmetrics MAP with faster-coco-eval backend` | Replace pycocotools evaluator internals with `torchmetrics.detection.MeanAveragePrecision(backend="faster_coco_eval")`; `DualPathEvaluator` API preserved; pycocotools dep dropped (added 2026-08-02, user request) | WP-044 oracle ladder green on new backend | 043,044 | ✅ |
| 070 | `perf(data): fused affine+letterbox single-warp` | Compose the letterbox affine into the random affine so the train geometric base resamples once (`FusedAffineLetterbox`); boxes/polygons byte-identical, image pixels non-antialiased (A32); 4.4x geometric-path speedup (added 2026-08-02, user request) | `tests/data/test_fused_warp.py`; data goldens recomputed | 011,013,014 | ✅ |
| 071 | `perf(data): packed + uint8 batch transport` | Ragged `list[Targets]` flattened to 8 dense tensors (`PackedTargets`) and images quantized uint8 for the DataLoader IPC hop (A33); `on_after_batch_transfer` restores float images at consumer precision + ragged targets on device; segments/batch ~600 -> 9, bytes/batch 4x down — clears containerized shm ceilings (added 2026-08-02, user request) | `tests/data/test_coco.py` pack/quantize round-trips | 014 | ✅ |
| 072 | `feat(optim): A8 LR schedule — warmup + linear decay` | Per-step LambdaLR: linear warmup over `warmup_epochs` then linear decay lr0 -> lr0*lrf (A8 revised, warmup folds in the A31 deferral); overfit-100 recipe pins the schedule off so its golden stays frozen (added 2026-08-03 after Det-A attempt 1 plateaued at 3.96 mAP on the constant-LR deferral) | `tests/optim/test_schedule.py`; overfit golden unchanged | 034,038 | ✅ |
| 073 | `perf(data): compact annotation store + val-loader worker cap` | `CocoDetectionDataset` precomputes per-image `Targets` at construction and drops the raw JSON annotation dicts — millions of small Python objects whose refcount traffic materializes copy-on-write pages in every persistent DataLoader worker until the host OOMs (observed: Colab 176 GB box killed at validation boundaries, e1 at 48 workers / e24 at 32). Val loader runs on `val_num_workers` (default `min(num_workers, 4)`) so the persistent val pool no longer doubles the worker population (added 2026-08-04, Det-A attempt-2 crash diagnosis) | `tests/data/test_coco.py` val-cap tests; dataset output unchanged (data goldens untouched) | 014 | ✅ |
| 076 | `perf(data): epoch-recycled loader workers` | `persistent_workers` now defaults **off** (opt-in constructor/CLI knob): respawning the pool costs seconds per 9-min epoch while a persistent pool accumulates per-worker memory — allocator high-water growth, arena fragmentation, residual copy-on-write pages — until long containerized runs OOM (observed: Colab kills at e24/e38 even after WP-073's compact store; recycling hard-resets all per-worker creep every epoch) (added 2026-08-04, Det-A attempt-2 crash saga) | `tests/data/test_coco.py` persistent-workers default/opt-in tests | 014,073 | ✅ |
| 077 | `feat(ptl): epoch val mAP in the progress bar` | `validation_step` decodes the one-to-one branch (E2E `TopKDecoder`) from the same forward as the loss and accumulates `torchmetrics` `MeanAveragePrecision` (`faster_coco_eval` backend); `on_validation_epoch_end` logs `val/mAP` to the progress bar. Letterbox-coordinate proxy for run monitoring (IoU invariant to per-image uniform scaling); acceptance figure stays `scripts/eval_det.py`. Metric states non-persistent — `state_dict` and older checkpoints unaffected (added 2026-08-04, user request) | `test_module.py` fast-dev-run logs `val/mAP`; state-dict-unchanged test | 041,043 | ✅ |
| 078 | `fix(losses): stride-normalized L1 term` | The L1 box term is measured in stride units (the head's native ltrb frame, the frame the legacy DFL-field gain 6.0 is calibrated in) instead of pixels — pixel-frame L1 ran 8-32x large and was **97.4%** of the val objective, starving classification (Det-A attempts 1-2 root cause: recall floor 17-24%, mAP 3.96/6.28 vs >25). A13 revised; `strides` threaded `module -> DualBranchLoss -> DetectionBranchLoss` (added 2026-08-04, attempt-2 diagnosis) | `test_detection_loss.py` stride-normalization test; overfit-100 golden revalidated | 027,028,034 | ✅ |
| 079 | `fix(data): per-worker, per-epoch augmentation RNG` | `worker_init_fn` re-seeds each worker's pipeline generator from `WorkerInfo.seed` (torch's `base_seed + worker_id`, redrawn from the loader generator every epoch). `_TrainPipeline` seeds one generator in the **parent**, which never advances when workers are used, so every worker replayed one identical augmentation-parameter stream (diversity `1/num_workers`) and — since WP-076 made workers non-persistent — restarted from that same state every epoch, replaying a single epoch's parameters for the whole run (measured: 1/3200 of intended draws at 32 workers x 100 epochs). Determinism per seed preserved (added 2026-08-05, Det-A attempt-3 pre-flight diagnosis) | `tests/data/test_coco.py` per-worker re-seed + thread-cap tests | 014,076 | ✅ |
| 080 | `feat(docs): training-curve vector figures` | `scripts/plot_training.py` renders a run's `metrics.csv` as a deterministic three-panel SVG (val/mAP, epoch-mean losses, one-to-one pre-gain components); fixed SVG hashsalt and suppressed date metadata so an unchanged run regenerates byte-identically. Panels degrade gracefully on runs predating a column (added 2026-08-06) | figure regenerates from `lightning_logs/version_8`; `epoch_means` doctest | 045 | ✅ |
| 081 | `docs: Det-A report section and detector model card` | `REPRODUCTION_REPORT.md` opens as the living D10 document with the 0.1.0 detection section (mechanism-to-module map, acceptance table, both root-cause narratives, assumption outcomes, four recorded deviations); `MODEL_CARD_DETECTION.md` covers intended and non-intended use, COCO composition limits, limitations, ethical considerations and the clean-room statement (added 2026-08-06) | both docs pinned in `test_docs_present.py` | 045,080 | ✅ |
| 082 | `test(losses): objective invariant gate` | Pins the coordinate frame the L1 term is measured in (`1/stride` against the pixel frame, parametrized per level) and its scale on a near-converged state (0.75 stride-units vs 5.99 in the broken frame); a loader-level test spawns real workers and asserts two epochs do not replay one augmentation stream. Deliberately omits an absolute share bound — on a converged fixture the broken frame reads 10.7% against 1.5%, so such a test would look like protection and catch nothing (added 2026-08-06, after WP-078/079 survived 474 tests and three tier runs) | `tests/losses/test_loss_invariants.py`; `test_coco.py` worker-epoch divergence | 078,079 | ✅ |
| 045 | `exp(det): Det-A smoke tier and report section` [GPU][HUMAN] | n-scale ~50 epochs; Det-A criteria; report + model card. **ACCEPTED 2026-08-06** (run v8, dev12, batch 128 x 50 ep, lr 0.02): val mAP50-95 **25.30** EMA-NMS / 25.33 raw-NMS vs >25; E2E deficit 1.46 EMA / 1.11 raw vs ~1.5; stable training; gates green. Attempts 1-2 (2026-08-03/04) not accepted at 3.96 / 6.28 — root cause pixel-frame L1 (WP-078), compounded by replayed augmentation streams (WP-079); records in `.experiments/det_a/` | Det-A acceptance met; artifacts archived with seeds/configs | 044,040,072,078,079 | ✅ |
| 083 | `test(eval): synthetic-shapes generalization golden` | The cheap stand-in for a tier run: 2000 generated scenes split 1800/200, n-scale, 6 epochs at 320 px, held-out val scored through both decode paths. Unlike the overfit-100 gate it never scores the images it trained on, so it measures generalization rather than loop composition; mosaic stays on for five of six epochs so a collapsed augmentation RNG (WP-079) is detectable. Frozen at NMS mAP50-95 0.8823 / E2E 0.8583 / mAP50 0.9842, three MPS runs identical to four decimals, 127-169 s wall clock. The annotation reader both this and `eval_det.py` need moved into `lucid_yolo.eval.annotations` rather than being copied (added 2026-08-06) | `goldens/gpu/shapes_regression_det.json` via `check_goldens.py --include-gpu`; `tests/eval/test_annotations.py` | 082 | ✅ |
| 084 | `ci(repo): commit-time gates, split workflows, dev group` | Three checks move to where they can catch something. The copyleft audit leaves its standalone CI job for `.pre-commit-config.yaml` (`always_run`, since it scans the installed environment rather than the diff, and a transitive copyleft bump touches no tracked file). The commit-trailer validator leaves CI entirely for a `commit-msg` hook: under a squash merge the per-commit messages CI validated are replaced by the squash message, so the job gated text that never lands. The `dev` extra becomes a PEP 735 `[dependency-groups]` entry so build/lint/test packages stop advertising themselves in the published wheel's metadata; installers move to `-e . --group dev`. Linting splits into `lint.yml` (job `precommit`) and `ci.yml` becomes `ci-tests.yml` (job `testing`). Test-suite and docs-updated checks are deliberately left unhooked (added 2026-08-06, user request) | `Provides-Extra: None` on the installed dist; the `commit-msg` hook rejects a malformed message and passes a valid one; `tests/meta/test_license_audit.py` keeps the rejection path covered | 004 | ✅ |
| 046 | `release: v0.1.0 detector` [HUMAN] | O3 cleared; CHANGELOG; weights published; goldens frozen to `0.1` | `release.yml` green on tag `v0.1.0` | 045 | ⬜ |

## Phase 7 — Instance segmentation, release 0.2.0 (WP-047…054)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 047 | `feat(models): mask coefficient branch` | K=32 tanh coefficients per location (A14, A16), opt-in behind `num_coeffs`: `None` builds no stem, so the accepted detector's module tree, 205600 parameters and forward output are bit-identical. Stem mirrors the class stem without the sigmoid prior-probability bias, registered as A34. Landed ahead of its listed dep 046, which is a release-ordering gate rather than a technical prerequisite (2026-08-06) | `test_segment_head.py::test_coeff_shapes`; off-by-default state-dict/parameter identity; `params_flops_det.json` unchanged | 046 | ✅ |
| 048 | `feat(models): multi-scale proto pathway` | Eq. 8: F_proto = X1 + sum U(phi_l(X_l)) as a standalone `ProtoFusion`, wired into nothing yet so the accepted detector cannot move. P3 is added unprojected (no phi_1); P4/P5 take bare 1x1 projections into P3's width and interpolate to P3's exact size rather than a fixed scale factor, so odd and non-square maps stay aligned. Projection form and upsample mode registered as A35 (2026-08-06) | `test_proto.py::test_fusion_eq8` (closed-form, mutation-checked); identity when projections vanish; odd/non-square target size | 047 | ✅ |
| 049 | `feat(models): prototype generation stack` | Eq. 9: `ProtoNet` maps the fused feature to K raw prototype maps at twice P3, i.e. 160x160 at 640 (A15). Three 3x3 `ConvBNAct` units, a 2x nearest upsample, a fourth 3x3 unit at the proto grid, then 1x1 to K — registered as A18, which replaces its former "protonet-style conv stack" placeholder. Output is deliberately unactivated: coefficients already carry tanh (A16), so squashing prototypes too would compress the linear mask combination twice. `scale_factor=2` is correct here where `ProtoFusion` interpolates to a target size, because A15 defines the grid as exactly twice P3 rather than as matching another tensor. Still wired into nothing (2026-08-06) | `test_proto.py::test_proto_resolution` (odd non-square case included); unactivated output; upsample-before-final-conv proven both structurally and by 2x2-block refinement | 048 | ✅ |
| 050 | `feat(models): auxiliary semantic branch (training-only)` | `SemanticAux`: dense per-class raw logits on F_proto at its own resolution, `forward` returning `None` whenever `self.training` is False. A17's "lightweight conv" is read as exactly one 1x1 `nn.Conv2d` and pinned numerically, so WP-052's fused-parameter gate knows precisely what disappears. Carries the A30 prior-probability bias — this is the dense sigmoid classifier over mostly-background that A30 was written for, unlike WP-047's tanh coefficients. `_CLS_PRIOR_PROB` became public `CLS_PRIOR_PROB` plus an `init_cls_prior_bias` helper so the formula has one home; proven behaviour-neutral against the previous revision (312 identical keys, 205600 identical params, identical init tensors, bit-identical forward). Still wired into nothing (2026-08-06) | `test_aux_semantic.py::test_eval_mode_inactive` (asserted under both grad-enabled eval and `no_grad`); single-conv parameter count; prior-init and raw-logit gates; `params_flops_det` unchanged | 049 | ✅ |
| 051 | `feat(losses): instance mask loss and BCE+Dice auxiliary` | Box-cropped BCE normalized by box area (A16); equal-weight BCE+Dice | `test_mask_loss.py`, `test_semantic_aux.py` | 050 | ⬜ |
| 052 | `feat(models): fuse() strips aux branch; seg fidelity gate` | Fused params = pre-fusion minus aux head; vs Table S9, all 5 scales | `test_param_flops.py::test_seg_vs_tableS9` — **golden frozen** | 051 | ⬜ |
| 053 | `feat(eval): segmentation decode and segm evaluation` | Masks assembled post-top-k; pycocotools bbox+segm, both paths | `test_seg_eval.py::test_bbox_and_segm` | 052 | ⬜ |
| 054 | `release: v0.2.0 segmentation` [GPU][HUMAN] | Seg-A tier + overfit-100 (mask IoU >=0.7); 0.1 goldens still green; tag | Seg-A acceptance; `release.yml` green on `v0.2.0` | 053 | ⬜ |

## Phase 8 — Oriented detection, release 0.3.0 (WP-055…064)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 055 | `feat(data): rotated geometry primitives` | Long-edge canonicalization (w>=h, theta in [-45,135) deg), point-in-rotated-rect, polygon conversion (A23) | `test_rotated_geom.py::test_theta_pi_equivalence`, `::test_canonicalization` | 054 | ⬜ |
| 056 | `feat(data): DOTA parsing and long-edge conversion` [DATA] | 15 classes; quad -> long-edge at load | `test_dota_parse.py`; `check-data` counts match | 055 | ⬜ |
| 057 | `feat(data): 1024 px overlapping crop tiling` [DATA] | 200 px overlap (A21); derived set is a build artifact | `test_tiling.py::test_coverage_no_gaps` | 056 | ⬜ |
| 058 | `feat(data): rotated-aware augmentation` | Affine/flip/mosaic on rotated boxes with re-canonicalization | `test_rotated_aug.py::test_roundtrip` | 057 | ⬜ |
| 059 | `feat(losses): ProbIoU rotated loss` | Gaussian-box ProbIoU (R17, A19) | `test_probiou.py::test_vs_shapely_oracle` | 055 | ⬜ |
| 060 | `feat(losses): square-object angle loss` | Eq. 14–15, lambda=3 (A22 weight=1.0); mod-pi wrap; omega aspect factor | `test_angle_loss.py::test_wrap_range`, `::test_omega_profile`, `::test_sin2_extrema`, `::test_boundary_continuity` | 059 | ⬜ |
| 061 | `feat(assign): rotated containment for TAL and STAL` | Point-in-rotated-rect candidates; STAL clamp on rotated (w,h) (A25) | `test_rotated_assign.py::test_tiny_rotated_gt` | 058,060 | ⬜ |
| 062 | `feat(models): OBB head, direct angle, NMS-free decode` | Angle branch (A20); theta = z (Eq. 13); one-to-one decode + canonicalization; vs Table S11 at 1024 | `test_obb_head.py`, `test_param_flops.py::test_obb_vs_tableS11` — **golden frozen** | 061 | ⬜ |
| 063 | `feat(eval): rotated mAP on DOTA val` | Exact polygon-intersection IoU (A24) with shapely oracle in tests | `test_dota_eval.py::test_vs_oracle` | 062 | ⬜ |
| 064 | `release: v0.3.0 oriented detection` [GPU][HUMAN] | OBB-A tier + overfit-100 (rotated mAP50 >=0.9); O4 resolved; 0.1/0.2 goldens green | OBB-A acceptance; `release.yml` green on `v0.3.0` | 063 | ⬜ |

## Phase 9 — Consolidation, release 0.4.0, then rolling (WP-065…067)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 065 | `docs(report): consolidated multi-task reproduction note` | Merge det/seg/obb sections; assumption outcomes; deviations and hypotheses | `test_docs_present.py::test_report_sections` | 064 | ⬜ |
| 066 | `feat(export): ONNX export smoke test for E2E paths` | Verifies the paper's export claim; no NMS ops in the graph | `test_onnx_export.py::test_e2e_graph_ops` | 065 | ⬜ |
| 067 | `release: v0.4.0 consolidated note and examples` [HUMAN] | `supervision` example notebook (boxes/masks/rboxes); tag; roadmap reopened for the next 0.MINOR | `release.yml` green on `v0.4.0` | 066 | ⬜ |
| 074 | `feat(data): RF100-VL incremental downloader and merged COCO layout` [DATA] | Extend `lucid-download` with a `rf100-vl` collection mode over the `rf100vl` pip package (Roboflow Universe API key; 100 COCO-JSON datasets, Apache-2.0, <https://github.com/roboflow/rf100-vl>): per-dataset download -> remap category ids into a union label space (dataset-qualified names to avoid cross-domain collisions) -> re-id images -> append into one merged COCO layout -> delete the per-dataset archive before fetching the next, bounding peak disk to O(one dataset + merged). Splits preserved; `RF20-VL` subset flag for smoke runs. Open design points recorded before code: union-label-space vs per-dataset eval, verify-module integration (added 2026-08-04, user request) | offline unit suite over a synthetic two-dataset fixture; merged-layout `check_data.py` parity | 068,073 | ⬜ |
| 075 | `exp(det): RF100-VL generalization tier` [GPU][HUMAN] | Fine-tune/eval the detector on the merged RF100-VL (RF20-VL smoke first); recipe, per-domain metric breakdown, report section | acceptance criteria set with the WP-074 design; artifacts archived with seeds/configs | 045,074 | ⬜ |

Beyond 0.4.0 the train continues on the same discipline — pose/RLE (R14),
classification, export matrix, RF100-VL generalization (WP-074/075) — each a
new phase of WPs and its own gated 0.MINOR. No 1.0, ever (ADR-002).

## Critical path and parallelism

Strictly sequential: 001 -> 007 -> 015 -> 023 -> 030 -> 033 -> 040 -> 046.
WP-016…019 (blocks), WP-024 (CIoU), and WP-031 (Newton–Schulz) have no
interdependencies and may be executed in any order once WP-003 lands. Phases 7
and 8 are strictly gated on releases 0.1.0 and 0.2.0 respectively — task heads
never land on an unproven detector.

## Failure budget

If a WP's DoD cannot be met after two documented assumption iterations,
escalate (AGENTS.md sec. 4). Fidelity-gate WPs (023, 052, 062) are the
expected escalation sites: they are where the papers' block-level ambiguity
meets a hard published number, and where an honest reproduction earns its
credibility.
