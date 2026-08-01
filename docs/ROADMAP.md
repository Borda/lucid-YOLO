# Work-Package Roadmap

The agent's work queue: 67 work packages, one commit each, executed in
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
| 011 | `feat(data): mosaic assembly` | 4-image mosaic (R9), border handling, target remap | `test_mosaic.py::test_bounds_and_counts` | 010 | ⬜ |
| 012 | `feat(data): mixup and copy-paste` | Table S3 probabilities, scale-aware policy | `test_mixup_copypaste.py` | 011 | ⬜ |
| 013 | `feat(data): HSV jitter and horizontal flip` | hsv_h/s/v, fliplr=0.5 with target mirroring | `test_photometric.py` | 010 | ⬜ |
| 014 | `feat(data): COCO dataset and LightningDataModule` [DATA] | Detection + polygon parsing, scale-aware augmentation policy, `make check-data` | `test_coco.py` (fixture-backed) + `check-data` on real COCO | 012,013 | ⬜ |
| 015 | `test(data): round-trip goldens and debug visualizer` | Augmented-batch checksums; annotated grid dump script | `goldens/data_checksums.json` frozen | 014 | ⬜ |

## Phase 2 — Architecture (WP-016…023)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 016 | `feat(models): Conv, DWConv, Bottleneck primitives` | Conv-BN-SiLU, depthwise variant, residual bottleneck | `test_blocks.py::test_primitives` | 003 | ✅ |
| 017 | `feat(models): C3k2 block` | CSP split, n inner blocks, e ratio, c3k switch (A3) | `test_blocks.py::test_c3k2_shapes` | 016 | ✅ |
| 018 | `feat(models): PSABlock and C2PSA` | Attention + FFN block; split/concat wrapper (A3) | `test_blocks.py::test_c2psa` | 016 | ✅ |
| 019 | `feat(models): SPPF with shortcut` | 1x1 -> 3x MaxPool(5) -> concat -> 1x1, plus input-output shortcut (A4) | `test_blocks.py::test_sppf_shortcut` | 016 | ⬜ |
| 020 | `feat(models): backbone` | Backbone stack with P3/P4/P5 taps | `test_backbone.py::test_tap_shapes` | 017,018,019 | ⬜ |
| 021 | `feat(models): neck with attention tail` | Top-down/bottom-up; final C3k2 n=1 e=0.5 attn=True | `test_neck.py::test_output_shapes` | 020 | ⬜ |
| 022 | `feat(models): dual detection head, reg_max=1` | o2o (300x6) + o2m (nc+4, 8400) branches, DFL-free ltrb regression (A9) | `test_head.py::test_dual_head_shapes` | 021 | ⬜ |
| 023 | `feat(models): scale registry, builder, param/FLOP fidelity gate` | 5-row dataclass registry, typed builders (ADR-001), fvcore counting | `test_param_flops.py::test_det_vs_table7` — plus/minus 2% params / 5% FLOPs, all 5 scales; **golden frozen** | 022 | ⬜ |

## Phase 3 — Assignment and losses (WP-024…030)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 024 | `feat(losses): CIoU` | CIoU per R10 (A1), batched, autograd-safe | `test_ciou.py::test_against_closed_form` | 003 | ✅ |
| 025 | `feat(assign): anchor grid and Task-Aligned Assigner` | Centers at (i+0.5)*stride (A11); t = s^1 * u^6 (A2); topk selection | `test_tal.py::test_alignment_and_topk` | 024 | ✅ |
| 026 | `feat(assign): STAL surrogate candidate filtering` | Eq. 4–6; per-dimension clamp d<8 -> 16; original box preserved for scoring/regression | `test_stal.py::test_tiny_box_gains_candidates`, `::test_per_dim_clamp`, `::test_targets_unchanged` | 025 | ✅ |
| 027 | `feat(losses): detection branch loss` | CIoU + L1 (dfl-gain field, A13) + BCE, TAL-weighted | `test_detection_loss.py::test_components` | 026 | ⬜ |
| 028 | `feat(losses): dual-branch composition` | o2m topk=10 / o2o topk=7->1 wiring; static alpha combination (schedule lands WP-035) | `test_dual_loss.py::test_one_positive_per_gt` | 027 | ⬜ |
| 029 | `test(assign): synthetic assignment goldens` | 6x6 px GT: STAL >=1 candidate, vanilla TAL exactly 0 at stride 8 | `goldens/assignment_cases.json` frozen | 028 | ⬜ |
| 030 | `test(train): single-batch overfit and gradient flow` | 200-step monotonic loss decrease; no NaN/Inf; all leaf grads populated | `test_overfit_batch.py` | 029 | ⬜ |

## Phase 4 — MuSGD (WP-031…033)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 031 | `feat(optim): Newton-Schulz orthogonalization` | Pure function from R7/R8, 5 iterations (A5), fp32 under AMP | `test_newton_schulz.py::test_orthogonality` | 003 | ✅ |
| 032 | `feat(optim): MuSGD with parameter-type split` | >=2D: w_muon*Muon + w_sgd*SGD (A6, A7); 1D: pure SGD, no weight decay (A12) | `test_musgd.py::test_param_split`, `::test_step_shapes` | 031 | ✅ |
| 033 | `test(optim): toy convergence golden vs SGD` | Fixed synthetic regression + micro-CNN; MuSGD reaches threshold in fewer steps | `goldens/optim_toy.json` frozen | 032 | ✅ |

## Phase 5 — Lightning training loop (WP-034…040)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 034 | `feat(ptl): LightningModule with task-conditional losses` | Automatic optimization; det losses active, seg/obb hooks inert | `test_module.py::test_training_step` | 030,032 | ⬜ |
| 035 | `feat(ptl): ProgressiveLossSchedule hook` | Eq. 3 in `on_train_epoch_start`, (0.8,0.2)->(0.1,0.9) | `test_proglos.py::test_alpha_at_t0_mid_end` | 034 | ⬜ |
| 036 | `feat(ptl): CloseMosaic callback` | Disables mosaic for final `close_mosaic` epochs | `test_close_mosaic.py::test_flip_epoch` | 034 | ⬜ |
| 037 | `feat(ptl): EMA callback` | Decay schedule, checkpointed, used for eval | `test_ema.py::test_shadow_updates` | 034 | ⬜ |
| 038 | `feat(ptl): LightningCLI entry and experiment configs` | `configs/` tier matrix (ADR-001); resolved config logged per run | `test_cli.py::test_yaml_roundtrip`; all configs dry-parse | 035,036,037 | ⬜ |
| 039 | `feat(ptl): deterministic checkpoint and resume` | Seeded resume reproduces the loss trajectory within tolerance | `test_resume.py::test_trajectory_match` | 038 | ⬜ |
| 040 | `test(lit): overfit-100 integration golden` [GPU] | n-scale on a 100-image subset -> >=0.95 recall at IoU 0.5 on train | `goldens/overfit_micro_det.json` frozen | 039 | ⬜ |

## Phase 6 — Evaluation, release 0.1.0 (WP-041…046)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 041 | `feat(decode): score-based top-k E2E decoding` | No IoU, no NMS, cap 300 (R3 sec. 4, A9) | `test_topk_e2e.py::test_no_nms_path` | 022 | ⬜ |
| 042 | `feat(decode): NMS path for the dense branch` | Conf threshold + class-wise NMS (torchvision) | `test_nms_path.py` | 041 | ⬜ |
| 043 | `feat(eval): pycocotools bbox evaluator, both paths` | One command evaluates E2E and non-E2E from one checkpoint | `test_coco_eval.py::test_dual_path_report` | 042 | ⬜ |
| 044 | `test(eval): oracle round-trip` | Perfect predictions -> mAP 1.0; shuffled classes -> approx 0 | `test_coco_eval.py::test_oracle` | 043 | ⬜ |
| 045 | `exp(det): Det-A smoke tier and report section` [GPU][HUMAN] | n-scale ~50 epochs; Det-A criteria; report + model card | Det-A acceptance met; artifacts archived with seeds/configs | 044,040 | ⬜ |
| 046 | `release: v0.1.0 detector` [HUMAN] | O3 cleared; CHANGELOG; weights published; goldens frozen to `0.1` | `release.yml` green on tag `v0.1.0` | 045 | ⬜ |

## Phase 7 — Instance segmentation, release 0.2.0 (WP-047…054)

| WP | Commit subject | Scope | DoD | Dep | Status |
|---|---|---|---|---|---|
| 047 | `feat(models): mask coefficient branch` | K=32 tanh coefficients per location (A14, A16) | `test_segment_head.py::test_coeff_shapes` | 046 | ⬜ |
| 048 | `feat(models): multi-scale proto pathway` | Eq. 8: F_proto = X1 + sum U(phi_l(X_l)) | `test_proto.py::test_fusion_eq8` | 047 | ⬜ |
| 049 | `feat(models): prototype generation stack` | Eq. 9 protonet, 160x160 at 640 (A15, A18) | `test_proto.py::test_proto_resolution` | 048 | ⬜ |
| 050 | `feat(models): auxiliary semantic branch (training-only)` | Dense per-class logits on F_proto (A17); inactive at eval | `test_aux_semantic.py::test_eval_mode_inactive` | 049 | ⬜ |
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

Beyond 0.4.0 the train continues on the same discipline — pose/RLE (R14),
classification, export matrix — each a new phase of WPs and its own gated
0.MINOR. No 1.0, ever (ADR-002).

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
