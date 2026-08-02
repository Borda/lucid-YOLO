# Assumption Register

Anywhere the papers underdetermine the implementation, the chosen assumption,
its public source, and its validation plan are recorded here **before** the
corresponding code lands. Rule: any new gap discovered during implementation
gets an entry before the code merges. Assumption revisions after a release ship
as PATCH (if within golden tolerances) or the next 0.MINOR (if results move).

Status legend: `open` = code not yet landed · `active` = in the codebase ·
`validated` = validation plan executed and passed · `revised` = superseded
(revision history kept inline).

| ID | Gap in papers | Assumption | Public source | Validation | Status |
|---|---|---|---|---|---|
| A1 | Box IoU loss variant unnamed | CIoU | R9, R10 | Ablation-neutral check at Tier A | open |
| A2 | TAL alpha/beta not restated | alpha=1, beta=6 | R4 | Overfit + Tier B trend | open |
| A3 | C3k2/C2PSA sub-block internals beyond Fig. S2 | Bottleneck/PSA internals per YOLO11 lineage; nested `C3k` inner-bottleneck count **1** (`_C3K_INNER_UNITS`, revised 2026-08-01 from 2 by WP-023 iteration 2 — at depth=1.0 the `l`/`x` variants carry two nested units per `c3k=True` stage and `n=2` overshot the R1 Table 7 FLOP budget; `n=1` lands all five scales within +/-5% FLOPs while keeping params within +/-2%) | R11, R1 Fig. S2 | Param/FLOP gate (Phase 2) — `tests/models/test_param_flops.py::test_det_vs_table7` passes all 5 scales | active |
| A4 | SPPF shortcut exact form | Input added to output of pooling stack | R3 sec. 4 | Param gate; ablate if mismatch | open |
| A5 | Newton–Schulz iteration count | 5 | R7, R8 | Orthogonality unit test | open |
| A6 | Muon step scaling | 0.2 * sqrt(max(A,B)) update-RMS scaling | R7 Eq. 4 (verbatim: 0.2 * O * sqrt(max(A,B))) | Toy convergence test | active |
| A7 | muon_w+sgd_w != 1 semantics | Independent additive gains | R1 Tables S4/S7 | Documented; sensitivity note | open |
| A8 | LR schedule shape (lrf semantics) | Linear decay to final LR = lr0*lrf | gap; convention widely restated in third-party YOLO-application literature | Trend-neutral by construction — identical schedule in all paired B-tier runs | open |
| A9 | One-to-one output tuple (...,6) | [x1,y1,x2,y2,score,class]; seg appends K coefficients | R1 sec. 3.2.1 | Eval round-trip test | open |
| A10 | Resize semantics | Letterbox, aspect-preserving | YOLO-lineage convention in third-party literature (e.g. R11) | Bbox/mask round-trip test | open |
| A11 | Anchor center placement | (i+0.5)*stride | R4, R5 | Assignment unit test | open |
| A12 | Weight-decay exclusions | 1D params excluded | Standard practice (e.g. R7 discussion) | Documented | open |
| A13 | From-scratch loss gains | Pretrain-style set (7.5/0.5/6.0) | R1 Table S2 | Tier A sanity | open |
| A14 | Prototype count K | K=32 | YOLACT R16 | Param/FLOP gate vs Table S9 | open |
| A15 | Proto spatial resolution | 160x160 at 640 input (2x upsample of P3) | YOLACT convention R16 | Param/FLOP gate; mask-quality check | open |
| A16 | Coefficient activation + instance-mask loss | tanh coefficients; per-pixel BCE on box-cropped masks, box-area normalized | YOLACT R16 | Overfit micro-set mask IoU | open |
| A17 | Aux semantic head structure | Lightweight conv to nc logits on F_proto | R1 sec. 3.4.1 ("training-only branch") | Fused-model param gate proves removal | open |
| A18 | Proto-generation stack internals | Protonet-style conv stack | YOLACT R16; R1 Eq. 9 | Param/FLOP gate | open |
| A19 | Rotated IoU loss variant | ProbIoU | R1 sec. 3.4.3 lineage; R17 | Tier OBB-B trend (Table 11 ranking) | open |
| A20 | Angle branch structure | Separate conv branch, 1 scalar/location | R1 sec. 3.4.3 ("separate branch") | Param/FLOP gate vs Table S11 | open |
| A21 | DOTA crop overlap | 200 px | DOTA devkit / MMRotate convention R13, R18 | Documented; coverage sanity test | open |
| A22 | L_angle scalar weight | 1.0 | gap | Sensitivity run at Tier OBB-A | open |
| A23 | Rotated decode normalization | Canonicalize to long-edge form post-decode | R13 | Boundary-continuity test | open |
| A24 | Rotated eval IoU | Exact polygon intersection (val protocol) | R18 devkit convention | Oracle test vs brute force | open |
| A25 | STAL/TAL containment for rotated GT | Point-in-rotated-rect; clamp on rotated (w,h) | R1 sec. 3.3.3 (generic formulation) | Synthetic rotated assignment cases | open |
| A28 | Detection-head stem structure and hidden width (Fig. S2 shows the stem pair, not widths) | Both box and class stems are **depthwise-separable** (two `DepthwiseConv+ConvBNAct` units + 1x1 output), shared hidden width `max(16, channels // 3)` (`_stem_width`); no `num_classes` floor (WP-023 iteration 1). Revised 2026-08-01 from the original full-conv box stem (`channels // 4`, two `3x3` convs) + class stem (`max(num_classes, channels // 2)`): the WP-023 param gate flagged the head as ~7-16x the reference budget (the box stem alone was ~3.7x the class stem despite emitting 4 vs `num_classes` channels, and the `num_classes` floor bloated the `n`/`s` head). The lightweight symmetric form lands all five scales within tolerance | R1 Fig. S2; YOLOv10/R6 lightweight-head lineage | Param/FLOP gate (Phase 2) — `test_det_vs_table7` | active |
| A29 | R1 Table 7 param/FLOP counting convention | GFLOPs = **2x** fvcore MACs (conventional one-multiply-plus-one-add FLOPs; raw MACs land ~48% low). **Params** count the full trained checkpoint (both dual-head branches); **FLOPs** count the deployed NMS-free inference model only (backbone+neck+one-to-one head — the one-to-many branch is training-only and never executed at E2E inference). Determined empirically by the WP-023 gate (trying raw-MAC and both-branch FLOPs is measurement, not iteration) | R6 (dual-assignment inference); R1 Table 7 | Param/FLOP gate (Phase 2) — `test_det_vs_table7` | active |
| A26 | Blueprint prescribes hand-annotated permissive fixture images; none exist | Test fixtures AND [DATA]-WP stand-ins are generated synthetically with fuse-augmentations (R21), seeded; byte-identical per seed **on a given platform** (revised 2026-08-01: libm last-bit rounding differs across OS/architecture, so cross-platform reproducibility is asserted via structural metrics with tolerance in the golden harness, not byte hashes — observed macOS arm64 vs ubuntu x86_64 CI divergence); det/seg 16 scenes with boxes+polygons, OBB rotated scenes. Real COCO/DOTA remain required for tier runs (Phases 6–8); synthetic stand-ins never substitute for tier acceptance | R21 | `tests/fixtures/test_fixtures_load.py` (same-platform byte determinism); `goldens/fixture_checksums.json` (cross-platform structural) | active |
| A27 | MuSGD hybrid: momentum-state sharing and Nesterov placement unspecified | one momentum buffer per param; both branch updates derived from Nesterov-adjusted g + mu*m; Muon branch per R7 Eq. 4 scaling | R1 3.3.1; R7 Eq. 4; R8 | WP-033 toy convergence golden; Tier B MuSGD-vs-SGD trend | active |
| A30 | Head initialization unspecified (R1 silent on init) | Classification-output 1x1 conv bias initialized to -log((1-pi)/pi), pi=0.01 (RetinaNet prior-probability init). Without it the summed TAL-normalized BCE opens at ~1.8M nats and the first MuSGD step kills the network (observed on Det-A launch 2026-08-02: loss 6.2e5 -> exact-zero dead predictor in 3 steps, reproduced on CPU and MPS) | R24 sec. 5.1 | `test_head.py::test_cls_bias_prior_init`; Det-A stable-training criterion | active |
| A31 | From-scratch stabilization beyond init not restated (warmup deferred with A8) | Global gradient-norm clipping at 10.0 via the trainer for tier/overfit runs; generic deep-learning practice, not lineage-specific. 30-step real-COCO check: descending loss, box predictions bounded within image scale | gap; standard practice | Det-A stable-training criterion; revisit when the A8 schedule (warmup) lands | active |

## Deviation notes

- **A26** is a deliberate deviation from blueprint section 7 (fixtures were
  specified as "16 permissively-licensed hand-annotated images"). Decision
  D12 (DECISIONS.md) records the rationale: synthetic scenes are fully
  redistributable, deterministic, and carry exact ground truth for boxes,
  polygons, and rotated boxes from one generator, removing annotation noise
  from unit gates. The dataset-contract rule "missing data blocks the WP, not
  worked around with synthetic substitutes" still applies to tier acceptance
  runs — A26 covers fixtures and offline development stand-ins only.
