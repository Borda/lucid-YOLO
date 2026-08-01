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
| A3 | C3k2/C2PSA sub-block internals beyond Fig. S2 | Bottleneck/PSA internals per YOLO11 lineage | R11, R1 Fig. S2 | Param/FLOP gate (Phase 2) | open |
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
| A26 | Blueprint prescribes hand-annotated permissive fixture images; none exist | Test fixtures AND [DATA]-WP stand-ins are generated synthetically with fuse-augmentations (R21), seeded; byte-identical per seed **on a given platform** (revised 2026-08-01: libm last-bit rounding differs across OS/architecture, so cross-platform reproducibility is asserted via structural metrics with tolerance in the golden harness, not byte hashes — observed macOS arm64 vs ubuntu x86_64 CI divergence); det/seg 16 scenes with boxes+polygons, OBB rotated scenes. Real COCO/DOTA remain required for tier runs (Phases 6–8); synthetic stand-ins never substitute for tier acceptance | R21 | `tests/fixtures/test_fixtures_load.py` (same-platform byte determinism); `goldens/fixture_checksums.json` (cross-platform structural) | active |
| A27 | MuSGD hybrid: momentum-state sharing and Nesterov placement unspecified | one momentum buffer per param; both branch updates derived from Nesterov-adjusted g + mu*m; Muon branch per R7 Eq. 4 scaling | R1 3.3.1; R7 Eq. 4; R8 | WP-033 toy convergence golden; Tier B MuSGD-vs-SGD trend | active |

## Deviation notes

- **A26** is a deliberate deviation from blueprint section 7 (fixtures were
  specified as "16 permissively-licensed hand-annotated images"). Decision
  D12 (DECISIONS.md) records the rationale: synthetic scenes are fully
  redistributable, deterministic, and carry exact ground truth for boxes,
  polygons, and rotated boxes from one generator, removing annotation noise
  from unit gates. The dataset-contract rule "missing data blocks the WP, not
  worked around with synthetic substitutes" still applies to tier acceptance
  runs — A26 covers fixtures and offline development stand-ins only.
