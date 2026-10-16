# 🧾 Consolidated note — detection, segmentation, oriented detection

This is not a fourth tier. No run stands behind it; it reads the three sections above against each other and states what only becomes visible once they sit side by side. Nothing here revises a claim those sections make — a correction to any of them belongs in a future section, named as one (D10) — and every number below already appears in `## 0.1.0`, `## 0.2.0` or `## 0.3.0`.

## One trunk, three heads

Det-smoke, Seg-smoke and OBB-smoke share the same DFL-free dual head, the same NMS-free one-to-one deploy path, the same TAL/STAL assignment, the same MuSGD optimizer, Progressive Loss, EMA, close-mosaic and warmup-then-linear-decay schedule. Each later tier adds a task-specific branch on top of that trunk rather than a new one: Seg-smoke adds prototype–coefficient masks (`ProtoNet`/`ProtoFusion`) and a training-only auxiliary semantic branch; OBB-smoke adds per-branch angle stems, a rotated ProbIoU term, a retargeted L1 target, and R1 Eq. 15's angle term.

The NMS-free deploy path's cost is the one quantity all three tiers could have reported in the same units, and reading the three rows together is what shows that only two of them did:

<!-- --8<-- [start:nms-cost] -->

| tier | path cost | source |
| -- | -- | -- |
| Det-smoke | 1.11 AP (raw), 1.46 AP (EMA) | "E2E within NMS path" acceptance row |
| Seg-smoke | 1.27 box AP, 0.81 segm AP | "The NMS-free path costs 1.27 box AP and 0.81 segm AP" |
| OBB-smoke | not measured | the run reports EMA and raw weights only; no NMS baseline was evaluated for the oriented path, so the one figure comparable across the first two tiers does not exist for the third |

<!-- --8<-- [end:nms-cost] -->

The absence in the third row is itself the finding, not a gap in this note: OBB-smoke deploys the same NMS-free architecture Det-smoke and Seg-smoke measure the cost of, and this project has never reported what that cost is for orientation.

No other cross-tier comparison is attempted here. Det-smoke and Seg-smoke differ in more than task — 16-mixed against bf16-mixed, different hardware, and the data-pipeline work that landed in between (0.2.0's own "offered as one" observation) — and OBB-smoke's figures are per tile, comparable to nothing outside this project (0.3.0's "not comparable to published DOTA results"). Consolidating three sections does not manufacture the comparisons those sections declined to make.

## Assumption outcomes, consolidated

Union of the three per-tier tables, 23 rows, one per id — no id repeats across tiers by direct table membership. Two ids cross tiers only by textual reference rather than a second table row: A11 is Det-smoke's anchor-centre rule, and Seg-smoke's own row says its mask crop leaves that rule "unchanged from detection" rather than defining a second one; A13 sets Det-smoke's loss gains, and OBB-smoke's A49 row names it directly — the rotated ProbIoU term is carried "at the gain A13 set for the axis-aligned term". Outcome wording is quoted from each tier's own table, not restated. `†` marks **held** or **not exercised at its recorded value** — a value that carried a run without being isolated, not one that was checked.

| id | subject | tier(s) | outcome, as recorded |
| -- | -- | -- | -- |
| A2 | TAL exponents | det | held at alpha 1, beta 6; not independently ablated at A-tier `†` |
| A8 | LR schedule shape | det | validated |
| A11 | anchor centre placement | det, seg | unchanged from detection; the mask crop reuses its `+0.5` half-open rule |
| A13 | loss gains and L1 coordinate frame | det (obb's A49 reuses its gain) | revised, then validated |
| A15 | prototype resolution, 160×160 at 640 input | seg | held — the size-bucket split is where its cost is visible `†` |
| A16 | tanh coefficients, box-cropped area-normalized BCE | seg | validated at the wiring scale |
| A17 | training-only auxiliary semantic branch | seg | validated |
| A21 | DOTA crop overlap | obb | not exercised at its recorded value — the tier ran at R18's 512 px, not the register's 200 px default `†` |
| A22 | angle-term weight, 0.25 | obb | held `†` |
| A24 | rotated-IoU metric and its protocol | obb | held `†` |
| A25 | rotated candidacy in TAL/STAL | obb | held `†` |
| A30 | classification bias prior init | det | validated |
| A31 | warmup convention | det | absorbed into A8 |
| A32 | train-time resampling filter | det | non-antialiased fused warp; no acceptance-visible effect |
| A33 | uint8 batch transport | det | validated — no acceptance-visible effect, 4× smaller IPC |
| A37 | mask-decode numerics the papers leave open | seg | held `†` |
| A38 | mask/semantic gains and which branch is supervised | seg | held `†` |
| A44 | how the angle composes with ltrb | obb | carries the run, and bounds it |
| A45 | oriented output tuple, width 7 | obb | held `†` |
| A48 | difficult-instance matching | obb | exercised at scale |
| A49 | Hellinger ProbIoU at gain 7.5 | obb | held `†` |
| A50 | retargeted L1 at gain 6.0 | obb | held `†` |
| A52 | object-free crops kept | obb | exercised |

Eleven rows carry `†`. That is not eleven weak results — the wiring gates and acceptance criteria those runs cleared are real — but it is eleven assumptions a passing run has not, by itself, isolated. A25's own row states the sharpest version of this for its own id: the wiring gate at 0.94 is "the evidence that positives reach the right anchors", which is a statement about candidacy, not proof that every clause of the assumption is correct. A21 is the one row where the record explicitly says the assumption itself remains unmeasured even though the run it was supposed to govern completed without incident.

## Deviations

**Common to all three tiers**, each stated once per section and repeated here because a deviation three runs share is a property of the project, not of any one run: ~50 epochs at n-scale against the paper's 500/600-epoch from-scratch schedules (a smoke tier by design in every case); no Objects365 pretraining and no evolutionary hyperparameter search (D2); single seed, seed 0, throughout every run.

**Specific to one tier:**

- Det-smoke — batch-scaled learning rate (lr 0.02 at batch 128, linear-scaled from the batch-64 recipe) with no clean measurement of the unscaled value, since the one run that carried it also carried the loss-frame defect; the E2E deficit itself measured wider than [R1 sec. 4.4]'s published 0.6–0.8 AP.
- Seg-smoke — bf16-mixed where Det-smoke ran 16-mixed, a hardware-driven choice never measured against the alternative; `lr` overridden to 0.02 against `seg_smoke.yaml`'s 0.01, the same batch-128 scaling Det-smoke settled on; the mask and semantic gains (2.5, 0.5) are this project's own, R1 giving neither; raw weights were not evaluated for this run, so the EMA contribution is unmeasured here where Det-smoke reports both.
- OBB-smoke — per-tile evaluation only, not the whole-image statistic [R1 Tables 10-11] report; batch 64 at 1024 px against 128 at 640 for the COCO tiers, 2.56× the pixels per image, with `lr` left at the config's 0.01 rather than scaled the way Det-smoke and Seg-smoke scaled theirs; 512 px tiling overlap, R18's own protocol figure rather than A21's 200 px register default; the angle gain (0.25) is this project's own, revised once already ([WP-093](../work_packages/wp-093.md)); the parameter gate has no published reference for this task family, unlike [R1 Table 7] and [R1 Table S9].

The learning-rate line is worth reading across tiers rather than within one: Det-smoke and Seg-smoke both scale lr with batch size from a batch-64 recipe; OBB-smoke, at half that batch and 2.56× the pixels per step, does not. No section argues the choice either way; it sits in three places as three separate facts until this note puts them next to each other.

## Hypotheses the record supports

Stated as hypotheses, each with the evidence already in this file and what would test it — none of them cleared to a conclusion here.

1. **The E2E-against-NMS deficit may be a genuine, reproducible excess over [R1 sec. 4.4]'s figure, not measurement noise.** Evidence: Det-smoke measures 1.11 AP (raw) / 1.46 AP (EMA) against the paper's reported 0.6–0.8 AP — right sign, right order of magnitude, wider. Test: the Det-smoke section already names this a Det-ablations question — whether the residual gap narrows at the paper's own 500/600-epoch schedule, which this project has not yet run.
2. **A15's prototype resolution (160×160 at 640 input) sets the small-object mask ceiling.** Evidence: the mask-to-box ratio by size bucket is 0.30 small / 0.70 medium / 0.91 large; at 160×160 one prototype cell covers 4 input pixels, so a 32-pixel object spans only 8 cells before quantization dominates whatever the coefficients predict. Test: Seg-smoke's own row names it as the first thing to ablate if masks are the target — rerun at a higher prototype resolution and re-read the same size-bucket split.
3. **A44's angle composition sets the oriented ceiling.** Evidence: `val/rotated_mAP50` is flat across the entire close-mosaic window (`-0.0001` from epoch 40 to 49), where the same window measurably lifted Det-smoke's accuracy. The section records two readings it cannot separate — the model has nothing left to gain at 50 epochs, or the ceiling is set by something the schedule does not touch — and names A44 as the candidate for the second. Test: an ablation of the composition itself (A44 records that the alternative readings — rotated-frame distances, or a rotated anchor offset — are foreclosed only by choice, not by R1), since more epochs is, in the section's own words, not the obvious next experiment.
4. **Not upgraded, stated as the confound it is: whether mask supervision costs box accuracy remains unmeasured.** Seg-smoke's box mAP is higher than Det-smoke's (26.12 against 25.30, EMA-NMS), but the two runs differ in precision, hardware and the data-pipeline work that landed between them, and each is a single seed — the section calls this "not a measurement of what mask supervision costs" and this note does not call it one either. Test: a paired run on matched hardware and precision, same seed set, isolating only the presence of the mask/semantic terms.
