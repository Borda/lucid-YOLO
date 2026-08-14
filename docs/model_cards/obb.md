# Model card — lucid-yolo oriented detector (n scale)

Covers the oriented detection model produced by the OBB-smoke tier run, released as `0.3.0`. That tier was accepted at its human gate (roadmap 064) on 2026-08-14, on the evidence recorded in `REPRODUCTION_REPORT.md`. One card per task family; detection and instance segmentation have their own.

## Model details

|  |  |
| -- | -- |
| Developed by | Independent reproduction, single operator; see `PROVENANCE.md` |
| Model type | Single-stage anchor-free oriented detector — DFL-free dual detection head with per-branch angle stems, NMS-free one-to-one deploy path |
| Scale | `n` — 2.56M parameters, 14.68 GFLOPs at 1024×1024 (`goldens/params_flops_obb.json`) |
| Input | RGB, letterboxed to 1024×1024, float32 in `[0, 1]` |
| Output | Per detection the A45 tuple `[cx, cy, w, h, theta, score, class]` — a long-edge rotated box, capped at 300 detections per image. The angle is gathered by the anchor index its own box was ranked by, never re-ranked separately (`models/heads/obb.py`) |
| Box composition | A44: the four ltrb distances give the axis-aligned rectangle, which the predicted `theta` then rotates about its own centre. The composition is minimal and its consequences are recorded under Limitations |
| Version | Run v10 checkpoint `epoch=49-step=23050.ckpt`, 2026-08-13, trained with `0.3.0.dev5` |
| License | Apache-2.0 (code and report). Weights derive from DOTA-v1.0 — see Licensing below, which is more restrictive than for the COCO-trained models |
| Paper | Methods per Ultralytics YOLO26, arXiv:2606.03748. Independent implementation; not affiliated with, endorsed by, or derived from Ultralytics or its codebase |

The oriented head is the detector's own dual head with `predict_angle` enabled (`models/build.py`); backbone and neck are unchanged. The task does not merely add a term — it **replaces** two. The dual loss is constructed with `box_gain = 0` and `l1_gain = 0` so its Complete-IoU and axis-aligned L1 are computed for logging and enter no total, and those gains are spent instead on the rotated ProbIoU of the assembled box (A49) and on an L1 retargeted onto that box's own `(cx, cy, w, h)` (A50). A reader of `metrics.csv` should know this: `val/o2o_box` and `val/o2o_l1` are live-looking curves that no gradient followed.

Parameter and FLOP fidelity is gated rather than asserted, but against a weaker reference than the other two tasks: [R1] publishes no oriented parameter table, so `test_param_flops.py` holds the oriented scales against this project's own frozen goldens. Those numbers pin the topology against drift; they do not corroborate it against the paper.

## Intended use

**Intended.** Reproduction research: verifying the paper's oriented-detection claims, ablating the A44 composition and the A49/A50 term choices, and serving as a readable from-scratch implementation of NMS-free oriented detection. As with the detector, the `e2e` path is the interesting artifact — oriented boxes come out of a forward containing no suppression op.

**Not intended.** Any operational use, and this is a stronger statement than for the COCO models. DOTA is aerial imagery; the categories are vehicles, ships, aircraft, storage tanks, harbours and sports facilities. Applications of aerial object detection include surveillance and targeting, and nothing about this model's licence, provenance or quality supports deployment for them. DOTA's own terms permit academic use only. Beyond that: production use of any kind, safety-critical or rights-affecting decisions, measurement of real-world object dimensions from predicted boxes, or any setting where a per-tile rotated mAP50 of 0.52 on 15 aerial categories would be read as a reliable perception system.

## Training data

DOTA-v1.0 (R18) `train` split — 1,411 images, 98,990 instances across 15 categories, oriented four-point annotations. No other data; no pretraining (D2). The `test` split's ground truth is withheld by the authors and was neither downloaded nor used.

Training does not read the original images. They run to several thousand pixels a side, and PNG has no random-access region decode, so each sample would decode a whole image to yield one 1024 px window; the tier trains on a 1024 px tiled layout built by `lucid-data build-tiles` (WP-094), with **512 px overlap** — R18's own protocol figure rather than A21's 200 px default.

DOTA's composition limits apply directly. It is satellite and aerial imagery of a particular set of places, with category frequencies that are severely long-tailed (`small-vehicle` dominates; `helicopter` and `ground-track-field` are rare), and object scale distributions unlike anything in ground-level photography. Nothing here corrects for it, and no evaluation across geographies has been run.

**Terms.** DOTA states: *"All images and their associated annotations in DOTA can be used for academic purposes only, but any commercial use is prohibited."* Google Earth imagery within it carries Google Earth's terms additionally. Weights trained on it inherit that restriction, which is why this card's Licensing section differs from the other two.

## Training procedure

|  |  |
| -- | -- |
| Recipe | batch 64 at 1024 px, 50 epochs (23,050 steps), MuSGD (`w_muon 0.5 / w_sgd 0.5`), lr 0.01 with 3-epoch warmup then linear decay to `lr0 × 0.01`, weight decay 5e-4, momentum 0.95, gradient clip 10.0 |
| Loss | classification 0.5 unchanged; rotated ProbIoU 7.5 in the IoU slot (A49, Hellinger form); rotated L1 6.0 in the L1 slot, stride-normalized (A50); angle 0.25 (A22, lowered from 1.0 by WP-093). Progressive branch weight `0.8 → 0.1` |
| Assignment | TAL (`alpha 1, beta 6`) with STAL small-target surrogates, one assignment shared by every term, with rotated **candidacy** only (A25) |
| Augmentation | mosaic `p=1.0` (disabled for the final 10 epochs), fused affine+letterbox, size-aware mixup and copy-paste, HSV jitter, horizontal flip `p=0.5` |
| Precision / seed | bf16-mixed, seed 0, `deterministic: true` |
| Hardware | one RTX PRO 6000, operator-reported — the run's own artifacts record no device, as with the earlier tiers |
| EMA | decay 0.9999, tau 2000 |

The wiring gate that precedes any DOTA launch is `scripts/overfit_micro.py --task obb`: 100 images, 592 instances, 4 classes, 100 epochs at 320 px, decoded through the deployed one-to-one path. It measures train rotated mAP50 **0.9390** against a 0.9 floor (`goldens/gpu/overfit_micro_obb.json`).

## Evaluation

Scored by `lucid-eval` on the DOTA-v1.0 **val** split tiled the same way training data was: 10,132 tiles, 101,209 instances, of which 16,528 are flagged difficult. Rotated mAP under the A24 protocol — exact polygon-intersection IoU, ten thresholds, 101-point interpolated recall, 300 detections per tile, R18's difficult rule.

| weights | rotated mAP50-95 | rotated mAP50 | rotated mAP75 | rotated mAR_300 |
| -- | -- | -- | -- | -- |
| EMA | **0.2914** | 0.5242 | 0.2770 | 0.5075 |
| raw | 0.2867 | 0.5140 | 0.2734 | 0.5013 |

EMA is worth `+0.0047` mAP50-95 and `+0.0102` mAP50 here — small, positive, and the opposite sign to Det-smoke, where raw weights scored marginally higher.

**Every number here is per tile.** Detections from overlapping tiles are not merged back onto whole images, because on an NMS-free path two tiles that both detect one object have nothing to suppress the duplicate, so the merge needs a policy decided rather than inherited (WP-064). A per-tile score never pays the duplicate-detection cost whole-image evaluation charges, so **these figures are not comparable to published DOTA results**, which are whole-image. That is a property of the measurement, not a caveat about its precision.

The same checkpoint scores 0.2914 / 0.5242 on Apple MPS and 0.2914 / 0.5243 on CUDA — agreement to four decimals across two accelerators and two operators, which is stronger evidence about the evaluator than about the model.

## Limitations

**The A44 composition is the model's defining constraint.** An oriented box is produced by rotating an axis-aligned rectangle about its own centre, so the centre and extents are predicted in the unrotated frame and the heading is a separate scalar. Objects whose rotated extent is poorly described by that construction are systematically harder for it than a formulation predicting the rotated box directly.

**Angle failure is invisible in axis-aligned terms.** This is why the run logs no `val/mAP` at all (WP-102): that figure reads the composition's pre-rotation rectangle, so a run with correct orientations and one with random orientations produce the identical curve.

**Per-tile, as above.** Objects spanning a tile boundary are seen as two clipped parts; parts below 70% of their original area are flagged difficult by the tiler and scored under R18's discard rule rather than counted.

**Long-tailed classes.** The reported mean is over classes with at least one non-difficult ground truth; rare categories rest on few instances, and the aggregate hides that spread.

**Smoke tier.** 50 epochs at the smallest scale against the paper's 500/600-epoch schedules, one seed, one benchmark.

## Licensing

Code and report: Apache-2.0.

**Weights are not released** (D14: releases ship no trained weights), and for this task family that policy does more work than for the others. A model trained on DOTA inherits DOTA's academic-use-only restriction, which is incompatible with the Apache-2.0 terms the code carries; publishing weights under this repository's licence would be misstating what a user is permitted to do with them.
