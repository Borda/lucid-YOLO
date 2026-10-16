# 🕺 0.5.0 — Keypoint detection

The Pose-smoke tier has run and was **accepted 2026-08-25** at roadmap 125. This section records what the run measured, and states plainly what it was and was not accepted on: the acceptance criterion is R14's mechanism claim — that RLE's learned residual density earns real accuracy over a non-flow control — not a pose figure, and not R1 Table 9's mAP=63.0, which no run in this project has ever targeted.

## What was reproduced

RLE (R14) composed onto the same dual head, NMS-free deploy path and training loop the other three tasks share, plus COCO's own OKS evaluation protocol (R12).

| mechanism | reference | implementation |
| -- | -- | -- |
| keypoint targets, left/right flip pairing | R12 (17-point schema), A64 | `data/targets.py` — `Targets.keypoints`/`keypoint_vis`; `HorizontalFlip(keypoint_flip_pairs=...)` swaps identity rather than mirroring coordinates alone, dataset-supplied rather than hardcoded |
| point head stem, decode | R14, [WP-122](https://borda.github.io/lucid-YOLO/work_packages/wp-122/) | `models/heads/detect.py` — a third stem per branch, raw `(x, y)` offset plus raw per-axis sigma; `decode_keypoints` composes with the anchor-centre/stride convention `decode_ltrb` already uses |
| RLE loss | R14 Eq. 8, Eq. 12, sec. 3.2–3.3, A65, A66, A71, A72 | `losses/rle_loss.py` — a hand-written 6-layer RealNVP flow over the per-axis-normalized residual, summed with a fixed unit Laplace NLL; the only trainable-weight loss in the project |
| Laplace-NLL control | R14 Table 7, [WP-135](https://borda.github.io/lucid-YOLO/work_packages/wp-135/) | `losses/keypoint_nll_loss.py` — the same residual likelihood with the flow term removed, everything else held; a `keypoint_loss` switch, default `rle` |
| OKS evaluation | R12, A67 | `eval/coco_eval.py::evaluate_keypoints` — `faster_coco_eval.COCOeval_faster(iouType="keypoints", ...)`, R12's 17-value sigma table when `num_keypoints == 17`, A67's uniform sigma otherwise; ground truth's own `area`/`iscrowd`/`num_keypoints`/`bbox` used directly rather than reconstructed |
| epoch-level pose signal | [WP-137](https://borda.github.io/lucid-YOLO/work_packages/wp-137/) | `val/oks_mAP`, accumulated per validation step and scored once per epoch, beside `val/mAP` rather than instead of it |
| standalone report | [WP-134](https://borda.github.io/lucid-YOLO/work_packages/wp-134/) | `lucid-eval` dispatches on the checkpoint's own `task`, scoring both decode paths from one forward |

No published or gated parameter/FLOP reference exists for this task family: R14 states a loss and an evaluation protocol, never an architecture, so unlike detection ([R1 Table 7]) and segmentation ([R1 Table S9]) there is nothing to hold the point stem's cost against beyond this project's own measurement — see `model_cards/keypoints.md`.

The fast wiring gate runs before any COCO launch: `scripts/overfit_micro.py --task keypoints` reaches train OKS AP **0.3357** over 548 instances against a 0.30 floor, on R21's synthetic `SymbolShape` schema (`goldens/gpu/overfit_micro_kp.json`) — not COCO's 17-point schema, since the gate's fixture family is geometric shapes throughout, and A67 records the sigma substitution that makes an OKS figure meaningful on it at all.

## The Pose-smoke run

**Run v11** — `0.5.0.dev1`, COCO `person_keypoints_{train,val}2017`, `task: keypoints`, `num_keypoints: 17`, `keypoint_loss: rle` (default), n scale, batch 128, lr 0.02, MuSGD, warmup 3 epochs then linear decay to `lr0 * 0.01`, close-mosaic for the final 10, EMA (decay 0.9999, tau 2000), gradient clip 10.0, `keypoint_gain = 1.0` (A68), bf16-mixed, seed 0, `deterministic: true`, 50 epochs = 46,250 steps on one unrecorded Colab GPU, 7h06m wall clock — about 8m30s an epoch, derived from artifact timestamps rather than logged, matching the other COCO tiers' own unrecorded-hardware pattern.

The Laplace-NLL control (`version_14`, read together with this run in Acceptance below) trained the next day under an identical config but for `keypoint_loss: laplace_nll`, at `0.5.0.dev2`; two attempts to launch it (`version_12`, `version_13`) wrote only a TensorBoard writer's opening record and nothing else — no config, no step — before exiting, 11 seconds apart. Their cause is not in what synced to Drive; the third attempt (`version_14`) ran to completion without incident.

Evaluated by `lucid-eval` on COCO val2017, all 5000 images, `person_keypoints_val2017` ground truth:


| path | box mAP50-95 | box mAP50 | box mAP75 | OKS AP | OKS AP50 | OKS AP75 |
| -- | -- | -- | -- | -- | -- | -- |
| E2E | **0.5030** | 0.7683 | 0.5323 | **0.2738** | 0.6094 | 0.2138 |
| NMS | 0.5119 | 0.7764 | 0.5417 | 0.2688 | 0.6066 | 0.2042 |


`e2e` is the row comparable to the training run's own `val/mAP` and `val/oks_mAP` — the same convention the earlier three tiers state for their own tables. The box branch's ranking (`nms` above `e2e`) and the point branch's (`e2e` above `nms`, by 0.005 AP) run opposite directions on this checkpoint; neither has been checked against a second seed.

## Reading the training curves

![Pose-smoke training curves](https://borda.github.io/lucid-YOLO/figures/pose_smoke_training.svg)

*Run v11, four panels. The validation box mAP50-95 (the same E2E proxy the other three tiers log); epoch-mean loss totals; the one-to-one branch's pre-gain box/classification/L1 components; and the RLE keypoint term, train and validation.*

**The metric is still rising at epoch 49.** `val/mAP` (box) has not plateaued the way Det-smoke's or Seg-smoke's had by this point in the schedule — its last epoch-to-epoch step is still positive. Fifty epochs may not be this run's ceiling the way it plausibly is for the oriented tier; no further epochs have been run to check.

**Validation loss descends almost without exception.** Two of forty-nine epoch-to-epoch steps rise, the larger by 0.30% — an order of magnitude cleaner than the oriented tier's own curve, and closer to Det-smoke's and Seg-smoke's monotone descent. The minimum sits at the final epoch, 49, unlike the oriented run's epoch 48.

**The keypoint term crosses zero, and that is expected, not a fault.** `val/keypoint` opens at 1.37 and closes at -0.79 — a negative log-likelihood, unlike every other logged term, is a normalized density's log and has no floor at zero the way a squared error or an IoU term does. A more negative value is a tighter fit, and this run's own trajectory is monotone toward it with the same two-rise exception the loss panel shows.

## Acceptance

Not accepted on a pose figure. `pose_nano_smoke.yaml` names training completion and stability, and — the criterion actually decisive for this tier — RLE's mechanism claim against [WP-135](https://borda.github.io/lucid-YOLO/work_packages/wp-135/)'s Laplace-NLL control, read in full in `../work_packages/wp-125.md#wp-125`.

| criterion | required | observed |  |
| -- | -- | -- | -- |
| wiring gate | train OKS AP ≥ 0.30 | **0.3357** | ✓ |
| training completed | 50/50 epochs, all logged | 50 validation rows, no gaps | ✓ |
| no divergence | — | `val/loss` 18.47 → 6.65, minimum at epoch 49; 2 of 49 steps rise, the largest 0.30% | ✓ |
| RLE beats the Laplace-NLL control | direction of effect, not a magnitude | e2e OKS AP **0.2738** vs **0.2527** (`version_14`) — same sign as R14 Table 7's 70.5-vs-67.4 AP | ✓ |
| `keypoint_flip_pairs` exercised live, tier scale | not a mechanism failure | trained without incident, both arms (A64) | ✓ |
| `keypoint_gain = 1.0` exercised, tier scale | not a mechanism failure | trained without incident, both arms; tuned adequacy unmeasured (A68) | ✓ |
| Phase ≤11 gates | green | full suite and 21/21 goldens at commit time | ✓ |
| pose figure itself | *not the criterion* — recorded, not cleared | 0.2738 e2e OKS AP, 0.5030 e2e box mAP | — |

## Assumption outcomes

Assumptions the keypoint phase exercised. Full register in `ASSUMPTIONS.md`.

| id | subject | outcome |
| -- | -- | -- |
| A64 | dataset-supplied flip pairing | **held, first exercised live at tier scale here** — resolved by [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/)'s own reading of the training path; this run is the first to train with it through real augmentation rather than through the overfit gate's augmentation-off path |
| A65 | sigma via sigmoid | **held** — resolved by [WP-123](https://borda.github.io/lucid-YOLO/work_packages/wp-123/) directly from R14 sec. 3.3; unexercised further by this run beyond what [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/)'s gate already confirmed |
| A66 | `v=0` excluded, `v≥1` included equally | **held, exercised at COCO scale** — this run is the first to train the policy against real annotation visibility rather than synthetic fixtures |
| A67 | uniform sigma for non-COCO schemas | **not exercised** — this run's `num_keypoints == 17` takes R12's own table, the branch A67 governs |
| A68 | `keypoint_gain = 1.0`, un-tuned | **the mechanism-failure exposure did not happen** — a real pose head trained in both arms rather than one swamped by or swamping the detection objective; tuned adequacy stays unmeasured, no sweep ran |
| A69 | one shared RLE flow across both branches | **untested against the alternative** — this run trains one flow, as every run does; nothing here compares it to two |
| A70 | off-canvas points carried through unchanged | **exercised, unmeasured** — real augmentation ran, so some points left the canvas on kept instances; no measurement of how often or with what effect |
| A71 | per-axis residual frame | **held** — resolved by [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/)'s numerical check; this run trained without the non-finite failure the un-normalized frame produced |
| A72 | RealNVP's tanh-bounded log-scale | **held** — resolved by [WP-123](https://borda.github.io/lucid-YOLO/work_packages/wp-123/)/WP-132; no runaway observed across 46,250 steps |
| A73 | keypoints stays hand-driven, not forked onto torchmetrics | **unaffected by this run** — a [WP-138](https://borda.github.io/lucid-YOLO/work_packages/wp-138/) decision about the standalone evaluator, not the training path this run exercises |

## Deviations from the paper, and from the earlier tiers

1. **Epoch budget.** 50 epochs at n scale, one seed — a smoke tier by design, and R14's own schedule is a from-scratch top-down crop regressor, not a directly comparable budget to begin with.
2. **No Objects365 pretraining, no evolutionary hyperparameter search** (D2).
3. **No comparison to any published pose figure.** R14 reports on a top-down, cropped-input architecture; this project's dense one-stage detector with a bottom-up point stem is not the same architecture family, and no attempt is made here to set this run's OKS beside R14's own numbers or R1 Table 9's mAP=63.0 (a different paper's different architecture). The acceptance criterion is the mechanism's direction of effect, read from the paired run in `../work_packages/wp-125.md#wp-125`, not an absolute figure.
4. **The keypoint gain is this project's, not the paper's.** R14 states the loss and never a multi-task weight — it trains a pose-only network. `keypoint_gain = 1.0` (A68) is this project's own un-tuned starting point, the same shape of choice A22 made for the oriented tier's angle gain.
5. **The parameter/FLOP gate has no published reference for this task family**, unlike detection ([R1 Table 7]) and segmentation ([R1 Table S9]) — R14 states no architecture at all.
6. **Single flow, not measured against two per branch** (A69) — untested here, as in every run so far.

## Reproducing this result

```bash
lucid-data download --data_root /content/coco2017 --splits '[train,val]' --verify true
lucid-data check    --data_root /content/coco2017

lucid-yolo fit --config pose_nano_smoke.yaml \
  --data.data_root /content/coco2017 \
  --data.batch_size 128 --data.num_workers 32 --data.prefetch_factor 1 \
  --model.lr 0.02 --trainer.max_epochs 50 --trainer.precision bf16-mixed \
  --trainer.default_root_dir /content/drive/MyDrive/lucid_runs

lucid-eval --checkpoint <checkpoint> --data_root /content/coco2017 --output pose_report.json
```

The Laplace-NLL control arm is the same command against `pose_nano_smoke_laplace_nll.yaml`, byte-identical but for `keypoint_loss`.

Seed 0 throughout. Cross-platform bitwise reproduction is not claimed (A26). The metric reports are archived beside each run's own checkpoint, and the run config, hyperparameters and epoch metrics under `lightning_logs/version_11/` (RLE) and `lightning_logs/version_14/` (Laplace-NLL control).
