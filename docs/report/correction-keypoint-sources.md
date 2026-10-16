# 📐 Correcting note — the 0.5.0 section's source claims

Appended, not merged (D10): the 0.5.0 section above is left byte-identical, and this note states which of its sentences are false and what is true instead. No figure moves; no run stands behind this note. Reported as issue 2 on the tracker, confirmed against the arXiv PDF on 2026-09-17, and recorded as D22 ([WP-178](../work_packages/wp-178.md)).

## What the section says, and what R1 says

The 0.5.0 section was written as if R1 had no pose section. It says the acceptance criterion is "not R1 Table 9's mAP=63.0" and, in the deviations list, calls that figure "a different paper's different architecture"; it says twice that "no published or gated parameter/FLOP reference exists for this task family" because "R14 states no architecture at all"; and its mechanism table sources the point stem and the RLE loss to R14 alone. Every one of those clauses rests on the premise that R1 addresses no keypoint task.

R1 sec. 3.4.2 "Pose Estimation" states the objective: "a parallel sigma branch predicts per-axis uncertainty sigma = (sigma_x, sigma_y) in (0, 1)^2 for each joint", the residual is normalized as Eq. 10, "a shared RealNVP normalizing flow estimates log phi(eps)", and Eq. 11 gives the RLE term; "training uses an Object Keypoint Similarity (OKS)-based loss", and "YOLO26 extends this scheme with Residual Log-Likelihood Estimation". R1 Table 9 ablates the weight split on YOLO26s — `(24, 1)` 63.0, RLE alone `(0, 1)` 61.9 and `(0, 2)` 62.4, OKS alone `(48, 0)` 61.5 — and R1 Table S10 publishes params and FLOPs for every pose scale at 640 px, `n` at 2.9 M / 7.5 G and 57.2 AP. All of it was inside R1's allowlisted role from the day the source was registered.

## What is true instead

| the section's claim | the correction |
| -- | -- |
| Table 9's 63.0 is "a different paper's different architecture" | R1 is *this* paper and Table S10's `n` row, 57.2, is the same architecture family at full budget under the full objective. The gap to this run's 0.27 is budget (50 epochs at `n`), objective (RLE alone) and stem width (A77); the run was never entitled to close it, and its acceptance — the mechanism's direction of effect — is unchanged |
| "no published or gated parameter/FLOP reference exists for this task family" (stated twice) | R1 Table S10 is that reference. Measured at its protocol (640 px, 1 class, 17 points) the point stem is 13.0% under the paper's `n` params and 26.5% under its FLOPs, narrowing to -3.0% / +1.0% at `x` — A77 holds every scale. No gate ships until a width is chosen against the table (roadmap WP-180) |
| point stem sourced to "R14, [WP-122](../work_packages/wp-122.md)" | R1 sec. 3.4.2 names the branch ("a parallel sigma branch") and gives no width; R14 remains the source of what the branch emits |
| RLE loss sourced to "R14 Eq. 8, Eq. 12, sec. 3.2–3.3" | Also R1 sec. 3.4.2, Eq. 10–11 — the composition this term belongs to. R1's printed Eq. 11 carries `log sigma` twice, R14 Eq. 8 once; the code carries R14's form and A76 records the discrepancy |
| "The keypoint gain is this project's, not the paper's ... R14 states the loss and never a multi-task weight" | R1 Table 9 does state a weight, `wRLE = 1` beside `wOKS = 24`. `keypoint_gain = 1.0` coincides with it in value only — Table 9's weights sit beside R1 Table S5's box/cls gains (9.83 / 0.65 at `s`), this project's beside 7.5 / 0.5 — and the OKS term it is weighed against was absent from this run (A68, A75) |
| "RLE (R14) composed onto the same dual head" is the whole of what was reproduced | That is the RLE half of R1 sec. 3.4.2's objective. The tier is a **partial** reproduction: R1's RLE term, transcribed from R14 where R1 draws it, without the OKS term R1 composes it with. `losses/oks_loss.py` lands that term at `oks_gain = 0.0` after this run (A75), so the objective every figure above was measured under is unchanged |

## What this note does not do

It does not re-score the run, move a golden, or change the acceptance the section records: [WP-125](../work_packages/wp-125.md)'s criterion was RLE's mechanism claim against a non-flow control, that comparison is R14 Table 7's ablation and is valid as run. The runs that would make the tier a full reproduction — the Table 9 split at smoke scale, the stem width against Table S10, Eq. 11's second `log sigma` — are roadmap Phase 17, each waiting on an accelerator or a principal.
