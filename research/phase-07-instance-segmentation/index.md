# 🖌️ Phase 7 — Instance segmentation, v0.2.0


## WP-052 — Segmenter wiring

<a id="wp-052"></a>

Absence of the training-only branches from `deploy()` is proved twice rather than asserted: by parameter identity and a module walk, and independently by arithmetic — `deployed == full - aux - o2m`, exact at **2,717,284 − 5,200 − 152,690 = 2,559,394**. Two proofs because the failure mode is a branch that is disabled rather than absent, which a module walk can miss and arithmetic cannot.



## WP-052b — Table S9 fidelity gate

<a id="wp-052b"></a>

R1 Table S9 was transcribed from the arXiv **PDF** (page 28) after the HTML rendering proved to truncate before section S5 — worth knowing before trusting any other appendix table read from HTML: n 2.7/9.1, s 10.4/34.2, m 23.6/121.5, l 28.0/139.8, x 62.8/313.5 (M params / B FLOPs).

The params tolerance is **±3%**, wider than detection's ±2%, and the widening is argued rather than fitted: the segmentation head's sizing rests on five registered assumptions (A14, A15, A18, A34, A35) instead of published structure. `s` is the binding scale at +2.5% and every other lands inside ±1.5%.

FLOPs hold ±5% on all five scales under the same 2×MACs convention Table 7 needed (A29). That is corroboration of A29 rather than a second assumption — the convention was fixed on detection and predicted segmentation correctly.

Two gaps in the source worth recording: S9 states neither its FLOP convention nor whether Params denotes the full or the deployed model (E2E and non-E2E share one pair per scale), and unlike Table 7 it attributes no source.



## WP-087b — batched mask-loss gather

<a id="wp-087b"></a>

Wiring the segmentation path made it trainable and slow: **1.59 s a step against detection's 0.39** on an RTX PRO 6000, GPU idle, main process on CPU.

Cause was not volume — ~2,340 masks, 0.24 GB. `_branch_mask_loss` selected each image's positives with a boolean mask index, whose output shape is data-dependent, so the device had to report the surviving count back to the host before the next op could be built. Inside a loop over the batch, run per branch, that is **2×B forced syncs a step — 64 at batch 32**.

Padding to the batch's largest positive count and selecting once costs two syncs a branch at any batch size. Positives keep ascending anchor order within an image and image order across the batch, so summation order — and therefore the value — is the loop's own.

Padding is where a perf change turns numerical, and it did: padded rows carry `gt_index = -1`, the unassigned sentinel, which is a legal negative index in Python and out of bounds in a gather. The existing gradient test caught it before any of the perf work ran.



## WP-087c — vectorised polygon rasterisation

<a id="wp-087c"></a>

The structural difference between the detection and segmentation training steps was one CPU loop: rasterising every instance polygon onto the prototype grid, in the main process between forward and loss, walking each ring's vertices in Python with a full `height × width` tensor op per vertex.

Cost is linear in **point count**, which is why earlier four-vertex benchmarks missed it entirely. At a 160-px grid: 0.11 ms at 4 vertices, 1.65 at 64, 3.32 at 128 — i.e. 0.37 s a step at batch 32 with seven instances an image, and mosaic quadruples the instance count.

That single fact explains every symptom the earlier hypotheses fit and then failed to predict: GPU idle waiting on CPU; `num_workers` irrelevant because this runs outside the workers; `OMP_NUM_THREADS` irrelevant because the cost is per-op overhead rather than arithmetic.

Three steps, each superseding the last:

| step | measurement |
| -- | -- |
| all of a ring's edges at once | 1.65 → 0.434 ms at 64 vertices; 3.32 → 0.700 at 128 |
| a whole image's rings in one crossing test | 1,138 real-COCO instances across 32 mosaic images at 23 points a ring: 10.14 → 4.87 ms an image |
| edges flattened with an instance index instead of padding every ring to the image's longest | seven 23-point rings 7.22 → 2.06 ms; the same seven with one 500-point ring 20.24 → 3.44; 35 rings with one of 300 55.95 → 6.03 |

The third step is where the ragged-versus-padded question was settled empirically: COCO rings are ragged enough that padding to the longest dominates. Output is identical at every step, not merely close.



## WP-087d — rasterise mask targets in the loader workers

<a id="wp-087d"></a>

Vectorised, the rasterisation still sat between forward and loss — serial with the GPU and competing with the optimiser's dispatch loop for cores. Measured on COCO at batch 32 with mosaic on: **1,005 ms rasterising and 1,475 ms in an optimiser that costs 94 ms in isolation**, the accelerator idle for most of a 2,250 ms step. The optimiser figure is the informative one: it is not slow, it is starved.

Moving the work into `collate_detection` puts it inside a DataLoader worker, so batches are rasterised concurrently and ahead of the step consuming them. Two design points that made it possible without handing the collate a model: the grid is the input canvas at a fixed stride of four, derivable because `ProtoNet` upsamples the stride-8 fused feature by two (A15); and masks travel as a bool `(sum_N, Hp, Wp)` stack split by `boxes_per_image`, exact because the values are `{0,1}` and a quarter the bytes.

That then put the collate on the critical path: **4,454 ms on one thread** for 64 mosaic images carrying 2,170 instances — 2.09 s of a 3.04 s step in `train_dataloader_next`. Fixed by testing each ring only inside its own bounding window, which **replaces** 087c's flattened-edge batching rather than composing with it: sharing one grid across an image's instances is what made that bookkeeping worth carrying, and a window removes the shared grid.

| instance span | before | after |
| -- | -- | -- |
| 30 px | 32.41 ms | 1.73 ms |
| 60 px | 35.00 ms | 3.27 ms |
| 120 px | 35.74 ms | 8.03 ms |

Cost now tracks the instance instead of being flat in the grid. Note the "before" column barely moves across a 4× span — that flatness *is* the defect, visible in the numbers before any explanation.

The correctness argument is stronger than "the window is large enough": outside it the crossing test is provably False rather than merely unlikely, and the window slices the coordinate ranges without translating the ring, so surviving pixels are bit-identical rather than equivalent.



## WP-087e — epoch mask mAP during validation

<a id="wp-087e"></a>

A `task="segment"` run logged `val/mAP` and nothing about its masks, so the branch the run exists for moved no epoch metric at all: a run whose masks were degenerate and one whose masks were right produced the same curve, and the first sign of either was an evaluation pass hours after the run ended.

Two measurement decisions are load-bearing and were argued from cost rather than taste:

**Scoring on the prototype grid**, where the loader's ground-truth masks already are. Upsampling both sides by four would cost sixteen times the memory to compare the same two fields at a finer sampling of one boundary. The metric is a proxy in exactly the sense `val/mAP` already is (WP-077); the acceptance figure stays the standalone evaluation.

**A second `MeanAveragePrecision` rather than `iou_type=("bbox", "segm")`** on the existing one. One metric holding both would take each instance's area from whichever frame torchmetrics picked, mis-bucketing the small/medium/large splits by a factor of sixteen. This is a case where the convenient API is wrong for a reason that would never surface as an error — only as quietly wrong area breakdowns.



## WP-087f — tier names

<a id="wp-087f"></a>

The configs were named `<task>_tier_<tier letter>_<scale letter>`, putting two letters from unrelated alphabets side by side, with the tier letter reading backwards (A the cheap smoke tier, B the longer paired-run one).

The scale letter was not merely confusing but **false**: it sets `variant` inside the file and `--variant` overrides it, so `seg_tier_a_n.yaml --variant m` is an unremarkable invocation whose filename says `n` while the run is `m`. A filename that can contradict the run it names is a defect, not a style question.

Renames propagated to historical mentions too — a reader meeting "Det-A attempt 2" in the A13 row has no file left to connect it to — with git history keeping the old names. Each config header keeps its `blueprint sec. 10 tier A` citation, because the blueprint is not in this repository and still speaks in letters, so the mapping has to survive somewhere explicit.

______________________________________________________________________



## WP-087 — factories, not a `Detector` object

<a id="wp-087"></a>

Composing the Phase 7 modules onto the shared backbone and neck went behind `build.py` factories rather than behind a `Detector` object holding them. The reason is the state dict: an object that owns the backbone and neck renames every parameter under it, and flat keys are what keep the accepted Det-smoke checkpoint loading **strict** after a second task lands beside it. That checkpoint is the artifact later phases load, so the DoD asserts the strict load directly rather than trusting the module tree to have stayed the same shape.



## WP-054 — a DoD clause with no referent

<a id="wp-054"></a>

The 0.2.0 release row owed "the 0.1 goldens still green", and that clause could not be met as written rather than merely being missed: no `v0.1.0` tag was ever cut, WP-046 was superseded, and `goldens/frozen/0.1` never existed to be green or red. The release froze `goldens/frozen/0.2` instead.

Worth keeping because the shape recurs in a roadmap transcribed ahead of the work: a definition of done can name an artifact that the intervening packages decided not to produce, and the honest outcome is to record that the clause lost its referent — not to satisfy a nearby clause and tick the row.

