# 🔬 Research Log

What executing the work packages taught about the reproduction itself: the paper-fidelity measurements, the modeling and data-protocol choices tried and abandoned, and the results that do not flatter the project. Repo-building findings -- CI, packaging, licensing, formatting, docs tooling -- live in `docs/ENGINEERING_LOG.md` instead; the two are split by claim, not by work package, so a single WP whose finding straddles both gets one entry in each, cross-linked.

**This file is not a register.** Where a finding fixed an implementation choice the papers left open, the choice lives in ASSUMPTIONS.md and the number lives there with it; those are cited by id here, never restated. Release-level results live in REPRODUCTION_REPORT.md for the same reason. What lands here is what has no other home: fidelity measurements, negative results, and the reasoning behind a rejection.

Sections are anchored by work package (`#wp-087d`) and the roadmap links to them. A package with nothing worth recording here has no section -- and a package whose whole finding is about tooling rather than fidelity has none here at all; see `ENGINEERING_LOG.md` for it.

**Phases 1-5 are backfilled** (2026-08-14) and read differently from the rest. Those roadmap rows were transcribed from the blueprint *before* the work and never rewritten afterwards, so nothing was carried out of them -- their entries here are reconstructed from the commit record, which is the durable artifact, and not from anyone's memory of the work. Phase 6 onward is the reverse: those rows had accumulated the retrospective detail in place, and this file is where it moved to. Phase 0's own findings were all tooling-kind and moved to `ENGINEERING_LOG.md` whole.

Dates are the day the finding was recorded.

______________________________________________________________________

## 🚰 Phase 1 — Data pipeline

### WP-014 — the DataLoader transport saga

<a id="wp-014"></a>

Five commits over one work package, each fixing the failure the previous one exposed. Worth reading as one story, because the later packages ([WP-071](#wp-071), [WP-073](#wp-073), [WP-076](#wp-076), [WP-103](#wp-103)) are all continuations of it.

1. **Starvation.** CUDA runs showed GPU utilization spikes: a mosaic sample alone decodes four images, and the loaders shipped with no `pin_memory`, prefetch depth 2, and every worker inheriting torch's full intra-op thread pool — `num_workers` processes each oversubscribing the CPU by the core count. Fixed with pin-memory-when-CUDA, prefetch 4, and a `worker_init_fn` capping each worker to one torch thread.
2. **Prefetch 4 was wrong.** It ENOMEMed containerized CUDA runs: the queue holds `num_workers × prefetch_factor` batches in POSIX shared memory, and at batch 64 (~315 MB of stacked 640 px images) × 12 workers × 4 that is **~15 GB of /dev/shm segments in flight**. The pin-memory thread dies first, on `mmap` with "Cannot allocate memory". Default returned to the DataLoader's own 2.
3. **Auto worker count was also wrong.** `min(batch_size, cores)` on a ~64-vCPU Colab at batch 64 launched ~64 workers and refilled shm — prefetch 2 alone had not closed the failure mode. `_shm_capped_workers` bounds the *auto* count so `workers × prefetch × batch bytes` fits half the currently free `/dev/shm` (via `statvfs`), floor one worker. Hosts without `/dev/shm` (macOS, Windows) and explicit counts are untouched. **This is the guard [WP-102](#wp-102) later removed for named counts and [WP-103](#wp-103) restored for inherited ones.**
4. **Segment count, not byte count.** Torch's `file_descriptor` sharing ships every tensor crossing the worker boundary as its own shm segment, each costing the consumer one `mmap`. The natural `(images, list[Targets])` batch is **~600 tiny segments at batch 64**, so in-flight batches blow the per-process `vm.max_map_count` (65530, read-only on Colab): observed "unable to mmap 96/144/336 bytes". Fixed by flattening the ragged batch into eight dense tensors and restoring it on the destination device.

The through-line: three distinct resources exhaust in this pipeline — shm **bytes**, shm **segments**, and CPU **threads** — and each has its own failure signature at a different layer. A fix aimed at one does not touch the others, which is why the same symptom ("the loader died at step 0") kept returning with a different cause.

### WP-013 — a defect deferred on purpose

<a id="wp-013"></a>

`HorizontalFlip` mirrors a rotated box by negating `theta`, and the commit documents re-canonicalization as a Phase 8 concern rather than doing it. That deferral was correct at the time and became a live defect the moment [WP-058](#wp-058) removed the guards — the note is what made it findable.

______________________________________________________________________

## 🏗️ Phase 2 — Architecture

### WP-023 — the param/FLOP fidelity gate

<a id="wp-023"></a>

The gate passed all five scales within ±2% params and ±5% FLOPs of R1 Table 7 (worst case `l` FLOPs +4.2%) after **exactly two assumption iterations**, both registered rather than tuned quietly: A28 (the original full-conv box stem left the head **7–16× the reference budget**) and A3 (nested `C3k` inner-bottleneck count 1 instead of 2, because at depth 1.0 the `l`/`x` variants overshot with two).

The third change was explicitly **not** an iteration: A29's counting convention (GFLOPs = 2× fvcore MACs, on the deployed NMS-free model; params over the full dual-branch checkpoint) was *determined empirically* by trying the candidates. Trying three conventions to discover which one a table used is measurement; changing the model until it matches a number is fitting. The distinction is recorded because the two are indistinguishable from the outside once the gate is green.

______________________________________________________________________

## ⚖️ Phase 3 — Assignment and losses

### WP-022 — the dead predictor

<a id="wp-022"></a>

The mechanism and its fix are A30. What belongs here is the failure signature, because it is the most misleading one this project has produced.

With zero-initialized class-conv biases every sigmoid opens at 0.5, the TAL-normalized summed BCE starts near **1.8M nats**, the first MuSGD step at lr 0.01 blows the weights out (**box predictions reached ±6000 px**), TAL then assigns no positives, and the network saturates to an **exact-zero loss with zero gradient**. The progress bar shows `train/loss 0.000` — which reads as a perfectly converged model rather than a dead one. Reproduced identically on CPU and MPS from real COCO batches.

Post-fix verification was a 30-step real-COCO check showing monotone descent **14k → 1.5k** with box predictions bounded within image scale — i.e. the fix was confirmed on the same signal that had lied, not on "the tests pass".

### WP-030 — the Phase 3 exit gate

<a id="wp-030"></a>

Proves the assignment-plus-loss stack is optimizable end to end **without the head**, which had not landed yet: learnable prediction tensors (logits, and anchor-centred softplus ltrb boxes so assigned ground truths are exactly representable) overfit a real letterboxed fixture batch for 200 Adam steps. Loss falls **182.3 → 6.9** with strictly decreasing window checkpoints, every value and gradient finite, and both alpha extremes (0 and 1) train.

Constructing the learnable tensors so the target is exactly representable is what makes a failure here attributable to the objective rather than to the parameterization.

______________________________________________________________________

## 📉 Phase 4 — MuSGD

### WP-033 — toy convergence golden

<a id="wp-033"></a>

MuSGD reaches the 0.1 MSE threshold in **47 steps against momentum-SGD's 90** on a fully seeded micro-CNN regression task.

The tolerance is the interesting part: step counts are pinned at **±15**, final losses at 0.005. Cross-platform torch float drift shifts trajectories slightly ([Float reproducibility](ENGINEERING_LOG.md#cross-float)), so a tight pin would be a golden about the machine. The 43-step separation is what keeps the *directional* claim decisive inside a tolerance that loose — the golden asserts the ordering it cares about rather than the number it happened to get.

______________________________________________________________________

## ⚡ Phase 5 — Lightning training loop

### WP-039 — a package whose finding is that no code was needed

<a id="wp-039"></a>

Deterministic checkpoint and resume was proved rather than implemented: a four-epoch reference run against a run stopped after two epochs and resumed produces the same per-step loss trajectory over the post-resume epochs (rel 1e-4), with `global_step`, `current_epoch` and every MuSGD momentum buffer restored. **No source changes were needed** — Lightning's checkpointing plus existing module and optimizer state already round-trips.

Two scoping choices made that provable: all runs share `max_epochs`, so the progressive-loss alpha ramp (a pure function of `current_epoch / max_epochs`, recomputed each epoch) is identical either side of the resume; and the data stream is a fixed unshuffled in-memory dataset, keeping sampler-state restoration out of scope. Without the first, a correct resume would still have produced divergent losses.

### WP-040 — overfit-100 on an accelerator

<a id="wp-040"></a>

Train-set recall **0.9747** against the 0.95 floor, MPS, 4:47 wall clock, decoded NMS-free through the deployed path.

The structural decision that outlived it: `goldens/gpu/` is excluded from default harness discovery and recomputed only under `--include-gpu`, which is what keeps the offline gate accelerator-free while still holding accelerator numbers under version control. The cost of that split is that a `gpu/` golden can go stale unnoticed — `shapes_regression_det.json` currently has an outstanding re-freeze.

*Superseded by [WP-167](ENGINEERING_LOG.md#wp-167).* The outstanding re-freeze is closed: all five files under `goldens/gpu/`, `shapes_regression_det.json` among them, were regenerated on an L4. The sentence above is left as written because the staleness it predicted is exactly what happened — `golden-gpu` returned 46/48 on the pre-WP-169 values — and a log that edits its own correct predictions into the past tense stops being evidence that anything was predicted.

______________________________________________________________________

## 🎯 Phase 6 — Evaluation, release 0.1.0

### WP-071 — packed uint8 batch transport

<a id="wp-071"></a>

The transport format and its error bound are A33. What the register does not carry is the shape of the win: flattening the ragged `list[Targets]` into eight dense tensors takes the DataLoader IPC hop from roughly **600 pickled segments per batch to 9**, with bytes down 4x from the uint8 quantization. Segment count, not byte count, is what makes a containerized shm ceiling bite.

### WP-073 — compact annotation store

<a id="wp-073"></a>

`CocoDetectionDataset` held the raw JSON annotation dicts after building its per-image `Targets`. Millions of small Python objects, each with a refcount touched on every access — and refcount traffic materializes copy-on-write pages in every persistent DataLoader worker, so a fork-shared dataset is only shared until Python reads it.

Observed on a Colab 176 GB host, killed at validation boundaries: **epoch 1 at 48 workers, epoch 24 at 32 workers**. The worker count sets how fast the leak accumulates, not whether it does.

Dropping the dicts fixed the growth; the val-loader worker cap (`min(num_workers, 4)`) addressed the second half, a persistent val pool doubling the worker population at exactly the moment the training pool is still resident. That cap is what [WP-102](#wp-102) later narrowed and [WP-103](#wp-103) restored for inherited counts.

### WP-076 — epoch-recycled loader workers

<a id="wp-076"></a>

WP-073's compact store did not end the OOMs: Colab still killed the run at **epoch 24 and epoch 38**. The residue is not one leak but a class of them — allocator high-water growth, arena fragmentation, and whatever copy-on-write pages survive — all of which are per-worker and none of which a dataset change reaches.

Respawning the pool costs seconds against a 9-minute epoch and hard-resets every one of them, so `persistent_workers` defaults **off**. This is a case where the cheap blunt fix dominates: no diagnosis of *which* allocator behaviour dominated was needed, because recycling ends all of them and the measurement said the cost was noise.

### WP-079 — per-worker, per-epoch augmentation RNG: what the collision cost

<a id="wp-079"></a>

Measured exposure: **1/3200 of the intended draws** at 32 workers over 100 epochs. Two DataLoader defects compose multiplicatively and neither is visible in a loss curve — the run trains, it simply trains on far less augmentation variety than the config describes. The mechanism is a `_TrainPipeline` bug fixed the same day; the full story is at [ENGINEERING_LOG.md#wp-079](ENGINEERING_LOG.md#wp-079).

### WP-082 — objective invariant gate

<a id="wp-082"></a>

Written after WP-078 and WP-079 survived 474 tests and three tier runs, which is the fact that justifies the package: the suite was large and did not contain the shape of test that would have caught either.

The useful negative result is the test **not** written. An obvious guard for WP-078's defect is "the L1 term must not exceed some share of the objective". On a converged fixture the broken pixel frame reads **10.7%** against the correct frame's **1.5%** — both comfortably under any threshold anyone would have picked. Such a test would look like protection and catch nothing. The gate pins the coordinate frame directly instead (`1/stride` against the pixel frame, per level) and its scale on a near-converged state (0.75 stride-units against 5.99 broken).

### WP-083 — synthetic-shapes generalization golden

<a id="wp-083"></a>

Frozen at NMS mAP50-95 **0.8823**, E2E **0.8583**, mAP50 **0.9842** — three MPS runs identical to four decimals, 127–169 s wall clock. The spread in wall clock against the exactness of the metric is the point: it is a cheap gate whose *number* is stable even though its *timing* is not. The harness this measures — 2000 generated scenes, held-out scoring, mosaic held on for five of six epochs specifically so a collapsed augmentation RNG ([WP-079](#wp-079)) would be detectable — is described at [ENGINEERING_LOG.md#wp-083](ENGINEERING_LOG.md#wp-083).

## 🖌️ Phase 7 — Instance segmentation, release 0.2.0

### WP-052 — Segmenter wiring

<a id="wp-052"></a>

Absence of the training-only branches from `deploy()` is proved twice rather than asserted: by parameter identity and a module walk, and independently by arithmetic — `deployed == full - aux - o2m`, exact at **2,717,284 − 5,200 − 152,690 = 2,559,394**. Two proofs because the failure mode is a branch that is disabled rather than absent, which a module walk can miss and arithmetic cannot.

### WP-052b — Table S9 fidelity gate

<a id="wp-052b"></a>

R1 Table S9 was transcribed from the arXiv **PDF** (page 28) after the HTML rendering proved to truncate before section S5 — worth knowing before trusting any other appendix table read from HTML: n 2.7/9.1, s 10.4/34.2, m 23.6/121.5, l 28.0/139.8, x 62.8/313.5 (M params / B FLOPs).

The params tolerance is **±3%**, wider than detection's ±2%, and the widening is argued rather than fitted: the segmentation head's sizing rests on five registered assumptions (A14, A15, A18, A34, A35) instead of published structure. `s` is the binding scale at +2.5% and every other lands inside ±1.5%.

FLOPs hold ±5% on all five scales under the same 2×MACs convention Table 7 needed (A29). That is corroboration of A29 rather than a second assumption — the convention was fixed on detection and predicted segmentation correctly.

Two gaps in the source worth recording: S9 states neither its FLOP convention nor whether Params denotes the full or the deployed model (E2E and non-E2E share one pair per scale), and unlike Table 7 it attributes no source.

### WP-087b — batched mask-loss gather

<a id="wp-087b"></a>

Wiring the segmentation path made it trainable and slow: **1.59 s a step against detection's 0.39** on an RTX PRO 6000, GPU idle, main process on CPU.

Cause was not volume — ~2,340 masks, 0.24 GB. `_branch_mask_loss` selected each image's positives with a boolean mask index, whose output shape is data-dependent, so the device had to report the surviving count back to the host before the next op could be built. Inside a loop over the batch, run per branch, that is **2×B forced syncs a step — 64 at batch 32**.

Padding to the batch's largest positive count and selecting once costs two syncs a branch at any batch size. Positives keep ascending anchor order within an image and image order across the batch, so summation order — and therefore the value — is the loop's own.

Padding is where a perf change turns numerical, and it did: padded rows carry `gt_index = -1`, the unassigned sentinel, which is a legal negative index in Python and out of bounds in a gather. The existing gradient test caught it before any of the perf work ran.

### WP-087c — vectorised polygon rasterisation

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

### WP-087d — rasterise mask targets in the loader workers

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

### WP-087e — epoch mask mAP during validation

<a id="wp-087e"></a>

A `task="segment"` run logged `val/mAP` and nothing about its masks, so the branch the run exists for moved no epoch metric at all: a run whose masks were degenerate and one whose masks were right produced the same curve, and the first sign of either was an evaluation pass hours after the run ended.

Two measurement decisions are load-bearing and were argued from cost rather than taste:

**Scoring on the prototype grid**, where the loader's ground-truth masks already are. Upsampling both sides by four would cost sixteen times the memory to compare the same two fields at a finer sampling of one boundary. The metric is a proxy in exactly the sense `val/mAP` already is (WP-077); the acceptance figure stays the standalone evaluation.

**A second `MeanAveragePrecision` rather than `iou_type=("bbox", "segm")`** on the existing one. One metric holding both would take each instance's area from whichever frame torchmetrics picked, mis-bucketing the small/medium/large splits by a factor of sixteen. This is a case where the convenient API is wrong for a reason that would never surface as an error — only as quietly wrong area breakdowns.

### WP-087f — tier names

<a id="wp-087f"></a>

The configs were named `<task>_tier_<tier letter>_<scale letter>`, putting two letters from unrelated alphabets side by side, with the tier letter reading backwards (A the cheap smoke tier, B the longer paired-run one).

The scale letter was not merely confusing but **false**: it sets `variant` inside the file and `--variant` overrides it, so `seg_tier_a_n.yaml --variant m` is an unremarkable invocation whose filename says `n` while the run is `m`. A filename that can contradict the run it names is a defect, not a style question.

Renames propagated to historical mentions too — a reader meeting "Det-A attempt 2" in the A13 row has no file left to connect it to — with git history keeping the old names. Each config header keeps its `blueprint sec. 10 tier A` citation, because the blueprint is not in this repository and still speaks in letters, so the mapping has to survive somewhere explicit.

______________________________________________________________________

### WP-087 — factories, not a `Detector` object

<a id="wp-087"></a>

Composing the Phase 7 modules onto the shared backbone and neck went behind `build.py` factories rather than behind a `Detector` object holding them. The reason is the state dict: an object that owns the backbone and neck renames every parameter under it, and flat keys are what keep the accepted Det-smoke checkpoint loading **strict** after a second task lands beside it. That checkpoint is the artifact later phases load, so the DoD asserts the strict load directly rather than trusting the module tree to have stayed the same shape.

### WP-054 — a DoD clause with no referent

<a id="wp-054"></a>

The 0.2.0 release row owed "the 0.1 goldens still green", and that clause could not be met as written rather than merely being missed: no `v0.1.0` tag was ever cut, WP-046 was superseded, and `goldens/frozen/0.1` never existed to be green or red. The release froze `goldens/frozen/0.2` instead.

Worth keeping because the shape recurs in a roadmap transcribed ahead of the work: a definition of done can name an artifact that the intervening packages decided not to produce, and the honest outcome is to record that the clause lost its referent — not to satisfy a nearby clause and tick the row.

## 🔄 Phase 8 — Oriented detection, release 0.3.0

### WP-055 — rotated geometry primitives

<a id="wp-055"></a>

Canonicalization is exactly idempotent rather than idempotent to tolerance, and the angle range is a guarantee rather than a near-certainty. The edge case that forces the final clamp: for an angle within an ulp of a range bound, shifting by multiples of pi cannot rescue it — the remainder rounds to pi and the shift returns its own input. A similarity round trip of a box at exactly −45° reaches this in practice, so a clamp closes it at a cost of one ulp.

The square tie-break keys on exact `w == h`, which means a quad reconstructed from DOTA's eight coordinates can return either pi/2-separated representative of the same rectangle. WP-060's angle loss, aimed precisely at near-square objects, must therefore not assume which — and that is why it is tested for bit-identical scoring of both representatives rather than agreement to seven digits.

### WP-058 — rotated-aware augmentation

<a id="wp-058"></a>

The three policy choices are A40. What belongs here is the **live defect it fixed**: mirroring negated `theta` without re-wrapping, so any box above 45° left the long-edge range. Nothing had ever hit it because no rotated box could reach a flip through a real pipeline — the augmentations raised `NotImplementedError` on non-empty `rboxes` until this package removed the guard.

A defect that is unreachable is still a defect, and it becomes reachable in the same commit that removes the guard. The general shape: when lifting a `NotImplementedError`, the code behind it has never executed and has never been tested, whatever its coverage says.

### WP-059 — ProbIoU

<a id="wp-059"></a>

R17's formulas are *evaluated* through cancellation-free identities (2×2 adjugate linearity for `B1`'s numerator, the `det(S1+S2)` expansion, `det S = w²h²/144`), and that is what makes float32 sufficient. The measurement that forced it: the **literal transcription loses 377% of `B_D` at a 1e-4 perturbation and returns `NaN` on parts of a 1000:1 aspect sweep**. Float64 internals are not an available fallback, because a regression loss runs on the training device and D12c puts training on MPS, which has no float64.

Both of R17's losses ship side by side — bounded Hellinger `L1` and unbounded `L2 = B_D` — because R17 suggests starting on the second and switching. Which one the oriented path uses, and the measured deficit of the alternative, is A49.

A DoD was **moved** here rather than met: the package originally owed a 1e-4 shapely agreement test. ProbIoU is a Hellinger-distance similarity between two Gaussians and R17 asserts no equality with polygon overlap, so that test could not pass and a correlation threshold would have been invented. The shapely oracle belongs to WP-063's exact rotated IoU (A24); ProbIoU got its own exact oracles instead (quadrature of the definition at 2e-14, literal float64 transcription at 1e-12). Recorded because "the DoD was wrong" is a legitimate outcome and the alternative — a tolerance chosen until it passes — is the failure this project is most exposed to.

### WP-060 — square-object angle loss

<a id="wp-060"></a>

R1 Table 11's own numbers set `lambda = 3`: **50.2 mAP against 49.0 with no angle loss, and 47.1 at `lambda = 5`** — i.e. the paper's own ablation shows the term can be worse than omitting it when mis-weighted. That is the context in which [WP-093](#wp-093) later found the gain A22 assumed was destabilising.

Two floating-point contracts rather than approximations: Eq. 14's `round` tie rule is unstated, so the wrap settles ties upwards into the half-open `[-pi/2, pi/2)` and is exactly idempotent; and the residual is folded once more into `[-pi/4, pi/4)`, a **Sterbenz-exact** shift under which the two representatives WP-055 leaves a square score bit-identically.

### WP-061 — rotated containment for TAL and STAL

<a id="wp-061"></a>

The axis-aligned path is an untouched early-return branch below the new one, and that was verified rather than assumed: assignments were snapshotted at the parent commit on a scene whose anchors sit **exactly on ground-truth boundaries**, then compared field by field with `torch.equal` — 20 tensors, 6,144 elements, bit-identical. "Its tests still pass" would not have distinguished a boundary case that moved by a rounding step.

### WP-088 — oriented training path

<a id="wp-088"></a>

This row was **left open with its criterion unmet**, which is the part worth recording. The training path verified clean — gradient reaches both branches' angle stems, each oriented gain has leverage, the detection step reproduces a snapshot replayed from a clean `fcf3040` export — and A49/A50 were measured rather than argued (their alternatives cost 0.5046 and 0.7112 rotated mAP50). But the overfit floor cleared on **0 of 3 seeds, mean 0.840, spread 0.091**.

The package stayed open rather than being closed on "the code is right", because the code being right was not what the DoD asked. It closed only when [WP-093](#wp-093)'s revised gain took the gate to 0.9390 at the recipe's own seed.

The criterion is met as written and not more than that: the five-seed picture behind that number is 4 of 5, which lives in WP-093's record rather than being folded into this one.

### WP-100 — parallel tiling

<a id="wp-100"></a>

A full DOTA build is **1,869 multi-megapixel PNG decodes**, each cropped into dozens of tiles and written back, and it ran silently on one core for about an hour — an operator had no way to tell a working build from a hung one.

Two constraints shaped the parallel form. Each worker pins itself to a **single torch thread**, since a pool of processes each fanning out over every core is oversubscription rather than parallelism. And ids are deliberately **not** assigned in the workers: a worker cannot know how many tiles preceded it, so numbering there would make the output JSON depend on which process finished first. The parent numbers every record in source order, and the test that protects this compares a four-image pooled build's JSON byte-for-byte against the serial one.

### WP-101 — JPEG tiles and a completion-order bar

<a id="wp-101"></a>

Two costs the first pooled build exposed.

**Decode cost at training time.** The tier decodes every tile once per epoch and mosaic assembles four source tiles per training sample, so a 64-image step at 1024 px is around **250 PNG decodes** — the loader, not the GPU, is what a smoke run waits on. `--suffix .jpg` writes through the JPEG encoder; PNG stays the default, because a tile is an exact crop of a lossless source and a build should not discard information on an operator's behalf. The choice is made once and collected every epoch.

**A bar that measures the wrong thing.** DOTA source images differ by an order of magnitude in size, and an ordered `map` left the bar frozen while one multi-thousand-pixel image held up results already finished behind it. Ordering is a requirement of the *numbering*, not of the *counting*: results are collected by index and replayed in order once the pool drains, and the bar counts completions.

Annotations are asserted byte-identical across the two encoders — nothing about a record passes through the encoder, so a suffix that ever reached the geometry is a bug the pixel difference would hide.

### WP-102 — validation that is not CPU-bound

<a id="wp-102"></a>

Validation left the GPU idle for most of its wall time, from two defaults nobody had revisited at 1024 px.

The val worker cap of 4 ([WP-073](#wp-073)) applied even when the operator had named `--data.num_workers 16`. A number an operator states says how much of the machine the run may use, and quietly validating on a quarter of it surfaces as an idle accelerator with **no message at all** — worse than the host-OOM the cap was avoiding, because it is silent. The cap now applies only to a count this class chose itself, and scales with pixel count rather than sitting at a constant reasoned about 640 px samples, where val was "a fraction of the train pipeline's work". At 1024 px the decode *is* the work.

Second, an `obb` run was accumulating the axis-aligned `val/mAP` beside the rotated one — a CPU pass over every detection of every batch, to produce a figure that reads the A44 composition's **pre-rotation** rectangle. It cannot separate a run whose orientations were right from one whose orientations were random. An oriented run now logs `val/rotated_mAP` and no `val/mAP`.

### WP-103 — a val worker count that fits the host

<a id="wp-103"></a>

**This package exists to fix a defect the previous one introduced.** WP-102 let a named `--data.num_workers` reach the validation loader and deleted the guard rather than the silence.

The arithmetic it deleted: at 1024 px a batch of 64 stacked float32 RGB images is `4 × 3 × 64 × 1024²` = **805 MB**. A named 32 workers at prefetch 2 queues about **51 GB for validation alone**, beside a training pool still resident at the epoch boundary. The result was the observed host-RAM crash on the first oriented tier run.

The division that was wrong the first time and is now explicit: an operator states parallelism with that flag; **the gigabytes it costs are this class's arithmetic**. A count inherited from a named train count is bounded by the memory budget and warns with the number and the override flag; a count stated as `--data.val_num_workers` is bounded by nothing, which is where an operator asks for the whole machine.

The general lesson, recorded because it was a real mistake and not a hypothetical: the principle behind WP-102 was right — a stated number should not be silently reduced — and the implementation deleted the wrong thing. Removing the *silence* was the fix; removing the *bound* was not.

### WP-104 — one-pass IoU threshold matching

<a id="wp-104"></a>

The oriented epoch boundary stalled for about a minute after validation reached 100%. Diagnosed before anything was changed: `evaluate_rotated_map` scoring ~10k val tiles at 300 detections each, **178 s at that scale, 58% of it in `_match_image`**.

The ten IoU thresholds share the overlap matrix and the score order and differ only in what counts as a hit, so ten passes re-walked the same detections to consume different subsets of the same ground truths. The availability state becomes `(T, G)` and the walk happens once. A second win falls out: detections reaching no ground truth at the lowest threshold skip the walk entirely — they cannot score, cannot be discarded and consume nothing at any threshold — which on an early-epoch model is nearly the whole loop.

**178 s → 38.6 s**, every pinned value unchanged.

**Rejected: a circumradius prefilter on the IoU kernel.** Written, measured at **12%**, reverted. The residual cost is per-call overhead across ~150k small calls, not pair count, so a cheaper rejection test attacks the wrong term. Carrying an extra geometric shortcut in hand-written rotated geometry is not worth 12% — the code it would live in is the code whose correctness the whole oriented evaluation rests on.

### WP-056 — one mask filters every modality

<a id="wp-056"></a>

What the DOTA parser guarantees beyond its shapes is an **alignment**: `rboxes[i]` is the same instance as `boxes[i]` and `labels[i]`, for every `i`. That is why it is stated as an invariant rather than left as an implementation detail — one boolean mask filters every modality at once, so a caller dropping difficult instances, or [WP-057](#wp-057) dropping instances a window lost, writes the filter once instead of three that are free to drift apart. Every stage of the oriented path downstream of the loader relies on it.

### WP-057 — tiling is what creates difficult instances

<a id="wp-057"></a>

DOTA ships a per-object `difficult` flag and [WP-056](#wp-056) declined to decide its fate at load, deferring to A39. Tiling is where that deferral turns load-bearing: R18's partial-object rule flags an instance difficult when its clipped area falls below 0.7 of its original, so **the tiler manufactures difficult instances the source annotations never carried**. A39 therefore governs a population that DOTA's own flag column does not describe, and the number the decision was made from has to survive the crop — which is why `U` is carried out as `visible_fraction` rather than consumed and discarded.

### WP-062 — the fidelity gate picked the stem width

<a id="wp-062"></a>

A20 records that R1 gives the angle branch no width: sec. 3.4.3 says only "separate branch", Fig. S2 draws one stem shape and no numbers, and Table S11 reports whole-model sizes a branch width can only be inferred back out of. This is the package where that inference was made, and it was made by the gate rather than by argument — the Table S11 comparison at 1024 px over 15 classes selected the **wider `channels // 2` stem**, and A20 was revised to it.

The margins say how much room the choice had: params bind at scale `s`, +3.27% against a ±3.5% band, and FLOPs at scale `n`, +4.84% against ±5%.

### WP-063 — four constants a library would otherwise have chosen

<a id="wp-063"></a>

R1 reports rotated mAP50-95 on DOTA-v1.0 val and defines none of the machinery that produces the number — not the IoU, the matching rule, the recall interpolation, the detection cap nor the class averaging (A24). Four constants therefore had to come from somewhere, and what is recorded here is that they come from the register rather than from whichever library the accumulator was built on: the 101-point recall grid (A46), the 300-detection per-image cap matching what the head emits (A47), the exclusion of classes with no non-difficult ground truth, and R18's difficult rule with an explicit precedence against the argmax match (A48).

A default inherited silently from a dependency is an unregistered assumption wearing a library's name, and it moves a published figure exactly as far as a registered one does. The difference is only that nobody can find it later.

### WP-097 — what R18's published totals actually describe

<a id="wp-097"></a>

R18's headline figures — 2,806 images, 188,282 instances, 15 categories — describe DOTA-v1.0 **whole**, and R18 releases ground truth for two of its three splits. An annotated root therefore holds about two thirds of that instance count, not the published total — the fact `check_dota_root` got backwards, fixed at [ENGINEERING_LOG.md#wp-097](ENGINEERING_LOG.md#wp-097).

### WP-064 — shipping a figure that compares to nothing

<a id="wp-064"></a>

0.3.0's oriented numbers are **per tile**. R1 and R18 report whole-image DOTA figures, so nothing this release published is comparable to anything outside it, and the release's own disclosure is what keeps that gap from reading as an omission: the reproduction report, the oriented model card and the README each say so.

The row also owned the whole-image tile merge and did not deliver it. It moved to [WP-107](#wp-107) rather than being marked done — a package that reports itself complete while a clause of its scope is unbuilt is how a roadmap stops describing the code, and this is the third time the merge had slipped, after WP-063 deferred it and WP-088's scope never took it up.

## 🔮 Phase 9 — Inference and generalization

### WP-093 — stable angle regression for elongated boxes

<a id="wp-093"></a>

The revised gain and its five-seed dose-response are A22. What lives here is the **diagnosis**, including the two mechanisms that were measured and ruled out — the part that would otherwise be lost, since a register records what was concluded and not what was eliminated.

Starting point: the oriented overfit gate cleared its 0.9 floor on **0 of 3 seeds at 100 epochs (mean 0.840)** and **2 of 3 at 200 (mean 0.910, spread *widened* to 0.148)**. A longer budget is not the answer, and the widening spread says the problem is instability rather than under-training.

Measured per class, the shortfall is carried entirely by the one class whose AP depends on orientation:

| class | behaviour |
| -- | -- |
| square, circle | symmetry-protected: 0.93–0.99 AP while carrying 40–49° mean angular residuals |
| triangle | near-square (w/h 1.15), tolerant |
| rectangle (w/h 2.00) | tracks angle precision directly — carries the whole shortfall |

Two hypotheses were tested and **rejected**:

- **A quarter-turn local minimum.** Rectangle carries the *least* angular-error mass beyond 60° of any class. The 90° mass sits in square, where it is a true symmetry and costs nothing. The distribution is bimodal rather than gradual, which is what suggested a local minimum in the first place.
- **Under-training.** Seed 0 *regressed* from 0.8959 to 0.8177 when the budget was doubled.

Cause, found by paired five-seed ablation: R1 Eq. 15's angle term itself, at the 1.0 gain this project had assumed.

Two things the record deliberately does not do. It does not lower the floor to fit the measurement. And it does not claim a clean fix: the revised gain reaches 4 of 5 seeds, not 5 of 5 — seed 1 lands at 0.8638 with 19.84° error. Only dropping R1's angle term entirely reached 5 of 5, and that is a paper deviation the register does not authorise, since A22 is a gap R1 leaves open while the term itself is not.

The residual instability sits at 2:1 aspect ratios, which is the band where `omega = 0.948` — a five percent reduction where R1's stated intent is that elongated boxes are "primarily constrained by the rotated IoU loss". At DOTA's 5:1 and 10:1 ratios `omega` genuinely falls (0.750, 0.555) and the term may behave as R1 intends. That remains unmeasured.

### WP-091 — the decode path that does not exist

<a id="wp-091"></a>

An oriented checkpoint can be read through one decode path, not two. The dense branch is not what is missing — an `obb` head builds both angle stems. What is missing is a **rotated suppression decoder**: running the axis-aligned `NMSDecoder` over columns that are a centre and two extents would suppress by upright overlap and keep or drop the wrong boxes with nothing in the output to show it had. `rotated_iou` exists in `eval/dota_eval.py` as the ingredient and no decode path uses it.

So `--decoder nms` raises rather than quietly running `e2e` instead. Substituting would be the worse failure of the two, because the report records the decoder it was *asked* for, and the file would then attest to a path that never ran. Recorded as 091b rather than fixed in passing: a suppression rule for rotated boxes is a new decode path with its own assumption to register, not a parameter. It landed as 091b directly afterwards, and the assumption is what the delay bought — see below.

The canonicalization the row named turned out to be already there, which is worth writing down because the row's phrasing invites the opposite reading. `decode_rboxes` ends in `canonicalize` on the dense `(B, A, 5)` output — A23 records exactly that — so a caller gets canonical angles from the existing decode. What this package had to establish is the other half of the guarantee: that canonical form *survives* the trip to original coordinates. A letterbox inverse is one isotropic scale plus a translation, so it scales both extents by a single positive number and turns no angle; `w >= h` and theta are invariant under it and nothing re-normalizes afterwards. Three places could host that normalization and two are wrong: in the head it would normalize an angle the loss trains against raw (R1 Eq. 13 emits the pre-activation), and after the inverse it would run once per entry point with each copy free to drift.

The test for it plants four raw angles that all describe one rectangle — already canonical, a half turn above the range, below the `-pi/4` floor, and the extents stated short edge first — and requires all four to come back as the same box, asserted after the inverse rather than before. A fifth plants 500 radians and asserts only the range and the long-edge order: computing which representative 500 folds to would restate `canonicalize` inside the test, and a restatement agrees with a broken implementation as readily as with a correct one.

### WP-074 — deferred, and what the search for a public YOLO corpus found

<a id="wp-074"></a>

Deferred on 2026-08-14 rather than built. `roboflow/rf100-vl` PR 29 is open and adds `combine_downloaded` / `download_and_combine` to the collection's own package — deterministic id remapping, category namespacing, a resumable manifest — which is this row's scope, written by the people who own the datasets. Building it here first means maintaining a second merge against a moving upstream and owning its correctness across 100 datasets; waiting costs a wrapper. A second route may retire the row outright: `probicheaux/rf100-vl` on Hugging Face is Apache-2.0 and ungated, 15 domains in COCO form, so the tier becomes a download plus a layout conversion with no API key involved.

The more useful finding came from asking the adjacent question — what public dataset could exercise the WP-099 YOLO reader at scale — and getting no good answer. The search is worth recording because the *shape* of the negative result explains WP-099's own central difficulty.

Rejected, each for a stated reason rather than a hunch: the `keremberke/*` mirrors are the largest set of public YOLOv8-format datasets on Hugging Face and state **no licence at all**, and unstated is not permissive. `CARD-Data/CARD-Germany-Batch1` is real-scale YOLO but is 315 GB, gated behind login plus terms acceptance, and its "CC BY 4.0" carries a no-military field-of-use rider — a licence with a use restriction bolted on is not the licence it names. `goodquestion1/RM26` is YOLO *pose*, whose extra keypoint fields the reader rejects by design. A Zenodo query for YOLO-format detection records under permissive terms returned nothing that was both an image dataset and detection.

What is left is the structural point. Public YOLO-format detection data at scale is almost entirely (a) Roboflow exports behind an API key and (b) Ultralytics-hosted mirrors, which this project's prime directive forbids reading. **The format has no neutral publisher, which is the same fact WP-099 met from the other side when it found no specification to write the reader from.** A convention owned by tooling vendors rather than by a standards body produces exactly this: universal adoption, no spec, and no corpus anyone can point at without accepting someone's terms. The consequence for this project is that the YOLO path's public story is a small verified example — the R32 export, CC BY 4.0, already on disk — plus documented instructions for bringing your own export, and not a headline dataset.

The design the row carried before it was deferred, kept here so a later attempt starts from it rather than from scratch. Per-dataset download, then category ids remapped into a **union label space** under dataset-qualified names — the collection spans domains, so bare category names collide across datasets and a collision is a silent label merge rather than an error — then images re-numbered, then appended into one merged COCO layout, with each per-dataset archive deleted before the next is fetched. That last step is what bounds peak disk to O(one dataset + merged) rather than O(all 100). Splits are preserved as published, and an `RF20-VL` subset flag exists so a smoke run does not pay for the whole collection. Two design points were deliberately left open rather than settled ahead of the code: whether evaluation reports over the union label space or per dataset, and how the verify module integrates.

Two citations for whoever picks this up: the collection is <https://github.com/roboflow/rf100-vl> and the upstream work is its PR 29, <https://github.com/roboflow/rf100-vl/pull/29>. Reconsider this row when that PR lands or is abandoned.

The licence question does **not** resolve with either route and is the thing most likely to kill the row. `rf100-vl` is Apache-2.0 **as tooling**; the 100 datasets it downloads carry their own licences, individually, and this project ships permissive-only. The Hugging Face alternative (`probicheaux/rf100-vl`, Apache-2.0, ungated, 15 domains, ~163k rows in COCO form) turns the tier into a download plus a parquet-to-COCO-layout conversion — no API key and no merge to own — but it does not answer what licence the underlying images carry.

**Dropped 2026-08-18.** The deferral above named the condition to watch (PR 29 landing or being abandoned) but not one this project controls or a date to stop waiting on — the licence question that named itself the row's likeliest killer never resolved either. Rows 074 and 075 close ⊘ rather than stay ⏸ indefinitely; the design and citations above are kept as-is for whoever reopens the tier, unedited by this note per this log's own convention of leaving a historical entry describing what was true when it was written.

### WP-091b — a threshold with nothing to cite

<a id="wp-091b"></a>

The interesting half is the threshold, and what it cost to find that there is nothing to cite for it — the decode rule itself (greedy class-wise suppression over `rotated_iou`) is described at [ENGINEERING_LOG.md#wp-091b](ENGINEERING_LOG.md#wp-091b). R1 states no suppression parameter at all — the one-to-one branch's freedom from NMS is the paper's claim, not a setting. R18's devkit scores detections it is handed and settles nothing on the decode side. R13 fixes the angle convention and no suppression rule. So the oriented comparison column has a knob no allowlisted source turns, and the register (A61) says exactly that in its source column rather than dressing a convention up as a citation.

What the value rests on instead is holding one variable still: 0.7 is carried over unchanged from the axis-aligned path, so the two comparison columns differ by the overlap *measure* alone and any gap between them is attributable to the geometry rather than to a second knob turned at the same time. The defence that was **rejected** is more interesting than the one taken. It would be natural to argue that 0.7 is safe because rotated IoU is the smaller measure, so carrying the threshold over can only suppress less. That is false: rotated IoU is usually smaller — mean 0.047 against 0.080 over 3000 random canonical pairs — but it *exceeds* the envelope IoU whenever both boxes lie along a shared diagonal, at 138 of those 3000 pairs and by up to +0.146. An argument that sounds like a bound and is only a tendency is worse than no argument, because it stops the next reader from checking.

The exposure is recorded with numbers rather than a hedge: 0.7 under-suppresses elongated objects, which is DOTA's characteristic shape. Two detections of one 10:1 object whose headings disagree by 5 degrees score 0.643 and both survive. That direction was accepted deliberately — a duplicate is inspectable in the output, where an over-suppressed object is simply absent — and the value is a constructor argument, so the alternative is one key away.

Looking for a lineage to inherit turned up that there was none. The axis-aligned `0.7` had shipped since WP-042 as a bare default with no register row, which means every "mAP (non-E2E)" number this project has published rested on an unregistered constant. It is now **A62**, registered at its shipped value rather than revised: re-picking it would silently move a comparison column that already has committed figures beside it, and that is a 0.MINOR change with goldens, not a tidy-up. The general shape: an assumption that goes looking for its own parentage is a good way to find the rows nobody wrote.

### WP-091c — what a refactor is allowed to notice

<a id="wp-091c"></a>

The move itself is unremarkable: `rotated_iou` and the seven private helpers it exclusively owns leave `eval/dota_eval.py` for `data/rotated_geom.py`, since an evaluation module had no business being on the decode path's import graph for a function that is pure geometry. What the package is worth recording for is the three things it declined to do.

**It did not merge the duplicate it found.** `rotated_geom` already had `_check_2d`, differing from the arriving `_check_rboxes` only in the letter naming the row count — and the two raise `must be (N, 5)` and `must be (K, 5)`, which two different test files match on. Merging them is therefore a test-visible behaviour change wearing the costume of a duplicate removal, and a refactor whose contract is "every existing test passes unchanged" cannot also change what an error says. It is recorded as 091d instead. A duplicate that is cheap to see is not always cheap to remove, and the difference is whether anything asserts on it.

The search also turned up `_check_rboxes` in `losses/probiou.py`, first reported as a third copy and corrected on re-reading: it guards `ndim == 0 or shape[-1] != 5` and raises `must be (..., 5)`, so it accepts arbitrary leading dimensions where both `rotated_geom` validators require exactly 2-D. Same name, same five, different contract — merging it into the other two would tighten what the loss path accepts, which is a behaviour change and not a rename. It is 091e, and its test coverage differs in the same direction: `tests/losses/test_probiou.py` matches the prefix `target must be` only, leaving the tail unpinned, where both `rotated_geom` messages are pinned tail and all. Three lookalikes, two of them the same check — the way to tell was to read each guard rather than each name.

**It did not deprecate the old name.** `pyDeprecate` is nowhere in this tree, and `warnings.warn` would have fired on both existing suites — which is not "tests pass unchanged" under any honest reading, and arms a trap for the day `filterwarnings = error` is set. The deeper reason is that `dota_eval.rotated_iou` is not a legacy alias: that module's docstring defines R18's protocol *in terms of* its overlap measure, so a caller reproducing the protocol wants the kernel where the protocol is. One function visible from two places is what `canonicalize` already is.

**It moved the argument, not just the code.** The "why there is no float64 here" section and its five-row error table went with the kernel, because that table is the *evidence* for the shift-before-expand ordering rather than commentary about it; a precision argument left behind in the module that no longer owns the arithmetic is how a constraint quietly stops being checked. Two claims in the destination's own docstring also had to change or become false — "no Python loop over boxes" and a blanket "float32 tensors" — which is the ordinary tax of moving code into prose written before it arrived.

The evidence that the move was inert is the goldens and only the goldens. `rotated_iou` sits on the scoring path, and a silently perturbed kernel would still clear the shapely oracle's 1e-4 tolerance; 20/20 unmoved, frozen subtrees included, is what says the arithmetic is bit-identical.

WP-091b had already verified there was no import cycle, from four entry orders, and left the module where it was rather than widening its own diff — which is why this is a package and not a line in that one.

### WP-107 — whole-image merge

<a id="wp-107"></a>

Recorded before the work, because the reason this is a package rather than a step inside another one is itself the finding: two overlapping 1024 px tiles that both detect one object produce two detections at full confidence, and an NMS-free path has **no suppression stage to remove the duplicate**. The merge is a policy that has to be decided and argued, which is precisely why it kept failing to happen as a side effect — WP-063 deferred it into WP-088, whose scope never took it up, and WP-064 shipped 0.3.0 without it.

What the work established is that the duplicate rule does not have to be a suppression rule. Core ownership (A59) keeps or drops a detection by **where it is**: the cores are built from the recorded windows before the model runs, nothing is compared against another detection or ranked by confidence, and a detection is dropped whether or not the neighbouring tile found anything. Run the model twice with different weights and the same detection is owned by the same tile. A confidence-ranked dedup by rotated IoU is the defensible alternative and would score *higher*, because it takes the union of the tiles' recalls rather than the owner's — it was rejected because the figure is meant to measure a suppression-free pipeline, and reintroducing suppression at the seam makes it measure something else. That rejection is the package's actual content; the code is the cheap part.

The cost is real and one-directional: whole-image recall is the owner's recall, so this merge can only *lower* a per-tile number. It is pinned by a test rather than left as prose, so that a later reader who finds the drop surprising cannot quietly fix it into NMS.

One residual runs the other way and was nearly missed. The first draft of the module claimed the difficult straggler "cannot inflate the score". That is wrong in exactly the direction this package exists to police: a detection landing on a surviving clipped copy is *discarded* under A48, where against the true whole-image annotation set it would have been a false positive, and discarding false positives raises precision. It is bounded to objects wider than the overlap and it is R18's own devkit behaviour rather than an invention here — but it is not zero, and it is now measured at **0.165 of `map_50`** on the unit fixture rather than argued away. A claim of the form "cannot" about one's own instrument deserves a measurement before it is written down.

The DoD's equality clause needed one guard the row did not ask for. A single-tile image must score identically through both paths, and an equality between two saturated `1.0`s would prove nothing at all — so the fixture is built to land every one of the four metric keys strictly inside `(0, 1)`, asserted separately, and the equality is then asserted with no tolerance.

Two smaller things the merge inherits rather than solves. `--limit` can end a tile prefix mid-source-image, so an incomplete trailing image is dropped rather than scored partially — a whole-image figure under `--limit N` therefore covers at most N tiles' worth of *complete* images. And A52's `--drop-empty-tiles` moves the core midpoints when windows are missing from the layout: still a partition, but the `v/2` circumradius guarantee weakens, which is a property of that off-protocol layout rather than of this rule.

### WP-111 — the whole-image figure runs higher, not lower, and both reasons are already on the register

<a id="wp-111"></a>

WP-107 built the merge and its own log entry above is explicit: "whole-image recall is the owner's recall, so this merge can only *lower* a per-tile number", pinned by a fixture test. The first real run — the accepted OBB-smoke checkpoint (`version_10`, 2026-08-14) re-scored on DOTA-v1.0 val, 458 source images — comes out the other way on every one of the four metrics: mAP50-95 0.2914 → 0.3146, mAP50 0.5242 → 0.5484, mAP75 0.2770 → 0.3111, mAR_300 0.5075 → 0.5193. That is not the merge mechanism breaking its own guarantee; it is a second, separately registered effect outrunning it.

WP-107's claim is about core ownership in isolation — on a fixture too small to ever reach a detection cap, dropping every non-owning tile's copy of an object can only cost recall. The real pipeline runs the merge alongside A60: per-tile scoring caps each 1024 px tile at 300 detections (`MAX_DETECTIONS`, A47 — what `o2o_rotated_topk` emits per forward pass), while the whole-image pass scores with that cap lifted, because a source image assembled from ~22 tiles (10,132 tiles / 458 images) is not one forward pass and re-imposing 300 on it would measure a truncation rather than the model — exactly the exposure A60's own text names ("a dense DOTA image carrying far more than 300 instances"). This is the first run that actually exercises that lifted cap against real data, and the direction is consistent with what A60 predicted: recall recovered by the uncapped pass outweighs what core ownership gives up.

The two effects run in the same pass here and are not separated by this measurement — a controlled comparison (whole-image scoring with the 300 cap re-imposed, versus lifted) would isolate how much of the `+0.012` mAR_300 is the cap and how much is anything else, and nothing in this run does that. Recorded as consistent with A60, not as proof of it.

### WP-091d — one letter, and what it costs to change one

<a id="wp-091d"></a>

091c declined this merge and said why: the two validators' messages are matched by two test files, so removing the duplicate changes what an error says, and a refactor contracted to leave every existing test passing cannot also do that. Done as its own package the visibility is the point rather than the obstacle. `_check_rboxes` is deleted, `rotated_iou` guards through `_check_2d`, and `tests/eval/test_dota_eval.py`'s `match=r"must be \(K, 5\)"` becomes `r"boxes_a must be \(N, 5\)"` — a *stricter* regex than the one it replaces, since it now pins the argument name as well as the shape. The other pinned message, in `tests/data/test_rotated_geom.py`, never moved: it was already the surviving spelling.

The letter itself was the only real question. `rotated_iou`'s signature does distinguish the two operands — `boxes_a` is `(M, 5)`, `boxes_b` is `(N, 5)`, and the result is `(M, N)`, so the distinction is load-bearing in the docstring. It is not load-bearing in the guard: each operand has to satisfy one and the same predicate, and a rejection message that spelled one of them `K` named a difference the check never tested. `N` reads as "some row count" in every message the module raises, which is what a shape rejection is actually saying.

What made the merge safe is narrow and worth stating, because the neighbouring lookalike fails it. The two conditions were character-identical — `ndim != 2 or shape[1] != 5` — so nothing but the string changed. `losses/probiou.py`'s guard passes arbitrary leading dimensions, and folding that one in would tighten a live contract rather than dedupe a message; that is 091e, and it is a different package for that reason and not for tidiness.

### WP-091e — a contract nothing in the tree exercises

<a id="wp-091e"></a>

The row asked which way to settle `losses/probiou.py`'s looser guard, and named the evidence it wanted: what the loss is actually called with. That evidence is one line. `oriented_loss.py:217` calls the resolved form on `(P, 5)` against `(P, 5)` and gets `(P,)` — aligned pairs, already matched by the assigner. **No caller in `src/` ever passes a leading dimension.** The pairwise `(4, 1, 5)` against `(1, 3, 5)` shape appears in `tests/losses/test_probiou.py` and nowhere else, and the one place in the tree that does want a pairwise rotated-overlap matrix — `decode/rotated_nms.py`, and the DOTA evaluator behind it — reaches for `rotated_geom.rotated_iou` instead, because it needs the polygon-exact measure and not R17's Gaussian surrogate.

So the honest reading is that the N-d contract is a property of the *public* function rather than of any in-tree use, and the decision turns on whether that is a reason to drop it. It is not. Everything in these two losses is elementwise in the leading shape — `_unpack` unbinds the last dimension and every term after it is arithmetic on those five tensors — so the broadcast is not a feature that was added, it is what the code does when the guard stops refusing. Narrowing to 2-D would be the validator inventing a restriction the arithmetic does not have, and it would take `probabilistic_iou`, an exported symbol whose docstring promises `(..., 5)`, with it. A guard should assert what the function needs, and this one needs the trailing five.

What the package changes is therefore not behaviour but what is asserted. The test matched `"target must be"` — the prefix only, leaving the shape the message names unpinned, so a future edit could have narrowed the promise from `(..., 5)` to `(N, 5)` with every test still green. It now matches the tail as well, and a new case runs a `(2, 3, 4, 5)` input through both losses and then through `canonicalize`, which rejects it: the divergence between the two validators is stated in one place, by a test that fails if either side of it moves. That is the answer to 091c's third finding — three lookalikes, two of them the same check, and now both facts are pinned rather than remembered.

## 📦 Phase 10 — Consolidation

### WP-065 — what three tiers say that none of them says alone

<a id="wp-065"></a>

The row said "merge det/seg/obb sections". It appends instead, and the reason is D10: the report is append-only, a later release correcting an earlier claim by adding to it rather than editing the record away. Merging three accepted sections would rewrite three records that three `[PRINCIPAL]` gates signed off on. So the consolidation is a fourth section that reads the other three against each other and revises none of them, and its own first line says it is not a fourth tier: no run stands behind it.

**The consolidation found an absence, and the absence is the result.** The NMS-free deploy path's cost against NMS is the one quantity all three tiers could have reported in the same units — Det-smoke measures 1.11 AP raw and 1.46 EMA, Seg-smoke 1.27 box and 0.81 segm. OBB-smoke reports EMA against raw weights and never evaluated an NMS baseline at all, so the project has no figure for what NMS-free deployment costs orientation, on a path built precisely to deploy without NMS. Three sections can each be complete and the set still have a hole in it; nothing but putting them in one table shows it.

The assumption tables turned out not to overlap the way the package expected. Twenty-three ids, and **no id appears in two tiers' tables** — the spawn brief assumed A11 did, and it does not: A11 is detection's anchor rule, and segmentation's table has its own row for it saying the mask crop reuses that rule rather than defining a second one. A13 crosses the same way, named inside the oriented A49 row as the gain the rotated term inherited. Two cross-references, zero duplicate rows, which is a tidier register than anyone had checked for.

What the union does show is a count that no single tier's table could: **eleven of the twenty-three are `held` or `not exercised at its recorded value`** — an assumption a passing run carried without isolating. That is not eleven weak results, and the note says so; the runs cleared their gates. It is eleven places where the evidence is "the run worked" rather than "this was measured", and A21 is the extreme case, where the tier ran at R18's 512 px overlap and the register's own 200 px value therefore remains unmeasured by the run that was supposed to exercise it.

The learning-rate line is the other thing only the side-by-side view produces. Det-smoke and Seg-smoke both scale lr from a batch-64 recipe; OBB-smoke, at half the batch and 2.56x the pixels per step, leaves it at the config value. Each section states its own choice and none argues it; three facts in three places become one open question in one.

Four hypotheses close the section, each with the evidence already in the file and the experiment that would settle it, and one of them is there specifically to *not* be upgraded: the segmentation run's box accuracy exceeds detection's, and the 0.2.0 section calls that confounded by precision, hardware and pipeline. A consolidation is exactly where a confounded observation gets quietly promoted to a finding by being restated one more time, so it is restated as the confound.

The gate is `test_report_sections`, reading headings only and never their content — which is what makes it compatible with append-only. A correction landing inside a section leaves every heading where it was; a section renamed out of the file does not, and that is the failure the test exists to catch.

### WP-066 — the exported graph confirms the one-to-one claim, not just its shape

<a id="wp-066"></a>

All three deploy paths export on the legacy TorchScript tracer at opset 20 and none of them contains a `NonMaxSuppression` node — a single `TopK` stands where suppression would be, and the two `Softmax` nodes belong to the attention block, not a distribution-focal-loss box head. The DFL-free, NMS-free one-to-one branch R1 claims shows up in the emitted graph itself, not only in a docstring. Verifying the graph's *arithmetic* — not just its shape — against the checkpoint it came from turned out to be almost entirely an export-tooling story rather than a fidelity one; it is recorded in full at [ENGINEERING_LOG.md#wp-066](ENGINEERING_LOG.md#wp-066).

## 🕺 Phase 12 — Keypoint detection, toward 0.5.0

### WP-125 — the flow beats its own control, at nano scale too

<a id="wp-125"></a>

R14's mechanism claim is that RLE's learned residual density buys real accuracy over the reparameterization, the sigmoid-bounded per-point scale and the `log sigma_hat` Jacobian it ships alongside — not that those three alone do the work. That claim needed a paired run differing only in the flow, which is what WP-135 built (`LaplaceNLLLoss`, R14 Table 7's own published control) and this row reads.

Both arms: `pose_nano_smoke.yaml` / `pose_nano_smoke_laplace_nll.yaml`, byte-identical but for `keypoint_loss`, nano variant, 50 epochs, COCO `person_keypoints_{train,val}2017`, same seed. RLE is Lightning run `version_11`; Laplace-NLL is `version_14` (two false starts, `version_12`/`version_13`, 11 seconds apart — each wrote only the TensorBoard writer's own opening record and nothing else, no `config.yaml`, no step, before exiting; the cause is not in what synced to Drive). Both scored with the identical `lucid-eval` pass, EMA weights, full val2017 split:

|  | RLE (`version_11`) | Laplace-NLL (`version_14`) |
| -- | -- | -- |
| e2e OKS AP | 0.2738 | 0.2527 |
| e2e OKS AP50 | 0.6094 | 0.5692 |
| e2e OKS AP75 | 0.2138 | 0.1933 |
| nms OKS AP | 0.2688 | 0.2525 |
| e2e box mAP | 0.5030 | 0.4471 |

RLE beats the ablation on every OKS statistic, both decode paths — a 8.3% relative gap on `e2e OKS AP` (0.2738 vs 0.2527), same sign as R14 Table 7's own 70.5-vs-67.4 AP (4.5% relative) at full COCO scale and full training budget. The absolute figures are not R14's — nano width, 50 epochs and no pretraining put a floor under both arms nothing here controls for — but the acceptance this row was scoped to is the mechanism's direction of effect, and a paired run differing only in `keypoint_loss` shows the flow term earning a real, consistent margin rather than none or a reversed one. `keypoint_flip_pairs` ran live in both arms — A64 was already resolved by WP-132's own reading of the training path, and this row is the first to exercise that resolved pairing at tier scale rather than through overfit-100's augmentation-off gate. `keypoint_gain = 1.0` (A68) trained a real pose head in both arms to a checkpoint that beats its own control rather than one swamped by or swamping the detection objective — the first evidence either way past a memorization-scale run, though the gain's own tuned adequacy stays unmeasured and A68 stays open on that question.
