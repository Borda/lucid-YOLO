# Research Log

What executing the work packages taught, kept out of the roadmap so that file can stay what it says it is: a queue. ROADMAP.md answers *what a package does*; this file answers *what it cost to find out* — the measurements behind a perf claim, the approaches that were tried and abandoned, and the results that do not flatter the project.

**This file is not a register.** Where a finding fixed an implementation choice the papers left open, the choice lives in ASSUMPTIONS.md and the number lives there with it; those are cited by id here, never restated. Release-level results live in REPRODUCTION_REPORT.md for the same reason. What lands here is what has no other home: engineering measurements, negative results, and the reasoning behind a rejection.

Sections are anchored by work package (`#wp-087d`) and the roadmap links to them. A package with nothing worth recording has no section.

**Phases 0–5 are backfilled** (2026-08-14) and read differently from the rest. Those roadmap rows were transcribed from the blueprint *before* the work and never rewritten afterwards, so nothing was carried out of them — their entries here are reconstructed from the commit record, which is the durable artifact, and not from anyone's memory of the work. Phase 6 onward is the reverse: those rows had accumulated the retrospective detail in place, and this file is where it moved to.

Dates are the day the finding was recorded.

______________________________________________________________________

## Cross-cutting

Findings that belong to no single package because they are properties of the environment every package runs in.

### Float reproducibility has two axes, not one

<a id="cross-float"></a>

A26 already records that the synthetic fixture generator is byte-identical per seed **on a given platform** and not across platforms — libm last-bit rounding differs, macOS arm64 against ubuntu x86-64 CI. That entry is about image generation. The same fault line runs through **torch reductions**, and it was found the hard way on 2026-08-14.

`test_obb_training.py` carried a snapshot of a `task="detect"` training step asserted with `==`, captured on arm64 at one intra-op thread. Three values of that same total have now been observed from source trees computing identical arithmetic:

| total | where |
| -- | -- |
| 22.80110550 | arm64, 1 intra-op thread (the recorded snapshot) |
| 22.80111694 | arm64, 12 threads |
| 22.80112076 | x86-64, ubuntu CI |

The spread is 1.5e-5 at magnitude 22.8 — about eight float32 ulps — and comes from the summation order a build chooses for the dense classification term. The test passed for the whole of Phase 8 because until 0.3.0 was pushed it had only ever run on the machine its snapshot came from; its first x86 execution failed it. It now compares at a relative 1e-4 with an absolute floor.

The general rule this establishes, beyond that one test: **bit-identity is a within-architecture property.** A gate that asserts it across machines is asserting something no library promises. What survives across architectures is a tolerance argued from ulp count, and what such a gate can still catch is a change to the objective — those move by percent. What it can no longer catch is a change of one rounding step. No tolerance choice recovers that.

### A green gate is only as trustworthy as the interpreter it ran under

<a id="cross-venv"></a>

Mid-session on 2026-08-13, `uv run` silently rebuilt `.venv` — Python 3.14.2, torch 2.13 — replacing the 3.11.16 environment a passing `make gate` had just run under. Nothing announced it; the first sign was unrelated behaviour changing. Restored with `make setup`, and every later invocation named `.venv/bin/python` explicitly rather than going through `uv run`.

The general point: "the gate is green" is a claim about an environment as much as about a tree, and a tool that can reconstruct that environment between two commands can invalidate the claim without touching a tracked file.

### Deterministic SVG is deterministic per matplotlib, not per run

<a id="cross-svg"></a>

WP-080 fixes `svg.hashsalt` and suppresses date metadata so an unchanged run regenerates its figure byte-identically. That holds — within one matplotlib. On 2026-08-14, regenerating the committed detection and segmentation figures produced files differing from the committed ones, and the cause was isolated by regenerating with the *git-stashed original* script: the committed script also differs, so the difference is the renderer, not the edit.

The committed figures were left untouched rather than churned. The consequence to know: a figure diff is not evidence that a plotting change altered a curve. Prove which by re-rendering with the pre-change script before reading anything into it.

### Two process failures that each cost a run

<a id="cross-process"></a>

Both recurred often enough to be written into AGENTS.md rather than left in notes.

**The gate certifies a tree, not an intention.** The loop runs `make gate` at step 4 and flips a roadmap row at step 6, and the gap between them is where a docs edit rides into a code commit that was never gated with it. That is how commit `fcf3040` left `main` red on two meta tests, and the next agent spent a full run reporting a blocker it was forbidden to fix. The invariant is a set equality between what was gated and what is committed; the docs-only exemption is the thing it is constantly confused with.

**A completion notification is not a completion.** A delegated work package returns marked *completed* whether the agent finished or stopped mid-sentence on a partial edit, and the notification carries a result field that reads like a report — so an unfinished run looks like a finished one that summarized badly. It happened **four times in Phase 8**. The check that catches it is one `git status` against the spec, and resuming the same agent beats respawning it, because its transcript is the context.

______________________________________________________________________

## Phase 0 — Foundation

### WP-001 — one place to write the version

<a id="wp-001"></a>

The version was declared twice and the two **had already drifted**: `pyproject.toml` said `0.1.0` while `lucid_yolo.__version__` said `0.0.1.dev0`, so a build published one number while every runtime consumer reported the other. `pyproject` now declares the version dynamic and reads the module attribute.

The attribute stays a plain string **literal** deliberately: setuptools resolves a literal by static analysis and only falls back to importing the package when it cannot, so a computed value would pull torch into every build and every source install. A meta test pins the literal form for that reason.

Also recorded — a test deliberately **not** written: `importlib.metadata` asserted against `__version__`. An editable install freezes its metadata at install time, so that assertion would fail after every bump until someone reinstalled, i.e. fail for a reason unrelated to what it claims to guard.

### WP-001 — markdown formatting, and what it found

<a id="wp-001-mdformat"></a>

Directly relevant to why this file exists.

`--compact-tables` was added on evidence rather than preference. **Table padding is charged per column at its widest cell**, and this repository's governance rows run past 2,500 characters, so padded output grew the tracked markdown from **162 kB to 375 kB — all of it trailing spaces**. Compact output is size-neutral at 161 kB. A single very wide cell taxes every other row in its table.

The GFM plugin is required, not optional: core mdformat is CommonMark-only, reads a GFM table as a paragraph, and `--wrap=no` would then collapse every row onto one line.

The content-preservation check — strip whitespace, cell pipes and backslash escapes, compare the remaining characters before and after — **found a real defect**. A roadmap row held an unescaped pipe inside a code span, and a pipe opens a table cell wherever it appears: the row parsed as **eight cells against a six-column header**, so GitHub had been rendering its tail into the wrong columns and silently dropping the overflow. A sweep found no other row in the repository whose cell count disagreed with its header.

### WP-004 — auditing what a wheel *bundles*, not what it declares

<a id="wp-004"></a>

The copyleft audit read what a distribution says about itself — `License`, `License-Expression`, OSI classifiers — and nothing about what it ships. A wheel may vendor a native library under a licence its own metadata never mentions, which is precisely the case the gate exists to catch.

Reading the `License-File` documents too, it found something on the first run:

| package | declares | bundles |
| -- | -- | -- |
| numpy | BSD-3-Clause | libgfortran, libgcc — GPL-3.0-or-later **with GCC Runtime Library Exception**; libquadmath — LGPL-2.1-or-later |
| shapely | BSD-3-Clause | GEOS binaries — LGPL-2.1 |

numpy is a runtime dependency, not a test-only one, and no audit this project had run could ever have seen it.

Three rules keep the result meaningful rather than merely red:

- **Only declaration lines are read, never licence prose.** A bundled copy of the LGPL names the GPL on dozens of its own lines, and a gate that fires on every wheel shipping a licence file teaches its reader to bypass it.
- **A recognized licence exception is permissive by construction**, matched by expression rather than by name, so a future gcc-built dependency needs no entry.
- **The exception is tested before the allowlist, not after.** Written the other way round the allowlist short-circuits first, the exception never runs, and a per-package entry quietly covers whatever that package vendors *next*. Mutation testing is what caught the ordering: with numpy delisted the audit reports libquadmath alone, and with the exception disabled it reports the GPL-3 line — evidence the two mechanisms act independently rather than one masking the other.

A related gate lesson from the same package: the decisions test moved from a hardcoded `D1–D14` to **contiguity plus a floor**. The hardcoded upper bound failed the moment a decision was added, which is a test asking for maintenance rather than reporting a defect; contiguity alone would not notice the last row being deleted, since what remains stays contiguous. The roadmap's `_WP_FLOOR` ratchet is the same pattern.

### WP-005 — goldens that survive a second machine

<a id="wp-005"></a>

The first fixture golden byte-hashed the generated annotation JSON. Those hashes are per-platform stable only — see [Float reproducibility](#cross-float) — so they were replaced with structural metrics: exact integer counts (images, annotations, categories, polygon points) plus bbox area and coordinate sums pinned with a small absolute tolerance.

The split that has held ever since: **integers exact, aggregates to tolerance**. Counts are platform-stable by construction; sums of floats are not. Every golden this project has frozen follows it.

______________________________________________________________________

## Phase 1 — Data pipeline

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

## Phase 2 — Architecture

### WP-023 — the param/FLOP fidelity gate

<a id="wp-023"></a>

The gate passed all five scales within ±2% params and ±5% FLOPs of R1 Table 7 (worst case `l` FLOPs +4.2%) after **exactly two assumption iterations**, both registered rather than tuned quietly: A28 (the original full-conv box stem left the head **7–16× the reference budget**) and A3 (nested `C3k` inner-bottleneck count 1 instead of 2, because at depth 1.0 the `l`/`x` variants overshot with two).

The third change was explicitly **not** an iteration: A29's counting convention (GFLOPs = 2× fvcore MACs, on the deployed NMS-free model; params over the full dual-branch checkpoint) was *determined empirically* by trying the candidates. Trying three conventions to discover which one a table used is measurement; changing the model until it matches a number is fitting. The distinction is recorded because the two are indistinguishable from the outside once the gate is green.

______________________________________________________________________

## Phase 3 — Assignment and losses

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

## Phase 4 — MuSGD

### WP-033 — toy convergence golden

<a id="wp-033"></a>

MuSGD reaches the 0.1 MSE threshold in **47 steps against momentum-SGD's 90** on a fully seeded micro-CNN regression task.

The tolerance is the interesting part: step counts are pinned at **±15**, final losses at 0.005. Cross-platform torch float drift shifts trajectories slightly ([Float reproducibility](#cross-float)), so a tight pin would be a golden about the machine. The 43-step separation is what keeps the *directional* claim decisive inside a tolerance that loose — the golden asserts the ordering it cares about rather than the number it happened to get.

______________________________________________________________________

## Phase 5 — Lightning training loop

### WP-038 — four defaults that only fail off the developer's machine

<a id="wp-038"></a>

Each was found by running somewhere other than a local checkout, and none is visible in a unit test.

**Strict determinism aborts on MPS.** Auto-picked MPS killed a tier run mid-step: `deterministic=True` hits `index_put_with_accumulate_mps`, which has no deterministic MPS kernel. The default is now accelerator-aware — `"warn_only"` where MPS is available, strict `True` on CPU/CUDA — and the shipped configs dropped their pins, since a config value would override the accelerator-aware default.

**Configs were not in the wheel.** They lived at the repo root and never entered the built distribution, so a wheel-only install (Colab) could not run `--config` at all. Moved to package data with a repo-root symlink so documented commands keep working, then given name resolution (`--config det_smoke` resolves onto the installed tree when no local file matches) and a default recipe, because a wheel install has no checkout to write a path into.

**Rich floods notebook output.** Lightning auto-picks `RichProgressBar` whenever `rich` is installed — it is, transitively — and Rich live rendering prints one line per refresh in a Colab cell. A newline flood every step. Default is now `tqdm.auto`, which renders a widget in notebooks and a single-line bar in terminals.

**Loggers.** An unset `trainer.logger` resolved to TensorBoard only, so a run left no plain-text metrics. It now resolves to a TensorBoard + CSV pair pinned to the same `version_N` directory — which is what makes every later `metrics.csv` analysis in this project possible at all.

### WP-039 — a package whose finding is that no code was needed

<a id="wp-039"></a>

Deterministic checkpoint and resume was proved rather than implemented: a four-epoch reference run against a run stopped after two epochs and resumed produces the same per-step loss trajectory over the post-resume epochs (rel 1e-4), with `global_step`, `current_epoch` and every MuSGD momentum buffer restored. **No source changes were needed** — Lightning's checkpointing plus existing module and optimizer state already round-trips.

Two scoping choices made that provable: all runs share `max_epochs`, so the progressive-loss alpha ramp (a pure function of `current_epoch / max_epochs`, recomputed each epoch) is identical either side of the resume; and the data stream is a fixed unshuffled in-memory dataset, keeping sampler-state restoration out of scope. Without the first, a correct resume would still have produced divergent losses.

### WP-040 — overfit-100 on an accelerator

<a id="wp-040"></a>

Train-set recall **0.9747** against the 0.95 floor, MPS, 4:47 wall clock, decoded NMS-free through the deployed path.

The structural decision that outlived it: `goldens/gpu/` is excluded from default harness discovery and recomputed only under `--include-gpu`, which is what keeps the offline gate accelerator-free while still holding accelerator numbers under version control. The cost of that split is that a `gpu/` golden can go stale unnoticed — `shapes_regression_det.json` currently has an outstanding re-freeze.

______________________________________________________________________

## Phase 6 — Evaluation, release 0.1.0

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

### WP-079 — per-worker, per-epoch augmentation RNG

<a id="wp-079"></a>

`_TrainPipeline` seeded one generator in the **parent** process. A parent's generator never advances when workers are used, so every worker replayed one identical augmentation-parameter stream — diversity `1/num_workers`. WP-076 then made workers non-persistent, so the pool restarted from that same state every epoch, replaying a single epoch's parameters for the entire run.

Measured exposure: **1/3200 of the intended draws** at 32 workers over 100 epochs. The two defects compose multiplicatively and neither is visible in a loss curve — the run trains, it simply trains on far less variety than the config describes.

### WP-082 — objective invariant gate

<a id="wp-082"></a>

Written after WP-078 and WP-079 survived 474 tests and three tier runs, which is the fact that justifies the package: the suite was large and did not contain the shape of test that would have caught either.

The useful negative result is the test **not** written. An obvious guard for WP-078's defect is "the L1 term must not exceed some share of the objective". On a converged fixture the broken pixel frame reads **10.7%** against the correct frame's **1.5%** — both comfortably under any threshold anyone would have picked. Such a test would look like protection and catch nothing. The gate pins the coordinate frame directly instead (`1/stride` against the pixel frame, per level) and its scale on a near-converged state (0.75 stride-units against 5.99 broken).

### WP-083 — synthetic-shapes generalization golden

<a id="wp-083"></a>

The cheap stand-in for a tier run: 2000 generated scenes split 1800/200, n-scale, 6 epochs at 320 px, scored on held-out images through both decode paths. Unlike the overfit-100 gate it never scores what it trained on, so it measures generalization rather than loop composition, and mosaic stays on for five of six epochs so a collapsed augmentation RNG ([WP-079](#wp-079)) is detectable rather than merely absent.

Frozen at NMS mAP50-95 **0.8823**, E2E **0.8583**, mAP50 **0.9842** — three MPS runs identical to four decimals, 127–169 s wall clock. The spread in wall clock against the exactness of the metric is the point: it is a cheap gate whose *number* is stable even though its *timing* is not.

### WP-084 — commit-time gates

<a id="wp-084"></a>

Three checks were moved to where they can catch something, and the reasoning generalizes past this repository.

The **commit-trailer validator** was a CI job. Under a squash merge the per-commit messages CI validated are replaced by the squash message — so the job gated text that never lands. It belongs in a `commit-msg` hook or nowhere.

The **copyleft audit** went the other way: it stays a standalone always-run job rather than a diff-triggered one, because it scans the installed environment. A transitive copyleft bump touches no tracked file, so a diff-scoped trigger would never fire on the case it exists for.

The **`dev` extra** became a PEP 735 `[dependency-groups]` entry, which is what stops build/lint/test packages advertising themselves in the published wheel's metadata (`Provides-Extra: None` on the installed dist is the check).

Test-suite and docs-updated checks were deliberately left unhooked — a pre-commit hook that runs the suite trains the operator to pass `--no-verify`, which is worse than not having it.

### WP-085 — doctests join the offline gate

<a id="wp-085"></a>

Every public function is required to carry an `Examples` section, and **no gate had ever run one**: `testpaths` pointed at `tests/` alone and `addopts` carried no `--doctest-modules`. Four examples had rotted unnoticed over roughly eighty work packages.

The worst was not stale — it was never true. `orthogonalize`'s example asserted near-exact orthogonality that the Newton–Schulz coefficients cannot reach, and which that function's own unit test documents as unattainable. It could not have passed at any seed, and was additionally unseeded. It now states the guarantee that actually holds (no singular value expanded). The other three pinned stale float reprs (`0.01` against `0.010000000000000009`; `0.01...` against `0.00999...`).

Wiring it up took the offline suite from **558 to 699 cases** and coverage from 96% to 97%. The lesson recorded for later gates: a requirement nothing executes is a documentation convention, not a gate, and the two are indistinguishable from a green run.

______________________________________________________________________

## Phase 7 — Instance segmentation, release 0.2.0

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

## Phase 8 — Oriented detection, release 0.3.0

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

### WP-105 — a report that survives

<a id="wp-105"></a>

Two defects the first real oriented evaluation exposed, both in the CLI rather than in what it measures.

`--output` into a directory that did not exist yet raised **after** the scoring pass finished and printed its numbers — minutes of accelerator work producing a traceback and nothing else. The parent is now created rather than required, because the file is the only durable form of the expensive part. The general rule: a long computation's output path should be validated or created *before* the computation, never discovered after it.

Second, a 10,132-tile pass showed one line at the start and nothing until it ended. The run has a known length; it now draws a progress bar, wrapped at the CLI rather than inside the evaluator, which tests drive in-process.

______________________________________________________________________

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

### WP-094 — the difficult flag died at the loader

<a id="wp-094"></a>

[WP-057](#wp-057) manufactures difficult instances at crop time and A51 defines the channel they are supposed to travel on, but nothing connected the two: the reader did not forward `difficult` out of the annotation record, so every instance the tiler had flagged arrived at the loss indistinguishable from a whole one. Nothing disagreed anywhere — the counts are right, the boxes are right, and the only thing wrong is that a protocol decision the register spent a row on was not in force.

A53's non-schema keys are what make the flag survivable through a COCO container at all, since COCO's schema has no field for it or for the tile's window provenance, and a standard reader ignores both and still sees an ordinary detection set.

### WP-095 — a run monitor is not an acceptance instrument

<a id="wp-095"></a>

The oriented tier reached its release row with no way to produce an acceptance number. What existed was the epoch metric `val/rotated_mAP` ([WP-102](#wp-102)), and that is a run monitor: it scores whatever the training loader hands it, under torchmetrics' own protocol rather than A46's recall grid, A47's cap and A48's difficult rule. Two figures computed under two protocols are not the same measurement, and the one a release quotes has to be the registered one.

One decision inside the instrument is worth recording. It scores in **letterbox coordinates, with the ground truth letterboxed alongside**, rather than inverting the predictions back to tile coordinates first. Both sides then pass through one geometry, which makes the comparison an equality rather than a proxy — the inverse is where a frame error would enter, and here it is not on the path at all. The same reasoning put checkpoint loading in `lucid_yolo.eval.checkpoint` instead of a second copy: one loader, one place for a task to be read wrong.

### WP-096 — a rename with a published cost

<a id="wp-096"></a>

Parsing moved to jsonargparse so that flags and help text derive from the **operation signatures** rather than from a hand-maintained parser that is free to disagree with the function it calls. The cost is real and is stated rather than absorbed: jsonargparse spells a parameter with underscores, so every documented flag changed shape in one commit.

`lucid-download` therefore survives as a deprecated alias rather than being dropped. It is named in published reproduction instructions, and an instruction that no longer runs is a worse outcome than a duplicate entry point — the alias keeps accepting its **old** flags, which is the part a copy-pasted command depends on.

### WP-097 — a published total no correct download can reach

<a id="wp-097"></a>

R18's headline figures — 2,806 images, 188,282 instances, 15 categories — describe DOTA-v1.0 **whole**, and R18 releases ground truth for two of its three splits. An annotated root therefore holds about two thirds of that instance count, so `check_dota_root` comparing its summed train+val totals against those numbers failed on every correct download, at the first command an operator is told to run.

The fix is not a looser tolerance but a different question. A count the caller did not state is **reported**, not compared: on an unstated expectation the check's job is to say what it found, and a number with no stated referent cannot fail. What still fails is a stated expectation that misses — and, new here, a DOTA expectation aimed at a COCO root, which had previously been accepted and silently ignored.

Recorded because the failure was in the gate rather than in the data, and a gate that fires on every correct input is the one shape of gate that teaches its operator to disbelieve gates.

### WP-064 — shipping a figure that compares to nothing

<a id="wp-064"></a>

0.3.0's oriented numbers are **per tile**. R1 and R18 report whole-image DOTA figures, so nothing this release published is comparable to anything outside it, and the release's own disclosure is what keeps that gap from reading as an omission: the reproduction report, the oriented model card and the README each say so.

The row also owned the whole-image tile merge and did not deliver it. It moved to [WP-107](#wp-107) rather than being marked done — a package that reports itself complete while a clause of its scope is unbuilt is how a roadmap stops describing the code, and this is the third time the merge had slipped, after WP-063 deferred it and WP-088's scope never took it up.

## Phase 9 — Inference and generalization

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

### WP-089 — the grid was already written three times

<a id="wp-089"></a>

The package exists because a letterbox inverse must not be written twice: two copies of one transform, each with its own passing tests, are free to disagree by a pad the day either side's rounding changes. That part went as intended — nothing in the predict path computes a ratio, a pad or a corner.

What the work found is that the *other* geometry had already made the mistake. The head's anchor grid — divide the canvas by the three strides, build the points, move them to the device — was written once in the dual-path evaluator, once in the oriented evaluator, and would have been written a third time here. The stride triple `(8, 16, 32)` had three separate declarations to match, two of them private. Nobody duplicated it carelessly; each consumer needed three lines and wrote them. That is how the duplication a project actively guards against still accumulates: the guarded case is the one everyone can see is dangerous, and the neighbouring case looks too small to bother with.

Consolidated into `assign/grid.py`, beside `make_anchor_points`. The general signature won: a canvas is `(height, width)`, and a square side cannot express a letterbox that is not square. Every golden was unchanged afterwards — 20/20 — which is the only evidence that a refactor of the scoring path changed nothing.

A second finding, recorded because it is a trap the next two packages walk into: `DetectionLitModule.forward` returns a `DualHeadOutput` for **every** task. A segmentation or oriented checkpoint handed to a detection predict path therefore produces boxes, silently, with its mask or angle branch never consulted — plausible output, no error, and the masks the caller asked for simply absent. The refusal is in the library rather than in the command for that reason: a wrong answer that looks right is worse than a traceback, and the command is not the only door.

Two citations that belong with the above rather than in the roadmap row. The duplicated-inverse hazard has a name in this project — it is the **WP-053a defect class**, the shared-decode function whose second caller found what its first never could ([WP-090b](#wp-090b) is the same class caught later). And the task string is read from the checkpoint by the same mechanism `lucid-eval` reads it, so the two commands cannot disagree about what a checkpoint is: a refusal that depended on which entry point asked would be worth less than no refusal at all.

### WP-090 — a crash the evaluator could never reach

<a id="wp-090"></a>

`decode_instance_masks` raises on a zero-length instance axis: `F.interpolate` rejects a `[1, 0, 16, 16]` tensor outright. The function has been in the tree since WP-053a and no test caught it, because from the evaluator it is unreachable — both decoders always hand it a fixed 300 rows, padding included, so the axis is never empty. Predict decodes only the survivors, and an image with nothing above the confidence threshold has none.

The general shape is worth keeping: a function whose only caller pads to a fixed length has never been tested at its degenerate input, and the untested case arrives with the second caller rather than with the first. Guarded at the call site here, with the crash pinned by a test on both decode paths; the shared function itself is left for its own package.

Two things this package had to get right that a shape check cannot see. The masks are assembled while the boxes are still on the letterboxed canvas, because that is the frame the A11 box crop is defined in — cropping after the inverse letterbox leaves every box right and every mask quietly wrong. And the mask path is **not** decoder-independent: the one-to-one coefficients are what the E2E decode reads and the dense ones are what the suppression path reads, so pairing either decoder's rows with the other branch's coefficients yields a plausible mask of a different object. The test plants different prototypes per branch, so a path reading the wrong one returns a mask on the other half of the picture and fails rather than passing with a mask nobody looks at.

For the report, COCO RLE through `faster_coco_eval.mask.encode` — already a declared dependency, and already the encoder every `segm_` statistic is measured through, so a predicted mask on disk and a scored mask are the same object in the same format. Rejected: a polygon contour needs a tracer and is lossy on masks with holes, which the box crop routinely produces; a sidecar `.npz` splits one prediction across two files, so a report can be archived into silently describing masks that are gone; mask-derived scalars answer a smaller question than the caller asked.

### WP-090b — the degenerate input arrives with the second caller

<a id="wp-090b"></a>

`decode_instance_masks` has raised on a zero-length instance axis since WP-053a wrote it. `F.interpolate` treats that axis as channels and rejects an empty one outright. No test caught it and no reviewer would have: from the evaluator the case is unreachable, because both decoders always hand the function a fixed 300 rows with their padding included, so the axis is never empty. Predict decodes only what survives the confidence cut, and an image can have nothing above it.

The rule worth keeping: **a function whose only caller pads to a fixed length has never been exercised at its degenerate input, and that input arrives with the second caller rather than the first.** The shape of the bug is not "someone forgot the empty case" — it is that the empty case did not exist while there was one caller, so no amount of care at the time would have surfaced it.

Returning the empty stack rather than raising is the shape contract, not politeness: a caller assembling per-image results needs a tensor of the right rank and dtype to stack, and an exception pushes the count check out into every such caller, which is where the copies start.

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

### WP-091b — a threshold with nothing to cite

<a id="wp-091b"></a>

The decoder itself is the small half. The rule is greedy class-wise suppression whose overlap is `rotated_iou` — the same exact polygon measure the oriented evaluator grades with — over boxes canonicalized first, so two detections stated a half turn apart cannot read as two objects. Class separation tests the label directly rather than borrowing `batched_nms`'s coordinate-offset trick, because there is no offset that separates rotated boxes without also turning them.

The interesting half is the threshold, and what it cost to find that there is nothing to cite for it. R1 states no suppression parameter at all — the one-to-one branch's freedom from NMS is the paper's claim, not a setting. R18's devkit scores detections it is handed and settles nothing on the decode side. R13 fixes the angle convention and no suppression rule. So the oriented comparison column has a knob no allowlisted source turns, and the register (A61) says exactly that in its source column rather than dressing a convention up as a citation.

What the value rests on instead is holding one variable still: 0.7 is carried over unchanged from the axis-aligned path, so the two comparison columns differ by the overlap *measure* alone and any gap between them is attributable to the geometry rather than to a second knob turned at the same time. The defence that was **rejected** is more interesting than the one taken. It would be natural to argue that 0.7 is safe because rotated IoU is the smaller measure, so carrying the threshold over can only suppress less. That is false: rotated IoU is usually smaller — mean 0.047 against 0.080 over 3000 random canonical pairs — but it *exceeds* the envelope IoU whenever both boxes lie along a shared diagonal, at 138 of those 3000 pairs and by up to +0.146. An argument that sounds like a bound and is only a tendency is worse than no argument, because it stops the next reader from checking.

The exposure is recorded with numbers rather than a hedge: 0.7 under-suppresses elongated objects, which is DOTA's characteristic shape. Two detections of one 10:1 object whose headings disagree by 5 degrees score 0.643 and both survive. That direction was accepted deliberately — a duplicate is inspectable in the output, where an over-suppressed object is simply absent — and the value is a constructor argument, so the alternative is one key away.

Looking for a lineage to inherit turned up that there was none. The axis-aligned `0.7` had shipped since WP-042 as a bare default with no register row, which means every "mAP (non-E2E)" number this project has published rested on an unregistered constant. It is now **A62**, registered at its shipped value rather than revised: re-picking it would silently move a comparison column that already has committed figures beside it, and that is a 0.MINOR change with goldens, not a tidy-up. The general shape: an assumption that goes looking for its own parentage is a good way to find the rows nobody wrote.

### WP-099 — a format with no specification

<a id="wp-099"></a>

The clean-room rule says the reader is written from the format as the datasets publish it, never from an implementation's reader. That is a clear instruction with a hole in the middle: the YOLO label format has no specification on the paper allowlist, and the obvious ways to learn it — a search, a docs page, somebody's loader — are all the thing the rule forbids. What closed it was a published dataset export already on the machine, CC BY 4.0, a data artifact with no implementation lineage: registered as R32, and read for what it actually contains rather than what a tutorial says it contains.

That got four facts and left four gaps, and the gaps are the useful part. The export's fields lie in `(0, 1]` and its clipped corners land on exactly `0` and `1`; `names` is an index-ordered list beside `nc`; split entries are written `../train/images` for a tree at `<root>/train/images`; every image has a label file and none is empty. What it could not settle: whether a ten-field oriented row exists (R18 appends a `difficult` flag to its own label lines, and the normalized variant has no published spelling for one), whether `data.yaml` may carry a `path:` key, which of the two directory spellings ranks first when a root satisfies both, and which image extensions are in scope. Each of those is rejected or documented as a precedence choice rather than guessed — A54 through A58 — because a guess that silently mis-parses is exactly the failure the clean-room clause exists to prevent, and it fails quietly by producing a dataset rather than an error.

Two decisions where the format's own limits set the line. **A missing label file is an error; an empty one is a background image** — writing zero bytes is a positive statement that the image holds no objects, while not writing one is silence, and a tree with no manifest cannot tell silence from a half-finished download. Reading silence as "no objects" turns a broken export into a dataset that trains on backgrounds without ever saying so. **A coordinate outside `[0, 1]` is rejected, not clamped, and compared with no epsilon** — it is not a slightly-off box but a file that was never normalized, typically written in pixels, and clamping turns it into a wall of degenerate edge boxes that would then be trained on.

The layout question was the one that looked smallest and was not. `resolve_split` requires a directory **and** an annotation file, and a YOLO split's annotations are a directory. Folding YOLO into the existing `CANDIDATES` would force that predicate to accept either kind of thing, and a root satisfying both conventions would then resolve by table order — handing a labels tree to the COCO reader, or an `instances_*.json` to this one. Which convention applies is a property of the reader asking, so the reader asks its own table.

What did not land: the datamodule still does not dispatch between the two readers, so a YOLO root is reachable from a library call and not yet from `lucid-yolo fit`. Recorded as 099b rather than absorbed into the row, because a package that reports itself done while a clause of its scope is unbuilt is how a roadmap stops describing the code.

Three details of the shipped surface, kept out of the roadmap row for length. The reader produces the same `Targets` container the COCO path does, which is the property that leaves assignment, loss and metric code untouched by a second dataset format entirely — the format question stops at the loader. Class names and split paths come from the dataset's own `data.yaml` rather than from a config, so a YOLO root carries its own label space. And a layout row was added alongside, so `--data.data_root` resolves a YOLO root the way it resolves every other, through `resolve_yolo_split` and `from_root`.

The clean-room instruction this was written under is AGENTS.md's prime directive, cited here because the register rows (A54-A58) record what was *unknown* and not what forbade looking it up.

### WP-091c — what a refactor is allowed to notice

<a id="wp-091c"></a>

The move itself is unremarkable: `rotated_iou` and the seven private helpers it exclusively owns leave `eval/dota_eval.py` for `data/rotated_geom.py`, since an evaluation module had no business being on the decode path's import graph for a function that is pure geometry. What the package is worth recording for is the three things it declined to do.

**It did not merge the duplicate it found.** `rotated_geom` already had `_check_2d`, differing from the arriving `_check_rboxes` only in the letter naming the row count — and the two raise `must be (N, 5)` and `must be (K, 5)`, which two different test files match on. Merging them is therefore a test-visible behaviour change wearing the costume of a duplicate removal, and a refactor whose contract is "every existing test passes unchanged" cannot also change what an error says. It is recorded as 091d instead. A duplicate that is cheap to see is not always cheap to remove, and the difference is whether anything asserts on it.

The search also turned up `_check_rboxes` in `losses/probiou.py`, first reported as a third copy and corrected on re-reading: it guards `ndim == 0 or shape[-1] != 5` and raises `must be (..., 5)`, so it accepts arbitrary leading dimensions where both `rotated_geom` validators require exactly 2-D. Same name, same five, different contract — merging it into the other two would tighten what the loss path accepts, which is a behaviour change and not a rename. It is 091e, and its test coverage differs in the same direction: `tests/losses/test_probiou.py` matches the prefix `target must be` only, leaving the tail unpinned, where both `rotated_geom` messages are pinned tail and all. Three lookalikes, two of them the same check — the way to tell was to read each guard rather than each name.

**It did not deprecate the old name.** `pyDeprecate` is nowhere in this tree, and `warnings.warn` would have fired on both existing suites — which is not "tests pass unchanged" under any honest reading, and arms a trap for the day `filterwarnings = error` is set. The deeper reason is that `dota_eval.rotated_iou` is not a legacy alias: that module's docstring defines R18's protocol *in terms of* its overlap measure, so a caller reproducing the protocol wants the kernel where the protocol is. One function visible from two places is what `canonicalize` already is.

**It moved the argument, not just the code.** The "why there is no float64 here" section and its five-row error table went with the kernel, because that table is the *evidence* for the shift-before-expand ordering rather than commentary about it; a precision argument left behind in the module that no longer owns the arithmetic is how a constraint quietly stops being checked. Two claims in the destination's own docstring also had to change or become false — "no Python loop over boxes" and a blanket "float32 tensors" — which is the ordinary tax of moving code into prose written before it arrived.

The evidence that the move was inert is the goldens and only the goldens. `rotated_iou` sits on the scoring path, and a silently perturbed kernel would still clear the shapely oracle's 1e-4 tolerance; 20/20 unmoved, frozen subtrees included, is what says the arithmetic is bit-identical.

WP-091b had already verified there was no import cycle, from four entry orders, and left the module where it was rather than widening its own diff — which is why this is a package and not a line in that one.

### WP-099b — the tie-break that must not exist

<a id="wp-099b"></a>

`resolve_split` already had a precedence rule: candidate spellings are tried in table order and the first match wins. Reusing that shape across the two *readers* is the obvious move and it is wrong, for a reason worth stating in one line — **within a table the loser is another spelling of one reader, and between the tables the loser is a different label space, a different image set and a different parser.** A first-match rule in the first case picks the wrong directory of the right dataset; in the second it trains on annotations nobody named and reports a plausible number while doing it. So the probe evaluates both tables in full and returns a pair of booleans: there is no order for the answer to depend on, and ambiguity is refused rather than resolved (A63).

The refusal only works if the operator has a way to answer it, which is what earned the explicit `layout` override its place. Two states are otherwise unreachable: an ambiguous root on a read-only dataset mount, where the only remedy would be moving files; and a YOLO root whose `data.yaml` points its splits outside the convention table, which `from_root` resolves — it is why `val: ../valid/images` works at all — and which no directory probe can see. A knob that exists because two legitimate datasets are otherwise unreadable is a different thing from a knob that exists because one was possible.

The construction-versus-`setup` split was forced by something outside this package. Refusing at construction is the better instinct and it collides with `tests/ptl/test_cli.py`, which instantiates every shipped config through `LightningCLI(run=False)` while `configs/det_smoke.yaml` carries a placeholder `data_root`. A filesystem verdict at construction fails that dry-parse for every run not yet pointed at data. The line drawn instead: everything decidable from the arguments alone raises at construction, and anything needing the disk raises at `setup()` — still before an image is decoded and before the trainer takes a step.

Two absences the YOLO path has to declare rather than discover. `mask_targets` is refused, because the format carries no per-instance rings and rasterising empty ones would produce a segmentation run supervised by nothing that still reports a detection number. Copy-paste is the quieter one: unlike the oriented path it does **not** raise on polygon-free targets — it decodes an entire extra source sample, finds no candidates and does nothing — so at `copy_paste = 0.1` every YOLO run would have paid for an augmentation that can never fire. The suppression keeps consuming the RNG draw, which is what leaves the COCO path byte-identical and the frozen data checksums unmoved. An augmentation that fails silently costs more than one that raises.

One more property of the override, which is what keeps it from costing anything: naming any COCO-shaped path — `train_images`, `val_annotations`, any of them — is itself read as stating the COCO layout. Every caller that predates the probe is therefore already off it, and the probe runs only for a root that has said nothing about its own shape.

### WP-107 — whole-image merge

<a id="wp-107"></a>

Recorded before the work, because the reason this is a package rather than a step inside another one is itself the finding: two overlapping 1024 px tiles that both detect one object produce two detections at full confidence, and an NMS-free path has **no suppression stage to remove the duplicate**. The merge is a policy that has to be decided and argued, which is precisely why it kept failing to happen as a side effect — WP-063 deferred it into WP-088, whose scope never took it up, and WP-064 shipped 0.3.0 without it.

What the work established is that the duplicate rule does not have to be a suppression rule. Core ownership (A59) keeps or drops a detection by **where it is**: the cores are built from the recorded windows before the model runs, nothing is compared against another detection or ranked by confidence, and a detection is dropped whether or not the neighbouring tile found anything. Run the model twice with different weights and the same detection is owned by the same tile. A confidence-ranked dedup by rotated IoU is the defensible alternative and would score *higher*, because it takes the union of the tiles' recalls rather than the owner's — it was rejected because the figure is meant to measure a suppression-free pipeline, and reintroducing suppression at the seam makes it measure something else. That rejection is the package's actual content; the code is the cheap part.

The cost is real and one-directional: whole-image recall is the owner's recall, so this merge can only *lower* a per-tile number. It is pinned by a test rather than left as prose, so that a later reader who finds the drop surprising cannot quietly fix it into NMS.

One residual runs the other way and was nearly missed. The first draft of the module claimed the difficult straggler "cannot inflate the score". That is wrong in exactly the direction this package exists to police: a detection landing on a surviving clipped copy is *discarded* under A48, where against the true whole-image annotation set it would have been a false positive, and discarding false positives raises precision. It is bounded to objects wider than the overlap and it is R18's own devkit behaviour rather than an invention here — but it is not zero, and it is now measured at **0.165 of `map_50`** on the unit fixture rather than argued away. A claim of the form "cannot" about one's own instrument deserves a measurement before it is written down.

The DoD's equality clause needed one guard the row did not ask for. A single-tile image must score identically through both paths, and an equality between two saturated `1.0`s would prove nothing at all — so the fixture is built to land every one of the four metric keys strictly inside `(0, 1)`, asserted separately, and the equality is then asserted with no tolerance.

Two smaller things the merge inherits rather than solves. `--limit` can end a tile prefix mid-source-image, so an incomplete trailing image is dropped rather than scored partially — a whole-image figure under `--limit N` therefore covers at most N tiles' worth of *complete* images. And A52's `--drop-empty-tiles` moves the core midpoints when windows are missing from the layout: still a partition, but the `v/2` circumradius guarantee weakens, which is a property of that off-protocol layout rather than of this rule.

### WP-099c — a pre-flight that asks the run's question

<a id="wp-099c"></a>

`check_dataset` defaulted to `coco`, which was correct for as long as COCO was the only layout a run could read. WP-099b ended that, and left the check answering a question its own run no longer asks: pointed at a YOLO root with no flag, it reported a missing `train2017` for a tree `fit` trains on without complaint. The fix is not a third branch but a deletion — an unstated `--dataset` now runs `detect_layout`, the same probe the datamodule dispatches on, so the pre-flight and the run resolve the root identically or not at all. `DATASETS` is built from `DatasetLayout` members rather than string literals, which is what stops the flag and the probe drifting apart later. DOTA stays outside it deliberately: no run reads a DOTA root — it is tiled to COCO first (WP-094) — so a probe covering it would advertise a path that does not exist, and the "satisfies no convention" report ends by naming `--dataset dota` for the operator who has one.

Both of the probe's undecidable states are verdicts about a disk, so both come back as report lines and exit 1. WP-097 already paid for that lesson once, on the first command an operator runs against a fresh provisioning: a traceback where a report belongs reads as a broken tool rather than an unusable dataset.

The extraction that mattered is `resolve_split_dirs`. A validator that resolves splits its own way validates a tree the reader will not open — and the YOLO layout has exactly the case that makes this bite, a `data.yaml` naming `../valid/images` for a directory no convention row spells (A58). Shared code between a feature and its checker is usually a smell, since the checker then inherits the feature's bugs; here it is the point, because the alternative is a check that passes on a different dataset than the one that trains. `scan_yolo_label_file` is the other half: the same grammar `load_yolo_targets` reads, stopping before the denormalization, so checking a split does not decode every image to obtain a pixel scale it immediately discards. Validation that costs an epoch is validation that gets skipped.

`DotaSplitCheck` became `PairedSplitCheck`, and the naming records what the two layouts actually share. It is not that both are "not COCO" — it is that both put one text file per image beside it, paired by stem, so both need the same two-directional pairing report and the same totals arithmetic. COCO's single manifest cannot be checked that way at all. The shape of the annotation decides the container, and the rename says so where a `Yolo`-prefixed twin would have said nothing.

No register row. A64 is still unconsumed, which is worth stating explicitly because the run made three decisions that look like assumption material and are not: the inference default is grounded in A63 and WP-097, the YOLO branch reuses A54–A58 through the reader rather than restating them, and the flag guard extends WP-097's own rule to one more layout. A register row for a decision that already has a documented parent is noise in the register that matters.

One absence left standing: `splits` is still not a CLI flag, for YOLO as it never was for DOTA, so a root shipping only `train` fails on the missing `val`. That is right for a `fit` pre-flight, which needs both, and wrong for anyone checking a partial download — a small follow-up rather than a defect, and recorded here so it is a choice rather than an oversight.

### WP-099d — the code is the specification

<a id="wp-099d"></a>

The gap this closed: `docs/DATASETS.md` said where COCO and DOTA come from and nothing about what either layout looks like on disk or what one annotation contains, while the YOLO layout the project had just started supporting appeared in it nowhere at all.

`docs/DATASETS.md` also carries a standing rule that a layout reference is the first real test of: the code is the specification and the document follows it. Applied here, that means every claim about a directory convention is cited to the module that enforces it — `data/layout.py`, `data/yolo.py`, `data/coco.py`, `data/dota.py` — rather than described from the outside from what the layouts are believed to be.

The alternative is not a worse document but a second specification. Two independent statements of a convention agree on the day they are written and disagree the first time a candidate table gains a row, and the one a reader trusts is the one that cannot be executed.

### WP-108 — a cell grows where its section is missing

<a id="wp-108"></a>

The Scope column had drifted to a 286-character median with a 1709-character worst case, and the drift is not random: of the 32 rows over 400, **12 had no RESEARCH_LOG section at all**. A row whose package never got a section is a row where the only place to write down why a thing was done is the row itself, so the reasoning lands there and stays. The length is the symptom; the missing section is the cause, which is why this package wrote the 12 sections before compressing anything.

The gate was considered and rejected, and the reasoning is worth keeping because it runs against this project's usual instinct. Every other convention here is enforced by a meta test — status icons, six columns, the header's package count, the register ids — and the argument for those is that a documented rule with no gate decays. A length gate is different in kind: the others fail on a fact about the file, while a length gate fails on a judgement about how much a row needs to say, and the cheapest way to make it pass is to delete a claim. A rule whose easiest satisfaction is losing substance should not be automatic. So the median is written into the header as guidance, and the enforcement is a reader noticing that a cell runs several times its column's median.

What the compression turned up is that most of the overflow was already recorded somewhere with an owner. Twenty rows needed no new writing at all: their long clauses restate what the log section, or the assumption register, already carries — A40's exact-under-similarity split, A59's core-ownership rule, A24's polygon measure. That is the duplication the header paragraph has always warned about, visible for the first time because something counted it. Three claims had no other home and were moved rather than cut, the sharpest being WP-064's: that the release discloses its per-tile figures in the report, the model card and the README. A disclosure claim living only in a roadmap cell is one edit from disappearing.

Two rows deliberately got no section: 091d and 091e were ⬜, and their content was a finding of 091c's, recorded there. A log section for unstarted work would be a stub asserting what the package will conclude, which is the opposite of what this file is for — so both rows pointed their `· log` tail at `#wp-091c` instead, and the sections got written when the packages produced something to record, which is the paragraph below this one.

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

### WP-099e — the gap was one layer above where it looked

<a id="wp-099e"></a>

WP-099c closed by naming this itself: `splits` was still not a CLI flag, so a root shipping only `train` fails on a missing `val`. That was recorded so it would be a choice rather than an oversight, and this is the follow-up. A third-party export whose validation split was never cut is a correct export, and the first command an operator runs against a fresh provisioning refused it over a directory nobody had promised.

What is worth writing down is where the parameter was actually missing. `check_yolo_root` and `check_dota_root` had both taken `splits` from the day each was written — `tests/data/test_check.py` had been calling `check_yolo_root(..., splits=("train",))` all along, because the fixtures only ever build one split. The capability existed; the route from the command line to it did not. So the fix is not a feature the layout checkers lacked but one argument `check_dataset` and `_check_root` dropped on the floor. A gap that presents as missing functionality is worth pricing against the layer below before implementing: the two root checkers were the expensive place to fix this, and the wrong one.

The default belongs to the layout rather than to the call. `_check_root` names `YOLO_SPLITS` or `DOTA_SPLITS` at the branch that already knows which layout it is dispatching to, instead of defaulting `splits` in `check_dataset`'s signature. The two constants are equal today and are documented for different reasons — DOTA's testing third has no public labels, YOLO's `test:` entry is a split no run of this project opens — and an entry-point default would have quietly made them one fact, so that changing either would have to be discovered rather than read.

COCO refuses the flag rather than defaulting it, on the reasoning that already governs `--expected_images` there: `check_coco_root` checks `train2017` and `val2017` against 118,287 and 5,000, and the names and the numbers are a single published fact. There is no subset of that to ask for. Following `_check_flags_apply`'s existing message shape was the whole of the design decision — a flag that silently does nothing makes the report read as though it had been honoured.

The empty tuple is the sharper case, because it inverts the failure mode the rest of the module guards. Every other bad argument here fails loudly; `splits=()` fails *cleanly* — no split checked means no problem found means `PASS: dataset layout valid` printed over a root nothing looked at, a false green from the one command whose purpose is to be believed. It is refused in `_require_splits`, called from `check_dataset` before the layout is even inferred and from both root checkers, because those two are documented as the importable core: guarding only the entry point would leave the false green reachable by the shorter route, and a library caller reading `ok` off the result has no report to notice the emptiness in. The entry-point call is what makes `--splits '[]'` fail at the command line, which is where an empty list is easiest to type by accident.

The annotation is `tuple[str, ...] | None`, and the CLI surface it produces was verified rather than assumed: jsonargparse renders it as a list literal, `--splits '[train]'`, which is the spelling `lucid-data download --splits` has always required — a bare `--splits train` is refused by both. One flag's syntax is not a second thing to learn. The `--help` line for it is less self-describing than `download`'s, the `| None` union suppressing the `[ITEM,...]` metavar; cosmetic, and not worth changing the annotation for.
