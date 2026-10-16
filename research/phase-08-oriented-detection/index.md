# 🔄 Phase 8 — Oriented detection, v0.3.0


## WP-055 — rotated geometry primitives

<a id="wp-055"></a>

Canonicalization is exactly idempotent rather than idempotent to tolerance, and the angle range is a guarantee rather than a near-certainty. The edge case that forces the final clamp: for an angle within an ulp of a range bound, shifting by multiples of pi cannot rescue it — the remainder rounds to pi and the shift returns its own input. A similarity round trip of a box at exactly −45° reaches this in practice, so a clamp closes it at a cost of one ulp.

The square tie-break keys on exact `w == h`, which means a quad reconstructed from DOTA's eight coordinates can return either pi/2-separated representative of the same rectangle. [WP-060](https://borda.github.io/lucid-YOLO/work_packages/wp-060/)'s angle loss, aimed precisely at near-square objects, must therefore not assume which — and that is why it is tested for bit-identical scoring of both representatives rather than agreement to seven digits.



## WP-058 — rotated-aware augmentation

<a id="wp-058"></a>

The three policy choices are A40. What belongs here is the **live defect it fixed**: mirroring negated `theta` without re-wrapping, so any box above 45° left the long-edge range. Nothing had ever hit it because no rotated box could reach a flip through a real pipeline — the augmentations raised `NotImplementedError` on non-empty `rboxes` until this package removed the guard.

A defect that is unreachable is still a defect, and it becomes reachable in the same commit that removes the guard. The general shape: when lifting a `NotImplementedError`, the code behind it has never executed and has never been tested, whatever its coverage says.



## WP-059 — ProbIoU

<a id="wp-059"></a>

R17's formulas are *evaluated* through cancellation-free identities (2×2 adjugate linearity for `B1`'s numerator, the `det(S1+S2)` expansion, `det S = w²h²/144`), and that is what makes float32 sufficient. The measurement that forced it: the **literal transcription loses 377% of `B_D` at a 1e-4 perturbation and returns `NaN` on parts of a 1000:1 aspect sweep**. Float64 internals are not an available fallback, because a regression loss runs on the training device and D12c puts training on MPS, which has no float64.

Both of R17's losses ship side by side — bounded Hellinger `L1` and unbounded `L2 = B_D` — because R17 suggests starting on the second and switching. Which one the oriented path uses, and the measured deficit of the alternative, is A49.

A DoD was **moved** here rather than met: the package originally owed a 1e-4 shapely agreement test. ProbIoU is a Hellinger-distance similarity between two Gaussians and R17 asserts no equality with polygon overlap, so that test could not pass and a correlation threshold would have been invented. The shapely oracle belongs to [WP-063](https://borda.github.io/lucid-YOLO/work_packages/wp-063/)'s exact rotated IoU (A24); ProbIoU got its own exact oracles instead (quadrature of the definition at 2e-14, literal float64 transcription at 1e-12). Recorded because "the DoD was wrong" is a legitimate outcome and the alternative — a tolerance chosen until it passes — is the failure this project is most exposed to.



## WP-060 — square-object angle loss

<a id="wp-060"></a>

R1 Table 11's own numbers set `lambda = 3`: **50.2 mAP against 49.0 with no angle loss, and 47.1 at `lambda = 5`** — i.e. the paper's own ablation shows the term can be worse than omitting it when mis-weighted. That is the context in which [WP-093](https://borda.github.io/lucid-YOLO/work_packages/wp-093/#wp-093) later found the gain A22 assumed was destabilising.

Two floating-point contracts rather than approximations: Eq. 14's `round` tie rule is unstated, so the wrap settles ties upwards into the half-open `[-pi/2, pi/2)` and is exactly idempotent; and the residual is folded once more into `[-pi/4, pi/4)`, a **Sterbenz-exact** shift under which the two representatives [WP-055](https://borda.github.io/lucid-YOLO/work_packages/wp-055/) leaves a square score bit-identically.



## WP-061 — rotated containment for TAL and STAL

<a id="wp-061"></a>

The axis-aligned path is an untouched early-return branch below the new one, and that was verified rather than assumed: assignments were snapshotted at the parent commit on a scene whose anchors sit **exactly on ground-truth boundaries**, then compared field by field with `torch.equal` — 20 tensors, 6,144 elements, bit-identical. "Its tests still pass" would not have distinguished a boundary case that moved by a rounding step.



## WP-088 — oriented training path

<a id="wp-088"></a>

This row was **left open with its criterion unmet**, which is the part worth recording. The training path verified clean — gradient reaches both branches' angle stems, each oriented gain has leverage, the detection step reproduces a snapshot replayed from a clean `fcf3040` export — and A49/A50 were measured rather than argued (their alternatives cost 0.5046 and 0.7112 rotated mAP50). But the overfit floor cleared on **0 of 3 seeds, mean 0.840, spread 0.091**.

The package stayed open rather than being closed on "the code is right", because the code being right was not what the DoD asked. It closed only when [WP-093](https://borda.github.io/lucid-YOLO/work_packages/wp-093/#wp-093)'s revised gain took the gate to 0.9390 at the recipe's own seed.

The criterion is met as written and not more than that: the five-seed picture behind that number is 4 of 5, which lives in [WP-093](https://borda.github.io/lucid-YOLO/work_packages/wp-093/)'s record rather than being folded into this one.



## WP-100 — parallel tiling

<a id="wp-100"></a>

A full DOTA build is **1,869 multi-megapixel PNG decodes**, each cropped into dozens of tiles and written back, and it ran silently on one core for about an hour — an operator had no way to tell a working build from a hung one.

Two constraints shaped the parallel form. Each worker pins itself to a **single torch thread**, since a pool of processes each fanning out over every core is oversubscription rather than parallelism. And ids are deliberately **not** assigned in the workers: a worker cannot know how many tiles preceded it, so numbering there would make the output JSON depend on which process finished first. The parent numbers every record in source order, and the test that protects this compares a four-image pooled build's JSON byte-for-byte against the serial one.



## WP-101 — JPEG tiles and a completion-order bar

<a id="wp-101"></a>

Two costs the first pooled build exposed.

**Decode cost at training time.** The tier decodes every tile once per epoch and mosaic assembles four source tiles per training sample, so a 64-image step at 1024 px is around **250 PNG decodes** — the loader, not the GPU, is what a smoke run waits on. `--suffix .jpg` writes through the JPEG encoder; PNG stays the default, because a tile is an exact crop of a lossless source and a build should not discard information on an operator's behalf. The choice is made once and collected every epoch.

**A bar that measures the wrong thing.** DOTA source images differ by an order of magnitude in size, and an ordered `map` left the bar frozen while one multi-thousand-pixel image held up results already finished behind it. Ordering is a requirement of the *numbering*, not of the *counting*: results are collected by index and replayed in order once the pool drains, and the bar counts completions.

Annotations are asserted byte-identical across the two encoders — nothing about a record passes through the encoder, so a suffix that ever reached the geometry is a bug the pixel difference would hide.



## WP-102 — validation that is not CPU-bound

<a id="wp-102"></a>

Validation left the GPU idle for most of its wall time, from two defaults nobody had revisited at 1024 px.

The val worker cap of 4 ([WP-073](https://borda.github.io/lucid-YOLO/work_packages/wp-073/#wp-073)) applied even when the operator had named `--data.num_workers 16`. A number an operator states says how much of the machine the run may use, and quietly validating on a quarter of it surfaces as an idle accelerator with **no message at all** — worse than the host-OOM the cap was avoiding, because it is silent. The cap now applies only to a count this class chose itself, and scales with pixel count rather than sitting at a constant reasoned about 640 px samples, where val was "a fraction of the train pipeline's work". At 1024 px the decode *is* the work.

Second, an `obb` run was accumulating the axis-aligned `val/mAP` beside the rotated one — a CPU pass over every detection of every batch, to produce a figure that reads the A44 composition's **pre-rotation** rectangle. It cannot separate a run whose orientations were right from one whose orientations were random. An oriented run now logs `val/rotated_mAP` and no `val/mAP`.



## WP-103 — a val worker count that fits the host

<a id="wp-103"></a>

**This package exists to fix a defect the previous one introduced.** [WP-102](https://borda.github.io/lucid-YOLO/work_packages/wp-102/) let a named `--data.num_workers` reach the validation loader and deleted the guard rather than the silence.

The arithmetic it deleted: at 1024 px a batch of 64 stacked float32 RGB images is `4 × 3 × 64 × 1024²` = **805 MB**. A named 32 workers at prefetch 2 queues about **51 GB for validation alone**, beside a training pool still resident at the epoch boundary. The result was the observed host-RAM crash on the first oriented tier run.

The division that was wrong the first time and is now explicit: an operator states parallelism with that flag; **the gigabytes it costs are this class's arithmetic**. A count inherited from a named train count is bounded by the memory budget and warns with the number and the override flag; a count stated as `--data.val_num_workers` is bounded by nothing, which is where an operator asks for the whole machine.

The general lesson, recorded because it was a real mistake and not a hypothetical: the principle behind [WP-102](https://borda.github.io/lucid-YOLO/work_packages/wp-102/) was right — a stated number should not be silently reduced — and the implementation deleted the wrong thing. Removing the *silence* was the fix; removing the *bound* was not.



## WP-104 — one-pass IoU threshold matching

<a id="wp-104"></a>

The oriented epoch boundary stalled for about a minute after validation reached 100%. Diagnosed before anything was changed: `evaluate_rotated_map` scoring ~10k val tiles at 300 detections each, **178 s at that scale, 58% of it in `_match_image`**.

The ten IoU thresholds share the overlap matrix and the score order and differ only in what counts as a hit, so ten passes re-walked the same detections to consume different subsets of the same ground truths. The availability state becomes `(T, G)` and the walk happens once. A second win falls out: detections reaching no ground truth at the lowest threshold skip the walk entirely — they cannot score, cannot be discarded and consume nothing at any threshold — which on an early-epoch model is nearly the whole loop.

**178 s → 38.6 s**, every pinned value unchanged.

**Rejected: a circumradius prefilter on the IoU kernel.** Written, measured at **12%**, reverted. The residual cost is per-call overhead across ~150k small calls, not pair count, so a cheaper rejection test attacks the wrong term. Carrying an extra geometric shortcut in hand-written rotated geometry is not worth 12% — the code it would live in is the code whose correctness the whole oriented evaluation rests on.



## WP-056 — one mask filters every modality

<a id="wp-056"></a>

What the DOTA parser guarantees beyond its shapes is an **alignment**: `rboxes[i]` is the same instance as `boxes[i]` and `labels[i]`, for every `i`. That is why it is stated as an invariant rather than left as an implementation detail — one boolean mask filters every modality at once, so a caller dropping difficult instances, or [WP-057](https://borda.github.io/lucid-YOLO/work_packages/wp-057/#wp-057) dropping instances a window lost, writes the filter once instead of three that are free to drift apart. Every stage of the oriented path downstream of the loader relies on it.



## WP-057 — tiling is what creates difficult instances

<a id="wp-057"></a>

DOTA ships a per-object `difficult` flag and [WP-056](https://borda.github.io/lucid-YOLO/work_packages/wp-056/#wp-056) declined to decide its fate at load, deferring to A39. Tiling is where that deferral turns load-bearing: R18's partial-object rule flags an instance difficult when its clipped area falls below 0.7 of its original, so **the tiler manufactures difficult instances the source annotations never carried**. A39 therefore governs a population that DOTA's own flag column does not describe, and the number the decision was made from has to survive the crop — which is why `U` is carried out as `visible_fraction` rather than consumed and discarded.



## WP-062 — the fidelity gate picked the stem width

<a id="wp-062"></a>

A20 records that R1 gives the angle branch no width: sec. 3.4.3 says only "separate branch", Fig. S2 draws one stem shape and no numbers, and Table S11 reports whole-model sizes a branch width can only be inferred back out of. This is the package where that inference was made, and it was made by the gate rather than by argument — the Table S11 comparison at 1024 px over 15 classes selected the **wider `channels // 2` stem**, and A20 was revised to it.

The margins say how much room the choice had: params bind at scale `s`, +3.27% against a ±3.5% band, and FLOPs at scale `n`, +4.84% against ±5%.



## WP-063 — four constants a library would otherwise have chosen

<a id="wp-063"></a>

R1 reports rotated mAP50-95 on DOTA-v1.0 val and defines none of the machinery that produces the number — not the IoU, the matching rule, the recall interpolation, the detection cap nor the class averaging (A24). Four constants therefore had to come from somewhere, and what is recorded here is that they come from the register rather than from whichever library the accumulator was built on: the 101-point recall grid (A46), the 300-detection per-image cap matching what the head emits (A47), the exclusion of classes with no non-difficult ground truth, and R18's difficult rule with an explicit precedence against the argmax match (A48).

A default inherited silently from a dependency is an unregistered assumption wearing a library's name, and it moves a published figure exactly as far as a registered one does. The difference is only that nobody can find it later.



## WP-097 — what R18's published totals actually describe

<a id="wp-097"></a>

R18's headline figures — 2,806 images, 188,282 instances, 15 categories — describe DOTA-v1.0 **whole**, and R18 releases ground truth for two of its three splits. An annotated root therefore holds about two thirds of that instance count, not the published total — the fact `check_dota_root` got backwards, fixed at [wp-097.md#wp-097-engineering](https://borda.github.io/lucid-YOLO/work_packages/wp-097/#wp-097-engineering).



## WP-064 — shipping a figure that compares to nothing

<a id="wp-064"></a>

0.3.0's oriented numbers are **per tile**. R1 and R18 report whole-image DOTA figures, so nothing this release published is comparable to anything outside it, and the release's own disclosure is what keeps that gap from reading as an omission: the reproduction report, the oriented model card and the README each say so.

The row also owned the whole-image tile merge and did not deliver it. It moved to [WP-107](https://borda.github.io/lucid-YOLO/work_packages/wp-107/#wp-107) rather than being marked done — a package that reports itself complete while a clause of its scope is unbuilt is how a roadmap stops describing the code, and this is the third time the merge had slipped, after [WP-063](https://borda.github.io/lucid-YOLO/work_packages/wp-063/) deferred it and [WP-088](https://borda.github.io/lucid-YOLO/work_packages/wp-088/)'s scope never took it up.

