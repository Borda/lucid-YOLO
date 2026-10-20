# WP-200 — replay native CUDA oriented networks

> Engineering log · [Phase 17 — R1 pose-objective parity](https://borda.github.io/lucid-YOLO/engineering/phase-17-pose-objective-parity/)
>
> 🔗 Depends on [WP-193d](https://borda.github.io/lucid-YOLO/work_packages/wp-193d/), [WP-199](https://borda.github.io/lucid-YOLO/work_packages/wp-199/) · Needed by none
>
> ⬅️ [WP-199](https://borda.github.io/lucid-YOLO/work_packages/wp-199/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/)


<a id="wp-200"></a>

## WP-200 — six native outputs, one recorded network

**The oriented head is part of the recording.** The existing `capture_network` option now accepts `task: obb`. Backbone, neck and both head branches run their existing expressions through R43's graphed-callable API. The adapter retains the four dense detection fields and both angle tensors, rebuilding them by name. Assignment, rotated decoding, classification, ProbIoU, rotated L1, angle scoring, schedules, clipping, optimizer, EMA and validation stay in their current loop. No method arithmetic, gain, convention or frozen golden changes.

**The complete native objective owns the gradient contract.** Its empty-target path retains zero gradients to both angle stems. An overridden objective can instead leave those gradients absent, so a changed training consumer, classification loss or oriented helper keeps the complete native call. Accumulation also stays native, as it does for pose: the existing shared recorder has no established accumulation contract. The ordinary network storage, hooks, precision, single-device CUDA and first-order constraints remain unchanged. One image signature is retained per fit; a temporary partial batch stays native and can return to the existing recording. The default stays off.

**The stage was priced before implementation.** Four native CUDA steps occupied a 1.466 s instrumented window, with about 24.1% covered by recorded device activity. Network forward dispatch took 378.113 ms, including 174.956 ms in backbone and neck and 202.463 ms in the head. A separate complete fit with lightweight host timers took 68.263 s: network forwards took 15.943 s, or 23.35%, across 208 training and 32 validation calls. That includes validation, which remains native, and is an upper bound rather than a speed claim. The 21.261 s backward timer includes loss work as well as network work and is not wholly removable. Profiler runs are separate from the final timing comparisons.

**The smaller recording did not clear the investigation's target.** A shared-backbone/neck-only screen preserved model, optimizer, scheduler, EMA and loss bytes, but changed complete-fit time from 68.277 to 63.800 s, a single-pair 6.56% reduction. It was set aside before the final cohort. The complete six-output screen preserved those same states and validation values and changed 68.742 to 59.853 s, a preliminary 12.93%; the repeated cohort, rather than that screen, determines the final claim.

**Workload.** The registered nano oriented model trains on 104 public DOTA-v1.0 tiles with 2,075 annotations and validates on 13 tiles with 616 annotations. Training and validation use disjoint original source images, 16 and 4 respectively. Eight epochs, batch 4, 1024 px, BF16, seed 123, deterministic kernels, four persistent loader workers and one NVIDIA L4 on a g2-standard-8. Both arms enable prepared source caching, device HSV, epoch prefetch and the existing orthogonalization/update-preparation replay. The comparison uses separate upstream and candidate source trees in fresh processes. Timing brackets include model/trainer construction, cache and recording setup, augmentation, training, EMA, validation and checkpoint writes, with device synchronization at both boundaries.

**[MEASURED] Five alternating fresh-process pairs.** One excluded warmup pair precedes the five measured pairs. Native and replay ranges do not overlap; every paired reduction is positive (11.669–13.572%). The exact model, optimizer, scheduler, EMA and per-step loss byte hashes agree in each pair and across native repeats, as do validation values. The package import path and hashes bind each arm to its separate source tree. The complete-fit median drops **12.011%**.

| Metric, median (range across five runs) | Native | Replay |
| -- | -- | -- |
| Complete fit, seconds | 68.642 (68.213–69.333) | 60.397 (59.924–60.784) |
| Whole process, seconds | 73.795 (73.379–74.561) | 65.608 (65.169–65.987) |
| Warmed training step, ms | 183.275 (179.156–184.821) | 139.500 (138.453–141.266) |

Warmed step diagnostics discard the first 64 steps and use CUDA events, synchronized before reading. Peak allocated memory is 2,572,018,688 bytes native and 1,889,886,208 bytes replay; peak reserved memory is 4,167,041,024 and 4,508,876,800 bytes, respectively, constant across the five runs. Extra graph-pool reservation is a tradeoff, not a capacity improvement. Dependency/compiler and OS caches are warm; each fit reconstructs its prepared image cache and CUDA recording. Package import and post-fit byte hashing are outside the fit timer and included in the separately reported process time. Runtime: Python 3.11.17, torch 2.9.1+cu129, Lightning 2.6.6, L4 driver 580.178.04.

**The removed work is observed at its owner.** Separate four-step CUDA traces show 31,193 native `cudaLaunchKernel` calls and 12,609 with replay, while graph launches rise from 4 to 12, adding the network forward/backward recordings beside the existing optimizer replay. Device-event union is 353.374 and 358.045 ms; these instrumented windows support the dispatch diagnosis, not a kernel-speed claim.

**A larger-batch probe limits the claim.** One batch-8 pair, with the same eight epochs and 1024 px inputs, changes complete-fit time from 60.494 to 58.014 s, only 4.099%. State and validation parity still hold, but that one pair is below the investigation's 8% target and is not a repeated performance claim. The five-pair headline remains scoped to batch 4.

**Validation.** Thirty oriented/constructor cases passed on both torch 2.9.1 and 2.4.0; the floor retains its existing BF16 eager fallback, while FP32 and FP16 exercise real replay. A silent eager reversion and an unsafe objective-guard bypass each make the committed Trainer tests fail. Five additional helper-replacement cases pass on CPU after review. Local CPU checks and source doctests, with those cases included, report 17 passed and 19 CUDA skips. Strict mypy and the full pre-commit hooks pass. No frozen golden is changed.

**Coverage and limits.** Focused tests compare all six fields, actual CUDA launches, empty-target gradient presence, temporary shape fallback, gradient clipping, accumulation retirement (native fallback), FP32/FP16/BF16, scaler overflow and recovery, native-objective fallback, EMA validation and fit cleanup. FP16 fixtures use a unit initial scale plus one injected overflow so at least one real optimizer update must occur. A silent native reversion must fail the launch assertions; bypassing the objective guard must be rejected by the absent-angle-gradient case. Full `make gate`, accelerator goldens and long-run acceptance accuracy are outside this validation run.

