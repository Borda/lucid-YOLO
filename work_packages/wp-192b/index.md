# 📦 WP-192b — `perf(data): cache decoded training images`

> Engineering log · [Phase 17 — R1 pose-objective parity](https://borda.github.io/lucid-YOLO/engineering/phase-17-pose-objective-parity/)
>
> 🔗 Depends on [WP-192](https://borda.github.io/lucid-YOLO/work_packages/wp-192/) · Needed by [WP-192c](https://borda.github.io/lucid-YOLO/work_packages/wp-192c/)
>
> ⬅️ [WP-192](https://borda.github.io/lucid-YOLO/work_packages/wp-192/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-192c](https://borda.github.io/lucid-YOLO/work_packages/wp-192c/) ➡️


## WP-192b — decode once, augment many times

<a id="wp-192b"></a>

**The input stalls, not the compiled loss, were the complete-path bottleneck.** Profiling twelve real oriented DOTA samples at 1024 px attributed 0.791 of 1.233 CPU seconds to `torchvision.io.decode_image`; the four-source mosaic makes that repeated work the dominant single stage. Compiling the oriented objective was rejected after the five-repeat steady-state result proved noisy: its paired median-step reduction was 43.39%, but the pairs ranged from 26.68% to 46.08%, the baseline median had 31.08% relative spread, and the repeat-median mean start-to-start interval worsened 24.68%. The intended repeated `Trainer.fit` wall-time run yielded no admissible measured pair after its discarded warm-up because SSH failed. The selected change acts on the stage the complete path was actually waiting for, and earns its claim from the complete repeated fit below.

**The cache is raw and opt-in.** Both readers can eagerly retain one RGB `uint8` tensor per distinct training file. The entry sits before mosaic and every random transform, and `_load_image` still allocates the float image each draw, so the cache removes decoding without freezing augmentation or handing mutable cache storage to a caller. `DetectionDataModule.cache_train_images` routes the switch to the training reader only; validation remains uncached. D27 keeps the default off because the memory bill is the uncompressed split, and limits the copy-on-write claim to forked workers.

**Public-data values are unchanged.** Two seeded pipelines, one cached and one uncached, produced bit-identical float-image `int32` views and exactly equal target fields over 100 real DOTA draws carrying 7,083 instances. The slice held 41 distinct decoded tensors using 125,370,774 bytes.

**The isolated CPU path establishes the stage ceiling.** On an archbench C3 VM (8 vCPU, Intel Xeon Platinum 8481C, torch 2.9.1+cpu), five measured 1,640-draw traversals after one warm-up included dataset construction and returned the same checksum in both arms. The uncached median was 52.713 s (52.520–53.127 s); cached was 9.440 s (7.936–9.494 s), with a paired median reduction of 82.14% (81.99–84.95%). The VM was deleted and an explicit instance query returned no row. This is the reader-stage bound, not the training headline.

**The training headline clears the complete-path bar on top of [WP-193](https://borda.github.io/lucid-YOLO/work_packages/wp-193/).** [WP-193](https://borda.github.io/lucid-YOLO/work_packages/wp-193/) is merged (`e8d632c`); both arms of the L4 comparison ran on that post-backward head so adjacent gains are not double-counted. Five measured 257-step `Trainer.fit` pairs after one discarded warm-up, oriented nano at batch 16 / 1024 px / BF16 with six workers and EMA, moved from a 436.922 s median (435.429–441.999 s) to 332.337 s (332.170–343.212 s): 23.94% from arm medians. Every pair improved by 21.97–23.97%, with a 23.68% paired median. The ordinary step median was actually 2.42% slower; the interval mean fell because the multi-second loader stalls disappeared, which is the measured mechanism rather than a GPU-kernel claim. First input and first loss matched in every pair, and peak CUDA allocation was exactly 11,174,604,800 bytes in both arms. The isolated `g2-standard-8` used an NVIDIA L4, driver 580.178.04, torch 2.9.1+cu129 and Python 3.12.3; archbench deleted it and both the helper cleanup and explicit inventory found no live instance.

**The headline is a small-slice, decode-bound-host result and does not transfer unchanged to a full-size split.** The L4 fit slice was the same 41 distinct tiles (125,370,774 bytes of cache) the parity paragraph cites, expanded to 4,100 records under fresh IDs per the PR description, so the cache build cost about 41 decodes rather than 4,100; the build sits inside the measured wall time and was nearly free. The host was an 8 vCPU `g2-standard-8` with six workers, the regime where the measured mechanism, the loader stalls that disappeared, applies. A real split costs about one decode per distinct file to build, and its RAM bill follows `sum(3*H*W)` over those distinct files (TRAINING.md) rather than the 125 MB measured here; neither that build time nor that memory was measured on a full-size split, so the 23.68% and 23.94% figures say nothing about either.

**Why the row is lettered 192b.** It was lettered from [WP-192](https://borda.github.io/lucid-YOLO/work_packages/wp-192/) so that the then-open [WP-193](https://borda.github.io/lucid-YOLO/work_packages/wp-193/) post-backward contribution kept its identifier; [WP-193](https://borda.github.io/lucid-YOLO/work_packages/wp-193/) has since merged.

