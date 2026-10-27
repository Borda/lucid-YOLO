# 📦 WP-195 — `perf(data): batch rotated-box clipping and refitting`

> Engineering log · [Phase 18 — CUDA graph replay and pipeline throughput](https://borda.github.io/lucid-YOLO/engineering/phase-18-cuda-replay-throughput/)
>
> 🔗 Depends on [WP-194](https://borda.github.io/lucid-YOLO/work_packages/wp-194/) · Needed by WP-196
>
> ⬅️ [WP-194](https://borda.github.io/lucid-YOLO/work_packages/wp-194/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-197](https://borda.github.io/lucid-YOLO/work_packages/wp-197/) ➡️


## WP-195 — rotated augmentation shares clipping and refitting dispatch

<a id="wp-195"></a>

**The profiler selected the whole operation.** An isolated one-thread CPU pass over 32 public DOTA augmentation draws at 640 pixels spent 1.15 s of its 5.67 s input interval in rotated clipping and refitting, about 20.3%. Optimizing only the half-plane helper did not move the enclosing input path enough; grouping the clips but leaving each refit scalar also fell below the training target. This row batches both parts. At 1024 pixels the same stage was only 6.8% of the input interval, so that profile does not support a comparable claim there.

**The batching follows the existing arithmetic.** Each of the four half-plane passes groups rings by their current vertex count, evaluates the scalar clipper's existing pointwise expressions over the group, and splits the surviving vertices back in their original ring order. Refitting groups the surviving polygons the same way: `vmap` applies the unchanged `_polygon_area` and `_fit_corners`, and the existing `polygons_to_rboxes` converts the resulting quadrilaterals together. There is no polygon padding, vertex cap, replacement fit, changed threshold or imported geometric convention. Results scatter back to their original instance rows. The original scalar clipper remains the reference and the route for single clips, accelerator inputs, other dtypes, gradients and non-finite inputs.

**The scope is CPU input work during oriented training.** It changes no model, loss or optimizer arithmetic. The public entry point and return layout are unchanged. Tests compare raw storage bytes with the scalar route, exercise changing vertex counts and empty parts, retain non-contiguous input ownership and check the fallback conditions. Full `make gate` and accelerator goldens are not claimed for this contribution; validation is focused on the changed geometry and a complete CUDA training comparison.

**The new test follows the ordinary staging path.** The notebook dataset ignore is anchored to `/data/`, matching its documented root-directory scope. The unanchored `data/` also ignored new files below `tests/data/` and `src/lucid_yolo/data/`, so `git add -A` skipped them. The root notebook-download exclusion remains; already tracked files keep their behaviour.

**The complete CPU input path confirms the mechanism.** Five paired input epochs on an isolated n2-standard-8 (8 vCPUs, Intel Xeon at 2.80 GHz, torch 2.9.1+cpu, one thread) traversed all 117 distinct public DOTA tiles and 2,691 annotations, including image-cache construction. Median wall time moved from 22.758 s (21.629–23.423) to 20.870 s (20.783–21.168); paired median reduction 8.27%, range 3.51–9.91%. Clipping/refitting accounted for 17.98% of the baseline input interval and its median cost fell from 6362.72 to 2856.17 microseconds per real-data call. Input images and every packed-target field matched at byte precision in all pairs. The instrumented input epoch includes checksum work; the separate complete CUDA fit carries the training claim.

**The training result is measured after the merged cache and state updates.** Five counterbalanced complete 20-epoch fits on an isolated g2-standard-8 with an NVIDIA L4 (driver 580.178.04, torch 2.9.1+cu129, Python 3.12.3, Lightning 2.6.6) used nano OBB, 640 pixels, batch 16, BF16, six persistent workers, cached images, mosaic and EMA. The first 20 sorted official DOTA validation images supplied 117 distinct lossless tiles and 2691 annotations; they are training-throughput inputs, with no held-out accuracy claim. Every fit traversed 2340 augmented samples and 160 optimizer steps; validation, logging and checkpointing were disabled. Baseline median 165.056s (164.518–169.835) became 149.772s (146.751–152.531); paired median reduction 10.19%, range 8.68–12.25%. Timing synchronizes CUDA around the whole fit and includes cache construction, worker startup, data and augmentation, transfer, forward/loss/backward, optimizer, EMA and Trainer teardown. Final model, EMA, optimizer and scheduler storage hashes, first input and sampled losses match in every pair. Peak CUDA allocation is unchanged at 3,583,640,576 bytes. One discarded two-epoch warmup precedes the five fresh-process pairs. No comparable 1024-pixel, larger-scale, MPS, DDP or convergence benefit is claimed. The paid VMs were deleted and independently verified.

