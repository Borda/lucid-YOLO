# 📦 WP-163 — `perf(ptl): validation ground truth crosses to the host once`

> Engineering log · [Phase 15 — Audit remediation, v0.8.0](https://borda.github.io/lucid-YOLO/engineering/phase-15-audit-remediation/)
>
> 🔗 Depends on [WP-158](https://borda.github.io/lucid-YOLO/work_packages/wp-158/) · Needed by [WP-164](https://borda.github.io/lucid-YOLO/work_packages/wp-164/)
>
> ⬅️ [WP-162](https://borda.github.io/lucid-YOLO/work_packages/wp-162/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-164](https://borda.github.io/lucid-YOLO/work_packages/wp-164/) ➡️


## WP-163 — one transfer per batch, and the three candidates that did not survive measurement

<a id="wp-163"></a>

This row is the surviving quarter of a survey of RF-DETR's 1.9.0..develop range (137 commits, ~40 `perf`-tagged, Apache-2.0, read under D13 and never copied). Most of what that survey found was either already implemented here or inapplicable; what follows is the part that measured well, and the part that measured badly, because the second is the more useful record.

**What the change is.** Every validation accumulator moved its ground truth to the host one image at a time: two tensors per image for detection, three for oriented detection and keypoints, and a mask stack for segmentation. Each of those transfers carries the accelerator's fixed per-transfer latency and a payload of a few kilobytes, so a batch of 16 paid that latency 32 to 48 times to move something that fits in a single transfer. `_host_split` concatenates along the batch axis, moves once, and slices the result back into the same per-image pieces. Values, dtypes and ordering are unchanged, so nothing the metrics read changes and no golden moves.

**Measured, not assumed.** On MPS, 16 images x 7 boxes: **2.823 ms to 0.210 ms, 92.6% less**, median of 40 timed runs after 15 warm-ups with the device synchronised inside the timed region. On CPU the same batch goes from 0.004 ms to 0.013 ms — a 9-microsecond regression, because on CPU there is no transfer to amortise and the concatenation is pure overhead. That trade is taken deliberately: the accelerator path is where validation actually runs, and 9 microseconds against 2.6 milliseconds is not a close call.

**Honest scope.** Per validation batch this is milliseconds. Across a 5000-image validation split at batch 16 it is roughly 0.8 s an epoch — under 1% of a training epoch. The row is worth landing because it is bit-identical, strictly less work, and removes a per-image pattern that would otherwise be copied into the next accumulator; it is not worth landing on the strength of its epoch-level number, and this entry does not claim one.

**Three candidates the measurement rejected.**

*Skipping the padding rows before the segmentation mask decode.* The validation decoder hands `decode_instance_masks` a fixed 100 rows per image and `decoded[keep]` discards the padding immediately after, so filtering first is bit-identical and strictly less work — and it is **slower on the accelerator**: CPU improves 32–67% depending on survivor count, MPS *regresses* 40–86% at 40 repeats. Boolean indexing plus the data-dependent sync costs more than decoding 100 rows of a 160x160 grid in parallel. Rejected rather than adopted device-guarded: the guard would exist to serve the offline path at the accelerator's expense, for a proxy metric measured in milliseconds. Upstream made the same call in the other direction for one of its own changes, excluding MPS explicitly.

*Hoisting torchmetrics' per-annotation score reads.* `CocoBackend._get_coco_format` hoists boxes and labels once per image but reads `area[image_id][k]` one annotation at a time, and it accepts `scores=None`, so the scores can be assigned per image afterwards. Measured **26.3% and 26.6% faster** at 32 and 128 images of 100 detections. Not adopted: reaching it requires calling a private method on a private attribute (`metric._coco_backend._get_coco_format`), which binds this project's validation path to torchmetrics internals that carry no compatibility promise, and the epoch-level saving is around 0.16%. The measurement is recorded so the trade can be re-taken if torchmetrics ever exposes it.

*The `uint8 / 255` divisor form.* Upstream found that CUDA evaluates a division by the Python int `255` as a multiplication by its reciprocal, landing one ULP from the host result for 126 of 256 byte values, and switched to a 0-dim tensor divisor to restore parity. This repository's consumer conversion uses the same Python-int form, so the question was live — and **it does not reproduce here**: across all 256 byte values, both divisor forms match the host bit-for-bit on CPU and on MPS, 0 of 256 differing. A26's cross-platform divergence is not this mechanism. Recorded as a closed question rather than left as a plausible-sounding hypothesis.

