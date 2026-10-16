# 📦 WP-112 — `refine(export): one exportable decode path`

> Engineering log · [Phase 11 — Rolling, toward v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-11-rolling/)
>
> 🔗 Depends on [WP-066](https://borda.github.io/lucid-YOLO/work_packages/wp-066/) · Needed by none
>
> ⬅️ [WP-111](https://borda.github.io/lucid-YOLO/work_packages/wp-111/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-113](https://borda.github.io/lucid-YOLO/work_packages/wp-113/) ➡️


## WP-112 — two compositions of the same decode, and the one that can't filter

<a id="wp-112"></a>

[WP-066](https://borda.github.io/lucid-YOLO/work_packages/wp-066/) needed one importable `nn.Module` per task so `torch.onnx.export` had something to trace, and wrote three small wrappers private to the test file to get one. `predict.py` composes the same pieces — a task's `deploy()` view plus its E2E decoder — for single-image inference, independently. [WP-066](https://borda.github.io/lucid-YOLO/work_packages/wp-066/)'s own log entry named the risk and declined to fix it: nothing forced the two compositions to keep agreeing as either side changed, and nothing shipped needed a third caller to justify the extraction. Roadmap 112 is that extraction, now that something does.

`src/lucid_yolo/export.py` reproduces `predict_image`/`predict_segmentation`/`predict_oriented`'s `"e2e"` compositions call for call — same decoder calls, same argument order, same `decode_instance_masks(prototypes, coefficients, boxes, image_size=...)` order, same `decode_rboxes` -> `o2o_rotated_topk` pairing — so the exported graph and the single-image path can be shown to agree rather than merely resemble each other. `predict.py` itself is untouched: it calls a raw `DetectionLitModule` and decodes an unbatched result with confidence filtering, a genuinely different call convention from a traced graph's fixed-shape `nn.Module`, and unifying the two call conventions was never this row's scope.

One divergence is real and stays, documented rather than engineered away. `predict_segmentation` drops padding rows (`anchor_index == PAD_ANCHOR_INDEX`) before gathering mask coefficients, since a single-image caller wants a mask per real object and a boolean filter is free to shrink the output. A traced graph's output shape is fixed at trace time, so `SegmentExportGraph` gathers every one of its `k` rows, padding included — which only stays safe because a padding row's index is a valid position in the coefficient tensor exactly when the canvas's anchor count exceeds `k`. Below that, gathering at the padding sentinel (`-1`) raises. This is the same constraint the ONNX export test's 128 px canvas (336 anchors, above the 300 cap) was already built to satisfy; `export.py`'s module docstring restates it because the module now has callers that test fixture does not.

