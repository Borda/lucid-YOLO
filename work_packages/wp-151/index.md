# 📦 WP-151 — `feat(models): keypoint detector ONNX export`

> Engineering log · [Phase 13 — Contributor admission, augmentation guard and keypoint parity, v0.6.0](https://borda.github.io/lucid-YOLO/engineering/phase-13-contributor-admission/)
>
> 🔗 Depends on [WP-139](https://borda.github.io/lucid-YOLO/work_packages/wp-139/) · Needed by [WP-152](https://borda.github.io/lucid-YOLO/work_packages/wp-152/)
>
> ⬅️ [WP-150](https://borda.github.io/lucid-YOLO/work_packages/wp-150/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-152](https://borda.github.io/lucid-YOLO/work_packages/wp-152/) ➡️


## WP-151 — the released task that could not leave PyTorch

<a id="wp-151"></a>

0.5.0 shipped keypoints as the fourth task and shipped no way to export it. Detection, segmentation and oriented detection each had a graph class in `lucid_yolo/export.py` and a case in the ONNX gate; keypoints had neither, so it was the only released task whose model could not leave PyTorch at all. `KeypointExportGraph` closes that, and the gate now runs four tasks where it ran three.

**The composition follows segmentation rather than detection, and the reason is structural.** A detection graph calls the decoder's plain `forward` and is done. The point stem, like the mask-coefficient stem, is *dense over anchors*, so decoding it needs the source anchor of each detection row — which `decode_with_indices` returns and `forward` discards. That single fact decides the whole shape of the class: `decode_with_indices` for the detections and their anchor indices, `decode_keypoints` over the entire dense tensor, then a gather. Decode first, gather second, which is the order the OBB angle branch established at [WP-062](https://borda.github.io/lucid-YOLO/work_packages/wp-062/) and the training module's own `val/oks_mAP` already uses. The reverse order is not merely slower; `decode_keypoints` needs the whole anchor grid and offers no per-detection indexing, so gathering first would mean reconstructing each kept row's anchor and stride by hand — a second implementation of the thing the index already names.

**Padding is where the two dense tasks part.** A segmentation graph *raises* on a padding row: gathering mask coefficients at index `-1` is an error, so that class documents a real precondition that the canvas carry more anchors than the detection cap. The keypoint gather does not have that precondition, because `gather_keypoints` clamps a padding index to a valid position and then zeroes the row, so a padding row yields the origin rather than the last anchor's pose. Reusing that function rather than writing a fresh gather is the point: [WP-109](https://borda.github.io/lucid-YOLO/work_packages/wp-109/)'s three-strides lesson was that a hand-copied padding rule drifts, and this is the third caller of the same rule rather than a third version of it.

**The gate was extended, not duplicated.** The keypoint case rides the existing fixture — the same doctored checkpoint, the same 128-pixel canvas, the same onnxruntime comparison against an eager decode of the checkpoint-loaded module — and adds one output-shape assertion and one parity assertion. The parity assertion carries a vacuity guard of its own, in the shape the existing mask comparison established: a gather at the wrong anchor still returns finite, in-canvas coordinates, so the point sets must be shown to differ from one another before comparing two orderings of them proves anything. Without it, a graph that returned the same pose for all 300 rows would match a reference that did the same.

**`K` is deliberately three rather than seventeen.** The point count is a constructor argument and the architecture is generic in it (A64); COCO's seventeen is one instantiation. A gate that only ever ran at seventeen would pass a shape accidentally baked to that number, which is exactly the class of defect an export gate is for. Three is small, is not COCO's, and puts `(1, 300, 3, 2)` in the asserted static output shape where a hardcoded seventeen would show up immediately.

