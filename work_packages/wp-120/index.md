# 📦 WP-120 — `feat(data): keypoint targets and fliplr index swap`

> Engineering log · [Phase 12 — Keypoint detection, v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-12-keypoints/)
>
> 🔗 Depends on WP-008, [WP-013](https://borda.github.io/lucid-YOLO/work_packages/wp-013/) · Needed by [WP-121](https://borda.github.io/lucid-YOLO/work_packages/wp-121/), [WP-122](https://borda.github.io/lucid-YOLO/work_packages/wp-122/)
>
> ⬅️ [WP-127](https://borda.github.io/lucid-YOLO/work_packages/wp-127/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-121](https://borda.github.io/lucid-YOLO/work_packages/wp-121/) ➡️


## WP-120 — a mirror does not know anatomy

<a id="wp-120"></a>

Phase 12 opens on a container change rather than a model change: `Targets` gains `keypoints` (`(N, K, 2)` float32) and `keypoint_vis` (`(N, K)` int64), and `HorizontalFlip` -- the one existing transform this WP's scope covers -- gains the matching mirror. Every other geometric transform (`RandomAffine`, `MosaicAssembly`, `Mixup`/`CopyPaste`, `Letterbox`) is untouched; keypoint support there is a later WP's concern, not this one's.

K is a constructor argument the way the class count already is: `keypoints`/`keypoint_vis` follow `polygons`' existing "count is 0 or N" convention (the shared instance axis every box-aligned modality already uses) rather than `rboxes`' independent-axis-plus-separate-mask one, since a keypoint set belongs to one box the way a polygon ring does, not to a second geometry entirely. `concat` additionally rejects differing `K` across items that do carry keypoints -- a check `_concat_polygons` has no analogue for, since ring length is per-instance there and K is fixed per dataset here, so two items disagreeing on K is corruption, not a valid ragged shape.

The harder decision is what `HorizontalFlip` does with left/right identity. A mirror reflects every point's x-coordinate unconditionally -- that part is geometry, not anatomy, and needed no new assumption. But swapping *which* index is now "left knee" after the flip is anatomical, dataset-specific knowledge the transform has no business hardcoding into a K-generic container's own mirror: nothing in this repository's K-point design says K=17 or names a COCO ordering, and baking the pairing into `HorizontalFlip` would silently narrow "generic K-point task" back down to "COCO human pose" the first time this class ran. `keypoint_flip_pairs: list[tuple[int, int]] | None = None` is the resolution -- a caller-supplied swap map, `None` meaning "mirror coordinates, keep column identity", the correct default for a K-point task with no left/right symmetry. A64 records COCO's own 17-point pairing as the value the demo dataset will actually supply, but the row is marked `open`: [WP-121](https://borda.github.io/lucid-YOLO/work_packages/wp-121/)'s COCO datamodule is what passes the concrete list in, not this one.

Implementation delegated to `codex:codex-rescue --write` against a design spec fixing the exact field names, shapes, validation rules, and test scenarios up front, rather than leaving the K-point/anatomy split to be discovered mid-implementation -- reviewed diff-by-diff against that spec before landing; one docstring grammar slip (`"A supplied, a dataset keypoint pair map..."`) was the only correction needed.

______________________________________________________________________

