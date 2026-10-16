# 📦 WP-074 — `feat(data): RF100-VL incremental downloader and merged COCO layout`

> Research log · [Phase 9 — Inference and generalization](https://borda.github.io/lucid-YOLO/research/phase-09-inference/)
>
> 🔗 Depends on WP-068, [WP-073](https://borda.github.io/lucid-YOLO/work_packages/wp-073/) · Needed by WP-075
>
> ⬅️ [WP-108](https://borda.github.io/lucid-YOLO/work_packages/wp-108/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-091c](https://borda.github.io/lucid-YOLO/work_packages/wp-091c/) ➡️


## WP-074 — deferred, and what the search for a public YOLO corpus found

<a id="wp-074"></a>

Deferred on 2026-08-14 rather than built. `roboflow/rf100-vl` PR 29 is open and adds `combine_downloaded` / `download_and_combine` to the collection's own package — deterministic id remapping, category namespacing, a resumable manifest — which is this row's scope, written by the people who own the datasets. Building it here first means maintaining a second merge against a moving upstream and owning its correctness across 100 datasets; waiting costs a wrapper. A second route may retire the row outright: `probicheaux/rf100-vl` on Hugging Face is Apache-2.0 and ungated, 15 domains in COCO form, so the tier becomes a download plus a layout conversion with no API key involved.

The more useful finding came from asking the adjacent question — what public dataset could exercise the [WP-099](https://borda.github.io/lucid-YOLO/work_packages/wp-099/) YOLO reader at scale — and getting no good answer. The search is worth recording because the *shape* of the negative result explains [WP-099](https://borda.github.io/lucid-YOLO/work_packages/wp-099/)'s own central difficulty.

Rejected, each for a stated reason rather than a hunch: the `keremberke/*` mirrors are the largest set of public YOLOv8-format datasets on Hugging Face and state **no licence at all**, and unstated is not permissive. `CARD-Data/CARD-Germany-Batch1` is real-scale YOLO but is 315 GB, gated behind login plus terms acceptance, and its "CC BY 4.0" carries a no-military field-of-use rider — a licence with a use restriction bolted on is not the licence it names. `goodquestion1/RM26` is YOLO *pose*, whose extra keypoint fields the reader rejects by design. A Zenodo query for YOLO-format detection records under permissive terms returned nothing that was both an image dataset and detection.

What is left is the structural point. Public YOLO-format detection data at scale is almost entirely (a) Roboflow exports behind an API key and (b) Ultralytics-hosted mirrors, which this project's prime directive forbids reading. **The format has no neutral publisher, which is the same fact [WP-099](https://borda.github.io/lucid-YOLO/work_packages/wp-099/) met from the other side when it found no specification to write the reader from.** A convention owned by tooling vendors rather than by a standards body produces exactly this: universal adoption, no spec, and no corpus anyone can point at without accepting someone's terms. The consequence for this project is that the YOLO path's public story is a small verified example — the R32 export, CC BY 4.0, already on disk — plus documented instructions for bringing your own export, and not a headline dataset.

The design the row carried before it was deferred, kept here so a later attempt starts from it rather than from scratch. Per-dataset download, then category ids remapped into a **union label space** under dataset-qualified names — the collection spans domains, so bare category names collide across datasets and a collision is a silent label merge rather than an error — then images re-numbered, then appended into one merged COCO layout, with each per-dataset archive deleted before the next is fetched. That last step is what bounds peak disk to O(one dataset + merged) rather than O(all 100). Splits are preserved as published, and an `RF20-VL` subset flag exists so a smoke run does not pay for the whole collection. Two design points were deliberately left open rather than settled ahead of the code: whether evaluation reports over the union label space or per dataset, and how the verify module integrates.

Two citations for whoever picks this up: the collection is <https://github.com/roboflow/rf100-vl> and the upstream work is its PR 29, <https://github.com/roboflow/rf100-vl/pull/29>. Reconsider this row when that PR lands or is abandoned.

The licence question does **not** resolve with either route and is the thing most likely to kill the row. `rf100-vl` is Apache-2.0 **as tooling**; the 100 datasets it downloads carry their own licences, individually, and this project ships permissive-only. The Hugging Face alternative (`probicheaux/rf100-vl`, Apache-2.0, ungated, 15 domains, ~163k rows in COCO form) turns the tier into a download plus a parquet-to-COCO-layout conversion — no API key and no merge to own — but it does not answer what licence the underlying images carry.

**Dropped 2026-08-18.** The deferral above named the condition to watch (PR 29 landing or being abandoned) but not one this project controls or a date to stop waiting on — the licence question that named itself the row's likeliest killer never resolved either. Rows 074 and 075 close ⊘ rather than stay ⏸ indefinitely; the design and citations above are kept as-is for whoever reopens the tier, unedited by this note per this log's own convention of leaving a historical entry describing what was true when it was written.

