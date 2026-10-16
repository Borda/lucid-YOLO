# 📦 WP-139 — `feat(models): keypoint detector builder and params/FLOPs golden`

> Engineering log · [Phase 12 — Keypoint detection, v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-12-keypoints/)
>
> 🔗 Depends on [WP-122](https://borda.github.io/lucid-YOLO/work_packages/wp-122/), [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/) · Needed by [WP-140](https://borda.github.io/lucid-YOLO/work_packages/wp-140/), [WP-151](https://borda.github.io/lucid-YOLO/work_packages/wp-151/)
>
> ⬅️ [WP-138](https://borda.github.io/lucid-YOLO/work_packages/wp-138/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-140](https://borda.github.io/lucid-YOLO/work_packages/wp-140/) ➡️


## WP-139 — the same blindness, one layer down

<a id="wp-139"></a>

**[WP-126](https://borda.github.io/lucid-YOLO/work_packages/wp-126/) above closes with a lesson, and this row is what running that lesson found.** Greping for the previous tasks' names across `src/` rather than across `docs/` turns up a structural gap the documentation sweep could not see: detection, segmentation and oriented detection each ship a composite `nn.Module`, a `build_*` factory, a `.deploy()` NMS-free inference view and a frozen params/FLOPs golden. Keypoints shipped none of the four. The keypoint model existed only as an inline `build_detection_stages(..., num_keypoints=K)` call inside `DetectionLitModule.__init__` — enough to train, evaluate and report, which is exactly why nothing complained.

**The golden harness cannot notice a golden that was never written.** `check_goldens.py` discovers its work by globbing `goldens/*.json` and `goldens/frozen/**/*.json`, so every golden present is checked and no count is asserted anywhere. Nothing in the repository enumerates "one size golden per task" and compares that list to the tasks that exist. This is the identical shape of blindness [WP-126](https://borda.github.io/lucid-YOLO/work_packages/wp-126/) recorded one layer up — a check that asks whether a named thing is intact, never whether a newly named thing was added beside its siblings — and it would have shipped a 0.5.0 whose headline feature was the one task absent from the release's own regression contract.

**What the measurement says.** The golden is taken at detection's own protocol — a 640-pixel input over 80 classes — rather than at a protocol of its own, specifically so that `params_flops_kp.json` minus `params_flops_det.json` is the point stems and nothing else. At `n` that difference is **+110,652 parameters and +0.1238 GFLOPs**, and the asymmetry between those two numbers is the A29 convention made visible: parameters count the whole checkpoint, so both head branches' stems are in the tally, while FLOPs count only the deployed one-to-one branch, so one branch's stems are. The per-branch stem cost is therefore half the parameter figure, and the two numbers are consistent rather than in tension.

**No published table, and that absence is recorded rather than filled.** \[**Struck 2026-09-17 (D22)**: there is one — R1 Table S10 — and this paragraph's care about not fabricating a table was spent on the wrong side of an unread section. The measurement against it is A77.\] R14 (RLE) specifies a loss and an OKS evaluation protocol and no architecture at all, so unlike detection (R1 Table 7), segmentation (Table S9) and oriented detection (Table S11), there is no size table in the literature to hold this model against. The three siblings each pair their golden with a published-table gate; this one has a golden alone, and `tests/models/test_param_flops.py` says so in a comment where a fourth `_TABLE_*` constant would otherwise be reached for. Inventing one — from a sibling table, from a third-party pose model, or from this project's own measurements — would fabricate a published claim, which is the failure mode the whole register system exists to prevent.

`num_keypoints` is keyword-only and required on both the class and the factory, with no default, matching the refusal `DetectionLitModule` already makes: `K` is a property of the dataset's annotation schema, so a default would be a silent claim about data the model has never seen. The shipped nano checkpoint's own figures in `docs/model_cards/keypoints.md` stay where they are and stay different — they are measured over COCO's single `person` category, which is a different question from the architecture's size.

**Verification.** `make gate` green: 2458 passed, 39 skipped, 3 deselected, **22/22 goldens** where 21 passed before. The new composite gates in `tests/models/test_keypoint_head.py::TestKeypointDetector` assert what the golden cannot — that the stems are run rather than merely built, that the deployed view holds no one-to-many parameter and shares rather than copies the rest, that its returned triple feeds `decode_keypoints` unadapted, and that omitting `num_keypoints` raises.

