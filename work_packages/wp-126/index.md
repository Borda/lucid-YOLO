# 📦 WP-126 — `docs(report): keypoint tier report, model card, and the fourth task everywhere`

> Engineering log · [Phase 12 — Keypoint detection, v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-12-keypoints/)
>
> 🔗 Depends on [WP-125](https://borda.github.io/lucid-YOLO/work_packages/wp-125/) · Needed by [WP-140](https://borda.github.io/lucid-YOLO/work_packages/wp-140/)
>
> ⬅️ [WP-125](https://borda.github.io/lucid-YOLO/work_packages/wp-125/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-131](https://borda.github.io/lucid-YOLO/work_packages/wp-131/) ➡️


## WP-126 — the fourth task existed everywhere except where a reader would look

<a id="wp-126"></a>

The tier's own artifacts landed cleanly: a model card, a `0.5.0` reproduction-report section, a fourth panel on `plot_training.py` for the keypoint loss term. What did not land, and was not noticed until the naming of one file was questioned, is that every general-audience document still described a three-task project. `README.md` said "Three tasks, one trunk" and listed three; `docs/index.md` had a heading reading "The three tasks"; `docs/TRAINING.md` carried a launch section for detection, segmentation and oriented detection and none for keypoints, and its wiring-gate block listed three of the four `overfit_micro.py` tasks. None of that is a gate failure — every register check, link check and section check passed throughout, because each of them asks whether a named thing exists, and no gate asks whether a *newly* named thing has been added everywhere its three siblings appear.

**The naming question that surfaced it is itself the more useful finding.** The card shipped as `model_cards/pose.md`, describing a "pose detector", which is precisely backwards: this project's keypoint task is `K`-generic by construction — `num_keypoints` is a constructor argument the way the class count is, nothing in the head, the loss or the decode path reads a point's meaning, and the wiring gate deliberately runs a 7-point synthetic symbol schema rather than a human one (A67 exists only because that schema has no annotator variance to derive sigmas from). Human pose is one instantiation, distinguished by having a published sigma table and a ground-truth file, not by being what the architecture is for. The file is `model_cards/keypoints.md`, and the general docs say keypoints and then name pose as the instantiation the shipped checkpoint trained on.

The task-specific prose stays in pose terms wherever it is actually describing the shipped checkpoint — its training data is `person_keypoints`, its ethical-considerations section is about predicting human skeletons, and softening those into schema-neutral language would make the card describe something safer than what was trained. The distinction the rename enforces is between the architecture (keypoints) and this checkpoint (pose), not a blanket find-and-replace.

**Three stale figures were corrected in passing**, each found by reading a line rather than by any check: the README claimed 63 registered assumptions against an actual 73, and 20 frozen golden files against an actual 14 (the number had been read off a total that included the seven live sets); and it described roadmap 111's whole-image oriented figure as not yet measured, which had stopped being true when that row closed and the reproduction report gained the number. A document that quotes a count is a document that goes stale silently, and none of the three would have been caught by anything in `make gate`.

The general lesson for the next task family, whenever there is one: adding a task is not done when its own artifacts exist. The check worth running by hand is to grep for the *previous* task's name across the tree and ask, at each hit, whether the new one belongs beside it.

