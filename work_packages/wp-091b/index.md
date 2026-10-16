# 📦 WP-091b — `feat(decode): rotated suppression decoder`

> Research log · [Phase 9 — Inference and generalization](https://borda.github.io/lucid-YOLO/research/phase-09-inference/)
>
> Engineering log · [Phase 9 — Inference and generalization](https://borda.github.io/lucid-YOLO/engineering/phase-09-inference/)
>
> 🔗 Depends on [WP-091](https://borda.github.io/lucid-YOLO/work_packages/wp-091/) · Needed by [WP-091c](https://borda.github.io/lucid-YOLO/work_packages/wp-091c/)
>
> ⬅️ [WP-091](https://borda.github.io/lucid-YOLO/work_packages/wp-091/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-093](https://borda.github.io/lucid-YOLO/work_packages/wp-093/) ➡️


## WP-091b — a threshold with nothing to cite

<a id="wp-091b"></a>

The interesting half is the threshold, and what it cost to find that there is nothing to cite for it — the decode rule itself (greedy class-wise suppression over `rotated_iou`) is described at [wp-091b.md#wp-091b-engineering](https://borda.github.io/lucid-YOLO/work_packages/wp-091b/#wp-091b-engineering). R1 states no suppression parameter at all — the one-to-one branch's freedom from NMS is the paper's claim, not a setting. R18's devkit scores detections it is handed and settles nothing on the decode side. R13 fixes the angle convention and no suppression rule. So the oriented comparison column has a knob no allowlisted source turns, and the register (A61) says exactly that in its source column rather than dressing a convention up as a citation.

What the value rests on instead is holding one variable still: 0.7 is carried over unchanged from the axis-aligned path, so the two comparison columns differ by the overlap *measure* alone and any gap between them is attributable to the geometry rather than to a second knob turned at the same time. The defence that was **rejected** is more interesting than the one taken. It would be natural to argue that 0.7 is safe because rotated IoU is the smaller measure, so carrying the threshold over can only suppress less. That is false: rotated IoU is usually smaller — mean 0.047 against 0.080 over 3000 random canonical pairs — but it *exceeds* the envelope IoU whenever both boxes lie along a shared diagonal, at 138 of those 3000 pairs and by up to +0.146. An argument that sounds like a bound and is only a tendency is worse than no argument, because it stops the next reader from checking.

The exposure is recorded with numbers rather than a hedge: 0.7 under-suppresses elongated objects, which is DOTA's characteristic shape. Two detections of one 10:1 object whose headings disagree by 5 degrees score 0.643 and both survive. That direction was accepted deliberately — a duplicate is inspectable in the output, where an over-suppressed object is simply absent — and the value is a constructor argument, so the alternative is one key away.

Looking for a lineage to inherit turned up that there was none. The axis-aligned `0.7` had shipped since WP-042 as a bare default with no register row, which means every "mAP (non-E2E)" number this project has published rested on an unregistered constant. It is now **A62**, registered at its shipped value rather than revised: re-picking it would silently move a comparison column that already has committed figures beside it, and that is a 0.MINOR change with goldens, not a tidy-up. The general shape: an assumption that goes looking for its own parentage is a good way to find the rows nobody wrote.


> Engineering log · [Phase 9 — Inference and generalization](https://borda.github.io/lucid-YOLO/engineering/phase-09-inference/)


## WP-091b — the tie-break decoder implementation

<a id="wp-091b-engineering"></a>

The decoder itself is the small half. The rule is greedy class-wise suppression whose overlap is `rotated_iou` — the same exact polygon measure the oriented evaluator grades with — over boxes canonicalized first, so two detections stated a half turn apart cannot read as two objects. Class separation tests the label directly rather than borrowing `batched_nms`'s coordinate-offset trick, because there is no offset that separates rotated boxes without also turning them.

Why its overlap threshold has no paper source to cite, and what that cost to establish, is recorded at [wp-091b.md#wp-091b](https://borda.github.io/lucid-YOLO/work_packages/wp-091b/#wp-091b).

