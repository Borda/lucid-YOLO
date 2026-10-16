# 📦 WP-183 — `refine(losses): dense box terms over every anchor`

> Engineering log · [Phase 17 — R1 pose-objective parity](https://borda.github.io/lucid-YOLO/engineering/phase-17-pose-objective-parity/)
>
> 🔗 Depends on WP-182 · Needed by WP-184, WP-186
>
> ⬅️ [WP-178](https://borda.github.io/lucid-YOLO/work_packages/wp-178/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-188](https://borda.github.io/lucid-YOLO/work_packages/wp-188/) ➡️


## WP-183 — the box terms scored densely, so the step traces as one graph

<a id="wp-183"></a>

**A boolean gather is a shape the compiler cannot know.** `pred_boxes[fg_mask]` lowers to `aten.nonzero`, whose output length is a property of the data, and `torch.compile` answers that with a graph break; the detection step carried seven of them, at the masked gathers. Recorded 2026-09-18. The replacement evaluates every term over the full `(B, A)` anchor grid — CIoU through a `reshape(-1, 4)` since its contract is two-dimensional, ProbIoU directly since it broadcasts, the L1 differences against the dense stride, the angle term over the whole branch — and selects the positives afterwards. The selection is a `torch.where`, not a multiply by the mask: a background anchor's target is the all-zero box, so the dense CIoU there is scored on a degenerate pair, and `0 * NaN` is still `NaN` where `where` is an exact selection in both the value and the backward pass. What remains is a dense sum in which every off-positive term is an exact zero, and `square_angle_loss` already documents whole-branch input with `q_i = 0` background as its intended second use.

**Loss-identical, measured rather than argued.** On a COCO batch the two forms differ by an absolute `0.0` on every term, the gradients stay finite on an image with no positives, and the pre-change CPU step snapshot holds at its existing tolerance; in eager the step is 2% faster, with the seven `nonzero` calls gone. The masked implementation is kept verbatim as a private oracle in each test module, and the dense form is held against it on random assignments — an image with no positives, an all-background batch, and degenerate predictions (zero-area, inverted, all-zero, both sides collapsed) planted on background anchors — to `1e-6` on values and gradients. The oriented oracle draws `gt_index` from `[0, N)` with `N ≥ 2`, so the gather over the padded instance axis is exercised on every column rather than only its first. WP-184 is what the one-graph step is for.

**What the accelerator gate said afterwards (2026-09-19).** On an A100-40GB, two of the five `goldens/gpu/` rows moved out of band under this change alone: keypoints `0.5696 → 0.4919` and oriented `0.9783 → 0.9088`, per-commit on the same machine, with WP-182's layout change accounting for `−0.021 / −0.040` of the path and detection, segmentation and `shapes_regression_det` holding. The arithmetic is the same; the summation order is not, and the overfit gate is a 100-epoch trajectory that amplifies a `1e-7` per-step difference into a different endpoint — the same sensitivity the [WP-177](https://borda.github.io/lucid-YOLO/work_packages/wp-177/) escalation measured across seeds on unchanged code. The principal resealed both rows at the measured values (ESCALATION 2026-09-19, WP-186); the oriented band is now `clamped_tolerance`-narrow at `± 0.0088`.

**And what three seeds said (2026-09-20).** The control the escalation asked for, run per commit on the same A100: the oriented move that looked monotone at HEAD splits into WP-182's `−0.020` (3 of 3 seeds down) and this change's `−0.014` (2 of 3), and keypoints span `0.49`–`0.73` across nine runs of the same objective. A layout change with no arithmetic in it producing the same signature as the loss rewrite is the evidence that the signature is the gate's, not the change's. Full table in ESCALATION 2026-09-19's follow-up.

