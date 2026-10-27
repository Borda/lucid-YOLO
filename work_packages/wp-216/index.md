# WP-216 — cross-task training stories

> Engineering log · [Phase 19 — Codebase audit remediation](https://borda.github.io/lucid-YOLO/engineering/phase-19-codebase-audit/)
>
> 🔗 Depends on [WP-215](https://borda.github.io/lucid-YOLO/work_packages/wp-215/) · Needed by [WP-217](https://borda.github.io/lucid-YOLO/work_packages/wp-217/)
>
> ⬅️ [WP-215](https://borda.github.io/lucid-YOLO/work_packages/wp-215/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-217](https://borda.github.io/lucid-YOLO/work_packages/wp-217/) ➡️


<a id="wp-216"></a>

## WP-216 — cross-task training stories

**Finding.** T3’s remaining training duplication repeats empty-image, branch-gradient and checkpoint-composition stories across task suites. Body comparison distinguishes data-only differences from task-specific actions and oracles: segmentation and keypoints share the empty-image oracle, all three tasks share the gradient oracle, and OBB and keypoints share the state-key oracle. Segmentation’s checkpoint-loading test also compares zero-gain training losses and is not the same story.

**Change.** Three parametrized tests retain each case’s model options, target generation, expected branches, log key and state-key exclusions. Existing target builders move into the shared training fixtures without changing their generated values; per-task suites import them by bare module name. Every original gradient assertion remains, including both head branches and the non-null pose flow. Assignment-order tests stay separate: segmentation compares mask losses exactly, OBB checks three zero and three positive terms, and keypoints compares against an RLE reference. OBB’s empty-image test retains its backward-connectivity assertions. No shipped code or changelog changes.

**Verification.**

- **CPU.** `make gate` green on this commit's tree. `test_task_gradients_reach_every_branch`, `test_task_image_without_instances_gives_finite_zero_term` and `test_task_state_dict_keeps_detection_keys` retain seven cases from seven original tests; the four distinct cases remain in their original suites.
- **Body comparison.** No pair is wholly identical. Gradient bodies differ by target builders, module options and branch names; the keypoint wrapper already installs its log recorder. Empty-image bodies for segmentation and keypoints differ by task options and expected log key. State-key bodies differ by stem substring and the flow prefix excluded for keypoints. The OBB empty-image actions and all three assignment-order oracles differ in behavior, so they are not merged.
- **Case counts.** Empty image: 3 → 3; assignment order: 3 → 3; branch gradients: 3 → 3; state keys: 2 → 2. The three original training files plus the shared story suite collect 109 → 105 items because four target-builder doctests move to `tests/fixtures/training.py`; those doctests remain collected by the full gate. Story cases remain 11 → 11. Segmentation has no standalone state-key case in the candidate files; its checkpoint-load and zero-gain-loss test remains unchanged.

