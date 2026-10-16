# 📦 WP-174 — `test(eval): the keypoint tier's shipped surface is executed`

> Engineering log · [Phase 15 — Audit remediation, v0.8.0](https://borda.github.io/lucid-YOLO/engineering/phase-15-audit-remediation/)
>
> 🔗 Depends on [WP-165b](https://borda.github.io/lucid-YOLO/work_packages/wp-165b/) · Needed by [WP-175](https://borda.github.io/lucid-YOLO/work_packages/wp-175/)
>
> ⬅️ [WP-173](https://borda.github.io/lucid-YOLO/work_packages/wp-173/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-175](https://borda.github.io/lucid-YOLO/work_packages/wp-175/) ➡️


## WP-174 — three shipped surfaces that a `callable(...)` Example stood in for

<a id="wp-174"></a>

`predict_keypoints` had **zero** executed lines in the gate. `pose_eval.py`, the keypoint tier's acceptance scorer, sat at 37%. `_overlay_ema` was unexecuted entirely. All three carried a docstring Example of the form `>>> callable(run)`, which satisfies `--doctest-modules` *and* the repository's own doctest-coverage audit while asserting nothing a broken implementation would fail. That is the pattern worth naming: a required Example is a coverage obligation, and an Example that only asserts the symbol exists discharges the obligation without discharging the purpose. `detect_eval.py` and `rotated_eval.py` carry the identical Example and sit at 97% and 96%, so the pattern only bites where nothing else happens to reach the code.

After: `predict.py` 76% → **99%**, `pose_eval.py` 37% → **100%**, `checkpoint.py` 47% → **94%**. 31 new test items.

**The keypoint expectations are hand-computed, not recorded.** On the 64×128 planted canvas the letterbox scale is `min(64/64, 64/128) = 0.5`, content is 32×64, `pad_top` is 16 and `pad_left` is 0, so the inverse is `x_orig = 2·x_canvas` and `y_orig = 2·(y_canvas − 16)`. Those literals are in the test with the arithmetic beside them. A test that asserts whatever the code returned would have passed against a broken inverse just as well.

**The fixture is built to fail in the interesting ways.** The one-to-one and dense branches carry *disjoint* point sets, so a path reading the other branch's stem returns a plausible pose of the wrong object beside a correct box — the failure the source docstring warns about, and one no shape assertion can see. A decoy set sits on anchor 0, which is never the selected anchor, while every other anchor carries zero offsets decoding to its own centre; a fixed-row gather therefore returns the decoy or a centre rather than the plant. That case is self-guarding: if the decoy anchor ever coincided with the selected one, the decoy would overwrite the plant and the primary assertion fails rather than passing vacuously. `K` is 3 throughout rather than 17, so a function that assumed the human schema fails.

**One brief corrected against the data.** The row asked for the `K != 17` refusal "at minimum"; that branch was already covered at `K=7`, which the audit's own coverage line implies — missing `130-175` means `117-129` runs. The real gap was the scoring body. Both landed: the scoring body is now executed, and the refusal gains `K=16`, the value adjacent to the boundary where an off-by-one or a table padded by one joint lets a foreign schema through. The pose split is hand-built in the layout `run` resolves, which is what pins the annotation *file name* — `person_keypoints_` rather than `instances_`, the substitution that divides a person detector's AP by eighty.

**`_overlay_ema` was the one place the tests found real defects.** Three, all in how the EMA claim was established rather than in arithmetic. The callback's key was resolved by substring in one place and by literal name in another, so a checkpoint written by a parametrized callback — whose `state_key` carries its arguments — passed the substring match, was overlaid, and then died on a bare `KeyError` *after* the module had been mutated. Two matching entries were resolved by iteration order, the last silently winning. And the copy walked the *shadow's* keys rather than the module's, so any module tensor absent from the shadow kept its raw value with nothing said — a run reporting a mixture of EMA and raw weights under the EMA name, which is neither of the two claims a report can make.

The key is now resolved once and handed back so the shadow and the update counter come from the same entry; more than one match raises; and coverage is checked **before** any copy rather than after. Validating after would have reproduced the defect's own shape — naming the problem while leaving the module in the state the message declares invalid. Validating first means no half-overlaid module ever exists, and the test asserts that stronger contract by snapshotting every float tensor and proving none moved after the refusal.

**What the row did not find is worth recording too.** `predict_keypoints` and `pose_eval.run` are correct as written — every hand-computed coordinate matched on the first functional run. Their gap was evidentiary, not behavioural. That is the honest result for two of the three surfaces, and it is why the row is `test(eval)` rather than `fix`.

