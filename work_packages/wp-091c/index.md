# 📦 WP-091c — `refine(decode): rotated geometry in the geometry module`

> Research log · [Phase 9 — Inference and generalization](https://borda.github.io/lucid-YOLO/research/phase-09-inference/)
>
> 🔗 Depends on [WP-091b](https://borda.github.io/lucid-YOLO/work_packages/wp-091b/) · Needed by [WP-091d](https://borda.github.io/lucid-YOLO/work_packages/wp-091d/)
>
> ⬅️ [WP-074](https://borda.github.io/lucid-YOLO/work_packages/wp-074/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-091d](https://borda.github.io/lucid-YOLO/work_packages/wp-091d/) ➡️


## WP-091c — what a refactor is allowed to notice

<a id="wp-091c"></a>

The move itself is unremarkable: `rotated_iou` and the seven private helpers it exclusively owns leave `eval/dota_eval.py` for `data/rotated_geom.py`, since an evaluation module had no business being on the decode path's import graph for a function that is pure geometry. What the package is worth recording for is the three things it declined to do.

**It did not merge the duplicate it found.** `rotated_geom` already had `_check_2d`, differing from the arriving `_check_rboxes` only in the letter naming the row count — and the two raise `must be (N, 5)` and `must be (K, 5)`, which two different test files match on. Merging them is therefore a test-visible behaviour change wearing the costume of a duplicate removal, and a refactor whose contract is "every existing test passes unchanged" cannot also change what an error says. It is recorded as 091d instead. A duplicate that is cheap to see is not always cheap to remove, and the difference is whether anything asserts on it.

The search also turned up `_check_rboxes` in `losses/probiou.py`, first reported as a third copy and corrected on re-reading: it guards `ndim == 0 or shape[-1] != 5` and raises `must be (..., 5)`, so it accepts arbitrary leading dimensions where both `rotated_geom` validators require exactly 2-D. Same name, same five, different contract — merging it into the other two would tighten what the loss path accepts, which is a behaviour change and not a rename. It is 091e, and its test coverage differs in the same direction: `tests/losses/test_probiou.py` matches the prefix `target must be` only, leaving the tail unpinned, where both `rotated_geom` messages are pinned tail and all. Three lookalikes, two of them the same check — the way to tell was to read each guard rather than each name.

**It did not deprecate the old name.** `pyDeprecate` is nowhere in this tree, and `warnings.warn` would have fired on both existing suites — which is not "tests pass unchanged" under any honest reading, and arms a trap for the day `filterwarnings = error` is set. The deeper reason is that `dota_eval.rotated_iou` is not a legacy alias: that module's docstring defines R18's protocol *in terms of* its overlap measure, so a caller reproducing the protocol wants the kernel where the protocol is. One function visible from two places is what `canonicalize` already is.

**It moved the argument, not just the code.** The "why there is no float64 here" section and its five-row error table went with the kernel, because that table is the *evidence* for the shift-before-expand ordering rather than commentary about it; a precision argument left behind in the module that no longer owns the arithmetic is how a constraint quietly stops being checked. Two claims in the destination's own docstring also had to change or become false — "no Python loop over boxes" and a blanket "float32 tensors" — which is the ordinary tax of moving code into prose written before it arrived.

The evidence that the move was inert is the goldens and only the goldens. `rotated_iou` sits on the scoring path, and a silently perturbed kernel would still clear the shapely oracle's 1e-4 tolerance; 20/20 unmoved, frozen subtrees included, is what says the arithmetic is bit-identical.

[WP-091b](https://borda.github.io/lucid-YOLO/work_packages/wp-091b/) had already verified there was no import cycle, from four entry orders, and left the module where it was rather than widening its own diff — which is why this is a package and not a line in that one.

