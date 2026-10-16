# 📦 WP-091d — `refine(data): one rotated-box shape check`

> Research log · [Phase 9 — Inference and generalization](https://borda.github.io/lucid-YOLO/research/phase-09-inference/)
>
> 🔗 Depends on [WP-091c](https://borda.github.io/lucid-YOLO/work_packages/wp-091c/) · Needed by [WP-091e](https://borda.github.io/lucid-YOLO/work_packages/wp-091e/)
>
> ⬅️ [WP-091c](https://borda.github.io/lucid-YOLO/work_packages/wp-091c/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-091e](https://borda.github.io/lucid-YOLO/work_packages/wp-091e/) ➡️


## WP-091d — one letter, and what it costs to change one

<a id="wp-091d"></a>

091c declined this merge and said why: the two validators' messages are matched by two test files, so removing the duplicate changes what an error says, and a refactor contracted to leave every existing test passing cannot also do that. Done as its own package the visibility is the point rather than the obstacle. `_check_rboxes` is deleted, `rotated_iou` guards through `_check_2d`, and `tests/eval/test_dota_eval.py`'s `match=r"must be \(K, 5\)"` becomes `r"boxes_a must be \(N, 5\)"` — a *stricter* regex than the one it replaces, since it now pins the argument name as well as the shape. The other pinned message, in `tests/data/test_rotated_geom.py`, never moved: it was already the surviving spelling.

The letter itself was the only real question. `rotated_iou`'s signature does distinguish the two operands — `boxes_a` is `(M, 5)`, `boxes_b` is `(N, 5)`, and the result is `(M, N)`, so the distinction is load-bearing in the docstring. It is not load-bearing in the guard: each operand has to satisfy one and the same predicate, and a rejection message that spelled one of them `K` named a difference the check never tested. `N` reads as "some row count" in every message the module raises, which is what a shape rejection is actually saying.

What made the merge safe is narrow and worth stating, because the neighbouring lookalike fails it. The two conditions were character-identical — `ndim != 2 or shape[1] != 5` — so nothing but the string changed. `losses/probiou.py`'s guard passes arbitrary leading dimensions, and folding that one in would tighten a live contract rather than dedupe a message; that is 091e, and it is a different package for that reason and not for tidiness.

