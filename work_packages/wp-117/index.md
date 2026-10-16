# 📦 WP-117 — `test(tests): doctest every helper, document every test`

> Engineering log · [Phase 11 — Rolling, toward v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-11-rolling/)
>
> 🔗 Depends on [WP-085](https://borda.github.io/lucid-YOLO/work_packages/wp-085/), [WP-116](https://borda.github.io/lucid-YOLO/work_packages/wp-116/) · Needed by [WP-128](https://borda.github.io/lucid-YOLO/work_packages/wp-128/), [WP-129](https://borda.github.io/lucid-YOLO/work_packages/wp-129/)
>
> ⬅️ [WP-116b](https://borda.github.io/lucid-YOLO/work_packages/wp-116b/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-128](https://borda.github.io/lucid-YOLO/work_packages/wp-128/) ➡️


## WP-117 — documented and verified are not the same claim

<a id="wp-117"></a>

[WP-085](https://borda.github.io/lucid-YOLO/work_packages/wp-085/) wired `--doctest-modules` over `src`, `scripts` and `tests`, which turns a helper's `Examples:` block into an executable check the moment that block exists -- but nothing forced it to exist. A fresh `ast` count over `tests/**/test_*.py` found 261 non-fixture, non-`test_` helper functions, every one already carrying a docstring, and only 1 of the 261 carrying a `>>>` line `--doctest-modules` could actually run. Prose describing behaviour a reader had to trust, not a line pytest ever executed. 23 of 678 `test_` functions carried no docstring at all, all 23 in one file, `tests/data/test_download.py`.

Closing the gap by hand, file by file, surfaced real bugs the doctests would otherwise have hidden behind a passing suite: `_write_split`'s and `build.convert_split`'s return values auto-printed inside a `with` block under doctest's "single" exec mode, needing `_ = ...` to suppress; `AssignResult` has no `.labels` attribute, only `.target_labels`; `Trainer` has no `.deterministic` attribute; `_decode_boxes`'s hand-derived expected output was numerically wrong until computed from a real run rather than worked out on paper. Each was a doctest that failed on first execution -- exactly the outcome [WP-085](https://borda.github.io/lucid-YOLO/work_packages/wp-085/)'s infrastructure exists to produce, and exactly what a prose-only docstring would never have surfaced.

A second, unrelated inconsistency turned up mid-pass: some docstrings wrote a bare `>>>` block, others wrapped it in a proper `Examples:` header matching the Napoleon style `predict.py` and `export.py` already use. An `ast`-based script fixed 166 docstrings across 56 files in one pass -- locate each docstring's span, detect a `>>>` line with no preceding `Examples:`, insert the header and re-indent -- verified against a doctest re-run before and after rather than assumed safe.

The "8 groups across 6 files" class-regrouping half of the original scope did not survive contact with measurement: a first-shared-word `ast` scan returns ~90 candidate groups, because this suite's own descriptive-sentence naming house style is indistinguishable from genuinely flat enumeration by any heuristic tried. Split off into [WP-128](https://borda.github.io/lucid-YOLO/work_packages/wp-128/) rather than guessed at.

