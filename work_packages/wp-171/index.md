# 📦 WP-171 — `fix(cli): entry points validate values, not only types`

> Engineering log · [Phase 15 — Audit remediation, v0.8.0](https://borda.github.io/lucid-YOLO/engineering/phase-15-audit-remediation/)
>
> 🔗 Depends on [WP-165b](https://borda.github.io/lucid-YOLO/work_packages/wp-165b/) · Needed by [WP-175](https://borda.github.io/lucid-YOLO/work_packages/wp-175/)
>
> ⬅️ [WP-170](https://borda.github.io/lucid-YOLO/work_packages/wp-170/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-172](https://borda.github.io/lucid-YOLO/work_packages/wp-172/) ➡️


## WP-171 — nine entry points that checked a type and not a domain

<a id="wp-171"></a>

One shape, nine instances: a value outside its domain reached arithmetic or dispatch instead of validation, and produced a plausible answer rather than an error. `--limit -5` scored every image *except* the last five while the banner printed the truncated count as though it were the request. `--batch_size 0` raised from a progress-bar total. A misspelled `decoder` took the one-to-many path and answered with another branch's boxes — on the oriented path, another branch's headings. `img_size=641` crashed inside the neck, because the upsampled P5 width is 42 while P4's is 41. An unknown `task` was scored as detection. `--device cuda` on a machine without CUDA was accepted, loaded the checkpoint, parsed the annotations, and failed minutes later inside the first `.to(device)` with a message about a tensor rather than about the flag.

**Three homes, so no rule is written twice.** A new public `lucid_yolo/validate.py` carries `require_at_least`, `require_in_range` and `require_one_of` in one voice — `name must <constraint>; got <value>` — and both numeric checks refuse NaN explicitly, since `value < minimum` is false for NaN. `assign/grid.py` carries `require_grid_side` and `require_grid_canvas`, because stride divisibility is architecture and belongs beside `HEAD_STRIDES`; positivity is a separate condition there, as `0 % 32` and `-640 % 32` are both zero. Where a file already had a validation voice — `MuSGD.__init__`, `orthogonalize` — the check matches its siblings rather than importing the helper.

**Refusals arrive before the work.** The `evaluate` doctest passes a nonexistent checkpoint path and still gets the flag's message rather than `FileNotFoundError`, which is the property that makes the validation worth having: the point is not that the run fails, it is that it fails at the flag, immediately, naming what to write instead.

**One placement was corrected mid-row.** The draft validated `conf_threshold` and `decoder` in `cli/predict.py`, where the audit pointed. Re-reading `cli/predict.py`'s own note — a library-owned refusal must not be restated in the command — moved both into the four library entry points, which also covers `scripts/draw_predictions.py` and direct callers. The command inherits the identical message.

**The vocabulary is read, not restated.** `task must be one of ('obb', 'detect', 'segment', 'keypoints')` takes its tuple from `DEFAULT_IMG_SIZE`'s keys rather than a hand-written list, and the now-unreachable `.get(task, 640)` fallbacks became `[task]` lookups, with a test pinning both per-task tables to the same key set and to values that are themselves usable. A vocabulary written twice drifts; the second copy is the one that goes stale.

**`pick_device`'s tests patch both probes in every case.** Asserting that `"cuda"` is refused would otherwise pass on this laptop and fail on a machine that has one — the same "green because of where it ran" failure the device flag itself produces. A separate case asserts the alternatives the refusal offers are exactly the spellings `pick_device` accepts, because a message is only useful if its list is a list of working commands.

**No existing test changed.** Verified before the guards went in: no test constructs `MuSGD` with a negative or NaN gain, no caller passes `steps < 1`, every canvas constant is stride-divisible, every `conf_threshold` in the tree is in `[0, 1]`, and no test calls `pick_device` with an accelerator name. Five test files gained cases; none had an assertion altered, loosened or skipped.

