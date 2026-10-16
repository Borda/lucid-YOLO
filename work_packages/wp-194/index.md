# 📦 WP-194 — `refine(tests): static checks to hooks, plain test names`

> Engineering log · [Phase 17 — R1 pose-objective parity](https://borda.github.io/lucid-YOLO/engineering/phase-17-pose-objective-parity/)
>
> 🔗 Depends on [WP-193](https://borda.github.io/lucid-YOLO/work_packages/wp-193/) · Needed by [WP-195](https://borda.github.io/lucid-YOLO/work_packages/wp-195/)
>
> ⬅️ [WP-193b](https://borda.github.io/lucid-YOLO/work_packages/wp-193b/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-195](https://borda.github.io/lucid-YOLO/work_packages/wp-195/) ➡️


## WP-194 — static checks leave pytest, test names lose their filler words

<a id="wp-194"></a>

**Pytest owns behaviour; hooks own file content.** A scan of `tests/` and `scripts/_tests/` found four tests in `tests/` that read a file as text and exercised no code, and fourteen in `scripts/_tests/` that asserted a hook passes on the live tree. The second group repeated what `lint.yml` and `make gate` already run, since both call `pre-commit run --all-files`, so those tests are deleted rather than moved. The first group moved: the matplotlib and `_viz` import boundary under `src/` is ruff `TID251` banned-api with per-file ignores for `_viz`, `scripts/`, `tests/` and `notebooks/`. A probe module holding two function-local imports showed the rule fires on both, which is the case the regex test existed for. The top-k decoder's no-suppression text pin and the two packaged-recipe pins (no `deterministic:`, `task:` outside `det_*`) became the `content-pins` hook, `scripts/lint/audit_content_pins.py`, with its functional core in `scripts/_tests/test_audit_content_pins.py`. The runtime `sys.modules` probes in `tests/viz/test_import_surface.py` stay in pytest because they import code. Two kept groups in `test_check_commit_trailers.py` also read the real register, `TestPlaceholderRows` and `TestAssumptionRegister`; they stay because they test how the parser reads that register. Removed alongside: the register-contiguity assertion, which `docs-present` already checks, and the docstring-path assertion on the trailer script.

**Names drop `a an the its their is are does that`.** 1057 test functions were renamed mechanically, and no two names collided in any file or class. The offline collection count is 2910 both before and after the rename. One name that read badly after the drop was renamed by hand: `test_rejects_row_naming_class_absent_from_data_yaml`. References in ROADMAP, ASSUMPTIONS and this log were rewritten to the new names. `Gate:` trailers in commit history still name the old ones, because history cannot be rewritten. This reverses the sentence-style naming that `audit_flat_test_groups.py` describes, at the principal's direction, and the flat-group hook stays green on the renamed tree. No guard hook stops filler words from returning, so it stays a convention.

