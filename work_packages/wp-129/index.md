# 📦 WP-129 — `refine(scripts): split scripts/ into core and lint, doctest audit as a hook`

> Engineering log · [Phase 11 — Rolling, toward v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-11-rolling/)
>
> 🔗 Depends on [WP-117](https://borda.github.io/lucid-YOLO/work_packages/wp-117/) · Needed by [WP-130](https://borda.github.io/lucid-YOLO/work_packages/wp-130/)
>
> ⬅️ [WP-128](https://borda.github.io/lucid-YOLO/work_packages/wp-128/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-130](https://borda.github.io/lucid-YOLO/work_packages/wp-130/) ➡️


## WP-129 — a check that scans the repo is not a unit test of it

<a id="wp-129"></a>

[WP-117](https://borda.github.io/lucid-YOLO/work_packages/wp-117/) landed its enforcement as `tests/meta/test_test_suite_quality.py`: a pytest test whose body is an `ast` walk over `tests/**/test_*.py`, asserting no helper lacks a doctest Example. Functionally that is a lint check -- it scans repo content for a house-style violation, the same shape as `audit_licenses.py` scanning the installed environment for a licence violation -- wearing a pytest test's clothes because that was the fastest place to put it at the time.

The fix is the split this project already draws for `audit_licenses.py` and `check_commit_trailers.py`: the check itself is a `scripts/` script with a `main(argv)` CLI entry point, invoked by a `.pre-commit-config.yaml` local hook; pytest keeps a meta-test exercising the script's own functions (`module_level_helpers`, `has_doctest_example`, `find_missing`) against synthetic `tmp_path` fixtures, the way `test_commit_trailers.py` exercises `validate_message` against synthetic commit messages rather than real repository history. Pytest stays scoped to the functional core -- does this function do what it claims, on cases chosen to exercise it -- and the hook owns the enforcement gate against the live tree.

Unlike `license-audit`, the new `test-doctest-audit` hook is not `always_run: true`: it scans pure repo content (`tests/**/test_*.py`), not installed-environment state that can drift without a tracked-file change, so it is gated on `files: ^tests/.*/test_.*\.py$` and only runs when a test file is actually part of the commit.

Moving the script surfaced one more inconsistency worth fixing in the same pass rather than leaving for a future WP to trip over: `scripts/` had grown to eleven files at its root with no separation between the two things a hook-invoked checker and a golden-regression producer actually are. `audit_licenses.py` and `check_commit_trailers.py` -- already hook-invoked -- move into a new `scripts/lint/` alongside the new script; `audit_licenses.py`'s own `PYPROJECT` path, resolved via `Path(__file__).resolve().parents[N]`, needed its `N` bumped for the extra directory level, caught immediately by the pre-commit hook itself failing on the next run rather than silently reading the wrong file. `check_goldens.py`, `golden_producers.py`, `shapes_regression.py` and the rest of the golden/training/release tooling stay at `scripts/`'s root: their dotted `scripts.<module>:<function>` paths are written into checked-in `goldens/**/*.json` `producer` fields as data, not just import statements, and rewriting those paths is a fixture migration across files that took minutes-each GPU runs to produce -- a materially different, materially riskier change than moving two files whose only callers are a Makefile line, a CI step and their own meta-tests.

______________________________________________________________________

