# `scripts/`

Development tooling. Nothing here ships — the four commands that do (`lucid-yolo`, `lucid-data`, `lucid-eval`, `lucid-predict`) live under `src/lucid_yolo/cli/` and cover a full tier run from an installed wheel alone (AGENTS.md sec. 2). This directory is what a developer working *in* the repository runs instead: golden regression, release gating, figure/report generation, and every repo-content check `make gate` enforces on commit.

Three subtrees, by what a script is for:

| Subtree | What lives there | Runs via |
| -- | -- | -- |
| `scripts/` (this level) | Core tooling: goldens, release gate, training-run outputs, dataset generalization | `make` targets, or direct CLI |
| `scripts/lint/` | Repo-content checkers | `.pre-commit-config.yaml` local hooks, individually callable |
| `scripts/_tests/` | Functional-core tests for every script above | `pytest` (part of `make test` / `make gate`) |

Every script below is a `main(argv) -> int` CLI with Google-style docstrings and doctested public functions (`make test` runs `--doctest-modules` over this whole directory, not just `src/`). None of them import from `github.com/ultralytics/*` — AGENTS.md's prime directive applies here exactly as it does to `src/`.

## Core (`scripts/*.py`)

| Script | Does | Invoked by | Test |
| -- | -- | -- | -- |
| `check_goldens.py` | Golden harness (WP-005): recomputes every `goldens/*.json` producer and compares against the frozen value, tolerance-checked. Also the `golden-check` manual pre-commit hook. | `make golden`, `make golden-gpu --include-gpu`, `pre-commit run --hook-stage manual golden-check` | `_tests/test_check_goldens.py` |
| `golden_producers.py` | The zero-argument, deterministic producer functions `check_goldens.py` discovers and recomputes — one per frozen golden. | Imported by `check_goldens.py`; also imported directly as a fixture source by several `tests/` suites (`tests/assign/`, `tests/data/`) | doctests only (no dedicated `_tests/test_golden_producers.py` — its producers are exercised transitively via `check_goldens.py` and the goldens it feeds) |
| `release_guard.py` | Release gate (WP-006): a candidate tag ships only if it's a shippable `v0.MINOR.PATCH` string (ADR-002: no 1.0, ever), its changelog section exists, and `make gate` is green. No-op (exit 0) when `HEAD` isn't exactly a tag. | Release WPs [HUMAN]; also the `release-guard` manual pre-commit hook | `_tests/test_release_guard.py` |
| `overfit_micro.py` | Overfit-100 integration golden (WP-040): trains the full stack on a fixed ~100-image synthetic slice and gates on train recall@0.5 ≥ 0.95. Needs an accelerator. | `make overfit`, `make overfit FREEZE=1` | `_tests/test_overfit_micro.py` (`gpu`/`data`-marked, run via `make test-gpu`) |
| `shapes_regression.py` | Synthetic-shapes generalization golden (WP-083): trains on 1,800 generated scenes, scores a held-out 200 through both decode paths. Needs an accelerator, ~2 min. | `make shapes`, `make shapes FREEZE=1` | `_tests/test_shapes_regression.py` (`gpu`/`data`-marked, run via `make test-gpu`) |
| `plot_training.py` | Renders a run's `lightning_logs/version_N/metrics.csv` as a deterministic three- or four-panel SVG for the reproduction report — fixed hashsalt, suppressed date metadata. | Release WPs, by hand | doctests only (no dedicated `_tests/test_plot_training.py`) |
| `draw_predictions.py` | Draws one checkpoint's predictions over the image they were made on — boxes, instance masks, or rotated quadrilaterals depending on task (WP-067). The release's worked example. | Release WPs, by hand | `_tests/test_draw_predictions.py` |
| `dump_debug_grid.py` | Debug visualizer (WP-015): dumps an annotated grid of `--samples` images run through the Phase-1 augmentation pipeline, for eyeballing mosaic/affine/mixup/copy-paste behaviour. | By hand, ad hoc | doctests only (no dedicated `_tests/test_dump_debug_grid.py`) |
| `absolutize_readme.py` | Rewrites the README's repo-relative links to absolute ones at packaging time, so PyPI's rendering doesn't 404 (WP-113b). Opt-in (`--ref` / `LUCID_YOLO_RELEASE_REF`), reverted after a local build. | `release.yml`, ahead of `uv build` | `_tests/test_absolutize_readme.py` |

## Lint (`scripts/lint/*.py`)

Every checker here used to be a `tests/meta/` pytest test scanning the live repo tree — a lint check wearing a unit test's clothes (WP-129/130). Each is now a standalone CLI plus a `.pre-commit-config.yaml` local hook, individually callable without running the rest of the suite: `pre-commit run <hook-id>`.

| Script | Hook id | Checks | Runs when |
| -- | -- | -- | -- |
| `audit_licenses.py` | `license-audit` | Every installed distribution's declared license is Apache-compatible; flags an undeclared license per tier; scans shipped native binaries for copyleft libraries a wheel vendors without declaring. | Every commit (`always_run` — installed-environment state can drift without a tracked-file diff) |
| `check_commit_trailers.py` | `commit-trailers` | The commit message matches AGENTS.md sec. 5: Conventional Commits subject, mandatory `WP:`/`Provenance:`/`Assumptions:`/`Gate:` trailers, provenance ids resolvable against `docs/PROVENANCE.md`. | `commit-msg` stage |
| `audit_test_doctests.py` | `test-doctest-audit` | Every non-fixture, non-`test_` module-level helper under `tests/` and `scripts/_tests/` carries a `>>>` doctest Example. | A `test_*.py` file changes, under either root |
| `audit_docs_present.py` | `docs-present` | The docs-governance register is structurally intact: required files exist, `ASSUMPTIONS.md`/`ROADMAP.md`/`DECISIONS.md`/`PROVENANCE.md` parse as contiguous registers, cross-links resolve. | A `docs/*.md`, `AGENTS.md`, or `README.md` changes |
| `audit_docs_site.py` | `docs-site` | The MkDocs Material site publishes the whole `docs/` tree: every page in the nav and vice versa, `tables` declared, identity fields matching `pyproject.toml`. | `mkdocs.yml`, `docs/**`, `pyproject.toml`, or the docs workflow changes |
| `audit_figure_captions.py` | `figure-captions` | A committed SVG's rendered caption text matches the source string that drew it — catches an edit to the XML comment that leaves the actual drawing untouched. | `docs/figures/*.svg` or `docs/REPRODUCTION_REPORT.md` changes |
| `audit_license_headers.py` | `license-headers` | `LICENSE` is Apache-2.0, `NOTICE` carries the Redmon/YOLO attribution, `README.md` carries the non-affiliation disclaimer, every `src/` file carries its SPDX header. | `LICENSE`, `NOTICE`, `README.md`, or a `src/*.py` file changes |
| `audit_version_single_source.py` | `version-single-source` | `pyproject.toml` declares the version dynamic and `lucid_yolo.__init__.__version__` is the one place it's actually written. | `pyproject.toml` or `src/lucid_yolo/__init__.py` changes |

Two more hooks live in `.pre-commit-config.yaml` under `repo: local` but audit core scripts rather than `scripts/lint/`, both `stages: [manual]` — never on a commit, only `pre-commit run --hook-stage manual <id>`:

- `golden-check` → `check_goldens.py` (already the comparison `make golden` runs)
- `release-guard` → `release_guard.py` (release-tag time only)

## Tests (`scripts/_tests/*.py`)

Every test here is the **functional core** of the script it's named for — `test_<script>.py`, script functions exercised against synthetic `tmp_path` fixtures (or, for the `scripts/lint/` checkers, loaded via `importlib.util.spec_from_file_location` and called directly). What runs against the *live* repo tree is the corresponding pre-commit hook or `make` target above, not pytest — pytest owns the functional core, the hook or target owns enforcement. A few files (`test_audit_test_doctests.py`, `test_check_goldens.py`) additionally carry one "the live tree is currently clean" sanity test, asserting the hook would pass right now.

`conftest.py` here is shared session setup for the drawing suites (`test_draw_predictions.py`): forces the `Agg` matplotlib backend before `pyplot` is imported anywhere in the session, and closes every figure a test opens so unclosed figures don't leak across tests.

This directory sits under `scripts/`, not `tests/`, because its subject does: a test of a `scripts/` module belongs beside the module it tests, the same way `tests/unit/` mirrors `src/`. `make test` and `make gate` already scan it — both pass `scripts` as a collection root alongside `src` and `tests`.
