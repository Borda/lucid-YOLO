# 📦 WP-038 — `feat(ptl): LightningCLI entry and experiment configs`

> Engineering log · [Phase 5 — Lightning training loop](https://borda.github.io/lucid-YOLO/engineering/phase-05-training-loop/)
>
> 🔗 Depends on WP-035, WP-036, WP-037 · Needed by [WP-039](https://borda.github.io/lucid-YOLO/work_packages/wp-039/), WP-072, [WP-119](https://borda.github.io/lucid-YOLO/work_packages/wp-119/)
>
> ⬅️ [WP-033](https://borda.github.io/lucid-YOLO/work_packages/wp-033/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-039](https://borda.github.io/lucid-YOLO/work_packages/wp-039/) ➡️


## WP-038 — four defaults that only fail off the developer's machine

<a id="wp-038"></a>

Each was found by running somewhere other than a local checkout, and none is visible in a unit test.

**Strict determinism aborts on MPS.** Auto-picked MPS killed a tier run mid-step: `deterministic=True` hits `index_put_with_accumulate_mps`, which has no deterministic MPS kernel. The default is now accelerator-aware — `"warn_only"` where MPS is available, strict `True` on CPU/CUDA — and the shipped configs dropped their pins, since a config value would override the accelerator-aware default.

**Configs were not in the wheel.** They lived at the repo root and never entered the built distribution, so a wheel-only install (Colab) could not run `--config` at all. Moved to package data with a repo-root symlink so documented commands keep working, then given name resolution (`--config det_smoke` resolves onto the installed tree when no local file matches) and a default recipe, because a wheel install has no checkout to write a path into.

**Rich floods notebook output.** Lightning auto-picks `RichProgressBar` whenever `rich` is installed — it is, transitively — and Rich live rendering prints one line per refresh in a Colab cell. A newline flood every step. Default is now `tqdm.auto`, which renders a widget in notebooks and a single-line bar in terminals.

**Loggers.** An unset `trainer.logger` resolved to TensorBoard only, so a run left no plain-text metrics. It now resolves to a TensorBoard + CSV pair pinned to the same `version_N` directory — which is what makes every later `metrics.csv` analysis in this project possible at all.

