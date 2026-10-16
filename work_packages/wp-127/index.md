# 📦 WP-127 — `refine(ptl): task and variant in the checkpoint path`

> Engineering log · [Phase 11 — Rolling, toward v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-11-rolling/)
>
> 🔗 Depends on [WP-040](https://borda.github.io/lucid-YOLO/work_packages/wp-040/) · Needed by none
>
> ⬅️ [WP-119](https://borda.github.io/lucid-YOLO/work_packages/wp-119/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-120](https://borda.github.io/lucid-YOLO/work_packages/wp-120/) ➡️


## WP-127 — a version number is not a run's identity

<a id="wp-127"></a>

Lightning's own default checkpoint path is `lightning_logs/version_N/checkpoints/epoch=X-step=Y.ckpt` -- `version_N` numbers the *run*, sequentially, across every task and scale a working tree has ever fit; nothing in the path names what the run actually trained. [WP-111](https://borda.github.io/lucid-YOLO/work_packages/wp-111/)'s re-scoring of [WP-064](https://borda.github.io/lucid-YOLO/work_packages/wp-064/)'s OBB-smoke checkpoint hit this directly: `version_10` said nothing until cross-referenced against its `hparams.yaml`'s `task: obb` field, a step that only works because the checkpoint happened to still carry its hyperparameters.

The fix stays inside `DetectionCLI.instantiate_trainer` (`cli/train.py`), the same injection seam [WP-038](https://borda.github.io/lucid-YOLO/work_packages/wp-038/) already uses for the progress bar and the default logger pair: unless a config places its own `ModelCheckpoint` in `trainer.callbacks`, one is now injected with `filename=_checkpoint_filename(self.model.task, variant)` -- a new pure helper prefixing `{task}_{variant}_` onto Lightning's own `{epoch}-{step}` template, left as a literal placeholder so `ModelCheckpoint`'s own metric-filling machinery still resolves it. `dirpath` is left at Lightning's default (`<version dir>/checkpoints`), so `lightning_logs/version_N` numbering is untouched and no already-written checkpoint moves or is renamed -- only new runs pick up the prefix. `task` comes from `self.model.task` (the instantiated module, already built by the time `instantiate_trainer` runs) rather than the raw config, since only the module's own default (`task="detect"`) is guaranteed correct once link-computed arguments are in play.

The injection duplicated the progress-bar callback's four-line "append without replacing the config's own list" pattern verbatim, so it moved into a small `_add_trainer_default_callback` method both call into, rather than being copy-pasted a second time.

