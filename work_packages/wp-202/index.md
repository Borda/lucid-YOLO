# WP-202 — CLI correctness and data-ingest fixes

> Engineering log · [Phase 19 — Codebase audit remediation](https://borda.github.io/lucid-YOLO/engineering/phase-19-codebase-audit/)
>
> 🔗 Depends on [WP-201](https://borda.github.io/lucid-YOLO/work_packages/wp-201/) · Needed by none
>
> ⬅️ [WP-201](https://borda.github.io/lucid-YOLO/work_packages/wp-201/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-203](https://borda.github.io/lucid-YOLO/work_packages/wp-203/) ➡️


<a id="wp-202"></a>

## WP-202 — the command line and the image readers, corrected in one commit

**Batch 2 of the codebase-audit roadmap.** The audit of 2026-10-21 found the four commands accepting arguments they ignored, a help text naming fewer tasks than the command serves, and the image readers dividing every decoded value by 255 whatever the source depth. The gate is `make gate`, as for every package (AGENTS.md sec. 1, D30); it will be run before the commit is made, and this page claims no result for it.

**Commands.** `lucid-yolo` exposes only `fit` and `validate`; Lightning's `test` and `predict` are no longer subcommands. `lucid-predict` names keypoints in its help and refuses a checkpoint task it has no default size for, where it used to fall back to 640 px. `lucid-eval` takes `masks`, `split`, `variant` and `eval_backend` as unset by default, so an explicitly passed value that the checkpoint's task does not use is refused by name (`split` and `variant` apply to `obb`, `masks` to `segment`, `eval_backend`, `ann_file` and `images_dir` to every task but `obb`); it gains `--ann_file` and `--images_dir` for the COCO tiers. `lucid-data build-tiles` accepts `--data_root` as an alias of `--root`. The `torch.jit.script` FutureWarning that fvcore raises at import is filtered in `lucid_yolo/cli/__init__.py`, so usage text prints clean. An EMA error names the checkpoint path passed to the loader, and the export docstrings name `KeypointExportGraph` beside the other three task graphs.

**Data ingest.** `lucid_yolo/data/_image.py` holds one decode-and-scale pair, `decode_rgb` and `to_unit_float`, used by the COCO and YOLO readers; a 16-bit PNG scales by its own maximum, and 8-bit sources stay bit-identical. `build-tiles` refuses a source that decodes above 8 bits per channel, naming the file. `lucid-data check` resolves each split through `resolve_split`: COCO 2017's `train2017`/`val2017` pair keeps its published names, counts and annotation-versus-disk parity, and a split in any other spelling of the layout table, a `build-tiles` output among them, is checked for presence and parseability only. `--keypoints true` additionally requires each split's `person_keypoints_*.json` to exist and parse. Every malformed annotation file — unreadable, not UTF-8, not JSON, nested too deeply, an integer past the parser's digit limit, or a top level, section or image entry of the wrong type — is reported as a located failure naming the file rather than raised, in both `check` and `--verify`, and `--verify` checks a `file_name` with a subdirectory against the path the reader opens. The letterboxed reader behind `lucid-eval` and `lucid-predict` scales through the same `to_unit_float`. The COCO, YOLO and DOTA readers build `Targets` through a shared `Targets.from_quads`, and the shared data constants moved to `lucid_yolo/data/_common.py`. The datamodule's refusal of a segmentation or keypoint request on a YOLO root is reworded.

**Documentation.** `docs/DATASETS.md` quotes the new refusal wording, describes `--keypoints` and the plain-spelled-root check, and notes the `build-tiles` refusal and alias; `docs/TRAINING.md` documents `--ann_file`, `--images_dir` and the refusals; the CHANGELOG Unreleased section lists each user-visible change.

**Limits.** The declared floor `jsonargparse>=4.28` predates `auto_cli` (4.36), which `lucid-eval` and `lucid-predict` already imported before this package; the floor is left for the batch that installs the built wheel in CI. Names and counts on a root that is not in COCO 2017's spelling are not checked against published figures; a plain root missing a split falls back to the 2017 names in its report. Copies of the data constants in `ptl/module.py`, `assign/`, `losses/` and `eval/` are left for a later batch. No frozen golden is touched.

