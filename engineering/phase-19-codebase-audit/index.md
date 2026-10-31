# 🔎 Phase 19 — Codebase audit remediation


<a id="wp-201"></a>

## WP-201 — the prose that was wrong, corrected in one commit

**Batch 1 of the codebase-audit roadmap.** The audit of 2026-10-21 found the contract text and the user-facing documentation disagreeing with the tree. Nothing here changes code behaviour, and the gate is unchanged: `make gate` is this package's gate, as for every package (AGENTS.md sec. 1, D30), with `make docs` (a strict build) alongside it. Per D30 the audit roadmap lands as one work package and one commit per batch, WP-201 to [WP-205](https://borda.github.io/lucid-YOLO/work_packages/wp-205/); this is the first.

**Contract record.** D30 records the principal's statement that O3's review is done, so O3 is cleared. With the repository public and [WP-125](https://borda.github.io/lucid-YOLO/work_packages/wp-125/) accepted, all three of D18's preconditions hold and D18 is in force. O3's entry under Open items is changed only in its status cell. `AGENTS.md` sec. 1 now says the relaxation is in force instead of "not in force", and `docs/CONTRIBUTING.md` conditions D18 on the same three facts. `AGENTS.md` sec. 2 no longer says the tree holds eight `gpu`/`data` marks: a recount at HEAD found 107 `pytest.mark.gpu` and 5 `pytest.mark.data` across 20 files, and `-m "not gpu and not data"` deselects 360 of 4210 collected tests, so the paragraph's conclusion about what an offline green run covers is reworded. Every other line of `AGENTS.md` is byte-identical to its pre-edit copy.

**Decided by the principal (2026-10-21).** D31 is approved: it supersedes D3's s-scale headline clause (every tier ran at n-scale; the n-scale smoke is the accepted tier), and D3 itself is not edited. The README and `docs/index.md` banner change is approved: the keypoint clause follows D22 and R1 sec. 3.4.2, and the evaluation wording is corrected too, so the loss is taken from RLE and the evaluation is COCO OKS. The banner is the text O3 reviewed, which is why it needed that approval. The install line uses `@main` as its ref, because no release tags exist.

**User-docs sweep.** The model cards state the current accelerator-golden value and tolerance for each task, read from `goldens/gpu/overfit_micro_*.json`. The README matplotlib sentence now says it is a runtime dependency, and `docs/PROVENANCE.md` R29 records the same fact (runtime since [WP-189](https://borda.github.io/lucid-YOLO/work_packages/wp-189/)). The README has one canonical Install section; its ref is `@main`, because no release tag exists. The `docs/index.md` tab count, the CHANGELOG Unreleased headings, the segmentation-report smoke-config note and the README bare doc paths are fixed; the README "Current release" paragraph names 0.8.0 and points at the CHANGELOG, and the index reading table is split into use-it and audit-it. `docs/DATASETS.md` and the README state that the YOLO text reader refuses segmentation and keypoints.

**Limits.** No hook yet asserts that a card figure equals its golden (batch 3). The CHANGELOG 0.9.0 cut stays a separate principal tag decision. `make docs` (`mkdocs build --strict`) and the full pre-commit run pass on the changed files; `make gate` is the gate for this package and no exemption is claimed.



<a id="wp-202"></a>

## WP-202 — the command line and the image readers, corrected in one commit

**Batch 2 of the codebase-audit roadmap.** The audit of 2026-10-21 found the four commands accepting arguments they ignored, a help text naming fewer tasks than the command serves, and the image readers dividing every decoded value by 255 whatever the source depth. The gate is `make gate`, as for every package (AGENTS.md sec. 1, D30); it will be run before the commit is made, and this page claims no result for it.

**Commands.** `lucid-yolo` exposes only `fit` and `validate`; Lightning's `test` and `predict` are no longer subcommands. `lucid-predict` names keypoints in its help and refuses a checkpoint task it has no default size for, where it used to fall back to 640 px. `lucid-eval` takes `masks`, `split`, `variant` and `eval_backend` as unset by default, so an explicitly passed value that the checkpoint's task does not use is refused by name (`split` and `variant` apply to `obb`, `masks` to `segment`, `eval_backend`, `ann_file` and `images_dir` to every task but `obb`); it gains `--ann_file` and `--images_dir` for the COCO tiers. `lucid-data build-tiles` accepts `--data_root` as an alias of `--root`. The `torch.jit.script` FutureWarning that fvcore raises at import is filtered in `lucid_yolo/cli/__init__.py`, so usage text prints clean. An EMA error names the checkpoint path passed to the loader, and the export docstrings name `KeypointExportGraph` beside the other three task graphs.

**Data ingest.** `lucid_yolo/data/_image.py` holds one decode-and-scale pair, `decode_rgb` and `to_unit_float`, used by the COCO and YOLO readers; a 16-bit PNG scales by its own maximum, and 8-bit sources stay bit-identical. `build-tiles` refuses a source that decodes above 8 bits per channel, naming the file. `lucid-data check` resolves each split through `resolve_split`: COCO 2017's `train2017`/`val2017` pair keeps its published names, counts and annotation-versus-disk parity, and a split in any other spelling of the layout table, a `build-tiles` output among them, is checked for presence and parseability only. `--keypoints true` additionally requires each split's `person_keypoints_*.json` to exist and parse. Every malformed annotation file — unreadable, not UTF-8, not JSON, nested too deeply, an integer past the parser's digit limit, or a top level, section or image entry of the wrong type — is reported as a located failure naming the file rather than raised, in both `check` and `--verify`, and `--verify` checks a `file_name` with a subdirectory against the path the reader opens. The letterboxed reader behind `lucid-eval` and `lucid-predict` scales through the same `to_unit_float`. The COCO, YOLO and DOTA readers build `Targets` through a shared `Targets.from_quads`, and the shared data constants moved to `lucid_yolo/data/_common.py`. The datamodule's refusal of a segmentation or keypoint request on a YOLO root is reworded.

**Documentation.** `docs/DATASETS.md` quotes the new refusal wording, describes `--keypoints` and the plain-spelled-root check, and notes the `build-tiles` refusal and alias; `docs/TRAINING.md` documents `--ann_file`, `--images_dir` and the refusals; the CHANGELOG Unreleased section lists each user-visible change.

**Limits.** The declared floor `jsonargparse>=4.28` predates `auto_cli` (4.36), which `lucid-eval` and `lucid-predict` already imported before this package; the floor is left for the batch that installs the built wheel in CI. Names and counts on a root that is not in COCO 2017's spelling are not checked against published figures; a plain root missing a split falls back to the 2017 names in its report. Copies of the data constants in `ptl/module.py`, `assign/`, `losses/` and `eval/` are left for a later batch. No frozen golden is touched.



<a id="wp-203"></a>

## WP-203 — the gate, the workflows and the test scaffolding, consolidated in one commit

**Batch 3 of the codebase-audit roadmap.** The audit of 2026-10-21 found a gate target that could skip its own check, the same helpers copied across the repo-content checkers and the test tree, and workflows that spent the critical path on serial pytest. Nothing here changes shipped behaviour except the declared `jsonargparse` floor. The gate is `make gate`, as for every package (AGENTS.md sec. 1, D30); it will be run before the commit is made, and this page claims no result for it.

**Gate and lint tooling.** `make golden` runs `scripts/check_goldens.py` unguarded, so a missing harness fails instead of printing a skip line. `make test` takes `COV` (default `--cov=lucid_yolo --cov-report=term`; a CI cell that measures nothing passes `COV=`) and `PYTEST_XARGS` (default `-n auto`). Shared helpers replace per-checker copies: `REPO_ROOT`, the `--repo-root` argument, `sorted_rglob` and the failure printer in `scripts/lint/_common.py`, and `REPO_ROOT` and one `digest` in `scripts/_common.py`; ten checkers plus `check_goldens.py` and `freeze_goldens.py` use them, and the `_load_audit` copies in the checker tests are one `load_script` in `scripts/_tests/script_loader.py`. Checker output and exit codes were diffed against HEAD over 25 invocations and are identical. The `no-ipynb-tracked` hook is now a stock `language: fail` hook on `\.ipynb$`, so `audit_no_notebooks_tracked.py` and its test are deleted and the hook runs on pre-commit.ci. Three hooks are new: `test-name-filler-adverbs` (the five padding adverbs of AGENTS.md sec. 2 in test names), `tqdm-auto-only` (over `tests/`, `scripts/` and `notebooks/`, which ruff's per-file ignores exempt) and `model-card-figures` (`scripts/lint/audit_model_card_figures.py`, the wiring-gate figure on each model card against the golden it cites). `tqdm.tqdm` joins ruff's banned-api table for `src/`. The `.pre-commit-config.yaml` comment that counted twelve local hooks no longer counts.

**Test tree.** The 45 autouse seed fixtures that only called `torch.manual_seed(0)` are deleted; `tests/conftest.py` already seeds every test at the same scope and value, and `tests/ptl/test_compile.py::isolate_dynamo`, which also resets dynamo, stays. `tests/fixtures/` gains `markers.py` (`requires_cuda`, `requires_accelerator`, the BF16 replay expectation), `determinism.py` (`deterministic_algorithms`, with two fixtures in `tests/ptl/conftest.py`), `training.py` (`LogRecorder`, `has_gradient`, `tiny_module`, `collate_unpacked`) and `tensors.py` (`assert_bytes`, `seeded_generator`), each replacing the copies the audit listed. Every in-body `pytest.skip()` became a per-case mark, so none is left under `tests/`, and the one cuda case of `test_module.py` that ran nowhere now carries `requires_cuda`. 91 sites in 18 files use the two markers. 39 test names lose a filler adverb; the one node id cited in a doc, in `docs/ROADMAP.md`, is updated, and two parametrized dtype tests change their ids (`dtypeN` to `float32` and so on). No assertion changed. On the final tree the offline selection reads 4668 of 5029 collected with 361 deselected, against 4671 of 5031 and 360 at the start of the batch. Helper consolidation alone accounts for 12 fewer selected items (19 doctest items of deleted duplicates, less 8 doctest items of the new helpers, plus the one cuda case that moved to the deselected side) and adds or removes no test function; the rest of the delta is the batch's own test changes: the model-card figure checker's tests and the fixture-cache race and lock tests are added, and the notebook checker's tests are deleted with the checker.

**Parallel suite.** `pytest-xdist>=3.5` joins the dev group and `make test` runs `-n auto`; `PYTEST_XARGS=` runs serially, which is how a failure is read, since xdist interleaves output. The synthetic fixture cache under `tests/fixtures/_generated` is rebuilt under one `fcntl.flock` per set and published by atomic rename, so a worker never reads a half-written set and a set any worker has returned is never moved away; a current set is read without taking the lock. The lock makes the test suite POSIX-only, which matches the Linux and macOS CI matrix. Wall time of `make test` on the 16-core development Mac, same tree and same 4612 passed / 47 skipped: 739 s serial (`PYTEST_XARGS=`), 207 s with `-n auto` on a warm cache, 228 s cold, 461 s at `-n 4` (the width of a hosted runner); coverage stays 96.52% under xdist. No GitHub-side timing exists yet.

**Workflows.** The test matrix is ubuntu on 3.11, 3.12 and 3.13 plus one macOS cell on 3.13 (the principal's decision), with coverage measured only on ubuntu 3.11 through `COV_ARGS`; goldens are their own `goldens` job and `testing-guardian` needs the testing, goldens, wheel and workflows jobs. A composite action, `.github/actions/wheel-smoke`, builds the check shared by `ci-tests.yml` and `release.yml`: `twine check`, a wheel install into a venv outside the checkout, the four `--help` calls and one packaged config loaded through `importlib.resources`. Every checkout drops credentials except the docs publish step; `docs.yml` splits into a read-only `build` and a `publish` job that installs only `ghp-import`; every job has a `timeout-minutes` set from the recent run history; `cache: pip` is gone from all five places. `ci-tests.yml` gains a `workflows` job running `actionlint` and `zizmor` at pinned versions through `uvx`, so a workflow-audit failure turns the required `testing-guardian` context red. New: `links.yml` (weekly lychee run over the README and `docs/`, not a required check) and `.github/dependabot.yml` (github-actions, monthly, one group, seven-day cooldown). `pyproject.toml` raises `jsonargparse[signatures]` from `>=4.28` to `>=4.36`, the first release that exports `auto_cli`, which `lucid-eval` and `lucid-predict` import; [WP-202](https://borda.github.io/lucid-YOLO/work_packages/wp-202/) had left that floor for this batch. A wheel built from this tree installed against `jsonargparse` 4.36.0 in a scratch venv printed usage for all four commands.

**Left deliberately.**

- **Lint helpers, partial.** Fence tracking is not unified: `audit_doctest_fences` follows language-aware fences with offsets and `audit_wp_links` toggles on three-backtick and tilde fences with a quote prefix, and merging them risks different output for no line saved. `REPO_ROOT` stays local in `audit_frozen_manifest`, `check_commit_trailers`, `absolutize_readme`, `golden_producers`, `overfit_micro`, `release_guard` and `shapes_regression`, where the bootstrap needed to import the helper costs more than it removes; `audit_frozen_manifest` and `check_commit_trailers` keep their own failure wording. `audit_frozen_manifest._load` and the `audit_licenses` fixture keep their own loaders.
- **Cross-task test stories** are not merged: `tiny_module` takes the task and its keyword arguments, and the obb, segment and keypoint files keep a thin binding of their own gains.
- **Skip gaps.** The four `test_overfit_micro.py` tests are `gpu` and `data` with no skip, the notebooks full run (`test_runs_in_full`) is the same, and the mps case in `test_module.py` is a skip with no `gpu` mark; all three are outside what this batch scoped and stay as they were.
- **Kept after review.** The `img_size` default pinned in `tests/predict/test_predict_obb.py`, the signature-bound spy in `tests/eval/test_eval_obb.py` and the `backend` default in `tests/eval/test_coco_eval.py` assert documented public constants or library-boundary defaults and were not rewritten as behaviour tests; the principal may revisit the first.
- **CI proof is local.** `actionlint` and `zizmor` are clean and the wheel-smoke steps ran by hand, but no GitHub-side run exists: the matrix `include`, the composite action, the `COV_ARGS` expression, the docs artifact hop and lychee are untested live. The dropped macOS 3.11 and 3.12 cells must not be required checks in branch protection; `testing-guardian` is the documented required one.

No frozen golden is touched.



<a id="wp-204"></a>

## WP-204 — one decode composition, one single-head model base, an export command and a Python API, consolidated in one commit

**Batch 4 of the codebase-audit roadmap.** The audit of 2026-10-21 found the per-task decode composed three times (prediction, evaluation, export), four near-identical single-head model classes with four deploy views, an exporter reachable only from a test, and no documented way to use a checkpoint from Python. The gate is `make gate`, as for every package (AGENTS.md sec. 1, D30); it will be run before the commit is made, and this page claims no result for it.

**Decode.** `decode/per_task.py` holds `decode_with_coefficients`, `decode_with_keypoints` and `gather_keypoint_sets`, and `decode/common.py` the constants `MAX_DETECTIONS` (300), `EVAL_CONF_THRESHOLD`, `DEFAULT_IOU_THRESHOLD` and `COCO_IMG_SIZE`, one `gather_by_anchor` that clamps a padding index and zeroes its row, and `points_to_letterboxed_original`. `RotatedTopKDecoder` pads to `k` rows and returns anchor indices, as `TopKDecoder` does. `predict.py` replaces four copied preambles with one `_Prepared` and one indexed decode; `export.py` and `eval/coco_eval.py` call the same functions, and `gather_keypoints` delegates to `gather_by_anchor` under its own parameter names, so a call by keyword still binds. A real-case oracle compared 42 outputs of random-initialised modules before and after — the four `predict_*` functions at two canvas sides, both decode paths and two confidence thresholds, the evaluator's batch prediction for three tasks, and the export graphs — and 41 were `torch.equal`; the one difference is the oriented export below.

**Export.** The three single-head models share a private `_SingleHeadModel` with one `deploy()` view, `_DeployedO2O`, in place of four `_Deployed*` classes; across five scales and four tasks the state-dict keys, module names, parameter counts and FLOPs are identical, and the 50 goldens pass unchanged. `variant` on the four task models is now `str | ScaleSpec`, which `export.graph_from_module(module, canvas)` needs to rebuild a checkpoint at its own multipliers; it checks the keys strictly, and a training-only `rle_loss.` prefix is the one thing it drops. `lucid-export` is the fifth command (`cli/export.py`) and needs the `onnx` package, which torch 2.14.1's legacy exporter imports unconditionally after the trace: with `onnx` hidden from the import system `export_graph` failed with `OnnxExporterError: Module onnx is not installed!`, so the dependency is the `export` extra and `onnxruntime` stays dev-only. The wheel-smoke action loops over five commands. Four export docstring paragraphs that described the pre-batch behaviour were corrected.

**Python API.** `eval/coco_score.py` factors the COCO scoring that `detect_eval.run` and `pose_eval.run` each carried into `score_coco`, which returns a `CocoScore` and prints nothing; both runners are thin wrappers, and stdout, exit code and JSON of six cases were identical before and after. The four `predict_*` functions take a file path or a decoded `(3, H, W)` tensor through one `letterbox_image_tensor`, and equal results were asserted for the file and both tensor forms. The package re-exports the loader, the predictors and the scorer lazily; `import lucid_yolo` was measured at 2062 µs and 64 modules against 164 µs and 53, with torch and Lightning still unloaded. `load_eval_module` reads a checkpoint once and builds the module from the payload through Lightning's public steps, so the two-read design its docstring justified is gone, and the user-facing explanation in `docs/TRAINING.md` is rewritten. No register row recorded that rationale. `lucid-yolo fit --ckpt_path` resume was tested through the command: step and epoch, the EMA shadow and count, the MuSGD momentum and the progressive-loss weight all continue.

**Documentation.** The training page splits into a recipe and an [opt-in CUDA performance page](https://borda.github.io/lucid-YOLO/advanced/cuda-performance/), and gains sections on resuming, using a checkpoint from Python and exporting to ONNX. The API tabs gain an export page and the decode helpers, and `inherited_members: [forward, deploy]` keeps those two methods rendering under `Detector`, `OrientedDetector` and `KeypointDetector` after the base-class move. The README lists five commands and a "Not here, and why" table.

**Limits and deferred.**

- **The `coco_eval.py` split moved to [WP-205](https://borda.github.io/lucid-YOLO/work_packages/wp-205/).** Separating the streaming scorers and the keypoint scoring from the bbox and segm functions was planned and not done; its test patch of `coco_eval.MeanAveragePrecision` must move with `_new_metric`.
- **Three literals stay.** `_DEFAULT_TOPK = 300` in `models/heads/detect.py` and `models/heads/obb.py`, and `_DEFAULT_IMG_SIZE = 640` in `models/build.py`, cannot import the shared constants: `models/` sits below `decode/` and any `lucid_yolo.decode` import runs `decode/__init__`, which imports `models.heads.detect` while still partially initialised. Two tests pin them equal through public signatures.
- **A segmentation export still refuses canvases under 300 anchors** (96 px a side and below). The shared gather would now give such a row an empty mask rather than fail, so lifting the refusal is possible; it is a behaviour decision for the principal.
- **No dynamic batch or canvas** in the exported graph, and no format other than ONNX at opset 17 through the legacy exporter.
- **Tensor predict takes one image.** Each image letterboxes to its own size and inverse, so a batch does not follow from the single-image path, and a batch is refused.
- **CS-10 behaviour change.** The oriented ONNX graph on a canvas under 300 anchors returns `(B, 300, 7)` instead of `(B, A, 7)`; rows `0..A-1` are byte-identical and the rest are zero. `predict_oriented` and `lucid-predict` are unchanged on every canvas, because their confidence filter drops the padding rows. The three callers of `o2o_rotated_topk` outside this batch's files stay unpadded and unaffected.
- **Scoring engines differ.** `score_coco` defaults to `faster_coco_eval` and `lucid-eval` to `auto`, which uses `hotcoco` when usable and `faster_coco_eval` otherwise; `score_coco` accepts only those two names, so a Python caller passes `eval_backend="hotcoco"` explicitly to match the command.
- **Checked on CPU only for the new paths.** The tensor predict input and the single-read loader were exercised on CPU, and the loader against Lightning 2.6.6, not the 2.4 floor.

No frozen golden is touched.



<a id="wp-205"></a>

## WP-205 — one task registry, three oversized files split by responsibility, and the small duplications removed, consolidated in one commit

**Batch 5 of the codebase-audit roadmap, the CPU part only.** The audit of 2026-10-21 found the closed task set `detect|segment|obb|keypoints` restated as name constants and per-task tables in five modules, `ptl/module.py` (3042 lines), `ptl/datamodule.py` (2065) and `eval/coco_eval.py` (1763) each holding several responsibilities, and a handful of local duplications inside them. The gate is `make gate`, as for every package (AGENTS.md sec. 1, D30); it will be run before the commit is made, and this page claims no result for it.

**Task registry.** `lucid_yolo/tasks.py` is a leaf module (standard library only; a fresh interpreter importing it loads neither torch nor numpy). `Task` is a `StrEnum` of the four tasks in the order `DetectionLitModule` already listed them, and `Task.spec` returns a frozen `TaskSpec` holding the default image side (640, and 1024 for oriented boxes) and the default `lucid-eval` batch size (32, and 8 for oriented boxes). `StrEnum` rather than a plain `(str, Enum)` because on Python 3.11 the latter formats as `Task.DETECT`, which would poison any f-string a leaked member reached. The tables that used to restate these facts now derive from the registry: `DEFAULT_IMG_SIZE` and `DEFAULT_BATCH_SIZE` in `cli/eval.py` (same keys, values and key order, oriented first, because two refusal messages print that order), `DEFAULT_ORIENTED_IMG_SIZE` and the four `_*_TASK` constants and entry-point table in `predict.py`, the three constants in `cli/predict.py`, the renderer and ground-truth tables and the image-size lookup in `_viz/overlay.py` (which no longer imports a CLI module), and `_TASKS` in `ptl/module.py`. The plain string stays at every boundary. `hparams["task"]` stays a `str` in the saved checkpoint, including when a subclass forwards a `Task` member to `DetectionLitModule.__init__`: Lightning lets a subclass's constructor locals overwrite the base's, so a normalisation placed before `save_hyperparameters()` was undone, and the member then failed `torch.load(..., weights_only=True)`. The base class now rewrites the recorded value and retakes the initial snapshot after the call (`_record_plain_task`), and six tests cover the named-parameter and `**kwargs` forms. The residual: a subclass that calls `save_hyperparameters()` again itself, after `super().__init__()`, recollects its own frame and can record the member; the base constructor cannot reach that call.

**`ptl/module.py`.** The padding and batch helpers (`pad_targets`, `pad_rboxes`, `pad_keypoints`, `StepBatch`, `_host_split`, `_split_batch` and their constants) moved to `ptl/_padding.py`, and the module-level keypoint-loss helpers (`normalize_keypoints_to_box`, `_StepContext`, `_KeypointPositives`, the OKS sigma lookup) to `ptl/_task_terms.py`; each was sliced verbatim and re-imported, `__all__` is unchanged, and `ptl/__init__.py` imports `normalize_keypoints_to_box` from its new home because mypy strict does not follow an implicit re-export. Two local simplifications: `_pad_instance_axis` replaces the five zero-then-fill loops over boxes, labels, oriented boxes, keypoint coordinates and visibility (the `if count` guard stays, since assigning an empty `(0, 0, 2)` keypoint set into a `(0, K, 2)` slot raises), and `_blend` replaces the expression `alpha * o2m + (1.0 - alpha) * o2o` at eight sites, written verbatim in the helper. `setup` and `validation_step` were shortened by extracting private helpers (about 99 and 74 lines before, 64 and 61 after, each measured from its `def` to the next); the capture-ignored warning keeps its order relative to the activation flag and its `stacklevel`, both pinned by tests. The per-task loss-term methods stay on the class: replay certification compares method identity (`_ORIENTED_METHODS`, `_EAGER_POSE_SCORING`) and `_oriented_helpers()` compares the module's own bindings of `decode_ltrb`, `decode_rboxes` and `oriented_branch_terms`, which three tests patch, so moving the callers would turn those patches into no-ops. They belong with the replay consolidation.

**`ptl/datamodule.py`.** The IPC transport (`PackedTargets`, `pack_targets`, `unpack_*`, the collate functions, `unpack_batch`) moved to `ptl/_transport.py` and the worker pipeline (`_DetectionSource`, `_init_worker`, `_PreparedTrainingSource`, `_TrainPipeline`) to `ptl/_pipeline.py`; `datamodule` imports from both and re-exports every name a test or script reads, so `lucid_yolo.ptl.PackedTargets` is the same object as before. Names the datamodule no longer looks up (`_jitter_hsv_batch`, `get_worker_info`) are deliberately not re-exported, so a stale patch on `datamodule` raises `AttributeError` instead of passing silently; the tests that patched them now target the module that looks them up. The `pack_targets` docstring said eleven dense tensors and the dataclass has twelve without a default (DP-21); it now says twelve and names the two optional fields the collate fills. `Targets._shared_keypoint_count` is the one place a batch's keypoint count `K` is read and checked; `Targets._concat_keypoints` and the transport's `_concat_keypoints` both call it. **Behaviour change:** a batch whose images carry different `K` now raises `ValueError` naming both counts from `pack_targets` and `collate_detection`; it used to raise `RuntimeError` from `torch.cat`. No test pinned the old type, and the output for every input the old code accepted is unchanged. `pad_keypoints` in the padding module still carries its own check.

**`eval/coco_eval.py`.** The split carried over from [WP-204](https://borda.github.io/lucid-YOLO/work_packages/wp-204/) landed: the streaming scorers and the hotcoco path moved to `eval/_coco_streaming.py`, the OKS keypoint scoring to `eval/_coco_keypoints.py`, by line-range slices checked contiguous and unduplicated. `coco_eval` keeps the bbox and segm functions and the orchestrator, `__all__` is byte-identical, and the names tests and callers read are re-exported. `MeanAveragePrecision` is deliberately not re-exported: nothing in `coco_eval` uses it after the move, so a patch of `coco_eval.MeanAveragePrecision` fails loudly (it did, on the unretargeted test) rather than becoming a silent no-op; that test now patches `_coco_streaming`. `gather_keypoints` stays in `coco_eval`: it is a decode gather delegating to `gather_by_anchor`, not keypoint scoring, and leaving it keeps the references in the keypoints model card, two docstrings and a patch site exact. `docs/report/keypoints.md` now points `evaluate_keypoints` at `eval/_coco_keypoints.py`.

**Size.** Line counts at HEAD and in the working tree, from `git show HEAD:<file> | wc -l` and `wc -l`:

| File | Before | After |
| -- | -- | -- |
| `ptl/module.py` | 3042 | 2697 |
| `ptl/datamodule.py` | 2065 | 1234 |
| `eval/coco_eval.py` | 1763 | 873 |
| `ptl/_padding.py` | new | 337 |
| `ptl/_task_terms.py` | new | 238 |
| `ptl/_transport.py` | new | 588 |
| `ptl/_pipeline.py` | new | 323 |
| `eval/_coco_streaming.py` | new | 522 |
| `eval/_coco_keypoints.py` | new | 451 |
| `tasks.py` | new | 132 |
| `data/targets.py` | 523 | 564 |

The three split files lose 3,636 lines between them and gain 2,458 in the six new modules and 41 in `targets.py` (the helper and its doctest); headers and import blocks account for most of the difference from a pure move.

**Verification the batch's agents recorded.** Each moved block was compared against the original rather than trusted: the padding and transport helpers and the coco_eval blocks by script against the pre-change text, `_pad_instance_axis` and `_blend` by old-against-new execution (160 random ragged batches `torch.equal` for the padding, and a two-image training step with every loss term and every parameter gradient `torch.equal` for detect, segment, oriented and keypoints, including the paired pose path). `make golden` ran 50/50 on the final trees of the module and datamodule steps. Every patch site on a moved name was enumerated by grep, and the retargeted ones were shown to fail when their guard was disabled.

**Coverage (T8).** The audit measured CI coverage at 94.65% against `fail_under = 96`. The local gate at the previous commit measured 96.56%. This batch adds regression tests for what it changed and no tests aimed at coverage, and it moves code between files without adding branches, so no change is expected; the figure on CI's ubuntu cell has not been measured for this tree and is not verified without a push. The batch does not claim to have closed T8.

**Limits and deferred.**

- **Batch 5b is not part of this row.** The replay consolidation (C-REPLAY, with the guarded lifecycle D5 and the stream-drain contract D6, and the P5 per-flag criteria) needs the principal's answer on gradient accumulation across the replay flags (the 2026-10-20 entry in [ESCALATION](https://borda.github.io/lucid-YOLO/ESCALATION/)) and CUDA parity runs. It has no number yet.
- **Moving the capture tests (T6)** to a `tests/ptl/capture/` directory waits on the same consolidation.
- **C-TASKS step 2 is not done.** The per-task strategy objects, and with them the `task == "obb"` style comparisons that remain in `ptl/module.py`, `cli/eval.py`, `cli/train.py` and `export.py`, stay as they were.
- **A fifth task is not yet one registry row.** `lucid-eval` derives its size and batch tables from the registry, so a fifth `Task` member would be accepted there and reach the final dispatch branch, `detect_eval.run`, which has no task guard and would score it as detection. No current input triggers it; a test pins exactly four members and its docstring lists every site a fifth task must extend. A guard in `cli/eval.py` belongs with step 2.
- **`lucid_yolo.tasks` is not documented.** It has no `docs/api` entry and is not re-exported from the top-level package, so this batch adds no CHANGELOG line for it.
- **Loss-input records (CS-17, DP-18) are batch 6**, and `decode/common.COCO_IMG_SIZE`, `models/build._DEFAULT_IMG_SIZE` and the restated subset in `eval/coco_score.py` still hold their own copies of the registry's facts, pinned equal by tests where one exists.
- **The oriented `detections.cpu()` copy stays in `validation_step`.** `preds` is still built under the `obb` task, where `val/mAP` is never updated, to keep behaviour identical; removing the device-to-host copy is a performance change for a later batch.
- **Not run on this host.** The two fused-HSV tests whose patches were retargeted to `_transport` (`test_literal_pixels_and_single_restore`, `test_transfer_hook_uses_fusion_and_keeps_targets`) need CUDA and Triton and were only exercised through a CPU stand-in, and no CUDA-marked capture test ran (243 skips in the module step's run), so `make gate-gpu` is the run that covers the replay paths and those two tests.
- **Moved docstrings keep their relative cross-references.** A few Sphinx roles in the new private modules name a sibling that no longer resolves module-locally; none of those modules is rendered on a `docs/api` page.

No frozen golden is touched.







<a id="wp-210"></a>

## WP-210 — capture tests in one package with shared scaffolds and private-state access

**Finding.** Audit T6 found nine CUDA replay test files mixed into `tests/ptl/`, repeated CPU routing scaffolds, scattered reads of private replay-holder state and inline recorder patches beside an inherited recording spy.

**Move.** The nine files retain their names under `tests/ptl/capture/`, without an `__init__.py`. The parent `tests/ptl/conftest.py` stays unchanged and its fixtures remain inherited. Live references follow the move; historical work-package prose retains its recorded paths.

**Shared scaffolds and helpers.** `tests/fixtures/capture_access.py` owns the identical CUDA-flagged tensor implementation, its image construction, the precision-query refusal, the ordered CUDA stream fence and the private-state accessors, imported by bare module name like the neighbouring fixture helpers. Five suites apply the inherited `deterministic_kernels` fixture through module-level `pytestmark`; the segmentation-network suite applies its distinct `deterministic_kernels_pinning_cudnn_benchmark` fixture the same way. No capture conftest or re-wrapped autouse fixture remains. Local image helper entry points retain their original doctests, delegating construction to the shared implementation so no collected case disappears. Constructors, byte comparisons and nested state comparisons with differing bodies stay local. Accessors centralize reads of the loss objective, network holder, pose child holders and head topology; deliberate private-state writes remain visible in the setup that needs them. Every behavioural byte and gradient equality assertion is retained.

**Recorder patches.** Four absence-only patches in `test_loss_capture.py` become call-count observations through the inherited `graph_recordings` fixture. Three distinct substitutions stay local: `_host_recorder` replaces CUDA with CPU autograd, the recording-failure test injects an exception, and the GC-policy test samples collection state inside the recording call. A wrapping call spy alone cannot provide those substitutions or the GC observation.

**Baseline CPU durations.** Before edits, the nine files passed 429 tests with 206 skipped in 25.28 s. The requested `--durations=25` output's final 40 lines contain the skip summary, so a second unchanged run retained the complete timing output: 429 passed, 206 skipped, 26 warnings in 24.81 s. These are CPU timings with CUDA cases skipped, not accelerator performance measurements. Suite names below refer to the original `tests/ptl/test_<suite>.py` paths.

| Seconds | Phase | Suite | Test |
| -- | -- | -- | -- |
| 1.48 | call | `network_capture_guards` | `TestPrecisionPolicy::test_new_api_policy_reads_and_distinguishes_values` |
| 1.05 | call | `pose_network_capture` | `tests.ptl.test_pose_network_capture._assert_same_gradients_and_state` |
| 0.99 | call | `segment_head_capture` | `test_old_checkpoint_loads_with_native_default` |
| 0.57 | call | `segment_head_capture` | `test_inactive_request_matches_native_train_forward_and_gradients[True]` |
| 0.43 | call | `segment_head_capture` | `test_inactive_request_matches_native_train_forward_and_gradients[False]` |
| 0.40 | call | `network_capture_guards` | `TestStageModuleBreach::test_names_cause[compiled]` |
| 0.38 | call | `network_capture` | `tests.ptl.test_network_capture._fit_and_discard` |
| 0.25 | call | `segment_head_capture` | `test_cpu_fit_wires_lifecycle_and_warns_once_without_replay` |
| 0.19 | call | `segment_network_capture` | `test_segment_host_routing_and_native_fallback[inactive]` |
| 0.17 | call | `segment_network_capture` | `test_frozen_segment_features_retire_before_cuda_submission` |
| 0.17 | call | `loss_capture` | `test_images_without_targets_match_native[cpu-first-image]` |
| 0.17 | call | `loss_capture` | `test_images_without_targets_match_native[cpu-every-image]` |
| 0.17 | call | `pose_network_capture` | `test_pose_routing_falls_back_when_features_are_absent` |
| 0.17 | call | `segment_network_capture` | `test_segment_head_hook_retires_feature_replay_and_stays_observable` |
| 0.17 | call | `segment_network_capture` | `test_segmentation_cli_keeps_feature_and_prediction_options_independent[True]` |
| 0.16 | call | `loss_capture` | `test_images_without_targets_match_native[cpu-second-image]` |
| 0.15 | call | `segment_network_capture` | `test_segment_host_routing_and_native_fallback[record]` |
| 0.15 | call | `oriented_network_capture` | `test_oriented_routing_leaves_cpu_images_native` |
| 0.14 | call | `oriented_network_capture` | `test_oriented_routing_retires_then_runs_native[objective-override]` |
| 0.14 | call | `pose_network_capture` | `test_pose_head_certificate_survives_copy_and_rejects_dispatch_overrides[forward-branch]` |
| 0.13 | call | `network_capture` | `test_cli_accepts_network_capture_option[False-True]` |
| 0.13 | call | `pose_network_capture` | `test_altered_pose_head_dispatch_fails_certificate[child-swap]` |
| 0.13 | call | `pose_network_capture` | `test_pose_routing_dispatches_shared_features_to_native_head` |
| 0.12 | call | `oriented_network_capture` | `test_oriented_default_and_cpu_fallback[validate]` |
| 0.11 | setup | `segment_head_capture_guards` | `TestRunnerCertificate::test_nontraining_calls_keep_native_without_retiring[no_grad]` |

**Verification.**

- **Collection.** Before: 4014 tests. After: 4022, the eight added being the doctests of the new `tests/fixtures/capture_access.py` helpers; no test is removed or renamed.
- **CPU.** `make gate` green on this commit's tree.
- **Assertions.** An AST comparison against the parent commit retains all 572 original assertions, every test function name, parametrization, capability marker and class decorator; four recorder call-count assertions are added.
- **CUDA.** `make gate-gpu` green on an A100 (py3.11, torch 2.9.1+cu129) at the stack's tip `72b5505`, the tree [WP-218](https://borda.github.io/lucid-YOLO/work_packages/wp-218/) left: 359 passed and 2 skipped (a `grad_dtype` case that needs a newer torch, and a replay test that needs two CUDA devices), every notebook in full, `demo_obb` on DOTA-v1.0 among them, and 55/55 accelerator goldens. `tests/ptl -m "gpu or data"` also passed on Colab's py3.13 / torch 2.11 (242 passed). The same selection on this package's own commit, `a1c72f4`, passed in that session too (242 passed). The moved suites run in both.



<a id="wp-211"></a>

## WP-211 — per-task strategies with exhaustive CLI dispatch

**Finding.** Audit C-TASKS step 2 found that the task registry declared the supported tasks while the Lightning module and three commands still selected behavior through task-string comparisons. A fifth task could therefore inherit detection behavior without an explicit implementation.

**Strategies.** One stateless private object per task lives in `ptl/_task_strategies.py`, selected once after task validation. The single `DetectionLitModule` and `Trainer` contract stays intact. Strategies own padding extras, extra-loss composition and logging, replay routing, metric construction and task-specific validation; registered modules, replay holders and accumulators remain on the Lightning module.

| Task | Target extras / forward | Extra loss and logs | Replay route | Validation |
| -- | -- | -- | -- | -- |
| `detect` | Boxes only | Scalar zero | Complete network; eager or compiled detection objective | Box mAP, distributed metric |
| `segment` | Mask/prototype forward | Mask + semantic terms | Existing segmentation features and head routes | Box + mask mAP, distributed metrics |
| `obb` | Padded rotated boxes | Rotated box + stride-normalized L1 + angle terms | Certified complete oriented network | Rotated mAP, rank-local lists; box metric stays unused |
| `keypoints` | Padded coordinates and visibility | Keypoint + OKS terms | Shared features and pose stems; certified training loss capture | Box mAP + OKS mAP, rank-local point lists |

**Compatibility.** Tensor arithmetic, operation order, dtypes, log keys and initialization order are retained. The task remains a plain string in hyperparameters and checkpoints; strategies register no parameters or buffers. Module methods checked by replay certificates remain the call-through boundary; the oriented certificate also pins the new strategy instance and extra-loss implementation, so changing that consumer cannot silently retain full-network replay. Segmentation retains its collective mask-presence decision at epoch end.

**Fifth-task guard.** Explicit `Task`-keyed tables select evaluation protocols and applicable options, prediction handlers and the three training-loader target flags. Exhaustiveness tests compare every table and the strategy mapping with `set(Task)`; module-resolution cases pin each task to its own strategy class. Prediction handlers retain the existing report formats and evaluation resolves the existing protocol functions at call time.

**Survivors.** No `self._task` or `self.task` equality, inequality or membership comparison remains in `ptl/module.py`. Constructor-local task checks remain to preserve head/loss registration and RNG order; argument validation remains before strategy lookup. These are construction and validation boundaries rather than step-time dispatch.

**Verification.**

- **CPU.** `make gate` green on this commit's tree, the strategy and CLI table exhaustiveness tests, the copy and pickle singleton tests and the oriented-certificate tests among it.
- **Parity.** Before/after CPU comparison covers all four tasks with ragged positive and empty batches: initial/final state dictionaries, hyperparameters, losses, raw outputs, ordered logs and gradients are byte-identical in all eight cases. Capture flags are disabled in this experiment; the existing replay suites cover CPU routing, and CUDA parity remains pending.
- **Goldens.** CPU sweep: 50/50 passed; no golden file is edited.
- **CUDA.** `make gate-gpu` green on an A100 (py3.11, torch 2.9.1+cu129) at the stack's tip `72b5505`, the tree [WP-218](https://borda.github.io/lucid-YOLO/work_packages/wp-218/) left: 359 passed and 2 skipped (a `grad_dtype` case that needs a newer torch, and a replay test that needs two CUDA devices), every notebook in full, `demo_obb` on DOTA-v1.0 among them, and 55/55 accelerator goldens. `tests/ptl -m "gpu or data"` also passed on Colab's py3.13 / torch 2.11 (242 passed). The same selection on this package's own commit, `c50ed22`, passed in that session too (242 passed).
- **Release notes.** No changelog entry: this package changes no user-visible behavior.



<a id="wp-212"></a>

## WP-212 — named loss inputs, recorded alpha

**Findings.** CS-17/DP-18 identified positional tensor groups in dual-branch scoring and loss capture, with four copies of the native fallback. CS-21 found that an explicit tensor override drove the total while `DualLossOutput.alpha` recorded the attribute. C-SMALL also identified long capture routing and sequential replay guards.

**Inputs and routing.** Public `BranchPrediction` pairs each branch's logits and decoded boxes. Private `_LossInputs` names the loss recorder's ten inputs in their existing order; the CUDA boundary still receives a flat tuple of tensors. One native fallback handles ineligible loss calls. Replay output ownership is separate from workspace preparation, and segmentation runner checks stop at the first reason while retaining the stage-replacement warning's existing precedence.

**Recorded weight.** Without an override, `DualLossOutput.alpha` remains the attribute float. With an override, it is the supplied tensor, including on replay, without a host read inside compilation or capture. The detection blend keeps its tensor expressions, operation order and dtypes. Extra task losses cast the recorded tensor weight per branch after forming its complement in double precision, preserving the former float-weight rounding and scalar dtype. Log keys, hyperparameters and state dictionaries are unchanged.

**Verification.**

- **CPU.** `make gate` green on this commit's tree. Regression: `test_tensor_alpha_blends_bit_equal_to_attribute_and_records_override`; capture contracts: `test_recorder_preserves_flat_input_order` and `test_replay_records_live_alpha_override`.
- **Goldens.** CPU sweep: 50/50 passed; no golden file is edited.
- **CUDA.** `make gate-gpu` green on an A100 (py3.11, torch 2.9.1+cu129) at the stack's tip `72b5505`, the tree [WP-218](https://borda.github.io/lucid-YOLO/work_packages/wp-218/) left: 359 passed and 2 skipped (a `grad_dtype` case that needs a newer torch, and a replay test that needs two CUDA devices), every notebook in full, `demo_obb` on DOTA-v1.0 among them, and 55/55 accelerator goldens. `tests/ptl -m "gpu or data"` also passed on Colab's py3.13 / torch 2.11 (242 passed). The same selection on this package's own commit, `c7e2ad7`, passed in that session too (242 passed). Both include the loss replay and compiled-step parity tests that exercise the tensor `alpha` blend.



<a id="wp-213"></a>

## WP-213 — distributed obb and keypoint validation

**Finding.** U6 found rotated and keypoint validation accumulated rank-local Python lists. Their strategies therefore refused multi-process setup rather than log each rank's shard as the split's AP. Averaging rank-local AP cannot reconstruct the whole split's confidence ranking.

**Change.** Private per-task torchmetrics metrics gather lossless tensor records through list states with `dist_reduce_fx="cat"`, then call the existing rotated mAP and OKS AP scorers. Image identities retain their first occurrence so DistributedSampler padding counts once. Direct metric callers retain input order; the module restores its existing sequential dataset-position order after gathering, preserving tied-score ordering. Private validation target and packed-target subclasses carry dataset positions across worker and device transfer without changing public signatures. All ranks participate when any rank has images; reset clears epoch state. The obb/keypoints setup refusal is lifted; unrelated replay and prefetch restrictions remain.

**Verification.**

- **CPU.** `make gate` green on this commit's tree. Regressions: `test_single_process_matches_list_accumulator`, `test_deduplication_keeps_first_image_and_input_order`, `test_record_transport_preserves_tensor_precision`, `test_validation_identity_survives_worker_and_device_transport` and `test_module_metric_retains_validation_identity`. The former refusal test becomes `test_setup_accepts_run`, whose multi-process rows cover fit and standalone validation for both tasks.
- **Distributed.** The multi-device proof is a CPU gloo two-process test, `test_gloo_matches_full_validation_set`: both tasks compare padded shards, conflicting duplicate predictions and an empty rank against full-set scoring with exact float equality. It runs in `make gate` (CPU-only, no `gpu` mark) and passes there.
- **Accelerators.** `make gate-gpu` green on an A100 (py3.11, torch 2.9.1+cu129) at the stack's tip `72b5505`, the tree [WP-218](https://borda.github.io/lucid-YOLO/work_packages/wp-218/) left: 359 passed and 2 skipped (a `grad_dtype` case that needs a newer torch, and a replay test that needs two CUDA devices), every notebook in full, `demo_obb` on DOTA-v1.0 among them, and 55/55 accelerator goldens. `tests/ptl -m "gpu or data"` also passed on Colab's py3.13 / torch 2.11 (242 passed). The same selection on this package's own commit, `197b846`, passed in that session too (242 passed). The single-device goldens stayed bit-exact. No two-GPU run was made, so the NCCL path itself is untested: the gloo proxy above covers the same gather and deduplication on CPU, and a real multi-GPU check remains outstanding.



<a id="wp-214"></a>

## WP-214 — fewer host syncs, chunked mask upsample

**Finding.** U8 found segmentation decode resized every retained mask to the full canvas together, creating full-resolution float temporaries for up to 300 detections. U9 found six count-tensor reads during batch restoration, followed by per-image writes to pad targets already transported as flat tensors.

**Change.** Eager mask decode retains the original prototype assembly and sigmoid, then resizes, crops and thresholds chunks of at most 32 masks into the final boolean stack. Upsampling still precedes cropping, preserving border numerics. ONNX export retains the original unchunked single-resize operations. Collation retains host count tuples and flat destination indices; restoration uses the tuples without reading device count tensors. Private consumer lists retain packed storage for one scatter per padded field, falling back to ordinary padding after target replacement or reordering. Mask padding scatters one float view of the packed masks, preserving caller edits and rebuilding from current slices after replacement. Existing count tensor fields, ragged-list APIs and rotated/keypoint validation remain available. Loss-replay synchronization is unchanged.

**Verification.**

- **CPU.** `make gate` green on this commit's tree. `test_collate_batch_small_constant_segment_count` pins 14 transported tensors, one more than before: the flat padded-row indices, one tensor per batch whatever the instance count. Regressions: `test_chunked_masks_equal_full_canvas_decode`, `test_mask_upsample_bounds_instance_axis`, `test_transfer_hook_reads_no_tensor_counts`, `test_packed_padding_equals_ragged_padding` and `test_packed_step_loss_equals_ragged_step`. Random decode cases include zero, one, 33 and 300 masks; packed equality includes empty images, 17 keypoints, rotated boxes and masks. The synchronization and bounded-resize checks fail before the change and pass afterwards.
- **Benchmark.** `scripts/bench_input_path.py` uses runtime entry points shared with the parent revision, times synchronized transfer/restoration/padding and 300-mask decode, and emits median milliseconds plus CUDA peak allocated MiB (`null` on CPU/MPS). A tiny CPU case exercises the command in `scripts/_tests/test_bench_input_path.py`.
- **Accelerators.** `make gate-gpu` green on an A100 (py3.11, torch 2.9.1+cu129) at the stack's tip `72b5505`, the tree [WP-218](https://borda.github.io/lucid-YOLO/work_packages/wp-218/) left: 359 passed and 2 skipped (a `grad_dtype` case that needs a newer torch, and a replay test that needs two CUDA devices), every notebook in full, `demo_obb` on DOTA-v1.0 among them, and 55/55 accelerator goldens. `tests/ptl -m "gpu or data"` also passed on Colab's py3.13 / torch 2.11 (242 passed). The same selection on this package's own commit, `af3567f`, passed in that session too (242 passed). The packed-padding and chunked-decode equality tests ran on CUDA in both.
- **A100 benchmark.** `scripts/bench_input_path.py`, three runs on the parent commit `197b846` and three on this one, medians of the three: transfer, restoration and padding 7.68 ms before and 7.06 ms after; 300-mask decode 3.87 ms before and 4.52 ms after; decode peak allocated memory 1,565 MiB before and 321 MiB after. The chunking cuts the peak to a fifth at about a sixth more decode time, which is the trade the package made: the temporary no longer grows with the detection count.



<a id="wp-215"></a>

## WP-215 — shared lint helpers

**Finding.** K3’s remaining tooling duplication was fence state tracking in the doctest and WP-link audits, seven script-local repository-root definitions and the docs-site audit’s recursive Markdown walk. The existing common helpers already owned root resolution, sorted recursive walks and the standard counted failure report.

**Change.** A private shared fence tracker toggles state using each caller’s existing regular expression, retaining the doctest audit’s language and closing-blank checks and the WP-link audit’s tilde and blockquote handling. The seven scripts import their root from the appropriate common module, retaining a patchable module attribute. The docs-site audit uses the shared sorted walk. Standard counted failure reports already use the common reporter; specialized license, frozen-manifest and commit-trailer reports retain their distinct wording. Test-local root constants remain unchanged. No hook or output format changes.

**Verification.**

- **CPU.** `make gate` green on this commit's tree. `scripts/_tests` remains unchanged; its existing doctest-fence, WP-link, docs-site, frozen-manifest, commit-trailer, README, release-guard, overfit and shapes-regression assertions cover the affected callers. The `FenceTracker` doctest exercises language retention, ordinary lines and quoted mixed-delimiter toggles. No new test functions or hooks.
- **Ownership.** Only the two common owners define a script-level `REPO_ROOT`; the other matches are unchanged test-fixture constants. Only the lint common helper calls `rglob`. Specialized failure formats remain local because the counted reporter cannot reproduce them byte for byte.



<a id="wp-216"></a>

## WP-216 — cross-task training stories

**Finding.** T3’s remaining training duplication repeats empty-image, branch-gradient and checkpoint-composition stories across task suites. Body comparison distinguishes data-only differences from task-specific actions and oracles: segmentation and keypoints share the empty-image oracle, all three tasks share the gradient oracle, and OBB and keypoints share the state-key oracle. Segmentation’s checkpoint-loading test also compares zero-gain training losses and is not the same story.

**Change.** Three parametrized tests retain each case’s model options, target generation, expected branches, log key and state-key exclusions. Existing target builders move into the shared training fixtures without changing their generated values; per-task suites import them by bare module name. Every original gradient assertion remains, including both head branches and the non-null pose flow. Assignment-order tests stay separate: segmentation compares mask losses exactly, OBB checks three zero and three positive terms, and keypoints compares against an RLE reference. OBB’s empty-image test retains its backward-connectivity assertions. No shipped code or changelog changes.

**Verification.**

- **CPU.** `make gate` green on this commit's tree. `test_task_gradients_reach_every_branch`, `test_task_image_without_instances_gives_finite_zero_term` and `test_task_state_dict_keeps_detection_keys` retain seven cases from seven original tests; the four distinct cases remain in their original suites.
- **Body comparison.** No pair is wholly identical. Gradient bodies differ by target builders, module options and branch names; the keypoint wrapper already installs its log recorder. Empty-image bodies for segmentation and keypoints differ by task options and expected log key. State-key bodies differ by stem substring and the flow prefix excluded for keypoints. The OBB empty-image actions and all three assignment-order oracles differ in behavior, so they are not merged.
- **Case counts.** Empty image: 3 → 3; assignment order: 3 → 3; branch gradients: 3 → 3; state keys: 2 → 2. The three original training files plus the shared story suite collect 109 → 105 items because four target-builder doctests move to `tests/fixtures/training.py`; those doctests remain collected by the full gate. Story cases remain 11 → 11. Segmentation has no standalone state-key case in the candidate files; its checkpoint-load and zero-gain-loss test remains unchanged.



<a id="wp-217"></a>

## WP-217 — audit residuals

**Finding.** C-CONST left model ranking caps and the FLOP-counter image side as literals because models cannot import the decode layer. [WP-205](https://borda.github.io/lucid-YOLO/work_packages/wp-205/) normalized checkpoint tasks only during base construction: a subclass saving hyperparameters again could restore a `Task` member, which restricted checkpoint loading refuses. Oriented validation also converted axis-aligned predictions to the host before passing them to a strategy that discards them.

**Change.** The private detection cap lives in `tasks.py`, the existing stdlib-only leaf importable from both layers; `decode.common.MAX_DETECTIONS` retains its public name and object identity. Model ranking defaults share that cap, and the FLOP-counter side derives from the detection task spec. Every `save_hyperparameters` call normalizes recorded `Task` members and the logger snapshot, preserving Lightning's original caller frame and explicit arguments. The host conversion of the axis-aligned detections moves into `_update_val_map`, so the strategy that keeps that metric pays for it and the oriented strategy, which keeps none, no longer does; the rotated decoder and metric path stay unchanged. No public names, documented assumptions or goldens change.

**Verification.**

- **CPU.** `make gate` green on this commit's tree. `TestSubclassCheckpointTask.test_saved_task_is_plain_str`, `test_initial_snapshot_task_is_plain_str` and `test_checkpoint_loads_through_load_eval_module` cover named and keyword-forwarding subclasses, including subclasses that save again. `test_detection_cap_matches_model_ranking_defaults` pins identity across all four ranking entry points, and `test_flop_measurement_side_matches_coco_img_size` pins the task-derived side.
- **Red before.** Both re-saving subclass cases failed `assert type(module.hparams["task"]) is str` and the logger-snapshot assertion with `<enum 'Task'>`; both command-loader round trips failed with `_pickle.UnpicklingError: Weights only load failed`. The ranking-cap identity assertion failed while its numeric value remained 300.
- **Cause check.** The original forwarding subclasses pass while both re-saving variants fail before checkpoint serialization. This rejects a loader-only fault and shows why constructor-only normalization is insufficient. The same assertions and restricted round trips pass after normalization moves to every save.
- **Goldens.** `scripts/check_goldens.py` passes all 50 live and frozen files without moving any golden.
- **Oriented logs.** One seeded real validation batch and epoch end retain exact equality of the returned loss and every `val/*` scalar before and after removing the conversion. `_OrientedStrategy.update_box_validation` discards axis-aligned predictions; `update_extra_validation` calls `_update_val_rotated`, which ranks and scores rotated boxes independently.
- **Not reproduced: single-file collection.** The reported `KeyError: 'keypoints'` in `test_sigma_choice_matches_coco_schema_only_at_seventeen_points` does not reproduce on the [WP-216](https://borda.github.io/lucid-YOLO/work_packages/wp-216/) tree: the exact `pytest --doctest-modules src/lucid_yolo/ptl/module.py tests/ptl/test_keypoint_training.py` collection passes (60 tests) serially, under `-n auto`, under `-n 2 -m "not gpu and not data"` and with `-p no:randomly`, on pytest 9.1.1 and doctestplus 1.7.1. The keypoint accumulator the test feeds was replaced by [WP-213](https://borda.github.io/lucid-YOLO/work_packages/wp-213/)'s `_KeypointAP` after the report was written, and no traceback survives to say whether that is what removed it, so no patch and no regression test is added. A recurrence reopens it with its traceback.



<a id="wp-218"></a>

## WP-218 — rules plus pointers

**Finding.** Audit I1 found an agent contract loaded into every session through `CLAUDE.md` carrying long explanatory paragraphs, historical measurements and drifting tree counts alongside operative rules. DCO instructions were repeated across the contract, contributor guide and deliberately redundant `CLAUDE.md` reminder.

**Change.** `AGENTS.md` becomes rules plus rationale pointers: 22,926 bytes before, 15,118 after, and the longest line falls from 3,343 characters to 592. Section headings 1–8, the ordered execution loop, the commit-format example and every obligation, prohibition, approval, threshold, stop condition and behaviour-bearing literal stay in the contract; what left it is explanation — the history behind a rule, measurements, incident counts and tree counts that drift. That text is archived in [Agent-contract rationale](https://borda.github.io/lucid-YOLO/engineering/agents-rationale/), one section per contract section and one heading per original bullet or paragraph, and each compressed bullet that lost its reasoning links to its heading. The page is in the MkDocs navigation. `AGENTS.md` sec. 5 owns the full DCO rule; the contributor guide points there and keeps its human-facing sign-off instructions. `CLAUDE.md` is untouched: it restates the commit rule on purpose.

The size target was about 10 KB. It was not met: the first compression reached 11,920 bytes by dropping rules along with their reasons, and restoring every rule a preservation review found missing — hook contracts, dataset counts, worked examples, the definition of what [PRINCIPAL] marks — brought the contract to 15,118. A rule is kept over a byte.

**Verification.**

- **CPU.** `make gate` green on this commit's tree.
- **Preservation.** A byte-exact backup of the original contract was taken before the change, and the result was compared against it twice: a sentence-level map of every original obligation to where it now lives, and an independent followability review asking where an agent reading only the new file would act differently. The review's 17 findings (rules present only in the rationale page, a broadened doctest exemption, an ambiguous skip condition, actors and examples dropped) were each restored in the contract, and a second review pass found none open. All rationale anchors resolve, and the commit-format example is byte-identical.
- **Not changed here.** The contract names a `trailers` job in `lint.yml` as the enforcing owner of commit trailers; the review found no job of that name in the current workflow. The sentence is preserved as written, since correcting it is a CI question rather than a compression.

