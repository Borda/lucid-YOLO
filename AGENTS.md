# AGENTS.md — Autonomous Execution Contract

Read this first to execute the roadmap (D12, `docs/DECISIONS.md`).

**Prime directive**: never open, download, import or consult any file, config, weight or code fragment from `github.com/ultralytics/*`, any mirror, package-registry copy, code-rendering page or docs.ultralytics.com. No recovery from violating this. ([why](docs/engineering/agents-rationale.md#prime-directive))

**Review tasks**: use [.github/skills/code-review/SKILL.md](.github/skills/code-review/SKILL.md) as the single owner of the review process. Review is read-only; the loop below governs implementation, not review. Keep the source, provenance, golden and principal boundaries; report violations or escalation needs instead of editing files, starting a WP or committing during review.

## 1. The loop

1. Read this file, `docs/DECISIONS.md`, `docs/ASSUMPTIONS.md`, `docs/ROADMAP.md`.
2. Select **lowest-numbered WP whose dependencies are all `done`**, not marked **[PRINCIPAL]**.
3. Implement **only that WP's scope**. Scope creep is a defect: unrelated fixes become their own WP. ([why](docs/engineering/agents-rationale.md#wp-scope))
4. Run `make gate`. Must be green, every previously frozen golden included.
5. Commit once (sec. 5 format) on `main` (D12a): one WP = one independently revertable commit; leave `main` releasable.
6. Flip WP row to `done` in `docs/ROADMAP.md`, same commit. Keep the Scope cell near the table's median (~285 chars) — guidance, not a limit; the reasoning goes in the linked `docs/work_packages/wp-XXX.md`, included by the log's phase page, per the log's header.
7. **Stop and report.** No chaining into the next WP without principal authorization for the session.

**Staged set must equal gate-tested set.** Any edit after step 4 before `git commit` invalidates green, including docs/step-6 roadmap flip. Docs-only *commits* exempt; docs in code commits are not. Re-run gate or at minimum meta tests before staging. ([why](docs/engineering/agents-rationale.md#tested-tree))

**[PRINCIPAL]** marks work belonging to whoever *drives* the project, not whoever *executes* it: calls from intent, ownership or risk appetite, which reading the code cannot answer. It is a role: an agent with delegated authority still executes; a principal writing the patch still decides. ([why](docs/engineering/agents-rationale.md#principal))

Pushes batched at phase boundaries, each requiring explicit principal confirmation (D12a). Release tags always [PRINCIPAL].

**D18 is in force (D30):** all four tiers accepted in the reproduction report; repository public, O3 cleared, WP-125 accepted. Changes altering no shipped behaviour, adding/removing no public symbol, moving no golden and changing no documented assumption: ordinary gated commit. All others, audit batches included, need a WP; the gate never relaxes.

A new model-family **task** restarts the full procedure: own phase, numbered WPs, smoke tier, principal gate, 0.MINOR. ([why](docs/engineering/agents-rationale.md#d18))

## 2. Environment

- Python 3.11; PyTorch >= 2.4. CPU wheels sufficient for WP-001…WP-044 except integration runs; [GPU] WPs attempt Apple MPS first (D12c).
- `make setup` — venv, editable install with dev extras, pre-commit hooks.
- `make gate` = pre-commit (all linters, D12d) + offline pytest + golden suite + frozen-golden regression; **sole commit acceptance command**. `make test`: pytest-xdist `-n auto`; read failures serially with `PYTEST_XARGS=`. ([why](docs/engineering/agents-rationale.md#gate))
- `make freeze-goldens MINOR=0.N` — release WPs only.
- All unit gates run **offline**: no network, dataset, or GPU. Only **[DATA]** and **[GPU]** WPs need more.
- **`gpu`/`data` markers do not delimit accelerator branches.** `tests/fixtures/markers.py` defines `requires_cuda` (`gpu` mark, skipped without CUDA) and `requires_accelerator` (`gpu` mark, skipped when neither CUDA nor MPS is present); every `data` test is also `gpu`. Offline `-m "not gpu and not data"` exercises no marked tests; an unmarked test runs on CPU because it was written to, and nothing says which accelerator branches it reaches.
  - `make gate-gpu` runs marked tests on an accelerator; not every commit needs it, but [GPU] WPs cannot close without green. Known marked failures leave `main` red until fixed (D32). ([why](docs/engineering/agents-rationale.md#markers))
- Tier pre-release wheels: PEP 440 `.devN` on the coming version, from zero (`0.3.0.dev0`); it sorts below that release, so a plain `pip install lucid-yolo` never resolves to one. Bump `N` for every wheel leaving this machine, same-tree rebuilds included; verify installed version before long runs. Bump in the wheel-tree commit, never separately. Build/upload by hand (`uv build`); never tag dev wheels (`release_guard.py`: `v0.MINOR.PATCH` only). ([why](docs/engineering/agents-rationale.md#wheels))
- Always `from tqdm.auto import tqdm`. Ruff's `TID251` ban covers `src/`; the `tqdm-auto-only` pygrep hook covers `tests/`, `scripts/` and `notebooks/`. ([why](docs/engineering/agents-rationale.md#tqdm))
- Fence every `src/` docstring `Examples:` doctest with ```` ```pycon ````, closing after exactly one blank line. The `doctest-fences` hook (`scripts/lint/audit_doctest_fences.py`) checks `src/` only; `make gate`'s `--doctest-modules` runs the examples either way. `scripts/` and `tests/` are not fenced and keep their existing form.
  - The API tab (`docs/api/`, rendered by `mkdocstrings`) highlights each block with the `pycon` lexer; `docs/javascripts/pycon_copy.js` makes its copy button yield the code alone, prompts and output lines dropped. A `>>>` in an ordinary string literal or comment is not a doctest and is not checked; keep `::` shell blocks (`$ python ...`) unchanged. ([why](docs/engineering/agents-rationale.md#doctests))
- A test name is its assertion written as an identifier, without filler: `test_casting_plugin_keeps_worker_hsv`, never `test_casting_plugin_still_keeps_the_worker_hsv`. Drop articles (`the`, `a`, `an`) and adverbs (`still`, `just`, `actually`, `properly`, `correctly`); keep claim-changing `only`, `never`, `not`, `once`. Put scenario/reason in docstring, not name.
  - Plain `bool`/`int`/`str` parameters stay bare (`[False, True]`), ids pytest-derived. `pytest.param(..., id=...)` only for unreadable defaults (function/container/opaque object) or `marks=`. `test-name-filler-adverbs` gates five adverbs in `tests/`, `scripts/_tests/`; rest is convention. ([why](docs/engineering/agents-rationale.md#names))
- Five commands ship, from `lucid_yolo/cli/`; `scripts/` (development tooling) does not: `lucid-yolo` train/validate; `lucid-data` `download` · `check` · `build-tiles`; `lucid-eval` acceptance on checkpoint task's protocol; `lucid-predict` one image on checkpoint's task; `lucid-export` one checkpoint to one ONNX file (`onnx`: `export` extra).
  - First four support wheel-only tiers. All use jsonargparse, underscored flags (`--data_root`, `--data.batch_size`), `--config`. `lucid-download` removed in 0.4.0 as 0.3.0 promised: only `lucid-data download`. ([why](docs/engineering/agents-rationale.md#commands))
- Put every repo-content checker in `scripts/lint/`: `main(argv)` CLI + `.pre-commit-config.yaml` local hook. Read directory for current set. Each hook needs distinct emoji in `name:`; verify WP-scoped fixes individually via `pre-commit run <hook-id>` (e.g. `pre-commit run license-audit`) before `make gate`.
  - `release-guard` (`scripts/release_guard.py`) runs on every commit and under `pre-commit run --all-files` (`stages: [pre-commit, manual]`), standalone as `pre-commit run --hook-stage manual release-guard`. Its hook passes `--metadata-only`; keep it, since the unsplit guard would ask for another `make gate` from inside one. Metadata-only runs the five file-reading checks and refuses on any of them with a full run's wording; it drops only the gate and says so in its verdict instead of printing the shipping line. The certifying run is `release.yml`'s, on a pushed tag, without the flag.
  - `golden-check` (`scripts/check_goldens.py`) is `stages: [manual]`: `make gate` runs it through its `golden` target, and `ci-tests.yml`'s `goldens` job holds CI's full sweep over every golden.
  - `commit-trailers` (`stages: [commit-msg]`) checks the message being written and cannot run under `--all-files`. `commit-trailers-history` is its `--all-files`-reachable sibling: `--unpushed` re-checks `@{upstream}..HEAD`, or `HEAD --not --remotes` on a branch with no resolvable upstream (never pushed, or a fork PR checked out by `gh pr checkout`), so on a pushed branch a green run usually examined nothing. `lint.yml`'s `trailers` job is the enforcing owner and fetches full history. ([why](docs/engineering/agents-rationale.md#checkers))
- Put every test of a `scripts/` module in `scripts/_tests/`, named `test_<script>.py`: functional-core synthetic-fixture tests, not live-tree scans; hook/DoD `make` target checks live tree. Map: `scripts/README.md`. ([why](docs/engineering/agents-rationale.md#script-tests))

## 3. Dataset contract

Never commit datasets or auto-download in tests. Roots: `src/lucid_yolo/configs/data/*.yaml`; validate layout/counts via `lucid-data check` (also `make check-data`) before any real-data [DATA] WP. Sources/layouts: `docs/DATASETS.md`.

- **COCO 2017** (R12): `train2017/`, `val2017/`, `annotations/instances_*.json`; 118,287 train / 5,000 val images. Keypoints: same archive, `annotations/person_keypoints_*.json`; both configs must explicitly override `train_ann_file`/`val_ann_file` (`docs/DATASETS.md`). ([why](docs/engineering/agents-rationale.md#coco))
- **DOTA-v1.0** (R18): original images + labelTxt; 2,806 images / 188,282 instances / 15 classes across all three splits; train/val annotated, testing truth withheld, neither downloaded nor needed. Provision manually (interactive Drive folders, not archive URLs); academic use only.
  - Build uncommitted 1024 px tiles: `lucid-data build-tiles --root <root> --out <tiles>` for `obb_nano_smoke.yaml`. Tier trains on tiles, never originals (WP-094). ([why](docs/engineering/agents-rationale.md#dota))
- Offline [DATA] development may use synthetic stand-in generator (A26/D12b); **tier acceptance requires real data**. Missing data blocks tier; no workaround.

## 4. Escalation protocol (the anti-guessing rule)

Stop and write a `docs/ESCALATION.md` entry — symptom, WP, hypotheses tried, sources consulted, decision requested — on any of:

1. Gate fails after **two** documented assumption iterations.
2. Answering a question would require a denylisted source.
3. WP spec conflicts with the technical specification or the papers.
4. A change would require altering a frozen golden.
5. A run would exceed 4 GPU-hours and is not already [PRINCIPAL].

## 5. Commit format (provenance-carrying)

Conventional Commits subject plus mandatory trailers:

```text
feat(models): add dual detection head with reg_max=1

Implements the one-to-one (topk=7 -> topk2=1, 300 outputs) and
one-to-many (topk=10, dense) branches sharing neck features, with
direct 4-scalar ltrb regression and no DFL module.

WP: 022
Provenance: R1 3.2.1, R1 3.2.2, R1 Fig. S2, R6
Assumptions: A9
Gate: tests/models/test_head.py::test_dual_head_shapes

Signed-off-by: Jane Doe <jane@example.com>
```

- `WP:` exactly one WP id, or `none` for work that belongs to no package — remediation of an audit finding, for one. `Provenance:` at least one source id from docs/PROVENANCE.md, papers/primary literature only. `Assumptions:` A-ids touched or `none`. `Gate:` test id(s) proving DoD. ([why](docs/engineering/agents-rationale.md#trailers))
- `Provenance: none` is a value, not an empty field, and may carry its reason: `none (implementation efficiency; R1 silent on target rasterisation)`. An id inside that reason is a reference: resolved against the register, never counted as naming a source. A trailer that neither cites nor denies is refused, and so is an id naming no register row or a row under Placeholders; R15 and R19 are legal evidence for documents, never implementation sources. ([why](docs/engineering/agents-rationale.md#denials))
- Types: `feat` `fix` `test` `ci` `docs` `chore` `perf` `refactor` `refine` `exp` `release`. `refine`: improves already-correct work (a rename, a clearer boundary, a sharpened comment); `refactor`: behaviour-preserving restructuring; `docs`: prose. Cross-code/config/prose rename is neither `refactor` nor `docs`. ([why](docs/engineering/agents-rationale.md#types))
- **Every commit needs DCO `Signed-off-by:`**, merges/cherry-picks/agent commits included. Use `git commit -s`, `git merge --signoff`, `git cherry-pick -s`; finish `--no-commit` merges with `git commit -s`. The DCO GitHub App checks every PR commit: sign at creation; never reword pushed commits to add it. ([why](docs/engineering/agents-rationale.md#dco))
- No Ultralytics repository references in commit messages; papers by arXiv id. No issue references (`#42`)/mentions (`@name`) in subject/body, except co-author trailer emails/`Signed-off-by:` address. Markdown headings are not references. ([why](docs/engineering/agents-rationale.md#references))

## 6. Naming hygiene

- Use descriptive internal identifiers from paper terminology (`DualAssignHead`, `SmallTargetAssigner`, `ProgressiveLossSchedule`, `MuSGD`, `PrototypeMaskHead`, `LongEdgeOBBHead`), never any implementation's symbol names. Write assignment/loss/optimizer/rotated-geometry modules by hand from equations. ([why](docs/engineering/agents-rationale.md#identifiers))

## 7. Standing prohibitions

Never:

- Consult/install `ultralytics` or any mirror; create a model-topology config format (ADR-001).
- Copy code from any external detection repository. Before consulting *any*, including third-party YOLO-seg/YOLO-OBB forks, check LICENSE and provenance.
- Plan, promise or tag a 1.0 (ADR-002); modify a frozen golden; start a [PRINCIPAL] WP.
- Commit datasets/downloaded weights; leave `main` red (D32: known failing `make gate-gpu` counts).
- Download, fine-tune, distill from or compare against released Ultralytics checkpoints.

`goldens/frozen/MANIFEST.sha256` pins every frozen file by digest and the `frozen-manifest` pre-commit hook asserts it (WP-168). Only after escalation trigger 4 is answered and move approved, use `python scripts/freeze_goldens.py --reseal`, sole sanctioned writer; never hand-edit digests. ([why](docs/engineering/agents-rationale.md#manifest))

WP-154b/WP-155: 2026-09-03 exceptions for `goldens/frozen/0.6/aug_invariants.json` in `docs/ESCALATION.md`, no standing waiver. Every next case needs fresh principal decision via escalation trigger 4 only; prohibition stands. ([why](docs/engineering/agents-rationale.md#exceptions))

## 8. Delegated work packages

Do not treat a completion notification as completion.

- Before accepting delegated WP, compare worktree `git status` to spec. If incomplete, resume naming missing work, never respawn. Spawn prompt "report when done or when stuck" does not replace check. ([why](docs/engineering/agents-rationale.md#completion))
