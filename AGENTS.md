# AGENTS.md — Autonomous Execution Contract

First file an agent (or engineer) executing the roadmap reads. Mirrors blueprint execution contract (sec. 14), amended by D12 (docs/DECISIONS.md) for single-operator local git flow.

**Prime directive**: no file, config, weight, or code fragment from `github.com/ultralytics/*` (or any mirror, package-registry copy, code-rendering page) is ever opened, downloaded, imported, or consulted. Includes docs.ultralytics.com — that site is generated from the same AGPL repository. No recovery from violating this.

## 1. The loop

1. Read this file, `docs/DECISIONS.md`, `docs/ASSUMPTIONS.md`, `docs/ROADMAP.md`.
2. Select **lowest-numbered WP whose dependencies are all `done`**, not marked **[PRINCIPAL]**.
3. Implement **only that WP's scope**. Scope creep is a defect: unrelated fixes become their own WP.
4. Run `make gate`. Must be green, every previously frozen golden included.
5. Commit once (sec. 5 format) on `main` (D12a — one WP = one commit; each independently revertable, leaving `main` releasable).
6. Flip WP row to `done` in `docs/ROADMAP.md`, same commit. Keep the Scope cell near the table's median (~285 chars) — guidance, not a limit, and the reasoning behind the scope belongs in the linked RESEARCH_LOG.md section, per that file's header.
7. **Stop and report.** No chaining into the next WP without principal authorization for the session.

**Staged set must equal gate-tested set.** Step 4 certifies a tree, not an intention: any edit between gate run and `git commit` invalidates the green — including a docs edit, which is the form this actually takes. Docs-only *commit* is exempt; a docs edit riding inside a code commit is not, nor is a step-6 roadmap flip made after step 4 ran. Re-run the gate, or at minimum the meta tests, before staging. Failure surfaces one commit later as a red `main` with no obvious owner: `fcf3040` shipped that way, and the next agent spent a full run reporting a blocker it was forbidden to fix.

**[PRINCIPAL]** marks work that belongs to whoever *drives* the project rather than whoever *executes* it — the calls that come from intent, ownership or risk appetite, and so cannot be answered by reading the code. It is a role, not a species: an agent acting with delegated authority is still executing, and a principal who writes the patch themselves is still deciding.

Pushes batched at phase boundaries, each requiring explicit principal confirmation (D12a). Release tags always [PRINCIPAL].

**This loop has an end condition (D18).** Once the reproduction report carries all four accepted tiers and the repository is public, a change that alters no shipped behaviour, adds or removes no public symbol, moves no golden and changes no documented assumption lands as an ordinary gated commit rather than a tracked WP. Everything else still opens one, the gate never relaxes, and adding a new **task** to the model family restarts the full procedure — its own phase, numbered WPs, smoke tier, principal gate and 0.MINOR — because a new task is a new reproduction claim. WP-141 landed the contributor guide and the three admission layers that make the relaxation usable by someone other than the agent; what it did not do is satisfy D18's own second condition. **The repository is not public yet** (O3, `docs/DECISIONS.md`), so the contract above still applies in full — the relaxation is written down and gated, not in force, and the row that changes that is the one making the repository public rather than any row here.

## 2. Environment

- Python 3.11; PyTorch >= 2.4. CPU wheels sufficient for WP-001…WP-044 except integration runs; [GPU] WPs attempt Apple MPS first (D12c).
- `make setup` — venv, editable install with dev extras, pre-commit hooks.
- `make gate` — pre-commit (all linters, D12d) + offline pytest + golden suite + frozen-golden regression. **The single command deciding whether a commit may land.**
- `make freeze-goldens MINOR=0.N` — release WPs only.
- All unit gates run **offline**: no network, dataset, or GPU. Only **[DATA]** and **[GPU]** WPs need more.
- **The `gpu` and `data` markers are not the accelerator boundary, and reading them as one overstates what the offline gate covers.** There are eight marks in the whole tree, all four tests of `scripts/_tests/test_overfit_micro.py`, so `-m "not gpu and not data"` deselects four tests and nothing else. Every other test runs on CPU because it was written to, not because anything checks that it can — and the accelerator branches themselves (MPS selection, the CUDA paths, the `goldens/gpu/` producers) are unmarked because no test reaches them at all. What the markers do is keep the four tests that would need a machine out of the offline run; what they do not do is delimit the code an offline green run has exercised.
- **Pre-release wheels for a tier run** carry a PEP 440 `.devN` suffix on the version they lead to, counting from zero (`0.3.0.dev0`); it sorts below that release, so a plain `pip install lucid-yolo` never resolves to one. Bump `N` for every wheel leaving this machine, rebuilds of the same tree included, and verify the installed version before launching anything long — a 15-hour run once trained on a stale wheel because the intended version was never published and pip silently took the newest existing. The bump rides with the commit whose tree the wheel is built from — never a commit of its own, which names no change and leaves the version describing a tree it was not written against. Dev wheels are built and uploaded by hand (`uv build`), never tagged: `release_guard.py` accepts `v0.MINOR.PATCH` only, and a tag claims a release.
- **Progress bars: `from tqdm.auto import tqdm`, always.** `tqdm.auto` resolves to terminal bar or notebook widget from where it runs; the plain import always draws the terminal bar, a wall of one line per update in a notebook — and notebooks are where this project's long builds and tier runs launch. Convention, not a gate: it is presentation, and a meta test over every import line costs more to carry than the mistake costs to fix.
- **A docstring `Examples:` block may be written bare or inside a ```` ```pycon ```` fence, and both are live doctests.** `make gate` runs `--doctest-modules`, which reads the `>>>` prompts either way, so the fence changes nothing a gate can see; nor does it change anything a reader sees, because the site build carries no `mkdocstrings` and renders no API page from a docstring at all. Both forms are in wide use under `src/`, the bare form in the clear majority, and a majority of them predate the fenced one; several modules carry both. **The convention is therefore to match the module you are editing, not to convert it.** A mixed module is the only real defect here, and it is fixed when that module is opened for another reason — never as a sweep, which would touch hundreds of docstrings to change no behaviour, no rendering and no gate result. Stated for the same reason `tqdm.auto` is, and gated for the same reason it is not.
- **Four commands ship; `scripts/` does not.** `lucid-yolo` (train/validate), `lucid-data` (`download` · `check` · `build-tiles`), `lucid-eval` (acceptance scoring, on the protocol the checkpoint's own task names) and `lucid-predict` (detections for one image, on the task the checkpoint names) live in `lucid_yolo/cli/` and cover a whole tier run from a wheel alone (WP-096, WP-089). `scripts/` keeps development tooling — goldens, overfit gates, figures, release guard, licence audit — which a remote run needs none of. All four parse with jsonargparse: flags underscored (`--data_root`) as `--data.batch_size` already was, every command takes `--config`. `lucid-download`, the deprecated alias, is removed in 0.4.0 as 0.3.0 said it would be: `lucid-data download` is the only spelling.
- **`scripts/lint/` holds every repo-content checker**, each a `main(argv)` CLI plus a `.pre-commit-config.yaml` local hook. Most of them arrived by migration — licence audit, commit trailers, doctest coverage, flat test groups, docs presence, docs site, figure captions, license headers, version single-source all used to run inline as a `tests/meta/` pytest test (WP-129) — and the directory has since grown checkers that were never tests at all: the frozen-manifest digest assertion (WP-168) and the DCO sign-off check (WP-142) were written as hooks from the start. Read the directory rather than this sentence for the current set; the count is a property of the tree, and a list written here drifts the moment a checker lands. Every hook carries a distinct emoji in its `name:` and is individually callable — `pre-commit run <hook-id>` (e.g. `pre-commit run license-audit`) — without running the rest of the suite, which is also how a WP-scoped fix is verified before `make gate`. `release-guard` (`scripts/release_guard.py`) runs on every commit and `pre-commit run --all-files` (`stages: [pre-commit, manual]`) as well as standalone (`pre-commit run --hook-stage manual release-guard`) — off a release tag it costs one `git describe` call, so always-run beats an exemption to remember. The hook passes `--metadata-only`, which is not a convenience: the guard's default gate command is `make gate`, `make gate` runs pre-commit, and pre-commit runs this hook, so on the one checkout where the guard has something to check — an exact release tag — the unsplit hook asked for another gate. Metadata-only runs the five file-reading checks and refuses on any of them with the same wording as a full run; what it drops is the gate, and it says so in its own verdict rather than printing the shipping line. The certifying run is `release.yml`'s, on a pushed tag, without the flag. `golden-check` (`scripts/check_goldens.py`) carried the same stages until WP-162 measured what "cheap" meant for it — 34.6 s on every commit — and moved it to `stages: [manual]`: `make gate` runs that harness through its own `golden` target, so the commit-stage copy repeated the gate instead of adding to it, and `ci-tests.yml` runs it as a step of its own, which is where CI's full sweep over every golden lives now (43 files when WP-162 measured it, 50 today — the number is a property of the release, so it is not restated here). `commit-trailers` (`stages: [commit-msg]`) validates the message actually being written and cannot run under `--all-files` at all, since no commit is in progress then; `commit-trailers-history` is its `--all-files`-reachable sibling, re-validating every commit ahead of `origin/main` via the same script's `--range` mode — which on a pushed branch, the normal state of a clean tree, is zero commits, so a green run there usually means nothing was examined rather than that everything passed. What it does catch is unpushed local work and an amended message before it leaves the machine; `lint.yml`'s `trailers` job is the enforcing owner, and it fetches full history to be one.
- **`scripts/_tests/` holds every test whose subject is a `scripts/` module** — the lint checkers above, plus `release_guard.py`, `check_goldens.py`, `absolutize_readme.py`, `draw_predictions.py`, `shapes_regression.py`, `overfit_micro.py`: each a functional-core test of the script's own functions on synthetic fixtures, not a live-tree scan (the hook or the `make` target the DoD table cites is what runs against the live tree). Named `test_<script>.py`, mirroring the script it covers. `scripts/README.md` is the map: what each script does and where its test lives.

## 3. Dataset contract

Datasets never committed, never auto-downloaded by test code. `src/lucid_yolo/configs/data/*.yaml` carries the root path; `lucid-data check` (also `make check-data`) validates layout and counts before any [DATA] WP runs against real data. **Where each dataset comes from and what has to be on disk: `docs/DATASETS.md`.**

- **COCO 2017** (R12): `train2017/`, `val2017/`, `annotations/instances_*.json`; 118,287 train / 5,000 val images.
- **DOTA-v1.0** (R18): original images + labelTxt; 2,806 images / 188,282 instances / 15 classes across all three splits, of which train and val are the annotated two thirds — testing ground truth is withheld by the authors, so it is neither downloaded nor needed. Provisioning is manual (interactive Drive folders, not archive URLs); terms are academic use only. 1024 px tiling is a build artifact, never committed — build with `lucid-data build-tiles --root <root> --out <tiles>`, which writes the layout `obb_nano_smoke.yaml` names. The tier trains on tiles, never the original tree: DOTA images run to several thousand pixels a side in PNG, which has no random-access region decode, so cropping per sample would decode a whole image to yield one 1024 px window (WP-094).
- Offline development against [DATA] WPs may use the synthetic stand-in generator (A26/D12b); **tier acceptance runs require real data** — missing real data blocks the tier, it is not worked around.

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
```

- `WP:` exactly one WP id, or `none` for work that belongs to no package — remediation of an audit finding, for one. `Provenance:` at least one source id from docs/PROVENANCE.md (papers and primary literature only). `Assumptions:` A-ids touched, or `none`. `Gate:` test id(s) proving the DoD.
- **`Provenance: none` is a value, not an empty field**, and it may carry its reason: `none (implementation efficiency; R1 silent on target rasterisation)`. An id written inside that explanation is a **reference** — resolved against the register like any other, and never counted as satisfying "name a source", because the line is a denial. A trailer that neither cites nor denies is refused. Both halves are checked: a well-formed id naming no register row fails, and so does an id whose row sits under Placeholders rather than the source allowlist — R15 and R19 are legal-evidence citations, admissible in a document and never as an implementation source.
- Types: `feat` `fix` `test` `ci` `docs` `chore` `perf` `refactor` `refine` `exp` `release`. `refine` is this project's own, for a change improving something already correct — a rename, a clearer boundary, a sharpened comment: `refactor` claims behaviour-preserving restructuring, `docs` claims prose, and a rename crossing code, configs and prose is neither.
- No commit message references the Ultralytics repository; papers cited by arXiv id. Hash-sign and at-sign characters never appear in subject or body (co-author trailer emails are the sole exception).

## 6. Naming hygiene

Internal identifiers use descriptive names derived from the papers' terminology (`DualAssignHead`, `SmallTargetAssigner`, `ProgressiveLossSchedule`, `MuSGD`, `PrototypeMaskHead`, `LongEdgeOBBHead`) — never any implementation's symbol names (we have not seen them, and it stays that way). Assignment, loss, optimizer and rotated-geometry modules are written by hand from the equations: AI assistants may regurgitate AGPL training data, and these are the highest-risk surface.

## 7. Standing prohibitions

Never: consult or install `ultralytics` or any mirror; create a model-topology config format (ADR-001); copy code from any external detection repository (check LICENSE and provenance before consulting *any* external detection repo, third-party YOLO-seg/YOLO-OBB forks included); plan, promise, or tag a 1.0 (ADR-002); modify a frozen golden; start a [PRINCIPAL] WP; commit datasets or downloaded weights; leave `main` red; download, fine-tune, distill from, or compare against released Ultralytics checkpoints.

**"Modify a frozen golden" is checked, not merely written here** (WP-168). `goldens/frozen/MANIFEST.sha256` pins every frozen file by digest and the `frozen-manifest` pre-commit hook asserts it; the golden harness could not, because it compares each frozen file against that same file's own stored values, so an edit moving the values and the tolerances together was green by construction. Once escalation trigger 4 has been answered and the move is approved, `python scripts/freeze_goldens.py --reseal` is what records it — the only sanctioned writer, and never a hand-edited digest.

**The prohibition has been overridden, twice, on the record** — both on `goldens/frozen/0.6/aug_invariants.json`, both dated 2026-09-03, both in `docs/ESCALATION.md`: WP-154b (the affine composition centre) and WP-155 (the letterbox resample). Written here because a rule stated as absolute while exceptions to it exist elsewhere teaches the next reader to distrust the rule rather than to find the exceptions. Neither is a standing waiver, and the second entry says so in its own resolution: the two are dated exceptions, and the next frozen-golden case is a fresh principal decision rather than one this precedent settles. The prohibition is what stands; the route through it is escalation trigger 4, and nothing else.

## 8. Delegated work packages

A completion notification is not a completion. The harness reports a delegated task `completed` whether the agent finished or stopped mid-sentence on a partial edit, and the notification carries a `result` field that looks like a report — so an unfinished run reads as a finished one that summarized badly. Happened four times in Phase 8, twice found only by inspecting the worktree.

Before treating a delegated WP as done, check the worktree's `git status` against what the spec asked for; a missing test file or untouched entry-point module is the tell. Resume with a message naming what is still missing rather than respawning — the agent picks up from its own transcript with full context. The spawn prompt's "report when done or when stuck" clause helps and does not prevent this.
