# AGENTS.md — Autonomous Execution Contract

First file an agent (or engineer) executing the roadmap reads. Mirrors blueprint execution contract (sec. 14), amended by D12 (docs/DECISIONS.md) for single-operator local git flow.

**Prime directive**: no file, config, weight, or code fragment from `github.com/ultralytics/*` (or any mirror, package-registry copy, code-rendering page) is ever opened, downloaded, imported, or consulted. Includes docs.ultralytics.com — that site is generated from the same AGPL repository. No recovery from violating this.

## 1. The loop

1. Read this file, `docs/DECISIONS.md`, `docs/ASSUMPTIONS.md`, `docs/ROADMAP.md`.
2. Select **lowest-numbered WP whose dependencies are all `done`**, not marked **[HUMAN]**.
3. Implement **only that WP's scope**. Scope creep is a defect: unrelated fixes become their own WP.
4. Run `make gate`. Must be green, every previously frozen golden included.
5. Commit once (sec. 5 format) on `main` (D12a — one WP = one commit; each independently revertable, leaving `main` releasable).
6. Flip WP row to `done` in `docs/ROADMAP.md`, same commit. Keep the Scope cell near the table's median (~285 chars) — guidance, not a limit, and the reasoning behind the scope belongs in the linked RESEARCH_LOG.md section, per that file's header.
7. **Stop and report.** No chaining into the next WP without operator authorization for the session.

**Staged set must equal gate-tested set.** Step 4 certifies a tree, not an intention: any edit between gate run and `git commit` invalidates the green — including a docs edit, which is the form this actually takes. Docs-only *commit* is exempt; a docs edit riding inside a code commit is not, nor is a step-6 roadmap flip made after step 4 ran. Re-run the gate, or at minimum the meta tests, before staging. Failure surfaces one commit later as a red `main` with no obvious owner: `fcf3040` shipped that way, and the next agent spent a full run reporting a blocker it was forbidden to fix.

Pushes batched at phase boundaries, each requiring explicit human confirmation (D12a). Release tags always [HUMAN].

## 2. Environment

- Python 3.11; PyTorch >= 2.4. CPU wheels sufficient for WP-001…WP-044 except integration runs; [GPU] WPs attempt Apple MPS first (D12c).
- `make setup` — venv, editable install with dev extras, pre-commit hooks.
- `make gate` — pre-commit (all linters, D12d) + offline pytest + golden suite + frozen-golden regression. **The single command deciding whether a commit may land.**
- `make freeze-goldens MINOR=0.N` — release WPs only.
- All unit gates run **offline**: no network, dataset, or GPU. Only **[DATA]** and **[GPU]** WPs need more.
- **Pre-release wheels for a tier run** carry a PEP 440 `.devN` suffix on the version they lead to, counting from zero (`0.3.0.dev0`); it sorts below that release, so a plain `pip install lucid-yolo` never resolves to one. Bump `N` for every wheel leaving this machine, rebuilds of the same tree included, and verify the installed version before launching anything long — a 15-hour run once trained on a stale wheel because the intended version was never published and pip silently took the newest existing. The bump rides with the commit whose tree the wheel is built from — never a commit of its own, which names no change and leaves the version describing a tree it was not written against. Dev wheels are built and uploaded by hand (`uv build`), never tagged: `release_guard.py` accepts `v0.MINOR.PATCH` only, and a tag claims a release.
- **Progress bars: `from tqdm.auto import tqdm`, always.** `tqdm.auto` resolves to terminal bar or notebook widget from where it runs; the plain import always draws the terminal bar, a wall of one line per update in a notebook — and notebooks are where this project's long builds and tier runs launch. Convention, not a gate: it is presentation, and a meta test over every import line costs more to carry than the mistake costs to fix.
- **Three commands ship; `scripts/` does not.** `lucid-yolo` (train/validate), `lucid-data` (`download` · `check` · `build-tiles`), `lucid-eval` (acceptance scoring, on the protocol the checkpoint's own task names) live in `lucid_yolo/cli/` and cover a whole tier run from a wheel alone (WP-096). `scripts/` keeps development tooling — goldens, overfit gates, figures, release guard, licence audit — which a remote run needs none of. All three parse with jsonargparse: flags underscored (`--data_root`) as `--data.batch_size` already was, every command takes `--config`. `lucid-download` survives as a deprecated alias with its original dashed flags, removed in 0.4.0.

## 3. Dataset contract

Datasets never committed, never auto-downloaded by test code. `configs/data/*.yaml` carries the root path; `lucid-data check` (also `make check-data`) validates layout and counts before any [DATA] WP runs against real data. **Where each dataset comes from and what has to be on disk: `docs/DATASETS.md`.**

- **COCO 2017** (R12): `train2017/`, `val2017/`, `annotations/instances_*.json`; 118,287 train / 5,000 val images.
- **DOTA-v1.0** (R18): original images + labelTxt; 2,806 images / 188,282 instances / 15 classes across all three splits, of which train and val are the annotated two thirds — testing ground truth is withheld by the authors, so it is neither downloaded nor needed. Provisioning is manual (interactive Drive folders, not archive URLs); terms are academic use only. 1024 px tiling is a build artifact, never committed — build with `lucid-data build-tiles --root <root> --out <tiles>`, which writes the layout `obb_smoke.yaml` names. The tier trains on tiles, never the original tree: DOTA images run to several thousand pixels a side in PNG, which has no random-access region decode, so cropping per sample would decode a whole image to yield one 1024 px window (WP-094).
- Offline development against [DATA] WPs may use the synthetic stand-in generator (A26/D12b); **tier acceptance runs require real data** — missing real data blocks the tier, it is not worked around.

## 4. Escalation protocol (the anti-guessing rule)

Stop and write a `docs/ESCALATION.md` entry — symptom, WP, hypotheses tried, sources consulted, decision requested — on any of:

1. Gate fails after **two** documented assumption iterations.
2. Answering a question would require a denylisted source.
3. WP spec conflicts with the technical specification or the papers.
4. A change would require altering a frozen golden.
5. A run would exceed 4 GPU-hours and is not already [HUMAN].

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

- `WP:` exactly one WP id. `Provenance:` at least one source id from docs/PROVENANCE.md (papers and primary literature only). `Assumptions:` A-ids touched, or `none`. `Gate:` test id(s) proving the DoD.
- Types: `feat` `fix` `test` `ci` `docs` `chore` `perf` `refactor` `refine` `exp` `release`. `refine` is this project's own, for a change improving something already correct — a rename, a clearer boundary, a sharpened comment: `refactor` claims behaviour-preserving restructuring, `docs` claims prose, and a rename crossing code, configs and prose is neither.
- No commit message references the Ultralytics repository; papers cited by arXiv id. Hash-sign and at-sign characters never appear in subject or body (co-author trailer emails are the sole exception).

## 6. Naming hygiene

Internal identifiers use descriptive names derived from the papers' terminology (`DualAssignHead`, `SmallTargetAssigner`, `ProgressiveLossSchedule`, `MuSGD`, `PrototypeMaskHead`, `LongEdgeOBBHead`) — never any implementation's symbol names (we have not seen them, and it stays that way). Assignment, loss, optimizer and rotated-geometry modules are written by hand from the equations: AI assistants may regurgitate AGPL training data, and these are the highest-risk surface.

## 7. Standing prohibitions

Never: consult or install `ultralytics` or any mirror; create a model-topology config format (ADR-001); copy code from any external detection repository (check LICENSE and provenance before consulting *any* external detection repo, third-party YOLO-seg/YOLO-OBB forks included); plan, promise, or tag a 1.0 (ADR-002); modify a frozen golden; start a [HUMAN] WP; commit datasets or downloaded weights; leave `main` red; download, fine-tune, distill from, or compare against released Ultralytics checkpoints.

## 8. Delegated work packages

A completion notification is not a completion. The harness reports a delegated task `completed` whether the agent finished or stopped mid-sentence on a partial edit, and the notification carries a `result` field that looks like a report — so an unfinished run reads as a finished one that summarized badly. Happened four times in Phase 8, twice found only by inspecting the worktree.

Before treating a delegated WP as done, check the worktree's `git status` against what the spec asked for; a missing test file or untouched entry-point module is the tell. Resume with a message naming what is still missing rather than respawning — the agent picks up from its own transcript with full context. The spawn prompt's "report when done or when stuck" clause helps and does not prevent this.
