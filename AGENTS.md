# AGENTS.md — Autonomous Execution Contract

This is the first file an agent (or engineer) executing the roadmap reads. It mirrors the governing blueprint's execution contract (sec. 14), as amended by D12 (docs/DECISIONS.md) for the single-operator local git flow.

**Prime directive**: no file, config, weight, or code fragment from `github.com/ultralytics/*` (or any mirror, package-registry copy, or code-rendering page thereof) is ever opened, downloaded, imported, or consulted. This includes docs.ultralytics.com — the docs site is generated from the same AGPL repository. There is no recovery from violating this rule.

## 1. The loop

1. Read this file, `docs/DECISIONS.md`, `docs/ASSUMPTIONS.md`, and `docs/ROADMAP.md`.
2. Select the **lowest-numbered WP whose dependencies are all `done`** and which is not marked **[HUMAN]**.
3. Implement **only that WP's scope**. Scope creep is a defect: unrelated fixes become their own WP.
4. Run `make gate`. It must be green, including every previously frozen golden.
5. Commit once (sec. 5 format) directly on `main` (D12a — one WP = one commit; every commit must be independently revertable and leave `main` releasable).
6. Flip the WP row to `done` in `docs/ROADMAP.md` (part of the same commit).
7. **Stop and report.** Do not chain into the next WP unless the operator has authorized continuous execution for the session.

**The staged set must equal the gate-tested set.** Step 4 certifies a tree, not an intention, so any edit made between the gate run and `git commit` invalidates the green — including a documentation edit, which is the form this actually takes. A docs-only *commit* is exempt from the gate; a docs edit riding along inside a code commit is not, and neither is a roadmap flip made at step 6 after step 4 already ran. Re-run the gate, or at minimum the meta tests, before staging. The failure surfaces one commit later on someone else's branch as a red `main` with no obvious owner: `fcf3040` shipped that way, and the next agent spent a full run reporting a blocker it was forbidden to fix.

Pushes to the remote are batched at phase boundaries and each push requires explicit human confirmation (D12a). Release tags are always [HUMAN].

## 2. Environment

- Python 3.11; PyTorch >= 2.4. CPU wheels sufficient for WP-001…WP-044 except integration runs; [GPU] WPs attempt Apple MPS first (D12c).
- `make setup` — venv, editable install with dev extras, pre-commit hooks.
- `make gate` — pre-commit (all linters, D12d) + offline pytest + golden suite + frozen-golden regression. **The single command that decides whether a commit may land.**
- `make freeze-goldens MINOR=0.N` — release WPs only.
- All unit gates run **offline**: no network, no dataset, no GPU. Only WPs marked **[DATA]** or **[GPU]** need more.
- **Pre-release wheels for a tier run** carry a PEP 440 `.devN` suffix on the version they lead to, counting from zero (`0.3.0.dev0`), which sorts below that release so a plain `pip install lucid-yolo` never resolves to one. Bump `N` for every wheel that leaves this machine, rebuilds of the same tree included, and verify the installed version before launching anything long: a 15-hour run once trained on a stale wheel because the intended version was never published and pip silently took the newest that existed. Dev wheels are built and uploaded by hand (`uv build`); they are **not** tagged, since `release_guard.py` accepts `v0.MINOR.PATCH` only and a tag is a claim about a release.
- **Three commands ship; `scripts/` does not.** `lucid-yolo` (train/validate), `lucid-data` (`download` · `check` · `build-tiles`) and `lucid-eval` (acceptance scoring, on the protocol the checkpoint's own task names) live in `lucid_yolo/cli/` and cover a whole tier run from a wheel alone (WP-096). What stays in `scripts/` is development tooling — goldens, overfit gates, figures, the release guard, the licence audit — and a remote run needs none of it. All three parse with jsonargparse, so flags are underscored (`--data_root`) exactly as `--data.batch_size` already was, and every command takes `--config`. `lucid-download` survives as a deprecated alias with its original dashed flags, removed in 0.4.0.

## 3. Dataset contract

Datasets are never committed and never auto-downloaded by test code. `configs/data/*.yaml` carries the root path; `lucid-data check` (also `make check-data`) validates layout and counts before any [DATA] WP runs against real data. **Where each dataset comes from and what has to be on disk: `docs/DATASETS.md`.**

- **COCO 2017** (R12): `train2017/`, `val2017/`, `annotations/instances_*.json`; 118,287 train / 5,000 val images.
- **DOTA-v1.0** (R18): original images + labelTxt; 2,806 images / 188,282 instances / 15 classes across all three splits, of which train and val are the annotated two thirds — the testing ground truth is withheld by the authors, so it is neither downloaded nor needed. Provisioning is manual (the distribution offers interactive Drive folders, not archive URLs) and the terms are academic use only. 1024 px tiling is a build artifact, never committed — build it with `lucid-data build-tiles --root <root> --out <tiles>`, which writes the COCO layout `obb_smoke.yaml` names. The tier trains on the tiles, never on the original tree: DOTA images run to several thousand pixels a side in PNG, which has no random-access region decode, so cropping per sample would decode the whole image to yield one 1024 px window (WP-094).
- Offline development against [DATA] WPs may use the synthetic stand-in generator (A26/D12b); **tier acceptance runs require real data** — missing real data blocks the tier, it is not worked around.

## 4. Escalation protocol (the anti-guessing rule)

Stop and write a `docs/ESCALATION.md` entry — symptom, WP, hypotheses tried, sources consulted, decision requested — when any of these occur:

1. A gate fails after **two** documented assumption iterations.
2. Answering a question would require a denylisted source.
3. A WP spec conflicts with the technical specification or the papers.
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

- `WP:` exactly one WP id. `Provenance:` at least one source id from docs/PROVENANCE.md (papers/primary literature only). `Assumptions:` A-ids touched, or `none`. `Gate:` the test id(s) proving the DoD.
- Types: `feat` `fix` `test` `ci` `docs` `chore` `perf` `refactor` `refine` `exp` `release`. `refine` is this project's own addition, for a change that improves something already correct — a rename, a clearer boundary, a sharpened comment: `refactor` claims behaviour-preserving restructuring and `docs` claims prose, and a rename crossing code, configs and prose is neither.
- No commit message references the Ultralytics repository; papers are cited by arXiv id. Hash-sign and at-sign characters never appear in subject or body (co-author trailer emails are the sole exception).

## 6. Naming hygiene

Internal identifiers use descriptive names derived from the papers' terminology (`DualAssignHead`, `SmallTargetAssigner`, `ProgressiveLossSchedule`, `MuSGD`, `PrototypeMaskHead`, `LongEdgeOBBHead`) — never from any implementation's symbol names (we have not seen them, and it stays that way). Generated code for assignment/loss/optimizer/rotated-geometry modules is written by hand from the equations — AI assistants may regurgitate AGPL training data, and these modules are the highest-risk surface.

## 7. Standing prohibitions

Never: consult or install `ultralytics` or any mirror; create a model-topology config format (ADR-001); copy code from any external detection repository (check LICENSE and provenance before consulting *any* external detection repo — this includes third-party YOLO-seg/YOLO-OBB forks); plan, promise, or tag a 1.0 (ADR-002); modify a frozen golden; start a [HUMAN] WP; commit datasets or downloaded weights; leave `main` red; download, fine-tune, distill from, or compare against released Ultralytics checkpoints.

## 8. Delegated work packages

A completion notification is not a completion. When a WP is delegated, the harness reports the task `completed` whether the agent finished or stopped mid-sentence on a partial edit — the notification carries a `result` field that looks like a report, so an unfinished run reads as a finished one that summarized badly. This happened four times in Phase 8, twice discovered only by inspecting the worktree.

Before treating a delegated WP as done, check the worktree's `git status` against what the spec asked for; a missing test file or an untouched entry-point module is the tell. Resume with a message naming what is still missing rather than respawning — the agent picks up from its own transcript with full context. The spawn prompt's "report when done or when stuck" clause helps and does not prevent this.
