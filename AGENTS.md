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

Pushes to the remote are batched at phase boundaries and each push requires explicit human confirmation (D12a). Release tags are always [HUMAN].

## 2. Environment

- Python 3.11; PyTorch >= 2.4. CPU wheels sufficient for WP-001…WP-044 except integration runs; [GPU] WPs attempt Apple MPS first (D12c).
- `make setup` — venv, editable install with dev extras, pre-commit hooks.
- `make gate` — pre-commit (all linters, D12d) + offline pytest + golden suite + frozen-golden regression. **The single command that decides whether a commit may land.**
- `make freeze-goldens MINOR=0.N` — release WPs only.
- All unit gates run **offline**: no network, no dataset, no GPU. Only WPs marked **[DATA]** or **[GPU]** need more.

## 3. Dataset contract

Datasets are never committed and never auto-downloaded by test code. `configs/data/*.yaml` carries the root path; `make check-data` validates layout and counts before any [DATA] WP runs against real data.

- **COCO 2017** (R12): `train2017/`, `val2017/`, `annotations/instances_*.json`; 118,287 train / 5,000 val images.
- **DOTA-v1.0** (R18): original images + labelTxt; 2,806 images / 188,282 instances / 15 classes. 1024 px tiling is a build artifact, never committed.
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
