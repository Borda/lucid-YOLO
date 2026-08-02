# Decision Record

Architecture- and policy-level decisions. D1–D11 are transcribed from the
governing blueprint (v1.5, 2026-07-10); D12 records execution-session
amendments (2026-07-31). ADRs expand the three decisions with lasting
architectural consequences.

## Decisions

| ID | Decision | Resolution |
|---|---|---|
| D1 | Scope | Detection + instance segmentation + OBB, delivered as sequential release-train milestones (0.1 → 0.2 → 0.3, see D10) on top of the fully gated detector. Pose and classification remain future 0.x work; YOLOE-26 out of scope entirely. |
| D2 | Reproduction claim | Faithful method reproduction (from-scratch training), not exact-number reproduction. Targets are the paper's relative/ablation claims and the from-scratch reference in R1 Table 4. No Objects365 pretraining, no evolutionary hyperparameter search. |
| D3 | Scale strategy | Family-generic code. Iterate/debug at n-scale, headline runs at s-scale (all paper ablations are s-scale, including seg and OBB). |
| D4 | Framework | PyTorch Lightning LightningModule + Trainer, automatic optimization. MuSGD is a single custom torch.optim.Optimizer; ProgLoss is an epoch hook; close-mosaic and EMA are callbacks. Task heads reuse the same LightningModule with task-conditional loss composition. |
| D5 | Source allowlist | Strict: the three papers + their cited primary literature + neutral dataset/tooling documentation only. All Ultralytics web properties — including docs.ultralytics.com — are inadmissible, alongside source code, YAML model configs, and released weights. Full policy in AGENTS.md and docs/PROVENANCE.md. |
| D6 | Muon core | Implement Newton–Schulz orthogonalization from scratch per R7/R8; no third-party optimizer dependency. |
| D7 | Identity | Repository/package name `lucid-yolo` (personal repository; employer sign-off required before any organizational nexus). "YOLO" used as the model-family/category term; naming evidence in docs/PROVENANCE.md. "YOLO26" and "Ultralytics" never appear in package, module, or model identifiers. |
| D8 | Verification | Published tables only. Fidelity gates use printed param counts, FLOPs, and mAP values from R1/R2 — Table 7 (detection), Table S9 (segmentation), Table S11 (OBB). No black-box runs of the reference package anywhere in the 0.x series. |
| D9 | Configuration strategy | Architecture in code; YAML for experiments only. See ADR-001. |
| D10 | Versioning and release policy | Perpetual 0.x — no 1.0 milestone is planned. See ADR-002. |
| D11 | Execution model | Autonomous agent execution under human gates. See ADR-003. |
| D12 | Execution-session amendments (2026-07-31) | (a) **Git flow**: work lands as one commit per WP directly on local `main`; `make gate` green is the merge gate (replacing the blueprint's PR/squash-merge flow — single-operator repository, no branch protection available locally); pushes to the remote are batched at phase boundaries and each push requires explicit human confirmation. (b) **Synthetic fixtures and data stand-ins**: test fixtures and offline [DATA]-WP development stand-ins are generated with fuse-augmentations (R21, Apache-2.0, pinned commit) instead of hand-annotated images — recorded as A26; real COCO/DOTA remain mandatory for tier acceptance. (c) **Accelerator**: [GPU] WPs attempt Apple MPS locally first; CUDA runs remain [HUMAN]-gated. (d) **Lint routing**: all linters (ruff check/format, mypy, hygiene hooks) run exclusively through `pre-commit run --all-files`; `make gate` = precommit + test + golden. (e) **Subpackage naming** (2026-08-01): the Lightning-integration subpackage is `src/lucid_yolo/ptl/` (blueprint sec. 7 named it `lit/`; renamed to avoid `lucid_yolo.lit` stutter); Phase 5 commit scopes use `feat(ptl)`. (f) **Package rename** (2026-08-02): distribution `lit-yolo` -> `lucid-yolo`, import `lit_yolo` -> `lucid_yolo`, console scripts `lucid-yolo`/`lucid-download`, variants `lucid-yolo-e2e-{n..x}`. Rationale: the `lit-` prefix is Lightning AI's own project-naming convention (lit-llama, lit-gpt) and the project must not borrow any third party's branding; `lucid-` states the project's education/readability goal. Alternatives ruled out during the decision: `open-yolo` (occupied PyPI name) and plural `*-yolos` forms (collide with the unrelated YOLOS ViT detector, hustvl "You Only Look at One Sequence"). Verified at decision time: `lucid-yolo` free on PyPI (404) and one inactive hobby repository on GitHub. Naming-evidence record and sec. 3.5 discipline unchanged. |

## ADR-001 — Architecture in code; YAML for experiments only (D9)

**Status**: accepted (blueprint v1.2).

**Context**: YOLO-lineage reference implementations describe model topology in
YAML DSLs. This project must neither consult nor resemble that expressive
artifact.

**Decision**: model topology and blocks are typed Python — builder functions
plus a 5-row scale-multiplier dataclass registry. YAML (via
LightningCLI/jsonargparse) covers only run-level configuration: data paths,
tier schedules, optimizer/loss gains, augmentation strengths.

**Consequences**: (i) clean-room — a layer-list YAML DSL would structurally
converge on the reference implementation's model-YAML format; (ii) the paper
defines exactly one topology with five multiplier rows — a configurable graph
engine is over-engineering; (iii) the Phase 2/7/8 param/FLOP gates test Python
constructors directly, and mypy covers what a DSL cannot. No model-topology
config format may ever be created (standing prohibition, AGENTS.md).

## ADR-002 — Perpetual 0.x release train (D10)

**Status**: accepted (blueprint v1.3).

**Context**: the project tracks a living specification — the paper plus this
project's assumption register. A 1.0 would imply an API-stability contract and
a completeness claim a research reproduction should not make.

**Decision**: SemVer 0.MINOR.PATCH forever. Each 0.MINOR is a gated capability
milestone (breaking changes permitted, called out in release notes); PATCH
covers fixes, docs, and assumption-register revisions whose results stay
within golden tolerances. Release train: 0.1.0 = detector (Phase 6 gate),
0.2.0 = +segmentation (Phase 7), 0.3.0 = +OBB (Phase 8), 0.4.0+ = rolling.

**Consequences**: each 0.MINOR freezes its golden metrics; later releases must
never regress any frozen golden — frozen goldens are immutable, a genuine
correction ships as the next 0.MINOR with an explicit changelog note. Every
release ships with resolved run configs + seeds, its report section, a
changelog, and weights where dataset licenses permit. No 1.0 is ever planned,
promised, or tagged.

## ADR-003 — Autonomous execution under human gates (D11)

**Status**: accepted (blueprint v1.5); amended by D12(a) for local git flow.

**Context**: the roadmap decomposes into 67 work packages (docs/ROADMAP.md),
each independently implementable and gateable.

**Decision**: an agent executes WPs in dependency order without per-step
approval, subject to the contract in AGENTS.md. Each WP is one commit that
must leave `main` green. Three classes require explicit human action and are
marked [HUMAN]: (a) compute-heavy tier runs (>4 GPU-hours), (b) release tags,
(c) legal/nexus review (O3) and dataset-licensed weight publication (O4).

**Consequences**: agents never widen the source allowlist, never create
model-topology configs, and stop rather than guess (escalation protocol in
AGENTS.md and docs/ESCALATION.md). The assumption register plus the
escalation log is itself research output a from-code port could not produce.

## Open items

| ID | Question | Deadline | Default |
|---|---|---|---|
| O1 | GPU budget → tier commitment per task | before Phase 6 | All three A+B tiers, B on rented spot instances, sequenced Det → Seg → OBB; each 0.MINOR ships at A-tier if B still queued |
| O2 | Project name | resolved 2026-08-02 (D12f) | `lucid-yolo`; YOLO26 implementation exposed under feature name `e2e` |
| O3 | Employer nexus + counsel review of README/NOTICE naming | before the repository or any tag is public (before v0.1.0) | blocked-on-human |
| O4 | OBB weight release policy given DOTA academic-use terms | before v0.3.0 | Code + report Apache-2.0; DOTA-trained weights withheld, documented reproduction recipe |
