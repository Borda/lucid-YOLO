# 🛠️ Engineering Log

What executing the work packages taught about the repository itself, split out of `docs/RESEARCH_LOG.md`: CI and commit-time gates, packaging and licensing, formatting and doc-generation tooling, environment reproducibility, and the CLI/report code the reproduction runs on top of but that carries no fidelity claim of its own. The two logs are split by claim, not by work package -- a WP whose finding straddles both gets one entry in each, cross-linked; most don't straddle and land in exactly one.

**This file is not a register**, for the same reason `RESEARCH_LOG.md` is not: a choice the papers left open belongs in ASSUMPTIONS.md, cited by id rather than restated. Release-level results live in REPRODUCTION_REPORT.md.

Sections are anchored by work package (`#wp-XXX`) and the roadmap links to them. A package with nothing worth recording here has no section -- and a package whose whole finding is a fidelity claim rather than tooling has none here at all; see `RESEARCH_LOG.md` for it.

Dates are the day the finding was recorded.

______________________________________________________________________

## 🧵 Cross-cutting

### Float reproducibility has two axes, not one

<a id="cross-float"></a>

A26 already records that the synthetic fixture generator is byte-identical per seed **on a given platform** and not across platforms — libm last-bit rounding differs, macOS arm64 against ubuntu x86-64 CI. That entry is about image generation. The same fault line runs through **torch reductions**, and it was found the hard way on 2026-08-14.

`test_obb_training.py` carried a snapshot of a `task="detect"` training step asserted with `==`, captured on arm64 at one intra-op thread. Three values of that same total have now been observed from source trees computing identical arithmetic:

| total | where |
| -- | -- |
| 22.80110550 | arm64, 1 intra-op thread (the recorded snapshot) |
| 22.80111694 | arm64, 12 threads |
| 22.80112076 | x86-64, ubuntu CI |

The spread is 1.5e-5 at magnitude 22.8 — about eight float32 ulps — and comes from the summation order a build chooses for the dense classification term. The test passed for the whole of Phase 8 because until 0.3.0 was pushed it had only ever run on the machine its snapshot came from; its first x86 execution failed it. It now compares at a relative 1e-4 with an absolute floor.

The general rule this establishes, beyond that one test: **bit-identity is a within-architecture property.** A gate that asserts it across machines is asserting something no library promises. What survives across architectures is a tolerance argued from ulp count, and what such a gate can still catch is a change to the objective — those move by percent. What it can no longer catch is a change of one rounding step. No tolerance choice recovers that.

### A green gate is only as trustworthy as the interpreter it ran under

<a id="cross-venv"></a>

Mid-session on 2026-08-13, `uv run` silently rebuilt `.venv` — Python 3.14.2, torch 2.13 — replacing the 3.11.16 environment a passing `make gate` had just run under. Nothing announced it; the first sign was unrelated behaviour changing. Restored with `make setup`, and every later invocation named `.venv/bin/python` explicitly rather than going through `uv run`.

The general point: "the gate is green" is a claim about an environment as much as about a tree, and a tool that can reconstruct that environment between two commands can invalidate the claim without touching a tracked file.

### Deterministic SVG is deterministic per matplotlib, not per run

<a id="cross-svg"></a>

WP-080 fixes `svg.hashsalt` and suppresses date metadata so an unchanged run regenerates its figure byte-identically. That holds — within one matplotlib. On 2026-08-14, regenerating the committed detection and segmentation figures produced files differing from the committed ones, and the cause was isolated by regenerating with the *git-stashed original* script: the committed script also differs, so the difference is the renderer, not the edit.

The committed figures were left untouched rather than churned. The consequence to know: a figure diff is not evidence that a plotting change altered a curve. Prove which by re-rendering with the pre-change script before reading anything into it.

### Two process failures that each cost a run

<a id="cross-process"></a>

Both recurred often enough to be written into AGENTS.md rather than left in notes.

**The gate certifies a tree, not an intention.** The loop runs `make gate` at step 4 and flips a roadmap row at step 6, and the gap between them is where a docs edit rides into a code commit that was never gated with it. That is how commit `fcf3040` left `main` red on two meta tests, and the next agent spent a full run reporting a blocker it was forbidden to fix. The invariant is a set equality between what was gated and what is committed; the docs-only exemption is the thing it is constantly confused with.

**A completion notification is not a completion.** A delegated work package returns marked *completed* whether the agent finished or stopped mid-sentence on a partial edit, and the notification carries a result field that reads like a report — so an unfinished run looks like a finished one that summarized badly. It happened **four times in Phase 8**. The check that catches it is one `git status` against the spec, and resuming the same agent beats respawning it, because its transcript is the context.

______________________________________________________________________

## 🧱 Phase 0 — Foundation

### WP-001 — one place to write the version

<a id="wp-001"></a>

The version was declared twice and the two **had already drifted**: `pyproject.toml` said `0.1.0` while `lucid_yolo.__version__` said `0.0.1.dev0`, so a build published one number while every runtime consumer reported the other. `pyproject` now declares the version dynamic and reads the module attribute.

The attribute stays a plain string **literal** deliberately: setuptools resolves a literal by static analysis and only falls back to importing the package when it cannot, so a computed value would pull torch into every build and every source install. A meta test pins the literal form for that reason.

Also recorded — a test deliberately **not** written: `importlib.metadata` asserted against `__version__`. An editable install freezes its metadata at install time, so that assertion would fail after every bump until someone reinstalled, i.e. fail for a reason unrelated to what it claims to guard.

### WP-001 — markdown formatting, and what it found

<a id="wp-001-mdformat"></a>

Directly relevant to why this file exists.

`--compact-tables` was added on evidence rather than preference. **Table padding is charged per column at its widest cell**, and this repository's governance rows run past 2,500 characters, so padded output grew the tracked markdown from **162 kB to 375 kB — all of it trailing spaces**. Compact output is size-neutral at 161 kB. A single very wide cell taxes every other row in its table.

The GFM plugin is required, not optional: core mdformat is CommonMark-only, reads a GFM table as a paragraph, and `--wrap=no` would then collapse every row onto one line.

The content-preservation check — strip whitespace, cell pipes and backslash escapes, compare the remaining characters before and after — **found a real defect**. A roadmap row held an unescaped pipe inside a code span, and a pipe opens a table cell wherever it appears: the row parsed as **eight cells against a six-column header**, so GitHub had been rendering its tail into the wrong columns and silently dropping the overflow. A sweep found no other row in the repository whose cell count disagreed with its header.

### WP-004 — auditing what a wheel *bundles*, not what it declares

<a id="wp-004"></a>

The copyleft audit read what a distribution says about itself — `License`, `License-Expression`, OSI classifiers — and nothing about what it ships. A wheel may vendor a native library under a licence its own metadata never mentions, which is precisely the case the gate exists to catch.

Reading the `License-File` documents too, it found something on the first run:

| package | declares | bundles |
| -- | -- | -- |
| numpy | BSD-3-Clause | libgfortran, libgcc — GPL-3.0-or-later **with GCC Runtime Library Exception**; libquadmath — LGPL-2.1-or-later |
| shapely | BSD-3-Clause | GEOS binaries — LGPL-2.1 |

numpy is a runtime dependency, not a test-only one, and no audit this project had run could ever have seen it.

Three rules keep the result meaningful rather than merely red:

- **Only declaration lines are read, never licence prose.** A bundled copy of the LGPL names the GPL on dozens of its own lines, and a gate that fires on every wheel shipping a licence file teaches its reader to bypass it.
- **A recognized licence exception is permissive by construction**, matched by expression rather than by name, so a future gcc-built dependency needs no entry.
- **The exception is tested before the allowlist, not after.** Written the other way round the allowlist short-circuits first, the exception never runs, and a per-package entry quietly covers whatever that package vendors *next*. Mutation testing is what caught the ordering: with numpy delisted the audit reports libquadmath alone, and with the exception disabled it reports the GPL-3 line — evidence the two mechanisms act independently rather than one masking the other.

A related gate lesson from the same package: the decisions test moved from a hardcoded `D1–D14` to **contiguity plus a floor**. The hardcoded upper bound failed the moment a decision was added, which is a test asking for maintenance rather than reporting a defect; contiguity alone would not notice the last row being deleted, since what remains stays contiguous. The roadmap's `_WP_FLOOR` ratchet is the same pattern.

### WP-005 — goldens that survive a second machine

<a id="wp-005"></a>

The first fixture golden byte-hashed the generated annotation JSON. Those hashes are per-platform stable only — see [Float reproducibility](#cross-float) — so they were replaced with structural metrics: exact integer counts (images, annotations, categories, polygon points) plus bbox area and coordinate sums pinned with a small absolute tolerance.

The split that has held ever since: **integers exact, aggregates to tolerance**. Counts are platform-stable by construction; sums of floats are not. Every golden this project has frozen follows it.

______________________________________________________________________

## ⚡ Phase 5 — Lightning training loop

### WP-038 — four defaults that only fail off the developer's machine

<a id="wp-038"></a>

Each was found by running somewhere other than a local checkout, and none is visible in a unit test.

**Strict determinism aborts on MPS.** Auto-picked MPS killed a tier run mid-step: `deterministic=True` hits `index_put_with_accumulate_mps`, which has no deterministic MPS kernel. The default is now accelerator-aware — `"warn_only"` where MPS is available, strict `True` on CPU/CUDA — and the shipped configs dropped their pins, since a config value would override the accelerator-aware default.

**Configs were not in the wheel.** They lived at the repo root and never entered the built distribution, so a wheel-only install (Colab) could not run `--config` at all. Moved to package data with a repo-root symlink so documented commands keep working, then given name resolution (`--config det_smoke` resolves onto the installed tree when no local file matches) and a default recipe, because a wheel install has no checkout to write a path into.

**Rich floods notebook output.** Lightning auto-picks `RichProgressBar` whenever `rich` is installed — it is, transitively — and Rich live rendering prints one line per refresh in a Colab cell. A newline flood every step. Default is now `tqdm.auto`, which renders a widget in notebooks and a single-line bar in terminals.

**Loggers.** An unset `trainer.logger` resolved to TensorBoard only, so a run left no plain-text metrics. It now resolves to a TensorBoard + CSV pair pinned to the same `version_N` directory — which is what makes every later `metrics.csv` analysis in this project possible at all.

## 🎯 Phase 6 — Evaluation, release 0.1.0

### WP-079 — per-worker, per-epoch augmentation RNG

<a id="wp-079"></a>

`_TrainPipeline` seeded one generator in the **parent** process. A parent's generator never advances when workers are used, so every worker replayed one identical augmentation-parameter stream — diversity `1/num_workers`. WP-076 then made workers non-persistent, so the pool restarted from that same state every epoch, replaying a single epoch's parameters for the entire run.

What the two defects together cost the training-data variety is measured at [RESEARCH_LOG.md#wp-079](RESEARCH_LOG.md#wp-079).

### WP-083 — synthetic-shapes generalization golden

<a id="wp-083"></a>

The cheap stand-in for a tier run: 2000 generated scenes split 1800/200, n-scale, 6 epochs at 320 px, scored on held-out images through both decode paths. Unlike the overfit-100 gate it never scores what it trained on, so it measures generalization rather than loop composition, and mosaic stays on for five of six epochs so a collapsed augmentation RNG ([WP-079](#wp-079)) is detectable rather than merely absent.

The frozen numbers this golden pins are recorded at [RESEARCH_LOG.md#wp-083](RESEARCH_LOG.md#wp-083).

### WP-084 — commit-time gates

<a id="wp-084"></a>

Three checks were moved to where they can catch something, and the reasoning generalizes past this repository.

The **commit-trailer validator** was a CI job. Under a squash merge the per-commit messages CI validated are replaced by the squash message — so the job gated text that never lands. It belongs in a `commit-msg` hook or nowhere.

The **copyleft audit** went the other way: it stays a standalone always-run job rather than a diff-triggered one, because it scans the installed environment. A transitive copyleft bump touches no tracked file, so a diff-scoped trigger would never fire on the case it exists for.

The **`dev` extra** became a PEP 735 `[dependency-groups]` entry, which is what stops build/lint/test packages advertising themselves in the published wheel's metadata (`Provides-Extra: None` on the installed dist is the check).

Test-suite and docs-updated checks were deliberately left unhooked — a pre-commit hook that runs the suite trains the operator to pass `--no-verify`, which is worse than not having it.

### WP-085 — doctests join the offline gate

<a id="wp-085"></a>

Every public function is required to carry an `Examples` section, and **no gate had ever run one**: `testpaths` pointed at `tests/` alone and `addopts` carried no `--doctest-modules`. Four examples had rotted unnoticed over roughly eighty work packages.

The worst was not stale — it was never true. `orthogonalize`'s example asserted near-exact orthogonality that the Newton–Schulz coefficients cannot reach, and which that function's own unit test documents as unattainable. It could not have passed at any seed, and was additionally unseeded. It now states the guarantee that actually holds (no singular value expanded). The other three pinned stale float reprs (`0.01` against `0.010000000000000009`; `0.01...` against `0.00999...`).

Wiring it up took the offline suite from **558 to 699 cases** and coverage from 96% to 97%. The lesson recorded for later gates: a requirement nothing executes is a documentation convention, not a gate, and the two are indistinguishable from a green run.

______________________________________________________________________

## 🔄 Phase 8 — Oriented detection, release 0.3.0

### WP-105 — a report that survives

<a id="wp-105"></a>

Two defects the first real oriented evaluation exposed, both in the CLI rather than in what it measures.

`--output` into a directory that did not exist yet raised **after** the scoring pass finished and printed its numbers — minutes of accelerator work producing a traceback and nothing else. The parent is now created rather than required, because the file is the only durable form of the expensive part. The general rule: a long computation's output path should be validated or created *before* the computation, never discovered after it.

Second, a 10,132-tile pass showed one line at the start and nothing until it ended. The run has a known length; it now draws a progress bar, wrapped at the CLI rather than inside the evaluator, which tests drive in-process.

______________________________________________________________________

### WP-094 — the difficult flag died at the loader

<a id="wp-094"></a>

[WP-057](RESEARCH_LOG.md#wp-057) manufactures difficult instances at crop time and A51 defines the channel they are supposed to travel on, but nothing connected the two: the reader did not forward `difficult` out of the annotation record, so every instance the tiler had flagged arrived at the loss indistinguishable from a whole one. Nothing disagreed anywhere — the counts are right, the boxes are right, and the only thing wrong is that a protocol decision the register spent a row on was not in force.

A53's non-schema keys are what make the flag survivable through a COCO container at all, since COCO's schema has no field for it or for the tile's window provenance, and a standard reader ignores both and still sees an ordinary detection set.

### WP-095 — a run monitor is not an acceptance instrument

<a id="wp-095"></a>

The oriented tier reached its release row with no way to produce an acceptance number. What existed was the epoch metric `val/rotated_mAP` ([WP-102](RESEARCH_LOG.md#wp-102)), and that is a run monitor: it scores whatever the training loader hands it, under torchmetrics' own protocol rather than A46's recall grid, A47's cap and A48's difficult rule. Two figures computed under two protocols are not the same measurement, and the one a release quotes has to be the registered one.

One decision inside the instrument is worth recording. It scores in **letterbox coordinates, with the ground truth letterboxed alongside**, rather than inverting the predictions back to tile coordinates first. Both sides then pass through one geometry, which makes the comparison an equality rather than a proxy — the inverse is where a frame error would enter, and here it is not on the path at all. The same reasoning put checkpoint loading in `lucid_yolo.eval.checkpoint` instead of a second copy: one loader, one place for a task to be read wrong.

### WP-096 — a rename with a published cost

<a id="wp-096"></a>

Parsing moved to jsonargparse so that flags and help text derive from the **operation signatures** rather than from a hand-maintained parser that is free to disagree with the function it calls. The cost is real and is stated rather than absorbed: jsonargparse spells a parameter with underscores, so every documented flag changed shape in one commit.

`lucid-download` therefore survives as a deprecated alias rather than being dropped. It is named in published reproduction instructions, and an instruction that no longer runs is a worse outcome than a duplicate entry point — the alias keeps accepting its **old** flags, which is the part a copy-pasted command depends on.

### WP-097 — a published total no correct download can reach

<a id="wp-097"></a>

`check_dota_root` compared its summed train+val totals against R18's whole-dataset figures ([RESEARCH_LOG.md#wp-097](RESEARCH_LOG.md#wp-097)), so it failed on every correct download, at the first command an operator is told to run.

The fix is not a looser tolerance but a different question. A count the caller did not state is **reported**, not compared: on an unstated expectation the check's job is to say what it found, and a number with no stated referent cannot fail. What still fails is a stated expectation that misses — and, new here, a DOTA expectation aimed at a COCO root, which had previously been accepted and silently ignored.

Recorded because the failure was in the gate rather than in the data, and a gate that fires on every correct input is the one shape of gate that teaches its operator to disbelieve gates.

## 🔮 Phase 9 — Inference and generalization

### WP-089 — the grid was already written three times

<a id="wp-089"></a>

The package exists because a letterbox inverse must not be written twice: two copies of one transform, each with its own passing tests, are free to disagree by a pad the day either side's rounding changes. That part went as intended — nothing in the predict path computes a ratio, a pad or a corner.

What the work found is that the *other* geometry had already made the mistake. The head's anchor grid — divide the canvas by the three strides, build the points, move them to the device — was written once in the dual-path evaluator, once in the oriented evaluator, and would have been written a third time here. The stride triple `(8, 16, 32)` had three separate declarations to match, two of them private. Nobody duplicated it carelessly; each consumer needed three lines and wrote them. That is how the duplication a project actively guards against still accumulates: the guarded case is the one everyone can see is dangerous, and the neighbouring case looks too small to bother with.

Consolidated into `assign/grid.py`, beside `make_anchor_points`. The general signature won: a canvas is `(height, width)`, and a square side cannot express a letterbox that is not square. Every golden was unchanged afterwards — 20/20 — which is the only evidence that a refactor of the scoring path changed nothing.

A second finding, recorded because it is a trap the next two packages walk into: `DetectionLitModule.forward` returns a `DualHeadOutput` for **every** task. A segmentation or oriented checkpoint handed to a detection predict path therefore produces boxes, silently, with its mask or angle branch never consulted — plausible output, no error, and the masks the caller asked for simply absent. The refusal is in the library rather than in the command for that reason: a wrong answer that looks right is worse than a traceback, and the command is not the only door.

Two citations that belong with the above rather than in the roadmap row. The duplicated-inverse hazard has a name in this project — it is the **WP-053a defect class**, the shared-decode function whose second caller found what its first never could ([WP-090b](#wp-090b) is the same class caught later). And the task string is read from the checkpoint by the same mechanism `lucid-eval` reads it, so the two commands cannot disagree about what a checkpoint is: a refusal that depended on which entry point asked would be worth less than no refusal at all.

### WP-090 — a crash the evaluator could never reach

<a id="wp-090"></a>

`decode_instance_masks` raises on a zero-length instance axis: `F.interpolate` rejects a `[1, 0, 16, 16]` tensor outright. The function has been in the tree since WP-053a and no test caught it, because from the evaluator it is unreachable — both decoders always hand it a fixed 300 rows, padding included, so the axis is never empty. Predict decodes only the survivors, and an image with nothing above the confidence threshold has none.

The general shape is worth keeping: a function whose only caller pads to a fixed length has never been tested at its degenerate input, and the untested case arrives with the second caller rather than with the first. Guarded at the call site here, with the crash pinned by a test on both decode paths; the shared function itself is left for its own package.

Two things this package had to get right that a shape check cannot see. The masks are assembled while the boxes are still on the letterboxed canvas, because that is the frame the A11 box crop is defined in — cropping after the inverse letterbox leaves every box right and every mask quietly wrong. And the mask path is **not** decoder-independent: the one-to-one coefficients are what the E2E decode reads and the dense ones are what the suppression path reads, so pairing either decoder's rows with the other branch's coefficients yields a plausible mask of a different object. The test plants different prototypes per branch, so a path reading the wrong one returns a mask on the other half of the picture and fails rather than passing with a mask nobody looks at.

For the report, COCO RLE through `faster_coco_eval.mask.encode` — already a declared dependency, and already the encoder every `segm_` statistic is measured through, so a predicted mask on disk and a scored mask are the same object in the same format. Rejected: a polygon contour needs a tracer and is lossy on masks with holes, which the box crop routinely produces; a sidecar `.npz` splits one prediction across two files, so a report can be archived into silently describing masks that are gone; mask-derived scalars answer a smaller question than the caller asked.

### WP-090b — the degenerate input arrives with the second caller

<a id="wp-090b"></a>

`decode_instance_masks` has raised on a zero-length instance axis since WP-053a wrote it. `F.interpolate` treats that axis as channels and rejects an empty one outright. No test caught it and no reviewer would have: from the evaluator the case is unreachable, because both decoders always hand the function a fixed 300 rows with their padding included, so the axis is never empty. Predict decodes only what survives the confidence cut, and an image can have nothing above it.

The rule worth keeping: **a function whose only caller pads to a fixed length has never been exercised at its degenerate input, and that input arrives with the second caller rather than the first.** The shape of the bug is not "someone forgot the empty case" — it is that the empty case did not exist while there was one caller, so no amount of care at the time would have surfaced it.

Returning the empty stack rather than raising is the shape contract, not politeness: a caller assembling per-image results needs a tensor of the right rank and dtype to stack, and an exception pushes the count check out into every such caller, which is where the copies start.

### WP-091b — the tie-break decoder implementation

<a id="wp-091b"></a>

The decoder itself is the small half. The rule is greedy class-wise suppression whose overlap is `rotated_iou` — the same exact polygon measure the oriented evaluator grades with — over boxes canonicalized first, so two detections stated a half turn apart cannot read as two objects. Class separation tests the label directly rather than borrowing `batched_nms`'s coordinate-offset trick, because there is no offset that separates rotated boxes without also turning them.

Why its overlap threshold has no paper source to cite, and what that cost to establish, is recorded at [RESEARCH_LOG.md#wp-091b](RESEARCH_LOG.md#wp-091b).

### WP-099 — a format with no specification

<a id="wp-099"></a>

The clean-room rule says the reader is written from the format as the datasets publish it, never from an implementation's reader. That is a clear instruction with a hole in the middle: the YOLO label format has no specification on the paper allowlist, and the obvious ways to learn it — a search, a docs page, somebody's loader — are all the thing the rule forbids. What closed it was a published dataset export already on the machine, CC BY 4.0, a data artifact with no implementation lineage: registered as R32, and read for what it actually contains rather than what a tutorial says it contains.

That got four facts and left four gaps, and the gaps are the useful part. The export's fields lie in `(0, 1]` and its clipped corners land on exactly `0` and `1`; `names` is an index-ordered list beside `nc`; split entries are written `../train/images` for a tree at `<root>/train/images`; every image has a label file and none is empty. What it could not settle: whether a ten-field oriented row exists (R18 appends a `difficult` flag to its own label lines, and the normalized variant has no published spelling for one), whether `data.yaml` may carry a `path:` key, which of the two directory spellings ranks first when a root satisfies both, and which image extensions are in scope. Each of those is rejected or documented as a precedence choice rather than guessed — A54 through A58 — because a guess that silently mis-parses is exactly the failure the clean-room clause exists to prevent, and it fails quietly by producing a dataset rather than an error.

Two decisions where the format's own limits set the line. **A missing label file is an error; an empty one is a background image** — writing zero bytes is a positive statement that the image holds no objects, while not writing one is silence, and a tree with no manifest cannot tell silence from a half-finished download. Reading silence as "no objects" turns a broken export into a dataset that trains on backgrounds without ever saying so. **A coordinate outside `[0, 1]` is rejected, not clamped, and compared with no epsilon** — it is not a slightly-off box but a file that was never normalized, typically written in pixels, and clamping turns it into a wall of degenerate edge boxes that would then be trained on.

The layout question was the one that looked smallest and was not. `resolve_split` requires a directory **and** an annotation file, and a YOLO split's annotations are a directory. Folding YOLO into the existing `CANDIDATES` would force that predicate to accept either kind of thing, and a root satisfying both conventions would then resolve by table order — handing a labels tree to the COCO reader, or an `instances_*.json` to this one. Which convention applies is a property of the reader asking, so the reader asks its own table.

What did not land: the datamodule still does not dispatch between the two readers, so a YOLO root is reachable from a library call and not yet from `lucid-yolo fit`. Recorded as 099b rather than absorbed into the row, because a package that reports itself done while a clause of its scope is unbuilt is how a roadmap stops describing the code.

Three details of the shipped surface, kept out of the roadmap row for length. The reader produces the same `Targets` container the COCO path does, which is the property that leaves assignment, loss and metric code untouched by a second dataset format entirely — the format question stops at the loader. Class names and split paths come from the dataset's own `data.yaml` rather than from a config, so a YOLO root carries its own label space. And a layout row was added alongside, so `--data.data_root` resolves a YOLO root the way it resolves every other, through `resolve_yolo_split` and `from_root`.

The clean-room instruction this was written under is AGENTS.md's prime directive, cited here because the register rows (A54-A58) record what was *unknown* and not what forbade looking it up.

### WP-099b — the tie-break that must not exist

<a id="wp-099b"></a>

`resolve_split` already had a precedence rule: candidate spellings are tried in table order and the first match wins. Reusing that shape across the two *readers* is the obvious move and it is wrong, for a reason worth stating in one line — **within a table the loser is another spelling of one reader, and between the tables the loser is a different label space, a different image set and a different parser.** A first-match rule in the first case picks the wrong directory of the right dataset; in the second it trains on annotations nobody named and reports a plausible number while doing it. So the probe evaluates both tables in full and returns a pair of booleans: there is no order for the answer to depend on, and ambiguity is refused rather than resolved (A63).

The refusal only works if the operator has a way to answer it, which is what earned the explicit `layout` override its place. Two states are otherwise unreachable: an ambiguous root on a read-only dataset mount, where the only remedy would be moving files; and a YOLO root whose `data.yaml` points its splits outside the convention table, which `from_root` resolves — it is why `val: ../valid/images` works at all — and which no directory probe can see. A knob that exists because two legitimate datasets are otherwise unreadable is a different thing from a knob that exists because one was possible.

The construction-versus-`setup` split was forced by something outside this package. Refusing at construction is the better instinct and it collides with `tests/ptl/test_cli.py`, which instantiates every shipped config through `LightningCLI(run=False)` while `configs/det_smoke.yaml` carries a placeholder `data_root`. A filesystem verdict at construction fails that dry-parse for every run not yet pointed at data. The line drawn instead: everything decidable from the arguments alone raises at construction, and anything needing the disk raises at `setup()` — still before an image is decoded and before the trainer takes a step.

Two absences the YOLO path has to declare rather than discover. `mask_targets` is refused, because the format carries no per-instance rings and rasterising empty ones would produce a segmentation run supervised by nothing that still reports a detection number. Copy-paste is the quieter one: unlike the oriented path it does **not** raise on polygon-free targets — it decodes an entire extra source sample, finds no candidates and does nothing — so at `copy_paste = 0.1` every YOLO run would have paid for an augmentation that can never fire. The suppression keeps consuming the RNG draw, which is what leaves the COCO path byte-identical and the frozen data checksums unmoved. An augmentation that fails silently costs more than one that raises.

One more property of the override, which is what keeps it from costing anything: naming any COCO-shaped path — `train_images`, `val_annotations`, any of them — is itself read as stating the COCO layout. Every caller that predates the probe is therefore already off it, and the probe runs only for a root that has said nothing about its own shape.

### WP-099c — a pre-flight that asks the run's question

<a id="wp-099c"></a>

`check_dataset` defaulted to `coco`, which was correct for as long as COCO was the only layout a run could read. WP-099b ended that, and left the check answering a question its own run no longer asks: pointed at a YOLO root with no flag, it reported a missing `train2017` for a tree `fit` trains on without complaint. The fix is not a third branch but a deletion — an unstated `--dataset` now runs `detect_layout`, the same probe the datamodule dispatches on, so the pre-flight and the run resolve the root identically or not at all. `DATASETS` is built from `DatasetLayout` members rather than string literals, which is what stops the flag and the probe drifting apart later. DOTA stays outside it deliberately: no run reads a DOTA root — it is tiled to COCO first (WP-094) — so a probe covering it would advertise a path that does not exist, and the "satisfies no convention" report ends by naming `--dataset dota` for the operator who has one.

Both of the probe's undecidable states are verdicts about a disk, so both come back as report lines and exit 1. WP-097 already paid for that lesson once, on the first command an operator runs against a fresh provisioning: a traceback where a report belongs reads as a broken tool rather than an unusable dataset.

The extraction that mattered is `resolve_split_dirs`. A validator that resolves splits its own way validates a tree the reader will not open — and the YOLO layout has exactly the case that makes this bite, a `data.yaml` naming `../valid/images` for a directory no convention row spells (A58). Shared code between a feature and its checker is usually a smell, since the checker then inherits the feature's bugs; here it is the point, because the alternative is a check that passes on a different dataset than the one that trains. `scan_yolo_label_file` is the other half: the same grammar `load_yolo_targets` reads, stopping before the denormalization, so checking a split does not decode every image to obtain a pixel scale it immediately discards. Validation that costs an epoch is validation that gets skipped.

`DotaSplitCheck` became `PairedSplitCheck`, and the naming records what the two layouts actually share. It is not that both are "not COCO" — it is that both put one text file per image beside it, paired by stem, so both need the same two-directional pairing report and the same totals arithmetic. COCO's single manifest cannot be checked that way at all. The shape of the annotation decides the container, and the rename says so where a `Yolo`-prefixed twin would have said nothing.

No register row. A64 is still unconsumed, which is worth stating explicitly because the run made three decisions that look like assumption material and are not: the inference default is grounded in A63 and WP-097, the YOLO branch reuses A54–A58 through the reader rather than restating them, and the flag guard extends WP-097's own rule to one more layout. A register row for a decision that already has a documented parent is noise in the register that matters.

One absence left standing: `splits` is still not a CLI flag, for YOLO as it never was for DOTA, so a root shipping only `train` fails on the missing `val`. That is right for a `fit` pre-flight, which needs both, and wrong for anyone checking a partial download — a small follow-up rather than a defect, and recorded here so it is a choice rather than an oversight.

### WP-099d — the code is the specification

<a id="wp-099d"></a>

The gap this closed: `docs/DATASETS.md` said where COCO and DOTA come from and nothing about what either layout looks like on disk or what one annotation contains, while the YOLO layout the project had just started supporting appeared in it nowhere at all.

`docs/DATASETS.md` also carries a standing rule that a layout reference is the first real test of: the code is the specification and the document follows it. Applied here, that means every claim about a directory convention is cited to the module that enforces it — `data/layout.py`, `data/yolo.py`, `data/coco.py`, `data/dota.py` — rather than described from the outside from what the layouts are believed to be.

The alternative is not a worse document but a second specification. Two independent statements of a convention agree on the day they are written and disagree the first time a candidate table gains a row, and the one a reader trusts is the one that cannot be executed.

### WP-108 — a cell grows where its section is missing

<a id="wp-108"></a>

The Scope column had drifted to a 286-character median with a 1709-character worst case, and the drift is not random: of the 32 rows over 400, **12 had no RESEARCH_LOG section at all**. A row whose package never got a section is a row where the only place to write down why a thing was done is the row itself, so the reasoning lands there and stays. The length is the symptom; the missing section is the cause, which is why this package wrote the 12 sections before compressing anything.

The gate was considered and rejected, and the reasoning is worth keeping because it runs against this project's usual instinct. Every other convention here is enforced by a meta test — status icons, six columns, the header's package count, the register ids — and the argument for those is that a documented rule with no gate decays. A length gate is different in kind: the others fail on a fact about the file, while a length gate fails on a judgement about how much a row needs to say, and the cheapest way to make it pass is to delete a claim. A rule whose easiest satisfaction is losing substance should not be automatic. So the median is written into the header as guidance, and the enforcement is a reader noticing that a cell runs several times its column's median.

What the compression turned up is that most of the overflow was already recorded somewhere with an owner. Twenty rows needed no new writing at all: their long clauses restate what the log section, or the assumption register, already carries — A40's exact-under-similarity split, A59's core-ownership rule, A24's polygon measure. That is the duplication the header paragraph has always warned about, visible for the first time because something counted it. Three claims had no other home and were moved rather than cut, the sharpest being WP-064's: that the release discloses its per-tile figures in the report, the model card and the README. A disclosure claim living only in a roadmap cell is one edit from disappearing.

Two rows deliberately got no section: 091d and 091e were ⬜, and their content was a finding of 091c's, recorded there. A log section for unstarted work would be a stub asserting what the package will conclude, which is the opposite of what this file is for — so both rows pointed their `· log` tail at `#wp-091c` instead, and the sections got written when the packages produced something to record, which is the paragraph below this one.

### WP-099e — the gap was one layer above where it looked

<a id="wp-099e"></a>

WP-099c closed by naming this itself: `splits` was still not a CLI flag, so a root shipping only `train` fails on a missing `val`. That was recorded so it would be a choice rather than an oversight, and this is the follow-up. A third-party export whose validation split was never cut is a correct export, and the first command an operator runs against a fresh provisioning refused it over a directory nobody had promised.

What is worth writing down is where the parameter was actually missing. `check_yolo_root` and `check_dota_root` had both taken `splits` from the day each was written — `tests/data/test_check.py` had been calling `check_yolo_root(..., splits=("train",))` all along, because the fixtures only ever build one split. The capability existed; the route from the command line to it did not. So the fix is not a feature the layout checkers lacked but one argument `check_dataset` and `_check_root` dropped on the floor. A gap that presents as missing functionality is worth pricing against the layer below before implementing: the two root checkers were the expensive place to fix this, and the wrong one.

The default belongs to the layout rather than to the call. `_check_root` names `YOLO_SPLITS` or `DOTA_SPLITS` at the branch that already knows which layout it is dispatching to, instead of defaulting `splits` in `check_dataset`'s signature. The two constants are equal today and are documented for different reasons — DOTA's testing third has no public labels, YOLO's `test:` entry is a split no run of this project opens — and an entry-point default would have quietly made them one fact, so that changing either would have to be discovered rather than read.

COCO refuses the flag rather than defaulting it, on the reasoning that already governs `--expected_images` there: `check_coco_root` checks `train2017` and `val2017` against 118,287 and 5,000, and the names and the numbers are a single published fact. There is no subset of that to ask for. Following `_check_flags_apply`'s existing message shape was the whole of the design decision — a flag that silently does nothing makes the report read as though it had been honoured.

The empty tuple is the sharper case, because it inverts the failure mode the rest of the module guards. Every other bad argument here fails loudly; `splits=()` fails *cleanly* — no split checked means no problem found means `PASS: dataset layout valid` printed over a root nothing looked at, a false green from the one command whose purpose is to be believed. It is refused in `_require_splits`, called from `check_dataset` before the layout is even inferred and from both root checkers, because those two are documented as the importable core: guarding only the entry point would leave the false green reachable by the shorter route, and a library caller reading `ok` off the result has no report to notice the emptiness in. The entry-point call is what makes `--splits '[]'` fail at the command line, which is where an empty list is easiest to type by accident.

The annotation is `tuple[str, ...] | None`, and the CLI surface it produces was verified rather than assumed: jsonargparse renders it as a list literal, `--splits '[train]'`, which is the spelling `lucid-data download --splits` has always required — a bare `--splits train` is refused by both. One flag's syntax is not a second thing to learn. The `--help` line for it is less self-describing than `download`'s, the `| None` union suppressing the `[ITEM,...]` metavar; cosmetic, and not worth changing the annotation for.

## 📦 Phase 10 — Consolidation

### WP-066 — an untrained checkpoint cannot tell you whether an export is right

<a id="wp-066"></a>

The architecture-fidelity half of this finding — the exported graph confirming the one-to-one, DFL-free claim — is recorded at [RESEARCH_LOG.md#wp-066](RESEARCH_LOG.md#wp-066). What is here is the export-and-comparison harness, which turned out to be where nearly all of the difficulty was.

The graph half of this package was the easy half. All three deploy paths export first try on the legacy TorchScript tracer at opset 20, in about 0.3 s each, and none of them contains a `NonMaxSuppression` node — 598 nodes and 30 op types for detection, with a single `TopK` where suppression would be. `dynamo=True`, the torch 2.13 default, is not used: it needs `onnxscript`, which nothing else here needs, and the legacy path already emits fully static dims. Subgraphs are walked recursively, since an absence claim a node nested in an `If` body could evade is not an absence claim; no control-flow nodes appear at all. The two `Softmax` nodes are the attention block's, not a distribution-focal-loss box head's — the DFL-free claim showing up in the emitted graph rather than in a docstring.

**The absence of an op proves the graph's shape and nothing about its arithmetic**, so the package also runs the exported model under onnxruntime against the checkpoint it came from. That is where the work was, and almost all of it went into discovering that a freshly initialized checkpoint makes the comparison meaningless while appearing to work.

An untrained BatchNorm in eval mode carries `running_var=1` and `running_mean=0`, so it rescales nothing and the signal attenuates through depth: the neck's P3 output measures 4.1e-05 and the classifier's contribution to its own logit lands near 1e-7. The prior-probability bias is `-log((1-pi)/pi) = -4.595` (A30), and float32 resolution *at* 4.595 is 4.8e-07. The contribution is an order of magnitude below the resolution of the number it is added to, so it vanishes: all 1344 class logits become one identical float32 value, `torch.unique` returns 1, and the top-k over 336 anchors is a 300-way tie. Torch and onnxruntime break that tie differently, both correctly, and a row-by-row comparison duly reported boxes 80 px apart. The head is not broken — the same model in train mode, on batch statistics, gives 1343 distinct logits of 1344 at std 0.372. Only the eval-mode statistics are degenerate, and only until something trains them.

Scaling the classification weights to force separation does not work either: a sweep at 1e2 through 1e5 saturates the sigmoid, and from 1e3 up all 336 confidences collapse to exactly 1.0 — the same degeneracy from the other end. What works is calibrating the BN running statistics with a few forward passes in train mode, then rescaling the stems to a *measured* logit spread rather than a guessed one.

**A second degeneracy was hiding behind the first, and it is the one worth remembering.** With the ranking fixed, the segmentation masks compared bit-exact: zero mismatches out of 4,915,200 pixels. They were all `False`. This head regresses ltrb directly with no DFL and no range cap, so an untrained box stem emits values straddling zero, every decoded box is inverted or empty, and `decode_instance_masks` crops each mask to nothing. A perfect score on an assertion comparing two empty tensors. Biasing the box stems positive gives 300 boxes with interiors and 5.07% mask foreground across 44 rows, and the same assertion then means something. The test asserts its own preconditions — score span, positive widths, non-empty masks — because the failure mode of this comparison is not a wrong answer, it is a vacuous one.

**The oriented head appeared to fail and did not.** Row-by-row it diverged by 88 px against 3e-05 for the other two, which reads exactly like a broken rotated decode. The tempting explanation was the A23 canonicalization swap, since `w >= h` flips discontinuously — and measuring killed it outright: the minimum `|w - h|` over 300 rows is 4.9e-02, no row within 1e-05 of the branch point. Comparing each column as a multiset instead showed every column agreeing to float32 noise, so the exported graph had been selecting the same detections all along. The fault was the comparison: rows tied on score were ordered by raw `cx`, which clusters tightly because near-square boxes put `cx` at the anchor centre and a whole grid column shares one value, so 1.5e-05 of backend noise reordered them. Rounding the ordering keys to three decimals — well above the noise, well below any real difference — pairs the rows identically on both sides. Recorded because the shape of the mistake generalizes: **an order-sensitive comparison of a tie-bearing output will accuse the implementation of an error the harness committed.**

What the comparison is worth was then measured rather than asserted. Three deliberate corruptions of the baked-in constants — transposed anchor points, an anchor grid built for the wrong canvas, and doubled strides — are each caught on every head, nine of nine, at box deltas between 1.2e+02 and 2.0e+02 against a 1e-3 tolerance. Boxes agree to 4.6e-05, scores to 1.2e-07, class ids exactly, and the masks bit-exactly at both one and the default intra-op thread count.

The anchor grid and strides are baked into the graph as buffers rather than taken as inputs, so the exported model's only input is the image and the graph is specific to one input size. That is the normal ONNX trade, and it is pinned by asserting the graph has exactly one input, so the choice cannot drift into being an accident. `onnx` and `onnxruntime` are dev-group dependencies: the export itself is `torch.onnx.export`, already a runtime dep, so nothing a consumer installs changes.

One thing this package deliberately leaves standing. The decode composition now exists twice — once in `predict.py`'s E2E path and once in the test's graph wrappers — with nothing forcing them to agree. A shared `export.py` composing deploy and decode would remove the drift risk, and it was not built here on the smallest-change rule; it is the natural home if a later package needs an exportable path it can ship.

### WP-109 — a wheel can ship a GPL binary and declare it nowhere

<a id="wp-109"></a>

The package exists because of what sourcing WP-067's example turned up. `supervision`, the obvious drawing library for boxes, masks and rotated boxes, hard-requires `av>=14.2`. The `av` wheel declares `BSD-3-Clause`, ships a `licenses/LICENSE.txt` in which `grep -ci gpl` returns 0, and ships `av/.dylibs/libx264.165.dylib`. x264 is GPL-2.0. Running this repository's own audit functions against a throwaway install of it — never the project venv — returned `declared: ['BSD-3-Clause'] | bundled: []`: both checks clean, on a wheel carrying a GPL binary. The gate whose entire purpose is seeing this saw nothing, and the reason it saw nothing is that a vendored binary need not be documented anywhere at all. D15 was found by reading a license document the wheel happened to ship; this one had no document to read.

**The third check reads the file list, and the honest description of it is that it narrows the hole rather than closing it.** Filenames are matched against a named table of copyleft libraries — x264, x265, mp3lame, GEOS, the eight FFmpeg libraries — so a copyleft library the table has never heard of passes exactly as `av` did before the table existed. The alternative was considered and is not available: `torch` alone ships hundreds of legitimate native libraries, and a check failing on every unrecognized one would fire on every commit and be switched off within the week. What the table can honestly claim is that no *listed* library is present, and the module docstring says that rather than implying more.

Two things made the scan work at all, and both were near-misses.

**The normalizer has to know what `auditwheel` does, or the check works on macOS and goes blind on Linux.** The `libgeos_c.1.19.2.dylib` macOS ships verbatim arrives on manylinux as `libgeos_c-abcdd5fa.so.1.19.2`: the repair tool grafts eight hex characters onto the name. A table keyed on the plain stem matches every local run and nothing on the CI runner — and the failure presents as a green gate, which is the worst shape a licence check can fail in. Verified against all 20 vendored libraries in the `shapely` and `pillow` manylinux wheels, the graft is uniformly eight lowercase hex characters; it is stripped before matching, and a live test on `shapely` fails if the convention ever changes.

**The table's licence notes were the part most at risk of being written from memory.** Each entry claims a licence and where that licence was read, which is worth exactly as much as the reading actually done. So they were read: VideoLAN's x264 page states x264 is released under the GNU GPL and separately available commercially; x265's own `COPYING` is the GPL version 2 text; FFmpeg's `LICENSE.md` states most of FFmpeg is LGPL v2.1-or-later and that `--enable-gpl` changes its licence to GPL v2+, naming libx264, libx265 and libxvid together as the GPL-v2 externals that flag admits; the LAME page states LGPL and names no version, so the entry names none either. GEOS needed no page at all — `shapely`'s installed `LICENSE_GEOS` opens `License: LGPLv2.1`, which is the same exposure D15 recorded reached by a second route, and it is what makes the table's behaviour observable in this environment rather than hypothetical.

The allowlist is keyed on `(distribution, library)` rather than on the package, which is the one place this check deliberately differs from the bundled-document allowlist beside it. That file's own rule, from D15, is that a package excused for one vendored library must not be excused for every other one it ships — and a package-keyed entry here would break exactly that rule, since allowing `shapely`'s GEOS would also wave through a libx264 it started shipping tomorrow.

The tests are two-sided on purpose: the synthetic `av` must be caught, and the live environment — torch, numpy, pillow, onnxruntime, matplotlib, shapely, 340 native libraries between them — must stay clean. Either side alone is worthless. A check firing on the current venv gets disabled by the next person; a check firing on nothing is decoration. The positive case is a hand-written `.dist-info` built through `Distribution.at`, so the real `RECORD` parser runs, because `av` is the one package this tree must never install to test against.

Cost is 0.09 s to 0.24 s on the pre-commit hook, ~0.15 s of it walking 87 wheels' `RECORD` files. Every distribution in this environment has one; a distribution installed without a `RECORD` ships nothing as far as this check can see, which is stated where the reader will need it rather than left to be discovered.

`supervision` is refused on the strength of this (D16) and WP-067's example is drawn with `matplotlib`, already a dependency (R29). The finding is the more useful artifact than the example would have been.

### WP-067 — the drawing library that could not be installed

<a id="wp-067"></a>

The example was going to use `supervision`, which draws boxes, masks and rotated boxes and is Apache-2.0 at its own metadata. It is not installed here, and the reason is [WP-109](#wp-109): it hard-requires `av>=14.2`, whose wheel ships `libx264` — GPL-2.0 — while declaring `BSD-3-Clause` and shipping a licence document that mentions no GPL anywhere. Neither existing licence check saw it (D16). So the example is drawn with `matplotlib`, which the repository already carries for the training figures (R29), and the refusal is the more durable artifact of the two: the example is a hundred lines anyone could rewrite, and the check that refuses the dependency now runs on every commit.

**A `dev` dependency must not become a runtime one by being convenient.** `matplotlib` is in the `dev` group and nothing under `src/lucid_yolo/` imports it at any level, so a wheel a consumer installs pulls no plotting stack. That is why this is `scripts/draw_predictions.py` beside `plot_training.py` rather than a `lucid_yolo.draw` module, and it is also why the deliverable is a script rather than the notebook the row originally named: a notebook is not reachable by `make gate`, and an example nothing runs is an example that rots.

The drawing functions take an already-computed prediction and an image array rather than a checkpoint. That is what makes the geometry assertable at all — D14 ships no trained weights, so there is no checkpoint to draw from in a test, and a drawing function that insisted on doing its own inference could only have been tested through one.

**The oriented case is the one with a wrong answer that looks right.** An A45 row's first four columns are `[cx, cy, w, h]`, which read as an `xyxy` box without complaint: a `[:, :4]` slice draws a plausible upright rectangle that is not the object the model reported, at the wrong place and the wrong size, and nothing in the figure says so. The corners come from `rboxes_to_polygons`, the same function the oriented report and the tile writer state an object with, so a drawn ring and a written ring are the same four points. The test asserts both halves — that the four vertices match the rotated corners, *and* that the ring is not axis-aligned — because the envelope of a rotated box contains all four of its corners too and would satisfy the first assertion alone.

Two smaller decisions worth the sentence each. Colour is a pure function of the class index, not a palette walked in encounter order, so two figures of the same scene are comparable by eye and a class missing from one image does not shift the colours around it. And masks are drawn one overlay per instance rather than one composited layer, because the question a segmentation figure is read to answer is *which* instance covers a pixel, and a single layer can only answer that some instance does — boxes and masks are filtered by one row mask computed once, since two filters are how a mask ends up drawn on its neighbour with the counts still agreeing.

What the tests cannot see is whether the figure reads. Artist assertions pin geometry, colour and count; legibility — where the label sits, whether the mask alpha leaves the object visible — was checked by rendering three previews and looking at them. That is stated rather than hidden, because the gate does not cover it and the next person changing the layout should know which half of this is machine-checked.

### WP-110 — a grep for a dead command cannot find its dead grammar

<a id="wp-110"></a>

WP-096 deprecated `lucid-download` in 0.3.0 and wrote the removal date into three places: `[project.scripts]`, AGENTS.md sec. 2, and the 0.3.0 changelog. 0.4.0 is that date. The alias existed so published reproduction instructions kept running, which is a real cost to removing it — and the way that cost is paid is not by keeping the alias forever but by making sure every string this repository prints names a command the reader can still run.

**The sweep was driven by grepping the alias name, and that method has a hole the grep cannot see.** A hint that names `lucid-download` is findable; a hint that uses the alias's *flag grammar* without naming it is not. Parse-testing every CLI fragment in the module's own docstrings found one: the checksum policy spelled `--sha256 val2017.zip=<hex>`, the bare argparse form, which the shipped jsonargparse command rejects outright — `error: Parser key "sha256": Expected a <class 'collections.abc.Sequence'>`. It needed `'[val2017.zip=<hex>]'`. Nothing that greps for a name would ever have surfaced it, and it would have shipped as documentation that fails when followed.

The same shape of mistake was in the test that guarded the repair hint: it asserted the hint *string appeared* on stderr. That assertion would have passed unchanged while the command it named was being deprecated out from under it — it tests presence where the thing worth testing is runnability. It now shell-splits the printed hint and re-parses it through the shipped parser, so a hint that stops parsing fails here rather than in an operator's terminal.

Nothing turned out to be shared. `lucid-data download` reaches `download_dataset` through jsonargparse's `add_function_arguments`, never `add_arguments`, never `main`, never an `argparse.ArgumentParser` — the deleted pair was referenced from exactly two places, each other and the test file. The near-trap was that `add_arguments`'s own docstring claimed it registered "the `lucid-data download` flags", untrue since WP-096; anyone trusting the docstring over the call graph would have kept it as shared code.

Removing the alias also removed the one exemption to the project's underscore-flag rule, and AGENTS.md sec. 2 was stale on a second count anyway: `lucid-predict` shipped in WP-089 and the bullet still counted three commands. Both corrected in the same edit, since they are one sentence describing one surface.

Two references are left standing deliberately. The reproduction report's "commands as run" blocks keep their dashed spellings — they record what was executed at the version named, and editing them would falsify the record rather than update it — with the surrounding prose changed to say the alias *was* removed and that the block must be read through its replacements rather than run as written.

## 🔁 Phase 11 — Rolling, toward 0.5.0

### WP-112 — two compositions of the same decode, and the one that can't filter

<a id="wp-112"></a>

WP-066 needed one importable `nn.Module` per task so `torch.onnx.export` had something to trace, and wrote three small wrappers private to the test file to get one. `predict.py` composes the same pieces — a task's `deploy()` view plus its E2E decoder — for single-image inference, independently. WP-066's own log entry named the risk and declined to fix it: nothing forced the two compositions to keep agreeing as either side changed, and nothing shipped needed a third caller to justify the extraction. Roadmap 112 is that extraction, now that something does.

`src/lucid_yolo/export.py` reproduces `predict_image`/`predict_segmentation`/`predict_oriented`'s `"e2e"` compositions call for call — same decoder calls, same argument order, same `decode_instance_masks(prototypes, coefficients, boxes, image_size=...)` order, same `decode_rboxes` -> `o2o_rotated_topk` pairing — so the exported graph and the single-image path can be shown to agree rather than merely resemble each other. `predict.py` itself is untouched: it calls a raw `DetectionLitModule` and decodes an unbatched result with confidence filtering, a genuinely different call convention from a traced graph's fixed-shape `nn.Module`, and unifying the two call conventions was never this row's scope.

One divergence is real and stays, documented rather than engineered away. `predict_segmentation` drops padding rows (`anchor_index == PAD_ANCHOR_INDEX`) before gathering mask coefficients, since a single-image caller wants a mask per real object and a boolean filter is free to shrink the output. A traced graph's output shape is fixed at trace time, so `SegmentExportGraph` gathers every one of its `k` rows, padding included — which only stays safe because a padding row's index is a valid position in the coefficient tensor exactly when the canvas's anchor count exceeds `k`. Below that, gathering at the padding sentinel (`-1`) raises. This is the same constraint the ONNX export test's 128 px canvas (336 anchors, above the 300 cap) was already built to satisfy; `export.py`'s module docstring restates it because the module now has callers that test fixture does not.

### WP-113 — an emoji changes the anchor a renderer derives

<a id="wp-113"></a>

Decoration with one load-bearing consequence. GitHub and Python-Markdown both slugify a heading by stripping what is not a word character and joining the rest with hyphens, so `## 🧠 How a modern YOLO works` no longer answers to `#how-a-modern-yolo-works` — the stripped emoji leaves the leading separator behind. The README's own audience table links to four of its H2 sections, and every one of those links would have gone dead in the same commit that made the page prettier, with nothing failing anywhere.

The fix is not to spell the new slug. The two renderers this project publishes through do not have to agree on what a slug becomes, and one that today keeps a leading hyphen is free to trim it tomorrow — so a link written against a derived anchor is a link written against a renderer's current behaviour. Explicit `<a id=>` tags above the four headings are what the research log already uses for exactly this reason: they are greppable, they survive a retitle, and they mean the same thing to both renderers.

**Two gates read heading text and had to stop.** `test_report_sections` pinned four `## `-prefixed literals from the reproduction report and `test_decisions_carry_all_ids` pinned `## ADR-00N`; both would have failed on a purely visual edit while reporting a missing section, which is a gate that misdescribes its own finding. They now compare undecorated titles — a lead token holding no ASCII alphanumeric is decoration, anything else is title — so the assertion is about which sections exist, which is what it was always for.

The emoji themselves are chosen per section and reused across files only where the subject is the same: 🎯 for detection, 🖌️ for segmentation, 🔄 for oriented detection, in the roadmap phase, the log phase, the model card and the training recipe alike. A palette walked in file order would have been faster to produce and would carry no information.

### WP-113b — a README cannot be relative in two places at once

<a id="wp-113b"></a>

`pyproject.toml` hands `README.md` to setuptools as the long description, and PyPI resolves its relative targets against `pypi.org`: the three training-curve figures render as broken images and every `docs/…` link 404s. The same file has to stay relative in the git tree, where relative is exactly what works. So the rewrite belongs to the moment of packaging rather than to the file, and it is opt-in per invocation — `--ref v0.4.0` or `LUCID_YOLO_RELEASE_REF` — so an ordinary `make build` or a `pip install .` cannot reach it.

**Two hosts, because the two link classes fail differently.** A figure served through `github.com/.../blob` renders an entire HTML page inside an `<img>`, which is a broken image; a document served through `raw.githubusercontent.com` hands the reader unrendered markdown. Images therefore resolve through `raw`, documents through `blob`, and the classifier keys on the markdown image bang *or* the target's suffix, since an HTML `<img src=>` carries no bang.

**A branch may not be pinned.** A released wheel's README is a snapshot of one tree, and a link into `main` describes whatever that branch holds when a reader clicks it — which is how a page for 0.4.0 ends up documenting code that shipped years later. Only a `v0.MINOR.PATCH` tag or a full commit sha is accepted.

**The slug is read from `project.urls.Homepage`, and reading it exposed a live defect.** The declared homepage was `Borda/lit-YOLOs`; the repository's canonical `full_name` is `Borda/lucid-YOLO`. GitHub redirects a renamed repository's HTML URLs, so the stale slug looks fine in a browser and every `blob` link would have worked — but `raw.githubusercontent.com` does not redirect, so precisely the three figures this package exists to publish would have 404'd, while the links beside them resolved. A rename is invisible until something reads the metadata programmatically.

What the tests cannot establish is whether the produced URLs resolve. The repository is private at the time of writing, so every generated link 404s for an anonymous reader regardless of correctness; the URLs become true when the repository is public and the tag exists. That is stated rather than asserted, because a test that fetched them would be a test of the repository's visibility settings.

### WP-114 — a default list that vanishes when you name one member

<a id="wp-114"></a>

The site publishes what the repository already holds: `mkdocs.yml` owns no prose beyond `docs/index.md`, no plugin generates a page from source, and every register stays a plain markdown file a reader can open on GitHub without the toolchain. The work was configuration, and configuration is where a wrong value reports success.

**Naming one member of a MkDocs default list replaces the whole list.** Supplying `plugins:` at all drops the implicit `search`, so a site whose search box finds nothing builds green; supplying `markdown_extensions:` drops `tables`, and this repository is wall-to-wall pipe tables — the assumption register, the roadmap, every acceptance table would render as literal pipe characters on a build that reports no error. Both are restated explicitly, and `tables` is asserted in `scripts/_tests/test_audit_docs_site.py` rather than trusted to stay listed.

**`--strict` is the whole value of building locally.** It fails on a link to a page that does not exist and on a page the nav never lists. A register nobody can navigate to is one nobody reads, and neither failure raises anything without the flag.

**A second config file needed a second hook instance, for the same reason `docs/` did.** Material's mermaid fence is enabled through a `!!python/name:` tag; MkDocs parses its config with `yaml.Loader` and constructs it, pre-commit's `check-yaml` uses `safe_load` and cannot. So `check-yaml` is now two instances — the default one excluding `mkdocs.yml`, and an `--unsafe` one scoped to it, which still validates the syntax of a 120-line nav without importing anything from it. `mdformat` split the same way and for the same shape of reason: `docs/` is now rendered by a dialect the rest of the tree is not written in, and running the mkdocs plugin over the README or `AGENTS.md` would apply that dialect to files no site renders.

**The audit has to run where the exposure enters.** The docs group is deliberately outside `make setup` and outside `dev`: no gate imports it, and a contributor who never builds the site never installs the tree. The consequence is that no other workflow's environment contains it, so the licence audit — which scans the installed environment rather than the diff (D15, D16) — runs in the docs job, ahead of the build. Measured over the installed tree: 106 distributions, 343 shipped binaries, no GPL-family licence declared, bundled or shipped. One transitive is not permissive and is named rather than buried: `certifi` is MPL-2.0, weak file-level copyleft, reached through `mkdocs-material` → `requests`; MPL-2.0 was already present via `pathspec`, which arrives with mypy.

**`configure-pages` reads by default, and cannot do otherwise with the token CI has.** Its `enablement` input defaults to `false`, and its own `action.yml` states that enabling requires a token other than `GITHUB_TOKEN`. So `pages: read` is the ceiling that job can use rather than a permission it is being denied, and the step is skipped on pull requests: it fails when Pages is not enabled, which would turn every PR red for a reason no PR can fix.

### WP-114b — an unlicensed dependency is not one the audit can see

<a id="wp-114b"></a>

Building the site prints a notice from the Material team about the upstream MkDocs 2.0 release. One line of it is a licence fact rather than an opinion about the release: "Currently unlicensed – unsuitable for production use". This project admits permissive licences only, and unlicensed is not a weaker version of permissive — it is stricter than the AGPL the audit bans, because the default with no declared licence is no grant at all.

**The audit would not have caught it.** `scripts/audit_licenses.py` matches `COPYLEFT_PATTERN` — a GPL-family regex — against what each installed distribution declares, and reports only what matches. A distribution that declares nothing matches nothing and passes silently, in the same run that reports the environment clean. The control this project relies on for licence exposure is shaped to catch a forbidden declaration, not an absent one, so the bound has to be a version cap: `mkdocs>=1.6,<2`, which keeps the resolution from happening rather than detecting it afterwards.

The cap is asserted in `scripts/_tests/test_audit_docs_site.py` because it is the kind of pin a routine dependency bump lifts without anyone deciding to. The assertion is keyed on the version operator directly after the name, since `mkdocs-material` shares the prefix and sits two lines away in the same list — without that, a deleted `mkdocs` pin would move the assertion silently onto its neighbour and keep passing. Resolution after the cap: mkdocs 1.6.1, mkdocs-material 9.7.7, 29 packages. Lift it when 2.x ships a permissive `LICENSE`, verified from the repository rather than from the notice.

### WP-115 — the check that could not see what it was capping

<a id="wp-115"></a>

`scripts/audit_licenses.py` read three surfaces and each of them matched a forbidden pattern against a declaration, which meant a distribution declaring nothing matched nothing and passed — in the same run that printed "license audit clean". WP-114b hit that wall directly: the exposure it wanted to prevent could only be bounded with a version cap, because the audit had no way to report it. The fourth surface is the absence of a licence.

**Reading a licence with no field to read it from.** One installed distribution declares nothing: `faster-coco-eval` 1.7.2, PEP 639 metadata with no `License`, no `License-Expression`, no `License ::` classifier, shipping `licenses/LICENSE` holding the Apache-2.0 text. So the naive rule — no field, no pass — would have failed on a genuinely Apache-2.0 base dependency on its first run, which is how a gate gets turned off. `bundled_license_indicators` was no help by design: it reads `License: ...` declaration lines rather than prose, because a bundled LGPL names GPL on dozens of its own lines, and the Apache text carries no such line at all. The recognizer added here identifies a document from a distinctive phrase in its own first 15 lines, and identification *is* the verdict.

**The copyleft pattern is deliberately not run over a recognized text as a second opinion.** MPL-2.0's own Secondary-License clause names the GNU General Public License, the GNU Lesser General Public License and the GNU Affero General Public License in consecutive lines. A belt-and-braces re-scan would therefore fail the audit on a licence this environment already carries through `certifi` and `pathspec`. The table lists permissive texts only, so the recognizer never has to decide that something is forbidden — an omission costs a false alarm, never a silent pass. The escalation runs the other way: a document that was *not* recognized and whose header names a GPL-family licence is a failure in any tier, because "no AGPL at any cost" does not care which group reached the package.

**Two tiers, and only the fourth check is graded.** A package in the `[project.dependencies]` closure is republished in this project's own wheel metadata and installed by everyone; an unreadable licence there fails. A package reachable only through `dev` or `docs` appears in no wheel metadata — PEP 735 groups are not published at all — is imported by `src/` never, and is vendored into no artifact; it is printed as a flag and the run passes. Both-tier packages take the stricter attribution, and so do packages the walk cannot attribute, so a hole in the resolver presents as a loud failure rather than as a quiet demotion.

**Attributing the tier was the larger half.** The audit walks a flat installed environment; the tiers are a property of `pyproject.toml`, so the closure has to be resolved the way an installer would. The first attempt dropped every requirement carrying an `extra` marker and left 19 of 105 distributions unattributed. Carrying the requested extras through the walk and evaluating each marker against them closes that: the residue is one distribution, `lucid-yolo` itself, which no dependency list mentions because it *is* the project. Final attribution: 56 base, 29 dev, 19 docs. `packaging` became a declared `dev` dependency in the process — it was already there transitively through both pytest and mkdocs, and a direct import is not a transitive dependency (R34).

The audit still prints one line on a clean run, and the flag lines only when there are any.

### WP-115b — the first thing the new check found was real

<a id="wp-115b"></a>

WP-115 shipped a check for the absence of a licence and it fired within a day, on a CUDA environment this laptop is not: `cuda-toolkit`, reported as a base-tier failure. The first question was whether the check was wrong. It is not. The wheel was downloaded and opened: `cuda_toolkit-13.3.1.dist-info/` contains `METADATA`, `WHEEL` and `RECORD`, and nothing else. No `License`, no `License-Expression`, no `License ::` classifier, no licence document declared or undeclared. Its every requirement is extras-gated, so a bare install pulls nothing at all. There is genuinely no statement of terms in that distribution to read.

**So the resolution is an allowlist, not a weaker check.** `UNREADABLE_ALLOWLIST` is the fourth allowlist in this file and is kept separate from the three beside it for the reason D16 already gave about the third: the exposures differ, and folding them together lets one package's silence excuse another package's declaration. `ALLOWLIST` excuses a package's own copyleft declaration, `BUNDLED_ALLOWLIST` and `BUNDLED_BINARY_ALLOWLIST` excuse a vendored copyleft library, and this one excuses metadata that says nothing. Entries require a DECISIONS.md row first, as all three others do; `cuda-toolkit` is the only one.

**CUDA is admitted as an environment, not as a licence** (D17). The NVIDIA CUDA EULA is proprietary, which the permissive-only policy would otherwise refuse outright. What makes it out of scope is that `pyproject.toml` declares no CUDA package, this project redistributes no part of the toolkit, and it publishes no trained weights (D14) -- so what the EULA covers is the machine an accelerator run happens on rather than anything this repository ships. That is written down rather than left as a silence, because an exemption nobody recorded is indistinguishable from an oversight.

**A second, smaller thing the finding exposed.** The report printed `(base)` for two different situations: a package the shipped closure actually reaches, and a package the walk never reached at all, which defaults to the strict tier by design. Both are failures, for opposite reasons, and spelling them identically sends a reader looking for a dependency declaration that does not exist -- which is exactly the wrong first move when the real answer is "nothing requires this; it was installed by hand". The unattributed case now says so in its own reason.

**And the suite stopped reading the installed environment.** Seven tests and nine doctests asserted over `metadata.distributions()` -- the whole venv is clean, `shapely` really does ship GEOS, `torch` attributes to base -- and none of those is an assertion about this code. A contributor who installs anything can fail them without touching a line of the audit, which is exactly what `cuda-toolkit` did. Had the allowlist landed on its own, `test_the_live_environment_has_no_unreadable_license` would have gone green and the next package installed on that box would have turned it red again. The live claim belongs to the pre-commit hook, which runs the audit against the real environment on every commit and is the actual gate; a test's job is that the audit *decides* correctly, and synthetic distributions answer that without depending on what anyone happens to have installed. The shapes those fixtures are written to were read off real wheels once and recorded in their docstrings -- the `cuda_toolkit` dist-info listing above is one of them.

Two things that went with it are worth naming. The GEOS true-positive test is gone; what stands in for it is the synthetic manylinux spelling (`libgeos_c-abcdd5fa.so.1.19.2` through `library_stem`) plus the parametrized normalizer cases, so the regression it guarded -- the stem normalizer going blind on Linux, where CI actually runs -- is still caught. And `audit = _load_audit()` at module level became a session-scoped fixture: loading a non-package script by path is fine, doing it as an import side effect at collection time is not.

### WP-116 — a formatter that reads `Eq.` as the end of a sentence

<a id="wp-116"></a>

docformatter joins the three formatters already on the commit hook, configured in `[tool.docformatter]` and ordered ahead of ruff-format so the docstring body is rewritten first and the quotes and indentation around it normalized second -- the reverse order converges too, but only on the commit after.

**Wrapping is off in both axes, and that is the load-bearing setting rather than a timid one.** `wrap-descriptions = 0` because this project's docstrings carry argued paragraphs, `**bold**` lead-ins, and Google `Args:`/`Returns:` blocks whose indentation is what Napoleon parses; re-flowing them to a column would run those blocks together and rewrite hand-chosen line breaks across the whole tree in a single commit nobody could review. `wrap-summaries = 0` for a smaller reason: a summary is one sentence by construction and has nothing to gain from being re-flowed.

**What it cost once was 32 summary lines.** docformatter decides where the summary ends by finding the first period it reads as a sentence end, and it reads every abbreviation that way. This repository writes the papers' own notation -- `R1 Eq. 15`, `blueprint sec. 5.9`, `Redmon et al.` -- in summary lines, so those summaries were cut mid-citation and everything after them re-indented as continuation text, which is what turns an `Args:` block into prose Napoleon cannot parse:

```
"""Initialize the two coarse-level projections required by Eq.

8.
        Args:
            in_channels: Per-level neck channel counts ``(N3, N4, N5)`` in
```

No option prevents it. `--wrap-summaries 0` disables wrapping, not the split; `--non-strict` governs reST list detection; `--docstring-length` would only choose which docstrings get mangled. So the source is what changed: `Eq.` and `sec.` are spelled out and `et al.` rephrased, **in summary lines only** -- the bodies keep the papers' abbreviations, since nothing reads them for sentence boundaries. One summary genuinely held two sentences and was split into a summary and a body paragraph, which is what PEP 257 asked for anyway.

**`non-cap` names 17 identifiers.** The formatter capitalizes a summary's first word, which is right for English and wrong for a name: `lucid-yolo:` became `Lucid-yolo:`, `mAP` became `MAP`, and `empty()`, `backward()`, `rboxes[i]`, `o2o` and the nine subpackage names were all renamed to something that does not exist. Each is listed rather than the check being disabled, so a future summary opening with an ordinary English word is still capitalized.

**The hook now reformats nothing, and that is the honest result.** Every line in this commit's diff is the one-time rephrase; docformatter's own second pass is empty, because these docstrings were already PEP 257-clean. Its value is prospective -- it is a gate on what gets written next, not a repair of what is there.

### WP-116b — a hook pinned to no interpreter in particular

<a id="wp-116b"></a>

WP-116's `docformatter` hook carried no `language_version`, so pre-commit built its isolated environment against whatever `python3` resolved first on `PATH` at hook-creation time. That happened to be a 3.11 install then; a later `python3.10` framework install on this machine moved ahead of it in `PATH`, and the next `pre-commit run` silently rebuilt the hook's environment against 3.10 -- a version below this project's own `requires-python = ">=3.11"` floor.

docformatter's own config reader needs `tomllib`, stdlib only since 3.11, to parse `[tool.docformatter]` out of `pyproject.toml`. Under 3.10 that import is simply absent, and the hook failed with `NameError: name 'tomllib' is not defined` rather than anything naming the real cause. Every other hook in the file that needs a specific interpreter says so explicitly -- `mypy` and `license-audit` both pin `entry: .venv/bin/python`. `docformatter` is a third-party hook rather than a local one, so it takes `language_version: python3.11` instead: the same guarantee, expressed the way pre-commit resolves environments for repos it clones rather than ones this project owns.

This is the same fault as `cross-venv` above -- an environment silently reconstructed between two commands -- reached through a different door: there it was `uv run` rebuilding `.venv`, here it is `pre-commit` rebuilding one hook's env off unpinned `PATH` resolution. Neither the tree nor the hook's own config changed; only what interpreter answered to `python3` did.

### WP-117 — documented and verified are not the same claim

<a id="wp-117"></a>

WP-085 wired `--doctest-modules` over `src`, `scripts` and `tests`, which turns a helper's `Examples:` block into an executable check the moment that block exists -- but nothing forced it to exist. A fresh `ast` count over `tests/**/test_*.py` found 261 non-fixture, non-`test_` helper functions, every one already carrying a docstring, and only 1 of the 261 carrying a `>>>` line `--doctest-modules` could actually run. Prose describing behaviour a reader had to trust, not a line pytest ever executed. 23 of 678 `test_` functions carried no docstring at all, all 23 in one file, `tests/data/test_download.py`.

Closing the gap by hand, file by file, surfaced real bugs the doctests would otherwise have hidden behind a passing suite: `_write_split`'s and `build.convert_split`'s return values auto-printed inside a `with` block under doctest's "single" exec mode, needing `_ = ...` to suppress; `AssignResult` has no `.labels` attribute, only `.target_labels`; `Trainer` has no `.deterministic` attribute; `_decode_boxes`'s hand-derived expected output was numerically wrong until computed from a real run rather than worked out on paper. Each was a doctest that failed on first execution -- exactly the outcome WP-085's infrastructure exists to produce, and exactly what a prose-only docstring would never have surfaced.

A second, unrelated inconsistency turned up mid-pass: some docstrings wrote a bare `>>>` block, others wrapped it in a proper `Examples:` header matching the Napoleon style `predict.py` and `export.py` already use. An `ast`-based script fixed 166 docstrings across 56 files in one pass -- locate each docstring's span, detect a `>>>` line with no preceding `Examples:`, insert the header and re-indent -- verified against a doctest re-run before and after rather than assumed safe.

The "8 groups across 6 files" class-regrouping half of the original scope did not survive contact with measurement: a first-shared-word `ast` scan returns ~90 candidate groups, because this suite's own descriptive-sentence naming house style is indistinguishable from genuinely flat enumeration by any heuristic tried. Split off into WP-128 rather than guessed at.

### WP-118 — one log grew two audiences

<a id="wp-118"></a>

`RESEARCH_LOG.md` had accumulated 85 entries answering two different questions -- what a reproduction claim cost to establish, and what the repo's own tooling cost to build -- under one charter that named only the first. The immediate trigger was smaller: rows 115, 115b and 116 had each grown to a paragraph of debugging narrative in the roadmap's own Scope column, because the log existed but the habit of writing to it first did not.

The split rule: does an entry record what the reproduction *claims* (model, loss, data protocol, eval numbers, paper ambiguity), or *how the repo builds, checks, and publishes itself* (CI, hooks, packaging, licence, formatting, CLI, docs tooling)? Most of the 85 sorted cleanly -- the whole of `Cross-cutting` and Phase 0 turned out to be tooling-kind, the whole of Phase 7 and most of Phase 8/9 stayed fidelity-kind. Five did not sort cleanly, because their finding genuinely was both: WP-066 (export tooling *and* an architecture-fidelity confirmation), WP-079 (a DataLoader bug *and* a measured cost to augmentation diversity), WP-083 (a golden-harness design *and* the frozen numbers it pins), WP-091b (a decoder implementation *and* an unsourced-threshold provenance investigation), WP-097 (a download-verify bug *and* what R18's published totals actually describe). Those five were not forced into one bucket; each is cut into two pieces, cross-linked both ways, sized to what each piece actually carries rather than split down the middle.

The mechanical risk in a retroactive split of this size is silent loss -- a paragraph dropped in the move, an anchor a roadmap row still cites that no longer resolves. The move itself was scripted rather than hand-edited per entry, asserting the parsed anchor set equalled the classification set before writing anything, so nothing could be silently skipped or double-counted; `scripts/_tests/test_audit_docs_present.py::test_every_log_link_resolves` (renamed from its single-file predecessor) now checks both files' anchors against the roadmap's citations of either.

### WP-119 — a symlink kept for a command nothing still runs literally

<a id="wp-119"></a>

WP-038 moved the experiment configs into `src/lucid_yolo/configs` as package data, so a wheel-only install could run `--config`, and kept a repo-root `configs -> src/lucid_yolo/configs` symlink so the commands already documented at the time -- ones that typed `configs/det_smoke.yaml` from repo root -- kept working. That was the right call then: a compatibility shim is cheaper than rewriting every example the moment a layout changes underneath them.

What made it safe to drop now is that nothing in the tree still needed it. Every path that resolves the configs directory in code does so package-relatively -- `Path(lucid_yolo.__file__).resolve().parent / "configs"` in `tests/ptl/test_cli.py` and `scripts/overfit_micro.py`, `Path(__file__).resolve().parents[1] / "configs"` in `packaged_config()` -- none of which touch the repo root at all. And `_resolve_config_args` (also WP-038) already does the harder version of what the symlink did: a `--config` value that names a packaged config by bare filename, with or without `.yaml`, resolves onto the installed `lucid_yolo/configs` tree when no such file exists locally, so `lucid-yolo fit --config det_smoke.yaml` already works identically from a checkout or a wheel install, no path and no symlink required.

The one place the symlink was load-bearing rather than redundant was prose: `train.py`'s own module docstring showed `python -m lucid_yolo.cli.train fit --config configs/det_smoke.yaml`, a literal root-relative path that only resolved because the symlink put a real directory at `configs/`. That is now the bare-name form the resolver was actually built for. `AGENTS.md`'s `configs/data/*.yaml` and two comments in `tests/ptl/test_cli.py` and `scripts/overfit_micro.py` that named the bare `configs/` directory are reworded to `src/lucid_yolo/configs/...` or `lucid_yolo/configs/...` for the same reason -- accurate once nothing at repo root answers to that name. Historical entries elsewhere (`ENGINEERING_LOG.md`'s own WP-099b entry, completed roadmap rows for WP-038 and WP-099c) describe what was true when they were written and are left alone; a backfilled record does not get edited to match a later decision.

### WP-127 — a version number is not a run's identity

<a id="wp-127"></a>

Lightning's own default checkpoint path is `lightning_logs/version_N/checkpoints/epoch=X-step=Y.ckpt` -- `version_N` numbers the *run*, sequentially, across every task and scale a working tree has ever fit; nothing in the path names what the run actually trained. WP-111's re-scoring of WP-064's OBB-smoke checkpoint hit this directly: `version_10` said nothing until cross-referenced against its `hparams.yaml`'s `task: obb` field, a step that only works because the checkpoint happened to still carry its hyperparameters.

The fix stays inside `DetectionCLI.instantiate_trainer` (`cli/train.py`), the same injection seam WP-038 already uses for the progress bar and the default logger pair: unless a config places its own `ModelCheckpoint` in `trainer.callbacks`, one is now injected with `filename=_checkpoint_filename(self.model.task, variant)` -- a new pure helper prefixing `{task}_{variant}_` onto Lightning's own `{epoch}-{step}` template, left as a literal placeholder so `ModelCheckpoint`'s own metric-filling machinery still resolves it. `dirpath` is left at Lightning's default (`<version dir>/checkpoints`), so `lightning_logs/version_N` numbering is untouched and no already-written checkpoint moves or is renamed -- only new runs pick up the prefix. `task` comes from `self.model.task` (the instantiated module, already built by the time `instantiate_trainer` runs) rather than the raw config, since only the module's own default (`task="detect"`) is guaranteed correct once link-computed arguments are in play.

The injection duplicated the progress-bar callback's four-line "append without replacing the config's own list" pattern verbatim, so it moved into a small `_add_trainer_default_callback` method both call into, rather than being copy-pasted a second time.

### WP-128 — a shared prefix is not a shared caller

<a id="wp-128"></a>

WP-117's own count -- "8 groups across 6 files" -- was a one-off human read, not a reproducible measurement, and the row it left behind said so: a first-shared-word `ast` scan over `test_` names returns roughly 90 candidate groups, because this suite's house style already writes full descriptive sentences (`test_the_boundary_sits_at_the_midpoint_of_the_overlap_band`) that share a topic noun with unrelated tests -- indistinguishable from genuinely flat, mechanical enumeration (`test_<subject>_<case>`) by prefix-matching alone.

The discriminator that breaks the tie: a candidate prefix confirms only if it is a contiguous token subsequence -- prefix, suffix, or infix -- of some `ast.Call` function-name target actually present in the same file (`Attribute.attr` or `Name.id`, underscore-split, leading-underscore tokens dropped). A house-style sentence sharing a topic noun calls nothing that looks like it and never confirms; `test_check_data_a/b/c` calling `check_data(...)` does. Confirmed groups need 3+ members to enforce, matching the size floor `tests/eval/test_tile_merge.py`'s own precedent already implied by example. Verified both directions against real cases in this tree: correctly includes `check_data`, `plan_archives`, `verify_split`, `configure_optimizers`, `warmup_decay`, `on_after_batch_transfer`, `detections_to_predictions`, and `rotated_topk` (confirmed as an infix of `o2o_rotated_topk`); correctly excludes `deployed_view`-style near-misses and the `a_named_worker_count_*` / `an_auto_chosen_*` house-style family WP-117's row already flagged as the hard case.

New `scripts/lint/audit_flat_test_groups.py` owns this rather than extending `audit_test_doctests.py` (129) -- that script's own docstring disclaims scanning for anything but missing doctest Examples, and stapling an unrelated structural check onto it would have made neither job legible from the file alone. Same shape as every other `scripts/lint/` checker: `main(argv) -> int` CLI, a `flat-test-group-audit` local hook gated on `files: ^(tests/.*/test_[^/]+\.py|scripts/_tests/test_[^/]+\.py)$` (pure repo content, not installed-environment state, so no `always_run`), and `scripts/_tests/test_audit_flat_test_groups.py` as the functional core -- synthetic `tmp_path` fixtures per confirmed/rejected case, plus the two "live tree is currently clean" sanity tests the hook itself depends on staying true.

Nine files regrouped under `class Test<Subject>:` once the discriminator was run against the whole `tests/` and `scripts/_tests/` tree rather than continuing WP-117's by-eye survey: `test_verify.py`, `test_download.py`, `test_coco_eval.py`, `test_schedule.py`, `test_dota_parse.py`, `test_coco.py`, `test_obb_head.py`, `test_tiling.py`, `test_proto.py`, `test_module.py`, and `scripts/_tests/test_audit_docs_present.py` (all 12 of its check-function pairs, for whole-file consistency, though the discriminator's crude grouping key only flagged 6) and `test_audit_docs_site.py`. Regrouping moves node ids, not test bodies -- Python resolves a module-level helper name at call time, not at class-body-definition time, so relocating a scattered function into a contiguous class needed no change to whatever module-level fixtures it called. Every citation of a moved node id in `docs/ROADMAP.md` (rows 044, 051, 110) and `docs/ASSUMPTIONS.md` (A45) was checked against a grep for the old id and updated; row 044's citation had already drifted stale before this WP touched the file (`test_oracle` moved under a pre-existing `TestOracleRoundTrip` by earlier, unrelated work) and was corrected in the same pass since the row was already open.

### WP-129 — a check that scans the repo is not a unit test of it

<a id="wp-129"></a>

WP-117 landed its enforcement as `tests/meta/test_test_suite_quality.py`: a pytest test whose body is an `ast` walk over `tests/**/test_*.py`, asserting no helper lacks a doctest Example. Functionally that is a lint check -- it scans repo content for a house-style violation, the same shape as `audit_licenses.py` scanning the installed environment for a licence violation -- wearing a pytest test's clothes because that was the fastest place to put it at the time.

The fix is the split this project already draws for `audit_licenses.py` and `check_commit_trailers.py`: the check itself is a `scripts/` script with a `main(argv)` CLI entry point, invoked by a `.pre-commit-config.yaml` local hook; pytest keeps a meta-test exercising the script's own functions (`module_level_helpers`, `has_doctest_example`, `find_missing`) against synthetic `tmp_path` fixtures, the way `test_commit_trailers.py` exercises `validate_message` against synthetic commit messages rather than real repository history. Pytest stays scoped to the functional core -- does this function do what it claims, on cases chosen to exercise it -- and the hook owns the enforcement gate against the live tree.

Unlike `license-audit`, the new `test-doctest-audit` hook is not `always_run: true`: it scans pure repo content (`tests/**/test_*.py`), not installed-environment state that can drift without a tracked-file change, so it is gated on `files: ^tests/.*/test_.*\.py$` and only runs when a test file is actually part of the commit.

Moving the script surfaced one more inconsistency worth fixing in the same pass rather than leaving for a future WP to trip over: `scripts/` had grown to eleven files at its root with no separation between the two things a hook-invoked checker and a golden-regression producer actually are. `audit_licenses.py` and `check_commit_trailers.py` -- already hook-invoked -- move into a new `scripts/lint/` alongside the new script; `audit_licenses.py`'s own `PYPROJECT` path, resolved via `Path(__file__).resolve().parents[N]`, needed its `N` bumped for the extra directory level, caught immediately by the pre-commit hook itself failing on the next run rather than silently reading the wrong file. `check_goldens.py`, `golden_producers.py`, `shapes_regression.py` and the rest of the golden/training/release tooling stay at `scripts/`'s root: their dotted `scripts.<module>:<function>` paths are written into checked-in `goldens/**/*.json` `producer` fields as data, not just import statements, and rewriting those paths is a fixture migration across files that took minutes-each GPU runs to produce -- a materially different, materially riskier change than moving two files whose only callers are a Makefile line, a CI step and their own meta-tests.

______________________________________________________________________

### WP-130 — a test of a script belongs beside the script

<a id="wp-130"></a>

WP-129 split the doctest-audit check itself out of `tests/meta/` but left its test where the whole family had always lived, `tests/meta/test_audit_test_doctests.py` -- and left the other eight `tests/meta/` files, plus `tests/scripts/`, `tests/train/test_shapes_regression.py` and `tests/integration/test_overfit_micro.py`, exactly where they were: every one of them a test whose subject is a `scripts/` module, sitting in `tests/` anyway because that is where tests conventionally go. The convention was wrong for this shape: `src/lucid_yolo/` and `tests/unit/` mirror each other by design, and `scripts/` had grown its own testable surface without growing its own mirror. Fixed by moving all fifteen files into a new `scripts/_tests/`, each renamed `test_<script>.py` after the module it exercises -- `test_license_audit.py` becomes `test_audit_licenses.py`, `test_golden_harness.py` becomes `test_check_goldens.py` -- so the pairing reads directly off the two filenames rather than needing the docstring to say which script a test covers.

The other five `tests/meta/` checkers still inline (docs presence, docs site, figure captions, license headers, version single source) convert the same way WP-129 converted the doctest audit: a `scripts/lint/<name>.py` script plus a `.pre-commit-config.yaml` hook, `scripts/_tests/test_<name>.py` left as the functional-core test. `release_guard.py` and `check_goldens.py` were already scripts with no hook -- both gain one, `stages: [manual]` like nothing else in the file, since a release tag and a golden recompute are not per-commit checks. `release_guard.py` needed one real change to be hook-shaped at all: `--tag` was `required=True`, unrunnable from a hook that fires with no arguments on every commit. It now defaults to `git describe --tags --exact-match HEAD` and no-ops when `HEAD` isn't exactly a tag -- true for nearly every commit, which is the point.

Every hook's `name:` field gained a distinct leading emoji chosen for the action it performs (🧠 mypy, ⚖️ license audit, 📋 docs presence, 🗺️ docs site, 🖼️ figure captions, 📄 license headers, 🔢 version single source, 📝 doctest audit, 🏷️ commit trailers, 🥇 golden check, 🛡️ release guard) -- eleven hooks read a lot faster on `pre-commit run --all-files`'s output when each line starts with a different glyph than when all eleven start with the same bullet.

The move itself broke two things that had never been exercised before, both structural rather than content bugs. **mypy**: `[tool.mypy]` scans `files = ["src", "scripts"]` with no `mypy_path` entry for the repo root, so a file inside `scripts/` got two identities at once -- `absolutize_readme` from the direct directory walk, `scripts.absolutize_readme` from `scripts/_tests/test_absolutize_readme.py`'s own `from scripts.absolutize_readme import ...`, now itself swept into the `scripts` scan for the first time. `explicit_package_bases = true` plus `mypy_path = "src:."` gives both paths the same resolution; `scripts/_tests/` itself is excluded from strict checking afterward, matching `tests/`'s own long-standing exemption for exactly the same fixture-heavy, `monkeypatch`-typed, dynamically-imported character. **pytest**: `scripts/_tests/conftest.py` and `tests/conftest.py` are both non-package `conftest.py` files with the bare module name `conftest`; under the default prepend import mode, loading them from two different top-level collection roots (`scripts` and `tests` in the same `pytest ... src scripts tests` invocation) rather than nested under a shared one raised `import file mismatch`. `--import-mode=importlib` fixes it structurally, keying each module's identity on its resolved path rather than its stem, and is now the addopts default for every invocation, not just this one.

`scripts/README.md` is new: a table per subtree (core, `lint/`, `_tests/`) naming every script, what it does, what invokes it, and which test covers it -- the crossroad a developer new to `scripts/` reads first, and the thing that makes "no dedicated test, doctests only" for `golden_producers.py`, `plot_training.py`, and `dump_debug_grid.py` a documented decision instead of a silent gap.

______________________________________________________________________

### WP-130b — an emoji requirement is not the same claim as "every local hook"

<a id="wp-130b"></a>

WP-130's emoji pass read "every hook" as "every `repo: local` hook" and stopped at eleven -- the other eleven, all third-party (`pre-commit-hooks`, `mdformat`, `docformatter`, `ruff-pre-commit`), kept their plain default or custom names. Fixed by giving each of those eleven a `name:` override where none existed (`end-of-file-fixer`, `check-yaml` x1, `check-toml`, `check-added-large-files`, `ruff-check`, `ruff-format`) or an emoji prefix on the existing custom one (`check-yaml`'s unsafe-tags instance, `mdformat` x2, `docformatter`). Twenty-two hooks, twenty-two distinct emoji, chosen per action rather than per category so the same glyph never covers two conceptually different checks.

Separately: `pre-commit run --all-files` was only ever exercising nineteen of the twenty-two hooks, by design at the time -- `commit-trailers` runs at `commit-msg` stage, `golden-check`/`release-guard` at `stages: [manual]`, none of which `--all-files` (a `pre-commit`-stage run) reaches. Asked to close that gap too. `golden-check` and `release-guard` are cheap enough off a release tag (one `git describe` call, one already-fast golden recompute) that `stages: [pre-commit, manual]` costs nothing meaningful on every commit -- the accepted trade is `make gate` now calls `check_goldens.py` twice, once inside `precommit`'s `--all-files` and once via the `golden` target, rather than carrying a second exemption to remember. `commit-trailers` cannot take the same fix: a `pre-commit`-stage run has no commit message to validate, since no commit is in progress. Its `--all-files`-reachable half is a new sibling hook, `commit-trailers-history`, wired to the same script's pre-existing `--range` mode (`--range origin/main..HEAD`) rather than new code -- it re-validates every commit not yet pushed, which is a strictly more useful check than re-reading the newest message alone, and was already there waiting to be pointed at from a hook.

______________________________________________________________________

### WP-120 — a mirror does not know anatomy

<a id="wp-120"></a>

Phase 12 opens on a container change rather than a model change: `Targets` gains `keypoints` (`(N, K, 2)` float32) and `keypoint_vis` (`(N, K)` int64), and `HorizontalFlip` -- the one existing transform this WP's scope covers -- gains the matching mirror. Every other geometric transform (`RandomAffine`, `MosaicAssembly`, `Mixup`/`CopyPaste`, `Letterbox`) is untouched; keypoint support there is a later WP's concern, not this one's.

K is a constructor argument the way the class count already is: `keypoints`/`keypoint_vis` follow `polygons`' existing "count is 0 or N" convention (the shared instance axis every box-aligned modality already uses) rather than `rboxes`' independent-axis-plus-separate-mask one, since a keypoint set belongs to one box the way a polygon ring does, not to a second geometry entirely. `concat` additionally rejects differing `K` across items that do carry keypoints -- a check `_concat_polygons` has no analogue for, since ring length is per-instance there and K is fixed per dataset here, so two items disagreeing on K is corruption, not a valid ragged shape.

The harder decision is what `HorizontalFlip` does with left/right identity. A mirror reflects every point's x-coordinate unconditionally -- that part is geometry, not anatomy, and needed no new assumption. But swapping *which* index is now "left knee" after the flip is anatomical, dataset-specific knowledge the transform has no business hardcoding into a K-generic container's own mirror: nothing in this repository's K-point design says K=17 or names a COCO ordering, and baking the pairing into `HorizontalFlip` would silently narrow "generic K-point task" back down to "COCO human pose" the first time this class ran. `keypoint_flip_pairs: list[tuple[int, int]] | None = None` is the resolution -- a caller-supplied swap map, `None` meaning "mirror coordinates, keep column identity", the correct default for a K-point task with no left/right symmetry. A64 records COCO's own 17-point pairing as the value the demo dataset will actually supply, but the row is marked `open`: WP-121's COCO datamodule is what passes the concrete list in, not this one.

Implementation delegated to `codex:codex-rescue --write` against a design spec fixing the exact field names, shapes, validation rules, and test scenarios up front, rather than leaving the K-point/anatomy split to be discovered mid-implementation -- reviewed diff-by-diff against that spec before landing; one docstring grammar slip (`"A supplied, a dataset keypoint pair map..."`) was the only correction needed.

______________________________________________________________________

### WP-121 — a reader is not a wiring

<a id="wp-121"></a>

`CocoDetectionDataset` gains `keypoints: bool = False`, matching `oriented`'s existing shape exactly: opt-in, defaulting to "leaves the reader exactly as it was". When set, each retained annotation's flat COCO `keypoints` field (`[x, y, v, ...]`, WP-120's `Targets.keypoints`/`keypoint_vis` on the receiving end) is split by a new `_parse_keypoints` helper into `(K, 2)` coordinates and `(K,)` visibility, stacked across an image's retained instances, and threaded into whichever of the axis-aligned or `_oriented_targets` construction paths `_build_targets` takes -- both grew optional `keypoints`/`keypoint_vis` parameters rather than staying axis-aligned-only, so a caller who sets both `oriented=True` and `keypoints=True` does not crash even though this repository's real pose data is never oriented. Crowd and RLE-skipped annotations are excluded from keypoint parsing exactly as they already are from boxes/labels/polygons -- the same retained-annotation loop populates all four, so the axes cannot drift apart. A mismatched `K` across two instances of one image raises naming the file and both observed counts, the same "cannot describe a rectangular tensor" failure `_oriented_targets` already raises for a non-quadrilateral ring.

WP-120's own log entry above states "WP-121's COCO datamodule is what passes the concrete [flip-pair] list in" -- that was wrong when written, not merely superseded: this row's DoD names only `tests/data/test_coco.py`, and the roadmap row's Scope never promised datamodule or `HorizontalFlip` wiring. Left standing in WP-120's own entry per this log's convention of not editing a historical paragraph to match a later correction, but A64 in `docs/ASSUMPTIONS.md` is corrected in this WP's own commit, since the assumption row itself (not a narrative entry) still needs to be accurate going forward: it now states that no roadmap row currently owns constructing `HorizontalFlip(keypoint_flip_pairs=...)` for a pose task, rather than pointing at a WP whose actual scope never included it.

Delegated to `codex:codex-rescue --write` against a fixed design spec, as WP-120 was. One environment-only test failure surfaced during codex's own verification pass (`test_worker_loader_augments_differently_each_epoch`, a pre-existing, unrelated DataLoader-worker test failing because codex's sandbox blocks `torch_shm_manager`'s multiprocessing calls) -- root-caused correctly by codex itself (failure occurs during worker process startup, before any WP-121 code runs; the same 62-test file otherwise passed) and left for the lead's full-access environment rather than patched around. Confirmed independently: the same test file passes 63/63 outside the sandbox, and the full `make gate` (2282 passed, 27/27 goldens) is green.

______________________________________________________________________

### WP-122 — a stem is architecture, its sigma is not

<a id="wp-122"></a>

`DualDetectionHead` gains a third opt-in stem set, `num_keypoints`, following `num_coeffs`'/`predict_angle`'s exact shape: absent by default, so the accepted detection/segmentation/oriented module trees are untouched byte-for-byte. `_build_keypoint_stem` reuses `_stem_width` (A28's `channels // 3`) rather than a bespoke width the way `_build_coeff_stem` already does -- unlike the angle stem, no R1 Table S11 or any allowlisted source measures a keypoint task at all, R14 being registered "future pose milestone only", so there is nothing to fit a dedicated width against. The final 1x1 emits `4 * K` raw channels in point-major order (`x, y, sigma_x, sigma_y` per point) and nothing follows it -- coordinates are unbounded offsets exactly like the angle branch, and per the scope boundary below, sigma is equally raw.

The scope line that mattered most here was what this WP explicitly does **not** decide: R14's RLE loss needs `sigma > 0` for its flow's density, but the specific function that guarantees that (softplus, exp, a scaled sigmoid) is tied to the loss equations themselves, and AGENTS.md sec. 6 reserves loss/assignment/optimizer/rotated-geometry modules for hand-written implementation from the equations, not generated code. A65 records the split explicitly: `DualDetectionHead` and the new `decode_keypoints` (`heads/keypoint.py`, `decode_ltrb`'s anchor-centre/stride convention generalized from a four-value ltrb offset to a direct two-value per-point offset) both leave sigma completely untouched, and WP-123 is where the mapping is decided, from R14 directly. The codex spec for this WP said so in as many words -- "if you find yourself reaching for such a transform, stop" -- and the delegated diff held the line: `heads/keypoint.py`'s decode function does not even accept a sigma argument, and the test suite's `test_sigma_output_remains_raw_unbounded_and_point_major` drives the stem's output convolution to explicit large-magnitude values and asserts they survive unchanged, which would fail against any accidentally-added activation.

Delegated to `codex:codex-rescue --write` as WP-120/121 were, but this one surfaced a real regression its own sandboxed test run correctly caught and correctly declined to fix: `_DetectionBranch.forward`'s return tuple grew from four elements to six (`cls, box, coeff, angle` to `..., keypoints, keypoint_sigma`), and three call sites in `models/build.py` (`Detector`, `OrientedDetector`, `Segmenter`) unpack that tuple positionally rather than through `DualHeadOutput`'s named fields -- `cls, box, _, _ = self.o2o(...)` and its two siblings. The spec had excluded `build.py` from scope entirely, which was the actual mistake: those three lines are a direct, unavoidable consequence of widening a tuple those call sites already depend on, not a new feature crossing a WP boundary. Codex's own gate ran `tests/models/` as a regression check, found 22 failures and 6 errors all tracing to those three lines, correctly refused to touch the excluded file, and reported the gap rather than guessing. Fixed by hand in three one-line edits (widening each unpack to six positions, matching the new tuple width) -- mechanical enough, and small enough, not to warrant a second delegation round; `make gate` (2290 passed, 27/27 goldens, including all four `params_flops_*` goldens the widened tuple had been silently breaking) confirms nothing else depended on the old width.

______________________________________________________________________

### WP-123 — a loss written by hand, from equations fetched three times

<a id="wp-123"></a>

Not delegated. AGENTS.md sec. 6 reserves loss/assignment/optimizer/rotated-geometry modules for hand-written implementation from the equations -- the highest AGPL-regurgitation-risk surface -- and R14's RLE loss is squarely a loss module, the first one this project's Phase 12 work has touched. Every other WP in this phase (120/121/122) went to `codex:codex-rescue`; this one did not, and no part of it was delegated except (implicitly) nothing -- the module, its RealNVP flow, and its test suite are all written directly against R14's own equations, fetched from arXiv's HTML rendering (the allowlisted primary-source endpoint) across three separate targeted `WebFetch` passes rather than one broad read, specifically to cross-corroborate the one detail secondary summarization is most likely to get subtly wrong: which direction the flow runs and what sits at its latent end. All three fetches agreed on Eq. 8 verbatim, R14 sec. 4's `K=6`/`Lfc=3`/`Nn=64`/Leaky-ReLU conditioner config, and Eq. 12's affine coupling formula; the third fetch additionally recovered R14 sec. 3.2's `z_bar ~ N(0, I)` latent statement and sec. 3.3's `sigma_hat` sigmoid sentence -- resolving A65, which WP-122 had deliberately left open rather than guessed at. No detection-repository code, including R14's own official implementation, was opened at any point (AGENTS.md sec. 7).

**The loss (Eq. 8, `log s` dropped per R14's own stated implementation choice):** `L_rle = -log Q(x_bar) - log G_phi(x_bar) + log sigma_hat`, where `x_bar = (mu_g - mu_hat) / sigma_hat` is the standardized residual, `Q` a fixed unit Laplace (no parameters, R14's "preset density"), and `G_phi` the density the RealNVP flow induces over `x_bar` via the standard change-of-variables identity. `Q`'s gradient reaching `mu_hat`/`sigma_hat` independently of the flow is R14's own stated "gradient shortcut" motivation for the additive split (sec. 3.2) rather than an incidental property -- `test_gradient_shortcut_reaches_mu_and_sigma_with_the_flow_frozen` freezes every flow parameter and confirms the regression head still receives nonzero gradient, a direct test of that specific paper claim rather than a generic backprop smoke test.

**The flow (Eq. 12, Appendix A):** six stacked affine coupling layers over the 2-D residual, each holding one coordinate fixed and transforming the other by a scale-and-shift pair from a 3-layer/64-unit/Leaky-ReLU conditioner. The paper states the coupling formula but not that the fixed/transformed partition alternates between layers -- that is RealNVP's own standard construction (Dinh et al., which R14 cites rather than restates), and without it one of the two coordinates could never move at all under the whole stack, making it degenerate rather than a valid bijection over R^2. Implemented as `synthesize` (the generative `z -> x` direction) and `invert` (`x -> z`, RealNVP's defining closed-form property, needed for density evaluation) as two directions of the same module rather than one direction plus a solved inverse.

**What a unit gate can and cannot prove here:** the mechanism-claim validation this row's roadmap text (and R14 itself) cares about -- RLE beating an OKS-only ablation -- needs the [GPU][HUMAN]-gated WP-125 tier run this session cannot execute. Every property a unit test *can* pin was therefore treated as load-bearing rather than incidental: `test_flow_density_integrates_to_one` numerically integrates the flow's induced density over a 161x161 grid and checks it lands within 0.05 of 1.0 -- the single most discriminating check of change-of-variables bookkeeping, since a base-density term without its accompanying log-determinant (or vice versa) passes every other test in the file but does not integrate to a proper probability density. `test_log_determinant_matches_autograd_jacobian` compares the analytic per-layer log-determinant against `torch.autograd.functional.jacobian`'s own determinant, independently of `invert` entirely -- self-consistency between `synthesize` and `invert` alone could not catch a sign error that happened to cancel between the two, and this check cannot make that mistake since it never calls `invert`.

**Design decisions this row records rather than silently makes (A65, A66):** sigma's positivity mapping (deferred by WP-122/A65) resolves here as `torch.sigmoid`, applied in the loss rather than the head, per R14 sec. 3.3's own stated sentence -- not chosen freehand. Visibility masking (A66, new): COCO `v=0` is excluded unconditionally (no ground truth exists to train against), `v=1`/`v=2` both contribute equally, since COCO's own OKS protocol already scores against `v>=1` regardless of occlusion and training against a narrower set than the metric evaluates against would be the inconsistent choice, not the safe one.

**A structural fix along the way:** every other loss in this repo (`instance_mask_loss`, `ciou_loss`, `probabilistic_iou`, ...) is a parameter-free pure function over already-produced network outputs. RLE's flow is the first loss component in this project with trainable weights of its own, and a first draft reached for a lazily-built module-level global singleton to hold it -- convenient for a bare function's signature, but a violation of this project's own "no global mutable state" rule, undiscoverable by an optimizer's `.parameters()`, and unable to move devices or checkpoint the way a registered submodule does. Corrected before landing: `RLELoss` is an `nn.Module` owning the flow as `self.flow`, the shape every other stateful component in this codebase (`DualDetectionHead`, `ProtoNet`) already takes.

### WP-124 — a premise checked against the installed package, not the roadmap text

<a id="wp-124"></a>

Delegated to `codex:codex-rescue`, `--write`, scoped to `src/lucid_yolo/eval/coco_eval.py` and a new `tests/eval/test_keypoint_eval.py` only -- no decoder, model, CLI, roadmap, or assumption edits, no commit, no full `make gate` inside the sandbox. Before dispatch this row's own Scope text was verified against the installed environment rather than trusted: `torchmetrics==1.9.0`'s `MeanAveragePrecision.__init__` accepts `iou_type` values of only `"bbox"`/`"segm"` (checked via direct signature/source inspection, not memory), so the row's premise -- "keypoint mAP wired the way WP-069 wired box mAP" -- was factually wrong for what is actually installed. This did not rise to an AGENTS.md sec. 4 escalation: the row's actual intent, a COCO-OKS-faithful engine wired against WP-121's ground truth and WP-122's decode, is still satisfiable, just through a different already-a-dependency library (`faster_coco_eval`, R23) called directly instead of through torchmetrics's wrapper. The spec handed to codex stated the corrected mechanism up front and explicitly instructed it to re-verify the premise itself against the installed package before writing any code, rather than trust the prompt.

**What codex built:** `keypoints_to_predictions` adapts a fixed-size `(B, N, K, 2)` decode-shaped batch into COCO prediction dicts, filtering the score-zero padding rows a fixed decoder shape produces and remapping contiguous class labels back to original COCO category ids -- the same shape of adapter `detections_to_predictions` already provides for boxes, not a new pattern. `evaluate_keypoints` builds a minimal in-memory COCO ground-truth/result document pair from WP-121's `keypoints`/`keypoint_vis` tensor convention directly (`v == 0` excluded from both OKS matching and the per-instance bounding-box area, `v >= 1` included -- A66 applied here exactly as WP-123 defined it, not re-decided) and drives `faster_coco_eval.COCOeval_faster(iouType="keypoints", kpt_oks_sigmas=...)` through its native `evaluate()` / `accumulate()` / `summarize()` sequence, one-shot rather than `DualPathEvaluator`-streaming since no keypoint decode pipeline exists yet for a streaming caller to feed. `COCO_KEYPOINT_OKS_SIGMAS` names R12's 17-point sigma table as a module constant rather than leaving it to the library's own `Params.setKpParams` default -- the values are identical today, but a project-owned constant keeps the number this project reports against from silently moving if the dependency's own default ever changes.

**Independent re-verification (lead, outside the sandbox):** `tests/eval/test_keypoint_eval.py` -- 5 passed (padding-drop/category-remap adapter contract; a perfect-match OKS oracle scoring near-1.0 AP; a far-miss oracle scoring exactly 0.0 AP, ruling out a category-only match; the empty-input all-zero ten-key contract, avoiding a degenerate call into `COCO.loadRes`'s empty-list case; corrupting only a `v=0` target coordinate and asserting the ten stats are bit-for-bit unchanged, reusing WP-123's own `TestVisibilityMasking` pattern). `coco_eval.py`'s module doctests -- 6 passed, 1 pre-existing skip. `mypy` on the file -- clean. Full `tests/eval/` regression -- 216 passed (211 pre-existing + 5 new, matching codex's own reported delta exactly). Full `make gate` -- 2319 passed, 37 skipped, 3 deselected, 27/27 goldens, run twice to confirm no stray failure line in the fuller output.

**No new assumption recorded.** The spec instructed codex not to add an `ASSUMPTIONS.md` row itself and to flag anything it found instead; nothing surfaced that qualifies -- the sigma-table caution above is a documented fragility comment, not an open gap, since the value is pinned as a citable constant rather than inherited implicitly either way.

### WP-121b — a segmentation ring only an oriented reading actually needs

<a id="wp-121b"></a>

Surfaced during Phase 12's own dogfooding rather than invented: validating a fuse-augmentations `Task.KEYPOINTS` COCO export against WP-121's reader (an offline check of the user's own upstream package, ahead of wiring a fixture from it) found that every annotation in the export parsed to zero instances. Root cause traced two layers deep. The proximate one was upstream -- fuse-augmentations' `CocoWriter` never emitted a `segmentation` field for `Task.KEYPOINTS`, even though the outline polygon it would need was already computed for every shape regardless of task -- and was fixed there, by the user, not here (AGENTS.md sec. 6/7: no fix to a dependency's own source from this repo). The deeper one was this repo's own: `_parse_annotation` (WP-014's original reader) required a *parseable* `segmentation` ring to retain any annotation at all, in **every** mode -- plain, oriented, and WP-121's own keypoints reading alike -- so the upstream bug's fix alone was not sufficient; a keypoints-only or plain-detect-only export genuinely has no ring to give, by design, and the reader treated that as identical to a crowd or RLE annotation it should legitimately drop.

**The fix, scoped precisely.** Two kinds of "no ring" are not the same claim. A `segmentation` value that is *present but unusable* -- an RLE crowd dict, an empty list, a ring under three points -- is never a real countable instance, in any mode; that half of WP-014's original policy is untouched, and it is exactly what `TestKeypointParsing`'s own existing fixture already exercises (one RLE annotation, no `iscrowd` flag set, expected to still be dropped) -- a first draft that relaxed on "ring is `None`" alone broke that test immediately (`KeyError` reading `ann["keypoints"]` off an annotation the keypoints branch was never meant to see), which is what forced the narrower `raw_segmentation is not None or self._oriented` condition actually landed. A `segmentation` key that is **absent entirely** is the other case, and it is fatal only for `oriented=True`, which has no other source for the rotated box (`_oriented_targets` derives both `boxes` and `rboxes` from the same quadrilateral ring); a plain or keypoints reading now keeps the box instead. `_build_targets` collapses the whole image's `polygons` to `[]` the moment any one retained instance lacks a ring, rather than leaving a ragged per-instance mix -- `Targets.polygons`'s own documented contract is "0 (no masks) or N," never in between.

**Why this does not reopen the silent-empty-mask hole.** `instance_mask_targets` already raises `ValueError` when `len(target.polygons) != len(target.boxes)` -- a guard written for exactly this shape of mismatch, already in place before this fix, for a different reason (WP-099b's YOLO-layout path, which never carries rings at all). Relaxing the parse-time gate does not need a new guard; it needs this existing one to still be reachable, and it is: requesting `mask_targets=True` against a ring-less image raises there precisely as it does for a genuinely maskless dataset.

**Verification.** `tests/data/test_coco.py::TestRinglessAnnotations` -- a plain reading keeps both ringless boxes with `polygons == []`; an oriented reading against the same file still returns zero instances, locking in that oriented mode's requirement did not loosen; `instance_mask_targets` against the same ringless targets still raises, naming "detection-only annotation cannot supervise masks." Every existing `CocoDetectionDataset` caller re-run as a regression sweep -- `test_coco.py`, `test_coco_eval.py`, `test_annotations.py`, `test_build_dota_tiles.py`, `test_overfit_batch.py`, `test_datamodule.py` -- 175 passed. `coco.py`'s own module doctests and `mypy --strict` both clean. Full `make gate` green (2322 passed, 37 skipped, 27/27 goldens), run against the project's actually-pinned `fuse-augmentations` wheel -- deliberately not the newer local checkout the upstream fix and the keypoints-fixture wiring both need, since that checkout's widened `Shape` vocabulary moves `class_names(ClassMode.SHAPE)`'s span for every task (det/obb/seg included) and would silently drag the frozen `*_num_categories` goldens along with it. That vocabulary-scoping gap is a separate, still-open finding, reported back rather than worked around: `class_names()` has no parameter to restrict its span to a shape family, so det/obb/seg cannot currently pin "geometric shapes only" against a `fuse-augmentations` version new enough to carry keypoints at all. The keypoints-fixture wiring this fix unblocks (`tests/fixtures/synthetic.py::generate_keypoints_fixtures`, a small fixed animal subset for an overfit-friendly category count) is validated end-to-end against a local dev-only override but held uncommitted pending that upstream resolution and the resulting pin bump.

### WP-131 — a dependency fixed upstream, checked rather than trusted

<a id="wp-131"></a>

The upstream half of WP-121b's "still-open finding" landed same-day: `class_names(class_mode, shapes=...)` (opt-in, defaulting to the old full-vocabulary span when omitted, so every existing caller is unaffected), a mirrored `SyntheticConfig.colors` field, and the `Task.KEYPOINTS` segmentation-emission fix WP-121b already depended on. Verified rather than assumed fixed: the first check found the new parameter existed and its own tests passed, but `generate_dataset` -- the convenience function this project's fixtures actually call, not the primitive directly -- still called the bare unscoped `class_names(config.class_mode)` at its one writer-construction call site, so `generate_dataset(..., shapes=DEFAULT_SHAPES, ...)` kept emitting all 16 categories regardless. Reported back with the exact line and the one-line fix it needed; landed within the hour as an amended commit (`74af985`), re-verified empirically both ways -- `shapes=DEFAULT_SHAPES` now yields exactly `['circle', 'rectangle', 'square', 'triangle']`, `shapes=animal_shapes(2)` yields exactly `['duck', 'elephant']` -- before trusting it for anything downstream.

**Pinned by commit, not by version (R21).** `fuse-augmentations`' version string had not moved (`0.10.0.dev0`) across every commit in this saga, and PyPI forbids re-uploading that string, so a bare `uv lock` re-resolve would have kept silently re-pinning the stale 2026-08-02 wheel. `pyproject.toml`'s dependency became a git+commit-SHA pin instead (`git+https://...@74af985...`) -- matching R21's own provenance row, which already cited a fixed commit rather than a floating version, closer to A26's original intent than the PyPI-registry pin the project had drifted to. `uv lock`'s re-resolution incidentally also picked up `onnx`/`onnxruntime` into the lock for the first time; both had been declared in the `dev` dependency group but were absent from `uv.lock` entirely, a pre-existing drift unrelated to this row that a plain `uv sync --frozen` had exposed by removing the two packages someone had evidently `pip install`ed by hand outside the lock.

**Goldens verified unchanged, not re-frozen.** `generate_detseg_fixtures`/`generate_obb_fixtures` gained an explicit `shapes=DEFAULT_SHAPES` argument -- defensive, not corrective, since that was already the implicit default -- so the fix restores exactly the 4-category vocabulary the frozen goldens were built against. Full `make gate` after the pin bump: 2325 passed (2322 plus the two new keypoints-fixture tests plus one doctest), 27/27 goldens passed with no `--freeze` needed, confirming the restored parity rather than assuming it.

**`generate_keypoints_fixtures`.** 12 images, `animal_shapes(2)` -- duck and elephant, not fuse-augmentations' full 12-animal vocabulary -- specifically to keep an eventual overfit-style milestone run's category count low; more animal classes means a wider vocabulary a memorization-style run has to fit before its own metric means anything. Mirrors `generate_detseg_fixtures`/`generate_obb_fixtures`'s existing shape exactly: idempotent, seeded (`KEYPOINTS_SEED`), a matching `keypoints_fixture_dir` session fixture in `conftest.py`. `test_keypoints_set_loads` pins every annotation's landmark table at fuse-augmentations' fixed 16-point animal schema and the category count at or under `KEYPOINTS_ANIMAL_COUNT`; `test_keypoints_generation_is_deterministic` mirrors the existing det/obb determinism tests byte-for-byte.

### WP-132 — every part built, nothing connected

<a id="wp-132"></a>

Phase 12 shipped five consecutive rows of keypoint work and never once ran a keypoint. WP-120 added the target channels, WP-121 the COCO reader, WP-122 the head stem, WP-123 the RLE loss, WP-124 the OKS scorer -- each landed green, each with its own passing unit suite, and the composition none of them owned did not exist. `DetectionLitModule`'s task enumeration still read `detect`/`segment`/`obb`, so the keypoint stem was never constructed by anything; `RLELoss` was exported from `losses/__init__.py` and had **zero call sites** outside the module that defined it; `keypoint` appeared nowhere in `ptl/module.py`, `ptl/datamodule.py`, `data/collate.py` or `models/build.py`. The suite was green because every test was a component test, and a component test cannot fail for a wire that was never run.

The trigger for finding it was a question about running the overfit gate, not a failing test -- which is the part worth recording. Nothing would have surfaced this until someone tried to train, because "wire absent" and "wire correct" are indistinguishable to a test suite that only ever exercises the parts. WP-087 and WP-088 exist for exactly this reason and are the precedent this row follows: each is a *training path* row that carries its own overfit-100 golden as DoD, rather than a component row trusting that a later integration will notice. The gate is the thing that makes a wire testable at all.

**Additive, not substitutive.** The RLE term enters the detection total at `keypoint_gain` (A68) alongside the box and class terms, which makes this WP-087's structural shape rather than WP-088's -- the oriented path is the one task that *replaces* terms, zeroing `box_gain`/`l1_gain` inside the dual loss and respending them on rotated equivalents. Keypoints add a channel of supervision without contesting any existing one, so the arrangement is the simpler one and the `keypoint_gain=0` bit-exactness test is what proves it: with the gain at zero the detection total is bit-identical to a detect run, so the term is genuinely additive and the gain genuinely consulted rather than folded in somewhere upstream.

**The whole transform layer, after a scoping that was wrong twice.** Every geometric transform rebuilds `Targets(...)` from scratch, and all of them dropped the keypoint fields on the floor -- `letterbox.py`, `mosaic.py`, `affine.py`, `mixup.py`. The first scoping fixed `Letterbox` alone and argued the boundary was principled: `_warp_targets` is a pure forward affine that resizes and pads, never crops, so no point can leave the canvas and there is no policy to decide, where mosaic/affine/mixup crop and would force a clamp-vs-demote decision an augmentation-off gate could not settle honestly. Every sentence of that is true and the conclusion was still wrong, because the premise it rested on was never checked: the claim "augmentation is off" was verified against the **validation** loader, which is letterbox-only, and generalized to the training loader, which is not. `_TrainPipeline` runs mosaic, `FusedAffineLetterbox`, mixup, copy-paste, HSV and flip unconditionally. The gate died on `pad_keypoints`' instance-count mismatch on its first batch, and the fix was the whole layer -- which is what made A70 this row's decision instead of WP-125's.

**The loud bug was hiding a silent one.** Re-reading the transform path after that failure -- rather than only the line that raised -- turned up `HorizontalFlip` constructed at `datamodule.py:767` with no `keypoint_flip_pairs` argument at all. The flip is K-generic by design and cannot know which index mirrors which, so the omission did not mean "no swap needed", it meant every mirrored sample supervised `flank_left` toward the location of `flank_right`. That trains, converges to a left/right-confused head, and reports nothing: no shape check fails, no loss value looks wrong, and the overfit gate would very likely still have passed. Worse, the first draft of A64 asserted the pairing *was* being threaded through -- a documentation claim written from what the code should have done rather than from what it did, and corrected only because the affine's hard error forced a second reading of the same path. A64 now resolves by reading the pairing off the dataset's own COCO category names (`left_eye`, `flank_left`), which is where the anatomy is actually stated.

**Symbols over animals, for reasons that outrank "simpler".** R21's `SymbolShape` family carries K=7 against the animals' K=16, and its landmarks are placed analytically -- three-quarters of the way from each outline's area centroid toward a named vertex -- rather than traced anatomically. Two consequences beyond the cheaper stem. Only 5 of 7 slots are populated on most symbols, so a symbol run carries `v=0` points natively and exercises A66's visibility exclusion inside a real training loop instead of only in `test_rle_loss.py`. And because the points are analytic, R12's OKS sigma table -- 17 values derived from measured *annotator* standard deviations on human anatomy -- describes a quantity that does not exist here, which is what A67 records: a uniform sigma is not a simplification of the real table but the only honest reading, since any per-point vector this project invented would fabricate a structure the data does not have.

**The pin bump was not fixture-neutral, and the frozen tree could not absorb it.** Reaching `SymbolShape` meant advancing R21 from `74af985` to `0a0cc64`, a commit that also redefines `GeomShape.TRIANGLE` from equilateral to obtuse-scalene and re-centres every outline on its area centroid. Both moved the existing fixtures substantially -- `obb_num_annotations` 44 to 61 (+38.6%), `data_checksums.bbox_area_sum` -45.2%, `image_std_sum` -35.5% -- and eight of 27 goldens went red. Six of those eight were in `goldens/frozen/0.2|0.3|0.4`, the per-release snapshots whose whole purpose (`check_goldens.py`: "a release's frozen goldens staying green *is* the frozen-golden regression") is that current code still satisfies every value a past release pinned. Re-freezing those would have rewritten what v0.2.0 actually produced, so the question went to the human rather than being resolved by the row that raised it. What decided it was that the failures partitioned exactly: every pure-code frozen golden (`assignment_cases`, `optim_toy`, `params_flops_*`) still passed, and the only two that failed were `data_checksums` and `fixture_checksums` -- the two whose producers render images through an external package. A generator-derived metric can never satisfy "current code must still satisfy every past value" once the generator changes, so those two were never really freezable; they were a category error in what got snapshotted, not a regression. They were removed from the frozen release tree (recoverable in history) and the two current copies re-frozen, leaving 21/21 green and the frozen-golden regression still meaning something for everything it can actually test.

**One flow or two, and a gate that cannot tell.** Wiring the RLE term forced a choice neither R14 nor R1 speaks to: the dual head has two branches, and a normalizing flow is the only trainable-weight loss in this project, so the term either shares one density across both branches or fits one per branch. One shared flow was taken (A69) on two arguments that are each weak alone -- it is what R14 actually describes, one density over one regressor's residuals, where two would assert the branches' errors differ in kind rather than merely in schedule; and it is the smaller parameter count. What matters more than the choice is that this row's own gate cannot check it: an overfit-100 run drives both branches to the same near-zero residual, which is precisely the regime where one flow and two are least distinguishable. So A69 ships `open` with WP-125 named as the first run whose branches disagree enough to tell, rather than being quietly ratified by a green gate that was never sensitive to it. The same caution applies to A68's `keypoint_gain = 1.0`: a memorization run will pass at almost any gain, so passing is not evidence the weight is right.

**Two numerical faults that no component test could have caught.** With the wiring done and the transforms carrying points, the gate still would not run -- it went non-finite inside the RLE flow, and the two causes are the clearest evidence this row could offer for why a *training path* is its own work package.

The first is a frame mismatch (A71). `decode_keypoints` returns absolute input pixels, deliberately and correctly -- it is the deployed decode. R14 forms the residual as `(mu_g - mu_hat) / sigma_hat` and bounds `sigma_hat` into `(0, 1)` with a sigmoid, which it can do because it regresses inside a top-down person crop: the crop is the normalization, so the paper never names a frame and this project had no reason to notice one was missing. Composed, the two make a residual whose *smallest* expressible value for a 40 px error is 40. Measured on the first batch, the keypoint term entered at **1300 of a 1342 total loss** -- three orders above the box and class terms it is supposed to sit beside -- and the flow overflowed on the second step. Normalizing both point tensors into the assigned box's frame first brings the median residual to **1.28**. Nothing about this is visible to `test_rle_loss.py`, which evaluates the loss at `O(1)` residuals because that is the regime the equations describe; the loss was never wrong, its input frame was unstated.

The second is a missing stabilization (A72). WP-123 built the coupling conditioner from R14's stated recipe -- 3 fully-connected layers, 64 units, Leaky-ReLU after each -- and emitted its raw linear output as the log-scale. R14 cites RealNVP for the coupling construction rather than restating it, and the parameterization of `s` is stated only in Dinh et al.: "To compute the scaling functions s, we use a hyperbolic tangent function multiplied by a learned scale", given there explicitly as a stability measure. A raw linear log-scale is not a simplification of that, it is a different layer, and the stack multiplies `exp(log_scale)` six times so the difference compounds. Measured after the frame was fixed: a residual of **27** arrived at the latent as **4e11** and at the loss as **7e22**. The general lesson is narrower than "read the citations": WP-123 read R14 closely and implemented what R14 states, and what bit it was a detail R14 *delegates* -- the one place a paper's own citation is load-bearing rather than contextual.

**The gate's number is small and that is the ruler, not the model.** With both faults fixed the gate runs its 100 epochs clean and scores **OKS AP 0.3357** over 548 instances -- next to the detection gate's 0.995, the oriented gate's 0.939 and the mask gate's 0.815, a number that reads like a failure. It was checked before it was believed, and the check says otherwise. Feeding the ground truth back through the scoring path as the prediction returns exactly **1.0**, so the harness, the sigma vector and the label space are all wired correctly and the ceiling is real. Displacing every point of every instance by a uniform offset then measures the slope: **+1 px scores 0.934, +3 px scores 0.269, +8 px scores 0.000**. On symbols a few dozen pixels across, scored at A67's uniform sigma, OKS AP is a cliff rather than a slope, and 0.3357 is where a landmark error of roughly 3 px lands. The floor was set from that measurement -- 0.30, with the same 0.05 tolerance the oriented gate uses -- rather than from the 0.75 that had been guessed before any number existed. One honest limit on the reading: the probe displaces every point *uniformly*, so it bounds the interpretation without proving it; the same AP is also consistent with a mixture of exact and badly-missed instances, which only a per-instance breakdown would separate.

Both faults were found by running the gate and neither by any test in the suite, which is the same shape as the wiring gap this row opened with. What is deliberately **not** claimed here is that the numerics are now healthy: the residual tail still spikes into the thousands on this data, the A70 off-canvas points meeting a flow whose base density is quadratic in the residual. The tanh removes the compounding feedback path; it does not make the tail bounded, and A72 says so rather than letting a green gate imply otherwise.

### WP-133 — an Unreleased section, and the rule it follows

<a id="wp-133"></a>

`CHANGELOG.md` had no `## [Unreleased]` section at all. 27 commits had landed since the 0.4.0 release commit (`0a3ef8b`) -- the dev-version bump to `0.5.0.dev0` and WP-132's own keypoint training path among them -- and every one of them carried zero changelog entries, so the file read as though nothing had shipped since 0.4.0. Surfaced by a direct question, not a check: nothing in `make gate` or the pre-commit hooks reads `CHANGELOG.md`'s content (`release_guard.py` checks a tag has *a* matching entry, not that intervening commits do), so the gap had no way to fail loud.

**Backfilled from `git log 0a3ef8b..HEAD`**, one bullet per user-visible change, grouped Added/Changed/Fixed to match the existing release sections' own convention. Internal-only commits -- test reorganization, roadmap bookkeeping, CI hook wiring, docstring formatting -- are omitted by the same judgment the existing sections already apply; none of the four shipped 0.x sections carries an entry for its own test suite. The keypoint task's five component WPs (120-124) plus its training-path composition (132) and the ringless-annotation fix (121b) collapse into one Added bullet rather than six, since a changelog reader wants "the keypoint task landed and what it cost to get right," not a WP-by-WP replay -- that granularity already exists in this table and in RESEARCH_LOG.md.

**The section states its own closing rule inline**, per the standing instruction this row also establishes: a commit that sets `__version__` to a plain, non-dev, non-rc value closes the `## [Unreleased]` section and opens the next one. Going forward, a WP's own commit should add its own `## [Unreleased]` bullet rather than the whole set being reconstructed after the fact the way this row had to -- the reconstruction is legible from git history exactly once; a habit of writing it live is not.

**Two pre-existing date mismatches surfaced and are recorded rather than silently corrected.** `[0.2.0] - 2026-08-10` against release commit `eb81caa`'s `2026-08-11`; `[0.4.0] - 2026-08-15` against `0a3ef8b`'s `2026-08-16`. Both one day earlier than the commit, both the same direction -- consistent with the date recorded being the day a run was accepted rather than the day the release commit landed, not a typo in either single case. `[0.3.0] - 2026-08-14` matches its commit exactly, so the convention is not applied consistently even under that reading. Left as a known discrepancy: correcting two dates nobody had complained about is a scope this row did not need, and the pattern is now written down for whoever next touches a release date to decide.

**`configs/pose_smoke.yaml` folded into this same commit, at the user's direction, rather than opening a separate WP.** WP-125 needed a launch config before a Colab session already underway could proceed; writing one is mechanical against the `det_smoke.yaml`/`seg_smoke.yaml` precedent (`task: keypoints`, `num_keypoints: 17` for COCO `person_keypoints`, `keypoint_gain: 1.0` per A68's un-tuned default, explicit `train_ann_file`/`val_ann_file` since `lucid_yolo.data.layout` only ever resolves the `instances_` spelling and `person_keypoints_{split}2017.json` sits beside it in the same annotations archive under a different name). Verified by dry-parse only -- `tests/ptl/test_cli.py::test_config_dry_parses[pose_smoke.yaml]`, the same DoD every other shipped config carries -- not by a training run. This is explicitly not WP-125's acceptance: that row is `[GPU][HUMAN]` and its criterion is RLE's mechanism claim against an OKS-only ablation, which a config file cannot settle and this entry does not claim to.

### WP-134 — a checkpoint reported at one eightieth of its own score

<a id="wp-134"></a>

WP-124 built the OKS scorer. WP-132 built the training path. Nothing ever joined them: `evaluate_keypoints` and `keypoints_to_predictions` had no caller anywhere outside their own module and its tests, and `cli/eval.py` branched on three tasks. WP-125's acceptance metric -- OKS keypoint mAP on COCO human pose -- was therefore not producible by any shipped command at all, which is the same shape of gap WP-132 opened with and the reason that row's own log ends where it does.

**The 0.006288 was not a decode fault, and finding that out first is what kept this row small.** A keypoints checkpoint run through `lucid-eval` fell through to `detect_eval` and reported `map=0.006288` against a training loop that had logged `val/mAP=0.48` at the same epoch. That reads as a broken decode, a broken un-letterbox, or a broken assignment, and any of the three would have made this a much larger row. It is none of them. `detect_eval.run` hardcodes `annotations/instances_val2017.json` regardless of what the checkpoint's task or class count is, and COCO's mAP averages average precision over **every category the ground truth contains**. A person-only checkpoint predicts contiguous class 0 and nothing else, so the other 79 categories each hold ground truth, receive no prediction, and score AP=0. The arithmetic closes: `0.006288 * 80 = 0.503`, against the 0.48 the epoch metric logged. Confirmed independently on a synthetic three-category case through `torchmetrics` directly -- perfect predictions in one category of three give `map=0.333` and `map_per_class=[1.0, 0.0, 0.0]` -- before any code here changed.

So the fix is the ground-truth file, not the mathematics. `pose_eval.run` reads `person_keypoints_val2017.json`, which names one category, and the dilution disappears as a side effect of scoring the protocol the checkpoint was trained for. `detect_eval`'s behaviour is untouched: for a `detect` or `segment` checkpoint the GT category count matches the model's own, and its average is over exactly the right set.

**Which row of the report compares.** Training's `val/mAP` is logged from the o2o branch alone -- `_val_decoder.decode_with_indices(head_out.o2o_cls, head_out.o2o_box, ...)` against the datamodule's already single-class targets. The comparable figure in a new report is the **e2e** entry, and the `nms` entry is the dense branch's own answer, expected to sit above it by R1 sec. 4.4's deficit. The runner prints that in words, because a report that does not say which figure it quotes is how the oriented tier confused two of them for a whole release.

**The keypoint mode enters `DualPathEvaluator` exactly where masks did, and differs where the pieces force it to.** Structural detection off `o2o_keypoints`/`o2m_keypoints` rather than a flag, so a detect, segment or obb checkpoint provably takes the path it already took -- those fields are `None` on its head output. Per-branch and per-path, so the E2E path reads the one-to-one branch's points at the E2E decoder's own anchor indices and the dense path reads its own. Two differences from the mask case, both dictated rather than chosen: `decode_keypoints` composes the raw offsets with the whole `(A, 2)` anchor grid and has no per-detection indexing, so this path decodes the dense batch and *then* gathers, where the mask path gathers coefficients and then decodes; and COCO's keypoint protocol is one matching over one document with no incremental update, so the points are accumulated for the split and scored once where the boxes stream. The second is affordable only because the sizes are opposite -- a split of masks is hundreds of gigabytes and a split of points is a few megabytes.

The `PAD_ANCHOR_INDEX` sentinel is `-1`, which `Tensor.gather` accepts and answers with the last anchor's row. It is clamped to a valid position and the result zeroed, the same two steps `_gather_coefficients` takes, and those rows carry score 0 and are dropped regardless -- belt and braces on a failure whose symptom would be a plausible pose at a correct box.

Points invert the letterbox through `Letterbox.inverse_map`, which is already point-shaped: `to_letterboxed_original` is that same call plus the score and class columns of a detection tuple. Reaching for the box function would carry two columns onto coordinates, and writing a second affine would be a second definition of what a letterbox is, which is the rule WP-053a set.

**The ground truth was the part that was quietly wrong, and it was wrong before this row.** `evaluate_keypoints` reconstructed each instance's `bbox` and `area` from the extent of its visible keypoints and forced `iscrowd=0` and `num_keypoints` to a recount. Every one of those four is a **supplied** field in COCO's protocol and each does real work: `area` is what OKS divides by *and* what draws the medium/large bucket boundary, and for a person it is the segmented area rather than the hull of the labeled joints; `iscrowd` and a declared `num_keypoints == 0` are the two flags the backend marks an instance ignored on, verified by reading `COCOeval_faster._prepare` in the installed `faster_coco_eval` 1.7.2 rather than recalling pycocotools' behaviour. Reconstructing them scores a different protocol under the same name. They are now read from the annotation when the target carries them, `annotations_to_target(with_keypoints=True)` carries them, and the reconstruction stays as the fallback -- so WP-124's one-shot callers, its doctests and every synthetic fixture keep the exact numbers they had.

The regression test makes the two readings disagree on purpose: an instance whose visible points span 2500 px² and whose annotation declares 20000 belongs to the medium bucket under one reading and the large bucket under the other, and exactly one of the two can be populated. Reconstruction reports `AP_medium=1.0, AP_large=-1.0`; the annotation reports the reverse.

**A K that is not 17 is refused rather than scored.** This protocol's ground truth, its point ordering and its sigma table are all COCO's person schema. A checkpoint predicting some other K has no correspondence to it -- point *i* of the prediction is not point *i* of the annotation -- and R12's sigmas measure annotator variance on joints that are not the ones being predicted. That is A67's argument read from the other side: A67 already refuses to fabricate a per-point table for analytically-placed synthetic landmarks and scores them on its own uniform sigma in its own gate, so this row needs no new assumption and invents no table.

**Validation is synthetic, and says so.** There is no COCO checkout and no GPU on this machine, so nothing here ran against real data. Each claim is separated by construction instead: the two branches carry offsets of opposite sign and each anchor a different magnitude, so of the four combinations of branch and index the evaluator could take, only the right one scores OKS 1.0. Both mutations were checked to fail -- crossing the branches and dropping the padding clamp each turn the suite red. What that does **not** establish is the figure a real pose checkpoint will report; the first honest reading of this instrument is WP-125's, on real data, and the arithmetic above is a prediction of what it should show rather than a measurement of it.

One promotion rode along: `_parse_keypoints` in `data/coco.py` becomes `parse_coco_keypoints`, because the evaluation reader now parses the same field off the same file format and a second copy would be free to disagree about the triplet stride or the visibility dtype -- after which a training run and its own acceptance score would read different ground truth from one file.

### WP-135 — a control arm, because one number states no direction

<a id="wp-135"></a>

WP-125's acceptance is not a pose figure. It is RLE's **mechanism** claim: that the *learned* residual density is what buys the improvement, rather than the reparameterization, the sigmoid-bounded per-point scale, or the `log sigma_hat` Jacobian that come with it. A claim of that shape cannot be settled by a run, however good the number, because one figure is equally consistent with the flow doing all the work, none of it, and two terms cancelling. It needs a paired run whose only difference is the mechanism under test -- and the tier had no second arm to pair against. This row is that arm, and nothing else: it builds the instrument and does not read it.

**The comparison is R14's own, not this project's invention.** R14 Table 7 ablates exactly this: "Laplace, learnable variance" -- the NLL over a learned per-point scale with no normalizing flow -- scores **67.4 AP** against full RLE's **70.5 AP** on COCO. Verified this session against the paper itself (arXiv:2107.11291, read through the ar5iv HTML rendering), not recalled. So the direction of effect is already published, and what a paired run here measures is whether it reproduces at this scale on this recipe -- a much smaller question than the one it would be if the baseline were invented. That is also why no assumption id accompanies this row: an assumption records an open project decision made without a source, and this one has a source with numbers in it.

**The loss is `RLELoss`'s formula with one term deleted.** R14 Eq. 8 (with the `log s` normalization R14's own implementation drops) is `-log Q(x_bar) - log G_phi(x_bar) + log sigma_hat` over the standardized residual `x_bar = (mu_g - mu_hat) / sigma_hat`. `LaplaceNLLLoss` is that expression without `G_phi`. Everything else is held: the residual is formed the same way, `sigma_hat` comes through the same R14 sec. 3.3 sigmoid (A65), the `log sigma_hat` term is the same reparameterization log-Jacobian, the sum is over the same two axes, A66's visibility mask is the same mask, and the zero-positives reduction is the same one. Holding all of it fixed is the entire point -- a control that differed in a second place would make the result unattributable, and every one of those clauses is a place it could have differed silently.

The identity is asserted rather than argued: `test_is_the_rle_loss_with_exactly_the_flow_term_removed` scores both losses on shared inputs and checks the difference equals the flow's own log-density term to `1e-5`. A drifted residual, a different sigma activation or a different reduction in either module turns that test red, and nothing else in either suite would have reported it.

**"Learnable variance" names what is kept.** `sigma_hat` is still a head output and still trained -- the `log sigma_hat` term is what gives the likelihood a reason to prefer a small scale where the model is accurate, and `test_gradient_reaches_both_the_location_and_the_learned_scale` pins that both `mu_hat` and `sigma_raw` receive gradient. What the ablation removes is the learned *shape* of the error distribution. The further-degenerate baseline with a fixed scale too is not this row, is not built here, and is just L1; R14 lists it separately.

**Zero parameters is the structural difference, and it is load-bearing twice.** `RLELoss` is the one loss in this repository carrying weights -- 42 flow tensors on the tiny test module -- which is why it is held as a submodule at all. `LaplaceNLLLoss` has none: `.parameters()` is empty, `state_dict()` is empty, and construction draws no RNG. The second consequence is the useful one. The module docstring's construction-order rule exists because the flow's `Linear` layers consume RNG and would perturb every parameter drawn after them; an ablation module draws nothing, so its weights are bit-for-bit a same-seed detection module's, and `test_the_ablation_adds_no_parameters_to_the_model` pins its state dict as a detection module's plus the point stems and no third thing.

**The attribute keeps the name `rle_loss`, deliberately.** It is an `nn.Module` attribute, so it prefixes the loss's state-dict keys, and renaming it to something task-neutral would move `rle_loss.flow.*` to a new prefix and stop every keypoints checkpoint produced before this row from loading -- including the one behind the frozen `goldens/gpu/overfit_micro_kp.json`. That is the same key-stability rule WP-087 established and this module keeps its stages flat for, arriving from a direction it had not been tested from. The name is now slightly wrong under one of two arms; a broken checkpoint is worse than a slightly wrong name, and the attribute says so in a comment rather than being quietly left to look like an oversight.

**Selection is additive, and the default is what makes it so.** `DetectionLitModule(keypoint_loss=...)` defaults to `"rle"`, so every shipped config, every checkpoint on disk and every accepted figure describes exactly what it described before. The two names are validated against one mapping that also builds them, mirroring how `task` and `rotated_iou_form` are validated in the same constructor -- one dict rather than a name tuple beside a constructor branch, so the accepted set and the thing each name builds cannot drift apart. An unrecognized value raises at construction, which matters more here than the pattern usually does: a silent fallback would produce an "ablation" run that trained the very objective it was meant to control for, and a comparison whose two arms are the same arm looks exactly like a null result.

`pose_smoke_ablation.yaml` is `pose_smoke.yaml` with one line added. Seed, variant, epochs, callbacks, every gain and the data placeholders are identical, `keypoint_gain` included -- it stays at A68's un-tuned `1.0` not because that is right but because moving it would make the arms differ twice. Verified by dry-parse only (`tests/ptl/test_cli.py::test_config_dry_parses[pose_smoke_ablation.yaml]`), the same DoD every shipped config carries.

**One claim this row makes about numerics, and its limit.** The ablation degrades more gently than RLE outside the `O(1)` residual range A71 normalizes into: the unit Laplace's log-density is linear in `|x_bar|`, where the flow's latent is exponential in a compounding log-scale and its base density quadratic in that latent -- A72's non-finite run. `test_a_far_out_residual_stays_finite_rather_than_overflowing` measures it at a residual two orders out rather than leaving it as an argument. What that does **not** say is that the ablation is safe unnormalized: a loss dominated by a handful of far-out points is still training the wrong thing, and both arms take the same A71 box frame for the same reason.

**Nothing here was run against real data.** There is no COCO checkout and no GPU on this machine, so this row ships a tool and a set of unit gates, not a result. The 67.4-vs-70.5 figures are R14's measurement quoted, not this project's; the arithmetic of the comparison is a prediction of what a paired run should show, and the first honest reading of it is WP-125's, which is `[GPU][HUMAN]`-tagged and belongs to the user's machine.

______________________________________________________________________

### WP-136 — a filename is not a scratch space for whatever axis is on your mind that day

<a id="wp-136"></a>

`src/lucid_yolo/configs/` had grown eight files under what read as one two-token scheme (`<task>_<tier>.yaml`), but a closer look found three separate inconsistencies rather than one: two files used a third token to mean something no sibling file's name carried at all, one word (`ablation`) meant two structurally different things across two files, and the scheme itself never named the one axis every config actually sets differently -- `variant`. User direction (mid-session, superseding this row's own first draft) was to stop patching the two collisions and instead put every config on one explicit scheme: `<task>_<variant>_<tier>[_<detail>].yaml`, variant spelled out in full rather than left as the bare letter the CLI and each file's own `variant:` field use.

**The full rename, driven by what each file's own `variant:` line already said (checked, not assumed):**

| old | new | why |
| -- | -- | -- |
| `det_smoke.yaml` | `det_nano_smoke.yaml` | `variant: n`, task truncated as it already was |
| `seg_smoke.yaml` | `seg_nano_smoke.yaml` | `variant: n` |
| `obb_smoke.yaml` | `obb_nano_smoke.yaml` | `variant: n` |
| `pose_smoke.yaml` | `pose_nano_smoke.yaml` | `variant: n`; `pose` kept as `task: keypoints`'s established domain term (roadmap's own "Pose-smoke tier" language, not a choice this row introduced) |
| `det_ablations.yaml` | `det_small_ablations.yaml` | `variant: s` -- the one config in the directory that is not `n`, previously invisible in its own filename |
| `overfit_100.yaml` | `det_nano_overfit_100.yaml` | had no task prefix at all, relying on `task`'s implicit `"detect"` default; `variant: n`; `100` kept as WP-040's own detail (image count, not a tier word) |
| `det_yolo_smoke.yaml` | *dropped* | see below |
| `pose_smoke_ablation.yaml` | `pose_nano_smoke_laplace_nll.yaml` | `ablation` collided with `det_ablations.yaml`'s different sense (a whole blueprint tier, sec. 10 tier B, ~120-epoch paired runs, its own sign-matching acceptance against R1 Tables 2/3/5/6) where this file is a same-budget arm of the *smoke* tier; `laplace_nll` names the mechanism swapped in, matching how the loss class and the `DetectionLitModule` docstring already refer to the arm |

**`det_yolo_smoke.yaml` was dropped, not renamed.** It was first renamed `det_smoke_layout.yaml` (naming the `--data.layout` axis, not the one value it took), then reconsidered a second time against `det_nano_smoke.yaml` sitting beside it in a listing -- "layout" alone read as no more specific than the word it replaced, so the candidate became `det_nano_smoke_yolo.yaml`, spelling the actual value. Diffing it against `det_nano_smoke.yaml` at that point (checked, not assumed) found the two differ in exactly three things: the `data:` block (a YOLO root and its `layout: yolo`), `model.num_classes` (`3` for the demo root vs COCO's `80` -- a property of the demoed root, not an independent axis), and header prose. Everything else -- schedule, gains, optimizer, `variant: n` -- is identical. A whole config file whose only content beyond a data override and its dependent class count is header prose is not a ninth point in the scheme; it is three CLI flags on `det_nano_smoke.yaml` that never needed a file of their own. The prose itself was not lost: `docs/DATASETS.md`'s "Dataset formats" section already carries the deeper mechanics (`layout` vs `format`, `YOLO_CANDIDATES`, `data.yaml` priority, why `segment` is unavailable), so the file's comments were largely restating what that section already says more completely. `docs/TRAINING.md` gains a "A YOLO-format root instead of COCO's" section instead: the three-flag override, why `--data.layout yolo` is stated rather than probed (A63's undecidable-tie case), the `num_classes` coupling to the root's own `data.yaml`, and the `segment` refusal, each in one paragraph rather than a whole file's worth of header comments.

**Variant spelled out, deliberately creating a second vocabulary.** Every config's own `variant:` field and the `--variant` CLI flag stay bare letters -- that is the real API and does not change here. The filename now spells `nano`/`small` for readability at the cost of a reader having to know `nano == n` when connecting a filename to the config's own content or an override. Confirmed with the user rather than assumed, since nothing else in this codebase (code, docs, or CLI) has ever spelled these out.

**What that scheme cannot promise, stated rather than left implicit.** A filename's variant token is the config's *default*; `--variant s` on `det_nano_smoke.yaml` still runs an s-scale job under an n-named file, exactly as `keypoint_gain 1.0` in a filename would not survive a `--model.keypoint_gain` override. The name documents the shipped default, not a runtime guarantee -- the same caveat every config's own header comment already states about `variant` ("this file's default scale, not a claim about the run").

**What did not move: the historical record.** `docs/REPRODUCTION_REPORT.md`'s rows are explicit about being "the exact commands as they were run at the time" (`docs/TRAINING.md`), and the `ROADMAP.md`/`ENGINEERING_LOG.md` rows for WP-040, WP-099c, and WP-135 describe what those WPs built and named *then*. WP-129 already set this project's precedent for exactly this situation: moving `audit_licenses.py`/`check_commit_trailers.py` into `scripts/lint/` did not rewrite WP-084's, WP-109's or WP-115's rows, which still cite the pre-move path. None of those earlier rows are touched here; this section is where all eight old names are cited for the last time.

**What did move: everything a reader or the tool itself would act on.** `README.md`, `docs/TRAINING.md`, `AGENTS.md`, `docs/DATASETS.md` (live usage docs, not a historical ledger -- a stale name there is just a broken command); `scripts/overfit_micro.py`'s and `scripts/shapes_regression.py`'s `_RECIPE_PATH` (functional, not prose -- the old name pointed at a file that no longer exists under it); `src/lucid_yolo/cli/train.py`'s `_DEFAULT_CONFIG` and its own doctests; `tests/ptl/test_cli.py`'s hardcoded `test_configs_dir_is_non_empty` set and every parametrized config-name literal; and every config's own internal header comment that named a sibling file. `CHANGELOG.md`'s still-open `## [Unreleased]` section also updated, on the same live-vs-historical reasoning -- it describes what the next release will ship, not what a past WP called something at the time, so a stale name there would mislead exactly the reader it exists for.

No test needed a structural change: `tests/ptl/test_cli.py`'s `_CONFIG_PATHS` is `_CONFIGS_DIR.glob("*.yaml")`, so all eight renamed files are discovered and dry-parsed under their new names automatically. The one hardcoded list (`test_configs_dir_is_non_empty`'s membership assertion) and the several tests parametrized on a literal config name still needed the literals updated by hand, since a glob does not know what a test author meant to assert about.
