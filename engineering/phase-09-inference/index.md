# 🔮 Phase 9 — Inference and generalization


## WP-089 — the grid was already written three times

<a id="wp-089"></a>

The package exists because a letterbox inverse must not be written twice: two copies of one transform, each with its own passing tests, are free to disagree by a pad the day either side's rounding changes. That part went as intended — nothing in the predict path computes a ratio, a pad or a corner.

What the work found is that the *other* geometry had already made the mistake. The head's anchor grid — divide the canvas by the three strides, build the points, move them to the device — was written once in the dual-path evaluator, once in the oriented evaluator, and would have been written a third time here. The stride triple `(8, 16, 32)` had three separate declarations to match, two of them private. Nobody duplicated it carelessly; each consumer needed three lines and wrote them. That is how the duplication a project actively guards against still accumulates: the guarded case is the one everyone can see is dangerous, and the neighbouring case looks too small to bother with.

Consolidated into `assign/grid.py`, beside `make_anchor_points`. The general signature won: a canvas is `(height, width)`, and a square side cannot express a letterbox that is not square. Every golden was unchanged afterwards — 20/20 — which is the only evidence that a refactor of the scoring path changed nothing.

A second finding, recorded because it is a trap the next two packages walk into: `DetectionLitModule.forward` returns a `DualHeadOutput` for **every** task. A segmentation or oriented checkpoint handed to a detection predict path therefore produces boxes, silently, with its mask or angle branch never consulted — plausible output, no error, and the masks the caller asked for simply absent. The refusal is in the library rather than in the command for that reason: a wrong answer that looks right is worse than a traceback, and the command is not the only door.

Two citations that belong with the above rather than in the roadmap row. The duplicated-inverse hazard has a name in this project — it is the **WP-053a defect class**, the shared-decode function whose second caller found what its first never could ([WP-090b](https://borda.github.io/lucid-YOLO/work_packages/wp-090b/#wp-090b) is the same class caught later). And the task string is read from the checkpoint by the same mechanism `lucid-eval` reads it, so the two commands cannot disagree about what a checkpoint is: a refusal that depended on which entry point asked would be worth less than no refusal at all.



## WP-090 — a crash the evaluator could never reach

<a id="wp-090"></a>

`decode_instance_masks` raises on a zero-length instance axis: `F.interpolate` rejects a `[1, 0, 16, 16]` tensor outright. The function has been in the tree since WP-053a and no test caught it, because from the evaluator it is unreachable — both decoders always hand it a fixed 300 rows, padding included, so the axis is never empty. Predict decodes only the survivors, and an image with nothing above the confidence threshold has none.

The general shape is worth keeping: a function whose only caller pads to a fixed length has never been tested at its degenerate input, and the untested case arrives with the second caller rather than with the first. Guarded at the call site here, with the crash pinned by a test on both decode paths; the shared function itself is left for its own package.

Two things this package had to get right that a shape check cannot see. The masks are assembled while the boxes are still on the letterboxed canvas, because that is the frame the A11 box crop is defined in — cropping after the inverse letterbox leaves every box right and every mask quietly wrong. And the mask path is **not** decoder-independent: the one-to-one coefficients are what the E2E decode reads and the dense ones are what the suppression path reads, so pairing either decoder's rows with the other branch's coefficients yields a plausible mask of a different object. The test plants different prototypes per branch, so a path reading the wrong one returns a mask on the other half of the picture and fails rather than passing with a mask nobody looks at.

For the report, COCO RLE through `faster_coco_eval.mask.encode` — already a declared dependency, and already the encoder every `segm_` statistic is measured through, so a predicted mask on disk and a scored mask are the same object in the same format. Rejected: a polygon contour needs a tracer and is lossy on masks with holes, which the box crop routinely produces; a sidecar `.npz` splits one prediction across two files, so a report can be archived into silently describing masks that are gone; mask-derived scalars answer a smaller question than the caller asked.



## WP-090b — the degenerate input arrives with the second caller

<a id="wp-090b"></a>

`decode_instance_masks` has raised on a zero-length instance axis since WP-053a wrote it. `F.interpolate` treats that axis as channels and rejects an empty one outright. No test caught it and no reviewer would have: from the evaluator the case is unreachable, because both decoders always hand the function a fixed 300 rows with their padding included, so the axis is never empty. Predict decodes only what survives the confidence cut, and an image can have nothing above it.

The rule worth keeping: **a function whose only caller pads to a fixed length has never been exercised at its degenerate input, and that input arrives with the second caller rather than the first.** The shape of the bug is not "someone forgot the empty case" — it is that the empty case did not exist while there was one caller, so no amount of care at the time would have surfaced it.

Returning the empty stack rather than raising is the shape contract, not politeness: a caller assembling per-image results needs a tensor of the right rank and dtype to stack, and an exception pushes the count check out into every such caller, which is where the copies start.



## WP-091b — the tie-break decoder implementation

<a id="wp-091b-engineering"></a>

The decoder itself is the small half. The rule is greedy class-wise suppression whose overlap is `rotated_iou` — the same exact polygon measure the oriented evaluator grades with — over boxes canonicalized first, so two detections stated a half turn apart cannot read as two objects. Class separation tests the label directly rather than borrowing `batched_nms`'s coordinate-offset trick, because there is no offset that separates rotated boxes without also turning them.

Why its overlap threshold has no paper source to cite, and what that cost to establish, is recorded at [wp-091b.md#wp-091b](https://borda.github.io/lucid-YOLO/work_packages/wp-091b/#wp-091b).



## WP-099 — a format with no specification

<a id="wp-099"></a>

The clean-room rule says the reader is written from the format as the datasets publish it, never from an implementation's reader. That is a clear instruction with a hole in the middle: the YOLO label format has no specification on the paper allowlist, and the obvious ways to learn it — a search, a docs page, somebody's loader — are all the thing the rule forbids. What closed it was a published dataset export already on the machine, CC BY 4.0, a data artifact with no implementation lineage: registered as R32, and read for what it actually contains rather than what a tutorial says it contains.

That got four facts and left four gaps, and the gaps are the useful part. The export's fields lie in `(0, 1]` and its clipped corners land on exactly `0` and `1`; `names` is an index-ordered list beside `nc`; split entries are written `../train/images` for a tree at `<root>/train/images`; every image has a label file and none is empty. What it could not settle: whether a ten-field oriented row exists (R18 appends a `difficult` flag to its own label lines, and the normalized variant has no published spelling for one), whether `data.yaml` may carry a `path:` key, which of the two directory spellings ranks first when a root satisfies both, and which image extensions are in scope. Each of those is rejected or documented as a precedence choice rather than guessed — A54 through A58 — because a guess that silently mis-parses is exactly the failure the clean-room clause exists to prevent, and it fails quietly by producing a dataset rather than an error.

Two decisions where the format's own limits set the line. **A missing label file is an error; an empty one is a background image** — writing zero bytes is a positive statement that the image holds no objects, while not writing one is silence, and a tree with no manifest cannot tell silence from a half-finished download. Reading silence as "no objects" turns a broken export into a dataset that trains on backgrounds without ever saying so. **A coordinate outside `[0, 1]` is rejected, not clamped, and compared with no epsilon** — it is not a slightly-off box but a file that was never normalized, typically written in pixels, and clamping turns it into a wall of degenerate edge boxes that would then be trained on.

The layout question was the one that looked smallest and was not. `resolve_split` requires a directory **and** an annotation file, and a YOLO split's annotations are a directory. Folding YOLO into the existing `CANDIDATES` would force that predicate to accept either kind of thing, and a root satisfying both conventions would then resolve by table order — handing a labels tree to the COCO reader, or an `instances_*.json` to this one. Which convention applies is a property of the reader asking, so the reader asks its own table.

What did not land: the datamodule still does not dispatch between the two readers, so a YOLO root is reachable from a library call and not yet from `lucid-yolo fit`. Recorded as 099b rather than absorbed into the row, because a package that reports itself done while a clause of its scope is unbuilt is how a roadmap stops describing the code.

Three details of the shipped surface, kept out of the roadmap row for length. The reader produces the same `Targets` container the COCO path does, which is the property that leaves assignment, loss and metric code untouched by a second dataset format entirely — the format question stops at the loader. Class names and split paths come from the dataset's own `data.yaml` rather than from a config, so a YOLO root carries its own label space. And a layout row was added alongside, so `--data.data_root` resolves a YOLO root the way it resolves every other, through `resolve_yolo_split` and `from_root`.

The clean-room instruction this was written under is AGENTS.md's prime directive, cited here because the register rows (A54-A58) record what was *unknown* and not what forbade looking it up.



## WP-099b — the tie-break that must not exist

<a id="wp-099b"></a>

`resolve_split` already had a precedence rule: candidate spellings are tried in table order and the first match wins. Reusing that shape across the two *readers* is the obvious move and it is wrong, for a reason worth stating in one line — **within a table the loser is another spelling of one reader, and between the tables the loser is a different label space, a different image set and a different parser.** A first-match rule in the first case picks the wrong directory of the right dataset; in the second it trains on annotations nobody named and reports a plausible number while doing it. So the probe evaluates both tables in full and returns a pair of booleans: there is no order for the answer to depend on, and ambiguity is refused rather than resolved (A63).

The refusal only works if the operator has a way to answer it, which is what earned the explicit `layout` override its place. Two states are otherwise unreachable: an ambiguous root on a read-only dataset mount, where the only remedy would be moving files; and a YOLO root whose `data.yaml` points its splits outside the convention table, which `from_root` resolves — it is why `val: ../valid/images` works at all — and which no directory probe can see. A knob that exists because two legitimate datasets are otherwise unreadable is a different thing from a knob that exists because one was possible.

The construction-versus-`setup` split was forced by something outside this package. Refusing at construction is the better instinct and it collides with `tests/ptl/test_cli.py`, which instantiates every shipped config through `LightningCLI(run=False)` while `configs/det_smoke.yaml` carries a placeholder `data_root`. A filesystem verdict at construction fails that dry-parse for every run not yet pointed at data. The line drawn instead: everything decidable from the arguments alone raises at construction, and anything needing the disk raises at `setup()` — still before an image is decoded and before the trainer takes a step.

Two absences the YOLO path has to declare rather than discover. `mask_targets` is refused, because the format carries no per-instance rings and rasterising empty ones would produce a segmentation run supervised by nothing that still reports a detection number. Copy-paste is the quieter one: unlike the oriented path it does **not** raise on polygon-free targets — it decodes an entire extra source sample, finds no candidates and does nothing — so at `copy_paste = 0.1` every YOLO run would have paid for an augmentation that can never fire. The suppression keeps consuming the RNG draw, which is what leaves the COCO path byte-identical and the frozen data checksums unmoved. An augmentation that fails silently costs more than one that raises.

One more property of the override, which is what keeps it from costing anything: naming any COCO-shaped path — `train_images`, `val_annotations`, any of them — is itself read as stating the COCO layout. Every caller that predates the probe is therefore already off it, and the probe runs only for a root that has said nothing about its own shape.



## WP-099c — a pre-flight that asks the run's question

<a id="wp-099c"></a>

`check_dataset` defaulted to `coco`, which was correct for as long as COCO was the only layout a run could read. [WP-099b](https://borda.github.io/lucid-YOLO/work_packages/wp-099b/) ended that, and left the check answering a question its own run no longer asks: pointed at a YOLO root with no flag, it reported a missing `train2017` for a tree `fit` trains on without complaint. The fix is not a third branch but a deletion — an unstated `--dataset` now runs `detect_layout`, the same probe the datamodule dispatches on, so the pre-flight and the run resolve the root identically or not at all. `DATASETS` is built from `DatasetLayout` members rather than string literals, which is what stops the flag and the probe drifting apart later. DOTA stays outside it deliberately: no run reads a DOTA root — it is tiled to COCO first ([WP-094](https://borda.github.io/lucid-YOLO/work_packages/wp-094/)) — so a probe covering it would advertise a path that does not exist, and the "satisfies no convention" report ends by naming `--dataset dota` for the operator who has one.

Both of the probe's undecidable states are verdicts about a disk, so both come back as report lines and exit 1. [WP-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/) already paid for that lesson once, on the first command an operator runs against a fresh provisioning: a traceback where a report belongs reads as a broken tool rather than an unusable dataset.

The extraction that mattered is `resolve_split_dirs`. A validator that resolves splits its own way validates a tree the reader will not open — and the YOLO layout has exactly the case that makes this bite, a `data.yaml` naming `../valid/images` for a directory no convention row spells (A58). Shared code between a feature and its checker is usually a smell, since the checker then inherits the feature's bugs; here it is the point, because the alternative is a check that passes on a different dataset than the one that trains. `scan_yolo_label_file` is the other half: the same grammar `load_yolo_targets` reads, stopping before the denormalization, so checking a split does not decode every image to obtain a pixel scale it immediately discards. Validation that costs an epoch is validation that gets skipped.

`DotaSplitCheck` became `PairedSplitCheck`, and the naming records what the two layouts actually share. It is not that both are "not COCO" — it is that both put one text file per image beside it, paired by stem, so both need the same two-directional pairing report and the same totals arithmetic. COCO's single manifest cannot be checked that way at all. The shape of the annotation decides the container, and the rename says so where a `Yolo`-prefixed twin would have said nothing.

No register row. A64 is still unconsumed, which is worth stating explicitly because the run made three decisions that look like assumption material and are not: the inference default is grounded in A63 and [WP-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/), the YOLO branch reuses A54–A58 through the reader rather than restating them, and the flag guard extends [WP-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/)'s own rule to one more layout. A register row for a decision that already has a documented parent is noise in the register that matters.

One absence left standing: `splits` is still not a CLI flag, for YOLO as it never was for DOTA, so a root shipping only `train` fails on the missing `val`. That is right for a `fit` pre-flight, which needs both, and wrong for anyone checking a partial download — a small follow-up rather than a defect, and recorded here so it is a choice rather than an oversight.



## WP-099d — the code is the specification

<a id="wp-099d"></a>

The gap this closed: `docs/DATASETS.md` said where COCO and DOTA come from and nothing about what either layout looks like on disk or what one annotation contains, while the YOLO layout the project had just started supporting appeared in it nowhere at all.

`docs/DATASETS.md` also carries a standing rule that a layout reference is the first real test of: the code is the specification and the document follows it. Applied here, that means every claim about a directory convention is cited to the module that enforces it — `data/layout.py`, `data/yolo.py`, `data/coco.py`, `data/dota.py` — rather than described from the outside from what the layouts are believed to be.

The alternative is not a worse document but a second specification. Two independent statements of a convention agree on the day they are written and disagree the first time a candidate table gains a row, and the one a reader trusts is the one that cannot be executed.



## WP-108 — a cell grows where its section is missing

<a id="wp-108"></a>

The Scope column had drifted to a 286-character median with a 1709-character worst case, and the drift is not random: of the 32 rows over 400, **12 had no RESEARCH_LOG section at all**. A row whose package never got a section is a row where the only place to write down why a thing was done is the row itself, so the reasoning lands there and stays. The length is the symptom; the missing section is the cause, which is why this package wrote the 12 sections before compressing anything.

The gate was considered and rejected, and the reasoning is worth keeping because it runs against this project's usual instinct. Every other convention here is enforced by a meta test — status icons, six columns, the header's package count, the register ids — and the argument for those is that a documented rule with no gate decays. A length gate is different in kind: the others fail on a fact about the file, while a length gate fails on a judgement about how much a row needs to say, and the cheapest way to make it pass is to delete a claim. A rule whose easiest satisfaction is losing substance should not be automatic. So the median is written into the header as guidance, and the enforcement is a reader noticing that a cell runs several times its column's median.

What the compression turned up is that most of the overflow was already recorded somewhere with an owner. Twenty rows needed no new writing at all: their long clauses restate what the log section, or the assumption register, already carries — A40's exact-under-similarity split, A59's core-ownership rule, A24's polygon measure. That is the duplication the header paragraph has always warned about, visible for the first time because something counted it. Three claims had no other home and were moved rather than cut, the sharpest being [WP-064](https://borda.github.io/lucid-YOLO/work_packages/wp-064/)'s: that the release discloses its per-tile figures in the report, the model card and the README. A disclosure claim living only in a roadmap cell is one edit from disappearing.

Two rows deliberately got no section: 091d and 091e were ⬜, and their content was a finding of 091c's, recorded there. A log section for unstarted work would be a stub asserting what the package will conclude, which is the opposite of what this file is for — so both rows pointed their `· log` tail at `#wp-091c` instead, and the sections got written when the packages produced something to record, which is the paragraph below this one.



## WP-099e — the gap was one layer above where it looked

<a id="wp-099e"></a>

[WP-099c](https://borda.github.io/lucid-YOLO/work_packages/wp-099c/) closed by naming this itself: `splits` was still not a CLI flag, so a root shipping only `train` fails on a missing `val`. That was recorded so it would be a choice rather than an oversight, and this is the follow-up. A third-party export whose validation split was never cut is a correct export, and the first command an operator runs against a fresh provisioning refused it over a directory nobody had promised.

What is worth writing down is where the parameter was actually missing. `check_yolo_root` and `check_dota_root` had both taken `splits` from the day each was written — `tests/data/test_check.py` had been calling `check_yolo_root(..., splits=("train",))` all along, because the fixtures only ever build one split. The capability existed; the route from the command line to it did not. So the fix is not a feature the layout checkers lacked but one argument `check_dataset` and `_check_root` dropped on the floor. A gap that presents as missing functionality is worth pricing against the layer below before implementing: the two root checkers were the expensive place to fix this, and the wrong one.

The default belongs to the layout rather than to the call. `_check_root` names `YOLO_SPLITS` or `DOTA_SPLITS` at the branch that already knows which layout it is dispatching to, instead of defaulting `splits` in `check_dataset`'s signature. The two constants are equal today and are documented for different reasons — DOTA's testing third has no public labels, YOLO's `test:` entry is a split no run of this project opens — and an entry-point default would have quietly made them one fact, so that changing either would have to be discovered rather than read.

COCO refuses the flag rather than defaulting it, on the reasoning that already governs `--expected_images` there: `check_coco_root` checks `train2017` and `val2017` against 118,287 and 5,000, and the names and the numbers are a single published fact. There is no subset of that to ask for. Following `_check_flags_apply`'s existing message shape was the whole of the design decision — a flag that silently does nothing makes the report read as though it had been honoured.

The empty tuple is the sharper case, because it inverts the failure mode the rest of the module guards. Every other bad argument here fails loudly; `splits=()` fails *cleanly* — no split checked means no problem found means `PASS: dataset layout valid` printed over a root nothing looked at, a false green from the one command whose purpose is to be believed. It is refused in `_require_splits`, called from `check_dataset` before the layout is even inferred and from both root checkers, because those two are documented as the importable core: guarding only the entry point would leave the false green reachable by the shorter route, and a library caller reading `ok` off the result has no report to notice the emptiness in. The entry-point call is what makes `--splits '[]'` fail at the command line, which is where an empty list is easiest to type by accident.

The annotation is `tuple[str, ...] | None`, and the CLI surface it produces was verified rather than assumed: jsonargparse renders it as a list literal, `--splits '[train]'`, which is the spelling `lucid-data download --splits` has always required — a bare `--splits train` is refused by both. One flag's syntax is not a second thing to learn. The `--help` line for it is less self-describing than `download`'s, the `| None` union suppressing the `[ITEM,...]` metavar; cosmetic, and not worth changing the annotation for.

