# 🔮 Phase 9 — Inference and generalization


## WP-093 — stable angle regression for elongated boxes

<a id="wp-093"></a>

The revised gain and its five-seed dose-response are A22. What lives here is the **diagnosis**, including the two mechanisms that were measured and ruled out — the part that would otherwise be lost, since a register records what was concluded and not what was eliminated.

Starting point: the oriented overfit gate cleared its 0.9 floor on **0 of 3 seeds at 100 epochs (mean 0.840)** and **2 of 3 at 200 (mean 0.910, spread *widened* to 0.148)**. A longer budget is not the answer, and the widening spread says the problem is instability rather than under-training.

Measured per class, the shortfall is carried entirely by the one class whose AP depends on orientation:

| class | behaviour |
| -- | -- |
| square, circle | symmetry-protected: 0.93–0.99 AP while carrying 40–49° mean angular residuals |
| triangle | near-square (w/h 1.15), tolerant |
| rectangle (w/h 2.00) | tracks angle precision directly — carries the whole shortfall |

Two hypotheses were tested and **rejected**:

- **A quarter-turn local minimum.** Rectangle carries the *least* angular-error mass beyond 60° of any class. The 90° mass sits in square, where it is a true symmetry and costs nothing. The distribution is bimodal rather than gradual, which is what suggested a local minimum in the first place.
- **Under-training.** Seed 0 *regressed* from 0.8959 to 0.8177 when the budget was doubled.

Cause, found by paired five-seed ablation: R1 Eq. 15's angle term itself, at the 1.0 gain this project had assumed.

Two things the record deliberately does not do. It does not lower the floor to fit the measurement. And it does not claim a clean fix: the revised gain reaches 4 of 5 seeds, not 5 of 5 — seed 1 lands at 0.8638 with 19.84° error. Only dropping R1's angle term entirely reached 5 of 5, and that is a paper deviation the register does not authorise, since A22 is a gap R1 leaves open while the term itself is not.

The residual instability sits at 2:1 aspect ratios, which is the band where `omega = 0.948` — a five percent reduction where R1's stated intent is that elongated boxes are "primarily constrained by the rotated IoU loss". At DOTA's 5:1 and 10:1 ratios `omega` genuinely falls (0.750, 0.555) and the term may behave as R1 intends. That remains unmeasured.



## WP-091 — the decode path that does not exist

<a id="wp-091"></a>

An oriented checkpoint can be read through one decode path, not two. The dense branch is not what is missing — an `obb` head builds both angle stems. What is missing is a **rotated suppression decoder**: running the axis-aligned `NMSDecoder` over columns that are a centre and two extents would suppress by upright overlap and keep or drop the wrong boxes with nothing in the output to show it had. `rotated_iou` exists in `eval/dota_eval.py` as the ingredient and no decode path uses it.

So `--decoder nms` raises rather than quietly running `e2e` instead. Substituting would be the worse failure of the two, because the report records the decoder it was *asked* for, and the file would then attest to a path that never ran. Recorded as 091b rather than fixed in passing: a suppression rule for rotated boxes is a new decode path with its own assumption to register, not a parameter. It landed as 091b directly afterwards, and the assumption is what the delay bought — see below.

The canonicalization the row named turned out to be already there, which is worth writing down because the row's phrasing invites the opposite reading. `decode_rboxes` ends in `canonicalize` on the dense `(B, A, 5)` output — A23 records exactly that — so a caller gets canonical angles from the existing decode. What this package had to establish is the other half of the guarantee: that canonical form *survives* the trip to original coordinates. A letterbox inverse is one isotropic scale plus a translation, so it scales both extents by a single positive number and turns no angle; `w >= h` and theta are invariant under it and nothing re-normalizes afterwards. Three places could host that normalization and two are wrong: in the head it would normalize an angle the loss trains against raw (R1 Eq. 13 emits the pre-activation), and after the inverse it would run once per entry point with each copy free to drift.

The test for it plants four raw angles that all describe one rectangle — already canonical, a half turn above the range, below the `-pi/4` floor, and the extents stated short edge first — and requires all four to come back as the same box, asserted after the inverse rather than before. A fifth plants 500 radians and asserts only the range and the long-edge order: computing which representative 500 folds to would restate `canonicalize` inside the test, and a restatement agrees with a broken implementation as readily as with a correct one.



## WP-074 — deferred, and what the search for a public YOLO corpus found

<a id="wp-074"></a>

Deferred on 2026-08-14 rather than built. `roboflow/rf100-vl` PR 29 is open and adds `combine_downloaded` / `download_and_combine` to the collection's own package — deterministic id remapping, category namespacing, a resumable manifest — which is this row's scope, written by the people who own the datasets. Building it here first means maintaining a second merge against a moving upstream and owning its correctness across 100 datasets; waiting costs a wrapper. A second route may retire the row outright: `probicheaux/rf100-vl` on Hugging Face is Apache-2.0 and ungated, 15 domains in COCO form, so the tier becomes a download plus a layout conversion with no API key involved.

The more useful finding came from asking the adjacent question — what public dataset could exercise the [WP-099](https://borda.github.io/lucid-YOLO/work_packages/wp-099/) YOLO reader at scale — and getting no good answer. The search is worth recording because the *shape* of the negative result explains [WP-099](https://borda.github.io/lucid-YOLO/work_packages/wp-099/)'s own central difficulty.

Rejected, each for a stated reason rather than a hunch: the `keremberke/*` mirrors are the largest set of public YOLOv8-format datasets on Hugging Face and state **no licence at all**, and unstated is not permissive. `CARD-Data/CARD-Germany-Batch1` is real-scale YOLO but is 315 GB, gated behind login plus terms acceptance, and its "CC BY 4.0" carries a no-military field-of-use rider — a licence with a use restriction bolted on is not the licence it names. `goodquestion1/RM26` is YOLO *pose*, whose extra keypoint fields the reader rejects by design. A Zenodo query for YOLO-format detection records under permissive terms returned nothing that was both an image dataset and detection.

What is left is the structural point. Public YOLO-format detection data at scale is almost entirely (a) Roboflow exports behind an API key and (b) Ultralytics-hosted mirrors, which this project's prime directive forbids reading. **The format has no neutral publisher, which is the same fact [WP-099](https://borda.github.io/lucid-YOLO/work_packages/wp-099/) met from the other side when it found no specification to write the reader from.** A convention owned by tooling vendors rather than by a standards body produces exactly this: universal adoption, no spec, and no corpus anyone can point at without accepting someone's terms. The consequence for this project is that the YOLO path's public story is a small verified example — the R32 export, CC BY 4.0, already on disk — plus documented instructions for bringing your own export, and not a headline dataset.

The design the row carried before it was deferred, kept here so a later attempt starts from it rather than from scratch. Per-dataset download, then category ids remapped into a **union label space** under dataset-qualified names — the collection spans domains, so bare category names collide across datasets and a collision is a silent label merge rather than an error — then images re-numbered, then appended into one merged COCO layout, with each per-dataset archive deleted before the next is fetched. That last step is what bounds peak disk to O(one dataset + merged) rather than O(all 100). Splits are preserved as published, and an `RF20-VL` subset flag exists so a smoke run does not pay for the whole collection. Two design points were deliberately left open rather than settled ahead of the code: whether evaluation reports over the union label space or per dataset, and how the verify module integrates.

Two citations for whoever picks this up: the collection is <https://github.com/roboflow/rf100-vl> and the upstream work is its PR 29, <https://github.com/roboflow/rf100-vl/pull/29>. Reconsider this row when that PR lands or is abandoned.

The licence question does **not** resolve with either route and is the thing most likely to kill the row. `rf100-vl` is Apache-2.0 **as tooling**; the 100 datasets it downloads carry their own licences, individually, and this project ships permissive-only. The Hugging Face alternative (`probicheaux/rf100-vl`, Apache-2.0, ungated, 15 domains, ~163k rows in COCO form) turns the tier into a download plus a parquet-to-COCO-layout conversion — no API key and no merge to own — but it does not answer what licence the underlying images carry.

**Dropped 2026-08-18.** The deferral above named the condition to watch (PR 29 landing or being abandoned) but not one this project controls or a date to stop waiting on — the licence question that named itself the row's likeliest killer never resolved either. Rows 074 and 075 close ⊘ rather than stay ⏸ indefinitely; the design and citations above are kept as-is for whoever reopens the tier, unedited by this note per this log's own convention of leaving a historical entry describing what was true when it was written.



## WP-091b — a threshold with nothing to cite

<a id="wp-091b"></a>

The interesting half is the threshold, and what it cost to find that there is nothing to cite for it — the decode rule itself (greedy class-wise suppression over `rotated_iou`) is described at [wp-091b.md#wp-091b-engineering](https://borda.github.io/lucid-YOLO/work_packages/wp-091b/#wp-091b-engineering). R1 states no suppression parameter at all — the one-to-one branch's freedom from NMS is the paper's claim, not a setting. R18's devkit scores detections it is handed and settles nothing on the decode side. R13 fixes the angle convention and no suppression rule. So the oriented comparison column has a knob no allowlisted source turns, and the register (A61) says exactly that in its source column rather than dressing a convention up as a citation.

What the value rests on instead is holding one variable still: 0.7 is carried over unchanged from the axis-aligned path, so the two comparison columns differ by the overlap *measure* alone and any gap between them is attributable to the geometry rather than to a second knob turned at the same time. The defence that was **rejected** is more interesting than the one taken. It would be natural to argue that 0.7 is safe because rotated IoU is the smaller measure, so carrying the threshold over can only suppress less. That is false: rotated IoU is usually smaller — mean 0.047 against 0.080 over 3000 random canonical pairs — but it *exceeds* the envelope IoU whenever both boxes lie along a shared diagonal, at 138 of those 3000 pairs and by up to +0.146. An argument that sounds like a bound and is only a tendency is worse than no argument, because it stops the next reader from checking.

The exposure is recorded with numbers rather than a hedge: 0.7 under-suppresses elongated objects, which is DOTA's characteristic shape. Two detections of one 10:1 object whose headings disagree by 5 degrees score 0.643 and both survive. That direction was accepted deliberately — a duplicate is inspectable in the output, where an over-suppressed object is simply absent — and the value is a constructor argument, so the alternative is one key away.

Looking for a lineage to inherit turned up that there was none. The axis-aligned `0.7` had shipped since WP-042 as a bare default with no register row, which means every "mAP (non-E2E)" number this project has published rested on an unregistered constant. It is now **A62**, registered at its shipped value rather than revised: re-picking it would silently move a comparison column that already has committed figures beside it, and that is a 0.MINOR change with goldens, not a tidy-up. The general shape: an assumption that goes looking for its own parentage is a good way to find the rows nobody wrote.



## WP-091c — what a refactor is allowed to notice

<a id="wp-091c"></a>

The move itself is unremarkable: `rotated_iou` and the seven private helpers it exclusively owns leave `eval/dota_eval.py` for `data/rotated_geom.py`, since an evaluation module had no business being on the decode path's import graph for a function that is pure geometry. What the package is worth recording for is the three things it declined to do.

**It did not merge the duplicate it found.** `rotated_geom` already had `_check_2d`, differing from the arriving `_check_rboxes` only in the letter naming the row count — and the two raise `must be (N, 5)` and `must be (K, 5)`, which two different test files match on. Merging them is therefore a test-visible behaviour change wearing the costume of a duplicate removal, and a refactor whose contract is "every existing test passes unchanged" cannot also change what an error says. It is recorded as 091d instead. A duplicate that is cheap to see is not always cheap to remove, and the difference is whether anything asserts on it.

The search also turned up `_check_rboxes` in `losses/probiou.py`, first reported as a third copy and corrected on re-reading: it guards `ndim == 0 or shape[-1] != 5` and raises `must be (..., 5)`, so it accepts arbitrary leading dimensions where both `rotated_geom` validators require exactly 2-D. Same name, same five, different contract — merging it into the other two would tighten what the loss path accepts, which is a behaviour change and not a rename. It is 091e, and its test coverage differs in the same direction: `tests/losses/test_probiou.py` matches the prefix `target must be` only, leaving the tail unpinned, where both `rotated_geom` messages are pinned tail and all. Three lookalikes, two of them the same check — the way to tell was to read each guard rather than each name.

**It did not deprecate the old name.** `pyDeprecate` is nowhere in this tree, and `warnings.warn` would have fired on both existing suites — which is not "tests pass unchanged" under any honest reading, and arms a trap for the day `filterwarnings = error` is set. The deeper reason is that `dota_eval.rotated_iou` is not a legacy alias: that module's docstring defines R18's protocol *in terms of* its overlap measure, so a caller reproducing the protocol wants the kernel where the protocol is. One function visible from two places is what `canonicalize` already is.

**It moved the argument, not just the code.** The "why there is no float64 here" section and its five-row error table went with the kernel, because that table is the *evidence* for the shift-before-expand ordering rather than commentary about it; a precision argument left behind in the module that no longer owns the arithmetic is how a constraint quietly stops being checked. Two claims in the destination's own docstring also had to change or become false — "no Python loop over boxes" and a blanket "float32 tensors" — which is the ordinary tax of moving code into prose written before it arrived.

The evidence that the move was inert is the goldens and only the goldens. `rotated_iou` sits on the scoring path, and a silently perturbed kernel would still clear the shapely oracle's 1e-4 tolerance; 20/20 unmoved, frozen subtrees included, is what says the arithmetic is bit-identical.

[WP-091b](https://borda.github.io/lucid-YOLO/work_packages/wp-091b/) had already verified there was no import cycle, from four entry orders, and left the module where it was rather than widening its own diff — which is why this is a package and not a line in that one.



## WP-107 — whole-image merge

<a id="wp-107"></a>

Recorded before the work, because the reason this is a package rather than a step inside another one is itself the finding: two overlapping 1024 px tiles that both detect one object produce two detections at full confidence, and an NMS-free path has **no suppression stage to remove the duplicate**. The merge is a policy that has to be decided and argued, which is precisely why it kept failing to happen as a side effect — [WP-063](https://borda.github.io/lucid-YOLO/work_packages/wp-063/) deferred it into [WP-088](https://borda.github.io/lucid-YOLO/work_packages/wp-088/), whose scope never took it up, and [WP-064](https://borda.github.io/lucid-YOLO/work_packages/wp-064/) shipped 0.3.0 without it.

What the work established is that the duplicate rule does not have to be a suppression rule. Core ownership (A59) keeps or drops a detection by **where it is**: the cores are built from the recorded windows before the model runs, nothing is compared against another detection or ranked by confidence, and a detection is dropped whether or not the neighbouring tile found anything. Run the model twice with different weights and the same detection is owned by the same tile. A confidence-ranked dedup by rotated IoU is the defensible alternative and would score *higher*, because it takes the union of the tiles' recalls rather than the owner's — it was rejected because the figure is meant to measure a suppression-free pipeline, and reintroducing suppression at the seam makes it measure something else. That rejection is the package's actual content; the code is the cheap part.

The cost is real and one-directional: whole-image recall is the owner's recall, so this merge can only *lower* a per-tile number. It is pinned by a test rather than left as prose, so that a later reader who finds the drop surprising cannot quietly fix it into NMS.

One residual runs the other way and was nearly missed. The first draft of the module claimed the difficult straggler "cannot inflate the score". That is wrong in exactly the direction this package exists to police: a detection landing on a surviving clipped copy is *discarded* under A48, where against the true whole-image annotation set it would have been a false positive, and discarding false positives raises precision. It is bounded to objects wider than the overlap and it is R18's own devkit behaviour rather than an invention here — but it is not zero, and it is now measured at **0.165 of `map_50`** on the unit fixture rather than argued away. A claim of the form "cannot" about one's own instrument deserves a measurement before it is written down.

The DoD's equality clause needed one guard the row did not ask for. A single-tile image must score identically through both paths, and an equality between two saturated `1.0`s would prove nothing at all — so the fixture is built to land every one of the four metric keys strictly inside `(0, 1)`, asserted separately, and the equality is then asserted with no tolerance.

Two smaller things the merge inherits rather than solves. `--limit` can end a tile prefix mid-source-image, so an incomplete trailing image is dropped rather than scored partially — a whole-image figure under `--limit N` therefore covers at most N tiles' worth of *complete* images. And A52's `--drop-empty-tiles` moves the core midpoints when windows are missing from the layout: still a partition, but the `v/2` circumradius guarantee weakens, which is a property of that off-protocol layout rather than of this rule.



## WP-111 — the whole-image figure runs higher, not lower, and both reasons are already on the register

<a id="wp-111"></a>

[WP-107](https://borda.github.io/lucid-YOLO/work_packages/wp-107/) built the merge and its own log entry above is explicit: "whole-image recall is the owner's recall, so this merge can only *lower* a per-tile number", pinned by a fixture test. The first real run — the accepted OBB-smoke checkpoint (`version_10`, 2026-08-14) re-scored on DOTA-v1.0 val, 458 source images — comes out the other way on every one of the four metrics: mAP50-95 0.2914 → 0.3146, mAP50 0.5242 → 0.5484, mAP75 0.2770 → 0.3111, mAR_300 0.5075 → 0.5193. That is not the merge mechanism breaking its own guarantee; it is a second, separately registered effect outrunning it.

[WP-107](https://borda.github.io/lucid-YOLO/work_packages/wp-107/)'s claim is about core ownership in isolation — on a fixture too small to ever reach a detection cap, dropping every non-owning tile's copy of an object can only cost recall. The real pipeline runs the merge alongside A60: per-tile scoring caps each 1024 px tile at 300 detections (`MAX_DETECTIONS`, A47 — what `o2o_rotated_topk` emits per forward pass), while the whole-image pass scores with that cap lifted, because a source image assembled from ~22 tiles (10,132 tiles / 458 images) is not one forward pass and re-imposing 300 on it would measure a truncation rather than the model — exactly the exposure A60's own text names ("a dense DOTA image carrying far more than 300 instances"). This is the first run that actually exercises that lifted cap against real data, and the direction is consistent with what A60 predicted: recall recovered by the uncapped pass outweighs what core ownership gives up.

The two effects run in the same pass here and are not separated by this measurement — a controlled comparison (whole-image scoring with the 300 cap re-imposed, versus lifted) would isolate how much of the `+0.012` mAR_300 is the cap and how much is anything else, and nothing in this run does that. Recorded as consistent with A60, not as proof of it.



## WP-091d — one letter, and what it costs to change one

<a id="wp-091d"></a>

091c declined this merge and said why: the two validators' messages are matched by two test files, so removing the duplicate changes what an error says, and a refactor contracted to leave every existing test passing cannot also do that. Done as its own package the visibility is the point rather than the obstacle. `_check_rboxes` is deleted, `rotated_iou` guards through `_check_2d`, and `tests/eval/test_dota_eval.py`'s `match=r"must be \(K, 5\)"` becomes `r"boxes_a must be \(N, 5\)"` — a *stricter* regex than the one it replaces, since it now pins the argument name as well as the shape. The other pinned message, in `tests/data/test_rotated_geom.py`, never moved: it was already the surviving spelling.

The letter itself was the only real question. `rotated_iou`'s signature does distinguish the two operands — `boxes_a` is `(M, 5)`, `boxes_b` is `(N, 5)`, and the result is `(M, N)`, so the distinction is load-bearing in the docstring. It is not load-bearing in the guard: each operand has to satisfy one and the same predicate, and a rejection message that spelled one of them `K` named a difference the check never tested. `N` reads as "some row count" in every message the module raises, which is what a shape rejection is actually saying.

What made the merge safe is narrow and worth stating, because the neighbouring lookalike fails it. The two conditions were character-identical — `ndim != 2 or shape[1] != 5` — so nothing but the string changed. `losses/probiou.py`'s guard passes arbitrary leading dimensions, and folding that one in would tighten a live contract rather than dedupe a message; that is 091e, and it is a different package for that reason and not for tidiness.



## WP-091e — a contract nothing in the tree exercises

<a id="wp-091e"></a>

The row asked which way to settle `losses/probiou.py`'s looser guard, and named the evidence it wanted: what the loss is actually called with. That evidence is one line. `oriented_loss.py:217` calls the resolved form on `(P, 5)` against `(P, 5)` and gets `(P,)` — aligned pairs, already matched by the assigner. **No caller in `src/` ever passes a leading dimension.** The pairwise `(4, 1, 5)` against `(1, 3, 5)` shape appears in `tests/losses/test_probiou.py` and nowhere else, and the one place in the tree that does want a pairwise rotated-overlap matrix — `decode/rotated_nms.py`, and the DOTA evaluator behind it — reaches for `rotated_geom.rotated_iou` instead, because it needs the polygon-exact measure and not R17's Gaussian surrogate.

So the honest reading is that the N-d contract is a property of the *public* function rather than of any in-tree use, and the decision turns on whether that is a reason to drop it. It is not. Everything in these two losses is elementwise in the leading shape — `_unpack` unbinds the last dimension and every term after it is arithmetic on those five tensors — so the broadcast is not a feature that was added, it is what the code does when the guard stops refusing. Narrowing to 2-D would be the validator inventing a restriction the arithmetic does not have, and it would take `probabilistic_iou`, an exported symbol whose docstring promises `(..., 5)`, with it. A guard should assert what the function needs, and this one needs the trailing five.

What the package changes is therefore not behaviour but what is asserted. The test matched `"target must be"` — the prefix only, leaving the shape the message names unpinned, so a future edit could have narrowed the promise from `(..., 5)` to `(N, 5)` with every test still green. It now matches the tail as well, and a new case runs a `(2, 3, 4, 5)` input through both losses and then through `canonicalize`, which rejects it: the divergence between the two validators is stated in one place, by a test that fails if either side of it moves. That is the answer to 091c's third finding — three lookalikes, two of them the same check, and now both facts are pinned rather than remembered.

