# Changelog

All notable changes to lucid-yolo are documented here, following the Keep a Changelog convention; versioning is a perpetual 0.x release train — no 1.0 is ever planned, promised, or tagged — per ADR-002 (docs/DECISIONS.md).

## [0.4.0] - 2026-08-15

### Added

- Single-image inference for all three tasks, as a library call and as the fourth console script `lucid-predict`: `predict_image` for detections, `predict_segmentation` for instance masks and `predict_oriented` for rotated boxes, each answering in the original image's coordinates through the letterbox inverse rather than in canvas coordinates. The three refuse each other by name — `forward` returns a `DualHeadOutput` for every task, so a segmentation checkpoint handed to the detection path produced boxes with its mask branch never consulted — and `--ema`, `--device` and `--output` keep `lucid-eval`'s spellings, so the two commands cannot disagree about what a flag means (WP-089, WP-090, WP-091).

- `RotatedNMSDecoder`: the oriented tier now has the suppression baseline the axis-aligned tier has reported since 0.1, greedy and class-wise over the one-to-many branch by the exact polygon `rotated_iou`, on canonicalized boxes. `predict_oriented(decoder="nms")` decodes where it used to raise, on an unchanged signature — the refusal WP-091 shipped is lifted rather than worked around (WP-091b, A24).

- `merge_whole_images`: tile detections are un-letterboxed, translated by the window origin recorded at tiling time and scored against whole-image ground truth reassembled from the tiles' own annotations, with seam duplicates resolved by core ownership rather than by suppression — an NMS-free path has no suppression stage in which to drop the duplicate. Every oriented figure this project published before this was a per-tile figure, and a per-tile score never pays the duplicate-detection cost a whole-image score charges (WP-107, A53, A59).

- A YOLO-format dataset reader: a `data.yaml` plus a `labels/<split>/*.txt` tree of normalized rows produces the same `Targets` container the COCO path does, so assignment, loss and metric code is untouched. Oriented rows arrive as eight normalized polygon coordinates and are canonicalized to the long-edge convention on load (WP-099, A54–A56).

- `DetectionDataModule` dispatches on layout: a YOLO root trains through `lucid-yolo fit` and not only through a library call, the reader chosen by which layout `data_root` actually satisfies rather than by a flag the operator has to remember (WP-099b).

- The pre-run check learns the YOLO tree, and an unstated `--dataset` probes instead of defaulting to COCO: pointed at a YOLO root with no flag it used to report a missing `train2017` for a tree `fit` trains on without complaint. The probe is `detect_layout`, the same one the datamodule dispatches on, so the pre-flight resolves a root exactly as the run does or fails the same way (WP-099c).

- `check_dataset(splits=...)`, threaded into both root checkers. A third-party export shipping only `train` — a correct export whose validation split was never cut — failed a pre-flight on a `val` nobody had promised. An empty `splits` is refused rather than honoured, since checking no split would report a pass over a root nothing had been read from (WP-099e).

- An export gate on the deployed one-to-one graph, all three heads: the exported ONNX carries no `NonMaxSuppression` node and does carry the `TopK` that stands in for suppression, at the static output shape `TopKDecoder` documents, and run under onnxruntime it reproduces the checkpoint-loaded module's own decode — class ids exactly, boxes to 4.6e-05. Nothing ships: `onnx` and `onnxruntime` are dev-group only, the export itself being `torch.onnx.export` (WP-066, A9, A23, A30, A37, A45).

- A third surface on the licence audit: the files each wheel's `RECORD` declares it installed, matched by filename against a named table of copyleft native libraries. A wheel can declare a permissive licence, ship licence documents that name no copyleft anywhere, and vendor a GPL binary regardless — `av`, which `supervision` hard-requires, ships `libx264` under BSD-3-Clause metadata, and both older checks pass it clean. The check narrows that hole rather than closing it, and says so: a library the table does not name passes exactly as `av` did (WP-109, D16).

- `scripts/draw_predictions.py`, the release's worked example: a checkpoint and an image to a figure showing what the model answered — boxes for a `detect` checkpoint, boxes and per-instance mask overlays for a `segment` one, rotated quadrilaterals for an `obb` one rather than their upright envelopes, which are a different rectangle from the one the model reported. The drawing functions take an already-computed prediction rather than a checkpoint, which is what makes the geometry assertable in a repository that ships no trained weights (D14). `matplotlib` stays a `dev` dependency: nothing under `src/lucid_yolo/` imports it, so a wheel a consumer installs pulls no plotting stack. The roadmap row named `supervision`, which is refused — it hard-requires `av`, whose wheel ships a GPL `libx264` under BSD-3-Clause metadata (WP-067, D16).

### Changed

- `rotated_iou` and the seven private helpers it exclusively owns move from `eval/dota_eval.py` to `data/rotated_geom.py`. WP-091b had put an evaluation module on the decode path's import graph for a function that is pure geometry with no evaluation content (WP-091c).

- One rotated-box shape check where `rotated_geom.py` carried two. Their conditions were character-identical and differed only in the letter naming the row count: `rotated_iou`'s signature distinguishes `(M, 5)` from `(N, 5)` because the result is `(M, N)`, but the predicate each operand satisfies is the same one, so a second letter in the rejection named a difference the check never tested (WP-091d).

- `losses/probiou.py`'s own shape guard is pinned as a contract rather than folded into that one: it accepts arbitrary leading dimensions where `rotated_geom` requires exactly 2-D, which is deliberate — the loss is elementwise in its leading shape and its in-tree caller passes `(P, 5)` from the assignment (WP-091e, A19).

### Removed

- `lucid-download` is removed, as 0.3.0 said it would be: `lucid-data download` is the only spelling. The alias took dashed flags (`--data-root`, `--splits train val`, a bare `--force`) where the replacement's are underscored and list-valued (`--data_root`, `--splits '[train,val]'`, `--force true`), so a copy-pasted 0.2.x command needs re-spelling rather than renaming. `python -m lucid_yolo.data.download` goes with it, having run the same parser; `python -m lucid_yolo.cli.data download` is the module-invocation equivalent. Every repair hint and docstring fragment that spelled the alias now names a command that parses — including one that used its flag grammar without naming it, which no grep for the command could have found — and the hint is asserted by re-parsing what it prints rather than by matching its text (WP-110).

### Fixed

- `decode_instance_masks` on an empty instance axis: `F.interpolate` treats that axis as channels and rejects a zero-length one outright, so an image with no kept detections raised a `RuntimeError` where it should have returned nothing. It returns the empty stack now — the case an image with no detections reaches on the ordinary path (WP-090b).

### Documentation

- The reproduction report gains a consolidated note reading the detection, segmentation and oriented sections against each other. It appends rather than merges, because D10 makes the report append-only and merging three sections would rewrite three records three human gates signed off on; it says in its own first line that it is not a fourth tier. What the consolidation produces is an absence: the NMS-free deploy path's cost against NMS is the one quantity all three tiers could have reported in the same units, and the oriented tier never evaluated an NMS baseline at all (WP-065).

- `docs/DATASETS.md` gains a reference section putting the COCO and YOLO trees side by side, with every citation naming the enclosing symbol instead of a line range — line numbers rotted three times while that one section was being written, two of them shipped, and the values were right each time with only the references wrong (WP-099d).

- The roadmap and the research log are two files rather than one column: Scope states what a package does, and the measurements, rejected approaches and negative results it taught move to `docs/RESEARCH_LOG.md`, linked per row. Thirty-two cells were then compressed toward the table's own median, a 286-character median against a 1709-character worst case (WP-003, WP-108).

- A `⏸` status for a row waiting on something outside this repository, distinct from `⬜`: rows 074 and 075 are deferred pending an upstream `rf100-vl` merge, and a queue that spells "not started" and "not startable from here" the same way loses the difference exactly when someone picks the next row (WP-074, WP-075).

- The README is rewritten as a poster rather than a manual: why the project exists and what it does not claim, four audiences each with a start-here link, the architecture as a diagram, the reproduced numbers per tier with their training-curve figures, and the six top-level decisions argued with evidence and cross-linked to the registers that hold it. Every top-level and second-level heading across the README and `docs/` then gained one topical emoji, and the four README sections other sections link to gained explicit anchor tags — a decorated heading no longer answers to the slug its text used to derive, so the links would have gone dead in the same commit that made the page readable (WP-113).

- `scripts/absolutize_readme.py`, for the build whose artifacts go to PyPI: `README.md` is the long description, and PyPI resolves its relative targets against `pypi.org`, so the three training-curve figures render as broken images and every `docs/…` link 404s. Figures are rewritten through `raw.githubusercontent.com` and documents through `github.com/.../blob`, both pinned to the tag being released — a branch is refused, since a released page's links would then describe whatever that branch holds when a reader clicks them. Opt-in per invocation, so an ordinary build ships the file exactly as committed; `make dist-pypi TAG=v0.M.P` reverts afterwards and `release.yml` rewrites ahead of `uv build`. Reading the slug from the declared homepage exposed a stale one: the repository was renamed, GitHub redirects HTML URLs but `raw.githubusercontent.com` does not, so precisely the figures would have 404'd while the links beside them worked (WP-113b).

- A MkDocs Material site over the existing `docs/` tree, published to GitHub Pages on every push to the default branch and built with `--strict` on every pull request. It owns no prose beyond `docs/index.md`, a front page that routes a reader to the register answering their question; every register stays a plain markdown file readable on GitHub without the toolchain. The docs dependency group sits outside `dev` and outside `make setup` — no gate imports it — so the workflow that installs it is the only place CI sees that tree, and runs the licence audit there. `mdformat` and `check-yaml` each became two hook instances: one for the tree, one for the dialect `docs/` and `mkdocs.yml` are now read in (WP-114).

- `mkdocs` is capped at `<2`, a licence bound rather than a compatibility one: the Material team's own build-time notice describes MkDocs 2.0 as "Currently unlicensed", and unlicensed is stricter than the AGPL this project bans, since the default is no grant at all. The licence audit cannot report it — it matches a GPL-family pattern against a declared licence, so a distribution declaring nothing passes in the same run that calls the environment clean (WP-114b).

## [0.3.0] - 2026-08-14

### Added

- Long-edge rotated-box primitives: canonicalization to `w >= h` with the angle on `[-45, 135)` degrees, quadrilateral conversion in both directions, and a vectorized point-in-rotated-rect test. Three conventions the paper leaves open are fixed here and inherited by the whole oriented path — the angle turns `+x` towards `+y`, containment is edge-inclusive, and an exact square folds towards zero (WP-055).

- DOTA-v1.0 label parsing: the 15 categories in the published order, metadata headers skipped, category names normalized across their hyphenated and underscored spellings, and each annotated quadrilateral converted to a canonical long-edge box. Rotated boxes are kept on the same instance axis as the axis-aligned envelopes and labels, so one mask filters every modality (WP-056).

- `make check-data DATASET=dota` — DOTA root layout, image/label pairing and published-count validation beside the existing COCO path (WP-056).

- Overlapping 1024 px crop tiling for aerial imagery, with the source paper's partial-object rule: an instance clipped to under 70% of its area is flagged difficult rather than dropped, and re-fitted to a long-edge box. Crop overlap is a parameter; the visible fraction is carried alongside each instance (WP-057).

- Rotated-aware augmentation: random affine, mosaic and mixup now carry rotated boxes instead of refusing them, warping each box through its four corners and re-fitting a long-edge box — exact under a similarity, an explicit fit under shear (WP-058).

- ProbIoU rotated-box loss: both forms the source paper proposes, the bounded Hellinger distance and the unbounded Bhattacharyya distance, evaluated in a cancellation-free form that holds float32 accuracy from square boxes out to 1000:1 elongation and stays finite on degenerate input (WP-059).

- Square-object angle loss: the auxiliary double-angle term that resolves what the rotated IoU loss cannot, weighted towards near-square targets by a log-Gaussian in the aspect ratio and zero at every quarter turn, so a square is never penalized for choosing either of its two indistinguishable orientations. The two representatives a square can arrive as score bit-identically, not merely to tolerance (WP-060).

- Rotated candidate selection in the Task-Aligned Assigner and its STAL subclass: an oriented ground truth now decides candidacy by point-in-rotated-rect containment instead of by its axis-aligned envelope, through one optional argument that leaves the accepted axis-aligned path bit-identical when omitted. The end-to-end one-to-one assigner inherits it with no code of its own (WP-061).

- Oriented detection head and NMS-free rotated decode: a third stem on each head branch predicts the angle directly, with the squashing nonlinearity of the previous versions removed, and the decode turns the ordinary box regression about its own centre and normalizes to the long-edge convention. An angle of zero reproduces the axis-aligned box exactly, and the two raw angles a half turn apart decode to the same box rather than two. Enabling the branch leaves the shipped detection head bit-identical, verified across 684 parameter and forward digests (WP-062).

- Rotated mAP50-95 for oriented detection: exact polygon-intersection IoU against a shapely oracle, the COCO threshold grid and 101-point interpolation, and the DOTA convention where an instance flagged difficult is neither credited nor penalized. The detection cap follows what the model emits rather than COCO's smaller default, so no part of the output is discarded and then counted as missed (WP-063).

- The license audit now reads the license documents a wheel bundles, not only the fields it declares. Its first run found two runtime libraries vendored under copyleft terms that no previous audit could see; a recognized license exception passes for any package, and the two remaining cases are allowlisted with the reasoning recorded (WP-063).

- Oriented supervision wired into the training step: `task="obb"` now trains the angle stems it had only been constructing. The rotated ProbIoU replaces the Complete-IoU term in its slot, the L1 term is retargeted onto the rotated box's own centre and extents rather than the axis-aligned envelope it no longer matches, and R1's double-angle term is added — all gathered from the one assignment the box terms were scored against, never a second one. A `task="detect"` step stays bit-identical to a snapshot replayed from the pre-change tree (WP-088, A49-A51).

- A `difficult` channel on the target container, so R18's per-instance flag survives the loader transport into the rotated metric instead of being filtered away at load, which would have scored every ignorable detection as a false positive (WP-088, A51).

- `python scripts/build_dota_tiles.py` turns a DOTA root into the tiled COCO layout the oriented recipe trains on. The tiling geometry had existed since WP-057 with no caller, so the recipe pointed at a directory nothing could produce. Object-free crops are kept by default, since they are the negative evidence a one-to-one branch needs, and a flag selects the other reading; the per-instance difficult flag, the source paper's visible-area fraction, and each tile's source image and window travel in the annotation file so whole-image evaluation can be built later without re-tiling (WP-094, A52-A53).

- The COCO reader forwards a `difficult` key onto the target container's flag channel on both the oriented and axis-aligned readings, defaulting to false when absent. Tiling *creates* difficult instances that appear in no label file, and without this every one of them reached the metric as an ordinary scored ground truth (WP-094, A53).

- `python scripts/eval_obb.py` scores an oriented checkpoint against a split of the tiled layout through the rotated accumulator, with the EMA weights and the exact recall grid, the emitted detection cap and the difficult rule that instrument brings. The figure is explicitly per tile and says so: an object crossing a tile boundary is counted in both, and the merge rule that would fix it is a decision the release work package owns rather than something an instrument may pick quietly (WP-095).

- A gate on the assumption register's own shape: a row written one column short is padded back by the formatter and reads as though a sourced assumption were unsourced, which two rows had already done (WP-061).

- `docs/DATASETS.md` and `docs/TRAINING.md`: where each dataset comes from and what has to be on disk, and the launch command for each of the three tiers. DOTA provisioning is manual because its distribution offers interactive Drive folders rather than archive URLs, and that had never been written down; the training guide leads with the rule the report's historical command blocks do not state, that `default_root_dir` is where both loggers and the checkpoints write and therefore the whole of what survives a disconnected runtime (WP-003).

### Changed

- One command surface, and all of it shipped. `lucid-data` covers `download`, `check` and `build-tiles`; `lucid-eval` scores a checkpoint on the protocol its own task names, with no task flag and with the letterbox side and batch size defaulted from the same place; `lucid-yolo` is unchanged. Two of these had lived under `scripts/`, which the wheel does not ship, so a remote tier run could install the package and still not reach its own data build or its acceptance figure. Parsing moves to jsonargparse — already there underneath the training CLI — so flags and help text come from the operation functions' signatures and docstrings instead of a hand-written parser, and every command takes `--config`. Flags are underscored as a result (`--data_root`), matching `--data.batch_size` (WP-096).
- `lucid-download` is deprecated in favour of `lucid-data download` and will be removed in 0.4.0. It keeps working, and keeps its original dashed flags, because published reproduction instructions name it (WP-096).
- Markdown formatted by mdformat at commit time, unwrapped and with tables left unpadded, so prose reflows in the editor rather than in the diff (WP-001).

### Fixed

- Validation no longer leaves the accelerator idle. The val loader's worker cap applied even to a count the operator had named on the command line; it now caps only what the datamodule chose for itself, and that ceiling scales with the letterbox side instead of sitting at a constant reasoned about 640 px samples. An oriented run also stops accumulating the axis-aligned `val/mAP` beside the rotated one — a CPU pass over every detection of every batch, producing a figure that reads the pre-rotation rectangle and so cannot tell right orientations from random ones (WP-102).
- `lucid-data build-tiles --suffix .jpg` writes JPEG tiles, with `--quality` to choose. The loader decodes every tile once per epoch and mosaic assembles four source tiles per sample, so a 64-image step at 1024 px is around 250 decodes and PNG is where a smoke run starves. PNG stays the default — a tile is an exact crop of a lossless source — and the annotations are identical either way. The progress bar now counts completions rather than positions, so one oversized source image no longer freezes it while smaller ones finish behind it (WP-101).
- `lucid-data build-tiles` runs across processes and shows a progress bar. A full DOTA build is 1,869 multi-megapixel decodes and it ran silently on one core, so an operator could not distinguish a working build from a hung one while the rest of the machine idled. Work is dispatched per source image, one worker per CPU by default and `--workers` to choose, each pinned to a single torch thread; `--progress false` turns the bar off. The output is byte-identical to a serial build, because ids are assigned by the parent in source order and never inside a worker (WP-100).
- A dataset root's split paths are resolved by convention instead of defaulting to COCO 2017's spelling. `train2017` and `annotations/instances_train2017.json` were hardcoded defaults, so every other COCO-format root — including the tiled oriented layout this project writes itself — reached the reader only by restating all four paths on the command line. Four candidate layouts are now tried in order and the first whose directory and annotation file both exist wins, so `--data.data_root` is sufficient for every tier; explicit per-split overrides still take precedence, and the oriented config drops the placeholders it used to ship (WP-098).
- `lucid-data check --dataset dota` no longer fails on a correct download. It compared the summed `train` and `val` counts against the published 2,806 images and 188,282 instances by default, and those describe DOTA-v1.0 whole: the source paper takes half the images as training, a sixth as validation and a third as testing, and releases ground truth for the first two only, so an annotated root holds about two thirds of them and can never sum to them. The totals are now reported on a `NOTE` line and compared only where the caller states one, with `--expected_images`, `--expected_instances` and `--expected_classes` to assert them once a provisioning's counts are known; the layout, pairing and parse checks are unchanged, and an expectation aimed at another layout raises rather than being ignored (WP-097).
- The angle term's weight is now measured rather than assumed: `0.25`, down from the `1.0` this project had picked for a gain R1 never states (A22). At `1.0` the term did not merely over-weight the objective, it destabilised angle regression on elongated targets — across five seeds the oriented overfit cleared its floor once, with a run-to-run spread of 0.667 and a worst run at 0.31; at `0.25` it clears four times with a spread of 0.096, and the mean angular error on elongated targets halves. The cause is visible in the term itself: `sin²(2Δθ)` is zero at both zero and a quarter turn, so for an elongated box a 90-degree error scores as a perfect answer and only the rotated IoU term objects (WP-093, A22).
- A gate on the roadmap's own table shape: an unescaped pipe inside a code span opens a cell, so a row renders its tail into the wrong columns and drops the overflow. Two rows had already shipped that way, and neither existing gate could see it — one reads only the first column, the other matches the last (WP-093).
- Exact recall sampling for the detection instrument: `evaluate_bbox` now supplies the 101-point recall grid as correctly rounded hundredths instead of accepting the metric's float32 default, which overshot `k/100` at 36 of the 101 indices. A class whose recall landed exactly on one of those boundaries forfeited that point and `1/101` of its average precision, always downward — an ordinary case rather than an exotic one, since a class with 5, 10, 20, 25, 50 or 100 ground truths lands on a grid point at every recall it can attain. Both mAP instruments now forfeit no boundary, and agree exactly where they used to differ (WP-092, A46).
- The validation worker count fits the host again. WP-102 let a named `--data.num_workers` reach the val loader and removed the guard rather than the silence: at 1024 px a batch of 64 stacked images is 805 MB, so a named 32 queued around 51 GB for validation alone, beside a training pool still resident at the epoch boundary, and the host died. A count inherited from a named training count now passes the shared-memory budget, measured when the val loader is built so the training queue is already counted, and says so with the number and the override flag. A count stated as `--data.val_num_workers` is capped by nothing, which is where an operator asks for the whole machine (WP-103).
- The oriented epoch boundary no longer stalls for a minute after validation reaches 100%. Scoring 10,000 tiles at 300 detections each measured 178 s, 58% of it in the matcher: the ten IoU thresholds share the overlap matrix and the score order and differ only in what counts as a hit, so ten passes re-walked the same detections to consume different subsets of the same ground truths. The availability state is now carried per threshold and the walk happens once, and detections that reach no ground truth at the lowest threshold skip it entirely — they cannot score, cannot be discarded and consume nothing at any threshold, which on an early-epoch model is nearly the whole loop. 178 s to 38.6 s, with every pinned value unchanged (WP-104).
- `lucid-eval --output` creates its parent directory instead of raising after the scoring pass has finished and printed its numbers, which turned minutes of accelerator work into a traceback. Both evaluation paths also draw a progress bar now: the length is known ahead of time, and four minutes of silence is indistinguishable from a hang (WP-105).
- Horizontal flip left rotated boxes outside the long-edge angle range: it negated the angle without re-wrapping, so any box past 45 degrees came out non-canonical (WP-058).

### Documentation

- The oriented tier's record, matching what the detection and segmentation releases carry: `docs/model_cards/obb.md`, the four-panel `docs/figures/obb_smoke_training.svg`, and the reproduction report's 0.3.0 section. The plotting script grows the panels an oriented run needs — such a run logs no `val/mAP` at all, so the headline panel falls back to the rotated pair and the fourth draws the terms that replaced the box pair — and the committed detection and segmentation figures are proved unchanged by the edit (WP-106).
- The three model cards move into `docs/model_cards/`, as `detection.md`, `segmentation.md` and `obb.md`. A meta test holds the directory and the required-file listing against each other in both directions, so a fourth card cannot arrive ungated and a listed one cannot vanish (WP-106).
- A roadmap row carried an unescaped pipe inside a code span, which opens a table cell: the row had eight cells against a six-column header, so GitHub had been rendering its tail into the wrong columns and dropping the overflow (WP-058).

## [0.2.0] - 2026-08-10

### Added

- Repository scaffold: `src/` package layout, pinned `pyproject.toml`, Makefile gate targets, and pre-commit configuration (WP-001).
- Legal and policy baseline: Apache-2.0 LICENSE with Redmon NOTICE attribution, a non-affiliation README, and the PROVENANCE / ASSUMPTIONS / DECISIONS / AGENTS / ROADMAP governance set (WP-002, WP-003).
- Continuous integration: a pull-request workflow running ruff, strict mypy, the offline pytest suite with coverage, a copyleft-dependency license audit, and a provenance-carrying commit-trailer validator (WP-004).
- Golden gate: a recompute-and-compare harness with per-metric tolerances and a `goldens/frozen/` regression path wired into `make gate` (WP-005).
- Tag-gated release workflow: a three-check guard (tag shape, CHANGELOG section present, `make gate` green) and this CHANGELOG's own scaffold (WP-006).
- Data foundation: seeded synthetic micro-dataset fixtures (boxes, polygons, and rotated scenes via fuse-augmentations) and the `Targets` container with its type-generic transform API (WP-007, WP-008).
- Augmentation pipeline: letterbox resize with an exact inverse, random affine for boxes and masks, mosaic assembly, mixup and copy-paste, and HSV jitter with horizontal flip (WP-009 through WP-013).
- COCO dataset and `LightningDataModule`, plus round-trip augmentation goldens and a debug visualizer (WP-014, WP-015).
- Model and optimizer primitives: Conv / DWConv / Bottleneck blocks, the CIoU regression loss, and Newton-Schulz orthogonalization for the MuSGD optimizer (WP-016, WP-024, WP-031).
- Architecture stack: C3k2, PSABlock/C2PSA, SPPF with a shortcut, a backbone with P3/P4/P5 taps, an attention-tail neck, a dual detection head with `reg_max=1`, and a scale registry with a param/FLOP fidelity gate against Table 7 (WP-017 through WP-023).
- Assignment and detection losses: the Task-Aligned Assigner, STAL surrogate candidate filtering for small targets, the detection branch loss, and the o2m/o2o dual-branch composition, plus synthetic assignment goldens and a single-batch overfit gradient-flow gate (WP-025 through WP-030).
- MuSGD optimizer with parameter-type split, and its toy convergence golden against plain SGD (WP-032, WP-033).
- Lightning training loop: task-conditional `LightningModule`, the ProgressiveLossSchedule hook, CloseMosaic and EMA callbacks, a LightningCLI entry with tiered experiment configs, deterministic checkpoint and resume, and the overfit-100 integration golden (WP-034 through WP-040).
- Detection decode and evaluation: score-based top-k end-to-end decoding, a class-wise NMS path, a dual-path pycocotools evaluator with an oracle round-trip gate, and a checkpoint eval script reporting both paths (WP-041 through WP-045).
- COCO 2017 downloader module and the `lucid-download` CLI (WP-068).
- CLI conveniences: packaged-config name resolution, the Det-smoke recipe as the default when no `--config` is given, and a default TensorBoard+CSV logger pair (WP-038).
- A8 learning-rate schedule: linear warmup followed by linear decay (WP-072).
- Epoch validation mAP surfaced in the training progress bar (WP-077).
- Regression coverage for the training objective: an objective-invariant gate pinning the loss coordinate frame and its scale, and a synthetic-shapes generalization golden scoring held-out data through both decode paths (WP-082, WP-083).
- Training-curve figures: a deterministic SVG script rendering a run's `metrics.csv` as a three-panel figure (a fourth panel for segmentation runs), gated against its own drawn output so a caption cannot silently diverge from what the figure draws (WP-080).

Instance segmentation:

- Mask-coefficient branch: K=32 tanh-activated coefficients added to both detection branches, opt-in and off by default so the accepted detector stays byte-for-byte unaffected (WP-047).
- Multi-scale prototype-fusion pathway (Eq. 8) and a four-layer prototype generation stack (Eq. 9), producing raw K-channel prototypes at twice the P3 resolution (WP-048, WP-049).
- Training-only auxiliary semantic branch, returning `None` whenever the module is in eval mode and sharing the detection head's prior-probability bias init (WP-050).
- Segmentation losses: box-cropped, area-normalized per-pixel instance mask BCE and a BCE+Dice auxiliary semantic loss, both built on one shared Eq. 7 mask-assembly implementation that the decode path also uses (WP-051).
- `Segmenter` model wiring: backbone, neck, dual head, prototype pathway, and auxiliary branch composed into one module, whose `deploy()` holds neither the o2m branch nor the auxiliary head, verified by both module-walk identity and exact parameter arithmetic; a Table S9 param/FLOP fidelity gate across all five scales (WP-052).
- Segmentation mask decode: sigmoid, bilinear upsample, box-crop, threshold, with masks inverted through the same letterbox geometry used for boxes, plus both-path (bbox and segm) COCO evaluation and a task-aware checkpoint eval script (WP-053).
- Segmentation training path: prototype, coefficient, and semantic heads wired into the Lightning module through shared `build.py` factories, mask supervision reusing the box loss's own assignment, and both branches' coefficients supervised under the shared alpha schedule; frozen at an overfit-100 mask IoU of 0.8146 (WP-087).
- Epoch `val/segm_mAP` logged beside `val/mAP` during segmentation validation, decoded through the same path deployment uses (WP-087).

### Changed

- Detection evaluator rebuilt on `torchmetrics.detection.MeanAveragePrecision` with the faster-coco-eval backend, dropping the pycocotools dependency (WP-069).
- Data-loading throughput: a fused affine+letterbox single warp (4.4x on the geometric path), packed ragged targets and uint8 image transport across the DataLoader IPC boundary, accelerator-aware `pin_memory`/prefetch/ thread-cap loader streaming, and an auto `num_workers` default capped by both core count and free `/dev/shm` (WP-014, WP-070, WP-071).
- Compact per-image annotation store and a separate, smaller validation-loader worker cap, closing a persistent-worker memory creep that killed long containerized runs (WP-073); loader workers now recycle every epoch instead of staying persistent, for the same reason (WP-076).
- Segmentation training throughput: batched mask-loss gather, replacing 64 forced device syncs a step with two per branch, and polygon rasterization vectorised, batched, and moved into the loader workers with each ring rasterised inside its own bounding window (WP-087).
- CI reorganized: the copyleft license audit and commit-trailer checks move to commit-time hooks, lint and test workflows split into their own files, and the `dev` extra becomes a PEP 735 dependency group so build/test tooling no longer ships in the published wheel's metadata (WP-084).
- Doctests joined the offline gate, first for `src` — 558 to 699 collected cases, and four Examples that had rotted unnoticed — and then for `scripts` as well, 767 to 819; a separate `gate-gpu` target now recomputes the GPU-marked tests and goldens on its own schedule (WP-085).
- Experiment tier configs and their historical mentions renamed to describe what they are (`det_smoke`, `det_ablations`, `seg_smoke`) instead of letters that read backwards (WP-087).

### Fixed

- S3 path-style URLs for COCO downloads, avoiding a TLS certificate mismatch on the official host's CNAME (WP-068).
- Accelerator-aware determinism default in the CLI, avoiding an MPS abort on a kernel with no deterministic implementation (WP-038).
- Prior-probability class-bias initialization and gradient clipping in the shipped recipes, fixing a dead-gradient launch failure (WP-022).
- Notebook-safe `tqdm` progress-bar default, avoiding a Rich live-render newline flood in notebook output; a spurious args/argv overlap warning from console-script runs removed, and TF32 matmuls enabled on CUDA (WP-038).
- Default DataLoader prefetch factor and a free-`/dev/shm` budget cap on the auto worker count, both closing shared-memory exhaustion on containerized hosts (WP-014).
- Stride-normalized L1 box-regression term, correcting a term that had grown to 97% of the training objective and starved classification (WP-078).
- Per-worker, per-epoch augmentation RNG re-seeding, correcting a generator that replayed one draw stream across every worker and every epoch (WP-079).
- Package version declared in one place, the module's `__version__` literal, instead of drifting between it and `pyproject.toml` — a build had shipped 0.1.0 while the runtime reported 0.0.1.dev0 (WP-001).

### Documentation

- ADR-004 (D13): the source allowlist widened to permit reading, never copying, permissively licensed detection implementations when diagnosing a structural defect.
- RF100-VL generalization work packages recorded on the roadmap (WP-074, WP-075).
- Policy that releases ship no trained weights, since the reproduction's claim rests on frozen goldens and fidelity gates rather than a retrainable binary (WP-086).
- Segmentation training-path and inference work packages added to the roadmap for Phases 7-9 (WP-087).
- Det-smoke report section and detector model card documenting the accepted run (val mAP50-95 25.30 EMA-NMS) and the two failed-attempt root causes across both decode paths (WP-045, WP-080, WP-081).
- Seg-smoke run write-up: the 0.2.0 `REPRODUCTION_REPORT.md` section (observed 26.12 box mAP, 0.728 segm mAP NMS / 0.732 E2E) and `MODEL_CARD_SEGMENTATION.md` (WP-054).
