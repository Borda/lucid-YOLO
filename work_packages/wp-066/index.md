# 📦 WP-066 — `test(export): the graph answers what the checkpoint answers`

> Research log · [Phase 10 — Consolidation](https://borda.github.io/lucid-YOLO/research/phase-10-consolidation/)
>
> Engineering log · [Phase 10 — Consolidation](https://borda.github.io/lucid-YOLO/engineering/phase-10-consolidation/)
>
> 🔗 Depends on [WP-065](https://borda.github.io/lucid-YOLO/work_packages/wp-065/) · Needed by [WP-067](https://borda.github.io/lucid-YOLO/work_packages/wp-067/), [WP-112](https://borda.github.io/lucid-YOLO/work_packages/wp-112/)
>
> ⬅️ [WP-065](https://borda.github.io/lucid-YOLO/work_packages/wp-065/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-109](https://borda.github.io/lucid-YOLO/work_packages/wp-109/) ➡️


## WP-066 — the exported graph confirms the one-to-one claim, not just its shape

<a id="wp-066"></a>

All three deploy paths export on the legacy TorchScript tracer at opset 20 and none of them contains a `NonMaxSuppression` node — a single `TopK` stands where suppression would be, and the two `Softmax` nodes belong to the attention block, not a distribution-focal-loss box head. The DFL-free, NMS-free one-to-one branch R1 claims shows up in the emitted graph itself, not only in a docstring. Verifying the graph's *arithmetic* — not just its shape — against the checkpoint it came from turned out to be almost entirely an export-tooling story rather than a fidelity one; it is recorded in full at [wp-066.md#wp-066-engineering](https://borda.github.io/lucid-YOLO/work_packages/wp-066/#wp-066-engineering).


> Engineering log · [Phase 10 — Consolidation](https://borda.github.io/lucid-YOLO/engineering/phase-10-consolidation/)


## WP-066 — an untrained checkpoint cannot tell you whether an export is right

<a id="wp-066-engineering"></a>

The architecture-fidelity half of this finding — the exported graph confirming the one-to-one, DFL-free claim — is recorded at [wp-066.md#wp-066](https://borda.github.io/lucid-YOLO/work_packages/wp-066/#wp-066). What is here is the export-and-comparison harness, which turned out to be where nearly all of the difficulty was.

The graph half of this package was the easy half. All three deploy paths export first try on the legacy TorchScript tracer at opset 20, in about 0.3 s each, and none of them contains a `NonMaxSuppression` node — 598 nodes and 30 op types for detection, with a single `TopK` where suppression would be. `dynamo=True`, the torch 2.13 default, is not used: it needs `onnxscript`, which nothing else here needs, and the legacy path already emits fully static dims. Subgraphs are walked recursively, since an absence claim a node nested in an `If` body could evade is not an absence claim; no control-flow nodes appear at all. The two `Softmax` nodes are the attention block's, not a distribution-focal-loss box head's — the DFL-free claim showing up in the emitted graph rather than in a docstring.

**The absence of an op proves the graph's shape and nothing about its arithmetic**, so the package also runs the exported model under onnxruntime against the checkpoint it came from. That is where the work was, and almost all of it went into discovering that a freshly initialized checkpoint makes the comparison meaningless while appearing to work.

An untrained BatchNorm in eval mode carries `running_var=1` and `running_mean=0`, so it rescales nothing and the signal attenuates through depth: the neck's P3 output measures 4.1e-05 and the classifier's contribution to its own logit lands near 1e-7. The prior-probability bias is `-log((1-pi)/pi) = -4.595` (A30), and float32 resolution *at* 4.595 is 4.8e-07. The contribution is an order of magnitude below the resolution of the number it is added to, so it vanishes: all 1344 class logits become one identical float32 value, `torch.unique` returns 1, and the top-k over 336 anchors is a 300-way tie. Torch and onnxruntime break that tie differently, both correctly, and a row-by-row comparison duly reported boxes 80 px apart. The head is not broken — the same model in train mode, on batch statistics, gives 1343 distinct logits of 1344 at std 0.372. Only the eval-mode statistics are degenerate, and only until something trains them.

Scaling the classification weights to force separation does not work either: a sweep at 1e2 through 1e5 saturates the sigmoid, and from 1e3 up all 336 confidences collapse to exactly 1.0 — the same degeneracy from the other end. What works is calibrating the BN running statistics with a few forward passes in train mode, then rescaling the stems to a *measured* logit spread rather than a guessed one.

**A second degeneracy was hiding behind the first, and it is the one worth remembering.** With the ranking fixed, the segmentation masks compared bit-exact: zero mismatches out of 4,915,200 pixels. They were all `False`. This head regresses ltrb directly with no DFL and no range cap, so an untrained box stem emits values straddling zero, every decoded box is inverted or empty, and `decode_instance_masks` crops each mask to nothing. A perfect score on an assertion comparing two empty tensors. Biasing the box stems positive gives 300 boxes with interiors and 5.07% mask foreground across 44 rows, and the same assertion then means something. The test asserts its own preconditions — score span, positive widths, non-empty masks — because the failure mode of this comparison is not a wrong answer, it is a vacuous one.

**The oriented head appeared to fail and did not.** Row-by-row it diverged by 88 px against 3e-05 for the other two, which reads exactly like a broken rotated decode. The tempting explanation was the A23 canonicalization swap, since `w >= h` flips discontinuously — and measuring killed it outright: the minimum `|w - h|` over 300 rows is 4.9e-02, no row within 1e-05 of the branch point. Comparing each column as a multiset instead showed every column agreeing to float32 noise, so the exported graph had been selecting the same detections all along. The fault was the comparison: rows tied on score were ordered by raw `cx`, which clusters tightly because near-square boxes put `cx` at the anchor centre and a whole grid column shares one value, so 1.5e-05 of backend noise reordered them. Rounding the ordering keys to three decimals — well above the noise, well below any real difference — pairs the rows identically on both sides. Recorded because the shape of the mistake generalizes: **an order-sensitive comparison of a tie-bearing output will accuse the implementation of an error the harness committed.**

What the comparison is worth was then measured rather than asserted. Three deliberate corruptions of the baked-in constants — transposed anchor points, an anchor grid built for the wrong canvas, and doubled strides — are each caught on every head, nine of nine, at box deltas between 1.2e+02 and 2.0e+02 against a 1e-3 tolerance. Boxes agree to 4.6e-05, scores to 1.2e-07, class ids exactly, and the masks bit-exactly at both one and the default intra-op thread count.

The anchor grid and strides are baked into the graph as buffers rather than taken as inputs, so the exported model's only input is the image and the graph is specific to one input size. That is the normal ONNX trade, and it is pinned by asserting the graph has exactly one input, so the choice cannot drift into being an accident. `onnx` and `onnxruntime` are dev-group dependencies: the export itself is `torch.onnx.export`, already a runtime dep, so nothing a consumer installs changes.

One thing this package deliberately leaves standing. The decode composition now exists twice — once in `predict.py`'s E2E path and once in the test's graph wrappers — with nothing forcing them to agree. A shared `export.py` composing deploy and decode would remove the drift risk, and it was not built here on the smallest-change rule; it is the natural home if a later package needs an exportable path it can ship.

